/* Compares the AdaLN schedule built from a curve-form file against the
 * dense one, for the schedules the app uses (needs the released weights;
 * not in `make test`).
 *
 * Usage (from the repository root):
 *   h3_adaln_curve_check <transformer dir> <curve file>
 */
#include "../h3_adaln_curve.h"
#include "../h3_dit_schedule.h"
#include "../h3_gpu.h"
#include "../h3_host.h"
#include "../h3_weights.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static float widen(uint16_t bits) {
    uint32_t value = (uint32_t)bits << 16;
    float result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

typedef struct {
    double rel_l2;
    double max_rel;
    double changed;
} diff;

static int compare(const h3_gpu_tensor *dense, const h3_gpu_tensor *curve,
                   size_t count, diff *out) {
    uint16_t *a = malloc(count * sizeof(*a));
    uint16_t *b = malloc(count * sizeof(*b));
    if (!a || !b || !h3_gpu_tensor_read_bf16(dense, a, count) ||
        !h3_gpu_tensor_read_bf16(curve, b, count)) {
        free(a);
        free(b);
        return 0;
    }
    double error = 0.0, norm = 0.0, peak = 0.0, worst = 0.0;
    size_t changed = 0;
    for (size_t index = 0; index < count; index++) {
        double x = widen(a[index]), y = widen(b[index]);
        error += (y - x) * (y - x);
        norm += x * x;
        if (fabs(x) > peak) peak = fabs(x);
        if (fabs(y - x) > worst) worst = fabs(y - x);
        changed += a[index] != b[index];
    }
    out->rel_l2 = norm > 0.0 ? sqrt(error / norm) : 0.0;
    out->max_rel = peak > 0.0 ? worst / peak : 0.0;
    out->changed = (double)changed / (double)count;
    free(a);
    free(b);
    return 1;
}


/* Exact (float64) AdaLN for some rows of one projection, from the dense
 * BF16 weight and the engine's BF16 time features, to tell which of the
 * two GPU results is closer to the true value. */
static int exact_rows(const h3_weight_store *store, h3_gpu *gpu,
                      const char *prefix, uint32_t outputs,
                      const float *times, uint32_t rows, double *out) {
    char name[128], error[256];
    const uint32_t width = H3_DIT_TIME_DIM;
    float *silu = malloc((size_t)rows * width * sizeof(*silu));
    double *input = malloc((size_t)rows * width * sizeof(*input));
    double *weight = malloc((size_t)outputs * width * sizeof(*weight));
    uint16_t *bits = malloc((size_t)outputs * width * sizeof(*bits));
    int ok = silu && input && weight && bits &&
        h3_dit_time_silu(store, gpu, times, rows, silu, error, sizeof(error));
    const h3_st_header *header = NULL;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    const h3_st_tensor *tensor = ok ? h3_weight_find(store, name, &header) : NULL;
    ok = ok && tensor && h3_st_read_data(header, tensor, bits,
        (size_t)outputs * width * sizeof(*bits), error, sizeof(error));
    for (size_t i = 0; ok && i < (size_t)outputs * width; i++) weight[i] = widen(bits[i]);
    for (size_t i = 0; ok && i < (size_t)rows * width; i++) input[i] = silu[i];
    if (ok) cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)rows,
                        (int)outputs, (int)width, 1.0, input, (int)width,
                        weight, (int)width, 0.0, out, (int)outputs);
    snprintf(name, sizeof(name), "%s.bias", prefix);
    tensor = ok ? h3_weight_find(store, name, &header) : NULL;
    ok = ok && tensor && h3_st_read_data(header, tensor, bits,
        (size_t)outputs * sizeof(*bits), error, sizeof(error));
    for (uint32_t r = 0; ok && r < rows; r++)
        for (uint32_t o = 0; o < outputs; o++)
            out[(size_t)r * outputs + o] += widen(bits[o]);
    free(silu); free(input); free(weight); free(bits);
    return ok;
}

static void against_exact(const h3_gpu_tensor *tensor, const double *exact,
                          size_t count, double *rel_l2, double *max_rel) {
    uint16_t *bits = malloc(count * sizeof(*bits));
    double error = 0.0, norm = 0.0, peak = 0.0, worst = 0.0;
    if (bits && h3_gpu_tensor_read_bf16(tensor, bits, count)) {
        for (size_t i = 0; i < count; i++) {
            double d = widen(bits[i]) - exact[i];
            error += d * d;
            norm += exact[i] * exact[i];
            if (fabs(exact[i]) > peak) peak = fabs(exact[i]);
            if (fabs(d) > worst) worst = fabs(d);
        }
    }
    free(bits);
    *rel_l2 = norm > 0.0 ? sqrt(error / norm) : 0.0;
    *max_rel = peak > 0.0 ? worst / peak : 0.0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <transformer dir> <curve file>\n", argv[0]);
        return 2;
    }
    char error[512];
    h3_weight_store *store = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!store) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    uint8_t model[32];
    h3_weight_store_fingerprint(store, model);
    h3_adaln_curve *curve = h3_adaln_curve_open(argv[2], model, error,
                                                sizeof(error));
    h3_gpu *gpu = curve ? h3_gpu_create("h3_shaders.metal", error,
                                        sizeof(error)) : NULL;
    if (!gpu) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    printf("curve rank %u, grid %u\n", h3_adaln_curve_rank(curve),
           h3_adaln_curve_grid(curve));

    struct { int evaluations; float audio_shift; int visual, audio; } cases[] = {
        {20, 3.0f, 0, 0}, {20, 3.0f, 1, 1}, {8, 3.0f, 1, 0},
        {4, 3.0f, 0, 0}, {4, 1.0f, 1, 1}, {30, 6.0f, 0, 1},
    };
    double worst_l2 = 0.0, worst_max = 0.0;
    int failures = 0;
    for (size_t c = 0; c < sizeof(cases) / sizeof(*cases); c++) {
        h3_sigma_schedule sigmas;
        if (!h3_serving_schedule_build_shifted(cases[c].evaluations,
                                               cases[c].audio_shift, &sigmas)) {
            fprintf(stderr, "cannot build schedule %zu\n", c);
            return 1;
        }
        double started = now();
        h3_dit_schedule *dense = h3_dit_schedule_precompute(
            store, gpu, &sigmas, cases[c].visual, cases[c].audio, NULL, NULL,
            error, sizeof(error));
        double dense_seconds = now() - started;
        started = now();
        h3_dit_schedule *fast = dense ? h3_dit_schedule_precompute_curve(
            curve, gpu, &sigmas, cases[c].visual, cases[c].audio, NULL, NULL,
            error, sizeof(error)) : NULL;
        double curve_seconds = now() - started;
        if (!fast) {
            fprintf(stderr, "%s\n", error);
            return 1;
        }
        uint32_t rows = h3_dit_schedule_time_rows(dense);
        if (rows != h3_dit_schedule_time_rows(fast)) {
            fprintf(stderr, "row count differs\n");
            return 1;
        }
        if (c == 1) {
            /* Rebuild this case's row times the way prepare_rows does. */
            float *times = calloc(rows, sizeof(*times));
            for (int step = 0; step < sigmas.steps; step++) {
                times[h3_dit_schedule_video_row(dense, step)] =
                    1.0f - sigmas.video[step];
                times[h3_dit_schedule_audio_row(dense, step)] =
                    1.0f - sigmas.audio[step];
                if (cases[c].visual &&
                    h3_dit_schedule_visual_condition_row(dense, step) !=
                    h3_dit_schedule_video_row(dense, step))
                    times[h3_dit_schedule_visual_condition_row(dense, step)] = 0.999f;
                if (cases[c].audio &&
                    h3_dit_schedule_audio_condition_row(dense, step) !=
                    h3_dit_schedule_audio_row(dense, step))
                    times[h3_dit_schedule_audio_condition_row(dense, step)] = 1.0f;
            }
            unsigned probes[] = {0, 5, 25, 49, H3_DIT_BLOCKS};
            for (size_t p = 0; p < sizeof(probes) / sizeof(*probes); p++) {
                unsigned block = probes[p];
                char prefix[96];
                uint32_t outputs = block < H3_DIT_BLOCKS ?
                    H3_DIT_MODALITIES * H3_DIT_ADALN_SLOTS * H3_DIT_HIDDEN :
                    2u * H3_DIT_HIDDEN;
                if (block < H3_DIT_BLOCKS)
                    snprintf(prefix, sizeof(prefix), "blocks.%u.adaln_proj.linear", block);
                else
                    snprintf(prefix, sizeof(prefix), "final_layer.adaln_proj.linear");
                double *exact = malloc((size_t)rows * outputs * sizeof(*exact));
                if (!exact || !exact_rows(store, gpu, prefix, outputs, times, rows, exact)) {
                    fprintf(stderr, "cannot compute exact %s\n", prefix);
                    return 1;
                }
                double dl2, dmax, cl2, cmax;
                const h3_gpu_tensor *a = block < H3_DIT_BLOCKS ?
                    h3_dit_schedule_block(dense, block) : h3_dit_schedule_final(dense);
                const h3_gpu_tensor *b = block < H3_DIT_BLOCKS ?
                    h3_dit_schedule_block(fast, block) : h3_dit_schedule_final(fast);
                against_exact(a, exact, (size_t)rows * outputs, &dl2, &dmax);
                against_exact(b, exact, (size_t)rows * outputs, &cl2, &cmax);
                printf("  vs exact float64, %-30s dense rel-L2 %.2e max %.2e | "
                       "curve rel-L2 %.2e max %.2e\n", prefix, dl2, dmax, cl2, cmax);
                free(exact);
            }
            free(times);
        }
        double case_l2 = 0.0, case_max = 0.0, case_changed = 0.0;
        for (unsigned block = 0; block <= H3_DIT_BLOCKS; block++) {
            const h3_gpu_tensor *a = block < H3_DIT_BLOCKS ?
                h3_dit_schedule_block(dense, block) : h3_dit_schedule_final(dense);
            const h3_gpu_tensor *b = block < H3_DIT_BLOCKS ?
                h3_dit_schedule_block(fast, block) : h3_dit_schedule_final(fast);
            size_t width = block < H3_DIT_BLOCKS ?
                (size_t)H3_DIT_MODALITIES * H3_DIT_ADALN_SLOTS * H3_DIT_HIDDEN :
                2u * H3_DIT_HIDDEN;
            diff d;
            if (!compare(a, b, (size_t)rows * width, &d)) {
                fprintf(stderr, "cannot read schedule values\n");
                return 1;
            }
            if (d.rel_l2 > case_l2) case_l2 = d.rel_l2;
            if (d.max_rel > case_max) case_max = d.max_rel;
            if (d.changed > case_changed) case_changed = d.changed;
        }
        printf("%2d steps audio shift %.1f visual %d audio %d (%u rows): "
               "worst rel-L2 %.2e, worst max %.2e, BF16 values changed <= "
               "%.1f%%; dense %.2fs, curve %.2fs\n",
               cases[c].evaluations, cases[c].audio_shift, cases[c].visual,
               cases[c].audio, rows, case_l2, case_max, 100.0 * case_changed,
               dense_seconds, curve_seconds);
        if (case_l2 > worst_l2) worst_l2 = case_l2;
        if (case_max > worst_max) worst_max = case_max;
        if (case_l2 > 1e-3) failures++;
        h3_dit_schedule_free(dense);
        h3_dit_schedule_free(fast);
    }
    printf("overall worst rel-L2 %.2e, worst max %.2e\n", worst_l2, worst_max);
    h3_adaln_curve_free(curve);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    return failures ? 1 : 0;
}
