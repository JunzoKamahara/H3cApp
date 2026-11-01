/* Dense SDPA execution modes (issue #2).
 *
 *   ./h3_sdpa_tests                 regression: every mode against a CPU
 *                                   reference, row- and head-major layouts,
 *                                   tails, f32 and bf16; exits 1 on failure
 *   ./h3_sdpa_tests --probe S H D   the shape a generation really sends
 *                                   (e.g. 17408 56 128): logs the device and
 *                                   call, times chunked, explicit and, last,
 *                                   the one-call path, and reports memory.
 *                                   On a machine that crashes in the
 *                                   one-call path, the chunked lines print
 *                                   first and the crash comes at the end.
 *
 * The reference is plain f32 attention on the CPU from the same bf16 (or
 * f32) inputs; every GPU mode must match it within bf16 output rounding and
 * contain no NaN or Inf. Chunked and full are also compared with each other.
 */
#include <mach/mach.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../h3_gpu.h"

static uint32_t rng_state = 0x9e3779b9u;
static float rng_normal(void) {
    /* Box-Muller on a 32-bit LCG; deterministic across runs. */
    rng_state = rng_state * 1664525u + 1013904223u;
    float u1 = ((rng_state >> 8) + 1.0f) / 16777217.0f;
    rng_state = rng_state * 1664525u + 1013904223u;
    float u2 = (rng_state >> 8) / 16777216.0f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (uint16_t)(bits >> 16);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double footprint_gib(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) !=
        KERN_SUCCESS) return 0.0;
    return (double)info.phys_footprint / (1024.0 * 1024.0 * 1024.0);
}

static int failures = 0;
#define CHECK(condition, ...) do {                                             \
    if (!(condition)) {                                                        \
        fprintf(stderr, "FAIL tests/test_sdpa.c:%d: ", __LINE__);              \
        fprintf(stderr, __VA_ARGS__);                                          \
        fprintf(stderr, "\n");                                                 \
        failures++;                                                            \
    }                                                                          \
} while (0)

/* Row-major [S, H, D] f32 values -> reference attention, row-major. */
static void reference_attention(const float *q, const float *k, const float *v,
                                uint32_t S, uint32_t H, uint32_t D, float scale,
                                float *out) {
    float *scores = malloc((size_t)S * sizeof(*scores));
    for (uint32_t h = 0; h < H; h++) {
        for (uint32_t i = 0; i < S; i++) {
            const float *qi = q + ((size_t)i * H + h) * D;
            float max_score = -INFINITY;
            for (uint32_t j = 0; j < S; j++) {
                const float *kj = k + ((size_t)j * H + h) * D;
                float dot = 0.0f;
                for (uint32_t d = 0; d < D; d++) dot += qi[d] * kj[d];
                scores[j] = dot * scale;
                if (scores[j] > max_score) max_score = scores[j];
            }
            double sum = 0.0;
            for (uint32_t j = 0; j < S; j++) {
                scores[j] = expf(scores[j] - max_score);
                sum += scores[j];
            }
            float *oi = out + ((size_t)i * H + h) * D;
            for (uint32_t d = 0; d < D; d++) oi[d] = 0.0f;
            for (uint32_t j = 0; j < S; j++) {
                float p = (float)(scores[j] / sum);
                const float *vj = v + ((size_t)j * H + h) * D;
                for (uint32_t d = 0; d < D; d++) oi[d] += p * vj[d];
            }
        }
    }
    free(scores);
}

typedef struct {
    double max_abs, mean_abs;
    size_t non_finite;
} error_stats;

static error_stats compare(const float *a, const float *b, size_t count) {
    error_stats e = {0.0, 0.0, 0};
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(a[i])) { e.non_finite++; continue; }
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > e.max_abs) e.max_abs = d;
        e.mean_abs += d;
    }
    e.mean_abs /= (double)count;
    return e;
}

/* Reads a GPU tensor back as f32 (row-major [S,H,D] or head-major). */
static float *read_output(h3_gpu *gpu, h3_gpu_tensor *tensor, size_t count,
                          int bf16, int head_major, uint32_t S, uint32_t H,
                          uint32_t D) {
    float *rows = malloc(count * sizeof(*rows));
    if (bf16) {
        uint16_t *raw = malloc(count * sizeof(*raw));
        if (!h3_gpu_tensor_read_bf16(tensor, raw, count)) {
            fprintf(stderr, "read bf16: %s\n", h3_gpu_error(gpu));
            exit(1);
        }
        for (size_t i = 0; i < count; i++) rows[i] = bf16_to_f32(raw[i]);
        free(raw);
    } else if (!h3_gpu_tensor_read_f32(tensor, rows, count)) {
        fprintf(stderr, "read f32: %s\n", h3_gpu_error(gpu));
        exit(1);
    }
    if (!head_major) return rows;
    float *fixed = malloc(count * sizeof(*fixed));
    for (uint32_t h = 0; h < H; h++)
        for (uint32_t s = 0; s < S; s++)
            memcpy(fixed + ((size_t)s * H + h) * D,
                   rows + ((size_t)h * S + s) * D, D * sizeof(float));
    free(rows);
    return fixed;
}

typedef struct {
    const char *name;
    h3_gpu_sdpa_mode mode;
    uint64_t budget;
} mode_case;

/* One shape through every mode and layout, against the CPU reference. */
static void run_shape(h3_gpu *gpu, uint32_t S, uint32_t H, uint32_t D,
                      int bf16) {
    size_t count = (size_t)S * H * D;
    float scale = 1.0f / sqrtf((float)D);
    float *q = malloc(count * sizeof(float)), *k = malloc(count * sizeof(float));
    float *v = malloc(count * sizeof(float)), *ref = malloc(count * sizeof(float));
    uint16_t *qb = malloc(count * 2), *kb = malloc(count * 2), *vb = malloc(count * 2);
    for (size_t i = 0; i < count; i++) {
        q[i] = rng_normal() * 0.7f; k[i] = rng_normal() * 0.7f; v[i] = rng_normal();
        if (bf16) {
            qb[i] = f32_to_bf16(q[i]); q[i] = bf16_to_f32(qb[i]);
            kb[i] = f32_to_bf16(k[i]); k[i] = bf16_to_f32(kb[i]);
            vb[i] = f32_to_bf16(v[i]); v[i] = bf16_to_f32(vb[i]);
        }
    }
    reference_attention(q, k, v, S, H, D, scale, ref);

    /* Head-major copies for the head-major input layout. */
    uint16_t *qh = malloc(count * 2), *kh = malloc(count * 2), *vh = malloc(count * 2);
    for (uint32_t s = 0; s < S; s++)
        for (uint32_t h = 0; h < H; h++) {
            memcpy(qh + ((size_t)h * S + s) * D, qb + ((size_t)s * H + h) * D, D * 2);
            memcpy(kh + ((size_t)h * S + s) * D, kb + ((size_t)s * H + h) * D, D * 2);
            memcpy(vh + ((size_t)h * S + s) * D, vb + ((size_t)s * H + h) * D, D * 2);
        }

    uint64_t one_head = (uint64_t)S * S * (bf16 ? 2 : 4);
    mode_case cases[] = {
        {"full", H3_GPU_SDPA_FULL, 0},
        {"chunked, heads grouped", H3_GPU_SDPA_CHUNKED, one_head * 3 + 1},
        {"chunked, one head", H3_GPU_SDPA_CHUNKED, one_head},
        {"chunked, row blocks", H3_GPU_SDPA_CHUNKED, one_head / 3},
        {"explicit, row blocks", H3_GPU_SDPA_EXPLICIT, one_head},
    };
    float *full_rows = NULL;
    double tolerance_max = bf16 ? 0.03 : 0.002;
    double tolerance_mean = bf16 ? 0.003 : 0.0002;
    for (size_t c = 0; c < sizeof(cases) / sizeof(*cases); c++) {
        for (int layout = 0; layout < 3; layout++) {
            /* 0: row-major in/out; 1: row-major in, head-major out;
             * 2: head-major in, head-major out (the M5 int8 pairing). bf16
             * only for 1 and 2 (the f32 entry points are row-major). */
            if (layout && !bf16) continue;
            int head_in = layout == 2, head_out = layout >= 1;
            h3_gpu_tensor *tq, *tk, *tv, *to;
            if (bf16) {
                tq = h3_gpu_tensor_from_bf16(gpu, head_in ? qh : qb, count);
                tk = h3_gpu_tensor_from_bf16(gpu, head_in ? kh : kb, count);
                tv = h3_gpu_tensor_from_bf16(gpu, head_in ? vh : vb, count);
                to = h3_gpu_tensor_new_bf16(gpu, count);
            } else {
                tq = h3_gpu_tensor_from_f32(gpu, q, count);
                tk = h3_gpu_tensor_from_f32(gpu, k, count);
                tv = h3_gpu_tensor_from_f32(gpu, v, count);
                to = h3_gpu_tensor_new_f32(gpu, count);
            }
            h3_gpu_set_sdpa_mode(gpu, cases[c].mode, cases[c].budget);
            uint32_t plan_heads = 0, plan_rows = 0;
            h3_gpu_sdpa_chunk_plan(gpu, S, S, H, bf16 ? H3_GPU_BF16 : H3_GPU_F32,
                                   &plan_heads, &plan_rows);
            h3_gpu_stats before; h3_gpu_get_stats(gpu, &before);
            int ok = h3_gpu_begin(gpu);
            if (ok) {
                if (head_in) h3_gpu_set_head_major_sdpa_inputs(gpu, 1);
                ok = bf16 ?
                    (head_out ?
                        h3_gpu_sdpa_bf16_head_major_output(gpu, to, tq, tk, tv, S, H, D, scale) :
                        h3_gpu_sdpa_bf16(gpu, to, tq, tk, tv, S, H, D, scale)) :
                    h3_gpu_sdpa_f32(gpu, to, tq, tk, tv, S, H, D, scale);
            }
            if (ok) ok = h3_gpu_submit(gpu);
            CHECK(ok, "%s layout %d S=%u H=%u D=%u %s: %s", cases[c].name,
                  layout, S, H, D, bf16 ? "bf16" : "f32", h3_gpu_error(gpu));
            if (!ok) { failures++; continue; }
            h3_gpu_stats after; h3_gpu_get_stats(gpu, &after);
            uint64_t pieces = after.mps_sdpa_chunk_dispatches -
                              before.mps_sdpa_chunk_dispatches;
            uint64_t expected_pieces = cases[c].mode == H3_GPU_SDPA_FULL ? 0 :
                (uint64_t)((H + plan_heads - 1) / plan_heads) *
                ((S + plan_rows - 1) / plan_rows);
            CHECK(pieces == expected_pieces,
                  "%s: %llu pieces, plan %u heads x %u rows -> %llu",
                  cases[c].name, (unsigned long long)pieces, plan_heads,
                  plan_rows, (unsigned long long)expected_pieces);
            float *rows = read_output(gpu, to, count, bf16, head_out, S, H, D);
            error_stats e = compare(rows, ref, count);
            printf("  %-24s layout %d S=%5u H=%2u D=%3u %s pieces %3llu "
                   "(%u heads x %u rows)  max %.5f mean %.6f non-finite %zu\n",
                   cases[c].name, layout, S, H, D, bf16 ? "bf16" : "f32 ",
                   (unsigned long long)pieces, plan_heads, plan_rows,
                   e.max_abs, e.mean_abs, e.non_finite);
            CHECK(e.non_finite == 0, "%s: non-finite outputs", cases[c].name);
            CHECK(e.max_abs <= tolerance_max && e.mean_abs <= tolerance_mean,
                  "%s layout %d: error vs reference too large (max %.5f mean %.6f)",
                  cases[c].name, layout, e.max_abs, e.mean_abs);
            if (c == 0 && layout == 0) full_rows = rows;
            else {
                if (full_rows) {
                    error_stats d = compare(rows, full_rows, count);
                    CHECK(d.max_abs <= (bf16 ? 0.02 : 0.001),
                          "%s layout %d: differs from the one-call result (max %.5f)",
                          cases[c].name, layout, d.max_abs);
                }
                free(rows);
            }
            h3_gpu_tensor_free(tq); h3_gpu_tensor_free(tk);
            h3_gpu_tensor_free(tv); h3_gpu_tensor_free(to);
        }
    }
    free(full_rows);
    free(q); free(k); free(v); free(ref);
    free(qb); free(kb); free(vb); free(qh); free(kh); free(vh);
}

/* GPU-only consistency at a real shape: chunked vs full for the row-major
 * and the head-major (M5 int8) layouts, without a CPU reference. */
static void probe_layouts(h3_gpu *gpu, uint32_t S, uint32_t H, uint32_t D) {
    size_t count = (size_t)S * H * D;
    float scale = 1.0f / sqrtf((float)D);
    uint16_t *data = malloc(count * 2);
    h3_gpu_tensor *t[3];
    for (int i = 0; i < 3; i++) {
        for (size_t j = 0; j < count; j++) data[j] = f32_to_bf16(rng_normal() * (i == 2 ? 1.0f : 0.7f));
        t[i] = h3_gpu_tensor_from_bf16(gpu, data, count);
    }
    free(data);
    h3_gpu_tensor *to = h3_gpu_tensor_new_bf16(gpu, count);
    for (int layout = 0; layout < 2; layout++) {
        float *full = NULL;
        for (int mode = 0; mode < 2; mode++) {
            h3_gpu_set_sdpa_mode(gpu, mode ? H3_GPU_SDPA_CHUNKED : H3_GPU_SDPA_FULL, 0);
            int ok = h3_gpu_begin(gpu);
            if (ok && layout) h3_gpu_set_head_major_sdpa_inputs(gpu, 1);
            if (ok) ok = layout ?
                h3_gpu_sdpa_bf16_head_major_output(gpu, to, t[0], t[1], t[2], S, H, D, scale) :
                h3_gpu_sdpa_bf16(gpu, to, t[0], t[1], t[2], S, H, D, scale);
            if (ok) ok = h3_gpu_submit(gpu);
            if (!ok) { printf("probe layouts: failed: %s\n", h3_gpu_error(gpu)); return; }
            float *rows = read_output(gpu, to, count, 1, 0, S, H, D);
            if (!full) full = rows;
            else {
                error_stats d = compare(rows, full, count);
                printf("probe: S=%u H=%u %s layout: chunked vs full max %.5f mean %.7f non-finite %zu\n",
                       S, H, layout ? "head-major in/out" : "row-major in/out",
                       d.max_abs, d.mean_abs, d.non_finite);
                free(rows);
            }
        }
        free(full);
    }
    for (int i = 0; i < 3; i++) h3_gpu_tensor_free(t[i]);
    h3_gpu_tensor_free(to);
}

static void probe(h3_gpu *gpu, uint32_t S, uint32_t H, uint32_t D, int reps) {
    size_t count = (size_t)S * H * D;
    float scale = 1.0f / sqrtf((float)D);
    uint16_t *data = malloc(count * 2);
    for (size_t i = 0; i < count; i++) data[i] = f32_to_bf16(rng_normal() * 0.7f);
    h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, data, count);
    for (size_t i = 0; i < count; i++) data[i] = f32_to_bf16(rng_normal() * 0.7f);
    h3_gpu_tensor *tk = h3_gpu_tensor_from_bf16(gpu, data, count);
    for (size_t i = 0; i < count; i++) data[i] = f32_to_bf16(rng_normal());
    h3_gpu_tensor *tv = h3_gpu_tensor_from_bf16(gpu, data, count);
    free(data);
    h3_gpu_tensor *to = h3_gpu_tensor_new_bf16(gpu, count);
    if (!tq || !tk || !tv || !to) { fprintf(stderr, "allocation failed\n"); exit(1); }
    setenv("H3_SDPA_LOG", "1", 0);
    printf("probe: S=%u H=%u D=%u bf16 row-major; auto mode here = %s; "
           "footprint before %.2f GiB\n", S, H, D,
           h3_gpu_sdpa_mode_name(h3_gpu_sdpa_effective_mode(gpu)),
           footprint_gib());
    mode_case cases[] = {
        {"chunked", H3_GPU_SDPA_CHUNKED, 0},
        {"explicit", H3_GPU_SDPA_EXPLICIT, 0},
        {"full", H3_GPU_SDPA_FULL, 0},
    };
    float *first = NULL;
    for (size_t c = 0; c < sizeof(cases) / sizeof(*cases); c++) {
        h3_gpu_set_sdpa_mode(gpu, cases[c].mode, 0);
        if (cases[c].mode == H3_GPU_SDPA_FULL)
            printf("probe: running the one-call path last (this is the call "
                   "that aborted in issue #2)\n");
        fflush(stdout);
        double best = INFINITY;
        for (int r = 0; r < reps; r++) {
            double start = now_seconds();
            int ok = h3_gpu_begin(gpu) &&
                     h3_gpu_sdpa_bf16(gpu, to, tq, tk, tv, S, H, D, scale) &&
                     h3_gpu_submit(gpu);
            double elapsed = now_seconds() - start;
            if (!ok) {
                printf("probe: %s FAILED: %s\n", cases[c].name, h3_gpu_error(gpu));
                break;
            }
            if (elapsed < best) best = elapsed;
        }
        h3_gpu_stats stats; h3_gpu_get_stats(gpu, &stats);
        printf("probe: %-8s best %.3f s/call  footprint %.2f GiB  "
               "peak live %.2f GiB  chunk pieces so far %llu\n", cases[c].name,
               best, footprint_gib(),
               (double)stats.peak_live_bytes / (1024.0 * 1024.0 * 1024.0),
               (unsigned long long)stats.mps_sdpa_chunk_dispatches);
        float *rows = read_output(gpu, to, count, 1, 0, S, H, D);
        error_stats self = compare(rows, rows, count);
        if (!first) first = rows;
        else {
            error_stats d = compare(rows, first, count);
            printf("probe: %-8s vs chunked: max %.5f mean %.6f non-finite %zu\n",
                   cases[c].name, d.max_abs, d.mean_abs, d.non_finite);
            free(rows);
        }
        if (self.non_finite) printf("probe: %s produced %zu non-finite values\n",
                                    cases[c].name, self.non_finite);
        fflush(stdout);
    }
    free(first);
}

int main(int argc, char **argv) {
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "FAIL tests/test_sdpa.c: %s\n", error); return 1; }
    if (argc >= 5 && !strcmp(argv[1], "--probe-layouts")) {
        probe_layouts(gpu, (uint32_t)atoi(argv[2]), (uint32_t)atoi(argv[3]),
                      (uint32_t)atoi(argv[4]));
        h3_gpu_free(gpu);
        return 0;
    }
    if (argc >= 5 && !strcmp(argv[1], "--probe")) {
        probe(gpu, (uint32_t)atoi(argv[2]), (uint32_t)atoi(argv[3]),
              (uint32_t)atoi(argv[4]), argc > 5 ? atoi(argv[5]) : 3);
        h3_gpu_free(gpu);
        return 0;
    }
    printf("auto SDPA mode on this machine: %s\n",
           h3_gpu_sdpa_mode_name(h3_gpu_sdpa_effective_mode(gpu)));
    /* Sequences that are not multiples of the row blocks, head counts that
     * are not multiples of the head groups, and the DiT's head_dim. */
    run_shape(gpu, 333, 5, 128, 1);
    run_shape(gpu, 700, 8, 128, 1);
    run_shape(gpu, 257, 4, 64, 0);
    h3_gpu_free(gpu);
    if (failures) { fprintf(stderr, "FAIL tests/test_sdpa.c: %d failures\n", failures); return 1; }
    printf("ok: dense SDPA modes (full, chunked, explicit) match the CPU reference\n");
    return 0;
}
