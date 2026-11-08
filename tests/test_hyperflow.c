/* HyperFlow H3 Flash (h3_hyperflow.h, HYPERFLOW_PLAN.ja.md): the fixed
 * step grids, the (t, r) row layout, and - with a checkpoint - the inspector
 * and the two-time AdaLN schedule against a double-precision CPU reference
 * computed from the same weights.
 *
 * H3_HYPERFLOW_DIR    a HyperFlow FL2VA transformer directory (a GGUF; only
 *                     the time embedders, AdaLN projections and headers are
 *                     read, so a sparse copy with just those tensors works)
 * H3_HYPERFLOW_STRICT=1  a missing checkpoint fails instead of skipping. */
#include "h3.h"
#include "h3_dit_schedule.h"
#include "h3_hyperflow.h"
#include "h3_host.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run;
#define CHECK(condition) do { tests_run++; if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); exit(1); } } while (0)

static int strict(void) {
    const char *s = getenv("H3_HYPERFLOW_STRICT");
    return s && *s && strcmp(s, "0") != 0;
}

static void skip(const char *what) {
    if (strict()) { fprintf(stderr, "FAIL: %s (H3_HYPERFLOW_STRICT=1)\n", what); exit(1); }
    printf("skip: %s\n", what);
}

/* The README's sd-cli --sigmas lists (the shifted video grids). */
static void test_grids(void) {
    static const double six[] = {1.0, 0.982655942, 0.914642572, 0.674755692,
                                 0.385223746, 0.159055322, 0.0};
    static const double four[] = {1.0, 0.962575793, 0.755781889, 0.312497854,
                                  0.0};
    static const double eight[] = {1.0, 0.993909776, 0.9842875, 0.966063738,
                                   0.923076928, 0.8349424, 0.696852028,
                                   0.468753308, 0.0};
    const double *expected[3] = {four, six, eight};
    int steps[3] = {4, 6, 8};
    for (int index = 0; index < 3; index++) {
        h3_sigma_schedule schedule;
        CHECK(h3_hyperflow_schedule_build(steps[index], &schedule));
        CHECK(schedule.steps == steps[index]);
        for (int step = 0; step <= steps[index]; step++) {
            CHECK(fabs(schedule.video[step] - expected[index][step]) < 1e-6);
            /* the same base grid with shift 3 */
            double video = schedule.video[step];
            double base = video / (12.0 - 11.0 * video);
            double audio = 3.0 * base / (1.0 + 2.0 * base);
            CHECK(fabs(schedule.audio[step] - audio) < 2e-6);
        }
        CHECK(schedule.video[steps[index]] == 0.0f &&
              schedule.audio[steps[index]] == 0.0f);
    }
    h3_sigma_schedule schedule;
    CHECK(!h3_hyperflow_schedule_build(5, &schedule));
    CHECK(!h3_hyperflow_schedule_build(20, &schedule));
    CHECK(!h3_hyperflow_schedule_build(6, NULL));
}

static void test_rows(void) {
    h3_sigma_schedule sigmas;
    CHECK(h3_hyperflow_schedule_build(6, &sigmas));
    h3_dit_time_pair pairs[14];
    uint32_t video[6], audio[6], visual[6], audio_condition[6];
    uint32_t rows = h3_dit_two_time_rows(&sigmas, 1, 1, pairs, 14, video,
                                         audio, visual, audio_condition);
    /* step 0: video (0, r_v) and audio (0, r_a) differ in r; every later
     * step has distinct times; then (0.999, 0.999) and (1, 1). */
    CHECK(rows == 14);
    for (int step = 0; step < 6; step++) {
        CHECK(video[step] == (uint32_t)(2 * step));
        CHECK(audio[step] == (uint32_t)(2 * step + 1));
        CHECK(pairs[video[step]].t == 1.0f - sigmas.video[step]);
        CHECK(pairs[video[step]].r == 1.0f - sigmas.video[step + 1]);
        CHECK(pairs[audio[step]].t == 1.0f - sigmas.audio[step]);
        CHECK(pairs[audio[step]].r == 1.0f - sigmas.audio[step + 1]);
        CHECK(visual[step] == 12 && audio_condition[step] == 13);
    }
    CHECK(pairs[0].t == 0.0f && pairs[1].t == 0.0f &&
          pairs[0].r != pairs[1].r);
    CHECK(pairs[10].r == 1.0f && pairs[11].r == 1.0f);
    CHECK(pairs[12].t == 0.999f && pairs[12].r == 0.999f);
    CHECK(pairs[13].t == 1.0f && pairs[13].r == 1.0f);

    /* Clean (forced) audio: every audio row is (1, 1), shared with the
     * audio condition row. */
    for (int step = 0; step <= 6; step++) sigmas.audio[step] = 0.0f;
    rows = h3_dit_two_time_rows(&sigmas, 0, 1, pairs, 14, video, audio, NULL,
                                audio_condition);
    CHECK(rows == 7);
    for (int step = 0; step < 6; step++) {
        CHECK(video[step] == (uint32_t)(step == 0 ? 0 : step + 1));
        CHECK(audio[step] == 1 && audio_condition[step] == 1);
    }
    CHECK(pairs[1].t == 1.0f && pairs[1].r == 1.0f);
    CHECK(!h3_dit_two_time_rows(&sigmas, 1, 0, pairs, 14, video, audio, NULL,
                                NULL));
    CHECK(!h3_dit_two_time_rows(&sigmas, 0, 0, pairs, 3, video, audio, NULL,
                                NULL));
}

static double bf16_value(uint16_t bits) {
    uint32_t widened = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &widened, sizeof(value));
    return value;
}

static void *read_tensor(const h3_weight_store *store, const char *name,
                         h3_dtype dtype, size_t elements) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    CHECK(tensor && tensor->dtype == dtype &&
          h3_st_tensor_elements(tensor) == elements);
    size_t bytes = elements * h3_dtype_size(dtype);
    void *data = malloc(bytes);
    char error[256];
    CHECK(data && h3_st_read_data(header, tensor, data, bytes, error,
                                  sizeof(error)));
    return data;
}

/* E(t) of one embedder in double: proj_out(silu(proj_in(features(t)))). */
static void embed(const h3_weight_store *store, const char *prefix, double t,
                  uint32_t time_dim, double *out) {
    char name[96];
    snprintf(name, sizeof(name), "%s.proj_in.weight", prefix);
    float *in_w = read_tensor(store, name, H3_DTYPE_F32, 5376u * 256u);
    snprintf(name, sizeof(name), "%s.proj_in.bias", prefix);
    float *in_b = read_tensor(store, name, H3_DTYPE_F32, 5376u);
    snprintf(name, sizeof(name), "%s.proj_out.weight", prefix);
    float *out_w = read_tensor(store, name, H3_DTYPE_F32,
                               (size_t)time_dim * 5376u);
    snprintf(name, sizeof(name), "%s.proj_out.bias", prefix);
    float *out_b = read_tensor(store, name, H3_DTYPE_F32, time_dim);
    double features[256], hidden[5376];
    for (int index = 0; index < 128; index++) {
        double frequency = exp(-log(10000.0) * index / 128.0);
        features[index] = cos(t * frequency);
        features[128 + index] = sin(t * frequency);
    }
    for (int row = 0; row < 5376; row++) {
        double sum = in_b[row];
        for (int column = 0; column < 256; column++)
            sum += (double)in_w[(size_t)row * 256 + column] * features[column];
        hidden[row] = sum / (1.0 + exp(-sum));
    }
    for (uint32_t row = 0; row < time_dim; row++) {
        double sum = out_b[row];
        for (int column = 0; column < 5376; column++)
            sum += (double)out_w[(size_t)row * 5376 + column] * hidden[column];
        out[row] = sum;
    }
    free(in_w); free(in_b); free(out_w); free(out_b);
}

static uint16_t bf16_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static uint64_t rng_state = 0x243f6a8885a308d3ull;
static float uniform(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (float)((rng_state >> 40) * (1.0 / 16777216.0)) * 2.0f - 1.0f;
}

static uint16_t *random_bf16(size_t count, float scale) {
    uint16_t *values = malloc(count * sizeof(*values));
    CHECK(values);
    for (size_t index = 0; index < count; index++)
        values[index] = bf16_bits(uniform() * scale);
    return values;
}

/* The split Q/K norm+RoPE and split SwiGLU kernels against the fused ones
 * they stand in for (same arithmetic: bit-identical), and a factored linear
 * up(down(x)) at the released ranks against a CPU reference. */
static void test_kernels(void) {
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) fprintf(stderr, "%s\n", error);
    CHECK(gpu);
    const uint32_t rows = 37, heads = 56, head_dim = 128, half = 48;
    const uint32_t inner = heads * head_dim;
    size_t count = (size_t)rows * inner;
    uint16_t *q = random_bf16(count, 3.0f), *k = random_bf16(count, 3.0f),
             *v = random_bf16(count, 3.0f);
    uint16_t *qn = random_bf16(head_dim, 1.0f), *kn = random_bf16(head_dim, 1.0f);
    uint16_t *cs = random_bf16((size_t)rows * half, 1.0f),
             *sn = random_bf16((size_t)rows * half, 1.0f);
    uint16_t *fused = malloc(count * 3 * sizeof(*fused));
    CHECK(fused);
    for (uint32_t row = 0; row < rows; row++) {
        memcpy(fused + (size_t)row * inner * 3, q + (size_t)row * inner,
               inner * sizeof(*q));
        memcpy(fused + (size_t)row * inner * 3 + inner, k + (size_t)row * inner,
               inner * sizeof(*k));
        memcpy(fused + (size_t)row * inner * 3 + 2 * inner,
               v + (size_t)row * inner, inner * sizeof(*v));
    }
    h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, q, count),
                  *tk = h3_gpu_tensor_from_bf16(gpu, k, count),
                  *tqkv = h3_gpu_tensor_from_bf16(gpu, fused, count * 3),
                  *tqn = h3_gpu_tensor_from_bf16(gpu, qn, head_dim),
                  *tkn = h3_gpu_tensor_from_bf16(gpu, kn, head_dim),
                  *tcs = h3_gpu_tensor_from_bf16(gpu, cs, (size_t)rows * half),
                  *tsn = h3_gpu_tensor_from_bf16(gpu, sn, (size_t)rows * half),
                  *oq1 = h3_gpu_tensor_new_bf16(gpu, count),
                  *ok1 = h3_gpu_tensor_new_bf16(gpu, count),
                  *ov1 = h3_gpu_tensor_new_bf16(gpu, count),
                  *oq2 = h3_gpu_tensor_new_bf16(gpu, count),
                  *ok2 = h3_gpu_tensor_new_bf16(gpu, count);
    CHECK(tq && tk && tqkv && tqn && tkn && tcs && tsn && oq1 && ok1 && ov1 &&
          oq2 && ok2);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_qkv_rope_bf16(gpu, oq1, ok1, ov1, tqkv, tqn, tkn, tcs, tsn,
                               rows, heads, head_dim, half, 1e-5f));
    if (!h3_gpu_qk_rope_split_bf16(gpu, oq2, ok2, tq, tk, tqn, tkn, tcs, tsn,
                                   rows, heads, head_dim, half, 1e-5f)) {
        fprintf(stderr, "%s\n", h3_gpu_error(gpu));
        CHECK(0);
    }
    CHECK(h3_gpu_submit(gpu));
    uint16_t *a = malloc(count * 2), *b = malloc(count * 2);
    CHECK(a && b);
    CHECK(h3_gpu_tensor_read_bf16(oq1, a, count) &&
          h3_gpu_tensor_read_bf16(oq2, b, count));
    CHECK(!memcmp(a, b, count * 2));
    CHECK(h3_gpu_tensor_read_bf16(ok1, a, count) &&
          h3_gpu_tensor_read_bf16(ok2, b, count));
    CHECK(!memcmp(a, b, count * 2));

    /* SwiGLU: fused [gate | up] rows vs separate halves, in place on up. */
    const uint32_t width = 14336, mrows = 5;
    size_t half_count = (size_t)mrows * width;
    uint16_t *gate = random_bf16(half_count, 4.0f), *up = random_bf16(half_count, 4.0f);
    uint16_t *both = malloc(half_count * 2 * sizeof(*both));
    CHECK(both);
    for (uint32_t row = 0; row < mrows; row++) {
        memcpy(both + (size_t)row * width * 2, gate + (size_t)row * width,
               width * 2);
        memcpy(both + (size_t)row * width * 2 + width, up + (size_t)row * width,
               width * 2);
    }
    h3_gpu_tensor *tboth = h3_gpu_tensor_from_bf16(gpu, both, half_count * 2),
                  *tgate = h3_gpu_tensor_from_bf16(gpu, gate, half_count),
                  *tup = h3_gpu_tensor_from_bf16(gpu, up, half_count),
                  *tout = h3_gpu_tensor_new_bf16(gpu, half_count);
    CHECK(tboth && tgate && tup && tout);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_swiglu_bf16(gpu, tout, tboth, mrows, width));
    CHECK(h3_gpu_swiglu_split_bf16(gpu, tup, tgate, tup, (uint32_t)half_count));
    CHECK(h3_gpu_submit(gpu));
    uint16_t *s1 = malloc(half_count * 2), *s2 = malloc(half_count * 2);
    CHECK(s1 && s2 && h3_gpu_tensor_read_bf16(tout, s1, half_count) &&
          h3_gpu_tensor_read_bf16(tup, s2, half_count));
    CHECK(!memcmp(s1, s2, half_count * 2));

    /* up(down(x)) through h3_gpu_linear_bf16 at a released shape (rank
     * 768, 5376 -> 7168) and a small rank (256) on few rows. */
    const uint32_t ranks[2] = {768, 256};
    for (int r = 0; r < 2; r++) {
        const uint32_t lrows = 40, in = 5376, out = 7168, rank = ranks[r];
        uint16_t *x = random_bf16((size_t)lrows * in, 1.0f);
        uint16_t *down = random_bf16((size_t)rank * in, 0.02f);
        uint16_t *upw = random_bf16((size_t)out * rank, 0.05f);
        h3_gpu_tensor *tx = h3_gpu_tensor_from_bf16(gpu, x, (size_t)lrows * in),
                      *td = h3_gpu_tensor_from_bf16(gpu, down, (size_t)rank * in),
                      *tu = h3_gpu_tensor_from_bf16(gpu, upw, (size_t)out * rank),
                      *tz = h3_gpu_tensor_new_bf16(gpu, (size_t)lrows * rank),
                      *ty = h3_gpu_tensor_new_bf16(gpu, (size_t)lrows * out);
        CHECK(tx && td && tu && tz && ty);
        CHECK(h3_gpu_begin(gpu));
        CHECK(h3_gpu_linear_bf16(gpu, tz, tx, td, NULL, lrows, in, rank));
        CHECK(h3_gpu_linear_bf16(gpu, ty, tz, tu, NULL, lrows, rank, out));
        CHECK(h3_gpu_submit(gpu));
        uint16_t *z = malloc((size_t)lrows * rank * 2),
                 *y = malloc((size_t)lrows * out * 2);
        CHECK(z && y && h3_gpu_tensor_read_bf16(tz, z, (size_t)lrows * rank) &&
              h3_gpu_tensor_read_bf16(ty, y, (size_t)lrows * out));
        double err = 0.0, ref = 0.0;
        for (uint32_t row = 0; row < lrows; row += 13)
            for (uint32_t column = 0; column < out; column += 97) {
                double sum = 0.0;
                for (uint32_t j = 0; j < rank; j++)
                    sum += bf16_value(z[(size_t)row * rank + j]) *
                           bf16_value(upw[(size_t)column * rank + j]);
                double got = bf16_value(y[(size_t)row * out + column]);
                err += (got - sum) * (got - sum);
                ref += sum * sum;
            }
        for (uint32_t row = 0; row < lrows; row += 13)
            for (uint32_t j = 0; j < rank; j += 31) {
                double sum = 0.0;
                for (uint32_t c = 0; c < in; c++)
                    sum += bf16_value(x[(size_t)row * in + c]) *
                           bf16_value(down[(size_t)j * in + c]);
                double got = bf16_value(z[(size_t)row * rank + j]);
                err += (got - sum) * (got - sum);
                ref += sum * sum;
            }
        CHECK(sqrt(err / ref) < 5e-3);
        free(x); free(down); free(upw); free(z); free(y);
        h3_gpu_tensor_free(tx); h3_gpu_tensor_free(td); h3_gpu_tensor_free(tu);
        h3_gpu_tensor_free(tz); h3_gpu_tensor_free(ty);
    }

    free(q); free(k); free(v); free(qn); free(kn); free(cs); free(sn);
    free(fused); free(a); free(b); free(gate); free(up); free(both);
    free(s1); free(s2);
    h3_gpu_tensor *all[] = {tq, tk, tqkv, tqn, tkn, tcs, tsn, oq1, ok1, ov1,
                            oq2, ok2, tboth, tgate, tup, tout};
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++)
        h3_gpu_tensor_free(all[index]);
    h3_gpu_free(gpu);
    printf("split Q/K RoPE and SwiGLU kernels bit-identical; factored linear ok\n");
}

static void test_checkpoint(void) {
    const char *directory = getenv("H3_HYPERFLOW_DIR");
    if (!directory || !*directory) {
        skip("HyperFlow checkpoint (set H3_HYPERFLOW_DIR)");
        return;
    }
    char error[512];
    h3_weight_store *store = h3_weight_store_open(directory, error,
                                                  sizeof(error));
    if (!store) {
        fprintf(stderr, "%s\n", error);
        CHECK(store);
    }
    h3_hyperflow_info info;
    if (!h3_hyperflow_inspect(store, &info, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        CHECK(0);
    }
    CHECK(info.two_time && info.time_dim == 16 && info.gate == 0.25f);
    printf("HyperFlow: %u blocks, %u factored, max rank %u, %u sigmas\n",
           info.block_count, info.factored_blocks, info.max_rank,
           info.sigma_count);
    CHECK(info.block_count == 47 || info.block_count == 49);
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/model.gguf", directory);
        h3_gguf_info gguf;
        if (h3_gguf_inspect(path, &gguf, error, sizeof(error))) {
            CHECK(gguf.role == H3_GGUF_DIT && gguf.hyperflow &&
                  !gguf.dense_adaln && !gguf.curve_adaln);
            CHECK(gguf.dit_blocks == (int)info.block_count);
            CHECK(gguf.factored_blocks == (int)info.factored_blocks);
            printf("GGUF inspect: HyperFlow DiT, %d blocks, %d factored, %s\n",
                   gguf.dit_blocks, gguf.factored_blocks, gguf.main_type);
        }
    }
    CHECK(info.factored_blocks > 0 && info.max_rank > 0);
    CHECK(h3_hyperflow_block_factored(&info, 1, H3_HF_Q, H3_HF_FC2));
    CHECK(!h3_hyperflow_block_factored(&info, 0, H3_HF_Q, H3_HF_FC2));
    h3_sigma_schedule sigmas;
    CHECK(h3_hyperflow_schedule_build(6, &sigmas));
    CHECK(info.sigma_count == 7);
    for (int step = 0; step <= 6; step++)
        CHECK(fabsf(info.sigmas[step] - sigmas.video[step]) < 1e-6f);

    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    CHECK(gpu);
    h3_dit_schedule *schedule = h3_dit_schedule_precompute_two_time(
        store, gpu, &sigmas, 1, 1, info.gate, info.time_dim,
        info.block_count, NULL, NULL, error, sizeof(error));
    if (!schedule) fprintf(stderr, "%s\n", error);
    CHECK(schedule);
    CHECK(h3_dit_schedule_time_rows(schedule) == 14);
    CHECK(h3_dit_schedule_block(schedule, info.block_count - 1) != NULL);
    CHECK(h3_dit_schedule_block(schedule, info.block_count) == NULL);

    /* CPU reference for a few rows of block 0, the last block and the
     * final layer. */
    h3_dit_time_pair pairs[14];
    uint32_t video[6], audio[6], visual[6], audio_condition[6];
    CHECK(h3_dit_two_time_rows(&sigmas, 1, 1, pairs, 14, video, audio, visual,
                               audio_condition) == 14);
    const uint32_t check_rows[] = {0, 1, 7, 11, 12, 13};
    const unsigned check_blocks[] = {0, info.block_count - 1, 999};
    double worst = 0.0;
    for (size_t b = 0; b < 3; b++) {
        int final = check_blocks[b] == 999;
        uint32_t outputs = final ? 10752u : 96768u;
        char name[128];
        if (final)
            snprintf(name, sizeof(name), "final_layer.adaln_proj.linear");
        else
            snprintf(name, sizeof(name), "blocks.%u.adaln_proj.linear",
                     check_blocks[b]);
        char weight_name[160], bias_name[160];
        snprintf(weight_name, sizeof(weight_name), "%s.weight", name);
        snprintf(bias_name, sizeof(bias_name), "%s.bias", name);
        uint16_t *weight = read_tensor(store, weight_name, H3_DTYPE_BF16,
                                       (size_t)outputs * info.time_dim);
        uint16_t *bias = read_tensor(store, bias_name, H3_DTYPE_BF16, outputs);
        const h3_gpu_tensor *tensor = final ? h3_dit_schedule_final(schedule) :
            h3_dit_schedule_block(schedule, check_blocks[b]);
        size_t count = (size_t)14 * outputs;
        uint16_t *values = malloc(count * sizeof(*values));
        CHECK(values && h3_gpu_tensor_read_bf16(tensor, values, count));
        for (size_t r = 0; r < sizeof(check_rows) / sizeof(*check_rows); r++) {
            uint32_t row = check_rows[r];
            double et[64], er[64], coordinate[64];
            embed(store, "time_embedder", pairs[row].t, info.time_dim, et);
            embed(store, "endpoint_time_embedder", pairs[row].r,
                  info.time_dim, er);
            for (uint32_t d = 0; d < info.time_dim; d++) {
                double mixed = et[d] + 0.25 * (er[d] - et[d]);
                coordinate[d] = mixed / (1.0 + exp(-mixed));
            }
            double *expected = malloc(outputs * sizeof(*expected));
            CHECK(expected);
            double error_sum = 0.0, reference_sum = 0.0, max_error = 0.0;
            for (uint32_t column = 0; column < outputs; column++) {
                double sum = bf16_value(bias[column]);
                for (uint32_t d = 0; d < info.time_dim; d++)
                    sum += bf16_value(weight[(size_t)column * info.time_dim + d]) *
                           coordinate[d];
                expected[column] = sum;
                reference_sum += sum * sum;
            }
            double rms = sqrt(reference_sum / outputs);
            for (uint32_t column = 0; column < outputs; column++) {
                double got = bf16_value(values[(size_t)row * outputs + column]);
                double difference = fabs(got - expected[column]);
                error_sum += difference * difference;
                /* BF16 rounding of the result plus F32 accumulation, which
                 * matters only next to cancellation (values << rms). */
                double allowed = fabs(expected[column]) / 256.0 + rms * 1e-4;
                if (difference / allowed > max_error)
                    max_error = difference / allowed;
            }
            free(expected);
            double relative = sqrt(error_sum / (reference_sum + 1e-30));
            if (relative > worst) worst = relative;
            if (relative >= 4e-3 || max_error > 1.0)
                fprintf(stderr, "%s row %u: rel-L2 %.3e, worst/allowed %.2f\n",
                        name, row, relative, max_error);
            CHECK(relative < 4e-3);
            CHECK(max_error <= 1.0);
        }
        free(weight); free(bias); free(values);
    }
    printf("two-time AdaLN vs CPU reference: worst rel-L2 %.2e\n", worst);
    h3_dit_schedule_host *host = h3_dit_schedule_export(schedule);
    CHECK(host);
    h3_dit_schedule *copy = h3_dit_schedule_import(host, gpu, error,
                                                   sizeof(error));
    CHECK(copy && h3_dit_schedule_block(copy, info.block_count) == NULL &&
          h3_dit_schedule_block(copy, info.block_count - 1) != NULL);
    h3_dit_schedule_free(copy);
    h3_dit_schedule_host_free(host);
    h3_dit_schedule_free(schedule);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
}

int main(void) {
    test_grids();
    test_rows();
    test_kernels();
    test_checkpoint();
    printf("h3 HyperFlow tests passed (%d checks)\n", tests_run);
    return 0;
}
