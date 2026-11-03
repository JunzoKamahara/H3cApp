/* Dense SDPA execution modes (issue #2).
 *
 *   ./h3_sdpa_tests                 regression: every mode against a CPU
 *                                   reference, row- and head-major layouts,
 *                                   tails, f32 and bf16; exits 1 on failure
 *   ./h3_sdpa_tests --probe-gqa S   Qwen's causal GQA (64/8 heads, D 128)
 *                                   at S tokens through the route long
 *                                   reference prompts take: chunked,
 *                                   explicit, then one call, with memory
 *   ./h3_sdpa_tests --probe S H D   the shape a generation really sends
 *                                   (e.g. 17408 56 128): logs the device and
 *                                   call, times chunked, explicit and, last,
 *                                   the one-call path, and reports memory.
 *                                   On a machine that crashes in the
 *                                   one-call path, the chunked lines print
 *                                   first and the crash comes at the end.
 *
 * The causal GQA of the text encoder is checked the same way: its kernel
 * and the MPSGraph route it takes past the kernel's threadgroup ceiling,
 * one call and in pieces.
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

/* Causal GQA (the Qwen text encoder), row-major q [S, QH, D] and k/v
 * [S, KH, D]: query head h reads KV head h / (QH / KH), row i keys 0..i. */
static void reference_gqa(const float *q, const float *k, const float *v,
                          uint32_t S, uint32_t QH, uint32_t KH, uint32_t D,
                          float scale, float *out) {
    float *scores = malloc((size_t)S * sizeof(*scores));
    uint32_t groups = QH / KH;
    for (uint32_t h = 0; h < QH; h++) {
        uint32_t g = h / groups;
        for (uint32_t i = 0; i < S; i++) {
            const float *qi = q + ((size_t)i * QH + h) * D;
            float max_score = -INFINITY;
            for (uint32_t j = 0; j <= i; j++) {
                const float *kj = k + ((size_t)j * KH + g) * D;
                float dot = 0.0f;
                for (uint32_t d = 0; d < D; d++) dot += qi[d] * kj[d];
                scores[j] = dot * scale;
                if (scores[j] > max_score) max_score = scores[j];
            }
            double sum = 0.0;
            for (uint32_t j = 0; j <= i; j++) {
                scores[j] = expf(scores[j] - max_score);
                sum += scores[j];
            }
            float *oi = out + ((size_t)i * QH + h) * D;
            for (uint32_t d = 0; d < D; d++) oi[d] = 0.0f;
            for (uint32_t j = 0; j <= i; j++) {
                float p = (float)(scores[j] / sum);
                const float *vj = v + ((size_t)j * KH + g) * D;
                for (uint32_t d = 0; d < D; d++) oi[d] += p * vj[d];
            }
        }
    }
    free(scores);
}

/* The long-sequence route of h3_gpu_gqa_causal_bf16 in every mode, and its
 * custom kernel, against the CPU reference. */
static void run_gqa(h3_gpu *gpu, uint32_t S, uint32_t QH, uint32_t KH,
                    uint32_t D) {
    size_t q_count = (size_t)S * QH * D, kv_count = (size_t)S * KH * D;
    float scale = 1.0f / sqrtf((float)D);
    float *q = malloc(q_count * sizeof(float)), *k = malloc(kv_count * sizeof(float));
    float *v = malloc(kv_count * sizeof(float)), *ref = malloc(q_count * sizeof(float));
    uint16_t *qb = malloc(q_count * 2), *kb = malloc(kv_count * 2), *vb = malloc(kv_count * 2);
    for (size_t i = 0; i < q_count; i++) {
        qb[i] = f32_to_bf16(rng_normal() * 0.7f); q[i] = bf16_to_f32(qb[i]);
    }
    for (size_t i = 0; i < kv_count; i++) {
        kb[i] = f32_to_bf16(rng_normal() * 0.7f); k[i] = bf16_to_f32(kb[i]);
        vb[i] = f32_to_bf16(rng_normal()); v[i] = bf16_to_f32(vb[i]);
    }
    reference_gqa(q, k, v, S, QH, KH, D, scale, ref);
    uint64_t one_row = (uint64_t)(QH / KH) * S * 2;
    mode_case cases[] = {
        {"kernel", H3_GPU_SDPA_AUTO, 0},
        {"graph full", H3_GPU_SDPA_FULL, 0},
        {"graph chunked, all rows", H3_GPU_SDPA_CHUNKED, 0},
        {"graph chunked, row blocks", H3_GPU_SDPA_CHUNKED, one_row * (S / 3 + 1)},
        {"graph chunked, tiny blocks", H3_GPU_SDPA_CHUNKED, one_row * 7},
        {"graph explicit, row blocks", H3_GPU_SDPA_EXPLICIT, one_row * 4 * (S / 3 + 1)},
    };
    float *kernel_rows = NULL;
    for (size_t c = 0; c < sizeof(cases) / sizeof(*cases); c++) {
        int kernel = c == 0;
        h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, qb, q_count);
        h3_gpu_tensor *tk = h3_gpu_tensor_from_bf16(gpu, kb, kv_count);
        h3_gpu_tensor *tv = h3_gpu_tensor_from_bf16(gpu, vb, kv_count);
        h3_gpu_tensor *to = h3_gpu_tensor_new_bf16(gpu, q_count);
        h3_gpu_set_sdpa_mode(gpu, cases[c].mode, cases[c].budget);
        uint32_t plan_rows = S;
        h3_gpu_gqa_chunk_plan(gpu, S, QH, KH, &plan_rows);
        h3_gpu_stats before; h3_gpu_get_stats(gpu, &before);
        int ok = h3_gpu_begin(gpu);
        if (ok) ok = kernel ?
            h3_gpu_gqa_causal_bf16(gpu, to, tq, tk, tv, S, QH, KH, D, scale) :
            h3_gpu_gqa_causal_graph_bf16(gpu, to, tq, tk, tv, S, QH, KH, D, scale);
        if (ok) ok = h3_gpu_submit(gpu);
        CHECK(ok, "GQA %s S=%u: %s", cases[c].name, S, h3_gpu_error(gpu));
        if (ok) {
            h3_gpu_stats after; h3_gpu_get_stats(gpu, &after);
            uint64_t pieces = after.mps_sdpa_chunk_dispatches -
                              before.mps_sdpa_chunk_dispatches;
            uint64_t expected = kernel || cases[c].mode == H3_GPU_SDPA_FULL ? 0 :
                (uint64_t)KH * ((S + plan_rows - 1) / plan_rows);
            CHECK(pieces == expected, "GQA %s: %llu pieces, expected %llu",
                  cases[c].name, (unsigned long long)pieces,
                  (unsigned long long)expected);
            float *rows = read_output(gpu, to, q_count, 1, 0, S, QH, D);
            error_stats e = compare(rows, ref, q_count);
            printf("  GQA %-27s S=%5u QH=%2u KH=%u D=%3u pieces %3llu "
                   "(%u rows)  max %.5f mean %.6f non-finite %zu\n",
                   cases[c].name, S, QH, KH, D, (unsigned long long)pieces,
                   plan_rows, e.max_abs, e.mean_abs, e.non_finite);
            CHECK(e.non_finite == 0, "GQA %s: non-finite outputs", cases[c].name);
            CHECK(e.max_abs <= 0.03 && e.mean_abs <= 0.003,
                  "GQA %s: error vs reference too large (max %.5f mean %.6f)",
                  cases[c].name, e.max_abs, e.mean_abs);
            if (kernel) kernel_rows = rows;
            else {
                error_stats d = compare(rows, kernel_rows, q_count);
                CHECK(d.max_abs <= 0.02,
                      "GQA %s: differs from the kernel (max %.5f)",
                      cases[c].name, d.max_abs);
                free(rows);
            }
        }
        h3_gpu_tensor_free(tq); h3_gpu_tensor_free(tk);
        h3_gpu_tensor_free(tv); h3_gpu_tensor_free(to);
    }
    h3_gpu_set_sdpa_mode(gpu, H3_GPU_SDPA_AUTO, 0);
    free(kernel_rows);
    free(q); free(k); free(v); free(ref); free(qb); free(kb); free(vb);
}

/* The device-memory GQA kernels (h3_gqa_causal_group_bf16 for eight query
 * heads per KV head, h3_gqa_causal_rows_bf16 otherwise) must give the
 * threadgroup-memory kernel's output bit for bit: generations made before
 * them stay reproducible. H3_GQA_ROWS picks the path for each run. */
static void run_gqa_exact(h3_gpu *gpu, uint32_t S, uint32_t QH, uint32_t KH,
                          uint32_t D) {
    size_t q_count = (size_t)S * QH * D, kv_count = (size_t)S * KH * D;
    float scale = 1.0f / sqrtf((float)D);
    uint16_t *qb = malloc(q_count * 2), *kb = malloc(kv_count * 2), *vb = malloc(kv_count * 2);
    for (size_t i = 0; i < q_count; i++) qb[i] = f32_to_bf16(rng_normal() * 1.5f);
    for (size_t i = 0; i < kv_count; i++) {
        kb[i] = f32_to_bf16(rng_normal() * 1.5f);
        vb[i] = f32_to_bf16(rng_normal());
    }
    h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, qb, q_count);
    h3_gpu_tensor *tk = h3_gpu_tensor_from_bf16(gpu, kb, kv_count);
    h3_gpu_tensor *tv = h3_gpu_tensor_from_bf16(gpu, vb, kv_count);
    h3_gpu_tensor *to = h3_gpu_tensor_new_bf16(gpu, q_count);
    const char *paths[][3] = {
        {"threadgroup kernel", "0", "1"},
        {"device rows, grouped", "1", "1"},
        {"device rows, per head", "1", "0"},
    };
    uint16_t *first = malloc(q_count * 2), *rows = malloc(q_count * 2);
    for (size_t p = 0; p < sizeof(paths) / sizeof(*paths); p++) {
        setenv("H3_GQA_ROWS", paths[p][1], 1);
        setenv("H3_GQA_GROUP", paths[p][2], 1);
        int ok = h3_gpu_begin(gpu) &&
                 h3_gpu_gqa_causal_bf16(gpu, to, tq, tk, tv, S, QH, KH, D, scale) &&
                 h3_gpu_submit(gpu) &&
                 h3_gpu_tensor_read_bf16(to, p ? rows : first, q_count);
        CHECK(ok, "GQA %s S=%u: %s", paths[p][0], S, h3_gpu_error(gpu));
        if (!ok || !p) continue;
        size_t differing = 0;
        for (size_t i = 0; i < q_count; i++) differing += rows[i] != first[i];
        printf("  GQA bit-exact %-22s S=%5u QH=%2u KH=%u D=%3u  %zu of %zu "
               "values differ\n", paths[p][0], S, QH, KH, D, differing, q_count);
        CHECK(differing == 0, "GQA %s S=%u: %zu values differ from the "
              "threadgroup kernel", paths[p][0], S, differing);
    }
    unsetenv("H3_GQA_ROWS");
    unsetenv("H3_GQA_GROUP");
    free(first); free(rows); free(qb); free(kb); free(vb);
    h3_gpu_tensor_free(tq); h3_gpu_tensor_free(tk);
    h3_gpu_tensor_free(tv); h3_gpu_tensor_free(to);
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

/* Qwen's causal GQA at a real length through its long-sequence route:
 * chunked, explicit, then the one-call graph last, as probe() does. */
static void probe_gqa(h3_gpu *gpu, uint32_t S, uint32_t QH, uint32_t KH,
                      uint32_t D, int reps) {
    size_t q_count = (size_t)S * QH * D, kv_count = (size_t)S * KH * D;
    float scale = 1.0f / sqrtf((float)D);
    uint16_t *data = malloc(q_count * 2);
    for (size_t i = 0; i < q_count; i++) data[i] = f32_to_bf16(rng_normal() * 0.7f);
    h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, data, q_count);
    for (size_t i = 0; i < kv_count; i++) data[i] = f32_to_bf16(rng_normal() * 0.7f);
    h3_gpu_tensor *tk = h3_gpu_tensor_from_bf16(gpu, data, kv_count);
    for (size_t i = 0; i < kv_count; i++) data[i] = f32_to_bf16(rng_normal());
    h3_gpu_tensor *tv = h3_gpu_tensor_from_bf16(gpu, data, kv_count);
    free(data);
    h3_gpu_tensor *to = h3_gpu_tensor_new_bf16(gpu, q_count);
    if (!tq || !tk || !tv || !to) { fprintf(stderr, "allocation failed\n"); exit(1); }
    setenv("H3_SDPA_LOG", "1", 0);
    printf("probe-gqa: S=%u QH=%u KH=%u D=%u causal bf16; auto mode here = %s; "
           "footprint before %.2f GiB\n", S, QH, KH, D,
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
            printf("probe-gqa: running the one-call path last\n");
        fflush(stdout);
        double best = INFINITY;
        for (int r = 0; r < reps; r++) {
            double start = now_seconds();
            int ok = h3_gpu_begin(gpu) &&
                     h3_gpu_gqa_causal_graph_bf16(gpu, to, tq, tk, tv, S, QH,
                                                  KH, D, scale) &&
                     h3_gpu_submit(gpu);
            double elapsed = now_seconds() - start;
            if (!ok) {
                printf("probe-gqa: %s FAILED: %s\n", cases[c].name, h3_gpu_error(gpu));
                break;
            }
            if (elapsed < best) best = elapsed;
        }
        h3_gpu_stats stats; h3_gpu_get_stats(gpu, &stats);
        printf("probe-gqa: %-8s best %.3f s/call  footprint %.2f GiB  "
               "peak live %.2f GiB  chunk pieces so far %llu\n", cases[c].name,
               best, footprint_gib(),
               (double)stats.peak_live_bytes / (1024.0 * 1024.0 * 1024.0),
               (unsigned long long)stats.mps_sdpa_chunk_dispatches);
        float *rows = read_output(gpu, to, q_count, 1, 0, S, QH, D);
        if (!first) first = rows;
        else {
            error_stats d = compare(rows, first, q_count);
            printf("probe-gqa: %-8s vs chunked: max %.5f mean %.6f non-finite %zu\n",
                   cases[c].name, d.max_abs, d.mean_abs, d.non_finite);
            free(rows);
        }
        fflush(stdout);
    }
    free(first);
    h3_gpu_tensor_free(tq); h3_gpu_tensor_free(tk);
    h3_gpu_tensor_free(tv); h3_gpu_tensor_free(to);
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
    if (argc >= 3 && !strcmp(argv[1], "--probe-gqa")) {
        probe_gqa(gpu, (uint32_t)atoi(argv[2]),
                  argc > 3 ? (uint32_t)atoi(argv[3]) : 64,
                  argc > 4 ? (uint32_t)atoi(argv[4]) : 8,
                  argc > 5 ? (uint32_t)atoi(argv[5]) : 128,
                  argc > 6 ? atoi(argv[6]) : 3);
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
    /* Qwen's head layout (8 query heads per KV head) with fewer heads, and
     * a tail row block. */
    run_gqa(gpu, 301, 16, 2, 128);
    run_gqa(gpu, 64, 8, 1, 64);
    /* Qwen's own shape (long enough for several score-row batches only at
     * real lengths; --probe-gqa covers those), a tail, and a group size
     * other than eight, which takes the per-head kernel. */
    run_gqa_exact(gpu, 1500, 64, 8, 128);
    run_gqa_exact(gpu, 333, 16, 2, 128);
    run_gqa_exact(gpu, 257, 12, 3, 64);
    h3_gpu_free(gpu);
    if (failures) { fprintf(stderr, "FAIL tests/test_sdpa.c: %d failures\n", failures); return 1; }
    printf("ok: dense SDPA modes (full, chunked, explicit) and causal GQA match the CPU reference\n");
    return 0;
}
