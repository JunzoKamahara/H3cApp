/* Micro-benchmark of h3_gpu_linear_int8_bf16 at the SwiftVR DiT's shapes
 * (and probes around them), to see why FC2 (K = 14336) runs at half the
 * efficiency of the others. Not part of `make test`.
 *   ./h3_int8_bench [rows]      (default 14280 = the 1080p FIRST chunk) */
#include "h3_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9; }

static void bench(h3_gpu *gpu, const char *label, uint32_t rows, uint32_t K, uint32_t N, int calls, int splits) {
    uint32_t padded = (rows + 127u) & ~127u;
    size_t in_n = (size_t)padded * K, w_n = (size_t)N * K;
    uint16_t *in = malloc(in_n * sizeof(uint16_t));
    int8_t *w = malloc(w_n);
    float *sc = malloc(N * sizeof(float));
    for (size_t i = 0; i < in_n; i++) in[i] = (uint16_t)(0x3f80 + (rand() & 0x7f));   /* ~1..2 */
    for (size_t i = 0; i < w_n; i++) w[i] = (int8_t)(rand() % 127 - 63);
    for (uint32_t i = 0; i < N; i++) sc[i] = 0.01f;
    h3_gpu_tensor *in_t = h3_gpu_tensor_from_bf16(gpu, in, in_n);
    h3_gpu_tensor *w_t = h3_gpu_tensor_from_i8(gpu, w, w_n);
    h3_gpu_tensor *sc_t = h3_gpu_tensor_from_f32(gpu, sc, N);
    h3_gpu_tensor *out_t = h3_gpu_tensor_new_bf16(gpu, (size_t)padded * N);
    h3_gpu_tensor *q_in = h3_gpu_tensor_new_i8(gpu, (size_t)padded * K);
    h3_gpu_tensor *q_sc = h3_gpu_tensor_new_f32(gpu, padded);
    if (!in_t || !w_t || !sc_t || !out_t || !q_in || !q_sc) { fprintf(stderr, "alloc failed\n"); exit(1); }
    h3_gpu_tensor *partial = splits > 1 ? h3_gpu_tensor_new_f32(gpu, (size_t)splits * padded * N) : NULL;
    if (splits < 0) partial = h3_gpu_tensor_new_f32(gpu, (size_t)padded * N);
    double best = 1e9;
    for (int pass = 0; pass < 3; pass++) {
        double t0 = now();
        if (!h3_gpu_begin(gpu)) exit(1);
        for (int i = 0; i < calls; i++) {
            int ok = splits > 1
                ? h3_gpu_linear_int8_bf16_split_k(gpu, out_t, partial, q_in, q_sc, in_t, w_t, sc_t, rows, K, N, (uint32_t)splits)
                : splits < 0
                ? (h3_gpu_linear_int8_partial_f32(gpu, partial, 0, q_in, q_sc, in_t, w_t, sc_t, rows, K, N) &&
                   h3_gpu_sum_partials_bf16(gpu, out_t, partial, rows, N, 1))
                : h3_gpu_linear_int8_bf16(gpu, out_t, q_in, q_sc, in_t, w_t, sc_t, rows, K, N, 0);
            if (!ok) { fprintf(stderr, "%s: %s\n", label, h3_gpu_error(gpu)); exit(1); }
        }
        if (!h3_gpu_submit(gpu)) exit(1);
        double secs = (now() - t0) / calls;
        if (secs < best) best = secs;
    }
    double flops = 2.0 * rows * (double)K * N;
    printf("  %-34s rows %5u K %5u N %5u: %7.2f ms  %5.1f TFLOP/s  (%u x %u groups)\n", label, rows, K, N,
           best * 1000.0, flops / best / 1e12, padded / 128, N / 128);
    free(in); free(w); free(sc);
    h3_gpu_tensor_free(in_t); h3_gpu_tensor_free(w_t); h3_gpu_tensor_free(sc_t); h3_gpu_tensor_free(out_t);
    h3_gpu_tensor_free(q_in); h3_gpu_tensor_free(q_sc); h3_gpu_tensor_free(partial);
}

static void bench_quant(h3_gpu *gpu, uint32_t rows, uint32_t K) {
    uint32_t padded = (rows + 127u) & ~127u;
    size_t in_n = (size_t)padded * K;
    uint16_t *in = malloc(in_n * sizeof(uint16_t));
    for (size_t i = 0; i < in_n; i++) in[i] = (uint16_t)(0x3f80 + (rand() & 0x7f));
    h3_gpu_tensor *in_t = h3_gpu_tensor_from_bf16(gpu, in, in_n);
    h3_gpu_tensor *q_in = h3_gpu_tensor_new_i8(gpu, in_n);
    h3_gpu_tensor *q_sc = h3_gpu_tensor_new_f32(gpu, padded);
    double best = 1e9;
    for (int pass = 0; pass < 3; pass++) {
        double t0 = now();
        if (!h3_gpu_begin(gpu)) exit(1);
        for (int i = 0; i < 4; i++)
            if (!h3_gpu_quantize_rows_bf16_int8(gpu, q_in, q_sc, in_t, rows, K)) { fprintf(stderr, "quant: %s\n", h3_gpu_error(gpu)); exit(1); }
        if (!h3_gpu_submit(gpu)) exit(1);
        double secs = (now() - t0) / 4;
        if (secs < best) best = secs;
    }
    printf("  %-34s rows %5u K %5u         : %7.2f ms  %5.0f GB/s read\n", "row quantizer", rows, K, best * 1000.0,
           (double)rows * K * 2 / best / 1e9);
    free(in); h3_gpu_tensor_free(in_t); h3_gpu_tensor_free(q_in); h3_gpu_tensor_free(q_sc);
}

int main(int argc, char **argv) {
    uint32_t rows = argc > 1 ? (uint32_t)atoi(argv[1]) : 14280;
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "%s\n", error); return 1; }
    if (!h3_gpu_has_int8_mlp(gpu)) { printf("skip: no TensorOps\n"); return 0; }
    srand(3);
    if (getenv("H3_BENCH_K_SWEEP")) {
        uint32_t ks[] = {4096, 6144, 8192, 10240, 12288, 13312, 14336, 14464, 14592, 15360, 16384, 20480};
        for (size_t i = 0; i < sizeof(ks) / sizeof(*ks); i++) {
            char label[64];
            snprintf(label, sizeof(label), "K sweep -> 3072");
            bench(gpu, label, rows, ks[i], 3072, 3, 1);
        }
        return 0;
    }
    bench_quant(gpu, rows, 3072);
    bench_quant(gpu, rows, 14336);
    bench(gpu, "qkv (3072 -> 9216)", rows, 3072, 9216, 4, 1);
    bench(gpu, "out (3072 -> 3072)", rows, 3072, 3072, 4, 1);
    bench(gpu, "fc1 (3072 -> 14336)", rows, 3072, 14336, 3, 1);
    bench(gpu, "fc2 (14336 -> 3072)", rows, 14336, 3072, 3, 1);
    bench(gpu, "fc2 split-K 2", rows, 14336, 3072, 3, 2);
    bench(gpu, "fc2 split-K 4", rows, 14336, 3072, 3, 4);
    bench(gpu, "fc2 split-K 7", rows, 14336, 3072, 3, 7);
    bench(gpu, "fc1 split-K 2", rows, 3072, 14336, 3, 2);
    bench(gpu, "fc2 half K (7168 -> 3072)", rows, 7168, 3072, 3, 1);
    bench(gpu, "fc2 half K via partial+sum", rows, 7168, 3072, 3, -1);
    bench(gpu, "fc1 half N (3072 -> 7168)", rows, 3072, 7168, 3, 1);
    bench(gpu, "fc2 quarter K (3584 -> 3072)", rows, 3584, 3072, 3, 1);
    bench(gpu, "fc2 wide N (14336 -> 6144)", rows, 14336, 6144, 3, 1);
    bench(gpu, "fc2 wide N (14336 -> 12288)", rows, 14336, 12288, 2, 1);
    bench(gpu, "K 7168 -> 5376 (H3 shape)", rows, 7168, 5376, 3, 1);
    bench(gpu, "fc2, 2048 rows", 2048, 14336, 3072, 4, 1);
    bench(gpu, "fc1, 2048 rows", 2048, 3072, 14336, 4, 1);
    return 0;
}
