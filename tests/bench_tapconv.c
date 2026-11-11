/* Speed of the TensorOps tap convolution (h3_gpu_vsr_tapconv_bf16) on the
 * SeedVR2 video VAE's shapes: one output frame of the decoder's blocks at
 * 2x of 672x384, 9 taps (image) and 27 taps (video: the taps reach two
 * frames back, so the input holds three). H3_TAPCONV_KERNEL / H3_TAPCONV_ORDER
 * pick the kernel variant and the dispatch order (h3_gpu.m). Development
 * only; skips without TensorOps.
 *
 *   ./h3_bench_tapconv [frames] */
#include "h3_gpu.h"
#include "h3_gpu_vsr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    int out_frames = argc > 1 ? atoi(argv[1]) : 1;
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "%s\n", error); return 1; }
    if (!h3_gpu_has_tensor_ops(gpu)) { printf("skip: no TensorOps\n"); return 0; }
    const char *kernel = getenv("H3_TAPCONV_KERNEL"), *order = getenv("H3_TAPCONV_ORDER");
    printf("kernel %s, order %s, %d output frame(s)\n", kernel ? kernel : "default", order ? order : "default (1)", out_frames);
    struct { int h, w, c; } shapes[] = {{768, 1344, 128}, {384, 672, 256}, {192, 336, 512}, {96, 168, 512}};
    for (size_t s = 0; s < sizeof(shapes) / sizeof(*shapes); s++) {
        int H = shapes[s].h, W = shapes[s].w, C = shapes[s].c;
        uint32_t wp = (uint32_t)W + 2, frame_rows = (uint32_t)(H + 2) * wp;
        uint32_t guard = (wp + 3 + 127) / 128 * 128, tail = guard;
        for (int taps = 9; taps <= 27; taps += 18) {
            int in_frames = out_frames + (taps == 27 ? 2 : 0);
            uint32_t rows = (uint32_t)out_frames * frame_rows;
            uint32_t x_rows = guard + (uint32_t)in_frames * frame_rows + tail + 256;
            uint32_t base = guard + (taps == 27 ? 2 * frame_rows : 0);   /* the first output row */
            int32_t offsets[27];
            int k = 0;
            for (int tt = 0; tt < (taps == 27 ? 3 : 1); tt++)
                for (int dy = 0; dy < 3; dy++)
                    for (int dx = 0; dx < 3; dx++)
                        offsets[k++] = (taps == 27 ? (tt - 2) * (int32_t)frame_rows : 0) + (dy - 1) * (int32_t)wp + (dx - 1);
            h3_gpu_tensor *x = h3_gpu_tensor_new_bf16(gpu, (size_t)x_rows * C), *y = h3_gpu_tensor_new_bf16(gpu, (size_t)x_rows * C);
            h3_gpu_tensor *w = h3_gpu_tensor_new_bf16(gpu, (size_t)taps * C * C), *b = h3_gpu_tensor_new_f32(gpu, C);
            if (!x || !y || !w || !b) { fprintf(stderr, "out of memory at %dx%d\n", W, H); return 1; }
            int reps = C >= 512 ? 8 : 4;
            double best = 1e9;
            for (int pass = 0; pass < 3; pass++) {
                double t0 = now();
                if (!h3_gpu_begin(gpu)) { fprintf(stderr, "%s\n", h3_gpu_error(gpu)); return 1; }
                for (int i = 0; i < reps; i++)
                    if (!h3_gpu_vsr_tapconv_bf16(gpu, y, x, NULL, w, NULL, b, NULL, rows, x_rows, base, (uint32_t)C, (uint32_t)C,
                                                 (uint32_t)taps, offsets, 0)) { fprintf(stderr, "%s\n", h3_gpu_error(gpu)); return 1; }
                if (!h3_gpu_submit(gpu)) { fprintf(stderr, "%s\n", h3_gpu_error(gpu)); return 1; }
                double secs = (now() - t0) / reps;
                if (pass && secs < best) best = secs;
            }
            double flops = 2.0 * (double)rows * taps * (double)C * C;
            printf("  %4dx%-4d %3d ch %2d taps: %7.2f ms  %5.1f TFLOP/s  (%.0f GB/s of input if every tap re-reads it)\n",
                   W, H, C, taps, best * 1000.0, flops / best / 1e12, (double)rows * C * 2 * taps * (C / 64) / best / 1e9);
            h3_gpu_tensor_free(x); h3_gpu_tensor_free(y); h3_gpu_tensor_free(w); h3_gpu_tensor_free(b);
        }
    }
    h3_gpu_free(gpu);
    return 0;
}
