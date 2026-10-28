/* The ReAE's TensorOps 3x3 convolution (h3_gpu_vsr_tapconv_bf16 on the
 * padded grid) against the F32 MPSGraph path and a double reference, plus
 * its speed on a decoder-sized MemBlock shape. Skips without TensorOps. */
#include "h3_vsr_internal.h"
#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int checks = 0;
#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); exit(1); } checks++; } while (0)

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9; }
static float rnd(void) { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

static uint16_t f32_to_bf16(float f) { union { uint32_t u; float f; } c; c.f = f; uint32_t u = c.u + 0x7fffu + ((c.u >> 16) & 1u); return (uint16_t)(u >> 16); }

/* conv weight [out, in_total, 3, 3] (the in_offset .. in_offset+in part) ->
 * the tap-major BF16 [9 * out_p, in_p] the kernel wants. */
static uint16_t *pack_taps(const float *w, int out, int in_total, int in_offset, int in, int out_p, int in_p) {
    uint16_t *packed = calloc((size_t)9 * out_p * in_p, sizeof(uint16_t));
    for (int t = 0; t < 9; t++)
        for (int o = 0; o < out; o++)
            for (int i = 0; i < in; i++)
                packed[((size_t)t * out_p + o) * in_p + i] =
                    f32_to_bf16(w[(((size_t)o * in_total + in_offset + i) * 9) + t]);
    return packed;
}

static double psnr(const float *a, const float *b, size_t n) {
    double err = 0.0, peak = 0.0;
    for (size_t i = 0; i < n; i++) { double d = a[i] - b[i]; err += d * d; if (fabs(b[i]) > peak) peak = fabs(b[i]); }
    return 10.0 * log10(peak * peak / (err / (double)n > 0 ? err / (double)n : 1e-30));
}

/* double reference: y = relu?(conv3x3(x, w) + conv3x3(past, w2) + b + res) */
static void reference(const float *x, const float *past, const float *w, const float *w2, const float *bias,
                      const float *res, int T, int H, int W, int C, int O, int relu, float *y) {
    for (int t = 0; t < T; t++)
        for (int yy = 0; yy < H; yy++)
            for (int xx = 0; xx < W; xx++)
                for (int o = 0; o < O; o++) {
                    double acc = bias ? bias[o] : 0.0;
                    for (int dy = 0; dy < 3; dy++)
                        for (int dx = 0; dx < 3; dx++) {
                            int sy = yy + dy - 1, sx = xx + dx - 1;
                            if (sy < 0 || sy >= H || sx < 0 || sx >= W) continue;
                            const float *px = x + (((size_t)t * H + sy) * W + sx) * C;
                            const float *pp = past ? past + (((size_t)t * H + sy) * W + sx) * C : NULL;
                            int in_total = past ? 2 * C : C;
                            for (int i = 0; i < C; i++) {
                                acc += (double)px[i] * w[(((size_t)o * in_total + i) * 3 + dy) * 3 + dx];
                                if (pp) acc += (double)pp[i] * w2[(((size_t)o * in_total + C + i) * 3 + dy) * 3 + dx];
                            }
                        }
                    size_t at = (((size_t)t * H + yy) * W + xx) * O + o;
                    if (res) acc += res[at];
                    if (relu && acc < 0) acc = 0;
                    y[at] = (float)acc;
                }
}

typedef struct { int T, H, W, C, Cp, O, Op; uint32_t rows, guard, x_rows; int32_t offsets[9]; } grid;

static grid make_grid(int T, int H, int W, int C, int O) {
    grid g = {T, H, W, C, (C + 63) / 64 * 64, O, (O + 63) / 64 * 64, 0, 0, 0, {0}};
    uint32_t wp = (uint32_t)W + 2;
    g.rows = (uint32_t)T * ((uint32_t)H + 2) * wp;
    g.guard = (wp + 3 + 127) / 128 * 128;
    g.x_rows = g.guard + g.rows + g.guard + 128;
    for (int dy = 0; dy < 3; dy++) for (int dx = 0; dx < 3; dx++) g.offsets[dy * 3 + dx] = (dy - 1) * (int)wp + (dx - 1);
    return g;
}

int main(void) {
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    CHECK(gpu != NULL);
    if (!h3_gpu_has_tensor_ops(gpu)) { printf("skip: no TensorOps on this GPU\n"); return 0; }
    srand(7);

    /* 1. plain conv (bias, no ReLU) vs double and vs the F32 MPSGraph path */
    {
        const int T = 2, H = 20, W = 24, C = 64, O = 64;
        grid g = make_grid(T, H, W, C, O);
        size_t xn = (size_t)T * H * W * C, yn = (size_t)T * H * W * O, wn = (size_t)O * C * 9;
        float *x = malloc(xn * sizeof(float)), *w = malloc(wn * sizeof(float)), *b = malloc(O * sizeof(float));
        for (size_t i = 0; i < xn; i++) x[i] = rnd();
        for (size_t i = 0; i < wn; i++) w[i] = rnd() * 0.1f;
        for (int i = 0; i < O; i++) b[i] = rnd();
        float *ref = malloc(yn * sizeof(float)), *got = malloc(yn * sizeof(float)), *mps = malloc(yn * sizeof(float));
        reference(x, NULL, w, NULL, b, NULL, T, H, W, C, O, 0, ref);
        uint16_t *wp = pack_taps(w, O, C, 0, C, g.Op, g.Cp);
        h3_gpu_tensor *x_t = h3_gpu_tensor_from_f32(gpu, x, xn);
        h3_gpu_tensor *xg = h3_gpu_tensor_new_bf16(gpu, (size_t)g.x_rows * g.Cp);
        h3_gpu_tensor *yg = h3_gpu_tensor_new_bf16(gpu, (size_t)g.x_rows * g.Op);
        h3_gpu_tensor *w_t = h3_gpu_tensor_from_bf16(gpu, wp, (size_t)9 * g.Op * g.Cp);
        h3_gpu_tensor *b_t = h3_gpu_tensor_from_f32(gpu, b, O);
        h3_gpu_tensor *y_t = h3_gpu_tensor_new_f32(gpu, yn);
        CHECK(x_t && xg && yg && w_t && b_t && y_t);
        float *bp = calloc(g.Op, sizeof(float)); memcpy(bp, b, O * sizeof(float));
        h3_gpu_tensor *bp_t = h3_gpu_tensor_from_f32(gpu, bp, g.Op);
        CHECK(h3_gpu_begin(gpu));
        CHECK(h3_gpu_vsr_pack_grid_bf16(gpu, xg, x_t, T, H, W, C, g.Cp, g.guard, g.x_rows, 1));
        CHECK(h3_gpu_vsr_tapconv_bf16(gpu, yg, xg, NULL, w_t, NULL, bp_t, NULL, g.rows, g.x_rows, g.guard, g.Cp, g.Op, 9, g.offsets, 0));
        CHECK(h3_gpu_vsr_unpack_grid_f32(gpu, y_t, yg, T, H, W, O, g.Op, g.guard, g.x_rows));
        CHECK(h3_gpu_submit(gpu));
        CHECK(h3_gpu_tensor_read_f32(y_t, got, yn));
        double db = psnr(got, ref, yn);
        printf("  conv 3x3 %dx%dx%d, %d->%d: TensorOps vs double %.1f dB\n", T, H, W, C, O, db);
        CHECK(db > 40.0);
        /* the F32 path for the same conv */
        h3_gpu_tensor *pad = h3_gpu_tensor_new_f32(gpu, (size_t)T * (H + 2) * (W + 2) * C);
        h3_gpu_tensor *wf = h3_gpu_tensor_from_f32(gpu, w, wn);
        h3_gpu_tensor *ym = h3_gpu_tensor_new_f32(gpu, yn);
        CHECK(h3_gpu_begin(gpu));
        CHECK(h3_gpu_vsr_pad_f32(gpu, pad, x_t, NULL, 1, T, H, W, C, 0, 1, 1));
        CHECK(h3_gpu_conv3d_as_2d_f32(gpu, ym, pad, wf, b_t, 1, T, H + 2, W + 2, C, O, 1, 3, 3, 1, 1, 1));
        CHECK(h3_gpu_submit(gpu));
        CHECK(h3_gpu_tensor_read_f32(ym, mps, yn));
        printf("  F32 MPSGraph path vs double %.1f dB\n", psnr(mps, ref, yn));
        CHECK(psnr(mps, ref, yn) > 80.0);
        free(x); free(w); free(b); free(ref); free(got); free(mps); free(wp); free(bp);
        h3_gpu_tensor_free(x_t); h3_gpu_tensor_free(xg); h3_gpu_tensor_free(yg); h3_gpu_tensor_free(w_t);
        h3_gpu_tensor_free(b_t); h3_gpu_tensor_free(y_t); h3_gpu_tensor_free(bp_t); h3_gpu_tensor_free(pad);
        h3_gpu_tensor_free(wf); h3_gpu_tensor_free(ym);
    }

    /* 2. MemBlock's first conv: cat[x, past], bias, residual, ReLU; odd channel counts */
    {
        const int T = 3, H = 9, W = 13, C = 48, O = 48;
        grid g = make_grid(T, H, W, C, O);
        size_t xn = (size_t)T * H * W * C, yn = (size_t)T * H * W * O, wn = (size_t)O * 2 * C * 9;
        float *x = malloc(xn * sizeof(float)), *past = malloc(xn * sizeof(float)), *w = malloc(wn * sizeof(float));
        float *b = malloc(O * sizeof(float)), *res = malloc(yn * sizeof(float));
        for (size_t i = 0; i < xn; i++) { x[i] = rnd(); past[i] = rnd(); }
        for (size_t i = 0; i < wn; i++) w[i] = rnd() * 0.1f;
        for (int i = 0; i < O; i++) b[i] = rnd();
        for (size_t i = 0; i < yn; i++) res[i] = rnd();
        float *ref = malloc(yn * sizeof(float)), *got = malloc(yn * sizeof(float));
        reference(x, past, w, w, b, res, T, H, W, C, O, 1, ref);
        uint16_t *wx = pack_taps(w, O, 2 * C, 0, C, g.Op, g.Cp), *wpst = pack_taps(w, O, 2 * C, C, C, g.Op, g.Cp);
        h3_gpu_tensor *x_t = h3_gpu_tensor_from_f32(gpu, x, xn), *p_t = h3_gpu_tensor_from_f32(gpu, past, xn);
        h3_gpu_tensor *r_t = h3_gpu_tensor_from_f32(gpu, res, yn);
        h3_gpu_tensor *xg = h3_gpu_tensor_new_bf16(gpu, (size_t)g.x_rows * g.Cp), *pg = h3_gpu_tensor_new_bf16(gpu, (size_t)g.x_rows * g.Cp);
        h3_gpu_tensor *rg = h3_gpu_tensor_new_bf16(gpu, (size_t)g.x_rows * g.Op), *yg = h3_gpu_tensor_new_bf16(gpu, (size_t)g.x_rows * g.Op);
        h3_gpu_tensor *wx_t = h3_gpu_tensor_from_bf16(gpu, wx, (size_t)9 * g.Op * g.Cp), *wp_t = h3_gpu_tensor_from_bf16(gpu, wpst, (size_t)9 * g.Op * g.Cp);
        float *bp = calloc(g.Op, sizeof(float)); memcpy(bp, b, O * sizeof(float));
        h3_gpu_tensor *b_t = h3_gpu_tensor_from_f32(gpu, bp, g.Op), *y_t = h3_gpu_tensor_new_f32(gpu, yn);
        CHECK(x_t && p_t && r_t && xg && pg && rg && yg && wx_t && wp_t && b_t && y_t);
        CHECK(h3_gpu_begin(gpu));
        CHECK(h3_gpu_vsr_pack_grid_bf16(gpu, xg, x_t, T, H, W, C, g.Cp, g.guard, g.x_rows, 1));
        CHECK(h3_gpu_vsr_pack_grid_bf16(gpu, pg, p_t, T, H, W, C, g.Cp, g.guard, g.x_rows, 1));
        CHECK(h3_gpu_vsr_pack_grid_bf16(gpu, rg, r_t, T, H, W, O, g.Op, g.guard, g.x_rows, 1));
        CHECK(h3_gpu_vsr_tapconv_bf16(gpu, yg, xg, pg, wx_t, wp_t, b_t, rg, g.rows, g.x_rows, g.guard, g.Cp, g.Op, 9, g.offsets, 1));
        CHECK(h3_gpu_vsr_zero_border_bf16(gpu, yg, T, H, W, g.Op, g.guard, g.x_rows));
        CHECK(h3_gpu_vsr_unpack_grid_f32(gpu, y_t, yg, T, H, W, O, g.Op, g.guard, g.x_rows));
        CHECK(h3_gpu_submit(gpu));
        CHECK(h3_gpu_tensor_read_f32(y_t, got, yn));
        double db = psnr(got, ref, yn);
        printf("  MemBlock conv (cat, bias, residual, ReLU) %dx%dx%d, %d ch: %.1f dB vs double\n", T, H, W, C, db);
        CHECK(db > 40.0);
        /* the border rows and the padded channels of the output are zero */
        uint16_t *raw = malloc((size_t)g.x_rows * g.Op * sizeof(uint16_t));
        CHECK(h3_gpu_tensor_read_bf16(yg, raw, (size_t)g.x_rows * g.Op));
        size_t nonzero = 0;
        for (uint32_t r = 0; r < g.rows; r++) {
            uint32_t f = r / ((H + 2) * (W + 2)), rem = r % ((H + 2) * (W + 2)), yy = rem / (W + 2), xx = rem % (W + 2);
            (void)f;
            int border = yy == 0 || yy == (uint32_t)H + 1 || xx == 0 || xx == (uint32_t)W + 1;
            for (int c = 0; c < g.Op; c++)
                if ((border || c >= O) && raw[((size_t)r + g.guard) * g.Op + c]) nonzero++;
        }
        CHECK(nonzero == 0);
        free(x); free(past); free(w); free(b); free(res); free(ref); free(got); free(wx); free(wpst); free(bp); free(raw);
        h3_gpu_tensor_free(x_t); h3_gpu_tensor_free(p_t); h3_gpu_tensor_free(r_t); h3_gpu_tensor_free(xg); h3_gpu_tensor_free(pg);
        h3_gpu_tensor_free(rg); h3_gpu_tensor_free(yg); h3_gpu_tensor_free(wx_t); h3_gpu_tensor_free(wp_t); h3_gpu_tensor_free(b_t); h3_gpu_tensor_free(y_t);
    }

    /* 3. speed on the decoder's 128-channel MemBlock stage (2 frames of 272x480) */
    {
        const int T = 2, H = 272, W = 480, C = 128, O = 128;
        grid g = make_grid(T, H, W, C, O);
        h3_gpu_tensor *xg = h3_gpu_tensor_new_bf16(gpu, (size_t)g.x_rows * g.Cp), *pg = h3_gpu_tensor_new_bf16(gpu, (size_t)g.x_rows * g.Cp);
        h3_gpu_tensor *yg = h3_gpu_tensor_new_bf16(gpu, (size_t)g.x_rows * g.Op);
        h3_gpu_tensor *w0 = h3_gpu_tensor_new_bf16(gpu, (size_t)9 * g.Op * g.Cp), *w1 = h3_gpu_tensor_new_bf16(gpu, (size_t)9 * g.Op * g.Cp);
        h3_gpu_tensor *b_t = h3_gpu_tensor_new_f32(gpu, g.Op);
        CHECK(xg && pg && yg && w0 && w1 && b_t);
        for (int pass = 0; pass < 2; pass++) {
            double t0 = now();
            CHECK(h3_gpu_begin(gpu));
            for (int i = 0; i < 5; i++) {
                CHECK(h3_gpu_vsr_tapconv_bf16(gpu, yg, xg, pg, w0, w1, b_t, NULL, g.rows, g.x_rows, g.guard, g.Cp, g.Op, 9, g.offsets, 1));
                CHECK(h3_gpu_vsr_tapconv_bf16(gpu, xg, yg, NULL, w0, NULL, b_t, NULL, g.rows, g.x_rows, g.guard, g.Cp, g.Op, 9, g.offsets, 1));
                CHECK(h3_gpu_vsr_tapconv_bf16(gpu, yg, xg, NULL, w0, NULL, b_t, xg, g.rows, g.x_rows, g.guard, g.Cp, g.Op, 9, g.offsets, 1));
            }
            CHECK(h3_gpu_submit(gpu));
            double secs = now() - t0;
            double flops = 5.0 * ((double)T * H * W) * 9.0 * 2.0 * ((double)2 * C * O + (double)C * O * 2);
            if (pass) printf("  MemBlock %dx%dx%d, %d ch: %.1f ms per block, %.1f TFLOP/s\n", T, H, W, C, secs / 5 * 1000.0, flops / secs / 1e12);
        }
        /* the same three convs on the F32 MPSGraph path */
        h3_gpu_tensor *xf = h3_gpu_tensor_new_f32(gpu, (size_t)T * (H + 2) * (W + 2) * 2 * C);
        h3_gpu_tensor *yf = h3_gpu_tensor_new_f32(gpu, (size_t)T * H * W * O);
        h3_gpu_tensor *wf = h3_gpu_tensor_new_f32(gpu, (size_t)O * 2 * C * 9), *wf2 = h3_gpu_tensor_new_f32(gpu, (size_t)O * C * 9);
        h3_gpu_tensor *bf = h3_gpu_tensor_new_f32(gpu, O);
        CHECK(xf && yf && wf && wf2 && bf);
        for (int pass = 0; pass < 2; pass++) {
            double t0 = now();
            CHECK(h3_gpu_begin(gpu));
            for (int i = 0; i < 5; i++) {
                CHECK(h3_gpu_conv3d_as_2d_f32(gpu, yf, xf, wf, bf, 1, T, H + 2, W + 2, 2 * C, O, 1, 3, 3, 1, 1, 1));
                CHECK(h3_gpu_conv3d_as_2d_f32(gpu, yf, xf, wf2, bf, 1, T, H + 2, W + 2, C, O, 1, 3, 3, 1, 1, 1));
                CHECK(h3_gpu_conv3d_as_2d_f32(gpu, yf, xf, wf2, bf, 1, T, H + 2, W + 2, C, O, 1, 3, 3, 1, 1, 1));
            }
            CHECK(h3_gpu_submit(gpu));
            if (pass) printf("  F32 MPSGraph, the same three convs: %.1f ms per block\n", (now() - t0) / 5 * 1000.0);
        }
    }
    printf("ok: %d checks (vsr TensorOps conv)\n", checks);
    return 0;
}
