/* Temporal SeedVR2 (SEEDVR2_TEMPORAL_PLAN): the engine's stages against
 * reference tensors from the official ByteDance-Seed/SeedVR code, run in
 * float32 on the CPU with the very weights the engine uses
 * (tools/dump_seedvr2_temporal_fixture.py, manifest in
 * tools/seedvr2_temporal_fixtures.md).
 *
 * H3_SEEDVR2_DIR      the MLX int8 model (default: as tests/test_seedvr2.c)
 * H3_SEEDVR2_TEMPORAL_FIXTURES   the folder with sv2t_T{1,5,9,17}_256.safetensors
 *                     (default ~/Documents/work/h3c/seedvr2-temporal/fixtures)
 * H3_SEEDVR2_STRICT=1 a missing model or fixture fails instead of skipping,
 *                     so "make test" passing is not mistaken for "verified".
 * H3_SV2_TEST_VAE_F32=1 forces the F32 (MPSGraph, M1-M4) VAE path (the
 *                     480x352 clip only with H3_SV2_TEST_LARGE=1: memory).
 * Arguments: --clips-only skips T=1; a second argument picks clips by name.
 * Run it alone: the T=17 clips peak at 18-20 GB. */
#include "h3_seedvr2.h"
#include "h3_safetensors.h"
#include "seedvr2_windows3.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int tests_run;
#define CHECK(condition) do { tests_run++; if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); exit(1); } } while (0)

static int strict(void) { const char *s = getenv("H3_SEEDVR2_STRICT"); return s && *s && strcmp(s, "0") != 0; }
static int exists(const char *path) { struct stat st; return stat(path, &st) == 0; }

static int skip(const char *what) {
    if (strict()) { fprintf(stderr, "FAIL: %s (H3_SEEDVR2_STRICT=1)\n", what); exit(1); }
    printf("skip: %s\n", what);
    return 0;
}

static float *load_tensor(const h3_st_header *h, const char *name, size_t expected) {
    const h3_st_tensor *t = h3_st_find(h, name);
    if (!t || t->dtype != H3_DTYPE_F32) { fprintf(stderr, "fixture: %s missing or not F32\n", name); exit(1); }
    size_t n = (size_t)h3_st_tensor_elements(t);
    if (expected && n != expected) { fprintf(stderr, "fixture: %s has %zu values, expected %zu\n", name, n, expected); exit(1); }
    float *data = malloc(n * sizeof(float));
    char error[256];
    if (!data || !h3_st_read_data(h, t, data, n * sizeof(float), error, sizeof(error))) {
        fprintf(stderr, "fixture: %s: %s\n", name, error); exit(1);
    }
    return data;
}

typedef struct { double max_abs, rms, ref_rms, psnr; } diff_stats;

static diff_stats compare(const float *a, const float *b, size_t n, double peak) {
    diff_stats d = {0, 0, 0, 0};
    double se = 0, sr = 0;
    for (size_t i = 0; i < n; i++) {
        double e = (double)a[i] - b[i];
        if (isnan(a[i]) || isinf(a[i])) { d.max_abs = INFINITY; continue; }
        if (fabs(e) > d.max_abs) d.max_abs = fabs(e);
        se += e * e; sr += (double)b[i] * b[i];
    }
    d.rms = sqrt(se / (double)n); d.ref_rms = sqrt(sr / (double)n);
    d.psnr = se > 0 ? 10 * log10(peak * peak * (double)n / se) : 999;
    return d;
}

static void report(const char *stage, diff_stats d, double limit_rel) {
    printf("  %-26s max abs %.5f  rms %.5f (ref rms %.4f, rel %.4f)  psnr %.1f dB\n",
           stage, d.max_abs, d.rms, d.ref_rms, d.ref_rms > 0 ? d.rms / d.ref_rms : 0, d.psnr);
    CHECK(d.max_abs < INFINITY);
    CHECK(d.ref_rms == 0 || d.rms / d.ref_rms < limit_rel);
}

/* The 3D window partition against tables from the reference's window.py,
 * and the number of distinct window lengths (the layout's classes) for the
 * sizes the app runs. Needs no model. */
static void test_windows3(void) {
    static int got[1024 * 6];
    for (size_t c = 0; c < sizeof(window_cases3) / sizeof(*window_cases3); c++) {
        int n = h3_seedvr2_debug_windows3(window_cases3[c].t, window_cases3[c].w, window_cases3[c].h,
                                          window_cases3[c].shifted, got, 1024);
        if (n != window_cases3[c].count)
            fprintf(stderr, "windows t=%d %dx%d shifted %d: %d, expected %d\n", window_cases3[c].t, window_cases3[c].w,
                    window_cases3[c].h, window_cases3[c].shifted, n, window_cases3[c].count);
        CHECK(n == window_cases3[c].count);
        CHECK(memcmp(got, window_table3[window_cases3[c].first], (size_t)n * 6 * sizeof(int)) == 0);
    }
    /* latent tokens (w, h) of 512x288 .. 1920x1088 outputs, 1..8 latent frames */
    static const int grids[][2] = {{32, 18}, {48, 27}, {84, 48}, {80, 46}, {120, 68}};
    int worst = 0;
    for (size_t g = 0; g < sizeof(grids) / sizeof(*grids); g++)
        for (int t = 1; t <= 8; t++)
            for (int shifted = 0; shifted < 2; shifted++) {
                int n = h3_seedvr2_debug_windows3(t, grids[g][0], grids[g][1], shifted, got, 1024), lengths[1024], k = 0;
                CHECK(n > 0 && n <= 1024);
                for (int i = 0; i < n; i++) {
                    const int *o = got + i * 6;
                    int len = (o[1] - o[0]) * (o[3] - o[2]) * (o[5] - o[4]), seen = 0;
                    for (int j = 0; j < k; j++) seen |= lengths[j] == len;
                    if (!seen) lengths[k++] = len;
                }
                if (k > worst) worst = k;
            }
    printf("ok: 3D windows match window.py (%zu cases); at most %d window lengths per layout\n",
           sizeof(window_cases3) / sizeof(*window_cases3), worst);
    CHECK(worst <= 64);   /* SV2_MAX_CLASSES */
}

/* Per-frame noise: a frame's values do not depend on the window it is drawn in. */
static void test_noise(void) {
    enum { LW = 12, LH = 8, P = LW * LH };
    static float a[16 * 6 * P], b[16 * 3 * P], one[16 * P];
    CHECK(h3_seedvr2_debug_noise(42, 0, 6, LW, LH, a));
    CHECK(h3_seedvr2_debug_noise(42, 2, 3, LW, LH, b));   /* frames 2..4 */
    for (int c = 0; c < 16; c++)
        for (int t = 0; t < 3; t++)
            CHECK(memcmp(b + ((size_t)c * 3 + t) * P, a + ((size_t)c * 6 + t + 2) * P, P * sizeof(float)) == 0);
    CHECK(h3_seedvr2_debug_noise(42, 0, 1, LW, LH, one));
    for (int c = 0; c < 16; c++) CHECK(memcmp(one + (size_t)c * P, a + (size_t)c * 6 * P, P * sizeof(float)) == 0);
    /* frames differ and look like N(0, 1) */
    double sum = 0, sq = 0, same = 0;
    for (size_t i = 0; i < sizeof(a) / sizeof(*a); i++) { sum += a[i]; sq += (double)a[i] * a[i]; }
    for (int i = 0; i < P; i++) same += a[i] == a[P + i];
    double n = (double)(sizeof(a) / sizeof(*a)), mean = sum / n, var = sq / n - mean * mean;
    CHECK(fabs(mean) < 0.05 && fabs(var - 1.0) < 0.1 && same == 0);
    printf("ok: per-frame noise (mean %.3f, var %.3f)\n", mean, var);
}

/* The memory model against the measured points (within 5 %), and its shape. */
static void test_memory_model(void) {
    const uint64_t R = 7015000000ull, DIT = 6668000000ull;   /* the model's resident bytes, its transformer blocks */
    struct { int nax, w, h, T, split; double measured_gb; } pts[] = {
        {1, 1344, 768, 5, 0, 9.89}, {1, 1344, 768, 17, 4, 16.72}, {1, 960, 544, 17, 4, 8.74},
        {0, 960, 544, 5, 0, 11.21}, {0, 960, 544, 17, 4, 13.03}, {0, 768, 432, 17, 4, 8.64}};
    for (size_t i = 0; i < sizeof(pts) / sizeof(*pts); i++) {
        double est = (double)sv2_video_peak_bytes(R, DIT, pts[i].nax, pts[i].w, pts[i].h, pts[i].T, pts[i].split, 1) / 1e9;
        double rel = fabs(est - pts[i].measured_gb) / pts[i].measured_gb;
        if (rel > 0.05) fprintf(stderr, "memory model: %dx%d T=%d split %d nax %d: %.2f GB, measured %.2f\n",
                                pts[i].w, pts[i].h, pts[i].T, pts[i].split, pts[i].nax, est, pts[i].measured_gb);
        CHECK(rel <= 0.05);
    }
    /* sliced memory does not grow with T; the whole window does; the transformer resident costs more */
    CHECK(sv2_video_peak_bytes(R, DIT, 1, 1344, 768, 29, 4, 1) - sv2_video_peak_bytes(R, DIT, 1, 1344, 768, 17, 4, 1) < 1000000000ull);
    CHECK(sv2_video_peak_bytes(R, DIT, 1, 1344, 768, 9, 0, 1) > sv2_video_peak_bytes(R, DIT, 1, 1344, 768, 5, 0, 1));
    CHECK(sv2_video_peak_bytes(R, DIT, 1, 1344, 768, 5, 0, 0) > sv2_video_peak_bytes(R, DIT, 1, 1344, 768, 5, 0, 1) + 6000000000ull);
    /* 17 frames at 1344x768 fit a 24 GB Mac (working set about 17.8 GiB = 19.1 GB, the engine keeps 90 %) */
    CHECK(sv2_video_peak_bytes(R, DIT, 1, 1344, 768, 17, 4, 1) < 17160000000ull);
    /* the model-less check (the app's sheet) follows the same model: a
     * window of 4n+1, no larger than the budget takes, monotone in frames
     * and size, and a small output on any Mac the tests run on */
    h3_seedvr2_info info = {0};
    info.resident_bytes = R; info.block_bytes = DIT; info.tensor_ops = 1;
    CHECK(!h3_seedvr2_video_output_fits(&info, 672, 384, 1344, 768, 4));
    CHECK(!h3_seedvr2_video_output_fits(&info, 672, 384, 1344, 768, 0));
    CHECK(!h3_seedvr2_video_output_fits(&info, 672, 384, 1344, 768, 125));
    CHECK(h3_seedvr2_video_output_fits(&info, 128, 128, 256, 256, 17));
    int fits5 = h3_seedvr2_video_output_fits(&info, 672, 384, 1344, 768, 5);
    int fits17 = h3_seedvr2_video_output_fits(&info, 672, 384, 1344, 768, 17);
    int fits29 = h3_seedvr2_video_output_fits(&info, 672, 384, 1344, 768, 29);
    int large17 = h3_seedvr2_video_output_fits(&info, 960, 544, 1920, 1088, 17);
    CHECK(fits5 >= fits17 && fits17 >= fits29 && fits17 >= large17);
    printf("ok: video memory model (%zu measured points), model-less check 1344x768 x5 %d x17 %d x29 %d, 1920x1088 x17 %d\n",
           sizeof(pts) / sizeof(*pts), fits5, fits17, fits29, large17);
}

/* The window session on a dry run (no model): every frame count comes out
 * once, in order, for every window/overlap/blend/total-known combination;
 * window starts are multiples of 4; a window's frames arrive as next_frames
 * asks. Each input frame carries its number in every byte. */
static void test_video_session(void) {
    static const int counts[] = {1, 2, 4, 5, 6, 16, 17, 18, 28, 29, 30, 33, 56, 124};
    static const int windows[] = {5, 17}, overlaps[] = {0, 1, 5, 9};
    const int IW = 8, IH = 4, OW = 16, OH = 16;
    int combos = 0;
    for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++)
        for (size_t wi = 0; wi < sizeof(windows) / sizeof(*windows); wi++)
            for (size_t oi = 0; oi < sizeof(overlaps) / sizeof(*overlaps); oi++)
                for (int blend = 0; blend < 2; blend++)
                    for (int known = 0; known < 2; known++) {
                        int N = counts[ci], W = windows[wi], O = overlaps[oi];
                        if (O >= W) continue;
                        h3_seedvr2_video_params p = {W, O, -1, blend, 1, -1};
                        char error[512];
                        h3_seedvr2_video_session *s = h3_seedvr2_video_session_open(NULL, IW, IH, OW, OH, known ? N : 0, &p, error, sizeof(error));
                        if (!s) fprintf(stderr, "open: %s\n", error);
                        CHECK(s != NULL);
                        uint8_t *in_px = malloc((size_t)N * IW * IH * 4);
                        for (int t = 0; t < N; t++) memset(in_px + (size_t)t * IW * IH * 4, t & 255, (size_t)IW * IH * 4);
                        uint8_t *out_px = malloc((size_t)(N + 1) * OW * OH * 4);
                        h3_vsr_image_out outs[256];
                        int sent = 0, got = 0, calls = 0;
                        while (sent < N || !got) {
                            int want = h3_seedvr2_video_session_next_frames(s);
                            CHECK(want > 0 || sent == N);
                            int give = want < N - sent ? want : N - sent;
                            int is_last = sent + give >= N;
                            h3_vsr_image ins[128];
                            for (int t = 0; t < give; t++)
                                ins[t] = (h3_vsr_image){in_px + (size_t)(sent + t) * IW * IH * 4, IW, IH, IW * 4, H3_VSR_BGRA8};
                            int capacity = h3_seedvr2_video_session_pending_frames(s) + give;
                            CHECK(capacity <= 256);
                            for (int t = 0; t < capacity; t++)
                                outs[t] = (h3_vsr_image_out){out_px + (size_t)(got + t) * OW * OH * 4, OW, OH, OW * 4, H3_VSR_BGRA8};
                            int produced = 0;
                            if (!h3_seedvr2_video_session_process(s, ins, give, is_last, outs, capacity, &produced, NULL, NULL, NULL, error, sizeof(error)))
                                fprintf(stderr, "N=%d W=%d O=%d blend %d known %d: %s\n", N, W, O, blend, known, error);
                            CHECK(produced >= 0 && produced <= capacity);
                            got += produced;
                            sent += give;
                            calls++;
                            if (is_last) break;
                            CHECK(calls < 1000);
                        }
                        if (got != N) fprintf(stderr, "N=%d W=%d O=%d blend %d known %d: %d frames out\n", N, W, O, blend, known, got);
                        CHECK(got == N);
                        CHECK(h3_seedvr2_video_session_next_frames(s) == 0 && h3_seedvr2_video_session_pending_frames(s) == 0);
                        for (int t = 0; t < N; t++)
                            for (size_t k = 0; k < (size_t)OW * OH * 4; k++) {
                                if (out_px[(size_t)t * OW * OH * 4 + k] != (t & 255)) {
                                    fprintf(stderr, "N=%d W=%d O=%d blend %d known %d: frame %d byte %zu = %d\n", N, W, O, blend, known, t, k, out_px[(size_t)t * OW * OH * 4 + k]);
                                    CHECK(0);
                                }
                            }
                        int frames_run, padded; double seconds;
                        int nw = h3_seedvr2_video_session_windows(s, &frames_run, &padded, &seconds);
                        CHECK(nw >= 1 && frames_run == nw * W);
                        /* the last window ends on the last frame: padding only when the video is shorter than a window */
                        if (N >= W) { if (padded) fprintf(stderr, "N=%d W=%d O=%d known %d: %d padded\n", N, W, O, known, padded); CHECK(padded == 0); }
                        else CHECK(padded == W - N);
                        int stride = W - O, expected_windows = N <= W ? 1 : (N - W + stride - 1) / stride + 1;
                        CHECK(nw == expected_windows);
                        h3_seedvr2_video_session_close(s);
                        free(in_px); free(out_px);
                        combos++;
                    }
    /* bad parameters are refused */
    char reason[256];
    h3_seedvr2_video_params bad1 = {4, 0, -1, 0, 0, -1}, bad2 = {9, 2, -1, 0, 0, -1}, bad3 = {5, 5, -1, 0, 0, -1}, ok = {9, 5, -1, 0, 0, -1};
    CHECK(!h3_seedvr2_video_check_params(&bad1, reason, sizeof(reason)));
    CHECK(!h3_seedvr2_video_check_params(&bad2, reason, sizeof(reason)));
    CHECK(!h3_seedvr2_video_check_params(&bad3, reason, sizeof(reason)));
    CHECK(h3_seedvr2_video_check_params(&ok, reason, sizeof(reason)));
    printf("ok: video session frame bookkeeping (%d combinations)\n", combos);
}

static const char *model_dir(char *buf, size_t size) {
    const char *dir = getenv("H3_SEEDVR2_DIR");
    if (dir) return dir;
    const char *home = getenv("HOME") ? getenv("HOME") : ".";
    snprintf(buf, size, "%s/models/seedvr2/SeedVR2-3B-mlx-int8", home);
    if (!exists(buf)) snprintf(buf, size, "%s/Library/Application Support/H3cApp/Upscalers/SeedVR2", home);
    return buf;
}

static const char *fixture_dir(char *buf, size_t size) {
    const char *dir = getenv("H3_SEEDVR2_TEMPORAL_FIXTURES");
    if (dir) return dir;
    snprintf(buf, size, "%s/Documents/work/h3c/seedvr2-temporal/fixtures", getenv("HOME") ? getenv("HOME") : ".");
    return buf;
}

/* T = 1 through the current (image) engine: the official temporal code with
 * one frame must agree with what the engine already does, stage by stage. */
static void test_single_frame(h3_seedvr2 *m, const char *path) {
    h3_st_header h;
    char error[512];
    CHECK(h3_st_read_header(path, &h, error, sizeof(error)) || (fprintf(stderr, "%s\n", error), 0));
    const h3_st_tensor *xt = h3_st_find(&h, "x");
    CHECK(xt && xt->ndim == 4 && xt->shape[0] == 3 && xt->shape[1] == 1);
    int H = (int)xt->shape[2], W = (int)xt->shape[3], lh = H / 8, lw = W / 8;
    size_t plane = (size_t)W * H, lplane = (size_t)lw * lh;
    printf("T=1 fixture %s: %dx%d, latent %dx%d\n", path, W, H, lw, lh);
    float *x = load_tensor(&h, "x", 3 * plane);

    static const char *stage_names[7] = {"enc_conv_in", "enc_down0", "enc_down1", "enc_down2", "enc_down3", "enc_mid", "enc_h"};
    for (int stage = 0; stage <= 6; stage++) {
        const h3_st_tensor *t = h3_st_find(&h, stage_names[stage]);
        CHECK(t && t->ndim == 4 && t->shape[1] == 1);
        size_t n = (size_t)h3_st_tensor_elements(t);
        float *ref = load_tensor(&h, stage_names[stage], n), *ours = malloc(n * sizeof(float));
        int ow, oh, oc;
        CHECK(h3_seedvr2_debug_encode_stage(m, x, W, H, stage, ours, &ow, &oh, &oc, n, error, sizeof(error)) ||
              (fprintf(stderr, "%s\n", error), 0));
        CHECK((size_t)ow * oh * oc == n && oc == (int)t->shape[0] && oh == (int)t->shape[2] && ow == (int)t->shape[3]);
        report(stage_names[stage], compare(ours, ref, n, 1.0), stage < 6 ? 0.2 : 0.05);
        free(ref); free(ours);
    }

    float *latent_ref = load_tensor(&h, "latent", 16 * lplane);
    float *enc = malloc(16 * lplane * sizeof(float));
    CHECK(h3_seedvr2_debug_encode(m, x, W, H, enc, error, sizeof(error)) || (fprintf(stderr, "%s\n", error), 0));
    report("encode latent", compare(enc, latent_ref, 16 * lplane, 1.0), 0.05);
    float *sliced = load_tensor(&h, "enc_mean_sliced", 16 * lplane);
    float *mean = load_tensor(&h, "enc_mean", 16 * lplane);
    report("reference sliced = whole", compare(sliced, mean, 16 * lplane, 1.0), 1e-5);
    free(sliced); free(mean);

    float *decoded_ref = load_tensor(&h, "decoded", 3 * plane);
    float *dec = malloc(3 * plane * sizeof(float));
    CHECK(h3_seedvr2_debug_decode(m, latent_ref, lw, lh, dec, error, sizeof(error)) || (fprintf(stderr, "%s\n", error), 0));
    report("decode (ref latent)", compare(dec, decoded_ref, 3 * plane, 2.0), 0.05);

    if (h3_st_find(&h, "pred")) {
        float *noise = load_tensor(&h, "noise", 16 * lplane);
        float *pred_ref = load_tensor(&h, "pred", 16 * lplane);
        float *z_ref = load_tensor(&h, "z", 16 * lplane);
        float *dz_ref = load_tensor(&h, "decoded_z", 3 * plane);
        float *dit_in = malloc(33 * lplane * sizeof(float)), *pred = malloc(16 * lplane * sizeof(float));
        memcpy(dit_in, noise, 16 * lplane * sizeof(float));
        memcpy(dit_in + 16 * lplane, latent_ref, 16 * lplane * sizeof(float));
        for (size_t i = 0; i < lplane; i++) dit_in[32 * lplane + i] = 1.0f;
        CHECK(h3_seedvr2_debug_dit(m, dit_in, lw, lh, pred, -1, error, sizeof(error)) || (fprintf(stderr, "%s\n", error), 0));
        report("dit prediction", compare(pred, pred_ref, 16 * lplane, 1.0), 0.05);
        float *pred_video = malloc(16 * lplane * sizeof(float));
        CHECK(h3_seedvr2_debug_dit_video(m, dit_in, 1, lw, lh, pred_video, -1, error, sizeof(error)) ||
              (fprintf(stderr, "%s\n", error), 0));
        CHECK(memcmp(pred_video, pred, 16 * lplane * sizeof(float)) == 0);
        printf("  %-26s bit-identical to the image path\n", "dit (1-frame clip)");
        free(pred_video);
        float *z = malloc(16 * lplane * sizeof(float));
        for (size_t i = 0; i < 16 * lplane; i++) z[i] = noise[i] - pred[i];
        report("z (noise - v)", compare(z, z_ref, 16 * lplane, 1.0), 0.05);
        CHECK(h3_seedvr2_debug_decode(m, z, lw, lh, dec, error, sizeof(error)));
        report("decode (our z)", compare(dec, dz_ref, 3 * plane, 2.0), 0.1);
        free(noise); free(pred_ref); free(z_ref); free(dz_ref); free(dit_in); free(pred); free(z);
    }
    free(x); free(latent_ref); free(enc); free(decoded_ref); free(dec);
    h3_st_free_header(&h);
}

/* A clip: every encoder and decoder stage of the whole clip against the
 * reference, the reference's sliced passes, and the engine's own sliced
 * passes against its whole-clip ones. */
static void test_clip(h3_seedvr2 *m, const char *path) {
    h3_st_header h;
    char error[512];
    CHECK(h3_st_read_header(path, &h, error, sizeof(error)) || (fprintf(stderr, "%s\n", error), 0));
    const h3_st_tensor *xt = h3_st_find(&h, "x");
    CHECK(xt && xt->ndim == 4 && xt->shape[0] == 3);
    int T = (int)xt->shape[1], H = (int)xt->shape[2], W = (int)xt->shape[3];
    const h3_st_tensor *lt = h3_st_find(&h, "latent");
    CHECK(lt && lt->ndim == 4);
    int L = (int)lt->shape[1], lh = (int)lt->shape[2], lw = (int)lt->shape[3];
    CHECK(L == (T - 1) / 4 + 1 && lh == H / 8 && lw == W / 8);
    size_t lsize = (size_t)16 * L * lh * lw, rgb = (size_t)3 * T * H * W;
    printf("clip %s: %d frames %dx%d, latent %d x %dx%d\n", path, T, W, H, L, lw, lh);
    float *x = load_tensor(&h, "x", rgb);
    int of, ow, oh, oc;

    static const char *enc_names[7] = {"enc_conv_in", "enc_down0", "enc_down1", "enc_down2", "enc_down3", "enc_mid", "enc_h"};
    for (int stage = 0; stage <= 6; stage++) {
        const h3_st_tensor *t = h3_st_find(&h, enc_names[stage]);
        if (!t) continue;
        size_t n = (size_t)h3_st_tensor_elements(t);
        float *ref = load_tensor(&h, enc_names[stage], n), *ours = malloc(n * sizeof(float));
        CHECK(h3_seedvr2_debug_encode_video(m, x, T, W, H, 0, stage, ours, &of, &ow, &oh, &oc, n, error, sizeof(error)) ||
              (fprintf(stderr, "%s\n", error), 0));
        CHECK(oc == (int)t->shape[0] && of == (int)t->shape[1] && oh == (int)t->shape[2] && ow == (int)t->shape[3]);
        report(enc_names[stage], compare(ours, ref, n, 1.0), stage < 6 ? 0.2 : 0.05);
        free(ref); free(ours);
    }
    /* the latent, whole and sliced (split 4) */
    float *latent_ref = load_tensor(&h, "latent", lsize);
    float *enc = malloc(2 * lsize * sizeof(float)), *enc_sliced = malloc(2 * lsize * sizeof(float));
    CHECK(h3_seedvr2_debug_encode_video(m, x, T, W, H, 0, 99, enc, &of, &ow, &oh, &oc, 2 * lsize, error, sizeof(error)) ||
          (fprintf(stderr, "%s\n", error), 0));
    CHECK(of == L && oc == 32);
    float *ours = malloc(lsize * sizeof(float));
    for (size_t i = 0; i < lsize; i++) ours[i] = enc[i] * 0.9152f;   /* the mean: channels 0..15 */
    report("encode latent", compare(ours, latent_ref, lsize, 1.0), 0.05);
    CHECK(h3_seedvr2_debug_encode_video(m, x, T, W, H, 4, 99, enc_sliced, &of, &ow, &oh, &oc, 2 * lsize, error, sizeof(error)) ||
          (fprintf(stderr, "%s\n", error), 0));
    CHECK(of == L && oc == 32);
    /* bit for bit on the grid; the F32 path keeps its causal state in BF16 (about 1e-4) */
    report("encode sliced = whole", compare(enc_sliced, enc, 2 * lsize, 1.0), 1e-3);
    float *mean_sliced_ref = load_tensor(&h, "enc_mean_sliced", lsize);
    report("encode sliced vs ref", compare(enc_sliced, mean_sliced_ref, lsize, 1.0), 0.05);
    free(mean_sliced_ref); free(enc); free(enc_sliced); free(ours);

    static const char *dec_names[7] = {"dec_conv_in", "dec_mid", "dec_up0", "dec_up1", "dec_up2", "dec_up3", "decoded"};
    for (int stage = 0; stage <= 6; stage++) {
        const h3_st_tensor *t = h3_st_find(&h, dec_names[stage]);
        if (!t) continue;
        size_t n = (size_t)h3_st_tensor_elements(t);
        float *ref = load_tensor(&h, dec_names[stage], n), *dec = malloc(n * sizeof(float));
        CHECK(h3_seedvr2_debug_decode_video(m, latent_ref, L, lw, lh, 0, stage, dec, &of, &ow, &oh, &oc, n, error, sizeof(error)) ||
              (fprintf(stderr, "%s\n", error), 0));
        CHECK(oc == (int)t->shape[0] && of == (int)t->shape[1] && oh == (int)t->shape[2] && ow == (int)t->shape[3]);
        report(dec_names[stage], compare(dec, ref, n, stage == 6 ? 2.0 : 1.0), stage == 6 ? 0.05 : 0.2);
        free(ref); free(dec);
    }
    float *decoded_ref = load_tensor(&h, "decoded", rgb), *decoded_sliced_ref = load_tensor(&h, "decoded_sliced", rgb);
    float *dec = malloc(rgb * sizeof(float)), *dec_sliced = malloc(rgb * sizeof(float));
    CHECK(h3_seedvr2_debug_decode_video(m, latent_ref, L, lw, lh, 0, 99, dec, &of, &ow, &oh, &oc, rgb, error, sizeof(error)) ||
          (fprintf(stderr, "%s\n", error), 0));
    CHECK(of == T && oc == 3 && oh == H && ow == W);
    CHECK(h3_seedvr2_debug_decode_video(m, latent_ref, L, lw, lh, 4, 99, dec_sliced, &of, &ow, &oh, &oc, rgb, error, sizeof(error)) ||
          (fprintf(stderr, "%s\n", error), 0));
    CHECK(of == T && oc == 3);
    report("decode sliced = whole", compare(dec_sliced, dec, rgb, 2.0), 1e-3);
    report("decode sliced vs ref", compare(dec_sliced, decoded_sliced_ref, rgb, 2.0), 0.05);
    report("ref sliced vs ref whole", compare(decoded_sliced_ref, decoded_ref, rgb, 2.0), 0.05);
    /* per frame: the worst frame of the sliced decode (a seam would show here) */
    size_t plane = (size_t)H * W;
    double worst = 0; int worst_t = 0;
    for (int t = 0; t < T; t++) {
        double se = 0, sr = 0;
        for (int c = 0; c < 3; c++)
            for (size_t i = 0; i < plane; i++) {
                size_t at = ((size_t)c * T + t) * plane + i;
                double e = (double)dec_sliced[at] - decoded_ref[at];
                se += e * e; sr += (double)decoded_ref[at] * decoded_ref[at];
            }
        double rel = sr > 0 ? sqrt(se / sr) : 0;
        if (rel > worst) { worst = rel; worst_t = t; }
    }
    printf("  worst sliced frame vs ref: %d (rel %.4f)\n", worst_t, worst);
    CHECK(worst < 0.05);

    if (h3_st_find(&h, "pred")) {
        float *noise = load_tensor(&h, "noise", lsize), *pred_ref = load_tensor(&h, "pred", lsize);
        float *z_ref = load_tensor(&h, "z", lsize), *dz_ref = load_tensor(&h, "decoded_z", rgb);
        float *dit_in = malloc((lsize / 16 * 33) * sizeof(float)), *pred = malloc(lsize * sizeof(float));
        memcpy(dit_in, noise, lsize * sizeof(float));
        memcpy(dit_in + lsize, latent_ref, lsize * sizeof(float));
        for (size_t i = 0; i < lsize / 16; i++) dit_in[2 * lsize + i] = 1.0f;
        CHECK(h3_seedvr2_debug_dit_video(m, dit_in, L, lw, lh, pred, -1, error, sizeof(error)) ||
              (fprintf(stderr, "%s\n", error), 0));
        report("dit prediction", compare(pred, pred_ref, lsize, 1.0), 0.05);
        /* per latent frame: the temporal windows must not favour any frame */
        size_t lplane = (size_t)lh * lw;
        for (int t = 0; t < L; t++) {
            double se = 0, sr = 0;
            for (int c = 0; c < 16; c++)
                for (size_t i = 0; i < lplane; i++) {
                    size_t at = ((size_t)c * L + t) * lplane + i;
                    double e = (double)pred[at] - pred_ref[at];
                    se += e * e; sr += (double)pred_ref[at] * pred_ref[at];
                }
            printf("    latent frame %d: rel %.4f\n", t, sr > 0 ? sqrt(se / sr) : 0);
            CHECK(sr == 0 || sqrt(se / sr) < 0.05);
        }
        float *z = malloc(lsize * sizeof(float));
        for (size_t i = 0; i < lsize; i++) z[i] = noise[i] - pred[i];
        report("z (noise - v)", compare(z, z_ref, lsize, 1.0), 0.05);
        CHECK(h3_seedvr2_debug_decode_video(m, z, L, lw, lh, 0, 99, dec, &of, &ow, &oh, &oc, rgb, error, sizeof(error)) ||
              (fprintf(stderr, "%s\n", error), 0));
        report("decode (our z)", compare(dec, dz_ref, rgb, 2.0), 0.1);
        free(noise); free(pred_ref); free(z_ref); free(dz_ref); free(dit_in); free(pred); free(z);
    }
    free(dec); free(dec_sliced); free(decoded_ref); free(decoded_sliced_ref); free(latent_ref); free(x);
    h3_st_free_header(&h);
}

int main(int argc, char **argv) {
    test_windows3();
    test_noise();
    test_memory_model();
    test_video_session();
    char mbuf[1024], fbuf[1024], cfg[1200], path[1200];
    const char *dir = model_dir(mbuf, sizeof(mbuf));
    const char *fixtures = fixture_dir(fbuf, sizeof(fbuf));
    snprintf(cfg, sizeof(cfg), "%s/config.json", dir);
    if (!exists(cfg)) return skip("SeedVR2 weights are not installed (H3_SEEDVR2_DIR)");
    snprintf(path, sizeof(path), "%s/sv2t_T1_256.safetensors", fixtures);
    if (!exists(path)) return skip("temporal fixtures are not installed (H3_SEEDVR2_TEMPORAL_FIXTURES)");

    char error[512];
    h3_seedvr2_open_params params = {"h3_shaders.metal", 1, 42, -1, getenv("H3_SV2_TEST_VAE_F32") != NULL, -1};
    h3_seedvr2 *m = h3_seedvr2_open(dir, &params, NULL, NULL, error, sizeof(error));
    if (!m) { fprintf(stderr, "open: %s\n", error); return 1; }
    if (!(argc > 1 && strcmp(argv[1], "--clips-only") == 0)) test_single_frame(m, path);
    static const char *clips[] = {"sv2t_T5_256.safetensors", "sv2t_T9_256.safetensors", "sv2t_T17_256.safetensors",
                                  "sv2t_T17_480x352_vae.safetensors"};
    for (size_t i = 0; i < sizeof(clips) / sizeof(*clips); i++) {
        snprintf(path, sizeof(path), "%s/%s", fixtures, clips[i]);
        if (!exists(path)) { skip(path); continue; }
        if (argc > 2 && !strstr(clips[i], argv[2])) continue;
        /* the whole 17-frame 480x352 clip peaks at 18 GiB on the grid; the F32
         * path keeps twice that and would swap a 24 GB Mac (phase 3 slices it) */
        if (strstr(clips[i], "480x352") && getenv("H3_SV2_TEST_VAE_F32") && !getenv("H3_SV2_TEST_LARGE")) {
            printf("skip: %s on the F32 path (H3_SV2_TEST_LARGE=1 runs it)\n", clips[i]);
            continue;
        }
        test_clip(m, path);
    }

    double gpu_seconds; uint64_t peak;
    h3_seedvr2_get_stats(m, &gpu_seconds, &peak);
    printf("gpu %.2f s, peak live %.2f GiB\n", gpu_seconds, (double)peak / 1073741824.0);
    h3_seedvr2_close(m);
    printf("ok: temporal SeedVR2 against the official reference (%d checks)\n", tests_run);
    return 0;
}
