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
 * H3_SV2_TEST_VAE_F32=1 forces the F32 (MPSGraph, M1-M4) VAE path. */
#include "h3_seedvr2.h"
#include "h3_safetensors.h"

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

int main(void) {
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
    test_single_frame(m, path);

    double gpu_seconds; uint64_t peak;
    h3_seedvr2_get_stats(m, &gpu_seconds, &peak);
    printf("gpu %.2f s, peak live %.2f GiB\n", gpu_seconds, (double)peak / 1073741824.0);
    h3_seedvr2_close(m);
    printf("ok: temporal SeedVR2 against the official reference (%d checks)\n", tests_run);
    return 0;
}
