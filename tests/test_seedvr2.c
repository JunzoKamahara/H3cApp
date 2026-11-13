/* SeedVR2 (h3_seedvr2.c): the window partition against mflux's rule (always),
 * and, when a model and a reference dump are installed, each stage against
 * the mflux/MLX-Swift reference: H3_SEEDVR2_DIR (default ~/models/seedvr2/
 * SeedVR2-3B-mlx-int8 or the app's Upscalers/SeedVR2 folder) and
 * H3_SEEDVR2_FIXTURE (a safetensors written by the dump tool, see
 * UPSCALE_PLAN appendix A.7). Skips the GPU part when either is missing. */
#include "h3_seedvr2.h"
#include "h3_seedvr2_internal.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int tests_run;
#define CHECK(condition) do { tests_run++; if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); exit(1); } } while (0)

static int windows_equal(int tw, int th, int shifted, const int *expected, int count) {
    int got[1024 * 4];
    int n = h3_seedvr2_debug_windows(tw, th, shifted, got, 1024);
    if (n != count) { fprintf(stderr, "  %dx%d shift %d: %d windows, expected %d\n", tw, th, shifted, n, count); return 0; }
    for (int i = 0; i < count * 4; i++)
        if (got[i] != expected[i]) { fprintf(stderr, "  %dx%d shift %d: window %d differs\n", tw, th, shifted, i / 4); return 0; }
    return 1;
}

static void test_windows(void) {
    /* from mflux's _make_windows (scratch windows.py): 512x288 -> 32x18 tokens */
    CHECK(windows_equal(32, 18, 0, (const int[]){0,15,0,27, 15,18,0,27, 0,15,27,32, 15,18,27,32}, 4));
    CHECK(windows_equal(32, 18, 1, (const int[]){0,7,0,13, 7,18,0,13, 0,7,13,32, 7,18,13,32}, 4));
    /* 672x384 -> 42x24 */
    CHECK(windows_equal(42, 24, 0, (const int[]){0,15,0,27, 15,24,0,27, 0,15,27,42, 15,24,27,42}, 4));
    CHECK(windows_equal(42, 24, 1, (const int[]){0,7,0,13, 7,22,0,13, 22,24,0,13, 0,7,13,40, 7,22,13,40,
                                                 22,24,13,40, 0,7,40,42, 7,22,40,42, 22,24,40,42}, 9));
    /* 1344x768 -> 84x48: 16 windows both parities, the first six */
    int got[1024 * 4];
    CHECK(h3_seedvr2_debug_windows(84, 48, 0, got, 1024) == 16);
    CHECK(got[0] == 0 && got[1] == 15 && got[2] == 0 && got[3] == 27);
    CHECK(got[12] == 45 && got[13] == 48 && got[14] == 0 && got[15] == 27);
    CHECK(got[16] == 0 && got[17] == 15 && got[18] == 27 && got[19] == 54);
    CHECK(h3_seedvr2_debug_windows(84, 48, 1, got, 1024) == 16);
    CHECK(got[0] == 0 && got[1] == 7 && got[2] == 0 && got[3] == 13);
    CHECK(got[12] == 37 && got[13] == 48 && got[14] == 0 && got[15] == 13);
    CHECK(got[16] == 0 && got[17] == 7 && got[18] == 13 && got[19] == 40);
    /* 1920x1088 -> 120x68: 25 / 30; 960x544 -> 60x34: 9 / 9 */
    CHECK(h3_seedvr2_debug_windows(120, 68, 0, got, 1024) == 25);
    CHECK(h3_seedvr2_debug_windows(120, 68, 1, got, 1024) == 30);
    CHECK(h3_seedvr2_debug_windows(60, 34, 0, got, 1024) == 9);
    CHECK(h3_seedvr2_debug_windows(60, 34, 1, got, 1024) == 9);
    /* every token in exactly one window */
    for (int shifted = 0; shifted < 2; shifted++) {
        int n = h3_seedvr2_debug_windows(84, 48, shifted, got, 1024);
        int covered[84 * 48] = {0};
        for (int i = 0; i < n; i++)
            for (int h = got[i * 4]; h < got[i * 4 + 1]; h++)
                for (int w = got[i * 4 + 2]; w < got[i * 4 + 3]; w++) covered[h * 84 + w]++;
        int ok = 1;
        for (int t = 0; t < 84 * 48; t++) if (covered[t] != 1) ok = 0;
        CHECK(ok);
    }
}

/* A safetensors file of zero-filled 2-D tensors. */
typedef struct { const char *name, *dtype; unsigned long long rows, cols; } st_entry;

static void write_safetensors(const char *path, const st_entry *e, int count) {
    char json[2048];
    size_t len = 0;
    unsigned long long offset = 0;
    len += (size_t)snprintf(json + len, sizeof(json) - len, "{");
    for (int i = 0; i < count; i++) {
        unsigned long long bytes = e[i].rows * e[i].cols * (strcmp(e[i].dtype, "U32") ? 2u : 4u);
        len += (size_t)snprintf(json + len, sizeof(json) - len,
                                "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%llu,%llu],\"data_offsets\":[%llu,%llu]}",
                                i ? "," : "", e[i].name, e[i].dtype, e[i].rows, e[i].cols, offset, offset + bytes);
        offset += bytes;
    }
    len += (size_t)snprintf(json + len, sizeof(json) - len, "}");
    CHECK(len < sizeof(json));
    unsigned char prefix[8];
    for (int i = 0; i < 8; i++) prefix[i] = (unsigned char)((unsigned long long)len >> (8 * i));
    void *zeros = calloc(1, offset ? offset : 1);
    FILE *f = fopen(path, "wb");
    CHECK(f && zeros);
    CHECK(fwrite(prefix, 1, 8, f) == 8 && fwrite(json, 1, len, f) == len && fwrite(zeros, 1, offset, f) == offset);
    fclose(f);
    free(zeros);
}

/* emb_in.proj_in (U32 [2560, 64] + F16 [2560, 4] scales / biases in the
 * export) with a tensor of the wrong size: open fails with `expected`
 * instead of the host dequantise reading past the file's buffers. */
static void expect_bad_emb(const char *weight_dims, const char *scales_dims, const char *biases_dims,
                           const char *expected) {
    char dir[1024], path[1200];
    snprintf(dir, sizeof(dir), "%s/h3_sv2_bad_emb.XXXXXX", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    CHECK(mkdtemp(dir) != NULL);
    snprintf(path, sizeof(path), "%s/config.json", dir);
    FILE *f = fopen(path, "wb");
    CHECK(f);
    fputs("{\"model_type\":\"seedvr2\",\"variant\":\"seedvr2-3b\",\"quantization\":{\"bits\":8,\"group_size\":64}}", f);
    fclose(f);
    st_entry e[3] = {{"emb_in.proj_in.weight", "U32", 1, 1}, {"emb_in.proj_in.scales", "F16", 1, 1},
                     {"emb_in.proj_in.biases", "F16", 1, 1}};
    const char *dims[3] = {weight_dims, scales_dims, biases_dims};
    for (int i = 0; i < 3; i++) CHECK(sscanf(dims[i], "%llux%llu", &e[i].rows, &e[i].cols) == 2);
    snprintf(path, sizeof(path), "%s/transformer.safetensors", dir);
    write_safetensors(path, e, 3);
    snprintf(path, sizeof(path), "%s/vae.safetensors", dir);
    write_safetensors(path, NULL, 0);
    snprintf(path, sizeof(path), "%s/pos_emb.safetensors", dir);
    write_safetensors(path, NULL, 0);
    char error[512] = "";
    h3_seedvr2_open_params params = {"h3_shaders.metal", 1, 42, -1, 0, -1};
    h3_seedvr2 *m = h3_seedvr2_open(dir, &params, NULL, NULL, error, sizeof(error));
    if (m) h3_seedvr2_close(m);
    if (!strstr(error, expected)) fprintf(stderr, "  open: \"%s\", expected \"%s\"\n", error, expected);
    CHECK(!m && strstr(error, expected));
    const char *names[] = {"config.json", "transformer.safetensors", "vae.safetensors", "pos_emb.safetensors"};
    for (int i = 0; i < 4; i++) { snprintf(path, sizeof(path), "%s/%s", dir, names[i]); remove(path); }
    remove(dir);
}

static void test_bad_emb(void) {
    /* h3_seedvr2_open makes its GPU before it reads emb_in. */
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { printf("skip: SeedVR2 mis-sized emb_in tensors (no GPU: %s)\n", error); return; }
    h3_gpu_free(gpu);
    expect_bad_emb("1x1", "1x1", "1x1", "emb_in.proj_in.weight is not a U32 [2560, 64] 8-bit weight");
    expect_bad_emb("2560x16", "2560x4", "2560x4", "emb_in.proj_in.weight is not a U32 [2560, 64] 8-bit weight");
    expect_bad_emb("2560x64", "1x1", "2560x4", "emb_in.proj_in.scales has an unexpected group size");
    expect_bad_emb("2560x64", "2560x4", "1x1", "emb_in.proj_in.biases has an unexpected group size");
    printf("ok: SeedVR2 rejects mis-sized emb_in tensors (%d checks)\n", tests_run);
}

/* A vae.safetensors holding only encoder.conv_in: weight (O, kt, kh, kw, I)
 * = [128, depth, 3, 3, 3] and bias [128], F16 zeros. */
static int write_conv_in(const char *path, int depth) {
    size_t wbytes = (size_t)128 * depth * 3 * 3 * 3 * 2, bbytes = 128 * 2;
    char json[512];
    int len = snprintf(json, sizeof(json),
                       "{\"encoder.conv_in.weight\":{\"dtype\":\"F16\",\"shape\":[128,%d,3,3,3],\"data_offsets\":[0,%zu]},"
                       "\"encoder.conv_in.bias\":{\"dtype\":\"F16\",\"shape\":[128],\"data_offsets\":[%zu,%zu]}}",
                       depth, wbytes, wbytes, wbytes + bbytes);
    if (len <= 0 || (size_t)len >= sizeof(json)) return 0;
    unsigned char prefix[8];
    for (int i = 0; i < 8; i++) prefix[i] = (unsigned char)((uint64_t)len >> (8 * i));
    void *zeros = calloc(wbytes + bbytes, 1);
    FILE *f = fopen(path, "wb");
    int ok = zeros && f && fwrite(prefix, 1, 8, f) == 8 && fwrite(json, 1, (size_t)len, f) == (size_t)len &&
             fwrite(zeros, 1, wbytes + bbytes, f) == wbytes + bbytes;
    if (f && fclose(f) != 0) ok = 0;
    free(zeros);
    return ok;
}

/* The grid conv (gconv_t) holds at most 3 * 3 * 3 taps: a temporal depth
 * above 3 is refused at load; depths 3 and 2 still load (this file ends
 * after conv_in, so the load stops at the first resnet's missing tensors). */
static void test_vae_kernel_depth(void) {
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { printf("skip: SeedVR2 VAE kernel depth (no GPU: %s)\n", error); return; }
    char dir[] = "/tmp/h3sv2-depth-test-XXXXXX", path[128];
    CHECK(mkdtemp(dir) != NULL);
    snprintf(path, sizeof(path), "%s/vae.safetensors", dir);
    static const int depths[3] = {4, 3, 2};
    for (int i = 0; i < 3; i++) {
        CHECK(write_conv_in(path, depths[i]));
        h3_st_header h;
        CHECK(h3_st_read_header(path, &h, error, sizeof(error)));
        for (int nax = 0; nax <= h3_gpu_has_tensor_ops(gpu); nax++)
            for (int temporal = 0; temporal < 2; temporal++) {
                h3_seedvr2 *m = calloc(1, sizeof(*m));
                CHECK(m != NULL);
                m->gpu = gpu; m->nax_vae = nax; m->vae_temporal = temporal;
                error[0] = '\0';
                CHECK(!sv2_load_vae(m, &h, error, sizeof(error)));
                if (depths[i] > 3)
                    CHECK(strstr(error, "encoder.conv_in.weight has 4 temporal taps, at most 3") != NULL);
                else
                    CHECK(strstr(error, "encoder.down_blocks.0.resnets.0.norm1.weight is missing") != NULL);
                sv2_free_vae(m);
                free(m);
            }
        h3_st_free_header(&h);
    }
    unlink(path);
    rmdir(dir);
    h3_gpu_free(gpu);
    printf("ok: SeedVR2 VAE temporal kernel depth (%d checks)\n", tests_run);
}


static float *load_tensor(const h3_st_header *h, const char *name, size_t expected) {
    const h3_st_tensor *t = h3_st_find(h, name);
    if (!t || t->dtype != H3_DTYPE_F32) { fprintf(stderr, "fixture: %s missing or not F32\n", name); exit(1); }
    size_t n = (size_t)h3_st_tensor_elements(t);
    if (expected && n != expected) { fprintf(stderr, "fixture: %s has %zu values, expected %zu\n", name, n, expected); exit(1); }
    float *data = malloc(n * sizeof(float));
    char error[256];
    if (!h3_st_read_data(h, t, data, n * sizeof(float), error, sizeof(error))) { fprintf(stderr, "fixture: %s\n", error); exit(1); }
    return data;
}

typedef struct { double max_abs, rms, ref_rms, psnr; } diff_stats;

static diff_stats compare(const float *a, const float *b, size_t n, double peak) {
    diff_stats d = {0, 0, 0, 0};
    double se = 0, sr = 0;
    for (size_t i = 0; i < n; i++) {
        double e = (double)a[i] - b[i];
        if (fabs(e) > d.max_abs) d.max_abs = fabs(e);
        se += e * e; sr += (double)b[i] * b[i];
        if (isnan(a[i]) || isinf(a[i])) { d.max_abs = INFINITY; }
    }
    d.rms = sqrt(se / (double)n); d.ref_rms = sqrt(sr / (double)n);
    d.psnr = se > 0 ? 10 * log10(peak * peak * (double)n / se) : 999;
    return d;
}

static void report(const char *stage, diff_stats d, double limit_rel) {
    printf("  %-22s max abs %.5f  rms %.5f (ref rms %.4f, rel %.4f)  psnr %.1f dB\n",
           stage, d.max_abs, d.rms, d.ref_rms, d.ref_rms > 0 ? d.rms / d.ref_rms : 0, d.psnr);
    CHECK(d.max_abs < INFINITY);
    CHECK(d.ref_rms == 0 || d.rms / d.ref_rms < limit_rel);
}

static int exists(const char *path) { struct stat st; return stat(path, &st) == 0; }

int main(void) {
    test_windows();
    printf("ok: SeedVR2 window partition (%d checks)\n", tests_run);
    test_bad_emb();
    test_vae_kernel_depth();
    const char *dir = getenv("H3_SEEDVR2_DIR");
    char fallback[1024];
    if (!dir) {
        snprintf(fallback, sizeof(fallback), "%s/models/seedvr2/SeedVR2-3B-mlx-int8", getenv("HOME") ? getenv("HOME") : ".");
        if (!exists(fallback))
            snprintf(fallback, sizeof(fallback), "%s/Library/Application Support/H3cApp/Upscalers/SeedVR2",
                     getenv("HOME") ? getenv("HOME") : ".");
        dir = fallback;
    }
    const char *fixture = getenv("H3_SEEDVR2_FIXTURE");
    char cfg[1200];
    snprintf(cfg, sizeof(cfg), "%s/config.json", dir);
    if (!exists(cfg) || !fixture || !exists(fixture)) {
        printf("skip: SeedVR2 weights (%s) or fixture (H3_SEEDVR2_FIXTURE) are not installed\n", dir);
        return 0;
    }
    h3_st_header h;
    char error[512];
    if (!h3_st_read_header(fixture, &h, error, sizeof(error))) { fprintf(stderr, "%s\n", error); return 1; }
    const h3_st_tensor *xt = h3_st_find(&h, "x");
    CHECK(xt && xt->ndim == 5);
    int H = (int)xt->shape[3], W = (int)xt->shape[4], lh = H / 8, lw = W / 8;
    size_t plane = (size_t)W * H, lplane = (size_t)lw * lh;
    float *x = load_tensor(&h, "x", 3 * plane);
    float *enc_ref = load_tensor(&h, "enc_latent", 16 * lplane);
    float *dit_in = load_tensor(&h, "dit_in", 33 * lplane);
    float *pred_ref = load_tensor(&h, "pred", 16 * lplane);
    float *latents_ref = load_tensor(&h, "latents", 16 * lplane);
    float *decoded_ref = load_tensor(&h, "decoded", 3 * plane);
    float *cc_ref = load_tensor(&h, "decoded_cc", 3 * plane);
    float *block0_ref = h3_st_find(&h, "block0_vid") ? load_tensor(&h, "block0_vid", 0) : NULL;
    printf("fixture %s: %dx%d, latent %dx%d\n", fixture, W, H, lw, lh);

    h3_seedvr2_open_params params = {"h3_shaders.metal", 1, 42, -1, getenv("H3_SV2_TEST_VAE_F32") != NULL, -1};
    h3_seedvr2 *m = h3_seedvr2_open(dir, &params, NULL, NULL, error, sizeof(error));
    if (!m) { fprintf(stderr, "open: %s\n", error); return 1; }
    h3_seedvr2_info info;
    CHECK(h3_seedvr2_inspect_dir(dir, &info, error, sizeof(error)));
    printf("model %s %s %s, %.2f GB on disk, %.2f GB resident, tensor ops %d\n", info.name, info.variant, info.dtype,
           (double)info.file_bytes / 1e9, (double)info.resident_bytes / 1e9, info.tensor_ops);

    /* 0. encoder stages, when the dump has them (enc_stage0..6, [1, C, 1, h, w]) */
    for (int stage = 0; stage <= 6; stage++) {
        char key[32];
        snprintf(key, sizeof(key), "enc_stage%d", stage);
        const h3_st_tensor *t = h3_st_find(&h, key);
        if (!t) continue;
        float *ref = load_tensor(&h, key, 0);
        size_t n = (size_t)h3_st_tensor_elements(t);
        float *ours = malloc(n * sizeof(float));
        int ow, oh, oc;
        CHECK(h3_seedvr2_debug_encode_stage(m, x, W, H, stage, ours, &ow, &oh, &oc, n, error, sizeof(error)) ||
              (fprintf(stderr, "%s\n", error), 0));
        CHECK((size_t)ow * oh * oc == n && oc == (int)t->shape[1] && oh == (int)t->shape[3] && ow == (int)t->shape[4]);
        report(key, compare(ours, ref, n, 1.0), stage < 6 ? 0.2 : 0.05);
        free(ref); free(ours);
    }

    /* 1. encoder */
    float *enc = malloc(16 * lplane * sizeof(float));
    CHECK(h3_seedvr2_debug_encode(m, x, W, H, enc, error, sizeof(error)) || (fprintf(stderr, "%s\n", error), 0));
    report("encode latent", compare(enc, enc_ref, 16 * lplane, 1.0), 0.05);

    /* 2. transformer on the reference's input (noise + its latent) */
    float *pred = malloc(16 * lplane * sizeof(float));
    const char *dump_env = getenv("H3_SV2_DUMP");
    int dump_block = block0_ref && dump_env ? 0 : -1;
    CHECK(h3_seedvr2_debug_dit(m, dit_in, lw, lh, pred, dump_block, error, sizeof(error)) || (fprintf(stderr, "%s\n", error), 0));
    if (dump_block == 0) {
        FILE *f = fopen(dump_env, "rb");
        size_t n = (size_t)(lw / 2) * (lh / 2) * 2560;
        float *b0 = malloc(n * sizeof(float));
        if (f && b0 && fread(b0, sizeof(float), n, f) == n)
            report("block 0 (vid rows)", compare(b0, block0_ref, n, 1.0), 0.1);
        if (f) fclose(f);
        free(b0);
    }
    double dit_tol = getenv("H3_SV2_TEST_TOL") ? atof(getenv("H3_SV2_TEST_TOL")) : 0.05;
    report("dit prediction", compare(pred, pred_ref, 16 * lplane, 1.0), dit_tol);
    /* z = noise - v */
    float *z = malloc(16 * lplane * sizeof(float));
    for (size_t i = 0; i < 16 * lplane; i++) z[i] = dit_in[i] - pred[i];
    report("latents (noise - v)", compare(z, latents_ref, 16 * lplane, 1.0), dit_tol);

    /* 3. decoder on the reference latents */
    float *dec = malloc(3 * plane * sizeof(float));
    CHECK(h3_seedvr2_debug_decode(m, latents_ref, lw, lh, dec, error, sizeof(error)) || (fprintf(stderr, "%s\n", error), 0));
    report("decode (ref latents)", compare(dec, decoded_ref, 3 * plane, 2.0), 0.05);

    /* 4. colour transfer on the reference's decoded frame */
    float *cc = malloc(3 * plane * sizeof(float));
    CHECK(h3_seedvr2_debug_color(m, decoded_ref, x, W, H, cc, error, sizeof(error)) || (fprintf(stderr, "%s\n", error), 0));
    report("colour transfer", compare(cc, cc_ref, 3 * plane, 2.0), 0.05);

    /* 5. the whole chain from our own stages: decode(noise - dit(enc)) */
    for (size_t i = 0; i < 16 * lplane; i++) dit_in[16 * lplane + i] = enc[i];
    CHECK(h3_seedvr2_debug_dit(m, dit_in, lw, lh, pred, -1, error, sizeof(error)));
    for (size_t i = 0; i < 16 * lplane; i++) z[i] = dit_in[i] - pred[i];
    CHECK(h3_seedvr2_debug_decode(m, z, lw, lh, dec, error, sizeof(error)));
    report("end to end (pre-colour)", compare(dec, decoded_ref, 3 * plane, 2.0), dit_tol * 2);
    CHECK(h3_seedvr2_debug_color(m, dec, x, W, H, cc, error, sizeof(error)));
    report("end to end (colour)", compare(cc, cc_ref, 3 * plane, 2.0), dit_tol * 2);

    double gpu_seconds; uint64_t peak;
    h3_seedvr2_get_stats(m, &gpu_seconds, &peak);
    printf("gpu %.2f s, peak live %.2f GiB\n", gpu_seconds, (double)peak / 1073741824.0);
    h3_seedvr2_close(m);
    h3_st_free_header(&h);
    printf("ok: SeedVR2 stages match the reference (%d checks)\n", tests_run);
    return 0;
}
