/* The ReAE autoencoder on the GPU against a plain double-precision
 * transcription of the official reae.py on synthetic weights: encoder and
 * decoder on a small clip, then the same clip in two chunks (and in frame
 * batches) with the MemBlock state carried, which must reproduce the
 * single-pass result. Needs a GPU, no real weights; runs in `make test`. */
#include "h3_vsr_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int tests_run;

#define CHECK(condition) do { \
    tests_run++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

/* ---- deterministic values ---- */
static uint64_t rng_state = 0x243F6A8885A308D3ull;
static float rnd(void) {
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((rng_state >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f;
}

/* ---- safetensors writer ---- */
typedef struct { char name[96]; int ndim; uint64_t shape[5]; float *data; size_t count; } tensor_spec;
static tensor_spec specs[256];
static int spec_count;

static float *add_tensor(const char *name, int ndim, const uint64_t *shape, float scale) {
    size_t n = 1;
    for (int d = 0; d < ndim; d++) n *= shape[d];
    float *v = malloc(n * sizeof(float));
    CHECK(v != NULL);
    for (size_t i = 0; i < n; i++) v[i] = rnd() * scale;
    tensor_spec *t = &specs[spec_count++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->ndim = ndim;
    memcpy(t->shape, shape, sizeof(uint64_t) * (size_t)ndim);
    t->data = v;
    t->count = n;
    return v;
}

static void write_safetensors(const char *path) {
    char header[65536];
    size_t at = 0;
    uint64_t offset = 0;
    at += (size_t)snprintf(header + at, sizeof(header) - at, "{");
    for (int i = 0; i < spec_count; i++) {
        tensor_spec *t = &specs[i];
        at += (size_t)snprintf(header + at, sizeof(header) - at, "%s\"%s\":{\"dtype\":\"F32\",\"shape\":[",
                               i ? "," : "", t->name);
        for (int d = 0; d < t->ndim; d++)
            at += (size_t)snprintf(header + at, sizeof(header) - at, "%s%llu", d ? "," : "",
                                   (unsigned long long)t->shape[d]);
        at += (size_t)snprintf(header + at, sizeof(header) - at, "],\"data_offsets\":[%llu,%llu]}",
                               (unsigned long long)offset, (unsigned long long)(offset + t->count * 4));
        offset += t->count * 4;
    }
    at += (size_t)snprintf(header + at, sizeof(header) - at, "}");
    while (at % 8) header[at++] = ' ';
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    uint64_t length = at;
    CHECK(fwrite(&length, 8, 1, f) == 1 && fwrite(header, 1, at, f) == at);
    for (int i = 0; i < spec_count; i++) CHECK(fwrite(specs[i].data, 4, specs[i].count, f) == specs[i].count);
    fclose(f);
}

/* ---- the synthetic ReAE ---- */
typedef struct { float *w, *b; } conv_w;            /* [O, I, kh, kw] (+ [O]) */
typedef struct { conv_w c0, c2, c4; } memblock_w;    /* c0 takes 2C channels */
typedef struct {
    conv_w e0, e3, e8, e13, e17;
    float *tpool2, *tpool7, *tpool12;                /* [C, sC, 1, 1] */
    memblock_w emb[9];
    conv_w d1, d8, d14, d20, d22;
    float *tgrow7;                                   /* [512, 512, 1, 1] */
    float *tgrow13, *tgrow19;                        /* [C, C, 3, 1, 1] */
    memblock_w dmb[9];
} reae_weights;

static conv_w conv_spec(const char *prefix, int out, int in, int bias) {
    char name[128];
    conv_w c;
    uint64_t s[4] = {(uint64_t)out, (uint64_t)in, 3, 3};
    snprintf(name, sizeof(name), "%s.weight", prefix);
    c.w = add_tensor(name, 4, s, 1.0f / sqrtf((float)in * 9.0f) * 1.7f);
    c.b = NULL;
    if (bias) {
        uint64_t sb[1] = {(uint64_t)out};
        snprintf(name, sizeof(name), "%s.bias", prefix);
        c.b = add_tensor(name, 1, sb, 0.05f);
    }
    return c;
}

static memblock_w memblock_spec(const char *prefix, int channels) {
    char name[128];
    memblock_w m;
    snprintf(name, sizeof(name), "%s.conv.0", prefix);
    m.c0 = conv_spec(name, channels, 2 * channels, 1);
    snprintf(name, sizeof(name), "%s.conv.2", prefix);
    m.c2 = conv_spec(name, channels, channels, 1);
    snprintf(name, sizeof(name), "%s.conv.4", prefix);
    m.c4 = conv_spec(name, channels, channels, 1);
    return m;
}

static float *tpool_spec(const char *prefix, int channels, int stride) {
    char name[128];
    uint64_t s[4] = {(uint64_t)channels, (uint64_t)stride * channels, 1, 1};
    snprintf(name, sizeof(name), "%s.conv.weight", prefix);
    return add_tensor(name, 4, s, 1.0f / sqrtf((float)(stride * channels)) * 1.5f);
}

static void build_model(const char *dir, reae_weights *w) {
    const int E = 64;
    spec_count = 0;
    w->e0 = conv_spec("encoder.0", E, 12, 1);
    w->tpool2 = tpool_spec("encoder.2", E, 2);
    w->e3 = conv_spec("encoder.3", E, E, 0);
    for (int i = 0; i < 3; i++) { char p[32]; snprintf(p, sizeof(p), "encoder.%d", 4 + i); w->emb[i] = memblock_spec(p, E); }
    w->tpool7 = tpool_spec("encoder.7", E, 2);
    w->e8 = conv_spec("encoder.8", E, E, 0);
    for (int i = 0; i < 3; i++) { char p[32]; snprintf(p, sizeof(p), "encoder.%d", 9 + i); w->emb[3 + i] = memblock_spec(p, E); }
    w->tpool12 = tpool_spec("encoder.12", E, 1);
    w->e13 = conv_spec("encoder.13", E, E, 0);
    for (int i = 0; i < 3; i++) { char p[32]; snprintf(p, sizeof(p), "encoder.%d", 14 + i); w->emb[6 + i] = memblock_spec(p, E); }
    w->e17 = conv_spec("encoder.17", 48, E, 1);
    w->d1 = conv_spec("decoder.1", 512, 48, 1);
    for (int i = 0; i < 3; i++) { char p[32]; snprintf(p, sizeof(p), "decoder.%d", 3 + i); w->dmb[i] = memblock_spec(p, 512); }
    { uint64_t s[4] = {512, 512, 1, 1}; w->tgrow7 = add_tensor("decoder.7.proj.weight", 4, s, 1.0f / sqrtf(512.0f) * 1.5f); }
    w->d8 = conv_spec("decoder.8", 256, 512, 0);
    for (int i = 0; i < 3; i++) { char p[32]; snprintf(p, sizeof(p), "decoder.%d", 9 + i); w->dmb[3 + i] = memblock_spec(p, 256); }
    { uint64_t s[5] = {256, 256, 3, 1, 1}; w->tgrow13 = add_tensor("decoder.13.conv3d.weight", 5, s, 1.0f / sqrtf(256.0f * 2.0f)); }
    w->d14 = conv_spec("decoder.14", 128, 256, 0);
    for (int i = 0; i < 3; i++) { char p[32]; snprintf(p, sizeof(p), "decoder.%d", 15 + i); w->dmb[6 + i] = memblock_spec(p, 128); }
    { uint64_t s[5] = {128, 128, 3, 1, 1}; w->tgrow19 = add_tensor("decoder.19.conv3d.weight", 5, s, 1.0f / sqrtf(128.0f * 2.0f)); }
    w->d20 = conv_spec("decoder.20", 64, 128, 0);
    w->d22 = conv_spec("decoder.22", 12, 64, 1);
    char path[4096];
    snprintf(path, sizeof(path), "%s/reae.safetensors", dir);
    write_safetensors(path);
}

/* ---- double reference, channels-last [T][H][W][C] ---- */
typedef struct { int t, h, w, c; double *v; } ref;

static ref ref_new(int t, int h, int w, int c) {
    ref r = {t, h, w, c, calloc((size_t)t * h * w * c, sizeof(double))};
    CHECK(r.v != NULL);
    return r;
}
static double *at(ref *r, int t, int y, int x, int c) {
    return r->v + (((size_t)t * r->h + y) * r->w + x) * r->c + c;
}

/* conv 3x3, zero pad 1, stride s, per frame; input channels may be a
 * concatenation given as two refs (x then past), for MemBlock. */
static ref ref_conv3x3(const ref *x, const ref *past_or_null, const conv_w *w, int out_ch, int stride) {
    int in_ch = x->c + (past_or_null ? past_or_null->c : 0);
    ref o = ref_new(x->t, x->h / stride, x->w / stride, out_ch);
    for (int t = 0; t < x->t; t++)
        for (int oy = 0; oy < o.h; oy++)
            for (int ox = 0; ox < o.w; ox++)
                for (int oc = 0; oc < out_ch; oc++) {
                    double acc = w->b ? w->b[oc] : 0.0;
                    for (int ky = 0; ky < 3; ky++) {
                        int iy = oy * stride - 1 + ky;
                        if (iy < 0 || iy >= x->h) continue;
                        for (int kx = 0; kx < 3; kx++) {
                            int ix = ox * stride - 1 + kx;
                            if (ix < 0 || ix >= x->w) continue;
                            for (int ic = 0; ic < in_ch; ic++) {
                                double value = ic < x->c
                                    ? *at((ref *)x, t, iy, ix, ic)
                                    : (t == 0 ? 0.0 : *at((ref *)past_or_null, t - 1, iy, ix, ic - x->c));
                                acc += value * w->w[(((size_t)oc * in_ch + ic) * 3 + ky) * 3 + kx];
                            }
                        }
                    }
                    *at(&o, t, oy, ox, oc) = acc;
                }
    return o;
}

static void ref_relu(ref *r) { size_t n = (size_t)r->t * r->h * r->w * r->c; for (size_t i = 0; i < n; i++) if (r->v[i] < 0) r->v[i] = 0; }

/* MemBlock(x): past = x shifted one frame (zeros first); the reference's
 * cat([x, past]) along channels is handled by ref_conv3x3's second input. */
static ref ref_memblock(const ref *x, const memblock_w *m) {
    ref a = ref_conv3x3(x, x, &m->c0, x->c, 1);
    ref_relu(&a);
    ref b = ref_conv3x3(&a, NULL, &m->c2, x->c, 1);
    ref_relu(&b);
    ref c = ref_conv3x3(&b, NULL, &m->c4, x->c, 1);
    size_t n = (size_t)x->t * x->h * x->w * x->c;
    for (size_t i = 0; i < n; i++) { c.v[i] += x->v[i]; if (c.v[i] < 0) c.v[i] = 0; }
    free(a.v); free(b.v);
    return c;
}

static ref ref_tpool(const ref *x, const float *w, int stride) {
    ref o = ref_new(x->t / stride, x->h, x->w, x->c);
    for (int t = 0; t < o.t; t++)
        for (int y = 0; y < x->h; y++)
            for (int xx = 0; xx < x->w; xx++)
                for (int oc = 0; oc < x->c; oc++) {
                    double acc = 0.0;
                    for (int f = 0; f < stride; f++)
                        for (int ic = 0; ic < x->c; ic++)
                            acc += *at((ref *)x, t * stride + f, y, xx, ic) * w[(size_t)oc * stride * x->c + (size_t)f * x->c + ic];
                    *at(&o, t, y, xx, oc) = acc;
                }
    return o;
}

static ref ref_pointwise(const ref *x, const float *w, int out_ch) {   /* [out, in, 1, 1] */
    ref o = ref_new(x->t, x->h, x->w, out_ch);
    for (int t = 0; t < x->t; t++)
        for (int y = 0; y < x->h; y++)
            for (int xx = 0; xx < x->w; xx++)
                for (int oc = 0; oc < out_ch; oc++) {
                    double acc = 0.0;
                    for (int ic = 0; ic < x->c; ic++) acc += *at((ref *)x, t, y, xx, ic) * w[(size_t)oc * x->c + ic];
                    *at(&o, t, y, xx, oc) = acc;
                }
    return o;
}

/* TGrow(2) literally: each frame -> [x, x] along a stride axis of length 2,
 * conv3d (3,1,1) pad (1,0,0) over that axis. */
static ref ref_tgrow2(const ref *x, const float *w) {
    ref o = ref_new(x->t * 2, x->h, x->w, x->c);
    for (int t = 0; t < x->t; t++)
        for (int d = 0; d < 2; d++)
            for (int y = 0; y < x->h; y++)
                for (int xx = 0; xx < x->w; xx++)
                    for (int oc = 0; oc < x->c; oc++) {
                        double acc = 0.0;
                        for (int k = 0; k < 3; k++) {
                            int src = d + k - 1;
                            if (src < 0 || src > 1) continue;
                            for (int ic = 0; ic < x->c; ic++)
                                acc += *at((ref *)x, t, y, xx, ic) * w[((size_t)oc * x->c + ic) * 3 + k];
                        }
                        *at(&o, t * 2 + d, y, xx, oc) = acc;
                    }
    return o;
}

static ref ref_upsample(const ref *x) {
    ref o = ref_new(x->t, x->h * 2, x->w * 2, x->c);
    for (int t = 0; t < x->t; t++) for (int y = 0; y < o.h; y++) for (int xx = 0; xx < o.w; xx++)
        for (int c = 0; c < x->c; c++) *at(&o, t, y, xx, c) = *at((ref *)x, t, y / 2, xx / 2, c);
    return o;
}

static ref ref_unshuffle(const ref *x) {
    ref o = ref_new(x->t, x->h / 2, x->w / 2, x->c * 4);
    for (int t = 0; t < x->t; t++) for (int y = 0; y < o.h; y++) for (int xx = 0; xx < o.w; xx++)
        for (int c = 0; c < x->c; c++) for (int ph = 0; ph < 2; ph++) for (int pw = 0; pw < 2; pw++)
            *at(&o, t, y, xx, (c * 2 + ph) * 2 + pw) = *at((ref *)x, t, y * 2 + ph, xx * 2 + pw, c);
    return o;
}

static ref ref_shuffle_clamp(const ref *x) {
    ref o = ref_new(x->t, x->h * 2, x->w * 2, x->c / 4);
    for (int t = 0; t < x->t; t++) for (int y = 0; y < x->h; y++) for (int xx = 0; xx < x->w; xx++)
        for (int c = 0; c < o.c; c++) for (int ph = 0; ph < 2; ph++) for (int pw = 0; pw < 2; pw++) {
            double v = *at((ref *)x, t, y, xx, (c * 2 + ph) * 2 + pw);
            *at(&o, t, y * 2 + ph, xx * 2 + pw, c) = v < 0 ? 0 : v > 1 ? 1 : v;
        }
    return o;
}

#define SWAP(a, b) do { ref tmp_ = (a); (a) = (b); free(tmp_.v); } while (0)

static ref ref_encode(const ref *frames, const reae_weights *w) {
    ref x = ref_unshuffle(frames);
    ref y = ref_conv3x3(&x, NULL, &w->e0, 64, 1); SWAP(x, y); ref_relu(&x);
    y = ref_tpool(&x, w->tpool2, 2); SWAP(x, y);
    y = ref_conv3x3(&x, NULL, &w->e3, 64, 2); SWAP(x, y);
    for (int i = 0; i < 3; i++) { y = ref_memblock(&x, &w->emb[i]); SWAP(x, y); }
    y = ref_tpool(&x, w->tpool7, 2); SWAP(x, y);
    y = ref_conv3x3(&x, NULL, &w->e8, 64, 2); SWAP(x, y);
    for (int i = 3; i < 6; i++) { y = ref_memblock(&x, &w->emb[i]); SWAP(x, y); }
    y = ref_tpool(&x, w->tpool12, 1); SWAP(x, y);
    y = ref_conv3x3(&x, NULL, &w->e13, 64, 2); SWAP(x, y);
    for (int i = 6; i < 9; i++) { y = ref_memblock(&x, &w->emb[i]); SWAP(x, y); }
    y = ref_conv3x3(&x, NULL, &w->e17, 48, 1); SWAP(x, y);
    return x;
}

static ref ref_decode(const ref *latent, const reae_weights *w) {
    ref x = ref_new(latent->t, latent->h, latent->w, latent->c);
    size_t n = (size_t)latent->t * latent->h * latent->w * latent->c;
    for (size_t i = 0; i < n; i++) x.v[i] = tanh(latent->v[i] / 3.0) * 3.0;
    ref y = ref_conv3x3(&x, NULL, &w->d1, 512, 1); SWAP(x, y); ref_relu(&x);
    for (int i = 0; i < 3; i++) { y = ref_memblock(&x, &w->dmb[i]); SWAP(x, y); }
    y = ref_upsample(&x); SWAP(x, y);
    y = ref_pointwise(&x, w->tgrow7, 512); SWAP(x, y);
    y = ref_conv3x3(&x, NULL, &w->d8, 256, 1); SWAP(x, y);
    for (int i = 3; i < 6; i++) { y = ref_memblock(&x, &w->dmb[i]); SWAP(x, y); }
    y = ref_upsample(&x); SWAP(x, y);
    y = ref_tgrow2(&x, w->tgrow13); SWAP(x, y);
    y = ref_conv3x3(&x, NULL, &w->d14, 128, 1); SWAP(x, y);
    for (int i = 6; i < 9; i++) { y = ref_memblock(&x, &w->dmb[i]); SWAP(x, y); }
    y = ref_upsample(&x); SWAP(x, y);
    y = ref_tgrow2(&x, w->tgrow19); SWAP(x, y);
    y = ref_conv3x3(&x, NULL, &w->d20, 64, 1); SWAP(x, y); ref_relu(&x);
    y = ref_conv3x3(&x, NULL, &w->d22, 12, 1); SWAP(x, y);
    y = ref_shuffle_clamp(&x); SWAP(x, y);
    return x;
}

/* ---- comparison ---- */
static void compare(const float *got, const double *want, size_t n, double tolerance, const char *what) {
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)got[i] - want[i]);
        if (d > worst) worst = d;
        if (fabs(want[i]) > scale) scale = fabs(want[i]);
    }
    printf("  %s: max |diff| %.3g (max |ref| %.3g)\n", what, worst, scale);
    CHECK(worst <= tolerance * (scale > 1.0 ? scale : 1.0));
}

static void compare_f32(const float *a, const float *b, size_t n, double tolerance, const char *what) {
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (d > worst) worst = d; }
    printf("  %s: max |diff| %.3g\n", what, worst);
    CHECK(worst <= tolerance);
}

static double psnr(const float *a, const float *b, size_t n) {
    double err = 0.0, peak = 0.0;
    for (size_t i = 0; i < n; i++) { double d = (double)a[i] - b[i]; err += d * d; if (fabs(b[i]) > peak) peak = fabs(b[i]); }
    return 10.0 * log10(peak * peak / (err / (double)n > 0 ? err / (double)n : 1e-30));
}

int main(void) {
    char dir[] = "/tmp/h3vsr-reae-test-XXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    /* The double transcription is checked on the exact F32 path; the
     * TensorOps (BF16) path is compared with it afterwards. */
    setenv("H3_VSR_REAE_F32", "1", 1);
    reae_weights w;
    build_model(dir, &w);

    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    CHECK(gpu != NULL);
    h3_vsr_reae *reae = h3_vsr_reae_load(dir, gpu, error, sizeof(error));
    if (!reae) fprintf(stderr, "%s\n", error);
    CHECK(reae != NULL);

    /* A smooth 8-frame 32x32 clip in [0, 1]. */
    const int T = 8, H = 32, W = 32;
    ref frames = ref_new(T, H, W, 3);
    float *frames_f = malloc((size_t)T * H * W * 3 * sizeof(float));
    for (int t = 0; t < T; t++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) for (int c = 0; c < 3; c++) {
        double v = 0.5 + 0.4 * sin(0.3 * x + 0.2 * y + 0.5 * t + c) * cos(0.17 * y - 0.11 * t);
        *at(&frames, t, y, x, c) = v;
        frames_f[(((size_t)t * H + y) * W + x) * 3 + c] = (float)v;
    }

    /* Encoder: whole clip vs reference. */
    ref latent_ref = ref_encode(&frames, &w);
    CHECK(latent_ref.t == T / 4 && latent_ref.h == H / 16 && latent_ref.w == W / 16 && latent_ref.c == 48);
    h3_gpu_tensor *frames_t = h3_gpu_tensor_from_f32(gpu, frames_f, (size_t)T * H * W * 3);
    size_t latent_n = (size_t)(T / 4) * (H / 16) * (W / 16) * 48;
    h3_gpu_tensor *latent_t = h3_gpu_tensor_new_f32(gpu, latent_n);
    CHECK(frames_t && latent_t);
    h3_vsr_reae_state *enc_state = h3_vsr_reae_state_new(reae, 1, H, W);
    CHECK(enc_state != NULL);
    if (!h3_vsr_reae_encode(reae, enc_state, frames_t, T, H, W, 0, latent_t, error, sizeof(error)))
        fprintf(stderr, "%s\n", error);
    CHECK(error[0] == '\0');
    float *latent_f = malloc(latent_n * sizeof(float));
    CHECK(h3_gpu_tensor_read_f32(latent_t, latent_f, latent_n));
    compare(latent_f, latent_ref.v, latent_n, 2e-4, "encoder, one pass");
    h3_vsr_reae_state_free(enc_state);

    /* Encoder in two chunks of 4 frames with the state carried, and in one
     * call with frame batches of 4: both equal the one-pass result. */
    for (int mode = 0; mode < 2; mode++) {
        h3_gpu_tensor *latent2 = h3_gpu_tensor_new_f32(gpu, latent_n);
        h3_vsr_reae_state *state = h3_vsr_reae_state_new(reae, 1, H, W);
        CHECK(latent2 && state);
        if (mode == 0) {
            h3_gpu_tensor *half = h3_gpu_tensor_new_f32(gpu, (size_t)4 * H * W * 3);
            h3_gpu_tensor *half_latent = h3_gpu_tensor_new_f32(gpu, latent_n / 2);
            CHECK(half && half_latent);
            for (int chunk = 0; chunk < 2; chunk++) {
                CHECK(h3_gpu_tensor_write_f32(half, frames_f + (size_t)chunk * 4 * H * W * 3, (size_t)4 * H * W * 3));
                CHECK(h3_vsr_reae_encode(reae, state, half, 4, H, W, 0, half_latent, error, sizeof(error)));
                float part[2048];
                CHECK(latent_n / 2 <= sizeof(part) / sizeof(part[0]));
                CHECK(h3_gpu_tensor_read_f32(half_latent, part, latent_n / 2));
                CHECK(h3_gpu_tensor_write_f32_range(latent2, (size_t)chunk * (latent_n / 2), part, latent_n / 2));
            }
            h3_gpu_tensor_free(half);
            h3_gpu_tensor_free(half_latent);
        } else {
            CHECK(h3_vsr_reae_encode(reae, state, frames_t, T, H, W, 4, latent2, error, sizeof(error)));
        }
        float *got = malloc(latent_n * sizeof(float));
        CHECK(h3_gpu_tensor_read_f32(latent2, got, latent_n));
        compare_f32(got, latent_f, latent_n, 1e-4, mode == 0 ? "encoder, two chunks" : "encoder, frame batches");
        free(got);
        h3_gpu_tensor_free(latent2);
        h3_vsr_reae_state_free(state);
    }

    /* Decoder: whole latent vs reference, then one latent step at a time. */
    ref frames_ref = ref_decode(&latent_ref, &w);
    CHECK(frames_ref.t == T && frames_ref.h == H && frames_ref.w == W && frames_ref.c == 3);
    size_t frames_n = (size_t)T * H * W * 3;
    h3_gpu_tensor *decoded = h3_gpu_tensor_new_f32(gpu, frames_n);
    h3_vsr_reae_state *dec_state = h3_vsr_reae_state_new(reae, 0, H, W);
    CHECK(decoded && dec_state);
    h3_gpu_tensor *latent_in = h3_gpu_tensor_from_f32(gpu, latent_f, latent_n);
    CHECK(latent_in != NULL);
    if (!h3_vsr_reae_decode(reae, dec_state, latent_in, T / 4, H / 16, W / 16, 0, decoded, error, sizeof(error)))
        fprintf(stderr, "%s\n", error);
    CHECK(error[0] == '\0');
    float *decoded_f = malloc(frames_n * sizeof(float));
    CHECK(h3_gpu_tensor_read_f32(decoded, decoded_f, frames_n));
    /* The reference decodes the double latent; feed the GPU the same values
     * (latent_f is the GPU encoder's output, close to the reference). */
    compare(decoded_f, frames_ref.v, frames_n, 5e-3, "decoder, one pass (vs double reference)");
    {
        h3_gpu_tensor *decoded2 = h3_gpu_tensor_new_f32(gpu, frames_n);
        h3_vsr_reae_state *state = h3_vsr_reae_state_new(reae, 0, H, W);
        CHECK(decoded2 && state);
        CHECK(h3_vsr_reae_decode(reae, state, latent_in, T / 4, H / 16, W / 16, 1, decoded2, error, sizeof(error)));
        float *got = malloc(frames_n * sizeof(float));
        CHECK(h3_gpu_tensor_read_f32(decoded2, got, frames_n));
        compare_f32(got, decoded_f, frames_n, 1e-4, "decoder, one latent step per batch");
        free(got);
        h3_gpu_tensor_free(decoded2);
        h3_vsr_reae_state_free(state);
    }
    /* Outputs are clamped to [0, 1]. */
    for (size_t i = 0; i < frames_n; i++) CHECK(decoded_f[i] >= 0.0f && decoded_f[i] <= 1.0f);

    /* The TensorOps path (M5): the same clip, within BF16 of the F32 path,
     * whole and with the state carried across two chunks. */
    if (h3_gpu_has_tensor_ops(gpu)) {
        unsetenv("H3_VSR_REAE_F32");
        h3_vsr_reae *nax = h3_vsr_reae_load(dir, gpu, error, sizeof(error));
        CHECK(nax != NULL);
        h3_gpu_tensor *latent2 = h3_gpu_tensor_new_f32(gpu, latent_n);
        h3_gpu_tensor *decoded2 = h3_gpu_tensor_new_f32(gpu, frames_n);
        float *got = malloc(frames_n * sizeof(float));
        CHECK(latent2 && decoded2 && got);
        for (int mode = 0; mode < 2; mode++) {
            h3_vsr_reae_state *state = h3_vsr_reae_state_new(nax, 1, H, W);
            CHECK(state != NULL);
            if (mode == 0) {
                CHECK(h3_vsr_reae_encode(nax, state, frames_t, T, H, W, 0, latent2, error, sizeof(error)));
            } else {
                h3_gpu_tensor *half = h3_gpu_tensor_new_f32(gpu, (size_t)4 * H * W * 3);
                h3_gpu_tensor *half_latent = h3_gpu_tensor_new_f32(gpu, latent_n / 2);
                CHECK(half && half_latent);
                for (int chunk = 0; chunk < 2; chunk++) {
                    CHECK(h3_gpu_tensor_write_f32(half, frames_f + (size_t)chunk * 4 * H * W * 3, (size_t)4 * H * W * 3));
                    CHECK(h3_vsr_reae_encode(nax, state, half, 4, H, W, 0, half_latent, error, sizeof(error)));
                    float part[2048];
                    CHECK(h3_gpu_tensor_read_f32(half_latent, part, latent_n / 2));
                    CHECK(h3_gpu_tensor_write_f32_range(latent2, (size_t)chunk * (latent_n / 2), part, latent_n / 2));
                }
                h3_gpu_tensor_free(half);
                h3_gpu_tensor_free(half_latent);
            }
            CHECK(h3_gpu_tensor_read_f32(latent2, got, latent_n));
            double db = psnr(got, latent_f, latent_n);
            printf("  TensorOps encoder, %s: %.1f dB vs the F32 path\n", mode ? "two chunks" : "one pass", db);
            CHECK(db > 40.0);
            h3_vsr_reae_state_free(state);
        }
        h3_vsr_reae_state_free(dec_state);
        dec_state = h3_vsr_reae_state_new(nax, 0, H, W);
        CHECK(dec_state != NULL);
        CHECK(h3_vsr_reae_decode(nax, dec_state, latent_in, T / 4, H / 16, W / 16, 0, decoded2, error, sizeof(error)));
        CHECK(h3_gpu_tensor_read_f32(decoded2, got, frames_n));
        double db = psnr(got, decoded_f, frames_n);
        printf("  TensorOps decoder: %.1f dB vs the F32 path\n", db);
        CHECK(db > 40.0);
        free(got);
        h3_gpu_tensor_free(latent2);
        h3_gpu_tensor_free(decoded2);
        h3_vsr_reae_free(nax);
    } else {
        h3_vsr_reae_state_free(dec_state);
    }

    h3_gpu_tensor_free(frames_t);
    h3_gpu_tensor_free(latent_t);
    h3_gpu_tensor_free(latent_in);
    h3_gpu_tensor_free(decoded);
    h3_vsr_reae_free(reae);
    h3_gpu_free(gpu);
    char command[4200];
    snprintf(command, sizeof(command), "rm -rf '%s'", dir);
    CHECK(system(command) == 0);
    printf("ok: %d checks (vsr ReAE vs double reference)\n", tests_run);
    return 0;
}
