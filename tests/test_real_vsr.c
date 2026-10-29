/* Parity of the SwiftVR engine against the official PyTorch implementation
 * run in float32 (tools/dump_swiftvr_fixture.py): window tables, RoPE, the
 * cache's precomputed constants, the DiT on the reference latents, and the
 * whole session (preprocess -> ReAE -> DiT -> ReAE -> 8-bit frames) on the
 * fixture clip, stage by stage through H3_VSR_DUMP_DIR.
 *
 * Skips (exit 0) when the fixture or the weights are not installed:
 *   H3_VSR_FIXTURE    fixture directory (default misc/fixtures/swiftvr-768x544)
 *   H3_VSR_MODEL_DIR  SwiftVR weights (default the app's Upscalers/SwiftVR)
 *   H3_VSR_PRECISION  auto (default: int8 on tensor hardware, else bf16), bf16, int8, or both
 *   H3_VSR_STAGES     comma list of host, reae, dit, session (default all)
 * The .h3vc cache is built next to the weights when it is missing. */
#include "h3_vsr_internal.h"
#include "h3_safetensors.h"
#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static int checks = 0;
#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); exit(1); } checks++; } while (0)

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9; }

typedef struct { void *data; h3_dtype dtype; int ndim; uint64_t shape[8]; size_t elements; } tensor;

static int load(const char *dir, const char *file, const char *name, tensor *t) {
    char path[4096], err[512] = {0};
    snprintf(path, sizeof(path), "%s/%s", dir, file);
    h3_st_header h;
    memset(t, 0, sizeof(*t));
    if (!h3_st_read_header(path, &h, err, sizeof(err))) { fprintf(stderr, "%s: %s\n", path, err); return 0; }
    const h3_st_tensor *st = h3_st_find(&h, name);
    if (!st) { fprintf(stderr, "%s: no tensor %s\n", path, name); h3_st_free_header(&h); return 0; }
    t->dtype = st->dtype; t->ndim = st->ndim;
    memcpy(t->shape, st->shape, sizeof(t->shape));
    t->elements = (size_t)h3_st_tensor_elements(st);
    size_t bytes = t->elements * h3_dtype_size(st->dtype);
    t->data = malloc(bytes);
    int ok = t->data && h3_st_read_data(&h, st, t->data, bytes, err, sizeof(err));
    if (!ok) fprintf(stderr, "%s: %s: %s\n", path, name, err);
    h3_st_free_header(&h);
    return ok;
}
static float *load_f32(const char *dir, const char *file, const char *name, tensor *t) {
    CHECK(load(dir, file, name, t) && t->dtype == H3_DTYPE_F32);
    return t->data;
}

typedef struct { double psnr, max_abs, rel_rms, peak; } metrics;

static metrics compare(const float *got, const float *ref, size_t n) {
    double err = 0.0, peak = 0.0, ref_sq = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)got[i] - ref[i];
        err += d * d; ref_sq += (double)ref[i] * ref[i];
        if (fabs(ref[i]) > peak) peak = fabs(ref[i]);
        if (fabs(d) > max_abs) max_abs = fabs(d);
    }
    metrics m;
    m.peak = peak; m.max_abs = max_abs;
    m.psnr = 10.0 * log10(peak * peak / (err / (double)n > 0 ? err / (double)n : 1e-30));
    m.rel_rms = sqrt(err / (ref_sq > 0 ? ref_sq : 1e-30));
    return m;
}
static void report(const char *what, metrics m) {
    printf("  %-44s %6.1f dB  max|d| %.3g  rel rms %.2e  (peak %.3g)\n", what, m.psnr, m.max_abs, m.rel_rms, m.peak);
}

/* [n, c, h, w] -> [n, h, w, c] */
static float *to_channels_last(const float *src, int n, int c, int h, int w) {
    float *dst = malloc((size_t)n * c * h * w * sizeof(float));
    CHECK(dst != NULL);
    for (int t = 0; t < n; t++)
        for (int ch = 0; ch < c; ch++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    dst[(((size_t)t * h + y) * w + x) * c + ch] = src[(((size_t)t * c + ch) * h + y) * w + x];
    return dst;
}

static float *read_raw_f32(const char *dir, int chunk, const char *stage, size_t elements) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/chunk%d_%s.f32", dir, chunk, stage);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "missing dump %s\n", path); return NULL; }
    float *data = malloc(elements * sizeof(float));
    size_t got = data ? fread(data, sizeof(float), elements, f) : 0;
    int extra = fgetc(f) != EOF;
    fclose(f);
    if (got != elements || extra) { fprintf(stderr, "%s: expected %zu floats, got %zu%s\n", path, elements, got, extra ? "+" : ""); free(data); return NULL; }
    return data;
}

static float bf16_to_f32(uint16_t v) { union { uint32_t u; float f; } c; c.u = (uint32_t)v << 16; return c.f; }

static int progress(const char *phase, int done, int total, void *opaque) {
    (void)opaque;
    if (done == 0 || done == total || done % 5 == 0) { printf("    %s %d/%d\r", phase, done, total); fflush(stdout); }
    if (done == total) printf("\n");
    return 0;
}

static int file_exists(const char *path) { struct stat st; return stat(path, &st) == 0; }

static int stage_on(const char *stage) {
    const char *stages = getenv("H3_VSR_STAGES");
    if (!stages || !stages[0]) return 1;
    size_t n = strlen(stage);
    for (const char *p = stages; (p = strstr(p, stage)) != NULL; p += n)
        if ((p == stages || p[-1] == ',') && (p[n] == '\0' || p[n] == ',')) return 1;
    return 0;
}

/* The session's input conversion: 8-bit RGB frames -> bilinear to the
 * output size, [0, 1], zero padding, channels-last F32 on the GPU. */
static h3_gpu_tensor *preprocess(h3_gpu *gpu, const uint8_t *rgb, int frames, int in_w, int in_h,
                                 int out_w, int out_h, int pad_w, int pad_h) {
    h3_gpu_tensor *pixels = h3_gpu_tensor_from_i8(gpu, (const int8_t *)rgb, (size_t)frames * in_h * in_w * 3);
    h3_gpu_tensor *out = h3_gpu_tensor_new_f32(gpu, (size_t)frames * (out_h + pad_h) * (out_w + pad_w) * 3);
    CHECK(pixels && out);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_vsr_pixels_to_f32(gpu, out, pixels, (uint32_t)frames, (uint32_t)in_w, (uint32_t)in_h,
                                   (uint32_t)in_w * 3, 0, (uint32_t)out_w, (uint32_t)out_h, 0, 0,
                                   (uint32_t)(out_w + pad_w), (uint32_t)(out_h + pad_h)));
    CHECK(h3_gpu_submit(gpu));
    h3_gpu_tensor_free(pixels);
    return out;
}

/* The fixture geometry, from the tensors' shapes. */
typedef struct {
    int frames, lq_w, lq_h, out_w, out_h, grid_h, grid_w, latent_h, latent_w;
    int chunk_latents[8], chunk_count;
} geometry;

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    /* parity with the official implementation, which has no colour match */
    setenv("H3_VSR_NO_COLOR_MATCH", "1", 1);
    const char *fixture = getenv("H3_VSR_FIXTURE");
    if (!fixture || !fixture[0]) fixture = "misc/fixtures/swiftvr-768x544";
    const char *model_dir = getenv("H3_VSR_MODEL_DIR");
    static char default_model[4096];
    if (!model_dir || !model_dir[0]) {
        snprintf(default_model, sizeof(default_model), "%s/Library/Application Support/H3cApp/Upscalers/SwiftVR", getenv("HOME") ? getenv("HOME") : ".");
        model_dir = default_model;
    }
    const char *precision_env = getenv("H3_VSR_PRECISION");
    char probe[4096];
    snprintf(probe, sizeof(probe), "%s/reae.safetensors", fixture);
    if (!file_exists(probe)) { printf("skip: SwiftVR fixture %s is not installed\n", fixture); return 0; }
    /* The DiT and output dumps come later in tools/dump_swiftvr_fixture.py;
     * the stages that need them are skipped while they are missing. */
    snprintf(probe, sizeof(probe), "%s/dit.safetensors", fixture);
    int have_dit = file_exists(probe);
    snprintf(probe, sizeof(probe), "%s/output.safetensors", fixture);
    int have_out = file_exists(probe);
    if (!have_dit) printf("note: no dit.safetensors in the fixture, skipping the window, RoPE, DiT and session checks\n");
    else if (!have_out) printf("note: no output.safetensors in the fixture, skipping the session check\n");
    snprintf(probe, sizeof(probe), "%s/reae.safetensors", model_dir);
    if (!file_exists(probe)) { printf("skip: SwiftVR weights are not installed in %s\n", model_dir); return 0; }
    snprintf(probe, sizeof(probe), "%s/transformer/diffusion_pytorch_model.safetensors", model_dir);
    int have_transformer = file_exists(probe);
    if (!have_transformer) printf("note: the transformer weights are not in %s, skipping the DiT and session checks\n", model_dir);

    /* ---- fixture ---------------------------------------------------- */
    tensor lq, hq, rgb_ref, z[3], zden[3], v0, v0_full, tp, k2, v2, rope_cos, pre4, rgb4;
    CHECK(load(fixture, "input.safetensors", "lq_rgb", &lq) && lq.dtype == H3_DTYPE_U8 && lq.ndim == 4);
    CHECK(load(fixture, "input.safetensors", "hq_rgb", &hq) && hq.dtype == H3_DTYPE_U8);
    geometry g;
    g.frames = (int)lq.shape[0]; g.lq_h = (int)lq.shape[1]; g.lq_w = (int)lq.shape[2];
    g.out_h = (int)hq.shape[1]; g.out_w = (int)hq.shape[2];
    memset(&rgb_ref, 0, sizeof(rgb_ref)); memset(&rgb4, 0, sizeof(rgb4));
    if (have_out) {
        CHECK(load(fixture, "output.safetensors", "rgb_out", &rgb_ref) && rgb_ref.dtype == H3_DTYPE_U8);
        CHECK((int)rgb_ref.shape[0] == g.frames && (int)rgb_ref.shape[1] == g.out_h && (int)rgb_ref.shape[2] == g.out_w);
        load_f32(fixture, "output.safetensors", "rgb0_f32_4", &rgb4);
    }
    g.chunk_count = 0;
    for (int c = 0; c < 3; c++) {
        char name[16];
        snprintf(name, sizeof(name), "z%d", c);
        load_f32(fixture, "reae.safetensors", name, &z[c]);
        memset(&zden[c], 0, sizeof(zden[c]));
        if (have_dit) {
            snprintf(name, sizeof(name), "zden%d", c);
            load_f32(fixture, "dit.safetensors", name, &zden[c]);
        }
        CHECK(z[c].ndim == 4 && z[c].shape[1] == H3_VSR_LATENT_CHANNELS);
        g.chunk_latents[c] = (int)z[c].shape[0];
        g.chunk_count++;
    }
    g.latent_h = (int)z[0].shape[2]; g.latent_w = (int)z[0].shape[3];
    g.grid_h = g.latent_h / 2; g.grid_w = g.latent_w / 2;
    CHECK(g.out_h % 32 == 0 && g.out_w % 32 == 0 && g.latent_h * 16 == g.out_h && g.latent_w * 16 == g.out_w);
    memset(&tp, 0, sizeof(tp)); memset(&k2, 0, sizeof(k2));
    if (have_dit) {
        load_f32(fixture, "dit.safetensors", "v0", &v0);
        load_f32(fixture, "dit.safetensors", "v0_full", &v0_full);
        load_f32(fixture, "dit.safetensors", "tp", &tp);
        load_f32(fixture, "dit.safetensors", "k2_0", &k2);
        load_f32(fixture, "dit.safetensors", "v2_0", &v2);
        load_f32(fixture, "dit.safetensors", "rope_cos0", &rope_cos);
    }
    load_f32(fixture, "reae.safetensors", "pre_clip4", &pre4);
    printf("fixture %s: %d frames %dx%d -> %dx%d, latent %dx%d, tokens %dx%d, chunks %d/%d/%d latents\n",
           fixture, g.frames, g.lq_w, g.lq_h, g.out_w, g.out_h, g.latent_w, g.latent_h, g.grid_w, g.grid_h,
           g.chunk_latents[0], g.chunk_latents[1], g.chunk_latents[2]);
    const int D = (int)tp.shape[1], S = (int)k2.shape[0];
    const size_t lplane = (size_t)g.latent_h * g.latent_w * H3_VSR_LATENT_CHANNELS;
    const size_t fplane = (size_t)g.out_h * g.out_w * 3;

    /* ---- 1. window tables ------------------------------------------- */
    int host = stage_on("host") && have_dit;
    if (host) printf("window tables\n");
    for (int t = 0; host && t < 2; t++) {
        int T = t == 0 ? g.chunk_latents[0] : g.chunk_latents[1];
        for (int shifted = 0; shifted < 2; shifted++) {
            char name[64];
            tensor lin, owner;
            snprintf(name, sizeof(name), "win_lin_%s_t%d", shifted ? "odd" : "even", T);
            CHECK(load(fixture, "dit.safetensors", name, &lin) && lin.dtype == H3_DTYPE_I32 && lin.ndim == 2);
            snprintf(name, sizeof(name), "win_owner_%s_t%d", shifted ? "odd" : "even", T);
            CHECK(load(fixture, "dit.safetensors", name, &owner) && owner.dtype == H3_DTYPE_I32);
            h3_vsr_window_layout layout;
            CHECK(h3_vsr_window_layout_build(T, g.grid_h, g.grid_w, H3_VSR_WINDOW, H3_VSR_WINDOW, shifted, &layout));
            CHECK(layout.window_count == (int)lin.shape[0] && layout.window_length == (int)lin.shape[1]);
            CHECK(layout.tokens == (int)owner.elements);
            CHECK(memcmp(layout.indices, lin.data, lin.elements * sizeof(int32_t)) == 0);
            CHECK(memcmp(layout.owner, owner.data, owner.elements * sizeof(int32_t)) == 0);
            printf("  T=%d %s: %d windows x %d, owner map identical\n", T, shifted ? "odd " : "even", layout.window_count, layout.window_length);
            h3_vsr_window_layout_free(&layout);
            free(lin.data); free(owner.data);
        }
    }

    /* ---- 2. RoPE ---------------------------------------------------- */
    if (host) {
        int T = g.chunk_latents[0];
        size_t tokens = (size_t)T * g.grid_h * g.grid_w, pairs = H3_VSR_HEAD_DIM / 2;
        CHECK(rope_cos.ndim == 2 && rope_cos.shape[0] == tokens && rope_cos.shape[1] == H3_VSR_HEAD_DIM);
        float *cs = malloc(tokens * pairs * sizeof(float)), *sn = malloc(tokens * pairs * sizeof(float));
        h3_vsr_rope_table(T, g.grid_h, g.grid_w, 0, H3_VSR_HEAD_DIM, cs, sn);
        const float *ref = rope_cos.data;
        double max_abs = 0.0;
        for (size_t i = 0; i < tokens; i++)
            for (size_t p = 0; p < pairs; p++) {
                double d = fabs((double)cs[i * pairs + p] - ref[i * H3_VSR_HEAD_DIM + 2 * p]);
                double d2 = fabs((double)ref[i * H3_VSR_HEAD_DIM + 2 * p] - ref[i * H3_VSR_HEAD_DIM + 2 * p + 1]);
                if (d > max_abs) max_abs = d;
                CHECK(d2 == 0.0);   /* repeat_interleave_real: pairs share the angle */
            }
        printf("rope: cos table max |d| %.2e vs official (float32 frequencies)\n", max_abs);
        CHECK(max_abs < 1e-3);
        free(cs); free(sn);
    }

    /* ---- 2b. the ReAE alone (real weights, no DiT) -------------------- */
    char error[512] = {0};
    if (stage_on("reae")) {
        tensor dec4;
        load_f32(fixture, "reae.safetensors", "dec_z0_4", &dec4);
        h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
        CHECK(gpu != NULL);
        double t0 = now();
        h3_vsr_reae *reae = h3_vsr_reae_load(model_dir, gpu, error, sizeof(error));
        if (!reae) fprintf(stderr, "%s\n", error);
        CHECK(reae != NULL);
        printf("reae: loaded in %.2f s\n", now() - t0);
        h3_vsr_reae_state *enc = h3_vsr_reae_state_new(reae, 1, g.out_h, g.out_w);
        CHECK(enc != NULL);
        int fed = 0;
        for (int c = 0; c < g.chunk_count; c++) {
            int latents = g.chunk_latents[c], enc_frames = latents * 4;
            /* frames past the clip are the last frame repeated (the LAST chunk's 3 tail copies) */
            uint8_t *rgb = malloc((size_t)enc_frames * g.lq_h * g.lq_w * 3);
            for (int f = 0; f < enc_frames; f++) {
                int source = fed + f < g.frames ? fed + f : g.frames - 1;
                memcpy(rgb + (size_t)f * g.lq_h * g.lq_w * 3, (uint8_t *)lq.data + (size_t)source * g.lq_h * g.lq_w * 3, (size_t)g.lq_h * g.lq_w * 3);
            }
            h3_gpu_tensor *frames = preprocess(gpu, rgb, enc_frames, g.lq_w, g.lq_h, g.out_w, g.out_h, 0, 0);
            h3_gpu_tensor *latent = h3_gpu_tensor_new_f32(gpu, (size_t)latents * lplane);
            CHECK(latent != NULL);
            t0 = now();
            if (!h3_vsr_reae_encode(reae, enc, frames, enc_frames, g.out_h, g.out_w, 0, latent, error, sizeof(error)))
                fprintf(stderr, "%s\n", error);
            CHECK(error[0] == '\0');
            double secs = now() - t0;
            float *got = malloc((size_t)latents * lplane * sizeof(float));
            CHECK(h3_gpu_tensor_read_f32(latent, got, (size_t)latents * lplane));
            float *z_cl = to_channels_last(z[c].data, latents, H3_VSR_LATENT_CHANNELS, g.latent_h, g.latent_w);
            char what[96];
            snprintf(what, sizeof(what), "ReAE encode chunk %d (%d frames, %.2f s)", c, enc_frames, secs);
            metrics m = compare(got, z_cl, (size_t)latents * lplane);
            report(what, m);
            CHECK(m.psnr > 40.0);
            free(got); free(z_cl); free(rgb);
            h3_gpu_tensor_free(frames); h3_gpu_tensor_free(latent);
            fed += enc_frames;
        }
        h3_vsr_reae_state_free(enc);
        /* decoder: chunk 0's raw latents, fresh state; frames 3..6 are outputs 0..3 */
        {
            int latents = g.chunk_latents[0];
            h3_vsr_reae_state *dec = h3_vsr_reae_state_new(reae, 0, g.out_h, g.out_w);
            CHECK(dec != NULL);
            float *z_cl = to_channels_last(z[0].data, latents, H3_VSR_LATENT_CHANNELS, g.latent_h, g.latent_w);
            h3_gpu_tensor *latent = h3_gpu_tensor_from_f32(gpu, z_cl, (size_t)latents * lplane);
            h3_gpu_tensor *decoded = h3_gpu_tensor_new_f32(gpu, (size_t)latents * 4 * fplane);
            CHECK(latent && decoded);
            t0 = now();
            if (!h3_vsr_reae_decode(reae, dec, latent, latents, g.latent_h, g.latent_w, 0, decoded, error, sizeof(error)))
                fprintf(stderr, "%s\n", error);
            CHECK(error[0] == '\0');
            double secs = now() - t0;
            float *got = malloc((size_t)latents * 4 * fplane * sizeof(float));
            CHECK(h3_gpu_tensor_read_f32(decoded, got, (size_t)latents * 4 * fplane));
            float *ref_cl = to_channels_last(dec4.data, 4, 3, g.out_h, g.out_w);
            char what[96];
            snprintf(what, sizeof(what), "ReAE decode chunk 0 (4 of %d frames, %.2f s)", latents * 4, secs);
            metrics m = compare(got + (size_t)H3_VSR_FRAMES_TO_TRIM * fplane, ref_cl, (size_t)4 * fplane);
            report(what, m);
            CHECK(m.psnr > 40.0);
            free(got); free(ref_cl); free(z_cl);
            h3_gpu_tensor_free(latent); h3_gpu_tensor_free(decoded);
            h3_vsr_reae_state_free(dec);
        }
        h3_vsr_reae_free(reae);
        h3_gpu_free(gpu);
        free(dec4.data);
    }

    /* ---- precisions ------------------------------------------------- */
    int precisions[2], precision_count = 0;
    /* Default: what the app uses on this Mac (int8 on tensor hardware,
     * BF16 elsewhere); only one cache need exist. */
    if (!precision_env || !precision_env[0] || !strcmp(precision_env, "auto"))
        precisions[precision_count++] = h3_vsr_int8_supported() ? H3_VSR_PRECISION_INT8 : H3_VSR_PRECISION_BF16;
    else if (!strcmp(precision_env, "bf16")) precisions[precision_count++] = H3_VSR_PRECISION_BF16;
    else if (!strcmp(precision_env, "int8")) precisions[precision_count++] = H3_VSR_PRECISION_INT8;
    else { precisions[precision_count++] = H3_VSR_PRECISION_BF16; precisions[precision_count++] = H3_VSR_PRECISION_INT8; }
    if (!h3_vsr_int8_supported()) {
        int n = 0;
        for (int i = 0; i < precision_count; i++) if (precisions[i] != H3_VSR_PRECISION_INT8) precisions[n++] = precisions[i];
        if (n < precision_count) printf("note: int8 is not supported on this GPU, skipping it\n");
        precision_count = n;
    }
    if (!have_dit || !have_transformer || (!stage_on("dit") && !stage_on("session"))) precision_count = 0;

    for (int pi = 0; pi < precision_count; pi++) {
        int precision = precisions[pi];
        const char *pname = precision == H3_VSR_PRECISION_INT8 ? "int8" : "bf16";
        double floor_db = precision == H3_VSR_PRECISION_INT8 ? 25.0 : 30.0;
        char cache_path[4096];
        CHECK(h3_vsr_cache_path(model_dir, precision, cache_path, sizeof(cache_path)));
        printf("== %s: cache %s\n", pname, cache_path);
        if (!h3_vsr_cache_matches(cache_path, model_dir)) {
            double t0 = now();
            printf("  building the cache from %s\n", model_dir);
            if (!h3_vsr_build_cache(model_dir, cache_path, precision, "h3_shaders.metal", progress, NULL, error, sizeof(error)))
                fprintf(stderr, "%s\n", error);
            CHECK(error[0] == '\0');
            printf("  built in %.0f s\n", now() - t0);
        }
        CHECK(h3_vsr_cache_matches(cache_path, model_dir));

        /* ---- 3. constants in the cache ------------------------------ */
        {
            h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
            CHECK(gpu != NULL);
            h3_vsr_cache cache;
            CHECK(h3_vsr_cache_open(cache_path, &cache, error, sizeof(error)));
            /* blocks.0.mod = scale_shift_table[0] + time_proj */
            tensor table;
            char st_path[4096];
            snprintf(st_path, sizeof(st_path), "%s/transformer", model_dir);
            CHECK(load(st_path, "diffusion_pytorch_model.safetensors", "blocks.0.scale_shift_table", &table) &&
                  table.dtype == H3_DTYPE_F32 && table.elements == (size_t)6 * D);
            float *mod = malloc((size_t)6 * D * sizeof(float)), *expect = malloc((size_t)6 * D * sizeof(float));
            CHECK(h3_vsr_cache_read_f32(&cache, "blocks.0.mod", mod, (size_t)6 * D, error, sizeof(error)));
            for (size_t i = 0; i < (size_t)6 * D; i++) expect[i] = ((float *)table.data)[i] + ((float *)tp.data)[i];
            metrics m = compare(mod, expect, (size_t)6 * D);
            report("blocks.0.mod (table + time_proj)", m);
            CHECK(m.psnr > 80.0);
            free(mod); free(expect); free(table.data);
            /* cross-attention K2/V2 of block 0 (BF16 in the cache) */
            const char *names[2] = {"blocks.0.attn2.k2", "blocks.0.attn2.v2"};
            const tensor *refs[2] = {&k2, &v2};
            for (int i = 0; i < 2; i++) {
                h3_gpu_tensor *t = h3_vsr_cache_load(&cache, gpu, names[i], error, sizeof(error));
                if (!t) fprintf(stderr, "%s\n", error);
                CHECK(t != NULL);
                uint16_t *raw = malloc((size_t)S * D * sizeof(uint16_t));
                float *vals = malloc((size_t)S * D * sizeof(float));
                CHECK(h3_gpu_tensor_read_bf16(t, raw, (size_t)S * D));
                for (size_t j = 0; j < (size_t)S * D; j++) vals[j] = bf16_to_f32(raw[j]);
                m = compare(vals, refs[i]->data, (size_t)S * D);
                report(names[i], m);
                CHECK(m.psnr > 40.0);
                free(raw); free(vals); h3_gpu_tensor_free(t);
            }
            h3_vsr_cache_close(&cache);
            h3_gpu_free(gpu);
        }

        /* ---- 4. the DiT alone on the reference latents -------------- */
        if (stage_on("dit")) {
            h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
            CHECK(gpu != NULL);
            double t0 = now();
            h3_vsr_wan *wan = h3_vsr_wan_load(gpu, cache_path, error, sizeof(error));
            if (!wan) fprintf(stderr, "%s\n", error);
            CHECK(wan != NULL && h3_vsr_wan_is_int8(wan) == (precision == H3_VSR_PRECISION_INT8));
            printf("  wan loaded in %.1f s\n", now() - t0);
            int T = g.chunk_latents[0];
            size_t tokens = (size_t)T * g.grid_h * g.grid_w, pairs = H3_VSR_HEAD_DIM / 2;
            float *z0_cl = to_channels_last(z[0].data, T, H3_VSR_LATENT_CHANNELS, g.latent_h, g.latent_w);
            float *v0_cl = to_channels_last(v0.data, T, H3_VSR_LATENT_CHANNELS, g.latent_h, g.latent_w);
            float *v0f_cl = to_channels_last(v0_full.data, T, H3_VSR_LATENT_CHANNELS, g.latent_h, g.latent_w);
            float *zden0_cl = to_channels_last(zden[0].data, T, H3_VSR_LATENT_CHANNELS, g.latent_h, g.latent_w);
            float *cs = malloc(tokens * pairs * sizeof(float)), *sn = malloc(tokens * pairs * sizeof(float));
            h3_vsr_rope_table(T, g.grid_h, g.grid_w, 0, H3_VSR_HEAD_DIM, cs, sn);
            h3_vsr_window_layout even, odd;
            CHECK(h3_vsr_window_layout_build(T, g.grid_h, g.grid_w, H3_VSR_WINDOW, H3_VSR_WINDOW, 0, &even));
            CHECK(h3_vsr_window_layout_build(T, g.grid_h, g.grid_w, H3_VSR_WINDOW, H3_VSR_WINDOW, 1, &odd));
            h3_gpu_tensor *z_lq_t = h3_gpu_tensor_from_f32(gpu, z0_cl, (size_t)T * lplane);
            h3_gpu_tensor *z_hq_t = h3_gpu_tensor_new_f32(gpu, (size_t)T * lplane);
            h3_gpu_tensor *cs_t = h3_gpu_tensor_from_f32(gpu, cs, tokens * pairs);
            h3_gpu_tensor *sn_t = h3_gpu_tensor_from_f32(gpu, sn, tokens * pairs);
            CHECK(z_lq_t && z_hq_t && cs_t && sn_t);
            float *got = malloc((size_t)T * lplane * sizeof(float)), *v_got = malloc((size_t)T * lplane * sizeof(float));
            char dit_dump[] = "/tmp/h3vsr-real-dit-XXXXXX";
            CHECK(mkdtemp(dit_dump) != NULL);
            for (int windowed = 1; windowed >= 0; windowed--) {
                t0 = now();
                if (windowed) { setenv("H3_VSR_DUMP_DIR", dit_dump, 1); setenv("H3_VSR_DUMP_TOKENS", "1", 1); }
                else { unsetenv("H3_VSR_DUMP_DIR"); unsetenv("H3_VSR_DUMP_TOKENS"); }
                if (!h3_vsr_wan_predict(wan, z_lq_t, T, g.latent_h, g.latent_w, windowed ? &even : NULL, windowed ? &odd : NULL,
                                        cs_t, sn_t, z_hq_t, NULL, NULL, error, sizeof(error)))
                    fprintf(stderr, "%s\n", error);
                CHECK(error[0] == '\0');
                double secs = now() - t0;
                CHECK(h3_gpu_tensor_read_f32(z_hq_t, got, (size_t)T * lplane));
                for (size_t i = 0; i < (size_t)T * lplane; i++) v_got[i] = z0_cl[i] - got[i];
                char what[96];
                snprintf(what, sizeof(what), "DiT %s v (chunk 0, %.1f s)", windowed ? "MFSWA" : "dense", secs);
                metrics mv = compare(v_got, windowed ? v0_cl : v0f_cl, (size_t)T * lplane);
                report(what, mv);
                if (windowed) {
                    metrics mz = compare(got, zden0_cl, (size_t)T * lplane);
                    report("DiT MFSWA z_hq (chunk 0)", mz);
                    /* the token activations of the first blocks vs the official ones */
                    const char *names[3] = {"x_in", "block0", "block1"};
                    const char *refs[3] = {"hs_in0", "block0_out0", "block1_out0"};
                    for (int i = 0; i < 3; i++) {
                        tensor ref;
                        load_f32(fixture, "dit.safetensors", refs[i], &ref);
                        CHECK(ref.elements == tokens * (size_t)D);
                        char path[4200];
                        snprintf(path, sizeof(path), "%s/dit_%s.f32", dit_dump, names[i]);
                        FILE *fp = fopen(path, "rb");
                        CHECK(fp != NULL);
                        float *act = malloc(ref.elements * sizeof(float));
                        CHECK(fread(act, sizeof(float), ref.elements, fp) == ref.elements);
                        fclose(fp);
                        metrics ma = compare(act, ref.data, ref.elements);
                        snprintf(what, sizeof(what), "DiT tokens after %s", names[i]);
                        report(what, ma);
                        free(act); free(ref.data);
                    }
                }
                if (!(mv.psnr > floor_db)) printf("  !! DiT %s below %.0f dB\n", windowed ? "MFSWA" : "dense", floor_db);
            }
            if (getenv("H3_VSR_KEEP_DUMP")) printf("  kept the DiT dump in %s\n", dit_dump);
            else {
                char cmd[4200];
                snprintf(cmd, sizeof(cmd), "rm -rf '%s'", dit_dump);
                (void)system(cmd);
            }
            free(got); free(v_got); free(cs); free(sn); free(z0_cl); free(v0_cl); free(v0f_cl); free(zden0_cl);
            h3_vsr_window_layout_free(&even); h3_vsr_window_layout_free(&odd);
            h3_gpu_tensor_free(z_lq_t); h3_gpu_tensor_free(z_hq_t); h3_gpu_tensor_free(cs_t); h3_gpu_tensor_free(sn_t);
            h3_vsr_wan_free(wan);
            h3_gpu_free(gpu);
        }

        /* ---- 5. the whole session on the clip ----------------------- */
        if (stage_on("session") && have_out) {
            char dump_dir[] = "/tmp/h3vsr-real-dump-XXXXXX";
            CHECK(mkdtemp(dump_dir) != NULL);
            setenv("H3_VSR_DUMP_DIR", dump_dir, 1);
            h3_vsr_open_params params = {0};
            params.precision = precision;
            params.cache_path = cache_path;
            params.shader_source_path = "h3_shaders.metal";
            if (getenv("H3_VSR_REAE_BATCH")) params.reae_frame_batch = atoi(getenv("H3_VSR_REAE_BATCH"));
            printf("  reae_frame_batch %d\n", params.reae_frame_batch);
            double t0 = now();
            h3_vsr *vsr = h3_vsr_open(model_dir, &params, NULL, NULL, error, sizeof(error));
            if (!vsr) fprintf(stderr, "%s\n", error);
            CHECK(vsr != NULL);
            printf("  model opened in %.1f s\n", now() - t0);
            h3_vsr_session *s = h3_vsr_session_open(vsr, g.lq_w, g.lq_h, g.out_w, g.out_h, g.frames, error, sizeof(error));
            if (!s) fprintf(stderr, "%s\n", error);
            CHECK(s != NULL);
            uint8_t *out = malloc((size_t)g.frames * fplane);
            memset(out, 0x55, (size_t)g.frames * fplane);
            h3_vsr_image in[32];
            h3_vsr_image_out outs[64];
            int fed = 0, total = 0, chunk = 0;
            t0 = now();
            while (fed < g.frames) {
                int want = h3_vsr_session_next_chunk_frames(s);
                int n = g.frames - fed < want ? g.frames - fed : want;
                int is_last = fed + n >= g.frames;
                for (int i = 0; i < n; i++) {
                    in[i].pixels = (uint8_t *)lq.data + (size_t)(fed + i) * g.lq_h * g.lq_w * 3;
                    in[i].width = g.lq_w; in[i].height = g.lq_h; in[i].stride_bytes = g.lq_w * 3;
                    in[i].format = H3_VSR_RGB24;
                }
                for (int i = 0; i < 64; i++) {
                    int frame = total + i;
                    outs[i].pixels = frame < g.frames ? out + (size_t)frame * fplane : NULL;
                    outs[i].width = g.out_w; outs[i].height = g.out_h; outs[i].stride_bytes = g.out_w * 3;
                    outs[i].format = H3_VSR_RGB24;
                }
                int capacity = g.frames - total < 64 ? g.frames - total : 64, got = 0;
                double tc = now();
                int ok = h3_vsr_session_process(s, in, n, is_last, outs, capacity, &got, NULL, NULL, NULL, error, sizeof(error));
                if (!ok) fprintf(stderr, "%s\n", error);
                CHECK(ok);
                printf("  chunk %d: %d frames in -> %d out, %.1f s\n", chunk, n, got, now() - tc);
                total += got; fed += n; chunk++;
            }
            CHECK(total == g.frames && h3_vsr_session_pending_frames(s) == 0);
            double dit_s, reae_s; uint64_t peak;
            CHECK(h3_vsr_get_stats(vsr, &dit_s, &reae_s, &peak));
            printf("  session: %.1f s wall, GPU dit %.1f s, reae %.1f s, peak live %.2f GiB\n",
                   now() - t0, dit_s, reae_s, (double)peak / (1024.0 * 1024.0 * 1024.0));
            h3_vsr_session_close(s);
            h3_vsr_close(vsr);
            unsetenv("H3_VSR_DUMP_DIR");

            /* 8-bit output vs the official frames, and both vs the ground truth. */
            {
                const uint8_t *ref = rgb_ref.data, *truth = hq.data;
                size_t n = (size_t)g.frames * fplane, over1 = 0, over4 = 0;
                double err = 0.0, err_truth = 0.0, err_ref_truth = 0.0;
                int max_abs = 0;
                for (size_t i = 0; i < n; i++) {
                    int d = (int)out[i] - (int)ref[i];
                    if (d < 0) d = -d;
                    err += (double)d * d;
                    if (d > 1) over1++;
                    if (d > 4) over4++;
                    if (d > max_abs) max_abs = d;
                    double dt = (double)out[i] - truth[i], dr = (double)ref[i] - truth[i];
                    err_truth += dt * dt; err_ref_truth += dr * dr;
                }
                double psnr = 10.0 * log10(255.0 * 255.0 / (err / (double)n > 0 ? err / (double)n : 1e-30));
                printf("  frames vs official: %.1f dB, max |d| %d, >1: %.3f%%, >4: %.4f%%\n", psnr, max_abs,
                       100.0 * (double)over1 / (double)n, 100.0 * (double)over4 / (double)n);
                printf("  vs ground truth: engine %.2f dB, official %.2f dB\n",
                       10.0 * log10(255.0 * 255.0 / (err_truth / (double)n)),
                       10.0 * log10(255.0 * 255.0 / (err_ref_truth / (double)n)));
                CHECK(psnr > floor_db);
            }

            /* Stage dumps of each chunk. */
            {
                int enc_frames0 = g.chunk_latents[0] * 4;
                float *frames0 = read_raw_f32(dump_dir, 0, "frames", (size_t)enc_frames0 * fplane);
                CHECK(frames0 != NULL);
                float *pre_cl = to_channels_last(pre4.data, 4, 3, g.out_h, g.out_w);
                metrics m = compare(frames0, pre_cl, (size_t)4 * fplane);
                report("preprocess (bilinear, 4 frames)", m);
                CHECK(m.max_abs < 2e-3);
                free(frames0); free(pre_cl);
                int prefix_last = (H3_VSR_CLIP_FRAMES / 4 + 1) - g.chunk_latents[2];
                for (int c = 0; c < g.chunk_count; c++) {
                    int latents = g.chunk_latents[c];
                    int prefix = c == g.chunk_count - 1 ? prefix_last : 0;
                    float *lat = read_raw_f32(dump_dir, c, "latent", (size_t)latents * lplane);
                    float *zhq = read_raw_f32(dump_dir, c, "z_hq", (size_t)(prefix + latents) * lplane);
                    CHECK(lat && zhq);
                    float *z_cl = to_channels_last(z[c].data, latents, H3_VSR_LATENT_CHANNELS, g.latent_h, g.latent_w);
                    float *zden_cl = to_channels_last(zden[c].data, latents, H3_VSR_LATENT_CHANNELS, g.latent_h, g.latent_w);
                    char what[96];
                    snprintf(what, sizeof(what), "chunk %d ReAE latent (%d)", c, latents);
                    m = compare(lat, z_cl, (size_t)latents * lplane);
                    report(what, m);
                    CHECK(m.psnr > 35.0);
                    snprintf(what, sizeof(what), "chunk %d DiT z_hq (own %d)", c, latents);
                    m = compare(zhq + (size_t)prefix * lplane, zden_cl, (size_t)latents * lplane);
                    report(what, m);
                    CHECK(m.psnr > floor_db);
                    free(lat); free(zhq); free(z_cl); free(zden_cl);
                }
                float *dec0 = read_raw_f32(dump_dir, 0, "decoded", (size_t)enc_frames0 * fplane);
                CHECK(dec0 != NULL);
                float *rgb4_cl = to_channels_last(rgb4.data, 4, 3, g.out_h, g.out_w);
                /* the decoder's first 3 frames are dropped; frames 3..6 are outputs 0..3 */
                m = compare(dec0 + (size_t)H3_VSR_FRAMES_TO_TRIM * fplane, rgb4_cl, (size_t)4 * fplane);
                report("chunk 0 decoded RGB (4 frames, F32)", m);
                CHECK(m.psnr > floor_db);
                free(dec0); free(rgb4_cl);
            }
            /* remove the dump */
            {
                char cmd[4200];
                snprintf(cmd, sizeof(cmd), "rm -rf '%s'", dump_dir);
                (void)system(cmd);
            }
            free(out);
        }
    }
    printf("ok: %d checks (real SwiftVR parity)\n", checks);
    return 0;
}
