/* The .h3vc weight cache round trip on a tiny synthetic Wan model: writes a
 * model directory (config.json, F32 safetensors, a BF16 prompt embedding),
 * builds the cache, and checks the header, the entries, the precomputed
 * constants against a plain double-precision recomputation, the int8 rows
 * against the F32 weights, and the fingerprint check. Needs a GPU (the
 * builder quantizes and converts there) but no real weights, so it always
 * runs in `make test`. */
#include "h3_vsr_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utime.h>

static int tests_run;

#define CHECK(condition) do { \
    tests_run++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

enum { LAYERS = 2, HEADS = 1, HEAD_DIM = 128, DIM = 128, FFN = 256, IN = 48,
       TEXT_DIM = 64, FREQ = 256, TOKENS = 8, PATCH_IN = IN * 4 };

/* Deterministic values in [-1, 1). */
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static float rnd(void) {
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((rng_state >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f;
}

/* A minimal safetensors writer. */
typedef struct { char name[128]; const char *dtype; int ndim; uint64_t shape[5]; void *data; size_t bytes; } tensor_spec;

static void write_safetensors(const char *path, tensor_spec *tensors, int count) {
    char header[65536];
    size_t at = 0;
    uint64_t offset = 0;
    at += (size_t)snprintf(header + at, sizeof(header) - at, "{");
    for (int i = 0; i < count; i++) {
        at += (size_t)snprintf(header + at, sizeof(header) - at, "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                               i ? "," : "", tensors[i].name, tensors[i].dtype);
        for (int d = 0; d < tensors[i].ndim; d++)
            at += (size_t)snprintf(header + at, sizeof(header) - at, "%s%llu", d ? "," : "",
                                   (unsigned long long)tensors[i].shape[d]);
        at += (size_t)snprintf(header + at, sizeof(header) - at, "],\"data_offsets\":[%llu,%llu]}",
                               (unsigned long long)offset, (unsigned long long)(offset + tensors[i].bytes));
        offset += tensors[i].bytes;
    }
    at += (size_t)snprintf(header + at, sizeof(header) - at, "}");
    while (at % 8) header[at++] = ' ';
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    uint64_t length = at;
    CHECK(fwrite(&length, 8, 1, f) == 1);
    CHECK(fwrite(header, 1, at, f) == at);
    for (int i = 0; i < count; i++) CHECK(fwrite(tensors[i].data, 1, tensors[i].bytes, f) == tensors[i].bytes);
    fclose(f);
}

static float *random_f32(size_t n, float scale) {
    float *v = malloc(n * sizeof(float));
    CHECK(v != NULL);
    for (size_t i = 0; i < n; i++) v[i] = rnd() * scale;
    return v;
}

static uint16_t to_bf16(float x) { uint32_t b; memcpy(&b, &x, 4); b += 0x7FFFu + ((b >> 16) & 1u); return (uint16_t)(b >> 16); }
static float from_bf16(uint16_t h) { uint32_t b = (uint32_t)h << 16; float x; memcpy(&x, &b, 4); return x; }

/* The synthetic model, kept so the test can recompute the constants. */
typedef struct {
    float *tw1, *tb1, *tw2, *tb2, *pw, *pb;      /* time embedder, time_proj */
    float *xw1, *xb1, *xw2, *xb2;                /* text embedder */
    float *prompt;                               /* [TOKENS, TEXT_DIM] (bf16-rounded) */
    float *sst[LAYERS];                          /* scale_shift_table [6, DIM] */
    float *k_w[LAYERS], *k_b[LAYERS], *k_norm[LAYERS];
    float *qkv_w[LAYERS][3];                     /* to_q/k/v weights */
    float *sst_out;                              /* [2, DIM] */
} model;

static int spec_count;
static tensor_spec specs[512];

static float *add_tensor(const char *name, int ndim, const uint64_t *shape, float scale) {
    size_t n = 1;
    for (int d = 0; d < ndim; d++) n *= shape[d];
    float *v = random_f32(n, scale);
    tensor_spec *t = &specs[spec_count++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->dtype = "F32"; t->ndim = ndim; memcpy(t->shape, shape, sizeof(uint64_t) * (size_t)ndim);
    t->data = v; t->bytes = n * 4;
    return v;
}
static float *add2(const char *name, uint64_t r, uint64_t c, float scale) { uint64_t s[2] = {r, c}; return add_tensor(name, 2, s, scale); }
static float *add1(const char *name, uint64_t n, float scale) { uint64_t s[1] = {n}; return add_tensor(name, 1, s, scale); }

static void write_model(const char *dir, model *m) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/transformer", dir);
    CHECK(mkdir(dir, 0755) == 0 && mkdir(path, 0755) == 0);
    snprintf(path, sizeof(path), "%s/transformer/config.json", dir);
    FILE *f = fopen(path, "w");
    CHECK(f != NULL);
    fprintf(f, "{\n  \"_class_name\": \"WanTransformer3DModel\",\n  \"attention_head_dim\": %d,\n"
               "  \"cross_attn_norm\": true,\n  \"eps\": 1e-06,\n  \"ffn_dim\": %d,\n  \"freq_dim\": %d,\n"
               "  \"in_channels\": %d,\n  \"num_attention_heads\": %d,\n  \"num_layers\": %d,\n"
               "  \"out_channels\": %d,\n  \"patch_size\": [\n    1,\n    2,\n    2\n  ],\n"
               "  \"text_dim\": %d\n}\n", HEAD_DIM, FFN, FREQ, IN, HEADS, LAYERS, IN, TEXT_DIM);
    fclose(f);

    spec_count = 0;
    m->tw1 = add2("condition_embedder.time_embedder.linear_1.weight", DIM, FREQ, 0.1f);
    m->tb1 = add1("condition_embedder.time_embedder.linear_1.bias", DIM, 0.1f);
    m->tw2 = add2("condition_embedder.time_embedder.linear_2.weight", DIM, DIM, 0.1f);
    m->tb2 = add1("condition_embedder.time_embedder.linear_2.bias", DIM, 0.1f);
    m->pw = add2("condition_embedder.time_proj.weight", 6 * DIM, DIM, 0.1f);
    m->pb = add1("condition_embedder.time_proj.bias", 6 * DIM, 0.1f);
    m->xw1 = add2("condition_embedder.text_embedder.linear_1.weight", DIM, TEXT_DIM, 0.1f);
    m->xb1 = add1("condition_embedder.text_embedder.linear_1.bias", DIM, 0.1f);
    m->xw2 = add2("condition_embedder.text_embedder.linear_2.weight", DIM, DIM, 0.1f);
    m->xb2 = add1("condition_embedder.text_embedder.linear_2.bias", DIM, 0.1f);
    {
        uint64_t s[5] = {DIM, IN, 1, 2, 2};
        add_tensor("patch_embedding.weight", 5, s, 0.1f);
        add1("patch_embedding.bias", DIM, 0.1f);
        add2("proj_out.weight", PATCH_IN, DIM, 0.1f);
        add1("proj_out.bias", PATCH_IN, 0.1f);
        uint64_t so[3] = {1, 2, DIM};
        m->sst_out = add_tensor("scale_shift_table", 3, so, 0.5f);
    }
    for (int b = 0; b < LAYERS; b++) {
        char name[160];
#define N(suffix) (snprintf(name, sizeof(name), "blocks.%d.%s", b, suffix), name)
        m->qkv_w[b][0] = add2(N("attn1.to_q.weight"), DIM, DIM, 0.2f);
        add1(N("attn1.to_q.bias"), DIM, 0.1f);
        m->qkv_w[b][1] = add2(N("attn1.to_k.weight"), DIM, DIM, 0.2f);
        add1(N("attn1.to_k.bias"), DIM, 0.1f);
        m->qkv_w[b][2] = add2(N("attn1.to_v.weight"), DIM, DIM, 0.2f);
        add1(N("attn1.to_v.bias"), DIM, 0.1f);
        add1(N("attn1.norm_q.weight"), DIM, 1.0f);
        add1(N("attn1.norm_k.weight"), DIM, 1.0f);
        add2(N("attn1.to_out.0.weight"), DIM, DIM, 0.2f);
        add1(N("attn1.to_out.0.bias"), DIM, 0.1f);
        add2(N("attn2.to_q.weight"), DIM, DIM, 0.2f);
        add1(N("attn2.to_q.bias"), DIM, 0.1f);
        m->k_w[b] = add2(N("attn2.to_k.weight"), DIM, DIM, 0.2f);
        m->k_b[b] = add1(N("attn2.to_k.bias"), DIM, 0.1f);
        add2(N("attn2.to_v.weight"), DIM, DIM, 0.2f);
        add1(N("attn2.to_v.bias"), DIM, 0.1f);
        add1(N("attn2.norm_q.weight"), DIM, 1.0f);
        m->k_norm[b] = add1(N("attn2.norm_k.weight"), DIM, 1.0f);
        add2(N("attn2.to_out.0.weight"), DIM, DIM, 0.2f);
        add1(N("attn2.to_out.0.bias"), DIM, 0.1f);
        add1(N("norm2.weight"), DIM, 1.0f);
        add1(N("norm2.bias"), DIM, 0.1f);
        add2(N("ffn.net.0.proj.weight"), FFN, DIM, 0.2f);
        add1(N("ffn.net.0.proj.bias"), FFN, 0.1f);
        add2(N("ffn.net.2.weight"), DIM, FFN, 0.2f);
        add1(N("ffn.net.2.bias"), DIM, 0.1f);
        uint64_t s[3] = {1, 6, DIM};
        m->sst[b] = add_tensor(N("scale_shift_table"), 3, s, 0.5f);
#undef N
    }
    snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors", dir);
    write_safetensors(path, specs, spec_count);

    /* prompt_emb as BF16 [1, TOKENS, TEXT_DIM]; keep the rounded values. */
    m->prompt = random_f32((size_t)TOKENS * TEXT_DIM, 1.0f);
    uint16_t *half = malloc((size_t)TOKENS * TEXT_DIM * 2);
    for (size_t i = 0; i < (size_t)TOKENS * TEXT_DIM; i++) { half[i] = to_bf16(m->prompt[i]); m->prompt[i] = from_bf16(half[i]); }
    tensor_spec prompt = {"prompt_emb", "BF16", 3, {1, TOKENS, TEXT_DIM, 0, 0}, half, (size_t)TOKENS * TEXT_DIM * 2};
    snprintf(path, sizeof(path), "%s/prompt_embedding.safetensors", dir);
    write_safetensors(path, &prompt, 1);
    free(half);

    float *reae = random_f32(48 * 12 * 9, 0.1f);
    tensor_spec reae_spec = {"decoder.1.weight", "F32", 4, {48, 12, 3, 3, 0}, reae, 48 * 12 * 9 * 4};
    snprintf(path, sizeof(path), "%s/reae.safetensors", dir);
    write_safetensors(path, &reae_spec, 1);
    free(reae);
}

/* Plain double-precision reference of the constants. */
static void ref_linear(const float *in, int rows, int in_dim, const float *w, const float *bias, int out_dim, double *out) {
    for (int r = 0; r < rows; r++)
        for (int o = 0; o < out_dim; o++) {
            double acc = bias ? bias[o] : 0.0;
            for (int i = 0; i < in_dim; i++) acc += (double)in[(size_t)r * in_dim + i] * w[(size_t)o * in_dim + i];
            out[(size_t)r * out_dim + o] = acc;
        }
}
static void to_f32(const double *in, float *out, size_t n) { for (size_t i = 0; i < n; i++) out[i] = (float)in[i]; }
static double silu(double x) { return x / (1.0 + exp(-x)); }
static double gelu_tanh(double x) { return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))); }

static double max_abs_diff(const float *a, const float *b, size_t n) {
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (d > worst) worst = d; }
    return worst;
}

static void check_cache(const char *dir, const char *cache_path, const model *m, int int8, h3_gpu *gpu) {
    char error[512] = {0};
    h3_vsr_cache cache;
    CHECK(h3_vsr_cache_open(cache_path, &cache, error, sizeof(error)));
    CHECK(cache.header.blocks == LAYERS && cache.header.dim == DIM && cache.header.ffn == FFN);
    CHECK(cache.header.heads == HEADS && cache.header.head_dim == HEAD_DIM);
    CHECK(cache.header.text_tokens == TOKENS && cache.header.patch_in == PATCH_IN);
    CHECK(cache.header.latent_channels == IN && cache.header.model_kind == H3_VSR_SWIFTVR);
    CHECK(cache.header.precision == (uint32_t)(int8 ? H3_VSR_PRECISION_INT8 : H3_VSR_PRECISION_BF16));
    /* Every payload offset is 64-byte aligned and inside the file. */
    struct stat st;
    CHECK(stat(cache_path, &st) == 0);
    for (size_t i = 0; i < cache.count; i++) {
        CHECK(cache.entries[i].offset % 64 == 0);
        CHECK(cache.entries[i].offset + cache.entries[i].bytes <= (uint64_t)st.st_size);
    }
    const h3_vsr_cache_entry *qkv = h3_vsr_cache_find(&cache, "blocks.0.qkv.weight");
    CHECK(qkv && qkv->rows == 3 * DIM && qkv->cols == DIM);
    CHECK(qkv->dtype == (uint32_t)(int8 ? H3_VSR_CACHE_I8 : H3_VSR_CACHE_BF16));
    CHECK(h3_vsr_cache_find(&cache, "blocks.1.ffn.fc2.weight") != NULL);
    CHECK((h3_vsr_cache_find(&cache, "blocks.0.qkv.scales") != NULL) == int8);
    CHECK(h3_vsr_cache_find(&cache, "blocks.0.attn2.k2")->dtype == H3_VSR_CACHE_BF16);
    CHECK(h3_vsr_cache_find(&cache, "patch_embedding.weight")->cols == PATCH_IN);
    CHECK(h3_vsr_cache_find(&cache, "proj_out.weight")->rows == PATCH_IN);
    CHECK(h3_vsr_cache_find(&cache, "nothing") == NULL);

    /* temb, blocks.0.mod, norm_out.mod against the double reference. */
    float emb[FREQ];
    h3_vsr_sinusoidal_embedding(1000.0, FREQ, emb);
    for (int i = 0; i < FREQ / 2; i++) {
        double f = exp(-log(10000.0) * i / (double)(FREQ / 2));
        CHECK(fabs(emb[i] - cos(1000.0 * f)) < 1e-5 && fabs(emb[FREQ / 2 + i] - sin(1000.0 * f)) < 1e-5);
    }
    double h[DIM], temb_d[DIM], proj_d[6 * DIM];
    float hf[DIM], temb_f[DIM];
    ref_linear(emb, 1, FREQ, m->tw1, m->tb1, DIM, h);
    for (int i = 0; i < DIM; i++) hf[i] = (float)silu(h[i]);
    ref_linear(hf, 1, DIM, m->tw2, m->tb2, DIM, temb_d);
    to_f32(temb_d, temb_f, DIM);
    float got_temb[DIM];
    CHECK(h3_vsr_cache_read_f32(&cache, "temb", got_temb, DIM, error, sizeof(error)));
    CHECK(max_abs_diff(got_temb, temb_f, DIM) < 1e-3);
    for (int i = 0; i < DIM; i++) hf[i] = (float)silu(temb_d[i]);
    ref_linear(hf, 1, DIM, m->pw, m->pb, 6 * DIM, proj_d);
    float got_mod[6 * DIM], want_mod[6 * DIM];
    CHECK(h3_vsr_cache_read_f32(&cache, "blocks.0.mod", got_mod, 6 * DIM, error, sizeof(error)));
    for (int i = 0; i < 6 * DIM; i++) want_mod[i] = (float)(proj_d[i] + m->sst[0][i]);
    CHECK(max_abs_diff(got_mod, want_mod, 6 * DIM) < 1e-3);
    float got_out[2 * DIM], want_out[2 * DIM];
    CHECK(h3_vsr_cache_read_f32(&cache, "norm_out.mod", got_out, 2 * DIM, error, sizeof(error)));
    for (int i = 0; i < 2 * DIM; i++) want_out[i] = (float)(temb_d[i % DIM] + m->sst_out[i]);
    CHECK(max_abs_diff(got_out, want_out, 2 * DIM) < 1e-3);

    /* context and block-0 K2 (RMSNorm_k applied) through the GPU loader. */
    double th[TOKENS * DIM], ctx_d[TOKENS * DIM], k2_d[TOKENS * DIM];
    float thf[TOKENS * DIM], ctx_f[TOKENS * DIM];
    ref_linear(m->prompt, TOKENS, TEXT_DIM, m->xw1, m->xb1, DIM, th);
    for (int i = 0; i < TOKENS * DIM; i++) thf[i] = (float)gelu_tanh(th[i]);
    ref_linear(thf, TOKENS, DIM, m->xw2, m->xb2, DIM, ctx_d);
    to_f32(ctx_d, ctx_f, TOKENS * DIM);
    h3_gpu_tensor *context = h3_vsr_cache_load(&cache, gpu, "context", error, sizeof(error));
    CHECK(context != NULL && h3_gpu_tensor_elements(context) == TOKENS * DIM);
    uint16_t got_bf16[TOKENS * DIM];
    CHECK(h3_gpu_tensor_read_bf16(context, got_bf16, TOKENS * DIM));
    for (int i = 0; i < TOKENS * DIM; i++)
        CHECK(fabs(from_bf16(got_bf16[i]) - ctx_f[i]) <= fabs(ctx_f[i]) * (1.0 / 128) + 1e-3);
    h3_gpu_tensor_free(context);
    ref_linear(ctx_f, TOKENS, DIM, m->k_w[0], m->k_b[0], DIM, k2_d);
    for (int r = 0; r < TOKENS; r++) {
        double sum = 0.0;
        for (int c = 0; c < DIM; c++) sum += k2_d[r * DIM + c] * k2_d[r * DIM + c];
        double scale = 1.0 / sqrt(sum / DIM + 1e-6);
        for (int c = 0; c < DIM; c++) k2_d[r * DIM + c] *= scale * m->k_norm[0][c];
    }
    h3_gpu_tensor *k2 = h3_vsr_cache_load(&cache, gpu, "blocks.0.attn2.k2", error, sizeof(error));
    CHECK(k2 != NULL && h3_gpu_tensor_read_bf16(k2, got_bf16, TOKENS * DIM));
    for (int i = 0; i < TOKENS * DIM; i++)
        CHECK(fabs(from_bf16(got_bf16[i]) - k2_d[i]) <= fabs(k2_d[i]) * (1.0 / 128) + 2e-3);
    h3_gpu_tensor_free(k2);

    /* The fused QKV weight: rows Q, K, V in order; int8 within half a scale
     * step (plus the BF16 rounding of the source), BF16 within one ulp. */
    h3_gpu_tensor *w = h3_vsr_cache_load(&cache, gpu, "blocks.0.qkv.weight", error, sizeof(error));
    CHECK(w != NULL);
    if (int8) {
        int8_t *q = malloc((size_t)3 * DIM * DIM);
        float scales[3 * DIM];
        CHECK(h3_gpu_tensor_read_i8(w, q, (size_t)3 * DIM * DIM));
        CHECK(h3_vsr_cache_read_f32(&cache, "blocks.0.qkv.scales", scales, 3 * DIM, error, sizeof(error)));
        for (int part = 0; part < 3; part++)
            for (int r = 0; r < DIM; r++) {
                float scale = scales[part * DIM + r];
                CHECK(scale > 0.0f);
                for (int c = 0; c < DIM; c++) {
                    float want = m->qkv_w[0][part][(size_t)r * DIM + c];
                    float got = (float)q[((size_t)part * DIM + r) * DIM + c] * scale;
                    CHECK(fabs(got - want) <= 0.5f * scale + fabsf(want) / 128.0f + 1e-6f);
                }
            }
        free(q);
    } else {
        uint16_t *hb = malloc((size_t)3 * DIM * DIM * 2);
        CHECK(h3_gpu_tensor_read_bf16(w, hb, (size_t)3 * DIM * DIM));
        for (int part = 0; part < 3; part++)
            for (size_t i = 0; i < (size_t)DIM * DIM; i++)
                CHECK(hb[(size_t)part * DIM * DIM + i] == to_bf16(m->qkv_w[0][part][i]));
        free(hb);
    }
    h3_gpu_tensor_free(w);
    h3_vsr_cache_close(&cache);

    /* Fingerprints: match, then not after the transformer file changes. */
    CHECK(h3_vsr_cache_matches(cache_path, dir));
    char weights[4096];
    snprintf(weights, sizeof(weights), "%s/transformer/diffusion_pytorch_model.safetensors", dir);
    /* A different stamp per pass: the int8 pass follows the bf16 one, whose
     * stamp the int8 cache was built against. */
    time_t stamp = 1000000000 + (int8 ? 3600 : 0);
    struct utimbuf times = {stamp, stamp};
    CHECK(utime(weights, &times) == 0);
    CHECK(!h3_vsr_cache_matches(cache_path, dir));
}

int main(void) {
    char dir[] = "/tmp/h3vsr-cache-test-XXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    if (getenv("H3_VSR_TEST_KEEP")) printf("model dir %s/model\n", dir);
    char model_dir[4096];
    snprintf(model_dir, sizeof(model_dir), "%s/model", dir);
    model m;
    write_model(model_dir, &m);

    char error[512] = {0};
    h3_vsr_info info;
    CHECK(h3_vsr_inspect_dir(model_dir, &info, error, sizeof(error)));
    CHECK(info.kind == H3_VSR_SWIFTVR && info.output_multiple == 32);
    CHECK(info.first_chunk_frames == 28 && info.chunk_frames == 24);
    CHECK(!info.commercial_use_allowed);
    CHECK(info.weight_bytes_bf16 > info.weight_bytes_int8);
    h3_vsr_wan_config config;
    CHECK(h3_vsr_read_wan_config(model_dir, &config, error, sizeof(error)));
    CHECK(config.layers == LAYERS && config.dim == DIM && config.text_tokens == TOKENS &&
          config.cross_attn_norm == 1 && config.patch_h == 2);

    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    CHECK(gpu != NULL);
    int int8 = h3_gpu_has_int8_mlp(gpu);

    char cache_path[4096];
    CHECK(h3_vsr_cache_path(model_dir, H3_VSR_PRECISION_BF16, cache_path, sizeof(cache_path)));
    CHECK(strstr(cache_path, "swiftvr-dit-bf16.h3vc") != NULL);
    CHECK(h3_vsr_build_cache(model_dir, cache_path, H3_VSR_PRECISION_BF16, "h3_shaders.metal",
                             NULL, NULL, error, sizeof(error)));
    check_cache(model_dir, cache_path, &m, 0, gpu);
    if (int8) {
        CHECK(h3_vsr_cache_path(model_dir, H3_VSR_PRECISION_INT8, cache_path, sizeof(cache_path)));
        CHECK(h3_vsr_build_cache(model_dir, cache_path, H3_VSR_PRECISION_INT8, "h3_shaders.metal",
                                 NULL, NULL, error, sizeof(error)));
        check_cache(model_dir, cache_path, &m, 1, gpu);
    } else {
        printf("note: no tensor hardware, int8 cache not exercised\n");
    }
    /* Precision refused where the hardware lacks it is an error, not a cache. */
    if (!int8)
        CHECK(!h3_vsr_build_cache(model_dir, cache_path, H3_VSR_PRECISION_INT8, "h3_shaders.metal",
                                  NULL, NULL, error, sizeof(error)));
    h3_gpu_free(gpu);
    if (getenv("H3_VSR_TEST_KEEP")) {
        printf("kept %s\n", dir);
    } else {
        char command[4200];
        snprintf(command, sizeof(command), "rm -rf '%s'", dir);
        CHECK(system(command) == 0);
    }
    printf("ok: %d checks (vsr weight cache, %s)\n", tests_run, int8 ? "bf16 + int8" : "bf16");
    return 0;
}
