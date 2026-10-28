/* The Wan DiT runner against a double-precision transcription of the
 * one-step forward pass (diffusers WanTransformer3DModel at t = 1000 with
 * a fixed prompt, mask-free shifted-window attention as LightX2V applies
 * it) on a tiny synthetic model, in full-attention and windowed modes, and
 * int8 against BF16 where the hardware has it. GPU, no weights; `make test`. */
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

enum { LAYERS = 2, HEADS = 2, HEAD_DIM = 64, DIM = 128, FFN = 256, IN = 48,
       TEXT_DIM = 64, FREQ = 256, TOKENS = 8, PATCH = IN * 4 };

static uint64_t rng_state = 0x13198A2E03707344ull;
static float rnd(void) {
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((rng_state >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f;
}

typedef struct { char name[128]; const char *dtype; int ndim; uint64_t shape[5]; void *data; size_t bytes; } tensor_spec;
static tensor_spec specs[512];
static int spec_count;

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
    CHECK(fwrite(&length, 8, 1, f) == 1 && fwrite(header, 1, at, f) == at);
    for (int i = 0; i < count; i++) CHECK(fwrite(tensors[i].data, 1, tensors[i].bytes, f) == tensors[i].bytes);
    fclose(f);
}

static float *add_tensor(const char *name, int ndim, const uint64_t *shape, float scale) {
    size_t n = 1;
    for (int d = 0; d < ndim; d++) n *= shape[d];
    float *v = malloc(n * sizeof(float));
    CHECK(v != NULL);
    for (size_t i = 0; i < n; i++) v[i] = rnd() * scale;
    tensor_spec *t = &specs[spec_count++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->dtype = "F32"; t->ndim = ndim; memcpy(t->shape, shape, sizeof(uint64_t) * (size_t)ndim);
    t->data = v; t->bytes = n * 4;
    return v;
}
static float *add2(const char *name, uint64_t r, uint64_t c, float scale) { uint64_t s[2] = {r, c}; return add_tensor(name, 2, s, scale); }
static float *add1(const char *name, uint64_t n, float scale) { uint64_t s[1] = {n}; return add_tensor(name, 1, s, scale); }
static uint16_t to_bf16(float x) { uint32_t b; memcpy(&b, &x, 4); b += 0x7FFFu + ((b >> 16) & 1u); return (uint16_t)(b >> 16); }
static float from_bf16(uint16_t h) { uint32_t b = (uint32_t)h << 16; float x; memcpy(&x, &b, 4); return x; }

typedef struct {
    float *wq, *bq, *wk, *bk, *wv, *bv, *nq, *nk, *wo, *bo;
    float *wq2, *bq2, *wk2, *bk2, *wv2, *bv2, *nq2, *nk2, *wo2, *bo2, *n2w, *n2b;
    float *w1, *b1, *w2, *b2, *sst;   /* sst [6, DIM] */
} block_w;

typedef struct {
    float *tw1, *tb1, *tw2, *tb2, *pw, *pb, *xw1, *xb1, *xw2, *xb2;
    float *prompt;          /* [TOKENS, TEXT_DIM], bf16-rounded */
    float *patch_w, *patch_b, *proj_w, *proj_b, *sst_out;
    block_w blocks[LAYERS];
} model;

static void write_model(const char *dir, model *m) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/transformer", dir);
    CHECK(mkdir(dir, 0755) == 0 && mkdir(path, 0755) == 0);
    snprintf(path, sizeof(path), "%s/transformer/config.json", dir);
    FILE *f = fopen(path, "w");
    CHECK(f != NULL);
    fprintf(f, "{\"attention_head_dim\": %d, \"cross_attn_norm\": true, \"eps\": 1e-06, \"ffn_dim\": %d,"
               " \"freq_dim\": %d, \"in_channels\": %d, \"num_attention_heads\": %d, \"num_layers\": %d,"
               " \"out_channels\": %d, \"patch_size\": [1, 2, 2], \"text_dim\": %d}\n",
            HEAD_DIM, FFN, FREQ, IN, HEADS, LAYERS, IN, TEXT_DIM);
    fclose(f);
    spec_count = 0;
    const float W = 0.08f;
    m->tw1 = add2("condition_embedder.time_embedder.linear_1.weight", DIM, FREQ, W);
    m->tb1 = add1("condition_embedder.time_embedder.linear_1.bias", DIM, 0.1f);
    m->tw2 = add2("condition_embedder.time_embedder.linear_2.weight", DIM, DIM, W);
    m->tb2 = add1("condition_embedder.time_embedder.linear_2.bias", DIM, 0.1f);
    m->pw = add2("condition_embedder.time_proj.weight", 6 * DIM, DIM, W);
    m->pb = add1("condition_embedder.time_proj.bias", 6 * DIM, 0.1f);
    m->xw1 = add2("condition_embedder.text_embedder.linear_1.weight", DIM, TEXT_DIM, W);
    m->xb1 = add1("condition_embedder.text_embedder.linear_1.bias", DIM, 0.1f);
    m->xw2 = add2("condition_embedder.text_embedder.linear_2.weight", DIM, DIM, W);
    m->xb2 = add1("condition_embedder.text_embedder.linear_2.bias", DIM, 0.1f);
    { uint64_t s[5] = {DIM, IN, 1, 2, 2}; m->patch_w = add_tensor("patch_embedding.weight", 5, s, W); }
    m->patch_b = add1("patch_embedding.bias", DIM, 0.1f);
    m->proj_w = add2("proj_out.weight", PATCH, DIM, W);
    m->proj_b = add1("proj_out.bias", PATCH, 0.1f);
    { uint64_t s[3] = {1, 2, DIM}; m->sst_out = add_tensor("scale_shift_table", 3, s, 0.3f); }
    for (int b = 0; b < LAYERS; b++) {
        block_w *k = &m->blocks[b];
        char name[160];
#define N(suffix) (snprintf(name, sizeof(name), "blocks.%d.%s", b, suffix), name)
        k->wq = add2(N("attn1.to_q.weight"), DIM, DIM, W); k->bq = add1(N("attn1.to_q.bias"), DIM, 0.1f);
        k->wk = add2(N("attn1.to_k.weight"), DIM, DIM, W); k->bk = add1(N("attn1.to_k.bias"), DIM, 0.1f);
        k->wv = add2(N("attn1.to_v.weight"), DIM, DIM, W); k->bv = add1(N("attn1.to_v.bias"), DIM, 0.1f);
        k->nq = add1(N("attn1.norm_q.weight"), DIM, 1.0f); k->nk = add1(N("attn1.norm_k.weight"), DIM, 1.0f);
        k->wo = add2(N("attn1.to_out.0.weight"), DIM, DIM, W); k->bo = add1(N("attn1.to_out.0.bias"), DIM, 0.1f);
        k->wq2 = add2(N("attn2.to_q.weight"), DIM, DIM, W); k->bq2 = add1(N("attn2.to_q.bias"), DIM, 0.1f);
        k->wk2 = add2(N("attn2.to_k.weight"), DIM, DIM, W); k->bk2 = add1(N("attn2.to_k.bias"), DIM, 0.1f);
        k->wv2 = add2(N("attn2.to_v.weight"), DIM, DIM, W); k->bv2 = add1(N("attn2.to_v.bias"), DIM, 0.1f);
        k->nq2 = add1(N("attn2.norm_q.weight"), DIM, 1.0f); k->nk2 = add1(N("attn2.norm_k.weight"), DIM, 1.0f);
        k->wo2 = add2(N("attn2.to_out.0.weight"), DIM, DIM, W); k->bo2 = add1(N("attn2.to_out.0.bias"), DIM, 0.1f);
        k->n2w = add1(N("norm2.weight"), DIM, 1.0f); k->n2b = add1(N("norm2.bias"), DIM, 0.1f);
        k->w1 = add2(N("ffn.net.0.proj.weight"), FFN, DIM, W); k->b1 = add1(N("ffn.net.0.proj.bias"), FFN, 0.1f);
        k->w2 = add2(N("ffn.net.2.weight"), DIM, FFN, W); k->b2 = add1(N("ffn.net.2.bias"), DIM, 0.1f);
        uint64_t s[3] = {1, 6, DIM};
        k->sst = add_tensor(N("scale_shift_table"), 3, s, 0.3f);
#undef N
    }
    snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors", dir);
    write_safetensors(path, specs, spec_count);
    m->prompt = malloc((size_t)TOKENS * TEXT_DIM * sizeof(float));
    uint16_t *half = malloc((size_t)TOKENS * TEXT_DIM * 2);
    for (size_t i = 0; i < (size_t)TOKENS * TEXT_DIM; i++) { half[i] = to_bf16(rnd()); m->prompt[i] = from_bf16(half[i]); }
    tensor_spec prompt = {"prompt_emb", "BF16", 3, {1, TOKENS, TEXT_DIM, 0, 0}, half, (size_t)TOKENS * TEXT_DIM * 2};
    snprintf(path, sizeof(path), "%s/prompt_embedding.safetensors", dir);
    write_safetensors(path, &prompt, 1);
    free(half);
    float dummy[16] = {0};
    tensor_spec reae = {"decoder.1.bias", "F32", 1, {16, 0, 0, 0, 0}, dummy, 64};
    snprintf(path, sizeof(path), "%s/reae.safetensors", dir);
    write_safetensors(path, &reae, 1);
}

/* ---- double reference ---- */
static void linear_d(const double *in, int rows, int in_dim, const float *w, const float *b, int out_dim, double *out) {
    for (int r = 0; r < rows; r++)
        for (int o = 0; o < out_dim; o++) {
            double acc = b ? b[o] : 0.0;
            for (int i = 0; i < in_dim; i++) acc += in[(size_t)r * in_dim + i] * w[(size_t)o * in_dim + i];
            out[(size_t)r * out_dim + o] = acc;
        }
}
static double silu(double x) { return x / (1.0 + exp(-x)); }
static double gelu_tanh(double x) { return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))); }
static void rms_rows(double *x, int rows, int dim, const float *w) {
    for (int r = 0; r < rows; r++) {
        double s = 0.0;
        for (int c = 0; c < dim; c++) s += x[(size_t)r * dim + c] * x[(size_t)r * dim + c];
        double inv = 1.0 / sqrt(s / dim + 1e-6);
        for (int c = 0; c < dim; c++) x[(size_t)r * dim + c] *= inv * w[c];
    }
}
static void layer_norm_rows(const double *x, int rows, int dim, double *out) {
    for (int r = 0; r < rows; r++) {
        double mean = 0.0, var = 0.0;
        for (int c = 0; c < dim; c++) mean += x[(size_t)r * dim + c];
        mean /= dim;
        for (int c = 0; c < dim; c++) { double d = x[(size_t)r * dim + c] - mean; var += d * d; }
        double inv = 1.0 / sqrt(var / dim + 1e-6);
        for (int c = 0; c < dim; c++) out[(size_t)r * dim + c] = (x[(size_t)r * dim + c] - mean) * inv;
    }
}
/* softmax(q k^T / sqrt(d)) v for one head over the given key rows. */
static void attention_d(const double *q, int q_rows, const double *k, const double *v, int kv_rows,
                        int heads, int head_dim, double *out) {
    double *scores = malloc((size_t)kv_rows * sizeof(double));
    double scale = 1.0 / sqrt((double)head_dim);
    for (int h = 0; h < heads; h++)
        for (int i = 0; i < q_rows; i++) {
            double mx = -1e300;
            for (int j = 0; j < kv_rows; j++) {
                double s = 0.0;
                for (int d = 0; d < head_dim; d++)
                    s += q[(size_t)i * heads * head_dim + h * head_dim + d] * k[(size_t)j * heads * head_dim + h * head_dim + d];
                scores[j] = s * scale;
                if (scores[j] > mx) mx = scores[j];
            }
            double sum = 0.0;
            for (int j = 0; j < kv_rows; j++) { scores[j] = exp(scores[j] - mx); sum += scores[j]; }
            for (int d = 0; d < head_dim; d++) {
                double acc = 0.0;
                for (int j = 0; j < kv_rows; j++) acc += scores[j] * v[(size_t)j * heads * head_dim + h * head_dim + d];
                out[(size_t)i * heads * head_dim + h * head_dim + d] = acc / sum;
            }
        }
    free(scores);
}

static void rope_d(double *x, int rows, int heads, int head_dim, const float *cs, const float *sn) {
    int pairs = head_dim / 2;
    for (int r = 0; r < rows; r++)
        for (int h = 0; h < heads; h++)
            for (int p = 0; p < pairs; p++) {
                double *a = x + (size_t)r * heads * head_dim + h * head_dim + 2 * p;
                double c = cs[(size_t)r * pairs + p], s = sn[(size_t)r * pairs + p];
                double x0 = a[0], x1 = a[1];
                a[0] = x0 * c - x1 * s;
                a[1] = x0 * s + x1 * c;
            }
}

/* Reference forward: z_lq [T,h,w,C] -> z_hq. Windowed attention when
 * layouts are given (even/odd by block parity), full attention otherwise. */
static void reference(const model *m, const float *z_lq, int T, int h, int w,
                      const h3_vsr_window_layout *even, const h3_vsr_window_layout *odd,
                      const float *cs, const float *sn, float *z_hq) {
    int h2 = h / 2, w2 = w / 2, N = T * h2 * w2;
    /* constants */
    float emb[FREQ];
    h3_vsr_sinusoidal_embedding(1000.0, FREQ, emb);
    double *e = malloc(FREQ * sizeof(double));
    for (int i = 0; i < FREQ; i++) e[i] = emb[i];
    double hid[DIM], temb[DIM], proj[6 * DIM];
    linear_d(e, 1, FREQ, m->tw1, m->tb1, DIM, hid);
    for (int i = 0; i < DIM; i++) hid[i] = silu(hid[i]);
    linear_d(hid, 1, DIM, m->tw2, m->tb2, DIM, temb);
    for (int i = 0; i < DIM; i++) hid[i] = silu(temb[i]);
    linear_d(hid, 1, DIM, m->pw, m->pb, 6 * DIM, proj);
    double *pr = malloc((size_t)TOKENS * TEXT_DIM * sizeof(double));
    for (size_t i = 0; i < (size_t)TOKENS * TEXT_DIM; i++) pr[i] = m->prompt[i];
    double *th = malloc((size_t)TOKENS * DIM * sizeof(double)), *ctx = malloc((size_t)TOKENS * DIM * sizeof(double));
    linear_d(pr, TOKENS, TEXT_DIM, m->xw1, m->xb1, DIM, th);
    for (size_t i = 0; i < (size_t)TOKENS * DIM; i++) th[i] = gelu_tanh(th[i]);
    linear_d(th, TOKENS, DIM, m->xw2, m->xb2, DIM, ctx);

    /* patchify (c, ph, pw) and embed */
    double *tok = malloc((size_t)N * PATCH * sizeof(double));
    for (int t = 0; t < T; t++) for (int hh = 0; hh < h2; hh++) for (int ww = 0; ww < w2; ww++)
        for (int c = 0; c < IN; c++) for (int ph = 0; ph < 2; ph++) for (int pw = 0; pw < 2; pw++)
            tok[((size_t)(t * h2 + hh) * w2 + ww) * PATCH + h3_vsr_patch_in_index(c, ph, pw)] =
                z_lq[(((size_t)t * h + hh * 2 + ph) * w + ww * 2 + pw) * IN + c];
    double *x = malloc((size_t)N * DIM * sizeof(double));
    linear_d(tok, N, PATCH, m->patch_w, m->patch_b, DIM, x);
    double *hn = malloc((size_t)N * DIM * sizeof(double));
    double *q = malloc((size_t)N * DIM * sizeof(double)), *k = malloc((size_t)N * DIM * sizeof(double));
    double *v = malloc((size_t)N * DIM * sizeof(double)), *att = malloc((size_t)N * DIM * sizeof(double));
    double *o = malloc((size_t)N * DIM * sizeof(double)), *ff = malloc((size_t)N * FFN * sizeof(double));
    double *k2 = malloc((size_t)TOKENS * DIM * sizeof(double)), *v2 = malloc((size_t)TOKENS * DIM * sizeof(double));
    for (int b = 0; b < LAYERS; b++) {
        const block_w *bw = &m->blocks[b];
        double mod[6 * DIM];
        for (int i = 0; i < 6 * DIM; i++) mod[i] = bw->sst[i] + proj[i];
        const double *shift = mod, *scale = mod + DIM, *gate = mod + 2 * DIM;
        const double *c_shift = mod + 3 * DIM, *c_scale = mod + 4 * DIM, *c_gate = mod + 5 * DIM;
        layer_norm_rows(x, N, DIM, hn);
        for (int r = 0; r < N; r++) for (int c = 0; c < DIM; c++) hn[(size_t)r * DIM + c] = hn[(size_t)r * DIM + c] * (1 + scale[c]) + shift[c];
        linear_d(hn, N, DIM, bw->wq, bw->bq, DIM, q);
        linear_d(hn, N, DIM, bw->wk, bw->bk, DIM, k);
        linear_d(hn, N, DIM, bw->wv, bw->bv, DIM, v);
        rms_rows(q, N, DIM, bw->nq);
        rms_rows(k, N, DIM, bw->nk);
        rope_d(q, N, HEADS, HEAD_DIM, cs, sn);
        rope_d(k, N, HEADS, HEAD_DIM, cs, sn);
        const h3_vsr_window_layout *layout = even ? (b % 2 ? odd : even) : NULL;
        if (!layout) {
            attention_d(q, N, k, v, N, HEADS, HEAD_DIM, att);
        } else {
            int L = layout->window_length;
            double *qw = malloc((size_t)L * DIM * sizeof(double)), *kw = malloc((size_t)L * DIM * sizeof(double));
            double *vw = malloc((size_t)L * DIM * sizeof(double)), *aw = malloc((size_t)L * DIM * sizeof(double));
            for (int win = 0; win < layout->window_count; win++) {
                const int32_t *rows = layout->indices + (size_t)win * L;
                for (int i = 0; i < L; i++) {
                    memcpy(qw + (size_t)i * DIM, q + (size_t)rows[i] * DIM, DIM * sizeof(double));
                    memcpy(kw + (size_t)i * DIM, k + (size_t)rows[i] * DIM, DIM * sizeof(double));
                    memcpy(vw + (size_t)i * DIM, v + (size_t)rows[i] * DIM, DIM * sizeof(double));
                }
                attention_d(qw, L, kw, vw, L, HEADS, HEAD_DIM, aw);
                for (int i = 0; i < L; i++)
                    if (layout->owner[rows[i]] == win * L + i)
                        memcpy(att + (size_t)rows[i] * DIM, aw + (size_t)i * DIM, DIM * sizeof(double));
            }
            free(qw); free(kw); free(vw); free(aw);
        }
        linear_d(att, N, DIM, bw->wo, bw->bo, DIM, o);
        for (int r = 0; r < N; r++) for (int c = 0; c < DIM; c++) x[(size_t)r * DIM + c] += o[(size_t)r * DIM + c] * gate[c];
        /* cross-attention */
        layer_norm_rows(x, N, DIM, hn);
        for (int r = 0; r < N; r++) for (int c = 0; c < DIM; c++) hn[(size_t)r * DIM + c] = hn[(size_t)r * DIM + c] * bw->n2w[c] + bw->n2b[c];
        linear_d(hn, N, DIM, bw->wq2, bw->bq2, DIM, q);
        rms_rows(q, N, DIM, bw->nq2);
        linear_d(ctx, TOKENS, DIM, bw->wk2, bw->bk2, DIM, k2);
        rms_rows(k2, TOKENS, DIM, bw->nk2);
        linear_d(ctx, TOKENS, DIM, bw->wv2, bw->bv2, DIM, v2);
        attention_d(q, N, k2, v2, TOKENS, HEADS, HEAD_DIM, att);
        linear_d(att, N, DIM, bw->wo2, bw->bo2, DIM, o);
        for (size_t i = 0; i < (size_t)N * DIM; i++) x[i] += o[i];
        /* ffn */
        layer_norm_rows(x, N, DIM, hn);
        for (int r = 0; r < N; r++) for (int c = 0; c < DIM; c++) hn[(size_t)r * DIM + c] = hn[(size_t)r * DIM + c] * (1 + c_scale[c]) + c_shift[c];
        linear_d(hn, N, DIM, bw->w1, bw->b1, FFN, ff);
        for (size_t i = 0; i < (size_t)N * FFN; i++) ff[i] = gelu_tanh(ff[i]);
        linear_d(ff, N, FFN, bw->w2, bw->b2, DIM, o);
        for (int r = 0; r < N; r++) for (int c = 0; c < DIM; c++) x[(size_t)r * DIM + c] += o[(size_t)r * DIM + c] * c_gate[c];
    }
    /* final norm (scale_shift_table[2] + temb: shift, scale), proj, unpatchify (ph, pw, c) */
    layer_norm_rows(x, N, DIM, hn);
    for (int r = 0; r < N; r++) for (int c = 0; c < DIM; c++)
        hn[(size_t)r * DIM + c] = hn[(size_t)r * DIM + c] * (1 + m->sst_out[DIM + c] + temb[c]) + m->sst_out[c] + temb[c];
    double *vout = malloc((size_t)N * PATCH * sizeof(double));
    linear_d(hn, N, DIM, m->proj_w, m->proj_b, PATCH, vout);
    for (int t = 0; t < T; t++) for (int y = 0; y < h; y++) for (int xx = 0; xx < w; xx++) for (int c = 0; c < IN; c++) {
        size_t token = ((size_t)t * h2 + y / 2) * w2 + xx / 2;
        size_t at = (((size_t)t * h + y) * w + xx) * IN + c;
        z_hq[at] = (float)(z_lq[at] - vout[token * PATCH + h3_vsr_patch_out_index(c, y % 2, xx % 2)]);
    }
    free(e); free(pr); free(th); free(ctx); free(tok); free(x); free(hn); free(q); free(k); free(v);
    free(att); free(o); free(ff); free(k2); free(v2); free(vout);
}

static double psnr(const float *a, const float *b, size_t n) {
    double err = 0.0, peak = 0.0;
    for (size_t i = 0; i < n; i++) { double d = a[i] - b[i]; err += d * d; if (fabs(b[i]) > peak) peak = fabs(b[i]); }
    err /= (double)n;
    return 10.0 * log10(peak * peak / (err > 0 ? err : 1e-30));
}

int main(void) {
    char dir[] = "/tmp/h3vsr-wan-test-XXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    char model_dir[4096], cache_path[4096], error[512] = {0};
    snprintf(model_dir, sizeof(model_dir), "%s/model", dir);
    model m;
    write_model(model_dir, &m);
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    CHECK(gpu != NULL);

    /* Latent grid: T=2, 6x40 cells -> 3x20 tokens, N=120; 16-wide windows
     * along w give two overlapping windows ({0,4} both parities). */
    const int T = 2, LH = 6, LW = 40, N = T * (LH / 2) * (LW / 2);
    size_t latent_n = (size_t)T * LH * LW * IN;
    float *z_lq = malloc(latent_n * sizeof(float));
    for (size_t i = 0; i < latent_n; i++) z_lq[i] = rnd() * 2.0f;
    h3_vsr_window_layout even, odd;
    CHECK(h3_vsr_window_layout_build(T, LH / 2, LW / 2, 16, 16, 0, &even));
    CHECK(h3_vsr_window_layout_build(T, LH / 2, LW / 2, 16, 16, 1, &odd));
    CHECK(even.window_count == 2 && even.tokens == N);
    float *cs = malloc((size_t)N * (HEAD_DIM / 2) * sizeof(float)), *sn = malloc((size_t)N * (HEAD_DIM / 2) * sizeof(float));
    h3_vsr_rope_table(T, LH / 2, LW / 2, 0, HEAD_DIM, cs, sn);
    float *ref_full = malloc(latent_n * sizeof(float)), *ref_win = malloc(latent_n * sizeof(float));
    reference(&m, z_lq, T, LH, LW, NULL, NULL, cs, sn, ref_full);
    reference(&m, z_lq, T, LH, LW, &even, &odd, cs, sn, ref_win);
    /* Windowed and full attention differ (two windows overlap only partly). */
    CHECK(psnr(ref_win, ref_full, latent_n) < 80.0);

    h3_gpu_tensor *z_lq_t = h3_gpu_tensor_from_f32(gpu, z_lq, latent_n);
    h3_gpu_tensor *z_hq_t = h3_gpu_tensor_new_f32(gpu, latent_n);
    h3_gpu_tensor *cs_t = h3_gpu_tensor_from_f32(gpu, cs, (size_t)N * (HEAD_DIM / 2));
    h3_gpu_tensor *sn_t = h3_gpu_tensor_from_f32(gpu, sn, (size_t)N * (HEAD_DIM / 2));
    CHECK(z_lq_t && z_hq_t && cs_t && sn_t);
    float *got = malloc(latent_n * sizeof(float)), *got_bf16_full = malloc(latent_n * sizeof(float));

    int int8 = h3_gpu_has_int8_mlp(gpu);
    for (int pass = 0; pass < (int8 ? 2 : 1); pass++) {
        h3_vsr_precision precision = pass ? H3_VSR_PRECISION_INT8 : H3_VSR_PRECISION_BF16;
        CHECK(h3_vsr_cache_path(model_dir, precision, cache_path, sizeof(cache_path)));
        if (!h3_vsr_build_cache(model_dir, cache_path, precision, "h3_shaders.metal", NULL, NULL, error, sizeof(error)))
            fprintf(stderr, "%s\n", error);
        CHECK(error[0] == '\0');
        h3_vsr_wan *wan = h3_vsr_wan_load(gpu, cache_path, error, sizeof(error));
        if (!wan) fprintf(stderr, "%s\n", error);
        CHECK(wan != NULL && h3_vsr_wan_is_int8(wan) == pass);
        for (int windowed = 0; windowed < 2; windowed++) {
            if (!h3_vsr_wan_predict(wan, z_lq_t, T, LH, LW, windowed ? &even : NULL, windowed ? &odd : NULL,
                                    cs_t, sn_t, z_hq_t, NULL, NULL, error, sizeof(error)))
                fprintf(stderr, "%s\n", error);
            CHECK(error[0] == '\0');
            CHECK(h3_gpu_tensor_read_f32(z_hq_t, got, latent_n));
            const float *ref = windowed ? ref_win : ref_full;
            double db = psnr(got, ref, latent_n);
            printf("  %s, %s attention: %.1f dB vs double reference\n", pass ? "int8" : "bf16",
                   windowed ? "windowed" : "full", db);
            CHECK(db > (pass ? 30.0 : 40.0));
            if (!pass && !windowed) memcpy(got_bf16_full, got, latent_n * sizeof(float));
        }
        if (pass) {
            CHECK(h3_vsr_wan_predict(wan, z_lq_t, T, LH, LW, NULL, NULL, cs_t, sn_t, z_hq_t, NULL, NULL, error, sizeof(error)));
            CHECK(h3_gpu_tensor_read_f32(z_hq_t, got, latent_n));
            printf("  int8 vs bf16 (full): %.1f dB\n", psnr(got, got_bf16_full, latent_n));
        }
        h3_vsr_wan_free(wan);
    }
    /* Deterministic: two runs agree bit for bit. */
    {
        CHECK(h3_vsr_cache_path(model_dir, H3_VSR_PRECISION_BF16, cache_path, sizeof(cache_path)));
        h3_vsr_wan *wan = h3_vsr_wan_load(gpu, cache_path, error, sizeof(error));
        CHECK(wan != NULL);
        float *again = malloc(latent_n * sizeof(float));
        CHECK(h3_vsr_wan_predict(wan, z_lq_t, T, LH, LW, &even, &odd, cs_t, sn_t, z_hq_t, NULL, NULL, error, sizeof(error)));
        CHECK(h3_gpu_tensor_read_f32(z_hq_t, got, latent_n));
        CHECK(h3_vsr_wan_predict(wan, z_lq_t, T, LH, LW, &even, &odd, cs_t, sn_t, z_hq_t, NULL, NULL, error, sizeof(error)));
        CHECK(h3_gpu_tensor_read_f32(z_hq_t, again, latent_n));
        CHECK(memcmp(got, again, latent_n * sizeof(float)) == 0);
        free(again);
        h3_vsr_wan_free(wan);
    }
    h3_vsr_window_layout_free(&even);
    h3_vsr_window_layout_free(&odd);
    h3_gpu_tensor_free(z_lq_t); h3_gpu_tensor_free(z_hq_t); h3_gpu_tensor_free(cs_t); h3_gpu_tensor_free(sn_t);
    h3_gpu_free(gpu);
    char command[4200];
    snprintf(command, sizeof(command), "rm -rf '%s'", dir);
    CHECK(system(command) == 0);
    printf("ok: %d checks (vsr Wan DiT vs double reference)\n", tests_run);
    return 0;
}
