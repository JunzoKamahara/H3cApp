/* The Wan DiT of the one-step VSR model (SwiftVR: Wan2.2-TI2V-5B shape),
 * run from an .h3vc cache. UPSCALE_PLAN §4.1 / §6.5. Per block:
 *   h = LN(x)*(1+scale)+shift -> qkv = h Wqkv ; + bias, RMSNorm(q), RMSNorm(k),
 *   RoPE -> [gather into windows] -> SDPA -> [scatter by owner] -> Wo ->
 *   x += (o + bo) * gate
 *   h = LN_affine(x) -> q2 = h Wq2 ; + bias, RMSNorm -> SDPA(q2, K2, V2) -> Wo2
 *   -> x += o2 + bo2
 *   h = LN(x)*(1+c_scale)+c_shift -> gelu(h W1 + b1) W2 -> x += (y + b2) * c_gate
 * then LN(x)*(1+s)+t -> proj_out, and z_hq = z_lq - (v + b). The text
 * side never runs: K2/V2 and every modulation vector come precomputed from
 * the cache. Linears are int8 (per-row, tensor hardware) or BF16 as the
 * cache was built; activations stay BF16 between kernels. */
#include "h3_vsr_internal.h"

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    h3_gpu_tensor *w, *scales;   /* BF16 [out, in], or I8 [out, in] + F32 [out] */
    uint32_t in, out;
} linear;

typedef struct {
    linear qkv, out, q2, out2, fc1, fc2;
    h3_gpu_tensor *qkv_bias, *out_bias, *q2_bias, *out2_bias, *fc1_bias, *fc2_bias;
    h3_gpu_tensor *norm_q, *norm_k, *norm_q2;
    h3_gpu_tensor *norm2_w, *norm2_b;          /* NULL without cross_attn_norm */
    h3_gpu_tensor *k2, *v2;                    /* BF16 [S, D] */
    h3_gpu_tensor *shift, *scale, *gate, *c_shift, *c_scale, *c_gate;   /* F32 [D] */
} wan_block;

struct h3_vsr_wan {
    h3_gpu *gpu;
    int int8;
    int layers, dim, heads, head_dim, ffn, text_tokens, patch_in, patch_out, latent_channels;
    wan_block *blocks;
    linear patch, proj;
    h3_gpu_tensor *patch_bias, *proj_bias, *out_shift, *out_scale;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

/* ---------------------------------------------------------------- */
/* Loading */

static int load_linear(h3_vsr_wan *wan, const h3_vsr_cache *cache, const char *entry,
                       uint32_t out, uint32_t in, int int8, linear *l,
                       char *error, size_t error_size) {
    char name[H3_VSR_CACHE_NAME + 16];
    snprintf(name, sizeof(name), "%s.weight", entry);
    const h3_vsr_cache_entry *e = h3_vsr_cache_find(cache, name);
    if (!e || e->rows != out || e->cols != in) {
        fail(error, error_size, "cache entry %s is missing or not [%u, %u]", name, out, in);
        return 0;
    }
    l->w = h3_vsr_cache_load(cache, wan->gpu, name, error, error_size);
    if (!l->w) return 0;
    l->in = in;
    l->out = out;
    if (int8) {
        snprintf(name, sizeof(name), "%s.scales", entry);
        l->scales = h3_vsr_cache_load(cache, wan->gpu, name, error, error_size);
        if (!l->scales) return 0;
    }
    return 1;
}

static h3_gpu_tensor *load_f32(h3_vsr_wan *wan, const h3_vsr_cache *cache, const char *name,
                               size_t elements, char *error, size_t error_size) {
    const h3_vsr_cache_entry *e = h3_vsr_cache_find(cache, name);
    if (!e || e->dtype != H3_VSR_CACHE_F32 || (size_t)e->rows * e->cols != elements) {
        fail(error, error_size, "cache entry %s is missing or not F32[%zu]", name, elements);
        return NULL;
    }
    return h3_vsr_cache_load(cache, wan->gpu, name, error, error_size);
}

/* Splits an F32 [rows, dim] entry into `rows` separate GPU vectors. */
static int load_rows(h3_vsr_wan *wan, const h3_vsr_cache *cache, const char *name, int rows,
                     h3_gpu_tensor **out, char *error, size_t error_size) {
    size_t dim = (size_t)wan->dim;
    float *host = malloc((size_t)rows * dim * sizeof(float));
    if (!host) return 0;
    if (!h3_vsr_cache_read_f32(cache, name, host, (size_t)rows * dim, error, error_size)) {
        free(host);
        return 0;
    }
    int ok = 1;
    for (int r = 0; r < rows && ok; r++) {
        out[r] = h3_gpu_tensor_from_f32(wan->gpu, host + (size_t)r * dim, dim);
        ok = out[r] != NULL;
    }
    free(host);
    if (!ok) fail(error, error_size, "cannot upload %s", name);
    return ok;
}

static void free_linear(linear *l) {
    h3_gpu_tensor_free(l->w);
    h3_gpu_tensor_free(l->scales);
}

void h3_vsr_wan_free(h3_vsr_wan *wan) {
    if (!wan) return;
    for (int b = 0; wan->blocks && b < wan->layers; b++) {
        wan_block *k = &wan->blocks[b];
        free_linear(&k->qkv); free_linear(&k->out); free_linear(&k->q2);
        free_linear(&k->out2); free_linear(&k->fc1); free_linear(&k->fc2);
        h3_gpu_tensor *vectors[] = {k->qkv_bias, k->out_bias, k->q2_bias, k->out2_bias,
            k->fc1_bias, k->fc2_bias, k->norm_q, k->norm_k, k->norm_q2, k->norm2_w,
            k->norm2_b, k->k2, k->v2, k->shift, k->scale, k->gate, k->c_shift, k->c_scale,
            k->c_gate};
        for (size_t i = 0; i < sizeof(vectors) / sizeof(*vectors); i++) h3_gpu_tensor_free(vectors[i]);
    }
    free(wan->blocks);
    free_linear(&wan->patch);
    free_linear(&wan->proj);
    h3_gpu_tensor_free(wan->patch_bias);
    h3_gpu_tensor_free(wan->proj_bias);
    h3_gpu_tensor_free(wan->out_shift);
    h3_gpu_tensor_free(wan->out_scale);
    free(wan);
}

int h3_vsr_wan_is_int8(const h3_vsr_wan *wan) { return wan && wan->int8; }

h3_vsr_wan *h3_vsr_wan_load(h3_gpu *gpu, const char *cache_path,
                            char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    h3_vsr_cache cache;
    if (!h3_vsr_cache_open(cache_path, &cache, error, error_size)) return NULL;
    h3_vsr_wan *wan = calloc(1, sizeof(*wan));
    if (!wan) {
        h3_vsr_cache_close(&cache);
        return NULL;
    }
    wan->gpu = gpu;
    wan->int8 = cache.header.precision == H3_VSR_PRECISION_INT8;
    wan->layers = (int)cache.header.blocks;
    wan->dim = (int)cache.header.dim;
    wan->heads = (int)cache.header.heads;
    wan->head_dim = (int)cache.header.head_dim;
    wan->ffn = (int)cache.header.ffn;
    wan->text_tokens = (int)cache.header.text_tokens;
    wan->patch_in = (int)cache.header.patch_in;
    wan->latent_channels = (int)cache.header.latent_channels;
    wan->patch_out = wan->latent_channels * H3_VSR_PATCH * H3_VSR_PATCH;
    if (wan->int8 && !h3_gpu_has_int8_mlp(gpu)) {
        fail(error, error_size, "%s is an int8 cache but this GPU has no int8 tensor path", cache_path);
        goto failed;
    }
    wan->blocks = calloc((size_t)wan->layers, sizeof(wan_block));
    if (!wan->blocks) goto failed;
    uint32_t D = (uint32_t)wan->dim, F = (uint32_t)wan->ffn, S = (uint32_t)wan->text_tokens;
    for (int b = 0; b < wan->layers; b++) {
        wan_block *k = &wan->blocks[b];
        char name[H3_VSR_CACHE_NAME];
#define E(suffix) (snprintf(name, sizeof(name), "blocks.%d.%s", b, suffix), name)
        if (!load_linear(wan, &cache, E("qkv"), 3 * D, D, wan->int8, &k->qkv, error, error_size) ||
            !(k->qkv_bias = load_f32(wan, &cache, E("qkv.bias"), 3 * D, error, error_size)) ||
            !(k->norm_q = load_f32(wan, &cache, E("attn1.norm_q"), D, error, error_size)) ||
            !(k->norm_k = load_f32(wan, &cache, E("attn1.norm_k"), D, error, error_size)) ||
            !load_linear(wan, &cache, E("attn1.out"), D, D, wan->int8, &k->out, error, error_size) ||
            !(k->out_bias = load_f32(wan, &cache, E("attn1.out.bias"), D, error, error_size)) ||
            !load_linear(wan, &cache, E("attn2.q"), D, D, wan->int8, &k->q2, error, error_size) ||
            !(k->q2_bias = load_f32(wan, &cache, E("attn2.q.bias"), D, error, error_size)) ||
            !(k->norm_q2 = load_f32(wan, &cache, E("attn2.norm_q"), D, error, error_size)) ||
            !(k->k2 = h3_vsr_cache_load(&cache, gpu, E("attn2.k2"), error, error_size)) ||
            !(k->v2 = h3_vsr_cache_load(&cache, gpu, E("attn2.v2"), error, error_size)) ||
            !load_linear(wan, &cache, E("attn2.out"), D, D, wan->int8, &k->out2, error, error_size) ||
            !(k->out2_bias = load_f32(wan, &cache, E("attn2.out.bias"), D, error, error_size)) ||
            !load_linear(wan, &cache, E("ffn.fc1"), F, D, wan->int8, &k->fc1, error, error_size) ||
            !(k->fc1_bias = load_f32(wan, &cache, E("ffn.fc1.bias"), F, error, error_size)) ||
            !load_linear(wan, &cache, E("ffn.fc2"), D, F, wan->int8, &k->fc2, error, error_size) ||
            !(k->fc2_bias = load_f32(wan, &cache, E("ffn.fc2.bias"), D, error, error_size)))
            goto failed;
        if (h3_vsr_cache_find(&cache, E("norm2.weight"))) {
            if (!(k->norm2_w = load_f32(wan, &cache, E("norm2.weight"), D, error, error_size)) ||
                !(k->norm2_b = load_f32(wan, &cache, E("norm2.bias"), D, error, error_size)))
                goto failed;
        }
        if (h3_gpu_tensor_elements(k->k2) != (size_t)S * D) {
            fail(error, error_size, "%s has the wrong size", name);
            goto failed;
        }
        h3_gpu_tensor *mod[6];
        if (!load_rows(wan, &cache, E("mod"), 6, mod, error, error_size)) goto failed;
        k->shift = mod[0]; k->scale = mod[1]; k->gate = mod[2];
        k->c_shift = mod[3]; k->c_scale = mod[4]; k->c_gate = mod[5];
#undef E
    }
    if (!load_linear(wan, &cache, "patch_embedding", D, (uint32_t)wan->patch_in, 0, &wan->patch, error, error_size) ||
        !(wan->patch_bias = load_f32(wan, &cache, "patch_embedding.bias", D, error, error_size)) ||
        !load_linear(wan, &cache, "proj_out", (uint32_t)wan->patch_out, D, 0, &wan->proj, error, error_size) ||
        !(wan->proj_bias = load_f32(wan, &cache, "proj_out.bias", (size_t)wan->patch_out, error, error_size)))
        goto failed;
    {
        h3_gpu_tensor *out_mod[2];
        if (!load_rows(wan, &cache, "norm_out.mod", 2, out_mod, error, error_size)) goto failed;
        wan->out_shift = out_mod[0];
        wan->out_scale = out_mod[1];
    }
    h3_vsr_cache_close(&cache);
    return wan;
failed:
    if (error && error_size && !error[0]) fail(error, error_size, "cannot load the Wan cache %s", cache_path);
    h3_vsr_cache_close(&cache);
    h3_vsr_wan_free(wan);
    return NULL;
}

/* ---------------------------------------------------------------- */
/* Prediction */

typedef struct {
    h3_gpu *gpu;
    uint32_t N, padded;
    h3_gpu_tensor *q_in;       /* I8 [padded, max_in] int8 activation scratch */
    h3_gpu_tensor *q_scales;   /* F32 [padded] */
    char *error;
    size_t error_size;
} runner;

static int run_linear(runner *r, const h3_vsr_wan *wan, const linear *l, h3_gpu_tensor *out,
                      const h3_gpu_tensor *in) {
    int ok;
    if (l->scales)
        ok = h3_gpu_linear_int8_bf16(r->gpu, out, r->q_in, r->q_scales, in, l->w, l->scales,
                                     r->N, l->in, l->out, 0);
    else
        ok = h3_gpu_linear_bf16(r->gpu, out, in, l->w, NULL, r->N, l->in, l->out);
    /* Padded rows hold whatever the quantizer made of uninitialized input;
     * nothing downstream reads them (every row-wise kernel is given N). */
    (void)wan;
    if (!ok) fail(r->error, r->error_size, "VSR linear [%u -> %u] failed: %s", l->in, l->out,
                  h3_gpu_error(r->gpu));
    return ok;
}

#define OP(expression) do { \
    if (!(expression)) { \
        if (error && error_size && !error[0]) \
            fail(error, error_size, "VSR DiT op failed: %s (%s)", #expression, h3_gpu_error(gpu)); \
        goto failed; \
    } \
} while (0)

static h3_gpu_tensor *upload_u32(h3_gpu *gpu, const int32_t *values, size_t n) {
    uint32_t *u = malloc(n * sizeof(uint32_t));
    if (!u) return NULL;
    for (size_t i = 0; i < n; i++) u[i] = (uint32_t)values[i];
    h3_gpu_tensor *t = h3_gpu_tensor_from_u32(gpu, u, n);
    free(u);
    return t;
}

typedef struct {
    h3_gpu_tensor *indices, *owner;   /* U32 */
    uint32_t windows, length;
} gpu_layout;

static int upload_layout(h3_gpu *gpu, const h3_vsr_window_layout *layout, gpu_layout *out) {
    out->indices = upload_u32(gpu, layout->indices, (size_t)layout->window_count * layout->window_length);
    out->owner = upload_u32(gpu, layout->owner, (size_t)layout->tokens);
    out->windows = (uint32_t)layout->window_count;
    out->length = (uint32_t)layout->window_length;
    return out->indices && out->owner;
}

static void free_layout(gpu_layout *l) {
    h3_gpu_tensor_free(l->indices);
    h3_gpu_tensor_free(l->owner);
}

int h3_vsr_wan_predict(h3_vsr_wan *wan, const h3_gpu_tensor *z_lq,
                       int frames, int height, int width,
                       const h3_vsr_window_layout *even,
                       const h3_vsr_window_layout *odd,
                       const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
                       h3_gpu_tensor *z_hq, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!wan || !z_lq || !z_hq || frames < 1 || height < 2 || width < 2 || height % 2 || width % 2 ||
        !rope_cos || !rope_sin || (even == NULL) != (odd == NULL)) {
        fail(error, error_size, "VSR DiT: bad arguments");
        return 0;
    }
    h3_gpu *gpu = wan->gpu;
    uint32_t D = (uint32_t)wan->dim, F = (uint32_t)wan->ffn, S = (uint32_t)wan->text_tokens;
    uint32_t heads = (uint32_t)wan->heads, head_dim = (uint32_t)wan->head_dim;
    uint32_t N = (uint32_t)frames * (uint32_t)(height / 2) * (uint32_t)(width / 2);
    if (even && (even->tokens != (int)N || odd->tokens != (int)N)) {
        fail(error, error_size, "VSR DiT: window layouts are for %d tokens, not %u", even->tokens, N);
        return 0;
    }
    float scale = 1.0f / sqrtf((float)head_dim);
    uint32_t max_rows = N;
    gpu_layout layouts[2] = {{0}};
    if (even) {
        if (!upload_layout(gpu, even, &layouts[0]) || !upload_layout(gpu, odd, &layouts[1])) {
            fail(error, error_size, "VSR DiT: cannot upload the window layouts");
            free_layout(&layouts[0]); free_layout(&layouts[1]);
            return 0;
        }
        uint32_t r0 = layouts[0].windows * layouts[0].length, r1 = layouts[1].windows * layouts[1].length;
        max_rows = r0 > r1 ? r0 : r1;
    }
    h3_gpu_tensor *identity = NULL;
    if (!even) {
        int32_t *ids = malloc(N * sizeof(int32_t));
        if (ids) for (uint32_t i = 0; i < N; i++) ids[i] = (int32_t)i;
        identity = ids ? upload_u32(gpu, ids, N) : NULL;
        free(ids);
        if (!identity) {
            fail(error, error_size, "VSR DiT: out of memory");
            return 0;
        }
    }

    /* The int8 linears want at least 128 rows and quantize per row, so the
     * token activations are allocated with the row count rounded up to 128
     * (the extra rows are never read by the row-wise kernels, which get N). */
    runner r = {gpu, N, (N + 127u) & ~127u, NULL, NULL, error, error_size};
    uint32_t rows = wan->int8 ? r.padded : N;
    uint32_t g_rows = max_rows > rows ? max_rows : rows;
    if (wan->int8) r.N = rows;
    h3_gpu_tensor *tokens = h3_gpu_tensor_new_bf16(gpu, (size_t)N * wan->patch_in);
    h3_gpu_tensor *x = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * D);
    h3_gpu_tensor *h = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * D);
    h3_gpu_tensor *qkv = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * 3 * D);
    h3_gpu_tensor *qg = h3_gpu_tensor_new_bf16(gpu, (size_t)g_rows * D);
    h3_gpu_tensor *kg = h3_gpu_tensor_new_bf16(gpu, (size_t)max_rows * D);
    h3_gpu_tensor *vg = h3_gpu_tensor_new_bf16(gpu, (size_t)max_rows * D);
    h3_gpu_tensor *ag = h3_gpu_tensor_new_bf16(gpu, (size_t)max_rows * D);
    h3_gpu_tensor *att = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * D);
    h3_gpu_tensor *o = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * D);
    h3_gpu_tensor *f = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * F);
    h3_gpu_tensor *v = h3_gpu_tensor_new_bf16(gpu, (size_t)N * wan->patch_out);
    if (wan->int8) {
        uint32_t max_in = F > D ? F : D;
        r.q_in = h3_gpu_tensor_new_i8(gpu, (size_t)r.padded * max_in);
        r.q_scales = h3_gpu_tensor_new_f32(gpu, r.padded);
    }
    if (!tokens || !x || !h || !qkv || !qg || !kg || !vg || !ag || !att || !o || !f || !v ||
        (wan->int8 && (!r.q_in || !r.q_scales))) {
        fail(error, error_size, "VSR DiT: out of GPU memory for %u tokens", N);
        goto failed;
    }

    OP(h3_gpu_begin(gpu));
    OP(h3_gpu_vsr_patchify_bf16(gpu, tokens, z_lq, (uint32_t)frames, (uint32_t)height,
                                (uint32_t)width, (uint32_t)wan->latent_channels));
    OP(h3_gpu_linear_bf16(gpu, x, tokens, wan->patch.w, NULL, N, (uint32_t)wan->patch_in, D));
    OP(h3_gpu_vsr_bias_gate_add_bf16(gpu, x, x, wan->patch_bias, NULL, NULL, N, D));
    for (int b = 0; b < wan->layers; b++) {
        wan_block *k = &wan->blocks[b];
        const gpu_layout *layout = even ? &layouts[b % 2] : NULL;
        /* Self-attention. */
        OP(h3_gpu_vsr_ln_mod_bf16(gpu, h, x, k->scale, k->shift, NULL, NULL, N, D, 1e-6f));
        OP(run_linear(&r, wan, &k->qkv, qkv, h));
        OP(h3_gpu_vsr_qkv_prepare_bf16(gpu, qkv, k->qkv_bias, k->norm_q, k->norm_k, rope_cos,
                                       rope_sin, N, D, heads, head_dim, 3, 1e-6f));
        if (layout) {
            uint32_t gathered = layout->windows * layout->length;
            OP(h3_gpu_vsr_gather_rows_bf16(gpu, qg, qkv, layout->indices, gathered, D, 3 * D, 0));
            OP(h3_gpu_vsr_gather_rows_bf16(gpu, kg, qkv, layout->indices, gathered, D, 3 * D, D));
            OP(h3_gpu_vsr_gather_rows_bf16(gpu, vg, qkv, layout->indices, gathered, D, 3 * D, 2 * D));
            OP(h3_gpu_sdpa_batched_bf16(gpu, ag, qg, kg, vg, layout->windows, layout->length,
                                        heads, head_dim, scale));
            OP(h3_gpu_vsr_gather_rows_bf16(gpu, att, ag, layout->owner, N, D, D, 0));
        } else {
            OP(h3_gpu_vsr_gather_rows_bf16(gpu, qg, qkv, identity, N, D, 3 * D, 0));
            OP(h3_gpu_vsr_gather_rows_bf16(gpu, kg, qkv, identity, N, D, 3 * D, D));
            OP(h3_gpu_vsr_gather_rows_bf16(gpu, vg, qkv, identity, N, D, 3 * D, 2 * D));
            OP(h3_gpu_sdpa_batched_bf16(gpu, att, qg, kg, vg, 1, N, heads, head_dim, scale));
        }
        OP(run_linear(&r, wan, &k->out, o, att));
        OP(h3_gpu_vsr_bias_gate_add_bf16(gpu, x, o, k->out_bias, k->gate, x, N, D));
        /* Cross-attention with the precomputed text K/V. */
        if (k->norm2_w)
            OP(h3_gpu_vsr_ln_mod_bf16(gpu, h, x, NULL, NULL, k->norm2_w, k->norm2_b, N, D, 1e-6f));
        else
            OP(h3_gpu_copy_bf16(gpu, h, 0, x, 0, (size_t)N * D));
        OP(run_linear(&r, wan, &k->q2, qg, h));
        OP(h3_gpu_vsr_qkv_prepare_bf16(gpu, qg, k->q2_bias, k->norm_q2, NULL, NULL, NULL, N, D,
                                       heads, head_dim, 1, 1e-6f));
        OP(h3_gpu_sdpa_cross_bf16(gpu, att, qg, k->k2, k->v2, N, S, heads, head_dim, scale));
        OP(run_linear(&r, wan, &k->out2, o, att));
        OP(h3_gpu_vsr_bias_gate_add_bf16(gpu, x, o, k->out2_bias, NULL, x, N, D));
        /* Feed-forward. */
        OP(h3_gpu_vsr_ln_mod_bf16(gpu, h, x, k->c_scale, k->c_shift, NULL, NULL, N, D, 1e-6f));
        OP(run_linear(&r, wan, &k->fc1, f, h));
        OP(h3_gpu_vsr_gelu_bias_bf16(gpu, f, k->fc1_bias, N, F));
        OP(run_linear(&r, wan, &k->fc2, o, f));
        OP(h3_gpu_vsr_bias_gate_add_bf16(gpu, x, o, k->fc2_bias, k->c_gate, x, N, D));
        OP(h3_gpu_continue(gpu));
    }
    OP(h3_gpu_vsr_ln_mod_bf16(gpu, h, x, wan->out_scale, wan->out_shift, NULL, NULL, N, D, 1e-6f));
    OP(h3_gpu_linear_bf16(gpu, v, h, wan->proj.w, NULL, N, D, (uint32_t)wan->patch_out));
    OP(h3_gpu_vsr_unpatchify_sub_f32(gpu, z_hq, z_lq, v, wan->proj_bias, (uint32_t)frames,
                                     (uint32_t)height, (uint32_t)width, (uint32_t)wan->latent_channels));
    OP(h3_gpu_submit(gpu));
    {
        h3_gpu_tensor *scratch[] = {tokens, x, h, qkv, qg, kg, vg, ag, att, o, f, v, r.q_in, r.q_scales, identity};
        for (size_t i = 0; i < sizeof(scratch) / sizeof(*scratch); i++) h3_gpu_tensor_free(scratch[i]);
    }
    free_layout(&layouts[0]);
    free_layout(&layouts[1]);
    return 1;
failed:
    {
        h3_gpu_tensor *scratch[] = {tokens, x, h, qkv, qg, kg, vg, ag, att, o, f, v, r.q_in, r.q_scales, identity};
        for (size_t i = 0; i < sizeof(scratch) / sizeof(*scratch); i++) h3_gpu_tensor_free(scratch[i]);
    }
    free_layout(&layouts[0]);
    free_layout(&layouts[1]);
    return 0;
}
