/* See h3_latent_upscale.h. Architecture (author node upscaler_node.py):
 *   conv_in Conv3d(24->512, 3^3, pad 1)
 *   in_blocks: 12 x ResBlockEmb3D, a TemporalConv after every even one
 *   trilinear resize to the target grid (align_corners=False)
 *   out_blocks: the same 12 + 6
 *   GroupNorm(32) -> SiLU -> conv_out Conv3d(512->24, 3^3, pad 1)
 * ResBlockEmb3D(x): h = Conv3d(SiLU(GN(x))); h = GN(h)*(1+scale)+shift;
 *   h = Conv3d(SiLU(h)); x + h, where (scale, shift) = Linear(SiLU(emb))
 *   and emb = Linear(SiLU(Linear([target scale - 1]))) is one 64-vector per
 *   call - so the FiLM folds into the second GroupNorm's affine terms.
 * TemporalConv(x): x + Conv1x1x1(DepthwiseConv3d(SiLU(GN(x)), (5,1,1))).
 * The depthwise temporal convolution runs as a dense conv3d with a
 * block-diagonal weight (512 x 512 x 5, 5 MB); the engine's GroupNorm
 * kernel reduces per "frame", so the whole clip is handed to it as one
 * frame of T*H rows, which is PyTorch's GroupNorm over (C/G, T, H, W). */
#include "h3_latent_upscale.h"
#include "h3_gpu.h"
#include "h3_weights.h"
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LC 24
#define CH 512
#define EMB 64
#define TK 5
#define GROUPS 32
#define GN_EPS 1e-5f
#define MAX_BLOCKS 32
#define H3_LATENT_FILE_MAGIC 0x544c3348u /* "H3LT" */

/* The author node's LATENTS_MEAN / LATENTS_STD (= the H3 VAE's). */
static const float LATENTS_MEAN[LC] = {
    0.858090341091156f, -0.9606591463088989f, 1.0661640167236328f, -0.5090325474739075f,
    -0.2727581858634949f, -1.3675414323806763f, -0.2553254961967468f, -0.26907554268836975f,
    -0.5376840829849243f, -0.0464097298681736f, 0.6657370328903198f, 0.19690127670764923f,
    -0.5460608005523682f, -0.4035342037677765f, -0.23683024942874908f, 0.25928452610969543f,
    -0.30133944749832153f, 0.211341992020607f, -1.1206848621368408f, 0.3581933379173279f,
    -0.04225143790245056f, 0.2604829967021942f, 0.22864092886447906f, 0.7056031823158264f};
static const float LATENTS_STD[LC] = {
    1.2223774194717407f, 1.2767263650894165f, 1.6831774711608887f, 1.7549455165863037f,
    1.5636216402053833f, 2.194143533706665f, 0.9653137922286987f, 1.0569885969161987f,
    0.841948926448822f, 0.7729952931404114f, 1.8955937623977661f, 0.946841835975647f,
    0.7996809482574463f, 0.44988900423049927f, 0.7197399735450745f, 0.6936293244361877f,
    2.961095094680786f, 2.7694199085235596f, 3.0496184825897217f, 2.1088054180145264f,
    3.276226282119751f, 3.1627357006073f, 2.2816812992095947f, 2.6127843856811523f};

typedef struct {
    int temporal;
    /* ResBlockEmb3D */
    h3_gpu_tensor *gn1_w, *gn1_b, *conv1_w, *conv1_b, *conv2_w, *conv2_b;
    float *gn2_w, *gn2_b;        /* host [512] */
    float *emb_w, *emb_b;        /* host [1024, 64], [1024] */
    h3_gpu_tensor *gn2_fold_w, *gn2_fold_b;   /* FiLM folded, per call */
    /* TemporalConv */
    h3_gpu_tensor *tn_w, *tn_b, *dw_w, *dw_b, *pw_w, *pw_b;
} block;

struct h3_latent_upscaler {
    h3_gpu *gpu;
    h3_gpu_tensor *conv_in_w, *conv_in_b, *norm_out_w, *norm_out_b,
                  *conv_out_w, *conv_out_b;
    float embed0_w[EMB], embed0_b[EMB], embed2_w[EMB * EMB], embed2_b[EMB];
    block in_blocks[MAX_BLOCKS], out_blocks[MAX_BLOCKS];
    int in_count, out_count;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

/* ---------------------------------------------------------------- */
/* Weights */

typedef struct {
    h3_latent_upscaler *u;
    h3_weight_store *store;
    char *error;
    size_t error_size;
    int failed;
} loader;

static h3_gpu_tensor *load(loader *l, const char *name, int ndim,
                           const uint64_t *shape) {
    if (l->failed) return NULL;
    h3_gpu_tensor *t = h3_weight_load_f32(l->store, l->u->gpu, name, ndim, shape,
                                          l->error, l->error_size);
    if (!t) l->failed = 1;
    return t;
}

static float *load_host(loader *l, const char *name, int ndim,
                        const uint64_t *shape) {
    h3_gpu_tensor *t = load(l, name, ndim, shape);
    if (!t) return NULL;
    size_t n = h3_gpu_tensor_elements(t);
    float *host = malloc(n * sizeof(float));
    if (!host || !h3_gpu_tensor_read_f32(t, host, n)) {
        fail(l->error, l->error_size, "cannot read %s", name);
        free(host);
        host = NULL;
        l->failed = 1;
    }
    h3_gpu_tensor_free(t);
    return host;
}

static int has(loader *l, const char *name) {
    return h3_weight_find(l->store, name, NULL) != NULL;
}

static int load_block(loader *l, const char *prefix, int index, block *b) {
    char name[160];
    uint64_t c1[1] = {CH}, c3[5] = {CH, CH, 3, 3, 3}, e1[2] = {2 * CH, EMB},
             e0[1] = {2 * CH}, dw[5] = {CH, 1, TK, 1, 1}, pw[5] = {CH, CH, 1, 1, 1};
#define NAME(suffix) (snprintf(name, sizeof(name), "%s.%d.%s", prefix, index, suffix), name)
    memset(b, 0, sizeof(*b));
    if (has(l, NAME("dwconv.weight"))) {
        b->temporal = 1;
        b->tn_w = load(l, NAME("norm.weight"), 1, c1);
        b->tn_b = load(l, NAME("norm.bias"), 1, c1);
        float *dwh = load_host(l, NAME("dwconv.weight"), 5, dw);
        b->dw_b = load(l, NAME("dwconv.bias"), 1, c1);
        b->pw_w = load(l, NAME("pwconv.weight"), 5, pw);
        b->pw_b = load(l, NAME("pwconv.bias"), 1, c1);
        if (dwh) {
            /* dense block-diagonal [out, in, 5, 1, 1]: only in == out. */
            size_t n = (size_t)CH * CH * TK;
            float *dense = calloc(n, sizeof(float));
            if (!dense) {
                fail(l->error, l->error_size, "out of memory");
                l->failed = 1;
            } else {
                for (int c = 0; c < CH; c++)
                    for (int k = 0; k < TK; k++)
                        dense[((size_t)c * CH + c) * TK + k] = dwh[(size_t)c * TK + k];
                b->dw_w = h3_gpu_tensor_from_f32(l->u->gpu, dense, n);
                if (!b->dw_w) l->failed = 1;
                free(dense);
            }
            free(dwh);
        }
        return !l->failed;
    }
    if (!has(l, NAME("in_layers.0.weight"))) return 0;
    b->gn1_w = load(l, NAME("in_layers.0.weight"), 1, c1);
    b->gn1_b = load(l, NAME("in_layers.0.bias"), 1, c1);
    b->conv1_w = load(l, NAME("in_layers.2.weight"), 5, c3);
    b->conv1_b = load(l, NAME("in_layers.2.bias"), 1, c1);
    b->emb_w = load_host(l, NAME("emb_layers.1.weight"), 2, e1);
    b->emb_b = load_host(l, NAME("emb_layers.1.bias"), 1, e0);
    b->gn2_w = load_host(l, NAME("out_norm.weight"), 1, c1);
    b->gn2_b = load_host(l, NAME("out_norm.bias"), 1, c1);
    b->conv2_w = load(l, NAME("out_layers.2.weight"), 5, c3);
    b->conv2_b = load(l, NAME("out_layers.2.bias"), 1, c1);
    b->gn2_fold_w = h3_gpu_tensor_new_f32(l->u->gpu, CH);
    b->gn2_fold_b = h3_gpu_tensor_new_f32(l->u->gpu, CH);
    if (!b->gn2_fold_w || !b->gn2_fold_b) l->failed = 1;
#undef NAME
    return !l->failed;
}

static void free_block(block *b) {
    h3_gpu_tensor *tensors[] = {b->gn1_w, b->gn1_b, b->conv1_w, b->conv1_b,
                                b->conv2_w, b->conv2_b, b->gn2_fold_w,
                                b->gn2_fold_b, b->tn_w, b->tn_b, b->dw_w,
                                b->dw_b, b->pw_w, b->pw_b};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++)
        h3_gpu_tensor_free(tensors[i]);
    free(b->gn2_w); free(b->gn2_b); free(b->emb_w); free(b->emb_b);
}

void h3_latent_upscaler_free(h3_latent_upscaler *u) {
    if (!u) return;
    for (int i = 0; i < u->in_count; i++) free_block(&u->in_blocks[i]);
    for (int i = 0; i < u->out_count; i++) free_block(&u->out_blocks[i]);
    h3_gpu_tensor_free(u->conv_in_w); h3_gpu_tensor_free(u->conv_in_b);
    h3_gpu_tensor_free(u->norm_out_w); h3_gpu_tensor_free(u->norm_out_b);
    h3_gpu_tensor_free(u->conv_out_w); h3_gpu_tensor_free(u->conv_out_b);
    h3_gpu_free(u->gpu);
    free(u);
}

h3_latent_upscaler *h3_latent_upscaler_open(const char *weights_dir,
                                            const char *shader_source_path,
                                            char *error, size_t error_size) {
    h3_latent_upscaler *u = calloc(1, sizeof(*u));
    if (!u) {
        fail(error, error_size, "out of memory");
        return NULL;
    }
    u->gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!u->gpu) {
        free(u);
        return NULL;
    }
    h3_weight_store *store = h3_weight_store_open(weights_dir, error, error_size);
    if (!store) {
        h3_latent_upscaler_free(u);
        return NULL;
    }
    loader l = {u, store, error, error_size, 0};
    uint64_t cin[5] = {CH, LC, 3, 3, 3}, cout[5] = {LC, CH, 3, 3, 3},
             c1[1] = {CH}, l1[1] = {LC}, e01[2] = {EMB, 1}, e0[1] = {EMB},
             e2[2] = {EMB, EMB};
    u->conv_in_w = load(&l, "conv_in.weight", 5, cin);
    u->conv_in_b = load(&l, "conv_in.bias", 1, c1);
    u->norm_out_w = load(&l, "norm_out.weight", 1, c1);
    u->norm_out_b = load(&l, "norm_out.bias", 1, c1);
    u->conv_out_w = load(&l, "conv_out.weight", 5, cout);
    u->conv_out_b = load(&l, "conv_out.bias", 1, l1);
    float *h;
    if ((h = load_host(&l, "embed.0.weight", 2, e01))) { memcpy(u->embed0_w, h, sizeof(u->embed0_w)); free(h); }
    if ((h = load_host(&l, "embed.0.bias", 1, e0))) { memcpy(u->embed0_b, h, sizeof(u->embed0_b)); free(h); }
    if ((h = load_host(&l, "embed.2.weight", 2, e2))) { memcpy(u->embed2_w, h, sizeof(u->embed2_w)); free(h); }
    if ((h = load_host(&l, "embed.2.bias", 1, e0))) { memcpy(u->embed2_b, h, sizeof(u->embed2_b)); free(h); }
    while (!l.failed && u->in_count < MAX_BLOCKS &&
           load_block(&l, "in_blocks", u->in_count, &u->in_blocks[u->in_count]))
        u->in_count++;
    while (!l.failed && u->out_count < MAX_BLOCKS &&
           load_block(&l, "out_blocks", u->out_count, &u->out_blocks[u->out_count]))
        u->out_count++;
    h3_weight_store_free(store);
    if (l.failed || u->in_count < 1 || u->out_count < 1) {
        if (!l.failed) fail(error, error_size, "%s holds no latent upscaler blocks", weights_dir);
        h3_latent_upscaler_free(u);
        return NULL;
    }
    return u;
}

/* ---------------------------------------------------------------- */
/* Forward */

typedef struct { h3_gpu_tensor *t; int T, H, W, C; } act;

static size_t act_elements(const act *a) { return (size_t)a->T * a->H * a->W * a->C; }

static int act_new(h3_gpu *gpu, act *a, int T, int H, int W, int C) {
    a->T = T; a->H = H; a->W = W; a->C = C;
    a->t = h3_gpu_tensor_new_f32(gpu, act_elements(a));
    return a->t != NULL;
}

static void act_free(act *a) {
    h3_gpu_tensor_free(a->t);
    a->t = NULL;
}

/* Tensors are released only after the command buffer that reads them has
 * been submitted (h3_gpu_tensor_free purges the buffer at once). */
#define GARBAGE 64
typedef struct {
    h3_latent_upscaler *u;
    char *error;
    size_t error_size;
    h3_gpu_tensor *garbage[GARBAGE];
    int garbage_count;
} run;

static void defer_free(run *r, act *a) {
    if (!a->t) return;
    if (r->garbage_count < GARBAGE) r->garbage[r->garbage_count++] = a->t;
    else h3_gpu_tensor_free(a->t);   /* never reached at 64 per block */
    a->t = NULL;
}

static void flush_garbage(run *r) {
    for (int i = 0; i < r->garbage_count; i++) h3_gpu_tensor_free(r->garbage[i]);
    r->garbage_count = 0;
}

#define GPU_OP(call) do { if (!(call)) { fail(r->error, r->error_size, "latent upscaler: %s failed: %s", #call, h3_gpu_error(r->u->gpu)); return 0; } } while (0)
#define OOM(call) do { if (!(call)) { fail(r->error, r->error_size, "latent upscaler: out of GPU memory"); return 0; } } while (0)

/* x [T,H,W,C] -> [T+front+back, H+2p, W+2p, C] zero padded. */
static int pad(run *r, const act *x, int front, int back, int p, act *out) {
    h3_gpu *gpu = r->u->gpu;
    act tail;
    OOM(act_new(gpu, &tail, x->T + back, x->H, x->W, x->C));
    GPU_OP(h3_gpu_zero_f32(gpu, tail.t, act_elements(&tail)));
    GPU_OP(h3_gpu_copy_f32(gpu, tail.t, 0, x->t, 0, act_elements(x)));
    OOM(act_new(gpu, out, x->T + front + back, x->H + 2 * p, x->W + 2 * p, x->C));
    GPU_OP(h3_gpu_vsr_pad_f32(gpu, out->t, tail.t, NULL, 1, (uint32_t)tail.T,
                              (uint32_t)x->H, (uint32_t)x->W, (uint32_t)x->C,
                              (uint32_t)front, (uint32_t)p, (uint32_t)p));
    defer_free(r, &tail);
    return 1;
}

/* 3x3x3 convolution, padding 1. */
static int conv3(run *r, const act *x, const h3_gpu_tensor *w,
                 const h3_gpu_tensor *b, int cout, act *out) {
    act padded;
    if (!pad(r, x, 1, 1, 1, &padded)) return 0;
    OOM(act_new(r->u->gpu, out, x->T, x->H, x->W, cout));
    GPU_OP((getenv("H3_LU_CONV3D") ? h3_gpu_conv3d_f32 : h3_gpu_conv3d_as_2d_f32)(r->u->gpu, out->t, padded.t, w, b, 1,
                             (uint32_t)padded.T, (uint32_t)padded.H,
                             (uint32_t)padded.W, (uint32_t)x->C, (uint32_t)cout,
                             3, 3, 3, 1, 1, 1));
    defer_free(r, &padded);
    return 1;
}

/* (5,1,1) temporal convolution, padding 2, dense weight. */
static int conv_t5(run *r, const act *x, const h3_gpu_tensor *w,
                   const h3_gpu_tensor *b, act *out) {
    act padded;
    if (!pad(r, x, 2, 2, 0, &padded)) return 0;
    OOM(act_new(r->u->gpu, out, x->T, x->H, x->W, x->C));
    GPU_OP((getenv("H3_LU_CONV3D") ? h3_gpu_conv3d_f32 : h3_gpu_conv3d_as_2d_f32)(r->u->gpu, out->t, padded.t, w, b, 1,
                             (uint32_t)padded.T, (uint32_t)padded.H,
                             (uint32_t)padded.W, (uint32_t)x->C, (uint32_t)x->C,
                             TK, 1, 1, 1, 1, 1));
    defer_free(r, &padded);
    return 1;
}

static int conv1(run *r, const act *x, const h3_gpu_tensor *w,
                 const h3_gpu_tensor *b, act *out) {
    OOM(act_new(r->u->gpu, out, x->T, x->H, x->W, x->C));
    GPU_OP(h3_gpu_conv3d_as_2d_f32(r->u->gpu, out->t, x->t, w, b, 1,
                                   (uint32_t)x->T, (uint32_t)x->H, (uint32_t)x->W,
                                   (uint32_t)x->C, (uint32_t)x->C, 1, 1, 1, 1, 1, 1));
    return 1;
}

/* GroupNorm(32) over (C/32, T, H, W) + SiLU: the clip as one frame. */
static int gn_silu(run *r, const act *x, const h3_gpu_tensor *w,
                   const h3_gpu_tensor *b, act *out) {
    OOM(act_new(r->u->gpu, out, x->T, x->H, x->W, x->C));
    GPU_OP(h3_gpu_vae_encoder_group_norm_silu_f32(
        r->u->gpu, out->t, x->t, w, b, 1, 1, (uint32_t)(x->T * x->H),
        (uint32_t)x->W, (uint32_t)x->C, GROUPS, GN_EPS));
    return 1;
}

static int add(run *r, const act *a, const act *b, act *out) {
    OOM(act_new(r->u->gpu, out, a->T, a->H, a->W, a->C));
    GPU_OP(h3_gpu_add_scaled_f32(r->u->gpu, out->t, a->t, b->t, 1.0f, 1.0f,
                                 (uint32_t)act_elements(a)));
    return 1;
}

static void dump_sub(h3_gpu *gpu, const act *x, int index, int on) {
    if (!on) return;
    char name[64];
    snprintf(name, sizeof(name), "sub_%d", index);
    const char *dir = getenv("H3_LU_DUMP_DIR");
    if (!dir) return;
    size_t n = act_elements(x);
    float *host = malloc(n * sizeof(float));
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.f32", dir, name);
    if (host && h3_gpu_submit(gpu) && h3_gpu_tensor_read_f32(x->t, host, n)) {
        FILE *file = fopen(path, "wb");
        if (file) { fwrite(host, sizeof(float), n, file); fclose(file); }
    }
    h3_gpu_begin(gpu);
    free(host);
}

/* One block in place: *x is replaced by the block's output. */
static int run_block(run *r, block *b, const float *emb_silu, act *x) {
    h3_gpu *gpu = r->u->gpu;
    act h1, h2, h3, y;
    int debug = getenv("H3_LU_DEBUG_BLOCK") && b == &r->u->out_blocks[atoi(getenv("H3_LU_DEBUG_BLOCK"))];
    if (b->temporal) {
        if (!gn_silu(r, x, b->tn_w, b->tn_b, &h1)) return 0;
        if (!conv_t5(r, &h1, b->dw_w, b->dw_b, &h2)) return 0;
        defer_free(r, &h1);
        if (!conv1(r, &h2, b->pw_w, b->pw_b, &h3)) return 0;
        defer_free(r, &h2);
        if (!add(r, x, &h3, &y)) return 0;
        defer_free(r, &h3);
    } else {
        /* FiLM: GN(h)*(1+scale)+shift == GN with weight*(1+scale),
         * bias*(1+scale)+shift. */
        float fold_w[CH], fold_b[CH];
        for (int c = 0; c < CH; c++) {
            float scale = b->emb_b[c], shift = b->emb_b[CH + c];
            for (int k = 0; k < EMB; k++) {
                scale += b->emb_w[(size_t)c * EMB + k] * emb_silu[k];
                shift += b->emb_w[(size_t)(CH + c) * EMB + k] * emb_silu[k];
            }
            fold_w[c] = b->gn2_w[c] * (1.0f + scale);
            fold_b[c] = b->gn2_b[c] * (1.0f + scale) + shift;
        }
        GPU_OP(h3_gpu_tensor_write_f32(b->gn2_fold_w, fold_w, CH));
        GPU_OP(h3_gpu_tensor_write_f32(b->gn2_fold_b, fold_b, CH));
        if (!gn_silu(r, x, b->gn1_w, b->gn1_b, &h1)) return 0;
        dump_sub(gpu, &h1, 0, debug);
        if (!conv3(r, &h1, b->conv1_w, b->conv1_b, CH, &h2)) return 0;
        dump_sub(gpu, &h2, 1, debug);
        defer_free(r, &h1);
        if (!gn_silu(r, &h2, b->gn2_fold_w, b->gn2_fold_b, &h3)) return 0;
        dump_sub(gpu, &h3, 2, debug);
        defer_free(r, &h2);
        if (!conv3(r, &h3, b->conv2_w, b->conv2_b, CH, &h1)) return 0;
        dump_sub(gpu, &h1, 3, debug);
        defer_free(r, &h3);
        if (!add(r, x, &h1, &y)) return 0;
        defer_free(r, &h1);
    }
    (void)gpu;
    defer_free(r, x);
    *x = y;
    return 1;
}

/* Trilinear resize (align_corners=False) of [T,H,W,C] to [T,oh,ow,C] on
 * the CPU; the time axis keeps its length, so it is bilinear in practice. */
static void resize_cpu(const float *in, int T, int H, int W, int C,
                       float *out, int oh, int ow) {
    for (int y = 0; y < oh; y++) {
        float sy = ((float)y + 0.5f) * (float)H / (float)oh - 0.5f;
        if (sy < 0.0f) sy = 0.0f;
        int y0 = (int)sy; if (y0 > H - 1) y0 = H - 1;
        int y1 = y0 + 1 < H ? y0 + 1 : H - 1;
        float fy = sy - (float)y0;
        for (int x = 0; x < ow; x++) {
            float sx = ((float)x + 0.5f) * (float)W / (float)ow - 0.5f;
            if (sx < 0.0f) sx = 0.0f;
            int x0 = (int)sx; if (x0 > W - 1) x0 = W - 1;
            int x1 = x0 + 1 < W ? x0 + 1 : W - 1;
            float fx = sx - (float)x0;
            float w00 = (1 - fy) * (1 - fx), w01 = (1 - fy) * fx,
                  w10 = fy * (1 - fx), w11 = fy * fx;
            for (int t = 0; t < T; t++) {
                const float *p00 = in + (((size_t)t * H + y0) * W + x0) * C;
                const float *p01 = in + (((size_t)t * H + y0) * W + x1) * C;
                const float *p10 = in + (((size_t)t * H + y1) * W + x0) * C;
                const float *p11 = in + (((size_t)t * H + y1) * W + x1) * C;
                float *o = out + (((size_t)t * oh + y) * ow + x) * C;
                for (int c = 0; c < C; c++)
                    o[c] = w00 * p00[c] + w01 * p01[c] + w10 * p10[c] + w11 * p11[c];
            }
        }
    }
}

/* H3_LU_DUMP_DIR=<dir>: writes every stage's activation ([T,H,W,C] F32)
 * for the parity check against the author node's forward hooks. */
static void dump_stage(h3_gpu *gpu, const act *x, int index) {
    const char *dir = getenv("H3_LU_DUMP_DIR");
    if (!dir || !*dir) return;
    size_t n = act_elements(x);
    float *host = malloc(n * sizeof(float));
    char path[1024];
    snprintf(path, sizeof(path), "%s/stage_%02d.f32", dir, index);
    FILE *file = host && h3_gpu_tensor_read_f32(x->t, host, n) ? fopen(path, "wb") : NULL;
    if (file) {
        fwrite(host, sizeof(float), n, file);
        fclose(file);
    }
    free(host);
    (void)gpu;
}

static int report(h3_progress_callback on_progress, void *opaque, int done,
                  int total, char *error, size_t error_size) {
    if (on_progress && on_progress("latent upscale", done, total, opaque)) {
        fail(error, error_size, "latent upscale cancelled");
        return 0;
    }
    return 1;
}

int h3_latent_upscale(h3_latent_upscaler *u, const float *latent, int T,
                      int H, int W, int oh, int ow, float *out,
                      h3_progress_callback on_progress, void *opaque,
                      char *error, size_t error_size) {
    if (!u || !latent || !out || T < 1 || H < 1 || W < 1 || oh < H || ow < W) {
        fail(error, error_size, "latent upscale: invalid arguments");
        return 0;
    }
    run rr = {u, error, error_size, {0}, 0};
    run *r = &rr;
    h3_gpu *gpu = u->gpu;
    int total = u->in_count + u->out_count + 3, done = 0;
    /* Condition: effective scale - 1 through the two embedding linears,
     * then SiLU once for every block's FiLM linear. */
    float scale = ((float)ow / (float)W + (float)oh / (float)H) * 0.5f;
    float e1[EMB], emb[EMB], emb_silu[EMB];
    for (int k = 0; k < EMB; k++)
        e1[k] = silu(u->embed0_w[k] * (scale - 1.0f) + u->embed0_b[k]);
    for (int j = 0; j < EMB; j++) {
        float v = u->embed2_b[j];
        for (int k = 0; k < EMB; k++) v += u->embed2_w[(size_t)j * EMB + k] * e1[k];
        emb[j] = v;
        emb_silu[j] = silu(v);
    }
    /* Normalize and go channels-last. */
    size_t plane = (size_t)T * H * W;
    float *host = malloc(plane * LC * sizeof(float));
    if (!host) {
        fail(error, error_size, "out of memory");
        return 0;
    }
    for (int c = 0; c < LC; c++)
        for (size_t i = 0; i < plane; i++)
            host[i * LC + c] = (latent[(size_t)c * plane + i] - LATENTS_MEAN[c]) /
                               LATENTS_STD[c];
    act x = {h3_gpu_tensor_from_f32(gpu, host, plane * LC), T, H, W, LC};
    free(host);
    if (!x.t) {
        fail(error, error_size, "latent upscaler: out of GPU memory");
        return 0;
    }
    act y;
    if (!h3_gpu_begin(gpu) || !conv3(r, &x, u->conv_in_w, u->conv_in_b, CH, &y) ||
        !h3_gpu_submit(gpu)) {
        flush_garbage(r);
        act_free(&x);
        return 0;
    }
    act_free(&x);
    flush_garbage(r);
    x = y;
    int stage = 0;
    dump_stage(gpu, &x, stage++);
    if (!report(on_progress, opaque, ++done, total, error, error_size)) { act_free(&x); return 0; }
    for (int i = 0; i < u->in_count; i++) {
        int ok = h3_gpu_begin(gpu) && run_block(r, &u->in_blocks[i], emb_silu, &x) &&
                 h3_gpu_submit(gpu);
        flush_garbage(r);
        if (!ok || !report(on_progress, opaque, ++done, total, error, error_size)) {
            act_free(&x);
            return 0;
        }
        dump_stage(gpu, &x, stage++);
    }
    /* Resize on the CPU. */
    size_t n_in = act_elements(&x);
    float *feat = malloc(n_in * sizeof(float));
    float *resized = malloc((size_t)T * oh * ow * CH * sizeof(float));
    if (!feat || !resized || !h3_gpu_tensor_read_f32(x.t, feat, n_in)) {
        fail(error, error_size, "latent upscaler: cannot read features back");
        free(feat); free(resized); act_free(&x);
        return 0;
    }
    resize_cpu(feat, T, H, W, CH, resized, oh, ow);
    free(feat);
    act_free(&x);
    x.t = h3_gpu_tensor_from_f32(gpu, resized, (size_t)T * oh * ow * CH);
    x.H = oh; x.W = ow;
    free(resized);
    if (!x.t) {
        fail(error, error_size, "latent upscaler: out of GPU memory");
        return 0;
    }
    dump_stage(gpu, &x, stage++);
    if (!report(on_progress, opaque, ++done, total, error, error_size)) { act_free(&x); return 0; }
    for (int i = 0; i < u->out_count; i++) {
        int ok = h3_gpu_begin(gpu) && run_block(r, &u->out_blocks[i], emb_silu, &x) &&
                 h3_gpu_submit(gpu);
        flush_garbage(r);
        if (!ok || !report(on_progress, opaque, ++done, total, error, error_size)) {
            act_free(&x);
            return 0;
        }
        dump_stage(gpu, &x, stage++);
    }
    act h, z;
    int ok = h3_gpu_begin(gpu) && gn_silu(r, &x, u->norm_out_w, u->norm_out_b, &h) &&
             conv3(r, &h, u->conv_out_w, u->conv_out_b, LC, &z) && h3_gpu_submit(gpu);
    flush_garbage(r);
    act_free(&x);
    if (!ok) return 0;
    act_free(&h);
    size_t oplane = (size_t)T * oh * ow;
    float *result = malloc(oplane * LC * sizeof(float));
    ok = result && h3_gpu_tensor_read_f32(z.t, result, oplane * LC);
    act_free(&z);
    if (!ok) {
        fail(error, error_size, "latent upscaler: cannot read the result");
        free(result);
        return 0;
    }
    /* De-normalize and back to [C, T, H, W]. */
    for (int c = 0; c < LC; c++)
        for (size_t i = 0; i < oplane; i++)
            out[(size_t)c * oplane + i] = result[i * LC + c] * LATENTS_STD[c] +
                                          LATENTS_MEAN[c];
    free(result);
    (void)emb;
    return report(on_progress, opaque, ++done, total, error, error_size);
}

int h3_latent_upscale_file(h3_latent_upscaler *u, const char *in_path,
                           const char *out_path, int oh, int ow,
                           h3_progress_callback on_progress, void *opaque,
                           char *error, size_t error_size) {
    int32_t header[9];
    FILE *file = fopen(in_path, "rb");
    if (!file || fread(header, sizeof(header), 1, file) != 1 ||
        header[0] != (int32_t)H3_LATENT_FILE_MAGIC || header[1] != 1 ||
        header[2] != LC || header[6] != 32) {
        if (file) fclose(file);
        fail(error, error_size, "%s is not an H3 clip latent file", in_path);
        return 0;
    }
    int T = header[3], H = header[4], W = header[5], audio_t = header[7];
    size_t video_count = (size_t)LC * T * H * W, audio_count = (size_t)32 * 2 * audio_t;
    size_t out_count = (size_t)LC * T * oh * ow;
    float *video = malloc(video_count * sizeof(float));
    float *audio = malloc(audio_count * sizeof(float));
    float *out = malloc(out_count * sizeof(float));
    int ok = video && audio && out &&
             fread(video, sizeof(float), video_count, file) == video_count &&
             fread(audio, sizeof(float), audio_count, file) == audio_count;
    fclose(file);
    if (!ok) {
        fail(error, error_size, "cannot read %s", in_path);
        goto done;
    }
    ok = h3_latent_upscale(u, video, T, H, W, oh, ow, out, on_progress, opaque,
                           error, error_size);
    if (!ok) goto done;
    header[4] = oh;
    header[5] = ow;
    file = fopen(out_path, "wb");
    ok = file && fwrite(header, sizeof(header), 1, file) == 1 &&
         fwrite(out, sizeof(float), out_count, file) == out_count &&
         fwrite(audio, sizeof(float), audio_count, file) == audio_count;
    if (file && fclose(file) != 0) ok = 0;
    if (!ok) {
        fail(error, error_size, "cannot write %s: %s", out_path, strerror(errno));
        remove(out_path);
    }
done:
    free(video); free(audio); free(out);
    return ok;
}
