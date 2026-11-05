/* SeedVR2-3B VAE (see h3_seedvr2.h): weights and the forward pass on the
 * F32 / MPSGraph path (M1-M4) and on the TensorOps grid (M5). */
#include "h3_seedvr2_internal.h"

/* --- VAE weights: every causal 3D convolution on one frame is the sum of
 * its temporal taps (the frame is replicated into the causal padding), so
 * each is loaded as a 2D OIHW convolution. Weights are stored (O, kt, kh, kw, I). --- */

static int round64(int value) { return (value + 63) / 64 * 64; }

/* OIHW F32 -> the tap-major BF16 weights of h3_gpu_vsr_tapconv_bf16 on
 * channels padded to 64, tap t = ky * k + kx; the bias padded to out_p. */
static int pack_nax_conv(h3_seedvr2 *m, sv2_conv *c, const float *w, char *error, size_t error_size) {
    int k = c->kernel, taps = k * k;
    c->in_p = round64(c->in); c->out_p = round64(c->out);
    size_t n = (size_t)taps * c->out_p * c->in_p;
    uint16_t *packed = calloc(n, sizeof(uint16_t));
    float *bias = calloc((size_t)c->out_p, sizeof(float));
    if (!packed || !bias) { free(packed); free(bias); fail(error, error_size, "SeedVR2: out of memory"); return 0; }
    for (int ky = 0; ky < k; ky++)
        for (int kx = 0; kx < k; kx++) {
            int t = ky * k + kx;
            for (int o = 0; o < c->out; o++)
                for (int i = 0; i < c->in; i++)
                    packed[((size_t)t * c->out_p + o) * c->in_p + i] =
                        f32_to_bf16(w[(((size_t)o * c->in + i) * k + ky) * k + kx]);
        }
    if (c->b) h3_gpu_tensor_read_f32(c->b, bias, (size_t)c->out);
    c->nw = h3_gpu_tensor_from_bf16(m->gpu, packed, n);
    c->nb = h3_gpu_tensor_from_f32(m->gpu, bias, (size_t)c->out_p);
    free(packed); free(bias);
    if (!c->nw || !c->nb) { fail(error, error_size, "SeedVR2: out of GPU memory for the TensorOps weights"); return 0; }
    return 1;
}

static int load_conv(h3_seedvr2 *m, const h3_st_header *h, const char *prefix, int out, int in,
                     int kernel, sv2_conv *c, char *error, size_t error_size) {
    char name[256];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    const h3_st_tensor *t = h3_st_find(h, name);
    if (!t || t->ndim != 5) { fail(error, error_size, "SeedVR2: %s is missing or not 5-D", name); return 0; }
    int kt = (int)t->shape[1], kh = (int)t->shape[2], kw = (int)t->shape[3];
    if ((int)t->shape[0] != out || (int)t->shape[4] != in || kh != kernel || kw != kernel) {
        fail(error, error_size, "SeedVR2: %s has shape [%d,%d,%d,%d,%d], expected [%d,*,%d,%d,%d]",
             name, (int)t->shape[0], kt, kh, kw, (int)t->shape[4], out, kernel, kernel, in);
        return 0;
    }
    float *w5 = sv2_read_f16_as_f32(h, name, (uint64_t)out * kt * kh * kw * in, error, error_size);
    if (!w5) return 0;
    size_t n = (size_t)out * in * kh * kw;
    float *w = calloc(n, sizeof(float));
    if (!w) { free(w5); fail(error, error_size, "SeedVR2: out of memory"); return 0; }
    for (int o = 0; o < out; o++)
        for (int tt = 0; tt < kt; tt++)
            for (int y = 0; y < kh; y++)
                for (int x = 0; x < kw; x++) {
                    const float *src = w5 + ((((size_t)o * kt + tt) * kh + y) * kw + x) * in;
                    float *dst = w + (((size_t)o * in) * kh + y) * kw + x;
                    for (int i = 0; i < in; i++) dst[(size_t)i * kh * kw] += src[i];
                }
    free(w5);
    snprintf(name, sizeof(name), "%s.bias", prefix);
    c->b = sv2_load_f16_f32(m->gpu, h, name, out, error, error_size);
    c->in = in; c->out = out; c->kernel = kernel;
    if (m->nax_vae) {
        if (!pack_nax_conv(m, c, w, error, error_size)) { free(w); return 0; }
    } else {
        c->w = sv2_upload_f32(m->gpu, w, n);
    }
    free(w);
    if ((!c->w && !c->nw) || !c->b) { fail(error, error_size, "SeedVR2: out of GPU memory for %s", prefix); return 0; }
    return 1;
}

static int load_norm(h3_seedvr2 *m, const h3_st_header *h, const char *prefix, int channels,
                     sv2_norm *n, char *error, size_t error_size) {
    char name[256];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    n->w = sv2_load_f16_f32(m->gpu, h, name, channels, error, error_size);
    snprintf(name, sizeof(name), "%s.bias", prefix);
    n->b = n->w ? sv2_load_f16_f32(m->gpu, h, name, channels, error, error_size) : NULL;
    return n->w && n->b;
}

static int load_resnet(h3_seedvr2 *m, const h3_st_header *h, const char *prefix, int in, int out,
                       sv2_resnet *r, char *error, size_t error_size) {
    char p[256];
    r->in = in; r->out = out;
    snprintf(p, sizeof(p), "%s.norm1", prefix);
    if (!load_norm(m, h, p, in, &r->norm1, error, error_size)) return 0;
    snprintf(p, sizeof(p), "%s.norm2", prefix);
    if (!load_norm(m, h, p, out, &r->norm2, error, error_size)) return 0;
    snprintf(p, sizeof(p), "%s.conv1", prefix);
    if (!load_conv(m, h, p, out, in, 3, &r->conv1, error, error_size)) return 0;
    snprintf(p, sizeof(p), "%s.conv2", prefix);
    if (!load_conv(m, h, p, out, out, 3, &r->conv2, error, error_size)) return 0;
    if (in != out) {
        snprintf(p, sizeof(p), "%s.conv_shortcut", prefix);
        if (!load_conv(m, h, p, out, in, 1, &r->shortcut, error, error_size)) return 0;
    }
    return 1;
}

static int load_attention(h3_seedvr2 *m, const h3_st_header *h, const char *prefix, int channels,
                          sv2_attention *a, char *error, size_t error_size) {
    char p[256];
    snprintf(p, sizeof(p), "%s.group_norm", prefix);
    if (!load_norm(m, h, p, channels, &a->norm, error, error_size)) return 0;
    struct { const char *name; h3_gpu_tensor **w, **b; } parts[4] = {
        {"to_q", &a->wq, &a->bq}, {"to_k", &a->wk, &a->bk}, {"to_v", &a->wv, &a->bv}, {"to_out.0", &a->wo, &a->bo}};
    for (int i = 0; i < 4; i++) {
        snprintf(p, sizeof(p), "%s.%s.weight", prefix, parts[i].name);
        *parts[i].w = sv2_load_f16_f32(m->gpu, h, p, (uint64_t)channels * channels, error, error_size);
        snprintf(p, sizeof(p), "%s.%s.bias", prefix, parts[i].name);
        *parts[i].b = *parts[i].w ? sv2_load_f16_f32(m->gpu, h, p, channels, error, error_size) : NULL;
        if (!*parts[i].b) return 0;
    }
    return 1;
}

int sv2_load_vae(h3_seedvr2 *m, const h3_st_header *h, char *error, size_t error_size) {
    static const int channels[4] = {128, 256, 512, 512};
    sv2_encoder *e = &m->enc;
    sv2_decoder *d = &m->dec;
    char p[256];
    if (!load_conv(m, h, "encoder.conv_in", 128, 3, 3, &e->conv_in, error, error_size)) return 0;
    int c_in = 128;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            snprintf(p, sizeof(p), "encoder.down_blocks.%d.resnets.%d", i, j);
            if (!load_resnet(m, h, p, j == 0 ? c_in : channels[i], channels[i], &e->down[i][j], error, error_size)) return 0;
        }
        c_in = channels[i];
        if (i < 3) {
            snprintf(p, sizeof(p), "encoder.down_blocks.%d.downsamplers.0.conv", i);
            if (!load_conv(m, h, p, channels[i], channels[i], 3, &e->downsample[i], error, error_size)) return 0;
        }
    }
    if (!load_resnet(m, h, "encoder.mid_block.resnets.0", 512, 512, &e->mid_a, error, error_size) ||
        !load_attention(m, h, "encoder.mid_block.attentions.0", 512, &e->mid_attn, error, error_size) ||
        !load_resnet(m, h, "encoder.mid_block.resnets.1", 512, 512, &e->mid_b, error, error_size) ||
        !load_norm(m, h, "encoder.conv_norm_out", 512, &e->norm_out, error, error_size) ||
        !load_conv(m, h, "encoder.conv_out", 2 * SV2_LATENT, 512, 3, &e->conv_out, error, error_size))
        return 0;
    /* decoder: reversed channels (512, 512, 256, 128), 3 resnets each */
    static const int rev[4] = {512, 512, 256, 128};
    if (!load_conv(m, h, "decoder.conv_in", 512, SV2_LATENT, 3, &d->conv_in, error, error_size) ||
        !load_resnet(m, h, "decoder.mid_block.resnets.0", 512, 512, &d->mid_a, error, error_size) ||
        !load_attention(m, h, "decoder.mid_block.attentions.0", 512, &d->mid_attn, error, error_size) ||
        !load_resnet(m, h, "decoder.mid_block.resnets.1", 512, 512, &d->mid_b, error, error_size))
        return 0;
    c_in = 512;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            snprintf(p, sizeof(p), "decoder.up_blocks.%d.resnets.%d", i, j);
            if (!load_resnet(m, h, p, j == 0 ? c_in : rev[i], rev[i], &d->up[i][j], error, error_size)) return 0;
        }
        c_in = rev[i];
        if (i < 3) {
            int tf = i < 2 ? 2 : 1;
            d->up_temporal[i] = tf;
            snprintf(p, sizeof(p), "decoder.up_blocks.%d.upsamplers.0.upscale_conv", i);
            /* 1x1x1 conv (O, 1, 1, 1, I) == linear [O, I] */
            char name[300];
            snprintf(name, sizeof(name), "%s.weight", p);
            d->up_w[i] = sv2_load_f16_f32(m->gpu, h, name, (uint64_t)rev[i] * 4 * tf * rev[i], error, error_size);
            snprintf(name, sizeof(name), "%s.bias", p);
            d->up_b[i] = d->up_w[i] ? sv2_load_f16_f32(m->gpu, h, name, (uint64_t)rev[i] * 4 * tf, error, error_size) : NULL;
            if (!d->up_b[i]) return 0;
            if (m->nax_vae) {
                sv2_conv *u = &d->up_nax[i];
                u->in = rev[i]; u->out = rev[i] * 4 * tf; u->kernel = 1; u->b = d->up_b[i];
                float *wu = malloc((size_t)u->out * u->in * sizeof(float));
                if (!wu || !h3_gpu_tensor_read_f32(d->up_w[i], wu, (size_t)u->out * u->in) ||
                    !pack_nax_conv(m, u, wu, error, error_size)) { free(wu); return 0; }
                free(wu);
                u->b = NULL;   /* owned by up_b */
            }
            snprintf(p, sizeof(p), "decoder.up_blocks.%d.upsamplers.0.conv", i);
            if (!load_conv(m, h, p, rev[i], rev[i], 3, &d->up_conv[i], error, error_size)) return 0;
        }
    }
    if (!load_norm(m, h, "decoder.conv_norm_out", 128, &d->norm_out, error, error_size) ||
        !load_conv(m, h, "decoder.conv_out", 3, 128, 3, &d->conv_out, error, error_size))
        return 0;
    return 1;
}

static void free_conv(sv2_conv *c) {
    h3_gpu_tensor_free(c->w); h3_gpu_tensor_free(c->b);
    h3_gpu_tensor_free(c->nw); h3_gpu_tensor_free(c->nb);
    memset(c, 0, sizeof(*c));
}
static void free_norm(sv2_norm *n) { h3_gpu_tensor_free(n->w); h3_gpu_tensor_free(n->b); memset(n, 0, sizeof(*n)); }
static void free_resnet(sv2_resnet *r) {
    free_norm(&r->norm1); free_norm(&r->norm2);
    free_conv(&r->conv1); free_conv(&r->conv2); free_conv(&r->shortcut);
}
static void free_attention(sv2_attention *a) {
    free_norm(&a->norm);
    h3_gpu_tensor *t[8] = {a->wq, a->bq, a->wk, a->bk, a->wv, a->bv, a->wo, a->bo};
    for (int i = 0; i < 8; i++) h3_gpu_tensor_free(t[i]);
    memset(a, 0, sizeof(*a));
}

void sv2_free_vae(h3_seedvr2 *m) {
    sv2_encoder *e = &m->enc;
    sv2_decoder *d = &m->dec;
    free_conv(&e->conv_in);
    for (int i = 0; i < 4; i++) for (int j = 0; j < 2; j++) free_resnet(&e->down[i][j]);
    for (int i = 0; i < 3; i++) free_conv(&e->downsample[i]);
    free_resnet(&e->mid_a); free_resnet(&e->mid_b); free_attention(&e->mid_attn);
    free_norm(&e->norm_out); free_conv(&e->conv_out);
    free_conv(&d->conv_in);
    free_resnet(&d->mid_a); free_resnet(&d->mid_b); free_attention(&d->mid_attn);
    for (int i = 0; i < 4; i++) for (int j = 0; j < 3; j++) free_resnet(&d->up[i][j]);
    for (int i = 0; i < 3; i++) {
        h3_gpu_tensor_free(d->up_w[i]); h3_gpu_tensor_free(d->up_b[i]);
        free_conv(&d->up_conv[i]); free_conv(&d->up_nax[i]);
    }
    free_norm(&d->norm_out); free_conv(&d->conv_out);
}

/* ---------------------------------------------------------------- */
/* VAE on one frame: F32 channels-last [H, W, C] planes */



int sv2_act_new(h3_seedvr2 *m, act *a, int h, int w, int c) {
    a->h = h; a->w = w; a->c = c;
    a->t = sv2_pool_take(m, (size_t)h * w * c, 0);
    return a->t != NULL;
}

void sv2_act_free(h3_seedvr2 *m, act *a) {
    sv2_pool_give(m, a->t);
    a->t = NULL;
}

/* 3x3 convolution with symmetric zero padding 1 (stride 1), or stride 2
 * with padding (0, 1) right/bottom (the downsampler); the padding is
 * applied inside the convolution graph. */
static int conv3x3(h3_seedvr2 *m, const act *x, const sv2_conv *c, int stride, act *y,
                   char *error, size_t error_size) {
    uint32_t top = stride == 2 ? 0 : 1, left = stride == 2 ? 0 : 1;
    int ph = x->h + (int)top + 1, pw = x->w + (int)left + 1;
    int oh = (ph - 3) / stride + 1, ow = (pw - 3) / stride + 1;
    if (!sv2_act_new(m, y, oh, ow, c->out)) { fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
    if (!h3_gpu_conv2d_padded_f32(m->gpu, y->t, x->t, c->w, c->b, (uint32_t)x->h, (uint32_t)x->w,
                                  (uint32_t)c->in, (uint32_t)c->out, 3, 3, (uint32_t)stride, (uint32_t)stride,
                                  top, 1, left, 1)) {
        fail(error, error_size, "SeedVR2 VAE conv: %s", h3_gpu_error(m->gpu));
        return 0;
    }
    return 1;
}

/* 1x1 convolution = linear over the pixels. */
static int conv1x1(h3_seedvr2 *m, const act *x, const h3_gpu_tensor *w, const h3_gpu_tensor *b,
                   int out, act *y, char *error, size_t error_size) {
    if (!sv2_act_new(m, y, x->h, x->w, out)) { fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
    GPU_OP(h3_gpu_linear_f32(m->gpu, y->t, x->t, w, b, (uint32_t)(x->h * x->w), (uint32_t)x->c, (uint32_t)out));
    return 1;
}

/* in_place: y reuses x's buffer (the kernel reads every element before it
 * writes it; x must not be needed afterwards). */
static int group_norm_silu(h3_seedvr2 *m, const act *x, const sv2_norm *n, act *y, int in_place,
                           char *error, size_t error_size) {
    if (in_place) *y = *x;
    else if (!sv2_act_new(m, y, x->h, x->w, x->c)) { fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
    GPU_OP(h3_gpu_vae_encoder_group_norm_silu_chunked_f32(m->gpu, y->t, x->t, n->w, n->b, 1, 1,
                                                          (uint32_t)x->h, (uint32_t)x->w, (uint32_t)x->c,
                                                          SV2_VAE_GROUPS, SV2_VAE_EPS));
    return 1;
}

static int resnet(h3_seedvr2 *m, const act *x, const sv2_resnet *r, act *y, char *error, size_t error_size) {
    act h1, c1, h2, c2, sc;
    if (!group_norm_silu(m, x, &r->norm1, &h1, 0, error, error_size)) return 0;
    int ok = conv3x3(m, &h1, &r->conv1, 1, &c1, error, error_size);
    sv2_act_free(m, &h1);
    if (!ok) return 0;
    ok = group_norm_silu(m, &c1, &r->norm2, &h2, 1, error, error_size);   /* in place */
    if (!ok) { sv2_act_free(m, &c1); return 0; }
    ok = conv3x3(m, &h2, &r->conv2, 1, &c2, error, error_size);
    sv2_act_free(m, &h2);
    if (!ok) return 0;
    const act *residual = x;
    if (r->shortcut.w) {
        if (!conv1x1(m, x, r->shortcut.w, r->shortcut.b, r->out, &sc, error, error_size)) { sv2_act_free(m, &c2); return 0; }
        residual = &sc;
    }
    *y = c2;   /* in place: y = c2 + residual */
    ok = h3_gpu_add_scaled_f32(m->gpu, y->t, c2.t, residual->t, 1.0f, 1.0f, (uint32_t)((size_t)y->h * y->w * y->c));
    if (r->shortcut.w) sv2_act_free(m, &sc);
    if (!ok) { fail(error, error_size, "SeedVR2 VAE: %s", h3_gpu_error(m->gpu)); return 0; }
    return 1;
}

static int attention(h3_seedvr2 *m, const act *x, const sv2_attention *a, act *y,
                     char *error, size_t error_size) {
    int C = x->c;
    uint32_t n = (uint32_t)(x->h * x->w);
    act g, q, k, v, o, p;
    if (!sv2_act_new(m, &g, x->h, x->w, C)) { fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
    GPU_OP(h3_gpu_sv2_group_norm_f32(m->gpu, g.t, x->t, a->norm.w, a->norm.b, 1, (uint32_t)x->h, (uint32_t)x->w,
                                     (uint32_t)C, SV2_VAE_GROUPS, SV2_VAE_EPS));
    int ok = conv1x1(m, &g, a->wq, a->bq, C, &q, error, error_size) &&
             conv1x1(m, &g, a->wk, a->bk, C, &k, error, error_size) &&
             conv1x1(m, &g, a->wv, a->bv, C, &v, error, error_size);
    sv2_act_free(m, &g);
    if (!ok) return 0;
    if (!sv2_act_new(m, &o, x->h, x->w, C)) { fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
    ok = h3_gpu_sdpa_f32(m->gpu, o.t, q.t, k.t, v.t, n, 1, (uint32_t)C, 1.0f / sqrtf((float)C));
    sv2_act_free(m, &q); sv2_act_free(m, &k); sv2_act_free(m, &v);
    if (!ok) { fail(error, error_size, "SeedVR2 VAE attention: %s", h3_gpu_error(m->gpu)); sv2_act_free(m, &o); return 0; }
    ok = conv1x1(m, &o, a->wo, a->bo, C, &p, error, error_size);
    sv2_act_free(m, &o);
    if (!ok) return 0;
    *y = p;
    GPU_OP(h3_gpu_add_scaled_f32(m->gpu, y->t, p.t, x->t, 1.0f, 1.0f, (uint32_t)((size_t)n * C)));
    return 1;
}

/* Encoder: RGB [H, W, 3] in [-1, 1] (multiples of 16) -> latent [h, w, 32]
 * (mean in the first 16 channels, unscaled). */
/* ---- The same VAE on the TensorOps grid (M5): BF16 rows [guard | (H+2) x
 * (W+2) | tail] with channels padded to 64; every convolution is one
 * h3_gpu_vsr_tapconv_bf16 (3x3: 9 taps over the padded frame, 1x1: 1 tap),
 * the residual added inside it; GroupNorm + SiLU in three chunked passes
 * on the grid; the attention block goes through the F32 path. ---- */

typedef struct { h3_gpu_tensor *t; int h, w, c, cp; uint32_t rows, guard, x_rows; } gact;

static int gact_new(h3_seedvr2 *m, gact *a, int h, int w, int c) {
    memset(a, 0, sizeof(*a));
    a->h = h; a->w = w; a->c = c; a->cp = round64(c);
    uint32_t wp = (uint32_t)w + 2;
    a->rows = (uint32_t)(h + 2) * wp;
    a->guard = (wp + 3 + 127) / 128 * 128;
    a->x_rows = a->guard + a->rows + a->guard + 128;
    a->t = sv2_pool_take(m, (size_t)a->x_rows * a->cp, 1);
    if (!a->t) return 0;
    return h3_gpu_vsr_zero_rows_bf16(m->gpu, a->t, 0, a->guard, (uint32_t)a->cp) &&
           h3_gpu_vsr_zero_rows_bf16(m->gpu, a->t, a->guard + a->rows, a->x_rows - a->guard - a->rows, (uint32_t)a->cp);
}

static void gact_free(h3_seedvr2 *m, gact *a) { sv2_pool_give(m, a->t); a->t = NULL; }

static void gact_offsets(const gact *a, int k, int32_t *offsets) {
    for (int dy = 0; dy < k; dy++)
        for (int dx = 0; dx < k; dx++)
            offsets[dy * k + dx] = (dy - k / 2) * (a->w + 2) + (dx - k / 2);
}

#define GRID_OP(call) do { if (!(call)) { fail(error, error_size, "SeedVR2 VAE (grid): %s", h3_gpu_error(m->gpu)); return 0; } } while (0)

/* out = conv(x) + bias (+ residual), same geometry. */
static int gconv(h3_seedvr2 *m, const gact *x, const sv2_conv *c, const gact *residual, gact *y,
                 char *error, size_t error_size) {
    if (x->cp != c->in_p) { fail(error, error_size, "SeedVR2 VAE (grid): %d padded channels, the conv wants %d", x->cp, c->in_p); return 0; }
    if (!gact_new(m, y, x->h, x->w, c->out)) { fail(error, error_size, "SeedVR2 VAE (grid): out of memory"); return 0; }
    int32_t offsets[9];
    gact_offsets(x, c->kernel, offsets);
    GRID_OP(h3_gpu_vsr_tapconv_bf16(m->gpu, y->t, x->t, NULL, c->nw, NULL, c->nb, residual ? residual->t : NULL,
                                    x->rows, x->x_rows, x->guard, (uint32_t)x->cp, (uint32_t)y->cp,
                                    (uint32_t)(c->kernel * c->kernel), offsets, 0));
    GRID_OP(h3_gpu_vsr_zero_border_bf16(m->gpu, y->t, 1, (uint32_t)y->h, (uint32_t)y->w, (uint32_t)y->cp, y->guard, y->x_rows));
    return 1;
}

/* The stride-2 3x3 convolution with padding (0, 1): the stride-1 result,
 * then its odd interior positions. */
static int gconv_s2(h3_seedvr2 *m, const gact *x, const sv2_conv *c, gact *y, char *error, size_t error_size) {
    gact full;
    if (!gconv(m, x, c, NULL, &full, error, error_size)) return 0;
    if (!gact_new(m, y, x->h / 2, x->w / 2, c->out)) { gact_free(m, &full); fail(error, error_size, "SeedVR2 VAE (grid): out of memory"); return 0; }
    int ok = h3_gpu_sv2_grid_select_bf16(m->gpu, y->t, (uint32_t)y->h, (uint32_t)y->w, y->guard, full.t,
                                         (uint32_t)full.h, (uint32_t)full.w, full.guard, (uint32_t)y->cp, 1, 1);
    gact_free(m, &full);
    if (!ok) { fail(error, error_size, "SeedVR2 VAE (grid): %s", h3_gpu_error(m->gpu)); return 0; }
    return 1;
}

static int ggn(h3_seedvr2 *m, const gact *x, const sv2_norm *n, gact *y, int in_place, int silu,
               char *error, size_t error_size) {
    if (in_place) *y = *x;
    else if (!gact_new(m, y, x->h, x->w, x->c)) { fail(error, error_size, "SeedVR2 VAE (grid): out of memory"); return 0; }
    GRID_OP(h3_gpu_sv2_grid_group_norm_bf16(m->gpu, y->t, x->t, n->w, n->b, (uint32_t)x->h, (uint32_t)x->w,
                                            (uint32_t)x->c, (uint32_t)x->cp, x->guard, x->x_rows,
                                            SV2_VAE_GROUPS, SV2_VAE_EPS, silu));
    /* the norm writes the interior only: a pooled buffer's border rows must
     * be zero before the next convolution's taps read them */
    if (!in_place)
        GRID_OP(h3_gpu_vsr_zero_border_bf16(m->gpu, y->t, 1, (uint32_t)y->h, (uint32_t)y->w, (uint32_t)y->cp, y->guard, y->x_rows));
    return 1;
}

static int gresnet(h3_seedvr2 *m, const gact *x, const sv2_resnet *r, gact *y, char *error, size_t error_size) {
    gact h1, c1, c2, sc;
    if (!ggn(m, x, &r->norm1, &h1, 0, 1, error, error_size)) return 0;
    int ok = gconv(m, &h1, &r->conv1, NULL, &c1, error, error_size);
    gact_free(m, &h1);
    if (!ok) return 0;
    gact h2;
    if (!ggn(m, &c1, &r->norm2, &h2, 1, 1, error, error_size)) { gact_free(m, &c1); return 0; }
    const gact *residual = x;
    if (r->shortcut.nw) {
        if (!gconv(m, x, &r->shortcut, NULL, &sc, error, error_size)) { gact_free(m, &c1); return 0; }
        residual = &sc;
    }
    ok = gconv(m, &h2, &r->conv2, residual, &c2, error, error_size);
    gact_free(m, &c1);
    if (r->shortcut.nw) gact_free(m, &sc);
    if (!ok) return 0;
    *y = c2;
    return 1;
}

/* F32 [h, w, c] <-> grid. */
static int gact_pack(h3_seedvr2 *m, const act *x, gact *g, char *error, size_t error_size) {
    if (!gact_new(m, g, x->h, x->w, x->c)) { fail(error, error_size, "SeedVR2 VAE (grid): out of memory"); return 0; }
    GRID_OP(h3_gpu_vsr_pack_grid_bf16(m->gpu, g->t, x->t, 1, (uint32_t)x->h, (uint32_t)x->w, (uint32_t)x->c,
                                      (uint32_t)g->cp, g->guard, g->x_rows, 0));
    return 1;
}

static int gact_unpack(h3_seedvr2 *m, const gact *g, act *x, char *error, size_t error_size) {
    if (!sv2_act_new(m, x, g->h, g->w, g->c)) { fail(error, error_size, "SeedVR2 VAE (grid): out of memory"); return 0; }
    GRID_OP(h3_gpu_vsr_unpack_grid_f32(m->gpu, x->t, g->t, 1, (uint32_t)g->h, (uint32_t)g->w, (uint32_t)g->c,
                                       (uint32_t)g->cp, g->guard, g->x_rows));
    return 1;
}

/* The mid block's attention through the F32 path (small: latent size). */
static int gattention(h3_seedvr2 *m, const gact *x, const sv2_attention *a, gact *y, char *error, size_t error_size) {
    act f, o;
    if (!gact_unpack(m, x, &f, error, error_size)) return 0;
    int ok = attention(m, &f, a, &o, error, error_size);
    sv2_act_free(m, &f);
    if (!ok) return 0;
    ok = gact_pack(m, &o, y, error, error_size);
    sv2_act_free(m, &o);
    return ok;
}

static int vae_encode_grid(h3_seedvr2 *m, const act *rgb, act *out, int stop, char *error, size_t error_size) {
    sv2_encoder *e = &m->enc;
    gact in, x, y;
    double mark = now_seconds();
    if (!gact_pack(m, rgb, &in, error, error_size)) return 0;
    int ok = gconv(m, &in, &e->conv_in, NULL, &x, error, error_size);
    gact_free(m, &in);
    if (!ok) return 0;
    if (stop == 0) { ok = gact_unpack(m, &x, out, error, error_size); gact_free(m, &x); return ok; }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            if (!gresnet(m, &x, &e->down[i][j], &y, error, error_size)) { gact_free(m, &x); return 0; }
            gact_free(m, &x); x = y;
        }
        if (i < 3) {
            if (!gconv_s2(m, &x, &e->downsample[i], &y, error, error_size)) { gact_free(m, &x); return 0; }
            gact_free(m, &x); x = y;
        }
        if (stop == i + 1) { ok = gact_unpack(m, &x, out, error, error_size); gact_free(m, &x); return ok; }
    }
    trace_sync(m->gpu, &mark, "enc down (grid)");
    if (!gresnet(m, &x, &e->mid_a, &y, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x); x = y;
    if (!gattention(m, &x, &e->mid_attn, &y, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x); x = y;
    if (!gresnet(m, &x, &e->mid_b, &y, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x); x = y;
    if (stop == 5) { ok = gact_unpack(m, &x, out, error, error_size); gact_free(m, &x); return ok; }
    if (!ggn(m, &x, &e->norm_out, &y, 1, 1, error, error_size)) { gact_free(m, &x); return 0; }
    x = y;
    if (!gconv(m, &x, &e->conv_out, NULL, &y, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x);
    ok = gact_unpack(m, &y, out, error, error_size);
    gact_free(m, &y);
    trace_sync(m->gpu, &mark, "enc mid/out (grid)");
    return ok;
}

static int vae_decode_grid(h3_seedvr2 *m, const act *z, act *out, char *error, size_t error_size) {
    sv2_decoder *d = &m->dec;
    gact in, x, y;
    double mark = now_seconds();
    if (!gact_pack(m, z, &in, error, error_size)) return 0;
    int ok = gconv(m, &in, &d->conv_in, NULL, &x, error, error_size);
    gact_free(m, &in);
    if (!ok) return 0;
    if (!gresnet(m, &x, &d->mid_a, &y, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x); x = y;
    if (!gattention(m, &x, &d->mid_attn, &y, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x); x = y;
    if (!gresnet(m, &x, &d->mid_b, &y, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x); x = y;
    trace_sync(m->gpu, &mark, "dec mid (grid)");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            if (!gresnet(m, &x, &d->up[i][j], &y, error, error_size)) { gact_free(m, &x); return 0; }
            gact_free(m, &x); x = y;
        }
        if (i < 3) {
            int C = x.c, tf = d->up_temporal[i];
            gact u, s;
            if (!gconv(m, &x, &d->up_nax[i], NULL, &u, error, error_size)) { gact_free(m, &x); return 0; }
            gact_free(m, &x);
            if (!gact_new(m, &s, 2 * u.h, 2 * u.w, C)) { gact_free(m, &u); fail(error, error_size, "SeedVR2 VAE (grid): out of memory"); return 0; }
            ok = h3_gpu_sv2_grid_shuffle_bf16(m->gpu, s.t, (uint32_t)s.cp, s.guard, u.t, (uint32_t)u.h, (uint32_t)u.w,
                                              (uint32_t)u.cp, u.guard, (uint32_t)C, (uint32_t)tf);
            gact_free(m, &u);
            if (!ok) { gact_free(m, &s); fail(error, error_size, "SeedVR2 VAE (grid): %s", h3_gpu_error(m->gpu)); return 0; }
            if (!gconv(m, &s, &d->up_conv[i], NULL, &x, error, error_size)) { gact_free(m, &s); return 0; }
            gact_free(m, &s);
            if (!h3_gpu_continue(m->gpu)) { gact_free(m, &x); fail(error, error_size, "SeedVR2 VAE (grid): %s", h3_gpu_error(m->gpu)); return 0; }
        }
        char label[32];
        snprintf(label, sizeof(label), "dec up%d (grid)", i);
        trace_sync(m->gpu, &mark, label);
    }
    if (!ggn(m, &x, &d->norm_out, &y, 1, 1, error, error_size)) { gact_free(m, &x); return 0; }
    x = y;
    if (!gconv(m, &x, &d->conv_out, NULL, &y, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x);
    ok = gact_unpack(m, &y, out, error, error_size);
    gact_free(m, &y);
    trace_sync(m->gpu, &mark, "dec out (grid)");
    return ok;
}

/* stop: 0 after conv_in, 1..4 after down block i-1, 5 after the mid block,
 * 6 (or more) the latent. */
int sv2_vae_encode_stage(h3_seedvr2 *m, const act *rgb, act *out, int stop, char *error, size_t error_size) {
    if (m->nax_vae) return vae_encode_grid(m, rgb, out, stop, error, error_size);
    sv2_encoder *e = &m->enc;
    act x, y;
    double mark = now_seconds();
    if (!conv3x3(m, rgb, &e->conv_in, 1, &x, error, error_size)) return 0;
    if (stop == 0) { *out = x; return 1; }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            if (!resnet(m, &x, &e->down[i][j], &y, error, error_size)) { sv2_act_free(m, &x); return 0; }
            sv2_act_free(m, &x); x = y;
        }
        if (i < 3) {
            if (!conv3x3(m, &x, &e->downsample[i], 2, &y, error, error_size)) { sv2_act_free(m, &x); return 0; }
            sv2_act_free(m, &x); x = y;
        }
        if (stop == i + 1) { *out = x; return 1; }
    }
    trace_sync(m->gpu, &mark, "enc down");
    if (!resnet(m, &x, &e->mid_a, &y, error, error_size)) { sv2_act_free(m, &x); return 0; }
    sv2_act_free(m, &x); x = y;
    if (!attention(m, &x, &e->mid_attn, &y, error, error_size)) { sv2_act_free(m, &x); return 0; }
    sv2_act_free(m, &x); x = y;
    if (!resnet(m, &x, &e->mid_b, &y, error, error_size)) { sv2_act_free(m, &x); return 0; }
    sv2_act_free(m, &x); x = y;
    if (stop == 5) { *out = x; return 1; }
    if (!group_norm_silu(m, &x, &e->norm_out, &y, 1, error, error_size)) { sv2_act_free(m, &x); return 0; }
    x = y;   /* in place: the same buffer */
    if (!conv3x3(m, &x, &e->conv_out, 1, out, error, error_size)) { sv2_act_free(m, &x); return 0; }
    sv2_act_free(m, &x);
    trace_sync(m->gpu, &mark, "enc mid/out");
    return 1;
}

int sv2_vae_encode(h3_seedvr2 *m, const act *rgb, act *out, char *error, size_t error_size) {
    return sv2_vae_encode_stage(m, rgb, out, 99, error, error_size);
}

/* Decoder: latent [h, w, 16] (already divided by the scaling factor) ->
 * RGB [8h, 8w, 3] in about [-1, 1]. */
int sv2_vae_decode(h3_seedvr2 *m, const act *z, act *out, char *error, size_t error_size) {
    if (m->nax_vae) return vae_decode_grid(m, z, out, error, error_size);
    sv2_decoder *d = &m->dec;
    act x, y;
    double mark = now_seconds();
    if (!conv3x3(m, z, &d->conv_in, 1, &x, error, error_size)) return 0;
    if (!resnet(m, &x, &d->mid_a, &y, error, error_size)) { sv2_act_free(m, &x); return 0; }
    sv2_act_free(m, &x); x = y;
    if (!attention(m, &x, &d->mid_attn, &y, error, error_size)) { sv2_act_free(m, &x); return 0; }
    sv2_act_free(m, &x); x = y;
    if (!resnet(m, &x, &d->mid_b, &y, error, error_size)) { sv2_act_free(m, &x); return 0; }
    sv2_act_free(m, &x); x = y;
    trace_sync(m->gpu, &mark, "dec mid");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            if (!resnet(m, &x, &d->up[i][j], &y, error, error_size)) { sv2_act_free(m, &x); return 0; }
            sv2_act_free(m, &x); x = y;
        }
        if (i < 3) {
            int C = x.c, tf = d->up_temporal[i];
            act u, s;
            if (!conv1x1(m, &x, d->up_w[i], d->up_b[i], C * 4 * tf, &u, error, error_size)) { sv2_act_free(m, &x); return 0; }
            sv2_act_free(m, &x);
            if (!sv2_act_new(m, &s, 2 * u.h, 2 * u.w, C)) { sv2_act_free(m, &u); fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
            int ok = h3_gpu_sv2_shuffle_f32(m->gpu, s.t, u.t, (uint32_t)u.h, (uint32_t)u.w, (uint32_t)C, (uint32_t)tf);
            sv2_act_free(m, &u);
            if (!ok) { sv2_act_free(m, &s); fail(error, error_size, "SeedVR2 VAE: %s", h3_gpu_error(m->gpu)); return 0; }
            if (!conv3x3(m, &s, &d->up_conv[i], 1, &x, error, error_size)) { sv2_act_free(m, &s); return 0; }
            sv2_act_free(m, &s);
            if (!h3_gpu_continue(m->gpu)) { sv2_act_free(m, &x); fail(error, error_size, "SeedVR2 VAE: %s", h3_gpu_error(m->gpu)); return 0; }
        }
        char label[32];
        snprintf(label, sizeof(label), "dec up%d", i);
        trace_sync(m->gpu, &mark, label);
    }
    if (!group_norm_silu(m, &x, &d->norm_out, &y, 1, error, error_size)) { sv2_act_free(m, &x); return 0; }
    x = y;   /* in place: the same buffer */
    if (!conv3x3(m, &x, &d->conv_out, 1, out, error, error_size)) { sv2_act_free(m, &x); return 0; }
    sv2_act_free(m, &x);
    trace_sync(m->gpu, &mark, "dec out");
    return 1;
}
