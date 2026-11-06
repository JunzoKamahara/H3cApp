/* SeedVR2-3B VAE (see h3_seedvr2.h): weights and the forward pass on the
 * F32 / MPSGraph path (M1-M4) and on the TensorOps grid (M5). */
#include "h3_seedvr2_internal.h"

/* --- VAE weights, stored (O, kt, kh, kw, I). For still images every causal
 * 3D convolution on one frame is the sum of its temporal taps (the frame is
 * replicated into the causal padding), so each is loaded folded as a 2D
 * OIHW convolution; for video (m->vae_temporal) the taps are kept, OIDHW on
 * the F32 path and 27 grid taps on the TensorOps path. --- */

static int round64(int value) { return (value + 63) / 64 * 64; }

/* OI(D)HW F32 (kt = c->kt temporal taps) -> the tap-major BF16 weights of
 * h3_gpu_vsr_tapconv_bf16 on channels padded to 64, tap t = (tt * k + ky)
 * * k + kx; the bias padded to out_p. */
static int pack_nax_conv(h3_seedvr2 *m, sv2_conv *c, const float *w, char *error, size_t error_size) {
    int k = c->kernel, kt = c->kt > 0 ? c->kt : 1, taps = kt * k * k;
    c->in_p = round64(c->in); c->out_p = round64(c->out);
    size_t n = (size_t)taps * c->out_p * c->in_p;
    uint16_t *packed = calloc(n, sizeof(uint16_t));
    float *bias = calloc((size_t)c->out_p, sizeof(float));
    if (!packed || !bias) { free(packed); free(bias); fail(error, error_size, "SeedVR2: out of memory"); return 0; }
    for (int tt = 0; tt < kt; tt++)
        for (int ky = 0; ky < k; ky++)
            for (int kx = 0; kx < k; kx++) {
                int t = (tt * k + ky) * k + kx;
                for (int o = 0; o < c->out; o++)
                    for (int i = 0; i < c->in; i++)
                        packed[((size_t)t * c->out_p + o) * c->in_p + i] =
                            f32_to_bf16(w[((((size_t)o * c->in + i) * kt + tt) * k + ky) * k + kx]);
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
    int keep = m->vae_temporal ? kt : 1;   /* temporal taps kept (video) or summed (image) */
    size_t n = (size_t)out * in * keep * kh * kw;
    float *w = calloc(n, sizeof(float));
    if (!w) { free(w5); fail(error, error_size, "SeedVR2: out of memory"); return 0; }
    if (keep == 1) {
        for (int o = 0; o < out; o++)
            for (int tt = 0; tt < kt; tt++)
                for (int y = 0; y < kh; y++)
                    for (int x = 0; x < kw; x++) {
                        const float *src = w5 + ((((size_t)o * kt + tt) * kh + y) * kw + x) * in;
                        float *dst = w + (((size_t)o * in) * kh + y) * kw + x;
                        for (int i = 0; i < in; i++) dst[(size_t)i * kh * kw] += src[i];
                    }
    } else {
        for (int o = 0; o < out; o++)
            for (int tt = 0; tt < kt; tt++)
                for (int y = 0; y < kh; y++)
                    for (int x = 0; x < kw; x++) {
                        const float *src = w5 + ((((size_t)o * kt + tt) * kh + y) * kw + x) * in;
                        for (int i = 0; i < in; i++)
                            w[((((size_t)o * in + i) * kt + tt) * kh + y) * kw + x] = src[i];
                    }
    }
    free(w5);
    snprintf(name, sizeof(name), "%s.bias", prefix);
    c->b = sv2_load_f16_f32(m->gpu, h, name, out, error, error_size);
    c->in = in; c->out = out; c->kernel = kernel; c->kt = keep;
    c->slot = m->conv_slots < SV2_MAX_CONVS ? m->conv_slots++ : SV2_MAX_CONVS - 1;
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
    m->conv_slots = 0;
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
                u->in = rev[i]; u->out = rev[i] * 4 * tf; u->kernel = 1; u->kt = 1; u->b = d->up_b[i];
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
/* VAE forward on F32 channels-last frames [f, H, W, C] (MPSGraph, every
 * Mac) or on the TensorOps grid (M5). ctx NULL is the still-image path: one
 * frame, folded weights and the original kernels, bit for bit. With a
 * causal context the frames of a clip go through the 3D causal convolutions
 * (temporal taps a frame apart, the head extended by copies of frame 0 or
 * by the previous slice's frames); GroupNorm and the mid-block attention
 * stay per frame, as in the reference. */

int sv2_act_new(h3_seedvr2 *m, act *a, int h, int w, int c) {
    a->h = h; a->w = w; a->c = c; a->f = 1;
    a->t = sv2_pool_take(m, (size_t)h * w * c, 0);
    return a->t != NULL;
}

static int act_new_f(h3_seedvr2 *m, act *a, int f, int h, int w, int c) {
    a->h = h; a->w = w; a->c = c; a->f = f;
    a->t = sv2_pool_take(m, (size_t)f * h * w * c, 0);
    return a->t != NULL;
}

void sv2_act_free(h3_seedvr2 *m, act *a) {
    sv2_pool_give(m, a->t);
    a->t = NULL;
}

void sv2_causal_reset(sv2_causal *ctx) {
    if (!ctx) return;
    for (int i = 0; i < SV2_MAX_CONVS; i++) {
        h3_gpu_tensor_free(ctx->mem[i]);
        ctx->mem[i] = NULL;
        ctx->mem_frames[i] = 0;
    }
    ctx->mode = SV2_CAUSAL_DISABLED;
}

/* Replaces conv `slot`'s state with `fresh` (k frames). */
static void causal_keep(h3_seedvr2 *m, sv2_causal *ctx, int slot, h3_gpu_tensor *fresh, int k) {
    sv2_pool_give(m, ctx->mem[slot]);
    ctx->mem[slot] = fresh;
    ctx->mem_frames[slot] = k;
}

/* 3x3 convolution with symmetric zero padding 1 (stride 1), or stride 2
 * with padding (0, 1) right/bottom (the downsampler); the padding is
 * applied inside the convolution graph. */
static int conv3x3_image(h3_seedvr2 *m, const act *x, const sv2_conv *c, int stride, act *y,
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

/* The causal 3D version on F32 frames, one output frame at a time: the kt
 * input frames of that frame (from the clip, the state, or copies of frame
 * 0) are gathered into a small staging buffer, spatially zero-padded
 * (h3_gpu_sv2_pad3d_f32 with no front frames) and convolved as one Conv3d
 * of depth kt (per-tap 2D convolutions). Memory stays at the input, the
 * output and a few frames, instead of a padded copy of the whole clip.
 * Stride 2 is spatial (padding (0, 1)) and, for a conv with temporal taps,
 * temporal too. The state (the last kt - stride input frames) is kept in
 * BF16, half the memory. */
static int conv3x3_clip(h3_seedvr2 *m, const act *x, const sv2_conv *c, int stride, act *y,
                        sv2_causal *ctx, char *error, size_t error_size) {
    int f = act_frames(x), kt = c->kt, st = kt > 1 ? stride : 1;
    uint32_t top = stride == 2 ? 0 : 1, left = top;
    int ph = x->h + (int)top + 1, pw = x->w + (int)left + 1;
    int oh = (ph - 3) / stride + 1, ow = (pw - 3) / stride + 1;
    size_t plane = (size_t)x->h * x->w * c->in, oplane = (size_t)oh * ow * c->out;
    if (kt == 1) {
        /* no temporal taps (1x3x3): every frame on its own, padded in the graph */
        if (!act_new_f(m, y, f, oh, ow, c->out)) { fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
        h3_gpu_tensor *in1 = sv2_pool_take(m, plane, 0), *out1 = sv2_pool_take(m, oplane, 0);
        int ok = in1 && out1;
        for (int t = 0; t < f && ok; t++)
            ok = h3_gpu_copy_f32(m->gpu, in1, 0, x->t, (size_t)t * plane, plane) &&
                 h3_gpu_conv2d_padded_f32(m->gpu, out1, in1, c->w, c->b, (uint32_t)x->h, (uint32_t)x->w,
                                          (uint32_t)c->in, (uint32_t)c->out, 3, 3, (uint32_t)stride, (uint32_t)stride,
                                          top, 1, left, 1) &&
                 h3_gpu_copy_f32(m->gpu, y->t, (size_t)t * oplane, out1, 0, oplane);
        sv2_pool_give(m, in1); sv2_pool_give(m, out1);
        if (!ok) { fail(error, error_size, "SeedVR2 VAE conv: %s", h3_gpu_error(m->gpu)); return 0; }
        return 1;
    }
    /* the frames before frame 0: the state (ACTIVE) or copies of frame 0 */
    h3_gpu_tensor *head = NULL;
    int front = kt - 1;
    if (ctx->mode == SV2_CAUSAL_ACTIVE && ctx->mem[c->slot]) {
        front = ctx->mem_frames[c->slot];
        head = sv2_pool_take(m, (size_t)front * plane, 0);
        if (!head || !h3_gpu_cast_bf16_to_f32(m->gpu, head, ctx->mem[c->slot], (uint32_t)((size_t)front * plane))) {
            sv2_pool_give(m, head);
            fail(error, error_size, "SeedVR2 VAE: %s", h3_gpu_error(m->gpu));
            return 0;
        }
    }
    int depth = front + f, of = (depth - kt) / st + 1;
    if (depth < kt) { sv2_pool_give(m, head); fail(error, error_size, "SeedVR2 VAE: %d frames for a %d-tap causal conv", depth, kt); return 0; }
    h3_gpu_tensor *window = sv2_pool_take(m, (size_t)kt * plane, 0);
    h3_gpu_tensor *padded = sv2_pool_take(m, (size_t)kt * ph * pw * c->in, 0);
    h3_gpu_tensor *out1 = sv2_pool_take(m, oplane, 0);
    int ok = window && padded && out1 && act_new_f(m, y, of, oh, ow, c->out);
    if (!ok) fail(error, error_size, "SeedVR2 VAE: out of memory");
    for (int o = 0; o < of && ok; o++) {
        for (int k = 0; k < kt && ok; k++) {
            int e = o * st + k;   /* frame of [head | x] */
            if (e >= front) ok = h3_gpu_copy_f32(m->gpu, window, (size_t)k * plane, x->t, (size_t)(e - front) * plane, plane);
            else if (head) ok = h3_gpu_copy_f32(m->gpu, window, (size_t)k * plane, head, (size_t)e * plane, plane);
            else ok = h3_gpu_copy_f32(m->gpu, window, (size_t)k * plane, x->t, 0, plane);
        }
        ok = ok &&
             h3_gpu_sv2_pad3d_f32(m->gpu, padded, window, NULL, 0, (uint32_t)kt, (uint32_t)x->h, (uint32_t)x->w,
                                  (uint32_t)c->in, 0, top, 1, left, 1) &&
             h3_gpu_conv3d_as_2d_f32(m->gpu, out1, padded, c->w, c->b, 1, (uint32_t)kt, (uint32_t)ph, (uint32_t)pw,
                                     (uint32_t)c->in, (uint32_t)c->out, (uint32_t)kt, 3, 3, 1, (uint32_t)stride,
                                     (uint32_t)stride) &&
             h3_gpu_copy_f32(m->gpu, y->t, (size_t)o * oplane, out1, 0, oplane);
    }
    sv2_pool_give(m, padded); sv2_pool_give(m, out1);
    if (!ok) { sv2_pool_give(m, window); sv2_pool_give(m, head); fail(error, error_size, "SeedVR2 VAE conv: %s", h3_gpu_error(m->gpu)); return 0; }
    /* the state: the last kt - st frames of [head | x], stored as BF16 */
    int k = kt - st;
    if (k > 0 && ctx->mode != SV2_CAUSAL_DISABLED) {
        h3_gpu_tensor *fresh = sv2_pool_take(m, (size_t)k * plane, 1);
        ok = fresh != NULL;
        for (int j = 0; j < k && ok; j++) {
            int e = depth - k + j;
            if (e >= front) ok = h3_gpu_copy_f32(m->gpu, window, (size_t)j * plane, x->t, (size_t)(e - front) * plane, plane);
            else if (head) ok = h3_gpu_copy_f32(m->gpu, window, (size_t)j * plane, head, (size_t)e * plane, plane);
            else ok = h3_gpu_copy_f32(m->gpu, window, (size_t)j * plane, x->t, 0, plane);
        }
        if (ok) ok = h3_gpu_cast_f32_to_bf16(m->gpu, fresh, window, (uint32_t)((size_t)k * plane));
        if (!ok) { sv2_pool_give(m, fresh); sv2_pool_give(m, window); sv2_pool_give(m, head); fail(error, error_size, "SeedVR2 VAE: %s", h3_gpu_error(m->gpu)); return 0; }
        causal_keep(m, ctx, c->slot, fresh, k);
    }
    sv2_pool_give(m, window); sv2_pool_give(m, head);
    return 1;
}

static int conv3x3(h3_seedvr2 *m, const act *x, const sv2_conv *c, int stride, act *y, sv2_causal *ctx,
                   char *error, size_t error_size) {
    if (!ctx) return conv3x3_image(m, x, c, stride, y, error, error_size);
    return conv3x3_clip(m, x, c, stride, y, ctx, error, error_size);
}

/* 1x1 convolution = linear over the pixels of every frame. */
static int conv1x1(h3_seedvr2 *m, const act *x, const h3_gpu_tensor *w, const h3_gpu_tensor *b,
                   int out, act *y, char *error, size_t error_size) {
    int f = act_frames(x);
    if (!act_new_f(m, y, f, x->h, x->w, out)) { fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
    GPU_OP(h3_gpu_linear_f32(m->gpu, y->t, x->t, w, b, (uint32_t)((size_t)f * x->h * x->w), (uint32_t)x->c, (uint32_t)out));
    return 1;
}

/* in_place: y reuses x's buffer (the kernel reads every element before it
 * writes it; x must not be needed afterwards). Statistics per frame. */
static int group_norm_silu(h3_seedvr2 *m, const act *x, const sv2_norm *n, act *y, int in_place,
                           char *error, size_t error_size) {
    int f = act_frames(x);
    if (in_place) *y = *x;
    else if (!act_new_f(m, y, f, x->h, x->w, x->c)) { fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
    GPU_OP(h3_gpu_vae_encoder_group_norm_silu_chunked_f32(m->gpu, y->t, x->t, n->w, n->b, (uint32_t)f, 1,
                                                          (uint32_t)x->h, (uint32_t)x->w, (uint32_t)x->c,
                                                          SV2_VAE_GROUPS, SV2_VAE_EPS));
    return 1;
}

static int resnet(h3_seedvr2 *m, const act *x, const sv2_resnet *r, act *y, sv2_causal *ctx,
                  char *error, size_t error_size) {
    act h1, c1, h2, c2, sc;
    if (!group_norm_silu(m, x, &r->norm1, &h1, 0, error, error_size)) return 0;
    int ok = conv3x3(m, &h1, &r->conv1, 1, &c1, ctx, error, error_size);
    sv2_act_free(m, &h1);
    if (!ok) return 0;
    ok = group_norm_silu(m, &c1, &r->norm2, &h2, 1, error, error_size);   /* in place */
    if (!ok) { sv2_act_free(m, &c1); return 0; }
    ok = conv3x3(m, &h2, &r->conv2, 1, &c2, ctx, error, error_size);
    sv2_act_free(m, &h2);
    if (!ok) return 0;
    const act *residual = x;
    if (r->shortcut.w) {
        if (!conv1x1(m, x, r->shortcut.w, r->shortcut.b, r->out, &sc, error, error_size)) { sv2_act_free(m, &c2); return 0; }
        residual = &sc;
    }
    *y = c2;   /* in place: y = c2 + residual */
    ok = h3_gpu_add_scaled_f32(m->gpu, y->t, c2.t, residual->t, 1.0f, 1.0f,
                               (uint32_t)((size_t)act_frames(y) * y->h * y->w * y->c));
    if (r->shortcut.w) sv2_act_free(m, &sc);
    if (!ok) { fail(error, error_size, "SeedVR2 VAE: %s", h3_gpu_error(m->gpu)); return 0; }
    return 1;
}

/* The mid block's self-attention, over the pixels of each frame. */
static int attention(h3_seedvr2 *m, const act *x, const sv2_attention *a, act *y,
                     char *error, size_t error_size) {
    int C = x->c, f = act_frames(x);
    uint32_t n = (uint32_t)(x->h * x->w);
    act g, q, k, v, o, p;
    if (!act_new_f(m, &g, f, x->h, x->w, C)) { fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
    GPU_OP(h3_gpu_sv2_group_norm_f32(m->gpu, g.t, x->t, a->norm.w, a->norm.b, (uint32_t)f, (uint32_t)x->h, (uint32_t)x->w,
                                     (uint32_t)C, SV2_VAE_GROUPS, SV2_VAE_EPS));
    int ok = conv1x1(m, &g, a->wq, a->bq, C, &q, error, error_size) &&
             conv1x1(m, &g, a->wk, a->bk, C, &k, error, error_size) &&
             conv1x1(m, &g, a->wv, a->bv, C, &v, error, error_size);
    sv2_act_free(m, &g);
    if (!ok) return 0;
    if (!act_new_f(m, &o, f, x->h, x->w, C)) { fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
    if (f == 1) {
        ok = h3_gpu_sdpa_f32(m->gpu, o.t, q.t, k.t, v.t, n, 1, (uint32_t)C, 1.0f / sqrtf((float)C));
    } else {
        size_t plane = (size_t)n * C;
        h3_gpu_tensor *qf = sv2_pool_take(m, plane, 0), *kf = sv2_pool_take(m, plane, 0);
        h3_gpu_tensor *vf = sv2_pool_take(m, plane, 0), *of = sv2_pool_take(m, plane, 0);
        ok = qf && kf && vf && of;
        for (int t = 0; t < f && ok; t++) {
            size_t at = (size_t)t * plane;
            ok = h3_gpu_copy_f32(m->gpu, qf, 0, q.t, at, plane) && h3_gpu_copy_f32(m->gpu, kf, 0, k.t, at, plane) &&
                 h3_gpu_copy_f32(m->gpu, vf, 0, v.t, at, plane) &&
                 h3_gpu_sdpa_f32(m->gpu, of, qf, kf, vf, n, 1, (uint32_t)C, 1.0f / sqrtf((float)C)) &&
                 h3_gpu_copy_f32(m->gpu, o.t, at, of, 0, plane);
        }
        sv2_pool_give(m, qf); sv2_pool_give(m, kf); sv2_pool_give(m, vf); sv2_pool_give(m, of);
    }
    sv2_act_free(m, &q); sv2_act_free(m, &k); sv2_act_free(m, &v);
    if (!ok) { fail(error, error_size, "SeedVR2 VAE attention: %s", h3_gpu_error(m->gpu)); sv2_act_free(m, &o); return 0; }
    ok = conv1x1(m, &o, a->wo, a->bo, C, &p, error, error_size);
    sv2_act_free(m, &o);
    if (!ok) return 0;
    *y = p;
    GPU_OP(h3_gpu_add_scaled_f32(m->gpu, y->t, p.t, x->t, 1.0f, 1.0f, (uint32_t)((size_t)f * n * C)));
    return 1;
}

/* ---- The same VAE on the TensorOps grid (M5): BF16 rows [guard | f x
 * (H+2) x (W+2) | tail] with channels padded to 64; every convolution is one
 * h3_gpu_vsr_tapconv_bf16 (3x3: 9 taps over the padded frame, 3x3x3: 27,
 * the temporal ones whole frames back; 1x1: 1 tap), the residual added
 * inside it; GroupNorm + SiLU in three chunked passes per frame; the
 * attention block goes through the F32 path. For a clip the guard holds
 * `front` (2) frame slots before frame 0 for the causal head. ---- */

typedef struct { h3_gpu_tensor *t; int h, w, c, cp, f, front; uint32_t rows, guard, x_rows, frame_rows; } gact;

static int gact_new_f(h3_seedvr2 *m, gact *a, int f, int front, int h, int w, int c) {
    memset(a, 0, sizeof(*a));
    a->h = h; a->w = w; a->c = c; a->cp = round64(c); a->f = f; a->front = front;
    uint32_t wp = (uint32_t)w + 2;
    a->frame_rows = (uint32_t)(h + 2) * wp;
    a->rows = (uint32_t)f * a->frame_rows;
    uint32_t tail = (wp + 3 + 127) / 128 * 128;
    a->guard = ((uint32_t)front * a->frame_rows + wp + 3 + 127) / 128 * 128;
    a->x_rows = a->guard + a->rows + tail + 128;
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

/* A clip's layers allocate GBs at encode time, far ahead of the GPU; a
 * buffer freed on the host stays until the command buffer that reads it
 * completes. Waiting for the GPU after each layer keeps the footprint at
 * the live bytes (the still-image path, ctx NULL, never waits). */
static int clip_sync(h3_seedvr2 *m, const sv2_causal *ctx, char *error, size_t error_size) {
    if (!ctx || !h3_gpu_command_open(m->gpu)) return 1;
    if (!h3_gpu_submit(m->gpu) || !h3_gpu_begin(m->gpu)) { fail(error, error_size, "SeedVR2 VAE: %s", h3_gpu_error(m->gpu)); return 0; }
    return 1;
}

/* Puts the causal head into x's front slots for a conv with temporal stride
 * st: the state's frames in ACTIVE mode, else kt - 1 copies of frame 0.
 * *front: the frames of the extended input before frame 0. */
static int gcausal_head(h3_seedvr2 *m, const gact *x, const sv2_conv *c, sv2_causal *ctx, int *front,
                        char *error, size_t error_size) {
    size_t frame = (size_t)x->frame_rows * x->cp;
    if (ctx->mode == SV2_CAUSAL_ACTIVE && ctx->mem[c->slot]) {
        *front = ctx->mem_frames[c->slot];
        if (*front > x->front) { fail(error, error_size, "SeedVR2 VAE (grid): %d state frames, %d slots", *front, x->front); return 0; }
        GRID_OP(h3_gpu_copy_bf16(m->gpu, x->t, (size_t)x->guard * x->cp - (size_t)*front * frame,
                                 ctx->mem[c->slot], 0, (size_t)*front * frame));
        return 1;
    }
    *front = c->kt - 1;
    if (*front > x->front) { fail(error, error_size, "SeedVR2 VAE (grid): a causal conv on a grid without head slots"); return 0; }
    for (int i = 1; i <= *front; i++)
        GRID_OP(h3_gpu_copy_bf16(m->gpu, x->t, (size_t)x->guard * x->cp - (size_t)i * frame, x->t,
                                 (size_t)x->guard * x->cp, frame));
    return 1;
}

/* Keeps the last kt - st frames of [head | x] (contiguous rows) as the state. */
static int gcausal_keep(h3_seedvr2 *m, const gact *x, const sv2_conv *c, int st, sv2_causal *ctx,
                        char *error, size_t error_size) {
    int k = c->kt - st;
    if (k <= 0 || ctx->mode == SV2_CAUSAL_DISABLED) return 1;
    size_t frame = (size_t)x->frame_rows * x->cp;
    h3_gpu_tensor *fresh = sv2_pool_take(m, (size_t)k * frame, 1);
    if (!fresh) { fail(error, error_size, "SeedVR2 VAE (grid): out of memory"); return 0; }
    if (!h3_gpu_copy_bf16(m->gpu, fresh, 0, x->t, (size_t)x->guard * x->cp + (size_t)(x->f - k) * frame - 0,
                          (size_t)k * frame)) {
        sv2_pool_give(m, fresh);
        fail(error, error_size, "SeedVR2 VAE (grid): %s", h3_gpu_error(m->gpu));
        return 0;
    }
    causal_keep(m, ctx, c->slot, fresh, k);
    return 1;
}

/* out = conv(x) + bias (+ residual), same geometry, stride 1 (a causal conv
 * with temporal stride st also keeps the state that stride needs). */
static int gconv_t(h3_seedvr2 *m, const gact *x, const sv2_conv *c, const gact *residual, gact *y,
                   int st, sv2_causal *ctx, int *front_used, char *error, size_t error_size) {
    if (x->cp != c->in_p) { fail(error, error_size, "SeedVR2 VAE (grid): %d padded channels, the conv wants %d", x->cp, c->in_p); return 0; }
    int32_t offsets[27];
    uint32_t taps;
    int front = 0;
    if (c->kt > 1) {
        if (!ctx) { fail(error, error_size, "SeedVR2 VAE (grid): temporal weights on a still image"); return 0; }
        if (!gcausal_head(m, x, c, ctx, &front, error, error_size)) return 0;
        int k = c->kernel;
        for (int tt = 0; tt < c->kt; tt++)
            for (int dy = 0; dy < k; dy++)
                for (int dx = 0; dx < k; dx++)
                    offsets[(tt * k + dy) * k + dx] = (tt - (c->kt - 1)) * (int32_t)x->frame_rows +
                                                      (dy - k / 2) * (x->w + 2) + (dx - k / 2);
        taps = (uint32_t)(c->kt * k * k);
    } else {
        gact_offsets(x, c->kernel, offsets);
        taps = (uint32_t)(c->kernel * c->kernel);
    }
    if (front_used) *front_used = front;
    if (!gact_new_f(m, y, x->f, x->front, x->h, x->w, c->out)) { fail(error, error_size, "SeedVR2 VAE (grid): out of memory"); return 0; }
    GRID_OP(h3_gpu_vsr_tapconv_bf16(m->gpu, y->t, x->t, NULL, c->nw, NULL, c->nb, residual ? residual->t : NULL,
                                    x->rows, x->x_rows, x->guard, (uint32_t)x->cp, (uint32_t)y->cp,
                                    taps, offsets, 0));
    GRID_OP(h3_gpu_vsr_zero_border_bf16(m->gpu, y->t, (uint32_t)y->f, (uint32_t)y->h, (uint32_t)y->w, (uint32_t)y->cp,
                                        y->guard, y->x_rows));
    if (c->kt > 1 && !gcausal_keep(m, x, c, st, ctx, error, error_size)) return 0;
    return 1;
}

static int gconv(h3_seedvr2 *m, const gact *x, const sv2_conv *c, const gact *residual, gact *y,
                 sv2_causal *ctx, char *error, size_t error_size) {
    return gconv_t(m, x, c, residual, y, 1, ctx, NULL, error, error_size);
}

/* The stride-2 3x3 convolution with padding (0, 1): the stride-1 result,
 * then its odd interior positions; with temporal taps also every second
 * frame, those whose window starts on the stride in [head | x]. */
static int gconv_s2(h3_seedvr2 *m, const gact *x, const sv2_conv *c, gact *y, sv2_causal *ctx,
                    char *error, size_t error_size) {
    gact full;
    int st = c->kt > 1 ? 2 : 1, front = 0;
    if (!gconv_t(m, x, c, NULL, &full, st, ctx, &front, error, error_size)) return 0;
    int frames = st == 2 ? (front + x->f - 3) / 2 + 1 : x->f;
    uint32_t phase_t = st == 2 ? (uint32_t)(2 - front) : 0;
    if (!gact_new_f(m, y, frames, x->front, x->h / 2, x->w / 2, c->out)) {
        gact_free(m, &full);
        fail(error, error_size, "SeedVR2 VAE (grid): out of memory");
        return 0;
    }
    int ok = h3_gpu_sv2_grid_select3d_bf16(m->gpu, y->t, (uint32_t)frames, (uint32_t)y->h, (uint32_t)y->w, y->guard,
                                           full.t, (uint32_t)full.f, (uint32_t)full.h, (uint32_t)full.w, full.guard,
                                           (uint32_t)y->cp, (uint32_t)st, phase_t, 1, 1);
    gact_free(m, &full);
    if (!ok) { fail(error, error_size, "SeedVR2 VAE (grid): %s", h3_gpu_error(m->gpu)); return 0; }
    return 1;
}

static int ggn(h3_seedvr2 *m, const gact *x, const sv2_norm *n, gact *y, int in_place, int silu,
               char *error, size_t error_size) {
    if (in_place) *y = *x;
    else if (!gact_new_f(m, y, x->f, x->front, x->h, x->w, x->c)) { fail(error, error_size, "SeedVR2 VAE (grid): out of memory"); return 0; }
    for (int t = 0; t < x->f; t++)
        GRID_OP(h3_gpu_sv2_grid_group_norm_bf16(m->gpu, y->t, x->t, n->w, n->b, (uint32_t)x->h, (uint32_t)x->w,
                                                (uint32_t)x->c, (uint32_t)x->cp, x->guard + (uint32_t)t * x->frame_rows,
                                                x->x_rows, SV2_VAE_GROUPS, SV2_VAE_EPS, silu));
    /* the norm writes the interior only: a pooled buffer's border rows must
     * be zero before the next convolution's taps read them */
    if (!in_place)
        GRID_OP(h3_gpu_vsr_zero_border_bf16(m->gpu, y->t, (uint32_t)y->f, (uint32_t)y->h, (uint32_t)y->w, (uint32_t)y->cp,
                                            y->guard, y->x_rows));
    return 1;
}

static int gresnet(h3_seedvr2 *m, const gact *x, const sv2_resnet *r, gact *y, sv2_causal *ctx,
                   char *error, size_t error_size) {
    gact h1, c1, c2, sc;
    if (!ggn(m, x, &r->norm1, &h1, 0, 1, error, error_size)) return 0;
    int ok = gconv(m, &h1, &r->conv1, NULL, &c1, ctx, error, error_size);
    gact_free(m, &h1);
    if (!ok) return 0;
    gact h2;
    if (!ggn(m, &c1, &r->norm2, &h2, 1, 1, error, error_size)) { gact_free(m, &c1); return 0; }
    const gact *residual = x;
    if (r->shortcut.nw) {
        if (!gconv(m, x, &r->shortcut, NULL, &sc, ctx, error, error_size)) { gact_free(m, &c1); return 0; }
        residual = &sc;
    }
    ok = gconv(m, &h2, &r->conv2, residual, &c2, ctx, error, error_size);
    gact_free(m, &c1);
    if (r->shortcut.nw) gact_free(m, &sc);
    if (!ok) return 0;
    *y = c2;
    return 1;
}

/* F32 [f, h, w, c] <-> grid (front: the head slots of a clip's grid). */
static int gact_pack(h3_seedvr2 *m, const act *x, gact *g, int front, char *error, size_t error_size) {
    int f = act_frames(x);
    if (!gact_new_f(m, g, f, front, x->h, x->w, x->c)) { fail(error, error_size, "SeedVR2 VAE (grid): out of memory"); return 0; }
    GRID_OP(h3_gpu_vsr_pack_grid_bf16(m->gpu, g->t, x->t, (uint32_t)f, (uint32_t)x->h, (uint32_t)x->w, (uint32_t)x->c,
                                      (uint32_t)g->cp, g->guard, g->x_rows, 0));
    return 1;
}

static int gact_unpack(h3_seedvr2 *m, const gact *g, act *x, char *error, size_t error_size) {
    if (!act_new_f(m, x, g->f, g->h, g->w, g->c)) { fail(error, error_size, "SeedVR2 VAE (grid): out of memory"); return 0; }
    GRID_OP(h3_gpu_vsr_unpack_grid_f32(m->gpu, x->t, g->t, (uint32_t)g->f, (uint32_t)g->h, (uint32_t)g->w, (uint32_t)g->c,
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
    ok = gact_pack(m, &o, y, x->front, error, error_size);
    sv2_act_free(m, &o);
    return ok;
}

static int vae_encode_grid(h3_seedvr2 *m, const act *rgb, act *out, int stop, sv2_causal *ctx,
                           char *error, size_t error_size) {
    sv2_encoder *e = &m->enc;
    gact in, x, y;
    double mark = now_seconds();
    if (!gact_pack(m, rgb, &in, ctx ? 2 : 0, error, error_size)) return 0;
    int ok = gconv(m, &in, &e->conv_in, NULL, &x, ctx, error, error_size);
    gact_free(m, &in);
    if (!ok) return 0;
    if (stop == 0) { ok = gact_unpack(m, &x, out, error, error_size); gact_free(m, &x); return ok; }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            if (!gresnet(m, &x, &e->down[i][j], &y, ctx, error, error_size)) { gact_free(m, &x); return 0; }
            gact_free(m, &x); x = y;
            if (!clip_sync(m, ctx, error, error_size)) { gact_free(m, &x); return 0; }
            if (ctx) sv2_pool_drain(m);
        }
        if (i < 3) {
            if (!gconv_s2(m, &x, &e->downsample[i], &y, ctx, error, error_size)) { gact_free(m, &x); return 0; }
            gact_free(m, &x); x = y;
        }
        if (stop == i + 1) { ok = gact_unpack(m, &x, out, error, error_size); gact_free(m, &x); return ok; }
        if (ctx) { if (!clip_sync(m, ctx, error, error_size)) { gact_free(m, &x); return 0; } sv2_pool_drain(m); }
    }
    trace_sync(m->gpu, &mark, "enc down (grid)");
    if (!gresnet(m, &x, &e->mid_a, &y, ctx, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x); x = y;
    if (!gattention(m, &x, &e->mid_attn, &y, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x); x = y;
    if (!gresnet(m, &x, &e->mid_b, &y, ctx, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x); x = y;
    if (stop == 5) { ok = gact_unpack(m, &x, out, error, error_size); gact_free(m, &x); return ok; }
    if (!ggn(m, &x, &e->norm_out, &y, 1, 1, error, error_size)) { gact_free(m, &x); return 0; }
    x = y;
    if (!gconv(m, &x, &e->conv_out, NULL, &y, ctx, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x);
    ok = gact_unpack(m, &y, out, error, error_size);
    gact_free(m, &y);
    trace_sync(m->gpu, &mark, "enc mid/out (grid)");
    return ok;
}

static int vae_decode_grid(h3_seedvr2 *m, const act *z, act *out, int stop, sv2_causal *ctx,
                           char *error, size_t error_size) {
    sv2_decoder *d = &m->dec;
    gact in, x, y;
    double mark = now_seconds();
    if (!gact_pack(m, z, &in, ctx ? 2 : 0, error, error_size)) return 0;
    int ok = gconv(m, &in, &d->conv_in, NULL, &x, ctx, error, error_size);
    gact_free(m, &in);
    if (!ok) return 0;
    if (stop == 0) { ok = gact_unpack(m, &x, out, error, error_size); gact_free(m, &x); return ok; }
    if (!gresnet(m, &x, &d->mid_a, &y, ctx, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x); x = y;
    if (!gattention(m, &x, &d->mid_attn, &y, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x); x = y;
    if (!gresnet(m, &x, &d->mid_b, &y, ctx, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x); x = y;
    trace_sync(m->gpu, &mark, "dec mid (grid)");
    if (stop == 1) { ok = gact_unpack(m, &x, out, error, error_size); gact_free(m, &x); return ok; }
    int drop_head = !ctx || ctx->mode != SV2_CAUSAL_ACTIVE;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            if (!gresnet(m, &x, &d->up[i][j], &y, ctx, error, error_size)) { gact_free(m, &x); return 0; }
            gact_free(m, &x); x = y;
            if (!clip_sync(m, ctx, error, error_size)) { gact_free(m, &x); return 0; }
            if (ctx) sv2_pool_drain(m);
        }
        if (i < 3) {
            int C = x.c, tf = d->up_temporal[i];
            int frames = (int)h3_gpu_sv2_shuffle_frames((uint32_t)x.f, (uint32_t)tf, drop_head);
            gact u, s;
            if (!gconv(m, &x, &d->up_nax[i], NULL, &u, ctx, error, error_size)) { gact_free(m, &x); return 0; }
            gact_free(m, &x);
            if (!gact_new_f(m, &s, ctx ? frames : 1, u.front, 2 * u.h, 2 * u.w, C)) { gact_free(m, &u); fail(error, error_size, "SeedVR2 VAE (grid): out of memory"); return 0; }
            ok = h3_gpu_sv2_grid_shuffle3d_bf16(m->gpu, s.t, (uint32_t)s.cp, s.guard, u.t, (uint32_t)u.f, (uint32_t)u.h,
                                                (uint32_t)u.w, (uint32_t)u.cp, u.guard, (uint32_t)C,
                                                (uint32_t)tf, drop_head);
            gact_free(m, &u);
            if (!ok) { gact_free(m, &s); fail(error, error_size, "SeedVR2 VAE (grid): %s", h3_gpu_error(m->gpu)); return 0; }
            if (!gconv(m, &s, &d->up_conv[i], NULL, &x, ctx, error, error_size)) { gact_free(m, &s); return 0; }
            gact_free(m, &s);
            if (!h3_gpu_continue(m->gpu)) { gact_free(m, &x); fail(error, error_size, "SeedVR2 VAE (grid): %s", h3_gpu_error(m->gpu)); return 0; }
        }
        char label[32];
        snprintf(label, sizeof(label), "dec up%d (grid)", i);
        trace_sync(m->gpu, &mark, label);
        if (stop == i + 2) { ok = gact_unpack(m, &x, out, error, error_size); gact_free(m, &x); return ok; }
        if (ctx) { if (!clip_sync(m, ctx, error, error_size)) { gact_free(m, &x); return 0; } sv2_pool_drain(m); }   /* a clip's buffers are large: keep none of another block's size */
    }
    if (!ggn(m, &x, &d->norm_out, &y, 1, 1, error, error_size)) { gact_free(m, &x); return 0; }
    x = y;
    if (!gconv(m, &x, &d->conv_out, NULL, &y, ctx, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x);
    ok = gact_unpack(m, &y, out, error, error_size);
    gact_free(m, &y);
    trace_sync(m->gpu, &mark, "dec out (grid)");
    return ok;
}

/* stop: 0 after conv_in, 1..4 after down block i-1, 5 after the mid block,
 * 6 (or more) the latent. */
int sv2_vae_encode_clip(h3_seedvr2 *m, const act *rgb, act *out, int stop, sv2_causal *ctx,
                        char *error, size_t error_size) {
    if (m->nax_vae) return vae_encode_grid(m, rgb, out, stop, ctx, error, error_size);
    sv2_encoder *e = &m->enc;
    act x, y;
    double mark = now_seconds();
    if (!conv3x3(m, rgb, &e->conv_in, 1, &x, ctx, error, error_size)) return 0;
    if (stop == 0) { *out = x; return 1; }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            if (!resnet(m, &x, &e->down[i][j], &y, ctx, error, error_size)) { sv2_act_free(m, &x); return 0; }
            sv2_act_free(m, &x); x = y;
            if (!clip_sync(m, ctx, error, error_size)) { sv2_act_free(m, &x); return 0; }
            if (ctx) sv2_pool_drain(m);
        }
        if (i < 3) {
            if (!conv3x3(m, &x, &e->downsample[i], 2, &y, ctx, error, error_size)) { sv2_act_free(m, &x); return 0; }
            sv2_act_free(m, &x); x = y;
        }
        if (stop == i + 1) { *out = x; return 1; }
        if (ctx) { if (!clip_sync(m, ctx, error, error_size)) { sv2_act_free(m, &x); return 0; } sv2_pool_drain(m); }
    }
    trace_sync(m->gpu, &mark, "enc down");
    if (!resnet(m, &x, &e->mid_a, &y, ctx, error, error_size)) { sv2_act_free(m, &x); return 0; }
    sv2_act_free(m, &x); x = y;
    if (!attention(m, &x, &e->mid_attn, &y, error, error_size)) { sv2_act_free(m, &x); return 0; }
    sv2_act_free(m, &x); x = y;
    if (!resnet(m, &x, &e->mid_b, &y, ctx, error, error_size)) { sv2_act_free(m, &x); return 0; }
    sv2_act_free(m, &x); x = y;
    if (stop == 5) { *out = x; return 1; }
    if (!group_norm_silu(m, &x, &e->norm_out, &y, 1, error, error_size)) { sv2_act_free(m, &x); return 0; }
    x = y;   /* in place: the same buffer */
    if (!conv3x3(m, &x, &e->conv_out, 1, out, ctx, error, error_size)) { sv2_act_free(m, &x); return 0; }
    sv2_act_free(m, &x);
    trace_sync(m->gpu, &mark, "enc mid/out");
    return 1;
}

int sv2_vae_encode_stage(h3_seedvr2 *m, const act *rgb, act *out, int stop, char *error, size_t error_size) {
    return sv2_vae_encode_clip(m, rgb, out, stop, NULL, error, error_size);
}

int sv2_vae_encode(h3_seedvr2 *m, const act *rgb, act *out, char *error, size_t error_size) {
    return sv2_vae_encode_clip(m, rgb, out, 99, NULL, error, error_size);
}

/* Decoder: latent [f, h, w, 16] (already divided by the scaling factor) ->
 * RGB [f', 8h, 8w, 3] in about [-1, 1]. */
int sv2_vae_decode_clip(h3_seedvr2 *m, const act *z, act *out, int stop, sv2_causal *ctx,
                        char *error, size_t error_size) {
    if (m->nax_vae) return vae_decode_grid(m, z, out, stop, ctx, error, error_size);
    sv2_decoder *d = &m->dec;
    act x, y;
    double mark = now_seconds();
    if (!conv3x3(m, z, &d->conv_in, 1, &x, ctx, error, error_size)) return 0;
    if (stop == 0) { *out = x; return 1; }
    if (!resnet(m, &x, &d->mid_a, &y, ctx, error, error_size)) { sv2_act_free(m, &x); return 0; }
    sv2_act_free(m, &x); x = y;
    if (!attention(m, &x, &d->mid_attn, &y, error, error_size)) { sv2_act_free(m, &x); return 0; }
    sv2_act_free(m, &x); x = y;
    if (!resnet(m, &x, &d->mid_b, &y, ctx, error, error_size)) { sv2_act_free(m, &x); return 0; }
    sv2_act_free(m, &x); x = y;
    trace_sync(m->gpu, &mark, "dec mid");
    if (stop == 1) { *out = x; return 1; }
    int drop_head = !ctx || ctx->mode != SV2_CAUSAL_ACTIVE;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            if (!resnet(m, &x, &d->up[i][j], &y, ctx, error, error_size)) { sv2_act_free(m, &x); return 0; }
            sv2_act_free(m, &x); x = y;
            if (!clip_sync(m, ctx, error, error_size)) { sv2_act_free(m, &x); return 0; }
            if (ctx) sv2_pool_drain(m);
        }
        if (i < 3) {
            int C = x.c, tf = d->up_temporal[i], f = act_frames(&x);
            int frames = ctx ? (int)h3_gpu_sv2_shuffle_frames((uint32_t)f, (uint32_t)tf, drop_head) : 1;
            act u, s;
            if (!conv1x1(m, &x, d->up_w[i], d->up_b[i], C * 4 * tf, &u, error, error_size)) { sv2_act_free(m, &x); return 0; }
            sv2_act_free(m, &x);
            if (!act_new_f(m, &s, frames, 2 * u.h, 2 * u.w, C)) { sv2_act_free(m, &u); fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
            int ok = h3_gpu_sv2_shuffle3d_f32(m->gpu, s.t, u.t, (uint32_t)f, (uint32_t)u.h, (uint32_t)u.w, (uint32_t)C,
                                              (uint32_t)tf, drop_head);
            sv2_act_free(m, &u);
            if (!ok) { sv2_act_free(m, &s); fail(error, error_size, "SeedVR2 VAE: %s", h3_gpu_error(m->gpu)); return 0; }
            if (!conv3x3(m, &s, &d->up_conv[i], 1, &x, ctx, error, error_size)) { sv2_act_free(m, &s); return 0; }
            sv2_act_free(m, &s);
            if (!h3_gpu_continue(m->gpu)) { sv2_act_free(m, &x); fail(error, error_size, "SeedVR2 VAE: %s", h3_gpu_error(m->gpu)); return 0; }
        }
        char label[32];
        snprintf(label, sizeof(label), "dec up%d", i);
        trace_sync(m->gpu, &mark, label);
        if (stop == i + 2) { *out = x; return 1; }
        if (ctx) { if (!clip_sync(m, ctx, error, error_size)) { sv2_act_free(m, &x); return 0; } sv2_pool_drain(m); }
    }
    if (!group_norm_silu(m, &x, &d->norm_out, &y, 1, error, error_size)) { sv2_act_free(m, &x); return 0; }
    x = y;   /* in place: the same buffer */
    if (!conv3x3(m, &x, &d->conv_out, 1, out, ctx, error, error_size)) { sv2_act_free(m, &x); return 0; }
    sv2_act_free(m, &x);
    trace_sync(m->gpu, &mark, "dec out");
    return 1;
}

int sv2_vae_decode(h3_seedvr2 *m, const act *z, act *out, char *error, size_t error_size) {
    return sv2_vae_decode_clip(m, z, out, 99, NULL, error, error_size);
}

/* ---- Video: the weights' form, and whole clips in causal slices ---- */

int sv2_vae_set_temporal(h3_seedvr2 *m, int temporal, char *error, size_t error_size) {
    temporal = temporal != 0;
    if (m->vae_temporal == temporal) return 1;
    char path[1200];
    snprintf(path, sizeof(path), "%s/vae.safetensors", m->model_dir);
    h3_st_header h;
    if (!h3_st_read_header(path, &h, error, error_size)) return 0;
    sv2_free_vae(m);
    m->vae_temporal = temporal;
    int ok = sv2_load_vae(m, &h, error, error_size);
    h3_st_free_header(&h);
    return ok;
}

/* frames [first, first + count) of x as a new activation */
static int act_slice(h3_seedvr2 *m, const act *x, int first, int count, act *y, char *error, size_t error_size) {
    size_t plane = (size_t)x->h * x->w * x->c;
    if (!act_new_f(m, y, count, x->h, x->w, x->c)) { fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
    GPU_OP(h3_gpu_copy_f32(m->gpu, y->t, 0, x->t, (size_t)first * plane, (size_t)count * plane));
    return 1;
}

typedef int (*sv2_clip_fn)(h3_seedvr2 *, const act *, act *, int, sv2_causal *, char *, size_t);

/* The reference's slicing_encode / slicing_decode: [0, 1 + step) with the
 * state initialised, then `step` frames at a time with the state carried. */
static int run_sliced(h3_seedvr2 *m, sv2_clip_fn fn, const act *x, act *out, int step, const char *phase,
                      h3_progress_callback on_progress, void *opaque, const volatile int *cancel,
                      char *error, size_t error_size) {
    if (!m->vae_temporal) { fail(error, error_size, "SeedVR2 VAE: the weights are not in temporal form"); return 0; }
    int f = act_frames(x);
    sv2_causal *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) { fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
    int ok = 1;
    if (step <= 0 || f <= step + 1) {
        ctx->mode = SV2_CAUSAL_DISABLED;
        ok = fn(m, x, out, 99, ctx, error, error_size);
        sv2_causal_reset(ctx);
        free(ctx);
        return ok;
    }
    int slices = 1 + (f - 1 - step + step - 1) / step;
    act *parts = calloc((size_t)slices, sizeof(act));
    if (!parts) { free(ctx); fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
    int done = 0, total = 0, n = 0;
    for (int s = 0; s < slices && ok; s++) {
        int first = s == 0 ? 0 : 1 + s * step, count = s == 0 ? 1 + step : (f - first < step ? f - first : step);
        if ((cancel && *cancel) || (on_progress && on_progress(phase, s, slices, opaque))) {
            fail(error, error_size, "vsr cancelled");
            ok = 0;
            break;
        }
        act piece;
        ctx->mode = s == 0 ? SV2_CAUSAL_INIT : SV2_CAUSAL_ACTIVE;
        ok = act_slice(m, x, first, count, &piece, error, error_size) &&
             fn(m, &piece, &parts[n], 99, ctx, error, error_size);
        if (piece.t) sv2_act_free(m, &piece);
        if (!ok) break;
        total += act_frames(&parts[n]);
        n++;
        done = first + count;
        if (!h3_gpu_continue(m->gpu)) { fail(error, error_size, "SeedVR2 VAE: %s", h3_gpu_error(m->gpu)); ok = 0; }
    }
    if (ok && done != f) { fail(error, error_size, "SeedVR2 VAE: slicing covered %d of %d frames", done, f); ok = 0; }
    if (ok && on_progress && on_progress(phase, slices, slices, opaque)) { fail(error, error_size, "vsr cancelled"); ok = 0; }
    if (ok) {
        size_t plane = (size_t)parts[0].h * parts[0].w * parts[0].c, at = 0;
        ok = act_new_f(m, out, total, parts[0].h, parts[0].w, parts[0].c);
        if (!ok) fail(error, error_size, "SeedVR2 VAE: out of memory");
        for (int i = 0; i < n && ok; i++) {
            size_t count = (size_t)act_frames(&parts[i]) * plane;
            ok = h3_gpu_copy_f32(m->gpu, out->t, at, parts[i].t, 0, count);
            at += count;
        }
        if (!ok && out->t) sv2_act_free(m, out);
    }
    for (int i = 0; i < n; i++) sv2_act_free(m, &parts[i]);
    free(parts);
    sv2_causal_reset(ctx);
    free(ctx);
    return ok;
}

int sv2_vae_encode_video(h3_seedvr2 *m, const act *rgb, act *out, int split,
                         h3_progress_callback on_progress, void *opaque, const volatile int *cancel,
                         char *error, size_t error_size) {
    return run_sliced(m, sv2_vae_encode_clip, rgb, out, split, "vsr encode", on_progress, opaque, cancel, error, error_size);
}

int sv2_vae_decode_video(h3_seedvr2 *m, const act *z, act *out, int split,
                         h3_progress_callback on_progress, void *opaque, const volatile int *cancel,
                         char *error, size_t error_size) {
    return run_sliced(m, sv2_vae_decode_clip, z, out, split > 0 ? (split / 4 > 1 ? split / 4 : 1) : 0, "vsr decode",
                      on_progress, opaque, cancel, error, error_size);
}
