/* SwiftVR's Restoration-aware Autoencoder (ReAE) on the engine's F32
 * channels-last conv3d path. Architecture: official swiftvr/models/reae.py
 * (= LightX2V reae.py); see UPSCALE_PLAN §4.2.
 *
 * Every layer is expressed with the existing Conv3d (no padding, so the
 * zero/state padding is a separate pass):
 *   conv3x3            pad(1,1) -> conv3d (1,3,3)
 *   conv3x3 stride 2   pad(1,1) -> conv3d (1,3,3) stride (1,2,2)
 *   MemBlock conv on cat[x, past]
 *                      pad(front 1 frame from the state, 1,1) -> conv3d (2,3,3):
 *                      depth tap 0 is the past frame (weight rows C..2C-1),
 *                      tap 1 the frame itself (rows 0..C-1)
 *   TPool(s)           conv3d (s,1,1) stride (s,1,1), weight [O, I, s, 1, 1]
 *   TGrow(1)           1x1 conv
 *   TGrow(2)           nearest x2 in time then Conv3d (3,1,1) pad (1,0,0) - the
 *                      two copies of a frame only see each other, so this is
 *                      one 1x1 conv to 2C channels (rows W1+W2 for the even
 *                      frame, W0+W1 for the odd one) followed by a frame split
 *   Upsample           nearest 2x (kernel)
 * The runner walks a layer table over frame batches, carrying every
 * MemBlock's last input frame in the state, so memory is bounded by the
 * batch and chunks join seamlessly. */
#include "h3_vsr_internal.h"
#include "h3_weights.h"

#include <float.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    L_CONV = 1,      /* 3x3, pad 1 */
    L_CONV_S2,       /* 3x3, stride 2, pad 1 */
    L_MEMBLOCK,
    L_TPOOL,         /* stride in `stride` */
    L_TGROW1,
    L_TGROW2,
    L_UPSAMPLE,
    L_RELU,
    L_CLAMP_TANH,
    L_S2D,           /* pixel_unshuffle(2) */
    L_D2S_CLAMP01    /* clamp [0,1] then pixel_shuffle(2) */
} layer_kind;

typedef struct {
    layer_kind kind;
    int in_ch, out_ch, stride;
    /* conv weights in conv3d layout; MemBlock: w[0..2], b[0..2] */
    h3_gpu_tensor *w[3], *b[3];
    int state_index;   /* MemBlock: slot in the state */
} layer;

#define MAX_LAYERS 32

struct h3_vsr_reae {
    h3_gpu *gpu;
    layer encoder[MAX_LAYERS];
    int encoder_count, encoder_states;
    layer decoder[MAX_LAYERS];
    int decoder_count, decoder_states;
    char error[512];
};

struct h3_vsr_reae_state {
    int encoder;
    int count;
    h3_gpu_tensor **frames;   /* one [H_l, W_l, C_l] frame per MemBlock */
    int *valid;               /* 0 until a chunk has passed */
    int height, width;        /* input frame size the shapes were derived for */
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

/* ---------------------------------------------------------------- */
/* Weights */

static float *host_f32(h3_weight_store *store, h3_gpu *gpu, const char *name,
                       int ndim, const uint64_t *shape, char *error, size_t error_size) {
    h3_gpu_tensor *t = h3_weight_load_f32(store, gpu, name, ndim, shape, error, error_size);
    if (!t) return NULL;
    size_t n = h3_gpu_tensor_elements(t);
    float *host = malloc(n * sizeof(float));
    if (!host || !h3_gpu_tensor_read_f32(t, host, n)) {
        fail(error, error_size, "cannot read %s", name);
        free(host);
        host = NULL;
    }
    h3_gpu_tensor_free(t);
    return host;
}

static h3_gpu_tensor *upload(h3_gpu *gpu, const float *values, size_t n) {
    return h3_gpu_tensor_from_f32(gpu, values, n);
}

/* conv 3x3 [O, I, 3, 3] (+ bias): the same bytes as conv3d [O, I, 1, 3, 3]. */
static int load_conv3x3(layer *l, h3_weight_store *store, h3_gpu *gpu, const char *prefix,
                        int in_ch, int out_ch, int bias, char *error, size_t error_size) {
    char name[160];
    uint64_t shape[4] = {(uint64_t)out_ch, (uint64_t)in_ch, 3, 3};
    snprintf(name, sizeof(name), "%s.weight", prefix);
    float *w = host_f32(store, gpu, name, 4, shape, error, error_size);
    if (!w) return 0;
    l->w[0] = upload(gpu, w, (size_t)out_ch * in_ch * 9);
    free(w);
    if (!l->w[0]) return 0;
    if (bias) {
        uint64_t bshape[1] = {(uint64_t)out_ch};
        snprintf(name, sizeof(name), "%s.bias", prefix);
        float *bv = host_f32(store, gpu, name, 1, bshape, error, error_size);
        if (!bv) return 0;
        l->b[0] = upload(gpu, bv, (size_t)out_ch);
        free(bv);
        if (!l->b[0]) return 0;
    }
    l->in_ch = in_ch;
    l->out_ch = out_ch;
    return 1;
}

/* MemBlock: conv.0 [C, 2C, 3, 3] -> conv3d [C, C, 2, 3, 3] with the past
 * half at depth tap 0; conv.2 and conv.4 are plain 3x3 (all with bias). */
static int load_memblock(layer *l, h3_weight_store *store, h3_gpu *gpu, const char *prefix,
                         int channels, char *error, size_t error_size) {
    char name[160];
    uint64_t shape[4] = {(uint64_t)channels, (uint64_t)2 * channels, 3, 3};
    snprintf(name, sizeof(name), "%s.conv.0.weight", prefix);
    float *w = host_f32(store, gpu, name, 4, shape, error, error_size);
    if (!w) return 0;
    size_t per_in = 9;
    float *w3 = malloc((size_t)channels * channels * 2 * 9 * sizeof(float));
    if (!w3) { free(w); return 0; }
    for (int o = 0; o < channels; o++)
        for (int i = 0; i < channels; i++) {
            /* w[o][i] (x half) -> tap 1; w[o][C + i] (past) -> tap 0 */
            memcpy(w3 + (((size_t)o * channels + i) * 2 + 1) * per_in,
                   w + ((size_t)o * 2 * channels + i) * per_in, per_in * sizeof(float));
            memcpy(w3 + (((size_t)o * channels + i) * 2 + 0) * per_in,
                   w + ((size_t)o * 2 * channels + channels + i) * per_in, per_in * sizeof(float));
        }
    free(w);
    l->w[0] = upload(gpu, w3, (size_t)channels * channels * 18);
    free(w3);
    if (!l->w[0]) return 0;
    uint64_t bshape[1] = {(uint64_t)channels};
    snprintf(name, sizeof(name), "%s.conv.0.bias", prefix);
    float *b0 = host_f32(store, gpu, name, 1, bshape, error, error_size);
    if (!b0) return 0;
    l->b[0] = upload(gpu, b0, (size_t)channels);
    free(b0);
    for (int k = 1; k < 3; k++) {
        char sub[160];
        snprintf(sub, sizeof(sub), "%s.conv.%d", prefix, k * 2);
        layer tmp = {0};
        if (!load_conv3x3(&tmp, store, gpu, sub, channels, channels, 1, error, error_size))
            return 0;
        l->w[k] = tmp.w[0];
        l->b[k] = tmp.b[0];
    }
    l->in_ch = l->out_ch = channels;
    return l->b[0] != NULL;
}

/* TPool: [C, s*C, 1, 1] -> conv3d [C, C, s, 1, 1]. */
static int load_tpool(layer *l, h3_weight_store *store, h3_gpu *gpu, const char *prefix,
                      int channels, int stride, char *error, size_t error_size) {
    char name[160];
    uint64_t shape[4] = {(uint64_t)channels, (uint64_t)stride * channels, 1, 1};
    snprintf(name, sizeof(name), "%s.conv.weight", prefix);
    float *w = host_f32(store, gpu, name, 4, shape, error, error_size);
    if (!w) return 0;
    float *w3 = malloc((size_t)channels * channels * stride * sizeof(float));
    if (!w3) { free(w); return 0; }
    for (int o = 0; o < channels; o++)
        for (int i = 0; i < channels; i++)
            for (int f = 0; f < stride; f++)
                w3[((size_t)o * channels + i) * stride + f] =
                    w[(size_t)o * stride * channels + (size_t)f * channels + i];
    free(w);
    l->w[0] = upload(gpu, w3, (size_t)channels * channels * stride);
    free(w3);
    l->in_ch = l->out_ch = channels;
    l->stride = stride;
    return l->w[0] != NULL;
}

/* TGrow(1): proj [C, C, 1, 1]; TGrow(2): conv3d [C, C, 3, 1, 1] -> 1x1 to 2C. */
static int load_tgrow(layer *l, h3_weight_store *store, h3_gpu *gpu, const char *prefix,
                      int channels, int stride, char *error, size_t error_size) {
    char name[160];
    if (stride == 1) {
        uint64_t shape[4] = {(uint64_t)channels, (uint64_t)channels, 1, 1};
        snprintf(name, sizeof(name), "%s.proj.weight", prefix);
        float *w = host_f32(store, gpu, name, 4, shape, error, error_size);
        if (!w) return 0;
        l->w[0] = upload(gpu, w, (size_t)channels * channels);
        free(w);
        l->in_ch = l->out_ch = channels;
        return l->w[0] != NULL;
    }
    uint64_t shape[5] = {(uint64_t)channels, (uint64_t)channels, 3, 1, 1};
    snprintf(name, sizeof(name), "%s.conv3d.weight", prefix);
    float *w = host_f32(store, gpu, name, 5, shape, error, error_size);
    if (!w) return 0;
    float *w2 = malloc((size_t)2 * channels * channels * sizeof(float));
    if (!w2) { free(w); return 0; }
    for (int o = 0; o < channels; o++)
        for (int i = 0; i < channels; i++) {
            const float *taps = w + ((size_t)o * channels + i) * 3;
            w2[(size_t)o * channels + i] = taps[1] + taps[2];                 /* even frame */
            w2[((size_t)channels + o) * channels + i] = taps[0] + taps[1];    /* odd frame */
        }
    free(w);
    l->w[0] = upload(gpu, w2, (size_t)2 * channels * channels);
    free(w2);
    l->in_ch = channels;
    l->out_ch = 2 * channels;
    return l->w[0] != NULL;
}

static void free_layers(layer *layers, int count) {
    for (int i = 0; i < count; i++)
        for (int k = 0; k < 3; k++) {
            h3_gpu_tensor_free(layers[i].w[k]);
            h3_gpu_tensor_free(layers[i].b[k]);
        }
}

void h3_vsr_reae_free(h3_vsr_reae *reae) {
    if (!reae) return;
    free_layers(reae->encoder, reae->encoder_count);
    free_layers(reae->decoder, reae->decoder_count);
    free(reae);
}

h3_vsr_reae *h3_vsr_reae_load(const char *model_dir, h3_gpu *gpu,
                              char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!model_dir || !gpu) return NULL;
    h3_weight_store *store = h3_weight_store_open(model_dir, error, error_size);
    if (!store) return NULL;
    h3_vsr_reae *reae = calloc(1, sizeof(*reae));
    if (!reae) {
        h3_weight_store_free(store);
        return NULL;
    }
    reae->gpu = gpu;
    int ok = 1;
    const int E = 64;
    int n = 0, states = 0;
    layer *L = reae->encoder;
#define ADD(kind_) (memset(&L[n], 0, sizeof(layer)), L[n].kind = (kind_), &L[n++])
    /* Encoder (official reae.py). */
    ADD(L_S2D)->in_ch = 3;
    L[n - 1].out_ch = 12;
    ok = ok && load_conv3x3(ADD(L_CONV), store, gpu, "encoder.0", 12, E, 1, error, error_size);
    ADD(L_RELU);
    ok = ok && load_tpool(ADD(L_TPOOL), store, gpu, "encoder.2", E, 2, error, error_size);
    ok = ok && load_conv3x3(ADD(L_CONV_S2), store, gpu, "encoder.3", E, E, 0, error, error_size);
    for (int i = 4; i <= 6 && ok; i++) {
        char p[32]; snprintf(p, sizeof(p), "encoder.%d", i);
        layer *l = ADD(L_MEMBLOCK); l->state_index = states++;
        ok = load_memblock(l, store, gpu, p, E, error, error_size);
    }
    ok = ok && load_tpool(ADD(L_TPOOL), store, gpu, "encoder.7", E, 2, error, error_size);
    ok = ok && load_conv3x3(ADD(L_CONV_S2), store, gpu, "encoder.8", E, E, 0, error, error_size);
    for (int i = 9; i <= 11 && ok; i++) {
        char p[32]; snprintf(p, sizeof(p), "encoder.%d", i);
        layer *l = ADD(L_MEMBLOCK); l->state_index = states++;
        ok = load_memblock(l, store, gpu, p, E, error, error_size);
    }
    ok = ok && load_tpool(ADD(L_TPOOL), store, gpu, "encoder.12", E, 1, error, error_size);
    ok = ok && load_conv3x3(ADD(L_CONV_S2), store, gpu, "encoder.13", E, E, 0, error, error_size);
    for (int i = 14; i <= 16 && ok; i++) {
        char p[32]; snprintf(p, sizeof(p), "encoder.%d", i);
        layer *l = ADD(L_MEMBLOCK); l->state_index = states++;
        ok = load_memblock(l, store, gpu, p, E, error, error_size);
    }
    ok = ok && load_conv3x3(ADD(L_CONV), store, gpu, "encoder.17", E, H3_VSR_LATENT_CHANNELS, 1, error, error_size);
    reae->encoder_count = n;
    reae->encoder_states = states;

    /* Decoder. */
    n = 0; states = 0;
    L = reae->decoder;
    const int W0 = 512, W1 = 256, W2 = 128, W3 = 64;
    ADD(L_CLAMP_TANH);
    ok = ok && load_conv3x3(ADD(L_CONV), store, gpu, "decoder.1", H3_VSR_LATENT_CHANNELS, W0, 1, error, error_size);
    ADD(L_RELU);
    for (int i = 3; i <= 5 && ok; i++) {
        char p[32]; snprintf(p, sizeof(p), "decoder.%d", i);
        layer *l = ADD(L_MEMBLOCK); l->state_index = states++;
        ok = load_memblock(l, store, gpu, p, W0, error, error_size);
    }
    ADD(L_UPSAMPLE);
    ok = ok && load_tgrow(ADD(L_TGROW1), store, gpu, "decoder.7", W0, 1, error, error_size);
    ok = ok && load_conv3x3(ADD(L_CONV), store, gpu, "decoder.8", W0, W1, 0, error, error_size);
    for (int i = 9; i <= 11 && ok; i++) {
        char p[32]; snprintf(p, sizeof(p), "decoder.%d", i);
        layer *l = ADD(L_MEMBLOCK); l->state_index = states++;
        ok = load_memblock(l, store, gpu, p, W1, error, error_size);
    }
    ADD(L_UPSAMPLE);
    ok = ok && load_tgrow(ADD(L_TGROW2), store, gpu, "decoder.13", W1, 2, error, error_size);
    ok = ok && load_conv3x3(ADD(L_CONV), store, gpu, "decoder.14", W1, W2, 0, error, error_size);
    for (int i = 15; i <= 17 && ok; i++) {
        char p[32]; snprintf(p, sizeof(p), "decoder.%d", i);
        layer *l = ADD(L_MEMBLOCK); l->state_index = states++;
        ok = load_memblock(l, store, gpu, p, W2, error, error_size);
    }
    ADD(L_UPSAMPLE);
    ok = ok && load_tgrow(ADD(L_TGROW2), store, gpu, "decoder.19", W2, 2, error, error_size);
    ok = ok && load_conv3x3(ADD(L_CONV), store, gpu, "decoder.20", W2, W3, 0, error, error_size);
    ADD(L_RELU);
    ok = ok && load_conv3x3(ADD(L_CONV), store, gpu, "decoder.22", W3, 12, 1, error, error_size);
    ADD(L_D2S_CLAMP01)->in_ch = 12;
    L[n - 1].out_ch = 3;
    reae->decoder_count = n;
    reae->decoder_states = states;
#undef ADD
    h3_weight_store_free(store);
    if (!ok) {
        if (error && error_size && !error[0]) fail(error, error_size, "cannot load the ReAE");
        h3_vsr_reae_free(reae);
        return NULL;
    }
    return reae;
}

/* ---------------------------------------------------------------- */
/* State */

/* Shapes of each MemBlock's input, walking the table with a unit frame. */
static void memblock_shapes(const layer *layers, int count, int height, int width,
                            int *heights, int *widths, int *channels) {
    int h = height, w = width, c = 3;
    for (int i = 0; i < count; i++) {
        const layer *l = &layers[i];
        switch (l->kind) {
        case L_S2D: h /= 2; w /= 2; c = 12; break;
        case L_D2S_CLAMP01: h *= 2; w *= 2; c = 3; break;
        case L_CONV: c = l->out_ch; break;
        case L_CONV_S2: c = l->out_ch; h /= 2; w /= 2; break;
        case L_MEMBLOCK:
            heights[l->state_index] = h;
            widths[l->state_index] = w;
            channels[l->state_index] = c;
            break;
        case L_TPOOL: break;
        case L_TGROW1: break;
        case L_TGROW2: break;   /* channels stay C after the frame split */
        case L_UPSAMPLE: h *= 2; w *= 2; break;
        case L_RELU: case L_CLAMP_TANH: break;
        }
    }
}

h3_vsr_reae_state *h3_vsr_reae_state_new(const h3_vsr_reae *reae, int encoder,
                                         int height, int width) {
    if (!reae) return NULL;
    const layer *layers = encoder ? reae->encoder : reae->decoder;
    int count = encoder ? reae->encoder_count : reae->decoder_count;
    int states = encoder ? reae->encoder_states : reae->decoder_states;
    h3_vsr_reae_state *state = calloc(1, sizeof(*state));
    if (!state) return NULL;
    state->encoder = encoder;
    state->count = states;
    state->height = height;
    state->width = width;
    state->frames = calloc((size_t)states, sizeof(h3_gpu_tensor *));
    state->valid = calloc((size_t)states, sizeof(int));
    int hs[MAX_LAYERS], ws[MAX_LAYERS], cs[MAX_LAYERS];
    memblock_shapes(layers, count, height, width, hs, ws, cs);
    int ok = state->frames && state->valid;
    for (int i = 0; ok && i < states; i++) {
        state->frames[i] = h3_gpu_tensor_new_f32(reae->gpu, (size_t)hs[i] * ws[i] * cs[i]);
        ok = state->frames[i] != NULL;
    }
    if (!ok) {
        h3_vsr_reae_state_free(state);
        return NULL;
    }
    return state;
}

void h3_vsr_reae_state_free(h3_vsr_reae_state *state) {
    if (!state) return;
    for (int i = 0; i < state->count; i++) h3_gpu_tensor_free(state->frames ? state->frames[i] : NULL);
    free(state->frames);
    free(state->valid);
    free(state);
}

/* ---------------------------------------------------------------- */
/* Runner */

typedef struct {
    h3_gpu_tensor *tensor;
    int t, h, w, c;
} activation;

static int act_alloc(h3_gpu *gpu, activation *a, int t, int h, int w, int c) {
    a->t = t; a->h = h; a->w = w; a->c = c;
    a->tensor = h3_gpu_tensor_new_f32(gpu, (size_t)t * h * w * c);
    return a->tensor != NULL;
}

static void act_free(activation *a) {
    h3_gpu_tensor_free(a->tensor);
    a->tensor = NULL;
}

#define GPU_OP(expression) do { \
    if (!(expression)) { \
        fail(error, error_size, "ReAE GPU op failed: %s (%s)", #expression, h3_gpu_error(gpu)); \
        goto failed; \
    } \
} while (0)

/* Runs one layer on `in` (one batch of frames) and leaves the result in
 * `out` (newly allocated). */
static int run_layer(h3_vsr_reae *reae, const layer *l, h3_vsr_reae_state *state,
                     const activation *in, activation *out, char *error, size_t error_size) {
    h3_gpu *gpu = reae->gpu;
    activation pad = {0}, tmp = {0}, tmp2 = {0};
    memset(out, 0, sizeof(*out));
    switch (l->kind) {
    case L_RELU:
        if (!act_alloc(gpu, out, in->t, in->h, in->w, in->c)) goto oom;
        GPU_OP(h3_gpu_clip_f32(gpu, out->tensor, in->tensor,
                               (uint32_t)((size_t)in->t * in->h * in->w * in->c), 0.0f, FLT_MAX));
        return 1;
    case L_CLAMP_TANH:
        if (!act_alloc(gpu, out, in->t, in->h, in->w, in->c)) goto oom;
        GPU_OP(h3_gpu_vsr_clamp_tanh_f32(gpu, out->tensor, in->tensor,
                                         (uint32_t)((size_t)in->t * in->h * in->w * in->c)));
        return 1;
    case L_S2D:
        if (!act_alloc(gpu, out, in->t, in->h / 2, in->w / 2, in->c * 4)) goto oom;
        GPU_OP(h3_gpu_vsr_space_to_depth_f32(gpu, out->tensor, in->tensor, 1, (uint32_t)in->t,
                                             (uint32_t)in->h, (uint32_t)in->w, (uint32_t)in->c));
        return 1;
    case L_D2S_CLAMP01:
        if (!act_alloc(gpu, &tmp, in->t, in->h, in->w, in->c)) goto oom;
        GPU_OP(h3_gpu_clip_f32(gpu, tmp.tensor, in->tensor,
                               (uint32_t)((size_t)in->t * in->h * in->w * in->c), 0.0f, 1.0f));
        if (!act_alloc(gpu, out, in->t, in->h * 2, in->w * 2, in->c / 4)) goto oom;
        GPU_OP(h3_gpu_vsr_depth_to_space_f32(gpu, out->tensor, tmp.tensor, 1, (uint32_t)in->t,
                                             (uint32_t)in->h, (uint32_t)in->w, (uint32_t)(in->c / 4)));
        act_free(&tmp);
        return 1;
    case L_UPSAMPLE:
        if (!act_alloc(gpu, out, in->t, in->h * 2, in->w * 2, in->c)) goto oom;
        GPU_OP(h3_gpu_vsr_upsample2_f32(gpu, out->tensor, in->tensor, 1, (uint32_t)in->t,
                                        (uint32_t)in->h, (uint32_t)in->w, (uint32_t)in->c));
        return 1;
    case L_CONV:
    case L_CONV_S2: {
        int stride = l->kind == L_CONV_S2 ? 2 : 1;
        if (!act_alloc(gpu, &pad, in->t, in->h + 2, in->w + 2, in->c)) goto oom;
        GPU_OP(h3_gpu_vsr_pad_f32(gpu, pad.tensor, in->tensor, NULL, 1, (uint32_t)in->t,
                                  (uint32_t)in->h, (uint32_t)in->w, (uint32_t)in->c, 0, 1, 1));
        if (!act_alloc(gpu, out, in->t, in->h / stride, in->w / stride, l->out_ch)) goto oom;
        GPU_OP(h3_gpu_conv3d_as_2d_f32(gpu, out->tensor, pad.tensor, l->w[0], l->b[0], 1,
                                       (uint32_t)in->t, (uint32_t)(in->h + 2), (uint32_t)(in->w + 2),
                                       (uint32_t)in->c, (uint32_t)l->out_ch, 1, 3, 3, 1,
                                       (uint32_t)stride, (uint32_t)stride));
        act_free(&pad);
        return 1;
    }
    case L_TPOOL:
        if (in->t % l->stride) {
            fail(error, error_size, "ReAE TPool: %d frames are not a multiple of %d", in->t, l->stride);
            goto failed;
        }
        if (!act_alloc(gpu, out, in->t / l->stride, in->h, in->w, l->out_ch)) goto oom;
        GPU_OP(h3_gpu_conv3d_f32(gpu, out->tensor, in->tensor, l->w[0], NULL, 1, (uint32_t)in->t,
                                 (uint32_t)in->h, (uint32_t)in->w, (uint32_t)in->c, (uint32_t)l->out_ch,
                                 (uint32_t)l->stride, 1, 1, (uint32_t)l->stride, 1, 1));
        return 1;
    case L_TGROW1:
        if (!act_alloc(gpu, out, in->t, in->h, in->w, l->out_ch)) goto oom;
        GPU_OP(h3_gpu_conv3d_as_2d_f32(gpu, out->tensor, in->tensor, l->w[0], NULL, 1, (uint32_t)in->t,
                                       (uint32_t)in->h, (uint32_t)in->w, (uint32_t)in->c,
                                       (uint32_t)l->out_ch, 1, 1, 1, 1, 1, 1));
        return 1;
    case L_TGROW2:
        if (!act_alloc(gpu, &tmp, in->t, in->h, in->w, l->out_ch)) goto oom;
        GPU_OP(h3_gpu_conv3d_as_2d_f32(gpu, tmp.tensor, in->tensor, l->w[0], NULL, 1, (uint32_t)in->t,
                                       (uint32_t)in->h, (uint32_t)in->w, (uint32_t)in->c,
                                       (uint32_t)l->out_ch, 1, 1, 1, 1, 1, 1));
        if (!act_alloc(gpu, out, in->t * 2, in->h, in->w, in->c)) goto oom;
        GPU_OP(h3_gpu_vsr_split_frames_f32(gpu, out->tensor, tmp.tensor, 1, (uint32_t)in->t,
                                           (uint32_t)in->h, (uint32_t)in->w, (uint32_t)in->c));
        act_free(&tmp);
        return 1;
    case L_MEMBLOCK: {
        h3_gpu_tensor *previous = state && state->valid[l->state_index]
            ? state->frames[l->state_index] : NULL;
        /* pad with the past frame in front, 1 pixel around; conv (2,3,3). */
        if (!act_alloc(gpu, &pad, in->t + 1, in->h + 2, in->w + 2, in->c)) goto oom;
        GPU_OP(h3_gpu_vsr_pad_f32(gpu, pad.tensor, in->tensor, previous, 1, (uint32_t)in->t,
                                  (uint32_t)in->h, (uint32_t)in->w, (uint32_t)in->c, 1, 1, 1));
        if (state) {
            /* The state becomes this batch's last input frame. */
            size_t frame = (size_t)in->h * in->w * in->c;
            GPU_OP(h3_gpu_copy_f32(gpu, state->frames[l->state_index], 0, in->tensor,
                                   (size_t)(in->t - 1) * frame, frame));
            state->valid[l->state_index] = 1;
        }
        if (!act_alloc(gpu, &tmp, in->t, in->h, in->w, in->c)) goto oom;
        GPU_OP(h3_gpu_conv3d_f32(gpu, tmp.tensor, pad.tensor, l->w[0], l->b[0], 1, (uint32_t)(in->t + 1),
                                 (uint32_t)(in->h + 2), (uint32_t)(in->w + 2), (uint32_t)in->c,
                                 (uint32_t)in->c, 2, 3, 3, 1, 1, 1));
        act_free(&pad);
        size_t count = (size_t)in->t * in->h * in->w * in->c;
        GPU_OP(h3_gpu_clip_f32(gpu, tmp.tensor, tmp.tensor, (uint32_t)count, 0.0f, FLT_MAX));
        for (int k = 1; k < 3; k++) {
            if (!act_alloc(gpu, &pad, in->t, in->h + 2, in->w + 2, in->c)) goto oom;
            GPU_OP(h3_gpu_vsr_pad_f32(gpu, pad.tensor, tmp.tensor, NULL, 1, (uint32_t)in->t,
                                      (uint32_t)in->h, (uint32_t)in->w, (uint32_t)in->c, 0, 1, 1));
            if (!act_alloc(gpu, &tmp2, in->t, in->h, in->w, in->c)) goto oom;
            GPU_OP(h3_gpu_conv3d_as_2d_f32(gpu, tmp2.tensor, pad.tensor, l->w[k], l->b[k], 1,
                                           (uint32_t)in->t, (uint32_t)(in->h + 2), (uint32_t)(in->w + 2),
                                           (uint32_t)in->c, (uint32_t)in->c, 1, 3, 3, 1, 1, 1));
            act_free(&pad);
            act_free(&tmp);
            tmp = tmp2;
            memset(&tmp2, 0, sizeof(tmp2));
            if (k == 1)
                GPU_OP(h3_gpu_clip_f32(gpu, tmp.tensor, tmp.tensor, (uint32_t)count, 0.0f, FLT_MAX));
        }
        /* relu(conv + x) */
        if (!act_alloc(gpu, out, in->t, in->h, in->w, in->c)) goto oom;
        GPU_OP(h3_gpu_add_scaled_f32(gpu, out->tensor, tmp.tensor, in->tensor, 1.0f, 1.0f, (uint32_t)count));
        GPU_OP(h3_gpu_clip_f32(gpu, out->tensor, out->tensor, (uint32_t)count, 0.0f, FLT_MAX));
        act_free(&tmp);
        return 1;
    }
    }
    fail(error, error_size, "ReAE: unknown layer");
    goto failed;
oom:
    fail(error, error_size, "ReAE: out of GPU memory");
failed:
    act_free(&pad);
    act_free(&tmp);
    act_free(&tmp2);
    act_free(out);
    return 0;
}

/* Runs the whole table over `frames` in batches, appending to `output`
 * (which the caller sized for the full result). */
static int run_table(h3_vsr_reae *reae, const layer *layers, int count,
                     h3_vsr_reae_state *state, const h3_gpu_tensor *input,
                     int in_t, int in_h, int in_w, int in_c, int batch,
                     h3_gpu_tensor *output, int out_t, int out_h, int out_w, int out_c,
                     char *error, size_t error_size) {
    h3_gpu *gpu = reae->gpu;
    if (batch <= 0 || batch > in_t) batch = in_t;
    size_t in_frame = (size_t)in_h * in_w * in_c;
    size_t out_frame = (size_t)out_h * out_w * out_c;
    int out_at = 0;
    if (!h3_gpu_begin(gpu)) {
        fail(error, error_size, "ReAE: cannot begin GPU work: %s", h3_gpu_error(gpu));
        return 0;
    }
    for (int start = 0; start < in_t; start += batch) {
        int frames = start + batch <= in_t ? batch : in_t - start;
        activation current = {0};
        if (!act_alloc(gpu, &current, frames, in_h, in_w, in_c)) {
            fail(error, error_size, "ReAE: out of GPU memory");
            return 0;
        }
        if (!h3_gpu_copy_f32(gpu, current.tensor, 0, input, (size_t)start * in_frame,
                             (size_t)frames * in_frame)) {
            fail(error, error_size, "ReAE: input copy failed: %s", h3_gpu_error(gpu));
            act_free(&current);
            return 0;
        }
        for (int i = 0; i < count; i++) {
            activation next;
            if (!run_layer(reae, &layers[i], state, &current, &next, error, error_size)) {
                act_free(&current);
                return 0;
            }
            act_free(&current);
            current = next;
            /* Keep command buffers short. */
            if (!h3_gpu_continue(gpu)) {
                fail(error, error_size, "ReAE: GPU continue failed: %s", h3_gpu_error(gpu));
                act_free(&current);
                return 0;
            }
        }
        if (current.h != out_h || current.w != out_w || current.c != out_c ||
            out_at + current.t > out_t) {
            fail(error, error_size, "ReAE: shape mismatch [%d,%d,%d,%d] vs [%d,%d,%d,%d]",
                 current.t, current.h, current.w, current.c, out_t, out_h, out_w, out_c);
            act_free(&current);
            return 0;
        }
        if (!h3_gpu_copy_f32(gpu, output, (size_t)out_at * out_frame, current.tensor, 0,
                             (size_t)current.t * out_frame)) {
            fail(error, error_size, "ReAE: output copy failed: %s", h3_gpu_error(gpu));
            act_free(&current);
            return 0;
        }
        out_at += current.t;
        act_free(&current);
    }
    if (!h3_gpu_submit(gpu)) {
        fail(error, error_size, "ReAE: GPU submit failed: %s", h3_gpu_error(gpu));
        return 0;
    }
    if (out_at != out_t) {
        fail(error, error_size, "ReAE: produced %d of %d frames", out_at, out_t);
        return 0;
    }
    return 1;
}

int h3_vsr_reae_encode(h3_vsr_reae *reae, h3_vsr_reae_state *state,
                       const h3_gpu_tensor *frames, int frame_count,
                       int height, int width, int frame_batch,
                       h3_gpu_tensor *latent, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!reae || !frames || !latent || frame_count < 4 || frame_count % 4 ||
        height % 32 || width % 32 || height < 32 || width < 32) {
        fail(error, error_size, "ReAE encode: frames must be 4k, size multiples of 32");
        return 0;
    }
    if (state && (!state->encoder || state->height != height || state->width != width)) {
        fail(error, error_size, "ReAE encode: state belongs to another size");
        return 0;
    }
    if (frame_batch > 0 && frame_batch % 4) frame_batch += 4 - frame_batch % 4;
    return run_table(reae, reae->encoder, reae->encoder_count, state, frames, frame_count,
                     height, width, 3, frame_batch, latent, frame_count / 4, height / 16,
                     width / 16, H3_VSR_LATENT_CHANNELS, error, error_size);
}

int h3_vsr_reae_decode(h3_vsr_reae *reae, h3_vsr_reae_state *state,
                       const h3_gpu_tensor *latent, int latent_count,
                       int latent_height, int latent_width, int latent_batch,
                       h3_gpu_tensor *frames, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!reae || !latent || !frames || latent_count < 1 || latent_height < 2 || latent_width < 2) {
        fail(error, error_size, "ReAE decode: bad arguments");
        return 0;
    }
    if (state && (state->encoder || state->height != latent_height * 16 ||
                  state->width != latent_width * 16)) {
        fail(error, error_size, "ReAE decode: state belongs to another size");
        return 0;
    }
    return run_table(reae, reae->decoder, reae->decoder_count, state, latent, latent_count,
                     latent_height, latent_width, H3_VSR_LATENT_CHANNELS, latent_batch, frames,
                     latent_count * 4, latent_height * 16, latent_width * 16, 3, error, error_size);
}
