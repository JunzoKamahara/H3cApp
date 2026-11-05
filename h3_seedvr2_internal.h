/* SeedVR2 internals shared by h3_seedvr2.c (load, transformer, colour,
 * sessions) and h3_seedvr2_vae.c (the VAE on the F32 / MPSGraph path and
 * the TensorOps grid). Not a public header. */
#ifndef H3_SEEDVR2_INTERNAL_H
#define H3_SEEDVR2_INTERNAL_H

#include "h3_seedvr2.h"
#include "h3_gpu.h"
#include "h3_metal.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


/* ---------------------------------------------------------------- */
/* Configuration (mflux SeedVR2Transformer / SeedVR2VAE defaults) */

#define SV2_DIM 2560
#define SV2_HEADS 20
#define SV2_HEAD_DIM 128
#define SV2_LAYERS 32
#define SV2_MM_LAYERS 10
#define SV2_HIDDEN 6912
#define SV2_TXT_TOKENS 58
#define SV2_TXT_IN 5120
#define SV2_VID_IN 33
#define SV2_VID_OUT 16
#define SV2_PATCH_IN (SV2_VID_IN * 4)      /* 132 */
#define SV2_PATCH_OUT (SV2_VID_OUT * 4)    /* 64 */
#define SV2_ROPE_FREQS 21                   /* rope_dim 128 / 3 axes / 2 */
#define SV2_ROPE_PAIRS (3 * SV2_ROPE_FREQS) /* 63 pairs = 126 rotated dims */
#define SV2_NORM_EPS 1e-5f
#define SV2_VAE_EPS 1e-6f
#define SV2_VAE_GROUPS 32
#define SV2_LATENT 16
#define SV2_SCALING 0.9152f
#define SV2_TIMESTEP 1000.0f
#define SV2_PIXEL_MULTIPLE 16
#define SV2_QUANT_GROUP 64

/* ---------------------------------------------------------------- */
/* Small host helpers */

static inline void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static inline double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline int trace_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("H3_SV2_TRACE") != NULL;
    return cached;
}

static inline void trace(double *mark, const char *label) {
    if (!trace_enabled()) return;
    double t = now_seconds();
    fprintf(stderr, "sv2 %-24s %8.1f ms\n", label, (t - *mark) * 1000.0);
    *mark = t;
}

/* H3_SV2_SYNC: the trace waits for the GPU first, so the stage times are
 * real (slower overall: the command buffer is split at every mark). */
static inline void trace_sync(h3_gpu *gpu, double *mark, const char *label) {
    if (!trace_enabled()) return;
    if (getenv("H3_SV2_SYNC") && h3_gpu_command_open(gpu)) { h3_gpu_submit(gpu); h3_gpu_begin(gpu); }
    trace(mark, label);
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1fu;
    uint32_t mantissa = h & 0x3ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 127 - 15 + 1;
            while (!(mantissa & 0x400u)) { mantissa <<= 1; exponent--; }
            mantissa &= 0x3ffu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static inline int round_up(int value, int multiple) {
    return (value + multiple - 1) / multiple * multiple;
}


/* ---------------------------------------------------------------- */
/* Weights */

typedef struct {
    h3_gpu_tensor *w;      /* BF16 [out, in]; NULL once quantized */
    h3_gpu_tensor *b;      /* F32 [out] or NULL */
    h3_gpu_tensor *q;      /* int8 [out, in] with F32 `scales` [out] (tensor hardware) */
    h3_gpu_tensor *scales;
    uint32_t in, out;
    int owned;             /* 0: shares another linear's tensors */
} sv2_linear;

typedef struct {
    int shared, last;
    sv2_linear qkv_vid, out_vid, qkv_txt, out_txt;
    sv2_linear fc1_vid, fc2_vid, fc1_txt, fc2_txt;   /* fc1 = [gate; up] */
    h3_gpu_tensor *nq_vid, *nk_vid, *nq_txt, *nk_txt;   /* F32 [128]; txt = vid when shared */
    /* attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate:
     * emb + the block's parameters, F32 [dim]; txt NULL in the last block */
    h3_gpu_tensor *mod_vid[6], *mod_txt[6];
} sv2_block;

typedef struct {
    h3_gpu_tensor *w;      /* F32 [out, in, kh, kw] (OIHW); NULL on the TensorOps path */
    h3_gpu_tensor *b;      /* F32 [out] */
    int in, out, kernel;
    /* TensorOps path: tap-major BF16 [taps * out_p, in_p] on channels padded
     * to 64, and the bias padded to out_p (h3_gpu_vsr_tapconv_bf16) */
    h3_gpu_tensor *nw, *nb;
    int in_p, out_p;
} sv2_conv;

typedef struct {
    h3_gpu_tensor *w, *b;  /* F32 [C] */
} sv2_norm;

typedef struct {
    int in, out;
    sv2_norm norm1, norm2;
    sv2_conv conv1, conv2, shortcut;   /* shortcut.w NULL when in == out */
} sv2_resnet;

typedef struct {
    sv2_norm norm;
    h3_gpu_tensor *wq, *bq, *wk, *bk, *wv, *bv, *wo, *bo;   /* F32 [C, C], [C] */
} sv2_attention;

typedef struct {
    sv2_conv conv_in;
    sv2_resnet down[4][2];
    sv2_conv downsample[3];            /* 3x3 stride 2 (temporal taps folded) */
    sv2_resnet mid_a, mid_b;
    sv2_attention mid_attn;
    sv2_norm norm_out;
    sv2_conv conv_out;                 /* 512 -> 32 */
} sv2_encoder;

typedef struct {
    sv2_conv conv_in;                  /* 16 -> 512 */
    sv2_resnet mid_a, mid_b;
    sv2_attention mid_attn;
    sv2_resnet up[4][3];
    h3_gpu_tensor *up_w[3], *up_b[3];  /* upscale_conv 1x1: F32 [C * 4 * tf, C] */
    sv2_conv up_nax[3];                /* the same as a 1-tap TensorOps conv */
    int up_temporal[3];
    sv2_conv up_conv[3];               /* the 3x3 after the shuffle */
    sv2_norm norm_out;
    sv2_conv conv_out;                 /* 128 -> 3 */
} sv2_decoder;

struct h3_seedvr2 {
    h3_gpu *gpu;
    h3_seedvr2_info info;
    h3_seedvr2_open_params params;
    char model_dir[1024];
    int nax_vae;                       /* the VAE on the TensorOps grid (M5) */
    int int8_dit;                      /* the transformer's linears per-row int8 (M5) */
    /* transformer */
    sv2_linear vid_in, vid_out;
    h3_gpu_tensor *txt0;               /* BF16 [58, 2560]: txt_in(pos_emb) */
    sv2_block blocks[SV2_LAYERS];
    h3_gpu_tensor *out_scale, *out_shift;   /* F32 [dim]: norm weight * (scale_a + out_scale), shift_a + out_shift */
    /* vae */
    sv2_encoder enc;
    sv2_decoder dec;
    /* activation pool (F32 and BF16 buffers reused across frames) */
    struct { h3_gpu_tensor *t; size_t bytes; int bf16; } pool[64];
    int pool_count;
    /* load-time temporaries freed after the next submit */
    h3_gpu_tensor *deferred[256];
    int deferred_count;
    char error[512];
};

#define GPU_OP(call) do { if (!(call)) { fail(error, error_size, "SeedVR2 VAE: %s", h3_gpu_error(m->gpu)); return 0; } } while (0)

/* VAE activation: F32 channels-last [h, w, c]. */
typedef struct { h3_gpu_tensor *t; int h, w, c; } act;

/* h3_seedvr2.c */
float *sv2_read_f16_as_f32(const h3_st_header *h, const char *name, uint64_t expected,
                           char *error, size_t error_size);
h3_gpu_tensor *sv2_upload_f32(h3_gpu *gpu, const float *values, size_t n);
h3_gpu_tensor *sv2_load_f16_f32(h3_gpu *gpu, const h3_st_header *h, const char *name,
                                uint64_t expected, char *error, size_t error_size);
h3_gpu_tensor *sv2_pool_take(h3_seedvr2 *m, size_t elements, int bf16);
void sv2_pool_give(h3_seedvr2 *m, h3_gpu_tensor *t);

/* h3_seedvr2_vae.c */
int sv2_load_vae(h3_seedvr2 *m, const h3_st_header *h, char *error, size_t error_size);
void sv2_free_vae(h3_seedvr2 *m);
int sv2_act_new(h3_seedvr2 *m, act *a, int h, int w, int c);
void sv2_act_free(h3_seedvr2 *m, act *a);
/* stop: 0 after conv_in, 1..4 after down block i-1, 5 after the mid block,
 * 6 (or more) the latent [h, w, 32] (mean in the first 16 channels, unscaled). */
int sv2_vae_encode_stage(h3_seedvr2 *m, const act *rgb, act *out, int stop, char *error, size_t error_size);
int sv2_vae_encode(h3_seedvr2 *m, const act *rgb, act *out, char *error, size_t error_size);
/* latent [h, w, 16] (already divided by the scaling factor) -> RGB [8h, 8w, 3]. */
int sv2_vae_decode(h3_seedvr2 *m, const act *z, act *out, char *error, size_t error_size);

#endif
