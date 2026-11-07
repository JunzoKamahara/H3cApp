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
    h3_gpu_tensor *w;      /* F32 [out, in, kt, kh, kw] (OIDHW; kt = 1: OIHW); NULL on the TensorOps path */
    h3_gpu_tensor *b;      /* F32 [out] */
    int in, out, kernel;
    int kt;                /* temporal taps as loaded: 1 (folded, or 1x1x1 / 1x3x3) or 3 */
    int slot;              /* index of the causal state (sv2_causal.mem) */
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
    int vae_temporal;                  /* VAE weights with their temporal taps (video) or folded (image) */
    int conv_slots;                    /* causal convolutions numbered at load */
    int dit_loaded;                    /* the 32 transformer blocks are resident */
    uint64_t dit_bytes;                /* what they take */
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
    int pool_bypass;                   /* clips: free at once (their buffers are GBs; the layers wait for the GPU) */
    int state_int8;                    /* this clip keeps the grid's causal state in per-row int8 (half the memory,
                                          not bit-exact): h3_seedvr2_clip_params.state_mode */
    /* H3_SV2_TRACE accounting of a clip's buffer traffic and GPU waits */
    struct { uint64_t allocs, alloc_bytes, pool_hits, pool_hit_bytes, cap_frees, cap_free_bytes, syncs; double sync_seconds; } clip_stats;
    /* load-time temporaries freed after the next submit */
    h3_gpu_tensor *deferred[256];
    int deferred_count;
    char error[512];
};

#define GPU_OP(call) do { if (!(call)) { fail(error, error_size, "SeedVR2 VAE: %s", h3_gpu_error(m->gpu)); return 0; } } while (0)

/* VAE activation: F32 channels-last [h, w, c]. */
typedef struct { h3_gpu_tensor *t; int h, w, c, f; } act;   /* F32 [f, h, w, c]; f 0 means 1 */
static inline int act_frames(const act *a) { return a->f > 0 ? a->f : 1; }

/* The causal state of a clip processed in temporal slices (the reference's
 * MemoryState): DISABLED is the whole clip at once (the head extended by
 * copies of frame 0), INIT the first slice (the same, keeping each causal
 * convolution's last kt - stride input frames), ACTIVE a later slice (those
 * frames in place of the copies; the temporal upsample keeps frame 1). */
enum { SV2_CAUSAL_DISABLED = 0, SV2_CAUSAL_INIT = 1, SV2_CAUSAL_ACTIVE = 2 };
#define SV2_MAX_CONVS 96
typedef struct {
    int mode;
    h3_gpu_tensor *mem[SV2_MAX_CONVS];  /* F32 [k, h, w, c] or a slim grid of k frames (BF16; I8 rows with `scale`) */
    int mem_frames[SV2_MAX_CONVS];
    h3_gpu_tensor *scale[SV2_MAX_CONVS];   /* F32 per-row scales when the grid state is kept in int8 (m->state_int8) */
    uint32_t mem_rows[SV2_MAX_CONVS];      /* rows of that slim grid */
    int ring[SV2_MAX_CONVS];               /* grid: the physical slot of the oldest frame (the state is a ring:
                                              a new frame overwrites the oldest slot instead of a new buffer) */
} sv2_causal;

/* The transformer's attention layout (h3_seedvr2.c) */
#define SV2_MAX_CLASSES 64

typedef struct { int hs, he, ws, we, ts, te; } sv2_window;

typedef struct {
    int tokens_w, tokens_h, tokens_t, N;
    int windows;                   /* W */
    int rows;                      /* every window's video rows + text rows */
    int classes;
    struct { int first, count, length, row_offset; } cls[SV2_MAX_CLASSES];
    h3_gpu_tensor *index[SV2_MAX_CLASSES], *cosine[SV2_MAX_CLASSES], *sine[SV2_MAX_CLASSES],
                  *dest[SV2_MAX_CLASSES];
    h3_gpu_tensor *q[SV2_MAX_CLASSES], *k[SV2_MAX_CLASSES], *v[SV2_MAX_CLASSES], *a[SV2_MAX_CLASSES];
    h3_gpu_tensor *base;           /* U32 [W]: N + w * 58, the text rows' home in `att` */
} sv2_layout;

/* h3_seedvr2.c */
void sv2_pool_drain(h3_seedvr2 *m);
void sv2_dit_unload(h3_seedvr2 *m);
int sv2_dit_ensure(h3_seedvr2 *m, char *error, size_t error_size);   /* loads again when unloaded */
int sv2_layout_build(h3_seedvr2 *m, sv2_layout *L, int tokens_w, int tokens_h, int tokens_t, int shifted,
                     char *error, size_t error_size);
void sv2_layout_free(h3_seedvr2 *m, sv2_layout *L);
/* tokens BF16 [N, 132] -> pred BF16 [N, 64]; layouts[0] even blocks, [1] odd */
int sv2_dit_forward(h3_seedvr2 *m, sv2_layout *layouts, const h3_gpu_tensor *tokens, h3_gpu_tensor *pred,
                    int dump_block, h3_progress_callback on_progress, void *opaque,
                    const volatile int *cancel_requested, char *error, size_t error_size);
void sv2_pack_tokens(const float *noise, const float *cond, int lw, int lh, int frames, uint16_t *tokens);
void sv2_unpack_pred(const uint16_t *pred, int lw, int lh, int frames, float *out);
/* wavelet + LAB colour transfer on one F32 [H, W, 3] frame (out may be content) */
int sv2_color_transfer(h3_seedvr2 *m, h3_gpu_tensor *out, const h3_gpu_tensor *content,
                       const h3_gpu_tensor *style, int width, int height, char *error, size_t error_size);
float *sv2_read_f16_as_f32(const h3_st_header *h, const char *name, uint64_t expected,
                           char *error, size_t error_size);
h3_gpu_tensor *sv2_upload_f32(h3_gpu *gpu, const float *values, size_t n);
h3_gpu_tensor *sv2_load_f16_f32(h3_gpu *gpu, const h3_st_header *h, const char *name,
                                uint64_t expected, char *error, size_t error_size);
h3_gpu_tensor *sv2_pool_take(h3_seedvr2 *m, size_t elements, int bf16);
void sv2_pool_give(h3_seedvr2 *m, h3_gpu_tensor *t);

/* latent frames [first_frame, first_frame + frames) of a clip's noise -> [16, frames, h, w] */
int sv2_fill_noise_clip(float *out, int first_frame, int frames, int lw, int lh, uint64_t seed);

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

/* Temporal VAE (video): the weights must be in temporal form
 * (sv2_vae_set_temporal). ctx carries the mode and the per-convolution
 * state across slices; *_stage stops early (encoder as above; decoder 0
 * after conv_in, 1 after the mid block, 2..5 after up block i-2, 6 or more
 * the RGB frames). */
int sv2_vae_set_temporal(h3_seedvr2 *m, int temporal, char *error, size_t error_size);
void sv2_causal_reset(sv2_causal *ctx);
int sv2_vae_encode_clip(h3_seedvr2 *m, const act *rgb, act *out, int stop, sv2_causal *ctx,
                        char *error, size_t error_size);
int sv2_vae_decode_clip(h3_seedvr2 *m, const act *z, act *out, int stop, sv2_causal *ctx,
                        char *error, size_t error_size);
/* The whole clip in slices of `split` frames after the first 1 + split
 * (encode), or of max(1, split / 4) latent frames after the first 1 + that
 * (decode), the state carried; split 0 (or a clip that short) is one pass. */
int sv2_vae_encode_video(h3_seedvr2 *m, const act *rgb, act *out, int split,
                         h3_progress_callback on_progress, void *opaque, const volatile int *cancel,
                         char *error, size_t error_size);
int sv2_vae_decode_video(h3_seedvr2 *m, const act *z, act *out, int split,
                         h3_progress_callback on_progress, void *opaque, const volatile int *cancel,
                         char *error, size_t error_size);

#endif
