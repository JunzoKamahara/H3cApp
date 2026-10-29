/* One-step Wan-family video super-resolution (SwiftVR first, FlashVSR-1.1
 * Tiny second): the engine side of the app's "AI 超解像" upscaler.
 *
 * Design: UPSCALE_PLAN.md (§6). This header has two layers:
 *   1. host-side geometry with no GPU dependency - the chunk protocol
 *      (h3_vsr_chunk.c) and the mask-free shifted-window attention layout,
 *      RoPE tables and patch index math (h3_vsr_window.c). Tested by
 *      tests/test_vsr.c with no weights;
 *   2. the model API (open / requirements / session / cache), implemented
 *      in h3_vsr.c and friends in later phases. Declared here so the Swift
 *      bridge and the tests see one stable interface.
 */
#ifndef H3_VSR_H
#define H3_VSR_H

#include "h3.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Model constants shared by SwiftVR and the ReAE autoencoder.          */

#define H3_VSR_CLIP_FRAMES 24      /* LightX2V clip_len default (multiple of 4) */
#define H3_VSR_TEMPORAL 4          /* ReAE frames per latent step */
#define H3_VSR_SPATIAL 16          /* ReAE pixels per latent cell */
#define H3_VSR_PATCH 2             /* DiT patch (1,2,2): latent cells per token side */
#define H3_VSR_TOKEN_PIXELS (H3_VSR_SPATIAL * H3_VSR_PATCH)  /* 32 px per token side */
#define H3_VSR_FRAMES_TO_TRIM 3    /* ReAE decoder: causal padding frames dropped at the start */
#define H3_VSR_WINDOW 16           /* MFSWA window side, in tokens */
#define H3_VSR_LATENT_CHANNELS 48
#define H3_VSR_HEAD_DIM 128

/* ------------------------------------------------------------------ */
/* Chunk protocol (h3_vsr_chunk.c) - LightX2V swiftvr/streaming.py.      */

typedef enum {
    H3_VSR_CHUNK_FIRST = 0,
    H3_VSR_CHUNK_MIDDLE = 1,
    H3_VSR_CHUNK_LAST = 2
} h3_vsr_chunk_type;

typedef struct {
    h3_vsr_chunk_type type;
    int index;              /* 0-based */
    int start;              /* first padded frame of the chunk */
    int frames;             /* frames fed to the ReAE encoder (before the
                               encoder's own 3 tail copies on the last chunk) */
    int latents;            /* (frames - 1) / 4 + 1 */
    int output_start;       /* real output frames this chunk is responsible for */
    int output_stop;        /* exclusive; clipped to the real frame count */
    int dit_prefix_latents; /* LAST only: latents borrowed from the previous
                               chunk (zeros when there is none) so the DiT sees
                               clip_frames / 4 + 1 latents */
    int rope_offset;        /* temporal RoPE offset of the first latent fed
                               to the DiT (dit_overlap 0) */
} h3_vsr_chunk;

/* Smallest 4k+1 count holding `frames` frames (frames >= 1). */
int h3_vsr_padded_frame_count(int frames);

/* Number of chunks for `raw_frames` real frames and a clip of `clip_frames`
 * (positive multiple of 4). 0 on bad arguments. */
int h3_vsr_chunk_count(int raw_frames, int clip_frames);

/* Fills up to `capacity` chunks; returns the count (0 on bad arguments or
 * when capacity is too small). Frames past the real count are the last real
 * frame repeated, the same as the reference reader. Output ranges partition
 * [0, raw_frames). */
int h3_vsr_chunk_plan(int raw_frames, int clip_frames,
                      h3_vsr_chunk *chunks, int capacity);

/* ------------------------------------------------------------------ */
/* Mask-free shifted-window attention layout (h3_vsr_window.c).        */

/* Window starts along one axis (LightX2V axis_window_starts): {0} when the
 * axis fits in one window; otherwise k*window - shift clamped to
 * [0, size - window], deduplicated, and interior starts already covered by
 * their neighbours dropped. shift is window / 2 when `shifted`. Returns the
 * count (<= capacity; 0 when capacity is too small). */
int h3_vsr_axis_window_starts(int size, int window, int shifted,
                              int *starts, int capacity);

typedef struct {
    int frames, height, width;     /* token grid (T, H/32, W/32) */
    int window_h, window_w;        /* effective window (clamped to the grid) */
    int shifted;
    int tokens;                    /* N = frames * height * width */
    int window_count;              /* Nw */
    int window_length;             /* Lw = frames * window_h * window_w */
    int32_t *indices;              /* [Nw * Lw]: token gathered at that row */
    int32_t *owner;                /* [N]: gathered row whose output the
                                      token takes (window * Lw + local) */
    int32_t *owned_offset;         /* [Nw + 1]: owned_local range per window */
    int32_t *owned_local;          /* [N]: local positions (0..Lw-1) a window
                                      owns, ascending, grouped per window */
    int32_t *owned_token;          /* [N]: the token at each owned_local entry */
} h3_vsr_window_layout;

/* Builds the layout for a token grid. Rows inside a window are ordered
 * frame-major, then window row, then window column (LightX2V). The owner of
 * a token covered by several windows is the front-most window when
 * `shifted` is 0 and the last window when it is 1 (LightX2V and the
 * official WanShiftWindow2DInferProcessor agree). Returns 1 on success. */
int h3_vsr_window_layout_build(int frames, int height, int width,
                               int window_h, int window_w, int shifted,
                               h3_vsr_window_layout *layout);
void h3_vsr_window_layout_free(h3_vsr_window_layout *layout);

/* ------------------------------------------------------------------ */
/* RoPE and patch index math (h3_vsr_window.c).                        */

/* Wan 3D RoPE for a (frames, height, width) token grid: per token head_dim/2
 * (cos, sin) pairs in F32, pair p rotating channels (2p, 2p+1). Pairs are
 * split t: head_dim/2 - 2*(head_dim/6), h: head_dim/6, w: head_dim/6 (44/42/42
 * channels at 128), theta 10000, positions t + t_offset, h, w. Output arrays
 * hold tokens * head_dim / 2 floats, token-major, in grid order (t, h, w). */
void h3_vsr_rope_table(int frames, int height, int width, int t_offset,
                       int head_dim, float *cos_out, float *sin_out);

/* Token index of latent cell (t, h, w) in the DiT sequence: (t*H + h)*W + w
 * over the token grid. */
static inline int h3_vsr_token_index(int t, int h, int w,
                                     int height, int width) {
    return (t * height + h) * width + w;
}

/* The DiT input vector (192 values) is the patch embedding's Conv3d weight
 * order (c, ph, pw): channel-major. */
static inline int h3_vsr_patch_in_index(int channel, int ph, int pw) {
    return (channel * H3_VSR_PATCH + ph) * H3_VSR_PATCH + pw;
}

/* The DiT output vector (192 values) follows diffusers' unpatchify
 * reshape(..., p_t, p_h, p_w, C): (ph, pw, c), channel fastest. */
static inline int h3_vsr_patch_out_index(int channel, int ph, int pw) {
    return (ph * H3_VSR_PATCH + pw) * H3_VSR_LATENT_CHANNELS + channel;
}

/* ------------------------------------------------------------------ */
/* Model API (h3_vsr.c, h3_vsr_weights.c; phases P2-P5 of the plan).    */

typedef struct h3_vsr h3_vsr;                 /* loaded model, reused across videos */
typedef struct h3_vsr_session h3_vsr_session; /* one video: carried causal state */

typedef enum { H3_VSR_SWIFTVR = 1, H3_VSR_FLASHVSR_TINY = 2 } h3_vsr_kind;
typedef enum {
    H3_VSR_PRECISION_AUTO = 0,
    H3_VSR_PRECISION_BF16 = 1,
    H3_VSR_PRECISION_INT8 = 2
} h3_vsr_precision;
typedef enum { H3_VSR_BGRA8 = 0, H3_VSR_RGB24 = 1 } h3_vsr_pixel_format;

typedef struct {
    h3_vsr_kind kind;
    char name[64];
    int fixed_factor_x100;      /* 0 = any output size (SwiftVR); 400 = 4x only */
    int output_multiple;        /* 32 (SwiftVR, internal pad + crop); 128 (FlashVSR) */
    int min_output_side, max_output_side;
    int first_chunk_frames, chunk_frames;   /* 28, 24 */
    uint64_t weight_bytes_bf16, weight_bytes_int8;
    int int8_supported;         /* tensor hardware (M5) on this machine */
    char license_id[64];        /* "SwiftVR-Research-License" / "Apache-2.0" */
    int commercial_use_allowed;
} h3_vsr_info;

/* Reads config.json and the safetensors headers only. */
/* Whether this Mac's GPU takes the int8 (tensor hardware) path, without
 * a model: what h3_vsr_info.int8_supported reports. */
int h3_vsr_int8_supported(void);

int h3_vsr_inspect_dir(const char *model_dir, h3_vsr_info *info,
                       char *error, size_t error_size);

/* Weight cache (.h3vc): built once on the device from the downloaded F32
 * safetensors; the raw weights are never run directly. */
int h3_vsr_cache_path(const char *model_dir, h3_vsr_precision precision,
                      char *path, size_t size);
int h3_vsr_cache_matches(const char *cache_path, const char *model_dir);
int h3_vsr_build_cache(const char *model_dir, const char *cache_path,
                       h3_vsr_precision precision,
                       const char *shader_source_path,
                       h3_progress_callback on_progress, void *opaque,
                       char *error, size_t error_size);

typedef struct {
    h3_vsr_precision precision;   /* AUTO: INT8 on tensor hardware, else BF16 */
    const char *cache_path;       /* required */
    const char *shader_source_path;
    int dit_overlap;              /* latents prepended from the previous chunk;
                                     0 = the reference restore_video */
    int reae_frame_batch;         /* 0 = auto from memory */
    int reae_separate_queue;      /* 1: the decoder runs on its own GPU
                                     context under the next chunk's DiT, so
                                     output lags one chunk (no faster where
                                     the DiT saturates the GPU); 0: one
                                     context */
    int keep_fc2_bf16;            /* quality fallback for int8 */
    int full_attention;           /* 1: dense attention instead of MFSWA */
    int color_match;              /* 0 (default) or 1: shift each output
                                     chunk's channel means to the input's
                                     (SwiftVR brightens by about 3/255);
                                     -1: no correction, the model's output */
} h3_vsr_open_params;

h3_vsr *h3_vsr_open(const char *model_dir, const h3_vsr_open_params *params,
                    h3_progress_callback on_progress, void *opaque,
                    char *error, size_t error_size);
void h3_vsr_close(h3_vsr *vsr);

typedef struct {
    int supported;
    char reason[256];
    uint64_t gpu_bytes, host_bytes;
    double estimated_seconds_per_chunk;
    int tokens_per_chunk, windows_even, windows_odd;
} h3_vsr_requirements;

int h3_vsr_check_requirements(const h3_vsr *vsr, int in_w, int in_h,
                              int out_w, int out_h, h3_vsr_requirements *out);

h3_vsr_session *h3_vsr_session_open(h3_vsr *vsr, int in_w, int in_h,
                                    int out_w, int out_h,
                                    int total_frames /* 0 = unknown */,
                                    char *error, size_t error_size);
int h3_vsr_session_next_chunk_frames(const h3_vsr_session *session);

typedef struct {
    const uint8_t *pixels;
    int width, height, stride_bytes;
    h3_vsr_pixel_format format;
} h3_vsr_image;

typedef struct {
    uint8_t *pixels;
    int width, height, stride_bytes;
    h3_vsr_pixel_format format;
} h3_vsr_image_out;

/* Feeds one input chunk and writes the finished output frames (in order;
 * with the decoder on its own context a call returns the previous chunk's
 * frames, and is_last flushes everything so total out == total in; size
 * `out` for h3_vsr_session_pending_frames() + in_frames).
 * Progress phases: "vsr encode", "vsr dit" (block / 30), "vsr decode". A
 * non-zero callback return or *cancel_requested stops at the next block or
 * phase boundary with the error "vsr cancelled". */
int h3_vsr_session_process(h3_vsr_session *session,
                           const h3_vsr_image *in, int in_frames, int is_last,
                           h3_vsr_image_out *out, int out_capacity,
                           int *out_frames,
                           h3_progress_callback on_progress, void *opaque,
                           const volatile int *cancel_requested,
                           char *error, size_t error_size);
int h3_vsr_session_pending_frames(const h3_vsr_session *session);
void h3_vsr_session_close(h3_vsr_session *session);

/* GPU time spent so far (seconds) on the DiT and ReAE contexts and the
 * larger of their peak live byte counts. */
int h3_vsr_get_stats(const h3_vsr *vsr, double *dit_seconds,
                     double *reae_seconds, uint64_t *peak_live_bytes);

#ifdef __cplusplus
}
#endif
#endif
