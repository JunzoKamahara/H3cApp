/* SeedVR2-3B (ByteDance Seed, Apache-2.0): the one-step diffusion image
 * restorer, run frame by frame as the app's slow, high-detail enlarger.
 * Architecture and weights follow mflux's MLX implementation (MIT) and the
 * mlx-community/SeedVR2-3B-mlx-int8 export (MLX affine 8-bit, group 64),
 * which this engine reads directly: the weights are dequantised to BF16 on
 * the GPU at load time, no cache file. UPSCALE_PLAN appendix A.7.
 *
 * Per frame: the input is resized to the output size with PIL's bicubic,
 * padded to multiples of 16 (black) and encoded by the causal 3D VAE (one
 * frame, so every temporal tap folds into a 2D convolution); the DiT sees
 * [noise, latent, mask] and predicts the velocity once at t = 1000;
 * z = noise - v is decoded, cropped and (by default) colour-matched to the
 * input with the reference's wavelet + L*a*b* histogram transfer. */
#ifndef H3_SEEDVR2_H
#define H3_SEEDVR2_H

#include "h3.h"
#include "h3_vsr.h"   /* h3_vsr_image / h3_vsr_image_out */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct h3_seedvr2 h3_seedvr2;
typedef struct h3_seedvr2_session h3_seedvr2_session;

typedef struct {
    char name[64];              /* "SeedVR2-3B" */
    char variant[32];           /* config.json "variant" */
    char dtype[16];             /* "int8" / "fp16" */
    int quant_bits, quant_group;
    uint64_t file_bytes;        /* the three safetensors */
    uint64_t resident_bytes;    /* BF16 transformer + F32 VAE on the GPU */
    uint64_t block_bytes;       /* of those, the transformer's blocks (BF16), which a
                                   video window frees around its VAE phases */
    int tensor_ops;             /* this Mac has tensor hardware (NAX linears) */
} h3_seedvr2_info;

/* Reads config.json and the safetensors headers. */
int h3_seedvr2_inspect_dir(const char *model_dir, h3_seedvr2_info *info,
                           char *error, size_t error_size);

typedef struct {
    const char *shader_source_path;
    int color_transfer;     /* 1: the reference's LAB transfer (default 1); 0: the model's output */
    uint64_t seed;          /* the per-frame noise (the same for every frame of a session) */
    int nax_linear;         /* -1 auto (on with tensor hardware), 0 off, 1 on */
    int vae_f32;            /* 1: the VAE on the F32 MPSGraph path even with tensor hardware */
    int int8;               /* 1: per-row int8 transformer linears on tensor hardware (faster, about
                               10 dB further from the reference); 0 or -1: BF16 (-1 honours H3_SV2_INT8) */
} h3_seedvr2_open_params;

h3_seedvr2 *h3_seedvr2_open(const char *model_dir, const h3_seedvr2_open_params *params,
                            h3_progress_callback on_progress, void *opaque,
                            char *error, size_t error_size);
void h3_seedvr2_close(h3_seedvr2 *model);

typedef struct {
    int supported;
    char reason[256];
    uint64_t gpu_bytes;             /* peak estimate */
    double estimated_seconds_per_frame;
    int tokens;                     /* DiT tokens per frame */
    int windows_even, windows_odd;
} h3_seedvr2_requirements;

int h3_seedvr2_check_requirements(const h3_seedvr2 *model, int in_w, int in_h,
                                  int out_w, int out_h, h3_seedvr2_requirements *out);

/* Whether h3_seedvr2_check_requirements would take this size, from what
 * h3_seedvr2_inspect_dir reports: the app's sheet asks before the model is
 * loaded. */
int h3_seedvr2_output_fits(const h3_seedvr2_info *info, int in_w, int in_h, int out_w, int out_h);

h3_seedvr2_session *h3_seedvr2_session_open(h3_seedvr2 *model, int in_w, int in_h,
                                            int out_w, int out_h,
                                            char *error, size_t error_size);
/* One frame in, one frame out. Progress phases: "vsr encode", "vsr dit"
 * (block / 32), "vsr decode"; a non-zero callback return or
 * *cancel_requested stops with the error "vsr cancelled". */
int h3_seedvr2_session_process(h3_seedvr2_session *session, const h3_vsr_image *in,
                               h3_vsr_image_out *out,
                               h3_progress_callback on_progress, void *opaque,
                               const volatile int *cancel_requested,
                               char *error, size_t error_size);
void h3_seedvr2_session_close(h3_seedvr2_session *session);

/* One window of a video: `frames` (4n+1) frames in, the same number out,
 * through the temporal VAE and the transformer's 3D windows (the reference's
 * inference loop for one window; the session built on it handles overlap
 * and the tail). vae_split: the causal slicing of the VAE (-1 or 4: the
 * reference's 1 + 4 frames then 4, decode 1 + 1 latent then 1; 0: the
 * whole window at once - more memory). first_latent_frame numbers the
 * noise so overlapping windows agree. seed 0 takes the model's;
 * color_transfer -1 the model's. noise_override ([16, L, h, w]) and
 * dump_block are for tests. Progress phases: "vsr encode" / "vsr decode"
 * (slices), "vsr dit" (blocks), "vsr output" (frames). */
typedef struct {
    int vae_split;
    int first_latent_frame;
    uint64_t seed;
    int color_transfer;
    const float *noise_override;
    int dump_block;
    int state_mode;         /* the causal state on the TensorOps grid: 0 BF16 unless the window only fits
                               with it in per-row int8 (half the memory), then in group int4 (a quarter;
                               neither bit-exact); 1 BF16 always (refused when it does not fit); 2 int8
                               always; 3 int4 always */
} h3_seedvr2_clip_params;
/* Whether a window of `window_frames` (4n+1) fits this Mac, with the VAE
 * form it would use (vae_split 0: the whole window; 4: causal slices) and
 * the peak GPU memory it needs, from measurements (h3_seedvr2_video.c).
 * Returns 0 only for bad arguments; `supported` says. */
typedef struct {
    int supported;
    char reason[256];
    uint64_t gpu_bytes;
    int vae_split;
    int state_bits;                 /* 0: the causal state in BF16; 8 / 4: fits only with it in int8 / int4 */
    double estimated_seconds_per_frame;
    int latent_frames, tokens;
} h3_seedvr2_video_requirements;
int h3_seedvr2_video_check_requirements(const h3_seedvr2 *model, int in_w, int in_h, int out_w, int out_h,
                                        int window_frames, h3_seedvr2_video_requirements *out);
/* Whether h3_seedvr2_video_check_requirements would take this size and
 * window, from what h3_seedvr2_inspect_dir read (no weights loaded; the
 * app's sheet asks before loading anything). */
int h3_seedvr2_video_output_fits(const h3_seedvr2_info *info, int in_w, int in_h, int out_w, int out_h,
                                 int window_frames);
int h3_seedvr2_restore_clip(h3_seedvr2 *model, const h3_vsr_image *in, int frames, int out_w, int out_h,
                            h3_vsr_image_out *out, const h3_seedvr2_clip_params *params,
                            h3_progress_callback on_progress, void *opaque,
                            const volatile int *cancel_requested, char *error, size_t error_size);

/* A video in overlapping windows (the same contract as h3_vsr_session_*):
 * window_frames 4n+1 (5 ... 121; 1 is one frame at a time through the
 * clip path), overlap_frames 0 or 4n+1 below the window; vae_slicing -1
 * lets the clip choose; blend 0 keeps the earlier window's frames in the
 * overlap, 1 cross-fades them (Hann); seed 0 and color_transfer -1 take
 * the model's. Starts are multiples of 4 so windows agree on the latent
 * grouping and the noise; with total_frames known the last window ends on
 * the last frame instead of being padded. next_frames says how many to
 * bring (fewer only with is_last); size `out` for pending + in_frames.
 * Progress phases are the clip's. A NULL model opens a dry run (tests). */
typedef struct {
    int window_frames;
    int overlap_frames;
    int vae_slicing;
    int blend;
    uint64_t seed;
    int color_transfer;
    int state_mode;         /* h3_seedvr2_clip_params.state_mode */
} h3_seedvr2_video_params;
typedef struct h3_seedvr2_video_session h3_seedvr2_video_session;
int h3_seedvr2_video_check_params(const h3_seedvr2_video_params *params, char *reason, size_t reason_size);
h3_seedvr2_video_session *h3_seedvr2_video_session_open(h3_seedvr2 *model, int in_w, int in_h, int out_w, int out_h,
                                                        int total_frames /* 0 = unknown */,
                                                        const h3_seedvr2_video_params *params,
                                                        char *error, size_t error_size);
int h3_seedvr2_video_session_next_frames(const h3_seedvr2_video_session *session);
int h3_seedvr2_video_session_process(h3_seedvr2_video_session *session, const h3_vsr_image *in, int in_frames,
                                     int is_last, h3_vsr_image_out *out, int out_capacity, int *out_frames,
                                     h3_progress_callback on_progress, void *opaque,
                                     const volatile int *cancel_requested, char *error, size_t error_size);
int h3_seedvr2_video_session_pending_frames(const h3_seedvr2_video_session *session);
/* Windows run so far, the frames they processed, how many of those were padding, and the seconds spent. */
int h3_seedvr2_video_session_windows(const h3_seedvr2_video_session *session, int *frames_run,
                                     int *padded_frames, double *seconds);
void h3_seedvr2_video_session_close(h3_seedvr2_video_session *session);

/* GPU seconds so far and the peak live bytes. */
int h3_seedvr2_get_stats(const h3_seedvr2 *model, double *gpu_seconds,
                         uint64_t *peak_live_bytes);

/* Stage entry points on host F32 tensors in the reference's layouts, for
 * tests/test_seedvr2.c against mflux dumps: RGB [3, H, W] in [-1, 1] with
 * H, W multiples of 16; latents [C, h, w] with h = H / 8, w = W / 8. */
int h3_seedvr2_debug_encode(h3_seedvr2 *model, const float *rgb, int width, int height,
                            float *latent /* [16, h, w], scaled */,
                            char *error, size_t error_size);
/* The encoder stopped after stage 0 (conv_in), 1-4 (down block), 5 (mid),
 * 6 (the 32-channel output); out is CHW F32 of the reported shape. */
int h3_seedvr2_debug_encode_stage(h3_seedvr2 *model, const float *rgb, int width, int height, int stage,
                                  float *out, int *out_w, int *out_h, int *out_c, size_t capacity,
                                  char *error, size_t error_size);
int h3_seedvr2_debug_dit(h3_seedvr2 *model, const float *dit_in /* [33, h, w] */,
                         int latent_w, int latent_h, float *pred /* [16, h, w] */,
                         int dump_block /* -1 none: writes vid rows of that block to H3_SV2_DUMP */,
                         char *error, size_t error_size);
int h3_seedvr2_debug_decode(h3_seedvr2 *model, const float *latent /* [16, h, w], scaled */,
                            int latent_w, int latent_h, float *rgb /* [3, H, W] */,
                            char *error, size_t error_size);
/* The window partition of a tokens_w x tokens_h grid (mflux's rule): up to
 * `capacity` windows as (h_start, h_end, w_start, w_end); returns the count. */
int h3_seedvr2_debug_windows(int tokens_w, int tokens_h, int shifted, int *windows, int capacity);
/* The 3D partition of `frames` latent frames (the reference's window.py):
 * up to `capacity` windows as (t_start, t_end, h_start, h_end, w_start, w_end). */
int h3_seedvr2_debug_windows3(int frames, int tokens_w, int tokens_h, int shifted, int *windows, int capacity);
/* The transformer on a clip: dit_in [33, frames, h, w] (noise, latent, mask)
 * -> pred [16, frames, h, w]; dump_block as h3_seedvr2_debug_dit. */
/* A clip's noise for latent frames [first_frame, first_frame + frames),
 * [16, frames, h, w]: the same values for the same absolute frame whatever
 * the window; frame 0 equals a still image's noise for that seed. */
/* The memory model behind h3_seedvr2_video_check_requirements (tests). */
uint64_t sv2_video_peak_bytes_state(uint64_t resident_bytes, uint64_t dit_bytes, int nax_vae, int full_w, int full_h,
                                    int frames, int vae_split, int unload_dit, int state_bits);
uint64_t sv2_video_peak_bytes(uint64_t resident_bytes, uint64_t dit_bytes, int nax_vae, int full_w, int full_h,
                              int frames, int vae_split, int unload_dit);
int h3_seedvr2_debug_noise(uint64_t seed, int first_frame, int frames, int latent_w, int latent_h, float *out);
int h3_seedvr2_debug_dit_video(h3_seedvr2 *model, const float *dit_in, int frames, int latent_w, int latent_h,
                               float *pred, int dump_block, char *error, size_t error_size);
int h3_seedvr2_debug_color(h3_seedvr2 *model, const float *content, const float *style,
                           int width, int height, float *out, char *error, size_t error_size);
/* The temporal (video) VAE on host F32 tensors in the reference's
 * [C, T, H, W] layout. The VAE weights switch to their temporal form (the
 * still-image entry points switch them back). split 0 runs the whole clip
 * at once, > 0 the reference's causal slicing (encode 1 + split frames,
 * then split; decode 1 + split/4 latent frames, then split/4) with the
 * state carried, for the final output only. Encoder stages as
 * h3_seedvr2_debug_encode_stage (6 or more: the 32 channels, unscaled);
 * decoder stages 0 conv_in, 1 mid block, 2..5 up block 0..3, 6 or more RGB.
 * The latent in is scaled (mean * 0.9152). */
int h3_seedvr2_debug_encode_video(h3_seedvr2 *model, const float *rgb, int frames, int width, int height,
                                  int split, int stage, float *out, int *out_frames, int *out_w,
                                  int *out_h, int *out_c, size_t capacity, char *error, size_t error_size);
int h3_seedvr2_debug_decode_video(h3_seedvr2 *model, const float *latent, int latent_frames,
                                  int latent_w, int latent_h, int split, int stage, float *out,
                                  int *out_frames, int *out_w, int *out_h, int *out_c, size_t capacity,
                                  char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
#endif
