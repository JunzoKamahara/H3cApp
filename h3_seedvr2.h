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
