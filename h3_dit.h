#ifndef H3_DIT_H
#define H3_DIT_H

#include "h3.h"
#include "h3_gpu.h"
#include "h3_host.h"
#include "h3_text_encoder.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_dit h3_dit;

typedef void (*h3_dit_progress)(const char *phase, int completed, int total,
                                void *opaque);

typedef int (*h3_dit_preview)(int completed_steps, int total_steps,
                              const float *video_latent,
                              size_t video_elements, void *opaque);

/* Seed-independent preparation kept between loads: the token refiner's
 * output (a few MB) and the AdaLN schedule (time rows x 50 blocks, a few
 * hundred MB). A load reuses a part only when everything it was computed
 * from matches exactly - the refiner: model, input embedding, row count and
 * LoRA files/strengths; AdaLN: model, sigma schedule and which conditions
 * are present - and stores what it computed for the next one. NULL, or a
 * part switched off, computes it as before. */
typedef struct h3_dit_prep h3_dit_prep;
h3_dit_prep *h3_dit_prep_new(void);
void h3_dit_prep_free(h3_dit_prep *prep);
/* Which parts to keep; switching one off frees it. */
void h3_dit_prep_set(h3_dit_prep *prep, int refined_text, int adaln);
/* Drops what is held, keeping which parts are on. */
void h3_dit_prep_clear(h3_dit_prep *prep);
/* Host bytes held by the refiner part (adaln 0) or the AdaLN part (1). */
size_t h3_dit_prep_bytes(const h3_dit_prep *prep, int adaln);

/* Load a text-only FL2VA transformer. Text refinement and AdaLN precomputation
 * happen before the persistent core is loaded. SSD streaming retains only the
 * small block norms and two alternating BF16 matrix slots. */
h3_dit *h3_dit_load_t2va(const char *weight_directory,
                         const char *shader_source_path,
                         const h3_text_embedding *text,
                         const h3_layout *layout,
                         const h3_sigma_schedule *sigmas,
                         unsigned active_blocks,
                         unsigned core_reuse_interval,
                         int token_reduction,
                         int ssd_streaming,
                         float spatial_rope_scale,
                         int use_slower_bf16_mlp,
                         int use_slower_bf16_qkv,
                         int use_slower_bf16_attention_output,
                         int use_slower_row_major_attention_output,
                         int use_slower_unfused_int8_inputs,
                         int use_slower_unfused_qkv_rope,
                         int use_slower_scalar_qkv_rms,
                         int use_slower_uncached_int8_scales,
                         int use_slower_dynamic_fc1_k,
                         int use_slower_grouped_quantizer,
                         int use_int8_row_fc2,
                         const h3_lora *loras, size_t lora_count,
                         h3_dit_prep *prep,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size);

/* Load FL2VA/Ref2VA packing. Condition inputs are already patchified F32 row
 * sources: visual rows have width 96 and audio rows width 32. Their element
 * counts must exactly match layout.img_cond_rows/audio_cond_rows. */
h3_dit *h3_dit_load_conditioned(
                         const char *weight_directory,
                         const char *shader_source_path,
                         const h3_text_embedding *text,
                         const h3_layout *layout,
                         const h3_sigma_schedule *sigmas,
                         unsigned active_blocks,
                         unsigned core_reuse_interval,
                         int token_reduction,
                         int ssd_streaming,
                         float spatial_rope_scale,
                         int use_slower_bf16_mlp,
                         int use_slower_bf16_qkv,
                         int use_slower_bf16_attention_output,
                         int use_slower_row_major_attention_output,
                         int use_slower_unfused_int8_inputs,
                         int use_slower_unfused_qkv_rope,
                         int use_slower_scalar_qkv_rms,
                         int use_slower_uncached_int8_scales,
                         int use_slower_dynamic_fc1_k,
                         int use_slower_grouped_quantizer,
                         int use_int8_row_fc2,
                         const h3_lora *loras, size_t lora_count,
                         h3_dit_prep *prep,
                         const float *condition_video_rows,
                         size_t condition_video_elements,
                         const float *condition_audio_rows,
                         size_t condition_audio_elements,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size);
void h3_dit_free(h3_dit *dit);
/* Releases attention-backend scratch that only the denoise loop uses (see
 * h3_gpu_ccv_release_scratch); call once denoising has finished. */
void h3_dit_release_backend_scratch(h3_dit *dit);

/* Which implementation the DiT's full attention uses. Chosen once per
 * generation by the caller (h3_generate) and set before denoising, also on
 * a reused prepared DiT, so nothing carries over from an earlier run. */
typedef enum {
    H3_DIT_ATTENTION_MPS = 0,         /* production MPSGraph SDPA */
    H3_DIT_ATTENTION_CCV_INT8,        /* ccv int8 via the FP16 row-major bridge */
    H3_DIT_ATTENTION_CCV_FP16,        /* ccv non-quantized FP16 (diagnostic) */
    H3_DIT_ATTENTION_CCV_INT8_DIRECT  /* ccv int8 on the BF16 head-major Q/K/V */
} h3_dit_attention_mode;
void h3_dit_set_attention_mode(h3_dit *dit, h3_dit_attention_mode mode);
/* Step rule for the following denoise calls: H3_SAMPLER_EULER (default)
 * or H3_SAMPLER_RENOISE, whose fresh noise is derived from seed. */
void h3_dit_set_sampler(h3_dit *dit, int sampler, uint64_t seed);
/* Forced audio for the following denoise calls: hold the audio stream on
 * this clean latent ([32,2,T], h3_dit_forward's audio layout,
 * h3_dit_audio_elements values) so that only the picture is generated,
 * conditioned on that sound at every step. With an all-zero audio sigma
 * schedule the audio is the clean latent throughout; otherwise it follows
 * the latent's noise trajectory. The audio passed to the denoise call is
 * its initial noise, as usual. NULL turns it off. */
/* A stop request the denoising loop checks before every step and DiT
 * block (h3_params.cancel_requested); NULL turns the check off. */
void h3_dit_set_cancel(h3_dit *dit, const volatile int *cancel_requested);

int h3_dit_set_forced_audio(h3_dit *dit, const float *clean_latent,
                            size_t elements, char *error, size_t error_size);

/* Reset mutable sampler state and replace seed-dependent condition rows before
 * reusing an otherwise identical prepared transformer. */
int h3_dit_reset_run(h3_dit *dit,
                     const float *condition_video_rows,
                     size_t condition_video_elements,
                     const float *condition_audio_rows,
                     size_t condition_audio_elements,
                     char *error, size_t error_size);

size_t h3_dit_video_elements(const h3_dit *dit);
size_t h3_dit_audio_elements(const h3_dit *dit);

/* One raw data-ward velocity evaluation. Input/output video layout is
 * [24,T,H,W], audio is [32,2,T], all F32 on the host boundary. */
int h3_dit_forward(h3_dit *dit, int step,
                   const float *video_latent, const float *audio_latent,
                   float *video_velocity, float *audio_velocity,
                   char *error, size_t error_size);

/* Run every configured RES multistep update in place. Solver state remains
 * F32 between transformer evaluations, matching the released MLX path. */
int h3_dit_denoise(h3_dit *dit, float *video_latent, float *audio_latent,
                   h3_dit_progress progress, void *progress_opaque,
                   char *error, size_t error_size);

/* Current serving sampler: independent video/audio shifted Euler grids. */
int h3_dit_denoise_euler(h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size);

/* Preview variant. The callback runs after every Euler transition with the
 * current channel-major F32 video latent. It is deliberately opt-in because it
 * introduces a synchronization point after each step. */
int h3_dit_denoise_euler_preview(
                         h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3_dit_progress progress, void *progress_opaque,
                         h3_dit_preview preview, void *preview_opaque,
                         char *error, size_t error_size);

/* Build the velocity-evaluation mask used by the serving sampler. Returns the
 * evaluation count, or -1 for invalid arguments. Public internally so the
 * quality-tuned aggressive schedule remains pinned by cheap host tests. */
int h3_dit_reuse_schedule(int steps, int reuse_interval, uint8_t *selected,
                          size_t selected_count);

int h3_dit_get_gpu_stats(const h3_dit *dit, h3_gpu_stats *stats);

/* Exact row-order conversions, public internally so they can be pinned by
 * cheap tests independently of the 62 GiB checkpoint. */
int h3_dit_patchify_video(const float *latent, int channels, int time,
                          int height, int width, float *rows,
                          size_t row_elements);
int h3_dit_unpatchify_video(const float *rows, int channels, int time,
                            int height, int width, float *latent,
                            size_t latent_elements);
int h3_dit_pack_audio(const float *latent, int channels, int time,
                      float *rows, size_t row_elements);
int h3_dit_unpack_audio(const float *rows, int channels, int time,
                        float *latent, size_t latent_elements);

#endif
