/* What the files of the generation flow share among themselves (h3.c and h3_*.c beside it): file-scope helpers and state. Not an API; include/h3.h is. */
#ifndef H3_PRIVATE_H
#define H3_PRIVATE_H

#include "h3_options.h"
#include "h3_internal.h"
#include "h3_audio_vae.h"
#include "h3_gpu.h"
#include "h3_host.h"
#include "h3_av_reader.h"
#include "h3_av_writer.h"
#include "h3_dit.h"
#include "h3_hyperflow.h"
#include "h3_model_family.h"
#include "h3_lora.h"
#include "h3_memory.h"
#include "h3_metal.h"
#include "h3_multimodal.h"
#include "h3_plan.h"
#include "h3_refmod.h"
#include "h3_safetensors.h"
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"
#include "h3_video_encoder.h"
#include "h3_video_vae.h"
#include "h3_vision_encoder.h"
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct {
    char *text;
    size_t length;
    size_t capacity;
} h3_key;

/* What a continuation pins, worked out from the earlier clip's latent file
 * header and the params: one condition step per pinned latent step, at its
 * own pixel frame, and an audio window that ends where those frames end. */
typedef struct {
    int active;
    int steps;              /* pinned latent steps */
    int keyframes[17];      /* their pixel frames in this clip */
    int audio_t;            /* pinned audio steps (0: none) */
    double audio_start;     /* first one's place, in audio steps */
    int file_t, file_h, file_w, file_audio_t, file_frames;
} h3_continuation;

typedef struct {
    h3_ctx *ctx;
    const h3_params *params;
    int cancelled;
} h3_generation_progress;

/* The RGB frames handed to the encoder, hashed as one stream in frame order,
 * so the value does not depend on how the decoder split the video into
 * chunks (streamed) or not (monolithic). */
typedef struct {
    uint64_t hash;
    size_t frames;
} h3_debug_video_hash;

/* h3_video_vae_decoder_decode_streamed callback: converts one temporal
 * chunk to u8, upscales it to the requested output size if needed,
 * delivers it through on_frame, and (if an output path was given) writes it
 * to the AVFoundation writer - all before the next chunk is decoded, so only
 * one chunk's worth of RGB is ever resident instead of the whole video. */
typedef struct {
    h3_av_writer *writer;
    h3_frame_callback on_frame;
    void *callback_opaque;
    h3_generation_progress *progress;
    int output_width;
    int output_height;
    int need_resize;
    int total_frames;
    int emitted;
    h3_debug_video_hash rgb_hash;
} h3_stream_encode_ctx;

/* File-scope functions and state used by more than one of the files. */
void h3_conditioning_cache_clear(h3_ctx *ctx);
int h3_cache_target(const h3_ctx *ctx, unsigned target);
int h3_key_append(h3_key *key, const char *format, ...);
int h3_latent_file_write(const char *path, const float *video,
                                int latent_t, int latent_h, int latent_w,
                                const float *audio, int audio_t, int frames,
                                char *error, size_t error_size);
int h3_continuation_plan(const h3_params *params, int latent_h,
                                int latent_w, h3_temporal_shape temporal,
                                h3_continuation *plan,
                                char *error, size_t error_size);
int h3_continuation_rows(const h3_params *params,
                                const h3_continuation *plan,
                                float **video_rows, size_t *video_elements,
                                float **audio_rows, size_t *audio_elements,
                                char *error, size_t error_size);
int h3_initial_latent_check(const h3_params *params, int latent_h,
                                   int latent_w, h3_temporal_shape temporal,
                                   char *error, size_t error_size);
int h3_initial_latent_apply(const char *path, float *video,
                                   size_t video_count, float sigma0,
                                   size_t audio_count, float **audio_out,
                                   char *error, size_t error_size);
int h3_join_rows(float **rows, size_t *elements,
                        const float *more, size_t count, int first);
char *h3_conditioning_key(const char *prompt, const h3_params *params,
                                 int render_width, int render_height,
                                 int ref2va);
float h3_audio_sigma_shift(const h3_params *params);
int h3_forced_audio_clean(const h3_params *params);
char *h3_prepared_key(const char *conditioning,
                             const h3_params *params,
                             int render_width, int render_height);
int h3_conditioning_cache_store(
        h3_ctx *ctx, const char *key, const h3_text_embedding *text,
        const float *video, size_t video_elements,
        const float *audio, size_t audio_elements,
        const h3_layout_ref *references, size_t reference_count,
        int conditioned);
int h3_conditioning_cache_load(
        const h3_ctx *ctx, h3_text_embedding *text,
        float **video, size_t *video_elements,
        float **audio, size_t *audio_elements,
        h3_layout_ref **references, size_t *reference_count,
        int *conditioned);
int h3_augment_conditions(const h3_params *params, int ref2va,
                                 int render_width, int render_height,
                                 const h3_layout_ref *references,
                                 size_t keyframe_count,
                                 size_t keyframe_audio_elements,
                                 float *video, size_t video_elements,
                                 float *audio, size_t audio_elements);
void h3_core_fail(char *error, size_t error_size, const char *format, ...);
char *h3_path(const char *root, const char *relative);
int h3_dir_has_gguf(const char *root, const char *component);
int h3_valid_params(h3_ctx *ctx, const h3_params *params);
void h3_debug_hash_frames(h3_debug_video_hash *state,
                                 const uint8_t *rgb, size_t frames,
                                 size_t frame_bytes);
void h3_progress_emit(h3_generation_progress *state, const char *phase,
                             int completed, int total);
uint8_t *h3_rgb_f32_to_u8(const float *rgb, size_t count);
float *h3_extract_vision_pair(const float *pixels, int frames,
                                     int height, int width,
                                     int first, int second);
int h3_stream_encode_chunk(void *opaque, const float *rgb,
                                  int frame_count, int height, int width,
                                  char *error, size_t error_size);
int h3_pooled_vision_encode(const char *weight_directory,
                                   const float *pixels, int frames,
                                   int height, int width,
                                   h3_vision_progress progress, void *opaque,
                                   h3_vision_output *output,
                                   char *error, size_t error_size);
int h3_pooled_video_encode(const char *vae_path, const float *pixels,
                                  int frames, int height, int width,
                                  h3_video_encoder_progress progress,
                                  void *opaque, h3_video_latent *latent,
                                  char *error, size_t error_size);
int h3_pooled_audio_encode(const char *audio_vae_path,
                                  const float *pcm, int samples,
                                  h3_audio_vae_progress progress, void *opaque,
                                  h3_audio_latent *latent,
                                  char *error, size_t error_size);
int h3_refmod_reference(const h3_reference *reference, size_t index,
                               h3_layout_ref *layout, h3_refmod_member *member,
                               char *error, size_t error_size);
int h3_reference_limits(const h3_params *params,
                               const h3_layout_ref *layout_references,
                               char *error, size_t error_size);
int h3_reference_canvas(const h3_ctx *ctx, const h3_params *params,
                               const h3_reference *reference,
                               int render_width, int render_height,
                               int source_width, int source_height,
                               int *width, int *height);

#endif
