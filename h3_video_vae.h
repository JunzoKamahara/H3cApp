#ifndef H3_VIDEO_VAE_H
#define H3_VIDEO_VAE_H

#include "h3_gpu.h"

#include <stddef.h>

typedef struct {
    int frames;
    int height;
    int width;
    /* Frame-major, row-major interleaved RGB F32 in [0,1]. */
    float *rgb;
    h3_gpu_stats gpu_stats;
} h3_video_frames;

typedef void (*h3_video_vae_progress)(int completed_blocks, int total_blocks,
                                      void *opaque);

typedef struct h3_video_vae_decoder h3_video_vae_decoder;

/* Resident tiled decoder used by live denoising previews. It decodes one
 * representative middle frame per preview call and can then produce the final
 * complete video without loading the 9.7 GiB weight set again. */
h3_video_vae_decoder *h3_video_vae_decoder_load(
                        const char *weight_directory,
                        const char *shader_source_path,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        char *error, size_t error_size);
int h3_video_vae_decoder_preview(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frames *output, int *output_frame_index,
                        char *error, size_t error_size);
int h3_video_vae_decoder_decode(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frames *output,
                        char *error, size_t error_size);

/* Pixel size a resident decoder will produce, before any chunk has been
 * decoded - needed by callers of the streamed decode below, which never
 * hands back a full h3_video_frames. */
void h3_video_vae_decoder_pixel_size(const h3_video_vae_decoder *decoder,
                        int *height, int *width);

/* rgb is frame-major, row-major interleaved RGB F32 in [0,1], owned by the
 * decoder and only valid for the duration of the call - copy it if you need
 * it past that. Returning 0 aborts the decode (mirrors *_progress callbacks
 * elsewhere: it's how a caller signals "stop", e.g. cancellation or a
 * downstream write failure), and the decode reports that through its own
 * ok/error return same as any other failure. */
typedef int (*h3_video_chunk_callback)(void *opaque, const float *rgb,
                        int frame_count, int height, int width,
                        char *error, size_t error_size);

/* Same decode as h3_video_vae_decoder_decode (same ~17-frame temporal
 * chunks, same overlap/blend rule), but hands each chunk to `callback` as
 * soon as it's ready instead of retaining the whole video in one buffer.
 * Peak host memory for the decoded video drops from the full output (e.g.
 * ~4.3 GiB at 512x512/362 frames) to about one chunk (~50 MiB at that
 * resolution), and it lets a caller start encoding/upscaling chunk N while
 * chunk N+1 is still being decoded. */
int h3_video_vae_decoder_decode_streamed(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_chunk_callback callback, void *callback_opaque,
                        h3_gpu_stats *out_stats,
                        char *error, size_t error_size);
void h3_video_vae_decoder_free(h3_video_vae_decoder *decoder);

/* Decode aligned H3 temporal chunks, using the released overlap/blend rules in
 * time and 256-pixel overlapping tiles in space. The two-token mode remains
 * available only for the diagnostic MLX fixture. */
/* A stop request (h3_params.cancel_requested) the decoders on the calling
 * thread check between chunks and tiles; NULL turns it off. */
void h3_video_vae_set_cancel(const volatile int *cancel_requested);

int h3_video_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent, int latent_time,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        h3_video_frames *output,
                        char *error, size_t error_size);
void h3_video_frames_free(h3_video_frames *frames);

#endif
