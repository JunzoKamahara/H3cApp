#ifndef H3_VIDEO_ENCODER_H
#define H3_VIDEO_ENCODER_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int time;
    int height;
    int width;
    /* Channel-major normalized F32: [24,time,height,width]. */
    float *values;
    h3_gpu_stats gpu_stats;
} h3_video_latent;

typedef void (*h3_video_encoder_progress)(int completed_tiles,
                                          int total_tiles, void *opaque);

/* Encode channel-major RGB [3,T,H,W] pixels in [0,1]. Spatial axes must be
 * multiples of 16. The released 256px/64px overlap tiling is preserved. */
int h3_video_vae_encode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *pixels, int frames, int height, int width,
                        h3_video_encoder_progress progress, void *progress_opaque,
                        h3_video_latent *output,
                        char *error, size_t error_size);
/* The same with each Conv3d run as 2D convolutions per time tap (about
 * 4.6x faster on an M5): equal up to the order of the F32 sums, so not
 * bit-identical to h3_video_vae_encode - kept for paths whose earlier
 * results needn't be reproduced exactly (continuing an imported video). */
int h3_video_vae_encode_fast(const char *weight_directory,
                             const char *shader_source_path,
                             const float *pixels, int frames, int height, int width,
                             h3_video_encoder_progress progress, void *progress_opaque,
                             h3_video_latent *output,
                             char *error, size_t error_size);
/* Memory over time (see h3_video_encoder.c). Both entry points above
 * encode a clip in one pass when h3_video_encoder_peak_bytes() fits
 * h3_video_encoder_default_budget() (a third of the RAM, or
 * H3_VAE_ENCODE_BUDGET_BYTES), and otherwise in chunks over time that fit
 * it; the chunks carry the causal convolutions' history, so the latent is
 * the same as in one pass (up to the order of F32 sums, if at all). */
uint64_t h3_video_encoder_peak_bytes(int frames, int height, int width);
uint64_t h3_video_encoder_chunked_peak_bytes(int chunk_frames, int height,
                                             int width);
uint64_t h3_video_encoder_default_budget(void);
/* chunk_frames: 0 for one pass, else a multiple of 4. */
int h3_video_encoder_plan(int frames, int height, int width,
                          uint64_t budget_bytes, int *chunk_frames);
/* An explicit plan: chunk_frames 0 is one pass; fast as in _encode_fast. */
int h3_video_vae_encode_chunked(const char *weight_directory,
                                const char *shader_source_path,
                                const float *pixels, int frames, int height,
                                int width, int chunk_frames, int fast,
                                h3_video_encoder_progress progress,
                                void *progress_opaque, h3_video_latent *output,
                                char *error, size_t error_size);
void h3_video_latent_free(h3_video_latent *latent);

#endif
