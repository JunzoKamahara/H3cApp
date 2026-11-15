#ifndef H3_IMAGE_VAE_H
#define H3_IMAGE_VAE_H

#include "h3_gpu.h"

#include <stddef.h>

/* The still-image VAE decoder of MiniMax H3 Image: a 2D vision transformer
 * (24 blocks, 1024 wide in the released weights) over 16x16 latent tiles,
 * one image at a time - no time axis and no temporal chunks, unlike the
 * video decoder (h3_video_vae.c), which it otherwise resembles block for
 * block. Its sizes are read from the weights, not fixed here. */

typedef struct {
    int width;
    int height;
    /* Row-major interleaved RGB F32 in [0,1]. */
    float *rgb;
    h3_gpu_stats gpu_stats;
} h3_image_pixels;

typedef void (*h3_image_vae_progress)(int completed_tiles, int total_tiles,
                                      void *opaque);

/* Decodes a normalized latent [24, latent_height, latent_width] (channel-
 * major, as the image DiT leaves it) into latent_height*16 x latent_width*16
 * pixels: 256-pixel tiles overlapping by at least 64 pixels, cross-faded,
 * as the released decoder does. `weight_directory` holds the VAE's
 * safetensors and, in it or one level up, the config.json with
 * latents_mean / latents_std. A set `cancel_requested` (may be NULL) stops
 * between tiles with the error "generation cancelled". */
int h3_image_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent,
                        int latent_height, int latent_width,
                        const volatile int *cancel_requested,
                        h3_image_vae_progress progress, void *progress_opaque,
                        h3_image_pixels *output,
                        char *error, size_t error_size);
void h3_image_pixels_free(h3_image_pixels *pixels);

#endif
