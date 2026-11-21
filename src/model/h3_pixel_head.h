#ifndef H3_PIXEL_HEAD_H
#define H3_PIXEL_HEAD_H

/* pi-H3's pixel head (DWD, "decoding without a decoder"): the video frames
 * straight from the DiT's last evaluation, in place of the last Euler step
 * and the video VAE decoder.
 *
 * The released head (Lakonik/pi-H3, LakonLab's SolH3DWDHead) is the last 8
 * of the Sol-H3-VAE decoder's 24 layers, driven by the DiT's last block:
 *
 *   rows of the last block's output for the generated video, [t*h*w, 5376]
 *     -> LayerNorm, * (1 + scale) + shift   the head's own AdaLN at the last
 *                                           step's time (modulation below)
 *     -> feature_expand                     a row becomes 2 x 2 tokens of
 *                                           2048, 16 x 16 pixels each
 *     -> 8 layers                           the video VAE decoder's layer
 *        with windows of 5 latent steps x 16 x 16 tokens (odd layers
 *        shifted by 2 steps and 8 tokens) and, for every 5 latent steps, 5
 *        global tokens (4 registers and a zero) that see all of them
 *     -> LayerNorm, projection              4 frames a token; the first
 *        latent step of every 5 has its own projection, 1 frame
 *     -> RGB = value * std + mean, clamped
 *
 * The whole clip is one token grid: nothing is decoded in overlapping
 * chunks. Activations stay resident (about 12 KiB a token: 0.12 GiB at
 * 384 x 384, 56 frames, 1.8 GiB at 1344 x 768, 124 frames); the linears run
 * a few thousand rows at a time. */

#include "h3_video_vae.h"

#include <stddef.h>
#include <stdint.h>

#define H3_PIXEL_HEAD_FEATURE 5376

/* Frames a clip of latent_t latent steps has: 1 for the first step of every
 * five, 4 for the others - h3_frame_per_token's rule. */
int h3_pixel_head_frames(int latent_t);

/* adapter_path: the pi-H3 adapter (its single-file ComfyUI layout).
 * rows: BF16 [latent_t * token_h * token_w, 5376], the last DiT block's
 * output for the generated video rows, in (t, y, x) order.
 * modulation: F32 [2 * 5376] - the head's AdaLN scale, then shift, for the
 * last step's time (h3_dit_pixel_head_input).
 * Frames go to `callback` one latent step at a time (1 or 4 frames, F32
 * RGB in [0, 1], 32 * token_h by 32 * token_w), in order. progress counts
 * the 8 layers. */
int h3_pixel_head_decode(const char *adapter_path,
                         const char *shader_source_path,
                         const uint16_t *rows, const float *modulation,
                         int latent_t, int token_h, int token_w,
                         h3_video_chunk_callback callback,
                         void *callback_opaque,
                         h3_video_vae_progress progress,
                         void *progress_opaque,
                         char *error, size_t error_size);

#endif
