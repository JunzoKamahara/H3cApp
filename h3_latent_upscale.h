/* LBH-123-AI's 3D-convolution H3 latent upscaler (LatentResizer3D from
 * Comfyui_Minimax_h3_latent_Upscaler, MIT; the network Open-H3-Regenerate-2K
 * uses as released) on the engine's F32 channels-last conv3d path.
 *
 * It maps a DiT-space latent z [24, T, H, W] (what a .h3lt holds, the VAE's
 * latents_mean/std already applied) to the same kind of latent on a larger
 * spatial grid. Convention checked against NVIDIA's pipeline and ComfyUI
 * (2026-10-31): the node normalizes its input once more with the VAE's own
 * 24-channel mean/std - the constants in the node are identical to the
 * VAE config's - runs the network, and de-normalizes the output; the
 * network was trained on exactly that input, so this port does the same.
 * See UPSCALE_PLAN §11.3. */
#ifndef H3_LATENT_UPSCALE_H
#define H3_LATENT_UPSCALE_H

#include <stddef.h>
#include "h3.h"

typedef struct h3_latent_upscaler h3_latent_upscaler;

/* weights_dir holds the single safetensors file (BF16, 322 tensors). */
h3_latent_upscaler *h3_latent_upscaler_open(const char *weights_dir,
                                            const char *shader_source_path,
                                            char *error, size_t error_size);
void h3_latent_upscaler_free(h3_latent_upscaler *upscaler);

/* latent [24, t, h, w] F32 -> out [24, t, out_h, out_w] (out_h >= h,
 * out_w >= w; the time axis is kept). Progress phase "latent upscale". */
int h3_latent_upscale(h3_latent_upscaler *upscaler, const float *latent,
                      int t, int h, int w, int out_h, int out_w, float *out,
                      h3_progress_callback on_progress, void *opaque,
                      char *error, size_t error_size);

/* The same on a clip latent file (.h3lt, see h3_params.save_latent_path):
 * the video latent is upscaled, the audio latent and frame count copied. */
int h3_latent_upscale_file(h3_latent_upscaler *upscaler, const char *in_path,
                           const char *out_path, int out_h, int out_w,
                           h3_progress_callback on_progress, void *opaque,
                           char *error, size_t error_size);

#endif
