#ifndef H3_IMAGE_H
#define H3_IMAGE_H

#include <stddef.h>
#include <stdint.h>

/* Still images with MiniMax H3 Image (plans/IMAGE_PLAN.ja.md). Apart from
 * the video API (h3.h): another model, another sampler - the checkpoint's
 * own six steps, one output head per step, no audio - and one picture as
 * the result. Text to image only, for now.
 *
 * A model folder holds
 *
 *   transformer/    the image DiT (safetensors; the ComfyUI int8 + ConvRot
 *                   re-publication is what has been run)
 *   vae/            the image VAE (safetensors) and its config.json
 *   text_encoder/   the same Qwen3-VL text encoder a video model uses
 *   tokenizer/      tokenizer.json
 *
 * The DiT's weights are read from disk block by block while it runs (two
 * blocks resident at a time), so memory stays in the low gigabytes whatever
 * the size of the checkpoint. */

typedef struct h3_image_ctx h3_image_ctx;

/* phase: "text encoder" (layers), "image model" (loading), "denoise"
 * (steps) or "decode" (tiles). */
typedef void (*h3_image_progress)(const char *phase, int completed, int total,
                                  void *opaque);

#define H3_IMAGE_MAX_STEPS 16
/* Prompts are cut to this many tokens, as the released pipeline does. */
#define H3_IMAGE_MAX_PROMPT_TOKENS 4096

typedef struct {
    const char *prompt;
    /* Pixels; both multiples of 32, from 256 to 4096. */
    int width;
    int height;
    uint64_t seed;
    /* A stop request, checked between the stages, before every DiT block
     * and between VAE tiles (NULL: none). A stopped generation fails with
     * the error "generation cancelled". */
    const volatile int *cancel_requested;
    h3_image_progress progress;
    void *progress_opaque;
    /* Testing: the initial noise, channel-major F32 [24, height/16,
     * width/16], instead of the one drawn from `seed`. */
    const float *initial_noise;
} h3_image_params;

typedef struct {
    int width;
    int height;
    /* Row-major interleaved RGB8. */
    uint8_t *rgb;
    /* The image as the DiT left it: channel-major F32 [24, latent_height,
     * latent_width], normalized (the VAE's latents_mean/std not applied). */
    float *latent;
    int latent_width;
    int latent_height;
    uint64_t seed;
    int prompt_tokens;
    int steps;
    float sigmas[H3_IMAGE_MAX_STEPS + 1];
    double text_seconds;
    double denoise_seconds;
    double decode_seconds;
} h3_image_result;

/* Checks that the folder holds an image model (its DiT is a still-image
 * checkpoint) and remembers where its parts are; nothing is loaded until an
 * image is generated. */
h3_image_ctx *h3_image_load_dir(const char *model_directory,
                                const char *shader_source_path,
                                char *error, size_t error_size);
void h3_image_free(h3_image_ctx *ctx);

/* Generates one image. Everything it loads is released again before it
 * returns. On success the caller frees the result with h3_image_result_free. */
int h3_image_generate(h3_image_ctx *ctx, const h3_image_params *params,
                      h3_image_result *result, char *error, size_t error_size);
void h3_image_result_free(h3_image_result *result);

#endif
