#ifndef H3_IMAGE_H
#define H3_IMAGE_H

#include "h3.h"

#include <stddef.h>
#include <stdint.h>

/* Still images with MiniMax H3 Image (plans/IMAGE_PLAN.ja.md). Apart from
 * the video API (h3.h): another model, another sampler - the checkpoint's
 * own six steps, one output head per step, no audio - and one picture as
 * the result: from a prompt alone, or from a prompt and reference images
 * (editing - the prompt says what to change, calling the references
 * <Picture 1>, <Picture 2>, ... in the order they are given).
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

/* phase: "vision" (the references through the vision tower), "text
 * encoder" (layers), "reference" (the references through the VAE encoder),
 * "image model" (loading), "denoise" (steps) or "decode" (tiles). */
typedef void (*h3_image_progress)(const char *phase, int completed, int total,
                                  void *opaque);

#define H3_IMAGE_MAX_STEPS 16
/* As many reference images as the released pipeline takes. */
#define H3_IMAGE_MAX_REFERENCES 16
/* A reference's longer side may be this many times its shorter one (the
 * vision encoder's budget in the released pipeline). */
#define H3_IMAGE_REFERENCE_MAX_ASPECT 12.25

typedef struct {
    /* An image file (whatever ImageIO decodes), used upright. */
    const char *path;
    /* Optional: the image's own latent, as the app keeps it beside an image
     * it generated (safetensors "latent" F32 [24, h, w], normalized). Taken
     * instead of encoding the picture when [h, w] is the grid the reference
     * is encoded at (h3_image_reference_canvas); otherwise ignored. */
    const char *latent_path;
} h3_image_reference;
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
    /* Editing: reference images, in the order the prompt numbers them
     * (reference_count 0: text to image). Each is seen twice - by the text
     * encoder's vision tower, at a short side of 1024 pixels, and by the
     * DiT as a latent, at the size h3_image_reference_canvas gives. */
    const h3_image_reference *references;
    int reference_count;
    /* The short side a reference's latent is encoded at is the picture's
     * own, kept within 1024 and this (0: 2048, as the released pipeline). */
    int reference_max_short_edge;
    /* h3_image_plan only: how many text-encoder layers the generation will
     * read ahead (0: what H3_QWEN_PREFETCH_DEPTH or the default gives now). */
    int plan_text_prefetch_depth;
    /* Optional: called after every denoising step but the last with the
     * model's estimate of the finished image (x0 = x_t + sigma * v):
     * channel-major F32 [24, latent_height, latent_width] in the DiT's
     * normalized latent space, valid only during the call; `step` is
     * 1-based. For a caller-side preview; no VAE runs. Gets
     * progress_opaque. */
    void (*latent_preview)(const float *latent, int latent_width,
                           int latent_height, int step, int steps,
                           void *opaque);
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
    /* Editing: the size each reference's latent was encoded at, whether
     * its own latent was taken instead, the rows of the text sequence
     * (prompt, labels and vision tokens) and the time the references took
     * (vision tower and VAE encoder; text_seconds is the language layers). */
    int reference_count;
    int reference_width[H3_IMAGE_MAX_REFERENCES];
    int reference_height[H3_IMAGE_MAX_REFERENCES];
    int reference_from_latent[H3_IMAGE_MAX_REFERENCES];
    int text_rows;
    double reference_seconds;
} h3_image_result;

/* The sizes a width x height reference is used at: encode_* for its latent
 * (short side within 1024 and max_short_edge - 0 is 2048 - and at most
 * 2048 x 2048 pixels in area), vision_* for the text encoder's vision tower
 * (short side 1024). All multiples of 32. Fails (0) for a picture whose
 * sides differ by more than H3_IMAGE_REFERENCE_MAX_ASPECT. */
int h3_image_reference_canvas(int width, int height, int max_short_edge,
                              int *encode_width, int *encode_height,
                              int *vision_width, int *vision_height);

/* P-H3-Image's sizes (diffusers' PH3ImageEditResizeStep): encode_* keeps
 * the picture's size when its area is within 1 to 8 megapixels and is
 * scaled to the nearer end otherwise; vision_* is about 1,000,000 pixels
 * (kept when within 20%). Same aspect, sides multiples of 32, and the same
 * aspect limit as above. */
int h3_image_reference_canvas_ph3(int width, int height,
                                  int *encode_width, int *encode_height,
                                  int *vision_width, int *vision_height);

/* Checks that the folder holds an image model - a still-image DiT (MiniMax
 * H3 Image), or MiniMax H3's audio-video DiT beside a text encoder with a
 * text connector (P-H3-Image) - and remembers where its parts are; nothing
 * is loaded until an image is generated. */
h3_image_ctx *h3_image_load_dir(const char *model_directory,
                                const char *shader_source_path,
                                char *error, size_t error_size);
void h3_image_free(h3_image_ctx *ctx);

/* Which image model a folder holds. */
typedef enum {
    /* MiniMax H3 Image: six steps, edits with references. */
    H3_IMAGE_MODEL_H3 = 0,
    /* P-H3-Image (Pruna AI): MiniMax H3's DiT with its image LoRA merged,
     * a Qwen3-VL-4B text encoder and connector; eight steps; edits with up
     * to H3_IMAGE_PH3_MAX_REFERENCES pictures; made for about 8
     * megapixels. */
    H3_IMAGE_MODEL_PH3 = 1
} h3_image_model;

typedef struct {
    h3_image_model model;
    const char *name;      /* "MiniMax H3 Image" or "P-H3-Image" */
    int steps;
    int can_edit;          /* takes reference images */
    int max_references;
} h3_image_info;

#define H3_IMAGE_PH3_MAX_REFERENCES 8

void h3_image_get_info(const h3_image_ctx *ctx, h3_image_info *info);

/* What h3_image_generate(ctx, params) will take, worked out without
 * loading any weights (it reads the references' sizes and loads the
 * tokenizer): the text encoder's and the DiT's tokens as the generation
 * will count them, and each stage's memory on this Mac, by the video
 * engine's own model (h3_plan, h3.h; the reference fields are the edit's
 * pictures). Returns 1 with *plan filled for every verdict, or 0 when the
 * request is invalid or a reference cannot be read (the reason in
 * plan->message).
 *
 * h3_image_generate runs this first and fails with plan->message when the
 * verdict is H3_PLAN_TOO_LARGE; H3_PREFLIGHT=0 in the environment skips
 * that check. */
int h3_image_plan(const h3_image_ctx *ctx, const h3_image_params *params,
                  h3_plan *plan);

/* Generates one image. Everything it loads is released again before it
 * returns. On success the caller frees the result with h3_image_result_free. */
int h3_image_generate(h3_image_ctx *ctx, const h3_image_params *params,
                      h3_image_result *result, char *error, size_t error_size);
void h3_image_result_free(h3_image_result *result);

#endif
