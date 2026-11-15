#include "h3_image.h"

#include "h3_dit.h"
#include "h3_dit_schedule.h"
#include "h3_host.h"
#include "h3_image_vae.h"
#include "h3_options.h"
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"
#include "h3_weights.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

enum { LATENT_CHANNELS = 24, MIN_SIDE = 256, MAX_SIDE = 4096 };

/* The released checkpoint's own six-step grid, for one that does not
 * carry it in its metadata. */
static const float default_sigmas[] = {
    1.0f, 0.9675236940383911f, 0.8875485062599182f, 0.75f,
    0.5327739119529724f, 0.23200887441635132f, 0.0f
};

struct h3_image_ctx {
    char *transformer;
    char *vae;
    char *text_encoder;
    char *tokenizer;
    char *shaders;
    int steps;
    float sigmas[H3_IMAGE_MAX_STEPS + 1];
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static double now(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

static char *join(const char *directory, const char *name) {
    size_t size = strlen(directory) + strlen(name) + 2;
    char *path = malloc(size);
    if (path) snprintf(path, size, "%s/%s", directory, name);
    return path;
}

static int exists(const char *path, int directory) {
    struct stat info;
    return path && stat(path, &info) == 0 &&
           (directory ? S_ISDIR(info.st_mode) : S_ISREG(info.st_mode));
}

/* The sampling grid a checkpoint names in its metadata ("config":
 * {"transformer": {"pdd_sigmas": [...]}}); 0 values when it names none. */
static int parse_sigmas(const char *config, float *sigmas) {
    const char *cursor = config ? strstr(config, "\"pdd_sigmas\"") : NULL;
    if (!cursor || !(cursor = strchr(cursor, '['))) return 0;
    cursor++;
    int count = 0;
    while (count <= H3_IMAGE_MAX_STEPS) {
        errno = 0;
        char *end = NULL;
        float value = strtof(cursor, &end);
        if (errno || end == cursor || !isfinite(value)) return 0;
        sigmas[count++] = value;
        cursor = end + strspn(end, " \t\r\n");
        if (*cursor == ']') break;
        if (*cursor++ != ',') return 0;
    }
    if (*cursor != ']' || count < 2) return 0;
    /* A grid runs down from 1 to 0. */
    for (int index = 1; index < count; index++)
        if (sigmas[index] >= sigmas[index - 1]) return 0;
    if (sigmas[0] > 1.0f || sigmas[count - 1] != 0.0f) return 0;
    return count;
}

/* That the DiT is a still-image checkpoint, and its sampling grid. */
static int inspect_transformer(h3_image_ctx *ctx, char *error,
                               size_t error_size) {
    h3_weight_store *weights =
        h3_weight_store_open(ctx->transformer, error, error_size);
    if (!weights) return 0;
    const h3_st_header *header = NULL;
    const h3_st_tensor *head =
        h3_weight_find(weights, "final_layer.video_out.weight", &header);
    int ok = 0;
    if (!head || !h3_weight_find(weights, "video_patch_proj.weight", NULL)) {
        fail(error, error_size, "%s does not hold an H3 transformer",
             ctx->transformer);
    } else if (h3_weight_find(weights, "audio_patch_proj.weight", NULL)) {
        fail(error, error_size, "the transformer in %s is a video model (it "
             "has an audio stream), not a still-image one", ctx->transformer);
    } else {
        int count = parse_sigmas(header ? h3_st_metadata(header, "config") : NULL,
                                 ctx->sigmas);
        if (!count) {
            count = (int)(sizeof(default_sigmas) / sizeof(*default_sigmas));
            memcpy(ctx->sigmas, default_sigmas, sizeof(default_sigmas));
        }
        ctx->steps = count - 1;
        ok = 1;
    }
    h3_weight_store_free(weights);
    return ok;
}

void h3_image_free(h3_image_ctx *ctx) {
    if (!ctx) return;
    free(ctx->transformer);
    free(ctx->vae);
    free(ctx->text_encoder);
    free(ctx->tokenizer);
    free(ctx->shaders);
    free(ctx);
}

h3_image_ctx *h3_image_load_dir(const char *model_directory,
                                const char *shader_source_path,
                                char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!model_directory || !shader_source_path) {
        fail(error, error_size, "invalid image model arguments");
        return NULL;
    }
    h3_image_ctx *ctx = calloc(1, sizeof(*ctx));
    if (ctx) {
        ctx->transformer = join(model_directory, "transformer");
        ctx->vae = join(model_directory, "vae");
        ctx->text_encoder = join(model_directory, "text_encoder");
        ctx->tokenizer = join(model_directory, "tokenizer/tokenizer.json");
        ctx->shaders = strdup(shader_source_path);
    }
    if (!ctx || !ctx->transformer || !ctx->vae || !ctx->text_encoder ||
        !ctx->tokenizer || !ctx->shaders) {
        fail(error, error_size, "out of memory opening the image model");
        h3_image_free(ctx);
        return NULL;
    }
    const char *missing =
        !exists(ctx->transformer, 1) ? "transformer" :
        !exists(ctx->vae, 1) ? "vae" :
        !exists(ctx->text_encoder, 1) ? "text_encoder" :
        !exists(ctx->tokenizer, 0) ? "tokenizer/tokenizer.json" : NULL;
    if (missing) {
        fail(error, error_size, "%s is not an image model folder: it has no %s",
             model_directory, missing);
        h3_image_free(ctx);
        return NULL;
    }
    if (!inspect_transformer(ctx, error, error_size)) {
        h3_image_free(ctx);
        return NULL;
    }
    return ctx;
}

void h3_image_result_free(h3_image_result *result) {
    if (!result) return;
    free(result->rgb);
    free(result->latent);
    memset(result, 0, sizeof(*result));
}

typedef struct {
    h3_image_progress progress;
    void *opaque;
} progress_state;

static void text_progress(int completed, int total, void *opaque) {
    progress_state *state = opaque;
    if (state->progress) state->progress("text encoder", completed, total, state->opaque);
}

static void dit_progress(const char *phase, int completed, int total,
                         void *opaque) {
    (void)phase;
    progress_state *state = opaque;
    if (state->progress) state->progress("image model", completed, total, state->opaque);
}

static void vae_progress(int completed, int total, void *opaque) {
    progress_state *state = opaque;
    if (state->progress) state->progress("decode", completed, total, state->opaque);
}

static int cancelled(const h3_image_params *params, char *error,
                     size_t error_size) {
    if (!params->cancel_requested || !*params->cancel_requested) return 0;
    fail(error, error_size, "generation cancelled");
    return 1;
}

int h3_image_generate(h3_image_ctx *ctx, const h3_image_params *params,
                      h3_image_result *result, char *error, size_t error_size) {
    if (result) memset(result, 0, sizeof(*result));
    if (error && error_size) error[0] = '\0';
    if (!ctx || !params || !result || !params->prompt) {
        fail(error, error_size, "invalid image generation arguments");
        return 0;
    }
    if (params->width < MIN_SIDE || params->height < MIN_SIDE ||
        params->width > MAX_SIDE || params->height > MAX_SIDE ||
        params->width % 32 || params->height % 32) {
        fail(error, error_size, "the image size must be multiples of 32 from "
             "%d to %d pixels a side", MIN_SIDE, MAX_SIDE);
        return 0;
    }
    int latent_h = params->height / 16, latent_w = params->width / 16;
    size_t latent_count = (size_t)LATENT_CHANNELS * (size_t)latent_h *
                          (size_t)latent_w;
    progress_state state = {params->progress, params->progress_opaque};

    h3_tokenizer *tokenizer = NULL;
    uint32_t *ids = NULL;
    size_t token_count = 0;
    h3_text_embedding text;
    memset(&text, 0, sizeof(text));
    h3_layout layout;
    memset(&layout, 0, sizeof(layout));
    h3_dit *dit = NULL;
    float *latent = malloc(latent_count * sizeof(*latent));
    float *velocity = malloc(latent_count * sizeof(*velocity));
    h3_image_pixels pixels;
    memset(&pixels, 0, sizeof(pixels));
    int ok = 0;
    if (!latent || !velocity) {
        fail(error, error_size, "out of memory for the image latent");
        goto done;
    }

    /* Text: the prompt as it is - no chat template, no special tokens. */
    double began = now();
    tokenizer = h3_tokenizer_load(ctx->tokenizer, error, error_size);
    if (!tokenizer ||
        !h3_tokenizer_encode(tokenizer, params->prompt, 0, &ids, &token_count,
                             error, error_size)) goto done;
    if (!token_count) {
        fail(error, error_size, "the prompt is empty");
        goto done;
    }
    if (token_count > H3_IMAGE_MAX_PROMPT_TOKENS)
        token_count = H3_IMAGE_MAX_PROMPT_TOKENS;
    if (cancelled(params, error, error_size)) goto done;
    if (!h3_text_encode_bf16(ctx->text_encoder, ctx->shaders, ids, token_count,
                             text_progress, &state, &text, error, error_size))
        goto done;
    result->text_seconds = now() - began;
    if (h3_option("H3_PROFILE")) {
        double energy = 0.0;
        for (size_t index = 0; index < text.tokens * text.width; index++) {
            uint32_t bits = (uint32_t)text.values[index] << 16;
            float value;
            memcpy(&value, &bits, sizeof(value));
            energy += (double)value * (double)value;
        }
        fprintf(stderr, "h3: image text embedding: %zu tokens, rms %.3f\n",
                text.tokens, sqrt(energy / (double)(text.tokens * text.width)));
    }
    result->prompt_tokens = (int)token_count;
    if (cancelled(params, error, error_size)) goto done;

    /* The DiT on [text | image]: one latent frame, no audio. */
    began = now();
    h3_sigma_schedule sigmas = {.steps = ctx->steps};
    for (int index = 0; index <= ctx->steps; index++)
        sigmas.video[index] = sigmas.audio[index] = ctx->sigmas[index];
    h3_layout_spec spec = {
        .text_len = (int)text.tokens, .latent_t = 1, .latent_h = latent_h,
        .latent_w = latent_w, .audio_t = 0, .frame_count = 5
    };
    if (!h3_layout_build(&spec, &layout, error, error_size)) goto done;
    dit = h3_dit_load_t2va(ctx->transformer, ctx->shaders, &text, &layout,
        &sigmas, H3_DIT_BLOCKS, 1, 0, 1 /* SSD streaming */, 1.0f,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, NULL, 0, NULL,
        dit_progress, &state, error, error_size);
    if (!dit) goto done;
    if (!h3_dit_is_image_model(dit)) {
        fail(error, error_size, "the transformer is not a still-image model");
        goto done;
    }
    h3_dit_set_cancel(dit, params->cancel_requested);
    if (params->initial_noise) {
        memcpy(latent, params->initial_noise, latent_count * sizeof(*latent));
    } else {
        h3_rng rng;
        h3_rng_seed(&rng, params->seed);
        h3_rng_fill_normal(&rng, latent, latent_count);
    }
    if (h3_option("H3_PROFILE")) {
        double energy = 0.0;
        for (size_t index = 0; index < latent_count; index++)
            energy += (double)latent[index] * (double)latent[index];
        fprintf(stderr, "h3: image noise: rms %.4f\n",
                sqrt(energy / (double)latent_count));
    }
    if (params->progress)
        params->progress("denoise", 0, ctx->steps, params->progress_opaque);
    for (int step = 0; step < ctx->steps; step++) {
        if (cancelled(params, error, error_size)) goto done;
        if (!h3_dit_forward_image(dit, step, latent, velocity, error,
                                  error_size)) goto done;
        /* The velocity points at the data: x += (sigma - sigma_next) * v. */
        float dt = ctx->sigmas[step] - ctx->sigmas[step + 1];
        if (h3_option("H3_PROFILE")) {
            size_t bad = 0;
            double sum = 0.0;
            for (size_t index = 0; index < latent_count; index++) {
                if (!isfinite(velocity[index])) bad++;
                else sum += (double)velocity[index] * (double)velocity[index];
            }
            fprintf(stderr, "h3: image velocity at step %d: rms %.4f, %zu of "
                    "%zu not finite\n", step + 1,
                    sqrt(sum / (double)latent_count), bad, latent_count);
        }
        double energy = 0.0;
        for (size_t index = 0; index < latent_count; index++) {
            latent[index] += dt * velocity[index];
            energy += (double)latent[index] * (double)latent[index];
        }
        if (!isfinite(energy)) {
            fail(error, error_size, "the image model produced a non-finite "
                 "value at step %d", step + 1);
            goto done;
        }
        if (h3_option("H3_PROFILE"))
            fprintf(stderr, "h3: image latent after step %d: rms %.4f\n",
                    step + 1, sqrt(energy / (double)latent_count));
        if (params->progress)
            params->progress("denoise", step + 1, ctx->steps,
                             params->progress_opaque);
    }
    /* The VAE gets the GPU to itself. */
    h3_dit_free(dit);
    dit = NULL;
    result->denoise_seconds = now() - began;
    if (cancelled(params, error, error_size)) goto done;

    began = now();
    if (!h3_image_vae_decode(ctx->vae, ctx->shaders, latent, latent_h,
                             latent_w, params->cancel_requested, vae_progress,
                             &state, &pixels, error, error_size)) goto done;
    result->decode_seconds = now() - began;
    size_t values = (size_t)pixels.width * (size_t)pixels.height * 3;
    result->rgb = malloc(values);
    if (!result->rgb) {
        fail(error, error_size, "out of memory for the image");
        goto done;
    }
    for (size_t index = 0; index < values; index++)
        result->rgb[index] = (uint8_t)lrintf(pixels.rgb[index] * 255.0f);
    result->width = pixels.width;
    result->height = pixels.height;
    result->latent = latent;
    latent = NULL;
    result->latent_width = latent_w;
    result->latent_height = latent_h;
    result->seed = params->seed;
    result->steps = ctx->steps;
    memcpy(result->sigmas, ctx->sigmas, sizeof(ctx->sigmas));
    ok = 1;

done:
    if (!ok) h3_image_result_free(result);
    h3_dit_free(dit);
    h3_image_pixels_free(&pixels);
    h3_layout_free(&layout);
    h3_text_embedding_free(&text);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tokenizer);
    free(latent);
    free(velocity);
    return ok;
}
