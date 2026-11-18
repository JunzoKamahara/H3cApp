#include "h3_image.h"

#include "h3_av_reader.h"
#include "h3_dit.h"
#include "h3_dit_schedule.h"
#include "h3_host.h"
#include "h3_image_vae.h"
#include "h3_metal.h"
#include "h3_multimodal.h"
#include "h3_options.h"
#include "h3_plan.h"
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"
#include "h3_video_encoder.h"
#include "h3_vision_encoder.h"
#include "h3_weights.h"

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

enum { LATENT_CHANNELS = 24, MIN_SIDE = 256, MAX_SIDE = 4096,
       PATCH_VALUES = 96 /* 24 channels x 2 x 2 */,
       REFERENCE_SHORT_EDGE = 1024, REFERENCE_MAX_SHORT_EDGE = 2048 };

/* h3_references.c: the latent an image was generated as, when it is of
 * this grid. */
int h3_frame_latent_load(const char *path, int latent_h, int latent_w,
                         float *values, char *reason, size_t reason_size);

/* The released checkpoint's own six-step grid, for one that does not
 * carry it in its metadata. */
static const float default_sigmas[] = {
    1.0f, 0.9675236940383911f, 0.8875485062599182f, 0.75f,
    0.5327739119529724f, 0.23200887441635132f, 0.0f
};

/* P-H3-Image's eight-step grid (its pipeline's sample_sigmas), used as it
 * is for the image and the two audio latents alike. */
static const float ph3_sigmas[] = {
    1.0f, 0.984753f, 0.967905f, 0.951688f, 0.925372f, 0.881923f, 0.810083f,
    0.528718f, 0.0f
};

enum { PH3_AUDIO_LATENTS = 2, AUDIO_CHANNELS = 32, AUDIO_STEREO = 2 };
/* diffusers' P-H3-Image edit sizes (modular_model_index.json). */
#define PH3_REFERENCE_MIN_PIXELS (1024.0 * 1024.0)
#define PH3_REFERENCE_MAX_PIXELS (8.0 * 1024.0 * 1024.0)
#define PH3_VISION_PIXELS 1000000.0

struct h3_image_ctx {
    /* P-H3-Image: MiniMax H3's audio-video DiT (FL2VA with Pruna's LoRA
     * merged), run on one latent frame and two audio latents, with a
     * Qwen3-VL-4B text encoder and its text connector. */
    int ph3;
    /* P-H3-Image as a LoRA: the one .safetensors in the folder's lora/,
     * applied at strength 1 to the transformer (then a video model's
     * FL2VA DiT); NULL when the transformer has it merged. */
    char *lora;
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

/* Whether the text encoder carries P-H3-Image's text connector. */
static int has_text_connector(const char *directory) {
    char ignored[256];
    h3_weight_store *store = h3_weight_store_open(directory, ignored,
                                                  sizeof(ignored));
    int found = store &&
        h3_weight_find(store, "text_connector.proj.weight", NULL) != NULL;
    h3_weight_store_free(store);
    return found;
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
        if (has_text_connector(ctx->text_encoder)) {
            ctx->ph3 = 1;
            ctx->steps = (int)(sizeof(ph3_sigmas) / sizeof(*ph3_sigmas)) - 1;
            memcpy(ctx->sigmas, ph3_sigmas, sizeof(ph3_sigmas));
            ok = 1;
        } else {
            fail(error, error_size, "the transformer in %s is a video model (it "
                 "has an audio stream), not a still-image one",
                 ctx->transformer);
        }
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
    free(ctx->lora);
    free(ctx);
}

void h3_image_get_info(const h3_image_ctx *ctx, h3_image_info *info) {
    if (!info) return;
    memset(info, 0, sizeof(*info));
    if (!ctx) return;
    info->model = ctx->ph3 ? H3_IMAGE_MODEL_PH3 : H3_IMAGE_MODEL_H3;
    info->name = ctx->ph3 ? "P-H3-Image" : "MiniMax H3 Image";
    info->steps = ctx->steps;
    info->can_edit = 1;
    info->max_references = ctx->ph3 ? H3_IMAGE_PH3_MAX_REFERENCES :
                                      H3_IMAGE_MAX_REFERENCES;
}

h3_image_ctx *h3_image_load_dir(const char *model_directory,
                                const char *shader_source_path,
                                char *error, size_t error_size) {
    return h3_image_load_parts(model_directory, NULL, shader_source_path,
                               error, error_size);
}

h3_image_ctx *h3_image_load_parts(const char *model_directory,
                                  const char *text_encoder_directory,
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
        ctx->text_encoder = text_encoder_directory
            ? strdup(text_encoder_directory)
            : join(model_directory, "text_encoder");
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
    if (missing && text_encoder_directory && !strcmp(missing, "text_encoder")) {
        fail(error, error_size, "the text encoder folder %s of the image model "
             "in %s is not there", text_encoder_directory, model_directory);
        h3_image_free(ctx);
        return NULL;
    }
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
    char *lora_folder = join(model_directory, "lora");
    DIR *folder = lora_folder ? opendir(lora_folder) : NULL;
    int adapters = 0;
    for (struct dirent *entry; folder && (entry = readdir(folder));) {
        size_t length = strlen(entry->d_name);
        if (entry->d_name[0] == '.' || length < 12 ||
            strcmp(entry->d_name + length - 12, ".safetensors")) continue;
        adapters++;
        if (!ctx->lora) ctx->lora = join(lora_folder, entry->d_name);
    }
    if (folder) closedir(folder);
    free(lora_folder);
    if (adapters && (!ctx->ph3 || adapters != 1 || !ctx->lora)) {
        fail(error, error_size, adapters != 1
             ? "an image model's lora folder holds one adapter, not %d"
             : "only P-H3-Image takes its DiT as a LoRA", adapters);
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

static int canvas_side(int value, double scale) {
    double side = nearbyint((double)value * scale / 32.0) * 32.0;
    return side < 32.0 ? 32 : (int)side;
}

int h3_image_reference_canvas(int width, int height, int max_short_edge,
                              int *encode_width, int *encode_height,
                              int *vision_width, int *vision_height) {
    if (width < 1 || height < 1 || !encode_width || !encode_height ||
        !vision_width || !vision_height) return 0;
    int low = width < height ? width : height;
    int high = width < height ? height : width;
    if ((double)high > H3_IMAGE_REFERENCE_MAX_ASPECT * (double)low) return 0;
    if (max_short_edge <= 0) max_short_edge = REFERENCE_MAX_SHORT_EDGE;
    /* The picture's own short side, enlarged to 1024 and reduced to the
     * limit, as the released pipeline sizes a reference for the VAE. */
    double target = low < REFERENCE_SHORT_EDGE ? REFERENCE_SHORT_EDGE : low;
    if (target > max_short_edge) target = max_short_edge;
    double scale = target / (double)low;
    /* Beyond the released rule: a wide picture at a long short side would
     * be many times the rows of the image being made. */
    double area = (double)width * scale * (double)height * scale;
    double limit = (double)REFERENCE_MAX_SHORT_EDGE *
                   (double)REFERENCE_MAX_SHORT_EDGE;
    if (area > limit) scale *= sqrt(limit / area);
    *encode_width = canvas_side(width, scale);
    *encode_height = canvas_side(height, scale);
    scale = (double)REFERENCE_SHORT_EDGE / (double)low;
    *vision_width = canvas_side(width, scale);
    *vision_height = canvas_side(height, scale);
    return 1;
}

int h3_image_reference_canvas_ph3(int width, int height,
                                  int *encode_width, int *encode_height,
                                  int *vision_width, int *vision_height) {
    if (width < 1 || height < 1 || !encode_width || !encode_height ||
        !vision_width || !vision_height) return 0;
    int low = width < height ? width : height;
    int high = width < height ? height : width;
    if ((double)high > H3_IMAGE_REFERENCE_MAX_ASPECT * (double)low) return 0;
    /* For the VAE: its own size, its area brought into 1 to 8 MP. */
    double area = (double)width * (double)height;
    double kept = area < PH3_REFERENCE_MIN_PIXELS ? PH3_REFERENCE_MIN_PIXELS :
                  area > PH3_REFERENCE_MAX_PIXELS ? PH3_REFERENCE_MAX_PIXELS : area;
    double scale = sqrt(kept / area);
    *encode_width = canvas_side(width, scale);
    *encode_height = canvas_side(height, scale);
    /* For the vision tower: about a megapixel, a picture already near it
     * (sides on the grid, within 20%) as it is. */
    if (width % 32 == 0 && height % 32 == 0 &&
        area >= 0.8 * PH3_VISION_PIXELS && area <= 1.2 * PH3_VISION_PIXELS) {
        *vision_width = width;
        *vision_height = height;
    } else {
        double ratio = (double)width / (double)height;
        *vision_width = canvas_side(1, sqrt(PH3_VISION_PIXELS * ratio));
        *vision_height = canvas_side(1, sqrt(PH3_VISION_PIXELS / ratio));
    }
    return 1;
}

/* The sizes the model in ctx uses a reference at. */
static int reference_canvas(const h3_image_ctx *ctx, int width, int height,
                            int max_short_edge, int *encode_width,
                            int *encode_height, int *vision_width,
                            int *vision_height) {
    return ctx->ph3
        ? h3_image_reference_canvas_ph3(width, height, encode_width,
                                        encode_height, vision_width,
                                        vision_height)
        : h3_image_reference_canvas(width, height, max_short_edge,
                                    encode_width, encode_height,
                                    vision_width, vision_height);
}

/* P-H3-Image's edit prompt: "Edit <Picture 1>. " in front, unless it
 * already starts that way or is an edit brief (diffusers'
 * PH3ImageEditTextEncoderStep). The caller frees it. */
static char *ph3_edit_prompt(const char *prompt) {
    const char *start = prompt + strspn(prompt, " \t\r\n\f\v");
    if (!strncmp(start, "Edit <Picture 1>.", 17) ||
        !strncmp(start, "subject_definitions:", 20)) return strdup(prompt);
    static const char prefix[] = "Edit <Picture 1>. ";
    size_t size = sizeof(prefix) + strlen(prompt);
    char *text = malloc(size);
    if (text) snprintf(text, size, "%s%s", prefix, prompt);
    return text;
}

typedef struct {
    h3_image_progress progress;
    void *opaque;
    /* Which reference of how many a vision or encoder callback is for. */
    int index;
    int count;
} progress_state;

static void vision_progress(int completed, int total, void *opaque) {
    progress_state *state = opaque;
    if (state->progress)
        state->progress("vision", state->index * total + completed,
                        state->count * total, state->opaque);
}

static void encoder_progress(int completed, int total, void *opaque) {
    progress_state *state = opaque;
    if (state->progress)
        state->progress("reference", state->index * total + completed,
                        state->count * total, state->opaque);
}

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

/* What is wrong with the request itself, before anything is read. */
static int check_request(const h3_image_ctx *ctx,
                         const h3_image_params *params, char *error,
                         size_t error_size) {
    int max_references = ctx->ph3 ? H3_IMAGE_PH3_MAX_REFERENCES :
                                     H3_IMAGE_MAX_REFERENCES;
    if (params->width < MIN_SIDE || params->height < MIN_SIDE ||
        params->width > MAX_SIDE || params->height > MAX_SIDE ||
        params->width % 32 || params->height % 32) {
        fail(error, error_size, "the image size must be multiples of 32 from "
             "%d to %d pixels a side", MIN_SIDE, MAX_SIDE);
        return 0;
    }
    if (params->reference_count < 0 ||
        params->reference_count > max_references ||
        (params->reference_count && !params->references)) {
        fail(error, error_size, "an image takes at most %d reference images",
             max_references);
        return 0;
    }
    if (params->init_image_path && *params->init_image_path &&
        params->reference_count) {
        fail(error, error_size, "a picture to redraw cannot be combined with "
             "reference images");
        return 0;
    }
    for (int index = 0; index < params->reference_count; index++)
        if (!params->references[index].path ||
            !*params->references[index].path) {
            fail(error, error_size, "reference %d has no file", index + 1);
            return 0;
        }
    return 1;
}

int h3_image_plan(const h3_image_ctx *ctx, const h3_image_params *params,
                  h3_plan *plan) {
    if (!plan) return 0;
    memset(plan, 0, sizeof(*plan));
    char *message = plan->message;
    size_t size = sizeof(plan->message);
    if (!ctx || !params || !params->prompt) {
        fail(message, size, "invalid image plan arguments");
        return 0;
    }
    if (!check_request(ctx, params, message, size)) return 0;
    int count = params->reference_count;
    char detail[384] = "";
    h3_device_info device;
    if (!h3_metal_probe(&device, detail, sizeof(detail))) {
        fail(message, size, "%s", detail);
        return 0;
    }
    h3_vision_output vision[H3_IMAGE_MAX_REFERENCES];
    h3_plan_visual visuals[H3_IMAGE_MAX_REFERENCES];
    memset(vision, 0, sizeof(vision));
    size_t vision_tokens = 0, reference_rows = 0;
    for (int index = 0; index < count; index++) {
        int source_w = 0, source_h = 0, encode_w = 0, encode_h = 0;
        int vision_w = 0, vision_h = 0;
        if (!h3_av_visual_size(params->references[index].path, &source_w,
                               &source_h, detail, sizeof(detail))) {
            fail(message, size, "reference %d: %s", index + 1, detail);
            return 0;
        }
        if (!reference_canvas(ctx, source_w, source_h,
                                       params->reference_max_short_edge,
                                       &encode_w, &encode_h, &vision_w,
                                       &vision_h)) {
            fail(message, size, "reference %d is %d x %d pixels: the longer "
                 "side may be at most %.2f times the shorter", index + 1,
                 source_w, source_h, H3_IMAGE_REFERENCE_MAX_ASPECT);
            return 0;
        }
        /* As the vision tower gives them: 16-pixel patches merged 2 x 2. */
        vision[index].grid_h = vision_h / 16;
        vision[index].grid_w = vision_w / 16;
        vision[index].tokens = (size_t)(vision_h / 32) * (size_t)(vision_w / 32);
        vision_tokens += vision[index].tokens;
        reference_rows += (size_t)(encode_h / 32) * (size_t)(encode_w / 32);
        /* A reference taken by its own latent is not encoded; counting it
         * keeps the estimate on the safe side. */
        visuals[index] = (h3_plan_visual){encode_w, encode_h, 1};
    }
    h3_tokenizer *tokenizer = h3_tokenizer_load(ctx->tokenizer, detail,
                                                sizeof(detail));
    size_t text_tokens = 0;
    int counted = 0;
    if (tokenizer && count) {
        char *prompt = ctx->ph3 ? ph3_edit_prompt(params->prompt) : NULL;
        counted = (!ctx->ph3 || prompt) &&
            h3_multimodal_image_edit_token_count(
                tokenizer, prompt ? prompt : params->prompt, vision,
                (size_t)count, ctx->ph3 ? H3_PICTURE_LABEL_VIDEO : NULL,
                H3_IMAGE_MAX_PROMPT_TOKENS, &text_tokens, detail,
                sizeof(detail));
        if (ctx->ph3 && !prompt) snprintf(detail, sizeof(detail), "out of memory");
        free(prompt);
    } else if (tokenizer) {
        uint32_t *ids = NULL;
        counted = h3_tokenizer_encode(tokenizer, params->prompt, 0, &ids,
                                      &text_tokens, detail, sizeof(detail));
        h3_tokenizer_ids_free(ids);
        if (counted && !text_tokens) {
            snprintf(detail, sizeof(detail), "the prompt is empty");
            counted = 0;
        }
        if (text_tokens > H3_IMAGE_MAX_PROMPT_TOKENS)
            text_tokens = H3_IMAGE_MAX_PROMPT_TOKENS;
    }
    h3_tokenizer_free(tokenizer);
    if (!counted) {
        fail(message, size, "%s", detail);
        return 0;
    }
    size_t generated = (size_t)(params->height / 32) * (size_t)(params->width / 32);
    /* As h3_gpu.m decides: dense attention in one call on an Apple10 GPU
     * with macOS 26 or later, unless H3_SDPA_MODE says. */
    const char *mode = h3_option("H3_SDPA_MODE");
    int full_sdpa = 0;
    if (mode && *mode) full_sdpa = !strcmp(mode, "full");
    else if (device.apple_gpu_family >= 10) {
        if (__builtin_available(macOS 26.0, *)) full_sdpa = 1;
    }
    const char *prefetch = h3_option("H3_QWEN_PREFETCH_DEPTH");
    int depth = params->plan_text_prefetch_depth > 0 ?
        params->plan_text_prefetch_depth :
        prefetch && *prefetch ? atoi(prefetch) :
        device.apple_gpu_family >= 10 ? 3 : 2;
    if (depth < 1) depth = 1;
    h3_plan_input input = {
        .memory_bytes = device.physical_memory, .ssd_streaming = 1,
        .full_sdpa = full_sdpa, .text_prefetch_depth = depth,
        .text_tokens = text_tokens, .vision_tokens = vision_tokens,
        .dit_tokens = text_tokens + reference_rows + generated +
                      (ctx->ph3 ? PH3_AUDIO_LATENTS * AUDIO_STEREO : 0),
        .dit_generated_tokens = generated,
        .dit_reference_tokens = reference_rows,
        .visuals = visuals, .visual_count = (size_t)count
    };
    h3_plan_compute(&input, plan);
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
    if (!check_request(ctx, params, error, error_size)) return 0;
    int reference_count = params->reference_count;
    {
        /* Refused before anything is loaded when it cannot fit; a request
         * the plan cannot read is left for the generation to report. */
        const char *preflight = h3_option("H3_PREFLIGHT");
        h3_plan plan;
        if ((!preflight || strcmp(preflight, "0")) &&
            h3_image_plan(ctx, params, &plan)) {
            fprintf(stderr, "h3: image plan: text encoder %zu tokens (%zu "
                    "from pictures), DiT %zu tokens (%zu from references, "
                    "cost x%.2f), peak about %.1f GiB in the %s\n",
                    plan.text_tokens, plan.vision_tokens, plan.dit_tokens,
                    plan.dit_reference_tokens, plan.dit_cost_ratio,
                    (double)plan.peak_bytes / (1024.0 * 1024.0 * 1024.0),
                    plan.peak_stage);
            if (plan.verdict == H3_PLAN_TOO_LARGE) {
                fail(error, error_size, "%s", plan.message);
                return 0;
            }
        }
    }
    int latent_h = params->height / 16, latent_w = params->width / 16;
    size_t latent_count = (size_t)LATENT_CHANNELS * (size_t)latent_h *
                          (size_t)latent_w;
    progress_state state = {params->progress, params->progress_opaque, 0,
                            reference_count};
    h3_vision_output vision[H3_IMAGE_MAX_REFERENCES];
    memset(vision, 0, sizeof(vision));
    h3_layout_ref layout_references[H3_IMAGE_MAX_REFERENCES];
    memset(layout_references, 0, sizeof(layout_references));
    int vision_w[H3_IMAGE_MAX_REFERENCES], vision_h[H3_IMAGE_MAX_REFERENCES];
    float *condition_rows = NULL;
    size_t condition_elements = 0;
    char detail[384] = "";

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
    /* The finished-image estimate a preview is given (latent_preview). */
    float *estimate = NULL;
    /* Image to image: the picture's latent and the level it starts at. */
    float *init = NULL;
    float init_sigma = params->init_strength < 0.05f ? 0.05f
                     : params->init_strength > 1.0f ? 1.0f : params->init_strength;
    /* P-H3-Image's two audio latents, denoised alongside and dropped. */
    enum { AUDIO_VALUES = AUDIO_CHANNELS * AUDIO_STEREO * PH3_AUDIO_LATENTS };
    float audio[AUDIO_VALUES], audio_velocity[AUDIO_VALUES];
    h3_image_pixels pixels;
    memset(&pixels, 0, sizeof(pixels));
    int ok = 0;
    if (!latent || !velocity) {
        fail(error, error_size, "out of memory for the image latent");
        goto done;
    }

    double began = now();
    tokenizer = h3_tokenizer_load(ctx->tokenizer, error, error_size);
    if (!tokenizer) goto done;
    if (reference_count) {
        /* Editing. The references through the vision tower first: the text
         * sequence is `Picture n: ` and the n-th picture's tokens for each,
         * then the prompt. */
        for (int index = 0; index < reference_count; index++) {
            const char *path = params->references[index].path;
            int source_w = 0, source_h = 0, encode_w = 0, encode_h = 0;
            if (!h3_av_visual_size(path, &source_w, &source_h, detail,
                                   sizeof(detail))) {
                fail(error, error_size, "reference %d: %s", index + 1, detail);
                goto done;
            }
            if (!reference_canvas(
                    ctx, source_w, source_h, params->reference_max_short_edge,
                    &encode_w, &encode_h, &vision_w[index], &vision_h[index])) {
                fail(error, error_size, "reference %d is %d x %d pixels: the "
                     "longer side may be at most %.2f times the shorter",
                     index + 1, source_w, source_h,
                     H3_IMAGE_REFERENCE_MAX_ASPECT);
                goto done;
            }
            result->reference_width[index] = encode_w;
            result->reference_height[index] = encode_h;
            layout_references[index] = (h3_layout_ref){
                .kind = H3_LAYOUT_REF_IMAGE, .latent_t = 1,
                .latent_h = encode_h / 16, .latent_w = encode_w / 16};
        }
        for (int index = 0; index < reference_count; index++) {
            float *picture = NULL;
            if (!h3_av_read_image_f32(params->references[index].path,
                                      vision_w[index], vision_h[index],
                                      H3_IMAGE_FIT_STRETCH, &picture, detail,
                                      sizeof(detail))) {
                fail(error, error_size, "reference %d: %s", index + 1, detail);
                goto done;
            }
            state.index = index;
            void *pool = h3_autorelease_push();
            int encoded = h3_vision_encode_bf16(
                ctx->text_encoder, ctx->shaders, picture, 1, vision_h[index],
                vision_w[index], vision_progress, &state, &vision[index],
                detail, sizeof(detail));
            h3_autorelease_pop(pool);
            free(picture);
            if (!encoded) {
                fail(error, error_size, "reference %d: %s", index + 1, detail);
                goto done;
            }
            if (cancelled(params, error, error_size)) goto done;
        }
        result->reference_seconds = now() - began;
        began = now();
        void *pool = h3_autorelease_push();
        /* P-H3-Image: `<Picture n>: ` labels and "Edit <Picture 1>. " in
         * front of the instruction. */
        char *prompt = ctx->ph3 ? ph3_edit_prompt(params->prompt) : NULL;
        int encoded = (!ctx->ph3 || prompt) &&
            h3_multimodal_encode_image_edit_bf16(
                tokenizer, ctx->text_encoder, ctx->shaders,
                prompt ? prompt : params->prompt, vision,
                (size_t)reference_count,
                ctx->ph3 ? H3_PICTURE_LABEL_VIDEO : NULL,
                H3_IMAGE_MAX_PROMPT_TOKENS, &token_count, text_progress,
                &state, &text, error, error_size);
        if (ctx->ph3 && !prompt) fail(error, error_size, "out of memory");
        free(prompt);
        h3_autorelease_pop(pool);
        for (int index = 0; index < reference_count; index++)
            h3_vision_output_free(&vision[index]);
        if (!encoded) goto done;
    } else {
        /* Text: the prompt as it is - no chat template, no special tokens. */
        if (!h3_tokenizer_encode(tokenizer, params->prompt, 0, &ids,
                                 &token_count, error, error_size)) goto done;
        if (!token_count) {
            fail(error, error_size, "the prompt is empty");
            goto done;
        }
        if (token_count > H3_IMAGE_MAX_PROMPT_TOKENS)
            token_count = H3_IMAGE_MAX_PROMPT_TOKENS;
        if (cancelled(params, error, error_size)) goto done;
        if (!h3_text_encode_bf16(ctx->text_encoder, ctx->shaders, ids,
                                 token_count, text_progress, &state, &text,
                                 error, error_size)) goto done;
    }
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
    result->text_rows = (int)text.tokens;
    result->reference_count = reference_count;
    if (cancelled(params, error, error_size)) goto done;

    /* The references as the DiT sees them: each one's latent, in the rows
     * of the image being made, with the released 0.1% of seeded noise. */
    if (reference_count) {
        began = now();
        for (int index = 0; index < reference_count; index++)
            condition_elements += (size_t)layout_references[index].latent_h *
                                  (size_t)layout_references[index].latent_w /
                                  4 * PATCH_VALUES;
        condition_rows = malloc(condition_elements * sizeof(*condition_rows));
        if (!condition_rows) {
            fail(error, error_size, "out of memory for the reference rows");
            goto done;
        }
        size_t offset = 0;
        for (int index = 0; index < reference_count; index++) {
            const h3_image_reference *reference = &params->references[index];
            int grid_h = layout_references[index].latent_h;
            int grid_w = layout_references[index].latent_w;
            size_t values = (size_t)LATENT_CHANNELS * (size_t)grid_h *
                            (size_t)grid_w;
            size_t elements = (size_t)grid_h * (size_t)grid_w / 4 * PATCH_VALUES;
            float *rows = condition_rows + offset;
            int have = 0;
            if (reference->latent_path && *reference->latent_path) {
                float *own = malloc(values * sizeof(*own));
                char reason[256] = "";
                have = own && h3_frame_latent_load(reference->latent_path,
                                                   grid_h, grid_w, own, reason,
                                                   sizeof(reason)) &&
                       h3_dit_patchify_video(own, LATENT_CHANNELS, 1, grid_h,
                                             grid_w, rows, elements);
                free(own);
                if (have)
                    fprintf(stderr, "h3: reference %d from its own latent "
                            "(%s)\n", index + 1, reference->latent_path);
                else
                    fprintf(stderr, "h3: reference %d: its latent is not used "
                            "(%s); encoding the image\n", index + 1,
                            reason[0] ? reason : "out of memory");
            }
            result->reference_from_latent[index] = have;
            if (!have) {
                float *picture = NULL;
                if (!h3_av_read_image_f32(reference->path, grid_w * 16,
                                          grid_h * 16, H3_IMAGE_FIT_STRETCH,
                                          &picture, detail, sizeof(detail))) {
                    fail(error, error_size, "reference %d: %s", index + 1,
                         detail);
                    goto done;
                }
                h3_video_latent encoded;
                memset(&encoded, 0, sizeof(encoded));
                state.index = index;
                /* The image VAE's encoder is the video VAE's. */
                void *pool = h3_autorelease_push();
                int ok_encode = h3_video_vae_encode(
                    ctx->vae, ctx->shaders, picture, 1, grid_h * 16, grid_w * 16,
                    encoder_progress, &state, &encoded, detail, sizeof(detail));
                h3_autorelease_pop(pool);
                free(picture);
                if (ok_encode && (encoded.time != 1 || encoded.height != grid_h ||
                                  encoded.width != grid_w)) {
                    snprintf(detail, sizeof(detail),
                             "the VAE gave an unexpected latent size");
                    ok_encode = 0;
                }
                if (ok_encode &&
                    !h3_dit_patchify_video(encoded.values, LATENT_CHANNELS, 1,
                                           grid_h, grid_w, rows, elements)) {
                    snprintf(detail, sizeof(detail),
                             "cannot arrange the latent in rows");
                    ok_encode = 0;
                }
                h3_video_latent_free(&encoded);
                if (!ok_encode) {
                    fail(error, error_size, "reference %d: %s", index + 1,
                         detail);
                    goto done;
                }
            }
            h3_rng rng;
            h3_rng_seed(&rng, params->seed);
            for (size_t value = 0; value < elements; value++)
                rows[value] = 0.999f * rows[value] +
                              0.001f * h3_rng_normal(&rng);
            offset += elements;
            if (cancelled(params, error, error_size)) goto done;
        }
        result->reference_seconds += now() - began;
    }

    /* Image to image: the picture encoded at the canvas, before the DiT
     * takes the GPU. */
    if (params->init_image_path && *params->init_image_path) {
        began = now();
        float *picture = NULL;
        if (!h3_av_read_image_f32(params->init_image_path, latent_w * 16,
                                  latent_h * 16, H3_IMAGE_FIT_STRETCH, &picture,
                                  detail, sizeof(detail))) {
            fail(error, error_size, "the picture to redraw: %s", detail);
            goto done;
        }
        h3_video_latent encoded;
        memset(&encoded, 0, sizeof(encoded));
        void *pool = h3_autorelease_push();
        int ok_encode = h3_video_vae_encode(
            ctx->vae, ctx->shaders, picture, 1, latent_h * 16, latent_w * 16,
            encoder_progress, &state, &encoded, detail, sizeof(detail));
        h3_autorelease_pop(pool);
        free(picture);
        if (ok_encode && (encoded.time != 1 || encoded.height != latent_h ||
                          encoded.width != latent_w)) {
            snprintf(detail, sizeof(detail), "the VAE gave an unexpected latent size");
            ok_encode = 0;
        }
        if (ok_encode) {
            init = malloc(latent_count * sizeof(*init));
            if (init) memcpy(init, encoded.values, latent_count * sizeof(*init));
            else snprintf(detail, sizeof(detail), "out of memory");
            ok_encode = init != NULL;
        }
        h3_video_latent_free(&encoded);
        if (!ok_encode) {
            fail(error, error_size, "the picture to redraw: %s", detail);
            goto done;
        }
        result->reference_seconds += now() - began;
        if (cancelled(params, error, error_size)) goto done;
    }

    /* The DiT on [text | references | image]: one latent frame, no audio;
     * P-H3-Image's on [text | audio | image], two audio latents. */
    began = now();
    h3_lora adapter = {.path = ctx->lora, .strength = 1.0f};
    const h3_lora *loras = ctx->lora ? &adapter : NULL;
    size_t lora_count = ctx->lora ? 1 : 0;
    /* The steps to run: the model's grid, or for an image to image its
     * start level and the grid below it. */
    int steps = ctx->steps;
    float grid[H3_IMAGE_MAX_STEPS + 1];
    memcpy(grid, ctx->sigmas, sizeof(grid));
    if (init) {
        steps = 0;
        grid[steps++] = init_sigma;
        for (int index = 1; index <= ctx->steps; index++)
            if (ctx->sigmas[index] < init_sigma - 1e-3f) grid[steps++] = ctx->sigmas[index];
        steps--;   /* the last entry, 0, ends the schedule */
    }
    h3_sigma_schedule sigmas = {.steps = steps};
    for (int index = 0; index <= steps; index++)
        sigmas.video[index] = sigmas.audio[index] = grid[index];
    h3_layout_spec spec = {
        .text_len = (int)text.tokens, .latent_t = 1, .latent_h = latent_h,
        .latent_w = latent_w, .audio_t = ctx->ph3 ? PH3_AUDIO_LATENTS : 0,
        .frame_count = 5,
        .references = reference_count ? layout_references : NULL,
        .reference_count = (size_t)reference_count
    };
    if (!h3_layout_build(&spec, &layout, error, error_size)) goto done;
    dit = reference_count
        ? h3_dit_load_conditioned(ctx->transformer, ctx->shaders, &text,
            &layout, &sigmas, H3_DIT_BLOCKS, 1, 0, 1 /* SSD streaming */, 1.0f,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, loras, lora_count, NULL,
            condition_rows, condition_elements, NULL, 0,
            dit_progress, &state, error, error_size)
        : h3_dit_load_t2va(ctx->transformer, ctx->shaders, &text, &layout,
            &sigmas, H3_DIT_BLOCKS, 1, 0, 1 /* SSD streaming */, 1.0f,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, loras, lora_count, NULL,
            dit_progress, &state, error, error_size);
    if (!dit) goto done;
    free(condition_rows);
    condition_rows = NULL;
    if (h3_dit_is_image_model(dit) == ctx->ph3) {
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
    if (init) {
        /* x_sigma = (1 - sigma) x_0 + sigma * noise, as the flow runs. */
        for (size_t index = 0; index < latent_count; index++)
            latent[index] = (1.0f - init_sigma) * init[index] + init_sigma * latent[index];
        free(init);
        init = NULL;
    }
    if (ctx->ph3) {
        /* The audio noise from the seed's own stream after the image's. */
        h3_rng rng;
        h3_rng_seed(&rng, params->seed ^ UINT64_C(0x9e3779b97f4a7c15));
        h3_rng_fill_normal(&rng, audio, AUDIO_VALUES);
    }
    if (h3_option("H3_PROFILE")) {
        double energy = 0.0;
        for (size_t index = 0; index < latent_count; index++)
            energy += (double)latent[index] * (double)latent[index];
        fprintf(stderr, "h3: image noise: rms %.4f\n",
                sqrt(energy / (double)latent_count));
    }
    if (params->progress)
        params->progress("denoise", 0, steps, params->progress_opaque);
    for (int step = 0; step < steps; step++) {
        if (cancelled(params, error, error_size)) goto done;
        if (ctx->ph3 ? !h3_dit_forward(dit, step, latent, audio, velocity,
                                       audio_velocity, error, error_size)
                     : !h3_dit_forward_image(dit, step, latent, velocity,
                                             error, error_size)) goto done;
        /* The velocity points at the data: x += (sigma - sigma_next) * v. */
        float dt = grid[step] - grid[step + 1];
        if (params->latent_preview && step + 1 < steps) {
            if (!estimate) estimate = malloc(latent_count * sizeof(*estimate));
            if (estimate) {
                float sigma = grid[step];
                for (size_t index = 0; index < latent_count; index++)
                    estimate[index] = latent[index] + sigma * velocity[index];
                params->latent_preview(estimate, latent_w, latent_h, step + 1,
                                       steps, params->progress_opaque);
            }
        }
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
        if (ctx->ph3)
            for (int index = 0; index < AUDIO_VALUES; index++)
                audio[index] += dt * audio_velocity[index];
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
            params->progress("denoise", step + 1, steps,
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
    result->steps = steps;
    memcpy(result->sigmas, grid, sizeof(grid));
    ok = 1;

done:
    if (!ok) h3_image_result_free(result);
    h3_dit_free(dit);
    h3_image_pixels_free(&pixels);
    h3_layout_free(&layout);
    h3_text_embedding_free(&text);
    for (int index = 0; index < reference_count; index++)
        h3_vision_output_free(&vision[index]);
    free(condition_rows);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tokenizer);
    free(latent);
    free(velocity);
    free(estimate);
    free(init);
    return ok;
}
