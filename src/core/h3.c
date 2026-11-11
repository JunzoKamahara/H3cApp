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

static char h3_global_error[512];

typedef struct {
    char *text;
    size_t length;
    size_t capacity;
} h3_key;

static void h3_conditioning_cache_clear(h3_ctx *ctx) {
    if (!ctx) return;
    free(ctx->conditioning_key);
    free(ctx->conditioning_values);
    free(ctx->conditioning_tags);
    free(ctx->conditioning_video_rows);
    free(ctx->conditioning_audio_rows);
    free(ctx->conditioning_references);
    ctx->conditioning_key = NULL;
    ctx->conditioning_values = NULL;
    ctx->conditioning_tags = NULL;
    ctx->conditioning_video_rows = NULL;
    ctx->conditioning_audio_rows = NULL;
    ctx->conditioning_references = NULL;
    ctx->conditioning_tokens = 0;
    ctx->conditioning_width = 0;
    ctx->conditioning_video_elements = 0;
    ctx->conditioning_audio_elements = 0;
    ctx->conditioning_reference_count = 0;
    ctx->conditioning_present = 0;
}

void h3_cache_clear(h3_ctx *ctx) {
    if (!ctx) return;
    h3_conditioning_cache_clear(ctx);
    h3_dit_prep_clear(ctx->dit_prep);
    h3_dit_free(ctx->dit);
    ctx->dit = NULL;
    free(ctx->dit_key);
    ctx->dit_key = NULL;
    h3_video_vae_decoder_free(ctx->video_decoder);
    ctx->video_decoder = NULL;
    free(ctx->video_decoder_key);
    ctx->video_decoder_key = NULL;
}

void h3_cache_set_targets(h3_ctx *ctx, unsigned targets) {
    if (!ctx) return;
    targets &= H3_CACHE_ALL | H3_CACHE_REFINED_TEXT | H3_CACHE_ADALN;
    int refined = (targets & H3_CACHE_REFINED_TEXT) != 0;
    int adaln = (targets & H3_CACHE_ADALN) != 0;
    if ((refined || adaln) && !ctx->dit_prep) ctx->dit_prep = h3_dit_prep_new();
    h3_dit_prep_set(ctx->dit_prep, refined, adaln);
    if (!(targets & H3_CACHE_CONDITIONING)) h3_conditioning_cache_clear(ctx);
    if (!(targets & H3_CACHE_DIT)) {
        h3_dit_free(ctx->dit);
        ctx->dit = NULL;
        free(ctx->dit_key);
        ctx->dit_key = NULL;
    }
    if (!(targets & H3_CACHE_DECODER)) {
        h3_video_vae_decoder_free(ctx->video_decoder);
        ctx->video_decoder = NULL;
        free(ctx->video_decoder_key);
        ctx->video_decoder_key = NULL;
    }
    ctx->cache_targets = targets;
}

void h3_cache_set_enabled(h3_ctx *ctx, int enabled) {
    h3_cache_set_targets(ctx, enabled ? H3_CACHE_ALL : 0u);
}

static int h3_cache_target(const h3_ctx *ctx, unsigned target) {
    return (ctx->cache_targets & target) != 0;
}

void h3_cache_get_info(const h3_ctx *ctx, h3_cache_info *info) {
    if (!info) return;
    memset(info, 0, sizeof(*info));
    if (!ctx) return;
    if (ctx->conditioning_key) {
        info->embedding_entries = 1;
        info->embedding_bytes =
            ctx->conditioning_tokens * ctx->conditioning_width *
                sizeof(*ctx->conditioning_values) +
            ctx->conditioning_tokens * sizeof(*ctx->conditioning_tags) +
            ctx->conditioning_video_elements *
                sizeof(*ctx->conditioning_video_rows) +
            ctx->conditioning_audio_elements *
                sizeof(*ctx->conditioning_audio_rows);
    }
    info->prepared_dit = ctx->dit != NULL;
    info->video_decoder = ctx->video_decoder != NULL;
    info->refined_text_bytes = h3_dit_prep_bytes(ctx->dit_prep, 0);
    info->adaln_bytes = h3_dit_prep_bytes(ctx->dit_prep, 1);
}

static int h3_key_append(h3_key *key, const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    va_list copy;
    va_copy(copy, arguments);
    int needed = vsnprintf(NULL, 0, format, copy);
    va_end(copy);
    if (needed < 0) {
        va_end(arguments);
        return 0;
    }
    size_t wanted = key->length + (size_t)needed + 1;
    if (wanted > key->capacity) {
        size_t capacity = key->capacity ? key->capacity : 256;
        while (capacity < wanted) {
            if (capacity > SIZE_MAX / 2) {
                va_end(arguments);
                return 0;
            }
            capacity *= 2;
        }
        char *grown = realloc(key->text, capacity);
        if (!grown) {
            va_end(arguments);
            return 0;
        }
        key->text = grown;
        key->capacity = capacity;
    }
    vsnprintf(key->text + key->length, key->capacity - key->length,
              format, arguments);
    va_end(arguments);
    key->length += (size_t)needed;
    return 1;
}

static int h3_key_file(h3_key *key, const char *role, const char *path) {
    if (!path) return h3_key_append(key, "|%s=none", role);
    struct stat status;
    if (stat(path, &status) != 0)
        return h3_key_append(key, "|%s=%zu:%s:missing", role,
                             strlen(path), path);
    return h3_key_append(key, "|%s=%zu:%s:%lld:%lld:%ld", role, strlen(path),
                         path, (long long)status.st_size,
                         (long long)status.st_mtimespec.tv_sec,
                         status.st_mtimespec.tv_nsec);
}

/* Clip latents for continuation: a 9-int32 header (magic, version, video
 * channels, latent_t, latent_h, latent_w, audio channels, audio_t, pixel
 * frames) and then the F32 video latent [24,T,H,W] and the normalized
 * audio latent [32,2,T], as they were before decoding. */
#define H3_LATENT_FILE_MAGIC 0x544c3348u /* "H3LT" */

static int h3_latent_file_write(const char *path, const float *video,
                                int latent_t, int latent_h, int latent_w,
                                const float *audio, int audio_t, int frames,
                                char *error, size_t error_size) {
    int32_t header[9] = {(int32_t)H3_LATENT_FILE_MAGIC, 1, 24, latent_t,
                         latent_h, latent_w, 32, audio_t, frames};
    size_t video_count = (size_t)24 * latent_t * latent_h * latent_w;
    size_t audio_count = (size_t)32 * 2 * audio_t;
    FILE *file = fopen(path, "wb");
    int ok = file &&
        fwrite(header, sizeof(header), 1, file) == 1 &&
        fwrite(video, sizeof(*video), video_count, file) == video_count &&
        fwrite(audio, sizeof(*audio), audio_count, file) == audio_count;
    if (file && fclose(file) != 0) ok = 0;
    if (!ok) {
        snprintf(error, error_size, "cannot write latents to %s: %s", path,
                 strerror(errno));
        remove(path);
    }
    return ok;
}

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

static int h3_continuation_plan(const h3_params *params, int latent_h,
                                int latent_w, h3_temporal_shape temporal,
                                h3_continuation *plan,
                                char *error, size_t error_size) {
    memset(plan, 0, sizeof(*plan));
    if (!params->continue_latent_path) return 1;
    int32_t header[9];
    FILE *file = fopen(params->continue_latent_path, "rb");
    int read = file && fread(header, sizeof(header), 1, file) == 1;
    if (file) fclose(file);
    if (!read || header[0] != (int32_t)H3_LATENT_FILE_MAGIC ||
        header[1] != 1 || header[2] != 24 || header[6] != 32) {
        snprintf(error, error_size, "%s is not an H3 clip latent file",
                 params->continue_latent_path);
        return 0;
    }
    plan->file_t = header[3];
    plan->file_h = header[4];
    plan->file_w = header[5];
    plan->file_audio_t = header[7];
    plan->file_frames = header[8];
    if (plan->file_h != latent_h || plan->file_w != latent_w) {
        snprintf(error, error_size,
                 "the clip to continue was made at another size "
                 "(latent %dx%d, this one %dx%d)", plan->file_w,
                 plan->file_h, latent_w, latent_h);
        return 0;
    }
    int frames = params->continue_frames;
    plan->steps = h3_video_latent_t(frames);
    if (plan->steps < 1 || plan->steps > 17 ||
        h3_video_step_frame(plan->steps) != frames ||
        plan->steps > plan->file_t || frames >= temporal.frame_count ||
        plan->file_frames != h3_video_step_frame(plan->file_t)) {
        snprintf(error, error_size,
                 "cannot pin %d frames of a %d-frame clip at the start of a "
                 "%d-frame one", frames, plan->file_frames,
                 temporal.frame_count);
        return 0;
    }
    for (int step = 0; step < plan->steps; step++)
        plan->keyframes[step] = h3_video_step_frame(step);
    if (!params->forced_audio_path) {
        int audio_frames = params->continue_audio_frames > 0
            ? params->continue_audio_frames : frames;
        plan->audio_t = (int)llround((double)audio_frames *
                                     H3_AUDIO_LATENT_FPS / H3_FPS);
        if (plan->audio_t < 1 || plan->audio_t > plan->file_audio_t) {
            snprintf(error, error_size, "cannot pin %d frames of sound",
                     audio_frames);
            return 0;
        }
        /* The earlier clip's audio step s lies at s - rescale * (its frame
         * count - pinned frames) on this clip's timeline. Its window ends
         * with that clip, snapped to this clip's own audio grid. */
        double shift = (double)H3_AUDIO_LATENT_FPS / H3_FPS *
                       (double)(plan->file_frames - frames);
        plan->audio_start = (double)llround((double)plan->file_audio_t - shift) -
                            plan->audio_t;
    }
    plan->active = 1;
    return 1;
}

/* Appends the pinned steps and sound of the continued clip to the
 * condition rows (patchified video steps, then packed audio rows). */
static int h3_continuation_rows(const h3_params *params,
                                const h3_continuation *plan,
                                float **video_rows, size_t *video_elements,
                                float **audio_rows, size_t *audio_elements,
                                char *error, size_t error_size) {
    size_t plane = (size_t)plan->file_h * plan->file_w;
    size_t video_count = (size_t)24 * plan->file_t * plane;
    size_t audio_count = (size_t)32 * 2 * plan->file_audio_t;
    float *video = malloc(video_count * sizeof(*video));
    float *audio = malloc(audio_count * sizeof(*audio));
    FILE *file = fopen(params->continue_latent_path, "rb");
    int ok = video && audio && file &&
        fseek(file, 9 * sizeof(int32_t), SEEK_SET) == 0 &&
        fread(video, sizeof(*video), video_count, file) == video_count &&
        fread(audio, sizeof(*audio), audio_count, file) == audio_count;
    if (file) fclose(file);
    if (!ok) {
        snprintf(error, error_size, "cannot read %s",
                 params->continue_latent_path);
        free(video); free(audio);
        return 0;
    }
    size_t step_rows = plane / 4 * 96;
    float *step = malloc(24 * plane * sizeof(*step));
    float *grown = realloc(*video_rows, (*video_elements +
                           (size_t)plan->steps * step_rows) * sizeof(*grown));
    if (!step || !grown) {
        snprintf(error, error_size, "out of memory pinning the continued clip");
        free(step); free(video); free(audio);
        if (grown) *video_rows = grown;
        return 0;
    }
    *video_rows = grown;
    for (int index = 0; ok && index < plan->steps; index++) {
        int time = plan->file_t - plan->steps + index;
        for (int channel = 0; channel < 24; channel++)
            memcpy(step + (size_t)channel * plane,
                   video + ((size_t)channel * plan->file_t + (size_t)time) * plane,
                   plane * sizeof(*step));
        ok = h3_dit_patchify_video(step, 24, 1, plan->file_h, plan->file_w,
                                   *video_rows + *video_elements, step_rows);
        *video_elements += step_rows;
    }
    free(step);
    if (ok && plan->audio_t) {
        size_t elements = (size_t)plan->audio_t * 2 * 32;
        float *more = realloc(*audio_rows,
                              (*audio_elements + elements) * sizeof(*more));
        if (!more) {
            ok = 0;
        } else {
            *audio_rows = more;
            float *rows = more + *audio_elements;
            int first = plan->file_audio_t - plan->audio_t;
            for (int stereo = 0; stereo < 2; stereo++)
                for (int time = 0; time < plan->audio_t; time++)
                    for (int channel = 0; channel < 32; channel++)
                        rows[((size_t)stereo * plan->audio_t + time) * 32 +
                             channel] =
                            audio[((size_t)channel * 2 + stereo) *
                                      plan->file_audio_t + first + time];
            *audio_elements += elements;
        }
    }
    free(video); free(audio);
    if (!ok) snprintf(error, error_size, "cannot pin the continued clip");
    return ok;
}

/* The latent a regeneration starts from: its header must describe exactly
 * this generation's latent grid, so the file's video latent can be blended
 * with the noise and its audio latent pinned as-is. */
static int h3_initial_latent_check(const h3_params *params, int latent_h,
                                   int latent_w, h3_temporal_shape temporal,
                                   char *error, size_t error_size) {
    if (!params->initial_latent_path) return 1;
    int32_t header[9];
    FILE *file = fopen(params->initial_latent_path, "rb");
    int read = file && fread(header, sizeof(header), 1, file) == 1;
    if (file) fclose(file);
    if (!read || header[0] != (int32_t)H3_LATENT_FILE_MAGIC ||
        header[1] != 1 || header[2] != 24 || header[6] != 32) {
        snprintf(error, error_size, "%s is not an H3 clip latent file",
                 params->initial_latent_path);
        return 0;
    }
    if (header[3] != temporal.video_t || header[4] != latent_h ||
        header[5] != latent_w || header[7] != temporal.audio_t ||
        header[8] != temporal.frame_count) {
        snprintf(error, error_size,
                 "the initial latent is %dx%d x %d steps (%d frames, %d audio "
                 "steps) but this generation needs %dx%d x %d steps (%d "
                 "frames, %d audio steps)", header[5], header[4], header[3],
                 header[8], header[7], latent_w, latent_h, temporal.video_t,
                 temporal.frame_count, temporal.audio_t);
        return 0;
    }
    return 1;
}

/* Moves the seeded noise in `video` onto the initial latent's trajectory
 * at sigma0 ((1 - sigma0) * latent + sigma0 * noise) and returns the
 * file's audio latent (malloc'd, audio_count values) through *audio_out. */
static int h3_initial_latent_apply(const char *path, float *video,
                                   size_t video_count, float sigma0,
                                   size_t audio_count, float **audio_out,
                                   char *error, size_t error_size) {
    float *latent = malloc(video_count * sizeof(*latent));
    float *audio = malloc(audio_count * sizeof(*audio));
    FILE *file = fopen(path, "rb");
    int ok = latent && audio && file &&
        fseek(file, 9 * sizeof(int32_t), SEEK_SET) == 0 &&
        fread(latent, sizeof(*latent), video_count, file) == video_count &&
        fread(audio, sizeof(*audio), audio_count, file) == audio_count;
    if (file) fclose(file);
    if (!ok) {
        snprintf(error, error_size, "cannot read the initial latent %s", path);
        free(latent); free(audio);
        return 0;
    }
    for (size_t index = 0; index < video_count; index++)
        video[index] = sigma0 * video[index] + (1.0f - sigma0) * latent[index];
    free(latent);
    *audio_out = audio;
    return 1;
}

/* Adds `count` values of `more` before (`first`) or after the `*elements`
 * values of `*rows`. */
static int h3_join_rows(float **rows, size_t *elements,
                        const float *more, size_t count, int first) {
    if (!count) return 1;
    float *joined = malloc((*elements + count) * sizeof(*joined));
    if (!joined) return 0;
    float *existing = joined + (first ? count : 0);
    if (*elements) memcpy(existing, *rows, *elements * sizeof(*joined));
    memcpy(first ? joined : joined + *elements, more, count * sizeof(*joined));
    free(*rows);
    *rows = joined;
    *elements += count;
    return 1;
}

static char *h3_conditioning_key(const char *prompt, const h3_params *params,
                                 int render_width, int render_height,
                                 int ref2va) {
    h3_key key = {0};
    if (!h3_key_append(&key, "mode=%d|prompt=%zu:%s", ref2va,
                       strlen(prompt), prompt)) goto failed;
    if (!ref2va && !params->first_frame && !params->last_frame &&
        !params->continue_latent_path) return key.text;
    if (!h3_key_append(&key, "|render=%dx%d|frames=%d|image-size=%d",
                       render_width, render_height, params->frames,
                       params->reference_image_size) ||
        !h3_key_file(&key, "first", params->first_frame) ||
        !h3_key_file(&key, "last", params->last_frame)) goto failed;
    if (params->continue_latent_path &&
        (!h3_key_append(&key, "|continue=%d:%d:%d", params->continue_frames,
                        params->continue_audio_frames,
                        params->forced_audio_path != NULL) ||
         !h3_key_file(&key, "continue", params->continue_latent_path)))
        goto failed;
    for (size_t index = 0; index < params->reference_count; index++) {
        const h3_reference *reference = &params->references[index];
        if (!h3_key_append(&key, "|ref=%d:%d", reference->kind,
                           reference->include_embedded_audio) ||
            !h3_key_file(&key, "media", reference->path) ||
            !h3_key_file(&key, "audio", reference->audio_path)) goto failed;
        if (reference->kind == H3_REFERENCE_REFMOD &&
            !h3_key_append(&key, "|strength=%.6g|member=%d",
                           (double)reference->strength, reference->member))
            goto failed;
    }
    return key.text;
failed:
    free(key.text);
    return NULL;
}

/* The sigma schedule is baked into a prepared DiT, so its audio shift is
 * part of the prepared key. */
static float h3_audio_sigma_shift(const h3_params *params) {
    return params->audio_sigma_shift > 0.0f ? params->audio_sigma_shift
                                            : (float)H3_AUDIO_SIGMA_SHIFT;
}

/* Forced audio reaches the DiT fully clean - audio sigma 0 at every step.
 * H3_FORCED_AUDIO_MODE=noised instead holds it on its noise trajectory, the
 * way ComfyUI's noise mask does; on the same seeds that synced the mouth
 * clearly worse (lipsync-test/REPORT.md). Changes the sigma schedule, so it
 * is part of the prepared key. */
static int h3_forced_audio_clean(const h3_params *params) {
    const char *mode = h3_option("H3_FORCED_AUDIO_MODE");
    if (params->forced_audio_path) return !(mode && !strcmp(mode, "noised"));
    /* An initial latent's own audio is held clean the same way. */
    return params->initial_latent_path != NULL;
}

static char *h3_prepared_key(const char *conditioning,
                             const h3_params *params,
                             int render_width, int render_height) {
    h3_key key = {0};
    if (!h3_key_append(
            &key,
            "%s|shape=%dx%dx%d|steps=%d|layers=%d|reuse-core=%d|reduce=%d"
            "|row-fc2=%d|reference-rope=%d|ssd-streaming=%d"
            "|slow=%d%d%d%d%d%d%d%d%d%d|audio-shift=%.6g|clean-audio=%d",
            conditioning, render_width, render_height, params->frames,
            params->steps, params->dit_layers, params->core_reuse,
            params->token_reduction, params->use_int8_row_fc2,
            params->use_reference_rope,
            params->ssd_streaming,
            params->use_slower_bf16_mlp,
            params->use_slower_bf16_qkv,
            params->use_slower_bf16_attention_output,
            params->use_slower_row_major_attention_output,
            params->use_slower_unfused_int8_inputs,
            params->use_slower_unfused_qkv_rope,
            params->use_slower_scalar_qkv_rms,
            params->use_slower_uncached_int8_scales,
            params->use_slower_dynamic_fc1_k,
            params->use_slower_grouped_quantizer,
            (double)h3_audio_sigma_shift(params),
            h3_forced_audio_clean(params))) {
        free(key.text);
        return NULL;
    }
    /* The DiT's weights carry its LoRA patches, so a different adapter set
     * or strength needs a different prepared DiT. */
    for (size_t index = 0; index < params->lora_count; index++)
        if (!h3_key_append(&key, "|lora=%s@%.6g", params->loras[index].path,
                           (double)params->loras[index].strength)) {
            free(key.text);
            return NULL;
        }
    /* A custom video schedule changes the baked AdaLN values too. */
    for (int index = 0; params->video_sigmas &&
                        index < params->video_sigma_count; index++)
        if (!h3_key_append(&key, "%s%.7g", index ? "," : "|sigmas=",
                           (double)params->video_sigmas[index])) {
            free(key.text);
            return NULL;
        }
    return key.text;
}

static int h3_text_embedding_copy(h3_text_embedding *destination,
                                  const h3_text_embedding *source) {
    memset(destination, 0, sizeof(*destination));
    if (!source || !source->tokens || !source->width || !source->values ||
        source->tokens > SIZE_MAX / source->width) return 0;
    size_t elements = source->tokens * source->width;
    if (elements > SIZE_MAX / sizeof(*destination->values)) return 0;
    destination->values = malloc(elements * sizeof(*destination->values));
    if (source->tags)
        destination->tags = malloc(source->tokens * sizeof(*destination->tags));
    if (!destination->values || (source->tags && !destination->tags)) {
        h3_text_embedding_free(destination);
        return 0;
    }
    memcpy(destination->values, source->values,
           elements * sizeof(*destination->values));
    if (source->tags)
        memcpy(destination->tags, source->tags,
               source->tokens * sizeof(*destination->tags));
    destination->tokens = source->tokens;
    destination->width = source->width;
    destination->gpu_stats = source->gpu_stats;
    return 1;
}

static int h3_conditioning_cache_store(
        h3_ctx *ctx, const char *key, const h3_text_embedding *text,
        const float *video, size_t video_elements,
        const float *audio, size_t audio_elements,
        const h3_layout_ref *references, size_t reference_count,
        int conditioned) {
    h3_text_embedding copy;
    if (!h3_text_embedding_copy(&copy, text)) return 0;
    float *video_copy = NULL;
    float *audio_copy = NULL;
    h3_layout_ref *reference_copy = NULL;
    char *key_copy = strdup(key);
    if (video_elements) {
        video_copy = malloc(video_elements * sizeof(*video_copy));
        if (video_copy) memcpy(video_copy, video,
                               video_elements * sizeof(*video_copy));
    }
    if (audio_elements) {
        audio_copy = malloc(audio_elements * sizeof(*audio_copy));
        if (audio_copy) memcpy(audio_copy, audio,
                               audio_elements * sizeof(*audio_copy));
    }
    if (reference_count) {
        reference_copy = malloc(reference_count * sizeof(*reference_copy));
        if (reference_copy) memcpy(reference_copy, references,
                                   reference_count * sizeof(*reference_copy));
    }
    if (!key_copy || (video_elements && !video_copy) ||
        (audio_elements && !audio_copy) ||
        (reference_count && !reference_copy)) {
        free(key_copy); free(video_copy); free(audio_copy); free(reference_copy);
        h3_text_embedding_free(&copy);
        return 0;
    }
    h3_conditioning_cache_clear(ctx);
    ctx->conditioning_key = key_copy;
    ctx->conditioning_tokens = copy.tokens;
    ctx->conditioning_width = copy.width;
    ctx->conditioning_values = copy.values;
    ctx->conditioning_tags = copy.tags;
    ctx->conditioning_video_rows = video_copy;
    ctx->conditioning_video_elements = video_elements;
    ctx->conditioning_audio_rows = audio_copy;
    ctx->conditioning_audio_elements = audio_elements;
    ctx->conditioning_references = reference_copy;
    ctx->conditioning_reference_count = reference_count;
    ctx->conditioning_present = conditioned;
    return 1;
}

static int h3_conditioning_cache_load(
        const h3_ctx *ctx, h3_text_embedding *text,
        float **video, size_t *video_elements,
        float **audio, size_t *audio_elements,
        h3_layout_ref **references, size_t *reference_count,
        int *conditioned) {
    h3_text_embedding source = {
        ctx->conditioning_tokens, ctx->conditioning_width,
        ctx->conditioning_values, {0}, ctx->conditioning_tags};
    if (!h3_text_embedding_copy(text, &source)) return 0;
    *video = NULL;
    *audio = NULL;
    *references = NULL;
    *video_elements = ctx->conditioning_video_elements;
    *audio_elements = ctx->conditioning_audio_elements;
    *reference_count = ctx->conditioning_reference_count;
    *conditioned = ctx->conditioning_present;
    if (*video_elements) {
        *video = malloc(*video_elements * sizeof(**video));
        if (*video) memcpy(*video, ctx->conditioning_video_rows,
                           *video_elements * sizeof(**video));
    }
    if (*audio_elements) {
        *audio = malloc(*audio_elements * sizeof(**audio));
        if (*audio) memcpy(*audio, ctx->conditioning_audio_rows,
                           *audio_elements * sizeof(**audio));
    }
    if (*reference_count) {
        *references = malloc(*reference_count * sizeof(**references));
        if (*references) memcpy(*references, ctx->conditioning_references,
                                *reference_count * sizeof(**references));
    }
    if ((*video_elements && !*video) || (*audio_elements && !*audio) ||
        (*reference_count && !*references)) {
        h3_text_embedding_free(text);
        free(*video); free(*audio); free(*references);
        *video = NULL; *audio = NULL; *references = NULL;
        return 0;
    }
    return 1;
}

static void h3_augment_span(float *values, size_t count, uint64_t seed) {
    h3_rng rng;
    h3_rng_seed(&rng, seed);
    for (size_t index = 0; index < count; index++)
        values[index] = 0.999f * values[index] +
                        0.001f * h3_rng_normal(&rng);
}

static int h3_augment_conditions(const h3_params *params, int ref2va,
                                 int render_width, int render_height,
                                 const h3_layout_ref *references,
                                 size_t keyframe_count,
                                 size_t keyframe_audio_elements,
                                 float *video, size_t video_elements,
                                 float *audio, size_t audio_elements) {
    size_t video_offset = 0;
    size_t audio_offset = 0;
    /* One-step keyframes (first/last frame, a continuation's pinned steps)
     * and a continuation's audio window come first, then any Ref2VA
     * references (the order h3_layout_build packs them in). */
    int latent_w, latent_h;
    h3_latent_canvas(render_width, render_height, &latent_w, &latent_h);
    size_t step_span = (size_t)h3_video_encoder_latent_t(1) *
                       (size_t)latent_h * (size_t)latent_w / 4 * 96;
    size_t count = keyframe_count;
    if (count && (step_span > SIZE_MAX / count ||
                  step_span * count > video_elements ||
                  (!ref2va && step_span * count != video_elements)))
        return 0;
    for (size_t index = 0; index < count; index++) {
        h3_augment_span(video + video_offset, step_span, params->seed);
        video_offset += step_span;
    }
    if (!ref2va) {
        if (audio_elements)
            h3_augment_span(audio, audio_elements, params->seed + 1);
        return video_offset == video_elements;
    }
    if (keyframe_audio_elements) {
        if (keyframe_audio_elements > audio_elements) return 0;
        h3_augment_span(audio, keyframe_audio_elements, params->seed + 1);
        audio_offset = keyframe_audio_elements;
    }
    for (size_t index = 0; index < params->reference_count; index++) {
        const h3_layout_ref *reference = &references[index];
        if (reference->kind != H3_LAYOUT_REF_AUDIO) {
            size_t span = (size_t)reference->latent_t *
                (size_t)reference->latent_h * (size_t)reference->latent_w /
                4 * 96;
            if (span > video_elements - video_offset) return 0;
            h3_augment_span(video + video_offset, span, params->seed);
            video_offset += span;
        }
        if (reference->audio_t) {
            size_t span = (size_t)reference->audio_t * 2 * 32;
            if (span > audio_elements - audio_offset) return 0;
            h3_augment_span(audio + audio_offset, span, params->seed + 1);
            audio_offset += span;
        }
    }
    return video_offset == video_elements && audio_offset == audio_elements;
}

void h3_set_error(h3_ctx *ctx, const char *format, ...) {
    char *destination = ctx ? ctx->error : h3_global_error;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(destination, 512, format, arguments);
    va_end(arguments);
}

/* Matches the local `fail()` helper every other module in this codebase
 * defines for its own `char *error, size_t error_size` out-parameters -
 * needed here now that h3_stream_encode_chunk (below) has to fill one of
 * its own rather than going through h3_set_error's ctx-bound buffer. */
static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int h3_is_file(const char *path) {
    struct stat status;
    return stat(path, &status) == 0 && S_ISREG(status.st_mode);
}

static char *h3_path(const char *root, const char *relative) {
    size_t size = strlen(root) + strlen(relative) + 2;
    char *result = malloc(size);
    if (result) snprintf(result, size, "%s/%s", root, relative);
    return result;
}

static int h3_require_file(h3_ctx *ctx, const char *relative) {
    char *path = h3_path(ctx->model_dir, relative);
    if (!path) {
        h3_set_error(ctx, "out of memory resolving model path");
        return 0;
    }
    int exists = h3_is_file(path);
    if (!exists) h3_set_error(ctx, "missing required model file: %s", path);
    free(path);
    return exists;
}

static int h3_inventory(h3_ctx *ctx, const char *relative,
                        h3_component_info *info) {
    char *path = h3_path(ctx->model_dir, relative);
    if (!path) {
        h3_set_error(ctx, "out of memory resolving component path");
        return 0;
    }
    char detail[384];
    int ok = h3_st_inventory_dir(path, info, detail, sizeof(detail));
    if (!ok) h3_set_error(ctx, "%s", detail);
    free(path);
    return ok;
}

/* 1 when a component directory holds a file with this suffix. */
static int h3_dir_has_suffix(const char *root, const char *component,
                             const char *suffix) {
    char *directory = h3_path(root, component);
    if (!directory) return 0;
    DIR *stream = opendir(directory);
    free(directory);
    if (!stream) return 0;
    int found = 0;
    size_t suffix_length = strlen(suffix);
    struct dirent *entry;
    while (!found && (entry = readdir(stream)) != NULL) {
        size_t length = strlen(entry->d_name);
        found = entry->d_name[0] != '.' && length > suffix_length &&
                !strcmp(entry->d_name + length - suffix_length, suffix);
    }
    closedir(stream);
    return found;
}

/* An imported (GGUF) Ref2VA transformer has no safetensors index. */
static int h3_dir_has_gguf(const char *root, const char *component) {
    return h3_dir_has_suffix(root, component, ".gguf");
}

h3_ctx *h3_load_dir(const char *model_dir) {
    h3_global_error[0] = '\0';
    if (!model_dir || !*model_dir) {
        h3_set_error(NULL, "model directory is required");
        return NULL;
    }
    h3_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        h3_set_error(NULL, "out of memory creating H3 context");
        return NULL;
    }
    ctx->model_dir = strdup(model_dir);
    if (!ctx->model_dir) {
        h3_set_error(NULL, "out of memory copying model path");
        free(ctx);
        return NULL;
    }
    if (!h3_require_file(ctx, "FL2VA/transformer/config.json") ||
        !h3_require_file(ctx, "FL2VA/tokenizer/tokenizer.json") ||
        !h3_inventory(ctx, "FL2VA/text_encoder", &ctx->model.text_encoder) ||
        !h3_inventory(ctx, "FL2VA/transformer", &ctx->model.fl2va_transformer) ||
        !h3_inventory(ctx, "FL2VA/video_vae/source", &ctx->model.video_vae) ||
        !h3_inventory(ctx, "FL2VA/audio_vae", &ctx->model.audio_vae)) {
        snprintf(h3_global_error, sizeof(h3_global_error), "%s", ctx->error);
        h3_free(ctx);
        return NULL;
    }
    /* Ref2VA is selected only by ordered-reference requests. Keep prompt-only
     * FL2VA usable while that optional 62 GiB checkpoint is not installed. */
    char *ref_index = h3_path(
        ctx->model_dir, "Ref2VA/transformer/model.safetensors.index.json");
    if (!ref_index) {
        h3_set_error(ctx, "out of memory resolving optional Ref2VA path");
        snprintf(h3_global_error, sizeof(h3_global_error), "%s", ctx->error);
        h3_free(ctx);
        return NULL;
    }
    /* ... nor has a single-file ComfyUI checkpoint (model.safetensors). */
    int has_ref2va = h3_is_file(ref_index) ||
                     h3_dir_has_gguf(ctx->model_dir, "Ref2VA/transformer") ||
                     h3_dir_has_suffix(ctx->model_dir, "Ref2VA/transformer",
                                       ".safetensors");
    free(ref_index);
    if (has_ref2va && !h3_inventory(
            ctx, "Ref2VA/transformer", &ctx->model.ref2va_transformer)) {
        snprintf(h3_global_error, sizeof(h3_global_error), "%s", ctx->error);
        h3_free(ctx);
        return NULL;
    }
    char metal_error[256];
    if (!h3_metal_probe(&ctx->device, metal_error, sizeof(metal_error))) {
        h3_set_error(ctx, "%s", metal_error);
        snprintf(h3_global_error, sizeof(h3_global_error), "%s", ctx->error);
        h3_free(ctx);
        return NULL;
    }
    return ctx;
}

void h3_free(h3_ctx *ctx) {
    if (!ctx) return;
    h3_cache_clear(ctx);
    h3_dit_prep_free(ctx->dit_prep);
    free(ctx->model_dir);
    free(ctx);
}

const char *h3_last_error(const h3_ctx *ctx) {
    return ctx ? ctx->error : h3_global_error;
}

const h3_device_info *h3_device(const h3_ctx *ctx) {
    return ctx ? &ctx->device : NULL;
}

const h3_model_info *h3_model(const h3_ctx *ctx) {
    return ctx ? &ctx->model : NULL;
}

static int h3_valid_params(h3_ctx *ctx, const h3_params *params) {
    if (!params) {
        h3_set_error(ctx, "generation parameters are required");
        return 0;
    }
    if (params->width < 32 || params->height < 32 ||
        params->width % H3_CANVAS_MULTIPLE ||
        params->height % H3_CANVAS_MULTIPLE) {
        h3_set_error(ctx, "width and height must be multiples of 32 and at least 32");
        return 0;
    }
    /* The released limit is on the canvas the DiT runs at. An internal
     * render canvas is upscaled on the host after decoding, so its output
     * may be larger: up to twice the limit's size per side. */
    int64_t output_pixels = (int64_t)params->width * params->height;
    int64_t dit_pixels = params->render_width
        ? (int64_t)params->render_width * params->render_height
        : output_pixels;
    if (dit_pixels > H3_MAX_PIXELS) {
        h3_set_error(ctx, "canvas exceeds the released 768*1344 pixel limit");
        return 0;
    }
    if (output_pixels > 4 * (int64_t)H3_MAX_PIXELS) {
        h3_set_error(ctx,
            "upscaled output exceeds twice the 768*1344 limit per side");
        return 0;
    }
    if ((params->render_width == 0) != (params->render_height == 0)) {
        h3_set_error(ctx, "render width and height must be set together");
        return 0;
    }
    if (params->render_width) {
        if (params->render_width < 32 || params->render_height < 32 ||
            params->render_width % H3_CANVAS_MULTIPLE ||
            params->render_height % H3_CANVAS_MULTIPLE ||
            params->render_width > params->width ||
            params->render_height > params->height ||
            (int64_t)params->render_width * params->height !=
                (int64_t)params->render_height * params->width) {
            h3_set_error(ctx,
                "internal render canvas must be same-aspect multiples of 32 "
                "no larger than the output canvas");
            return 0;
        }
    }
    if (params->frames < 5 || h3_align_frame_count(params->frames) > 362) {
        h3_set_error(ctx, "frames must align within the released 5..362 range");
        return 0;
    }
    if (params->steps < 2 || params->steps > H3_MAX_STEPS) {
        h3_set_error(ctx, "denoising steps must be in [2, 1000]");
        return 0;
    }
    if (params->denoise_reuse < 1 || params->denoise_reuse > 3) {
        h3_set_error(ctx, "denoise reuse must be in [1, 3]");
        return 0;
    }
    if (params->dit_layers < H3_MIN_DIT_LAYERS ||
        params->dit_layers > H3_DEFAULT_DIT_LAYERS) {
        h3_set_error(ctx, "DiT layers must be in [35, 50]");
        return 0;
    }
    if (params->core_reuse < 1 || params->core_reuse > 6) {
        h3_set_error(ctx, "core reuse must be in [1, 6]");
        return 0;
    }
    if (params->token_reduction != 0 && params->token_reduction != 1) {
        h3_set_error(ctx, "token reduction must be zero or one");
        return 0;
    }
    if (params->use_int8_row_fc2 != 0 &&
        params->use_int8_row_fc2 != 1) {
        h3_set_error(ctx, "int8 row FC2 must be zero or one");
        return 0;
    }
    if (params->use_reference_rope != 0 &&
        params->use_reference_rope != 1) {
        h3_set_error(ctx, "reference RoPE must be zero or one");
        return 0;
    }
    if (params->ssd_streaming != 0 && params->ssd_streaming != 1) {
        h3_set_error(ctx, "SSD streaming must be zero or one");
        return 0;
    }
    if (params->ssd_streaming && params->use_int8_row_fc2) {
        h3_set_error(ctx, "SSD streaming uses original BF16 weights and cannot "
                         "be combined with int8 row FC2");
        return 0;
    }
    if (params->use_int8_row_fc2 && params->use_slower_bf16_mlp) {
        h3_set_error(ctx, "int8 row FC2 cannot be combined with the BF16 MLP");
        return 0;
    }
    if (params->use_int8_row_fc2 && !h3_device(ctx)->metal4) {
        h3_set_error(ctx, "int8 row FC2 requires an M5-class Metal 4 GPU");
        return 0;
    }
    if (params->preview_denoise != 0 && params->preview_denoise != 1) {
        h3_set_error(ctx, "denoising preview must be zero or one");
        return 0;
    }
    if (params->latent_preview_every < 0 ||
        (params->latent_preview_every > 0 && !params->on_latent_preview)) {
        h3_set_error(ctx, "a latent preview needs a positive interval and a callback");
        return 0;
    }
    if (params->preview_denoise && !params->on_frame) {
        h3_set_error(ctx, "denoising preview requires a frame callback");
        return 0;
    }
    if (params->core_reuse > 1 && params->denoise_reuse > 1) {
        h3_set_error(ctx, "core reuse and denoiser reuse cannot be combined");
        return 0;
    }
    if (params->fast_attention != 0 && params->fast_attention != 1) {
        h3_set_error(ctx, "fast attention must be zero or one");
        return 0;
    }
    const h3_sampler_info *sampler = h3_sampler(params->sampler);
    if (!sampler) {
        h3_set_error(ctx, "sampler must be Euler or re-noise");
        return 0;
    }
    if (sampler->evaluates_every_step && params->denoise_reuse != 1) {
        h3_set_error(ctx, "the re-noise sampler evaluates every step and "
                         "cannot be combined with denoiser reuse");
        return 0;
    }
    if (params->save_latent_path && !*params->save_latent_path) {
        h3_set_error(ctx, "save_latent_path is empty");
        return 0;
    }
    if (params->continue_latent_path) {
        int frames = params->continue_frames;
        if (!*params->continue_latent_path ||
            (frames != 5 && frames != 22 && frames != 39 && frames != 56) ||
            params->continue_audio_frames < 0 ||
            params->continue_audio_frames > 240) {
            h3_set_error(ctx, "continuation needs a latent file, 5, 22, 39 "
                              "or 56 frames, and 0-240 frames of sound");
            return 0;
        }
        /* References are fine: they shape who and what appears, while the
         * pinned steps fix where the clip starts (as ComfyUI combines
         * keyframes with Ref2VA references). */
        if (params->first_frame) {
            h3_set_error(ctx, "a continuation already fixes the start of the "
                              "clip; it cannot take a first frame");
            return 0;
        }
    }
    if (params->forced_audio_path &&
        (!*params->forced_audio_path ||
         !isfinite(params->forced_audio_offset) ||
         params->forced_audio_offset < 0.0)) {
        h3_set_error(ctx, "forced audio needs a file and an offset >= 0");
        return 0;
    }
    if (params->initial_latent_path &&
        (!*params->initial_latent_path || params->continue_latent_path)) {
        h3_set_error(ctx, "an initial latent needs a file and cannot be "
                          "combined with a continuation");
        return 0;
    }
    if (params->video_sigmas || params->video_sigma_count) {
        int count = params->video_sigma_count;
        int ok = params->video_sigmas && count >= 3 &&
                 count - 1 <= H3_MAX_STEPS && params->steps == count - 1 &&
                 isfinite(params->video_sigmas[0]) &&
                 params->video_sigmas[0] > 0.0f &&
                 params->video_sigmas[0] <= 1.0f &&
                 params->video_sigmas[count - 1] == 0.0f;
        for (int index = 1; ok && index < count; index++)
            ok = isfinite(params->video_sigmas[index]) &&
                 params->video_sigmas[index] < params->video_sigmas[index - 1];
        if (!ok) {
            h3_set_error(ctx, "video sigmas must be steps + 1 finite values, "
                              "strictly decreasing from (0, 1] to exactly 0");
            return 0;
        }
    }
    if (!(params->audio_sigma_shift == 0.0f ||
          (params->audio_sigma_shift > 0.0f &&
           params->audio_sigma_shift <= 100.0f))) {
        h3_set_error(ctx, "audio sigma shift must be 0 (default) or in (0, 100]");
        return 0;
    }
    if (params->fast_attention && !h3_fast_attention_available()) {
        h3_set_error(ctx, "fast attention mode is not available: this build "
                     "lacks the ccv backend or this GPU lacks neural "
                     "matrix accelerators");
        return 0;
    }
    if (params->lora_count && !params->loras) {
        h3_set_error(ctx, "lora_count is nonzero but loras is NULL");
        return 0;
    }
    if (params->lora_count > H3_MAX_LORAS) {
        h3_set_error(ctx, "at most %d LoRA adapters can be stacked",
                     H3_MAX_LORAS);
        return 0;
    }
    for (size_t index = 0; index < params->lora_count; index++) {
        if (!params->loras[index].path || !*params->loras[index].path ||
            !isfinite(params->loras[index].strength)) {
            h3_set_error(ctx, "LoRA %zu needs a path and a finite strength",
                         index + 1);
            return 0;
        }
    }
    if (params->reference_count && !params->references) {
        h3_set_error(ctx, "reference_count is nonzero but references is NULL");
        return 0;
    }
    if (params->reference_count > 12) {
        h3_set_error(ctx, "Ref2VA supports at most 12 references");
        return 0;
    }
    if (params->reference_image_size != H3_REFERENCE_IMAGE_MATCH &&
        params->reference_image_size != H3_REFERENCE_IMAGE_MAX) {
        h3_set_error(ctx, "unknown reference image sizing policy");
        return 0;
    }
    if (params->reference_count && (params->first_frame || params->last_frame)) {
        h3_set_error(ctx, "full references cannot be combined with frame anchors");
        return 0;
    }
    size_t images = 0, videos = 0, audio_inputs = 0, visual = 0;
    for (size_t index = 0; index < params->reference_count; index++) {
        const h3_reference *reference = &params->references[index];
        if (!reference->path || !*reference->path) {
            h3_set_error(ctx, "reference %zu has no input path", index + 1);
            return 0;
        }
        switch (reference->kind) {
        case H3_REFERENCE_IMAGE:
            images++; visual++;
            break;
        case H3_REFERENCE_VIDEO:
            videos++; visual++;
            if (reference->include_embedded_audio) audio_inputs++;
            break;
        case H3_REFERENCE_AUDIO:
            audio_inputs++;
            break;
        case H3_REFERENCE_VIDEO_AUDIO:
            videos++; visual++; audio_inputs++;
            if (!reference->audio_path || !*reference->audio_path) {
                h3_set_error(ctx,
                    "video+audio reference %zu has no soundtrack path",
                    index + 1);
                return 0;
            }
            break;
        case H3_REFERENCE_REFMOD:
            /* Counted against the image/video/audio limits once its file
             * says what it holds (h3_refmod_reference); here it stands in
             * for the visual Ref2VA needs. */
            visual++;
            if (!isfinite(reference->strength) || reference->strength <= 0.0f ||
                reference->strength > 1.0f) {
                h3_set_error(ctx,
                    "RefMod %zu needs a strength above 0 and at most 1",
                    index + 1);
                return 0;
            }
            if (reference->member < -1 ||
                reference->member >= H3_REFMOD_MAX_MEMBERS) {
                h3_set_error(ctx, "RefMod %zu has an invalid member index",
                             index + 1);
                return 0;
            }
            break;
        default:
            h3_set_error(ctx, "reference %zu has an unknown type", index + 1);
            return 0;
        }
    }
    if (images > 9 || videos > 3 || audio_inputs > 3) {
        h3_set_error(ctx,
            "Ref2VA limits are 9 images, 3 videos, and 3 audio inputs");
        return 0;
    }
    if (params->reference_count && !visual) {
        h3_set_error(ctx, "reference audio requires an image or video reference");
        return 0;
    }
    return 1;
}

typedef struct {
    h3_ctx *ctx;
    const h3_params *params;
    int cancelled;
} h3_generation_progress;

/* H3_DEBUG_HASHES=1: log an FNV-1a hash of each intermediate result, to find
 * where two runs with the same request and seed first diverge. Off by
 * default; costs one pass over each buffer when on. */
static int h3_debug_hashes_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) enabled = h3_option("H3_DEBUG_HASHES") != NULL;
    return enabled;
}

static uint64_t h3_fnv1a(uint64_t hash, const void *data, size_t bytes) {
    const unsigned char *bytes_ptr = data;
    for (size_t i = 0; i < bytes; i++) {
        hash ^= bytes_ptr[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static void h3_debug_hash(const char *name, const void *data, size_t bytes) {
    if (!h3_debug_hashes_enabled() || !data) return;
    fprintf(stderr, "h3: hash %s %016llx (%zu bytes)\n", name,
            (unsigned long long)h3_fnv1a(1469598103934665603ULL, data, bytes),
            bytes);
}

/* The RGB frames handed to the encoder, hashed as one stream in frame order,
 * so the value does not depend on how the decoder split the video into
 * chunks (streamed) or not (monolithic). */
typedef struct {
    uint64_t hash;
    size_t frames;
} h3_debug_video_hash;

static void h3_debug_hash_frames(h3_debug_video_hash *state,
                                 const uint8_t *rgb, size_t frames,
                                 size_t frame_bytes) {
    if (!h3_debug_hashes_enabled() || !rgb) return;
    if (!state->frames) state->hash = 1469598103934665603ULL;
    state->hash = h3_fnv1a(state->hash, rgb, frames * frame_bytes);
    state->frames += frames;
}

static void h3_debug_hash_video_done(const h3_debug_video_hash *state) {
    if (!h3_debug_hashes_enabled()) return;
    fprintf(stderr, "h3: hash rgb video %016llx (%zu frames)\n",
            (unsigned long long)state->hash, state->frames);
}

static void h3_progress_emit(h3_generation_progress *state, const char *phase,
                             int completed, int total) {
    if (!state || state->cancelled) return;
    h3_memory_phase(phase);
    if (state->params->cancel_requested && *state->params->cancel_requested) {
        state->cancelled = 1;
        h3_set_error(state->ctx, "generation cancelled during %s", phase);
        return;
    }
    if (!state->params->on_progress) return;
    if (state->params->on_progress(phase, completed, total,
                                   state->params->callback_opaque)) {
        state->cancelled = 1;
        h3_set_error(state->ctx, "generation cancelled during %s", phase);
    }
}

static void h3_text_progress_bridge(int completed, int total, void *opaque) {
    h3_progress_emit(opaque, "text encoder", completed, total);
}

static void h3_dit_progress_bridge(const char *phase, int completed, int total,
                                   void *opaque) {
    h3_progress_emit(opaque, phase, completed, total);
}

static void h3_vae_progress_bridge(int completed, int total, void *opaque) {
    h3_progress_emit(opaque, "video VAE load", completed, total);
}

static void h3_preview_vae_progress_bridge(int completed, int total,
                                           void *opaque) {
    h3_progress_emit(opaque, "preview VAE load", completed, total);
}

static void h3_audio_vae_progress_bridge(int completed, int total,
                                         void *opaque) {
    h3_progress_emit(opaque, "audio VAE", completed, total);
}

static void h3_audio_encoder_progress_bridge(int completed, int total,
                                             void *opaque) {
    h3_progress_emit(opaque, "audio VAE encoder", completed, total);
}

static void h3_video_encoder_progress_bridge(int completed, int total,
                                             void *opaque) {
    h3_progress_emit(opaque, "video VAE encoder", completed, total);
}

static h3_video_vae_decoder *h3_acquire_video_decoder(
        h3_ctx *ctx, const char *key, const char *weight_directory,
        int latent_height, int latent_width, h3_video_vae_progress progress,
        void *progress_opaque, int *cached, char *error, size_t error_size) {
    *cached = 0;
    if (h3_cache_target(ctx, H3_CACHE_DECODER) && ctx->video_decoder &&
        ctx->video_decoder_key && !strcmp(ctx->video_decoder_key, key)) {
        *cached = 1;
        fprintf(stderr, "h3: video VAE cache hit\n");
        return ctx->video_decoder;
    }
    if (h3_cache_target(ctx, H3_CACHE_DECODER)) {
        h3_video_vae_decoder_free(ctx->video_decoder);
        ctx->video_decoder = NULL;
        free(ctx->video_decoder_key);
        ctx->video_decoder_key = NULL;
    }
    h3_video_vae_decoder *decoder = h3_video_vae_decoder_load(
        weight_directory, "h3_shaders.metal", latent_height, latent_width,
        progress, progress_opaque, error, error_size);
    if (!decoder || !h3_cache_target(ctx, H3_CACHE_DECODER)) return decoder;
    char *key_copy = strdup(key);
    if (!key_copy) {
        fprintf(stderr, "h3: warning: could not retain video VAE cache key\n");
        return decoder;
    }
    ctx->video_decoder = decoder;
    ctx->video_decoder_key = key_copy;
    *cached = 1;
    fprintf(stderr, "h3: video VAE cache miss; decoder retained\n");
    return decoder;
}

static void h3_vision_progress_bridge(int completed, int total, void *opaque) {
    h3_progress_emit(opaque, "Qwen vision", completed, total);
}

static uint8_t *h3_rgb_f32_to_u8(const float *rgb, size_t count) {
    uint8_t *output = malloc(count);
    if (!output) return NULL;
    for (size_t index = 0; index < count; index++) {
        float scaled = rgb[index] * 255.0f;
        if (scaled < 0.0f) scaled = 0.0f;
        if (scaled > 255.0f) scaled = 255.0f;
        output[index] = (uint8_t)lrintf(scaled);
    }
    return output;
}

typedef struct {
    h3_generation_progress *progress;
    h3_video_vae_decoder *decoder;
    int latent_t;
    int latent_h;
    int latent_w;
    int output_frames;
    int output_width;
    int output_height;
    int failed;
} h3_live_preview;

/* h3_dit's latent preview handed on as h3_params.on_latent_preview. */
static int h3_deliver_latent_preview(int completed_steps, int total_steps,
                                     const float *clean_video, int time,
                                     int height, int width, void *opaque) {
    const h3_params *params = opaque;
    h3_latent_preview preview = {clean_video, 24, time, height, width,
                                 completed_steps, total_steps};
    return params->on_latent_preview(&preview, params->callback_opaque);
}

static int h3_deliver_denoise_preview(int completed_steps, int total_steps,
                                      const float *video_latent,
                                      size_t video_elements, void *opaque) {
    h3_live_preview *preview = opaque;
    if (!preview || !preview->progress || !preview->decoder || !video_latent) {
        if (preview && preview->progress)
            h3_set_error(preview->progress->ctx,
                         "invalid denoising preview latent");
        if (preview) preview->failed = 1;
        return 1;
    }
    size_t expected = (size_t)24 * (size_t)preview->latent_t *
                      (size_t)preview->latent_h * (size_t)preview->latent_w;
    if (video_elements != expected) {
        h3_set_error(preview->progress->ctx,
                     "invalid denoising preview latent size");
        preview->failed = 1;
        return 1;
    }
    char detail[512];
    h3_video_frames decoded;
    memset(&decoded, 0, sizeof(decoded));
    int frame_index = 0;
    if (!h3_video_vae_decoder_preview(
            preview->decoder, video_latent, preview->latent_t,
            &decoded, &frame_index, detail, sizeof(detail))) {
        h3_set_error(preview->progress->ctx,
                     "cannot decode denoising preview: %s", detail);
        preview->failed = 1;
        return 1;
    }
    size_t count = (size_t)decoded.width * (size_t)decoded.height * 3;
    uint8_t *rgb = h3_rgb_f32_to_u8(decoded.rgb, count);
    h3_video_frames_free(&decoded);
    if (!rgb) {
        h3_set_error(preview->progress->ctx,
                     "out of memory converting denoising preview");
        preview->failed = 1;
        return 1;
    }
    int source_width = preview->latent_w * H3_VAE_SPATIAL_RATIO;
    int source_height = preview->latent_h * H3_VAE_SPATIAL_RATIO;
    if (source_width != preview->output_width ||
        source_height != preview->output_height) {
        uint8_t *resized = NULL;
        if (!h3_resize_rgb24_high_quality(
                rgb, 1, source_width, source_height,
                preview->output_width, preview->output_height, &resized)) {
            free(rgb);
            h3_set_error(preview->progress->ctx,
                         "cannot resize denoising preview");
            preview->failed = 1;
            return 1;
        }
        free(rgb);
        rgb = resized;
    }
    h3_frame frame = {
        preview->output_width, preview->output_height,
        preview->output_width * 3, rgb, frame_index,
        preview->output_frames, completed_steps - 1, total_steps
    };
    int cancelled = preview->progress->params->on_frame(
        &frame, preview->progress->params->callback_opaque);
    free(rgb);
    if (cancelled) {
        h3_set_error(preview->progress->ctx,
                     "generation cancelled during denoising preview %d",
                     completed_steps);
        preview->failed = 1;
        return 1;
    }
    return 0;
}

/* Qwen consumes reference video as time-major two-frame blocks, while the
 * visual VAE and media boundary retain channel-major [3,T,H,W]. */
static float *h3_extract_vision_pair(const float *pixels, int frames,
                                     int height, int width,
                                     int first, int second) {
    if (!pixels || frames < 1 || height < 1 || width < 1 || first < 0 ||
        second < 0 || first >= frames || second >= frames) return NULL;
    size_t area = (size_t)height * (size_t)width;
    if (area > SIZE_MAX / 6 || 6 * area > SIZE_MAX / sizeof(float)) return NULL;
    float *pair = malloc(6 * area * sizeof(*pair));
    if (!pair) return NULL;
    const int times[2] = {first, second};
    for (int time = 0; time < 2; time++)
        for (int channel = 0; channel < 3; channel++) {
            size_t source = ((size_t)channel * (size_t)frames +
                             (size_t)times[time]) * area;
            size_t destination = ((size_t)time * 3 +
                                  (size_t)channel) * area;
            memcpy(pair + destination, pixels + source,
                   area * sizeof(*pair));
        }
    return pair;
}

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

static int h3_stream_encode_chunk(void *opaque, const float *rgb,
                                  int frame_count, int height, int width,
                                  char *error, size_t error_size) {
    h3_stream_encode_ctx *stream = opaque;
    size_t count = (size_t)frame_count * (size_t)height * (size_t)width * 3;
    uint8_t *chunk8 = h3_rgb_f32_to_u8(rgb, count);
    if (!chunk8) {
        fail(error, error_size,
             "out of memory converting generated RGB frames");
        return 0;
    }
    if (stream->need_resize) {
        uint8_t *resized = NULL;
        if (!h3_resize_rgb24_high_quality(chunk8, frame_count, width, height,
                stream->output_width, stream->output_height, &resized)) {
            free(chunk8);
            fail(error, error_size, "cannot resize generated RGB frames");
            return 0;
        }
        free(chunk8);
        chunk8 = resized;
    }
    h3_debug_hash_frames(&stream->rgb_hash, chunk8, (size_t)frame_count,
                         (size_t)stream->output_width *
                             (size_t)stream->output_height * 3);
    if (stream->on_frame) {
        size_t frame_bytes = (size_t)stream->output_width *
            (size_t)stream->output_height * 3;
        for (int index = 0; index < frame_count; index++) {
            h3_frame frame = {
                stream->output_width, stream->output_height,
                stream->output_width * 3,
                chunk8 + (size_t)index * frame_bytes,
                stream->emitted + index, stream->total_frames, -1, 0
            };
            if (stream->on_frame(&frame, stream->callback_opaque)) {
                free(chunk8);
                fail(error, error_size,
                     "generation cancelled while delivering frame %d",
                     stream->emitted + index);
                return 0;
            }
        }
    }
    int ok = 1;
    if (stream->writer)
        ok = h3_av_writer_write_video(stream->writer, chunk8, frame_count,
                                      error, error_size);
    free(chunk8);
    if (!ok) return 0;
    stream->emitted += frame_count;
    if (stream->writer)
        h3_progress_emit(stream->progress, "encode", stream->emitted,
                         stream->total_frames);
    if (stream->progress->cancelled) {
        fail(error, error_size, "generation cancelled");
        return 0;
    }
    return 1;
}

int h3_continuation_latent_from_video(const char *model_dir,
                                      const char *video_path,
                                      int render_width, int render_height,
                                      int carry_frames,
                                      const char *latent_path,
                                      char *error, size_t error_size) {
    enum { CHUNK_FRAMES = 17 };
    if (error && error_size) error[0] = '\0';
    if (!model_dir || !video_path || !latent_path ||
        render_width < 16 || render_height < 16 ||
        (carry_frames != 22 && carry_frames != 39 && carry_frames != 56)) {
        snprintf(error, error_size, "invalid continuation source arguments");
        return 0;
    }
    int latent_w, latent_h;
    h3_latent_canvas(render_width, render_height, &latent_w, &latent_h);
    const int width = latent_w * H3_VAE_SPATIAL_RATIO;
    const int height = latent_h * H3_VAE_SPATIAL_RATIO;
    const int frames = carry_frames;
    const int steps = h3_video_latent_t(frames);          /* 7, 12 or 17 */
    const h3_temporal_shape temporal = h3_temporal(frames);
    const int audio_t = temporal.audio_t;                 /* 37, 65 or 93 */
    const size_t area = (size_t)width * height;
    const size_t plane = (size_t)latent_w * latent_h;

    char *vae_path = h3_path(model_dir, "FL2VA/video_vae/source");
    char *audio_vae_path = h3_path(model_dir, "FL2VA/audio_vae");
    float *pixels = NULL, *chunk = NULL, *video = NULL, *pcm = NULL, *audio = NULL;
    h3_audio_latent sound;
    memset(&sound, 0, sizeof(sound));
    int ok = 0;
    double start = 0;
    if (!vae_path || !audio_vae_path) {
        snprintf(error, error_size, "out of memory resolving model paths");
        goto done;
    }
    if (!h3_av_read_video_tail_f32(video_path, width, height, frames, &pixels,
                                   &start, error, error_size)) goto done;

    /* Generated clips are 17-frame chunks of 1 + 4 + 4 + 4 + 4 frames per
     * latent step (h3_video_step_frame), each decoded on its own: encode
     * the frames as such chunks and the start (5 frames) of the next. */
    video = malloc((size_t)24 * steps * plane * sizeof(*video));
    chunk = malloc((size_t)3 * CHUNK_FRAMES * area * sizeof(*chunk));
    if (!video || !chunk) {
        snprintf(error, error_size, "out of memory encoding the video's end");
        goto done;
    }
    for (int first = 0, step = 0; first < frames; ) {
        int count = frames - first < CHUNK_FRAMES ? frames - first : CHUNK_FRAMES;
        for (int channel = 0; channel < 3; channel++)
            memcpy(chunk + (size_t)channel * count * area,
                   pixels + ((size_t)channel * frames + first) * area,
                   (size_t)count * area * sizeof(*chunk));
        h3_video_latent latent;
        memset(&latent, 0, sizeof(latent));
        if (!h3_video_vae_encode_fast(vae_path, "h3_shaders.metal", chunk, count,
                                      height, width, NULL, NULL, &latent,
                                      error, error_size)) {
            h3_video_latent_free(&latent);
            goto done;
        }
        int chunk_steps = h3_video_encoder_latent_t(count);
        if (latent.time != chunk_steps || latent.height != latent_h ||
            latent.width != latent_w || step + chunk_steps > steps) {
            h3_video_latent_free(&latent);
            snprintf(error, error_size, "the video VAE gave an unexpected latent shape");
            goto done;
        }
        for (int channel = 0; channel < 24; channel++)
            memcpy(video + ((size_t)channel * steps + step) * plane,
                   latent.values + (size_t)channel * chunk_steps * plane,
                   (size_t)chunk_steps * plane * sizeof(*video));
        h3_video_latent_free(&latent);
        first += count;
        step += chunk_steps;
    }

    /* The sound under those frames, encoded with as much before them as a
     * generated clip has (175 frames, about 7.3 s) where the video is that
     * long: the audio VAE's latents depend on the whole stretch it encodes,
     * and a generated clip's come from the clip as a whole. Only the last
     * audio_t steps are kept. */
    int window_t = h3_temporal(frames > 175 ? frames : 175).audio_t;
    int lead = (int)floor(start * H3_AUDIO_LATENT_FPS);
    if (lead > window_t - audio_t) lead = window_t - audio_t;
    int samples = (lead + audio_t) * 800;
    char audio_error[512];
    if (!h3_av_read_audio_range_f32(video_path, start - (double)lead / H3_AUDIO_LATENT_FPS,
                                    samples, &pcm, audio_error, sizeof(audio_error))) {
        fprintf(stderr, "h3: continuing %s without its sound (%s)\n", video_path, audio_error);
        pcm = calloc((size_t)2 * samples, sizeof(*pcm));
        if (!pcm) {
            snprintf(error, error_size, "out of memory for silence");
            goto done;
        }
    }
    if (!h3_audio_vae_encode(audio_vae_path, "h3_shaders.metal", pcm, samples,
                             NULL, NULL, &sound, error, error_size)) goto done;
    if (sound.channels != 32 || sound.stereo != 2 || sound.length < audio_t) {
        snprintf(error, error_size, "the audio VAE gave an unexpected latent shape");
        goto done;
    }
    audio = malloc((size_t)64 * audio_t * sizeof(*audio));
    if (!audio) {
        snprintf(error, error_size, "out of memory encoding the sound");
        goto done;
    }
    for (int row = 0; row < 64; row++)
        memcpy(audio + (size_t)row * audio_t,
               sound.values + (size_t)row * sound.length + (sound.length - audio_t),
               (size_t)audio_t * sizeof(*audio));
    ok = h3_latent_file_write(latent_path, video, steps, latent_h, latent_w,
                              audio, audio_t, frames, error, error_size);
done:
    h3_audio_latent_free(&sound);
    free(audio); free(pcm); free(video); free(chunk); free(pixels);
    free(vae_path); free(audio_vae_path);
    return ok;
}

int h3_fast_attention_available(void) {
    return h3_gpu_ccv_dense_attention_available() &&
           h3_gpu_ccv_attention_supported();
}

/* The attention path for one generation, decided once up front: the
 * caller's fast_attention flag; otherwise, for diagnostics only, the
 * H3_ATTENTION_BACKEND / H3_CCV_DIRECT environment (never set by the app). */
static h3_dit_attention_mode h3_resolve_attention_mode(const h3_params *params) {
    if (params->fast_attention) return H3_DIT_ATTENTION_CCV_INT8_DIRECT;
    const char *backend = h3_option("H3_ATTENTION_BACKEND");
    if (backend && !strcmp(backend, "ccv_dense")) {
        const char *direct = h3_option("H3_CCV_DIRECT");
        return direct && *direct == '1' ? H3_DIT_ATTENTION_CCV_INT8_DIRECT
                                        : H3_DIT_ATTENTION_CCV_INT8;
    }
    if (backend && !strcmp(backend, "ccv_fp16")) return H3_DIT_ATTENTION_CCV_FP16;
    return H3_DIT_ATTENTION_MPS;
}

/* The conditioning encoders, each in an autorelease pool of its own (see
 * h3_autorelease_push): without one, every call's autoreleased Metal
 * objects stay until the whole generation ends. */
static int h3_pooled_vision_encode(const char *weight_directory,
                                   const float *pixels, int frames,
                                   int height, int width,
                                   h3_vision_progress progress, void *opaque,
                                   h3_vision_output *output,
                                   char *error, size_t error_size) {
    void *pool = h3_autorelease_push();
    int ok = h3_vision_encode_bf16(weight_directory, "h3_shaders.metal",
                                   pixels, frames, height, width, progress,
                                   opaque, output, error, error_size);
    h3_autorelease_pop(pool);
    return ok;
}

static int h3_pooled_video_encode(const char *vae_path, const float *pixels,
                                  int frames, int height, int width,
                                  h3_video_encoder_progress progress,
                                  void *opaque, h3_video_latent *latent,
                                  char *error, size_t error_size) {
    void *pool = h3_autorelease_push();
    int ok = h3_video_vae_encode(vae_path, "h3_shaders.metal", pixels, frames,
                                 height, width, progress, opaque, latent,
                                 error, error_size);
    h3_autorelease_pop(pool);
    return ok;
}

static int h3_pooled_audio_encode(const char *audio_vae_path,
                                  const float *pcm, int samples,
                                  h3_audio_vae_progress progress, void *opaque,
                                  h3_audio_latent *latent,
                                  char *error, size_t error_size) {
    void *pool = h3_autorelease_push();
    int ok = h3_audio_vae_encode(audio_vae_path, "h3_shaders.metal", pcm,
                                 samples, progress, opaque, latent,
                                 error, error_size);
    h3_autorelease_pop(pool);
    return ok;
}

/* Resolves a RefMod reference to its layout entry from the file's header
 * alone, for h3_generate and h3_plan_generation alike: the member's stored
 * latent geometry is all the layout needs. An audio member's audio_t is
 * left for the audio pass to fill. */
static int h3_refmod_reference(const h3_reference *reference, size_t index,
                               h3_layout_ref *layout, h3_refmod_member *member,
                               char *error, size_t error_size) {
    h3_refmod_info info;
    char detail[512];
    if (!h3_refmod_inspect(reference->path, &info, detail, sizeof(detail))) {
        snprintf(error, error_size, "RefMod %zu: %s", index + 1, detail);
        return 0;
    }
    int ok = 0;
    if (info.bundle && reference->member < 0) {
        snprintf(error, error_size,
                 "RefMod %zu is a bundle of %zu members: choose one",
                 index + 1, info.member_count);
    } else if (reference->member >= (int)info.member_count) {
        snprintf(error, error_size, "RefMod %zu has %zu member%s, not %d",
                 index + 1, info.member_count,
                 info.member_count == 1 ? "" : "s", reference->member + 1);
    } else {
        const h3_refmod_member *chosen =
            &info.members[reference->member < 0 ? 0 : reference->member];
        if (!chosen->supported) {
            snprintf(error, error_size, "RefMod %zu member \"%s\" is not "
                     "supported", index + 1, chosen->name);
        } else {
            if (chosen->kind == H3_REFMOD_AUDIO)
                *layout = (h3_layout_ref){H3_LAYOUT_REF_AUDIO, 0, 0, 0, 0};
            else
                *layout = (h3_layout_ref){
                    chosen->kind == H3_REFMOD_IMAGE ? H3_LAYOUT_REF_IMAGE :
                                                      H3_LAYOUT_REF_VIDEO,
                    chosen->latent_t, chosen->latent_h, chosen->latent_w, 0};
            if (member) *member = *chosen;
            ok = 1;
        }
    }
    h3_refmod_info_free(&info);
    return ok;
}

/* The 9 images / 3 videos / 3 audio inputs rule (h3_valid_params) once
 * every RefMod's contents are known: a RefMod counts as what its member
 * is, read from the layout entry h3_refmod_reference filled. */
static int h3_reference_limits(const h3_params *params,
                               const h3_layout_ref *layout_references,
                               char *error, size_t error_size) {
    size_t images = 0, videos = 0, audio_inputs = 0;
    for (size_t index = 0; index < params->reference_count; index++) {
        const h3_reference *reference = &params->references[index];
        switch (reference->kind) {
        case H3_REFERENCE_IMAGE: images++; break;
        case H3_REFERENCE_VIDEO:
            videos++;
            if (reference->include_embedded_audio) audio_inputs++;
            break;
        case H3_REFERENCE_AUDIO: audio_inputs++; break;
        case H3_REFERENCE_VIDEO_AUDIO: videos++; audio_inputs++; break;
        case H3_REFERENCE_REFMOD:
            if (layout_references[index].kind == H3_LAYOUT_REF_IMAGE) images++;
            else if (layout_references[index].kind == H3_LAYOUT_REF_VIDEO) videos++;
            else audio_inputs++;
            break;
        }
    }
    if (images > 9 || videos > 3 || audio_inputs > 3) {
        snprintf(error, error_size,
                 "Ref2VA limits are 9 images, 3 videos, and 3 audio inputs");
        return 0;
    }
    return 1;
}

/* The canvas a visual Ref2VA reference is read at, for h3_generate and
 * h3_plan_generation alike. */
static int h3_reference_canvas(const h3_ctx *ctx, const h3_params *params,
                               const h3_reference *reference,
                               int render_width, int render_height,
                               int source_width, int source_height,
                               int *width, int *height) {
    if (reference->kind == H3_REFERENCE_IMAGE)
        return h3_reference_image_canvas(
            source_width, source_height, render_width, render_height,
            params->reference_image_size == H3_REFERENCE_IMAGE_MAX ? 2048 : 0,
            width, height);
    return h3_reference_video_canvas(
        source_width, source_height,
        h3_reference_max_pixels(h3_device(ctx)->physical_memory),
        width, height);
}

/* Whether h3_gpu.m will run dense attention as one MPSGraph call (its
 * policy: Apple10 GPUs on macOS 26 or later, unless H3_SDPA_MODE says). */
static int h3_plan_full_sdpa(const h3_ctx *ctx) {
    const char *mode = h3_option("H3_SDPA_MODE");
    if (mode && *mode) return !strcmp(mode, "full");
    if (h3_device(ctx)->apple_gpu_family < 10) return 0;
    if (__builtin_available(macOS 26.0, *)) return 1;
    return 0;
}

/* Resident DiT block bytes of a HyperFlow H3 Flash transformer (0 for
 * any other checkpoint), the way h3_dit.c loads it: dense linears int8 on a
 * GPU with tensor units (BF16 elsewhere), factored linears and the dense
 * half of a half-factored MLP in BF16. Matches the 18 GiB plain-H3 figure,
 * which also counts block weights only. */
static uint64_t h3_hyperflow_resident_bytes(const h3_ctx *ctx, int ref2va) {
    if (!h3_dir_has_gguf(ctx->model_dir, ref2va ? "Ref2VA/transformer"
                                                : "FL2VA/transformer"))
        return 0;
    char *path = h3_path(ctx->model_dir, ref2va ? "Ref2VA/transformer"
                                                : "FL2VA/transformer");
    char detail[512];
    h3_weight_store *store = path ?
        h3_weight_store_open(path, detail, sizeof(detail)) : NULL;
    free(path);
    h3_hyperflow_info info;
    int ok = store && h3_hyperflow_inspect(store, &info, detail,
                                           sizeof(detail)) && info.two_time;
    h3_weight_store_free(store);
    if (!ok) return 0;
    enum { HIDDEN = 5376, INNER = 7168, FFN = 14336 };
    static const uint64_t in[H3_HF_LINEARS] = {
        HIDDEN, HIDDEN, HIDDEN, INNER, HIDDEN, FFN};
    static const uint64_t out[H3_HF_LINEARS] = {
        INNER, INNER, INNER, HIDDEN, 2 * FFN, HIDDEN};
    uint64_t dense = h3_device(ctx)->apple_gpu_family >= 10 ? 1 : 2;
    uint64_t bytes = 0;
    for (uint32_t block = 0; block < info.block_count; block++) {
        const uint32_t *rank = info.rank[block];
        int mixed_mlp = rank[H3_HF_FC1] || rank[H3_HF_FC2];
        for (int linear = 0; linear < H3_HF_LINEARS; linear++) {
            int mlp = linear == H3_HF_FC1 || linear == H3_HF_FC2;
            if (rank[linear])
                bytes += 2ull * rank[linear] * (in[linear] + out[linear]);
            else
                bytes += (mlp && mixed_mlp ? 2 : dense) *
                         in[linear] * out[linear];
        }
    }
    return bytes;
}

int h3_plan_generation(const h3_ctx *ctx, const char *prompt,
                       const h3_params *params, h3_plan *plan) {
    if (!plan) return 0;
    memset(plan, 0, sizeof(*plan));
    char *message = plan->message;
    size_t message_size = sizeof(plan->message);
    if (!ctx || !prompt || !*prompt || !params) {
        snprintf(message, message_size,
                 "planning needs a context, a prompt and parameters");
        return 0;
    }
    int render_width = params->render_width ? params->render_width :
                                               params->width;
    int render_height = params->render_height ? params->render_height :
                                                 params->height;
    if (render_width < 32 || render_height < 32 ||
        render_width % H3_CANVAS_MULTIPLE || render_height % H3_CANVAS_MULTIPLE ||
        params->frames < 5 || h3_align_frame_count(params->frames) > 362 ||
        (params->reference_count && !params->references)) {
        snprintf(message, message_size,
                 "invalid canvas, frame count or references");
        return 0;
    }
    int ref2va = params->reference_count != 0;
    h3_temporal_shape temporal = h3_temporal(params->frames);
    int latent_w, latent_h;
    h3_latent_canvas(render_width, render_height, &latent_w, &latent_h);
    h3_continuation continuation;
    if (!h3_continuation_plan(params, latent_h, latent_w, temporal,
                              &continuation, message, message_size))
        return 0;

    size_t reference_count = ref2va ? params->reference_count : 0;
    size_t visual_capacity = ref2va ? reference_count :
        (size_t)(params->first_frame != NULL) +
        (size_t)(params->last_frame != NULL);
    /* At most this many Qwen vision calls: one per image, one per
     * two-frame block of a video (362 frames: 31 samples, 16 blocks). */
    size_t vision_capacity = visual_capacity * 16 + 1;
    h3_plan_visual *visuals = calloc(visual_capacity + 1, sizeof(*visuals));
    h3_vision_output *visions = calloc(vision_capacity, sizeof(*visions));
    double *timestamps = calloc(vision_capacity, sizeof(*timestamps));
    h3_reference_presentation *presentations =
        calloc(reference_count + 1, sizeof(*presentations));
    h3_layout_ref *layout_references =
        calloc(reference_count + 1, sizeof(*layout_references));
    char *tokenizer_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/tokenizer/tokenizer.json" : "FL2VA/tokenizer/tokenizer.json");
    h3_tokenizer *tokenizer = NULL;
    h3_layout layout;
    memset(&layout, 0, sizeof(layout));
    int ok = 0;
    if (!visuals || !visions || !timestamps || !presentations ||
        !layout_references || !tokenizer_path) {
        snprintf(message, message_size, "out of memory planning the generation");
        goto done;
    }

    size_t visual_count = 0, vision_count = 0, vision_tokens = 0;
    size_t reference_rows = 0;
    size_t audio_samples_total = 0;
    char detail[512];
    for (size_t index = 0; index < reference_count; index++) {
        const h3_reference *reference = &params->references[index];
        if (!reference->path || !*reference->path) {
            snprintf(message, message_size, "reference %zu has no file",
                     index + 1);
            goto done;
        }
        int64_t samples = 0;
        if (reference->kind == H3_REFERENCE_REFMOD) {
            /* A saved latent: its geometry comes from the file, it costs no
             * encoder or Qwen work, and Qwen gets no label for it. */
            h3_refmod_member member;
            if (!h3_refmod_reference(reference, index, &layout_references[index],
                                     &member, detail, sizeof(detail))) {
                snprintf(message, message_size, "%s", detail);
                goto done;
            }
            presentations[index].kind = H3_PRESENTATION_HIDDEN;
            if (member.kind == H3_REFMOD_AUDIO)
                samples = (int64_t)member.latent_t * 800;
            else
                reference_rows += member.tokens;
        } else if (reference->kind == H3_REFERENCE_AUDIO) {
            if (!h3_av_audio_sample_count(reference->path, &samples,
                                          detail, sizeof(detail))) {
                snprintf(message, message_size, "%s", detail);
                goto done;
            }
            presentations[index].kind = H3_PRESENTATION_AUDIO;
            presentations[index].has_audio = 1;
            layout_references[index] = (h3_layout_ref){
                H3_LAYOUT_REF_AUDIO, 0, 0, 0, 0};
        } else {
            int source_width, source_height, width, height;
            if (!h3_av_visual_size(reference->path, &source_width,
                                   &source_height, detail, sizeof(detail))) {
                snprintf(message, message_size, "%s", detail);
                goto done;
            }
            if (!h3_reference_canvas(ctx, params, reference, render_width,
                                     render_height, source_width,
                                     source_height, &width, &height)) {
                snprintf(message, message_size,
                         "cannot resolve reference %zu canvas", index + 1);
                goto done;
            }
            size_t block_tokens = (size_t)(width / 32) * (size_t)(height / 32);
            int frames = 1;
            if (reference->kind == H3_REFERENCE_IMAGE) {
                presentations[index].kind = H3_PRESENTATION_IMAGE;
                presentations[index].vision = &visions[vision_count];
                presentations[index].vision_count = 1;
                visions[vision_count++] = (h3_vision_output){
                    height / 16, width / 16, block_tokens, NULL, {NULL}, {0}};
                vision_tokens += block_tokens;
            } else {
                if (!h3_av_video_frame_count(reference->path, &frames,
                                             detail, sizeof(detail))) {
                    snprintf(message, message_size, "%s", detail);
                    goto done;
                }
                if (frames > temporal.frame_count) frames = temporal.frame_count;
                if (frames < 5) {
                    snprintf(message, message_size,
                             "reference videos require at least 5 decoded "
                             "frames");
                    goto done;
                }
                while (frames % 17 != 5) frames--;
                /* As h3_generate samples Qwen's view of the video. */
                size_t sampled = ((size_t)frames + 11) / 12;
                size_t blocks = (sampled + 1) / 2;
                presentations[index].kind = H3_PRESENTATION_VIDEO;
                presentations[index].vision = &visions[vision_count];
                presentations[index].vision_count = blocks;
                presentations[index].timestamps = &timestamps[vision_count];
                for (size_t block = 0; block < blocks; block++) {
                    size_t first = 2 * block;
                    size_t second = first + 1 < sampled ? first + 1 : first;
                    timestamps[vision_count] =
                        ((double)first + (double)second) / 4.0;
                    visions[vision_count++] = (h3_vision_output){
                        height / 16, width / 16, block_tokens, NULL, {NULL},
                        {0}};
                    vision_tokens += block_tokens;
                }
                const char *audio_path =
                    reference->kind == H3_REFERENCE_VIDEO_AUDIO ?
                        reference->audio_path :
                    reference->include_embedded_audio ? reference->path : NULL;
                if (audio_path) {
                    int64_t limit = llround((double)frames * 32000.0 / H3_FPS);
                    if (limit < 64000) {
                        snprintf(message, message_size,
                                 "video soundtrack %zu requires at least 2 "
                                 "seconds; request at least 56 output frames",
                                 index + 1);
                        goto done;
                    }
                    if (!h3_av_audio_sample_count(audio_path, &samples,
                                                  detail, sizeof(detail))) {
                        snprintf(message, message_size, "%s", detail);
                        goto done;
                    }
                    if (samples > limit) samples = limit;
                    presentations[index].has_audio = 1;
                }
            }
            visuals[visual_count++] = (h3_plan_visual){width, height, frames};
            layout_references[index] = (h3_layout_ref){
                reference->kind == H3_REFERENCE_IMAGE ?
                    H3_LAYOUT_REF_IMAGE : H3_LAYOUT_REF_VIDEO,
                h3_video_encoder_latent_t(frames), height / 16, width / 16, 0};
            reference_rows += (size_t)layout_references[index].latent_t *
                              (size_t)(height / 32) * (size_t)(width / 32);
        }
        if (samples > 0) {
            if ((uint64_t)samples > (uint64_t)32000 * 15 - audio_samples_total) {
                snprintf(message, message_size,
                         "ordered reference audio exceeds 15 seconds in total");
                goto done;
            }
            audio_samples_total += (size_t)samples;
            /* The AudioVAE encoder's hop is 800 samples (2*4*4*5*5). */
            layout_references[index].audio_t = (int)((samples + 799) / 800);
            reference_rows += (size_t)layout_references[index].audio_t * 2;
        }
    }
    if (reference_count &&
        !h3_reference_limits(params, layout_references, message, message_size))
        goto done;

    int keyframes[2 + 17] = {0};
    size_t keyframe_count = 0;
    if (!ref2va) {
        const char *anchors[2] = {params->first_frame, params->last_frame};
        for (int anchor = 0; anchor < 2; anchor++) {
            if (!anchors[anchor]) continue;
            int source_width, source_height;
            if (!h3_av_visual_size(anchors[anchor], &source_width,
                                   &source_height, detail, sizeof(detail))) {
                snprintf(message, message_size, "%s", detail);
                goto done;
            }
            keyframes[keyframe_count++] =
                anchor ? temporal.frame_count - 1 : 0;
            size_t tokens = (size_t)(render_width / 32) *
                            (size_t)(render_height / 32);
            visions[vision_count++] = (h3_vision_output){
                render_height / 16, render_width / 16, tokens, NULL, {NULL},
                {0}};
            vision_tokens += tokens;
            visuals[visual_count++] =
                (h3_plan_visual){render_width, render_height, 1};
        }
    }
    for (int step = 0; continuation.active && step < continuation.steps; step++)
        keyframes[keyframe_count++] = continuation.keyframes[step];

    tokenizer = h3_tokenizer_load(tokenizer_path, detail, sizeof(detail));
    if (!tokenizer) {
        snprintf(message, message_size, "%s", detail);
        goto done;
    }
    size_t text_tokens = 0;
    int counted;
    if (ref2va) {
        counted = h3_multimodal_ref2va_token_count(
            tokenizer, prompt, presentations, reference_count, &text_tokens,
            detail, sizeof(detail));
    } else if (vision_count) {
        counted = h3_multimodal_fl2va_token_count(
            tokenizer, prompt, visions, vision_count, &text_tokens,
            detail, sizeof(detail));
    } else {
        uint32_t *ids = NULL;
        counted = h3_tokenizer_encode(tokenizer, prompt, 1, &ids, &text_tokens,
                                      detail, sizeof(detail));
        h3_tokenizer_ids_free(ids);
    }
    if (!counted || text_tokens > INT_MAX) {
        snprintf(message, message_size, "%s",
                 counted ? "the prompt is too long" : detail);
        goto done;
    }

    h3_layout_spec spec = {(int)text_tokens, temporal.video_t, latent_h,
                           latent_w, temporal.audio_t, temporal.frame_count,
                           keyframes, keyframe_count, layout_references,
                           reference_count, continuation.audio_t,
                           continuation.audio_start};
    if (!h3_layout_build(&spec, &layout, detail, sizeof(detail))) {
        snprintf(message, message_size, "%s", detail);
        goto done;
    }
    size_t generated_rows = 0;
    for (size_t segment = 0; segment < layout.segment_count; segment++) {
        const h3_segment *part = &layout.segments[segment];
        if (part->kind == H3_SEG_VIDEO || part->kind == H3_SEG_AUDIO)
            generated_rows += part->stop - part->start;
    }
    /* As h3_text_encoder.c reads ahead: 3 layers on M5, 2 elsewhere. */
    const char *prefetch = h3_option("H3_QWEN_PREFETCH_DEPTH");
    int prefetch_depth = params->plan_text_prefetch_depth > 0 ?
        params->plan_text_prefetch_depth :
        prefetch && *prefetch ? atoi(prefetch) :
        h3_device(ctx)->apple_gpu_family >= 10 ? 3 : 2;
    if (prefetch_depth < 1) prefetch_depth = 1;
    if (prefetch_depth > 6) prefetch_depth = 6;
    /* Resident mode: neither SSD streaming nor the attention cache. Int8
     * on a GPU with tensor units (~18 GiB, like the cache file), BF16
     * elsewhere (~37 GiB). */
    const char *cache = h3_option("H3_ATTENTION_CACHE");
    int cached = params->plan_attention_cache ?
        params->plan_attention_cache > 0 : (cache && *cache);
    uint64_t hyperflow_resident = h3_hyperflow_resident_bytes(ctx, ref2va);
    /* HyperFlow never uses the attention cache (h3_dit.c ignores it). */
    if (hyperflow_resident) cached = 0;
    uint64_t resident = params->ssd_streaming || cached ? 0 :
        hyperflow_resident ? hyperflow_resident :
        h3_device(ctx)->apple_gpu_family >= 10 ? 18ull << 30 : 37ull << 30;
    h3_plan_input input = {
        h3_device(ctx)->physical_memory, params->ssd_streaming,
        h3_plan_full_sdpa(ctx), prefetch_depth, resident,
        text_tokens, vision_tokens, layout.seq_len,
        generated_rows, reference_rows,
        visuals, visual_count};
    h3_plan_compute(&input, plan);
    ok = 1;

done:
    h3_layout_free(&layout);
    h3_tokenizer_free(tokenizer);
    free(tokenizer_path);
    free(layout_references);
    free(presentations);
    free(timestamps);
    free(visions);
    free(visuals);
    return ok;
}

h3_result *h3_generate(h3_ctx *ctx, const char *prompt,
                       const h3_params *params) {
    if (!ctx) return NULL;
    ctx->error[0] = '\0';
    if (!prompt || !*prompt) {
        h3_set_error(ctx, "prompt must not be empty");
        return NULL;
    }
    if (!h3_valid_params(ctx, params)) return NULL;
    h3_memory_watch_start();
    int planned = 0;
    size_t planned_text_tokens = 0, planned_dit_tokens = 0;
    {
        /* Refuse what cannot run before loading anything - a reference
         * that only fails at the text encoder would otherwise do so after
         * encoding it for many minutes, and one past this Mac's memory can
         * take the whole system down. A plan that cannot be made (say, an
         * unreadable file) is left for the generation to report. */
        const char *preflight = h3_option("H3_PREFLIGHT");
        h3_plan plan;
        planned = (!preflight || strcmp(preflight, "0")) &&
            h3_plan_generation(ctx, prompt, params, &plan);
        if (planned) {
            planned_text_tokens = plan.text_tokens;
            planned_dit_tokens = plan.dit_tokens;
            fprintf(stderr, "h3: plan: text encoder %zu tokens (%zu from "
                    "images and video), DiT %zu tokens (%zu from references, "
                    "cost x%.2f), peak about %.1f GiB in the %s\n",
                    plan.text_tokens, plan.vision_tokens, plan.dit_tokens,
                    plan.dit_reference_tokens, plan.dit_cost_ratio,
                    (double)plan.peak_bytes / (1024.0 * 1024.0 * 1024.0),
                    plan.peak_stage);
            if (plan.verdict == H3_PLAN_TOO_LARGE) {
                h3_set_error(ctx, "%s", plan.message);
                return NULL;
            }
            if (plan.verdict == H3_PLAN_TIGHT)
                fprintf(stderr, "h3: plan: %s\n", plan.message);
        }
    }
    int render_width = params->render_width ? params->render_width :
                                               params->width;
    int render_height = params->render_height ? params->render_height :
                                                 params->height;
    if (h3_align_frame_count(params->frames) < 22) {
        h3_set_error(ctx,
            "generation requires at least one trained 22-frame decoder chunk");
        return NULL;
    }
    int ref2va = params->reference_count != 0;
    if (ref2va && !ctx->model.ref2va_transformer.files) {
        h3_set_error(ctx, "ordered references require the Ref2VA checkpoint");
        return NULL;
    }
    h3_generation_progress progress = {ctx, params, 0};
    h3_temporal_shape temporal = h3_temporal(params->frames);
    int latent_w, latent_h;
    h3_latent_canvas(render_width, render_height, &latent_w, &latent_h);
    h3_tokenizer *tokenizer = NULL;
    uint32_t *ids = NULL;
    size_t token_count = 0;
    size_t visual_capacity = ref2va ? params->reference_count :
        (size_t)(params->first_frame != NULL) +
        (size_t)(params->last_frame != NULL);
    size_t visual_count = 0;
    float **condition_pixels = NULL;
    int *condition_widths = NULL;
    int *condition_heights = NULL;
    int *condition_frames = NULL;
    size_t *visual_reference_indices = NULL;
    size_t *reference_visual_indices = NULL;
    h3_vision_output *vision_outputs = NULL;
    size_t vision_output_count = 0;
    h3_reference_presentation *presentations = NULL;
    double **presentation_timestamps = NULL;
    h3_layout_ref *layout_references = NULL;
    int keyframes[2 + 17] = {0};
    h3_continuation continuation;
    memset(&continuation, 0, sizeof(continuation));
    size_t keyframe_count = 0;
    float *condition_video_rows = NULL;
    size_t condition_video_elements = 0;
    float *condition_audio_rows = NULL;
    size_t condition_audio_elements = 0;
    h3_text_embedding text;
    memset(&text, 0, sizeof(text));
    h3_layout layout;
    memset(&layout, 0, sizeof(layout));
    h3_dit *dit = NULL;
    h3_video_vae_decoder *preview_decoder = NULL;
    h3_live_preview live_preview;
    memset(&live_preview, 0, sizeof(live_preview));
    float *video = NULL, *audio = NULL;
    float *forced_pcm = NULL, *forced_latent = NULL;
    int forced_samples = 0;
    h3_video_frames frames;
    memset(&frames, 0, sizeof(frames));
    h3_audio_waveform waveform;
    memset(&waveform, 0, sizeof(waveform));
    uint8_t *rgb8 = NULL;
    h3_result *result = NULL;
    char *conditioning_key = NULL;
    char *prepared_key = NULL;
    char *decoder_key = NULL;
    int conditioning_hit = 0;
    int conditioned = 0;
    int dit_is_cached = 0;
    int decoder_is_cached = 0;
    char *tokenizer_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/tokenizer/tokenizer.json" : "FL2VA/tokenizer/tokenizer.json");
    char *text_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/text_encoder" : "FL2VA/text_encoder");
    char *dit_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/transformer" : "FL2VA/transformer");
    char *vae_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/video_vae/source" : "FL2VA/video_vae/source");
    char *audio_vae_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/audio_vae" : "FL2VA/audio_vae");
    if (!tokenizer_path || !text_path || !dit_path || !vae_path ||
        !audio_vae_path) {
        h3_set_error(ctx, "out of memory resolving generation model paths");
        goto cleanup;
    }
    /* The transformer's family (h3_model_family.h). HyperFlow H3 Flash (a
     * GGUF transformer) brings its own 4/6/8-step sampler and supports none
     * of the H3 speed-ups that change blocks or steps; whatever the family
     * cannot do is refused before any work. */
    const h3_model_family_desc *family = h3_model_family_of(0);
    char hyperflow_detail[512] = {0};
    if (h3_dir_has_gguf(ctx->model_dir, ref2va ? "Ref2VA/transformer"
                                               : "FL2VA/transformer")) {
        h3_weight_store *store = h3_weight_store_open(dit_path, hyperflow_detail,
                                                      sizeof(hyperflow_detail));
        h3_hyperflow_info hyperflow_info;
        int inspected = store && h3_hyperflow_inspect(
            store, &hyperflow_info, hyperflow_detail, sizeof(hyperflow_detail));
        h3_weight_store_free(store);
        if (!inspected) {
            h3_set_error(ctx, "%s", hyperflow_detail);
            goto cleanup;
        }
        family = h3_model_family_of(hyperflow_info.two_time);
    }
    {
        const char *problem = h3_model_family_problem(&family->info, params);
        if (problem) {
            h3_set_error(ctx, "%s checkpoint: %s", family->info.label, problem);
            goto cleanup;
        }
    }
    char plan_error[512];
    if (!h3_continuation_plan(params, latent_h, latent_w, temporal,
                              &continuation, plan_error, sizeof(plan_error)) ||
        !h3_initial_latent_check(params, latent_h, latent_w, temporal,
                                 plan_error, sizeof(plan_error))) {
        h3_set_error(ctx, "%s", plan_error);
        goto cleanup;
    }
    conditioning_key = h3_conditioning_key(
        prompt, params, render_width, render_height, ref2va);
    if (!conditioning_key) {
        h3_set_error(ctx, "out of memory constructing conditioning cache key");
        goto cleanup;
    }
    prepared_key = h3_prepared_key(
        conditioning_key, params, render_width, render_height);
    if (!prepared_key) {
        h3_set_error(ctx, "out of memory constructing prepared-model cache key");
        goto cleanup;
    }
    h3_key decoder_cache_key = {0};
    if (!h3_key_append(&decoder_cache_key, "%s|%dx%d", vae_path,
                       latent_h, latent_w)) {
        h3_set_error(ctx, "out of memory constructing decoder cache key");
        goto cleanup;
    }
    decoder_key = decoder_cache_key.text;
    if (ctx->video_decoder && (!h3_cache_target(ctx, H3_CACHE_DECODER) ||
        !ctx->video_decoder_key || strcmp(ctx->video_decoder_key, decoder_key))) {
        h3_video_vae_decoder_free(ctx->video_decoder);
        ctx->video_decoder = NULL;
        free(ctx->video_decoder_key);
        ctx->video_decoder_key = NULL;
    }
    if (ctx->dit && (!h3_cache_target(ctx, H3_CACHE_DIT) ||
        !ctx->dit_key || strcmp(ctx->dit_key, prepared_key))) {
        h3_dit_free(ctx->dit);
        ctx->dit = NULL;
        free(ctx->dit_key);
        ctx->dit_key = NULL;
    }
    if (ctx->conditioning_key && !h3_cache_target(ctx, H3_CACHE_CONDITIONING))
        h3_conditioning_cache_clear(ctx);
    conditioning_hit = h3_cache_target(ctx, H3_CACHE_CONDITIONING) &&
        ctx->conditioning_key &&
        !strcmp(ctx->conditioning_key, conditioning_key);
    char detail[512];
    if (conditioning_hit) {
        size_t cached_reference_count = 0;
        if (!h3_conditioning_cache_load(
                ctx, &text, &condition_video_rows, &condition_video_elements,
                &condition_audio_rows, &condition_audio_elements,
                &layout_references, &cached_reference_count, &conditioned) ||
            cached_reference_count != (ref2va ? params->reference_count : 0)) {
            h3_set_error(ctx, "cannot restore cached conditioning");
            goto cleanup;
        }
        if (!ref2va) {
            if (params->first_frame) keyframes[keyframe_count++] = 0;
            if (params->last_frame)
                keyframes[keyframe_count++] = temporal.frame_count - 1;
        }
        for (int step = 0; continuation.active &&
                           step < continuation.steps; step++)
            keyframes[keyframe_count++] = continuation.keyframes[step];
        fprintf(stderr, "h3: conditioning cache hit\n");
    } else {
    if (visual_capacity) {
        condition_pixels = calloc(visual_capacity, sizeof(*condition_pixels));
        condition_widths = calloc(visual_capacity, sizeof(*condition_widths));
        condition_heights = calloc(visual_capacity, sizeof(*condition_heights));
        condition_frames = calloc(visual_capacity, sizeof(*condition_frames));
        visual_reference_indices = calloc(
            visual_capacity, sizeof(*visual_reference_indices));
        reference_visual_indices = malloc(
            params->reference_count * sizeof(*reference_visual_indices));
        if (reference_visual_indices)
            for (size_t index = 0; index < params->reference_count; index++)
                reference_visual_indices[index] = SIZE_MAX;
        if (ref2va) {
            layout_references = calloc(visual_capacity,
                                       sizeof(*layout_references));
            presentations = calloc(params->reference_count,
                                   sizeof(*presentations));
            presentation_timestamps = calloc(
                params->reference_count, sizeof(*presentation_timestamps));
        }
        if (!condition_pixels || !condition_widths || !condition_heights ||
            !condition_frames || !visual_reference_indices ||
            !reference_visual_indices ||
            (ref2va && (!layout_references || !presentations ||
                        !presentation_timestamps))) {
            h3_set_error(ctx, "out of memory preparing visual references");
            goto cleanup;
        }
    }
    h3_progress_emit(&progress, "tokenizer", 0, 1);
    tokenizer = h3_tokenizer_load(tokenizer_path, detail, sizeof(detail));
    if (!tokenizer) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    h3_progress_emit(&progress, "tokenizer", 1, 1);
    if (progress.cancelled) goto cleanup;
    if (ref2va) {
        for (size_t index = 0; index < params->reference_count; index++) {
            const h3_reference *reference = &params->references[index];
            if (reference->kind == H3_REFERENCE_REFMOD) {
                h3_refmod_member member;
                if (!h3_refmod_reference(reference, index,
                                         &layout_references[index], &member,
                                         detail, sizeof(detail))) {
                    h3_set_error(ctx, "%s", detail);
                    goto cleanup;
                }
                presentations[index].kind = H3_PRESENTATION_HIDDEN;
                (void)member;
                continue;
            }
            if (reference->kind == H3_REFERENCE_AUDIO) {
                presentations[index].kind = H3_PRESENTATION_AUDIO;
                layout_references[index] = (h3_layout_ref){
                    H3_LAYOUT_REF_AUDIO, 0, 0, 0, 0};
                continue;
            }
            int source_width, source_height, media_width, media_height;
            if (!h3_av_visual_size(reference->path,
                                        &source_width, &source_height,
                                        detail, sizeof(detail))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            if (!h3_reference_canvas(ctx, params, reference, render_width,
                                     render_height, source_width,
                                     source_height, &media_width,
                                     &media_height)) {
                h3_set_error(ctx, "cannot resolve reference %s %zu canvas",
                             reference->kind == H3_REFERENCE_IMAGE ?
                                 "image" : "video", index + 1);
                goto cleanup;
            }
            if (reference->kind == H3_REFERENCE_IMAGE) {
                if (!h3_av_read_image_f32(
                        reference->path, media_width, media_height,
                        H3_IMAGE_FIT_STRETCH, &condition_pixels[visual_count],
                        detail, sizeof(detail))) {
                    h3_set_error(ctx, "%s", detail);
                    goto cleanup;
                }
                condition_frames[visual_count] = 1;
                presentations[index].kind = H3_PRESENTATION_IMAGE;
                presentations[index].vision_count = 1;
                vision_output_count++;
            } else {
                if (!h3_av_read_video_f32(
                        reference->path, media_width, media_height,
                        temporal.frame_count, &condition_pixels[visual_count],
                        &condition_frames[visual_count],
                        detail, sizeof(detail))) {
                    h3_set_error(ctx, "%s", detail);
                    goto cleanup;
                }
                size_t samples = ((size_t)condition_frames[visual_count] + 11) / 12;
                size_t blocks = (samples + 1) / 2;
                if (vision_output_count > SIZE_MAX - blocks) {
                    h3_set_error(ctx, "reference video vision count overflows");
                    goto cleanup;
                }
                presentation_timestamps[index] = malloc(
                    blocks * sizeof(*presentation_timestamps[index]));
                if (!presentation_timestamps[index]) {
                    h3_set_error(ctx,
                        "out of memory allocating reference video timestamps");
                    goto cleanup;
                }
                for (size_t block = 0; block < blocks; block++) {
                    size_t first = 2 * block;
                    size_t second = first + 1 < samples ? first + 1 : first;
                    presentation_timestamps[index][block] =
                        ((double)first + (double)second) / 4.0;
                }
                presentations[index].kind = H3_PRESENTATION_VIDEO;
                presentations[index].vision_count = blocks;
                presentations[index].timestamps = presentation_timestamps[index];
                vision_output_count += blocks;
            }
            condition_widths[visual_count] = media_width;
            condition_heights[visual_count] = media_height;
            visual_reference_indices[visual_count] = index;
            reference_visual_indices[index] = visual_count;
            int ref_latent_w, ref_latent_h;
            h3_latent_canvas(media_width, media_height,
                             &ref_latent_w, &ref_latent_h);
            layout_references[index] = (h3_layout_ref){
                reference->kind == H3_REFERENCE_IMAGE ?
                    H3_LAYOUT_REF_IMAGE : H3_LAYOUT_REF_VIDEO,
                h3_video_encoder_latent_t(condition_frames[visual_count]),
                ref_latent_h, ref_latent_w, 0};
            visual_count++;
        }

        if (!h3_reference_limits(params, layout_references,
                                 detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        size_t total_audio_samples = 0;
        for (size_t index = 0; index < params->reference_count; index++) {
            const h3_reference *reference = &params->references[index];
            const char *audio_path = NULL;
            int truncate = 0;
            int max_samples = 32000 * 15;
            if (reference->kind == H3_REFERENCE_REFMOD) {
                if (layout_references[index].kind != H3_LAYOUT_REF_AUDIO)
                    continue;
                /* An audio RefMod: the audio VAE's latent, already in the
                 * normalized space the rows carry; packed exactly as an
                 * encoded soundtrack is below. */
                h3_refmod_latent refmod;
                if (!h3_refmod_load(reference->path, reference->member,
                                    reference->strength, &refmod, NULL,
                                    detail, sizeof(detail))) {
                    h3_set_error(ctx, "RefMod %zu: %s", index + 1, detail);
                    goto cleanup;
                }
                size_t refmod_samples = (size_t)refmod.audio_t * 800;
                size_t elements = (size_t)refmod.audio_t * 2 * 32;
                if (refmod_samples > (size_t)32000 * 15 - total_audio_samples) {
                    h3_refmod_latent_free(&refmod);
                    h3_set_error(ctx,
                        "ordered reference audio exceeds 15 seconds in total");
                    goto cleanup;
                }
                float *grown = realloc(
                    condition_audio_rows,
                    (condition_audio_elements + elements) * sizeof(*grown));
                if (!grown) {
                    h3_refmod_latent_free(&refmod);
                    h3_set_error(ctx,
                        "out of memory packing reference audio conditions");
                    goto cleanup;
                }
                condition_audio_rows = grown;
                float *rows = condition_audio_rows + condition_audio_elements;
                int length = refmod.audio_t;
                for (int stereo = 0; stereo < 2; stereo++)
                    for (int time = 0; time < length; time++)
                        for (int channel = 0; channel < 32; channel++)
                            rows[((size_t)stereo * length + time) * 32 + channel] =
                                refmod.values[((size_t)channel * 2 + stereo) *
                                              length + time];
                condition_audio_elements += elements;
                total_audio_samples += refmod_samples;
                layout_references[index].audio_t = length;
                h3_refmod_latent_free(&refmod);
                continue;
            }
            if (reference->kind == H3_REFERENCE_AUDIO) {
                audio_path = reference->path;
            } else if (reference->kind == H3_REFERENCE_VIDEO_AUDIO) {
                audio_path = reference->audio_path;
                truncate = 1;
            } else if (reference->kind == H3_REFERENCE_VIDEO &&
                       reference->include_embedded_audio) {
                audio_path = reference->path;
                truncate = 1;
            }
            if (!audio_path) continue;
            if (truncate) {
                size_t visual = reference_visual_indices[index];
                if (visual == SIZE_MAX) {
                    h3_set_error(ctx,
                        "video soundtrack %zu has no decoded video", index + 1);
                    goto cleanup;
                }
                max_samples = (int)llround(
                    (double)condition_frames[visual] * 32000.0 / H3_FPS);
                if (max_samples < 64000) {
                    h3_set_error(ctx,
                        "video soundtrack %zu requires at least 2 seconds; "
                        "request at least 56 output frames", index + 1);
                    goto cleanup;
                }
            }
            float *pcm = NULL;
            int samples = 0;
            if (!h3_av_read_audio_f32(
                    audio_path, max_samples, truncate, &pcm, &samples,
                    detail, sizeof(detail))) {
                free(pcm);
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            if ((size_t)samples > (size_t)32000 * 15 - total_audio_samples) {
                free(pcm);
                h3_set_error(ctx,
                    "ordered reference audio exceeds 15 seconds in total");
                goto cleanup;
            }
            h3_audio_latent latent;
            memset(&latent, 0, sizeof(latent));
            if (!h3_pooled_audio_encode(
                    audio_vae_path, pcm, samples,
                    h3_audio_encoder_progress_bridge, &progress, &latent,
                    detail, sizeof(detail))) {
                free(pcm);
                h3_audio_latent_free(&latent);
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            free(pcm);
            if (latent.channels != 32 || latent.stereo != 2 ||
                latent.length < 1 ||
                (size_t)latent.length > SIZE_MAX / 2 / 32) {
                h3_audio_latent_free(&latent);
                h3_set_error(ctx,
                    "reference audio encoder produced invalid geometry");
                goto cleanup;
            }
            size_t elements = (size_t)latent.length * 2 * 32;
            if (condition_audio_elements > SIZE_MAX - elements ||
                condition_audio_elements + elements >
                    SIZE_MAX / sizeof(*condition_audio_rows)) {
                h3_audio_latent_free(&latent);
                h3_set_error(ctx, "reference audio row count overflows");
                goto cleanup;
            }
            float *grown = realloc(
                condition_audio_rows,
                (condition_audio_elements + elements) * sizeof(*grown));
            if (!grown) {
                h3_audio_latent_free(&latent);
                h3_set_error(ctx,
                    "out of memory packing reference audio conditions");
                goto cleanup;
            }
            condition_audio_rows = grown;
            float *rows = condition_audio_rows + condition_audio_elements;
            for (int stereo = 0; stereo < 2; stereo++)
                for (int time = 0; time < latent.length; time++)
                    for (int channel = 0; channel < 32; channel++) {
                        size_t source = ((size_t)channel * 2 +
                                         (size_t)stereo) * latent.length +
                                        (size_t)time;
                        size_t destination = ((size_t)stereo * latent.length +
                                              (size_t)time) * 32 +
                                             (size_t)channel;
                        rows[destination] = latent.values[source];
                    }
            condition_audio_elements += elements;
            total_audio_samples += (size_t)samples;
            layout_references[index].audio_t = latent.length;
            presentations[index].has_audio = 1;
            h3_audio_latent_free(&latent);
            if (progress.cancelled) goto cleanup;
        }
    } else {
        if (params->first_frame) {
            keyframes[keyframe_count++] = 0;
            /* Center-crop to the render aspect ratio instead of stretching,
             * matching last_frame below - an input whose aspect differs
             * from render_width/render_height no longer gets squashed. */
            if (!h3_av_read_image_f32(
                    params->first_frame, render_width, render_height,
                    H3_IMAGE_FIT_COVER, &condition_pixels[visual_count],
                    detail, sizeof(detail))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            condition_widths[visual_count] = render_width;
            condition_heights[visual_count] = render_height;
            condition_frames[visual_count] = 1;
            visual_count++;
        }
        if (params->last_frame) {
            keyframes[keyframe_count++] = temporal.frame_count - 1;
            if (!h3_av_read_image_f32(
                    params->last_frame, render_width, render_height,
                    H3_IMAGE_FIT_COVER, &condition_pixels[visual_count],
                    detail, sizeof(detail))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            condition_widths[visual_count] = render_width;
            condition_heights[visual_count] = render_height;
            condition_frames[visual_count] = 1;
            visual_count++;
        }
        vision_output_count = visual_count;
    }
    if (vision_output_count) {
        vision_outputs = calloc(vision_output_count, sizeof(*vision_outputs));
        if (!vision_outputs) {
            h3_set_error(ctx, "out of memory allocating Qwen vision outputs");
            goto cleanup;
        }
    }

    if (visual_count || ref2va) {
        /* Reference rows in request order: Ref2VA walks the references
         * (the DiT consumes its reference segments in that order), FL2VA
         * its frame anchors. */
        size_t pass_count = ref2va ? params->reference_count : visual_count;
        for (size_t pass = 0; pass < pass_count; pass++) {
            size_t rows;
            if (ref2va) {
                const h3_layout_ref *reference = &layout_references[pass];
                if (reference->kind == H3_LAYOUT_REF_AUDIO) continue;
                rows = (size_t)reference->latent_t *
                       (size_t)reference->latent_h *
                       (size_t)reference->latent_w / 4;
            } else {
                int image_latent_w, image_latent_h;
                h3_latent_canvas(condition_widths[pass], condition_heights[pass],
                                 &image_latent_w, &image_latent_h);
                rows = (size_t)h3_video_encoder_latent_t(condition_frames[pass]) *
                       (size_t)image_latent_h * (size_t)image_latent_w / 4;
            }
            if (rows > SIZE_MAX / 96 ||
                condition_video_elements > SIZE_MAX - rows * 96) {
                h3_set_error(ctx, "condition row count overflows");
                goto cleanup;
            }
            condition_video_elements += rows * 96;
        }
        if (condition_video_elements > SIZE_MAX /
                                       sizeof(*condition_video_rows)) {
            h3_set_error(ctx, "condition storage size overflows");
            goto cleanup;
        }
        if (condition_video_elements) {
            condition_video_rows = malloc(condition_video_elements *
                                          sizeof(*condition_video_rows));
            if (!condition_video_rows) {
                h3_set_error(ctx,
                    "out of memory allocating visual condition rows");
                goto cleanup;
            }
        }
        size_t condition_offset = 0;
        for (size_t pass = 0; pass < pass_count; pass++) {
            size_t image = pass;
            if (ref2va) {
                const h3_reference *reference = &params->references[pass];
                const h3_layout_ref *entry = &layout_references[pass];
                if (entry->kind == H3_LAYOUT_REF_AUDIO) continue;
                if (reference->kind == H3_REFERENCE_REFMOD) {
                    size_t row_elements = (size_t)entry->latent_t *
                                          (size_t)entry->latent_h *
                                          (size_t)entry->latent_w / 4 * 96;
                    h3_refmod_latent refmod;
                    if (!h3_refmod_load(reference->path, reference->member,
                                        reference->strength, &refmod, NULL,
                                        detail, sizeof(detail))) {
                        h3_set_error(ctx, "RefMod %zu: %s", pass + 1, detail);
                        goto cleanup;
                    }
                    if (refmod.kind == H3_REFMOD_AUDIO ||
                        refmod.time != entry->latent_t ||
                        refmod.height != entry->latent_h ||
                        refmod.width != entry->latent_w) {
                        h3_refmod_latent_free(&refmod);
                        h3_set_error(ctx,
                            "RefMod %zu changed between planning and loading",
                            pass + 1);
                        goto cleanup;
                    }
                    float *rows = condition_video_rows + condition_offset;
                    if (!h3_dit_patchify_video(
                            refmod.values, 24, refmod.time, refmod.height,
                            refmod.width, rows, row_elements)) {
                        h3_refmod_latent_free(&refmod);
                        h3_set_error(ctx, "cannot patchify RefMod latent");
                        goto cleanup;
                    }
                    h3_refmod_latent_free(&refmod);
                    condition_offset += row_elements;
                    continue;
                }
                image = reference_visual_indices[pass];
                if (image == SIZE_MAX) {
                    h3_set_error(ctx, "reference %zu has no decoded media",
                                 pass + 1);
                    goto cleanup;
                }
            }
            int image_latent_w, image_latent_h;
            h3_latent_canvas(condition_widths[image], condition_heights[image],
                             &image_latent_w, &image_latent_h);
            int image_latent_t = h3_video_encoder_latent_t(
                condition_frames[image]);
            size_t row_elements = (size_t)image_latent_t *
                                  (size_t)image_latent_h *
                                  (size_t)image_latent_w / 4 * 96;
            h3_video_latent latent;
            memset(&latent, 0, sizeof(latent));
            if (!h3_pooled_video_encode(
                    vae_path, condition_pixels[image],
                    condition_frames[image], condition_heights[image],
                    condition_widths[image],
                    h3_video_encoder_progress_bridge, &progress,
                    &latent, detail, sizeof(detail))) {
                h3_video_latent_free(&latent);
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            if (latent.time != image_latent_t ||
                latent.height != image_latent_h ||
                latent.width != image_latent_w) {
                h3_video_latent_free(&latent);
                h3_set_error(ctx,
                    "visual condition VAE produced unexpected latent geometry");
                goto cleanup;
            }
            float *rows = condition_video_rows + condition_offset;
            if (!h3_dit_patchify_video(
                    latent.values, 24, image_latent_t,
                    image_latent_h, image_latent_w,
                    rows, row_elements)) {
                h3_video_latent_free(&latent);
                h3_set_error(ctx, "cannot patchify visual condition latent");
                goto cleanup;
            }
            h3_video_latent_free(&latent);
            condition_offset += row_elements;
            if (progress.cancelled) goto cleanup;
        }
        if (condition_offset != condition_video_elements) {
            h3_set_error(ctx, "visual condition packing size mismatch");
            goto cleanup;
        }
        size_t vision_cursor = 0;
        for (size_t image = 0; image < visual_count; image++) {
            size_t reference_index = visual_reference_indices[image];
            if (!ref2va ||
                params->references[reference_index].kind == H3_REFERENCE_IMAGE) {
                if (!h3_pooled_vision_encode(
                        text_path, condition_pixels[image],
                        1, condition_heights[image], condition_widths[image],
                        h3_vision_progress_bridge, &progress,
                        &vision_outputs[vision_cursor], detail, sizeof(detail))) {
                    h3_set_error(ctx, "%s", detail);
                    goto cleanup;
                }
                if (ref2va)
                    presentations[reference_index].vision =
                        &vision_outputs[vision_cursor];
                vision_cursor++;
            } else {
                size_t blocks = presentations[reference_index].vision_count;
                presentations[reference_index].vision =
                    &vision_outputs[vision_cursor];
                size_t samples = ((size_t)condition_frames[image] + 11) / 12;
                for (size_t block = 0; block < blocks; block++) {
                    size_t first_sample = 2 * block;
                    size_t second_sample = first_sample + 1 < samples ?
                                           first_sample + 1 : first_sample;
                    int first = (int)(first_sample * 12);
                    int second = (int)(second_sample * 12);
                    float *pair = h3_extract_vision_pair(
                        condition_pixels[image], condition_frames[image],
                        condition_heights[image], condition_widths[image],
                        first, second);
                    if (!pair) {
                        h3_set_error(ctx,
                            "out of memory extracting Qwen video pair");
                        goto cleanup;
                    }
                    int ok = h3_pooled_vision_encode(
                        text_path, pair, 2,
                        condition_heights[image], condition_widths[image],
                        h3_vision_progress_bridge, &progress,
                        &vision_outputs[vision_cursor], detail, sizeof(detail));
                    free(pair);
                    if (!ok) {
                        h3_set_error(ctx, "%s", detail);
                        goto cleanup;
                    }
                    vision_cursor++;
                    if (progress.cancelled) goto cleanup;
                }
            }
            free(condition_pixels[image]);
            condition_pixels[image] = NULL;
            if (progress.cancelled) goto cleanup;
        }
        if (vision_cursor != vision_output_count) {
            h3_set_error(ctx, "Qwen reference vision count mismatch");
            goto cleanup;
        }
        for (size_t image = 0; image < vision_output_count; image++) {
            size_t bytes = vision_outputs[image].tokens * H3_VISION_OUTPUT_WIDTH *
                           sizeof(uint16_t);
            h3_debug_hash("vision merged", vision_outputs[image].merged, bytes);
            h3_debug_hash("vision deepstack0", vision_outputs[image].deepstack[0], bytes);
        }
        h3_progress_emit(&progress, "text encoder", 0, 50);
        void *text_pool = h3_autorelease_push();
        int text_ok = ref2va ? h3_multimodal_encode_ref2va_bf16(
                tokenizer, text_path, "h3_shaders.metal", prompt,
                presentations, params->reference_count,
                h3_text_progress_bridge, &progress, &text,
                detail, sizeof(detail)) :
            h3_multimodal_encode_fl2va_bf16(
                tokenizer, text_path, "h3_shaders.metal", prompt,
                vision_outputs, visual_count,
                h3_text_progress_bridge, &progress, &text,
                detail, sizeof(detail));
        h3_autorelease_pop(text_pool);
        if (!text_ok) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        for (size_t image = 0; image < vision_output_count; image++)
            h3_vision_output_free(&vision_outputs[image]);
    } else {
        if (!h3_tokenizer_encode(tokenizer, prompt, 1, &ids, &token_count,
                                 detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        h3_progress_emit(&progress, "text encoder", 0, 50);
        void *text_pool = h3_autorelease_push();
        int text_ok = h3_text_encode_bf16(
                text_path, "h3_shaders.metal", ids, token_count,
                h3_text_progress_bridge, &progress, &text,
                detail, sizeof(detail));
        h3_autorelease_pop(text_pool);
        if (!text_ok) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
    }
    if (continuation.active) {
        /* With Ref2VA references the pinned steps go before the reference
         * rows; otherwise after a last frame's - the order the keyframes
         * (and h3_layout_build) have them in. */
        float *pinned_video = NULL, *pinned_audio = NULL;
        size_t pinned_video_elements = 0, pinned_audio_elements = 0;
        detail[0] = '\0';
        if (!h3_continuation_rows(params, &continuation,
                                  &pinned_video, &pinned_video_elements,
                                  &pinned_audio, &pinned_audio_elements,
                                  detail, sizeof(detail)) ||
            !h3_join_rows(&condition_video_rows, &condition_video_elements,
                          pinned_video, pinned_video_elements, ref2va) ||
            !h3_join_rows(&condition_audio_rows, &condition_audio_elements,
                          pinned_audio, pinned_audio_elements, ref2va)) {
            free(pinned_video);
            free(pinned_audio);
            h3_set_error(ctx, "%s", detail[0] ? detail :
                         "out of memory pinning the continued clip");
            goto cleanup;
        }
        free(pinned_video);
        free(pinned_audio);
        for (int step = 0; step < continuation.steps; step++)
            keyframes[keyframe_count++] = continuation.keyframes[step];
        fprintf(stderr, "h3: continuing a %d-frame clip: %d frames pinned "
                "(%d latent steps), %d audio steps\n",
                continuation.file_frames, params->continue_frames,
                continuation.steps, continuation.audio_t);
    }
    conditioned = visual_count != 0 || condition_video_elements != 0 ||
                  condition_audio_elements != 0;
    if (h3_cache_target(ctx, H3_CACHE_CONDITIONING)) {
        if (!h3_conditioning_cache_store(
                ctx, conditioning_key, &text,
                condition_video_rows, condition_video_elements,
                condition_audio_rows, condition_audio_elements,
                layout_references, ref2va ? params->reference_count : 0,
                conditioned))
            fprintf(stderr, "h3: warning: could not retain conditioning cache\n");
        else
            fprintf(stderr, "h3: conditioning cache miss; stored exact BF16\n");
    }
    }
    h3_debug_hash("text", text.values,
                  text.tokens * text.width * sizeof(*text.values));
    h3_debug_hash("condition video", condition_video_rows,
                  condition_video_elements * sizeof(*condition_video_rows));
    h3_debug_hash("condition audio", condition_audio_rows,
                  condition_audio_elements * sizeof(*condition_audio_rows));
    if (conditioned && !h3_augment_conditions(
            params, ref2va, render_width, render_height, layout_references,
            keyframe_count,
            continuation.active ? (size_t)continuation.audio_t * 2 * 32 : 0,
            condition_video_rows, condition_video_elements,
            condition_audio_rows, condition_audio_elements)) {
        h3_set_error(ctx, "cannot apply seeded condition augmentation");
        goto cleanup;
    }
    if (conditioned)
        h3_debug_hash("augmented condition video", condition_video_rows,
                      condition_video_elements * sizeof(*condition_video_rows));
    if (progress.cancelled) goto cleanup;

    if (params->forced_audio_path) {
        /* As long as the decoder's own soundtrack would be, so the writer
         * muxes it exactly where generated audio would have gone. */
        forced_samples = temporal.audio_t * 800;
        if (!h3_av_read_audio_range_f32(
                params->forced_audio_path, params->forced_audio_offset,
                forced_samples, &forced_pcm, detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        h3_audio_latent latent;
        memset(&latent, 0, sizeof(latent));
        if (!h3_pooled_audio_encode(
                audio_vae_path, forced_pcm,
                forced_samples, h3_audio_encoder_progress_bridge, &progress,
                &latent, detail, sizeof(detail))) {
            h3_audio_latent_free(&latent);
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        if (latent.channels != 32 || latent.stereo != 2 ||
            latent.length != temporal.audio_t) {
            h3_audio_latent_free(&latent);
            h3_set_error(ctx, "forced audio encoder produced invalid geometry");
            goto cleanup;
        }
        forced_latent = latent.values;
        latent.values = NULL;
        h3_audio_latent_free(&latent);
        h3_debug_hash("forced audio", forced_latent,
                      (size_t)temporal.audio_t * 2 * 32 *
                          sizeof(*forced_latent));
        if (progress.cancelled) goto cleanup;
    }

    h3_layout_spec spec = {(int)text.tokens, temporal.video_t, latent_h,
                           latent_w, temporal.audio_t, temporal.frame_count,
                           keyframes, keyframe_count,
                           layout_references,
                           ref2va ? params->reference_count : 0,
                           continuation.audio_t, continuation.audio_start};
    if (!h3_layout_build(&spec, &layout, detail, sizeof(detail))) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    if (planned && (planned_text_tokens != text.tokens ||
                    planned_dit_tokens != layout.seq_len))
        fprintf(stderr, "h3: warning: the plan counted %zu text and %zu DiT "
                "tokens, the generation has %zu and %zu\n",
                planned_text_tokens, planned_dit_tokens, text.tokens,
                layout.seq_len);
    h3_sigma_schedule sigmas;
    if (!family->build_schedule(params->steps, h3_audio_sigma_shift(params),
                                &sigmas)) {
        h3_set_error(ctx, "cannot construct the requested sigma schedule");
        goto cleanup;
    }
    if (family->log_name)
        fprintf(stderr, "h3: %s %d-step schedule\n", family->log_name,
                params->steps);
    if (params->video_sigmas) {
        for (int step = 0; step <= sigmas.steps; step++)
            sigmas.video[step] = params->video_sigmas[step];
        fprintf(stderr, "h3: custom video sigmas, starting at %.6g\n",
                (double)sigmas.video[0]);
    }
    if (h3_forced_audio_clean(params)) {
        for (int step = 0; step <= sigmas.steps; step++) sigmas.audio[step] = 0.0f;
        fprintf(stderr, "h3: %s is held clean (audio sigma 0)\n",
                params->forced_audio_path ? "forced audio"
                                          : "the initial latent's audio");
    }
    float spatial_rope_scale = !params->use_reference_rope &&
        render_width == 256 && render_height == 256 ? 0.5f : 1.0f;
    h3_dit_prep *dit_prep =
        h3_cache_target(ctx, H3_CACHE_REFINED_TEXT | H3_CACHE_ADALN)
            ? ctx->dit_prep : NULL;
    if (h3_cache_target(ctx, H3_CACHE_DIT) && ctx->dit && ctx->dit_key &&
        !strcmp(ctx->dit_key, prepared_key)) {
        dit = ctx->dit;
        dit_is_cached = 1;
        if (!h3_dit_reset_run(
                dit, condition_video_rows, condition_video_elements,
                condition_audio_rows, condition_audio_elements,
                detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        fprintf(stderr, "h3: prepared DiT cache hit\n");
    } else if (conditioned) {
        dit = h3_dit_load_conditioned(
            dit_path, "h3_shaders.metal", &text, &layout, &sigmas,
            (unsigned)params->dit_layers, (unsigned)params->core_reuse,
            params->token_reduction,
            params->ssd_streaming,
            spatial_rope_scale,
            params->use_slower_bf16_mlp,
            params->use_slower_bf16_qkv,
            params->use_slower_bf16_attention_output,
            params->use_slower_row_major_attention_output,
            params->use_slower_unfused_int8_inputs,
            params->use_slower_unfused_qkv_rope,
            params->use_slower_scalar_qkv_rms,
            params->use_slower_uncached_int8_scales,
            params->use_slower_dynamic_fc1_k,
            params->use_slower_grouped_quantizer,
            params->use_int8_row_fc2,
            params->loras, params->lora_count, dit_prep,
            condition_video_rows, condition_video_elements,
            condition_audio_rows, condition_audio_elements,
            h3_dit_progress_bridge, &progress, detail, sizeof(detail));
    } else {
        dit = h3_dit_load_t2va(
            dit_path, "h3_shaders.metal", &text, &layout, &sigmas,
            (unsigned)params->dit_layers, (unsigned)params->core_reuse,
            params->token_reduction,
            params->ssd_streaming,
            spatial_rope_scale,
            params->use_slower_bf16_mlp,
            params->use_slower_bf16_qkv,
            params->use_slower_bf16_attention_output,
            params->use_slower_row_major_attention_output,
            params->use_slower_unfused_int8_inputs,
            params->use_slower_unfused_qkv_rope,
            params->use_slower_scalar_qkv_rms,
            params->use_slower_uncached_int8_scales,
            params->use_slower_dynamic_fc1_k,
            params->use_slower_grouped_quantizer,
            params->use_int8_row_fc2,
            params->loras, params->lora_count, dit_prep,
            h3_dit_progress_bridge, &progress, detail, sizeof(detail));
    }
    if (!dit) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    if (h3_cache_target(ctx, H3_CACHE_DIT) && !dit_is_cached) {
        char *key_copy = strdup(prepared_key);
        if (!key_copy) {
            fprintf(stderr, "h3: warning: could not retain prepared DiT key\n");
        } else {
            ctx->dit = dit;
            ctx->dit_key = key_copy;
            dit_is_cached = 1;
            fprintf(stderr, "h3: prepared DiT cache miss; model retained\n");
        }
    }
    h3_text_embedding_free(&text);
    free(condition_video_rows);
    condition_video_rows = NULL;
    free(condition_audio_rows);
    condition_audio_rows = NULL;
    if (progress.cancelled) goto cleanup;
    if (params->preview_denoise) {
        h3_progress_emit(&progress, "preview VAE load", 0, 36);
        preview_decoder = h3_acquire_video_decoder(
            ctx, decoder_key, vae_path, latent_h, latent_w,
            h3_preview_vae_progress_bridge, &progress,
            &decoder_is_cached, detail, sizeof(detail));
        if (preview_decoder && ctx->video_decoder == preview_decoder)
            h3_progress_emit(&progress, "preview VAE load", 36, 36);
        if (!preview_decoder) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        live_preview.progress = &progress;
        live_preview.decoder = preview_decoder;
        live_preview.latent_t = temporal.video_t;
        live_preview.latent_h = latent_h;
        live_preview.latent_w = latent_w;
        live_preview.output_frames = temporal.frame_count;
        live_preview.output_width = params->width;
        live_preview.output_height = params->height;
        if (progress.cancelled) goto cleanup;
    }
    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);
    video = malloc(video_count * sizeof(*video));
    audio = malloc(audio_count * sizeof(*audio));
    if (!video || !audio) {
        h3_set_error(ctx, "out of memory allocating joint H3 noise");
        goto cleanup;
    }
    /* The released server initializes each modality from a separate generator
     * carrying the same requested seed. */
    h3_rng video_rng, audio_rng;
    h3_rng_seed(&video_rng, params->seed);
    h3_rng_seed(&audio_rng, params->seed);
    h3_rng_fill_normal(&video_rng, video, video_count);
    h3_rng_fill_normal(&audio_rng, audio, audio_count);
    h3_debug_hash("noise video", video, video_count * sizeof(*video));
    h3_debug_hash("noise audio", audio, audio_count * sizeof(*audio));
    if (params->initial_latent_path) {
        float *initial_audio = NULL;
        if (video_count != (size_t)24 * temporal.video_t * latent_h * latent_w ||
            audio_count != (size_t)32 * 2 * temporal.audio_t) {
            h3_set_error(ctx, "the DiT's latent layout does not match the "
                              "initial latent file");
            goto cleanup;
        }
        if (!h3_initial_latent_apply(params->initial_latent_path, video,
                                     video_count, sigmas.video[0],
                                     audio_count, &initial_audio,
                                     detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        if (forced_latent) {
            free(initial_audio);
        } else {
            forced_latent = initial_audio;
            memcpy(audio, forced_latent, audio_count * sizeof(*audio));
        }
        h3_debug_hash("initial video", video, video_count * sizeof(*video));
    }
    h3_dit_set_attention_mode(dit, h3_resolve_attention_mode(params));
    h3_dit_set_sampler(dit, params->sampler, params->seed);
    /* Set on every run, so a retained DiT never carries it over. */
    if (!h3_dit_set_forced_audio(dit, forced_latent,
                                 forced_latent ? audio_count : 0,
                                 detail, sizeof(detail))) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    uint64_t ccv_calls_before = h3_gpu_ccv_attention_dispatch_count();
    uint64_t ccv_direct_before = h3_gpu_ccv_attention_direct_count();
    h3_dit_set_cancel(dit, params->cancel_requested);
    /* Set on every run (and cleared below), so a retained DiT never keeps a
     * callback into an earlier request. */
    h3_dit_set_latent_preview(dit, params->latent_preview_every,
                              params->latent_preview_every > 0
                                  ? h3_deliver_latent_preview : NULL,
                              (void *)params);
    int denoised = h3_dit_denoise_euler_preview(
            dit, video, audio, params->denoise_reuse,
            h3_dit_progress_bridge, &progress,
            preview_decoder ? h3_deliver_denoise_preview : NULL,
            preview_decoder ? &live_preview : NULL,
            detail, sizeof(detail));
    h3_dit_set_latent_preview(dit, 0, NULL, NULL);
    if (!denoised) {
        if (!live_preview.failed) h3_set_error(ctx, "%s", detail);
        h3_dit_set_cancel(dit, NULL);
        h3_dit_release_backend_scratch(dit);
        if (dit_is_cached) {
            ctx->dit = NULL;
            free(ctx->dit_key);
            ctx->dit_key = NULL;
            dit_is_cached = 0;
        }
        goto cleanup;
    }
    h3_dit_set_cancel(dit, NULL);
    h3_dit_release_backend_scratch(dit);
    h3_dit_set_forced_audio(dit, NULL, 0, NULL, 0);
    h3_debug_hash("denoised video", video, video_count * sizeof(*video));
    h3_debug_hash("denoised audio", audio, audio_count * sizeof(*audio));
    if (params->save_latent_path &&
        !h3_latent_file_write(params->save_latent_path, video,
                              temporal.video_t, latent_h, latent_w, audio,
                              temporal.audio_t, temporal.frame_count,
                              detail, sizeof(detail))) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    uint64_t ccv_calls = h3_gpu_ccv_attention_dispatch_count() -
                         ccv_calls_before;
    uint64_t ccv_direct_calls = h3_gpu_ccv_attention_direct_count() -
                                ccv_direct_before;
    if (!dit_is_cached) h3_dit_free(dit);
    dit = NULL;
    if (progress.cancelled) goto cleanup;
    h3_video_vae_set_cancel(params->cancel_requested);
    const char *decode_forced = h3_option("H3_FORCED_AUDIO_DECODE");
    if (forced_pcm && !(decode_forced && *decode_forced &&
                        strcmp(decode_forced, "0"))) {
        /* The file's own audio, not its trip through the audio VAE (set
         * H3_FORCED_AUDIO_DECODE=1 to hear the decoded latent instead). */
        waveform.channels = 2;
        waveform.samples = forced_samples;
        waveform.sample_rate = 32000;
        waveform.pcm = forced_pcm;
        forced_pcm = NULL;
    } else {
        h3_progress_emit(&progress, "audio VAE", 0, 7);
        if (!h3_audio_vae_decode(audio_vae_path, "h3_shaders.metal", audio,
                                 temporal.audio_t,
                                 h3_audio_vae_progress_bridge, &progress,
                                 &waveform, detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
    }
    /* The PCM handed to the encoder (the AAC track itself is not bit-exact). */
    h3_debug_hash("audio waveform", waveform.pcm,
                  (size_t)waveform.channels * (size_t)waveform.samples *
                      sizeof(*waveform.pcm));
    free(audio);
    audio = NULL;
    if (progress.cancelled) goto cleanup;
    if (!preview_decoder && h3_cache_target(ctx, H3_CACHE_DECODER)) {
        h3_progress_emit(&progress, "video VAE load", 0, 36);
        preview_decoder = h3_acquire_video_decoder(
            ctx, decoder_key, vae_path, latent_h, latent_w,
            h3_vae_progress_bridge, &progress,
            &decoder_is_cached, detail, sizeof(detail));
        if (preview_decoder && ctx->video_decoder == preview_decoder)
            h3_progress_emit(&progress, "video VAE load", 36, 36);
        if (!preview_decoder) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
    }
    if (preview_decoder) {
        /* Chunked path: decode, upscale, and mux/deliver one ~17-frame
         * chunk at a time (see h3_stream_encode_chunk above) instead of
         * retaining the whole video, so encoding chunk N overlaps with
         * decoding chunk N+1 and peak host memory for the video drops from
         * the full output to about one chunk. */
        int native_height = 0, native_width = 0;
        h3_video_vae_decoder_pixel_size(preview_decoder, &native_height,
                                        &native_width);
        int output_width = params->width, output_height = params->height;
        int chunks = (temporal.video_t - 2) / 5;
        int total_frames = chunks * 17 + 5;
        h3_av_writer *writer = NULL;
        if (params->output_path && *params->output_path) {
            writer = h3_av_writer_open(
                params->output_path, output_width, output_height, H3_FPS,
                waveform.pcm, waveform.samples, waveform.channels,
                waveform.sample_rate, params->video_hevc,
                params->video_quality, detail, sizeof(detail));
            if (!writer) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            h3_progress_emit(&progress, "encode", 0, total_frames);
        }
        h3_stream_encode_ctx stream = {
            writer, params->on_frame, params->callback_opaque, &progress,
            output_width, output_height,
            native_width != output_width || native_height != output_height,
            total_frames, 0, {0, 0}
        };
        int video_ok = h3_video_vae_decoder_decode_streamed(
            preview_decoder, video, temporal.video_t,
            h3_stream_encode_chunk, &stream, NULL, detail, sizeof(detail));
        free(video);
        video = NULL;
        if (writer) {
            char close_detail[512];
            close_detail[0] = '\0';
            if (!h3_av_writer_close(writer, close_detail,
                                        sizeof(close_detail)) && video_ok) {
                video_ok = 0;
                snprintf(detail, sizeof(detail), "%s", close_detail);
            }
        }
        if (!video_ok) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        h3_debug_hash_video_done(&stream.rgb_hash);
        if (writer)
            h3_progress_emit(&progress, "encode", total_frames, total_frames);
        if (progress.cancelled) goto cleanup;
        result = calloc(1, sizeof(*result));
        if (!result) {
            h3_set_error(ctx, "out of memory creating generation result");
            goto cleanup;
        }
        result->width = output_width;
        result->height = output_height;
        result->frames = total_frames;
        result->fps = H3_FPS;
        result->sample_rate = waveform.sample_rate;
        result->seed = params->seed;
        result->ccv_attention_calls = ccv_calls;
        result->ccv_attention_direct_calls = ccv_direct_calls;
    } else {
        /* No resident decoder (video VAE caching disabled): fall back to
         * the original monolithic decode-then-mux path. */
        if (!h3_video_vae_decode(
                vae_path, "h3_shaders.metal", video,
                temporal.video_t, latent_h, latent_w,
                h3_vae_progress_bridge, &progress, &frames,
                detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        free(video);
        video = NULL;
        if (progress.cancelled) goto cleanup;
        size_t rgb_count = (size_t)frames.frames * (size_t)frames.height *
                           (size_t)frames.width * 3;
        rgb8 = h3_rgb_f32_to_u8(frames.rgb, rgb_count);
        if (!rgb8) {
            h3_set_error(ctx, "out of memory converting generated RGB frames");
            goto cleanup;
        }
        int output_width = frames.width;
        int output_height = frames.height;
        if (output_width != params->width || output_height != params->height) {
            uint8_t *resized = NULL;
            if (!h3_resize_rgb24_high_quality(
                    rgb8, frames.frames, output_width, output_height,
                    params->width, params->height, &resized)) {
                h3_set_error(ctx, "cannot resize generated RGB frames");
                goto cleanup;
            }
            free(rgb8);
            rgb8 = resized;
            output_width = params->width;
            output_height = params->height;
        }
        h3_debug_video_hash rgb_hash = {0, 0};
        h3_debug_hash_frames(&rgb_hash, rgb8, (size_t)frames.frames,
                             (size_t)output_width * (size_t)output_height * 3);
        h3_debug_hash_video_done(&rgb_hash);
        if (params->on_frame) {
            size_t frame_bytes = (size_t)output_width * (size_t)output_height * 3;
            for (int index = 0; index < frames.frames; index++) {
                h3_frame frame = {output_width, output_height, output_width * 3,
                                  rgb8 + (size_t)index * frame_bytes,
                                  index, frames.frames, -1, 0};
                if (params->on_frame(&frame, params->callback_opaque)) {
                    h3_set_error(ctx, "generation cancelled while delivering frame %d",
                                 index);
                    goto cleanup;
                }
            }
        }
        if (params->output_path && *params->output_path) {
            h3_progress_emit(&progress, "encode", 0, frames.frames);
            if (!h3_av_write_rgb24_f32(
                    params->output_path, rgb8, frames.frames, output_width,
                    output_height, H3_FPS, waveform.pcm, waveform.samples,
                    waveform.channels, waveform.sample_rate,
                    params->video_hevc, params->video_quality, detail,
                    sizeof(detail))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            h3_progress_emit(&progress, "encode", frames.frames, frames.frames);
        }
        result = calloc(1, sizeof(*result));
        if (!result) {
            h3_set_error(ctx, "out of memory creating generation result");
            goto cleanup;
        }
        result->width = output_width;
        result->height = output_height;
        result->frames = frames.frames;
        result->fps = H3_FPS;
        result->sample_rate = waveform.sample_rate;
        result->seed = params->seed;
        result->ccv_attention_calls = ccv_calls;
        result->ccv_attention_direct_calls = ccv_direct_calls;
    }

cleanup:
    h3_memory_phase(NULL);
    h3_video_vae_set_cancel(NULL);
    free(forced_pcm);
    free(forced_latent);
    free(conditioning_key);
    free(prepared_key);
    free(decoder_key);
    free(tokenizer_path); free(text_path); free(dit_path); free(vae_path);
    free(audio_vae_path);
    h3_tokenizer_free(tokenizer);
    h3_tokenizer_ids_free(ids);
    for (size_t image = 0; image < visual_capacity; image++) {
        if (condition_pixels) free(condition_pixels[image]);
    }
    for (size_t image = 0; image < vision_output_count; image++)
        if (vision_outputs) h3_vision_output_free(&vision_outputs[image]);
    for (size_t index = 0; index < params->reference_count; index++)
        if (presentation_timestamps) free(presentation_timestamps[index]);
    free(condition_pixels);
    free(condition_widths);
    free(condition_heights);
    free(condition_frames);
    free(visual_reference_indices);
    free(reference_visual_indices);
    free(vision_outputs);
    free(presentations);
    free(presentation_timestamps);
    free(layout_references);
    free(condition_video_rows);
    free(condition_audio_rows);
    h3_text_embedding_free(&text);
    h3_layout_free(&layout);
    if (!dit_is_cached) h3_dit_free(dit);
    if (!decoder_is_cached) h3_video_vae_decoder_free(preview_decoder);
    free(video); free(audio); free(rgb8);
    h3_video_frames_free(&frames);
    h3_audio_waveform_free(&waveform);
    return result;
}

void h3_result_free(h3_result *result) {
    free(result);
}

int h3_debug_ccv_warmup(const char *shader_source_path, char *error,
                        size_t error_size) {
    h3_gpu *gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!gpu) return 0;
    uint32_t rows = 3211, heads = 56, head_dim = 128;
    size_t count = (size_t)rows * heads * head_dim;
    h3_gpu_tensor *q = h3_gpu_tensor_new_bf16(gpu, count);
    h3_gpu_tensor *k = h3_gpu_tensor_new_bf16(gpu, count);
    h3_gpu_tensor *v = h3_gpu_tensor_new_bf16(gpu, count);
    h3_gpu_tensor *o = h3_gpu_tensor_new_bf16(gpu, count);
    uint16_t *zeros = calloc(count, sizeof(uint16_t));
    if (!q || !k || !v || !o || !zeros) {
        free(zeros);
        h3_gpu_free(gpu);
        snprintf(error, error_size, "h3_debug_ccv_warmup: allocation failed");
        return 0;
    }
    h3_gpu_tensor_write_bf16(q, zeros, count);
    h3_gpu_tensor_write_bf16(k, zeros, count);
    h3_gpu_tensor_write_bf16(v, zeros, count);
    free(zeros);
    if (!h3_gpu_begin(gpu)) {
        snprintf(error, error_size, "h3_debug_ccv_warmup: h3_gpu_begin failed");
        h3_gpu_free(gpu);
        return 0;
    }
    int ok = h3_gpu_ccv_dense_attention_bf16(
        gpu, o, q, k, v, rows, heads, head_dim, 1.0f / sqrtf((float)head_dim),
        1, NULL);
    if (!ok) snprintf(error, error_size, "h3_debug_ccv_warmup: ccv call failed");
    h3_gpu_free(gpu);
    return ok;
}
