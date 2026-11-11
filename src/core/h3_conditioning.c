/* What a generation keeps between runs and carries between clips: the conditioning cache and its keys, latent files, continuation and joining of clips. Split out of h3.c. */
#include "h3_private.h"
void h3_conditioning_cache_clear(h3_ctx *ctx) {
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

int h3_cache_target(const h3_ctx *ctx, unsigned target) {
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

int h3_key_append(h3_key *key, const char *format, ...) {
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

int h3_latent_file_write(const char *path, const float *video,
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

int h3_continuation_plan(const h3_params *params, int latent_h,
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
int h3_continuation_rows(const h3_params *params,
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
int h3_initial_latent_check(const h3_params *params, int latent_h,
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
int h3_initial_latent_apply(const char *path, float *video,
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
int h3_join_rows(float **rows, size_t *elements,
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

char *h3_conditioning_key(const char *prompt, const h3_params *params,
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
float h3_audio_sigma_shift(const h3_params *params) {
    return params->audio_sigma_shift > 0.0f ? params->audio_sigma_shift
                                            : (float)H3_AUDIO_SIGMA_SHIFT;
}

/* Forced audio reaches the DiT fully clean - audio sigma 0 at every step.
 * H3_FORCED_AUDIO_MODE=noised instead holds it on its noise trajectory, the
 * way ComfyUI's noise mask does; on the same seeds that synced the mouth
 * clearly worse (lipsync-test/REPORT.md). Changes the sigma schedule, so it
 * is part of the prepared key. */
int h3_forced_audio_clean(const h3_params *params) {
    const char *mode = h3_option("H3_FORCED_AUDIO_MODE");
    if (params->forced_audio_path) return !(mode && !strcmp(mode, "noised"));
    /* An initial latent's own audio is held clean the same way. */
    return params->initial_latent_path != NULL;
}

char *h3_prepared_key(const char *conditioning,
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

int h3_conditioning_cache_store(
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

int h3_conditioning_cache_load(
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

int h3_augment_conditions(const h3_params *params, int ref2va,
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
