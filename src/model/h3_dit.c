#include "h3_dit_private.h"
void h3_dit_fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

unsigned h3_dit_command_block_interval(const h3_dit *dit) {
    const char *value = h3_option("H3_DIT_COMMAND_BLOCKS");
    if (value && *value) {
        char *end = NULL;
        long parsed = strtol(value, &end, 10);
        return end != value && !*end && parsed >= 0 &&
               parsed <= H3_DIT_BLOCKS ? (unsigned)parsed : 0;
    }
    if (h3_gpu_is_m5(dit->gpu))
        return dit->active_block_count * 3 / 5;
    return dit->active_block_count == H3_DIT_BLOCKS ? 30u : 0u;
}

int h3_dit_gpu_op(h3_dit *dit, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    h3_dit_fail(error, error_size, "%s: %s", operation, h3_gpu_error(dit->gpu));
    return 0;
}

void h3_dit_report(h3_dit_progress progress, void *opaque, const char *phase,
                   int completed, int total) {
    if (progress) progress(phase, completed, total, opaque);
}

void h3_dit_set_cancel(h3_dit *dit, const volatile int *cancel_requested) {
    if (dit) dit->cancel_requested = cancel_requested;
}

void h3_dit_set_latent_preview(h3_dit *dit, int every,
                               h3_dit_latent_preview preview, void *opaque) {
    if (!dit) return;
    dit->latent_preview_every = preview ? every : 0;
    dit->latent_preview = every > 0 ? preview : NULL;
    dit->latent_preview_opaque = opaque;
    dit->latent_preview_evaluations = 0;
}

/* Called once per evaluated step (0-based): whether it delivers a latent
 * preview. */
static int wants_latent_preview(h3_dit *dit, int step) {
    if (!dit->latent_preview || dit->latent_preview_every <= 0) return 0;
    dit->latent_preview_evaluations++;
    return dit->latent_preview_evaluations % dit->latent_preview_every == 0 &&
           step + 1 < dit->sigmas.steps;
}

/* clean = sample + sigma * velocity, channel-major, handed to the preview. */
static int deliver_latent_preview(h3_dit *dit, int step, const float *clean,
                                  char *error, size_t error_size) {
    if (dit->latent_preview(step + 1, dit->sigmas.steps, clean,
                            dit->latent_t, dit->latent_h, dit->latent_w,
                            dit->latent_preview_opaque)) {
        h3_dit_fail(error, error_size, "denoising preview stopped at step %d",
             step + 1);
        return 0;
    }
    return 1;
}

/* A stop was asked for: let the GPU finish what is already encoded (a
 * command left open would break the next h3_gpu_begin), then h3_dit_fail. */
int h3_dit_stop_if_cancelled(h3_dit *dit, char *error, size_t error_size) {
    if (!dit->cancel_requested || !*dit->cancel_requested) return 0;
    if (h3_gpu_command_open(dit->gpu)) (void)h3_gpu_submit(dit->gpu);
    h3_dit_fail(error, error_size, "generation cancelled");
    return 1;
}

void h3_dit_free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

h3_gpu_tensor *h3_dit_bf1(h3_dit *dit, const char *name, uint64_t width,
                          char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_bf16(dit->weights, dit->gpu, name, 1, shape,
                               error, error_size);
}

h3_gpu_tensor *h3_dit_bf2(h3_dit *dit, const char *name, uint64_t rows,
                          uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_bf16(dit->weights, dit->gpu, name, 2, shape,
                               error, error_size);
}

/* h3_dit_bf2() for one of the four LoRA-able projections: the LoRA delta is added
 * on the GPU right after upload, so any later int8 quantization
 * (h3_dit_quantize_block_qkv() etc.) sees the patched weight exactly as it would a
 * plain load. */
h3_gpu_tensor *h3_dit_bf2_patched(h3_dit *dit, const char *name,
                                  uint64_t rows, uint64_t columns,
                                  int refiner, unsigned block, int projection,
                                  char *error, size_t error_size) {
    h3_gpu_tensor *weight = h3_dit_bf2(dit, name, rows, columns, error, error_size);
    if (!weight ||
        !h3_lora_set_covers(dit->loras, refiner, block, projection))
        return weight;
    if (!h3_gpu_begin(dit->gpu) ||
        !h3_lora_set_apply_bf16(dit->loras, dit->gpu, refiner, block,
                                projection, weight) ||
        !h3_gpu_submit(dit->gpu)) {
        h3_dit_fail(error, error_size, "cannot apply LoRA to %s: %s", name,
             h3_gpu_error(dit->gpu));
        h3_gpu_tensor_free(weight);
        return NULL;
    }
    return weight;
}

h3_gpu_tensor *h3_dit_f1(h3_dit *dit, const char *name, uint64_t width,
                         char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_f32(dit->weights, dit->gpu, name, 1, shape,
                              error, error_size);
}

h3_gpu_tensor *h3_dit_f2(h3_dit *dit, const char *name, uint64_t rows,
                         uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_f32(dit->weights, dit->gpu, name, 2, shape,
                              error, error_size);
}

int h3_dit_copy_layout(h3_dit *dit, const h3_layout *layout,
                       char *error, size_t error_size) {
    dit->layout = *layout;
    dit->layout.segments = NULL;
    dit->layout.positions = NULL;
    if (layout->segment_count) {
        dit->layout.segments = malloc(layout->segment_count *
                                      sizeof(*layout->segments));
        if (!dit->layout.segments) goto oom;
        memcpy(dit->layout.segments, layout->segments,
               layout->segment_count * sizeof(*layout->segments));
    }
    if (layout->seq_len) {
        dit->layout.positions = malloc(layout->seq_len *
                                       sizeof(*layout->positions));
        if (!dit->layout.positions) goto oom;
        memcpy(dit->layout.positions, layout->positions,
               layout->seq_len * sizeof(*layout->positions));
    }
    return 1;
oom:
    h3_dit_fail(error, error_size, "out of memory copying packed H3 layout");
    h3_layout_free(&dit->layout);
    return 0;
}

int h3_dit_validate_layout(h3_dit *dit, const h3_text_embedding *text,
                           char *error, size_t error_size) {
    const h3_layout *layout = &dit->layout;
    if (!text || !text->values || text->width != TEXT_DIM || !text->tokens ||
        layout->signature[0] != (int)text->tokens ||
        !layout->segments || layout->segment_count < 3 ||
        layout->segments[0].kind != H3_SEG_TEXT ||
        layout->segments[layout->segment_count - 1].kind != H3_SEG_VIDEO ||
        layout->signature[1] < 1 || layout->signature[2] < 2 ||
        layout->signature[3] < 2 || layout->signature[4] < 1 ||
        layout->signature[2] % 2 || layout->signature[3] % 2 ||
        layout->seq_len > UINT32_MAX || text->tokens > UINT32_MAX ||
        layout->img_cond_rows > UINT32_MAX ||
        layout->audio_cond_rows > UINT32_MAX ||
        layout->audio_target_rows > UINT32_MAX ||
        layout->img_target_rows > UINT32_MAX) {
        h3_dit_fail(error, error_size,
             "DiT requires a valid contiguous H3 packed layout");
        return 0;
    }
    size_t cursor = 0, text_rows = 0, video_condition = 0;
    size_t audio_condition = 0, video_target = 0, audio_target = 0;
    unsigned target_video_segments = 0, target_audio_segments = 0;
    for (size_t index = 0; index < layout->segment_count; index++) {
        const h3_segment *segment = &layout->segments[index];
        if (segment->start != cursor || segment->stop < segment->start ||
            segment->stop > layout->seq_len) {
            h3_dit_fail(error, error_size, "DiT layout segments are not contiguous");
            return 0;
        }
        size_t rows = segment->stop - segment->start;
        switch (segment->kind) {
        case H3_SEG_TEXT: text_rows += rows; break;
        case H3_SEG_COND:
        case H3_SEG_REF_IMAGE: video_condition += rows; break;
        case H3_SEG_REF_AUDIO: audio_condition += rows; break;
        case H3_SEG_AUDIO:
            audio_target += rows;
            target_audio_segments++;
            dit->audio_target_start = (uint32_t)segment->start;
            break;
        case H3_SEG_VIDEO:
            video_target += rows;
            target_video_segments++;
            dit->video_target_start = (uint32_t)segment->start;
            break;
        default:
            h3_dit_fail(error, error_size, "DiT layout contains an unknown segment");
            return 0;
        }
        cursor = segment->stop;
    }
    if (cursor != layout->seq_len || text_rows != text->tokens ||
        video_condition != layout->img_cond_rows ||
        audio_condition != layout->audio_cond_rows ||
        video_target != layout->img_target_rows ||
        audio_target != layout->audio_target_rows ||
        target_video_segments != 1 || target_audio_segments != 1 ||
        video_condition > UINT32_MAX - video_target ||
        audio_condition > UINT32_MAX - audio_target) {
        h3_dit_fail(error, error_size, "DiT layout row-source counts are inconsistent");
        return 0;
    }
    if (text->tags) {
        for (size_t index = 0; index < text->tokens; index++) {
            if (text->tags[index] >= H3_DIT_MODALITIES) {
                h3_dit_fail(error, error_size, "DiT text presentation has an invalid tag");
                return 0;
            }
        }
    }
    dit->latent_t = layout->signature[1];
    dit->latent_h = layout->signature[2];
    dit->latent_w = layout->signature[3];
    dit->audio_t = layout->signature[4];
    dit->text_rows = (uint32_t)text->tokens;
    dit->video_condition_rows = (uint32_t)video_condition;
    dit->audio_condition_rows = (uint32_t)audio_condition;
    dit->audio_rows = (uint32_t)layout->audio_target_rows;
    dit->video_rows = (uint32_t)layout->img_target_rows;
    dit->video_total_rows = (uint32_t)(video_condition + video_target);
    dit->audio_total_rows = (uint32_t)(audio_condition + audio_target);
    dit->sequence = (uint32_t)layout->seq_len;
    return 1;
}

int h3_dit_configure_token_reduction(h3_dit *dit, int requested,
                                     char *error, size_t error_size) {
    const char *enabled = h3_option("H3_TOKEN_REDUCTION");
    if (!requested &&
        (!enabled || !*enabled || !strcmp(enabled, "0"))) return 1;
    unsigned begin = 4, end = 30;
    const char *range = h3_option("H3_TOKEN_REDUCTION_BLOCKS");
    if (range && *range) {
        char *middle = NULL;
        unsigned long parsed_begin = strtoul(range, &middle, 10);
        if (middle == range || *middle != ':') {
            h3_dit_fail(error, error_size,
                 "H3_TOKEN_REDUCTION_BLOCKS must be BEGIN:END");
            return 0;
        }
        char *tail = NULL;
        unsigned long parsed_end = strtoul(middle + 1, &tail, 10);
        if (tail == middle + 1 || *tail || parsed_begin >= parsed_end ||
            parsed_end > H3_DIT_BLOCKS) {
            h3_dit_fail(error, error_size,
                 "token-reduction block range must satisfy 0 <= BEGIN < END <= 50");
            return 0;
        }
        begin = (unsigned)parsed_begin;
        end = (unsigned)parsed_end;
    }
    /* Coarse structure is tolerant of a deeper reduced stack while the first
     * noisy samples form. Restore earlier once fine detail starts resolving. */
    unsigned early_steps = end < 40 ? 10 : 0;
    unsigned early_end = end < 40 ? 40 : end;
    const char *early = h3_option("H3_TOKEN_REDUCTION_EARLY");
    if (early && *early) {
        if (!strcmp(early, "0")) {
            early_steps = 0;
            early_end = end;
        } else {
            char *middle = NULL;
            unsigned long parsed_steps = strtoul(early, &middle, 10);
            if (middle == early || *middle != ':') {
                h3_dit_fail(error, error_size,
                     "H3_TOKEN_REDUCTION_EARLY must be STEPS:END");
                return 0;
            }
            char *tail = NULL;
            unsigned long parsed_end = strtoul(middle + 1, &tail, 10);
            if (tail == middle + 1 || *tail || !parsed_steps ||
                parsed_steps > 1000 || parsed_end <= end ||
                parsed_end > H3_DIT_BLOCKS) {
                h3_dit_fail(error, error_size,
                     "early token reduction requires STEPS > 0 and "
                     "base END < END <= 50");
                return 0;
            }
            early_steps = (unsigned)parsed_steps;
            early_end = (unsigned)parsed_end;
        }
    }
    float scale = 1.0f;
    const char *scale_text = h3_option("H3_TOKEN_REDUCTION_SCALE");
    if (scale_text && *scale_text) {
        char *tail = NULL;
        scale = strtof(scale_text, &tail);
        if (tail == scale_text || *tail || !isfinite(scale) ||
            scale < 0.0f || scale > 2.0f) {
            h3_dit_fail(error, error_size,
                 "H3_TOKEN_REDUCTION_SCALE must be in [0, 2]");
            return 0;
        }
    }
    uint32_t spatial_height = (uint32_t)dit->latent_h / 2;
    uint32_t spatial_width = (uint32_t)dit->latent_w / 2;
    uint32_t reduced_width = (spatial_width + 1) / 2;
    uint64_t reduced_video =
        (uint64_t)(uint32_t)dit->latent_t * spatial_height * reduced_width;
    if (!spatial_height || !spatial_width ||
        (uint64_t)(uint32_t)dit->latent_t * spatial_height * spatial_width !=
            dit->video_rows ||
        dit->video_target_start + dit->video_rows != dit->sequence ||
        reduced_video > UINT32_MAX ||
        reduced_video > UINT32_MAX - dit->video_target_start) {
        h3_dit_fail(error, error_size,
             "token reduction requires the target video to end the packed layout");
        return 0;
    }
    dit->token_reduction = 1;
    dit->token_reduction_begin = begin;
    dit->token_reduction_end = end;
    dit->token_reduction_early_steps = early_steps;
    dit->token_reduction_early_end = early_end;
    dit->token_reduction_scale = scale;
    dit->reduced_video_rows = (uint32_t)reduced_video;
    dit->token_baseline_rows = dit->video_rows - dit->reduced_video_rows;
    dit->reduced_sequence = dit->video_target_start +
                            dit->reduced_video_rows;
    return 1;
}

void h3_dit_token_pool_sources(const h3_dit *dit, uint32_t reduced_row,
                               uint32_t *first, uint32_t *second) {
    if (reduced_row < dit->video_target_start) {
        *first = reduced_row;
        *second = reduced_row;
        return;
    }
    uint32_t spatial_width = (uint32_t)dit->latent_w / 2;
    uint32_t reduced_width = (spatial_width + 1) / 2;
    uint32_t local = reduced_row - dit->video_target_start;
    uint32_t source = dit->video_target_start +
        (local / reduced_width) * spatial_width +
        (local % reduced_width) * 2;
    *first = source;
    *second = source + ((source - dit->video_target_start) % spatial_width + 1 <
                        spatial_width ? 1u : 0u);
}

uint32_t h3_dit_token_reduced_parent(const h3_dit *dit, uint32_t full_row) {
    if (full_row < dit->video_target_start) return full_row;
    uint32_t spatial_width = (uint32_t)dit->latent_w / 2;
    uint32_t reduced_width = (spatial_width + 1) / 2;
    uint32_t local = full_row - dit->video_target_start;
    return dit->video_target_start + (local / spatial_width) * reduced_width +
           (local % spatial_width) / 2;
}

size_t h3_dit_video_elements(const h3_dit *dit) {
    return dit ? (size_t)VIDEO_CHANNELS * (size_t)dit->latent_t *
        (size_t)dit->latent_h * (size_t)dit->latent_w : 0;
}

size_t h3_dit_audio_elements(const h3_dit *dit) {
    return dit ? (size_t)AUDIO_CHANNELS * AUDIO_STREAMS *
        (size_t)dit->audio_t : 0;
}

int h3_dit_reset_run(h3_dit *dit,
                     const float *condition_video_rows,
                     size_t condition_video_elements,
                     const float *condition_audio_rows,
                     size_t condition_audio_elements,
                     char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit) {
        h3_dit_fail(error, error_size, "prepared DiT is absent");
        return 0;
    }
    size_t wanted_video =
        (size_t)dit->video_condition_rows * VIDEO_PATCH;
    size_t wanted_audio =
        (size_t)dit->audio_condition_rows * AUDIO_CHANNELS;
    if (condition_video_elements != wanted_video ||
        condition_audio_elements != wanted_audio ||
        (wanted_video && !condition_video_rows) ||
        (wanted_audio && !condition_audio_rows)) {
        h3_dit_fail(error, error_size, "prepared DiT condition rows do not match");
        return 0;
    }
    if ((wanted_video && !h3_gpu_tensor_write_f32_range(
             dit->video_input, 0, condition_video_rows, wanted_video)) ||
        (wanted_audio && !h3_gpu_tensor_write_f32_range(
             dit->audio_input, 0, condition_audio_rows, wanted_audio))) {
        h3_dit_fail(error, error_size, "cannot refresh prepared DiT conditions");
        return 0;
    }
    dit->core_forward_count = 0;
    dit->core_residual_ready = 0;
    return 1;
}

int h3_dit_forward(h3_dit *dit, int step,
                   const float *video_latent, const float *audio_latent,
                   float *video_velocity, float *audio_velocity,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || step < 0 || step >= h3_dit_schedule_steps(dit->schedule) ||
        !video_latent || !audio_latent || !video_velocity || !audio_velocity) {
        h3_dit_fail(error, error_size, "invalid DiT forward arguments");
        return 0;
    }
    size_t video_row_elements = (size_t)dit->video_rows * VIDEO_PATCH;
    size_t audio_row_elements = (size_t)dit->audio_rows * AUDIO_CHANNELS;
    float *video_rows = malloc(video_row_elements * sizeof(*video_rows));
    float *audio_rows = malloc(audio_row_elements * sizeof(*audio_rows));
    uint16_t *video_out = malloc(video_row_elements * sizeof(*video_out));
    uint16_t *audio_out = malloc(audio_row_elements * sizeof(*audio_out));
    float *video_f32 = malloc(video_row_elements * sizeof(*video_f32));
    float *audio_f32 = malloc(audio_row_elements * sizeof(*audio_f32));
    if (!video_rows || !audio_rows || !video_out || !audio_out ||
        !video_f32 || !audio_f32) {
        h3_dit_fail(error, error_size, "out of memory packing DiT latents");
        free(video_rows); free(audio_rows); free(video_out); free(audio_out);
        free(video_f32); free(audio_f32);
        return 0;
    }
    int ok = h3_dit_patchify_video(video_latent, VIDEO_CHANNELS,
        dit->latent_t, dit->latent_h, dit->latent_w, video_rows,
        video_row_elements) &&
        h3_dit_pack_audio(audio_latent, AUDIO_CHANNELS, dit->audio_t,
                          audio_rows, audio_row_elements) &&
        h3_gpu_tensor_write_f32_range(
            dit->video_input,
            (size_t)dit->video_condition_rows * VIDEO_PATCH,
            video_rows, video_row_elements) &&
        h3_gpu_tensor_write_f32_range(
            dit->audio_input,
            (size_t)dit->audio_condition_rows * AUDIO_CHANNELS,
            audio_rows, audio_row_elements);
    if (!ok) h3_dit_fail(error, error_size, "cannot pack/write DiT input latents");
    if (ok) ok = h3_dit_encode_forward(dit, step, 1, 1, 0, error, error_size);
    if (ok) ok = h3_gpu_tensor_read_bf16(dit->video_output_bf16, video_out,
                                         video_row_elements) &&
                 h3_gpu_tensor_read_bf16(dit->audio_output_bf16, audio_out,
                                         audio_row_elements);
    if (!ok && (!error || !*error)) h3_dit_fail(error, error_size, "cannot read DiT output");
    if (ok) {
        for (size_t index = 0; index < video_row_elements; index++) {
            uint32_t bits = (uint32_t)video_out[index] << 16;
            memcpy(&video_f32[index], &bits, sizeof(bits));
        }
        for (size_t index = 0; index < audio_row_elements; index++) {
            uint32_t bits = (uint32_t)audio_out[index] << 16;
            memcpy(&audio_f32[index], &bits, sizeof(bits));
        }
    }
    if (ok) ok = h3_dit_unpatchify_video(video_f32, VIDEO_CHANNELS,
        dit->latent_t, dit->latent_h, dit->latent_w, video_velocity,
        h3_dit_video_elements(dit)) &&
        h3_dit_unpack_audio(audio_f32, AUDIO_CHANNELS, dit->audio_t,
                            audio_velocity, h3_dit_audio_elements(dit));
    if (!ok && (!error || !*error)) h3_dit_fail(error, error_size, "cannot unpack DiT output");
    free(video_rows); free(audio_rows); free(video_out); free(audio_out);
    free(video_f32); free(audio_f32);
    return ok;
}

int h3_dit_get_gpu_stats(const h3_dit *dit, h3_gpu_stats *stats) {
    return dit && h3_gpu_get_stats(dit->gpu, stats);
}

static float extrapolation_ratio(float current_sigma, float last_sigma,
                                 float previous_sigma, int have_previous) {
    if (!have_previous) return 0.0f;
    float denominator = last_sigma - previous_sigma;
    float ratio = denominator != 0.0f
        ? (current_sigma - last_sigma) / denominator : 0.0f;
    /* Reuse intervals are deliberately small. This guard prevents malformed
     * custom schedules from turning one cached evaluation into an explosion. */
    if (ratio < -2.0f) ratio = -2.0f;
    if (ratio > 2.0f) ratio = 2.0f;
    return ratio;
}

static void extrapolate_velocity(float *output, const float *last,
                                 const float *previous, size_t count,
                                 float current_sigma, float last_sigma,
                                 float previous_sigma, int have_previous) {
    if (!have_previous) {
        memcpy(output, last, count * sizeof(*output));
        return;
    }
    float ratio = extrapolation_ratio(current_sigma, last_sigma,
                                      previous_sigma, have_previous);
    for (size_t index = 0; index < count; index++)
        output[index] = last[index] +
                        ratio * (last[index] - previous[index]);
}

int h3_dit_reuse_schedule(int steps, int reuse_interval, uint8_t *selected,
                          size_t selected_count) {
    if (steps < 1 || reuse_interval < 1 || reuse_interval > 32 || !selected ||
        selected_count < (size_t)steps) return -1;
    memset(selected, 0, (size_t)steps);

    int count = 0;
    for (int step = 0; step < steps; step++) {
        if (reuse_interval == 1 || step == 0 || step == steps - 1 ||
            step % reuse_interval == 0) {
            selected[step] = 1;
            count++;
        }
    }
    return count;
}

static int parse_reuse_steps(int steps, uint8_t *selected) {
    const char *text = h3_option("H3_REUSE_STEPS");
    if (!text || !*text) return 0;
    memset(selected, 0, (size_t)steps);
    int count = 0;
    int previous = -1;
    while (*text) {
        char *end = NULL;
        long value = strtol(text, &end, 10);
        if (end == text || value < 0 || value >= steps ||
            value <= previous) return -1;
        selected[value] = 1;
        previous = (int)value;
        count++;
        if (!*end) break;
        if (*end != ',') return -1;
        text = end + 1;
        if (!*text) return -1;
    }
    return selected[0] && selected[steps - 1] ? count : -1;
}

static int gpu_sampler_requested(const h3_dit *dit) {
    const char *cpu = h3_option("H3_CPU_SAMPLER");
    if (cpu && *cpu && strcmp(cpu, "0")) return 0;
    const char *value = h3_option("H3_GPU_SAMPLER");
    if (value) return *value && strcmp(value, "0");
    return h3_gpu_is_m5(dit->gpu);
}

static unsigned gpu_sampler_window(void) {
    const char *value = h3_option("H3_GPU_SAMPLER_WINDOW");
    if (!value || !*value) return 1;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    return end != value && !*end && parsed >= 0 && parsed <= H3_MAX_STEPS
        ? (unsigned)parsed : 1;
}

static int ensure_previous_velocities(h3_dit *dit, char *error,
                                      size_t error_size) {
    if (dit->previous_video_velocity && dit->previous_audio_velocity) return 1;
    h3_dit_free_tensor(&dit->previous_video_velocity);
    h3_dit_free_tensor(&dit->previous_audio_velocity);
    dit->previous_video_velocity = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)dit->video_rows * VIDEO_PATCH);
    dit->previous_audio_velocity = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)dit->audio_rows * AUDIO_CHANNELS);
    if (dit->previous_video_velocity && dit->previous_audio_velocity) return 1;
    h3_dit_free_tensor(&dit->previous_video_velocity);
    h3_dit_free_tensor(&dit->previous_audio_velocity);
    h3_dit_fail(error, error_size, "cannot allocate GPU Euler velocity cache: %s",
         h3_gpu_error(dit->gpu));
    return 0;
}

static int denoise_euler_gpu(h3_dit *dit, float *video_latent,
                             float *audio_latent, int reuse_interval,
                             h3_dit_progress progress, void *progress_opaque,
                             h3_dit_preview preview, void *preview_opaque,
                             char *error, size_t error_size) {
    int renoise = dit->sampler == H3_SAMPLER_RENOISE;
    uint8_t selected[H3_MAX_STEPS] = {0};
    int selected_count = h3_dit_reuse_schedule(
        dit->sigmas.steps, reuse_interval, selected, sizeof(selected));
    int custom_count = reuse_interval > 1 ?
        parse_reuse_steps(dit->sigmas.steps, selected) : 0;
    if (selected_count < 0 || custom_count < 0) {
        h3_dit_fail(error, error_size,
             "H3_REUSE_STEPS must be increasing and include 0 and %d",
             dit->sigmas.steps - 1);
        return 0;
    }
    if (custom_count > 0) selected_count = custom_count;
    if (reuse_interval > 1 && h3_option("H3_PROFILE"))
        fprintf(stderr, "h3: %s GPU reuse schedule has %d evaluations\n",
                custom_count > 0 ? "custom" : "selected", selected_count);
    unsigned window = gpu_sampler_window();
    int disable_command_split = window == 1 &&
                                h3_option("H3_DIT_COMMAND_BLOCKS") == NULL;
    if (h3_option("H3_PROFILE"))
        fprintf(stderr, "h3: GPU sampler encode window is %s; internal split "
                "%s\n", window ? "bounded" : "unbounded",
                disable_command_split ? "disabled" : "enabled");

    size_t video_count = (size_t)dit->video_rows * VIDEO_PATCH;
    size_t audio_count = (size_t)dit->audio_rows * AUDIO_CHANNELS;
    size_t video_offset = (size_t)dit->video_condition_rows * VIDEO_PATCH;
    size_t audio_offset = (size_t)dit->audio_condition_rows * AUDIO_CHANNELS;
    if (video_count > UINT32_MAX || audio_count > UINT32_MAX ||
        video_offset > UINT32_MAX - video_count ||
        audio_offset > UINT32_MAX - audio_count ||
        (reuse_interval > 1 &&
         !ensure_previous_velocities(dit, error, error_size))) return 0;

    float *video_rows = malloc(video_count * sizeof(*video_rows));
    float *audio_rows = malloc(audio_count * sizeof(*audio_rows));
    if (!video_rows || !audio_rows) {
        h3_dit_fail(error, error_size, "out of memory packing GPU Euler latents");
        free(video_rows);
        free(audio_rows);
        return 0;
    }
    /* Forced audio: start the audio on the clean latent's trajectory and
     * keep the velocity that holds it there in forced_audio_velocity; it
     * replaces the model's audio velocity after every evaluation. */
    int forced = dit->forced_audio != NULL;
    /* Audio sigma 0 throughout (h3.c's default for forced audio): the audio
     * simply is the clean latent and is never stepped. */
    int forced_clean = forced && dit->sigmas.audio[0] == 0.0f;
    float *forced_velocity = forced
        ? malloc(audio_count * sizeof(*forced_velocity)) : NULL;
    float *forced_rows = forced
        ? malloc(audio_count * sizeof(*forced_rows)) : NULL;
    if (forced && !dit->forced_audio_velocity)
        dit->forced_audio_velocity = h3_gpu_tensor_new_f32(dit->gpu,
                                                           audio_count);
    if (forced && (!forced_velocity || !forced_rows ||
                   !dit->forced_audio_velocity ||
                   !h3_forced_trajectory_start(
                       audio_latent, forced_velocity, dit->forced_audio,
                       h3_dit_audio_elements(dit), dit->sigmas.audio[0]) ||
                   !h3_dit_pack_audio(forced_velocity, AUDIO_CHANNELS,
                                      dit->audio_t, forced_rows,
                                      audio_count) ||
                   !h3_gpu_tensor_write_f32(dit->forced_audio_velocity,
                                            forced_rows, audio_count))) {
        h3_dit_fail(error, error_size, "cannot prepare GPU forced audio");
        free(forced_velocity);
        free(forced_rows);
        free(video_rows);
        free(audio_rows);
        return 0;
    }
    free(forced_velocity);
    free(forced_rows);
    int ok = h3_dit_patchify_video(video_latent, VIDEO_CHANNELS,
        dit->latent_t, dit->latent_h, dit->latent_w, video_rows, video_count) &&
        h3_dit_pack_audio(audio_latent, AUDIO_CHANNELS, dit->audio_t,
                          audio_rows, audio_count) &&
        h3_gpu_tensor_write_f32_range(dit->video_input, video_offset,
                                      video_rows, video_count) &&
        h3_gpu_tensor_write_f32_range(dit->audio_input, audio_offset,
                                      audio_rows, audio_count);
    if (!ok) h3_dit_fail(error, error_size, "cannot pack/write GPU Euler latents");

    int last_evaluated = -1;
    int previous_evaluated = -1;
    unsigned pending_evaluations = 0;
    int command_active = 0;
    for (int step = 0; step < dit->sigmas.steps && ok; step++) {
        if (h3_dit_stop_if_cancelled(dit, error, error_size)) {
            ok = 0;
            break;
        }
        h3_dit_report(progress, progress_opaque, "denoise enqueue", step,
               dit->sigmas.steps);
        if (!command_active) {
            ok = h3_dit_gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                        "begin GPU Euler command chain");
            command_active = ok;
        }
        if (!ok) break;
        int evaluate = selected[step];
        if (evaluate) {
            if (last_evaluated >= 0 && reuse_interval > 1) {
                ok = h3_dit_gpu_op(dit, h3_gpu_copy_bf16(
                    dit->gpu, dit->previous_video_velocity, 0,
                    dit->video_output_bf16, 0, video_count),
                    error, error_size, "cache previous video velocity") &&
                    h3_dit_gpu_op(dit, h3_gpu_copy_bf16(
                    dit->gpu, dit->previous_audio_velocity, 0,
                    dit->audio_output_bf16, 0, audio_count),
                    error, error_size, "cache previous audio velocity");
                if (ok) previous_evaluated = last_evaluated;
            }
            if (ok) ok = h3_dit_encode_forward(dit, step, 0, 0,
                                        disable_command_split,
                                        error, error_size);
            if (ok && forced)
                ok = h3_dit_gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                    dit->gpu, dit->audio_output_bf16,
                    dit->forced_audio_velocity, (uint32_t)audio_count),
                    error, error_size, "force audio velocity");
            if (ok) {
                last_evaluated = step;
                pending_evaluations++;
            }
        }
        if (!ok) break;
        if (evaluate && wants_latent_preview(dit, step)) {
            /* Before the update: x_t and this step's fresh velocity give the
             * clean estimate for Euler and re-noise alike. */
            uint16_t *velocity = malloc(video_count * sizeof(*velocity));
            float *clean = malloc(h3_dit_video_elements(dit) * sizeof(*clean));
            ok = h3_dit_gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                        "submit GPU preview step");
            command_active = 0;
            pending_evaluations = 0;
            if (ok && (!velocity || !clean)) {
                h3_dit_fail(error, error_size, "out of memory for the latent preview");
                ok = 0;
            }
            if (ok && !(h3_gpu_tensor_read_f32_range(
                            dit->video_input, video_offset, video_rows,
                            video_count) &&
                        h3_gpu_tensor_read_bf16(dit->video_output_bf16,
                                                velocity, video_count))) {
                h3_dit_fail(error, error_size,
                     "cannot read the latent preview at step %d", step);
                ok = 0;
            }
            if (ok) {
                float sigma = dit->sigmas.video[step];
                for (size_t index = 0; index < video_count; index++) {
                    uint32_t bits = (uint32_t)velocity[index] << 16;
                    float v;
                    memcpy(&v, &bits, sizeof(v));
                    video_rows[index] += sigma * v;
                }
                ok = h3_dit_unpatchify_video(
                         video_rows, VIDEO_CHANNELS, dit->latent_t,
                         dit->latent_h, dit->latent_w, clean,
                         h3_dit_video_elements(dit)) &&
                     deliver_latent_preview(dit, step, clean,
                                            error, error_size);
            }
            free(velocity);
            free(clean);
            if (ok) {
                ok = h3_dit_gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                            "begin GPU Euler command chain");
                command_active = ok;
            }
            if (!ok) break;
        }

        float video_ratio = evaluate ? 0.0f : extrapolation_ratio(
            dit->sigmas.video[step], dit->sigmas.video[last_evaluated],
            previous_evaluated >= 0
                ? dit->sigmas.video[previous_evaluated] : 0.0f,
            previous_evaluated >= 0);
        float audio_ratio = evaluate ? 0.0f : extrapolation_ratio(
            dit->sigmas.audio[step], dit->sigmas.audio[last_evaluated],
            previous_evaluated >= 0
                ? dit->sigmas.audio[previous_evaluated] : 0.0f,
            previous_evaluated >= 0);
        const h3_gpu_tensor *previous_video = previous_evaluated >= 0
            ? dit->previous_video_velocity : dit->video_output_bf16;
        const h3_gpu_tensor *previous_audio = previous_evaluated >= 0
            ? dit->previous_audio_velocity : dit->audio_output_bf16;
        if (renoise) {
            uint32_t video_keys[2], audio_keys[2];
            h3_renoise_keys(dit->sampler_seed, step, 0, video_keys);
            h3_renoise_keys(dit->sampler_seed, step, 1, audio_keys);
            ok = h3_dit_gpu_op(dit, h3_gpu_renoise_bf16(
                    dit->gpu, dit->video_input, video_offset,
                    dit->video_output_bf16, (uint32_t)video_count,
                    dit->sigmas.video[step], dit->sigmas.video[step + 1],
                    video_keys), error, error_size, "GPU video re-noise step") &&
                 (forced_clean ? 1 : forced
                  /* Forced audio stays on its own trajectory: fresh noise
                   * would leave it, while an Euler step with the held
                   * velocity lands on the next trajectory point. */
                  ? h3_dit_gpu_op(dit, h3_gpu_euler_bf16(
                        dit->gpu, dit->audio_input, audio_offset,
                        dit->audio_output_bf16, dit->audio_output_bf16,
                        (uint32_t)audio_count,
                        dit->sigmas.audio[step] - dit->sigmas.audio[step + 1],
                        0.0f), error, error_size, "GPU forced audio step")
                  : h3_dit_gpu_op(dit, h3_gpu_renoise_bf16(
                        dit->gpu, dit->audio_input, audio_offset,
                        dit->audio_output_bf16, (uint32_t)audio_count,
                        dit->sigmas.audio[step], dit->sigmas.audio[step + 1],
                        audio_keys), error, error_size,
                        "GPU audio re-noise step"));
        } else {
            ok = h3_dit_gpu_op(dit, h3_gpu_euler_bf16(
                    dit->gpu, dit->video_input, video_offset,
                    dit->video_output_bf16, previous_video,
                    (uint32_t)video_count,
                    dit->sigmas.video[step] - dit->sigmas.video[step + 1],
                    video_ratio), error, error_size, "GPU video Euler step") &&
                 (forced_clean ? 1 : h3_dit_gpu_op(dit, h3_gpu_euler_bf16(
                    dit->gpu, dit->audio_input, audio_offset,
                    dit->audio_output_bf16, previous_audio,
                    (uint32_t)audio_count,
                    dit->sigmas.audio[step] - dit->sigmas.audio[step + 1],
                    audio_ratio), error, error_size, "GPU audio Euler step"));
        }
        if (ok && (evaluate || preview)) {
            int finish = preview || step + 1 == dit->sigmas.steps ||
                         (window && pending_evaluations >= window);
            ok = h3_dit_gpu_op(dit, finish ? h3_gpu_submit(dit->gpu)
                                    : h3_gpu_continue(dit->gpu),
                        error, error_size,
                        finish ? "submit GPU Euler window"
                               : "continue GPU Euler command chain");
            if (ok && finish) {
                command_active = 0;
                pending_evaluations = 0;
            }
        }
        if (ok && preview) {
            ok = h3_gpu_tensor_read_f32_range(
                     dit->video_input, video_offset, video_rows, video_count) &&
                 h3_dit_unpatchify_video(
                     video_rows, VIDEO_CHANNELS, dit->latent_t, dit->latent_h,
                     dit->latent_w, video_latent,
                     h3_dit_video_elements(dit));
            if (!ok) {
                h3_dit_fail(error, error_size,
                     "cannot read GPU Euler preview latent at step %d", step);
            } else if (preview(step + 1, dit->sigmas.steps, video_latent,
                               h3_dit_video_elements(dit), preview_opaque)) {
                h3_dit_fail(error, error_size,
                     "denoising preview stopped at step %d", step + 1);
                ok = 0;
            }
        }
        if (ok) h3_dit_report(progress, progress_opaque, "denoise enqueue", step + 1,
                       dit->sigmas.steps);
    }
    if (ok && command_active)
        ok = h3_dit_gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                    "submit GPU Euler denoise");
    if (ok) ok = h3_gpu_tensor_read_f32_range(
                     dit->video_input, video_offset, video_rows, video_count) &&
                 h3_gpu_tensor_read_f32_range(
                     dit->audio_input, audio_offset, audio_rows, audio_count);
    if (!ok && (!error || !*error))
        h3_dit_fail(error, error_size, "cannot read GPU Euler latents");
    if (ok) ok = h3_dit_unpatchify_video(
                     video_rows, VIDEO_CHANNELS, dit->latent_t, dit->latent_h,
                     dit->latent_w, video_latent, h3_dit_video_elements(dit)) &&
                 h3_dit_unpack_audio(audio_rows, AUDIO_CHANNELS, dit->audio_t,
                                     audio_latent,
                                     h3_dit_audio_elements(dit));
    if (!ok && (!error || !*error))
        h3_dit_fail(error, error_size, "cannot unpack GPU Euler latents");
    free(video_rows);
    free(audio_rows);
    if (ok) h3_dit_report(progress, progress_opaque, "denoise", dit->sigmas.steps,
                   dit->sigmas.steps);
    h3_gpu_profile_mark(dit->gpu, "GPU Euler denoise");
    return ok;
}

int h3_dit_denoise(h3_dit *dit, float *video_latent, float *audio_latent,
                   h3_dit_progress progress, void *progress_opaque,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !video_latent || !audio_latent ||
        dit->sigmas.steps != h3_dit_schedule_steps(dit->schedule)) {
        h3_dit_fail(error, error_size, "invalid DiT denoising arguments");
        return 0;
    }
    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);
    float *video_velocity = malloc(video_count * sizeof(*video_velocity));
    float *audio_velocity = malloc(audio_count * sizeof(*audio_velocity));
    float *video_denoised = malloc(video_count * sizeof(*video_denoised));
    float *audio_denoised = malloc(audio_count * sizeof(*audio_denoised));
    float *old_video = malloc(video_count * sizeof(*old_video));
    float *old_audio = malloc(audio_count * sizeof(*old_audio));
    float *video_next = malloc(video_count * sizeof(*video_next));
    float *audio_next = malloc(audio_count * sizeof(*audio_next));
    if (!video_velocity || !audio_velocity || !video_denoised ||
        !audio_denoised || !old_video || !old_audio || !video_next ||
        !audio_next) {
        h3_dit_fail(error, error_size, "out of memory allocating RES solver state");
        free(video_velocity); free(audio_velocity); free(video_denoised);
        free(audio_denoised); free(old_video); free(old_audio);
        free(video_next); free(audio_next);
        return 0;
    }
    int ok = 1;
    for (int step = 0; step < dit->sigmas.steps && ok; step++) {
        if (h3_dit_stop_if_cancelled(dit, error, error_size)) {
            ok = 0;
            break;
        }
        h3_dit_report(progress, progress_opaque, "denoise", step, dit->sigmas.steps);
        ok = h3_dit_forward(dit, step, video_latent, audio_latent,
                            video_velocity, audio_velocity,
                            error, error_size);
        float sigma = dit->sigmas.video[step];
        float timestep = 1.0f - sigma;
        float sigma_from_timestep = 1.0f - timestep;
        float audio_slope = (float)h3_time_shift_slope(
            sigma, H3_VIDEO_SIGMA_SHIFT, H3_AUDIO_SIGMA_SHIFT);
        if (ok) {
            for (size_t index = 0; index < video_count; index++)
                video_denoised[index] = video_latent[index] +
                    sigma_from_timestep * video_velocity[index];
            for (size_t index = 0; index < audio_count; index++)
                audio_denoised[index] = audio_latent[index] +
                    sigma_from_timestep * audio_velocity[index] * audio_slope;
            ok = h3_res_step(video_next, video_latent, video_denoised,
                             step ? old_video : NULL, video_count,
                             dit->sigmas.video, step, dit->sigmas.steps) &&
                 h3_res_step(audio_next, audio_latent, audio_denoised,
                             step ? old_audio : NULL, audio_count,
                             dit->sigmas.video, step, dit->sigmas.steps);
            if (!ok) h3_dit_fail(error, error_size, "RES solver rejected step %d", step);
        }
        if (ok) {
            memcpy(video_latent, video_next,
                   video_count * sizeof(*video_latent));
            memcpy(audio_latent, audio_next,
                   audio_count * sizeof(*audio_latent));
            memcpy(old_video, video_denoised,
                   video_count * sizeof(*old_video));
            memcpy(old_audio, audio_denoised,
                   audio_count * sizeof(*old_audio));
            h3_dit_report(progress, progress_opaque, "denoise", step + 1,
                   dit->sigmas.steps);
        }
    }
    free(video_velocity); free(audio_velocity); free(video_denoised);
    free(audio_denoised); free(old_video); free(old_audio);
    free(video_next); free(audio_next);
    h3_gpu_profile_mark(dit->gpu, "RES denoise");
    return ok;
}

int h3_dit_denoise_euler_preview(
                         h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3_dit_progress progress, void *progress_opaque,
                         h3_dit_preview preview, void *preview_opaque,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !video_latent || !audio_latent || reuse_interval < 1 ||
        reuse_interval > 32 ||
        dit->sigmas.steps != h3_dit_schedule_steps(dit->schedule)) {
        h3_dit_fail(error, error_size, "invalid Euler denoising arguments");
        return 0;
    }
    if (dit->sampler == H3_SAMPLER_RENOISE && reuse_interval != 1) {
        h3_dit_fail(error, error_size,
             "the re-noise sampler cannot be combined with denoiser reuse");
        return 0;
    }
    if (gpu_sampler_requested(dit))
        return denoise_euler_gpu(dit, video_latent, audio_latent,
                                 reuse_interval, progress, progress_opaque,
                                 preview, preview_opaque,
                                 error, error_size);
    uint8_t selected[H3_MAX_STEPS] = {0};
    int selected_count = h3_dit_reuse_schedule(
        dit->sigmas.steps, reuse_interval, selected, sizeof(selected));
    int custom_count = reuse_interval > 1 ?
        parse_reuse_steps(dit->sigmas.steps, selected) : 0;
    if (selected_count < 0 || custom_count < 0) {
        h3_dit_fail(error, error_size,
             "H3_REUSE_STEPS must be increasing and include 0 and %d",
             dit->sigmas.steps - 1);
        return 0;
    }
    if (custom_count > 0) selected_count = custom_count;
    if (reuse_interval > 1 && h3_option("H3_PROFILE"))
        fprintf(stderr, "h3: %s reuse schedule has %d evaluations\n",
                custom_count > 0 ? "custom" : "selected", selected_count);
    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);
    float *video_velocity = malloc(video_count * sizeof(*video_velocity));
    float *audio_velocity = malloc(audio_count * sizeof(*audio_velocity));
    float *last_video = reuse_interval > 1
        ? malloc(video_count * sizeof(*last_video)) : NULL;
    float *previous_video = reuse_interval > 1
        ? malloc(video_count * sizeof(*previous_video)) : NULL;
    float *last_audio = reuse_interval > 1
        ? malloc(audio_count * sizeof(*last_audio)) : NULL;
    float *previous_audio = reuse_interval > 1
        ? malloc(audio_count * sizeof(*previous_audio)) : NULL;
    if (!video_velocity || !audio_velocity ||
        (reuse_interval > 1 &&
         (!last_video || !previous_video || !last_audio || !previous_audio))) {
        h3_dit_fail(error, error_size, "out of memory allocating Euler velocities");
        free(video_velocity);
        free(audio_velocity);
        free(last_video);
        free(previous_video);
        free(last_audio);
        free(previous_audio);
        return 0;
    }
    int ok = 1;
    int forced = dit->forced_audio != NULL;
    int forced_clean = forced && dit->sigmas.audio[0] == 0.0f;
    float *forced_velocity = forced
        ? malloc(audio_count * sizeof(*forced_velocity)) : NULL;
    if (forced && (!forced_velocity ||
                   !h3_forced_trajectory_start(
                       audio_latent, forced_velocity, dit->forced_audio,
                       audio_count, dit->sigmas.audio[0]))) {
        h3_dit_fail(error, error_size, "cannot prepare forced audio");
        ok = 0;
    }
    int last_evaluated = -1;
    int previous_evaluated = -1;
    for (int step = 0; step < dit->sigmas.steps && ok; step++) {
        if (h3_dit_stop_if_cancelled(dit, error, error_size)) {
            ok = 0;
            break;
        }
        h3_dit_report(progress, progress_opaque, "denoise", step, dit->sigmas.steps);
        int evaluate = selected[step];
        if (evaluate) {
            ok = h3_dit_forward(dit, step, video_latent, audio_latent,
                                video_velocity, audio_velocity,
                                error, error_size);
            if (ok && reuse_interval > 1) {
                if (last_evaluated >= 0) {
                    memcpy(previous_video, last_video,
                           video_count * sizeof(*previous_video));
                    memcpy(previous_audio, last_audio,
                           audio_count * sizeof(*previous_audio));
                    previous_evaluated = last_evaluated;
                }
                memcpy(last_video, video_velocity,
                       video_count * sizeof(*last_video));
                memcpy(last_audio, audio_velocity,
                       audio_count * sizeof(*last_audio));
                last_evaluated = step;
            }
        } else {
            extrapolate_velocity(
                video_velocity, last_video, previous_video, video_count,
                dit->sigmas.video[step], dit->sigmas.video[last_evaluated],
                previous_evaluated >= 0
                    ? dit->sigmas.video[previous_evaluated] : 0.0f,
                previous_evaluated >= 0);
            extrapolate_velocity(
                audio_velocity, last_audio, previous_audio, audio_count,
                dit->sigmas.audio[step], dit->sigmas.audio[last_evaluated],
                previous_evaluated >= 0
                    ? dit->sigmas.audio[previous_evaluated] : 0.0f,
                previous_evaluated >= 0);
        }
        if (ok && forced)
            memcpy(audio_velocity, forced_velocity,
                   audio_count * sizeof(*audio_velocity));
        if (ok && evaluate && wants_latent_preview(dit, step)) {
            float *clean = malloc(video_count * sizeof(*clean));
            float sigma = dit->sigmas.video[step];
            if (!clean) {
                h3_dit_fail(error, error_size, "out of memory for the latent preview");
                ok = 0;
            } else {
                for (size_t index = 0; index < video_count; index++)
                    clean[index] = video_latent[index] +
                                   sigma * video_velocity[index];
                ok = deliver_latent_preview(dit, step, clean,
                                            error, error_size);
                free(clean);
            }
        }
        if (ok && dit->sampler == H3_SAMPLER_RENOISE) {
            uint32_t video_keys[2], audio_keys[2];
            h3_renoise_keys(dit->sampler_seed, step, 0, video_keys);
            h3_renoise_keys(dit->sampler_seed, step, 1, audio_keys);
            ok = h3_renoise_velocity_step(
                     video_latent, video_velocity, video_count,
                     dit->sigmas.video[step], dit->sigmas.video[step + 1],
                     video_keys) &&
                 (forced_clean ? 1 : forced
                  ? h3_euler_velocity_step(
                        audio_latent, audio_velocity, audio_count,
                        dit->sigmas.audio[step], dit->sigmas.audio[step + 1])
                  : h3_renoise_velocity_step(
                        audio_latent, audio_velocity, audio_count,
                        dit->sigmas.audio[step], dit->sigmas.audio[step + 1],
                        audio_keys));
            if (!ok) h3_dit_fail(error, error_size,
                          "re-noise solver rejected step %d", step);
        } else if (ok) {
            ok = h3_euler_velocity_step(
                     video_latent, video_velocity, video_count,
                     dit->sigmas.video[step], dit->sigmas.video[step + 1]) &&
                 (forced_clean || h3_euler_velocity_step(
                     audio_latent, audio_velocity, audio_count,
                     dit->sigmas.audio[step], dit->sigmas.audio[step + 1]));
            if (!ok) h3_dit_fail(error, error_size,
                          "Euler solver rejected step %d", step);
        }
        if (ok && preview &&
            preview(step + 1, dit->sigmas.steps, video_latent, video_count,
                    preview_opaque)) {
            h3_dit_fail(error, error_size, "denoising preview stopped at step %d",
                 step + 1);
            ok = 0;
        }
        if (ok) h3_dit_report(progress, progress_opaque, "denoise", step + 1,
                       dit->sigmas.steps);
    }
    free(video_velocity);
    free(audio_velocity);
    free(last_video);
    free(previous_video);
    free(last_audio);
    free(previous_audio);
    free(forced_velocity);
    h3_gpu_profile_mark(dit->gpu, "Euler denoise");
    return ok;
}

int h3_dit_denoise_euler(h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size) {
    return h3_dit_denoise_euler_preview(
        dit, video_latent, audio_latent, reuse_interval,
        progress, progress_opaque, NULL, NULL, error, error_size);
}

void h3_dit_set_attention_mode(h3_dit *dit, h3_dit_attention_mode mode) {
    if (dit) dit->attention_mode = mode;
}

void h3_dit_set_sampler(h3_dit *dit, int sampler, uint64_t seed) {
    if (!dit) return;
    dit->sampler = sampler;
    dit->sampler_seed = seed;
}

int h3_dit_set_forced_audio(h3_dit *dit, const float *clean_latent,
                            size_t elements, char *error, size_t error_size) {
    if (!dit) return 0;
    free(dit->forced_audio);
    dit->forced_audio = NULL;
    if (!clean_latent) return 1;
    if (elements != h3_dit_audio_elements(dit)) {
        h3_dit_fail(error, error_size, "forced audio has %zu values, expected %zu",
             elements, h3_dit_audio_elements(dit));
        return 0;
    }
    dit->forced_audio = malloc(elements * sizeof(*dit->forced_audio));
    if (!dit->forced_audio) {
        h3_dit_fail(error, error_size, "out of memory storing forced audio");
        return 0;
    }
    memcpy(dit->forced_audio, clean_latent,
           elements * sizeof(*dit->forced_audio));
    return 1;
}

void h3_dit_release_backend_scratch(h3_dit *dit) {
    if (dit && !h3_gpu_ccv_release_scratch(dit->gpu))
        fprintf(stderr, "h3: warning: ccv scratch kept, GPU work pending\n");
}

void h3_dit_free(h3_dit *dit) {
    if (!dit) return;
    h3_lora_set_free(dit->loras);
    int steps = h3_dit_schedule_steps(dit->schedule);
    if (dit->row_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->row_maps[step]);
    if (dit->reduced_row_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->reduced_row_maps[step]);
    if (dit->final_audio_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->final_audio_maps[step]);
    if (dit->final_video_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->final_video_maps[step]);
    free(dit->row_maps);
    free(dit->reduced_row_maps);
    free(dit->final_audio_maps);
    free(dit->final_video_maps);
    h3_dit_free_tensor(&dit->refined_text);
    h3_dit_free_tensor(&dit->rope_cos);
    h3_dit_free_tensor(&dit->rope_sin);
    h3_dit_free_tensor(&dit->reduced_rope_cos);
    h3_dit_free_tensor(&dit->reduced_rope_sin);
    h3_dit_free_tensor(&dit->video_patch_w); h3_dit_free_tensor(&dit->video_patch_b);
    h3_dit_free_tensor(&dit->audio_patch_w); h3_dit_free_tensor(&dit->audio_patch_b);
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        h3_dit_free_block(&dit->blocks[block]);
    h3_dit_free_stream_slots(dit);
    h3_dit_free_attention_slot(&dit->attention_slots[0]);
    h3_dit_free_attention_slot(&dit->attention_slots[1]);
    free(dit->attention_cache_path);
    h3_dit_free_tensor(&dit->final_norm);
    h3_dit_free_tensor(&dit->final_video_w); h3_dit_free_tensor(&dit->final_video_b);
    h3_dit_free_tensor(&dit->final_audio_w); h3_dit_free_tensor(&dit->final_audio_b);
#define FREE(field) h3_dit_free_tensor(&dit->field)
    if (dit->activation_aliases) {
        dit->attention_heads = NULL;
        dit->mod_mlp = NULL;
    }
    FREE(video_input); FREE(audio_input);
    FREE(video_projected_f32); FREE(audio_projected_f32);
    FREE(video_projected); FREE(audio_projected);
    FREE(video_projection_map); FREE(audio_projection_map); FREE(hidden);
    FREE(core_input); FREE(core_residual);
    FREE(mod_attention); FREE(qkv); FREE(query); FREE(key); FREE(value);
    FREE(attention_heads); FREE(attention_output);
    FREE(token_pool_pairs); FREE(token_baseline_indices);
    FREE(token_expand_parents); FREE(token_original); FREE(mod_mlp); FREE(fc1);
    FREE(activated); FREE(mlp_output); FREE(int8_activation);
    FREE(lowrank_tmp);
    FREE(int8_activation_scales); FREE(final_audio_input);
    FREE(final_video_input); FREE(final_audio_inverse);
    FREE(final_video_inverse); FREE(final_audio_norm); FREE(final_video_norm);
    FREE(final_audio_f32); FREE(final_video_f32); FREE(audio_output);
    FREE(video_output);
    FREE(audio_output_bf16); FREE(video_output_bf16);
    FREE(previous_audio_velocity); FREE(previous_video_velocity);
    FREE(forced_audio_velocity);
#undef FREE
    free(dit->forced_audio);
    h3_dit_schedule_free(dit->schedule);
    if (dit->ssd_streaming && h3_option("H3_PROFILE")) {
        double gib = (double)dit->stream_bytes / (1024.0 * 1024.0 * 1024.0);
        fprintf(stderr,
                "h3: BF16 SSD stream %.3f GiB read in %.3fs (%.3f GiB/s), "
                "unhidden wait %.3fs\n",
                gib, dit->stream_read_seconds,
                dit->stream_read_seconds > 0.0
                    ? gib / dit->stream_read_seconds : 0.0,
                dit->stream_wait_seconds);
    }
    if (dit->attention_stream && h3_option("H3_PROFILE")) {
        double gib = (double)dit->attn_stream_bytes /
            (1024.0 * 1024.0 * 1024.0);
        fprintf(stderr,
                "h3: attention int8 stream %.3f GiB read in %.3fs "
                "(%.3f GiB/s), unhidden wait %.3fs\n",
                gib, dit->attn_stream_read_seconds,
                dit->attn_stream_read_seconds > 0.0
                    ? gib / dit->attn_stream_read_seconds : 0.0,
                dit->attn_stream_wait_seconds);
    }
    h3_gpu_free(dit->gpu);
    h3_weight_store_free(dit->weights);
    h3_layout_free(&dit->layout);
    free(dit);
}

static int video_shape(int channels, int time, int height, int width,
                       size_t *latent_count, size_t *row_count) {
    if (channels < 1 || time < 1 || height < 2 || width < 2 ||
        height % 2 || width % 2) return 0;
    size_t c = (size_t)channels, t = (size_t)time;
    size_t h = (size_t)height, w = (size_t)width;
    if (c > SIZE_MAX / t || c * t > SIZE_MAX / h ||
        c * t * h > SIZE_MAX / w) return 0;
    *latent_count = c * t * h * w;
    *row_count = t * (h / 2) * (w / 2) * c * 4;
    return 1;
}

int h3_dit_patchify_video(const float *latent, int channels, int time,
                          int height, int width, float *rows,
                          size_t row_elements) {
    size_t latent_count, expected;
    if (!latent || !rows ||
        !video_shape(channels, time, height, width, &latent_count, &expected) ||
        row_elements != expected || latent_count != expected) return 0;
    size_t output = 0;
    for (int t = 0; t < time; t++)
        for (int h = 0; h < height; h += 2)
            for (int w = 0; w < width; w += 2)
                for (int c = 0; c < channels; c++)
                    for (int dh = 0; dh < 2; dh++)
                        for (int dw = 0; dw < 2; dw++) {
                            size_t input = (((size_t)c * (size_t)time +
                                (size_t)t) * (size_t)height + (size_t)(h + dh)) *
                                (size_t)width + (size_t)(w + dw);
                            rows[output++] = latent[input];
                        }
    return output == row_elements;
}

int h3_dit_unpatchify_video(const float *rows, int channels, int time,
                            int height, int width, float *latent,
                            size_t latent_elements) {
    size_t expected, row_count;
    if (!rows || !latent ||
        !video_shape(channels, time, height, width, &expected, &row_count) ||
        latent_elements != expected || row_count != expected) return 0;
    size_t input = 0;
    for (int t = 0; t < time; t++)
        for (int h = 0; h < height; h += 2)
            for (int w = 0; w < width; w += 2)
                for (int c = 0; c < channels; c++)
                    for (int dh = 0; dh < 2; dh++)
                        for (int dw = 0; dw < 2; dw++) {
                            size_t output = (((size_t)c * (size_t)time +
                                (size_t)t) * (size_t)height + (size_t)(h + dh)) *
                                (size_t)width + (size_t)(w + dw);
                            latent[output] = rows[input++];
                        }
    return input == row_count;
}

int h3_dit_pack_audio(const float *latent, int channels, int time,
                      float *rows, size_t row_elements) {
    if (!latent || !rows || channels < 1 || time < 1 ||
        (size_t)channels > SIZE_MAX / (2 * (size_t)time) ||
        row_elements != (size_t)channels * 2 * (size_t)time) return 0;
    size_t output = 0;
    for (int stream = 0; stream < 2; stream++)
        for (int t = 0; t < time; t++)
            for (int channel = 0; channel < channels; channel++) {
                size_t input = ((size_t)channel * 2 + (size_t)stream) *
                               (size_t)time + (size_t)t;
                rows[output++] = latent[input];
            }
    return output == row_elements;
}

int h3_dit_unpack_audio(const float *rows, int channels, int time,
                        float *latent, size_t latent_elements) {
    if (!rows || !latent || channels < 1 || time < 1 ||
        (size_t)channels > SIZE_MAX / (2 * (size_t)time) ||
        latent_elements != (size_t)channels * 2 * (size_t)time) return 0;
    size_t input = 0;
    for (int stream = 0; stream < 2; stream++)
        for (int t = 0; t < time; t++)
            for (int channel = 0; channel < channels; channel++) {
                size_t output = ((size_t)channel * 2 + (size_t)stream) *
                                (size_t)time + (size_t)t;
                latent[output] = rows[input++];
            }
    return input == latent_elements;
}

int h3_dit_debug_set_refined_text(h3_dit *dit, const uint16_t *rows,
                                  size_t count, char *error,
                                  size_t error_size) {
    if (!dit || !rows || !dit->refined_text ||
        count != (size_t)dit->text_rows * HIDDEN) {
        h3_dit_fail(error, error_size, "refined text override needs %zu values",
             dit ? (size_t)dit->text_rows * HIDDEN : (size_t)0);
        return 0;
    }
    if (!h3_gpu_tensor_write_bf16(dit->refined_text, rows, count)) {
        h3_dit_fail(error, error_size, "cannot write the refined text override");
        return 0;
    }
    return 1;
}
