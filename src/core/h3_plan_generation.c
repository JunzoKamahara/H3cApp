/* h3_plan_generation: what a generation will need, worked out the way h3_generate will (token counts, reference geometry, memory per stage) without loading weights. The arithmetic is in h3_plan.c. Split out of h3.c. */
#include "h3_private.h"
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
