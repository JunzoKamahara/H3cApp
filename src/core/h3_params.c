/* Checking h3_params before any work (h3_valid_params). Split out of h3.c. */
#include "h3_private.h"
int h3_valid_params(h3_ctx *ctx, const h3_params *params) {
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
