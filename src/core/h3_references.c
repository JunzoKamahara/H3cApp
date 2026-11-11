/* Encoding what a generation is given: reference images, video and audio through the vision tower and the VAEs, RefMods, and a continuation latent from a video. Split out of h3.c. */
#include "h3_private.h"
/* Qwen consumes reference video as time-major two-frame blocks, while the
 * visual VAE and media boundary retain channel-major [3,T,H,W]. */
float *h3_extract_vision_pair(const float *pixels, int frames,
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

int h3_stream_encode_chunk(void *opaque, const float *rgb,
                                  int frame_count, int height, int width,
                                  char *error, size_t error_size) {
    h3_stream_encode_ctx *stream = opaque;
    size_t count = (size_t)frame_count * (size_t)height * (size_t)width * 3;
    uint8_t *chunk8 = h3_rgb_f32_to_u8(rgb, count);
    if (!chunk8) {
        h3_core_fail(error, error_size,
             "out of memory converting generated RGB frames");
        return 0;
    }
    if (stream->need_resize) {
        uint8_t *resized = NULL;
        if (!h3_resize_rgb24_high_quality(chunk8, frame_count, width, height,
                stream->output_width, stream->output_height, &resized)) {
            free(chunk8);
            h3_core_fail(error, error_size, "cannot resize generated RGB frames");
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
                h3_core_fail(error, error_size,
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
        h3_core_fail(error, error_size, "generation cancelled");
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

/* The conditioning encoders, each in an autorelease pool of its own (see
 * h3_autorelease_push): without one, every call's autoreleased Metal
 * objects stay until the whole generation ends. */
int h3_pooled_vision_encode(const char *weight_directory,
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

int h3_pooled_video_encode(const char *vae_path, const float *pixels,
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

int h3_pooled_audio_encode(const char *audio_vae_path,
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
int h3_refmod_reference(const h3_reference *reference, size_t index,
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
int h3_reference_limits(const h3_params *params,
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
int h3_reference_canvas(const h3_ctx *ctx, const h3_params *params,
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
