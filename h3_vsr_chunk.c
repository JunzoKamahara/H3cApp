/* Chunk protocol of the one-step VSR streamer: LightX2V
 * lightx2v/models/networks/swiftvr/streaming.py (build_video_chunks,
 * padded_frame_count, VideoChunk.output_range, StreamingTransformer.
 * prepare_input) transcribed for the host. No GPU, no allocation. */
#include "h3_vsr.h"

int h3_vsr_padded_frame_count(int frames) {
    if (frames < 1) return 0;
    return ((frames - 1 + 3) / 4) * 4 + 1;
}

static int valid_clip(int clip_frames) {
    return clip_frames > 0 && clip_frames % 4 == 0;
}

int h3_vsr_chunk_count(int raw_frames, int clip_frames) {
    if (raw_frames < 1 || !valid_clip(clip_frames)) return 0;
    int padded = h3_vsr_padded_frame_count(raw_frames);
    if (padded <= clip_frames + 4) return 1;
    int count = 1;
    int start = clip_frames + 4;
    while (start < padded) {
        int remaining = padded - start;
        count++;
        start += remaining <= clip_frames ? remaining : clip_frames;
    }
    return count;
}

static void fill_chunk(h3_vsr_chunk *chunk, h3_vsr_chunk_type type, int index,
                       int start, int frames, int raw_frames, int clip_frames,
                       int previous_exists) {
    chunk->type = type;
    chunk->index = index;
    chunk->start = start;
    chunk->frames = frames;
    chunk->latents = (frames - 1) / H3_VSR_TEMPORAL + 1;
    int trim = H3_VSR_FRAMES_TO_TRIM;
    int output_start = start - trim;
    if (output_start < 0) output_start = 0;
    int output_stop = start + frames - (type == H3_VSR_CHUNK_LAST ? 0 : trim);
    if (output_stop > raw_frames) output_stop = raw_frames;
    chunk->output_start = output_start;
    chunk->output_stop = output_stop;
    int clip_latents = clip_frames / H3_VSR_TEMPORAL;
    int offset = start / H3_VSR_TEMPORAL;
    if (type == H3_VSR_CHUNK_LAST) {
        int padding = clip_latents + 1 - chunk->latents;
        if (padding < 0) padding = 0;
        chunk->dit_prefix_latents = padding;
        /* Zeros stand in when no previous chunk exists; the offset rule is
         * the same either way (prepare_input). */
        (void)previous_exists;
        chunk->rope_offset = offset - padding < 0 ? 0 : offset - padding;
    } else {
        chunk->dit_prefix_latents = 0;
        chunk->rope_offset = offset;
    }
}

int h3_vsr_chunk_plan(int raw_frames, int clip_frames,
                      h3_vsr_chunk *chunks, int capacity) {
    int count = h3_vsr_chunk_count(raw_frames, clip_frames);
    if (count == 0 || chunks == NULL || capacity < count) return 0;
    int padded = h3_vsr_padded_frame_count(raw_frames);
    if (count == 1) {
        fill_chunk(&chunks[0], H3_VSR_CHUNK_LAST, 0, 0, padded, raw_frames,
                   clip_frames, 0);
        return 1;
    }
    fill_chunk(&chunks[0], H3_VSR_CHUNK_FIRST, 0, 0, clip_frames + 4,
               raw_frames, clip_frames, 0);
    int start = clip_frames + 4;
    int index = 1;
    while (start < padded) {
        int remaining = padded - start;
        h3_vsr_chunk_type type = remaining <= clip_frames
            ? H3_VSR_CHUNK_LAST : H3_VSR_CHUNK_MIDDLE;
        int frames = type == H3_VSR_CHUNK_LAST ? remaining : clip_frames;
        fill_chunk(&chunks[index], type, index, start, frames, raw_frames,
                   clip_frames, 1);
        start += frames;
        index++;
    }
    return index;
}
