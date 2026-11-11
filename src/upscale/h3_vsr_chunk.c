/* Chunk protocol of the one-step VSR streamer: LightX2V
 * lightx2v/models/networks/swiftvr/streaming.py (build_video_chunks,
 * padded_frame_count, VideoChunk.output_range, StreamingTransformer.
 * prepare_input) transcribed for the host. No GPU, no allocation. */
#include "h3_vsr.h"

#include <stdlib.h>

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

/* ---------------------------------------------------------------- */
/* Spatial tiles. */

static int round_up_32(int value) { return (value + 31) / 32 * 32; }

int h3_vsr_tile_plan_build(int full_w, int full_h, int columns, int rows,
                           int overlap, h3_vsr_tile_plan *plan) {
    if (!plan || full_w < 32 || full_h < 32 || full_w % 32 || full_h % 32 ||
        columns < 1 || rows < 1 || columns > H3_VSR_MAX_TILES_PER_AXIS ||
        rows > H3_VSR_MAX_TILES_PER_AXIS || overlap < 32 || overlap % 32)
        return 0;
    int counts[2] = {columns, rows}, sizes[2] = {full_w, full_h}, tile[2];
    int *origins[2] = {plan->x, plan->y};
    for (int axis = 0; axis < 2; axis++) {
        int n = counts[axis], size = sizes[axis];
        if (n == 1) {
            tile[axis] = size;
            origins[axis][0] = 0;
            continue;
        }
        /* n tiles of t cover size with n-1 overlaps of at least `overlap`:
         * n*t >= size + (n-1)*overlap. */
        int t = round_up_32((size + (n - 1) * overlap + n - 1) / n);
        if (t >= size) return 0;
        tile[axis] = t;
        for (int i = 0; i < n; i++) {
            int at = i == n - 1 ? size - t : (int)((long long)i * (size - t) / (n - 1)) / 32 * 32;
            if (i && at <= origins[axis][i - 1]) return 0;
            origins[axis][i] = at;
        }
    }
    plan->full_w = full_w;
    plan->full_h = full_h;
    plan->columns = columns;
    plan->rows = rows;
    plan->tile_w = tile[0];
    plan->tile_h = tile[1];
    plan->overlap = overlap;
    return 1;
}

/* The normalised ramp of tile `index` along one axis: tile size `t`,
 * origins `at[n]`. ramp[i] for the tile's local pixel i. */
static void axis_ramp(const int *at, int n, int t, int index, float *ramp) {
    int left = index > 0 ? at[index - 1] + t - at[index] : 0;        /* shared with the previous tile */
    int right = index < n - 1 ? at[index] + t - at[index + 1] : 0;   /* shared with the next */
    for (int i = 0; i < t; i++) {
        double w = 1.0;
        if (i < left) w = (i + 0.5) / left;
        if (i >= t - right) { double r = (t - i - 0.5) / right; if (r < w) w = r; }
        /* Normalise by the sum over every tile covering this pixel. */
        int x = at[index] + i;
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            int local = x - at[j];
            if (local < 0 || local >= t) continue;
            int jl = j > 0 ? at[j - 1] + t - at[j] : 0, jr = j < n - 1 ? at[j] + t - at[j + 1] : 0;
            double v = 1.0;
            if (local < jl) v = (local + 0.5) / jl;
            if (local >= t - jr) { double r = (t - local - 0.5) / jr; if (r < v) v = r; }
            sum += v;
        }
        ramp[i] = (float)(w / sum);
    }
}

void h3_vsr_tile_weights(const h3_vsr_tile_plan *plan, int column, int row,
                         float *weights) {
    if (!plan || !weights || column < 0 || column >= plan->columns || row < 0 || row >= plan->rows) return;
    float *rx = malloc((size_t)plan->tile_w * sizeof(float));
    float *ry = malloc((size_t)plan->tile_h * sizeof(float));
    if (!rx || !ry) { free(rx); free(ry); return; }
    axis_ramp(plan->x, plan->columns, plan->tile_w, column, rx);
    axis_ramp(plan->y, plan->rows, plan->tile_h, row, ry);
    for (int y = 0; y < plan->tile_h; y++)
        for (int x = 0; x < plan->tile_w; x++)
            weights[(size_t)y * plan->tile_w + x] = ry[y] * rx[x];
    free(rx);
    free(ry);
}
