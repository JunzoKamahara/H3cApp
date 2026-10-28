/* Mask-free shifted-window attention geometry, Wan RoPE tables: the host
 * side of SwiftVR's self-attention (LightX2V swiftvr/attention.py
 * axis_window_starts / WindowLayout, official transformer.py
 * WanShiftWindow2DInferProcessor, diffusers WanRotaryPosEmbed). */
#include "h3_vsr.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int h3_vsr_axis_window_starts(int size, int window, int shifted,
                              int *starts, int capacity) {
    if (size <= 0 || window <= 0 || capacity < 1 || starts == NULL) return 0;
    if (size <= window) {
        starts[0] = 0;
        return 1;
    }
    int shift = shifted ? window / 2 : 0;
    int max_start = size - window;
    int candidates = (size + window - 1) / window + 2;
    /* Candidates are non-decreasing, so dedup while generating. */
    int count = 0;
    int scratch[1024];
    for (int k = 0; k < candidates && count < 1024; k++) {
        int start = k * window - shift;
        if (start < 0) start = 0;
        if (start > max_start) start = max_start;
        if (count == 0 || scratch[count - 1] != start) scratch[count++] = start;
    }
    /* Drop interior starts whose span is already covered by both
     * neighbours: keep[i] = starts[i+1] > starts[i-1] + window. */
    int kept = 0;
    for (int i = 0; i < count; i++) {
        int keep = 1;
        if (i > 0 && i < count - 1)
            keep = scratch[i + 1] > scratch[i - 1] + window;
        if (keep) {
            if (kept >= capacity) return 0;
            starts[kept++] = scratch[i];
        }
    }
    return kept;
}

void h3_vsr_window_layout_free(h3_vsr_window_layout *layout) {
    if (layout == NULL) return;
    free(layout->indices);
    free(layout->owner);
    free(layout->owned_offset);
    free(layout->owned_local);
    free(layout->owned_token);
    memset(layout, 0, sizeof(*layout));
}

int h3_vsr_window_layout_build(int frames, int height, int width,
                               int window_h, int window_w, int shifted,
                               h3_vsr_window_layout *layout) {
    if (layout == NULL || frames < 1 || height < 1 || width < 1 ||
        window_h < 1 || window_w < 1)
        return 0;
    memset(layout, 0, sizeof(*layout));
    if (window_h > height) window_h = height;
    if (window_w > width) window_w = width;
    int h_starts[1024], w_starts[1024];
    int nh = h3_vsr_axis_window_starts(height, window_h, shifted, h_starts, 1024);
    int nw = h3_vsr_axis_window_starts(width, window_w, shifted, w_starts, 1024);
    if (nh == 0 || nw == 0) return 0;

    int tokens = frames * height * width;
    int window_count = nh * nw;
    int window_length = frames * window_h * window_w;
    layout->frames = frames;
    layout->height = height;
    layout->width = width;
    layout->window_h = window_h;
    layout->window_w = window_w;
    layout->shifted = shifted;
    layout->tokens = tokens;
    layout->window_count = window_count;
    layout->window_length = window_length;
    layout->indices = malloc(sizeof(int32_t) * (size_t)window_count * (size_t)window_length);
    layout->owner = malloc(sizeof(int32_t) * (size_t)tokens);
    layout->owned_offset = malloc(sizeof(int32_t) * (size_t)(window_count + 1));
    layout->owned_local = malloc(sizeof(int32_t) * (size_t)tokens);
    layout->owned_token = malloc(sizeof(int32_t) * (size_t)tokens);
    if (!layout->indices || !layout->owner || !layout->owned_offset ||
        !layout->owned_local || !layout->owned_token) {
        h3_vsr_window_layout_free(layout);
        return 0;
    }

    /* Gather rows: windows h-major then w; rows inside a window frame-major,
     * then window row, then window column. */
    int plane = height * width;
    for (int wi = 0; wi < nh; wi++) {
        for (int wj = 0; wj < nw; wj++) {
            int window = wi * nw + wj;
            int32_t *rows = layout->indices + (size_t)window * window_length;
            int local = 0;
            for (int t = 0; t < frames; t++)
                for (int hh = 0; hh < window_h; hh++)
                    for (int ww = 0; ww < window_w; ww++)
                        rows[local++] = t * plane
                            + (h_starts[wi] + hh) * width + (w_starts[wj] + ww);
        }
    }

    /* Owners: iterate windows descending when unshifted (the front-most
     * window wins), ascending when shifted (the last wins); later writes
     * overwrite. */
    for (int i = 0; i < tokens; i++) layout->owner[i] = -1;
    for (int step = 0; step < window_count; step++) {
        int window = shifted ? step : window_count - 1 - step;
        const int32_t *rows = layout->indices + (size_t)window * window_length;
        for (int local = 0; local < window_length; local++)
            layout->owner[rows[local]] = window * window_length + local;
    }

    /* Owned locals per window, ascending. */
    int32_t *counts = calloc((size_t)window_count, sizeof(int32_t));
    if (!counts) {
        h3_vsr_window_layout_free(layout);
        return 0;
    }
    for (int i = 0; i < tokens; i++) counts[layout->owner[i] / window_length]++;
    layout->owned_offset[0] = 0;
    for (int w = 0; w < window_count; w++)
        layout->owned_offset[w + 1] = layout->owned_offset[w] + counts[w];
    memset(counts, 0, sizeof(int32_t) * (size_t)window_count);
    for (int window = 0; window < window_count; window++) {
        const int32_t *rows = layout->indices + (size_t)window * window_length;
        for (int local = 0; local < window_length; local++) {
            int token = rows[local];
            if (layout->owner[token] != window * window_length + local) continue;
            int at = layout->owned_offset[window] + counts[window]++;
            layout->owned_local[at] = local;
            layout->owned_token[at] = token;
        }
    }
    free(counts);
    return 1;
}

void h3_vsr_rope_table(int frames, int height, int width, int t_offset,
                       int head_dim, float *cos_out, float *sin_out) {
    /* diffusers WanRotaryPosEmbed: h_dim = w_dim = 2 * (head_dim / 6),
     * t_dim = head_dim - h_dim - w_dim; get_1d_rotary_pos_embed with
     * repeat_interleave_real: pair p of an axis of `dim` channels rotates by
     * pos * theta^(-2p / dim). */
    const double theta = 10000.0;
    int hw_dim = 2 * (head_dim / 6);
    int t_dim = head_dim - 2 * hw_dim;
    int t_pairs = t_dim / 2, hw_pairs = hw_dim / 2;
    int pairs = head_dim / 2;
    size_t token = 0;
    for (int t = 0; t < frames; t++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++, token++) {
                float *c = cos_out + token * (size_t)pairs;
                float *s = sin_out + token * (size_t)pairs;
                int p = 0;
                for (int i = 0; i < t_pairs; i++, p++) {
                    double angle = (double)(t + t_offset)
                        * pow(theta, -2.0 * i / t_dim);
                    c[p] = (float)cos(angle);
                    s[p] = (float)sin(angle);
                }
                for (int i = 0; i < hw_pairs; i++, p++) {
                    double angle = (double)h * pow(theta, -2.0 * i / hw_dim);
                    c[p] = (float)cos(angle);
                    s[p] = (float)sin(angle);
                }
                for (int i = 0; i < hw_pairs; i++, p++) {
                    double angle = (double)w * pow(theta, -2.0 * i / hw_dim);
                    c[p] = (float)cos(angle);
                    s[p] = (float)sin(angle);
                }
            }
        }
    }
}
