/* Host-side tests of the one-step VSR geometry: chunk protocol, mask-free
 * shifted-window layout, Wan RoPE tables, patch index math. No GPU, no
 * weights; always part of `make test`. */
#include "h3_vsr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run;

#define CHECK(condition) do { \
    tests_run++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static int starts_equal(int size, int window, int shifted,
                        const int *expected, int count) {
    int got[64];
    int n = h3_vsr_axis_window_starts(size, window, shifted, got, 64);
    if (n != count) return 0;
    for (int i = 0; i < count; i++)
        if (got[i] != expected[i]) return 0;
    return 1;
}

static void test_axis_starts(void) {
    /* 1920x1088 -> token grid 60x34 (UPSCALE_PLAN §6.4). */
    CHECK(starts_equal(34, 16, 0, (const int[]){0, 16, 18}, 3));
    CHECK(starts_equal(34, 16, 1, (const int[]){0, 8, 18}, 3));
    CHECK(starts_equal(60, 16, 0, (const int[]){0, 16, 32, 44}, 4));
    CHECK(starts_equal(60, 16, 1, (const int[]){0, 8, 24, 40, 44}, 5));
    /* The paper's 1440p grid 45x80: 3x5 windows even, 4x6 odd. */
    CHECK(h3_vsr_axis_window_starts(45, 16, 0, (int[64]){0}, 64) == 3);
    CHECK(h3_vsr_axis_window_starts(80, 16, 0, (int[64]){0}, 64) == 5);
    CHECK(h3_vsr_axis_window_starts(45, 16, 1, (int[64]){0}, 64) == 4);
    CHECK(h3_vsr_axis_window_starts(80, 16, 1, (int[64]){0}, 64) == 6);
    /* An axis that fits in one window has a single start. */
    CHECK(starts_equal(16, 16, 0, (const int[]){0}, 1));
    CHECK(starts_equal(16, 16, 1, (const int[]){0}, 1));
    CHECK(starts_equal(8, 16, 1, (const int[]){0}, 1));
    /* Exact multiples: no clamped duplicates. */
    CHECK(starts_equal(32, 16, 0, (const int[]){0, 16}, 2));
    /* Shifted 32: 8 is covered by 0 and 16 together, so it is dropped. */
    CHECK(starts_equal(32, 16, 1, (const int[]){0, 16}, 2));
    /* Bad arguments. */
    CHECK(h3_vsr_axis_window_starts(0, 16, 0, (int[4]){0}, 4) == 0);
    CHECK(h3_vsr_axis_window_starts(60, 16, 1, (int[2]){0}, 2) == 0);
}

/* Every token is owned exactly once, by a window that contains it, and the
 * owned lists partition the tokens. */
static void check_layout(int frames, int height, int width, int shifted,
                         int expected_windows) {
    h3_vsr_window_layout layout;
    CHECK(h3_vsr_window_layout_build(frames, height, width, 16, 16, shifted,
                                     &layout));
    CHECK(layout.window_count == expected_windows);
    CHECK(layout.tokens == frames * height * width);
    CHECK(layout.window_length == frames * layout.window_h * layout.window_w);
    int *seen = calloc((size_t)layout.tokens, sizeof(int));
    CHECK(seen != NULL);
    for (int token = 0; token < layout.tokens; token++) {
        int32_t owner = layout.owner[token];
        CHECK(owner >= 0 && owner < layout.window_count * layout.window_length);
        CHECK(layout.indices[owner] == token);
    }
    CHECK(layout.owned_offset[0] == 0);
    CHECK(layout.owned_offset[layout.window_count] == layout.tokens);
    for (int window = 0; window < layout.window_count; window++) {
        int previous = -1;
        for (int at = layout.owned_offset[window];
             at < layout.owned_offset[window + 1]; at++) {
            int local = layout.owned_local[at];
            CHECK(local > previous);
            previous = local;
            int token = layout.owned_token[at];
            CHECK(layout.indices[window * layout.window_length + local] == token);
            CHECK(layout.owner[token] == window * layout.window_length + local);
            seen[token]++;
        }
    }
    for (int token = 0; token < layout.tokens; token++) CHECK(seen[token] == 1);
    /* Rows inside a window: frame-major, then row, then column. */
    const int32_t *rows = layout.indices;
    if (layout.window_w > 1) CHECK(rows[1] == rows[0] + 1);
    if (layout.window_h > 1) CHECK(rows[layout.window_w] == rows[0] + width);
    if (frames > 1)
        CHECK(rows[layout.window_h * layout.window_w] == rows[0] + height * width);
    free(seen);
    h3_vsr_window_layout_free(&layout);
}

static void test_window_layout(void) {
    check_layout(6, 34, 60, 0, 12);
    check_layout(6, 34, 60, 1, 15);
    check_layout(7, 34, 60, 0, 12);
    check_layout(6, 45, 80, 0, 15);
    check_layout(6, 45, 80, 1, 24);
    check_layout(2, 16, 16, 0, 1);
    check_layout(2, 16, 16, 1, 1);
    check_layout(3, 8, 8, 0, 1);
    check_layout(1, 17, 33, 1, 2 * 3);  /* starts {0,1} x {0,8,17} */
    /* Owner widths at 34x60 (plan §6.4): even layers h 16/16/2, w 16/16/12/16. */
    h3_vsr_window_layout layout;
    CHECK(h3_vsr_window_layout_build(6, 34, 60, 16, 16, 0, &layout));
    int owned_first = layout.owned_offset[1] - layout.owned_offset[0];
    CHECK(owned_first == 6 * 16 * 16);
    /* Window (h start 18, w start 44) owns rows 32-33 and columns 48-59:
     * 2 x 12 per frame (front-most windows win the overlaps). */
    int last = layout.window_count - 1;
    CHECK(layout.owned_offset[last + 1] - layout.owned_offset[last] == 6 * 2 * 12);
    h3_vsr_window_layout_free(&layout);
    CHECK(h3_vsr_window_layout_build(6, 34, 60, 16, 16, 1, &layout));
    /* Shifted: the last window wins, so the first window keeps 8 x 8 x T. */
    CHECK(layout.owned_offset[1] - layout.owned_offset[0] == 6 * 8 * 8);
    h3_vsr_window_layout_free(&layout);
}

static void test_chunk_plan(void) {
    CHECK(h3_vsr_padded_frame_count(1) == 1);
    CHECK(h3_vsr_padded_frame_count(2) == 5);
    CHECK(h3_vsr_padded_frame_count(5) == 5);
    CHECK(h3_vsr_padded_frame_count(24) == 25);
    CHECK(h3_vsr_padded_frame_count(124) == 125);
    CHECK(h3_vsr_padded_frame_count(0) == 0);
    /* 124 frames pad to 125: 28 + 4 x 24 + 1 (a MIDDLE chunk is cut whenever
     * more than clip_frames remain, so the tail can be a single frame). */
    CHECK(h3_vsr_chunk_count(124, 24) == 6);
    CHECK(h3_vsr_chunk_count(1440, 24) == 60);
    CHECK(h3_vsr_chunk_count(25, 24) == 1);   /* padded 25 <= 28 */
    CHECK(h3_vsr_chunk_count(28, 24) == 2);   /* padded 29: 28 + 1 */
    CHECK(h3_vsr_chunk_count(29, 24) == 2);
    CHECK(h3_vsr_chunk_count(10, 23) == 0);
    h3_vsr_chunk chunks[128];
    CHECK(h3_vsr_chunk_plan(124, 24, chunks, 128) == 6);
    CHECK(chunks[0].type == H3_VSR_CHUNK_FIRST && chunks[0].frames == 28 &&
          chunks[0].latents == 7 && chunks[0].output_start == 0 &&
          chunks[0].output_stop == 25 && chunks[0].rope_offset == 0);
    CHECK(chunks[1].type == H3_VSR_CHUNK_MIDDLE && chunks[1].start == 28 &&
          chunks[1].frames == 24 && chunks[1].latents == 6 &&
          chunks[1].output_start == 25 && chunks[1].output_stop == 49 &&
          chunks[1].rope_offset == 7);
    CHECK(chunks[4].type == H3_VSR_CHUNK_MIDDLE && chunks[4].start == 100 &&
          chunks[4].output_start == 97 && chunks[4].output_stop == 121);
    CHECK(chunks[5].type == H3_VSR_CHUNK_LAST && chunks[5].start == 124 &&
          chunks[5].frames == 1 && chunks[5].latents == 1 &&
          chunks[5].output_start == 121 && chunks[5].output_stop == 124 &&
          chunks[5].dit_prefix_latents == 6 && chunks[5].rope_offset == 31 - 6);
    /* 1440 frames: the last chunk has 20 -> 21 frames, 6 latents, one
     * borrowed latent, and its RoPE offset moves back by one. */
    CHECK(h3_vsr_chunk_plan(1440, 24, chunks, 128) == 60);
    CHECK(chunks[59].frames == 21 && chunks[59].latents == 6 &&
          chunks[59].dit_prefix_latents == 1 &&
          chunks[59].rope_offset == chunks[59].start / 4 - 1);
    /* A single short video is one LAST chunk padded to 4k+1. */
    CHECK(h3_vsr_chunk_plan(1, 24, chunks, 128) == 1);
    CHECK(chunks[0].type == H3_VSR_CHUNK_LAST && chunks[0].frames == 1 &&
          chunks[0].latents == 1 && chunks[0].dit_prefix_latents == 6 &&
          chunks[0].output_start == 0 && chunks[0].output_stop == 1);
    /* Output ranges partition [0, raw) for every length. */
    for (int raw = 1; raw <= 200; raw++) {
        int count = h3_vsr_chunk_plan(raw, 24, chunks, 128);
        CHECK(count == h3_vsr_chunk_count(raw, 24));
        int expected = 0;
        for (int i = 0; i < count; i++) {
            CHECK(chunks[i].index == i);
            CHECK(chunks[i].output_start == expected);
            CHECK(chunks[i].output_stop >= chunks[i].output_start);
            expected = chunks[i].output_stop;
            CHECK(chunks[i].frames >= 1);
            CHECK(chunks[i].latents == (chunks[i].frames - 1) / 4 + 1);
            if (i + 1 < count)
                CHECK(chunks[i + 1].start == chunks[i].start + chunks[i].frames);
        }
        CHECK(expected == raw);
        CHECK(chunks[count - 1].type == H3_VSR_CHUNK_LAST);
        CHECK(chunks[count - 1].start + chunks[count - 1].frames ==
              h3_vsr_padded_frame_count(raw));
    }
    CHECK(h3_vsr_chunk_plan(124, 24, chunks, 4) == 0);
}

static void test_rope_table(void) {
    const int frames = 2, height = 3, width = 4, head_dim = 128;
    int tokens = frames * height * width, pairs = head_dim / 2;
    float *c = malloc(sizeof(float) * (size_t)tokens * pairs);
    float *s = malloc(sizeof(float) * (size_t)tokens * pairs);
    CHECK(c && s);
    h3_vsr_rope_table(frames, height, width, 5, head_dim, c, s);
    for (int i = 0; i < tokens * pairs; i++)
        CHECK(fabs((double)c[i] * c[i] + (double)s[i] * s[i] - 1.0) < 1e-5);
    /* Token (t=1, h=2, w=3) with t_offset 5: t pairs use position 6 with
     * frequencies theta^(-2i/44), h pairs position 2 with theta^(-2i/42),
     * w pairs position 3. */
    int token = h3_vsr_token_index(1, 2, 3, height, width);
    CHECK(token == (1 * 3 + 2) * 4 + 3);
    const float *ct = c + (size_t)token * pairs, *st = s + (size_t)token * pairs;
    for (int i = 0; i < 22; i++) {
        double angle = 6.0 * pow(10000.0, -2.0 * i / 44.0);
        CHECK(fabs(ct[i] - cos(angle)) < 1e-5 && fabs(st[i] - sin(angle)) < 1e-5);
    }
    for (int i = 0; i < 21; i++) {
        double ah = 2.0 * pow(10000.0, -2.0 * i / 42.0);
        double aw = 3.0 * pow(10000.0, -2.0 * i / 42.0);
        CHECK(fabs(ct[22 + i] - cos(ah)) < 1e-5 && fabs(st[22 + i] - sin(ah)) < 1e-5);
        CHECK(fabs(ct[43 + i] - cos(aw)) < 1e-5 && fabs(st[43 + i] - sin(aw)) < 1e-5);
    }
    /* Pair 0 of every axis rotates by the raw position (frequency 1). */
    CHECK(fabs(ct[0] - cos(6.0)) < 1e-5);
    CHECK(fabs(ct[22] - cos(2.0)) < 1e-5);
    CHECK(fabs(ct[43] - cos(3.0)) < 1e-5);
    free(c);
    free(s);
}

static void test_patch_indices(void) {
    /* Input order (c, ph, pw); output order (ph, pw, c); both cover 0..191. */
    int seen_in[192] = {0}, seen_out[192] = {0};
    for (int ch = 0; ch < 48; ch++)
        for (int ph = 0; ph < 2; ph++)
            for (int pw = 0; pw < 2; pw++) {
                seen_in[h3_vsr_patch_in_index(ch, ph, pw)]++;
                seen_out[h3_vsr_patch_out_index(ch, ph, pw)]++;
            }
    for (int i = 0; i < 192; i++) CHECK(seen_in[i] == 1 && seen_out[i] == 1);
    CHECK(h3_vsr_patch_in_index(1, 0, 0) == 4);
    CHECK(h3_vsr_patch_out_index(1, 0, 0) == 1);
    CHECK(h3_vsr_patch_out_index(0, 1, 0) == 96);
    CHECK(H3_VSR_TOKEN_PIXELS == 32);
}

int main(void) {
    test_axis_starts();
    test_window_layout();
    test_chunk_plan();
    test_rope_table();
    test_patch_indices();
    printf("ok: %d checks (vsr host geometry)\n", tests_run);
    return 0;
}
