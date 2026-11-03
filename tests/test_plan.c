/* The generation planner's arithmetic (h3_plan.c): verdicts on both sides
 * of each limit, which stage it blames, and the DiT cost ratio. Needs no
 * weights or media; h3_plan_generation itself (reading references and
 * counting tokens) is checked against real generations, which warn when
 * the plan's token counts differ from theirs.
 *
 *   ./h3_plan_tests */
#include "h3_plan.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GIB (1024.0 * 1024.0 * 1024.0)
static const uint64_t ram24 = 24ull << 30;
static int checks;

static void check(int condition, const char *what) {
    checks++;
    if (condition) return;
    fprintf(stderr, "FAIL tests/test_plan.c: %s\n", what);
    exit(1);
}

static h3_plan run(uint64_t ram, size_t text, size_t vision, size_t dit,
                   size_t generated, size_t reference,
                   const h3_plan_visual *visuals, size_t visual_count) {
    h3_plan_input input = {ram, 1, 1, 1, 0, text, vision, dit, generated,
                           reference, visuals, visual_count};
    h3_plan plan;
    memset(&plan, 0x5a, sizeof(plan));
    h3_plan_compute(&input, &plan);
    return plan;
}

int main(void) {
    check(fabs(h3_plan_dit_cost_ratio(5000, 5000) - 1.0) < 1e-12,
          "the same sequence costs the same");
    check(fabs(h3_plan_dit_cost_ratio(53760, 26880) - 3.0) < 1e-9,
          "twice the crossover sequence costs three times");
    check(h3_plan_dit_cost_ratio(85400, 42700) > 2.5,
          "doubling a 42700-token sequence costs more than 2.5x");
    check(h3_plan_dit_cost_ratio(100, 0) == 1.0, "no base is a ratio of 1");

    /* A short text-to-video job fits anywhere. */
    h3_plan plan = run(ram24, 40, 0, 5700, 5660, 0, NULL, 0);
    check(plan.verdict == H3_PLAN_OK && !plan.message[0], "small job is OK");
    check(plan.video_encoder_bytes == 0 && plan.vision_encoder_bytes == 0,
          "no references, no encoder stages");
    check(plan.dit_cost_ratio == 1.0, "no reference rows, ratio 1");
    check(plan.peak_bytes >= plan.dit_bytes &&
          plan.peak_bytes >= plan.text_encoder_bytes, "peak is the largest");
    check(plan.text_fast_limit == H3_PLAN_TEXT_FAST_LIMIT &&
          plan.text_token_limit == H3_PLAN_TEXT_TOKEN_LIMIT, "limits reported");

    /* The text encoder's hard limit, on both sides. */
    plan = run(ram24, H3_PLAN_TEXT_TOKEN_LIMIT, 16000, 60000, 40000, 20000,
               NULL, 0);
    check(plan.verdict != H3_PLAN_TOO_LARGE || !strstr(plan.message, "at most"),
          "exactly the text limit is not refused for its length");
    plan = run(ram24, H3_PLAN_TEXT_TOKEN_LIMIT + 1, 16000, 60000, 40000,
               20000, NULL, 0);
    check(plan.verdict == H3_PLAN_TOO_LARGE && strstr(plan.message, "16384"),
          "one past the text limit is refused, naming it");

    /* Past the fast kernel: at least a warning, on a big Mac too. */
    plan = run(256ull << 30, H3_PLAN_TEXT_FAST_LIMIT, 7800, 30000, 20000,
               9000, NULL, 0);
    check(plan.verdict == H3_PLAN_OK, "up to the fast limit on a big Mac");
    plan = run(256ull << 30, H3_PLAN_TEXT_FAST_LIMIT + 1, 7800, 30000, 20000,
               9000, NULL, 0);
    check(plan.verdict == H3_PLAN_TIGHT && strstr(plan.message, "fallback"),
          "one past the fast limit warns about the fallback");
    check(plan.text_encoder_bytes > run(256ull << 30, H3_PLAN_TEXT_FAST_LIMIT,
          7800, 30000, 20000, 9000, NULL, 0).text_encoder_bytes,
          "the fallback costs memory");
    {
        /* Off the measured one-call path the fallback is counted as if it
         * held every head's scores: a 16000-token sequence then needs more
         * than 24 GB and is refused. */
        h3_plan_input input = {ram24, 1, 0, 1, 0, 16000, 15900, 60000, 40000,
                               20000, NULL, 0};
        h3_plan chunked;
        h3_plan_compute(&input, &chunked);
        check(chunked.verdict == H3_PLAN_TOO_LARGE &&
              !strcmp(chunked.peak_stage, "text encoder"),
              "unmeasured fallback at 16000 tokens is refused on 24 GB");
        input.full_sdpa = 1;
        h3_plan_compute(&input, &chunked);
        check(chunked.verdict == H3_PLAN_TIGHT,
              "measured fallback at 16000 tokens only warns");
    }

    /* A DiT sequence far past 24 GB is refused and blamed on the DiT;
     * a smaller Mac turns the same job from OK to tight to refused. */
    plan = run(ram24, 400, 0, 200000, 199600, 0, NULL, 0);
    check(plan.verdict == H3_PLAN_TOO_LARGE && !strcmp(plan.peak_stage, "DiT") &&
          strstr(plan.message, "DiT"), "huge DiT sequence is refused");
    h3_plan big = run(256ull << 30, 400, 0, 60000, 59600, 0, NULL, 0);
    check(big.verdict == H3_PLAN_OK, "60000 DiT tokens fit 256 GB");
    uint64_t peak = big.peak_bytes;
    plan = run((uint64_t)((double)peak / 0.9), 400, 0, 60000, 59600, 0, NULL, 0);
    check(plan.verdict == H3_PLAN_TIGHT, "peak at 90% of RAM is tight");
    plan = run((uint64_t)((double)peak / 1.5), 400, 0, 60000, 59600, 0, NULL, 0);
    check(plan.verdict == H3_PLAN_TOO_LARGE, "peak at 150% of RAM is refused");

    /* A long reference video: chunked VAE encode, its pixels held. */
    h3_plan_visual video = {1120, 640, 243};
    plan = run(ram24, 7860, 7700, 85400, 42700, 42700, &video, 1);
    check(plan.reference_video_frames == 243 &&
          plan.reference_video_width == 1120 &&
          plan.reference_video_height == 640, "longest video reported");
    check(plan.reference_video_chunk_frames > 0 &&
          plan.reference_video_chunk_frames % 4 == 0 &&
          plan.reference_video_chunk_frames < 243,
          "a 243-frame 1120x640 video is encoded in chunks");
    check(plan.video_encoder_bytes > (uint64_t)(243.0 * 1120 * 640 * 12),
          "the encoder stage holds the decoded video");
    check(plan.dit_cost_ratio > 2.5, "doubling the DiT sequence costs >2.5x");
    check(fabs(plan.reference_pixel_frames - 243.0 * 1120 * 640) < 1.0,
          "pixel-frames counted");

    /* Images count once each; the largest sets the vision stage. */
    h3_plan_visual images[2] = {{512, 512, 1}, {2048, 2048, 1}};
    h3_plan small_image = run(ram24, 300, 256, 6000, 5700, 256, images, 1);
    h3_plan two_images = run(ram24, 4400, 4352, 10000, 5700, 4352, images, 2);
    check(two_images.vision_encoder_bytes > small_image.vision_encoder_bytes,
          "a 2048x2048 image needs more vision memory");
    check(two_images.reference_video_frames == 0, "images are not videos");

    printf("ok: %d checks (generation plan)\n", checks);
    return 0;
}
