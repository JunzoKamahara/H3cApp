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
                           reference, visuals, visual_count, 0, 0, 0.0, 0, 0};
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
    check(plan.video_decoder_bytes == 0 &&
          strcmp(plan.peak_stage, "video decoder") != 0,
          "nothing decoded to video (a still image), no video decoder stage");
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
        /* Off the one-call path the fallback runs in pieces, so a
         * 16000-token sequence fits 24 GB there too, and costs more than
         * the one call's mask. */
        h3_plan_input input = {ram24, 1, 0, 1, 0, 16000, 15900, 60000, 40000,
                               20000, NULL, 0, 0, 0, 0.0, 0, 0};
        h3_plan chunked;
        h3_plan_compute(&input, &chunked);
        check(chunked.verdict != H3_PLAN_TOO_LARGE,
              "chunked fallback at 16000 tokens is not refused on 24 GB");
        uint64_t pieces_bytes = chunked.text_encoder_bytes;
        input.full_sdpa = 1;
        h3_plan_compute(&input, &chunked);
        check(chunked.verdict == H3_PLAN_TIGHT,
              "measured fallback at 16000 tokens only warns");
        check(pieces_bytes > chunked.text_encoder_bytes,
              "the pieces are counted above the one call's mask");
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

    {
        /* The int8 path keeps fewer activation buffers a token; its
         * estimate still covers the measured peaks (3.88 GiB at 9895
         * tokens and 20 steps, 4.87 GiB at 21735 tokens and 2 steps). */
        h3_plan_input input = {ram24, 0, 1, 1, 0, 30, 0, 9895, 9865, 0,
                               NULL, 0, 0, 0, 0.0, 0, 0};
        h3_plan bf16, int8;
        h3_plan_compute(&input, &bf16);
        input.dit_int8 = 1;
        h3_plan_compute(&input, &int8);
        check(int8.dit_bytes < bf16.dit_bytes,
              "the int8 DiT needs less memory a token");
        check((double)int8.dit_bytes > 3.88 * GIB,
              "the int8 estimate covers 9895 measured tokens");
        input.dit_tokens = 21735;
        h3_plan_compute(&input, &int8);
        check((double)int8.dit_bytes > 4.87 * GIB,
              "the int8 estimate covers 21735 measured tokens");
    }

    {
        /* The video VAE decoder, after the DiT: its weights and one tile
         * whatever the canvas, and the decoded frames beside them. The
         * estimates cover what was measured (int8 5.44 GiB and F32 12.11
         * at 960x544 and 362 frames, F32 11.76 at 512x512 and 49). */
        h3_plan_input input = {ram24, 1, 0, 1, 0, 9, 0, 4547, 4538, 0,
                               NULL, 0, 0, 0, 512.0 * 512 * 39, 0, 0};
        h3_plan f32, int8, longer;
        h3_plan_compute(&input, &f32);
        check((double)f32.video_decoder_bytes > 11.76 * GIB,
              "the F32 decoder's estimate covers a measured short video");
        check(!strcmp(f32.peak_stage, "video decoder") &&
              f32.peak_bytes == f32.video_decoder_bytes,
              "a short video peaks in the F32 decoder");
        check(f32.verdict == H3_PLAN_OK, "the F32 decoder fits 24 GB");
        input.video_decoder_int8 = 1;
        h3_plan_compute(&input, &int8);
        check(int8.video_decoder_bytes < f32.video_decoder_bytes / 2,
              "the int8 decoder needs less than half");
        input.decoded_pixel_frames = 960.0 * 544 * 362;
        h3_plan_compute(&input, &longer);
        check(longer.video_decoder_bytes > int8.video_decoder_bytes &&
              (double)longer.video_decoder_bytes > 5.44 * GIB,
              "kept frames count, and cover 362 measured frames in int8");
        input.video_decoder_int8 = 0;
        h3_plan_compute(&input, &longer);
        check((double)longer.video_decoder_bytes > 12.11 * GIB,
              "the F32 estimate covers 362 measured frames");

        /* Shown during the denoise, its weights are beside the DiT. */
        input.decoded_pixel_frames = 512.0 * 512 * 22;
        h3_plan without, with;
        h3_plan_compute(&input, &without);
        input.video_decoder_with_dit = 1;
        h3_plan_compute(&input, &with);
        check((double)(with.dit_bytes - without.dit_bytes) > 9.0 * GIB &&
              !strcmp(with.peak_stage, "DiT"),
              "a decoder showing the denoise is counted with the DiT");

        /* 16 GB without tensor units holds it; 8 GB is refused for the
         * decoder itself, and is not told to make the video smaller. */
        input.video_decoder_with_dit = 0;
        input.memory_bytes = 16ull << 30;
        h3_plan_compute(&input, &f32);
        check(f32.verdict == H3_PLAN_OK, "the F32 decoder fits 16 GB");
        input.memory_bytes = 8ull << 30;
        h3_plan_compute(&input, &f32);
        check(f32.verdict == H3_PLAN_TOO_LARGE &&
              strstr(f32.message, "video decoder") &&
              strstr(f32.message, "any size") &&
              !strstr(f32.message, "smaller"),
              "8 GB is refused for the F32 decoder, whatever the video");
        input.video_decoder_int8 = 1;
        h3_plan_compute(&input, &int8);
        check(int8.verdict != H3_PLAN_TOO_LARGE,
              "8 GB is not refused for the int8 decoder");
    }

    {
        /* A pi-H3 pixel head stands where the decoder would: its estimate
         * grows with the clip and covers what was measured (1.49, 2.14 and 4.04 GiB),
         * and it is the stage named. */
        h3_plan_input input = {ram24, 0, 1, 1, 0, 23, 0, 2657, 2634, 0,
                               NULL, 0, 1, 1, 384.0 * 384 * 22, 0, 0};
        h3_plan decoder, head;
        h3_plan_compute(&input, &decoder);
        input.pixel_head_tokens = 17 * 24 * 24;
        h3_plan_compute(&input, &head);
        check(head.video_decoder_bytes < decoder.video_decoder_bytes &&
              (double)head.video_decoder_bytes > 1.49 * GIB,
              "a short clip's pixel head needs less than the int8 decoder");
        check(head.dit_bytes == decoder.dit_bytes,
              "the pixel head changes nothing of the DiT's estimate");
        input.pixel_head_tokens = 37 * 30 * 52;
        h3_plan_compute(&input, &head);
        check((double)head.video_decoder_bytes > 2.14 * GIB,
              "the pixel head's estimate covers 832x480 and 124 frames");
        input.pixel_head_tokens = 37 * 48 * 84;
        h3_plan_compute(&input, &head);
        check((double)head.video_decoder_bytes > 4.04 * GIB &&
              (double)head.video_decoder_bytes < 4.5 * GIB,
              "the pixel head's estimate covers 1344x768 and 124 frames");
        /* Four times that clip on a Mac the DiT's stage still fits. */
        input.pixel_head_tokens *= 4;
        input.memory_bytes = 8ull << 30;
        h3_plan_compute(&input, &head);
        check(head.verdict == H3_PLAN_TOO_LARGE &&
              !strstr(head.message, "any size") &&
              !strcmp(head.peak_stage, "pixel head") &&
              strstr(head.message, "pixel head") &&
              strstr(head.message, "smaller"),
              "a pixel head too large is told to make the video smaller");
        check(strcmp(head.peak_stage, "video decoder") != 0,
              "with a pixel head no stage is the video decoder");
    }

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
