#include "h3_plan.h"
#include "h3_video_encoder.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define GIB (1024.0 * 1024.0 * 1024.0)

/* Memory model. Each stage is a fixed part (weights, buffers that do not
 * depend on the request) plus a part per token. The per-token parts come
 * from the stages' own allocations; the fixed parts are measured process
 * footprints (H3_MEMORY_TRACE=1) on a 24 GB M5 with the app's settings
 * (H3_QWEN_PREFETCH_DEPTH=1), rounded up so every measured run came in
 * under its estimate. Footprints were measured with the GGUF import under
 * SSD streaming. The int8 attention cache streams its blocks too (not
 * measured here); the resident mode's whole transformer comes in through
 * h3_plan_input.dit_resident_bytes. */

/* Qwen3-VL language model: 4.8 GiB reading one layer ahead, and 1.8 GiB
 * more for each further layer read ahead (8.4 GiB at the CLI's default of
 * 3 on M5). Per token its own buffers take 176 KiB (4 hidden-sized, 2
 * query-, 2 KV- and 2 intermediate-sized BF16 rows, h3_text_encoder.c),
 * and the vision outputs it injects, its F32 output and the rest bring
 * that to the measured 253 KiB (5.82 GiB at 2325 tokens, 6.96 at 6949,
 * with references); a job with images or video also starts it about
 * 0.5 GiB higher, with the earlier stages' leftovers. */
static const double text_fixed = 4.8 * GIB;
static const double text_per_prefetch = 1.83 * GIB;
static const double text_per_token = 256.0 * 1024.0;
static const double text_after_visuals = 0.5 * GIB;
/* The GQA kernels keep their F32 score rows in a device scratch of at most
 * 256 MiB (64 heads x tokens^2 floats below that). */
static const double text_score_rows_cap = 256.0 * 1024.0 * 1024.0;
static const double text_score_rows_heads = 64.0;
/* The MPSGraph fallback past H3_PLAN_TEXT_FAST_LIMIT. On an M5 with
 * macOS 26 or later (full_sdpa) it is one attention call with an explicit
 * BF16 causal mask, which took the mask and about 0.25 GiB more (7.84 GiB
 * at 9261 tokens). Elsewhere it runs in pieces (h3_gpu_gqa_chunked): head-
 * major copies of Q, K, V and the output (64 + 8 + 8 + 64 heads of 128
 * BF16 values per token), one piece's scores within the 1 GiB budget and
 * its in-graph mask (an eighth of that, eight query heads per KV head). In
 * a probe on the M5 forced to pieces, 16384 tokens took 1.3 GiB beyond the
 * inputs; this counts 1.7. */
static const double text_fallback_extra = 0.25 * GIB;
static const double text_fallback_chunked_per_token = 144.0 * 128.0 * 2.0;
static const double text_fallback_chunked_fixed = 1.125 * GIB;

/* Qwen's vision tower runs one image (or two-frame video block) at a
 * time: 16-pixel patches, 1152 wide, 10 hidden-sized BF16 buffers and a
 * 4304-wide MLP per patch (h3_vision_encoder.c), while the outputs so far
 * stay (four 5120-wide BF16 rows a token). Measured 1.40 GiB for one
 * 1536x1536 image (9216 patches), 1.70 for three, and 2.08 for the three
 * 1120x640 blocks of a 56-frame video (its decoded frames held), weights
 * and the process included. */
static const double vision_fixed = 1.5 * GIB;
static const double vision_per_patch = 2.0 * (10 * 1152 + 2 * 4304);
static const double vision_output_per_token = 4.0 * 5120 * 2;

/* DiT: streamed blocks, AdaLN and tables, plus per token the hidden,
 * modulation and attention output (5376 wide), QKV and its three heads
 * (7168 wide each), and the BF16 MLP's FC1 and activation (14336 wide, two
 * and one) - 199.5 KiB (h3_dit.c allocate_activations). Measured under SSD
 * streaming: 3.70 GiB at 536 tokens, 7.54 at 21740, 10.45 at 35985 (190
 * KiB a token), and 6.62 at 14383 with three reference images; the int8
 * paths of the other modes need less a token. Those were with the MLP in
 * one piece and the text encoder's freed scratch still held by malloc.
 * With the MLP in pieces of 4096 rows at most (the default since) and the
 * scratch returned, 2 steps measured 5.01 GiB at 19293 tokens on the
 * one-call attention path and 5.80 in pieces (about 142 and 174 KiB a
 * token; docs/perf/DIFFUSERS_FFN_SPLIT.ja.md). The estimate is left where
 * it was, above both: neither was measured on a 16 GB Mac or on an M1 to
 * M4. */
static const double dit_fixed = 4.0 * GIB;
static const double dit_per_token =
    2.0 * (3 * 5376 + 6 * 7168 + 3 * 14336);
/* On the int8 path (h3_plan_input.dit_int8) the QKV projection writes its
 * three heads directly and the MLP's activation lives in the QKV buffer,
 * which token reduction can grow by a hidden-sized snapshot; the quantized
 * activation takes a byte a value: 126 KiB. Measured with the int8
 * attention cache streaming every block (docs/perf/
 * DIFFUSERS_BASELINE.ja.md): 116.4 KiB a token and 2.45 GiB beside them at
 * 2 steps, 2.78 at 20 (3.56 GiB at 9895 tokens, 4.87 at 21735, 3.88 at
 * 9895 with 20 steps). Core reuse keeps two more hidden-sized buffers,
 * which dit_fixed's margin over those covers up to about 60000 tokens. */
static const double dit_int8_per_token =
    2.0 * (4 * 5376 + 3 * 7168 + 14336) + 14336 + 64;
/* Text rows cost more: the refined text and the text embedding are kept
 * beside them (7.98 GiB at 18999 tokens with 9261 text rows). */
static const double dit_per_text_token = 64.0 * 1024.0;
/* Off the one-call path, dense attention runs in pieces whose scores fit
 * a 1 GiB budget (h3_gpu.m). */
static const double chunked_sdpa_scores = 1.0 * GIB;

/* The video VAE encoder stage: h3_video_encoder.c's own estimate plus the
 * decoded references (F32 RGB) and this much more. In 16-frame chunks at
 * 1120x640, a 56-frame reference measured 8.15 GiB against 8.14 from those
 * two and a 243-frame one 10.2 against 9.64: each of its 18 tiles x 16
 * chunks leaves a little behind until the encode ends. */
static const double video_encoder_margin = 0.6 * GIB;

/* The video VAE decoder stage, after the DiT has gone. Its 36 blocks stay
 * loaded while it decodes 256-pixel tiles of 17 frames, so what it needs
 * does not grow with the canvas: the weights (F32 9.7 GiB; as int8 where
 * the GPU has tensor units, 2.7), one tile's activations, and the decoded
 * frames so far (F32 RGB, 12 bytes a pixel) where the whole video is kept
 * until it is written. Loading peaks above that for a moment, as the last
 * blocks arrive (and, for int8, are quantized from F32).
 *
 * Measured on the 24 GB M5 (docs/perf/DIFFUSERS_FFN_SPLIT.ja.md, 13), int8: the
 * load 3.2 to 4.9 GiB (5.1 after a resident DiT), and at 960x544 4.8 GiB
 * for 243 frames and 5.3 to 5.4 for 362, which is 3.3 to 3.4 beside the
 * frames. F32 (tensor units off, as on an M1 to M4): the load 11.1 to 11.8
 * at 512x512, and at 960x544 10.7 to 11.1 for 124 frames, 11.4 for 243 and
 * 12.1 for 362, which is 10.0 beside the frames. Not measured on an M1 to
 * M4. */
static const double video_decoder_load_int8 = 5.2 * GIB;
static const double video_decoder_load_f32 = 12.0 * GIB;
static const double video_decoder_held_int8 = 3.6 * GIB;
static const double video_decoder_held_f32 = 10.4 * GIB;
static const double decoded_frame_bytes = 3.0 * 4.0;
/* Loaded before the denoise to show its steps (preview_denoise), its
 * weights stay beside the DiT. Counted, not measured. */
static const double video_decoder_weights_int8 = 2.7 * GIB;
static const double video_decoder_weights_f32 = 9.7 * GIB;

/* A pi-H3 pixel head in the video decoder's place (h3_pixel_head.c): its
 * own GPU context, one of its 8 layers' weights at a time, 512 rows of
 * scratch, and for the whole clip at once its activations (BF16 hidden, Q,
 * K and V rows of 2048, 16 KiB a token) and the DiT rows it reads (2.6
 * KiB); the frames leave as they come. Measured on the 24 GB M5
 * (plans/PI_H3_PLAN.ja.md, 8.7): 1.49 GiB at 384x384 and 56 frames (9792
 * tokens), 2.14 at 832x480 and 124 (57720), 4.04 at 1344x768 and 124
 * (149184, the head alone). */
static const double pixel_head_fixed = 1.5 * GIB;
static const double pixel_head_per_token = 19.0 * 1024.0;

/* Verdicts against RAM: past `tight` macOS starts compressing and swapping
 * heavily; past `too_large` the run risks exhausting swap, which is what
 * restarted this Mac twice (kernel watchdog). */
static const double tight_fraction = 0.85;
static const double too_large_fraction = 1.25;

double h3_plan_dit_cost_ratio(size_t tokens, size_t base) {
    if (!base) return 1.0;
    /* Per token and block: QKV, output and MLP projections take
     * 5376 * (3 * 7168 + 7168 + 2 * 14336 + 14336) multiply-adds; attention
     * takes 2 * 7168 per pair of tokens. Equal at 26880 tokens. */
    const double crossover = 26880.0;
    double n = (double)tokens, b = (double)base;
    return n * (1.0 + n / crossover) / (b * (1.0 + b / crossover));
}

static void stage(h3_plan *plan, uint64_t bytes, const char *name) {
    if (bytes <= plan->peak_bytes) return;
    plan->peak_bytes = bytes;
    snprintf(plan->peak_stage, sizeof(plan->peak_stage), "%s", name);
}

void h3_plan_compute(const h3_plan_input *input, h3_plan *plan) {
    plan->verdict = H3_PLAN_OK;
    plan->message[0] = '\0';
    plan->peak_bytes = 0;
    plan->peak_stage[0] = '\0';
    plan->memory_bytes = input->memory_bytes;
    plan->text_tokens = input->text_tokens;
    plan->vision_tokens = input->vision_tokens;
    plan->text_fast_limit = H3_PLAN_TEXT_FAST_LIMIT;
    plan->text_token_limit = H3_PLAN_TEXT_TOKEN_LIMIT;
    plan->dit_tokens = input->dit_tokens;
    plan->dit_reference_tokens = input->dit_reference_tokens;
    plan->dit_generated_tokens = input->dit_generated_tokens;
    /* Without the references their rows go, and so do the image tokens
     * they put in the text rows (FL2VA anchors keep theirs). */
    size_t removed = input->dit_reference_tokens +
        (input->dit_reference_tokens ? input->vision_tokens : 0);
    plan->dit_cost_ratio = h3_plan_dit_cost_ratio(
        input->dit_tokens,
        input->dit_tokens > removed ? input->dit_tokens - removed : 1);

    /* Video VAE encoder: every decoded reference is held (F32 RGB) while
     * each is encoded in turn; the largest encode sets the peak. */
    double pixels_held = 0.0, largest_encode = 0.0, largest_video = 0.0;
    double largest_patches = 0.0;
    uint64_t budget = h3_video_encoder_default_budget();
    plan->reference_pixel_frames = 0.0;
    plan->reference_video_frames = 0;
    plan->reference_video_width = plan->reference_video_height = 0;
    plan->reference_video_chunk_frames = 0;
    for (size_t index = 0; index < input->visual_count; index++) {
        const h3_plan_visual *visual = &input->visuals[index];
        double pixel_frames = (double)visual->width * visual->height *
                              visual->frames;
        pixels_held += 3.0 * 4.0 * pixel_frames;
        plan->reference_pixel_frames += pixel_frames;
        int chunk = 0;
        uint64_t encode;
        if (visual->frames > 1 && h3_video_encoder_plan(
                visual->frames, visual->height, visual->width, budget, &chunk) &&
            chunk)
            encode = h3_video_encoder_chunked_peak_bytes(
                chunk, visual->height, visual->width);
        else
            encode = h3_video_encoder_peak_bytes(
                visual->frames, visual->height, visual->width);
        if ((double)encode > largest_encode) largest_encode = (double)encode;
        if (visual->frames > 1 && pixel_frames > largest_video) {
            largest_video = pixel_frames;
            plan->reference_video_frames = visual->frames;
            plan->reference_video_width = visual->width;
            plan->reference_video_height = visual->height;
            plan->reference_video_chunk_frames = chunk;
        }
        double patches = (double)(visual->width / 16) * (visual->height / 16);
        if (patches > largest_patches) largest_patches = patches;
    }
    plan->video_encoder_bytes = input->visual_count ?
        (uint64_t)(pixels_held + largest_encode + video_encoder_margin) : 0;
    plan->vision_encoder_bytes = input->visual_count ? (uint64_t)(
        pixels_held + vision_fixed + largest_patches * vision_per_patch +
        (double)input->vision_tokens * vision_output_per_token +
        (input->full_sdpa ? 0.0 : chunked_sdpa_scores)) : 0;

    double text_tokens = (double)input->text_tokens;
    double fallback = 0.0;
    if (input->text_tokens > H3_PLAN_TEXT_FAST_LIMIT)
        fallback = input->full_sdpa ?
            text_tokens * text_tokens * 2.0 + text_fallback_extra :
            text_fallback_chunked_fixed +
                text_tokens * text_fallback_chunked_per_token;
    double score_rows = text_tokens > H3_PLAN_TEXT_FAST_LIMIT ? 0.0 :
        fmin(text_score_rows_cap,
             text_score_rows_heads * text_tokens * text_tokens * 4.0);
    int depth = input->text_prefetch_depth > 1 ? input->text_prefetch_depth : 1;
    plan->text_encoder_bytes = (uint64_t)(text_fixed +
        (depth - 1) * text_per_prefetch + text_tokens * text_per_token +
        (input->visual_count ? text_after_visuals : 0.0) + fallback +
        score_rows);

    plan->dit_bytes = (uint64_t)(dit_fixed + (double)input->dit_resident_bytes +
        (double)input->dit_tokens *
            (input->dit_int8 ? dit_int8_per_token : dit_per_token) +
        (double)input->text_tokens * dit_per_text_token +
        (input->full_sdpa ? 0.0 : chunked_sdpa_scores) +
        (!input->video_decoder_with_dit ? 0.0 :
         input->video_decoder_int8 ? video_decoder_weights_int8 :
                                     video_decoder_weights_f32));

    /* Only where a video is decoded: a still image (h3_image.c) has its own
     * decoder, and gives no frames here. */
    double decoder_load = input->decoded_pixel_frames <= 0.0 ? 0.0 :
        input->video_decoder_int8 ? video_decoder_load_int8 :
                                    video_decoder_load_f32;
    plan->video_decoder_bytes = input->decoded_pixel_frames <= 0.0 ? 0 :
        (uint64_t)fmax(decoder_load,
            (input->video_decoder_int8 ? video_decoder_held_int8 :
                                         video_decoder_held_f32) +
                input->decoded_pixel_frames * decoded_frame_bytes);
    /* The pixel head stands where the decoder would: nothing of the decoder
     * is loaded after the DiT. */
    if (input->pixel_head_tokens) {
        decoder_load = 0.0;
        plan->video_decoder_bytes = (uint64_t)(pixel_head_fixed +
            (double)input->pixel_head_tokens * pixel_head_per_token);
    }

    stage(plan, plan->video_encoder_bytes, "video encoder");
    stage(plan, plan->vision_encoder_bytes, "vision encoder");
    stage(plan, plan->text_encoder_bytes, "text encoder");
    stage(plan, plan->dit_bytes, "DiT");
    stage(plan, plan->video_decoder_bytes,
          input->pixel_head_tokens ? "pixel head" : "video decoder");

    double ram = (double)input->memory_bytes;
    double peak = (double)plan->peak_bytes;
    char *message = plan->message;
    size_t size = sizeof(plan->message);
    if (input->text_tokens > H3_PLAN_TEXT_TOKEN_LIMIT) {
        plan->verdict = H3_PLAN_TOO_LARGE;
        snprintf(message, size,
                 "The references need %zu text-encoder tokens (%zu of them "
                 "from images and video); it takes at most %d. Use a shorter "
                 "or smaller reference video, or fewer reference images.",
                 input->text_tokens, input->vision_tokens,
                 H3_PLAN_TEXT_TOKEN_LIMIT);
    } else if (ram > 0.0 && decoder_load > too_large_fraction * ram &&
               plan->peak_bytes == plan->video_decoder_bytes) {
        /* Nothing about the request changes what loading the decoder
         * takes. */
        plan->verdict = H3_PLAN_TOO_LARGE;
        snprintf(message, size,
                 "The video decoder would need about %.1f GiB, more than this "
                 "Mac's %.0f GiB can hold without risking the whole system. "
                 "It needs that much for a video of any size.",
                 peak / GIB, ram / GIB);
    } else if (ram > 0.0 && peak > too_large_fraction * ram) {
        plan->verdict = H3_PLAN_TOO_LARGE;
        snprintf(message, size,
                 "The %s would need about %.1f GiB, more than this Mac's "
                 "%.0f GiB can hold without risking the whole system. Use a "
                 "shorter or smaller reference, fewer frames or a smaller "
                 "canvas.", plan->peak_stage, peak / GIB, ram / GIB);
    } else if (ram > 0.0 && peak > tight_fraction * ram) {
        plan->verdict = H3_PLAN_TIGHT;
        snprintf(message, size,
                 "The %s would need about %.1f GiB of this Mac's %.0f GiB: "
                 "expect swapping and a much slower run.",
                 plan->peak_stage, peak / GIB, ram / GIB);
    } else if (input->text_tokens > H3_PLAN_TEXT_FAST_LIMIT) {
        plan->verdict = H3_PLAN_TIGHT;
        snprintf(message, size,
                 "The references need %zu text-encoder tokens, past the %d "
                 "its fast attention takes; a slower fallback runs instead.",
                 input->text_tokens, H3_PLAN_TEXT_FAST_LIMIT);
    }
}
