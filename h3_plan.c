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
 * paths of the other modes need less a token. */
static const double dit_fixed = 4.0 * GIB;
static const double dit_per_token =
    2.0 * (3 * 5376 + 6 * 7168 + 3 * 14336);
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
        (double)input->dit_tokens * dit_per_token +
        (double)input->text_tokens * dit_per_text_token +
        (input->full_sdpa ? 0.0 : chunked_sdpa_scores));

    stage(plan, plan->video_encoder_bytes, "video encoder");
    stage(plan, plan->vision_encoder_bytes, "vision encoder");
    stage(plan, plan->text_encoder_bytes, "text encoder");
    stage(plan, plan->dit_bytes, "DiT");

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
