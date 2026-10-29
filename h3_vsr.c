/* The one-step VSR model and its streaming session: the public API of
 * h3_vsr.h over the cache (h3_vsr_weights.c), the ReAE (h3_vsr_reae.c),
 * the DiT (h3_vsr_wan.c) and the chunk / window geometry. UPSCALE_PLAN §6.
 *
 * A session upscales one video chunk by chunk, exactly as LightX2V's
 * SwiftVR runner does: frames are resized to the target size and
 * zero-padded to multiples of 32, encoded by the ReAE, restored by one DiT
 * pass (z_hq = z_lq - v), decoded, cropped and written out. The first
 * chunk has 28 frames, the next ones 24; the ReAE's causal state and the
 * last LQ latents (which pad the DiT input of the final chunk) are carried.
 * Output lags the input by the decoder's 3 causal frames, so a call
 * returns the frames of the chunk it processed minus that delay and the
 * last call flushes the rest. The three stages run in sequence on one GPU
 * context. A frame whose chunk would not fit the GPU's memory is restored
 * as a grid of overlapping tiles (h3_vsr_tile_plan), each with its own
 * ReAE state, and the tiles are blended into the frame with linear ramps
 * across the overlaps. */
#include "h3_vsr_internal.h"
#include "h3_metal.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

struct h3_vsr {
    h3_gpu *gpu;
    h3_vsr_wan *wan;
    h3_vsr_reae *reae;
    /* The decoder on its own context: a chunk's frames are decoded while
     * the next chunk's DiT runs (NULL = everything on one context). */
    h3_gpu *gpu_b;
    h3_vsr_reae *reae_b;
    h3_vsr_info info;
    h3_vsr_open_params params;
    int clip_latents;    /* 6 */
};

typedef struct {
    int frames;          /* latent frames the layout is for (0 = unused) */
    h3_vsr_window_layout even, odd;
} layout_slot;

/* One spatial tile of a session (the whole frame when there is one): its
 * origin in the padded output frame, the part of it inside the real
 * output, the ReAE's causal state and the last LQ latents (the final
 * chunk's DiT prefix), and its blend weights. */
typedef struct {
    int x, y;
    int crop_w, crop_h;
    h3_vsr_reae_state *enc_state, *dec_state;
    h3_gpu_tensor *prev_lq;        /* F32 [prev_count, lh, lw, 48] */
    int prev_count;
    h3_gpu_tensor *weights;        /* F32 [tile_h, tile_w]; NULL with one tile */
} session_tile;

struct h3_vsr_session {
    h3_vsr *vsr;
    int in_w, in_h, out_w, out_h;
    int full_w, full_h;            /* out rounded up to multiples of 32 */
    h3_vsr_tile_plan plan;         /* 1 x 1: the whole padded frame */
    int tile_w, tile_h;            /* what the ReAE and the DiT see (multiples of 32) */
    int latent_h, latent_w;        /* tile / 16 */
    session_tile *tiles;
    int tile_count;
    int sync_decode;               /* several tiles: decode on the main context */
    int chunk_index;
    int start;                     /* padded frame index the next chunk starts at */
    int real_in;                   /* real frames fed so far */
    int real_out;                  /* real frames written so far */
    int done;
    layout_slot layouts[4];
    /* A chunk whose decode is running on the second context (one tile). */
    struct {
        int valid, chunk_index, count, skip, latents;
        h3_gpu_tensor *own;        /* its latents */
        h3_gpu_tensor *decoded;    /* [latents * 4, tile_h, tile_w, 3] */
        float input_means[3];      /* of the chunk's input frames (colour match) */
    } pending;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

/* Development trace (H3_VSR_TRACE_LIVE): the GPU's live tensor bytes. */
/* Development trace (H3_VSR_TRACE_STAGES): wall time since the previous
 * mark, printed at points where the GPU has been waited for. */
static double trace_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9; }
static void trace_stage(double *mark, const char *label) {
    if (!getenv("H3_VSR_TRACE_STAGES")) return;
    double now = trace_now();
    fprintf(stderr, "vsr stage %-28s %7.1f ms\n", label, (now - *mark) * 1000.0);
    *mark = now;
}

static void trace_live(h3_gpu *gpu, const char *label) {
    h3_gpu_stats st;
    if (getenv("H3_VSR_TRACE_LIVE") && h3_gpu_get_stats(gpu, &st))
        fprintf(stderr, "vsr live %.2f GiB: %s\n", (double)st.live_bytes / 1073741824.0, label);
}

static int round_up(int value, int multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

/* ---------------------------------------------------------------- */
/* Model */

void h3_vsr_close(h3_vsr *vsr) {
    if (!vsr) return;
    h3_vsr_wan_free(vsr->wan);
    h3_vsr_reae_free(vsr->reae);
    h3_vsr_reae_free(vsr->reae_b);
    h3_gpu_free(vsr->gpu_b);
    h3_gpu_free(vsr->gpu);
    free(vsr);
}

h3_vsr *h3_vsr_open(const char *model_dir, const h3_vsr_open_params *params,
                    h3_progress_callback on_progress, void *opaque,
                    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!model_dir || !params || !params->cache_path) {
        fail(error, error_size, "h3_vsr_open: a model directory and a cache path are required");
        return NULL;
    }
    h3_vsr *vsr = calloc(1, sizeof(*vsr));
    if (!vsr) return NULL;
    vsr->params = *params;
    /* 0 = "auto": 4 frames per ReAE pass (one latent step when decoding)
     * bounds the activations; larger batches measured no faster (bench_vsr). */
    if (vsr->params.reae_frame_batch <= 0) vsr->params.reae_frame_batch = 4;
    if (getenv("H3_VSR_NO_COLOR_MATCH")) vsr->params.color_match = -1;
    vsr->clip_latents = H3_VSR_CLIP_FRAMES / H3_VSR_TEMPORAL;
    if (!h3_vsr_inspect_dir(model_dir, &vsr->info, error, error_size)) goto failed;
    if (!h3_vsr_cache_matches(params->cache_path, model_dir)) {
        fail(error, error_size, "%s does not match the weights in %s (rebuild the cache)",
             params->cache_path, model_dir);
        goto failed;
    }
    if (on_progress && on_progress("vsr load", 0, 3, opaque)) goto cancelled;
    vsr->gpu = h3_gpu_create(params->shader_source_path ? params->shader_source_path : "h3_shaders.metal",
                             error, error_size);
    if (!vsr->gpu) goto failed;
    if (on_progress && on_progress("vsr load", 1, 3, opaque)) goto cancelled;
    vsr->wan = h3_vsr_wan_load(vsr->gpu, params->cache_path, error, error_size);
    if (!vsr->wan) goto failed;
    trace_live(vsr->gpu, "after wan load");
    if (on_progress && on_progress("vsr load", 2, 3, opaque)) goto cancelled;
    vsr->reae = h3_vsr_reae_load(model_dir, vsr->gpu, error, error_size);
    if (!vsr->reae) goto failed;
    trace_live(vsr->gpu, "after reae load");
    /* Off by default: measured no faster at 1080p on M5, where the DiT
     * already saturates the GPU (UPSCALE_PLAN appendix A.2). */
    if (params->reae_separate_queue > 0 || getenv("H3_VSR_OVERLAP")) {
        vsr->gpu_b = h3_gpu_create(params->shader_source_path ? params->shader_source_path : "h3_shaders.metal",
                                   error, error_size);
        if (!vsr->gpu_b) goto failed;
        vsr->reae_b = h3_vsr_reae_load(model_dir, vsr->gpu_b, error, error_size);
        if (!vsr->reae_b) goto failed;
    }
    if (on_progress && on_progress("vsr load", 3, 3, opaque)) goto cancelled;
    return vsr;
cancelled:
    fail(error, error_size, "vsr load cancelled");
failed:
    h3_vsr_close(vsr);
    return NULL;
}

int h3_vsr_get_stats(const h3_vsr *vsr, double *dit_seconds,
                     double *reae_seconds, uint64_t *peak_live_bytes) {
    if (!vsr) return 0;
    h3_gpu_stats stats;
    if (!h3_gpu_get_stats(vsr->gpu, &stats)) return 0;
    /* The main context runs the encoder and the DiT; the decoder has its
     * own when the chunks overlap. */
    if (dit_seconds) *dit_seconds = stats.gpu_seconds;
    if (reae_seconds) *reae_seconds = 0.0;
    if (peak_live_bytes) *peak_live_bytes = stats.peak_live_bytes;
    if (vsr->gpu_b) {
        h3_gpu_stats b;
        if (h3_gpu_get_stats(vsr->gpu_b, &b)) {
            if (reae_seconds) *reae_seconds = b.gpu_seconds;
            if (peak_live_bytes) *peak_live_bytes = stats.peak_live_bytes + b.peak_live_bytes;
        }
    }
    return 1;
}

int h3_vsr_gpu_stats(const h3_vsr *vsr, h3_gpu_stats *stats) {
    return vsr && stats && h3_gpu_get_stats(vsr->gpu, stats);
}

/* ---------------------------------------------------------------- */
/* Requirements and the tile plan */

/* The least overlap between tiles, in output pixels (a multiple of 32). */
static int tile_overlap(void) {
    const char *env = getenv("H3_VSR_TILE_OVERLAP");
    int overlap = env && atoi(env) >= 32 ? atoi(env) / 32 * 32 : H3_VSR_TILE_OVERLAP;
    return overlap;
}

/* The memory a chunk takes on one tile_w x tile_h tile: about 150 KB per
 * DiT token of activations, the ReAE's 4-frame passes on the BF16 grid
 * (three 64-channel activations live at full resolution: about 1.5 KB per
 * pixel) and the F32 frames of one chunk in and out. With the weights this
 * said 11.7 GB at 1920x1088 on int8 where 9.9 GiB were measured live
 * (UPSCALE_PLAN appendix A.2). */
static uint64_t tile_bytes(int latents, int tile_w, int tile_h) {
    uint64_t pixels = (uint64_t)tile_w * tile_h;
    uint64_t tokens = (uint64_t)latents * (tile_w / H3_VSR_TOKEN_PIXELS) * (tile_h / H3_VSR_TOKEN_PIXELS);
    uint64_t frames = H3_VSR_CLIP_FRAMES + 4 + 3;
    return tokens * 150000 + pixels * 1536 + frames * pixels * 3 * 4 * 2;
}

/* The peak estimate of a plan: the weights, one tile's chunk, the chunk's
 * 8-bit input and, with several tiles, their weights, the F32 composite of
 * a chunk's output frames and the further tiles' ReAE states (about 210 B
 * per tile pixel of MemBlock history; one tile's is inside tile_bytes).
 * Measured peak live 17.6 GB against 16.4 GB for 2 x 2 tiles of 1152x1984
 * and 12.5 GB against 11.5 GB for one tile of 1088x1920 (appendix A.3). */
static uint64_t plan_bytes(const h3_vsr *vsr, const h3_vsr_tile_plan *plan,
                           int in_w, int in_h, int out_w, int out_h) {
    int int8 = h3_vsr_wan_is_int8(vsr->wan);
    uint64_t bytes = int8 ? vsr->info.weight_bytes_int8 : vsr->info.weight_bytes_bf16;
    uint64_t tile_pixels = (uint64_t)plan->tile_w * plan->tile_h;
    int tiles = plan->columns * plan->rows;
    bytes += tile_bytes(vsr->clip_latents + 1, plan->tile_w, plan->tile_h);
    bytes += (uint64_t)(H3_VSR_CLIP_FRAMES + 4 + 3) * in_w * in_h * 4;
    if (tiles > 1)
        bytes += (uint64_t)(H3_VSR_CLIP_FRAMES + 4) * out_w * out_h * 3 * 4 +
                 (uint64_t)tiles * tile_pixels * 4 + (uint64_t)(tiles - 1) * tile_pixels * 210;
    return bytes;
}

/* The fewest tiles whose estimate fits `budget` (0: no limit, one tile);
 * among as many tiles, the smallest estimate. H3_VSR_TILES=CxR forces a
 * grid. Returns 0 when none fits. */
static int choose_plan(const h3_vsr *vsr, int in_w, int in_h, int out_w, int out_h,
                       int full_w, int full_h, uint64_t budget,
                       h3_vsr_tile_plan *plan, uint64_t *bytes) {
    int overlap = tile_overlap();
    const char *forced = getenv("H3_VSR_TILES");
    int fc = 0, fr = 0;
    if (forced && sscanf(forced, "%dx%d", &fc, &fr) == 2 && fc >= 1 && fr >= 1) {
        if (!h3_vsr_tile_plan_build(full_w, full_h, fc, fr, overlap, plan)) return 0;
        *bytes = plan_bytes(vsr, plan, in_w, in_h, out_w, out_h);
        return 1;
    }
    for (int total = 1; total <= H3_VSR_MAX_TILES_PER_AXIS; total++) {
        int found = 0;
        h3_vsr_tile_plan best;
        uint64_t best_bytes = 0;
        for (int rows = 1; rows <= total; rows++) {
            if (total % rows) continue;
            int columns = total / rows;
            h3_vsr_tile_plan candidate;
            if (!h3_vsr_tile_plan_build(full_w, full_h, columns, rows, overlap, &candidate)) continue;
            uint64_t b = plan_bytes(vsr, &candidate, in_w, in_h, out_w, out_h);
            if (!found || b < best_bytes) { best = candidate; best_bytes = b; found = 1; }
        }
        if (found && (!budget || best_bytes <= budget)) {
            *plan = best;
            *bytes = best_bytes;
            return 1;
        }
    }
    return 0;
}

int h3_vsr_check_requirements(const h3_vsr *vsr, int in_w, int in_h,
                              int out_w, int out_h, h3_vsr_requirements *out) {
    if (!vsr || !out) return 0;
    memset(out, 0, sizeof(*out));
    const h3_vsr_info *info = &vsr->info;
    if (in_w < 8 || in_h < 8) {
        snprintf(out->reason, sizeof(out->reason), "the input is too small");
        return 1;
    }
    if (out_w < info->min_output_side || out_h < info->min_output_side) {
        snprintf(out->reason, sizeof(out->reason), "the output must be at least %d pixels a side",
                 info->min_output_side);
        return 1;
    }
    if (out_w > info->max_output_side || out_h > info->max_output_side) {
        snprintf(out->reason, sizeof(out->reason), "the output must be at most %d pixels a side",
                 info->max_output_side);
        return 1;
    }
    int full_w = round_up(out_w, H3_VSR_TOKEN_PIXELS), full_h = round_up(out_h, H3_VSR_TOKEN_PIXELS);
    int latents = vsr->clip_latents + 1;   /* the first chunk's 7 */
    uint64_t frames = (uint64_t)(H3_VSR_CLIP_FRAMES + 4 + 3);
    out->host_bytes = frames * (uint64_t)full_w * full_h * 4 * 2;
    h3_device_info device;
    char probe_error[256];
    uint64_t budget = 0;
    if (h3_metal_probe(&device, probe_error, sizeof(probe_error)))
        budget = device.recommended_working_set ? device.recommended_working_set : device.physical_memory;
    budget = budget * 9 / 10;
    if (getenv("H3_VSR_IGNORE_BUDGET")) budget = 0;   /* measuring: one tile, whatever it takes */
    h3_vsr_tile_plan plan;
    uint64_t bytes = 0;
    if (!choose_plan(vsr, in_w, in_h, out_w, out_h, full_w, full_h, budget, &plan, &bytes)) {
        h3_vsr_tile_plan whole;
        h3_vsr_tile_plan_build(full_w, full_h, 1, 1, H3_VSR_TILE_OVERLAP, &whole);
        out->gpu_bytes = plan_bytes(vsr, &whole, in_w, in_h, out_w, out_h);
        snprintf(out->reason, sizeof(out->reason),
                 "about %.1f GB of GPU memory is needed, more than %.1f GB even in tiles; this Mac offers %.1f GB",
                 (double)(out->gpu_bytes / 1000000) / 1e3, (double)(budget / 1000000) / 1e3,
                 (double)(budget / 9 * 10 / 1000000) / 1e3);
        return 1;
    }
    int grid_w = plan.tile_w / H3_VSR_TOKEN_PIXELS, grid_h = plan.tile_h / H3_VSR_TOKEN_PIXELS;
    int tiles = plan.columns * plan.rows;
    int tokens = latents * grid_w * grid_h;
    int starts[1024];
    int he = h3_vsr_axis_window_starts(grid_h, H3_VSR_WINDOW, 0, starts, 1024);
    int we = h3_vsr_axis_window_starts(grid_w, H3_VSR_WINDOW, 0, starts, 1024);
    int ho = h3_vsr_axis_window_starts(grid_h, H3_VSR_WINDOW, 1, starts, 1024);
    int wo = h3_vsr_axis_window_starts(grid_w, H3_VSR_WINDOW, 1, starts, 1024);
    out->tokens_per_chunk = tokens * tiles;
    out->windows_even = he * we;
    out->windows_odd = ho * wo;
    out->tile_columns = plan.columns;
    out->tile_rows = plan.rows;
    out->tile_width = plan.tile_w;
    out->tile_height = plan.tile_h;
    out->gpu_bytes = bytes;
    /* Time: an unmeasured first guess (bench_vsr replaces it): 165 ms per
     * block per 12,240 tokens on int8, 291 ms on BF16, plus 1.5 s ReAE per
     * 1920x1088 of tile. */
    int int8 = h3_vsr_wan_is_int8(vsr->wan);
    double per_token_block = (int8 ? 0.165 : 0.291) / 12240.0;
    out->estimated_seconds_per_chunk = per_token_block * tokens * tiles * 30 +
                                       1.5 * tiles * ((double)(uint32_t)plan.tile_w * (double)(uint32_t)plan.tile_h) / (1920.0 * 1088.0);
    out->supported = 1;
    return 1;
}

/* ---------------------------------------------------------------- */
/* Session */

static layout_slot *session_layouts(h3_vsr_session *s, int frames) {
    for (size_t i = 0; i < sizeof(s->layouts) / sizeof(*s->layouts); i++)
        if (s->layouts[i].frames == frames) return &s->layouts[i];
    for (size_t i = 0; i < sizeof(s->layouts) / sizeof(*s->layouts); i++) {
        layout_slot *slot = &s->layouts[i];
        if (slot->frames) continue;
        int grid_h = s->tile_h / H3_VSR_TOKEN_PIXELS, grid_w = s->tile_w / H3_VSR_TOKEN_PIXELS;
        if (!h3_vsr_window_layout_build(frames, grid_h, grid_w, H3_VSR_WINDOW, H3_VSR_WINDOW, 0, &slot->even) ||
            !h3_vsr_window_layout_build(frames, grid_h, grid_w, H3_VSR_WINDOW, H3_VSR_WINDOW, 1, &slot->odd)) {
            h3_vsr_window_layout_free(&slot->even);
            h3_vsr_window_layout_free(&slot->odd);
            return NULL;
        }
        slot->frames = frames;
        return slot;
    }
    return NULL;
}

void h3_vsr_session_close(h3_vsr_session *s) {
    if (!s) return;
    if (s->pending.valid) {
        char ignored[64];
        h3_vsr_reae_finish(s->vsr->reae_b, ignored, sizeof(ignored));
        h3_gpu_tensor_free(s->pending.own);
        h3_gpu_tensor_free(s->pending.decoded);
        s->pending.valid = 0;
    }
    for (int i = 0; s->tiles && i < s->tile_count; i++) {
        session_tile *tile = &s->tiles[i];
        h3_vsr_reae_state_free(tile->enc_state);
        h3_vsr_reae_state_free(tile->dec_state);
        h3_gpu_tensor_free(tile->prev_lq);
        h3_gpu_tensor_free(tile->weights);
    }
    free(s->tiles);
    for (size_t i = 0; i < sizeof(s->layouts) / sizeof(*s->layouts); i++)
        if (s->layouts[i].frames) {
            h3_vsr_window_layout_free(&s->layouts[i].even);
            h3_vsr_window_layout_free(&s->layouts[i].odd);
        }
    free(s);
}

h3_vsr_session *h3_vsr_session_open(h3_vsr *vsr, int in_w, int in_h,
                                    int out_w, int out_h, int total_frames,
                                    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    (void)total_frames;   /* the chunk protocol needs no total: see process() */
    if (!vsr) return NULL;
    h3_vsr_requirements req;
    if (!h3_vsr_check_requirements(vsr, in_w, in_h, out_w, out_h, &req) || !req.supported) {
        fail(error, error_size, "%s", req.reason[0] ? req.reason : "unsupported sizes");
        return NULL;
    }
    h3_vsr_session *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->vsr = vsr;
    s->in_w = in_w; s->in_h = in_h; s->out_w = out_w; s->out_h = out_h;
    s->full_w = round_up(out_w, H3_VSR_TOKEN_PIXELS);
    s->full_h = round_up(out_h, H3_VSR_TOKEN_PIXELS);
    if (!h3_vsr_tile_plan_build(s->full_w, s->full_h, req.tile_columns, req.tile_rows, tile_overlap(), &s->plan)) {
        fail(error, error_size, "cannot lay %d x %d tiles over %d x %d", req.tile_columns, req.tile_rows, s->full_w, s->full_h);
        h3_vsr_session_close(s);
        return NULL;
    }
    s->tile_w = s->plan.tile_w;
    s->tile_h = s->plan.tile_h;
    s->latent_h = s->tile_h / H3_VSR_SPATIAL;
    s->latent_w = s->tile_w / H3_VSR_SPATIAL;
    s->tile_count = s->plan.columns * s->plan.rows;
    s->sync_decode = s->tile_count > 1;
    s->tiles = calloc((size_t)s->tile_count, sizeof(session_tile));
    if (!s->tiles) { h3_vsr_session_close(s); return NULL; }
    float *weights = s->tile_count > 1 ? malloc((size_t)s->tile_h * s->tile_w * sizeof(float)) : NULL;
    for (int i = 0; i < s->tile_count; i++) {
        session_tile *tile = &s->tiles[i];
        int column = i % s->plan.columns, row = i / s->plan.columns;
        tile->x = s->plan.x[column];
        tile->y = s->plan.y[row];
        tile->crop_w = s->tile_w < out_w - tile->x ? s->tile_w : out_w - tile->x;
        tile->crop_h = s->tile_h < out_h - tile->y ? s->tile_h : out_h - tile->y;
        tile->enc_state = h3_vsr_reae_state_new(vsr->reae, 1, s->tile_h, s->tile_w);
        tile->dec_state = h3_vsr_reae_state_new(vsr->reae_b && !s->sync_decode ? vsr->reae_b : vsr->reae, 0,
                                                s->tile_h, s->tile_w);
        if (!tile->enc_state || !tile->dec_state) {
            fail(error, error_size, "cannot allocate the ReAE state");
            free(weights);
            h3_vsr_session_close(s);
            return NULL;
        }
        if (weights) {
            h3_vsr_tile_weights(&s->plan, column, row, weights);
            tile->weights = h3_gpu_tensor_from_f32(vsr->gpu, weights, (size_t)s->tile_h * s->tile_w);
            if (!tile->weights) {
                fail(error, error_size, "cannot allocate the tile weights");
                free(weights);
                h3_vsr_session_close(s);
                return NULL;
            }
        }
    }
    free(weights);
    if (getenv("H3_VSR_TRACE_STAGES") && s->tile_count > 1)
        fprintf(stderr, "vsr tiles: %d x %d of %d x %d over %d x %d (overlap %d)\n", s->plan.columns, s->plan.rows,
                s->tile_w, s->tile_h, s->full_w, s->full_h, s->plan.overlap);
    trace_live(vsr->gpu, "session opened");
    return s;
}

int h3_vsr_session_next_chunk_frames(const h3_vsr_session *s) {
    if (!s || s->done) return 0;
    return s->chunk_index == 0 ? H3_VSR_CLIP_FRAMES + 4 : H3_VSR_CLIP_FRAMES;
}

int h3_vsr_session_pending_frames(const h3_vsr_session *s) {
    return s ? s->real_in - s->real_out : 0;
}

#define STEP(expression, what) do { \
    if (!(expression)) { \
        if (error && error_size && !error[0]) \
            fail(error, error_size, "%s: %s", what, h3_gpu_error(gpu)); \
        goto failed; \
    } \
} while (0)

/* The DiT's per-block callback: the caller's progress over all tiles of
 * the chunk ("vsr dit", tile * blocks + block, tiles * blocks) and the
 * cancel flag, so a 中止 lands within a block rather than after the DiT. */
typedef struct {
    h3_progress_callback on_progress;
    void *opaque;
    const volatile int *cancel_requested;
    int tile, tiles;
} block_watch;

static int block_watch_callback(const char *phase, int done, int total, void *opaque) {
    block_watch *watch = opaque;
    if (watch->cancel_requested && *watch->cancel_requested) return 1;
    if (watch->on_progress &&
        watch->on_progress(phase, watch->tile * total + done, watch->tiles * total, watch->opaque)) return 1;
    return 0;
}

/* Development hook: with H3_VSR_DUMP_DIR set, every chunk's latents and
 * the first chunk's frames are written there as raw F32
 * (tests/test_real_vsr.c compares them with the official SwiftVR dump).
 * Off in the app; one tile only. */
static void dump_f32(int chunk, const char *stage, h3_gpu_tensor *tensor, size_t elements) {
    const char *dir = getenv("H3_VSR_DUMP_DIR");
    if (!dir || !dir[0] || !tensor) return;
    float *host = malloc(elements * sizeof(float));
    if (!host) return;
    if (h3_gpu_tensor_read_f32(tensor, host, elements)) {
        char path[4096];
        snprintf(path, sizeof(path), "%s/chunk%d_%s.f32", dir, chunk, stage);
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(host, sizeof(float), elements, f); fclose(f); }
    }
    free(host);
}

/* Crops `count` frames (from `first`) of the F32 [.., height, width, 3]
 * `decoded` to the target size as 8-bit pixels and copies them into
 * out[*written ..]. `gpu` is the context the frames are on. */
static int emit_frames(h3_vsr_session *s, h3_gpu *gpu, h3_gpu_tensor *decoded, int width, int height,
                       int first, int count, h3_vsr_image_out *out, int *written,
                       char *error, size_t error_size) {
    h3_gpu_tensor *out_pixels = NULL;
    int8_t *out_bytes = NULL;
    if (count <= 0) return 1;
    int out_bpp = out[*written].format == H3_VSR_BGRA8 ? 4 : 3;
    size_t out_stride = (size_t)s->out_w * out_bpp;
    out_pixels = h3_gpu_tensor_new_i8(gpu, (size_t)count * s->out_h * out_stride);
    if (!out_pixels) { fail(error, error_size, "vsr: out of memory"); goto failed; }
    STEP(h3_gpu_begin(gpu), "vsr begin");
    STEP(h3_gpu_vsr_f32_to_pixels(gpu, out_pixels, decoded, (uint32_t)first, (uint32_t)count, (uint32_t)width,
                                  (uint32_t)height, (uint32_t)s->out_w, (uint32_t)s->out_h,
                                  (uint32_t)out_stride, out_bpp == 4, 1), "vsr pixels");
    STEP(h3_gpu_submit(gpu), "vsr submit");
    out_bytes = malloc((size_t)count * s->out_h * out_stride);
    if (!out_bytes) { fail(error, error_size, "vsr: out of memory"); goto failed; }
    STEP(h3_gpu_tensor_read_i8(out_pixels, out_bytes, (size_t)count * s->out_h * out_stride),
         "vsr read pixels");
    for (int f = 0; f < count; f++) {
        h3_vsr_image_out *image = &out[*written + f];
        if (image->width != s->out_w || image->height != s->out_h ||
            (image->format == H3_VSR_BGRA8 ? 4 : 3) != out_bpp ||
            image->stride_bytes < (int)out_stride) {
            fail(error, error_size, "vsr: output frame %d has the wrong size or format", *written + f);
            goto failed;
        }
        for (int y = 0; y < s->out_h; y++)
            memcpy(image->pixels + (size_t)y * image->stride_bytes,
                   out_bytes + ((size_t)f * s->out_h + y) * out_stride, out_stride);
    }
    *written += count;
    free(out_bytes);
    h3_gpu_tensor_free(out_pixels);
    return 1;
failed:
    free(out_bytes);
    h3_gpu_tensor_free(out_pixels);
    return 0;
}

/* Shifts `decoded`'s channel means (over its real crop_h x crop_w part) to
 * `input_means` (the chunk's input frames): the model's output comes out
 * about 3/255 brighter than its input, uniformly, and this takes it back. */
static int color_match(h3_gpu *gpu, h3_gpu_tensor *decoded, int frames, int width, int height,
                       int crop_w, int crop_h, const float input_means[3], char *error, size_t error_size) {
    float out_means[3], offsets[3];
    if (!h3_gpu_vsr_channel_means_f32(gpu, decoded, (uint32_t)frames, (uint32_t)height, (uint32_t)width,
                                      (uint32_t)crop_h, (uint32_t)crop_w, out_means)) {
        fail(error, error_size, "vsr colour match: %s", h3_gpu_error(gpu));
        return 0;
    }
    for (int c = 0; c < 3; c++) offsets[c] = input_means[c] - out_means[c];
    if (getenv("H3_VSR_TRACE_STAGES"))
        fprintf(stderr, "vsr colour match: offsets %+.4f %+.4f %+.4f (x255: %+.1f %+.1f %+.1f)\n",
                offsets[0], offsets[1], offsets[2], offsets[0] * 255, offsets[1] * 255, offsets[2] * 255);
    if (!h3_gpu_begin(gpu) ||
        !h3_gpu_vsr_add_channel_offsets_f32(gpu, decoded, (uint32_t)((size_t)frames * height * width), offsets) ||
        !h3_gpu_submit(gpu)) {
        fail(error, error_size, "vsr colour match: %s", h3_gpu_error(gpu));
        return 0;
    }
    return 1;
}

/* Waits for the chunk decoding on the second context and writes its
 * frames. No-op without one. */
static int finish_pending(h3_vsr_session *s, h3_vsr_image_out *out, int out_capacity, int *written,
                          h3_progress_callback on_progress, void *opaque, char *error, size_t error_size) {
    if (!s->pending.valid) return 1;
    h3_vsr *vsr = s->vsr;
    size_t fplane = (size_t)s->tile_h * s->tile_w * 3;
    int ok = h3_vsr_reae_finish(vsr->reae_b, error, error_size);
    if (ok && vsr->params.color_match >= 0)
        ok = color_match(vsr->gpu_b, s->pending.decoded, s->pending.latents * 4, s->tile_w, s->tile_h,
                         s->tiles[0].crop_w, s->tiles[0].crop_h, s->pending.input_means, error, error_size);
    if (ok && *written + s->pending.count > out_capacity) {
        fail(error, error_size, "vsr: %d output frames do not fit in %d", *written + s->pending.count, out_capacity);
        ok = 0;
    }
    if (ok && s->pending.chunk_index == 0)
        dump_f32(0, "decoded", s->pending.decoded, (size_t)s->pending.latents * 4 * fplane);
    if (ok) ok = emit_frames(s, vsr->gpu_b, s->pending.decoded, s->tile_w, s->tile_h, s->pending.skip,
                             s->pending.count, out, written, error, error_size);
    h3_gpu_tensor_free(s->pending.own);
    h3_gpu_tensor_free(s->pending.decoded);
    memset(&s->pending, 0, sizeof(s->pending));
    if (ok && on_progress && on_progress("vsr decode", 1, 1, opaque)) {
        fail(error, error_size, "vsr cancelled");
        return 0;
    }
    return ok;
}

/* Runs one chunk: `frames` padded input frames taken from `in` (real
 * frames 0..in_count-1, the last one repeated past them), tile by tile,
 * and writes its output frames to `out` from *written on. */
static int run_chunk(h3_vsr_session *s, const h3_vsr_chunk *chunk,
                     const h3_vsr_image *in, int in_count, int in_offset,
                     h3_vsr_image_out *out, int out_capacity, int *written,
                     h3_progress_callback on_progress, void *opaque,
                     const volatile int *cancel_requested, char *error, size_t error_size) {
    h3_vsr *vsr = s->vsr;
    h3_gpu *gpu = vsr->gpu;
    const int C = H3_VSR_LATENT_CHANNELS;
    int tail = chunk->type == H3_VSR_CHUNK_LAST ? H3_VSR_FRAMES_TO_TRIM : 0;
    int enc_frames = chunk->frames + tail;            /* a multiple of 4 */
    int latents = enc_frames / H3_VSR_TEMPORAL;
    int prefix = chunk->dit_prefix_latents;
    int dit_latents = prefix + latents;
    size_t lplane = (size_t)s->latent_h * s->latent_w * C;
    size_t fplane = (size_t)s->tile_h * s->tile_w * 3;
    int bpp = in[0].format == H3_VSR_BGRA8 ? 4 : 3;
    size_t in_stride = (size_t)s->in_w * bpp;
    double mark = trace_now();
    h3_gpu_tensor *pixels = NULL, *frames = NULL, *latent = NULL, *z_lq = NULL, *z_hq = NULL;
    h3_gpu_tensor *decoded = NULL, *rope_cos = NULL, *rope_sin = NULL, *own = NULL, *composite = NULL;
    int8_t *packed = NULL;
    float *cs = NULL, *sn = NULL;
    int decoded_frames = latents * H3_VSR_TEMPORAL;
    int skip = chunk->type == H3_VSR_CHUNK_FIRST ? H3_VSR_FRAMES_TO_TRIM : 0;
    int count = chunk->output_stop - chunk->output_start;
    int deferred = vsr->reae_b && !s->sync_decode;    /* decode under the next chunk's DiT */
    int one_tile = s->tile_count == 1;
    if (count < 0) count = 0;
    if (count > decoded_frames - skip) count = decoded_frames - skip;
    if (cancel_requested && *cancel_requested) {
        fail(error, error_size, "vsr cancelled");
        return 0;
    }
    /* What this call writes: the pending chunk's frames and, when the
     * decode is not deferred (one context, or a LAST chunk), this one's. */
    int will_write = (s->pending.valid ? s->pending.count : 0) +
                     (deferred && chunk->type != H3_VSR_CHUNK_LAST ? 0 : count);
    if (*written + will_write > out_capacity) {
        fail(error, error_size, "vsr: %d output frames do not fit in %d", *written + will_write, out_capacity);
        return 0;
    }

    /* 1. Pack the chunk's 8-bit frames (last real frame repeated): every
     * tile is resized from them. */
    packed = malloc((size_t)enc_frames * s->in_h * in_stride);
    if (!packed) goto oom;
    for (int f = 0; f < enc_frames; f++) {
        int source = in_offset + f;
        if (source > in_count - 1) source = in_count - 1;
        const h3_vsr_image *image = &in[source];
        for (int y = 0; y < s->in_h; y++)
            memcpy(packed + ((size_t)f * s->in_h + y) * in_stride,
                   image->pixels + (size_t)y * image->stride_bytes, in_stride);
    }
    pixels = h3_gpu_tensor_from_i8(gpu, packed, (size_t)enc_frames * s->in_h * in_stride);
    if (!pixels) goto oom;
    free(packed);
    packed = NULL;
    /* RoPE tables and window layouts for this latent count (every tile has
     * the same token grid). */
    int grid_h = s->tile_h / H3_VSR_TOKEN_PIXELS, grid_w = s->tile_w / H3_VSR_TOKEN_PIXELS;
    size_t tokens = (size_t)dit_latents * grid_h * grid_w, pairs = H3_VSR_HEAD_DIM / 2;
    cs = malloc(tokens * pairs * sizeof(float));
    sn = malloc(tokens * pairs * sizeof(float));
    if (!cs || !sn) goto oom;
    h3_vsr_rope_table(dit_latents, grid_h, grid_w, chunk->rope_offset, H3_VSR_HEAD_DIM, cs, sn);
    rope_cos = h3_gpu_tensor_from_f32(gpu, cs, tokens * pairs);
    rope_sin = h3_gpu_tensor_from_f32(gpu, sn, tokens * pairs);
    if (!rope_cos || !rope_sin) goto oom;
    free(cs); free(sn);
    cs = sn = NULL;
    layout_slot *slot = NULL;
    if (!vsr->params.full_attention) {
        slot = session_layouts(s, dit_latents);
        if (!slot) {
            fail(error, error_size, "vsr: cannot build the window layout for %d latents", dit_latents);
            goto failed;
        }
    }
    /* With several tiles, their decoded frames are blended into this F32
     * frame of the chunk's output (count frames at the target size). */
    if (!one_tile && count > 0) {
        composite = h3_gpu_tensor_new_f32(gpu, (size_t)count * s->out_h * s->out_w * 3);
        if (!composite) goto oom;
        STEP(h3_gpu_begin(gpu), "vsr begin");
        STEP(h3_gpu_zero_f32(gpu, composite, (size_t)count * s->out_h * s->out_w * 3), "vsr composite zero");
        STEP(h3_gpu_submit(gpu), "vsr submit");
    }
    trace_live(gpu, "chunk tensors new");

    for (int t = 0; t < s->tile_count; t++) {
        session_tile *tile = &s->tiles[t];
        if (cancel_requested && *cancel_requested) goto cancelled;
        /* 2. Resize the tile from the 8-bit frames and encode it. */
        frames = h3_gpu_tensor_new_f32(gpu, (size_t)enc_frames * fplane);
        latent = h3_gpu_tensor_new_f32(gpu, (size_t)latents * lplane);
        if (!frames || !latent) goto oom;
        STEP(h3_gpu_begin(gpu), "vsr begin");
        STEP(h3_gpu_vsr_pixels_to_f32(gpu, frames, pixels, (uint32_t)enc_frames, (uint32_t)s->in_w,
                                      (uint32_t)s->in_h, (uint32_t)in_stride, bpp == 4,
                                      (uint32_t)s->out_w, (uint32_t)s->out_h, (uint32_t)tile->x,
                                      (uint32_t)tile->y, (uint32_t)s->tile_w, (uint32_t)s->tile_h), "vsr resize");
        STEP(h3_gpu_submit(gpu), "vsr resize submit");
        trace_stage(&mark, "pack + resize");
        float input_means[3] = {0, 0, 0};
        if (vsr->params.color_match >= 0 &&
            !h3_gpu_vsr_channel_means_f32(gpu, frames, (uint32_t)enc_frames, (uint32_t)s->tile_h, (uint32_t)s->tile_w,
                                          (uint32_t)tile->crop_h, (uint32_t)tile->crop_w, input_means)) {
            fail(error, error_size, "vsr colour match: %s", h3_gpu_error(gpu));
            goto failed;
        }
        if (one_tile && chunk->index == 0) dump_f32(chunk->index, "frames", frames, (size_t)enc_frames * fplane);
        trace_live(gpu, "chunk buffers allocated");
        if (on_progress && on_progress("vsr encode", 0, 1, opaque)) goto cancelled;
        if (!h3_vsr_reae_encode(vsr->reae, tile->enc_state, frames, enc_frames, s->tile_h, s->tile_w,
                                vsr->params.reae_frame_batch, latent, error, error_size))
            goto failed;
        h3_gpu_tensor_free(frames);
        frames = NULL;
        if (one_tile) dump_f32(chunk->index, "latent", latent, (size_t)latents * lplane);
        trace_live(gpu, "after encode");
        trace_stage(&mark, "reae encode");
        if (on_progress && on_progress("vsr encode", 1, 1, opaque)) goto cancelled;

        /* 3. DiT input: the tile's previous chunk's last `prefix` LQ latents
         * (zeros when there is none) then this chunk's. */
        z_lq = h3_gpu_tensor_new_f32(gpu, (size_t)dit_latents * lplane);
        z_hq = h3_gpu_tensor_new_f32(gpu, (size_t)dit_latents * lplane);
        if (!z_lq || !z_hq) goto oom;
        STEP(h3_gpu_begin(gpu), "vsr begin");
        if (prefix) {
            if (tile->prev_lq && tile->prev_count >= prefix) {
                STEP(h3_gpu_copy_f32(gpu, z_lq, 0, tile->prev_lq, (size_t)(tile->prev_count - prefix) * lplane,
                                     (size_t)prefix * lplane), "vsr prefix copy");
            } else {
                float *zeros = calloc((size_t)prefix * lplane, sizeof(float));
                int ok = zeros && h3_gpu_tensor_write_f32_range(z_lq, 0, zeros, (size_t)prefix * lplane);
                free(zeros);
                STEP(ok, "vsr prefix zeros");
            }
        }
        STEP(h3_gpu_copy_f32(gpu, z_lq, (size_t)prefix * lplane, latent, 0, (size_t)latents * lplane),
             "vsr latent copy");
        STEP(h3_gpu_submit(gpu), "vsr submit");
        h3_gpu_tensor_free(latent);
        latent = NULL;
        {
            block_watch watch = { on_progress, opaque, cancel_requested, t, s->tile_count };
            if (!h3_vsr_wan_predict(vsr->wan, z_lq, dit_latents, s->latent_h, s->latent_w,
                                    slot ? &slot->even : NULL, slot ? &slot->odd : NULL,
                                    rope_cos, rope_sin, z_hq, block_watch_callback, &watch,
                                    error, error_size))
                goto failed;
        }
        trace_live(gpu, "after dit");
        trace_stage(&mark, "dit (gpu complete)");
        if (one_tile) {
            dump_f32(chunk->index, "z_lq", z_lq, (size_t)dit_latents * lplane);
            dump_f32(chunk->index, "z_hq", z_hq, (size_t)dit_latents * lplane);
        }
        /* Remember the LQ latents for the next chunk's prefix. */
        {
            int keep = dit_latents < vsr->clip_latents + 1 ? dit_latents : vsr->clip_latents + 1;
            h3_gpu_tensor *kept = h3_gpu_tensor_new_f32(gpu, (size_t)keep * lplane);
            if (!kept) goto oom;
            STEP(h3_gpu_begin(gpu), "vsr begin");
            STEP(h3_gpu_copy_f32(gpu, kept, 0, z_lq, (size_t)(dit_latents - keep) * lplane,
                                 (size_t)keep * lplane), "vsr keep latents");
            STEP(h3_gpu_submit(gpu), "vsr submit");
            h3_gpu_tensor_free(tile->prev_lq);
            tile->prev_lq = kept;
            tile->prev_count = keep;
        }
        h3_gpu_tensor_free(z_lq);
        z_lq = NULL;

        /* 4. This chunk's latents for the decoder (the prefix is dropped). */
        h3_gpu *dgpu = deferred ? vsr->gpu_b : gpu;
        own = h3_gpu_tensor_new_f32(dgpu, (size_t)latents * lplane);
        if (!own) goto oom;
        STEP(h3_gpu_begin(gpu), "vsr begin");
        STEP(h3_gpu_copy_f32(gpu, own, 0, z_hq, (size_t)prefix * lplane, (size_t)latents * lplane),
             "vsr z_hq copy");
        STEP(h3_gpu_submit(gpu), "vsr submit");
        h3_gpu_tensor_free(z_hq);
        z_hq = NULL;
        /* 5. The previous chunk's frames: decoded on the second context
         * while this chunk's DiT ran. */
        if (!finish_pending(s, out, out_capacity, written, on_progress, opaque, error, error_size)) goto failed;
        /* 6. Decode: deferred onto the second context, or right now. */
        decoded = h3_gpu_tensor_new_f32(dgpu, (size_t)decoded_frames * fplane);
        if (!decoded) goto oom;
        if (deferred) {
            if (!h3_vsr_reae_decode_begin(vsr->reae_b, tile->dec_state, own, latents, s->latent_h, s->latent_w,
                                          vsr->params.reae_frame_batch > 0 ? 1 : 0, decoded, error, error_size))
                goto failed;
            s->pending.valid = 1;
            s->pending.chunk_index = chunk->index;
            s->pending.count = count;
            s->pending.skip = skip;
            s->pending.latents = latents;
            s->pending.own = own;
            s->pending.decoded = decoded;
            memcpy(s->pending.input_means, input_means, sizeof(input_means));
            own = decoded = NULL;
            trace_stage(&mark, "reae decode (launched)");
            if (chunk->type == H3_VSR_CHUNK_LAST &&
                !finish_pending(s, out, out_capacity, written, on_progress, opaque, error, error_size))
                goto failed;
        } else {
            if (!h3_vsr_reae_decode(vsr->reae, tile->dec_state, own, latents, s->latent_h, s->latent_w,
                                    vsr->params.reae_frame_batch > 0 ? 1 : 0, decoded, error, error_size))
                goto failed;
            h3_gpu_tensor_free(own);
            own = NULL;
            trace_live(gpu, "after decode");
            trace_stage(&mark, "reae decode");
            if (vsr->params.color_match >= 0 &&
                !color_match(gpu, decoded, decoded_frames, s->tile_w, s->tile_h, tile->crop_w, tile->crop_h,
                             input_means, error, error_size)) goto failed;
            if (one_tile) {
                if (on_progress && on_progress("vsr decode", 1, 1, opaque)) goto cancelled;
                if (chunk->index == 0) dump_f32(chunk->index, "decoded", decoded, (size_t)decoded_frames * fplane);
                if (!emit_frames(s, gpu, decoded, s->tile_w, s->tile_h, skip, count, out, written, error, error_size))
                    goto failed;
                trace_stage(&mark, "output pixels");
            } else if (count > 0) {
                STEP(h3_gpu_begin(gpu), "vsr begin");
                STEP(h3_gpu_vsr_tile_accumulate_f32(gpu, composite, (uint32_t)s->out_w, (uint32_t)s->out_h, decoded,
                                                    (uint32_t)skip, (uint32_t)count, (uint32_t)s->tile_w,
                                                    (uint32_t)s->tile_h, (uint32_t)tile->x, (uint32_t)tile->y,
                                                    tile->weights), "vsr tile blend");
                STEP(h3_gpu_submit(gpu), "vsr submit");
                trace_stage(&mark, "tile blend");
            }
            h3_gpu_tensor_free(decoded);
            decoded = NULL;
        }
    }
    if (composite) {
        if (on_progress && on_progress("vsr decode", 1, 1, opaque)) goto cancelled;
        if (!emit_frames(s, gpu, composite, s->out_w, s->out_h, 0, count, out, written, error, error_size))
            goto failed;
        trace_stage(&mark, "output pixels");
    }
    h3_gpu_tensor_free(pixels); h3_gpu_tensor_free(composite);
    h3_gpu_tensor_free(rope_cos); h3_gpu_tensor_free(rope_sin);
    return 1;
oom:
    fail(error, error_size, "vsr: out of memory");
    goto failed;
cancelled:
    fail(error, error_size, "vsr cancelled");
failed:
    free(packed); free(cs); free(sn);
    h3_gpu_tensor_free(pixels); h3_gpu_tensor_free(frames); h3_gpu_tensor_free(latent);
    h3_gpu_tensor_free(z_lq); h3_gpu_tensor_free(z_hq); h3_gpu_tensor_free(decoded); h3_gpu_tensor_free(own);
    h3_gpu_tensor_free(composite); h3_gpu_tensor_free(rope_cos); h3_gpu_tensor_free(rope_sin);
    return 0;
}

int h3_vsr_session_process(h3_vsr_session *s,
                           const h3_vsr_image *in, int in_frames, int is_last,
                           h3_vsr_image_out *out, int out_capacity, int *out_frames,
                           h3_progress_callback on_progress, void *opaque,
                           const volatile int *cancel_requested,
                           char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (out_frames) *out_frames = 0;
    if (!s || !in || in_frames < 1 || !out || !out_frames) {
        fail(error, error_size, "vsr: bad arguments");
        return 0;
    }
    if (s->done) {
        fail(error, error_size, "vsr: the session has already finished");
        return 0;
    }
    int expected = h3_vsr_session_next_chunk_frames(s);
    if (!is_last && in_frames != expected) {
        fail(error, error_size, "vsr: this chunk needs exactly %d frames (got %d)", expected, in_frames);
        return 0;
    }
    if (is_last && in_frames > expected) {
        fail(error, error_size, "vsr: the last chunk takes at most %d frames (got %d)", expected, in_frames);
        return 0;
    }
    for (int f = 0; f < in_frames; f++)
        if (in[f].width != s->in_w || in[f].height != s->in_h || in[f].format != in[0].format ||
            in[f].stride_bytes < s->in_w * (in[0].format == H3_VSR_BGRA8 ? 4 : 3) || !in[f].pixels) {
            fail(error, error_size, "vsr: input frame %d has the wrong size or format", f);
            return 0;
        }
    int real_total = s->real_in + in_frames;
    int written = 0;

    /* The chunks this call runs (h3_vsr_chunk.c's rules on the padded
     * frame count): FIRST / MIDDLE for a regular call; for the last call,
     * the remaining frames padded to 4b+1 - a MIDDLE of 24 before a LAST of
     * 1 when that padded count would exceed the clip. */
    h3_vsr_chunk chunks[2];
    int chunk_count = 0;
    int in_offsets[2] = {0, 0};
    if (!is_last) {
        h3_vsr_chunk *c = &chunks[chunk_count++];
        memset(c, 0, sizeof(*c));
        c->type = s->chunk_index == 0 ? H3_VSR_CHUNK_FIRST : H3_VSR_CHUNK_MIDDLE;
        c->index = s->chunk_index;
        c->start = s->start;
        c->frames = in_frames;
        c->latents = (in_frames - 1) / H3_VSR_TEMPORAL + 1;
        c->output_start = c->start - H3_VSR_FRAMES_TO_TRIM < 0 ? 0 : c->start - H3_VSR_FRAMES_TO_TRIM;
        c->output_stop = c->start + c->frames - H3_VSR_FRAMES_TO_TRIM;
        c->dit_prefix_latents = 0;
        c->rope_offset = c->start / H3_VSR_TEMPORAL;
    } else {
        int padded = h3_vsr_padded_frame_count(in_frames);
        int start = s->start;
        if (s->chunk_index > 0 && padded > H3_VSR_CLIP_FRAMES) {
            h3_vsr_chunk *c = &chunks[chunk_count++];
            memset(c, 0, sizeof(*c));
            c->type = H3_VSR_CHUNK_MIDDLE;
            c->index = s->chunk_index;
            c->start = start;
            c->frames = H3_VSR_CLIP_FRAMES;
            c->latents = H3_VSR_CLIP_FRAMES / H3_VSR_TEMPORAL;
            c->output_start = start - H3_VSR_FRAMES_TO_TRIM;
            c->output_stop = start + H3_VSR_CLIP_FRAMES - H3_VSR_FRAMES_TO_TRIM;
            if (c->output_stop > real_total) c->output_stop = real_total;
            c->rope_offset = start / H3_VSR_TEMPORAL;
            start += H3_VSR_CLIP_FRAMES;
            padded -= H3_VSR_CLIP_FRAMES;
            in_offsets[chunk_count] = H3_VSR_CLIP_FRAMES;
        }
        h3_vsr_chunk *c = &chunks[chunk_count++];
        memset(c, 0, sizeof(*c));
        c->type = H3_VSR_CHUNK_LAST;
        c->index = s->chunk_index + chunk_count - 1;
        c->start = start;
        c->frames = padded;
        c->latents = (padded - 1) / H3_VSR_TEMPORAL + 1;
        c->output_start = start - H3_VSR_FRAMES_TO_TRIM < 0 ? 0 : start - H3_VSR_FRAMES_TO_TRIM;
        c->output_stop = real_total;
        int clip_latents = s->vsr->clip_latents;
        int padding = clip_latents + 1 - c->latents;
        c->dit_prefix_latents = padding > 0 ? padding : 0;
        int offset = start / H3_VSR_TEMPORAL;
        c->rope_offset = offset - c->dit_prefix_latents < 0 ? 0 : offset - c->dit_prefix_latents;
    }
    for (int i = 0; i < chunk_count; i++) {
        if (!run_chunk(s, &chunks[i], in, in_frames, in_offsets[i], out, out_capacity,
                       &written, on_progress, opaque, cancel_requested, error, error_size)) {
            *out_frames = written;
            return 0;
        }
        s->chunk_index++;
        s->start += chunks[i].frames;
    }
    s->real_in = real_total;
    s->real_out += written;
    *out_frames = written;
    if (is_last) s->done = 1;
    return 1;
}
