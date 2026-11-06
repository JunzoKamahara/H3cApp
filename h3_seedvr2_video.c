/* SeedVR2 on clips (see h3_seedvr2.h): one window of 4n+1 frames through
 * the temporal VAE and the transformer's 3D windows, the reference's
 * inference loop for one window (inference_seedvr2_3b.py): resize, encode
 * in causal slices, z = noise - v at t = 1000 with the latent as the
 * condition, decode in causal slices, the colour transfer per frame. The
 * window session (overlap, tail padding) is built on this. */
#include "h3_seedvr2_internal.h"
#include "h3_metal.h"

#define CLIP_FAIL(...) do { fail(error, error_size, __VA_ARGS__); goto done; } while (0)

/* H3_SV2_TRACE: the stage time and the GPU's peak live bytes so far. */
static void trace_peak(h3_seedvr2 *m, double *mark, const char *label) {
    if (!trace_enabled()) return;
    h3_gpu_stats st;
    if (h3_gpu_get_stats(m->gpu, &st))
        fprintf(stderr, "sv2 %-24s %8.1f ms  live %.2f GiB, peak %.2f GiB (dit %.2f GiB, %s)\n", label, (now_seconds() - *mark) * 1000.0,
                (double)st.live_bytes / 1073741824.0, (double)st.peak_live_bytes / 1073741824.0,
                (double)m->dit_bytes / 1073741824.0, m->dit_loaded ? "loaded" : "unloaded");
    *mark = now_seconds();
}

static int clip_progress(h3_progress_callback on_progress, void *opaque, const volatile int *cancel,
                         const char *phase, int step, int steps) {
    if (cancel && *cancel) return 0;
    if (on_progress && on_progress(phase, step, steps, opaque)) return 0;
    return 1;
}

/* Peak live GPU memory of one window, from measurements on an M5 (24 GB,
 * 2026-11-06; the clip pipeline below, every large buffer freed as soon as
 * the GPU is done with it, the transformer's blocks freed around the VAE
 * phases, blocks 2-3 of a sliced decode streamed frame by frame; peak live
 * GiB -> GB, process footprint in brackets):
 *
 *   path  size      T  VAE     peak            phase that sets it
 *   grid  1344x768  5  whole    9.89 (12.6)    decode: block 3, T+2 frame slots
 *   grid  1344x768 17  sliced  16.72 (18.2)    decode: every conv's state + one frame
 *   grid  960x544  17  sliced   8.74 (12.9)
 *   F32   960x544   5  whole   11.21 (12.8)
 *   F32   960x544  17  sliced  13.03 (15.5)
 *   F32   768x432  17  sliced   8.64 (13.1)
 *
 * Per megapixel of output: decode whole 1.29 (grid) / 2.9 (F32) GB per
 * frame slot (T + 2), sliced 15.5 / 24 GB whatever T; encode whole 0.92 /
 * 1.93 per slot, sliced 12 / 20. The transformer phase is the resident
 * weights plus 113 KB per token. Before this policy a 17-frame 1344x768
 * run reached 36 GiB and swapped: never run an unmeasured configuration
 * without the estimate. */
static uint64_t phase_max(uint64_t a, uint64_t b) { return a > b ? a : b; }

static uint64_t video_peak_bytes(uint64_t resident, uint64_t dit_bytes, int nax_vae, int full_w, int full_h,
                                 int frames, int split, int unload_dit) {
    double mp = (double)full_w * full_h / 1e6;
    int L = (frames - 1) / 4 + 1;
    double tokens = (double)L * (full_w / 16) * (full_h / 16);
    double rgb = 24e6 * frames * mp, common = 0.1e9 + rgb;
    double vae_base = (unload_dit ? (double)(resident - (dit_bytes < resident ? dit_bytes : 0)) : (double)resident) + common;
    double enc = split > 0 ? (nax_vae ? 12.0e9 : 20.0e9) * mp : (nax_vae ? 0.92e9 : 1.93e9) * (frames + 2) * mp;
    double dec = split > 0 ? (nax_vae ? 15.5e9 : 24.0e9) * mp : (nax_vae ? 1.29e9 : 2.9e9) * (frames + 2) * mp;
    double dit = (double)resident + common + tokens * 113e3;
    return phase_max(phase_max((uint64_t)(vae_base + enc), (uint64_t)(vae_base + dec)), (uint64_t)dit);
}

uint64_t sv2_video_peak_bytes(uint64_t resident, uint64_t dit_bytes, int nax_vae, int full_w, int full_h,
                              int frames, int vae_split, int unload_dit) {
    return video_peak_bytes(resident, dit_bytes, nax_vae, full_w, full_h, frames, vae_split, unload_dit);
}

/* 90 % of the GPU's working set (the device's memory when it names none). */
static uint64_t video_budget(void) {
    h3_device_info device;
    char probe[256];
    uint64_t budget = 0;
    if (h3_metal_probe(&device, probe, sizeof(probe)))
        budget = device.recommended_working_set ? device.recommended_working_set : device.physical_memory;
    return budget * 9 / 10;
}

int h3_seedvr2_video_check_requirements(const h3_seedvr2 *m, int in_w, int in_h, int out_w, int out_h,
                                        int window_frames, h3_seedvr2_video_requirements *out) {
    if (!m || !out) return 0;
    memset(out, 0, sizeof(*out));
    if (in_w < 16 || in_h < 16) { snprintf(out->reason, sizeof(out->reason), "the input is too small"); return 1; }
    if (out_w < 64 || out_h < 64) { snprintf(out->reason, sizeof(out->reason), "the output must be at least 64 pixels a side"); return 1; }
    if (out_w > 4096 || out_h > 4096) { snprintf(out->reason, sizeof(out->reason), "the output must be at most 4096 pixels a side"); return 1; }
    if (window_frames < 1 || window_frames % 4 != 1 || window_frames > 121) {
        snprintf(out->reason, sizeof(out->reason), "the window must be 4n+1 frames (5, 9, 13, ...)");
        return 1;
    }
    int full_w = round_up(out_w, SV2_PIXEL_MULTIPLE), full_h = round_up(out_h, SV2_PIXEL_MULTIPLE);
    out->latent_frames = (window_frames - 1) / 4 + 1;
    out->tokens = (full_w / 16) * (full_h / 16) * out->latent_frames;
    uint64_t whole = video_peak_bytes(m->info.resident_bytes, m->dit_bytes, m->nax_vae, full_w, full_h, window_frames, 0, 1);
    uint64_t sliced = video_peak_bytes(m->info.resident_bytes, m->dit_bytes, m->nax_vae, full_w, full_h, window_frames, 4, 1);
    uint64_t budget = video_budget();
    /* the cheaper form that fits; both give the same output */
    int split = whole <= sliced ? 0 : 4;
    uint64_t need = whole <= sliced ? whole : sliced;
    if (budget && need > budget && split == 0 && sliced <= budget) { split = 4; need = sliced; }
    out->vae_split = split;
    out->gpu_bytes = need;
    if (budget && need > budget) {
        snprintf(out->reason, sizeof(out->reason),
                 "about %.1f GB of GPU memory is needed for %d frames at %dx%d, this Mac has %.1f GB",
                 (double)need / 1e9, window_frames, out_w, out_h, (double)budget / 0.9 / 1e9);
        return 1;
    }
    /* measured on an M5 (the table above): grid 5.3 s/frame at 1344x768 (T=5),
     * 2.6 s/frame at 960x544 (T=17); F32 5.8 s/frame at 960x544 (T=5) on the
     * same GPU, and other Apple GPUs are slower still */
    double mp = (double)full_w * full_h / 1e6;
    double per_frame = m->nax_vae ? 5.0 : 11.0;
    out->estimated_seconds_per_frame = mp * per_frame * (window_frames <= 5 ? 1.0 : 0.8);
    out->supported = 1;
    return 1;
}

int h3_seedvr2_restore_clip(h3_seedvr2 *m, const h3_vsr_image *in, int frames, int out_w, int out_h,
                            h3_vsr_image_out *out, const h3_seedvr2_clip_params *params,
                            h3_progress_callback on_progress, void *opaque,
                            const volatile int *cancel, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!m || !in || !out || frames < 1 || frames % 4 != 1 || out_w < 16 || out_h < 16) {
        fail(error, error_size, "SeedVR2 clip: %d frames (4n+1 needed) or bad sizes", frames);
        return 0;
    }
    h3_seedvr2_clip_params p = {4, 0, 0, -1, NULL, -1};
    if (params) p = *params;
    int full_w = round_up(out_w, SV2_PIXEL_MULTIPLE), full_h = round_up(out_h, SV2_PIXEL_MULTIPLE);
    if (p.vae_split < 0) {   /* the cheaper form (the same output either way) */
        uint64_t whole = video_peak_bytes(m->info.resident_bytes, m->dit_bytes, m->nax_vae, full_w, full_h, frames, 0, 1);
        uint64_t sliced = video_peak_bytes(m->info.resident_bytes, m->dit_bytes, m->nax_vae, full_w, full_h, frames, 4, 1);
        p.vae_split = whole <= sliced ? 0 : 4;
    }
    uint64_t seed = p.seed ? p.seed : m->params.seed;
    int color = p.color_transfer < 0 ? m->params.color_transfer : p.color_transfer;
    int lw = full_w / 8, lh = full_h / 8, L = (frames - 1) / 4 + 1;
    int N = (lw / 2) * (lh / 2) * L;
    size_t plane = (size_t)full_w * full_h * 3, lplane = (size_t)lw * lh, latent = (size_t)SV2_LATENT * L * lplane;
    for (int t = 0; t < frames; t++) {
        if (!in[t].pixels || !out[t].pixels || in[t].width != in[0].width || in[t].height != in[0].height ||
            out[t].width != out_w || out[t].height != out_h) {
            fail(error, error_size, "SeedVR2 clip: frame %d has another size", t);
            return 0;
        }
    }
    if (!sv2_vae_set_temporal(m, 1, error, error_size)) return 0;
    sv2_pool_drain(m);
    m->pool_bypass = 1;   /* exact live memory: clip_sync in the VAE makes the frees real */
    /* the transformer's 6 GB are idle during the VAE: free them when the
     * window would otherwise take more than 70 % of the GPU's memory */
    uint64_t budget = video_budget();
    int unload_dit = budget && video_peak_bytes(m->info.resident_bytes, m->dit_bytes, m->nax_vae, full_w, full_h, frames,
                                                p.vae_split, 0) > budget * 7 / 10;
    if (unload_dit) sv2_dit_unload(m);
    sv2_pool_drain(m);

    sv2_layout layouts[2];
    memset(layouts, 0, sizeof(layouts));
    h3_gpu_tensor *rgb_in = NULL, *frame = NULL, *style = NULL, *pixels = NULL, *tokens = NULL, *pred = NULL;
    act enc = {0}, dec = {0}, zt = {0};
    float *cond = malloc(latent * sizeof(float)), *noise = malloc(latent * sizeof(float));
    float *pred_host = malloc(latent * sizeof(float)), *z = malloc(latent * sizeof(float));
    float *enc_host = malloc(2 * latent * sizeof(float));
    uint16_t *tokens_host = malloc((size_t)N * SV2_PATCH_IN * sizeof(uint16_t));
    uint16_t *pred_bf16 = malloc((size_t)N * SV2_PATCH_OUT * sizeof(uint16_t));
    int ok = 0;
    double mark = now_seconds();
    if (!cond || !noise || !pred_host || !z || !enc_host || !tokens_host || !pred_bf16) CLIP_FAIL("SeedVR2 clip: out of memory");

    /* resize every frame into F32 [T, full_h, full_w, 3] in [-1, 1] (black padding) */
    rgb_in = h3_gpu_tensor_new_f32(m->gpu, (size_t)frames * plane);
    frame = sv2_pool_take(m, plane, 0);
    if (!rgb_in || !frame) CLIP_FAIL("SeedVR2 clip: out of GPU memory");
    for (int t = 0; t < frames; t++) {
        size_t in_bytes = (size_t)in[t].stride_bytes * in[t].height;
        pixels = h3_gpu_tensor_from_i8(m->gpu, (const int8_t *)in[t].pixels, in_bytes);
        if (!pixels) CLIP_FAIL("SeedVR2 clip: out of GPU memory");
        if (!h3_gpu_begin(m->gpu) ||
            !h3_gpu_sv2_bicubic_pixels_f32(m->gpu, frame, pixels, (uint32_t)in[t].width, (uint32_t)in[t].height,
                                           (uint32_t)in[t].stride_bytes, in[t].format == H3_VSR_BGRA8,
                                           (uint32_t)out_w, (uint32_t)out_h, (uint32_t)full_w, (uint32_t)full_h, -1.0f) ||
            !h3_gpu_copy_f32(m->gpu, rgb_in, (size_t)t * plane, frame, 0, plane) ||
            !h3_gpu_submit(m->gpu)) CLIP_FAIL("SeedVR2 resize: %s", h3_gpu_error(m->gpu));
        h3_gpu_tensor_free(pixels);
        pixels = NULL;
    }
    trace_peak(m, &mark, "resize (clip)");

    /* encode (sliced): [T, H, W, 3] -> [L, lh, lw, 32], the mean scaled as the condition */
    {
        act rgb = {rgb_in, full_h, full_w, 3, frames};
        if (!h3_gpu_begin(m->gpu)) CLIP_FAIL("SeedVR2: %s", h3_gpu_error(m->gpu));
        if (!sv2_vae_encode_video(m, &rgb, &enc, p.vae_split, on_progress, opaque, cancel, error, error_size)) {
            h3_gpu_submit(m->gpu);
            goto done;
        }
        if (!h3_gpu_submit(m->gpu)) CLIP_FAIL("SeedVR2 encode: %s", h3_gpu_error(m->gpu));
        if (act_frames(&enc) != L || !h3_gpu_tensor_read_f32(enc.t, enc_host, 2 * latent))
            CLIP_FAIL("SeedVR2: cannot read the latent");
        sv2_act_free(m, &enc);
        for (int t = 0; t < L; t++)
            for (size_t i = 0; i < lplane; i++)
                for (int c = 0; c < SV2_LATENT; c++)
                    cond[((size_t)c * L + t) * lplane + i] = enc_host[((size_t)t * lplane + i) * 2 * SV2_LATENT + c] * SV2_SCALING;
        sv2_pool_drain(m);
        trace_peak(m, &mark, "encode (clip)");
    }

    /* transformer: one Euler step from t = 1 (its blocks reload if they were freed) */
    if (!sv2_dit_ensure(m, error, error_size)) goto done;
    if (!sv2_layout_build(m, &layouts[0], lw / 2, lh / 2, L, 0, error, error_size) ||
        !sv2_layout_build(m, &layouts[1], lw / 2, lh / 2, L, 1, error, error_size)) goto done;
    if (p.noise_override) memcpy(noise, p.noise_override, latent * sizeof(float));
    else if (!sv2_fill_noise_clip(noise, p.first_latent_frame, L, lw, lh, seed)) CLIP_FAIL("SeedVR2 clip: out of memory");
    sv2_pack_tokens(noise, cond, lw, lh, L, tokens_host);
    tokens = h3_gpu_tensor_from_bf16(m->gpu, tokens_host, (size_t)N * SV2_PATCH_IN);
    pred = sv2_pool_take(m, (size_t)N * SV2_PATCH_OUT, 1);
    if (!tokens || !pred) CLIP_FAIL("SeedVR2 clip: out of GPU memory");
    if (!sv2_dit_forward(m, layouts, tokens, pred, p.dump_block, on_progress, opaque, cancel, error, error_size)) goto done;
    if (!h3_gpu_tensor_read_bf16(pred, pred_bf16, (size_t)N * SV2_PATCH_OUT)) CLIP_FAIL("SeedVR2: cannot read the prediction");
    h3_gpu_tensor_free(tokens); tokens = NULL;
    sv2_pool_give(m, pred); pred = NULL;
    sv2_unpack_pred(pred_bf16, lw, lh, L, pred_host);
    for (int c = 0; c < SV2_LATENT; c++)
        for (int t = 0; t < L; t++)
            for (size_t i = 0; i < lplane; i++) {
                size_t at = ((size_t)c * L + t) * lplane + i;
                z[((size_t)t * lplane + i) * SV2_LATENT + c] = (noise[at] - pred_host[at]) / SV2_SCALING;
            }
    sv2_layout_free(m, &layouts[0]); sv2_layout_free(m, &layouts[1]);   /* only the transformer needs them */
    sv2_pool_drain(m);
    trace_peak(m, &mark, "dit (clip)");

    /* decode (sliced): [L, lh, lw, 16] -> [T, H, W, 3] */
    if (unload_dit) sv2_dit_unload(m);
    zt = (act){h3_gpu_tensor_from_f32(m->gpu, z, latent), lh, lw, SV2_LATENT, L};
    if (!zt.t) CLIP_FAIL("SeedVR2 clip: out of GPU memory");
    if (!h3_gpu_begin(m->gpu)) CLIP_FAIL("SeedVR2: %s", h3_gpu_error(m->gpu));
    if (!sv2_vae_decode_video(m, &zt, &dec, p.vae_split, on_progress, opaque, cancel, error, error_size)) {
        h3_gpu_submit(m->gpu);
        goto done;
    }
    if (!h3_gpu_submit(m->gpu)) CLIP_FAIL("SeedVR2 decode: %s", h3_gpu_error(m->gpu));
    h3_gpu_tensor_free(zt.t); zt.t = NULL;
    if (act_frames(&dec) != frames) CLIP_FAIL("SeedVR2 clip: decoded %d frames of %d", act_frames(&dec), frames);
    trace_peak(m, &mark, "decode (clip)");

    /* colour transfer against the resized input, then 8-bit frames */
    style = sv2_pool_take(m, plane, 0);
    if (!style) CLIP_FAIL("SeedVR2 clip: out of GPU memory");
    for (int t = 0; t < frames; t++) {
        if (!clip_progress(on_progress, opaque, cancel, "vsr output", t, frames)) CLIP_FAIL("vsr cancelled");
        size_t out_bytes = (size_t)out[t].stride_bytes * out[t].height;
        if (!pixels || h3_gpu_tensor_elements(pixels) != out_bytes) {
            h3_gpu_tensor_free(pixels);
            pixels = h3_gpu_tensor_new_i8(m->gpu, out_bytes);
            if (!pixels) CLIP_FAIL("SeedVR2 clip: out of GPU memory");
        }
        if (!h3_gpu_begin(m->gpu) ||
            !h3_gpu_copy_f32(m->gpu, frame, 0, dec.t, (size_t)t * plane, plane) ||
            (color && !h3_gpu_copy_f32(m->gpu, style, 0, rgb_in, (size_t)t * plane, plane)) ||
            !h3_gpu_submit(m->gpu)) CLIP_FAIL("SeedVR2 clip: %s", h3_gpu_error(m->gpu));
        if (color && !sv2_color_transfer(m, frame, frame, style, full_w, full_h, error, error_size)) goto done;
        if (!h3_gpu_begin(m->gpu) ||
            !h3_gpu_sv2_f32_to_pixels(m->gpu, pixels, frame, (uint32_t)full_w, (uint32_t)full_h, (uint32_t)out_w,
                                      (uint32_t)out_h, (uint32_t)out[t].stride_bytes, out[t].format == H3_VSR_BGRA8) ||
            !h3_gpu_submit(m->gpu)) CLIP_FAIL("SeedVR2 output: %s", h3_gpu_error(m->gpu));
        if (!h3_gpu_tensor_read_i8(pixels, (int8_t *)out[t].pixels, out_bytes)) CLIP_FAIL("SeedVR2: cannot read the output frame");
    }
    clip_progress(on_progress, opaque, cancel, "vsr output", frames, frames);
    trace_peak(m, &mark, "colour + output (clip)");
    ok = 1;
done:
    h3_gpu_tensor_free(rgb_in); h3_gpu_tensor_free(pixels); h3_gpu_tensor_free(tokens); h3_gpu_tensor_free(zt.t);
    sv2_pool_give(m, frame); sv2_pool_give(m, style); sv2_pool_give(m, pred);
    if (enc.t) sv2_act_free(m, &enc);
    if (dec.t) sv2_act_free(m, &dec);
    sv2_layout_free(m, &layouts[0]); sv2_layout_free(m, &layouts[1]);
    sv2_pool_drain(m);
    m->pool_bypass = 0;
    free(cond); free(noise); free(pred_host); free(z); free(enc_host); free(tokens_host); free(pred_bf16);
    return ok;
}

/* ---------------------------------------------------------------- */
/* The window session: a video in windows of `window_frames` (4n+1) that
 * overlap by `overlap_frames`, each through h3_seedvr2_restore_clip; the
 * overlap is taken from the earlier window (which had the frames' past) or
 * cross-faded. Window starts are multiples of 4 (the stride is), so the
 * frame -> latent grouping and the per-latent-frame noise agree between
 * windows; the last window is moved back to end on the last frame instead
 * of being padded (its start may then be unaligned: only its new frames
 * are used, so nothing depends on the agreement). Input frames are copied
 * and kept until their window has run; outputs come back in order. */

#define SV2_VIDEO_MAX_WINDOW 121

static int pixel_bytes(h3_vsr_pixel_format format) { return format == H3_VSR_RGB24 ? 3 : 4; }

struct h3_seedvr2_video_session {
    h3_seedvr2 *m;                 /* NULL: a dry run (tests) that copies the input frames up */
    int in_w, in_h, out_w, out_h;
    h3_seedvr2_video_params p;
    int total;                     /* frames in the video, 0 unknown */
    int stride;                    /* window - overlap */
    h3_vsr_pixel_format in_format, out_format;
    int in_pixel, out_pixel;       /* bytes per pixel of the copies */
    /* input frames kept: absolute frames [in_first, in_first + in_count) */
    uint8_t **in_frames;
    int in_cap, in_first, in_count;
    int received, emitted;         /* absolute counts */
    int next_start;                /* the next window's first frame */
    int windows_done, frames_run, padded;   /* statistics */
    /* the earlier window's output for the overlap, absolute [pend_start, +pend_count) */
    uint8_t **pend;
    int pend_start, pend_count;
    uint8_t **win_out;             /* the window's output frames */
    int done, failed;
    double seconds;                /* in the windows */
};

int h3_seedvr2_video_check_params(const h3_seedvr2_video_params *p, char *reason, size_t reason_size) {
    if (!p) return 0;
    int W = p->window_frames, O = p->overlap_frames;
    if (W < 1 || W % 4 != 1 || W > SV2_VIDEO_MAX_WINDOW) { snprintf(reason, reason_size, "the window must be 4n+1 frames (5, 9, ..., %d)", SV2_VIDEO_MAX_WINDOW); return 0; }
    if (O < 0 || O >= W || (O != 0 && O % 4 != 1)) { snprintf(reason, reason_size, "the overlap must be 0 or 4n+1 frames below the window"); return 0; }
    if (W == 1 && O != 0) { snprintf(reason, reason_size, "a 1-frame window has no overlap"); return 0; }
    return 1;
}

h3_seedvr2_video_session *h3_seedvr2_video_session_open(h3_seedvr2 *m, int in_w, int in_h, int out_w, int out_h,
                                                        int total_frames, const h3_seedvr2_video_params *params,
                                                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    char reason[256];
    if (in_w < 1 || in_h < 1 || out_w < 16 || out_h < 16 || total_frames < 0) { fail(error, error_size, "SeedVR2 video: bad sizes"); return NULL; }
    if (!h3_seedvr2_video_check_params(params, reason, sizeof(reason))) { fail(error, error_size, "SeedVR2 video: %s", reason); return NULL; }
    if (m) {
        h3_seedvr2_video_requirements req;
        if (!h3_seedvr2_video_check_requirements(m, in_w, in_h, out_w, out_h, params->window_frames, &req) || !req.supported) {
            fail(error, error_size, "SeedVR2 video: %s", req.reason[0] ? req.reason : "unsupported");
            return NULL;
        }
    }
    h3_seedvr2_video_session *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->m = m; s->in_w = in_w; s->in_h = in_h; s->out_w = out_w; s->out_h = out_h;
    s->p = *params; s->total = total_frames;
    s->stride = s->p.window_frames - s->p.overlap_frames;
    s->in_cap = 2 * s->p.window_frames + s->stride;   /* the last W frames stay: the last window may move back */
    s->in_frames = calloc((size_t)s->in_cap, sizeof(*s->in_frames));
    s->pend = calloc((size_t)s->p.window_frames, sizeof(*s->pend));
    s->win_out = calloc((size_t)s->p.window_frames, sizeof(*s->win_out));
    if (!s->in_frames || !s->pend || !s->win_out) { h3_seedvr2_video_session_close(s); fail(error, error_size, "SeedVR2 video: out of memory"); return NULL; }
    s->in_format = H3_VSR_BGRA8; s->out_format = H3_VSR_BGRA8;
    return s;
}

void h3_seedvr2_video_session_close(h3_seedvr2_video_session *s) {
    if (!s) return;
    if (s->in_frames) { for (int i = 0; i < s->in_cap; i++) free(s->in_frames[i]); free(s->in_frames); }
    if (s->pend) { for (int i = 0; i < s->p.window_frames; i++) free(s->pend[i]); free(s->pend); }
    if (s->win_out) { for (int i = 0; i < s->p.window_frames; i++) free(s->win_out[i]); free(s->win_out); }
    free(s);
}

int h3_seedvr2_video_session_pending_frames(const h3_seedvr2_video_session *s) {
    return s ? s->received - s->emitted : 0;
}

int h3_seedvr2_video_session_next_frames(const h3_seedvr2_video_session *s) {
    if (!s || s->done || s->failed) return 0;
    int need = s->next_start + s->p.window_frames - s->received;
    if (s->total > 0 && need > s->total - s->received) need = s->total - s->received;
    return need > 0 ? need : 0;
}

int h3_seedvr2_video_session_windows(const h3_seedvr2_video_session *s, int *frames_run, int *padded_frames, double *seconds) {
    if (!s) return 0;
    if (frames_run) *frames_run = s->frames_run;
    if (padded_frames) *padded_frames = s->padded;
    if (seconds) *seconds = s->seconds;
    return s->windows_done;
}

/* The start of the window after one at `start`: a stride on, or, when
 * that window would run past the end of a video whose length is known
 * (`total`), the start that ends on the last frame - unless that is no
 * further than `start`, in which case the window is padded. */
static int next_window_start(const h3_seedvr2_video_session *s, int start, int total) {
    int W = s->p.window_frames, ns = start + s->stride;
    if (total > 0 && ns + W > total) {
        int back = total - W;
        if (back > start) ns = back;
    }
    return ns;
}

static uint8_t *frame_alloc(int w, int h, int pixel) { return malloc((size_t)w * h * pixel); }

/* in[t] copied into the store as absolute frame `abs`. */
static int store_input(h3_seedvr2_video_session *s, const h3_vsr_image *in, int abs) {
    int slot = abs - s->in_first;
    if (slot < 0 || slot >= s->in_cap) return 0;
    if (!s->in_frames[slot]) s->in_frames[slot] = frame_alloc(s->in_w, s->in_h, s->in_pixel);
    if (!s->in_frames[slot]) return 0;
    size_t row = (size_t)s->in_w * s->in_pixel;
    for (int y = 0; y < s->in_h; y++)
        memcpy(s->in_frames[slot] + (size_t)y * row, in->pixels + (size_t)y * in->stride_bytes, row);
    if (slot >= s->in_count) s->in_count = slot + 1;
    return 1;
}

/* Drops stored frames before `abs`. */
static void forget_input(h3_seedvr2_video_session *s, int abs) {
    int drop = abs - s->in_first;
    if (drop <= 0) return;
    if (drop > s->in_count) drop = s->in_count;
    for (int i = 0; i < drop; i++) free(s->in_frames[i]);
    memmove(s->in_frames, s->in_frames + drop, (size_t)(s->in_cap - drop) * sizeof(*s->in_frames));
    memset(s->in_frames + s->in_cap - drop, 0, (size_t)drop * sizeof(*s->in_frames));
    s->in_first += drop;
    s->in_count -= drop;
}

static int emit_frame(h3_seedvr2_video_session *s, const uint8_t *pixels, h3_vsr_image_out *out, int capacity,
                      int *produced, char *error, size_t error_size) {
    if (*produced >= capacity) { fail(error, error_size, "SeedVR2 video: no room for output frame %d", s->emitted); return 0; }
    h3_vsr_image_out *o = &out[*produced];
    if (o->width != s->out_w || o->height != s->out_h || o->format != s->out_format) {
        fail(error, error_size, "SeedVR2 video: output frame %d is %dx%d, the session %dx%d", s->emitted, o->width, o->height, s->out_w, s->out_h);
        return 0;
    }
    size_t row = (size_t)s->out_w * s->out_pixel;
    for (int y = 0; y < s->out_h; y++) memcpy(o->pixels + (size_t)y * o->stride_bytes, pixels + (size_t)y * row, row);
    (*produced)++;
    s->emitted++;
    return 1;
}

/* The overlap: the earlier window's frame, or a cross-fade (Hann weights
 * over the overlap) with the later one's. i of n frames into the overlap. */
static void blend_frames(const h3_seedvr2_video_session *s, uint8_t *earlier, const uint8_t *later, int i, int n) {
    if (s->p.blend != 1) return;
    double w = 0.5 - 0.5 * cos(M_PI * (double)(i + 1) / (double)(n + 1));
    size_t bytes = (size_t)s->out_w * s->out_h * s->out_pixel;
    for (size_t k = 0; k < bytes; k++)
        earlier[k] = (uint8_t)lrint((1.0 - w) * earlier[k] + w * later[k]);
}

/* A dry run's window: nearest-neighbour copies of the input (tests). */
static void dry_window(const h3_seedvr2_video_session *s, const h3_vsr_image *in, int frames, h3_vsr_image_out *out) {
    for (int t = 0; t < frames; t++)
        for (int y = 0; y < s->out_h; y++) {
            int sy = (int)((int64_t)y * s->in_h / s->out_h);
            for (int x = 0; x < s->out_w; x++) {
                int sx = (int)((int64_t)x * s->in_w / s->out_w);
                memcpy(out[t].pixels + (size_t)y * out[t].stride_bytes + (size_t)x * s->out_pixel,
                       in[t].pixels + (size_t)sy * in[t].stride_bytes + (size_t)sx * s->in_pixel, (size_t)s->out_pixel);
            }
        }
}

/* Runs the window at `start` on the stored frames (the tail repeated to the
 * window length), then emits and keeps what the policy says. */
static int run_window(h3_seedvr2_video_session *s, int start, int last_window, h3_vsr_image_out *out, int capacity,
                      int *produced, h3_progress_callback on_progress, void *opaque, const volatile int *cancel,
                      char *error, size_t error_size) {
    int W = s->p.window_frames;
    int real = s->received - start;
    if (real > W) real = W;
    if (real < 1 || start < s->in_first) { fail(error, error_size, "SeedVR2 video: window at %d has no frames", start); return 0; }
    h3_vsr_image in[SV2_VIDEO_MAX_WINDOW];
    h3_vsr_image_out wo[SV2_VIDEO_MAX_WINDOW];
    for (int t = 0; t < W; t++) {
        int src = t < real ? t : real - 1;
        in[t] = (h3_vsr_image){s->in_frames[start - s->in_first + src], s->in_w, s->in_h, s->in_w * s->in_pixel, s->in_format};
        if (!s->win_out[t]) s->win_out[t] = frame_alloc(s->out_w, s->out_h, s->out_pixel);
        if (!s->win_out[t]) { fail(error, error_size, "SeedVR2 video: out of memory"); return 0; }
        wo[t] = (h3_vsr_image_out){s->win_out[t], s->out_w, s->out_h, s->out_w * s->out_pixel, s->out_format};
    }
    double t0 = now_seconds();
    if (s->m) {
        h3_seedvr2_clip_params cp = {s->p.vae_slicing, start / 4, s->p.seed, s->p.color_transfer, NULL, -1};
        if (!h3_seedvr2_restore_clip(s->m, in, W, s->out_w, s->out_h, wo, &cp, on_progress, opaque, cancel, error, error_size))
            return 0;
    } else {
        dry_window(s, in, W, wo);
    }
    s->seconds += now_seconds() - t0;
    s->frames_run += W;
    s->padded += W - real;
    s->windows_done++;
    /* the frames after the next window's start wait for it; the rest go out,
     * the earlier window's overlap frames blended with ours */
    int ns = last_window ? start + real : next_window_start(s, start, s->total);
    for (int t = 0; t < real; t++) {
        int abs = start + t;
        if (abs < s->emitted) continue;                       /* already out (the window was moved back) */
        if (abs >= ns && !last_window) break;
        uint8_t *frame = s->win_out[t];
        if (abs >= s->pend_start && abs < s->pend_start + s->pend_count) {
            uint8_t *earlier = s->pend[abs - s->pend_start];
            blend_frames(s, earlier, frame, abs - s->pend_start, s->pend_count);
            frame = earlier;
        }
        if (!emit_frame(s, frame, out, capacity, produced, error, error_size)) return 0;
    }
    /* the new pending overlap: our frames from ns on (swap buffers in) */
    for (int i = 0; i < s->pend_count; i++) { free(s->pend[i]); s->pend[i] = NULL; }
    s->pend_count = 0;
    if (!last_window) {
        s->pend_start = ns;
        for (int t = ns - start; t < real; t++) {
            s->pend[s->pend_count++] = s->win_out[t];
            s->win_out[t] = NULL;
        }
    }
    s->next_start = ns;
    forget_input(s, ns < s->received - W ? ns : s->received - W);
    return 1;
}

int h3_seedvr2_video_session_process(h3_seedvr2_video_session *s, const h3_vsr_image *in, int in_frames, int is_last,
                                     h3_vsr_image_out *out, int out_capacity, int *out_frames,
                                     h3_progress_callback on_progress, void *opaque, const volatile int *cancel,
                                     char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (out_frames) *out_frames = 0;
    if (!s || (in_frames > 0 && !in) || in_frames < 0 || (!out && out_capacity > 0)) { fail(error, error_size, "SeedVR2 video: bad arguments"); return 0; }
    if (s->failed) { fail(error, error_size, "SeedVR2 video: the session failed earlier"); return 0; }
    if (s->done) { if (in_frames) { fail(error, error_size, "SeedVR2 video: the session is finished"); return 0; } return 1; }
    int expected = h3_seedvr2_video_session_next_frames(s);
    if (!is_last && in_frames != expected) { fail(error, error_size, "SeedVR2 video: this call needs %d frames (got %d)", expected, in_frames); s->failed = 1; return 0; }
    if (in_frames > expected) { fail(error, error_size, "SeedVR2 video: at most %d frames now (got %d)", expected, in_frames); s->failed = 1; return 0; }
    if (in_frames > 0) {
        if (s->received == 0) {
            s->in_format = in[0].format; s->in_pixel = pixel_bytes(s->in_format);
            s->out_format = out_capacity > 0 ? out[0].format : H3_VSR_BGRA8; s->out_pixel = pixel_bytes(s->out_format);
        }
        for (int t = 0; t < in_frames; t++) {
            if (in[t].width != s->in_w || in[t].height != s->in_h || in[t].format != s->in_format) {
                fail(error, error_size, "SeedVR2 video: frame %d is %dx%d, the session %dx%d", s->received, in[t].width, in[t].height, s->in_w, s->in_h);
                s->failed = 1;
                return 0;
            }
            if (!store_input(s, &in[t], s->received)) { fail(error, error_size, "SeedVR2 video: out of memory"); s->failed = 1; return 0; }
            s->received++;
        }
    }
    if (s->total > 0 && s->received >= s->total) is_last = 1;
    int produced = 0, W = s->p.window_frames;
    for (;;) {
        int start = s->next_start;
        if (start >= s->received) break;
        int complete = s->received >= start + W;
        if (!complete && !is_last) break;
        if (!complete && s->windows_done > 0 && s->received - W > start - s->stride && s->received - W >= s->in_first) {
            start = s->received - W;   /* the end is known now: end on the last frame instead of padding */
            s->next_start = start;
            complete = 1;
        }
        int last_window = (s->total > 0 && start + W >= s->total) || (is_last && s->received <= start + W);
        if (!run_window(s, start, last_window, out, out_capacity, &produced, on_progress, opaque, cancel, error, error_size)) {
            s->failed = 1;
            if (out_frames) *out_frames = produced;
            return 0;
        }
        if (last_window) { s->done = 1; break; }
    }
    if (is_last && !s->done) {
        /* no window left to run: whatever waits in the overlap is final */
        for (int i = 0; i < s->pend_count; i++) {
            if (s->pend_start + i < s->emitted) continue;
            if (!emit_frame(s, s->pend[i], out, out_capacity, &produced, error, error_size)) { s->failed = 1; return 0; }
        }
        for (int i = 0; i < s->pend_count; i++) { free(s->pend[i]); s->pend[i] = NULL; }
        s->pend_count = 0;
        s->done = 1;
    }
    if (s->done && s->emitted != s->received) {
        fail(error, error_size, "SeedVR2 video: %d frames in, %d out", s->received, s->emitted);
        s->failed = 1;
        return 0;
    }
    if (out_frames) *out_frames = produced;
    return 1;
}
