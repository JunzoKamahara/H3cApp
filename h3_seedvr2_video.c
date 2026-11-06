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
