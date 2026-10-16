// Plain-C generation driver, mirroring native/H3Spike/Sources/H3Spike/main.swift
// closely enough to reproduce the same run, but without Swift/SwiftPM - built
// to sidestep a real, still-unexplained interaction between the Swift runtime
// and ccv's Metal shader JIT-compilation (see SPEEDUP_ROADMAP.md item 5):
// the exact same H3_ATTENTION_BACKEND=ccv_dense call that works from a plain
// C or Objective-C++ process fails with a garbled/corrupted-looking Metal
// shader compile error every time it is made from H3Spike (a Swift binary),
// even in the most minimal possible reproduction (no model loaded, called
// immediately after h3_gpu_create). Root cause not found after extensive
// isolation (ruled out: GPU memory pressure up to ~62GB in both few-large and
// many-small buffer patterns, buffer array size bugs, h3's own shader library
// being compiled first, thread races, matching -mmacosx-version-min, and
// merely linking libswiftXPC/libswiftCore without an actual Swift runtime).
// This tool lets real end-to-end generation-level validation proceed without
// waiting on that root cause.
//
// Build: see tools/ccv_eval/README.md (same CCV_DIR pattern as the Makefile).
// Usage: same env vars as H3Spike (H3SPIKE_SIZE/FRAMES/STEPS/SEED/...),
//   H3_ATTENTION_BACKEND=ccv_dense to select the experimental backend,
//   H3SPIKE_DUMP=<path> to also write raw packed RGB24 frames for PSNR
//   comparison against a baseline run, plus this repo's other H3_* env
//   vars (H3_ATTENTION_CACHE, H3_DUMP_ATTENTION_QKV, etc.) all work as-is
//   since this drives the exact same public h3.h API H3Spike does.
#include "h3.h"
#include "h3_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}
static double g_start;

static FILE *g_dump = NULL;
static int g_frames_received = 0;
static int g_previews_received = 0;

static int on_progress(const char *phase, int completed, int total, void *opaque) {
    (void)opaque;
    printf("[%7.2fs] [progress] %s %d/%d\n", now() - g_start, phase, completed, total);
    return 0;
}

static int on_frame(const h3_frame *frame, void *opaque) {
    (void)opaque;
    if (frame->denoise_step >= 0) {
        g_previews_received++;
        printf("[preview] denoise step %d/%d\n", frame->denoise_step + 1, frame->denoise_steps);
        return 0;
    }
    g_frames_received++;
    printf("[%7.2fs] [frame] %d/%d\n", now() - g_start, frame->frame_index + 1, frame->frame_count);
    if (g_dump && frame->rgb) {
        int row_bytes = frame->width * 3;
        for (int row = 0; row < frame->height; row++)
            fwrite(frame->rgb + (size_t)row * frame->stride, 1, (size_t)row_bytes, g_dump);
    }
    return 0;
}

static int env_int(const char *name, int fallback) {
    const char *v = getenv(name);
    return v && *v ? atoi(v) : fallback;
}

int main(int argc, char **argv) {
    g_start = now();
    setvbuf(stdout, NULL, _IOLBF, 0);

    const char *model_dir = argc > 1 ? argv[1] : NULL;
    char default_model_dir[1024];
    if (!model_dir) {
        snprintf(default_model_dir, sizeof(default_model_dir), "%s/models/MiniMax-H3",
                 getenv("HOME") ? getenv("HOME") : "");
        model_dir = default_model_dir;
    }
    const char *output_path = argc > 2 ? argv[2] : "/tmp/h3cli_output.mp4";
    const char *prompt = argc > 3 ? argv[3] : "A cat playing with a ball of yarn.";

    printf("== h3c-app plain-C generation CLI ==\n");
    printf("Model dir: %s\n", model_dir);

    h3_ctx *ctx = h3_load_dir(model_dir);
    if (!ctx) { fprintf(stderr, "h3_load_dir returned NULL\n"); return 1; }

    const h3_device_info *device = h3_device(ctx);
    if (device) {
        printf("Device: %s (%s)\n", device->name, device->architecture);
        printf("  unified memory: %d, metal4: %d\n", device->unified_memory, device->metal4);
    }
    const h3_model_info *model = h3_model(ctx);
    if (model) {
        printf("FL2VA transformer: %llu bytes, %zu tensors\n",
               (unsigned long long)model->fl2va_transformer.bytes,
               model->fl2va_transformer.tensors);
    }

    if (!getenv("H3_ATTENTION_CACHE") && !getenv("H3SPIKE_SSD")) {
        char cache_path[1024];
        snprintf(cache_path, sizeof(cache_path), "%s/models/cache/dit_int8_v2.cache",
                 getenv("HOME") ? getenv("HOME") : "");
        setenv("H3_ATTENTION_CACHE", cache_path, 0);
    }

    const char *dump_path = getenv("H3SPIKE_DUMP");
    if (dump_path) g_dump = fopen(dump_path, "wb");

    h3_params params = H3_PARAMS_DEFAULT;
    params.output_path = output_path;
    int size = env_int("H3SPIKE_SIZE", 256);
    params.width = env_int("H3SPIKE_WIDTH", size);
    params.height = env_int("H3SPIKE_HEIGHT", size);
    params.first_frame = getenv("H3SPIKE_FIRST_FRAME");
    /* Forced audio (lip sync): H3SPIKE_FORCED_AUDIO=<file>, optional
     * H3SPIKE_FORCED_AUDIO_OFFSET=<seconds>. */
    params.forced_audio_path = getenv("H3SPIKE_FORCED_AUDIO");
    if (getenv("H3SPIKE_FORCED_AUDIO_OFFSET"))
        params.forced_audio_offset = atof(getenv("H3SPIKE_FORCED_AUDIO_OFFSET"));
    /* Clip continuation: H3SPIKE_SAVE_LATENT=<file> keeps this clip's
     * latents; H3SPIKE_CONTINUE_LATENT=<file> continues one (pinning
     * H3SPIKE_CONTINUE_FRAMES, default 22, and optionally
     * H3SPIKE_CONTINUE_AUDIO_FRAMES of sound). */
    params.save_latent_path = getenv("H3SPIKE_SAVE_LATENT");
    params.continue_latent_path = getenv("H3SPIKE_CONTINUE_LATENT");
    params.continue_frames = env_int("H3SPIKE_CONTINUE_FRAMES", 22);
    params.continue_audio_frames = env_int("H3SPIKE_CONTINUE_AUDIO_FRAMES", 0);
    params.frames = env_int("H3SPIKE_FRAMES", 9);
    params.steps = env_int("H3SPIKE_STEPS", 4);
    params.seed = (uint64_t)env_int("H3SPIKE_SEED", 42);
    params.dit_layers = env_int("H3SPIKE_LAYERS", 50);
    params.ssd_streaming = getenv("H3SPIKE_SSD") && !strcmp(getenv("H3SPIKE_SSD"), "1");
    params.denoise_reuse = env_int("H3SPIKE_REUSE", 1);
    params.core_reuse = env_int("H3SPIKE_CORE_REUSE", 1);
    params.token_reduction = env_int("H3SPIKE_TOKEN_REDUCTION", 0);
    params.fast_attention = env_int("H3SPIKE_FAST_ATTENTION", 0);
    params.use_int8_row_fc2 = env_int("H3SPIKE_INT8_ROW_FC2", 0);
    /* Ref2VA with one reference image: H3SPIKE_REF_IMAGE=<file>. */
    h3_reference reference = {H3_REFERENCE_IMAGE, getenv("H3SPIKE_REF_IMAGE"),
                              NULL, 0};
    if (reference.path && *reference.path) {
        params.references = &reference;
        params.reference_count = 1;
    }
    /* One LoRA: H3SPIKE_LORA=<file>, H3SPIKE_LORA_STRENGTH (default 1);
     * H3SPIKE_SAMPLER=1 for the re-noise step rule, H3SPIKE_AUDIO_SHIFT. */
    h3_lora lora = {getenv("H3SPIKE_LORA"),
                    getenv("H3SPIKE_LORA_STRENGTH") ?
                        (float)atof(getenv("H3SPIKE_LORA_STRENGTH")) : 1.0f};
    if (lora.path && *lora.path) {
        params.loras = &lora;
        params.lora_count = 1;
    }
    params.sampler = env_int("H3SPIKE_SAMPLER", params.sampler);
    if (getenv("H3SPIKE_AUDIO_SHIFT"))
        params.audio_sigma_shift = (float)atof(getenv("H3SPIKE_AUDIO_SHIFT"));
    params.reference_image_size = H3_REFERENCE_IMAGE_MATCH;
    params.on_progress = on_progress;
    params.on_frame = on_frame;
    params.callback_opaque = NULL;

    printf("Generating %dx%d, %d frames, %d steps...\n", params.width, params.height,
           params.frames, params.steps);
    h3_result *result = h3_generate(ctx, prompt, &params);
    if (!result) {
        fprintf(stderr, "h3_generate failed: %s\n", h3_last_error(ctx));
        h3_free(ctx);
        return 1;
    }
    printf("Done in %.1fs\n", now() - g_start);
    printf("Result: %d frames @ %dfps, seed=%llu\n", result->frames, result->fps,
           (unsigned long long)result->seed);
    printf("Attention: %llu ccv calls (%llu direct)\n",
           (unsigned long long)result->ccv_attention_calls,
           (unsigned long long)result->ccv_attention_direct_calls);
    printf("Frames delivered via on_frame: %d, previews: %d\n", g_frames_received,
           g_previews_received);
    if (getenv("H3_ATTENTION_BACKEND"))
        printf("ccv attention dispatch count: %llu\n",
               (unsigned long long)h3_gpu_ccv_attention_dispatch_count());
    printf("Output written to: %s\n", output_path);
    h3_result_free(result);
    if (g_dump) fclose(g_dump);
    h3_free(ctx);
    return 0;
}
