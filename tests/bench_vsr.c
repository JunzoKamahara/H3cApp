/* Stage timing of the SwiftVR engine on the real weights: feeds synthetic
 * frames of a given size through h3_vsr_session_process and reports, per
 * chunk, the wall time of the resize + ReAE encode, the DiT (and its
 * per-block pace), the ReAE decode and the 8-bit output, plus the GPU
 * context's own counters. Not part of `make test`.
 *
 *   ./h3_vsr_bench [in_w in_h out_w out_h] [chunks] [int8|bf16|auto] [full]
 *   defaults: 960 544 1920 1088, 2 chunks (28 + 24 frames), auto, MFSWA
 * Environment: H3_VSR_MODEL_DIR (default the app's Upscalers/SwiftVR),
 * H3_VSR_REAE_BATCH (ReAE frames per pass; 0 = whole chunk). */
#include "h3_vsr_internal.h"
#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9; }

typedef struct {
    double start, encode_begin, encode_end, dit_begin, dit_end, decode_end;
    double block_first, block_last;
    int blocks;
} chunk_times;

static int on_progress(const char *phase, int done, int total, void *opaque) {
    chunk_times *t = opaque;
    double at = now();
    if (!strcmp(phase, "vsr encode")) {
        if (done == 0) t->encode_begin = at; else t->encode_end = at;
    } else if (!strcmp(phase, "vsr dit")) {
        if (done == 0) { t->dit_begin = at; t->block_first = at; }
        if (done == total) t->dit_end = at;
        else t->block_last = at;
        t->blocks = total;
    } else if (!strcmp(phase, "vsr decode")) {
        t->decode_end = at;
    }
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    int in_w = 960, in_h = 544, out_w = 1920, out_h = 1088, chunks = 2;
    const char *precision_name = "auto";
    int full = 0, arg = 1;
    if (argc >= 5) { in_w = atoi(argv[1]); in_h = atoi(argv[2]); out_w = atoi(argv[3]); out_h = atoi(argv[4]); arg = 5; }
    if (argc > arg) chunks = atoi(argv[arg++]);
    if (argc > arg) precision_name = argv[arg++];
    if (argc > arg) full = !strcmp(argv[arg++], "full");
    const char *model_dir = getenv("H3_VSR_MODEL_DIR");
    static char default_model[4096];
    if (!model_dir || !model_dir[0]) {
        snprintf(default_model, sizeof(default_model), "%s/Library/Application Support/H3cApp/Upscalers/SwiftVR", getenv("HOME") ? getenv("HOME") : ".");
        model_dir = default_model;
    }
    h3_vsr_precision precision = !strcmp(precision_name, "int8") ? H3_VSR_PRECISION_INT8 :
                                 !strcmp(precision_name, "bf16") ? H3_VSR_PRECISION_BF16 :
                                 h3_vsr_int8_supported() ? H3_VSR_PRECISION_INT8 : H3_VSR_PRECISION_BF16;
    char cache_path[4096], error[512] = {0};
    if (!h3_vsr_cache_path(model_dir, precision, cache_path, sizeof(cache_path))) return 1;
    if (!h3_vsr_cache_matches(cache_path, model_dir)) {
        fprintf(stderr, "no matching cache %s (build_vsr_cache first)\n", cache_path);
        return 1;
    }
    h3_vsr_open_params params = {0};
    params.precision = precision;
    params.cache_path = cache_path;
    params.shader_source_path = "h3_shaders.metal";
    params.full_attention = full;
    if (getenv("H3_VSR_REAE_BATCH")) params.reae_frame_batch = atoi(getenv("H3_VSR_REAE_BATCH"));
    double t0 = now();
    h3_vsr *vsr = h3_vsr_open(model_dir, &params, NULL, NULL, error, sizeof(error));
    if (!vsr) { fprintf(stderr, "%s\n", error); return 1; }
    printf("open: %.1f s (%s, %s attention, reae batch %d)\n", now() - t0, precision == H3_VSR_PRECISION_INT8 ? "int8" : "bf16", full ? "dense" : "MFSWA", params.reae_frame_batch);
    h3_vsr_requirements req;
    if (h3_vsr_check_requirements(vsr, in_w, in_h, out_w, out_h, &req))
        printf("requirements: %s, tokens/chunk %d, windows %d/%d, estimate %.1f s/chunk, gpu %.1f GB\n",
               req.supported ? "supported" : req.reason, req.tokens_per_chunk, req.windows_even, req.windows_odd,
               req.estimated_seconds_per_chunk, (double)req.gpu_bytes / 1e9);
    int frames = H3_VSR_CLIP_FRAMES + 4 + (chunks - 1) * H3_VSR_CLIP_FRAMES;
    h3_vsr_session *s = h3_vsr_session_open(vsr, in_w, in_h, out_w, out_h, frames, error, sizeof(error));
    if (!s) { fprintf(stderr, "%s\n", error); return 1; }
    /* Smooth synthetic content (a moving gradient with a grid), RGB24. */
    size_t in_frame = (size_t)in_w * in_h * 3, out_frame = (size_t)out_w * out_h * 3;
    uint8_t *video = malloc((size_t)frames * in_frame);
    for (int f = 0; f < frames; f++)
        for (int y = 0; y < in_h; y++)
            for (int x = 0; x < in_w; x++) {
                uint8_t *p = video + (size_t)f * in_frame + ((size_t)y * in_w + x) * 3;
                int grid = ((x + f) / 16 + y / 16) & 1 ? 40 : 0;
                p[0] = (uint8_t)((x * 255) / in_w); p[1] = (uint8_t)((y * 255) / in_h); p[2] = (uint8_t)(128 + grid);
            }
    uint8_t *out = malloc((size_t)frames * out_frame);
    h3_vsr_image in[32];
    h3_vsr_image_out outs[64];
    int fed = 0, total = 0, chunk = 0;
    double wall0 = now(), gpu_prev = 0.0;
    while (fed < frames) {
        int want = h3_vsr_session_next_chunk_frames(s);
        int n = frames - fed < want ? frames - fed : want;
        int is_last = fed + n >= frames;
        for (int i = 0; i < n; i++) {
            in[i].pixels = video + (size_t)(fed + i) * in_frame;
            in[i].width = in_w; in[i].height = in_h; in[i].stride_bytes = in_w * 3; in[i].format = H3_VSR_RGB24;
        }
        for (int i = 0; i < 64; i++) {
            int frame = total + i;
            outs[i].pixels = frame < frames ? out + (size_t)frame * out_frame : NULL;
            outs[i].width = out_w; outs[i].height = out_h; outs[i].stride_bytes = out_w * 3; outs[i].format = H3_VSR_RGB24;
        }
        int capacity = frames - total < 64 ? frames - total : 64, got = 0;
        chunk_times t = {0};
        t.start = now();
        if (!h3_vsr_session_process(s, in, n, is_last, outs, capacity, &got, on_progress, &t, NULL, error, sizeof(error))) {
            fprintf(stderr, "%s\n", error);
            return 1;
        }
        double end = now();
        double dit_s, reae_s; uint64_t peak;
        h3_vsr_get_stats(vsr, &dit_s, &reae_s, &peak);
        /* The callbacks fire while the CPU encodes, so these splits are
         * encode pace, not GPU time: H3_VSR_TRACE_STAGES=1 for that. */
        double decode_s = t.decode_end > 0 ? t.decode_end - t.dit_end : 0.0;
        printf("chunk %d (%d frames in, %d out): total %.2f s = resize+encode %.2f, dit %.2f (%.0f ms/block over %d), decode %.2f, output %.2f; gpu busy %.2f s, peak live %.2f GiB\n",
               chunk, n, got, end - t.start, t.encode_end - t.start, t.dit_end - t.dit_begin,
               t.blocks > 1 ? 1000.0 * (t.block_last - t.block_first) / (t.blocks - 1) : 0.0, t.blocks,
               decode_s, t.decode_end > 0 ? end - t.decode_end : end - t.dit_end, dit_s - gpu_prev, (double)peak / 1073741824.0);
        gpu_prev = dit_s;
        h3_gpu_stats st;
        if (h3_vsr_gpu_stats(vsr, &st))
            printf("        encode %.2f s, wait %.2f s, %llu submissions, %llu conv, %llu sdpa, %llu linear, %llu direct dispatches (cumulative)\n",
                   st.command_encode_seconds, st.command_wait_seconds, (unsigned long long)st.submissions,
                   (unsigned long long)st.mps_conv_dispatches, (unsigned long long)st.mps_sdpa_dispatches,
                   (unsigned long long)st.mps_linear_dispatches, (unsigned long long)st.direct_dispatches);
        total += got; fed += n; chunk++;
    }
    printf("all: %.2f s wall for %d frames (%d chunks) -> %.2f s per middle chunk, %.1f frames/s\n",
           now() - wall0, frames, chunks, chunks > 1 ? (now() - wall0) / chunks : now() - wall0, frames / (now() - wall0));
    h3_vsr_session_close(s);
    h3_vsr_close(vsr);
    free(video); free(out);
    return 0;
}
