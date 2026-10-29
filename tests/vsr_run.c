/* Runs the SwiftVR engine on raw RGB24 frames (the quality evaluation of
 * UPSCALE_PLAN §9.5): precision and attention are chosen here, which the
 * app does not expose. Not part of `make test`.
 *   ./h3_vsr_run in.raw W H FRAMES out.raw OUT_W OUT_H [int8|bf16|auto] [window|dense]
 * Environment: H3_VSR_MODEL_DIR (default the app's Upscalers/SwiftVR). */
#include "h3_vsr_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage: %s in.raw W H FRAMES out.raw OUT_W OUT_H [int8|bf16|auto] [window|dense]\n", argv[0]); return 2; }
    const char *in_path = argv[1]; int in_w = atoi(argv[2]), in_h = atoi(argv[3]), frames = atoi(argv[4]);
    const char *out_path = argv[5]; int out_w = atoi(argv[6]), out_h = atoi(argv[7]);
    const char *precision_name = argc > 8 ? argv[8] : "auto";
    int dense = argc > 9 && !strcmp(argv[9], "dense");
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
    if (!h3_vsr_cache_matches(cache_path, model_dir)) { fprintf(stderr, "no matching cache %s\n", cache_path); return 1; }
    size_t in_frame = (size_t)in_w * in_h * 3, out_frame = (size_t)out_w * out_h * 3;
    uint8_t *video = malloc((size_t)frames * in_frame), *out = malloc((size_t)frames * out_frame);
    FILE *f = fopen(in_path, "rb");
    if (!f || fread(video, in_frame, (size_t)frames, f) != (size_t)frames) { fprintf(stderr, "cannot read %s\n", in_path); return 1; }
    fclose(f);
    h3_vsr_open_params params = {0};
    params.precision = precision; params.cache_path = cache_path; params.shader_source_path = "h3_shaders.metal";
    params.full_attention = dense;
    h3_vsr *vsr = h3_vsr_open(model_dir, &params, NULL, NULL, error, sizeof(error));
    if (!vsr) { fprintf(stderr, "%s\n", error); return 1; }
    h3_vsr_session *s = h3_vsr_session_open(vsr, in_w, in_h, out_w, out_h, frames, error, sizeof(error));
    if (!s) { fprintf(stderr, "%s\n", error); return 1; }
    h3_vsr_image in[32]; h3_vsr_image_out outs[96];
    int fed = 0, total = 0;
    double t0 = now();
    while (fed < frames) {
        int want = h3_vsr_session_next_chunk_frames(s);
        int n = frames - fed < want ? frames - fed : want;
        int is_last = fed + n >= frames;
        for (int i = 0; i < n; i++) {
            in[i].pixels = video + (size_t)(fed + i) * in_frame; in[i].width = in_w; in[i].height = in_h;
            in[i].stride_bytes = in_w * 3; in[i].format = H3_VSR_RGB24;
        }
        for (int i = 0; i < 96; i++) {
            int frame = total + i;
            outs[i].pixels = frame < frames ? out + (size_t)frame * out_frame : NULL;
            outs[i].width = out_w; outs[i].height = out_h; outs[i].stride_bytes = out_w * 3; outs[i].format = H3_VSR_RGB24;
        }
        int capacity = frames - total < 96 ? frames - total : 96, got = 0;
        if (!h3_vsr_session_process(s, in, n, is_last, outs, capacity, &got, NULL, NULL, NULL, error, sizeof(error))) {
            fprintf(stderr, "%s\n", error); return 1;
        }
        total += got; fed += n;
    }
    if (total != frames) { fprintf(stderr, "got %d of %d frames\n", total, frames); return 1; }
    fprintf(stderr, "%s %s: %d frames %dx%d -> %dx%d in %.1f s\n", precision == H3_VSR_PRECISION_INT8 ? "int8" : "bf16",
            dense ? "dense" : "window", frames, in_w, in_h, out_w, out_h, now() - t0);
    f = fopen(out_path, "wb");
    if (!f || fwrite(out, out_frame, (size_t)frames, f) != (size_t)frames) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }
    fclose(f);
    h3_vsr_session_close(s); h3_vsr_close(vsr);
    return 0;
}
