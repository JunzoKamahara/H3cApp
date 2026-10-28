/* One-time tool: converts a downloaded SwiftVR model directory (F32
 * safetensors) into the .h3vc weight cache the VSR engine runs from.
 *
 *   build_vsr_cache <model_dir> [int8|bf16|auto] [output.h3vc]
 *
 * <model_dir> holds reae.safetensors, prompt_embedding.safetensors and
 * transformer/{config.json,diffusion_pytorch_model.safetensors}. The
 * default output is <model_dir>/swiftvr-dit-<precision>.h3vc; auto picks
 * int8 on tensor hardware (M5) and BF16 elsewhere. See UPSCALE_PLAN.md §6.3. */
#include "h3_vsr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int progress(const char *phase, int completed, int total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "\r%s: %d / %d", phase, completed, total);
    if (completed == total) fputc('\n', stderr);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s <model_dir> [int8|bf16|auto] [output.h3vc]\n", argv[0]);
        return 2;
    }
    const char *model_dir = argv[1];
    h3_vsr_precision precision = H3_VSR_PRECISION_AUTO;
    if (argc >= 3) {
        if (!strcmp(argv[2], "int8")) precision = H3_VSR_PRECISION_INT8;
        else if (!strcmp(argv[2], "bf16")) precision = H3_VSR_PRECISION_BF16;
        else if (strcmp(argv[2], "auto")) {
            fprintf(stderr, "unknown precision %s\n", argv[2]);
            return 2;
        }
    }
    char error[1024] = {0};
    h3_vsr_info info;
    if (!h3_vsr_inspect_dir(model_dir, &info, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    if (precision == H3_VSR_PRECISION_AUTO)
        precision = info.int8_supported ? H3_VSR_PRECISION_INT8 : H3_VSR_PRECISION_BF16;
    char output[4096];
    if (argc == 4) snprintf(output, sizeof(output), "%s", argv[3]);
    else if (!h3_vsr_cache_path(model_dir, precision, output, sizeof(output))) return 1;
    fprintf(stderr, "%s (%s): %s cache -> %s\n", info.name, info.license_id,
            precision == H3_VSR_PRECISION_INT8 ? "int8" : "bf16", output);
    time_t started = time(NULL);
    if (!h3_vsr_build_cache(model_dir, output, precision, "h3_shaders.metal",
                            progress, NULL, error, sizeof(error))) {
        fprintf(stderr, "\n%s\n", error);
        return 1;
    }
    fprintf(stderr, "done in %ld s; matches model: %s\n", (long)(time(NULL) - started),
            h3_vsr_cache_matches(output, model_dir) ? "yes" : "NO");
    return 0;
}
