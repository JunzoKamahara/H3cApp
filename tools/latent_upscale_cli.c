/* h3_latent_upscale <in.h3lt> <out.h3lt> <out width px> <out height px>
 *                   [weights dir]
 * Upscales a clip latent file's video latent to the given pixel size (a
 * multiple of 16) with the LBH 3D-conv latent upscaler (see
 * h3_latent_upscale.h). Weights default to ~/models/open-h3/latent_upsampler. */
#include "h3_latent_upscale.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int on_progress(const char *phase, int done, int total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "[%6.1fs] %s %d/%d\n", now() - *(double *)opaque, phase, done, total);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s in.h3lt out.h3lt out_width out_height [weights dir]\n", argv[0]);
        return 2;
    }
    int width = atoi(argv[3]), height = atoi(argv[4]);
    if (width % 16 || height % 16 || width < 16 || height < 16) {
        fprintf(stderr, "the output size must be a multiple of 16 pixels\n");
        return 2;
    }
    char weights[1024];
    if (argc > 5) snprintf(weights, sizeof(weights), "%s", argv[5]);
    else snprintf(weights, sizeof(weights), "%s/models/open-h3/latent_upsampler",
                  getenv("HOME") ? getenv("HOME") : "");
    char error[512] = {0};
    double start = now();
    h3_latent_upscaler *u = h3_latent_upscaler_open(weights, "h3_shaders.metal", error, sizeof(error));
    if (!u) {
        fprintf(stderr, "cannot open the latent upscaler: %s\n", error);
        return 1;
    }
    fprintf(stderr, "[%6.1fs] weights loaded\n", now() - start);
    if (!h3_latent_upscale_file(u, argv[1], argv[2], height / 16, width / 16,
                                on_progress, &start, error, sizeof(error))) {
        fprintf(stderr, "latent upscale failed: %s\n", error);
        h3_latent_upscaler_free(u);
        return 1;
    }
    fprintf(stderr, "[%6.1fs] wrote %s\n", now() - start, argv[2]);
    h3_latent_upscaler_free(u);
    return 0;
}
