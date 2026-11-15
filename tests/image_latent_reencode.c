/* Phase 4 of plans/IMAGE_PLAN.ja.md: how far the latent an image was
 * generated as is from the latent the video VAE's encoder gives for the
 * decoded picture - the two things a video could take as its first frame.
 * Not in `make test` (needs the VAE weights).
 *
 *   h3_image_latent_reencode <video VAE dir> <image.png> <latent.safetensors> [shaders.metal]
 *
 * <latent.safetensors> holds "latent" F32 [24, H/16, W/16], normalized, as
 * the app keeps it next to a generated image. */
#include "h3_av_reader.h"
#include "h3_safetensors.h"
#include "h3_video_encoder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <video VAE dir> <image.png> "
                        "<latent.safetensors> [shaders.metal]\n", argv[0]);
        return 2;
    }
    const char *shaders = argc > 4 ? argv[4] : "h3_shaders.metal";
    char error[512] = "";
    h3_st_header header;
    if (!h3_st_read_header(argv[3], &header, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    const h3_st_tensor *tensor = h3_st_find(&header, "latent");
    if (!tensor || tensor->dtype != H3_DTYPE_F32 || tensor->ndim != 3 ||
        tensor->shape[0] != 24) {
        fprintf(stderr, "%s has no \"latent\" F32 [24, h, w]\n", argv[3]);
        return 1;
    }
    int height = (int)tensor->shape[1], width = (int)tensor->shape[2];
    size_t plane = (size_t)height * (size_t)width, count = 24 * plane;
    float *generated = malloc(count * sizeof(*generated));
    if (!generated || !h3_st_read_data(&header, tensor, generated,
                                       count * sizeof(*generated), error,
                                       sizeof(error))) {
        fprintf(stderr, "cannot read the latent: %s\n", error);
        return 1;
    }
    float *pixels = NULL;
    if (!h3_av_read_image_f32(argv[2], width * 16, height * 16,
                              H3_IMAGE_FIT_COVER, &pixels, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    h3_video_latent encoded;
    memset(&encoded, 0, sizeof(encoded));
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    if (!h3_video_vae_encode(argv[1], shaders, pixels, 1, height * 16,
                             width * 16, NULL, NULL, &encoded, error,
                             sizeof(error))) {
        fprintf(stderr, "encode: %s\n", error);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    if (encoded.time != 1 || encoded.height != height || encoded.width != width) {
        fprintf(stderr, "unexpected encoded latent [24,%d,%d,%d]\n",
                encoded.time, encoded.height, encoded.width);
        return 1;
    }
    double dot = 0, gg = 0, ee = 0, dd = 0, worst = 0;
    int worst_channel = 0;
    for (int channel = 0; channel < 24; channel++) {
        double cg = 0, cd = 0;
        for (size_t index = 0; index < plane; index++) {
            double g = generated[(size_t)channel * plane + index];
            double e = encoded.values[(size_t)channel * plane + index];
            dot += g * e; gg += g * g; ee += e * e;
            cg += g * g; cd += (e - g) * (e - g);
        }
        dd += cd;
        if (cg > 0 && sqrt(cd / cg) > worst) {
            worst = sqrt(cd / cg);
            worst_channel = channel;
        }
    }
    printf("%dx%d pixels (latent %dx%d): encoded in %.2f s\n", width * 16,
           height * 16, width, height,
           (double)(end.tv_sec - begin.tv_sec) +
           (double)(end.tv_nsec - begin.tv_nsec) * 1e-9);
    printf("re-encoded against generated: relative L2 %.4f, cosine %.5f, "
           "rms %.4f against %.4f, worst channel %d (%.4f)\n",
           sqrt(dd / gg), dot / sqrt(gg * ee), sqrt(ee / (double)count),
           sqrt(gg / (double)count), worst_channel, worst);
    return 0;
}
