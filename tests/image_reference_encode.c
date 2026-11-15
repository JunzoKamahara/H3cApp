/* Phase 5 of plans/IMAGE_PLAN.ja.md: a reference image as the image engine
 * encodes it for the DiT - the size h3_image_reference_canvas gives, the
 * image VAE's encoder (the video VAE's), the posterior mean, normalized -
 * written out to be compared with the reference implementation's
 * (tools/image_phase5/compare_edit.py). Not in `make test`.
 *
 *   h3_image_reference_encode <vae dir> <image> <out.safetensors>
 *                             [max short edge] [shaders.metal]
 *
 * Writes "latent" F32 [24, h, w]. */
#include "h3_av_reader.h"
#include "h3_image.h"
#include "h3_video_encoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <vae dir> <image> <out.safetensors> "
                        "[max short edge] [shaders.metal]\n", argv[0]);
        return 2;
    }
    int limit = argc > 4 ? atoi(argv[4]) : 0;
    const char *shaders = argc > 5 ? argv[5] : "h3_shaders.metal";
    char error[512] = "";
    int width, height, encode_w, encode_h, vision_w, vision_h;
    float *pixels = NULL;
    if (!h3_av_visual_size(argv[2], &width, &height, error, sizeof(error)) ||
        !h3_image_reference_canvas(width, height, limit, &encode_w, &encode_h,
                                   &vision_w, &vision_h) ||
        !h3_av_read_image_f32(argv[2], encode_w, encode_h, H3_IMAGE_FIT_STRETCH,
                              &pixels, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error[0] ? error : "unusable size");
        return 1;
    }
    h3_video_latent latent;
    memset(&latent, 0, sizeof(latent));
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    if (!h3_video_vae_encode(argv[1], shaders, pixels, 1, encode_h, encode_w,
                             NULL, NULL, &latent, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("%dx%d encoded at %dx%d (latent %dx%d) in %.2f s\n", width, height,
           encode_w, encode_h, latent.width, latent.height,
           (double)(end.tv_sec - begin.tv_sec) +
           (double)(end.tv_nsec - begin.tv_nsec) * 1e-9);
    size_t bytes = (size_t)24 * (size_t)latent.height * (size_t)latent.width *
                   sizeof(float);
    char header[256];
    int length = snprintf(header, sizeof(header),
        "{\"latent\":{\"dtype\":\"F32\",\"shape\":[24,%d,%d],\"data_offsets\":[0,%zu]}}",
        latent.height, latent.width, bytes);
    while (length % 8) header[length++] = ' ';
    uint64_t header_length = (uint64_t)length;
    FILE *out = fopen(argv[3], "wb");
    int ok = out && fwrite(&header_length, 8, 1, out) == 1 &&
             fwrite(header, 1, (size_t)length, out) == (size_t)length &&
             fwrite(latent.values, 1, bytes, out) == bytes;
    if (out) ok = fclose(out) == 0 && ok;
    if (!ok) {
        fprintf(stderr, "cannot write %s\n", argv[3]);
        return 1;
    }
    printf("wrote %s\n", argv[3]);
    return 0;
}
