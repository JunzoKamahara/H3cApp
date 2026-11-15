/* Phase 2a of plans/IMAGE_PLAN.ja.md: the image VAE decoder on a latent the
 * reference implementation produced. Not in `make test` (needs the weights).
 *
 *   h3_image_vae_check <vae dir> <latent.safetensors> <out.ppm> [shaders.metal]
 *
 * <latent.safetensors> holds "latent" F32 [24, H/16, W/16], normalized - what
 * h3-image-mlx's DiT stage writes. The PPM is 8-bit RGB, rounded as the
 * reference rounds; tools/image_phase2/image_compare.py compares it with the
 * reference's PNG. */
#include "h3_image_vae.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == total || completed % 20 == 0)
        fprintf(stderr, "tile %d/%d\n", completed, total);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <vae dir> <latent.safetensors> <out.ppm> "
                        "[shaders.metal]\n", argv[0]);
        return 2;
    }
    const char *shaders = argc > 4 ? argv[4] : "h3_shaders.metal";
    char error[512] = "";
    h3_st_header header;
    if (!h3_st_read_header(argv[2], &header, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    const h3_st_tensor *tensor = h3_st_find(&header, "latent");
    if (!tensor || tensor->dtype != H3_DTYPE_F32 || tensor->ndim != 3 ||
        tensor->shape[0] != 24) {
        fprintf(stderr, "%s has no \"latent\" F32 [24, h, w]\n", argv[2]);
        return 1;
    }
    int height = (int)tensor->shape[1], width = (int)tensor->shape[2];
    size_t count = (size_t)24 * (size_t)height * (size_t)width;
    float *latent = malloc(count * sizeof(*latent));
    if (!latent || !h3_st_read_data(&header, tensor, latent,
                                    count * sizeof(*latent), error,
                                    sizeof(error))) {
        fprintf(stderr, "cannot read the latent from %s: %s\n", argv[2], error);
        return 1;
    }

    h3_image_pixels pixels;
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    if (!h3_image_vae_decode(argv[1], shaders, latent, height, width, NULL,
                             progress, NULL, &pixels, error, sizeof(error))) {
        fprintf(stderr, "decode: %s\n", error);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("latent %dx%d -> %dx%d pixels in %.2f s\n", width, height,
           pixels.width, pixels.height,
           (double)(end.tv_sec - begin.tv_sec) +
           (double)(end.tv_nsec - begin.tv_nsec) * 1e-9);

    FILE *out = fopen(argv[3], "wb");
    if (!out) {
        fprintf(stderr, "cannot create %s\n", argv[3]);
        return 1;
    }
    fprintf(out, "P6\n%d %d\n255\n", pixels.width, pixels.height);
    size_t values = (size_t)pixels.width * (size_t)pixels.height * 3;
    unsigned char *bytes = malloc(values);
    for (size_t index = 0; index < values; index++)
        bytes[index] = (unsigned char)lrintf(pixels.rgb[index] * 255.0f);
    int ok = bytes && fwrite(bytes, 1, values, out) == values;
    ok = fclose(out) == 0 && ok;
    free(bytes);
    free(latent);
    h3_image_pixels_free(&pixels);
    h3_st_free_header(&header);
    if (!ok) {
        fprintf(stderr, "cannot write %s\n", argv[3]);
        return 1;
    }
    printf("wrote %s\n", argv[3]);
    return 0;
}
