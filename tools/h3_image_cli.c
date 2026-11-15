/* Text to image with MiniMax H3 Image (include/h3_image.h), from the command
 * line: for checking the engine without the app.
 *
 *   h3_image_cli <model dir> <prompt> <out.ppm> [WxH] [seed] [shaders.metal]
 *
 * <model dir> holds transformer/, vae/, text_encoder/ and tokenizer/ (see
 * h3_image.h). Writes an 8-bit PPM and, next to it, <out>.latent.f32: the
 * normalized latent [24, H/16, W/16] as raw little-endian F32. */
#include "h3_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void progress(const char *phase, int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == total || !strcmp(phase, "denoise"))
        fprintf(stderr, "%s %d/%d\n", phase, completed, total);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model dir> <prompt> <out.ppm> [WxH] [seed] "
                        "[shaders.metal]\n", argv[0]);
        return 2;
    }
    int width = 1024, height = 1024;
    if (argc > 4 && sscanf(argv[4], "%dx%d", &width, &height) != 2) {
        fprintf(stderr, "the size is <width>x<height>, e.g. 1024x1024\n");
        return 2;
    }
    unsigned long long seed = argc > 5 ? strtoull(argv[5], NULL, 10) : 0;
    const char *shaders = argc > 6 ? argv[6] : "h3_shaders.metal";
    char error[512] = "";
    h3_image_ctx *ctx = h3_image_load_dir(argv[1], shaders, error, sizeof(error));
    if (!ctx) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    h3_image_params params = {.prompt = argv[2], .width = width, .height = height,
                              .seed = seed, .progress = progress};
    h3_image_result result;
    if (!h3_image_generate(ctx, &params, &result, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        h3_image_free(ctx);
        return 1;
    }
    printf("%dx%d, seed %llu, %d prompt tokens, %d steps: text %.1f s, "
           "denoise %.1f s, decode %.1f s\n", result.width, result.height,
           (unsigned long long)result.seed, result.prompt_tokens, result.steps,
           result.text_seconds, result.denoise_seconds, result.decode_seconds);
    FILE *out = fopen(argv[3], "wb");
    size_t values = (size_t)result.width * (size_t)result.height * 3;
    int ok = out && fprintf(out, "P6\n%d %d\n255\n", result.width, result.height) > 0 &&
             fwrite(result.rgb, 1, values, out) == values;
    if (out) ok = fclose(out) == 0 && ok;
    char latent_path[2048];
    snprintf(latent_path, sizeof(latent_path), "%s.latent.f32", argv[3]);
    FILE *latent = fopen(latent_path, "wb");
    size_t count = (size_t)24 * (size_t)result.latent_height * (size_t)result.latent_width;
    ok = latent && fwrite(result.latent, sizeof(float), count, latent) == count && ok;
    if (latent) ok = fclose(latent) == 0 && ok;
    h3_image_result_free(&result);
    h3_image_free(ctx);
    if (!ok) {
        fprintf(stderr, "cannot write %s\n", argv[3]);
        return 1;
    }
    printf("wrote %s\n", argv[3]);
    return 0;
}
