/* Runs pi-H3's pixel head (h3_pixel_head.h) on rows a generation dumped
 * (H3_DUMP_FINAL_VIDEO) and writes its frames, to set beside
 * tools/pi_h3/pixel_head_ref.py's - the same computation in numpy. Needs the
 * adapter and a dump; not in `make test`.
 *
 *   h3_pixel_head_probe <adapter.safetensors> <dump prefix> <modulation.f32>
 *                       <latent_t> <token_h> <token_w> <out.rgb>
 *
 * <dump prefix>.hidden.bf16 holds the rows before the final AdaLN;
 * <modulation.f32> is the head's AdaLN (10752 F32: scale, then shift), which
 * pixel_head_ref.py --features writes. <out.rgb> gets RGB24 frames. */
#include "h3_pixel_head.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int write_frames(void *opaque, const float *rgb, int frame_count,
                        int height, int width, char *error, size_t error_size) {
    (void)error; (void)error_size;
    size_t count = (size_t)frame_count * (size_t)height * (size_t)width * 3;
    unsigned char *bytes = malloc(count);
    if (!bytes) return 0;
    for (size_t index = 0; index < count; index++)
        bytes[index] = (unsigned char)lrintf(rgb[index] * 255.0f);
    int ok = fwrite(bytes, 1, count, opaque) == count;
    free(bytes);
    return ok;
}

static void *read_file(const char *path, size_t bytes) {
    FILE *file = fopen(path, "rb");
    void *data = file ? malloc(bytes) : NULL;
    if (data && fread(data, 1, bytes, file) != bytes) {
        free(data);
        data = NULL;
    }
    if (file) fclose(file);
    if (!data) fprintf(stderr, "cannot read %zu bytes of %s\n", bytes, path);
    return data;
}

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr, "usage: %s <adapter> <dump prefix> <modulation.f32> "
                "<latent_t> <token_h> <token_w> <out.rgb>\n", argv[0]);
        return 2;
    }
    int latent_t = atoi(argv[4]), token_h = atoi(argv[5]), token_w = atoi(argv[6]);
    size_t rows = (size_t)latent_t * (size_t)token_h * (size_t)token_w;
    char path[1024];
    snprintf(path, sizeof(path), "%s.hidden.bf16", argv[2]);
    uint16_t *hidden = read_file(path, rows * H3_PIXEL_HEAD_FEATURE * 2);
    float *modulation = read_file(argv[3], 2 * H3_PIXEL_HEAD_FEATURE * sizeof(float));
    FILE *out = fopen(argv[7], "wb");
    if (!hidden || !modulation || !out) return 1;
    char error[512];
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    int ok = h3_pixel_head_decode(argv[1], "h3_shaders.metal", hidden,
                                  modulation, latent_t, token_h, token_w,
                                  write_frames, out, NULL, NULL, error,
                                  sizeof(error));
    clock_gettime(CLOCK_MONOTONIC, &end);
    fclose(out);
    if (!ok) {
        fprintf(stderr, "pixel head failed: %s\n", error);
        return 1;
    }
    printf("%d frames of %dx%d in %.2f s -> %s\n",
           h3_pixel_head_frames(latent_t), token_w * 32, token_h * 32,
           (double)(end.tv_sec - begin.tv_sec) +
               (double)(end.tv_nsec - begin.tv_nsec) / 1e9, argv[7]);
    return 0;
}
