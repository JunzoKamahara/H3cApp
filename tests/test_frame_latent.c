/* h3_frame_latent_load (h3_params.first_frame_latent): which files it takes
 * as a frame's own latent, and which it leaves to the VAE encoder; and that
 * the conditioning cache's key follows H3_TEXT_PROJECTION. No GPU, no
 * weights. */
#include "h3_private.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int checks, failures;
#define CHECK(condition) do { checks++; if (!(condition)) { failures++; \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); } } while (0)

enum { H = 4, W = 6, COUNT = 24 * H * W };

/* A safetensors file with one tensor "latent" and the given metadata. */
static void write_file(const char *path, const char *metadata, const char *dtype,
                       int height, int width, const void *data, size_t bytes) {
    char header[512];
    int length = snprintf(header, sizeof(header),
        "{%s\"latent\":{\"dtype\":\"%s\",\"shape\":[24,%d,%d],\"data_offsets\":[0,%zu]}}",
        metadata, dtype, height, width, bytes);
    while (length % 8) header[length++] = ' ';
    uint64_t header_length = (uint64_t)length;
    FILE *file = fopen(path, "wb");
    if (!file || fwrite(&header_length, 8, 1, file) != 1 ||
        fwrite(header, 1, (size_t)length, file) != (size_t)length ||
        fwrite(data, 1, bytes, file) != bytes || fclose(file)) {
        fprintf(stderr, "cannot write %s\n", path);
        exit(1);
    }
}

int main(void) {
    const char *good = "\"__metadata__\":{\"latent_space\":\"minimax-h3-vae-24ch-16x\","
                       "\"normalization\":\"latents_mean_std\"},";
    char directory[512], path[600], reason[256];
    snprintf(directory, sizeof(directory), "%s/h3-frame-latent-XXXXXX",
             getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (!mkdtemp(directory)) { perror("mkdtemp"); return 1; }
    snprintf(path, sizeof(path), "%s/latent.safetensors", directory);
    float values[COUNT], loaded[COUNT];
    for (int index = 0; index < COUNT; index++) values[index] = (float)index * 0.01f - 2.0f;

    /* A latent of the canvas, of the H3 VAE's normalized space: taken. */
    write_file(path, good, "F32", H, W, values, sizeof(values));
    memset(loaded, 0, sizeof(loaded));
    CHECK(h3_frame_latent_load(path, H, W, loaded, reason, sizeof(reason)));
    CHECK(!memcmp(values, loaded, sizeof(values)));

    /* Another canvas: the image is encoded instead, and the reason says why. */
    reason[0] = '\0';
    CHECK(!h3_frame_latent_load(path, H, W + 2, loaded, reason, sizeof(reason)));
    CHECK(strstr(reason, "6x4") && strstr(reason, "8x4"));
    CHECK(!h3_frame_latent_load(path, H + 1, W, loaded, reason, sizeof(reason)));

    /* A file that does not say what its numbers are, or says another space. */
    write_file(path, "", "F32", H, W, values, sizeof(values));
    CHECK(!h3_frame_latent_load(path, H, W, loaded, reason, sizeof(reason)));
    write_file(path, "\"__metadata__\":{\"latent_space\":\"some-other-vae\","
                     "\"normalization\":\"latents_mean_std\"},", "F32", H, W, values, sizeof(values));
    CHECK(!h3_frame_latent_load(path, H, W, loaded, reason, sizeof(reason)));
    write_file(path, "\"__metadata__\":{\"latent_space\":\"minimax-h3-vae-24ch-16x\","
                     "\"normalization\":\"raw\"},", "F32", H, W, values, sizeof(values));
    CHECK(!h3_frame_latent_load(path, H, W, loaded, reason, sizeof(reason)));

    /* Another type, a damaged value, no file at all. */
    write_file(path, good, "F16", H, W, values, sizeof(values) / 2);
    CHECK(!h3_frame_latent_load(path, H, W, loaded, reason, sizeof(reason)));
    values[17] = NAN;
    write_file(path, good, "F32", H, W, values, sizeof(values));
    CHECK(!h3_frame_latent_load(path, H, W, loaded, reason, sizeof(reason)));
    CHECK(strstr(reason, "non-finite") != NULL);
    unlink(path);
    CHECK(!h3_frame_latent_load(path, H, W, loaded, reason, sizeof(reason)));
    rmdir(directory);

    /* Text rows from a projection (H3_TEXT_PROJECTION) are other rows than
     * the text encoder's, and than another projection's or another small
     * model's: none of them may be answered from the others' cache. */
    h3_params params = {0};
    unsetenv("H3_TEXT_PROJECTION");
    unsetenv("H3_TEXT_PROJECTION_ENCODER");
    char *own = h3_conditioning_key("a prompt", &params, 512, 512, 0);
    setenv("H3_TEXT_PROJECTION", "/nowhere/a.safetensors", 1);
    setenv("H3_TEXT_PROJECTION_ENCODER", "/nowhere/4b", 1);
    char *projected = h3_conditioning_key("a prompt", &params, 512, 512, 0);
    setenv("H3_TEXT_PROJECTION", "/nowhere/b.safetensors", 1);
    char *other_file = h3_conditioning_key("a prompt", &params, 512, 512, 0);
    setenv("H3_TEXT_PROJECTION_ENCODER", "/nowhere/4b-int8", 1);
    char *other_model = h3_conditioning_key("a prompt", &params, 512, 512, 0);
    unsetenv("H3_TEXT_PROJECTION");
    unsetenv("H3_TEXT_PROJECTION_ENCODER");
    char *own_again = h3_conditioning_key("a prompt", &params, 512, 512, 0);
    CHECK(own && projected && other_file && other_model && own_again);
    if (own && projected && other_file && other_model && own_again) {
        CHECK(strcmp(own, projected) != 0);
        CHECK(strcmp(projected, other_file) != 0);
        CHECK(strcmp(other_file, other_model) != 0);
        CHECK(strcmp(own, own_again) == 0);
    }
    free(own); free(projected); free(other_file); free(other_model);
    free(own_again);

    if (failures) {
        fprintf(stderr, "%d of %d frame latent checks failed\n", failures, checks);
        return 1;
    }
    printf("ok: %d checks (frame latent)\n", checks);
    return 0;
}
