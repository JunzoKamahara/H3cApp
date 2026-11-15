/* Phase 2c of plans/IMAGE_PLAN.ja.md: the image DiT against a run of the
 * reference implementation (h3-image-mlx, traced by tools/image_phase0/
 * stage_trace.py). Not in `make test` (needs the 21 GB checkpoint).
 *
 *   h3_image_dit_check <transformer dir> <reference run dir> <out latent.safetensors>
 *                      [shaders.metal] [forced|free|both] [bf16|int8] [text encoder dir]
 *
 * The run dir holds embedding.safetensors ("embedding" BF16 [tokens, 5120])
 * and trace.safetensors (sigmas, noise, patches_after_step_N, head_out_step_N).
 *
 * forced: each step starts from the reference's own latent before that step,
 *   and the velocity is compared with the reference's - one step's error.
 * free:   the six steps run on from the reference's initial noise; the latent
 *   after each step is compared, and the last is written out for the VAE.
 * bf16:   the block linears in BF16 instead of the engine's int8 kernels
 *   (slower; tells the kernels' rounding from the rest).
 * A text encoder folder as the last argument runs the text encoder in this
 * process first, as a whole generation does; H3_CHECK_RNG=1 in the
 * environment replaces the reference's noise with the engine's own. Both
 * only tell where a failure of the whole pipeline comes from. */
#include "h3_dit.h"
#include "h3_dit_schedule.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { CHANNELS = 24, PATCH = 96, TEXT_DIM = 5120, STEPS = 6 };

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void *read_tensor(const h3_st_header *header, const char *name,
                         h3_dtype dtype, int ndim, uint64_t *shape) {
    char error[256] = "";
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor || tensor->dtype != dtype || tensor->ndim != ndim) {
        fprintf(stderr, "%s: missing or of another type\n", name);
        exit(1);
    }
    size_t bytes = (size_t)h3_st_tensor_elements(tensor) * h3_dtype_size(dtype);
    void *data = malloc(bytes);
    if (!data || !h3_st_read_data(header, tensor, data, bytes, error, sizeof(error))) {
        fprintf(stderr, "%s: %s\n", name, error);
        exit(1);
    }
    for (int axis = 0; axis < ndim; axis++) shape[axis] = tensor->shape[axis];
    return data;
}

static void compare(const char *label, const float *reference, const float *mine,
                    size_t count) {
    double dot = 0, rr = 0, mm = 0, dd = 0;
    for (size_t index = 0; index < count; index++) {
        double r = reference[index], m = mine[index];
        dot += r * m; rr += r * r; mm += m * m; dd += (m - r) * (m - r);
    }
    printf("%-28s relative L2 %.5f  cosine %.6f  rms %.4f (reference %.4f)\n",
           label, sqrt(dd / rr), dot / sqrt(rr * mm), sqrt(mm / (double)count),
           sqrt(rr / (double)count));
}

static void progress(const char *phase, int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == total) fprintf(stderr, "  %s %d/%d\n", phase, completed, total);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <transformer dir> <reference run dir> "
                        "<out latent.safetensors> [shaders.metal] [forced|free|both]\n", argv[0]);
        return 2;
    }
    const char *shaders = argc > 4 ? argv[4] : "h3_shaders.metal";
    const char *mode = argc > 5 ? argv[5] : "both";
    int bf16 = argc > 6 && !strcmp(argv[6], "bf16");
    char path[2048], error[512] = "";
    h3_st_header embedding_file, trace;
    snprintf(path, sizeof(path), "%s/embedding.safetensors", argv[2]);
    if (!h3_st_read_header(path, &embedding_file, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error); return 1;
    }
    snprintf(path, sizeof(path), "%s/trace.safetensors", argv[2]);
    if (!h3_st_read_header(path, &trace, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error); return 1;
    }
    uint64_t shape[8];
    h3_text_embedding text = {0};
    text.values = read_tensor(&embedding_file, "embedding", H3_DTYPE_BF16, 2, shape);
    text.tokens = (size_t)shape[0];
    text.width = (size_t)shape[1];
    float *sigma_values = read_tensor(&trace, "sigmas", H3_DTYPE_F32, 1, shape);
    if (shape[0] != STEPS + 1 || text.width != TEXT_DIM) {
        fprintf(stderr, "unexpected reference shapes\n"); return 1;
    }
    float *noise = read_tensor(&trace, "noise", H3_DTYPE_F32, 3, shape);
    int height = (int)shape[1], width = (int)shape[2];
    size_t latent_count = (size_t)CHANNELS * (size_t)height * (size_t)width;
    size_t rows = (size_t)(height / 2) * (size_t)(width / 2);
    printf("%zu text tokens, latent %dx%d (%zu image rows)\n", text.tokens,
           width, height, rows);
    if (getenv("H3_CHECK_RNG")) {
        /* Diagnostic: the engine's own noise in place of the reference's. */
        h3_rng rng;
        h3_rng_seed(&rng, 42);
        h3_rng_fill_normal(&rng, noise, latent_count);
        printf("using h3_rng noise\n");
    }

    h3_sigma_schedule sigmas = {.steps = STEPS};
    for (int index = 0; index <= STEPS; index++)
        sigmas.video[index] = sigmas.audio[index] = sigma_values[index];
    h3_layout_spec spec = {.text_len = (int)text.tokens, .latent_t = 1,
                           .latent_h = height, .latent_w = width,
                           .audio_t = 0, .frame_count = 5};
    h3_layout layout;
    if (!h3_layout_build(&spec, &layout, error, sizeof(error))) {
        fprintf(stderr, "layout: %s\n", error); return 1;
    }
    if (argc > 7) {
        /* Diagnostic: run the text encoder in this process first, as a whole
         * generation does, and use its embedding if argv[8] says so. */
        uint32_t ids[] = {32, 13753, 78540};
        h3_text_embedding warm = {0};
        if (!h3_text_encode_bf16(argv[7], shaders, ids, 3, NULL, NULL, &warm, error, sizeof(error))) {
            fprintf(stderr, "text encoder: %s\n", error); return 1;
        }
        printf("ran the text encoder first (%zu tokens)\n", warm.tokens);
        h3_text_embedding_free(&warm);
    }
    double began = now();
    h3_dit *dit = h3_dit_load_t2va(argv[1], shaders, &text, &layout, &sigmas,
        H3_DIT_BLOCKS, 1, 0, 1, 1.0f, bf16, bf16, bf16, 0, 0, 0, 0, 0, 0, 0, 0,
        NULL, 0, NULL, progress, NULL, error, sizeof(error));
    if (!dit) { fprintf(stderr, "load: %s\n", error); return 1; }
    printf("loaded in %.1f s (image model: %d)\n", now() - began,
           h3_dit_is_image_model(dit));

    float *latent = malloc(latent_count * sizeof(*latent));
    float *velocity = malloc(latent_count * sizeof(*velocity));
    float *mine = malloc(rows * PATCH * sizeof(*mine));
    char name[64];
    if (strcmp(mode, "free")) {
        printf("-- forced: one step from the reference's latent\n");
        for (int step = 0; step < STEPS; step++) {
            if (step == 0) {
                memcpy(latent, noise, latent_count * sizeof(*latent));
            } else {
                snprintf(name, sizeof(name), "patches_after_step_%d", step);
                float *before = read_tensor(&trace, name, H3_DTYPE_F32, 2, shape);
                h3_dit_unpatchify_video(before, CHANNELS, 1, height, width,
                                        latent, latent_count);
                free(before);
            }
            double start = now();
            if (!h3_dit_forward_image(dit, step, latent, velocity, error, sizeof(error))) {
                fprintf(stderr, "step %d: %s\n", step + 1, error); return 1;
            }
            double seconds = now() - start;
            h3_dit_patchify_video(velocity, CHANNELS, 1, height, width, mine, rows * PATCH);
            snprintf(name, sizeof(name), "head_out_step_%d", step + 1);
            float *reference = read_tensor(&trace, name, H3_DTYPE_F32, 2, shape);
            snprintf(name, sizeof(name), "velocity, step %d (%.1f s)", step + 1, seconds);
            compare(name, reference, mine, rows * PATCH);
            free(reference);
        }
    }
    if (strcmp(mode, "forced")) {
        printf("-- free: six steps from the reference's noise\n");
        memcpy(latent, noise, latent_count * sizeof(*latent));
        double start = now();
        for (int step = 0; step < STEPS; step++) {
            if (!h3_dit_forward_image(dit, step, latent, velocity, error, sizeof(error))) {
                fprintf(stderr, "step %d: %s\n", step + 1, error); return 1;
            }
            float dt = sigma_values[step] - sigma_values[step + 1];
            for (size_t index = 0; index < latent_count; index++)
                latent[index] += dt * velocity[index];
            h3_dit_patchify_video(latent, CHANNELS, 1, height, width, mine, rows * PATCH);
            snprintf(name, sizeof(name), "patches_after_step_%d", step + 1);
            float *reference = read_tensor(&trace, name, H3_DTYPE_F32, 2, shape);
            snprintf(name, sizeof(name), "latent after step %d", step + 1);
            compare(name, reference, mine, rows * PATCH);
            free(reference);
        }
        printf("six steps in %.1f s\n", now() - start);
        char header[256];
        int length = snprintf(header, sizeof(header),
            "{\"latent\":{\"dtype\":\"F32\",\"shape\":[%d,%d,%d],\"data_offsets\":[0,%zu]}}",
            CHANNELS, height, width, latent_count * sizeof(float));
        while (length % 8) header[length++] = ' ';
        uint64_t header_length = (uint64_t)length;
        FILE *out = fopen(argv[3], "wb");
        if (!out || fwrite(&header_length, 8, 1, out) != 1 ||
            fwrite(header, 1, (size_t)length, out) != (size_t)length ||
            fwrite(latent, sizeof(float), latent_count, out) != latent_count ||
            fclose(out)) {
            fprintf(stderr, "cannot write %s\n", argv[3]); return 1;
        }
        printf("wrote %s\n", argv[3]);
    }
    h3_dit_free(dit);
    return 0;
}
