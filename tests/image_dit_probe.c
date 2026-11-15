/* Phase-0 probe for plans/IMAGE_PLAN.ja.md: how far the existing loaders get
 * with a MiniMax H3 Image DiT (ComfyUI's single-file int8 + ConvRot,
 * curve-form AdaLN checkpoint), and where they stop. Not in `make test`.
 *
 *   h3_image_dit_probe <transformer dir> [shaders.metal] [load]
 *   h3_image_dit_probe <transformer dir> <shaders.metal> dump <tensor> <out>
 *
 * It reads headers and small tensors only, so a sparse stand-in of the
 * checkpoint (real header, real small tensors, holes for the block
 * matrices) is enough. "load" also runs the video loader (h3_dit_load_t2va,
 * SSD streaming) on it to record the error it ends with. "dump" needs the
 * real checkpoint: it writes one weight as the store restores it (int8 +
 * ConvRot undone, QKV rows reordered) as raw BF16, to compare with the
 * reference implementation's reading of the same bytes. */
#include "h3.h"
#include "h3_adaln_curve.h"
#include "h3_dit.h"
#include "h3_dit_schedule.h"
#include "h3_gpu.h"
#include "h3_hyperflow.h"
#include "h3_model_family.h"
#include "h3_weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { HIDDEN = 5376, INNER = 7168, FFN = 14336, HEAD_DIM = 128,
       VIDEO_PATCH = 96, AUDIO_CHANNELS = 32, TEXT_DIM = 5120,
       PDD_STEPS = 6 };

/* The released sampler's sigmas (the checkpoint's "config" metadata). */
static const float pdd_sigmas[PDD_STEPS + 1] = {
    1.0f, 0.9675236940383911f, 0.8875485062599182f, 0.75f,
    0.5327739119529724f, 0.23200887441635132f, 0.0f,
};

static int failures;

static void shape_text(const h3_st_tensor *tensor, char *out, size_t size) {
    size_t used = 0;
    out[0] = '\0';
    for (int d = 0; d < tensor->ndim && used < size; d++)
        used += (size_t)snprintf(out + used, size - used, "%s%llu", d ? "," : "",
                                 (unsigned long long)tensor->shape[d]);
}

/* What the video loader asks the store for, by name and shape. */
static void expect(const h3_weight_store *store, const char *name, int ndim,
                   uint64_t rows, uint64_t columns) {
    const h3_st_tensor *tensor = h3_weight_find(store, name, NULL);
    char want[64], have[64];
    if (ndim == 2)
        snprintf(want, sizeof(want), "%llu,%llu", (unsigned long long)rows,
                 (unsigned long long)columns);
    else
        snprintf(want, sizeof(want), "%llu", (unsigned long long)rows);
    if (!tensor) {
        printf("  MISSING  %-40s wants [%s]\n", name, want);
        failures++;
        return;
    }
    shape_text(tensor, have, sizeof(have));
    int same = tensor->ndim == ndim && tensor->shape[0] == rows &&
               (ndim == 1 || tensor->shape[1] == columns);
    printf("  %-8s %-40s %s [%s]%s%s%s\n", same ? "ok" : "SHAPE", name,
           h3_dtype_name(tensor->dtype), have, same ? "" : ", wants [",
           same ? "" : want, same ? "" : "]");
    if (!same) failures++;
}

static void report_tensor(const char *what, h3_gpu_tensor *tensor,
                          size_t count, const char *error) {
    if (!tensor) {
        printf("  FAILED   %s: %s\n", what, error);
        failures++;
        return;
    }
    float *values = malloc(count * sizeof(*values));
    double sum = 0.0;
    size_t bad = 0;
    if (values && h3_gpu_tensor_read_f32(tensor, values, count))
        for (size_t i = 0; i < count; i++) {
            if (!isfinite(values[i])) bad++;
            else sum += (double)values[i] * values[i];
        }
    printf("  ok       %s: %zu values, L2 %.4g, non-finite %zu\n", what, count,
           sqrt(sum), bad);
    free(values);
    h3_gpu_tensor_free(tensor);
}

static void load_progress(const char *phase, int completed, int total,
                          void *opaque) {
    (void)opaque;
    if (completed == total) fprintf(stderr, "  [%s %d/%d]\n", phase, completed, total);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <transformer dir> [shaders.metal] [load]\n",
                argv[0]);
        return 2;
    }
    const char *directory = argv[1];
    const char *shaders = argc > 2 ? argv[2] : "build/bin/h3_shaders.metal";
    int run_load = argc > 3 && !strcmp(argv[3], "load");
    char error[512] = "";

    h3_weight_store *store = h3_weight_store_open(directory, error, sizeof(error));
    if (!store) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    if (argc > 5 && !strcmp(argv[3], "dump")) {
        const h3_st_tensor *tensor = h3_weight_find(store, argv[4], NULL);
        h3_gpu *dump_gpu = h3_gpu_create(shaders, error, sizeof(error));
        if (!tensor || !dump_gpu) {
            fprintf(stderr, "%s\n", tensor ? error : "no such tensor");
            return 1;
        }
        size_t count = 1;
        for (int d = 0; d < tensor->ndim; d++) count *= (size_t)tensor->shape[d];
        h3_gpu_tensor *weight = h3_weight_load_bf16(store, dump_gpu, argv[4],
                                                    tensor->ndim, tensor->shape,
                                                    error, sizeof(error));
        uint16_t *values = weight ? malloc(count * sizeof(*values)) : NULL;
        FILE *out = values ? fopen(argv[5], "wb") : NULL;
        if (!out || !h3_gpu_tensor_read_bf16(weight, values, count) ||
            fwrite(values, sizeof(*values), count, out) != count || fclose(out)) {
            fprintf(stderr, "cannot dump %s: %s\n", argv[4], error);
            return 1;
        }
        printf("%s: %zu BF16 values -> %s\n", argv[4], count, argv[5]);
        return 0;
    }
    puts("== weight store");
    const h3_st_header *header = NULL;
    h3_weight_store_tensor(store, 0, &header);
    printf("  %zu shard(s), %zu tensors, first shard %s\n",
           h3_weight_store_shards(store), h3_weight_store_tensor_count(store),
           header ? header->path : "?");
    const char *config = header ? h3_st_metadata(header, "config") : NULL;
    printf("  metadata config: %s\n", config ? config : "(none)");

    puts("== h3_gguf_inspect (what an importer sees)");
    h3_gguf_info info;
    if (!header || !h3_gguf_inspect(header->path, &info, error, sizeof(error))) {
        printf("  FAILED   %s\n", error);
        failures++;
    } else {
        printf("  role %d (1 = DiT), %s, %d blocks, curve AdaLN %d, dense AdaLN %d, "
               "comfy_int8 %d, hyperflow %d, still image %d\n", (int)info.role,
               info.main_type, info.dit_blocks, info.curve_adaln,
               info.dense_adaln, info.comfy_int8, info.hyperflow,
               info.still_image);
    }

    puts("== model family");
    h3_hyperflow_info hyperflow;
    if (!h3_hyperflow_inspect(store, &hyperflow, error, sizeof(error))) {
        printf("  FAILED   %s\n", error);
        failures++;
    } else {
        const h3_model_family_desc *family = h3_model_family_of(hyperflow.two_time);
        printf("  two_time %d, %u blocks -> family \"%s\"\n", hyperflow.two_time,
               hyperflow.block_count, family->info.label);
    }

    puts("== tensors the video loader reads");
    expect(store, "condition_proj.weight", 2, HIDDEN, TEXT_DIM);
    expect(store, "condition_proj.bias", 1, HIDDEN, 0);
    static const char *const refiner[] = {"0", "1"};
    static const unsigned blocks[] = {0, 49};
    char name[128];
    for (size_t i = 0; i < 2 + sizeof(blocks) / sizeof(*blocks); i++) {
        char prefix[64];
        if (i < 2)
            snprintf(prefix, sizeof(prefix), "token_refiner.blocks.%s.", refiner[i]);
        else
            snprintf(prefix, sizeof(prefix), "blocks.%u.", blocks[i - 2]);
#define EXPECT(suffix, ndim, rows, columns) \
    (snprintf(name, sizeof(name), "%s" suffix, prefix), \
     expect(store, name, ndim, rows, columns))
        EXPECT("norm1.weight", 1, HIDDEN, 0);
        EXPECT("norm2.weight", 1, HIDDEN, 0);
        EXPECT("attn.qkv_proj.weight", 2, INNER * 3, HIDDEN);
        EXPECT("attn.q_norm.weight", 1, HEAD_DIM, 0);
        EXPECT("attn.k_norm.weight", 1, HEAD_DIM, 0);
        EXPECT("attn.out_proj.weight", 2, HIDDEN, INNER);
        EXPECT("mlp.fc1.weight", 2, FFN * 2, HIDDEN);
        EXPECT("mlp.fc2.weight", 2, HIDDEN, FFN);
#undef EXPECT
    }
    expect(store, "token_refiner.final_norm.weight", 1, HIDDEN, 0);
    expect(store, "rope.inv_freq", 1, 16, 0);
    expect(store, "video_patch_proj.weight", 2, HIDDEN, VIDEO_PATCH);
    expect(store, "video_patch_proj.bias", 1, HIDDEN, 0);
    expect(store, "audio_patch_proj.weight", 2, HIDDEN, AUDIO_CHANNELS);
    expect(store, "audio_patch_proj.bias", 1, HIDDEN, 0);
    expect(store, "final_layer.norm.weight", 1, HIDDEN, 0);
    expect(store, "final_layer.video_out.weight", 2, VIDEO_PATCH, HIDDEN);
    expect(store, "final_layer.video_out.bias", 1, VIDEO_PATCH, 0);
    expect(store, "final_layer.audio_out.weight", 2, AUDIO_CHANNELS, HIDDEN);
    expect(store, "final_layer.audio_out.bias", 1, AUDIO_CHANNELS, 0);
    int name_failures = failures;

    h3_gpu *gpu = h3_gpu_create(shaders, error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }

    puts("== ComfyUI int8 + ConvRot");
    static const char *const linears[] = {
        "blocks.0.attn.qkv_proj.weight", "blocks.0.attn.out_proj.weight",
        "blocks.0.mlp.fc1.weight", "blocks.0.mlp.fc2.weight",
        "token_refiner.blocks.0.attn.qkv_proj.weight",
    };
    for (size_t i = 0; i < sizeof(linears) / sizeof(*linears); i++) {
        const h3_st_tensor *tensor = h3_weight_find(store, linears[i], NULL);
        if (!tensor) continue;
        uint32_t inner = 0, head_dim = 0, group = 0;
        int int8 = h3_weight_is_comfy_int8(store, tensor);
        int reorder = h3_weight_qkv_reorder(store, linears[i], tensor, &inner,
                                            &head_dim);
        h3_gpu_tensor *scales = NULL;
        float first[4] = {0};
        int got = int8 && h3_weight_comfy_int8_scales(store, gpu, tensor, 0, 4,
                                                      &group, &scales, error,
                                                      sizeof(error)) &&
                  h3_gpu_tensor_read_f32(scales, first, 4);
        printf("  %-44s %s int8 %d, qkv reorder %d (inner %u, head %u)",
               linears[i], h3_dtype_name(tensor->dtype), int8, reorder, inner,
               head_dim);
        if (int8 && got)
            printf(", ConvRot group %u, scales %.3g %.3g %.3g %.3g\n", group,
                   (double)first[0], (double)first[1], (double)first[2],
                   (double)first[3]);
        else if (int8)
            printf(", scales FAILED: %s\n", error), failures++;
        else
            printf("\n");
        h3_gpu_tensor_free(scales);
    }

    puts("== curve-form AdaLN");
    printf("  h3_adaln_curve_in_store: %d\n", h3_adaln_curve_in_store(store));
    h3_adaln_curve *curve = h3_adaln_curve_from_store(store, error, sizeof(error));
    if (!curve) {
        printf("  FAILED   %s\n", error);
        failures++;
    } else {
        printf("  grid %u, rank %u\n", h3_adaln_curve_grid(curve),
               h3_adaln_curve_rank(curve));
        uint32_t rank = h3_adaln_curve_rank(curve);
        const uint32_t block_rows =
            H3_DIT_MODALITIES * H3_DIT_ADALN_SLOTS * H3_DIT_HIDDEN;
        struct { const char *prefix; uint32_t rows; } projections[] = {
            {"blocks.0.adaln_proj.linear", block_rows},
            {"blocks.49.adaln_proj.linear", block_rows},
            {"final_layer.adaln_proj.linear", 2u * H3_DIT_HIDDEN},
        };
        for (size_t i = 0; i < sizeof(projections) / sizeof(*projections); i++) {
            char what[160];
            snprintf(what, sizeof(what), "%s.weight", projections[i].prefix);
            report_tensor(what, h3_adaln_curve_load_weight(
                              curve, gpu, projections[i].prefix,
                              projections[i].rows, error, sizeof(error)),
                          (size_t)projections[i].rows * rank, error);
            snprintf(what, sizeof(what), "%s.bias", projections[i].prefix);
            report_tensor(what, h3_adaln_curve_load_bias(
                              curve, gpu, projections[i].prefix,
                              projections[i].rows, error, sizeof(error)),
                          projections[i].rows, error);
        }
        /* The PDD sigmas as an ordinary schedule; the audio grid is unused
         * by an image model and only mirrors the video one here. */
        h3_sigma_schedule sigmas = {.steps = PDD_STEPS};
        memcpy(sigmas.video, pdd_sigmas, sizeof(pdd_sigmas));
        memcpy(sigmas.audio, pdd_sigmas, sizeof(pdd_sigmas));
        for (int visual = 0; visual <= 1; visual++) {
            h3_dit_schedule *schedule = h3_dit_schedule_precompute_curve(
                NULL, curve, gpu, &sigmas, visual, 0, NULL, NULL, error,
                sizeof(error));
            if (!schedule) {
                printf("  FAILED   PDD schedule (visual condition %d): %s\n",
                       visual, error);
                failures++;
                continue;
            }
            printf("  ok       PDD schedule, visual condition %d: %d steps, "
                   "%u time rows, video rows", visual,
                   h3_dit_schedule_steps(schedule),
                   h3_dit_schedule_time_rows(schedule));
            for (int step = 0; step < PDD_STEPS; step++)
                printf(" %u", h3_dit_schedule_video_row(schedule, step));
            if (visual)
                printf(", condition row %u",
                       h3_dit_schedule_visual_condition_row(schedule, 0));
            printf("\n");
            h3_dit_schedule_free(schedule);
        }
    }

    puts("== output head");
    {
        uint64_t one[2] = {VIDEO_PATCH, HIDDEN};
        h3_gpu_tensor *single = h3_weight_load_f32(
            store, gpu, "final_layer.video_out.weight", 2, one, error,
            sizeof(error));
        printf("  as one head [%d,%d]: %s%s\n", VIDEO_PATCH, HIDDEN,
               single ? "loaded" : "refused - ", single ? "" : error);
        h3_gpu_tensor_free(single);
        uint64_t six[2] = {VIDEO_PATCH * PDD_STEPS, HIDDEN};
        h3_gpu_tensor *heads = h3_weight_load_f32(
            store, gpu, "final_layer.video_out.weight", 2, six, error,
            sizeof(error));
        size_t count = (size_t)VIDEO_PATCH * PDD_STEPS * HIDDEN;
        float *values = heads ? malloc(count * sizeof(*values)) : NULL;
        if (!values || !h3_gpu_tensor_read_f32(heads, values, count)) {
            printf("  FAILED   six heads [%d,%d]: %s\n", VIDEO_PATCH * PDD_STEPS,
                   HIDDEN, error);
            failures++;
        } else {
            /* Six full heads would have similar norms; a first head plus
             * differences shows up as small later chunks. */
            printf("  as six heads [%d,%d]: loaded; L2 per head",
                   VIDEO_PATCH * PDD_STEPS, HIDDEN);
            for (int head = 0; head < PDD_STEPS; head++) {
                double sum = 0.0;
                const float *rows = values + (size_t)head * VIDEO_PATCH * HIDDEN;
                for (size_t i = 0; i < (size_t)VIDEO_PATCH * HIDDEN; i++)
                    sum += (double)rows[i] * rows[i];
                printf(" %.4g", sqrt(sum));
            }
            printf("\n");
        }
        free(values);
        h3_gpu_tensor_free(heads);
    }
    h3_adaln_curve_free(curve);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);

    if (run_load) {
        puts("== h3_dit_load_t2va (SSD streaming) on this checkpoint");
        enum { TEXT_ROWS = 6 };
        uint16_t *text_values = malloc((size_t)TEXT_ROWS * TEXT_DIM * sizeof(uint16_t));
        if (!text_values) return 1;
        for (size_t i = 0; i < (size_t)TEXT_ROWS * TEXT_DIM; i++)
            text_values[i] = (uint16_t)(i % 3 ? 0x3C24 : 0xBC24); /* +-0.01 */
        h3_text_embedding text = {TEXT_ROWS, TEXT_DIM, text_values, {0}, NULL};
        /* The smallest layout the video path builds: 5 frames, 2 audio steps. */
        h3_layout_spec spec = {.text_len = TEXT_ROWS, .latent_t = 2, .latent_h = 2,
                               .latent_w = 2, .audio_t = 8, .frame_count = 5};
        h3_layout layout;
        h3_sigma_schedule sigmas = {.steps = PDD_STEPS};
        memcpy(sigmas.video, pdd_sigmas, sizeof(pdd_sigmas));
        memcpy(sigmas.audio, pdd_sigmas, sizeof(pdd_sigmas));
        if (!h3_layout_build(&spec, &layout, error, sizeof(error))) {
            printf("  layout: %s\n", error);
            return 1;
        }
        h3_dit *dit = h3_dit_load_t2va(directory, shaders, &text, &layout,
                                       &sigmas, H3_DIT_BLOCKS, 1, 0, 1, 1.0f,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       NULL, 0, NULL, load_progress, NULL,
                                       error, sizeof(error));
        printf("  %s%s\n", dit ? "loaded" : "stopped: ", dit ? "" : error);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values);
    }
    printf("== %d problem(s), %d of them tensors the video loader would miss\n",
           failures, name_failures);
    return 0;
}
