/* HyperFlow H3 Flash on real weights against a ComfyUI reference dump
 * (HYPERFLOW_PLAN.ja.md appendix A.2; fixture made by the reference kit's
 * make_fixture.py from a t2v run: 256x256, 39 frames, 6 steps).
 *
 * H3_HYPERFLOW_DIR      transformer directory with model.gguf (FL2VA)
 * H3_HYPERFLOW_FIXTURE  t2v_fixture.safetensors
 * H3_HYPERFLOW_STRICT=1 missing inputs fail instead of skipping
 * H3_HYPERFLOW_INT8=1   the production int8 resident path (looser bounds)
 * H3_HYPERFLOW_SSD=1    SSD streaming: BF16 matrices expanded per block, the
 *                       only BF16 path that fits a 24 GB Mac
 * Without either, BF16 resident (about 27 GB of weights).
 *
 * The refined text is taken from the dump (ComfyUI refines it outside the
 * hooked module), so this checks the patch projections, the two-time AdaLN,
 * every dense and factored block and the final layer, steps 0 and 1. */
#include "h3_dit.h"
#include "h3_host.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { TEXT = 24, HIDDEN = 5376, LT = 12, LH = 16, LW = 16, AT = 65,
       FRAMES = 39, VIDEO_ROWS = 768, AUDIO_ROWS = 130, SEQ = 922 };

static int failures;

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_hyperflow_real.c: %s\n", message);
    exit(1);
}

static int strict(void) {
    const char *s = getenv("H3_HYPERFLOW_STRICT");
    return s && *s && strcmp(s, "0") != 0;
}

static float bf16_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

/* Any fixture tensor as F32 (BF16 or F32 stored). */
static float *load(const h3_st_header *header, const char *name,
                   size_t expected) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor || h3_st_tensor_elements(tensor) != expected) {
        fprintf(stderr, "fixture tensor %s missing or wrong size\n", name);
        die("bad fixture");
    }
    float *out = malloc(expected * sizeof(*out));
    char error[512];
    if (!out) die("out of memory");
    if (tensor->dtype == H3_DTYPE_F32) {
        if (!h3_st_read_data(header, tensor, out, expected * sizeof(*out),
                             error, sizeof(error))) die(error);
    } else if (tensor->dtype == H3_DTYPE_BF16) {
        uint16_t *raw = malloc(expected * sizeof(*raw));
        if (!raw || !h3_st_read_data(header, tensor, raw,
                                     expected * sizeof(*raw), error,
                                     sizeof(error))) die("fixture read");
        for (size_t i = 0; i < expected; i++) out[i] = bf16_f32(raw[i]);
        free(raw);
    } else {
        die("fixture dtype");
    }
    return out;
}

static uint16_t *load_bf16_raw(const h3_st_header *header, const char *name,
                               size_t expected) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    char error[512];
    if (!tensor || tensor->dtype != H3_DTYPE_BF16 ||
        h3_st_tensor_elements(tensor) != expected) die("bad BF16 fixture");
    uint16_t *raw = malloc(expected * sizeof(*raw));
    if (!raw || !h3_st_read_data(header, tensor, raw, expected * sizeof(*raw),
                                 error, sizeof(error))) die(error);
    return raw;
}

/* rel-L2 and the cosine of got against want, over [begin, end) rows of
 * width columns; bound 0 only reports. */
static double compare(const char *label, const float *got, const float *want,
                      size_t begin, size_t end, size_t width, double bound) {
    double err = 0, ref = 0, dot = 0, own = 0;
    for (size_t i = begin * width; i < end * width; i++) {
        double d = (double)got[i] - want[i];
        err += d * d; ref += (double)want[i] * want[i];
        dot += (double)got[i] * want[i]; own += (double)got[i] * got[i];
    }
    double rel = sqrt(err / (ref > 1e-30 ? ref : 1e-30));
    double cosine = dot / sqrt((own > 1e-30 ? own : 1e-30) *
                               (ref > 1e-30 ? ref : 1e-30));
    int bad = bound > 0 && !(rel < bound);
    printf("  %-26s rel-L2 %.4e  cos %.6f%s\n", label, rel, cosine,
           bad ? "  <-- over bound" : "");
    if (bad) failures++;
    return rel;
}

static float *read_dump(const char *path, size_t count) {
    uint16_t *raw = malloc(count * sizeof(*raw));
    float *out = malloc(count * sizeof(*out));
    FILE *file = fopen(path, "rb");
    if (!raw || !out || !file || fread(raw, sizeof(*raw), count, file) != count)
        die("cannot read a block dump");
    fclose(file);
    for (size_t i = 0; i < count; i++) out[i] = bf16_f32(raw[i]);
    free(raw);
    return out;
}

int main(void) {
    const char *dir = getenv("H3_HYPERFLOW_DIR");
    const char *fixture_path = getenv("H3_HYPERFLOW_FIXTURE");
    if (!dir || !*dir || !fixture_path || !*fixture_path ||
        access(fixture_path, R_OK) != 0) {
        if (strict()) die("set H3_HYPERFLOW_DIR and H3_HYPERFLOW_FIXTURE");
        puts("SKIP: HyperFlow real-weight fixture (set H3_HYPERFLOW_DIR and "
             "H3_HYPERFLOW_FIXTURE)");
        return 0;
    }
    int int8 = getenv("H3_HYPERFLOW_INT8") && strcmp(getenv("H3_HYPERFLOW_INT8"), "0");
    int ssd = getenv("H3_HYPERFLOW_SSD") && strcmp(getenv("H3_HYPERFLOW_SSD"), "0");
    if (ssd) int8 = 0;
    char error[1024];
    h3_st_header fx;
    if (!h3_st_read_header(fixture_path, &fx, error, sizeof(error))) die(error);

    const size_t VE = 24u * LT * LH * LW, AE = 32u * 2u * AT;
    float *video[2] = {load(&fx, "step0.video_latent", VE),
                       load(&fx, "step1.video_latent", VE)};
    float *audio[2] = {load(&fx, "step0.audio_latent", AE),
                       load(&fx, "step1.audio_latent", AE)};
    float *want_video[2] = {load(&fx, "step0.final_video", VIDEO_ROWS * 96u),
                            load(&fx, "step1.final_video", VIDEO_ROWS * 96u)};
    float *want_audio[2] = {load(&fx, "step0.final_audio", AUDIO_ROWS * 32u),
                            load(&fx, "step1.final_audio", AUDIO_ROWS * 32u)};
    float *sigmas_ref = load(&fx, "sample_sigmas", 7);
    uint16_t *context = load_bf16_raw(&fx, "context", (size_t)TEXT * HIDDEN);

    /* Patch layouts: our row packing must equal the reference projection
     * inputs, otherwise every later comparison is meaningless. */
    {
        float *rows = malloc(VIDEO_ROWS * 96u * sizeof(*rows));
        float *arows = malloc(AUDIO_ROWS * 32u * sizeof(*arows));
        float *want_v = load(&fx, "video_patch_in", VIDEO_ROWS * 96u);
        float *want_a = load(&fx, "audio_patch_in", AUDIO_ROWS * 32u);
        if (!h3_dit_patchify_video(video[0], 24, LT, LH, LW, rows,
                                   VIDEO_ROWS * 96u) ||
            !h3_dit_pack_audio(audio[0], 32, AT, arows, AUDIO_ROWS * 32u))
            die("patchify");
        puts("patch packing vs reference projection inputs:");
        compare("video patch rows", rows, want_v, 0, VIDEO_ROWS, 96, 1e-6);
        compare("audio rows", arows, want_a, 0, AUDIO_ROWS, 32, 1e-6);
        free(rows); free(arows); free(want_v); free(want_a);
    }

    h3_sigma_schedule sigmas;
    if (!h3_hyperflow_schedule_build(6, &sigmas)) die("schedule");
    {
        double worst = 0;
        for (int i = 0; i < 7; i++) {
            double d = fabs((double)sigmas.video[i] - sigmas_ref[i]);
            if (d > worst) worst = d;
        }
        printf("video sigma grid vs reference: max |diff| %.3g\n", worst);
        if (worst > 1e-6) failures++;
    }

    uint16_t *dummy = calloc((size_t)TEXT * 5120u, sizeof(*dummy));
    h3_text_embedding text = {TEXT, 5120, dummy, {0}, NULL};
    h3_layout_spec spec = {TEXT, LT, LH, LW, AT, FRAMES, NULL, 0, NULL, 0,
                           0, 0.0};
    h3_layout layout;
    if (!h3_layout_build(&spec, &layout, error, sizeof(error))) die(error);
    if (layout.seq_len != SEQ) die("sequence length");

    char dump_prefix[512];
    snprintf(dump_prefix, sizeof(dump_prefix), "%s/h3hf-%d", 
             getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", (int)getpid());
    setenv("H3_DUMP_BLOCK_HIDDEN", dump_prefix, 1);
    unsetenv("H3_ATTENTION_CACHE");
    unsetenv("H3_ATTENTION_CACHE_DIR");
    if (!int8) {
        setenv("H3_DISABLE_FUSED_MLP", "1", 1);
        setenv("H3_DIT_F32_FINAL", "1", 1);
    }
    int bf16 = !int8;
    h3_dit *dit = h3_dit_load_t2va(
        dir, "h3_shaders.metal", &text, &layout, &sigmas, 50, 1, 0, ssd, 1.0f,
        bf16, bf16, bf16, 0, 0, 0, 0, 0, 0, 0, 0, NULL, 0, NULL, NULL, NULL,
        error, sizeof(error));
    if (!dit) die(error);
    if (h3_dit_video_elements(dit) != VE || h3_dit_audio_elements(dit) != AE)
        die("latent geometry");
    if (!h3_dit_debug_set_refined_text(dit, context, (size_t)TEXT * HIDDEN,
                                       error, sizeof(error))) die(error);
    printf("mode: %s\n", int8 ? "int8 resident (production)" :
           ssd ? "SSD streaming (BF16 matrices)" : "BF16 resident");

    /* Measured 2026-11-09 (plan appendix A.3/A.4): BF16 streaming worst
     * 5.2 % (block 46) and 5.4 % (step-1 audio head), int8 resident 8.9 %
     * and 7.9 %. BF16 rounding of the residual stream alone grows to a few
     * percent over 47 blocks; the released H3 parity test allows 25 %. */
    double head_bound = int8 ? 0.15 : 0.08;
    double block_bound = int8 ? 0.15 : 0.08;
    float *vel_v = malloc(VE * sizeof(float)), *vel_a = malloc(AE * sizeof(float));
    float *rows_v = malloc(VIDEO_ROWS * 96u * sizeof(float));
    float *rows_a = malloc(AUDIO_ROWS * 32u * sizeof(float));
    for (int step = 0; step < 2; step++) {
        if (!h3_dit_forward(dit, step, video[step], audio[step], vel_v, vel_a,
                            error, sizeof(error))) die(error);
        if (step == 0) {
            static const int blocks[] = {0, 1, 2, 3, 4, 5, 18, 23, 46};
            puts("step 0 block outputs (rows: text | audio | video):");
            for (size_t i = 0; i < sizeof(blocks) / sizeof(blocks[0]); i++) {
                char path[600], name[32], label[64];
                snprintf(path, sizeof(path), "%s.block%02d.bf16", dump_prefix,
                         blocks[i]);
                snprintf(name, sizeof(name), "block%02d.out", blocks[i]);
                float *got = read_dump(path, (size_t)SEQ * HIDDEN);
                float *want = load(&fx, name, (size_t)SEQ * HIDDEN);
                snprintf(label, sizeof(label), "block %2d all", blocks[i]);
                compare(label, got, want, 0, SEQ, HIDDEN, block_bound);
                snprintf(label, sizeof(label), "block %2d text", blocks[i]);
                compare(label, got, want, 0, TEXT, HIDDEN, 0);
                snprintf(label, sizeof(label), "block %2d audio", blocks[i]);
                compare(label, got, want, TEXT, TEXT + AUDIO_ROWS, HIDDEN, 0);
                snprintf(label, sizeof(label), "block %2d video", blocks[i]);
                compare(label, got, want, TEXT + AUDIO_ROWS, SEQ, HIDDEN, 0);
                free(got); free(want);
            }
            for (unsigned b = 0; b < 50; b++) {
                char path[600];
                snprintf(path, sizeof(path), "%s.block%02u.bf16", dump_prefix, b);
                unlink(path);
            }
        }
        if (!h3_dit_patchify_video(vel_v, 24, LT, LH, LW, rows_v,
                                   VIDEO_ROWS * 96u) ||
            !h3_dit_pack_audio(vel_a, 32, AT, rows_a, AUDIO_ROWS * 32u))
            die("patchify velocity");
        /* The engine's velocity is the reference final-layer output itself
         * (the ComfyUI wrapper negates and rescales it afterwards). */
        printf("step %d final heads:\n", step);
        compare("video head", rows_v, want_video[step], 0, VIDEO_ROWS, 96,
                head_bound);
        compare("audio head", rows_a, want_audio[step], 0, AUDIO_ROWS, 32,
                head_bound);
    }
    h3_gpu_stats stats;
    if (h3_dit_get_gpu_stats(dit, &stats))
        printf("GPU: %.2f GiB allocated, %.2f s\n",
               (double)stats.allocated_bytes / (1u << 30), stats.gpu_seconds);
    h3_dit_free(dit);
    h3_layout_free(&layout);
    h3_st_free_header(&fx);
    if (failures) {
        fprintf(stderr, "FAIL: %d comparison(s) over bound\n", failures);
        return 1;
    }
    puts("h3 HyperFlow real-weight fixture passed");
    return 0;
}
