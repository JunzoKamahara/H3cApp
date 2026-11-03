/* The video encoder over time (h3_video_encoder.c, "Memory over time").
 *
 *   ./h3_video_encoder_chunks_test
 *       plan checks only (no weights)
 *   ./h3_video_encoder_chunks_test <video_vae/source dir> <video file>
 *       also encodes clips in one pass and in chunks and checks they agree:
 *       96x64 x 90 frames (one tile) and 320x64 x 39 frames (two tiles, so
 *       the history is reset between tiles). H3_CHUNKS_WIDTH/HEIGHT/FRAMES
 *       run one other size instead; H3_CHUNKS_ONLY=<chunk frames> skips the
 *       one-pass encode and reports the chunked run alone (for clips whose
 *       one pass would not fit). The clip's first frame, as a still image
 *       at 320x256 (two by two tiles), must also encode bit for bit the
 *       same with its causal zero frames left out (the default for stills)
 *       and convolved (H3_ENCODER_STILL=0), in both encoders.
 *
 * The one-pass encode holds every frame of a tile at once, so its estimate
 * is checked against a fixed ceiling here, before anything is allocated:
 * an oversized run stalls the whole Mac in swap. */
#include "h3_av_reader.h"
#include "h3_host.h"
#include "h3_video_encoder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GIB (1024.0 * 1024.0 * 1024.0)
static const uint64_t ceiling_bytes = 6ull << 30;

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_video_encoder_chunks.c: %s\n", message);
    exit(1);
}

static int env_int(const char *name, int fallback) {
    const char *value = getenv(name);
    return value && *value ? atoi(value) : fallback;
}

static void expect_plan(int frames, int height, int width, uint64_t budget,
                        int chunk) {
    int got = -1;
    if (!h3_video_encoder_plan(frames, height, width, budget, &got))
        die("plan rejected valid input");
    if (got != chunk) {
        fprintf(stderr, "plan(%d frames, %dx%d, %.2f GiB): chunks of %d, want %d\n",
                frames, width, height, (double)budget / GIB, got, chunk);
        die("unexpected encoder plan");
    }
    if (chunk > 4 &&
        h3_video_encoder_chunked_peak_bytes(chunk, height, width) > budget)
        die("a chunk can exceed the budget");
}

static void plan_checks(void) {
    const uint64_t eight = 8ull << 30;   /* the default on a 24 GB Mac */
    /* A 256x256 tile costs 352 MiB per frame over a 1.5 GiB base. */
    if (h3_video_encoder_peak_bytes(1, 640, 1120) != (3ull << 29) + (352ull << 20))
        die("unexpected per-frame estimate");
    if (h3_video_encoder_peak_bytes(243, 640, 1120) < 51ull << 30)
        die("the 243-frame estimate is below what was observed (51+ GiB)");
    expect_plan(1, 640, 1120, eight, 0);               /* still images */
    expect_plan(4, 640, 1120, 1ull << 30, 0);          /* one step */
    expect_plan(17, 640, 1120, eight, 0);              /* continuation: one pass */
    expect_plan(39, 640, 1120,
                h3_video_encoder_peak_bytes(39, 640, 1120), 0); /* exact fit */
    expect_plan(22, 640, 1120, eight, 16);
    expect_plan(243, 640, 1120, eight, 16);            /* the crash */
    expect_plan(362, 640, 1120, eight, 16);
    expect_plan(243, 96, 128, eight, 96);              /* smaller tiles */
    expect_plan(243, 640, 1120, 1ull << 30, 4);        /* tiny budget */
    expect_plan(243, 640, 1120, 0, 0);                 /* no budget */
    int chunk;
    if (h3_video_encoder_plan(0, 640, 1120, eight, &chunk) ||
        h3_video_encoder_plan(5, 640, 1120, eight, NULL))
        die("plan accepted invalid input");
    if (h3_video_encoder_default_budget() < (1ull << 30))
        die("default budget is implausibly small");
    puts("ok: video encoder plans keep every chunk within the budget");
}

static float *read_clip(const char *video, int width, int height, int frames) {
    char error[512];
    float *pixels = NULL;
    int read_frames = 0;
    if (!h3_av_read_video_f32(video, width, height, frames, &pixels,
                              &read_frames, error, sizeof(error)))
        die(error);
    if (read_frames != frames)
        die("the video gave another frame count (reference clips are 5 + 17k)");
    return pixels;
}

static void encode(const char *weights, const float *pixels, int frames,
                   int height, int width, int chunk, int fast,
                   h3_video_latent *out) {
    char error[512];
    if (!h3_video_vae_encode_chunked(weights, "h3_shaders.metal", pixels,
                                     frames, height, width, chunk, fast, NULL,
                                     NULL, out, error, sizeof(error)))
        die(error);
    if (out->time != h3_video_encoder_latent_t(frames) ||
        out->height != height / 16 || out->width != width / 16)
        die("encoder returned the wrong shape");
}

static size_t latent_count(const h3_video_latent *latent) {
    return (size_t)24 * latent->time * latent->height * latent->width;
}

static void compare(const char *label, const h3_video_latent *got,
                    const h3_video_latent *want) {
    double maximum = 0, error = 0, value = 0;
    size_t count = latent_count(want), differing = 0;
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(got->values[index])) die("non-finite latent");
        double delta = (double)got->values[index] - want->values[index];
        if (delta != 0) differing++;
        if (fabs(delta) > maximum) maximum = fabs(delta);
        error += delta * delta;
        value += (double)want->values[index] * want->values[index];
    }
    double relative = sqrt(error / (value > 1e-24 ? value : 1e-24));
    printf("  %-28s max abs %.3g, relative L2 %.3g, %zu of %zu values differ, "
           "peak live %.2f GiB\n", label, maximum, relative, differing, count,
           (double)got->gpu_stats.peak_live_bytes / GIB);
    if (relative > 1e-5) die("chunked encode differs from one pass");
}

static void check_clip(const char *weights, const char *video, int width,
                       int height, int frames) {
    uint64_t whole_bytes = h3_video_encoder_peak_bytes(frames, height, width);
    printf("clip %dx%d, %d frames, %d steps; one-pass estimate %.2f GiB\n",
           width, height, frames, h3_video_encoder_latent_t(frames),
           (double)whole_bytes / GIB);
    if (whole_bytes > ceiling_bytes)
        die("one-pass estimate is over the 6 GiB test ceiling; use a smaller "
            "clip or H3_CHUNKS_ONLY");
    float *pixels = read_clip(video, width, height, frames);
    h3_video_latent whole, fast_whole;
    encode(weights, pixels, frames, height, width, 0, 0, &whole);
    printf("  one pass: peak live %.2f GiB\n",
           (double)whole.gpu_stats.peak_live_bytes / GIB);
    if (whole.gpu_stats.peak_live_bytes > whole_bytes)
        die("one-pass peak is above the estimate the plan relies on");
    static const int chunks[] = {4, 8, 16, 28};
    for (size_t index = 0; index < sizeof(chunks) / sizeof(*chunks); index++) {
        if (chunks[index] >= frames) continue;
        h3_video_latent part;
        encode(weights, pixels, frames, height, width, chunks[index], 0, &part);
        char label[64];
        snprintf(label, sizeof(label), "chunks of %d frames:", chunks[index]);
        compare(label, &part, &whole);
        if (part.gpu_stats.peak_live_bytes >
            h3_video_encoder_chunked_peak_bytes(chunks[index], height, width))
            die("a chunked encode's peak is above its estimate");
        h3_video_latent_free(&part);
    }
    /* The 2D-convolution encoder (continuation) chunks the same way. */
    encode(weights, pixels, frames, height, width, 0, 1, &fast_whole);
    h3_video_latent fast_part;
    encode(weights, pixels, frames, height, width, 8, 1, &fast_part);
    compare("fast, chunks of 8 frames:", &fast_part, &fast_whole);
    h3_video_latent_free(&fast_part);
    h3_video_latent_free(&fast_whole);

    /* The public entry point follows the plan: a budget that only fits
     * chunks of 8 frames must give exactly the explicit encode. */
    char budget[32], error[512];
    snprintf(budget, sizeof(budget), "%llu", (unsigned long long)
             h3_video_encoder_chunked_peak_bytes(8, height, width));
    setenv("H3_VAE_ENCODE_BUDGET_BYTES", budget, 1);
    int planned_chunk = 0;
    if (!h3_video_encoder_plan(frames, height, width,
                               h3_video_encoder_default_budget(), &planned_chunk) ||
        planned_chunk != 8)
        die("budget override did not plan chunks of 8 frames");
    h3_video_latent planned, explicit_chunks;
    if (!h3_video_vae_encode(weights, "h3_shaders.metal", pixels, frames, height,
                             width, NULL, NULL, &planned, error, sizeof(error)))
        die(error);
    unsetenv("H3_VAE_ENCODE_BUDGET_BYTES");
    encode(weights, pixels, frames, height, width, 8, 0, &explicit_chunks);
    if (planned.time != explicit_chunks.time ||
        memcmp(planned.values, explicit_chunks.values,
               latent_count(&planned) * sizeof(float)))
        die("h3_video_vae_encode does not follow its plan");
    h3_video_latent_free(&planned);
    h3_video_latent_free(&explicit_chunks);
    h3_video_latent_free(&whole);
    free(pixels);
}

static void check_still(const char *weights, const char *video) {
    const int width = 320, height = 256;
    float *clip = read_clip(video, width, height, 5);
    size_t area = (size_t)width * height;
    float *pixels = malloc(3 * area * sizeof(*pixels));
    if (!pixels) die("out of memory");
    for (int channel = 0; channel < 3; channel++)   /* [3,5,H,W] -> [3,1,H,W] */
        memcpy(pixels + channel * area, clip + (size_t)channel * 5 * area,
               area * sizeof(*pixels));
    free(clip);
    for (int fast = 0; fast <= 1; fast++) {
        h3_video_latent skipped, convolved;
        encode(weights, pixels, 1, height, width, 0, fast, &skipped);
        setenv("H3_ENCODER_STILL", "0", 1);
        encode(weights, pixels, 1, height, width, 0, fast, &convolved);
        unsetenv("H3_ENCODER_STILL");
        if (memcmp(skipped.values, convolved.values,
                   latent_count(&skipped) * sizeof(float)))
            die("a still image encodes differently without its zero frames");
        printf("still %dx%d (%s): the same latent without the zero frames, "
               "peak live %.2f vs %.2f GiB\n", width, height,
               fast ? "2D convs" : "Conv3d",
               (double)skipped.gpu_stats.peak_live_bytes / GIB,
               (double)convolved.gpu_stats.peak_live_bytes / GIB);
        h3_video_latent_free(&skipped);
        h3_video_latent_free(&convolved);
    }
    free(pixels);
}

int main(int argc, char **argv) {
    plan_checks();
    if (argc < 3) {
        puts("skip: real-weights chunk comparison (pass <video_vae/source> <video>)");
        return 0;
    }
    const char *weights = argv[1], *video = argv[2];
    check_still(weights, video);
    int only = env_int("H3_CHUNKS_ONLY", 0);
    if (getenv("H3_CHUNKS_WIDTH") || only) {
        int width = env_int("H3_CHUNKS_WIDTH", 96);
        int height = env_int("H3_CHUNKS_HEIGHT", 64);
        int frames = env_int("H3_CHUNKS_FRAMES", 90);
        if (!only) {
            check_clip(weights, video, width, height, frames);
        } else {
            uint64_t bytes = h3_video_encoder_chunked_peak_bytes(only, height, width);
            printf("clip %dx%d, %d frames in chunks of %d; estimate %.2f GiB\n",
                   width, height, frames, only, (double)bytes / GIB);
            if (bytes > (13ull << 30)) die("chunk estimate is over 13 GiB");
            float *pixels = read_clip(video, width, height, frames);
            h3_video_latent part;
            encode(weights, pixels, frames, height, width, only, 0, &part);
            printf("  peak live %.2f GiB\n",
                   (double)part.gpu_stats.peak_live_bytes / GIB);
            h3_video_latent_free(&part);
            free(pixels);
        }
    } else {
        check_clip(weights, video, 96, 64, 90);
        check_clip(weights, video, 320, 64, 39);
    }
    puts("ok: chunked video encodes match one pass");
    return 0;
}
