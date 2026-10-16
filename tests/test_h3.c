#include "h3_adaln_curve.h"
#include "h3_host.h"
#include "h3_dit.h"
#include "h3_lora.h"
#include "h3_metal.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

static int tests_run;

#define CHECK(condition) do { \
    tests_run++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static int close_enough(double value, double expected, double tolerance) {
    return fabs(value - expected) <= tolerance;
}

static void test_temporal_and_canvas(void) {
    const struct { int requested, frames, video_t, audio_t; } cases[] = {
        {5, 5, 2, 8}, {6, 22, 7, 37}, {22, 22, 7, 37},
        {23, 39, 12, 65}, {56, 56, 17, 93}, {124, 124, 37, 207},
        {361, 362, 107, 603}, {362, 362, 107, 603}
    };
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        h3_temporal_shape got = h3_temporal(cases[index].requested);
        CHECK(got.frame_count == cases[index].frames);
        CHECK(got.video_t == cases[index].video_t);
        CHECK(got.audio_t == cases[index].audio_t);
    }
    CHECK(h3_video_encoder_latent_t(1) == 1);
    CHECK(h3_video_encoder_latent_t(5) == 2);
    CHECK(h3_video_encoder_latent_t(22) == 6);
    CHECK(h3_video_encoder_latent_t(39) == 10);
    int width, height;
    CHECK(h3_adapt_canvas(1920, 1080, H3_MAX_PIXELS, &width, &height));
    CHECK(width == 1344 && height == 768);
    CHECK(h3_adapt_canvas(1080, 1920, H3_MAX_PIXELS, &width, &height));
    CHECK(width == 768 && height == 1344);
    CHECK(h3_adapt_canvas(864, 480, H3_MAX_PIXELS, &width, &height));
    CHECK(width == 1376 && height == 768);
    CHECK(h3_adapt_canvas(32, 32, H3_MAX_PIXELS, &width, &height));
    CHECK(width == 768 && height == 768);
    CHECK(h3_reference_image_canvas(1920, 1080, 512, 512, 0,
                                    &width, &height));
    CHECK(width == 672 && height == 384);
    CHECK(h3_reference_image_canvas(640, 480, 1024, 1024, 0,
                                    &width, &height));
    CHECK(width == 640 && height == 480);
    CHECK(h3_reference_image_canvas(4096, 2160, 512, 512, 2048,
                                    &width, &height));
    CHECK(width == 3872 && height == 2048);
    CHECK(!h3_reference_image_canvas(0, 480, 512, 512, 0,
                                     &width, &height));
    CHECK(h3_reference_video_canvas(1920, 1080, H3_MAX_PIXELS, &width, &height));
    CHECK(width == 1344 && height == 768);
    CHECK(h3_reference_video_canvas(640, 360, H3_MAX_PIXELS, &width, &height));
    CHECK(width == 640 && height == 352);
    CHECK(!h3_reference_video_canvas(0, 360, H3_MAX_PIXELS, &width, &height));
}

static void test_schedule(void) {
    h3_params defaults = H3_PARAMS_DEFAULT;
    CHECK(defaults.steps == 20);
    CHECK(defaults.use_reference_rope == 0);

    h3_sigma_schedule schedule;
    CHECK(h3_schedule_build(20, &schedule));
    CHECK(schedule.steps == 20);
    CHECK(schedule.video[0] == 1.0f && schedule.audio[0] == 1.0f);
    CHECK(schedule.video[20] == 0.0f && schedule.audio[20] == 0.0f);
    CHECK(close_enough(schedule.video[1], 0.995633185, 1e-7));
    CHECK(close_enough(schedule.audio[1], 0.982758582, 1e-7));
    CHECK(close_enough(schedule.video[19], 0.387096792, 1e-7));
    CHECK(close_enough(schedule.audio[19], 0.136363640, 1e-7));
    for (int index = 0; index < schedule.steps; index++) {
        CHECK(schedule.video[index] > schedule.video[index + 1]);
        CHECK(schedule.audio[index] > schedule.audio[index + 1]);
    }
    double shifted = h3_time_shift_sigma(0.5, 12.0, 3.0);
    double back = h3_time_shift_sigma(shifted, 3.0, 12.0);
    CHECK(close_enough(back, 0.5, 1e-12));
    CHECK(h3_time_shift_slope(0.5, 12.0, 3.0) > 0.0);

    CHECK(h3_serving_schedule_build(50, &schedule));
    CHECK(schedule.steps == 50);
    CHECK(schedule.video[0] == 1.0f && schedule.audio[0] == 1.0f);
    CHECK(schedule.video[50] == 0.0f && schedule.audio[50] == 0.0f);
    float base = 0.5f;
    CHECK(close_enough(schedule.video[25],
        12.0f * base / (1.0f + 11.0f * base), 1e-7));
    CHECK(close_enough(schedule.audio[25],
        3.0f * base / (1.0f + 2.0f * base), 1e-7));
    for (int index = 0; index < schedule.steps; index++) {
        CHECK(schedule.video[index] > schedule.video[index + 1]);
        CHECK(schedule.audio[index] > schedule.audio[index + 1]);
    }

    CHECK(h3_serving_schedule_build(4, &schedule));
    CHECK(schedule.steps == 4);
    CHECK(schedule.video[0] == 1.0f && schedule.audio[0] == 1.0f);
    CHECK(close_enough(schedule.video[1], 36.0 / 37.0, 1e-7));
    CHECK(close_enough(schedule.audio[1], 9.0 / 10.0, 1e-7));
    CHECK(schedule.video[4] == 0.0f && schedule.audio[4] == 0.0f);
    for (int index = 0; index < schedule.steps; index++) {
        CHECK(schedule.video[index] > schedule.video[index + 1]);
        CHECK(schedule.audio[index] > schedule.audio[index + 1]);
    }
    CHECK(!h3_serving_schedule_build(1, &schedule));
    CHECK(!h3_serving_schedule_build(H3_MAX_STEPS + 1, &schedule));
}

static void test_dit_reuse_schedule(void) {
    uint8_t selected[50];
    const int aggressive[] = {0, 3, 6, 9, 12, 15, 18, 19};
    CHECK(h3_dit_reuse_schedule(20, 3, selected, sizeof(selected)) == 8);
    for (int step = 0; step < 20; step++) {
        int expected = 0;
        for (size_t index = 0;
             index < sizeof(aggressive) / sizeof(*aggressive); index++)
            expected |= step == aggressive[index];
        CHECK(selected[step] == expected);
    }
    CHECK(h3_dit_reuse_schedule(20, 2, selected, sizeof(selected)) == 11);
    for (int step = 0; step < 20; step++)
        CHECK(selected[step] == (step % 2 == 0 || step == 19));
    CHECK(h3_dit_reuse_schedule(50, 3, selected, sizeof(selected)) == 18);
    for (int step = 0; step < 50; step++)
        CHECK(selected[step] == (step % 3 == 0 || step == 49));
    CHECK(h3_dit_reuse_schedule(20, 1, selected, sizeof(selected)) == 20);
    CHECK(h3_dit_reuse_schedule(20, 3, selected, 19) == -1);
}

static void check_segments(const h3_layout *layout,
                           const size_t (*bounds)[2],
                           const h3_segment_kind *kinds, size_t count) {
    CHECK(layout->segment_count == count);
    for (size_t index = 0; index < count; index++) {
        CHECK(layout->segments[index].start == bounds[index][0]);
        CHECK(layout->segments[index].stop == bounds[index][1]);
        CHECK(layout->segments[index].kind == kinds[index]);
        if (index) CHECK(layout->segments[index - 1].stop == bounds[index][0]);
    }
}

static void position_checksums(const h3_layout *layout, double sums[3],
                               double weighted[3]) {
    memset(sums, 0, 3 * sizeof(*sums));
    memset(weighted, 0, 3 * sizeof(*weighted));
    for (size_t index = 0; index < layout->seq_len; index++) {
        const double values[3] = {layout->positions[index].t,
                                  layout->positions[index].h,
                                  layout->positions[index].w};
        for (int axis = 0; axis < 3; axis++) {
            sums[axis] += values[axis];
            weighted[axis] += (double)(index + 1) * values[axis];
        }
    }
}

static void test_layout_tiny(void) {
    h3_layout_spec spec = {12, 2, 2, 2, 8, 5, NULL, 0, NULL, 0, 0, 0.0};
    h3_layout layout;
    char error[256];
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.seq_len == 30);
    const size_t bounds[][2] = {{0, 12}, {12, 28}, {28, 30}};
    const h3_segment_kind kinds[] = {H3_SEG_TEXT, H3_SEG_AUDIO, H3_SEG_VIDEO};
    check_segments(&layout, bounds, kinds, 3);
    CHECK(layout.img_target_rows == 2 && layout.audio_target_rows == 16);
    double sums[3], weighted[3];
    position_checksums(&layout, sums, weighted);
    CHECK(close_enough(sums[0], 339.6666666666667, 1e-10));
    CHECK(close_enough(weighted[0], 6498.0, 1e-10));
    CHECK(sums[1] == 0.0 && sums[2] == 0.0);
    h3_layout_free(&layout);
}

static void test_layout_fl2va(void) {
    int keyframes[] = {0, 55};
    h3_layout_spec spec = {128, 17, 30, 54, 93, 56,
                           keyframes, 2, NULL, 0, 0, 0.0};
    h3_layout layout;
    char error[256];
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.seq_len == 8009);
    const size_t bounds[][2] = {
        {0, 128}, {128, 533}, {533, 938}, {938, 1124}, {1124, 8009}
    };
    const h3_segment_kind kinds[] = {
        H3_SEG_TEXT, H3_SEG_COND, H3_SEG_COND, H3_SEG_AUDIO, H3_SEG_VIDEO
    };
    check_segments(&layout, bounds, kinds, 5);
    CHECK(layout.img_cond_rows == 810);
    CHECK(layout.img_target_rows == 6885);
    double sums[3], weighted[3];
    position_checksums(&layout, sums, weighted);
    CHECK(close_enough(sums[0], 1360927.0, 1e-6));
    CHECK(close_enough(weighted[0], 5883121758.0, 1e-4));
    CHECK(close_enough(sums[1], 117002.11801356057, 1e-7));
    CHECK(close_enough(sums[2], 119830.2393846486, 1e-7));
    h3_layout_free(&layout);
}

/* Clip continuation: 22 pinned frames are 7 one-step keyframes at their
 * own frames (0, 1, 5, 9, 13, 17, 18), each at text_len + frame * 5/3, and
 * an audio window on the clip's own timeline. */
static void test_layout_continuation(void) {
    int keyframes[7];
    for (int step = 0; step < 7; step++) keyframes[step] = h3_video_step_frame(step);
    CHECK(keyframes[1] == 1 && keyframes[2] == 5 && keyframes[5] == 17 &&
          keyframes[6] == 18 && h3_video_step_frame(7) == 22);
    h3_layout_spec spec = {128, 17, 30, 54, 93, 56,
                           keyframes, 7, NULL, 0, 37, 0.0};
    h3_layout layout;
    char error[256];
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.img_cond_rows == 7 * 405);
    CHECK(layout.audio_cond_rows == 74);
    CHECK(layout.segment_count == 1 + 7 + 1 + 2);
    CHECK(layout.segments[8].kind == H3_SEG_REF_AUDIO);
    for (int step = 0; step < 7; step++) {
        size_t row = layout.segments[1 + step].start;
        CHECK(close_enough(layout.positions[row].t,
                           128.0 + 5.0 / 3.0 * keyframes[step], 1e-12));
    }
    /* The pinned frames line up with the clip's own first frames: the
     * target video's first latent step sits where the first keyframe is. */
    size_t video = layout.segments[layout.segment_count - 1].start;
    CHECK(layout.positions[video].t == layout.positions[layout.segments[1].start].t);
    size_t audio = layout.segments[8].start;
    CHECK(layout.positions[audio].t == 128.0 &&
          layout.positions[audio + 36].t == 164.0);
    h3_layout_free(&layout);
}

static void test_layout_ref2va(void) {
    h3_layout_ref references[] = {
        {H3_LAYOUT_REF_IMAGE, 0, 16, 24, 0},
        {H3_LAYOUT_REF_VIDEO, 7, 16, 24, 48},
        {H3_LAYOUT_REF_AUDIO, 0, 0, 0, 80}
    };
    h3_layout_spec spec = {192, 17, 30, 54, 93, 56,
                           NULL, 0, references, 3, 0, 0.0};
    h3_layout layout;
    char error[256];
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.seq_len == 8287);
    const size_t bounds[][2] = {
        {0, 192}, {192, 288}, {288, 384}, {384, 1056},
        {1056, 1216}, {1216, 1402}, {1402, 8287}
    };
    const h3_segment_kind kinds[] = {
        H3_SEG_TEXT, H3_SEG_REF_IMAGE, H3_SEG_REF_AUDIO, H3_SEG_REF_IMAGE,
        H3_SEG_REF_AUDIO, H3_SEG_AUDIO, H3_SEG_VIDEO
    };
    check_segments(&layout, bounds, kinds, 7);
    CHECK(layout.img_cond_rows == 768);
    CHECK(layout.audio_cond_rows == 256);
    double sums[3], weighted[3];
    position_checksums(&layout, sums, weighted);
    CHECK(close_enough(sums[0], 2818905.0, 1e-5));
    CHECK(close_enough(weighted[0], 12788836714.0, 1e-3));
    CHECK(close_enough(sums[1], 115719.96684277552, 1e-7));
    CHECK(close_enough(sums[2], 122360.11344760543, 1e-7));
    h3_layout_free(&layout);
}

static void write_all(int descriptor, const void *data, size_t size) {
    const unsigned char *bytes = data;
    while (size) {
        ssize_t written = write(descriptor, bytes, size);
        CHECK(written > 0);
        bytes += (size_t)written;
        size -= (size_t)written;
    }
}

static void test_safetensors(void) {
    char path[] = "/tmp/h3_safetensors_XXXXXX";
    int descriptor = mkstemp(path);
    CHECK(descriptor >= 0);
    const char header_json[] =
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]},"
        "\"scalar\":{\"dtype\":\"BF16\",\"shape\":[],\"data_offsets\":[24,26]}}";
    uint64_t length = sizeof(header_json) - 1;
    unsigned char prefix[8];
    for (unsigned index = 0; index < 8; index++) prefix[index] = (unsigned char)(length >> (8 * index));
    unsigned char payload[26] = {0};
    write_all(descriptor, prefix, sizeof(prefix));
    write_all(descriptor, header_json, (size_t)length);
    write_all(descriptor, payload, sizeof(payload));
    CHECK(close(descriptor) == 0);

    h3_st_header header;
    char error[256];
    CHECK(h3_st_read_header(path, &header, error, sizeof(error)));
    CHECK(header.tensor_count == 2);
    const h3_st_tensor *x = h3_st_find(&header, "x");
    CHECK(x && x->dtype == H3_DTYPE_F32 && x->ndim == 2);
    CHECK(x->shape[0] == 2 && x->shape[1] == 3);
    CHECK(x->data_end - x->data_begin == 24);
    CHECK(h3_st_tensor_elements(x) == 6);
    unsigned char readback[24];
    CHECK(h3_st_read_data(&header, x, readback, sizeof(readback), error,
                          sizeof(error)));
    for (size_t index = 0; index < sizeof(readback); index++) {
        CHECK(readback[index] == 0);
    }
    CHECK(!h3_st_read_data(&header, x, readback, sizeof(readback) - 1, error,
                           sizeof(error)));
    const h3_st_tensor *scalar = h3_st_find(&header, "scalar");
    CHECK(scalar && scalar->dtype == H3_DTYPE_BF16 && scalar->ndim == 0);
    CHECK(scalar->data_end - scalar->data_begin == 2);
    CHECK(h3_st_tensor_elements(scalar) == 1);
    h3_st_free_header(&header);
    CHECK(unlink(path) == 0);
}

/* A minimal valid shard (one F32 tensor) followed by `padding` extra
 * bytes, with a fixed mtime, so tests control exactly what the weight
 * fingerprint sees. */
static void write_fake_shard(const char *path, size_t padding, time_t mtime) {
    FILE *file = fopen(path, "wb");
    CHECK(file != NULL);
    const char header_json[] =
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    uint64_t length = sizeof(header_json) - 1;
    unsigned char prefix[8];
    for (unsigned index = 0; index < 8; index++) prefix[index] = (unsigned char)(length >> (8 * index));
    CHECK(fwrite(prefix, sizeof(prefix), 1, file) == 1);
    CHECK(fwrite(header_json, (size_t)length, 1, file) == 1);
    for (size_t index = 0; index < 4 + padding; index++) CHECK(fputc(0, file) != EOF);
    CHECK(fclose(file) == 0);
    struct timeval times[2] = {{mtime, 0}, {mtime, 0}};
    CHECK(utimes(path, times) == 0);
}

static void fingerprint_directory(const char *directory, uint8_t id[32],
                                  uint8_t legacy[32]) {
    char error[256];
    h3_weight_store *store = h3_weight_store_open(directory, error, sizeof(error));
    CHECK(store != NULL);
    CHECK(h3_weight_store_shards(store) == 2);
    h3_weight_store_fingerprint(store, id);
    h3_weight_store_fingerprint_legacy(store, legacy);
    h3_weight_store_free(store);
}

static void test_weight_fingerprint(void) {
    char root[] = "/tmp/h3_fingerprint_XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    char first[256], second[256], transformer[256];
    char shard1[320], shard2[320], renamed[320];
    snprintf(first, sizeof(first), "%s/models-a", root);
    snprintf(second, sizeof(second), "%s/models-b", root);
    snprintf(transformer, sizeof(transformer), "%s/transformer", first);
    CHECK(mkdir(first, 0700) == 0);
    CHECK(mkdir(transformer, 0700) == 0);
    snprintf(shard1, sizeof(shard1), "%s/model-00001-of-00002.safetensors", transformer);
    snprintf(shard2, sizeof(shard2), "%s/model-00002-of-00002.safetensors", transformer);
    write_fake_shard(shard1, 0, 1700000000);
    write_fake_shard(shard2, 16, 1700000100);

    uint8_t original[32], original_legacy[32], id[32], legacy[32];
    fingerprint_directory(transformer, original, original_legacy);
    static const uint8_t zero[32];
    CHECK(memcmp(original, zero, 32) != 0);
    /* The two formats never collide: legacy ids are zero past byte 8. */
    CHECK(memcmp(original + 8, zero, 24) != 0);
    CHECK(memcmp(original_legacy + 8, zero, 24) == 0);
    CHECK(memcmp(original, original_legacy, 32) != 0);

    /* Same directory again: both ids are stable. */
    fingerprint_directory(transformer, id, legacy);
    CHECK(memcmp(id, original, 32) == 0);
    CHECK(memcmp(legacy, original_legacy, 32) == 0);

    /* Move the whole model folder: the new id survives, the legacy one
     * (what pre-2026-10-05 caches hold) does not. */
    CHECK(rename(first, second) == 0);
    snprintf(transformer, sizeof(transformer), "%s/transformer", second);
    snprintf(shard1, sizeof(shard1), "%s/model-00001-of-00002.safetensors", transformer);
    snprintf(shard2, sizeof(shard2), "%s/model-00002-of-00002.safetensors", transformer);
    fingerprint_directory(transformer, id, legacy);
    CHECK(memcmp(id, original, 32) == 0);
    CHECK(memcmp(legacy, original_legacy, 32) != 0);

    /* Different weights must still change it: an mtime, a size, a name. */
    write_fake_shard(shard2, 16, 1700000101);
    fingerprint_directory(transformer, id, legacy);
    CHECK(memcmp(id, original, 32) != 0);
    write_fake_shard(shard2, 17, 1700000100);
    fingerprint_directory(transformer, id, legacy);
    CHECK(memcmp(id, original, 32) != 0);
    write_fake_shard(shard2, 16, 1700000100);
    fingerprint_directory(transformer, id, legacy);
    CHECK(memcmp(id, original, 32) == 0);
    snprintf(renamed, sizeof(renamed), "%s/model-00002-of-00003.safetensors", transformer);
    CHECK(rename(shard2, renamed) == 0);
    fingerprint_directory(transformer, id, legacy);
    CHECK(memcmp(id, original, 32) != 0);

    CHECK(unlink(shard1) == 0);
    CHECK(unlink(renamed) == 0);
    CHECK(rmdir(transformer) == 0);
    CHECK(rmdir(second) == 0);
    CHECK(rmdir(root) == 0);
}

static void test_rng_and_solver(void) {
    h3_rng a, b;
    h3_rng_seed(&a, 42);
    h3_rng_seed(&b, 42);
    for (int index = 0; index < 64; index++) CHECK(h3_rng_u32(&a) == h3_rng_u32(&b));
    h3_rng_seed(&a, 7);
    float values[1024];
    h3_rng_fill_normal(&a, values, 1024);
    double sum = 0.0;
    for (size_t index = 0; index < 1024; index++) {
        CHECK(isfinite(values[index]));
        sum += values[index];
    }
    CHECK(fabs(sum / 1024.0) < 0.15);

    const float sigmas[] = {1.0f, 0.75f, 0.25f, 0.0f};
    float sample[] = {0.0f};
    float denoised[] = {1.0f};
    float old[] = {0.5f};
    float output[1];
    CHECK(h3_res_step(output, sample, denoised, NULL, 1, sigmas, 0, 3));
    CHECK(close_enough(output[0], 0.25, 1e-7));
    CHECK(h3_res_step(output, sample, denoised, old, 1, sigmas, 1, 3));
    CHECK(isfinite(output[0]));
    float velocity[] = {2.0f, -4.0f};
    float euler[] = {1.0f, 3.0f};
    CHECK(h3_euler_velocity_step(euler, velocity, 2, 0.75f, 0.25f));
    CHECK(euler[0] == 2.0f && euler[1] == 1.0f);
    CHECK(!h3_euler_velocity_step(euler, velocity, 2, 0.25f, 0.25f));
}

static void test_rgb_resize(void) {
    const uint8_t constant[] = {
        17, 33, 201, 17, 33, 201,
        17, 33, 201, 17, 33, 201
    };
    uint8_t *identity = NULL;
    CHECK(h3_resize_rgb24_high_quality(constant, 1, 2, 2, 2, 2, &identity));
    CHECK(identity != NULL);
    CHECK(memcmp(identity, constant, sizeof(constant)) == 0);
    free(identity);

    uint8_t *upscaled = NULL;
    CHECK(h3_resize_rgb24_high_quality(constant, 1, 2, 2, 8, 8, &upscaled));
    CHECK(upscaled != NULL);
    for (size_t pixel = 0; pixel < 64; pixel++) {
        CHECK(upscaled[3 * pixel] == 17);
        CHECK(upscaled[3 * pixel + 1] == 33);
        CHECK(upscaled[3 * pixel + 2] == 201);
    }
    free(upscaled);
    CHECK(!h3_resize_rgb24_high_quality(NULL, 1, 2, 2, 8, 8, &upscaled));
}

static void test_dit_row_conversions(void) {
    enum { C = 3, T = 2, H = 4, W = 6, VIDEO = C * T * H * W };
    float latent[VIDEO], rows[VIDEO], roundtrip[VIDEO];
    for (size_t index = 0; index < VIDEO; index++) latent[index] = (float)index;
    CHECK(h3_dit_patchify_video(latent, C, T, H, W, rows, VIDEO));
    /* First row is pixel patch (h=0..1,w=0..1), channel-major inside it. */
    const float first[] = {0, 1, 6, 7, 48, 49, 54, 55, 96, 97, 102, 103};
    for (size_t index = 0; index < sizeof(first) / sizeof(*first); index++)
        CHECK(rows[index] == first[index]);
    CHECK(h3_dit_unpatchify_video(rows, C, T, H, W, roundtrip, VIDEO));
    CHECK(memcmp(latent, roundtrip, sizeof(latent)) == 0);
    CHECK(!h3_dit_patchify_video(latent, C, T, H, W, rows, VIDEO - 1));

    enum { AC = 4, AT = 3, AUDIO = AC * 2 * AT };
    float audio[AUDIO], packed[AUDIO], unpacked[AUDIO];
    for (size_t index = 0; index < AUDIO; index++) audio[index] = (float)index;
    CHECK(h3_dit_pack_audio(audio, AC, AT, packed, AUDIO));
    const float expected[] = {0, 6, 12, 18, 1, 7, 13, 19,
                              2, 8, 14, 20, 3, 9, 15, 21,
                              4, 10, 16, 22, 5, 11, 17, 23};
    CHECK(memcmp(packed, expected, sizeof(expected)) == 0);
    CHECK(h3_dit_unpack_audio(packed, AC, AT, unpacked, AUDIO));
    CHECK(memcmp(audio, unpacked, sizeof(audio)) == 0);
}

static void check_lora_key(const char *name, int refiner, int block,
                           int projection, h3_lora_part part,
                           h3_lora_role role, h3_lora_style style) {
    h3_lora_key key;
    CHECK(h3_lora_parse_key(name, &key));
    CHECK(key.refiner == refiner);
    CHECK(key.block == block);
    CHECK(key.projection == projection);
    CHECK(key.part == part);
    CHECK(key.role == role);
    CHECK(key.style == style);
}

/* Names taken from published H3 adapters (lightx2v Turbo, fal, drbaph,
 * TaoMate, larryvrh, alibaba-pai, Cseti) - one per layout. */
static void test_lora_keys_and_rows(void) {
    check_lora_key("transformer_blocks.7.attn.to_q.lora_A.default.weight",
                   0, 7, H3_LORA_QKV, H3_LORA_PART_Q, H3_LORA_ROLE_A,
                   H3_LORA_STYLE_DIFFUSERS);
    check_lora_key("transformer_blocks.7.attn.to_v.lora_B.default.weight",
                   0, 7, H3_LORA_QKV, H3_LORA_PART_V, H3_LORA_ROLE_B,
                   H3_LORA_STYLE_DIFFUSERS);
    check_lora_key("transformer_blocks.0.ff.net.0.proj.lora_B.default.weight",
                   0, 0, H3_LORA_FC1, H3_LORA_PART_FC1_VALUE_GATE,
                   H3_LORA_ROLE_B, H3_LORA_STYLE_DIFFUSERS);
    check_lora_key("transformer_blocks.49.ff.net.2.lora_A.default.weight",
                   0, 49, H3_LORA_FC2, H3_LORA_PART_ALL, H3_LORA_ROLE_A,
                   H3_LORA_STYLE_DIFFUSERS);
    check_lora_key("token_refiner.refiner_blocks.1.attn.to_out.0.lora_B.default.weight",
                   1, 1, H3_LORA_OUT, H3_LORA_PART_ALL, H3_LORA_ROLE_B,
                   H3_LORA_STYLE_DIFFUSERS);
    check_lora_key("diffusion_model.blocks.12.attn.qkv_proj.lora_A.weight",
                   0, 12, H3_LORA_QKV, H3_LORA_PART_ALL, H3_LORA_ROLE_A,
                   H3_LORA_STYLE_COMFYUI);
    check_lora_key("diffusion_model.blocks.12.attn.qkv_proj.alpha",
                   0, 12, H3_LORA_QKV, H3_LORA_PART_ALL, H3_LORA_ROLE_ALPHA,
                   H3_LORA_STYLE_COMFYUI);
    check_lora_key("diffusion_model.token_refiner.blocks.0.mlp.fc1.lora_B.weight",
                   1, 0, H3_LORA_FC1, H3_LORA_PART_ALL, H3_LORA_ROLE_B,
                   H3_LORA_STYLE_COMFYUI);
    check_lora_key("diffusion_model.blocks.3.mlp.fc2.lora_up.weight",
                   0, 3, H3_LORA_FC2, H3_LORA_PART_ALL, H3_LORA_ROLE_B,
                   H3_LORA_STYLE_COMFYUI);
    check_lora_key("blocks.20.attn.out_proj.lora_A.weight",
                   0, 20, H3_LORA_OUT, H3_LORA_PART_ALL, H3_LORA_ROLE_A,
                   H3_LORA_STYLE_NATIVE);
    check_lora_key("blocks.20.mlp.fc1.lora_B.default.weight",
                   0, 20, H3_LORA_FC1, H3_LORA_PART_ALL, H3_LORA_ROLE_B,
                   H3_LORA_STYLE_NATIVE);
    check_lora_key("token_refiner.refiner_blocks.0.attn.to_k.lora_down",
                   1, 0, H3_LORA_QKV, H3_LORA_PART_K, H3_LORA_ROLE_A,
                   H3_LORA_STYLE_DIFFUSERS);
    check_lora_key("lora_unet_blocks_10_attn_qkv_proj.lora_down.weight",
                   0, 10, H3_LORA_QKV, H3_LORA_PART_ALL, H3_LORA_ROLE_A,
                   H3_LORA_STYLE_KOHYA);
    check_lora_key("lora_unet_blocks_10_mlp_fc2.alpha",
                   0, 10, H3_LORA_FC2, H3_LORA_PART_ALL, H3_LORA_ROLE_ALPHA,
                   H3_LORA_STYLE_KOHYA);

    h3_lora_key key;
    CHECK(!h3_lora_parse_key("diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight", &key));
    CHECK(!h3_lora_parse_key("final_layer.adaln_proj.linear.lora_B.weight", &key));
    CHECK(!h3_lora_parse_key("proj_out.weight", &key));
    CHECK(!h3_lora_parse_key("diffusion_model.blocks.50.attn.qkv_proj.lora_A.weight", &key));
    CHECK(!h3_lora_parse_key("token_refiner.blocks.2.mlp.fc1.lora_A.weight", &key));
    CHECK(!h3_lora_parse_key("blocks.0.mlp.fc1.weight", &key));

    /* Verified against the ComfyUI and official checkpoints: ComfyUI's
     * contiguous row 128 (q, head 1) is official row 384, its row 7168
     * (k, head 0) official row 128 and its row 14336 (v, head 0) row 256. */
    CHECK(h3_lora_group_count(H3_LORA_QKV) == 3);
    CHECK(h3_lora_group_rows(H3_LORA_QKV) == H3_LORA_INNER);
    CHECK(h3_lora_engine_row(H3_LORA_QKV, 0, 0) == 0);
    CHECK(h3_lora_engine_row(H3_LORA_QKV, 0, 128) == 384);
    CHECK(h3_lora_engine_row(H3_LORA_QKV, 1, 0) == 128);
    CHECK(h3_lora_engine_row(H3_LORA_QKV, 2, 0) == 256);
    CHECK(h3_lora_engine_row(H3_LORA_QKV, 2, H3_LORA_INNER - 1) ==
          3u * H3_LORA_INNER - 1);
    CHECK(h3_lora_group_count(H3_LORA_FC1) == 1);
    CHECK(h3_lora_group_rows(H3_LORA_FC1) == 2u * H3_LORA_FFN);
    CHECK(h3_lora_engine_row(H3_LORA_FC1, 0, 77) == 77);
    CHECK(h3_lora_columns(H3_LORA_OUT) == H3_LORA_INNER);
    CHECK(h3_lora_columns(H3_LORA_FC2) == H3_LORA_FFN);
}

static void test_metal_probe(void) {
    h3_device_info info;
    char error[256];
    CHECK(h3_metal_probe(&info, error, sizeof(error)));
    CHECK(info.name[0] != '\0');
    CHECK(info.physical_memory >= UINT64_C(8) * 1024 * 1024 * 1024);
    CHECK(info.max_buffer_length > 0);
    CHECK(info.apple_gpu_family > 0);
}

static void write_curve_file(const char *path, const char *metadata,
                             const float *table, size_t values) {
    char header_json[512];
    int length = snprintf(header_json, sizeof(header_json),
        "{\"__metadata__\":{%s},\"adaln_t_table\":{\"dtype\":\"F32\","
        "\"shape\":[3,2],\"data_offsets\":[0,%zu]}}", metadata,
        values * sizeof(float));
    int descriptor = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(descriptor >= 0 && length > 0);
    uint64_t size = (uint64_t)length;
    unsigned char prefix[8];
    for (unsigned index = 0; index < 8; index++)
        prefix[index] = (unsigned char)(size >> (8 * index));
    write_all(descriptor, prefix, sizeof(prefix));
    write_all(descriptor, header_json, (size_t)length);
    write_all(descriptor, table, values * sizeof(float));
    CHECK(close(descriptor) == 0);
}

static void test_adaln_curve(void) {
    /* ComfyUI's interpolation on a 3-point, rank-2 table. */
    const float table[] = {0.0f, 10.0f, 2.0f, 20.0f, 6.0f, 60.0f};
    float out[2];
    h3_adaln_curve_lerp(table, 3, 2, 0.0f, out);
    CHECK(out[0] == 0.0f && out[1] == 10.0f);
    h3_adaln_curve_lerp(table, 3, 2, 0.25f, out);
    CHECK(fabsf(out[0] - 1.0f) < 1e-6f && fabsf(out[1] - 15.0f) < 1e-5f);
    h3_adaln_curve_lerp(table, 3, 2, 0.5f, out);
    CHECK(out[0] == 2.0f && out[1] == 20.0f);
    h3_adaln_curve_lerp(table, 3, 2, 0.75f, out);
    CHECK(fabsf(out[0] - 4.0f) < 1e-6f && fabsf(out[1] - 40.0f) < 1e-5f);
    h3_adaln_curve_lerp(table, 3, 2, 1.0f, out);   /* last interval, fraction 1 */
    CHECK(out[0] == 6.0f && out[1] == 60.0f);
    h3_adaln_curve_lerp(table, 3, 2, 1.5f, out);   /* clamped */
    CHECK(out[0] == 6.0f && out[1] == 60.0f);
    h3_adaln_curve_lerp(table, 3, 2, -1.0f, out);
    CHECK(out[0] == 0.0f && out[1] == 10.0f);
    h3_adaln_curve_lerp(table, 3, 2, NAN, out);
    CHECK(out[0] == 0.0f && out[1] == 10.0f);

    char path[] = "/tmp/h3_adaln_curve_XXXXXX";
    int descriptor = mkstemp(path);
    CHECK(descriptor >= 0);
    CHECK(close(descriptor) == 0);
    uint8_t model[32] = {0};
    model[0] = 0xab;
    model[31] = 0x01;
    const char *good =
        "\"format\":\"h3-adaln-curve\",\"version\":\"1\",\"grid\":\"3\","
        "\"rank\":\"2\",\"model_id\":"
        "\"ab00000000000000000000000000000000000000000000000000000000000001\"";
    write_curve_file(path, good, table, 6);
    char error[256];
    h3_adaln_curve *curve = h3_adaln_curve_open(path, model, error,
                                                sizeof(error));
    CHECK(curve != NULL);
    if (curve) {
        CHECK(h3_adaln_curve_rank(curve) == 2 && h3_adaln_curve_grid(curve) == 3);
        const float times[] = {0.25f, 1.0f};
        float coordinates[4];
        h3_adaln_curve_coordinates(curve, times, 2, coordinates);
        CHECK(fabsf(coordinates[0] - 1.0f) < 1e-6f &&
              coordinates[2] == 6.0f && coordinates[3] == 60.0f);
        uint8_t id[32];
        h3_adaln_curve_fingerprint(curve, id);
        CHECK(!memcmp(id + 8, "h3curve1", 8));
        h3_adaln_curve_free(curve);
    }
    /* Any other checkpoint is refused; no expected id skips the check. */
    model[31] = 0x02;
    CHECK(h3_adaln_curve_open(path, model, error, sizeof(error)) == NULL);
    CHECK(strstr(error, "different transformer") != NULL);
    curve = h3_adaln_curve_open(path, NULL, error, sizeof(error));
    CHECK(curve != NULL);
    h3_adaln_curve_free(curve);
    /* Wrong format tag, or a table that disagrees with grid/rank. */
    write_curve_file(path, "\"format\":\"other\",\"version\":\"1\","
                     "\"grid\":\"3\",\"rank\":\"2\"", table, 6);
    CHECK(h3_adaln_curve_open(path, NULL, error, sizeof(error)) == NULL);
    write_curve_file(path, "\"format\":\"h3-adaln-curve\",\"version\":\"1\","
                     "\"grid\":\"4\",\"rank\":\"2\"", table, 6);
    CHECK(h3_adaln_curve_open(path, NULL, error, sizeof(error)) == NULL);
    CHECK(unlink(path) == 0);
}

int main(void) {
    test_temporal_and_canvas();
    test_schedule();
    test_dit_reuse_schedule();
    test_layout_tiny();
    test_layout_fl2va();
    test_layout_continuation();
    test_layout_ref2va();
    test_safetensors();
    test_weight_fingerprint();
    test_adaln_curve();
    test_rng_and_solver();
    test_rgb_resize();
    test_dit_row_conversions();
    test_lora_keys_and_rows();
    test_metal_probe();
    printf("ok: %d checks\n", tests_run);
    return 0;
}
