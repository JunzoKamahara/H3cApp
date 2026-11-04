/* RefMod files (h3_refmod.c) without any model weights: the reader on
 * hand-written single and bundle files in the upstream format, legacy
 * metadata placements, the rejection of malformed shapes, and the strength
 * and pooling arithmetic against closed-form cases. The Ref2VA label test
 * runs only when a tokenizer path is given:
 *
 *   ./h3_refmod_tests [MiniMax-H3/Ref2VA/tokenizer/tokenizer.json] [fixture]
 *
 * The numeric fixture (misc/fixtures/h3_refmod_fixture.safetensors, made by
 * tools/dump_refmod_fixture.py with torch) checks the pooling, upsampling,
 * strength mix and Adam refinement against PyTorch; it is skipped when the
 * file is missing. */
#include "h3_refmod.h"
#include "h3_dit.h"
#include "h3_multimodal.h"
#include "h3_quant.h"
#include "h3_safetensors.h"
#include "h3_tokenizer.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int checks;

#define CHECK(condition) do { \
    checks++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static char directory[1024];

static const char *path_in_directory(const char *name) {
    static char path[1200];
    snprintf(path, sizeof(path), "%s/%s", directory, name);
    return path;
}

static void write_all(int descriptor, const void *data, size_t bytes) {
    const unsigned char *cursor = data;
    while (bytes) {
        ssize_t wrote = write(descriptor, cursor, bytes);
        CHECK(wrote > 0);
        cursor += wrote;
        bytes -= (size_t)wrote;
    }
}

/* A safetensors file from a header JSON body (without the 8-byte prefix)
 * and the concatenated tensor data. */
static void write_safetensors(const char *path, const char *header_json,
                              const void *data, size_t bytes) {
    int descriptor = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(descriptor >= 0);
    uint64_t size = strlen(header_json);
    unsigned char prefix[8];
    for (unsigned index = 0; index < 8; index++)
        prefix[index] = (unsigned char)(size >> (8 * index));
    write_all(descriptor, prefix, sizeof(prefix));
    write_all(descriptor, header_json, (size_t)size);
    if (bytes) write_all(descriptor, data, bytes);
    CHECK(close(descriptor) == 0);
}

static void write_text(const char *path, const char *text) {
    FILE *file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(fputs(text, file) >= 0);
    CHECK(fclose(file) == 0);
}

/* A deterministic latent: distinct, small values that F16 keeps exactly
 * enough for a 1e-3 comparison. */
static void fill_pattern(float *values, size_t count, float scale) {
    for (size_t index = 0; index < count; index++)
        values[index] = scale * (float)((int)(index % 97) - 48) / 32.0f;
}

static uint16_t *to_half(const float *values, size_t count) {
    uint16_t *halves = malloc(count * sizeof(*halves));
    CHECK(halves != NULL);
    for (size_t index = 0; index < count; index++)
        halves[index] = h3_float_to_half(values[index]);
    return halves;
}

/* JSON-escapes a metadata object for use as the "refmod_meta" string. */
static void escape_json(const char *text, char *out, size_t size) {
    size_t length = 0;
    for (; *text && length + 2 < size; text++) {
        if (*text == '"' || *text == '\\') out[length++] = '\\';
        out[length++] = *text;
    }
    out[length] = '\0';
}

static void write_single(const char *name, const char *meta_object,
                         const char *meta_key, const char *shape,
                         const float *values, size_t count) {
    char escaped[2048], header[4096];
    escape_json(meta_object, escaped, sizeof(escaped));
    uint16_t *halves = to_half(values, count);
    if (meta_key)
        snprintf(header, sizeof(header),
                 "{\"__metadata__\":{\"%s\":\"%s\"},\"latent\":{\"dtype\":\"F16\","
                 "\"shape\":%s,\"data_offsets\":[0,%zu]}}", meta_key, escaped,
                 shape, count * 2);
    else
        snprintf(header, sizeof(header),
                 "{\"latent\":{\"dtype\":\"F16\",\"shape\":%s,"
                 "\"data_offsets\":[0,%zu]}}", shape, count * 2);
    write_safetensors(path_in_directory(name), header, halves, count * 2);
    free(halves);
}

static float max_abs_difference(const float *a, const float *b, size_t count) {
    float worst = 0.0f;
    for (size_t index = 0; index < count; index++) {
        float difference = fabsf(a[index] - b[index]);
        if (difference > worst) worst = difference;
    }
    return worst;
}

static void test_single_image(void) {
    enum { T = 1, H = 4, W = 6, COUNT = 24 * T * H * W };
    float values[COUNT];
    fill_pattern(values, COUNT, 1.0f);
    write_single("image.safetensors",
        "{\"name\": \"face\", \"kind\": \"image\", \"latent_h\": 4, \"latent_w\": 6,"
        " \"latent_t\": 1, \"mode\": \"encode\", \"source\": \"image\","
        " \"source_shape\": \"1x4x6\", \"pool\": \"full-res 96x64px (short-edge"
        " cap 1024px)\", \"optimize_steps\": 0, \"tags\": [\"1 img, 0 vid\"],"
        " \"description\": \"a test\", \"concept_type\": \"generic\","
        " \"_format_version\": 4, \"sample_rate\": 32000}",
        "refmod_meta", "[1,24,1,4,6]", values, COUNT);
    h3_refmod_info info;
    char error[512];
    CHECK(h3_refmod_inspect(path_in_directory("image.safetensors"), &info,
                            error, sizeof(error)));
    CHECK(info.format_version == 4 && !info.bundle && info.member_count == 1);
    CHECK(!strcmp(info.name, "face"));
    const h3_refmod_member *member = &info.members[0];
    CHECK(member->kind == H3_REFMOD_IMAGE && member->mode == H3_REFMOD_FULL);
    CHECK(member->latent_t == 1 && member->latent_h == 4 && member->latent_w == 6);
    CHECK(member->tokens == 6 && member->supported);
    CHECK(!strcmp(member->source, "image") && !strcmp(member->tensor, "latent"));
    CHECK(!strcmp(member->description, "a test") && member->format_version == 4);
    h3_refmod_info_free(&info);

    h3_refmod_latent latent;
    h3_refmod_member meta;
    CHECK(h3_refmod_load(path_in_directory("image.safetensors"), -1, 1.0f,
                         &latent, &meta, error, sizeof(error)));
    CHECK(latent.kind == H3_REFMOD_IMAGE && latent.time == 1 &&
          latent.height == 4 && latent.width == 6);
    CHECK(max_abs_difference(latent.values, values, COUNT) < 2e-3f);
    CHECK(meta.tokens == 6);
    /* The row count the DiT sees: one 96-wide row per 2x2 patch. */
    float rows[6 * 96];
    CHECK(h3_dit_patchify_video(latent.values, 24, 1, 4, 6, rows, sizeof(rows) / sizeof(*rows)));
    CHECK(rows[0] == latent.values[0] && rows[1] == latent.values[1]);
    h3_refmod_latent_free(&latent);
    CHECK(latent.values == NULL);

    /* A bundle member index on a single file is an error, as is a strength
     * outside (0, 1]. */
    CHECK(!h3_refmod_load(path_in_directory("image.safetensors"), 1, 1.0f,
                          &latent, NULL, error, sizeof(error)));
    CHECK(strstr(error, "1 RefMod member") != NULL);
    CHECK(!h3_refmod_load(path_in_directory("image.safetensors"), -1, 0.0f,
                          &latent, NULL, error, sizeof(error)));
    CHECK(!h3_refmod_load(path_in_directory("image.safetensors"), -1, 1.5f,
                          &latent, NULL, error, sizeof(error)));
    /* Member 0 on a single file is the same latent. */
    CHECK(h3_refmod_load(path_in_directory("image.safetensors"), 0, 1.0f,
                         &latent, NULL, error, sizeof(error)));
    h3_refmod_latent_free(&latent);
}

static void test_rejections(void) {
    float values[24 * 2 * 4 * 6];
    fill_pattern(values, sizeof(values) / sizeof(*values), 1.0f);
    h3_refmod_info info;
    char error[512];
    /* An image with two latent frames. */
    write_single("bad-t.safetensors",
        "{\"kind\": \"image\", \"name\": \"x\", \"_format_version\": 4}",
        "refmod_meta", "[1,24,2,4,6]", values, 24 * 2 * 4 * 6);
    CHECK(!h3_refmod_inspect(path_in_directory("bad-t.safetensors"), &info,
                             error, sizeof(error)));
    CHECK(strstr(error, "latent frames") != NULL);
    /* An odd width. */
    write_single("bad-w.safetensors",
        "{\"kind\": \"video\", \"name\": \"x\", \"_format_version\": 4}",
        "refmod_meta", "[1,24,2,4,5]", values, 24 * 2 * 4 * 5);
    CHECK(!h3_refmod_inspect(path_in_directory("bad-w.safetensors"), &info,
                             error, sizeof(error)));
    CHECK(strstr(error, "even") != NULL);
    /* Metadata that disagrees with the tensor. */
    write_single("bad-meta.safetensors",
        "{\"kind\": \"video\", \"name\": \"x\", \"latent_t\": 3, \"latent_h\": 4,"
        " \"latent_w\": 6, \"_format_version\": 4}",
        "refmod_meta", "[1,24,2,4,6]", values, 24 * 2 * 4 * 6);
    CHECK(!h3_refmod_inspect(path_in_directory("bad-meta.safetensors"), &info,
                             error, sizeof(error)));
    CHECK(strstr(error, "disagrees") != NULL);
    /* No metadata at all. */
    write_single("plain.safetensors", "", NULL, "[1,24,2,4,6]", values,
                 24 * 2 * 4 * 6);
    CHECK(!h3_refmod_inspect(path_in_directory("plain.safetensors"), &info,
                             error, sizeof(error)));
    CHECK(strstr(error, "not a RefMod") != NULL);
    /* An audio tensor with three "stereo" channels. */
    write_single("bad-audio.safetensors",
        "{\"kind\": \"audio\", \"name\": \"x\", \"_format_version\": 4}",
        "refmod_meta", "[1,32,3,6]", values, 32 * 3 * 6);
    CHECK(!h3_refmod_inspect(path_in_directory("bad-audio.safetensors"), &info,
                             error, sizeof(error)));
    CHECK(strstr(error, "[1,32,2,T]") != NULL);
}

static void test_legacy_placements(void) {
    enum { COUNT = 24 * 2 * 4 * 6 };
    float values[COUNT];
    fill_pattern(values, COUNT, 0.5f);
    h3_refmod_info info;
    char error[512];
    /* Format 2: mode "pooled", no latent_* fields - geometry from the
     * tensor, mode mapped to compressed. */
    write_single("legacy.safetensors",
        "{\"name\": \"old\", \"kind\": \"video\", \"mode\": \"pooled\","
        " \"_format_version\": 2}",
        "refmod_meta", "[1,24,2,4,6]", values, COUNT);
    CHECK(h3_refmod_inspect(path_in_directory("legacy.safetensors"), &info,
                            error, sizeof(error)));
    CHECK(info.format_version == 2 && info.member_count == 1);
    CHECK(info.members[0].mode == H3_REFMOD_COMPRESSED);
    CHECK(info.members[0].latent_t == 2 && info.members[0].latent_h == 4 &&
          info.members[0].latent_w == 6 && info.members[0].tokens == 12);
    h3_refmod_info_free(&info);
    /* The audio pack's header key. */
    write_single("audiokey.safetensors",
        "{\"name\": \"hum\", \"kind\": \"audio\", \"mode\": \"encode\","
        " \"_format_version\": 4}",
        "audio_refmod_meta", "[1,32,2,6]", values, 32 * 2 * 6);
    CHECK(h3_refmod_inspect(path_in_directory("audiokey.safetensors"), &info,
                            error, sizeof(error)));
    CHECK(info.members[0].kind == H3_REFMOD_AUDIO &&
          info.members[0].latent_t == 6 && info.members[0].tokens == 12);
    h3_refmod_info_free(&info);
    /* A sidecar JSON beside a file whose header carries no metadata. */
    write_single("sidecar.safetensors", "", NULL, "[1,24,1,4,6]", values,
                 24 * 4 * 6);
    write_text(path_in_directory("sidecar.json"),
               "{\"name\": \"side\", \"kind\": \"image\", \"mode\": \"full\","
               " \"_format_version\": 3}");
    CHECK(h3_refmod_inspect(path_in_directory("sidecar.safetensors"), &info,
                            error, sizeof(error)));
    CHECK(!strcmp(info.name, "side") && info.members[0].kind == H3_REFMOD_IMAGE &&
          info.members[0].mode == H3_REFMOD_FULL && info.format_version == 3);
    h3_refmod_info_free(&info);
}

static void test_bundle(void) {
    enum { IMAGE = 24 * 1 * 4 * 6, VIDEO = 24 * 3 * 8 * 12, AUDIO = 32 * 2 * 10 };
    static float image[IMAGE], video[VIDEO], audio[AUDIO];
    fill_pattern(image, IMAGE, 1.0f);
    fill_pattern(video, VIDEO, 0.25f);
    fill_pattern(audio, AUDIO, 2.0f);
    uint16_t *image_half = to_half(image, IMAGE);
    uint16_t *video_half = to_half(video, VIDEO);
    uint16_t *audio_half = to_half(audio, AUDIO);
    size_t bytes = (IMAGE + VIDEO + AUDIO) * 2;
    unsigned char *data = malloc(bytes);
    CHECK(data != NULL);
    memcpy(data, image_half, IMAGE * 2);
    memcpy(data + IMAGE * 2, video_half, VIDEO * 2);
    memcpy(data + (IMAGE + VIDEO) * 2, audio_half, AUDIO * 2);
    const char *meta =
        "{\"_format_version\": 5, \"kind\": \"bundle\", \"name\": \"trio\","
        " \"members\": ["
        "{\"name\": \"face\", \"kind\": \"image\", \"latent_t\": 1, \"latent_h\": 4,"
        " \"latent_w\": 6, \"mode\": \"encode\", \"_format_version\": 4,"
        " \"tags\": [\"1 img\"], \"refmod_config\": \"{\\\"retention\\\": 1.0}\"},"
        "{\"name\": \"walk\", \"kind\": \"video\", \"latent_t\": 3, \"latent_h\": 8,"
        " \"latent_w\": 12, \"mode\": \"training\", \"optimize_steps\": 500,"
        " \"pool\": \"3x8x12\", \"_format_version\": 4},"
        "{\"name\": \"hum\", \"kind\": \"audio\", \"latent_t\": 10, \"latent_h\": 0,"
        " \"latent_w\": 0, \"mode\": \"encode\", \"_format_version\": 4}]}";
    char escaped[4096], header[8192];
    escape_json(meta, escaped, sizeof(escaped));
    snprintf(header, sizeof(header),
             "{\"__metadata__\":{\"refmod_meta\":\"%s\"},"
             "\"ref_0\":{\"dtype\":\"F16\",\"shape\":[1,24,1,4,6],\"data_offsets\":[0,%zu]},"
             "\"ref_1\":{\"dtype\":\"F16\",\"shape\":[1,24,3,8,12],\"data_offsets\":[%zu,%zu]},"
             "\"ref_2\":{\"dtype\":\"F16\",\"shape\":[1,32,2,10],\"data_offsets\":[%zu,%zu]}}",
             escaped, (size_t)IMAGE * 2, (size_t)IMAGE * 2,
             (size_t)(IMAGE + VIDEO) * 2, (size_t)(IMAGE + VIDEO) * 2, bytes);
    write_safetensors(path_in_directory("bundle.safetensors"), header, data, bytes);
    free(data); free(image_half); free(video_half); free(audio_half);

    h3_refmod_info info;
    char error[512];
    CHECK(h3_refmod_inspect(path_in_directory("bundle.safetensors"), &info,
                            error, sizeof(error)));
    CHECK(info.format_version == 5 && info.bundle && info.member_count == 3);
    CHECK(!strcmp(info.name, "trio"));
    CHECK(info.members[0].kind == H3_REFMOD_IMAGE && info.members[0].tokens == 6);
    CHECK(info.members[1].kind == H3_REFMOD_VIDEO && info.members[1].tokens == 72 &&
          info.members[1].mode == H3_REFMOD_COMPRESSED &&
          info.members[1].optimize_steps == 500 &&
          !strcmp(info.members[1].pool, "3x8x12"));
    CHECK(info.members[2].kind == H3_REFMOD_AUDIO && info.members[2].tokens == 20 &&
          info.members[2].latent_t == 10);
    CHECK(!strcmp(info.members[1].tensor, "ref_1"));
    h3_refmod_info_free(&info);

    h3_refmod_latent latent;
    CHECK(!h3_refmod_load(path_in_directory("bundle.safetensors"), -1, 1.0f,
                          &latent, NULL, error, sizeof(error)));
    CHECK(strstr(error, "bundle") != NULL);
    CHECK(!h3_refmod_load(path_in_directory("bundle.safetensors"), 3, 1.0f,
                          &latent, NULL, error, sizeof(error)));
    CHECK(h3_refmod_load(path_in_directory("bundle.safetensors"), 1, 1.0f,
                         &latent, NULL, error, sizeof(error)));
    CHECK(latent.kind == H3_REFMOD_VIDEO && latent.time == 3 &&
          latent.height == 8 && latent.width == 12);
    CHECK(max_abs_difference(latent.values, video, VIDEO) < 1e-3f);
    h3_refmod_latent_free(&latent);
    CHECK(h3_refmod_load(path_in_directory("bundle.safetensors"), 2, 1.0f,
                         &latent, NULL, error, sizeof(error)));
    CHECK(latent.kind == H3_REFMOD_AUDIO && latent.audio_t == 10 &&
          latent.time == 0);
    CHECK(max_abs_difference(latent.values, audio, AUDIO) < 4e-3f);
    h3_refmod_latent_free(&latent);
}

static void test_pooling_and_interpolation(void) {
    /* adaptive_avg_pool3d of a 4x6 ramp to 2x3: 2x2 blocks. */
    float ramp[24];
    for (int index = 0; index < 24; index++) ramp[index] = (float)index;
    float pooled[6];
    CHECK(h3_refmod_pool3d(ramp, 1, 1, 4, 6, 1, 2, 3, pooled));
    CHECK(pooled[0] == (0 + 1 + 6 + 7) / 4.0f);
    CHECK(pooled[2] == (4 + 5 + 10 + 11) / 4.0f);
    CHECK(pooled[5] == (16 + 17 + 22 + 23) / 4.0f);
    /* Uneven bins: 5 -> 2 covers [0,3) and [2,5), as PyTorch does. */
    float five[5] = {0, 10, 20, 30, 40}, two[2];
    CHECK(h3_refmod_pool3d(five, 1, 1, 1, 5, 1, 1, 2, two));
    CHECK(two[0] == 10.0f && two[1] == 30.0f);
    /* Linear interpolation 2 -> 4 with align_corners=False. */
    float pair[2] = {1.0f, 3.0f}, four[4];
    CHECK(h3_refmod_trilinear(pair, 1, 1, 1, 2, 1, 1, 4, four));
    CHECK(four[0] == 1.0f && fabsf(four[1] - 1.5f) < 1e-6f &&
          fabsf(four[2] - 2.5f) < 1e-6f && four[3] == 3.0f);
    /* Same size is the identity. */
    float same[6];
    CHECK(h3_refmod_trilinear(pooled, 1, 1, 2, 3, 1, 2, 3, same));
    CHECK(memcmp(same, pooled, sizeof(same)) == 0);
    /* 1 -> 3 repeats the single value. */
    float one = 7.0f, three[3];
    CHECK(h3_refmod_trilinear(&one, 1, 1, 1, 1, 1, 1, 3, three));
    CHECK(three[0] == 7.0f && three[1] == 7.0f && three[2] == 7.0f);
}

static void test_strength(void) {
    enum { C = 24, T = 2, H = 4, W = 6, COUNT = C * T * H * W };
    static float original[COUNT], latent[COUNT];
    fill_pattern(original, COUNT, 1.0f);
    memcpy(latent, original, sizeof(latent));
    CHECK(h3_refmod_apply_strength(latent, C, T, H, W, 1.0f));
    CHECK(memcmp(latent, original, sizeof(latent)) == 0);
    /* Strength 0 is the blur alone: with h/8 and w/8 both 1, every frame
     * of a channel becomes its own mean. */
    CHECK(h3_refmod_apply_strength(latent, C, T, H, W, 0.0f));
    for (int channel = 0; channel < C; channel++)
        for (int t = 0; t < T; t++) {
            double mean = 0.0;
            const float *frame = original + ((size_t)channel * T + t) * H * W;
            for (int index = 0; index < H * W; index++) mean += frame[index];
            mean /= H * W;
            const float *blurred = latent + ((size_t)channel * T + t) * H * W;
            for (int index = 0; index < H * W; index++)
                CHECK(fabs(blurred[index] - mean) < 1e-5);
        }
    /* Strength 0.4 mixes linearly. */
    static float mixed[COUNT];
    memcpy(mixed, original, sizeof(mixed));
    CHECK(h3_refmod_apply_strength(mixed, C, T, H, W, 0.4f));
    for (size_t index = 0; index < COUNT; index++)
        CHECK(fabsf(mixed[index] - (0.4f * original[index] + 0.6f * latent[index])) < 1e-5f);
    /* Out-of-range strengths are refused. */
    CHECK(!h3_refmod_apply_strength(mixed, C, T, H, W, -0.1f));
    CHECK(!h3_refmod_apply_strength(mixed, C, T, H, W, 1.1f));

    /* Audio: a single latent frame is untouched; 16 frames pool to 2. */
    static float audio[32 * 2 * 16], audio_original[32 * 2 * 16];
    fill_pattern(audio_original, 32 * 2 * 16, 1.0f);
    memcpy(audio, audio_original, sizeof(audio));
    CHECK(h3_refmod_apply_strength_audio(audio, 1, 0.0f));
    CHECK(memcmp(audio, audio_original, sizeof(audio)) == 0);
    CHECK(h3_refmod_apply_strength_audio(audio, 16, 0.0f));
    for (int series = 0; series < 64; series++) {
        const float *source = audio_original + (size_t)series * 16;
        float pooled[2], blur[16];
        CHECK(h3_refmod_pool3d(source, 1, 1, 1, 16, 1, 1, 2, pooled));
        CHECK(h3_refmod_trilinear(pooled, 1, 1, 1, 2, 1, 1, 16, blur));
        for (int index = 0; index < 16; index++)
            CHECK(fabsf(audio[(size_t)series * 16 + index] - blur[index]) < 1e-6f);
    }
    /* And through the loader, a 0.5 strength on the bundle's video. */
    h3_refmod_latent loaded;
    char error[512];
    CHECK(h3_refmod_load(path_in_directory("bundle.safetensors"), 1, 0.5f,
                         &loaded, NULL, error, sizeof(error)));
    static float expect[24 * 3 * 8 * 12];
    fill_pattern(expect, 24 * 3 * 8 * 12, 0.25f);
    for (size_t index = 0; index < 24 * 3 * 8 * 12; index++)
        expect[index] = h3_half_to_float(h3_float_to_half(expect[index]));
    CHECK(h3_refmod_apply_strength(expect, 24, 3, 8, 12, 0.5f));
    CHECK(max_abs_difference(loaded.values, expect, 24 * 3 * 8 * 12) < 1e-5f);
    h3_refmod_latent_free(&loaded);
}

static void test_extraction_rules(void) {
    int w, h;
    CHECK(h3_refmod_extract_canvas(1920, 1080, 1024, &w, &h));
    CHECK(w == 1824 && h == 1024);
    CHECK(h3_refmod_extract_canvas(200, 100, 1024, &w, &h));
    CHECK(w == 640 && h == 320);
    CHECK(h3_refmod_extract_canvas(512, 512, 1024, &w, &h));
    CHECK(w == 512 && h == 512);
    CHECK(h3_refmod_extract_canvas(4000, 3000, 0, &w, &h));
    CHECK(w == 1376 && h == 1024);
    CHECK(h3_refmod_extract_canvas(1000, 2000, 1024, &w, &h));
    CHECK(w == 992 && h == 1984);

    int indices[64], count;
    CHECK(h3_refmod_extract_frames(100, 16, indices, &count));
    CHECK(count == 5 && indices[0] == 0 && indices[4] == 99);
    /* 16 evenly spaced of 100 (0, 7, 13, 20, ...), then 5 of those. */
    CHECK(indices[1] == 26 && indices[2] == 53 && indices[3] == 73);
    CHECK(h3_refmod_extract_frames(3, 16, indices, &count));
    CHECK(count == 1 && indices[0] == 0);
    CHECK(h3_refmod_extract_frames(40, 40, indices, &count));
    CHECK(count == 39 && indices[0] == 0 && indices[38] == 39);
    CHECK(h3_refmod_extract_frames(22, 0, indices, &count));
    CHECK(count == 5);
    for (int index = 1; index < count; index++) CHECK(indices[index] > indices[index - 1]);

    int gh, gw;
    h3_refmod_aspect_grid(16, 34, 60, &gh, &gw);
    CHECK(gw == 16 && gh == 10);
    h3_refmod_aspect_grid(16, 64, 64, &gh, &gw);
    CHECK(gw == 16 && gh == 16);
    h3_refmod_aspect_grid(16, 60, 34, &gh, &gw);
    CHECK(gh == 16 && gw == 10);
    h3_refmod_aspect_grid(8, 4, 64, &gh, &gw);
    CHECK(gw == 8 && gh == 2);
    h3_refmod_aspect_grid(16, 4, 6, &gh, &gw);
    CHECK(gh == 4 && gw == 6);
}

static void test_write_round_trip(void) {
    enum { T = 2, H = 6, W = 8, COUNT = 24 * T * H * W };
    static float values[COUNT];
    fill_pattern(values, COUNT, 0.75f);
    h3_refmod_latent latent = {H3_REFMOD_VIDEO, T, H, W, 0, values};
    const char *tags[] = {"0 img, 1 vid", "x1"};
    h3_refmod_meta meta = {H3_REFMOD_VIDEO, H3_REFMOD_COMPRESSED, 500,
                           "walk \"two\"", "video", "7x34x60", "2x6x8",
                           "a note\nwith a newline", NULL, tags, 2};
    char error[512];
    CHECK(h3_refmod_write(path_in_directory("written.safetensors"), &latent,
                          &meta, error, sizeof(error)));
    h3_refmod_info info;
    CHECK(h3_refmod_inspect(path_in_directory("written.safetensors"), &info,
                            error, sizeof(error)));
    CHECK(info.format_version == 4 && !info.bundle && info.member_count == 1);
    CHECK(!strcmp(info.name, "walk \"two\""));
    CHECK(info.members[0].kind == H3_REFMOD_VIDEO &&
          info.members[0].mode == H3_REFMOD_COMPRESSED &&
          info.members[0].optimize_steps == 500 &&
          info.members[0].latent_t == T && info.members[0].latent_h == H &&
          info.members[0].latent_w == W && info.members[0].tokens == T * (H / 2) * (W / 2));
    CHECK(!strcmp(info.members[0].pool, "2x6x8") &&
          !strcmp(info.members[0].source_shape, "7x34x60") &&
          !strcmp(info.members[0].description, "a note\nwith a newline") &&
          !strcmp(info.members[0].concept_type, "generic"));
    h3_refmod_info_free(&info);
    h3_refmod_latent loaded;
    CHECK(h3_refmod_load(path_in_directory("written.safetensors"), -1, 1.0f,
                         &loaded, NULL, error, sizeof(error)));
    CHECK(loaded.time == T && loaded.height == H && loaded.width == W);
    CHECK(max_abs_difference(loaded.values, values, COUNT) < 1e-3f);
    h3_refmod_latent_free(&loaded);
    /* The header is 8-byte aligned, as the format requires. */
    FILE *file = fopen(path_in_directory("written.safetensors"), "rb");
    CHECK(file != NULL);
    uint64_t header_size = 0;
    CHECK(fread(&header_size, 8, 1, file) == 1 && header_size % 8 == 0);
    fclose(file);
    /* No .partial is left behind. */
    CHECK(access(path_in_directory("written.safetensors.partial"), F_OK) != 0);

    /* Audio. */
    static float audio[32 * 2 * 40];
    fill_pattern(audio, 32 * 2 * 40, 1.5f);
    h3_refmod_latent audio_latent = {H3_REFMOD_AUDIO, 0, 0, 0, 40, audio};
    h3_refmod_meta audio_meta = {H3_REFMOD_AUDIO, H3_REFMOD_FULL, 0, "hum", "audio",
                                 "40x0x0", "4x4x1", NULL, NULL, NULL, 0};
    CHECK(h3_refmod_write(path_in_directory("written-audio.safetensors"),
                          &audio_latent, &audio_meta, error, sizeof(error)));
    CHECK(h3_refmod_inspect(path_in_directory("written-audio.safetensors"), &info,
                            error, sizeof(error)));
    CHECK(info.members[0].kind == H3_REFMOD_AUDIO && info.members[0].latent_t == 40 &&
          info.members[0].tokens == 80);
    h3_refmod_info_free(&info);
    CHECK(h3_refmod_load(path_in_directory("written-audio.safetensors"), -1, 1.0f,
                         &loaded, NULL, error, sizeof(error)));
    CHECK(loaded.kind == H3_REFMOD_AUDIO && loaded.audio_t == 40);
    CHECK(max_abs_difference(loaded.values, audio, 32 * 2 * 40) < 2e-3f);
    h3_refmod_latent_free(&loaded);
    /* An image latent with two frames is refused. */
    h3_refmod_meta image_meta = {H3_REFMOD_IMAGE, H3_REFMOD_FULL, 0, "x", NULL,
                                 NULL, NULL, NULL, NULL, NULL, 0};
    CHECK(!h3_refmod_write(path_in_directory("bad.safetensors"), &latent,
                           &image_meta, error, sizeof(error)));
}

static void test_refine(void) {
    /* A small latent refined toward a full one it can reproduce exactly
     * (the full is itself a trilinear upsample of a target): the loss must
     * fall below the pooled start. */
    enum { C = 2, ST = 1, SH = 2, SW = 3, T = 1, H = 8, W = 12 };
    float target[C * ST * SH * SW], full[C * T * H * W];
    fill_pattern(target, C * ST * SH * SW, 1.0f);
    CHECK(h3_refmod_trilinear(target, C, ST, SH, SW, T, H, W, full));
    float small[C * ST * SH * SW];
    CHECK(h3_refmod_pool3d(full, C, T, H, W, ST, SH, SW, small));
    float up[C * T * H * W];
    double before = 0.0, after = 0.0;
    CHECK(h3_refmod_trilinear(small, C, ST, SH, SW, T, H, W, up));
    for (size_t index = 0; index < C * T * H * W; index++)
        before += (up[index] - full[index]) * (up[index] - full[index]);
    char error[512];
    int cancel = 0;
    CHECK(h3_refmod_refine(small, C, ST, SH, SW, full, T, H, W, 200, 0.02f,
                           NULL, NULL, &cancel, error, sizeof(error)));
    CHECK(h3_refmod_trilinear(small, C, ST, SH, SW, T, H, W, up));
    for (size_t index = 0; index < C * T * H * W; index++)
        after += (up[index] - full[index]) * (up[index] - full[index]);
    CHECK(after < before * 0.05);
    /* Zero steps is a no-op; a raised cancel flag stops it. */
    float copy[C * ST * SH * SW];
    memcpy(copy, small, sizeof(copy));
    CHECK(h3_refmod_refine(small, C, ST, SH, SW, full, T, H, W, 0, 0.02f,
                           NULL, NULL, NULL, error, sizeof(error)));
    CHECK(memcmp(copy, small, sizeof(copy)) == 0);
    cancel = 1;
    CHECK(!h3_refmod_refine(small, C, ST, SH, SW, full, T, H, W, 10, 0.02f,
                            NULL, NULL, &cancel, error, sizeof(error)));
    CHECK(strstr(error, "cancelled") != NULL);
}

/* Against torch (tools/dump_refmod_fixture.py), when the fixture exists. */
static void test_fixture(const char *path) {
    h3_st_header header;
    char error[512];
    if (!h3_st_read_header(path, &header, error, sizeof(error))) {
        printf("skip: RefMod numeric fixture %s is not installed\n", path);
        return;
    }
    enum { C = 24, T = 3, H = 16, W = 24, PT = 2, PH = 8, PW = 12 };
    enum { FULL = C * T * H * W, SMALL = C * PT * PH * PW };
    static float z[FULL], expect_blur[FULL], expect_mix[FULL], expect_up[FULL],
                 pooled[SMALL], expect_pooled[SMALL], expect_refined[SMALL],
                 expect_grad[SMALL], work[FULL];
    #define LOAD(name, buffer, count) do { \
        const h3_st_tensor *tensor = h3_st_find(&header, name); \
        CHECK(tensor && tensor->dtype == H3_DTYPE_F32 && \
              h3_st_tensor_elements(tensor) == (count)); \
        CHECK(h3_st_read_data(&header, tensor, buffer, (count) * sizeof(float), \
                              error, sizeof(error))); \
    } while (0)
    LOAD("z", z, FULL);
    LOAD("blur", expect_blur, FULL);
    LOAD("mix_0_4", expect_mix, FULL);
    LOAD("pooled_2x8x12", expect_pooled, SMALL);
    LOAD("up_from_pooled", expect_up, FULL);
    LOAD("refined_20", expect_refined, SMALL);
    LOAD("grad_0", expect_grad, SMALL);
    #undef LOAD
    h3_st_free_header(&header);
    /* Pooling and the upsample back. */
    CHECK(h3_refmod_pool3d(z, C, T, H, W, PT, PH, PW, pooled));
    CHECK(max_abs_difference(pooled, expect_pooled, SMALL) < 1e-5f);
    CHECK(h3_refmod_trilinear(expect_pooled, C, PT, PH, PW, T, H, W, work));
    CHECK(max_abs_difference(work, expect_up, FULL) < 1e-5f);
    /* The strength blur and mix. */
    memcpy(work, z, sizeof(work));
    CHECK(h3_refmod_apply_strength(work, C, T, H, W, 0.0f));
    CHECK(max_abs_difference(work, expect_blur, FULL) < 1e-5f);
    memcpy(work, z, sizeof(work));
    CHECK(h3_refmod_apply_strength(work, C, T, H, W, 0.4f));
    CHECK(max_abs_difference(work, expect_mix, FULL) < 1e-5f);
    /* The first Adam step's gradient (the trilinear adjoint of the loss)
     * and 20 steps of refinement from the F16-rounded pool. */
    static float small[SMALL];
    for (size_t index = 0; index < SMALL; index++)
        small[index] = h3_half_to_float(h3_float_to_half(expect_pooled[index]));
    static float start[SMALL];
    memcpy(start, small, sizeof(start));
    CHECK(h3_refmod_refine(small, C, PT, PH, PW, z, T, H, W, 1, 0.02f,
                           NULL, NULL, NULL, error, sizeof(error)));
    /* After one Adam step every parameter moves by lr in the direction of
     * minus the gradient (m_hat/sqrt(v_hat) = sign(g)), so the gradient's
     * sign is checked through the step itself. */
    for (size_t index = 0; index < SMALL; index++) {
        /* Adam's epsilon blunts the step where the gradient is tiny. */
        if (fabsf(expect_grad[index]) < 1e-5f) continue;
        float moved = small[index] - start[index];
        CHECK(fabsf(moved + 0.02f * (expect_grad[index] > 0 ? 1.0f : -1.0f)) < 1e-4f);
    }
    memcpy(small, start, sizeof(start));
    CHECK(h3_refmod_refine(small, C, PT, PH, PW, z, T, H, W, 20, 0.02f,
                           NULL, NULL, NULL, error, sizeof(error)));
    CHECK(max_abs_difference(small, expect_refined, SMALL) < 1e-3f);
    printf("ok: RefMod arithmetic matches the torch fixture\n");
}

static void test_half_round_trip(void) {
    const float samples[] = {0.0f, 1.0f, -1.0f, 0.5f, 65504.0f, 1e-7f,
                             3.14159f, -0.33333f, 1024.5f, 6.1e-5f};
    for (size_t index = 0; index < sizeof(samples) / sizeof(*samples); index++) {
        float value = samples[index];
        float back = h3_half_to_float(h3_float_to_half(value));
        float tolerance = fabsf(value) * (1.0f / 1024.0f) + 1e-7f;
        CHECK(fabsf(back - value) <= tolerance);
    }
    CHECK(h3_float_to_half(1e10f) == 0x7c00u);
    CHECK(h3_float_to_half(-1e10f) == 0xfc00u);
    /* Round to nearest even at a tie: 1 + 2^-11 is halfway between two
     * halves and rounds to the even one (1.0). */
    CHECK(h3_float_to_half(1.0f + 1.0f / 2048.0f) == 0x3c00u);
    CHECK(h3_float_to_half(1.0f + 3.0f / 2048.0f) == 0x3c02u);
}

static void test_ref2va_labels(const char *tokenizer_path) {
    char error[512];
    h3_tokenizer *tokenizer = h3_tokenizer_load(tokenizer_path, error, sizeof(error));
    CHECK(tokenizer != NULL);
    h3_vision_output vision = {4, 6, 6, NULL, {NULL}, {0}};
    h3_reference_presentation image = {H3_PRESENTATION_IMAGE, 0, &vision, 1, NULL};
    h3_reference_presentation hidden = {H3_PRESENTATION_HIDDEN, 0, NULL, 0, NULL};
    h3_reference_presentation audio = {H3_PRESENTATION_AUDIO, 1, NULL, 0, NULL};
    h3_reference_presentation with_hidden[3] = {image, hidden, audio};
    h3_reference_presentation without[2] = {image, audio};
    size_t tokens_with = 0, tokens_without = 0, tokens_alone = 0;
    CHECK(h3_multimodal_ref2va_token_count(tokenizer, "a cat", with_hidden, 3,
                                           &tokens_with, error, sizeof(error)));
    CHECK(h3_multimodal_ref2va_token_count(tokenizer, "a cat", without, 2,
                                           &tokens_without, error, sizeof(error)));
    CHECK(tokens_with == tokens_without);
    /* A hidden reference alone is accepted: the prompt and nothing else. */
    CHECK(h3_multimodal_ref2va_token_count(tokenizer, "a cat", &hidden, 1,
                                           &tokens_alone, error, sizeof(error)));
    CHECK(tokens_alone > 0 && tokens_alone < tokens_without);
    /* An audio label alone still needs a visual. */
    CHECK(!h3_multimodal_ref2va_token_count(tokenizer, "a cat", &audio, 1,
                                            &tokens_alone, error, sizeof(error)));
    h3_tokenizer_free(tokenizer);
    printf("ok: Ref2VA labels skip hidden references (%zu tokens)\n", tokens_with);
}

int main(int argc, char **argv) {
    const char *base = getenv("TMPDIR");
    snprintf(directory, sizeof(directory), "%s/h3-refmod-tests-XXXXXX",
             base && *base ? base : "/tmp");
    CHECK(mkdtemp(directory) != NULL);
    test_half_round_trip();
    test_single_image();
    test_rejections();
    test_legacy_placements();
    test_bundle();
    test_pooling_and_interpolation();
    test_strength();
    test_extraction_rules();
    test_write_round_trip();
    test_refine();
    test_fixture(argc > 2 ? argv[2] : "misc/fixtures/h3_refmod_fixture.safetensors");
    if (argc > 1 && *argv[1]) test_ref2va_labels(argv[1]);
    else printf("skip: Ref2VA label test needs a tokenizer path\n");
    char command[1400];
    snprintf(command, sizeof(command), "rm -rf '%s'", directory);
    (void)system(command);
    printf("ok: %d RefMod checks\n", checks);
    return 0;
}
