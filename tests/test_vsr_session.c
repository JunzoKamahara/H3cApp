/* End to end: a synthetic SwiftVR (tiny Wan + a full-shape random ReAE)
 * through h3_vsr_open / session / process on a few video lengths, checking
 * the chunk accounting (every real frame comes out once, in order), the
 * pixel formats, determinism, cancellation and the error paths. GPU, no
 * weights; runs in `make test`. */
#include "h3_vsr_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int tests_run;

#define CHECK(condition) do { \
    tests_run++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

enum { LAYERS = 2, HEADS = 2, HEAD_DIM = 64, DIM = 128, FFN = 256, IN = 48,
       TEXT_DIM = 64, FREQ = 256, TOKENS = 8, PATCH = IN * 4 };

static uint64_t rng_state = 0xA4093822299F31D0ull;
static float rnd(void) {
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((rng_state >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f;
}

typedef struct { char name[128]; const char *dtype; int ndim; uint64_t shape[5]; void *data; size_t bytes; } tensor_spec;
static tensor_spec specs[512];
static int spec_count;

static void write_safetensors(const char *path, tensor_spec *tensors, int count) {
    char *header = malloc(1 << 20);
    size_t at = 0;
    uint64_t offset = 0;
    at += (size_t)sprintf(header + at, "{");
    for (int i = 0; i < count; i++) {
        at += (size_t)sprintf(header + at, "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[", i ? "," : "",
                              tensors[i].name, tensors[i].dtype);
        for (int d = 0; d < tensors[i].ndim; d++)
            at += (size_t)sprintf(header + at, "%s%llu", d ? "," : "", (unsigned long long)tensors[i].shape[d]);
        at += (size_t)sprintf(header + at, "],\"data_offsets\":[%llu,%llu]}", (unsigned long long)offset,
                              (unsigned long long)(offset + tensors[i].bytes));
        offset += tensors[i].bytes;
    }
    at += (size_t)sprintf(header + at, "}");
    while (at % 8) header[at++] = ' ';
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    uint64_t length = at;
    CHECK(fwrite(&length, 8, 1, f) == 1 && fwrite(header, 1, at, f) == at);
    for (int i = 0; i < count; i++) CHECK(fwrite(tensors[i].data, 1, tensors[i].bytes, f) == tensors[i].bytes);
    fclose(f);
    free(header);
}

static float *add_tensor(const char *name, int ndim, const uint64_t *shape, float scale) {
    size_t n = 1;
    for (int d = 0; d < ndim; d++) n *= shape[d];
    float *v = malloc(n * sizeof(float));
    CHECK(v != NULL);
    for (size_t i = 0; i < n; i++) v[i] = rnd() * scale;
    tensor_spec *t = &specs[spec_count++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->dtype = "F32"; t->ndim = ndim; memcpy(t->shape, shape, sizeof(uint64_t) * (size_t)ndim);
    t->data = v; t->bytes = n * 4;
    return v;
}
static void add2(const char *name, uint64_t r, uint64_t c, float scale) { uint64_t s[2] = {r, c}; add_tensor(name, 2, s, scale); }
static void add1(const char *name, uint64_t n, float scale) { uint64_t s[1] = {n}; add_tensor(name, 1, s, scale); }
static void conv(const char *prefix, int out, int in, int bias) {
    char name[160]; uint64_t s[4] = {(uint64_t)out, (uint64_t)in, 3, 3};
    snprintf(name, sizeof(name), "%s.weight", prefix); add_tensor(name, 4, s, 1.5f / sqrtf((float)(in * 9)));
    if (bias) { snprintf(name, sizeof(name), "%s.bias", prefix); add1(name, (uint64_t)out, 0.05f); }
}
static void memblock(const char *prefix, int c) {
    char n[160];
    snprintf(n, sizeof(n), "%s.conv.0", prefix); conv(n, c, 2 * c, 1);
    snprintf(n, sizeof(n), "%s.conv.2", prefix); conv(n, c, c, 1);
    snprintf(n, sizeof(n), "%s.conv.4", prefix); conv(n, c, c, 1);
}
static void free_specs(void) { for (int i = 0; i < spec_count; i++) free(specs[i].data); spec_count = 0; }

static void write_model(const char *dir) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/transformer", dir);
    CHECK(mkdir(dir, 0755) == 0 && mkdir(path, 0755) == 0);
    snprintf(path, sizeof(path), "%s/transformer/config.json", dir);
    FILE *f = fopen(path, "w");
    CHECK(f != NULL);
    fprintf(f, "{\"attention_head_dim\": %d, \"cross_attn_norm\": true, \"eps\": 1e-06, \"ffn_dim\": %d,"
               " \"freq_dim\": %d, \"in_channels\": %d, \"num_attention_heads\": %d, \"num_layers\": %d,"
               " \"out_channels\": %d, \"patch_size\": [1, 2, 2], \"text_dim\": %d}\n",
            HEAD_DIM, FFN, FREQ, IN, HEADS, LAYERS, IN, TEXT_DIM);
    fclose(f);
    /* The DiT. */
    spec_count = 0;
    const float W = 0.08f;
    add2("condition_embedder.time_embedder.linear_1.weight", DIM, FREQ, W); add1("condition_embedder.time_embedder.linear_1.bias", DIM, 0.1f);
    add2("condition_embedder.time_embedder.linear_2.weight", DIM, DIM, W); add1("condition_embedder.time_embedder.linear_2.bias", DIM, 0.1f);
    add2("condition_embedder.time_proj.weight", 6 * DIM, DIM, W); add1("condition_embedder.time_proj.bias", 6 * DIM, 0.1f);
    add2("condition_embedder.text_embedder.linear_1.weight", DIM, TEXT_DIM, W); add1("condition_embedder.text_embedder.linear_1.bias", DIM, 0.1f);
    add2("condition_embedder.text_embedder.linear_2.weight", DIM, DIM, W); add1("condition_embedder.text_embedder.linear_2.bias", DIM, 0.1f);
    { uint64_t s[5] = {DIM, IN, 1, 2, 2}; add_tensor("patch_embedding.weight", 5, s, W); }
    add1("patch_embedding.bias", DIM, 0.1f);
    add2("proj_out.weight", PATCH, DIM, W); add1("proj_out.bias", PATCH, 0.1f);
    { uint64_t s[3] = {1, 2, DIM}; add_tensor("scale_shift_table", 3, s, 0.3f); }
    for (int b = 0; b < LAYERS; b++) {
        char name[160];
#define N(suffix) (snprintf(name, sizeof(name), "blocks.%d.%s", b, suffix), name)
        add2(N("attn1.to_q.weight"), DIM, DIM, W); add1(N("attn1.to_q.bias"), DIM, 0.1f);
        add2(N("attn1.to_k.weight"), DIM, DIM, W); add1(N("attn1.to_k.bias"), DIM, 0.1f);
        add2(N("attn1.to_v.weight"), DIM, DIM, W); add1(N("attn1.to_v.bias"), DIM, 0.1f);
        add1(N("attn1.norm_q.weight"), DIM, 1.0f); add1(N("attn1.norm_k.weight"), DIM, 1.0f);
        add2(N("attn1.to_out.0.weight"), DIM, DIM, W); add1(N("attn1.to_out.0.bias"), DIM, 0.1f);
        add2(N("attn2.to_q.weight"), DIM, DIM, W); add1(N("attn2.to_q.bias"), DIM, 0.1f);
        add2(N("attn2.to_k.weight"), DIM, DIM, W); add1(N("attn2.to_k.bias"), DIM, 0.1f);
        add2(N("attn2.to_v.weight"), DIM, DIM, W); add1(N("attn2.to_v.bias"), DIM, 0.1f);
        add1(N("attn2.norm_q.weight"), DIM, 1.0f); add1(N("attn2.norm_k.weight"), DIM, 1.0f);
        add2(N("attn2.to_out.0.weight"), DIM, DIM, W); add1(N("attn2.to_out.0.bias"), DIM, 0.1f);
        add1(N("norm2.weight"), DIM, 1.0f); add1(N("norm2.bias"), DIM, 0.1f);
        add2(N("ffn.net.0.proj.weight"), FFN, DIM, W); add1(N("ffn.net.0.proj.bias"), FFN, 0.1f);
        add2(N("ffn.net.2.weight"), DIM, FFN, W); add1(N("ffn.net.2.bias"), DIM, 0.1f);
        uint64_t s[3] = {1, 6, DIM};
        add_tensor(N("scale_shift_table"), 3, s, 0.3f);
#undef N
    }
    snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors", dir);
    write_safetensors(path, specs, spec_count);
    free_specs();
    /* The prompt. */
    uint16_t *half = malloc((size_t)TOKENS * TEXT_DIM * 2);
    for (size_t i = 0; i < (size_t)TOKENS * TEXT_DIM; i++) { float x = rnd(); uint32_t b; memcpy(&b, &x, 4); half[i] = (uint16_t)(b >> 16); }
    tensor_spec prompt = {"prompt_emb", "BF16", 3, {1, TOKENS, TEXT_DIM, 0, 0}, half, (size_t)TOKENS * TEXT_DIM * 2};
    snprintf(path, sizeof(path), "%s/prompt_embedding.safetensors", dir);
    write_safetensors(path, &prompt, 1);
    free(half);
    /* The ReAE at its real shapes (random). */
    spec_count = 0;
    conv("encoder.0", 64, 12, 1);
    { uint64_t s[4] = {64, 128, 1, 1}; add_tensor("encoder.2.conv.weight", 4, s, 0.1f); }
    conv("encoder.3", 64, 64, 0);
    for (int i = 4; i <= 6; i++) { char p[32]; snprintf(p, sizeof(p), "encoder.%d", i); memblock(p, 64); }
    { uint64_t s[4] = {64, 128, 1, 1}; add_tensor("encoder.7.conv.weight", 4, s, 0.1f); }
    conv("encoder.8", 64, 64, 0);
    for (int i = 9; i <= 11; i++) { char p[32]; snprintf(p, sizeof(p), "encoder.%d", i); memblock(p, 64); }
    { uint64_t s[4] = {64, 64, 1, 1}; add_tensor("encoder.12.conv.weight", 4, s, 0.12f); }
    conv("encoder.13", 64, 64, 0);
    for (int i = 14; i <= 16; i++) { char p[32]; snprintf(p, sizeof(p), "encoder.%d", i); memblock(p, 64); }
    conv("encoder.17", 48, 64, 1);
    conv("decoder.1", 512, 48, 1);
    for (int i = 3; i <= 5; i++) { char p[32]; snprintf(p, sizeof(p), "decoder.%d", i); memblock(p, 512); }
    { uint64_t s[4] = {512, 512, 1, 1}; add_tensor("decoder.7.proj.weight", 4, s, 0.06f); }
    conv("decoder.8", 256, 512, 0);
    for (int i = 9; i <= 11; i++) { char p[32]; snprintf(p, sizeof(p), "decoder.%d", i); memblock(p, 256); }
    { uint64_t s[5] = {256, 256, 3, 1, 1}; add_tensor("decoder.13.conv3d.weight", 5, s, 0.05f); }
    conv("decoder.14", 128, 256, 0);
    for (int i = 15; i <= 17; i++) { char p[32]; snprintf(p, sizeof(p), "decoder.%d", i); memblock(p, 128); }
    { uint64_t s[5] = {128, 128, 3, 1, 1}; add_tensor("decoder.19.conv3d.weight", 5, s, 0.07f); }
    conv("decoder.20", 64, 128, 0);
    conv("decoder.22", 12, 64, 1);
    snprintf(path, sizeof(path), "%s/reae.safetensors", dir);
    write_safetensors(path, specs, spec_count);
    free_specs();
}

/* A deterministic 8-bit test video. */
static uint8_t *make_video(int frames, int w, int h, int bgra) {
    int bpp = bgra ? 4 : 3;
    uint8_t *v = malloc((size_t)frames * h * w * bpp);
    CHECK(v != NULL);
    for (int t = 0; t < frames; t++) for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        uint8_t *p = v + (((size_t)t * h + y) * w + x) * bpp;
        uint8_t r = (uint8_t)(128 + 100 * sin(0.2 * x + 0.1 * t)), g = (uint8_t)(128 + 100 * cos(0.15 * y - 0.1 * t));
        uint8_t b = (uint8_t)((x * 7 + y * 3 + t * 11) & 255);
        if (bgra) { p[0] = b; p[1] = g; p[2] = r; p[3] = 255; } else { p[0] = r; p[1] = g; p[2] = b; }
    }
    return v;
}

/* Feeds `frames` frames in chunks and returns the concatenated output. */
static uint8_t *run_video(h3_vsr *vsr, const uint8_t *video, int frames, int in_w, int in_h,
                          int out_w, int out_h, int bgra, int *out_total, int cancel_at_chunk) {
    char error[512] = {0};
    h3_vsr_session *s = h3_vsr_session_open(vsr, in_w, in_h, out_w, out_h, frames, error, sizeof(error));
    if (!s) fprintf(stderr, "%s\n", error);
    CHECK(s != NULL);
    int bpp = bgra ? 4 : 3;
    uint8_t *out = malloc((size_t)frames * out_h * out_w * bpp);
    memset(out, 0x55, (size_t)frames * out_h * out_w * bpp);
    h3_vsr_image in[32];
    h3_vsr_image_out outs[64];
    int fed = 0, total = 0, chunk = 0;
    volatile int cancel = 0;
    while (fed < frames) {
        int want = h3_vsr_session_next_chunk_frames(s);
        CHECK(want == (chunk == 0 ? 28 : 24));
        int n = frames - fed < want ? frames - fed : want;
        int is_last = fed + n >= frames;
        for (int i = 0; i < n; i++) {
            in[i].pixels = video + (size_t)(fed + i) * in_h * in_w * bpp;
            in[i].width = in_w; in[i].height = in_h; in[i].stride_bytes = in_w * bpp;
            in[i].format = bgra ? H3_VSR_BGRA8 : H3_VSR_RGB24;
        }
        for (int i = 0; i < 64; i++) {
            int frame = total + i;
            outs[i].pixels = frame < frames ? out + (size_t)frame * out_h * out_w * bpp : NULL;
            outs[i].width = out_w; outs[i].height = out_h; outs[i].stride_bytes = out_w * bpp;
            outs[i].format = bgra ? H3_VSR_BGRA8 : H3_VSR_RGB24;
        }
        int capacity = frames - total < 64 ? frames - total : 64;
        int got = 0;
        if (cancel_at_chunk == chunk) cancel = 1;
        int ok = h3_vsr_session_process(s, in, n, is_last, outs, capacity, &got, NULL, NULL, &cancel,
                                        error, sizeof(error));
        if (cancel_at_chunk == chunk) {
            CHECK(!ok && strstr(error, "cancelled") != NULL);
            h3_vsr_session_close(s);
            free(out);
            *out_total = -1;
            return NULL;
        }
        if (!ok) fprintf(stderr, "%s\n", error);
        CHECK(ok);
        CHECK(got >= 0 && total + got <= frames);
        total += got;
        fed += n;
        chunk++;
    }
    CHECK(h3_vsr_session_pending_frames(s) == 0);
    h3_vsr_session_close(s);
    *out_total = total;
    return out;
}

int main(void) {
    char dir[] = "/tmp/h3vsr-session-test-XXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    char model_dir[4096], cache_path[4096], error[512] = {0};
    snprintf(model_dir, sizeof(model_dir), "%s/model", dir);
    write_model(model_dir);
    CHECK(h3_vsr_cache_path(model_dir, H3_VSR_PRECISION_BF16, cache_path, sizeof(cache_path)));
    CHECK(h3_vsr_build_cache(model_dir, cache_path, H3_VSR_PRECISION_BF16, "h3_shaders.metal", NULL, NULL,
                             error, sizeof(error)));

    h3_vsr_open_params params = {0};
    params.precision = H3_VSR_PRECISION_BF16;
    params.cache_path = cache_path;
    params.shader_source_path = "h3_shaders.metal";
    h3_vsr *vsr = h3_vsr_open(model_dir, &params, NULL, NULL, error, sizeof(error));
    if (!vsr) fprintf(stderr, "%s\n", error);
    CHECK(vsr != NULL);

    /* Requirements: sizes outside the model's range are refused with a reason. */
    h3_vsr_requirements req;
    CHECK(h3_vsr_check_requirements(vsr, 64, 48, 160, 128, &req) && req.supported);
    CHECK(req.tokens_per_chunk == 7 * (160 / 32) * (128 / 32));
    CHECK(req.windows_even == 1 && req.windows_odd == 1);
    CHECK(h3_vsr_check_requirements(vsr, 64, 48, 96, 64, &req) && !req.supported && req.reason[0]);
    CHECK(h3_vsr_check_requirements(vsr, 64, 48, 5000, 128, &req) && !req.supported);
    CHECK(h3_vsr_check_requirements(vsr, 64, 48, 150, 130, &req) && req.supported);   /* padded to 160x160 */

    const int in_w = 64, in_h = 48, out_w = 150, out_h = 130;   /* an odd target: pad + crop */
    int lengths[] = {7, 29, 52, 70};
    for (size_t i = 0; i < sizeof(lengths) / sizeof(*lengths); i++) {
        int frames = lengths[i];
        int bgra = i % 2 == 0;
        uint8_t *video = make_video(frames, in_w, in_h, bgra);
        int total = 0;
        uint8_t *out = run_video(vsr, video, frames, in_w, in_h, out_w, out_h, bgra, &total, -1);
        CHECK(total == frames);
        int bpp = bgra ? 4 : 3;
        /* Every output frame was written (no 0x55 fill rows left) and BGRA alpha is 255. */
        for (int t = 0; t < frames; t++) {
            const uint8_t *frame = out + (size_t)t * out_h * out_w * bpp;
            int untouched = 1;
            for (size_t k = 0; k < (size_t)out_h * out_w * bpp && untouched; k++) if (frame[k] != 0x55) untouched = 0;
            CHECK(!untouched);
            if (bgra) for (int k = 0; k < out_h * out_w; k++) CHECK(frame[k * 4 + 3] == 255);
        }
        /* Deterministic. */
        int total2 = 0;
        uint8_t *again = run_video(vsr, video, frames, in_w, in_h, out_w, out_h, bgra, &total2, -1);
        CHECK(total2 == frames && memcmp(out, again, (size_t)frames * out_h * out_w * bpp) == 0);
        printf("  %d frames (%s): %d out, deterministic\n", frames, bgra ? "BGRA" : "RGB24", total);
        free(out); free(again); free(video);
    }
    /* Cancellation stops the second chunk with the "cancelled" error. */
    {
        uint8_t *video = make_video(52, in_w, in_h, 1);
        int total = 0;
        uint8_t *out = run_video(vsr, video, 52, in_w, in_h, out_w, out_h, 1, &total, 1);
        CHECK(out == NULL && total == -1);
        free(video);
    }
    /* A regular call with the wrong frame count is refused. */
    {
        h3_vsr_session *s = h3_vsr_session_open(vsr, in_w, in_h, out_w, out_h, 0, error, sizeof(error));
        CHECK(s != NULL);
        uint8_t *video = make_video(5, in_w, in_h, 1);
        h3_vsr_image in[5];
        for (int i = 0; i < 5; i++) { in[i].pixels = video + (size_t)i * in_h * in_w * 4; in[i].width = in_w; in[i].height = in_h; in[i].stride_bytes = in_w * 4; in[i].format = H3_VSR_BGRA8; }
        h3_vsr_image_out outs[8];
        uint8_t scratch[8][150 * 130 * 4];
        for (int i = 0; i < 8; i++) { outs[i].pixels = scratch[i]; outs[i].width = out_w; outs[i].height = out_h; outs[i].stride_bytes = out_w * 4; outs[i].format = H3_VSR_BGRA8; }
        int got = 0;
        CHECK(!h3_vsr_session_process(s, in, 5, 0, outs, 8, &got, NULL, NULL, NULL, error, sizeof(error)));
        CHECK(strstr(error, "exactly 28") != NULL);
        h3_vsr_session_close(s);
        free(video);
    }
    double dit = 0, reae = 0; uint64_t peak = 0;
    CHECK(h3_vsr_get_stats(vsr, &dit, &reae, &peak) && dit > 0.0 && peak > 0);
    /* The decoder on its own context (output one chunk behind): the same
     * frames, bit for bit. */
    {
        h3_vsr_open_params overlap = params;
        overlap.reae_separate_queue = 1;
        h3_vsr *vsr2 = h3_vsr_open(model_dir, &overlap, NULL, NULL, error, sizeof(error));
        if (!vsr2) fprintf(stderr, "%s\n", error);
        CHECK(vsr2 != NULL);
        uint8_t *video = make_video(52, in_w, in_h, 0);
        int total = 0, total2 = 0;
        uint8_t *one = run_video(vsr, video, 52, in_w, in_h, out_w, out_h, 0, &total, -1);
        uint8_t *two = run_video(vsr2, video, 52, in_w, in_h, out_w, out_h, 0, &total2, -1);
        CHECK(total == 52 && total2 == 52 && memcmp(one, two, (size_t)52 * out_h * out_w * 3) == 0);
        double dit2 = 0, reae2 = 0; uint64_t peak2 = 0;
        CHECK(h3_vsr_get_stats(vsr2, &dit2, &reae2, &peak2) && reae2 > 0.0);
        printf("  decoder on a second context: 52 frames identical\n");
        free(one); free(two); free(video);
        h3_vsr_close(vsr2);
    }
    /* Spatial tiles (H3_VSR_TILES forces a grid the memory would not ask
     * for): the frame is 160x160 padded, so 2 x 2 tiles of 96 with 32 of
     * overlap. Every frame comes out, deterministic, and close to the
     * one-tile output: the ReAE is convolutional and the DiT's windows
     * only change which tokens attend to which. */
    {
        uint8_t *video = make_video(29, in_w, in_h, 1);
        int total = 0;
        uint8_t *whole = run_video(vsr, video, 29, in_w, in_h, out_w, out_h, 1, &total, -1);
        CHECK(total == 29);
        setenv("H3_VSR_TILES", "2x2", 1);
        setenv("H3_VSR_TILE_OVERLAP", "32", 1);
        CHECK(h3_vsr_check_requirements(vsr, in_w, in_h, out_w, out_h, &req) && req.supported);
        CHECK(req.tile_columns == 2 && req.tile_rows == 2 && req.tile_width == 96 && req.tile_height == 96);
        CHECK(req.tokens_per_chunk == 4 * 7 * 3 * 3);
        int total2 = 0, total3 = 0;
        uint8_t *tiled = run_video(vsr, video, 29, in_w, in_h, out_w, out_h, 1, &total2, -1);
        uint8_t *again = run_video(vsr, video, 29, in_w, in_h, out_w, out_h, 1, &total3, -1);
        CHECK(total2 == 29 && total3 == 29);
        CHECK(memcmp(tiled, again, (size_t)29 * out_h * out_w * 4) == 0);
        double se = 0.0;
        int alpha_ok = 1;
        for (size_t k = 0; k < (size_t)29 * out_h * out_w; k++) {
            for (int c = 0; c < 3; c++) {
                double d = (double)tiled[k * 4 + c] - (double)whole[k * 4 + c];
                se += d * d;
            }
            if (tiled[k * 4 + 3] != 255) alpha_ok = 0;
        }
        double psnr = 10.0 * log10(255.0 * 255.0 / (se / ((double)29 * out_h * out_w * 3) + 1e-12));
        printf("  2 x 2 tiles: 29 frames, %.1f dB against one tile\n", psnr);
        CHECK(alpha_ok);
        CHECK(psnr > 20.0);
        /* A grid the frame cannot hold (8 tiles of at least 64 on 160) is refused. */
        setenv("H3_VSR_TILES", "8x8", 1);
        CHECK(h3_vsr_check_requirements(vsr, in_w, in_h, out_w, out_h, &req) && !req.supported && req.reason[0]);
        CHECK(h3_vsr_session_open(vsr, in_w, in_h, out_w, out_h, 0, error, sizeof(error)) == NULL);
        unsetenv("H3_VSR_TILES");
        unsetenv("H3_VSR_TILE_OVERLAP");
        free(whole); free(tiled); free(again); free(video);
    }
    h3_vsr_close(vsr);
    char command[4200];
    snprintf(command, sizeof(command), "rm -rf '%s'", dir);
    CHECK(system(command) == 0);
    printf("ok: %d checks (vsr session end to end)\n", tests_run);
    return 0;
}
