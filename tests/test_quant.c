/* Block-quantized weights: the reference dequantizers on hand-built
 * blocks, a synthetic GGUF file read through the weight store, and the GPU
 * expansion (h3_gpu_dequantize) against the reference for every format on
 * random blocks. */
#include "h3_gpu.h"
#include "h3_quant.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int checks = 0;

#define CHECK(condition) do { \
    checks++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static uint16_t float_to_half(float value) {
    /* Exact for the small values the tests use. */
    _Float16 half = (_Float16)value;
    uint16_t bits;
    memcpy(&bits, &half, sizeof(bits));
    return bits;
}

static void put_u16(uint8_t *at, uint16_t value) {
    at[0] = (uint8_t)value;
    at[1] = (uint8_t)(value >> 8);
}

static void test_reference(void) {
    CHECK(h3_quant_block_bytes(H3_QUANT_Q8_0) == 34);
    CHECK(h3_quant_block_bytes(H3_QUANT_Q4_K) == 144);
    CHECK(h3_quant_block_bytes(H3_QUANT_Q6_K) == 210);
    CHECK(h3_quant_block_bytes(H3_QUANT_Q3_K) == 110);
    CHECK(h3_quant_block_bytes(H3_QUANT_Q2_K) == 84);
    CHECK(h3_quant_block_bytes(H3_QUANT_Q5_K) == 176);
    CHECK(h3_quant_bytes(H3_QUANT_Q8_0, 64) == 68);
    CHECK(h3_quant_bytes(H3_QUANT_Q8_0, 33) == 0);
    CHECK(h3_quant_from_ggml(8) == H3_QUANT_Q8_0);
    CHECK(h3_quant_from_ggml(12) == H3_QUANT_Q4_K);
    CHECK(h3_quant_from_ggml(30) == H3_QUANT_BF16);
    CHECK(h3_quant_from_ggml(16) == H3_QUANT_NONE); /* IQ2_XXS */

    /* Q8_0: d = 0.5, q = j - 16. */
    uint8_t q8[34];
    put_u16(q8, float_to_half(0.5f));
    for (int j = 0; j < 32; j++) q8[2 + j] = (uint8_t)(int8_t)(j - 16);
    float out[256];
    CHECK(h3_quant_dequantize(H3_QUANT_Q8_0, q8, 32, out));
    for (int j = 0; j < 32; j++) CHECK(out[j] == 0.5f * (float)(j - 16));

    /* Q4_0: d = 2, low nibbles j, high nibbles 15 - j. */
    uint8_t q4[18];
    put_u16(q4, float_to_half(2.0f));
    for (int j = 0; j < 16; j++) q4[2 + j] = (uint8_t)(j | ((15 - j) << 4));
    CHECK(h3_quant_dequantize(H3_QUANT_Q4_0, q4, 32, out));
    for (int j = 0; j < 16; j++) {
        CHECK(out[j] == 2.0f * (float)(j - 8));
        CHECK(out[j + 16] == 2.0f * (float)(15 - j - 8));
    }

    /* Q4_K: d = 1, dmin = 0.5, sub-block j scale j + 1, min j; nibbles
     * low 3 / high 5 everywhere. */
    uint8_t q4k[144];
    memset(q4k, 0, sizeof(q4k));
    put_u16(q4k, float_to_half(1.0f));
    put_u16(q4k + 2, float_to_half(0.5f));
    uint8_t *scales = q4k + 4;
    for (int j = 0; j < 8; j++) {
        uint8_t sc = (uint8_t)(j + 1), mn = (uint8_t)j;
        if (j < 4) {
            scales[j] = sc;
            scales[j + 4] = mn;
        } else {
            scales[j + 4] = (uint8_t)((sc & 0xF) | ((mn & 0xF) << 4));
            scales[j - 4] |= (uint8_t)((sc >> 4) << 6);
            scales[j] |= (uint8_t)((mn >> 4) << 6);
        }
    }
    memset(q4k + 16, 0x53, 128);
    CHECK(h3_quant_dequantize(H3_QUANT_Q4_K, q4k, 256, out));
    for (int sub = 0; sub < 8; sub++) {
        float expected = (float)(sub + 1) * ((sub % 2) ? 5.0f : 3.0f) - 0.5f * (float)sub;
        for (int l = 0; l < 32; l++) CHECK(out[sub * 32 + l] == expected);
    }

    /* Q6_K: d = 0.25, scale 2 everywhere, every 6-bit value 33 (q = 1). */
    uint8_t q6k[210];
    memset(q6k, 0, sizeof(q6k));
    memset(q6k, 0x11, 128);        /* low nibbles 1 */
    memset(q6k + 128, 0xAA, 64);   /* high bits 2 -> +32 */
    memset(q6k + 192, 2, 16);
    put_u16(q6k + 208, float_to_half(0.25f));
    CHECK(h3_quant_dequantize(H3_QUANT_Q6_K, q6k, 256, out));
    for (int j = 0; j < 256; j++) CHECK(out[j] == 0.25f * 2.0f * 1.0f);

    /* F16 and MLX affine 4-bit. */
    uint16_t halves[2] = {float_to_half(1.5f), float_to_half(-0.25f)};
    CHECK(h3_quant_dequantize(H3_QUANT_F16, halves, 2, out));
    CHECK(out[0] == 1.5f && out[1] == -0.25f);
    /* MLX affine: 4-bit words 0x76543210 hold 0..7; scale 0.5, bias -1. */
    uint32_t packed[8];
    for (int w = 0; w < 8; w++) packed[w] = 0x76543210u;
    uint16_t scale = 0x3f00, bias = 0xbf80; /* BF16 0.5 and -1 */
    CHECK(h3_mlx_dequantize_row(packed, &scale, &bias, 64, 4, 64, out));
    for (int c = 0; c < 64; c++) CHECK(out[c] == 0.5f * (float)(c % 8) - 1.0f);
    /* 8-bit: the same words hold 0x10, 0x32, 0x54, 0x76. */
    CHECK(h3_mlx_dequantize_row(packed, &scale, &bias, 32, 8, 32, out));
    for (int c = 0; c < 32; c++) {
        static const float bytes[4] = {16, 50, 84, 118};
        CHECK(out[c] == 0.5f * bytes[c % 4] - 1.0f);
    }
    CHECK(!h3_mlx_dequantize_row(packed, &scale, &bias, 64, 3, 64, out));
}

static void write_bytes(FILE *file, const void *data, size_t size) {
    CHECK(fwrite(data, 1, size, file) == size);
}

static void write_u32(FILE *file, uint32_t value) { write_bytes(file, &value, 4); }
static void write_u64(FILE *file, uint64_t value) { write_bytes(file, &value, 8); }
static void write_str(FILE *file, const char *text) {
    write_u64(file, strlen(text));
    write_bytes(file, text, strlen(text));
}

/* A two-tensor GGUF (Q8_0 [2, 64] and F32 [3]) with a string KV, a u32
 * alignment of 64, and a skipped string array, read via the store. */
static void test_gguf(h3_gpu *gpu) {
    char root[] = "/tmp/h3_gguf_XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    char path[256];
    snprintf(path, sizeof(path), "%s/model.gguf", root);
    FILE *file = fopen(path, "wb");
    CHECK(file != NULL);
    write_bytes(file, "GGUF", 4);
    write_u32(file, 3);
    write_u64(file, 2);  /* tensors */
    write_u64(file, 3);  /* key/values */
    write_str(file, "general.architecture");
    write_u32(file, 8);
    write_str(file, "minimax_h3");
    write_str(file, "general.alignment");
    write_u32(file, 4);
    write_u32(file, 64);
    write_str(file, "tokenizer.tokens");
    write_u32(file, 9);
    write_u32(file, 8);
    write_u64(file, 2);
    write_str(file, "a");
    write_str(file, "bc");
    /* tensor infos: dims fastest-first */
    write_str(file, "w");
    write_u32(file, 2);
    write_u64(file, 64);
    write_u64(file, 2);
    write_u32(file, 8);  /* Q8_0 */
    write_u64(file, 0);
    write_str(file, "b");
    write_u32(file, 1);
    write_u64(file, 3);
    write_u32(file, 0);  /* F32 */
    write_u64(file, 192);
    long position = ftell(file);
    long data_start = (position + 63) / 64 * 64;
    for (long pad = position; pad < data_start; pad++) fputc(0, file);
    uint8_t blocks[4][34];
    for (int block = 0; block < 4; block++) {
        put_u16(blocks[block], float_to_half(0.125f * (float)(block + 1)));
        for (int j = 0; j < 32; j++) blocks[block][2 + j] = (uint8_t)(int8_t)(j - block * 8);
    }
    write_bytes(file, blocks, sizeof(blocks));
    for (long pad = (long)sizeof(blocks); pad < 192; pad++) fputc(0, file);
    float bias[3] = {1.0f, -2.0f, 0.5f};
    write_bytes(file, bias, sizeof(bias));
    CHECK(fclose(file) == 0);

    h3_st_header header;
    char error[256];
    CHECK(h3_gguf_read_header(path, &header, error, sizeof(error)));
    CHECK(header.tensor_count == 2);
    const h3_st_tensor *w = h3_st_find(&header, "w");
    CHECK(w && w->ndim == 2 && w->shape[0] == 2 && w->shape[1] == 64);
    CHECK(w->quant == H3_QUANT_Q8_0 && w->dtype == H3_DTYPE_UNKNOWN);
    CHECK(w->file_offset == (uint64_t)data_start);
    CHECK(w->data_end - w->data_begin == 136);
    const h3_st_tensor *b = h3_st_find(&header, "b");
    CHECK(b && b->dtype == H3_DTYPE_F32 && b->file_offset == (uint64_t)data_start + 192);
    CHECK(h3_st_metadata(&header, "general.architecture") &&
          !strcmp(h3_st_metadata(&header, "general.architecture"), "minimax_h3"));
    h3_st_free_header(&header);

    if (gpu) {
        h3_weight_store *store = h3_weight_store_open(root, error, sizeof(error));
        CHECK(store != NULL);
        uint64_t shape[2] = {2, 64};
        h3_gpu_tensor *bf16 = h3_weight_load_bf16(store, gpu, "w", 2, shape,
                                                  error, sizeof(error));
        h3_gpu_tensor *f32 = h3_weight_load_f32(store, gpu, "w", 2, shape,
                                                error, sizeof(error));
        if (!bf16 || !f32) fprintf(stderr, "load: %s\n", error);
        CHECK(bf16 && f32);
        float values[128];
        uint16_t bits[128];
        CHECK(h3_gpu_tensor_read_f32(f32, values, 128));
        CHECK(h3_gpu_tensor_read_bf16(bf16, bits, 128));
        for (int index = 0; index < 128; index++) {
            int block = index / 32, j = index % 32;
            float expected = 0.125f * (float)(block + 1) * (float)(j - block * 8);
            CHECK(values[index] == expected);
            uint32_t widened = (uint32_t)bits[index] << 16;
            float rounded;
            memcpy(&rounded, &widened, sizeof(rounded));
            CHECK(rounded == expected); /* all exactly representable */
        }
        uint64_t bias_shape[1] = {3};
        h3_gpu_tensor *bias_tensor = h3_weight_load_f32(store, gpu, "b", 1,
                                                        bias_shape, error,
                                                        sizeof(error));
        CHECK(bias_tensor && h3_gpu_tensor_read_f32(bias_tensor, values, 3));
        CHECK(values[0] == 1.0f && values[1] == -2.0f && values[2] == 0.5f);
        h3_gpu_tensor_free(bf16);
        h3_gpu_tensor_free(f32);
        h3_gpu_tensor_free(bias_tensor);
        h3_weight_store_free(store);
    }
    CHECK(unlink(path) == 0);
    CHECK(rmdir(root) == 0);
}

/* Random blocks (their float16 scale fields replaced by sane values) of
 * every format, expanded on the GPU to F32 and BF16 and compared with the
 * reference. */
static void test_gpu_matches_reference(h3_gpu *gpu) {
    static const struct { h3_quant_type type; int halves[2]; } formats[] = {
        {H3_QUANT_F16, {-2, -1}}, {H3_QUANT_Q4_0, {0, -1}}, {H3_QUANT_Q4_1, {0, 2}},
        {H3_QUANT_Q5_0, {0, -1}}, {H3_QUANT_Q5_1, {0, 2}}, {H3_QUANT_Q8_0, {0, -1}},
        {H3_QUANT_Q2_K, {80, 82}}, {H3_QUANT_Q3_K, {108, -1}},
        {H3_QUANT_Q4_K, {0, 2}}, {H3_QUANT_Q5_K, {0, 2}}, {H3_QUANT_Q6_K, {208, -1}},
    };
    uint64_t state = 0x1234567887654321ull;
    for (size_t f = 0; f < sizeof(formats) / sizeof(*formats); f++) {
        h3_quant_type type = formats[f].type;
        uint32_t per_block = h3_quant_block_elements(type);
        uint32_t block_bytes = h3_quant_block_bytes(type);
        uint32_t blocks = type == H3_QUANT_F16 ? 4096 : 97;
        size_t bytes = (size_t)blocks * block_bytes;
        size_t count = (size_t)blocks * per_block;
        uint8_t *raw = malloc(bytes);
        float *expected = malloc(count * sizeof(float));
        float *got = malloc(count * sizeof(float));
        uint16_t *got16 = malloc(count * sizeof(uint16_t));
        CHECK(raw && expected && got && got16);
        for (size_t index = 0; index < bytes; index++) {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            raw[index] = (uint8_t)(state >> 56);
        }
        for (uint32_t block = 0; block < blocks; block++) {
            uint8_t *b = raw + (size_t)block * block_bytes;
            for (int h = 0; h < 2; h++) {
                int at = formats[f].halves[h];
                if (at == -1) continue;
                if (at == -2) at = 0;
                state = state * 6364136223846793005ull + 1442695040888963407ull;
                float value = ((float)((state >> 40) % 2001) - 1000.0f) / 4096.0f;
                put_u16(b + at, float_to_half(value));
            }
        }
        CHECK(h3_quant_dequantize(type, raw, count, expected));
        h3_gpu_tensor *raw_gpu = h3_gpu_tensor_from_i8(gpu, (const int8_t *)raw, bytes);
        h3_gpu_tensor *f32 = h3_gpu_tensor_new_f32(gpu, count);
        h3_gpu_tensor *bf16 = h3_gpu_tensor_new_bf16(gpu, count);
        CHECK(raw_gpu && f32 && bf16);
        CHECK(h3_gpu_begin(gpu));
        CHECK(h3_gpu_dequantize(gpu, f32, raw_gpu, (uint32_t)type, blocks,
                                per_block, block_bytes, 0, 0, 0));
        CHECK(h3_gpu_dequantize(gpu, bf16, raw_gpu, (uint32_t)type, blocks,
                                per_block, block_bytes, 0, 0, 0));
        CHECK(h3_gpu_submit(gpu));
        CHECK(h3_gpu_tensor_read_f32(f32, got, count));
        CHECK(h3_gpu_tensor_read_bf16(bf16, got16, count));
        double worst = 0.0;
        for (size_t index = 0; index < count; index++) {
            double scale = fabs(expected[index]) > 1.0 ? fabs(expected[index]) : 1.0;
            double diff = fabs((double)got[index] - expected[index]) / scale;
            if (diff > worst) worst = diff;
            uint32_t widened = (uint32_t)got16[index] << 16;
            float rounded;
            memcpy(&rounded, &widened, sizeof(rounded));
            CHECK(fabs((double)rounded - got[index]) <= fabs(got[index]) / 128.0 + 1e-30);
        }
        printf("quant %-5s GPU vs reference: worst rel diff %.2e over %zu values\n",
               h3_quant_name(type), worst, count);
        CHECK(worst < 1e-6);
        h3_gpu_tensor_free(raw_gpu);
        h3_gpu_tensor_free(f32);
        h3_gpu_tensor_free(bf16);
        free(raw);
        free(expected);
        free(got);
        free(got16);
    }
}

int main(void) {
    test_reference();
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) fprintf(stderr, "skip: no Metal device (%s)\n", error);
    test_gguf(gpu);
    if (gpu) {
        test_gpu_matches_reference(gpu);
        h3_gpu_free(gpu);
    }
    printf("ok: %d quant checks\n", checks);
    return 0;
}
