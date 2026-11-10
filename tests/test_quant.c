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

/* --- ComfyUI int8 + ConvRot ------------------------------------------- */

static uint32_t lcg_state = 12345u;
static float lcg_float(void) { /* about [-1, 1) */
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (float)(lcg_state >> 8) / 8388608.0f - 1.0f;
}

/* H_group built the slow way: the Kronecker power of H4 as a dense matrix. */
static void dense_hadamard(double *h, uint32_t group) {
    static const double h4[4][4] = {{.5, .5, .5, -.5}, {.5, .5, -.5, .5},
                                    {.5, -.5, .5, .5}, {-.5, .5, .5, .5}};
    for (uint32_t row = 0; row < group; row++) {
        for (uint32_t column = 0; column < group; column++) {
            double value = 1.0;
            for (uint32_t r = row, c = column, size = group; size > 1; size /= 4) {
                value *= h4[r % 4][c % 4];
                r /= 4;
                c /= 4;
            }
            h[row * group + column] = value;
        }
    }
}

static void test_convrot_reference(void) {
    enum { GROUP = 256, COUNT = 512 };
    double *h = malloc(sizeof(double) * GROUP * GROUP);
    CHECK(h != NULL);
    dense_hadamard(h, GROUP);
    /* H is symmetric and its own inverse. */
    for (uint32_t row = 0; row < GROUP; row += 37) {
        for (uint32_t column = 0; column < GROUP; column += 29) {
            double sum = 0.0;
            for (uint32_t k = 0; k < GROUP; k++)
                sum += h[row * GROUP + k] * h[k * GROUP + column];
            CHECK(fabs(sum - (row == column ? 1.0 : 0.0)) < 1e-12);
            CHECK(h[row * GROUP + column] == h[column * GROUP + row]);
        }
    }
    float values[COUNT], rotated[COUNT];
    for (int i = 0; i < COUNT; i++) rotated[i] = values[i] = lcg_float();
    CHECK(h3_convrot_rotate(rotated, COUNT, GROUP));
    double worst = 0.0;
    for (uint32_t i = 0; i < COUNT; i++) {
        /* (w H)[i] within the group i belongs to */
        uint32_t base = i / GROUP * GROUP;
        double sum = 0.0;
        for (uint32_t k = 0; k < GROUP; k++)
            sum += (double)values[base + k] * h[k * GROUP + (i - base)];
        worst = fmax(worst, fabs(sum - rotated[i]));
    }
    CHECK(worst < 1e-5);
    CHECK(h3_convrot_rotate(rotated, COUNT, GROUP)); /* and back */
    for (int i = 0; i < COUNT; i++) CHECK(fabsf(rotated[i] - values[i]) < 1e-5f);
    CHECK(!h3_convrot_rotate(rotated, COUNT, 128)); /* not a power of 4 */
    CHECK(!h3_convrot_rotate(rotated, 300, GROUP)); /* partial group */
    float four[4] = {1.0f, 0.0f, 0.0f, 0.0f}; /* first column of H4 */
    CHECK(h3_convrot_rotate(four, 4, 4));
    CHECK(four[0] == 0.5f && four[1] == 0.5f && four[2] == 0.5f && four[3] == -0.5f);
    int8_t q[8] = {1, -2, 3, -4, 5, -6, 7, -8};
    float row[8];
    CHECK(h3_comfy_int8_dequantize_row(q, 0.5f, 8, 0, row));
    CHECK(row[0] == 0.5f && row[7] == -4.0f);
    CHECK(h3_comfy_int8_dequantize_row(q, 0.5f, 8, 4, row));
    /* (0.5, -1, 1.5, -2): t = -0.5 -> (1.5, -2, 0.5, -1) */
    CHECK(row[0] == 1.5f && row[1] == -2.0f && row[2] == 0.5f && row[3] == -1.0f);
    free(h);
}

typedef struct {
    const char *name, *dtype;
    uint64_t shape[2];
    int ndim;
    const void *data;
    size_t bytes;
} st_entry;

static void write_safetensors(const char *path, const st_entry *entries,
                              size_t count) {
    char json[4096];
    size_t used = (size_t)snprintf(json, sizeof(json),
        "{\"__metadata__\":{\"format\":\"pt\","
        "\"merge_strategy\":\"FL2VA_Ref2VA_Hybrid_test\"}");
    uint64_t offset = 0;
    for (size_t i = 0; i < count; i++) {
        char shape[64];
        if (entries[i].ndim == 2)
            snprintf(shape, sizeof(shape), "%llu,%llu",
                     (unsigned long long)entries[i].shape[0],
                     (unsigned long long)entries[i].shape[1]);
        else
            snprintf(shape, sizeof(shape), "%llu",
                     (unsigned long long)entries[i].shape[0]);
        used += (size_t)snprintf(json + used, sizeof(json) - used,
            ",\"%s\":{\"dtype\":\"%s\",\"shape\":[%s],\"data_offsets\":[%llu,%llu]}",
            entries[i].name, entries[i].dtype, shape, (unsigned long long)offset,
            (unsigned long long)(offset + entries[i].bytes));
        offset += entries[i].bytes;
        CHECK(used < sizeof(json) - 16);
    }
    json[used++] = '}';
    while (used % 8) json[used++] = ' ';
    FILE *file = fopen(path, "wb");
    CHECK(file != NULL);
    write_u64(file, used);
    write_bytes(file, json, used);
    for (size_t i = 0; i < count; i++)
        write_bytes(file, entries[i].data, entries[i].bytes);
    CHECK(fclose(file) == 0);
}

static float bf16_value(uint16_t bits) {
    uint32_t widened = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &widened, sizeof(value));
    return value;
}

/* A ComfyUI-layout checkpoint (curve AdaLN table, so fused QKV rows are
 * [Q; K; V]) with a ConvRot QKV weight, a ConvRot FC2 weight and a plain
 * int8 weight with one scale, read through the weight store: the loads
 * must return the weights that were rotated and quantized, QKV reordered
 * per head. */
static void test_comfy_int8(h3_gpu *gpu) {
    enum { QKV_ROWS = 768, QKV_COLUMNS = 256, FC_ROWS = 6, FC_COLUMNS = 512,
           PLAIN_ROWS = 3, PLAIN_COLUMNS = 8 };
    static const char convrot[] =
        "{\"format\": \"int8_tensorwise\", \"convrot\": true, \"convrot_groupsize\": 256}";
    static const char plain[] = "{\"format\": \"int8_tensorwise\"}";
    static const struct { const char *weight, *scale, *quant; uint32_t rows, columns, group; int one_scale; } layers[3] = {
        {"blocks.0.attn.qkv_proj.weight", "blocks.0.attn.qkv_proj.weight_scale",
         "blocks.0.attn.qkv_proj.comfy_quant", QKV_ROWS, QKV_COLUMNS, 256, 0},
        {"blocks.0.mlp.fc2.weight", "blocks.0.mlp.fc2.weight_scale",
         "blocks.0.mlp.fc2.comfy_quant", FC_ROWS, FC_COLUMNS, 256, 0},
        {"condition_proj.weight", "condition_proj.weight_scale",
         "condition_proj.comfy_quant", PLAIN_ROWS, PLAIN_COLUMNS, 0, 1},
    };
    float *original[3];
    int8_t *quantized[3];
    float *scales[3];
    float table[8] = {0};
    st_entry entries[11];
    size_t count = 0;
    entries[count++] = (st_entry){"adaln_t_table", "F32", {1, 8}, 2, table, sizeof(table)};
    entries[count++] = (st_entry){"video_patch_proj.weight", "F32", {1, 8}, 2, table, sizeof(table)};
    for (int l = 0; l < 3; l++) {
        uint32_t rows = layers[l].rows, columns = layers[l].columns;
        original[l] = malloc(sizeof(float) * rows * columns);
        quantized[l] = malloc((size_t)rows * columns);
        scales[l] = malloc(sizeof(float) * rows);
        float *rotated = malloc(sizeof(float) * columns);
        CHECK(original[l] && quantized[l] && scales[l] && rotated);
        float tensor_peak = 0.0f;
        for (uint32_t i = 0; i < rows * columns; i++) {
            original[l][i] = lcg_float() * (1.0f + (float)(i / columns % 5));
            tensor_peak = fmaxf(tensor_peak, fabsf(original[l][i]));
        }
        for (uint32_t row = 0; row < rows; row++) {
            memcpy(rotated, original[l] + (size_t)row * columns, sizeof(float) * columns);
            if (layers[l].group) CHECK(h3_convrot_rotate(rotated, columns, layers[l].group));
            float peak = 0.0f;
            for (uint32_t c = 0; c < columns; c++) peak = fmaxf(peak, fabsf(rotated[c]));
            float scale = (layers[l].one_scale ? tensor_peak : peak) / 127.0f;
            scales[l][row] = scale;
            for (uint32_t c = 0; c < columns; c++)
                quantized[l][(size_t)row * columns + c] = (int8_t)lrintf(rotated[c] / scale);
        }
        free(rotated);
        entries[count++] = (st_entry){layers[l].weight, "I8", {rows, columns}, 2,
                                      quantized[l], (size_t)rows * columns};
        entries[count++] = layers[l].one_scale ?
            (st_entry){layers[l].scale, "F32", {1, 0}, 1, scales[l], sizeof(float)} :
            (st_entry){layers[l].scale, "F32", {rows, 1}, 2, scales[l], sizeof(float) * rows};
        const char *json = layers[l].group ? convrot : plain;
        entries[count++] = (st_entry){layers[l].quant, "U8", {strlen(json), 0}, 1,
                                      json, strlen(json)};
    }
    char root[] = "/tmp/h3_comfy_int8_XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    char path[256];
    snprintf(path, sizeof(path), "%s/model.safetensors", root);
    write_safetensors(path, entries, count);
    char error[512] = "";
    h3_weight_store *store = h3_weight_store_open(root, error, sizeof(error));
    CHECK(store != NULL);
    const h3_st_tensor *qkv = h3_weight_find(store, layers[0].weight, NULL);
    CHECK(qkv && h3_weight_is_comfy_int8(store, qkv));
    CHECK(!h3_weight_is_comfy_int8(store, h3_weight_find(store, "adaln_t_table", NULL)));
    /* What the importer sees in the same file. */
    h3_gguf_info info;
    CHECK(h3_gguf_inspect(path, &info, error, sizeof(error)));
    CHECK(info.role == H3_GGUF_DIT && info.comfy_int8 == 2 && info.curve_adaln &&
          info.both_transformers && info.dit_blocks == 1 &&
          !strcmp(info.main_type, "INT8 ConvRot"));
    for (int l = 0; gpu && l < 3; l++) {
        uint32_t rows = layers[l].rows, columns = layers[l].columns;
        size_t elements = (size_t)rows * columns;
        uint64_t shape[2] = {rows, columns};
        h3_gpu_tensor *bf16 = h3_weight_load_bf16(store, gpu, layers[l].weight, 2,
                                                  shape, error, sizeof(error));
        h3_gpu_tensor *f32 = h3_weight_load_f32(store, gpu, layers[l].weight, 2,
                                                shape, error, sizeof(error));
        if (!bf16 || !f32) fprintf(stderr, "%s\n", error);
        CHECK(bf16 && f32);
        uint16_t *got16 = malloc(elements * sizeof(*got16));
        float *got = malloc(elements * sizeof(*got));
        CHECK(got16 && got && h3_gpu_tensor_read_bf16(bf16, got16, elements) &&
              h3_gpu_tensor_read_f32(f32, got, elements));
        double worst = 0.0, peak = 0.0;
        for (uint32_t row = 0; row < rows; row++) {
            uint32_t destination = row;
            if (l == 0) { /* [Q; K; V] with 128 rows a head -> per-head [q; k; v] */
                uint32_t inner = rows / 3, part = row / inner, within = row % inner;
                destination = within / 128 * 384 + part * 128 + within % 128;
            }
            /* quantization error of one row is at most its scale / 2 a value
             * before the rotation; the rotation keeps the row's L2 norm */
            double square = 0.0;
            for (uint32_t c = 0; c < columns; c++) {
                float want = original[l][(size_t)row * columns + c];
                float have = got[(size_t)destination * columns + c];
                square += ((double)want - have) * ((double)want - have);
                peak = fmax(peak, fabs((double)want));
                float rounded = bf16_value(got16[(size_t)destination * columns + c]);
                CHECK(fabsf(rounded - have) <= fabsf(have) / 128.0f + 1e-30f);
            }
            double bound = 0.5 * scales[l][row] * sqrt((double)columns);
            CHECK(sqrt(square) <= bound * 1.001 + 1e-6);
            worst = fmax(worst, sqrt(square / columns));
        }
        /* The SSD-streaming route - the int8 payload as it is in the file,
         * restored on the GPU - gives the same bits as the load. */
        const h3_st_tensor *tensor = h3_weight_find(store, layers[l].weight, NULL);
        h3_gpu_tensor *scale_tensor = NULL;
        uint32_t group = 99;
        CHECK(h3_weight_comfy_int8_scales(store, gpu, tensor, 0, rows, &group,
                                          &scale_tensor, error, sizeof(error)));
        CHECK(group == layers[l].group);
        h3_gpu_tensor *raw = h3_gpu_tensor_from_i8(gpu, quantized[l], elements);
        h3_gpu_tensor *expanded = h3_gpu_tensor_new_bf16(gpu, elements);
        CHECK(raw && expanded);
        CHECK(h3_gpu_begin(gpu));
        CHECK(h3_gpu_comfy_int8_expand(gpu, expanded, raw, scale_tensor, rows, columns,
                                       group, l == 0 ? rows / 3 : 0, l == 0 ? 128 : 0));
        CHECK(h3_gpu_submit(gpu));
        uint16_t *streamed = malloc(elements * sizeof(*streamed));
        CHECK(streamed && h3_gpu_tensor_read_bf16(expanded, streamed, elements));
        CHECK(memcmp(streamed, got16, elements * sizeof(*streamed)) == 0);
        /* A run of rows read on its own (SSD streaming reads the SwiGLU
         * halves of an FC1 weight apart) equals those rows of the whole. */
        if (l != 0) {
            uint32_t begin = rows / 2, part_rows = rows - begin;
            size_t part = (size_t)part_rows * columns;
            h3_gpu_tensor *part_scales = NULL;
            uint32_t part_group = 99;
            CHECK(h3_weight_comfy_int8_scales(store, gpu, tensor, begin, part_rows, &part_group,
                                              &part_scales, error, sizeof(error)));
            CHECK(part_group == group);
            CHECK(!h3_weight_comfy_int8_scales(store, gpu, tensor, begin, part_rows + 1,
                                               &part_group, &part_scales, error, sizeof(error)));
            h3_gpu_tensor *part_raw = h3_gpu_tensor_from_i8(
                gpu, quantized[l] + (size_t)begin * columns, part);
            h3_gpu_tensor *part_out = h3_gpu_tensor_new_bf16(gpu, part);
            CHECK(part_raw && part_out && h3_gpu_begin(gpu));
            CHECK(h3_gpu_comfy_int8_expand(gpu, part_out, part_raw, part_scales, part_rows,
                                           columns, group, 0, 0));
            CHECK(h3_gpu_submit(gpu));
            CHECK(h3_gpu_tensor_read_bf16(part_out, streamed, part));
            CHECK(memcmp(streamed, got16 + (size_t)begin * columns,
                         part * sizeof(*streamed)) == 0);
            h3_gpu_tensor_free(part_raw);
            h3_gpu_tensor_free(part_out);
            h3_gpu_tensor_free(part_scales);
        }
        free(streamed);
        h3_gpu_tensor_free(raw);
        h3_gpu_tensor_free(expanded);
        h3_gpu_tensor_free(scale_tensor);
        printf("comfy int8 %-30s rms error %.3e (peak value %.2f)\n",
               layers[l].weight, worst, peak);
        CHECK(worst < peak / 100.0);
        free(got16);
        free(got);
        h3_gpu_tensor_free(bf16);
        h3_gpu_tensor_free(f32);
    }
    h3_weight_store_free(store);
    for (int l = 0; l < 3; l++) {
        free(original[l]);
        free(quantized[l]);
        free(scales[l]);
    }
    CHECK(unlink(path) == 0);
    CHECK(rmdir(root) == 0);
}

int main(void) {
    test_reference();
    test_convrot_reference();
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) fprintf(stderr, "skip: no Metal device (%s)\n", error);
    test_gguf(gpu);
    test_comfy_int8(gpu);
    if (gpu) {
        test_gpu_matches_reference(gpu);
        h3_gpu_free(gpu);
    }
    printf("ok: %d quant checks\n", checks);
    return 0;
}
