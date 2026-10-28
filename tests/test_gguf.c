/* GGUF header parsing on hand-built files: flat metadata arrays and
 * orig_shape restore, and arrays of arrays - accepted up to the reader's
 * nesting limit, refused past it however deep (a file nesting them without
 * end once ran the recursive reader off the stack). CPU only: no Metal
 * device or model weights needed. */
#include "h3_safetensors.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

enum { GGUF_U32 = 4, GGUF_STRING = 8, GGUF_ARRAY = 9, GGUF_U64 = 10 };

static void bytes(FILE *file, const void *data, size_t size) {
    CHECK(fwrite(data, 1, size, file) == size);
}

static void u32(FILE *file, uint32_t value) {
    uint8_t data[4];
    for (int i = 0; i < 4; i++) data[i] = (uint8_t)(value >> (8 * i));
    bytes(file, data, sizeof(data));
}

static void u64(FILE *file, uint64_t value) {
    uint8_t data[8];
    for (int i = 0; i < 8; i++) data[i] = (uint8_t)(value >> (8 * i));
    bytes(file, data, sizeof(data));
}

static void string(FILE *file, const char *value) {
    u64(file, strlen(value));
    bytes(file, value, strlen(value));
}

static void prefix(FILE *file, uint64_t tensors, uint64_t metadata) {
    bytes(file, "GGUF", 4);
    u32(file, 3);
    u64(file, tensors);
    u64(file, metadata);
}

/* One 2x2 F32 tensor "w", stored flat, with general.architecture, a string
 * array and comfy's orig_shape to restore its shape. */
static void test_flat_arrays(const char *path) {
    FILE *file = fopen(path, "wb");
    CHECK(file != NULL);
    prefix(file, 1, 3);
    string(file, "general.architecture");
    u32(file, GGUF_STRING);
    string(file, "minimax_h3");
    string(file, "tokenizer.tokens");
    u32(file, GGUF_ARRAY);
    u32(file, GGUF_STRING);
    u64(file, 2);
    string(file, "a");
    string(file, "bc");
    string(file, "comfy.gguf.orig_shape.w");
    u32(file, GGUF_ARRAY);
    u32(file, GGUF_U64);
    u64(file, 2);
    u64(file, 2);
    u64(file, 2);
    string(file, "w");
    u32(file, 1); /* stored flat */
    u64(file, 4);
    u32(file, 0); /* F32 */
    u64(file, 0);
    while (ftell(file) % 32) CHECK(fputc(0, file) != EOF);
    for (uint32_t i = 0; i < 4; i++) u32(file, i);
    CHECK(fclose(file) == 0);

    h3_st_header header;
    char error[512];
    CHECK(h3_gguf_read_header(path, &header, error, sizeof(error)));
    CHECK(header.tensor_count == 1);
    const char *architecture = h3_st_metadata(&header, "general.architecture");
    CHECK(architecture && !strcmp(architecture, "minimax_h3"));
    const h3_st_tensor *tensor = h3_st_find(&header, "w");
    CHECK(tensor && tensor->ndim == 2);
    CHECK(tensor->shape[0] == 2 && tensor->shape[1] == 2);
    CHECK(tensor->data_end - tensor->data_begin == 16);
    uint8_t data[16];
    CHECK(h3_st_read_data(&header, tensor, data, sizeof(data), error, sizeof(error)));
    for (int i = 0; i < 4; i++) {
        CHECK(data[i * 4] == (uint8_t)i);
        CHECK(data[i * 4 + 1] == 0 && data[i * 4 + 2] == 0 && data[i * 4 + 3] == 0);
    }
    h3_st_free_header(&header);
}

/* `key` holding `arrays` arrays one inside the next, the innermost holding
 * one u32 - or, with `arrays` 0, an empty array of arrays. Returns whether
 * the header was read; on failure `error` says why. */
static int read_nested(const char *path, const char *key, uint64_t arrays,
                       char *error, size_t error_size) {
    FILE *file = fopen(path, "wb");
    CHECK(file != NULL);
    prefix(file, 0, 1);
    string(file, key);
    u32(file, GGUF_ARRAY);
    if (arrays == 0) {
        u32(file, GGUF_ARRAY);
        u64(file, 0);
    } else {
        for (uint64_t i = 0; i < arrays; i++) {
            u32(file, i + 1 < arrays ? GGUF_ARRAY : GGUF_U32);
            u64(file, 1);
        }
        u32(file, 0);
    }
    CHECK(fclose(file) == 0);

    h3_st_header header;
    int ok = h3_gguf_read_header(path, &header, error, error_size);
    if (!ok) CHECK(header.path == NULL && header.tensors == NULL && header.tensor_count == 0);
    h3_st_free_header(&header);
    return ok;
}

static void test_nesting(const char *path, const char *key) {
    char error[512];
    /* As deep as the reader goes (GGUF_MAX_ARRAY_DEPTH, 8), and empty. */
    CHECK(read_nested(path, key, 8, error, sizeof(error)));
    CHECK(read_nested(path, key, 0, error, sizeof(error)));
    /* One past it, and far past it. */
    CHECK(!read_nested(path, key, 9, error, sizeof(error)));
    CHECK(strstr(error, "nested too deeply") != NULL);
    CHECK(!read_nested(path, key, 150000, error, sizeof(error)));
    CHECK(strstr(error, "nested too deeply") != NULL);
}

int main(void) {
    char path[] = "/tmp/h3_gguf_header_XXXXXX";
    int descriptor = mkstemp(path);
    CHECK(descriptor >= 0);
    CHECK(close(descriptor) == 0);
    test_flat_arrays(path);
    test_nesting(path, "metadata");
    test_nesting(path, "comfy.gguf.orig_shape.w");
    CHECK(unlink(path) == 0);
    puts("ok: GGUF headers (flat arrays, orig_shape, nesting up to the limit, refused past it)");
    return 0;
}
