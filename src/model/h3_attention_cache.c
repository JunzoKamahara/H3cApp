/* h3_build_attention_cache()'s implementation - the reusable core also
 * used by the build_attention_cache CLI tool (h3_build_attention_cache.c,
 * now a thin argument-parsing wrapper around this) and, via h3.h, by the
 * native app so it can build a missing cache itself instead of just
 * telling the user it's missing. See h3.h for the public contract.
 *
 * Cache layout: a 64-byte header, then per block (in block order):
 *   qkv_int8[INNER*3*HIDDEN]  qkv_scales[INNER*3]   (f32)
 *   out_int8[HIDDEN*INNER]    out_scales[HIDDEN]    (f32)
 *   fc1_int8[FFN*2*HIDDEN]    fc1_scales[FFN*2]     (f32)
 *   fc2_int8[HIDDEN*FFN]      fc2_scales[HIDDEN]    (f32)
 * Quantization uses the exact same GPU routine (h3_gpu_quantize_weight_int8)
 * as the existing resident-int8 path, so results match it bit for bit; they
 * are not expected to match a BF16-only run.
 *
 * The v3 header adds model_kind (FL2VA=1/Ref2VA=2, matching h3_dit.c's
 * h3_cache_model_kind) and model_id (h3_weight_store_fingerprint() of the
 * transformer directory quantized: shard names, sizes and mtimes, not
 * their absolute paths, so moving the model folder keeps the cache valid)
 * so h3_dit.c can refuse a cache built for the wrong model rather than
 * silently streaming mismatched weights -
 * a v2 cache (no such tagging) simply fails h3_dit.c's version check and
 * must be rebuilt. */
#include "h3.h"
#include "h3_gpu.h"
#include "h3_weights.h"

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum {
    HIDDEN = 5376,
    HEADS = 56,
    HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM,
    FFN = 14336,
    DIT_BLOCKS = 50,
};

#define CACHE_MAGIC "H3AC"
#define CACHE_VERSION 3u

typedef enum {
    MODEL_UNKNOWN = 0,
    MODEL_FL2VA = 1,
    MODEL_REF2VA = 2,
} model_kind;

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t block_count;
    uint32_t hidden;
    uint32_t inner;
    uint32_t ffn;
    uint32_t model_kind;
    uint8_t model_id[32];
    uint32_t reserved[1];
} cache_header;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

/* Mirrors h3_dit.c's detect_model_kind(): reads back the layout h3.c's
 * own dit_path selection always produces, rather than guessing at one. */
static model_kind detect_model_kind(const char *transformer_dir) {
    size_t length = strlen(transformer_dir);
    static const char ref2va_suffix[] = "Ref2VA/transformer";
    static const char fl2va_suffix[] = "FL2VA/transformer";
    if (length >= sizeof(ref2va_suffix) - 1 &&
        !strcmp(transformer_dir + length - (sizeof(ref2va_suffix) - 1),
                ref2va_suffix))
        return MODEL_REF2VA;
    if (length >= sizeof(fl2va_suffix) - 1 &&
        !strcmp(transformer_dir + length - (sizeof(fl2va_suffix) - 1),
                fl2va_suffix))
        return MODEL_FL2VA;
    return MODEL_UNKNOWN;
}

static int quantize_and_write(h3_gpu *gpu, h3_weight_store *store,
                              const char *name, uint32_t rows,
                              uint32_t columns, FILE *out,
                              char *error, size_t error_size) {
    uint64_t shape[2] = { rows, columns };
    h3_gpu_tensor *bf16 = h3_weight_load_bf16(store, gpu, name, 2, shape,
                                              error, error_size);
    if (!bf16) return 0;

    size_t elements = (size_t)rows * columns;
    h3_gpu_tensor *i8 = h3_gpu_tensor_new_i8(gpu, elements);
    h3_gpu_tensor *scales = h3_gpu_tensor_new_f32(gpu, rows);
    int ok = i8 && scales &&
             h3_gpu_begin(gpu) &&
             h3_gpu_quantize_weight_int8(gpu, i8, scales, bf16, rows,
                                         columns) &&
             h3_gpu_submit(gpu);
    h3_gpu_tensor_free(bf16);
    if (!ok) {
        if (error && error_size && !error[0])
            snprintf(error, error_size, "cannot quantize %s: %s", name,
                     h3_gpu_error(gpu));
        h3_gpu_tensor_free(i8);
        h3_gpu_tensor_free(scales);
        return 0;
    }

    int8_t *i8_host = malloc(elements);
    float *scale_host = malloc((size_t)rows * sizeof(float));
    int result = i8_host && scale_host &&
        h3_gpu_tensor_read_i8(i8, i8_host, elements) &&
        h3_gpu_tensor_read_f32(scales, scale_host, rows) &&
        fwrite(i8_host, 1, elements, out) == elements &&
        fwrite(scale_host, sizeof(float), rows, out) == rows;
    free(i8_host);
    free(scale_host);
    h3_gpu_tensor_free(i8);
    h3_gpu_tensor_free(scales);
    if (!result && error && error_size)
        snprintf(error, error_size, "cannot write cache payload for %s: %s",
                 name, strerror(errno));
    return result;
}

int h3_build_attention_cache(const char *transformer_dir,
                             const char *output_path,
                             const char *shader_source_path,
                             h3_progress_callback on_progress,
                             void *callback_opaque,
                             char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!transformer_dir || !*transformer_dir || !output_path ||
        !*output_path) {
        fail(error, error_size, "invalid attention cache build arguments");
        return 0;
    }

    h3_gpu *gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!gpu) return 0;
    if (!h3_gpu_has_int8_mlp(gpu)) {
        fail(error, error_size,
             "this GPU lacks the int8 tensor-op path the cache is built for "
             "(needs an M5-class GPU)");
        h3_gpu_free(gpu);
        return 0;
    }

    h3_weight_store *store = h3_weight_store_open(transformer_dir, error,
                                                   error_size);
    if (!store) {
        h3_gpu_free(gpu);
        return 0;
    }

    FILE *out = fopen(output_path, "wb");
    if (!out) {
        fail(error, error_size, "cannot open %s: %s", output_path,
             strerror(errno));
        h3_weight_store_free(store);
        h3_gpu_free(gpu);
        return 0;
    }

    cache_header header = {0};
    memcpy(header.magic, CACHE_MAGIC, 4);
    header.version = CACHE_VERSION;
    header.block_count = DIT_BLOCKS;
    header.hidden = HIDDEN;
    header.inner = INNER;
    header.ffn = FFN;
    header.model_kind = (uint32_t)detect_model_kind(transformer_dir);
    h3_weight_store_fingerprint(store, header.model_id);

    int ok = fwrite(&header, sizeof(header), 1, out) == 1;
    if (!ok) fail(error, error_size, "cannot write cache header: %s",
                  strerror(errno));

    for (uint32_t block = 0; ok && block < DIT_BLOCKS; block++) {
        char name[160];
        struct { const char *suffix; uint32_t rows, columns; } projections[] = {
            {"attn.qkv_proj.weight", INNER * 3, HIDDEN},
            {"attn.out_proj.weight", HIDDEN, INNER},
            {"mlp.fc1.weight", FFN * 2, HIDDEN},
            {"mlp.fc2.weight", HIDDEN, FFN},
        };
        for (size_t p = 0; ok && p < sizeof(projections) / sizeof(*projections); p++) {
            if (error) error[0] = '\0';
            snprintf(name, sizeof(name), "blocks.%u.%s", block,
                    projections[p].suffix);
            ok = quantize_and_write(gpu, store, name, projections[p].rows,
                                    projections[p].columns, out, error,
                                    error_size);
        }
        if (ok && on_progress &&
            on_progress("attention cache", (int)block + 1, DIT_BLOCKS,
                       callback_opaque)) {
            fail(error, error_size, "attention cache build cancelled");
            ok = 0;
        }
    }

    fclose(out);
    h3_weight_store_free(store);
    h3_gpu_free(gpu);
    if (!ok) {
        unlink(output_path); /* don't leave a truncated cache file behind */
        return 0;
    }
    return 1;
}
