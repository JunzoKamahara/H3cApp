/* Engine-internal declarations of the one-step VSR: the Wan configuration,
 * the .h3vc weight cache and its constant precomputation (h3_vsr_weights.c).
 * Not part of the app-facing API in h3_vsr.h; tests include it. */
#ifndef H3_VSR_INTERNAL_H
#define H3_VSR_INTERNAL_H

#include "h3_vsr.h"
#include "h3_gpu.h"

#include <stdint.h>
#include <stdio.h>

/* transformer/config.json of a diffusers WanTransformer3DModel. */
typedef struct {
    int layers;          /* num_layers */
    int heads;           /* num_attention_heads */
    int head_dim;        /* attention_head_dim */
    int dim;             /* heads * head_dim */
    int ffn;             /* ffn_dim */
    int in_channels;     /* 48 (SwiftVR), 16 (FlashVSR) */
    int out_channels;
    int text_dim;        /* 4096 */
    int freq_dim;        /* 256 */
    int patch_t, patch_h, patch_w;
    int text_tokens;     /* from prompt_embedding.safetensors (512) */
    int cross_attn_norm; /* norm2 affine */
} h3_vsr_wan_config;

int h3_vsr_read_wan_config(const char *model_dir, h3_vsr_wan_config *config,
                           char *error, size_t error_size);

/* ---------------------------------------------------------------- */
/* .h3vc cache file: 128-byte header, payload (each 64-byte aligned),
 * then a name table. Both models share the format; names are listed in
 * h3_vsr_weights.c. */

#define H3_VSR_CACHE_MAGIC "H3VC"
#define H3_VSR_CACHE_VERSION 1u
#define H3_VSR_CACHE_NAME 96
#define H3_VSR_CACHE_ALIGN 64u

typedef enum {
    H3_VSR_CACHE_F32 = 1,
    H3_VSR_CACHE_BF16 = 2,
    H3_VSR_CACHE_I8 = 3
} h3_vsr_cache_dtype;

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t model_kind;      /* h3_vsr_kind */
    uint32_t precision;       /* h3_vsr_precision (BF16 or INT8) */
    uint32_t blocks, dim, ffn, heads, head_dim, text_tokens;
    uint32_t patch_in;        /* in_channels * patch_h * patch_w (192) */
    uint32_t latent_channels;
    uint8_t fingerprint[32];      /* h3_weight_store_fingerprint(transformer/) */
    uint8_t reae_fingerprint[32]; /* same hash over reae + prompt_embedding */
    uint64_t table_offset;
    uint32_t table_count;
    uint32_t reserved[1];
} h3_vsr_cache_header;             /* 128 bytes */

typedef struct {
    char name[H3_VSR_CACHE_NAME];
    uint32_t dtype;           /* h3_vsr_cache_dtype */
    uint32_t rows, cols;      /* cols 1 for vectors */
    uint32_t reserved;
    uint64_t offset;          /* absolute file offset, 64-byte aligned */
    uint64_t bytes;
} h3_vsr_cache_entry;             /* 128 bytes */

typedef struct {
    char *path;
    h3_vsr_cache_header header;
    h3_vsr_cache_entry *entries;
    size_t count;
} h3_vsr_cache;

int h3_vsr_cache_open(const char *path, h3_vsr_cache *cache,
                      char *error, size_t error_size);
void h3_vsr_cache_close(h3_vsr_cache *cache);
const h3_vsr_cache_entry *h3_vsr_cache_find(const h3_vsr_cache *cache,
                                            const char *name);
/* Allocates a GPU tensor of the entry's dtype and preads its payload
 * (F_NOCACHE). */
h3_gpu_tensor *h3_vsr_cache_load(const h3_vsr_cache *cache, h3_gpu *gpu,
                                 const char *name, char *error,
                                 size_t error_size);
/* Reads a small F32 entry into host memory (constants, biases). */
int h3_vsr_cache_read_f32(const h3_vsr_cache *cache, const char *name,
                          float *values, size_t elements,
                          char *error, size_t error_size);

/* The fingerprint of reae.safetensors + prompt_embedding.safetensors in a
 * model directory (same hash as the weight store's). */
int h3_vsr_reae_fingerprint(const char *model_dir, uint8_t out[32]);

/* ---------------------------------------------------------------- */
/* Constant conditioning of the one-step model (CPU, double/F32), exposed
 * for the builder and its test. All outputs are row-major F32. */

/* diffusers Timesteps(freq_dim, flip_sin_to_cos=True, shift=0) at t:
 * [cos(t * f_i) ..., sin(t * f_i) ...], f_i = 10000^(-i / (freq_dim/2)). */
void h3_vsr_sinusoidal_embedding(double t, int freq_dim, float *out);
/* y = silu(x) in place. */
void h3_vsr_silu_f32(float *x, size_t n);
/* y = gelu_tanh(x) in place. */
void h3_vsr_gelu_tanh_f32(float *x, size_t n);
/* out[rows, out_dim] = in[rows, in_dim] * W[out_dim, in_dim]^T + b (Accelerate). */
void h3_vsr_linear_f32(const float *in, int rows, int in_dim,
                       const float *weight, const float *bias, int out_dim,
                       float *out);
/* x[rows, dim] *= rsqrt(mean(x^2) + eps) * weight, per row. */
void h3_vsr_rms_norm_rows_f32(float *x, int rows, int dim,
                              const float *weight, float eps);

#endif
