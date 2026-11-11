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

/* ---------------------------------------------------------------- */
/* ReAE autoencoder (h3_vsr_reae.c): SwiftVR's TAEHV-family encoder and
 * decoder on the F32 channels-last conv3d path. Frames and latents are
 * channels-last [T, H, W, C] F32 GPU tensors; RGB in [0, 1]. The encoder
 * takes 4k frames (the caller adds the last chunk's 3 tail copies) with H
 * and W multiples of 32 and gives [T/4, H/16, W/16, 48]; the decoder gives
 * [4L, 16h, 16w, 3] (the caller drops the first chunk's first 3 frames).
 * A state carries every MemBlock's previous-frame input across chunks and
 * across the frame batches the runner uses to bound memory. */
typedef struct h3_vsr_reae h3_vsr_reae;
typedef struct h3_vsr_reae_state h3_vsr_reae_state;

h3_vsr_reae *h3_vsr_reae_load(const char *model_dir, h3_gpu *gpu,
                              char *error, size_t error_size);
void h3_vsr_reae_free(h3_vsr_reae *reae);

/* encoder != 0 for the encoder's state. States are tied to a frame size. */
h3_vsr_reae_state *h3_vsr_reae_state_new(const h3_vsr_reae *reae, int encoder,
                                         int height, int width);
void h3_vsr_reae_state_free(h3_vsr_reae_state *state);

/* frame_batch: input frames per pass (0 = whole chunk); encoder batches
 * must be multiples of 4, decoder batches count latent steps. */
int h3_vsr_reae_encode(h3_vsr_reae *reae, h3_vsr_reae_state *state,
                       const h3_gpu_tensor *frames, int frame_count,
                       int height, int width, int frame_batch,
                       h3_gpu_tensor *latent, char *error, size_t error_size);
int h3_vsr_reae_decode(h3_vsr_reae *reae, h3_vsr_reae_state *state,
                       const h3_gpu_tensor *latent, int latent_count,
                       int latent_height, int latent_width, int latent_batch,
                       h3_gpu_tensor *frames, char *error, size_t error_size);
/* The same decode committed without waiting: the context keeps running
 * it (e.g. under the next chunk's DiT on another context) until
 * h3_vsr_reae_finish waits for it. Nothing else may use that reae
 * in between. */
int h3_vsr_reae_decode_begin(h3_vsr_reae *reae, h3_vsr_reae_state *state,
                             const h3_gpu_tensor *latent, int latent_count,
                             int latent_height, int latent_width, int latent_batch,
                             h3_gpu_tensor *frames, char *error, size_t error_size);
int h3_vsr_reae_finish(h3_vsr_reae *reae, char *error, size_t error_size);

/* ---------------------------------------------------------------- */
/* Wan DiT runner (h3_vsr_wan.c): the 30 blocks of the one-step model from
 * an .h3vc cache, on BF16 activations with F32 constants. predict() turns
 * the F32 channels-last LQ latent [T, h, w, C] into z_hq = z_lq - v. The
 * window layouts select mask-free shifted-window attention (even and odd
 * blocks); both NULL means full attention. rope_cos/sin are the F32
 * [tokens, head_dim/2] tables of h3_vsr_rope_table on the GPU. */
typedef struct h3_vsr_wan h3_vsr_wan;

h3_vsr_wan *h3_vsr_wan_load(h3_gpu *gpu, const char *cache_path,
                            char *error, size_t error_size);
void h3_vsr_wan_free(h3_vsr_wan *wan);
int h3_vsr_wan_is_int8(const h3_vsr_wan *wan);
/* `on_block`, when given, is called as ("vsr dit", b, layers) before each
 * block and once after the last; a non-zero return stops with the error
 * "vsr cancelled". */
int h3_vsr_wan_predict(h3_vsr_wan *wan, const h3_gpu_tensor *z_lq,
                       int frames, int height, int width,
                       const h3_vsr_window_layout *even,
                       const h3_vsr_window_layout *odd,
                       const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
                       h3_gpu_tensor *z_hq,
                       h3_progress_callback on_block, void *opaque,
                       char *error, size_t error_size);

/* The DiT/ReAE context's raw counters (tests/bench_vsr.c). */
int h3_vsr_gpu_stats(const h3_vsr *vsr, h3_gpu_stats *stats);

#endif /* H3_VSR_INTERNAL_H */
