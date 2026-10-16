#ifndef H3_WEIGHTS_H
#define H3_WEIGHTS_H

#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_weight_store h3_weight_store;

/* Open every safetensors header in a component directory without reading
 * tensor payloads. */
h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size);
void h3_weight_store_free(h3_weight_store *store);
size_t h3_weight_store_shards(const h3_weight_store *store);

/* A lightweight, non-cryptographic fingerprint of this store's shards
 * (each shard's file name - not its directory - file size, and mtime,
 * hashed together) - cheap enough to recompute at every run (a stat() per
 * shard, no payload reads), unlike hashing the ~18GB of actual weight
 * bytes. Moving the model folder keeps it (mv preserves mtime); a copy
 * that does not preserve mtime changes it. Meant for staleness detection
 * (a cache built against different/rewritten weights), not as a security
 * property. Always fills all 32 bytes of `out`: the hash in bytes 0..7, a
 * format tag in 8..15, zeros beyond. */
void h3_weight_store_fingerprint(const h3_weight_store *store, uint8_t out[32]);

/* The fingerprint attention caches carried before 2026-10-05: the same
 * hash but over each shard's full path instead of its file name (bytes
 * 8..31 zero), so it only matches while the model stays where the cache
 * was built. Kept so those caches still validate without a rebuild. */
void h3_weight_store_fingerprint_legacy(const h3_weight_store *store,
                                        uint8_t out[32]);

/* Rows per attention head in the H3 DiT and its token refiner. */
#define H3_QKV_HEAD_DIM 128u

/* 1 when `name` is a fused QKV weight this store keeps in ComfyUI's
 * [all Q; all K; all V] row order, which loads reorder to the released
 * per-head [q; k; v] (h3_gpu_dequantize does it on the way); sets the Q
 * row count and rows per head. */
int h3_weight_qkv_reorder(const h3_weight_store *store, const char *name,
                          const h3_st_tensor *tensor, uint32_t *inner,
                          uint32_t *head_dim);

/* Every tensor in the store, shard by shard (for tools that walk them). */
size_t h3_weight_store_tensor_count(const h3_weight_store *store);
const h3_st_tensor *h3_weight_store_tensor(const h3_weight_store *store,
                                           size_t index,
                                           const h3_st_header **header);

/* Looks a tensor up by the released checkpoints' name, also under the
 * spellings quantized re-publications use (llama.cpp's blk.N.attn_q,
 * mlx-vlm's language_model.model.*, transformers' model.layers.*). */
const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header);

/* Validate an exact BF16 shape, allocate a shared Metal buffer, and read the
 * payload directly into that buffer with no intermediate host allocation. */
h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size);
h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size);

#endif
