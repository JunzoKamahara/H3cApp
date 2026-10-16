#include "h3_weights.h"

#include "h3_quant.h"

#include <dirent.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

struct h3_weight_store {
    h3_st_header *headers;
    size_t count;
    /* Fused QKV weights stored as [all Q; all K; all V] (ComfyUI's layout,
     * and the quantized files made from its checkpoints) rather than the
     * released per-head [q; k; v]; loads reorder them. */
    int qkv_split;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static const h3_st_tensor *find_exact(const h3_weight_store *store,
                                      const char *name,
                                      const h3_st_header **header);

static int has_suffix(const char *name, const char *suffix) {
    size_t length = strlen(name), suffix_length = strlen(suffix);
    return length > suffix_length &&
           strcmp(name + length - suffix_length, suffix) == 0;
}

/* A component directory may hold safetensors shards and/or GGUF files
 * (h3_gguf.c); both read into the same tensor table. */
static int safetensors_name(const char *name) {
    return has_suffix(name, ".safetensors") || has_suffix(name, ".gguf");
}

static int compare_paths(const void *left, const void *right) {
    const char *const *a = left;
    const char *const *b = right;
    return strcmp(*a, *b);
}

static void free_paths(char **paths, size_t count) {
    if (!paths) return;
    for (size_t index = 0; index < count; index++) free(paths[index]);
    free(paths);
}

h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size) {
    if (!directory || !*directory) {
        fail(error, error_size, "weight directory is required");
        return NULL;
    }
    DIR *stream = opendir(directory);
    if (!stream) {
        fail(error, error_size, "cannot open weight directory: %s", directory);
        return NULL;
    }
    char **paths = NULL;
    size_t count = 0;
    size_t capacity = 0;
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        if (!safetensors_name(entry->d_name)) continue;
        if (count == capacity) {
            size_t next = capacity ? capacity * 2 : 8;
            char **grown = realloc(paths, next * sizeof(*grown));
            if (!grown) {
                fail(error, error_size, "out of memory listing weight shards");
                closedir(stream);
                free_paths(paths, count);
                return NULL;
            }
            paths = grown;
            capacity = next;
        }
        size_t length = strlen(directory) + strlen(entry->d_name) + 2;
        paths[count] = malloc(length);
        if (!paths[count]) {
            fail(error, error_size, "out of memory resolving a weight shard");
            closedir(stream);
            free_paths(paths, count);
            return NULL;
        }
        snprintf(paths[count], length, "%s/%s", directory, entry->d_name);
        count++;
    }
    closedir(stream);
    if (!count) {
        fail(error, error_size, "no safetensors or GGUF files in %s", directory);
        free(paths);
        return NULL;
    }
    qsort(paths, count, sizeof(*paths), compare_paths);
    h3_weight_store *store = calloc(1, sizeof(*store));
    if (!store) {
        fail(error, error_size, "out of memory creating weight store");
        free_paths(paths, count);
        return NULL;
    }
    store->headers = calloc(count, sizeof(*store->headers));
    if (!store->headers) {
        fail(error, error_size, "out of memory allocating weight headers");
        free(store);
        free_paths(paths, count);
        return NULL;
    }
    store->count = count;
    for (size_t index = 0; index < count; index++) {
        char detail[384];
        int read = has_suffix(paths[index], ".gguf") ?
            h3_gguf_read_header(paths[index], &store->headers[index], detail,
                                sizeof(detail)) :
            h3_st_read_header(paths[index], &store->headers[index], detail,
                              sizeof(detail));
        if (!read) {
            fail(error, error_size, "%s", detail);
            free_paths(paths, count);
            h3_weight_store_free(store);
            return NULL;
        }
    }
    free_paths(paths, count);
    /* ComfyUI-derived checkpoints are recognizable by its curve-form AdaLN
     * (adaln_t_table, no time embedder); H3_QKV_LAYOUT=split|interleaved
     * overrides the guess. */
    store->qkv_split = find_exact(store, "adaln_t_table", NULL) &&
                       !find_exact(store, "time_embedder.proj_in.weight", NULL);
    const char *layout = getenv("H3_QKV_LAYOUT");
    if (layout && !strcmp(layout, "split")) store->qkv_split = 1;
    if (layout && !strcmp(layout, "interleaved")) store->qkv_split = 0;
    return store;
}

int h3_weight_qkv_reorder(const h3_weight_store *store, const char *name,
                          const h3_st_tensor *tensor, uint32_t *inner,
                          uint32_t *head_dim) {
    static const char suffix[] = "attn.qkv_proj.weight";
    if (!store || !store->qkv_split || !name || !tensor ||
        !has_suffix(name, suffix) || tensor->ndim != 2 ||
        tensor->shape[0] % (3 * H3_QKV_HEAD_DIM) || tensor->shape[0] > UINT32_MAX)
        return 0;
    if (inner) *inner = (uint32_t)(tensor->shape[0] / 3);
    if (head_dim) *head_dim = H3_QKV_HEAD_DIM;
    return 1;
}

void h3_weight_store_free(h3_weight_store *store) {
    if (!store) return;
    for (size_t index = 0; index < store->count; index++) {
        h3_st_free_header(&store->headers[index]);
    }
    free(store->headers);
    free(store);
}

size_t h3_weight_store_shards(const h3_weight_store *store) {
    return store ? store->count : 0;
}

static uint64_t fingerprint_fnv1a64(uint64_t hash, const void *data,
                                    size_t bytes) {
    const unsigned char *p = data;
    for (size_t i = 0; i < bytes; i++) {
        hash ^= p[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static int64_t fingerprint_mtime(const char *path) {
    struct stat status;
    if (path && stat(path, &status) == 0) return (int64_t)status.st_mtime;
    return 0;
}

/* Marks bytes 8..15 of a path-independent fingerprint, so it can never
 * equal a legacy one (whose bytes 8..31 are all zero). */
static const char fingerprint_v2_tag[8] = "h3fp-v2";

void h3_weight_store_fingerprint(const h3_weight_store *store, uint8_t out[32]) {
    memset(out, 0, 32);
    if (!store) return;
    uint64_t hash = 1469598103934665603ull;
    for (size_t index = 0; index < store->count; index++) {
        const char *path = store->headers[index].path;
        const char *slash = path ? strrchr(path, '/') : NULL;
        const char *name = slash ? slash + 1 : (path ? path : "");
        uint64_t size = store->headers[index].file_size;
        int64_t mtime = fingerprint_mtime(path);
        /* The terminating NUL separates one name from the next field. */
        hash = fingerprint_fnv1a64(hash, name, strlen(name) + 1);
        hash = fingerprint_fnv1a64(hash, &size, sizeof(size));
        hash = fingerprint_fnv1a64(hash, &mtime, sizeof(mtime));
    }
    memcpy(out, &hash, sizeof(hash));
    memcpy(out + 8, fingerprint_v2_tag, sizeof(fingerprint_v2_tag));
}

void h3_weight_store_fingerprint_legacy(const h3_weight_store *store,
                                        uint8_t out[32]) {
    memset(out, 0, 32);
    if (!store) return;
    uint64_t hash = 1469598103934665603ull;
    for (size_t index = 0; index < store->count; index++) {
        const char *path = store->headers[index].path;
        uint64_t size = store->headers[index].file_size;
        int64_t mtime = fingerprint_mtime(path);
        hash = fingerprint_fnv1a64(hash, path, path ? strlen(path) : 0);
        hash = fingerprint_fnv1a64(hash, &size, sizeof(size));
        hash = fingerprint_fnv1a64(hash, &mtime, sizeof(mtime));
    }
    memcpy(out, &hash, sizeof(hash));
}

static const h3_st_tensor *find_exact(const h3_weight_store *store,
                                      const char *name,
                                      const h3_st_header **header) {
    for (size_t index = 0; index < store->count; index++) {
        const h3_st_tensor *tensor = h3_st_find(&store->headers[index], name);
        if (tensor) {
            if (header) *header = &store->headers[index];
            return tensor;
        }
    }
    return NULL;
}

/* The engine asks for the released checkpoints' names; quantized
 * re-publications of the same weights name them differently. Text encoder
 * (Qwen3-VL) per-layer tensors as llama.cpp writes them. */
static const struct { const char *hf, *gguf; } llama_layer_names[] = {
    {"input_layernorm.weight", "attn_norm.weight"},
    {"self_attn.q_proj.weight", "attn_q.weight"},
    {"self_attn.k_proj.weight", "attn_k.weight"},
    {"self_attn.v_proj.weight", "attn_v.weight"},
    {"self_attn.o_proj.weight", "attn_output.weight"},
    {"self_attn.q_norm.weight", "attn_q_norm.weight"},
    {"self_attn.k_norm.weight", "attn_k_norm.weight"},
    {"post_attention_layernorm.weight", "ffn_norm.weight"},
    {"mlp.gate_proj.weight", "ffn_gate.weight"},
    {"mlp.up_proj.weight", "ffn_up.weight"},
    {"mlp.down_proj.weight", "ffn_down.weight"},
};

/* Qwen3-VL vision tensors as llama.cpp's mmproj files name them; `rest` is
 * the name after "model.visual.". Deepstack mergers are numbered by the
 * vision layer they tap (8, 16, 24) instead of 0..2. The temporal patch
 * embedding is split into two tensors there (load_tensor reassembles it),
 * so it has no single alias. */
static int clip_name(const char *rest, char *out, size_t size) {
    static const struct { const char *hf, *clip; } block_names[] = {
        {"norm1.", "ln1."}, {"norm2.", "ln2."}, {"attn.qkv.", "attn_qkv."},
        {"attn.proj.", "attn_out."}, {"mlp.linear_fc1.", "ffn_up."},
        {"mlp.linear_fc2.", "ffn_down."},
    };
    static const struct { const char *hf, *clip; } merger_names[] = {
        {"norm.", "norm."}, {"linear_fc1.", "fc1."}, {"linear_fc2.", "fc2."},
    };
    static const int deepstack_layers[] = {8, 16, 24};
    int index = -1, used = 0;
    if (sscanf(rest, "blocks.%d.%n", &index, &used) == 1 && used > 0) {
        for (size_t i = 0; i < sizeof(block_names) / sizeof(*block_names); i++) {
            size_t length = strlen(block_names[i].hf);
            if (!strncmp(rest + used, block_names[i].hf, length)) {
                snprintf(out, size, "v.blk.%d.%s%s", index, block_names[i].clip,
                         rest + used + length);
                return 1;
            }
        }
        return 0;
    }
    if (sscanf(rest, "deepstack_merger_list.%d.%n", &index, &used) == 1 &&
        used > 0 && index >= 0 && index < 3) {
        for (size_t i = 0; i < sizeof(merger_names) / sizeof(*merger_names); i++) {
            size_t length = strlen(merger_names[i].hf);
            if (!strncmp(rest + used, merger_names[i].hf, length)) {
                snprintf(out, size, "v.deepstack.%d.%s%s",
                         deepstack_layers[index], merger_names[i].clip,
                         rest + used + length);
                return 1;
            }
        }
        return 0;
    }
    static const struct { const char *hf, *clip; } fixed[] = {
        {"merger.norm.weight", "v.post_ln.weight"},
        {"merger.norm.bias", "v.post_ln.bias"},
        {"merger.linear_fc1.weight", "mm.0.weight"},
        {"merger.linear_fc1.bias", "mm.0.bias"},
        {"merger.linear_fc2.weight", "mm.2.weight"},
        {"merger.linear_fc2.bias", "mm.2.bias"},
        {"pos_embed.weight", "v.position_embd.weight"},
        {"patch_embed.proj.bias", "v.patch_embd.bias"},
    };
    for (size_t i = 0; i < sizeof(fixed) / sizeof(*fixed); i++) {
        if (!strcmp(rest, fixed[i].hf)) {
            snprintf(out, size, "%s", fixed[i].clip);
            return 1;
        }
    }
    return 0;
}

/* Writes the n-th alternative spelling of name to out; 0 when there are
 * no more. */
static int alias_name(const char *name, int n, char *out, size_t size) {
    static const char text_prefix[] = "model.language_model.";
    static const char visual_prefix[] = "model.visual.";
    size_t text_length = sizeof(text_prefix) - 1;
    size_t visual_length = sizeof(visual_prefix) - 1;
    if (!strncmp(name, text_prefix, text_length)) {
        const char *rest = name + text_length;
        switch (n) {
        case 0: /* unsloth / transformers without the language_model level */
            snprintf(out, size, "model.%s", rest);
            return 1;
        case 1: /* mlx-vlm */
            snprintf(out, size, "language_model.model.%s", rest);
            return 1;
        case 2: { /* llama.cpp */
            int layer = -1, used = 0;
            if (!strcmp(rest, "embed_tokens.weight")) {
                snprintf(out, size, "token_embd.weight");
                return 1;
            }
            if (sscanf(rest, "layers.%d.%n", &layer, &used) == 1 && used > 0) {
                for (size_t i = 0; i < sizeof(llama_layer_names) /
                                       sizeof(*llama_layer_names); i++) {
                    if (!strcmp(rest + used, llama_layer_names[i].hf)) {
                        snprintf(out, size, "blk.%d.%s", layer,
                                 llama_layer_names[i].gguf);
                        return 1;
                    }
                }
            }
            out[0] = '\0';
            return 1; /* no llama.cpp spelling; try nothing for this n */
        }
        default:
            return 0;
        }
    }
    if (!strncmp(name, visual_prefix, visual_length)) {
        const char *rest = name + visual_length;
        switch (n) {
        case 0: snprintf(out, size, "visual.%s", rest); return 1;
        case 1: snprintf(out, size, "vision_tower.%s", rest); return 1;
        case 2: /* llama.cpp's mmproj (clip) file */
            if (!clip_name(rest, out, size)) out[0] = '\0';
            return 1;
        default: return 0;
        }
    }
    return 0;
}

size_t h3_weight_store_tensor_count(const h3_weight_store *store) {
    size_t count = 0;
    for (size_t index = 0; store && index < store->count; index++)
        count += store->headers[index].tensor_count;
    return count;
}

const h3_st_tensor *h3_weight_store_tensor(const h3_weight_store *store,
                                           size_t index,
                                           const h3_st_header **header) {
    for (size_t shard = 0; store && shard < store->count; shard++) {
        if (index < store->headers[shard].tensor_count) {
            if (header) *header = &store->headers[shard];
            return &store->headers[shard].tensors[index];
        }
        index -= store->headers[shard].tensor_count;
    }
    return NULL;
}

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header) {
    if (header) *header = NULL;
    if (!store || !name) return NULL;
    const h3_st_tensor *tensor = find_exact(store, name, header);
    char alias[256];
    for (int n = 0; !tensor && alias_name(name, n, alias, sizeof(alias)); n++)
        if (alias[0]) tensor = find_exact(store, alias, header);
    return tensor;
}

/* Reads a quantized payload's raw bytes and expands them on the GPU in a
 * command of its own. When the caller already has a command open, the
 * expansion runs on the CPU instead (h3_quant.c): freeing the raw buffer
 * discards its contents at once, which a not yet executed GPU expansion
 * in the caller's command would then read. */
static h3_gpu_tensor *expand_tensor(h3_gpu *gpu, const h3_st_header *header,
                                    const h3_st_tensor *tensor,
                                    h3_quant_type quant, size_t elements,
                                    h3_dtype dtype, uint32_t qkv_inner,
                                    uint32_t qkv_head_dim, char *error,
                                    size_t error_size) {
    uint32_t columns = tensor->ndim == 2 ? (uint32_t)tensor->shape[1] : 0;
    uint32_t block_elements = h3_quant_block_elements(quant);
    uint32_t block_bytes = h3_quant_block_bytes(quant);
    uint64_t bytes = h3_quant_bytes(quant, elements);
    uint64_t blocks = block_elements ? elements / block_elements : 0;
    if (!bytes || bytes != tensor->data_end - tensor->data_begin ||
        blocks > UINT32_MAX) {
        fail(error, error_size, "weight %s: %s payload does not match its shape",
             tensor->name, h3_quant_name(quant));
        return NULL;
    }
    char detail[256] = "";
    if (h3_gpu_command_open(gpu)) {
        void *raw = malloc((size_t)bytes);
        float *values = malloc(elements * sizeof(*values));
        uint16_t *bf16 = dtype == H3_DTYPE_BF16 ?
            malloc(elements * sizeof(*bf16)) : NULL;
        int ok = raw && values && (dtype != H3_DTYPE_BF16 || bf16) &&
            h3_st_read_data(header, tensor, raw, (size_t)bytes, detail,
                            sizeof(detail)) &&
            h3_quant_dequantize(quant, raw, elements, values);
        if (ok && qkv_inner) {
            /* Same reorder as the GPU path: [Q; K; V] -> per-head [q; k; v]. */
            float *ordered = malloc(elements * sizeof(*ordered));
            ok = ordered != NULL;
            for (size_t row = 0; ok && row < (size_t)3 * qkv_inner; row++) {
                size_t part = row / qkv_inner, within = row % qkv_inner;
                size_t head = within / qkv_head_dim, d = within % qkv_head_dim;
                size_t destination = head * 3 * qkv_head_dim + part * qkv_head_dim + d;
                memcpy(ordered + destination * columns, values + row * columns,
                       columns * sizeof(float));
            }
            free(values);
            values = ordered;
        }
        h3_gpu_tensor *result = NULL;
        if (ok && bf16) {
            for (size_t index = 0; index < elements; index++) {
                uint32_t bits;
                memcpy(&bits, &values[index], sizeof(bits));
                bits += 0x7fffu + ((bits >> 16) & 1u); /* round to nearest even */
                bf16[index] = (uint16_t)(bits >> 16);
            }
            result = h3_gpu_tensor_from_bf16(gpu, bf16, elements);
        } else if (ok) {
            result = h3_gpu_tensor_from_f32(gpu, values, elements);
        }
        free(raw);
        free(values);
        free(bf16);
        if (!result)
            fail(error, error_size, "cannot expand %s (%s): %s", tensor->name,
                 h3_quant_name(quant), detail[0] ? detail : "out of memory");
        return result;
    }
    h3_gpu_tensor *raw = h3_gpu_tensor_new_i8(gpu, (size_t)bytes);
    h3_gpu_tensor *result = dtype == H3_DTYPE_BF16 ?
        h3_gpu_tensor_new_bf16(gpu, elements) : h3_gpu_tensor_new_f32(gpu, elements);
    int ok = raw && result &&
        h3_gpu_tensor_read_file_i8(raw, header->path, tensor->file_offset,
                                   (size_t)bytes, detail, sizeof(detail)) &&
        h3_gpu_begin(gpu) &&
        h3_gpu_dequantize(gpu, result, raw, (uint32_t)quant, (uint32_t)blocks,
                          block_elements, block_bytes, columns, qkv_inner,
                          qkv_head_dim) &&
        h3_gpu_submit(gpu);
    h3_gpu_tensor_free(raw);
    if (!ok) {
        fail(error, error_size, "cannot expand %s (%s): %s", tensor->name,
             h3_quant_name(quant), detail[0] ? detail : h3_gpu_error(gpu));
        h3_gpu_tensor_free(result);
        return NULL;
    }
    return result;
}

/* llama.cpp's mmproj stores the [out, 3, 2, 16, 16] temporal patch
 * embedding as two [out, 3, 16, 16] tensors, one per frame of the pair
 * (v.patch_embd.weight and .weight.1). Reassembled on the CPU - 2 MB. */
static h3_gpu_tensor *load_split_patch_embed(const h3_weight_store *store,
                                             h3_gpu *gpu, int ndim,
                                             const uint64_t *shape,
                                             h3_dtype dtype, char *error,
                                             size_t error_size) {
    const h3_st_header *headers[2] = {NULL, NULL};
    const h3_st_tensor *parts[2] = {
        find_exact(store, "v.patch_embd.weight", &headers[0]),
        find_exact(store, "v.patch_embd.weight.1", &headers[1]),
    };
    if (!parts[0] || !parts[1] || ndim != 5 || shape[2] != 2) {
        fail(error, error_size,
             "required weight is absent: model.visual.patch_embed.proj.weight");
        return NULL;
    }
    size_t per_frame_inner = (size_t)(shape[3] * shape[4]);
    size_t channels = (size_t)(shape[0] * shape[1]);
    size_t half = channels * per_frame_inner, elements = 2 * half;
    float *frames[2] = {NULL, NULL};
    float *joined = malloc(elements * sizeof(float));
    int ok = joined != NULL;
    for (int t = 0; ok && t < 2; t++) {
        const h3_st_tensor *part = parts[t];
        h3_quant_type quant = part->quant ? (h3_quant_type)part->quant :
            part->dtype == H3_DTYPE_F32 ? H3_QUANT_F32 :
            part->dtype == H3_DTYPE_F16 ? H3_QUANT_F16 :
            part->dtype == H3_DTYPE_BF16 ? H3_QUANT_BF16 : H3_QUANT_NONE;
        uint64_t bytes = h3_quant_bytes(quant, half);
        void *raw = bytes && bytes == part->data_end - part->data_begin ?
            malloc((size_t)bytes) : NULL;
        frames[t] = malloc(half * sizeof(float));
        char detail[256];
        ok = raw && frames[t] && part->ndim == 4 && part->shape[0] == shape[0] &&
             part->shape[1] == shape[1] && part->shape[2] == shape[3] &&
             part->shape[3] == shape[4] &&
             h3_st_read_data(headers[t], part, raw, (size_t)bytes, detail,
                             sizeof(detail)) &&
             h3_quant_dequantize(quant, raw, half, frames[t]);
        free(raw);
    }
    for (size_t c = 0; ok && c < channels; c++)
        for (int t = 0; t < 2; t++)
            memcpy(joined + (c * 2 + (size_t)t) * per_frame_inner,
                   frames[t] + c * per_frame_inner,
                   per_frame_inner * sizeof(float));
    h3_gpu_tensor *result = NULL;
    if (ok && dtype == H3_DTYPE_F32) {
        result = h3_gpu_tensor_from_f32(gpu, joined, elements);
    } else if (ok) {
        uint16_t *bf16 = malloc(elements * sizeof(*bf16));
        for (size_t i = 0; bf16 && i < elements; i++) {
            uint32_t bits;
            memcpy(&bits, &joined[i], sizeof(bits));
            bits += 0x7fffu + ((bits >> 16) & 1u);
            bf16[i] = (uint16_t)(bits >> 16);
        }
        result = bf16 ? h3_gpu_tensor_from_bf16(gpu, bf16, elements) : NULL;
        free(bf16);
    }
    free(frames[0]);
    free(frames[1]);
    free(joined);
    if (!result)
        fail(error, error_size, "cannot reassemble the split patch embedding");
    return result;
}

static h3_gpu_tensor *load_tensor(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape, h3_dtype dtype,
                                  char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor && !strcmp(name, "model.visual.patch_embed.proj.weight"))
        return load_split_patch_embed(store, gpu, ndim, shape, dtype, error,
                                      error_size);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return NULL;
    }
    /* A quantized (GGUF block or F16) payload is expanded on the GPU to
     * the dtype asked for. */
    h3_quant_type quant = (h3_quant_type)tensor->quant;
    /* Quantized and F16 payloads expand to the dtype asked for; F32 and BF16
     * convert into each other (GGUF files keep norms in F32 where the
     * released checkpoints have BF16). */
    if (quant == H3_QUANT_NONE && tensor->dtype != dtype &&
        (tensor->dtype == H3_DTYPE_F32 || tensor->dtype == H3_DTYPE_BF16))
        quant = tensor->dtype == H3_DTYPE_F32 ? H3_QUANT_F32 : H3_QUANT_BF16;
    int expand = tensor->dtype != dtype && quant != H3_QUANT_NONE;
    uint32_t qkv_inner = 0, qkv_head_dim = 0;
    if (h3_weight_qkv_reorder(store, name, tensor, &qkv_inner, &qkv_head_dim) &&
        (expand || tensor->dtype == dtype)) {
        /* A plain BF16/F32 tensor goes through the expansion kernel as a
         * copy, just to reorder its rows. */
        if (!expand)
            quant = tensor->dtype == H3_DTYPE_BF16 ? H3_QUANT_BF16 : H3_QUANT_F32;
        expand = 1;
    }
    if ((tensor->dtype != dtype && !expand) || tensor->ndim != ndim) {
        fail(error, error_size, "weight %s has dtype/rank %s/%d, expected %s/%d",
             name, quant != H3_QUANT_NONE ? h3_quant_name(quant) :
             h3_dtype_name(tensor->dtype), tensor->ndim,
             h3_dtype_name(dtype), ndim);
        return NULL;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size, "weight %s shape mismatch at dimension %d",
                 name, dimension);
            return NULL;
        }
        if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
            fail(error, error_size, "weight %s shape overflows", name);
            return NULL;
        }
        elements *= shape[dimension];
    }
    if (elements > SIZE_MAX) {
        fail(error, error_size, "weight %s is too large for this process", name);
        return NULL;
    }
    if (expand)
        return expand_tensor(gpu, header, tensor, quant, (size_t)elements,
                             dtype, qkv_inner, qkv_head_dim, error,
                             error_size);
    h3_gpu_tensor *result = dtype == H3_DTYPE_BF16 ?
        h3_gpu_tensor_load_bf16(gpu, header->path, tensor->file_offset,
                                (size_t)elements) :
        h3_gpu_tensor_load_f32(gpu, header->path, tensor->file_offset,
                               (size_t)elements);
    if (!result) {
        fail(error, error_size, "cannot load %s: %s", name, h3_gpu_error(gpu));
    }
    return result;
}

h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_BF16,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_F32,
                       error, error_size);
}
