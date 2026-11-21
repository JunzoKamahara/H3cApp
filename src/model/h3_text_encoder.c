#define ACCELERATE_NEW_LAPACK
#include "h3_memory.h"
#include "h3_options.h"
#include "h3_text_encoder.h"
#include "h3_quant.h"
#include "h3_text_projection.h"

#include "h3_weights.h"

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    /* H3 reads the released first 50 layers of Qwen3-VL-32B. */
    TEXT_RELEASED_LAYERS = 50,
    TEXT_MAX_LAYERS = 64,
    TEXT_VOCAB = 151936,
    /* Qwen3-VL's own tokens end below this ID. H3's tokenizer adds its tags
     * (<d>, </d>, ...) from here on: rows of the embedding a stock Qwen3-VL
     * never trained. */
    TEXT_STOCK_VOCAB = 151669,
    TEXT_HEAD_DIM = 128,
    TEXT_ROPE_HALF = TEXT_HEAD_DIM / 2,
    TEXT_DEFERRED_WEIGHTS = 1 + TEXT_MAX_LAYERS * 11,
    /* P-H3-Image's text connector: five hidden states of its 36-layer
     * Qwen3-VL-4B, concatenated, to the 5120 values the DiT reads. */
    CONNECTOR_TAPS = 5,
    CONNECTOR_LAYERS = 36,
    CONNECTOR_WIDTH = H3_TEXT_HIDDEN_SIZE
};

/* The language model's sizes, from the shapes of its weights. */
typedef struct {
    uint32_t layers, hidden, intermediate, query_heads, kv_heads;
    uint32_t query_dim, kv_dim;
    /* P-H3-Image: every layer, the final norm and the text connector. */
    int connector;
} text_shape;

/* hidden_states indices of the connector (0 is the embedding output, i the
 * output of layer i, the last one after the final norm). */
static const int connector_taps[CONNECTOR_TAPS] = {13, 19, 25, 31, 36};

static const float TEXT_RMS_EPSILON = 1e-6f;
static const float TEXT_ROPE_THETA = 5000000.0f;

static float round_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= UINT32_C(0xffff0000);
    memcpy(&value, &bits, sizeof(value));
    return value;
}

typedef struct {
    h3_gpu_tensor *input_norm;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *query_norm;
    h3_gpu_tensor *key_norm;
    h3_gpu_tensor *attention_output;
    h3_gpu_tensor *post_norm;
    h3_gpu_tensor *gate;
    h3_gpu_tensor *up;
    h3_gpu_tensor *down;
} text_layer_weights;

typedef struct {
    const h3_weight_store *store;
    h3_gpu *gpu;
    h3_gpu_tensor *deferred[TEXT_DEFERRED_WEIGHTS];
    size_t deferred_count;
    char *error;
    size_t error_size;
    const text_shape *shape;
} load_context;

typedef struct {
    const h3_weight_store *store;
    const text_shape *shape;
    int layer;
    text_layer_weights *weights;
    int lane;
    int lanes;
    int ok;
    char error[512];
} text_layer_prefetch;

typedef struct {
    int occupied;
    int active;
    int layer;
    int lanes;
    load_context load;
    text_layer_weights weights;
    text_layer_prefetch prefetch[8];
    pthread_t threads[8];
    int started[8];
} text_prefetch_slot;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static h3_gpu_tensor *defer(load_context *load, h3_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    if (load->deferred_count >= TEXT_DEFERRED_WEIGHTS) {
        fail(load->error, load->error_size, "Qwen deferred weight registry overflow");
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    load->deferred[load->deferred_count++] = tensor;
    return tensor;
}

static void retire_deferred(load_context *load) {
    for (size_t index = 0; index < load->deferred_count; index++)
        h3_gpu_tensor_free(load->deferred[index]);
    load->deferred_count = 0;
}

static h3_gpu_tensor *load_1d(load_context *load, const char *name,
                              uint64_t width) {
    uint64_t shape[] = {width};
    return defer(load, h3_weight_load_bf16(load->store, load->gpu, name, 1,
                                            shape, load->error,
                                            load->error_size));
}

static h3_gpu_tensor *load_2d(load_context *load, const char *name,
                              uint64_t rows, uint64_t columns) {
    uint64_t shape[] = {rows, columns};
    return defer(load, h3_weight_load_bf16(load->store, load->gpu, name, 2,
                                            shape, load->error,
                                            load->error_size));
}

static int text_prefetch_threads(void) {
    const char *value = h3_option("H3_QWEN_PREFETCH");
    if (!value || !*value) return 8;
    if (!strcmp(value, "0")) return 0;
    char *tail = NULL;
    long threads = strtol(value, &tail, 10);
    if (!tail || *tail || threads < 1) threads = 1;
    if (threads > 8) threads = 8;
    return (int)threads;
}

static int text_prefetch_depth(const h3_gpu *gpu) {
    const char *value = h3_option("H3_QWEN_PREFETCH_DEPTH");
    if (!value || !*value) return h3_gpu_is_m5(gpu) ? 3 : 2;
    char *tail = NULL;
    long depth = strtol(value, &tail, 10);
    if (!tail || *tail || depth < 1) depth = 1;
    if (depth > 6) depth = 6;
    return (int)depth;
}

static int layer_weights_load(load_context *load, int layer,
                              text_layer_weights *weights) {
    const text_shape *S = load->shape;
    char prefix[96];
    int length = snprintf(prefix, sizeof(prefix),
                          "model.language_model.layers.%d.", layer);
    if (length < 0 || (size_t)length >= sizeof(prefix)) {
        fail(load->error, load->error_size, "cannot format Qwen layer name");
        return 0;
    }
#define LOAD_1D(field, suffix, width) do {                                      \
    char name[192];                                                             \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                     \
    weights->field = load_1d(load, name, width);                                \
    if (!weights->field) return 0;                                              \
} while (0)
#define LOAD_2D(field, suffix, rows, columns) do {                              \
    char name[192];                                                             \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                     \
    weights->field = load_2d(load, name, rows, columns);                        \
    if (!weights->field) return 0;                                              \
} while (0)
    LOAD_1D(input_norm, "input_layernorm.weight", S->hidden);
    LOAD_2D(query, "self_attn.q_proj.weight", S->query_dim, S->hidden);
    LOAD_2D(key, "self_attn.k_proj.weight", S->kv_dim, S->hidden);
    LOAD_2D(value, "self_attn.v_proj.weight", S->kv_dim, S->hidden);
    LOAD_1D(query_norm, "self_attn.q_norm.weight", TEXT_HEAD_DIM);
    LOAD_1D(key_norm, "self_attn.k_norm.weight", TEXT_HEAD_DIM);
    LOAD_2D(attention_output, "self_attn.o_proj.weight", S->hidden,
            S->query_dim);
    LOAD_1D(post_norm, "post_attention_layernorm.weight", S->hidden);
    LOAD_2D(gate, "mlp.gate_proj.weight", S->intermediate, S->hidden);
    LOAD_2D(up, "mlp.up_proj.weight", S->intermediate, S->hidden);
    LOAD_2D(down, "mlp.down_proj.weight", S->hidden, S->intermediate);
#undef LOAD_1D
#undef LOAD_2D
    return 1;
}

static h3_gpu_tensor *allocate_bf16(load_context *load, size_t elements) {
    h3_gpu_tensor *tensor = defer(
        load, h3_gpu_tensor_new_bf16(load->gpu, elements));
    if (!tensor) {
        fail(load->error, load->error_size,
             "cannot allocate prefetched Qwen weight: %s",
             h3_gpu_error(load->gpu));
    }
    return tensor;
}

static int layer_weights_allocate(load_context *load,
                                  text_layer_weights *weights) {
    const text_shape *S = load->shape;
#define ALLOCATE(field, elements) do {                                         \
    weights->field = allocate_bf16(load, (elements));                          \
    if (!weights->field) return 0;                                              \
} while (0)
    ALLOCATE(input_norm, S->hidden);
    ALLOCATE(query, (size_t)S->query_dim * S->hidden);
    ALLOCATE(key, (size_t)S->kv_dim * S->hidden);
    ALLOCATE(value, (size_t)S->kv_dim * S->hidden);
    ALLOCATE(query_norm, TEXT_HEAD_DIM);
    ALLOCATE(key_norm, TEXT_HEAD_DIM);
    ALLOCATE(attention_output, (size_t)S->hidden * S->query_dim);
    ALLOCATE(post_norm, S->hidden);
    ALLOCATE(gate, (size_t)S->intermediate * S->hidden);
    ALLOCATE(up, (size_t)S->intermediate * S->hidden);
    ALLOCATE(down, (size_t)S->hidden * S->intermediate);
#undef ALLOCATE
    return 1;
}

static int read_weight_bf16(const h3_weight_store *store, const char *name,
                            int ndim, const uint64_t *shape,
                            h3_gpu_tensor *destination,
                            char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return 0;
    }
    if (h3_weight_is_comfy_int8(store, tensor)) {
        /* A ComfyUI int8 text encoder (P-H3-Image's int8_convrot file):
         * restored on this prefetch thread like a GGUF one. */
        uint64_t elements = ndim == 2 ? shape[0] * shape[1] : 0;
        if (ndim != 2 || tensor->shape[0] != shape[0] ||
            tensor->shape[1] != shape[1] || elements > SIZE_MAX) {
            fail(error, error_size, "weight %s shape mismatch", name);
            return 0;
        }
        size_t bf16_bytes = (size_t)elements * sizeof(uint16_t);
        uint16_t *bf16 = h3_memory_scratch(bf16_bytes);
        int ok = bf16 &&
            h3_weight_comfy_int8_read_bf16(store, header, tensor, bf16,
                                           (size_t)elements, error,
                                           error_size) &&
            h3_gpu_tensor_write_bf16(destination, bf16, (size_t)elements);
        h3_memory_scratch_free(bf16, bf16_bytes);
        if (!ok && error && error_size && !error[0])
            fail(error, error_size, "cannot expand %s", name);
        return ok;
    }
    h3_quant_type quant = (h3_quant_type)tensor->quant;
    if (quant == H3_QUANT_NONE && tensor->dtype == H3_DTYPE_F32)
        quant = H3_QUANT_F32; /* converted below like any expansion */
    int quantized = tensor->dtype != H3_DTYPE_BF16 && quant != H3_QUANT_NONE &&
                    quant != H3_QUANT_BF16;
    if ((tensor->dtype != H3_DTYPE_BF16 && !quantized) || tensor->ndim != ndim) {
        fail(error, error_size,
             "weight %s has dtype/rank %s/%d, expected BF16/%d", name,
             quantized || quant ? h3_quant_name(quant) :
             h3_dtype_name(tensor->dtype), tensor->ndim, ndim);
        return 0;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size,
                 "weight %s shape mismatch at dimension %d", name,
                 dimension);
            return 0;
        }
        if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
            fail(error, error_size, "weight %s shape overflows", name);
            return 0;
        }
        elements *= shape[dimension];
    }
    if (quantized && elements <= SIZE_MAX) {
        /* A quantized text encoder (GGUF): expanded on this prefetch thread
         * on the CPU, straight into the BF16 destination. The scratch is
         * reused from layer to layer and returned when the encoder is done
         * (h3_memory_scratch): malloc kept a layer's worth of it for every
         * layer read ahead, through the denoise. */
        uint64_t bytes = h3_quant_bytes(quant, elements);
        size_t bf16_bytes = (size_t)elements * sizeof(uint16_t);
        void *raw = bytes == tensor->data_end - tensor->data_begin ?
            h3_memory_scratch((size_t)bytes) : NULL;
        uint16_t *bf16 = raw ? h3_memory_scratch(bf16_bytes) : NULL;
        int ok = bf16 &&
            h3_st_read_data(header, tensor, raw, (size_t)bytes, error,
                            error_size) &&
            h3_quant_dequantize_bf16(quant, raw, elements, bf16) &&
            h3_gpu_tensor_write_bf16(destination, bf16, (size_t)elements);
        h3_memory_scratch_free(raw, (size_t)bytes);
        h3_memory_scratch_free(bf16, bf16_bytes);
        if (!ok && error && error_size && !error[0])
            fail(error, error_size, "cannot expand %s (%s)", name,
                 h3_quant_name(quant));
        return ok;
    }
    if (elements > SIZE_MAX ||
        !h3_gpu_tensor_read_file_bf16(destination, header->path,
                                      tensor->file_offset, (size_t)elements,
                                      error, error_size)) {
        if (error && error_size && !error[0])
            fail(error, error_size, "cannot prefetch %s", name);
        return 0;
    }
    return 1;
}

static int layer_weights_read_lane(const h3_weight_store *store,
                                   const text_shape *S, int layer,
                                   text_layer_weights *weights,
                                   int lane, int lanes,
                                   char *error, size_t error_size) {
    char prefix[96];
    int length = snprintf(prefix, sizeof(prefix),
                          "model.language_model.layers.%d.", layer);
    if (length < 0 || (size_t)length >= sizeof(prefix)) {
        fail(error, error_size, "cannot format Qwen layer name");
        return 0;
    }
#define READ_1D(field, suffix, width) do {                                     \
    int selected = item++ % lanes == lane;                                     \
    char name[192];                                                             \
    uint64_t shape[] = {width};                                                 \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    if (selected && !read_weight_bf16(                                         \
                          store, name, 1, shape, weights->field,               \
                          error, error_size)) return 0;                         \
} while (0)
#define READ_2D(field, suffix, rows, columns) do {                             \
    int selected = item++ % lanes == lane;                                     \
    char name[192];                                                             \
    uint64_t shape[] = {rows, columns};                                         \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    if (selected && !read_weight_bf16(                                         \
                          store, name, 2, shape, weights->field,               \
                          error, error_size)) return 0;                         \
} while (0)
    int item = 0;
    READ_1D(input_norm, "input_layernorm.weight", S->hidden);
    READ_2D(query, "self_attn.q_proj.weight", S->query_dim, S->hidden);
    READ_2D(key, "self_attn.k_proj.weight", S->kv_dim, S->hidden);
    READ_2D(value, "self_attn.v_proj.weight", S->kv_dim, S->hidden);
    READ_1D(query_norm, "self_attn.q_norm.weight", TEXT_HEAD_DIM);
    READ_1D(key_norm, "self_attn.k_norm.weight", TEXT_HEAD_DIM);
    READ_2D(attention_output, "self_attn.o_proj.weight", S->hidden,
            S->query_dim);
    READ_1D(post_norm, "post_attention_layernorm.weight", S->hidden);
    READ_2D(gate, "mlp.gate_proj.weight", S->intermediate, S->hidden);
    READ_2D(up, "mlp.up_proj.weight", S->intermediate, S->hidden);
    READ_2D(down, "mlp.down_proj.weight", S->hidden, S->intermediate);
#undef READ_1D
#undef READ_2D
    return 1;
}

static void *layer_prefetch_main(void *opaque) {
    text_layer_prefetch *prefetch = opaque;
    prefetch->ok = layer_weights_read_lane(
        prefetch->store, prefetch->shape, prefetch->layer, prefetch->weights,
        prefetch->lane, prefetch->lanes,
        prefetch->error, sizeof(prefetch->error));
    return NULL;
}

static void prefetch_slot_retire(text_prefetch_slot *slot) {
    if (!slot || !slot->occupied) return;
    if (slot->active) {
        for (int lane = 0; lane < slot->lanes; lane++) {
            if (slot->started[lane])
                pthread_join(slot->threads[lane], NULL);
        }
    }
    retire_deferred(&slot->load);
    memset(slot, 0, sizeof(*slot));
}

static int prefetch_slot_start(text_prefetch_slot *slot,
                               const h3_weight_store *store,
                               const text_shape *shape, h3_gpu *gpu,
                               int layer, int lanes,
                               char *error, size_t error_size) {
    if (!slot || slot->occupied || lanes < 1 || lanes > 8) return 0;
    memset(slot, 0, sizeof(*slot));
    slot->occupied = 1;
    slot->active = 1;
    slot->layer = layer;
    slot->lanes = lanes;
    slot->load.store = store;
    slot->load.shape = shape;
    slot->load.gpu = gpu;
    slot->load.error = error;
    slot->load.error_size = error_size;
    if (!layer_weights_allocate(&slot->load, &slot->weights)) {
        prefetch_slot_retire(slot);
        return 0;
    }
    for (int lane = 0; lane < lanes; lane++) {
        text_layer_prefetch *prefetch = &slot->prefetch[lane];
        prefetch->store = store;
        prefetch->shape = shape;
        prefetch->layer = layer;
        prefetch->weights = &slot->weights;
        prefetch->lane = lane;
        prefetch->lanes = lanes;
        if (pthread_create(&slot->threads[lane], NULL,
                           layer_prefetch_main, prefetch) == 0) {
            slot->started[lane] = 1;
        } else {
            prefetch->ok = layer_weights_read_lane(
                store, shape, layer, &slot->weights, lane, lanes,
                prefetch->error, sizeof(prefetch->error));
        }
    }
    return 1;
}

static int prefetch_slot_take(text_prefetch_slot *slot,
                              load_context *load,
                              text_layer_weights *weights,
                              char *error, size_t error_size) {
    if (!slot || !slot->occupied || !load || !weights) return 0;
    if (slot->active) {
        for (int lane = 0; lane < slot->lanes; lane++) {
            if (slot->started[lane])
                pthread_join(slot->threads[lane], NULL);
        }
        slot->active = 0;
    }
    for (int lane = 0; lane < slot->lanes; lane++) {
        if (!slot->prefetch[lane].ok) {
            fail(error, error_size,
                 "Qwen layer %d prefetch lane %d failed: %s",
                 slot->layer, lane,
                 slot->prefetch[lane].error[0] ?
                     slot->prefetch[lane].error : "unknown error");
            prefetch_slot_retire(slot);
            return 0;
        }
    }
    *load = slot->load;
    *weights = slot->weights;
    slot->load.deferred_count = 0;
    memset(slot, 0, sizeof(*slot));
    return 1;
}

static void prefetch_slots_retire(text_prefetch_slot *slots, int count) {
    for (int index = 0; index < count; index++)
        prefetch_slot_retire(&slots[index]);
}

/* H3_DEBUG_HASHES=1: log a hash of the hidden state after each layer (see
 * h3_debug_hash in h3.c). */
static int debug_hashes_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) enabled = h3_option("H3_DEBUG_HASHES") != NULL;
    return enabled;
}

static void debug_hash_tensor(const h3_gpu_tensor *tensor, size_t count,
                              int layer, const char *label) {
    if (!debug_hashes_enabled()) return;
    uint16_t *values = malloc(count * sizeof(*values));
    if (!values) return;
    if (h3_gpu_tensor_read_bf16(tensor, values, count)) {
        uint64_t hash = 1469598103934665603ULL;
        const unsigned char *bytes = (const unsigned char *)values;
        for (size_t i = 0; i < count * sizeof(*values); i++) {
            hash ^= bytes[i];
            hash *= 1099511628211ULL;
        }
        fprintf(stderr, "h3: hash qwen layer %d %s %016llx\n", layer, label,
                (unsigned long long)hash);
    }
    free(values);
}

static void debug_hash_hidden(const h3_gpu_tensor *hidden, size_t count,
                              int layer) {
    debug_hash_tensor(hidden, count, layer, "out");
}

static int gpu_operation(h3_gpu *gpu, int ok, char *error, size_t error_size,
                         const char *operation, int layer) {
    if (ok) return 1;
    if (layer >= 0) {
        fail(error, error_size, "Qwen layer %d %s failed: %s", layer,
             operation, h3_gpu_error(gpu));
    } else {
        fail(error, error_size, "Qwen %s failed: %s", operation,
             h3_gpu_error(gpu));
    }
    return 0;
}

static int encode_layer(h3_gpu *gpu, const text_shape *S,
                        const text_layer_weights *weight,
                        uint32_t tokens, h3_gpu_tensor *hidden,
                        h3_gpu_tensor *norm, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        h3_gpu_tensor *attention_heads,
                        h3_gpu_tensor *attention_output,
                        h3_gpu_tensor *gate, h3_gpu_tensor *up,
                        h3_gpu_tensor *mlp_output,
                        h3_gpu_tensor *rope_cos, h3_gpu_tensor *rope_sin,
                        int layer, char *error, size_t error_size) {
#define OP(call, label) do {                                                    \
    if (!gpu_operation(gpu, (call), error, error_size, label, layer)) return 0; \
} while (0)
    OP(h3_gpu_rms_norm_bf16(gpu, norm, hidden, weight->input_norm,
                             tokens, S->hidden, TEXT_RMS_EPSILON),
       "input RMSNorm");
    OP(h3_gpu_linear_bf16(gpu, query, norm, weight->query, NULL, tokens,
                           S->hidden, S->query_dim), "query projection");
    OP(h3_gpu_linear_bf16(gpu, key, norm, weight->key, NULL, tokens,
                           S->hidden, S->kv_dim), "key projection");
    OP(h3_gpu_linear_bf16(gpu, value, norm, weight->value, NULL, tokens,
                           S->hidden, S->kv_dim), "value projection");
    OP(h3_gpu_head_rms_norm_bf16(gpu, query, weight->query_norm, tokens,
                                  S->query_heads, TEXT_HEAD_DIM,
                                  TEXT_RMS_EPSILON), "query RMSNorm");
    OP(h3_gpu_head_rms_norm_bf16(gpu, key, weight->key_norm, tokens,
                                  S->kv_heads, TEXT_HEAD_DIM,
                                  TEXT_RMS_EPSILON), "key RMSNorm");
    OP(h3_gpu_rope_text_bf16(gpu, query, key, rope_cos, rope_sin, tokens,
                              S->query_heads, S->kv_heads, TEXT_HEAD_DIM),
       "RoPE");
    OP(h3_gpu_gqa_causal_bf16(gpu, attention_heads, query, key, value, tokens,
                               S->query_heads, S->kv_heads, TEXT_HEAD_DIM,
                               1.0f / sqrtf((float)TEXT_HEAD_DIM)),
       "causal GQA");
    OP(h3_gpu_linear_bf16(gpu, attention_output, attention_heads,
                           weight->attention_output, NULL, tokens,
                           S->query_dim, S->hidden),
       "attention output projection");
    OP(h3_gpu_add_bf16(gpu, hidden, hidden, attention_output,
                        tokens * S->hidden), "attention residual");
    OP(h3_gpu_rms_norm_bf16(gpu, norm, hidden, weight->post_norm, tokens,
                             S->hidden, TEXT_RMS_EPSILON),
       "post-attention RMSNorm");
    OP(h3_gpu_linear_bf16(gpu, gate, norm, weight->gate, NULL, tokens,
                           S->hidden, S->intermediate), "MLP gate");
    OP(h3_gpu_linear_bf16(gpu, up, norm, weight->up, NULL, tokens,
                           S->hidden, S->intermediate), "MLP up");
    OP(h3_gpu_silu_mul_bf16(gpu, gate, gate, up,
                             tokens * S->intermediate), "fused SwiGLU");
    OP(h3_gpu_linear_bf16(gpu, mlp_output, gate, weight->down, NULL, tokens,
                           S->intermediate, S->hidden), "MLP down");
    OP(h3_gpu_add_bf16(gpu, hidden, hidden, mlp_output,
                        tokens * S->hidden), "MLP residual");
#undef OP
    return 1;
}

void h3_text_embedding_free(h3_text_embedding *embedding) {
    if (!embedding) return;
    free(embedding->values);
    free(embedding->tags);
    memset(embedding, 0, sizeof(*embedding));
}

/* The language model's sizes from its weights: H3's Qwen3-VL-32B (read to
 * its released 50 layers) or, with a text connector beside it,
 * P-H3-Image's Qwen3-VL-4B (all of its layers and its final norm). With a
 * projection it is the small Qwen3-VL the projection was fitted on, read
 * to the hidden state the projection takes; a text connector beside it is
 * not used. */
static int read_text_shape(const h3_weight_store *store,
                           const h3_text_projection *projection,
                           text_shape *shape, char *error, size_t error_size) {
    memset(shape, 0, sizeof(*shape));
    const h3_st_tensor *embed = h3_weight_find(
        store, "model.language_model.embed_tokens.weight", NULL);
    const h3_st_tensor *query = h3_weight_find(
        store, "model.language_model.layers.0.self_attn.q_proj.weight", NULL);
    const h3_st_tensor *key = h3_weight_find(
        store, "model.language_model.layers.0.self_attn.k_proj.weight", NULL);
    const h3_st_tensor *gate = h3_weight_find(
        store, "model.language_model.layers.0.mlp.gate_proj.weight", NULL);
    if (!embed || !query || !key || !gate || embed->ndim != 2 ||
        query->ndim != 2 || key->ndim != 2 || gate->ndim != 2) {
        fail(error, error_size, "the text encoder's Qwen weights are missing");
        return 0;
    }
    uint64_t hidden = embed->shape[1];
    if (embed->shape[0] != TEXT_VOCAB || hidden < TEXT_HEAD_DIM ||
        hidden > 8192 || query->shape[1] != hidden || key->shape[1] != hidden ||
        gate->shape[1] != hidden || query->shape[0] % TEXT_HEAD_DIM ||
        key->shape[0] % TEXT_HEAD_DIM || !key->shape[0] ||
        (query->shape[0] / TEXT_HEAD_DIM) % (key->shape[0] / TEXT_HEAD_DIM) ||
        gate->shape[0] < 1 || gate->shape[0] > 65536) {
        fail(error, error_size, "the text encoder has an unsupported shape");
        return 0;
    }
    shape->hidden = (uint32_t)hidden;
    shape->intermediate = (uint32_t)gate->shape[0];
    shape->query_heads = (uint32_t)(query->shape[0] / TEXT_HEAD_DIM);
    shape->kv_heads = (uint32_t)(key->shape[0] / TEXT_HEAD_DIM);
    shape->query_dim = shape->query_heads * TEXT_HEAD_DIM;
    shape->kv_dim = shape->kv_heads * TEXT_HEAD_DIM;
    shape->connector = !projection &&
        h3_weight_find(store, "text_connector.proj.weight", NULL) != NULL;
    if (!shape->connector && !projection) {
        if (hidden != H3_TEXT_HIDDEN_SIZE) {
            fail(error, error_size, "the text encoder is %llu wide, not the "
                 "%u of H3's Qwen3-VL", (unsigned long long)hidden,
                 H3_TEXT_HIDDEN_SIZE);
            return 0;
        }
        shape->layers = TEXT_RELEASED_LAYERS;
        return 1;
    }
    int layers = 0;
    for (; layers <= TEXT_MAX_LAYERS; layers++) {
        char name[96];
        snprintf(name, sizeof(name),
                 "model.language_model.layers.%d.input_layernorm.weight", layers);
        if (!h3_weight_find(store, name, NULL)) break;
    }
    if (projection) {
        if (hidden != projection->input_width) {
            fail(error, error_size, "the text projection is for a Qwen3-VL "
                 "%zu wide, this text encoder is %llu wide",
                 projection->input_width, (unsigned long long)hidden);
            return 0;
        }
        /* hidden_states[layers] is after the final norm: not a layer's
         * input, and no projection reads it. */
        if (projection->tap >= layers) {
            fail(error, error_size, "the text projection reads hidden state "
                 "%d, this text encoder has %d layers", projection->tap,
                 layers);
            return 0;
        }
        shape->layers = (uint32_t)projection->tap;
        return 1;
    }
    if (layers != CONNECTOR_LAYERS ||
        !h3_weight_find(store, "model.language_model.norm.weight", NULL)) {
        fail(error, error_size, "the text encoder beside a text connector has "
             "%d layers, not the %d of P-H3-Image's Qwen3-VL-4B", layers,
             CONNECTOR_LAYERS);
        return 0;
    }
    shape->layers = (uint32_t)layers;
    return 1;
}

/* An F32 tensor of the store, on the host. */
static float *read_f32(const h3_weight_store *store, const char *name,
                       uint64_t rows, uint64_t columns, char *error,
                       size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    uint64_t count = rows * columns;
    if (!tensor || tensor->dtype != H3_DTYPE_F32 || tensor->quant ||
        !((tensor->ndim == 1 && columns == 1 && tensor->shape[0] == rows) ||
          (tensor->ndim == 2 && tensor->shape[0] == rows &&
           tensor->shape[1] == columns))) {
        fail(error, error_size, "%s is missing or not F32 [%llu, %llu]", name,
             (unsigned long long)rows, (unsigned long long)columns);
        return NULL;
    }
    float *values = malloc((size_t)count * sizeof(*values));
    if (!values) {
        fail(error, error_size, "out of memory reading %s", name);
        return NULL;
    }
    if (!h3_st_read_data(header, tensor, values, (size_t)count * sizeof(*values),
                         error, error_size)) {
        free(values);
        return NULL;
    }
    return values;
}

static float bf16_value(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

static uint16_t bf16_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

/* P-H3-Image's text connector, as diffusers' PH3ImageTextConnector: the
 * taps (BF16 [tokens, hidden] each) concatenated per token, standardized,
 * projected in F32 and destandardized; the first row becomes the attention
 * sink. The result is the DiT's BF16 text rows. */
static int apply_connector(const h3_weight_store *store, const text_shape *S,
                           uint16_t *const taps[CONNECTOR_TAPS], size_t tokens,
                           uint16_t *output, char *error, size_t error_size) {
    size_t width = (size_t)CONNECTOR_TAPS * S->hidden;
    float *in_mean = read_f32(store, "text_connector.input_mean", width, 1,
                              error, error_size);
    float *in_std = in_mean ? read_f32(store, "text_connector.input_std", width,
                                       1, error, error_size) : NULL;
    float *proj = in_std ? read_f32(store, "text_connector.proj.weight",
                                    CONNECTOR_WIDTH, width, error,
                                    error_size) : NULL;
    float *out_mean = proj ? read_f32(store, "text_connector.output_mean",
                                      CONNECTOR_WIDTH, 1, error,
                                      error_size) : NULL;
    float *out_std = out_mean ? read_f32(store, "text_connector.output_std",
                                         CONNECTOR_WIDTH, 1, error,
                                         error_size) : NULL;
    float *sink = out_std ? read_f32(store, "text_connector.sink_embedding",
                                     CONNECTOR_WIDTH, 1, error,
                                     error_size) : NULL;
    float *input = sink ? malloc(tokens * width * sizeof(*input)) : NULL;
    float *projected = input ?
        malloc(tokens * CONNECTOR_WIDTH * sizeof(*projected)) : NULL;
    int ok = projected != NULL;
    if (sink && !ok) fail(error, error_size, "out of memory for the text connector");
    if (ok) {
        for (size_t token = 0; token < tokens; token++)
            for (int tap = 0; tap < CONNECTOR_TAPS; tap++)
                for (size_t channel = 0; channel < S->hidden; channel++) {
                    size_t column = (size_t)tap * S->hidden + channel;
                    input[token * width + column] =
                        (bf16_value(taps[tap][token * S->hidden + channel]) -
                         in_mean[column]) / in_std[column];
                }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)tokens,
                    CONNECTOR_WIDTH, (int)width, 1.0f, input, (int)width, proj,
                    (int)width, 0.0f, projected, CONNECTOR_WIDTH);
        for (size_t token = 0; ok && token < tokens; token++)
            for (size_t column = 0; column < CONNECTOR_WIDTH; column++) {
                float value = token == 0 ? sink[column] :
                    projected[token * CONNECTOR_WIDTH + column] *
                    out_std[column] + out_mean[column];
                if (!isfinite(value)) {
                    fail(error, error_size, "the text connector produced a "
                         "non-finite value");
                    ok = 0;
                    break;
                }
                output[token * CONNECTOR_WIDTH + column] = bf16_bits(value);
            }
    }
    free(in_mean); free(in_std); free(proj); free(out_mean); free(out_std);
    free(sink); free(input); free(projected);
    return ok;
}

static int text_encode_bf16_impl(
                        const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        const h3_text_vision_span *spans, size_t span_count,
                        const uint32_t *position_ids, const uint8_t *tags,
                        int layer_count,
                        const h3_text_projection *projection,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (!weight_directory || !shader_source_path || !token_ids || !token_count ||
        !output || token_count > UINT32_MAX || layer_count < 1 ||
        (projection && span_count)) {
        fail(error, error_size, "invalid Qwen text encoder arguments");
        return 0;
    }
    for (size_t index = 0; index < token_count; index++) {
        if (token_ids[index] >= TEXT_VOCAB) {
            fail(error, error_size, "Qwen token ID %u is outside the vocabulary",
                 token_ids[index]);
            return 0;
        }
        if (projection && token_ids[index] >= TEXT_STOCK_VOCAB) {
            fail(error, error_size, "token ID %u is one of H3's own tags "
                 "(such as <d>), which the Qwen3-VL behind a text projection "
                 "never learned; this prompt needs H3's own text encoder",
                 token_ids[index]);
            return 0;
        }
        if (tags && tags[index] > 2) {
            fail(error, error_size, "Qwen presentation tag is invalid");
            return 0;
        }
    }
    size_t span_cursor = 0;
    for (size_t index = 0; index < span_count; index++) {
        const h3_text_vision_span *span = &spans[index];
        if (!span->tokens || span->start < span_cursor ||
            span->start > token_count || span->tokens > token_count - span->start ||
            !span->embeddings || !span->deepstack[0] || !span->deepstack[1] ||
            !span->deepstack[2]) {
            fail(error, error_size, "invalid Qwen vision presentation span");
            return 0;
        }
        span_cursor = span->start + span->tokens;
    }

    h3_weight_store *store = h3_weight_store_open(weight_directory, error,
                                                   error_size);
    if (!store) return 0;
    text_shape shape;
    const text_shape *S = &shape;
    if (!read_text_shape(store, projection, &shape, error, error_size)) {
        h3_weight_store_free(store);
        return 0;
    }
    if (layer_count > (int)shape.layers) layer_count = (int)shape.layers;
    if (token_count > UINT32_MAX / S->intermediate ||
        token_count > UINT32_MAX / S->query_dim) {
        fail(error, error_size, "the prompt is too long for the text encoder");
        h3_weight_store_free(store);
        return 0;
    }
    for (size_t index = 0; index < span_count; index++) {
        size_t width = spans[index].width ? spans[index].width :
                                            H3_TEXT_HIDDEN_SIZE;
        if (width != S->hidden) {
            fail(error, error_size, "the vision tower's rows are %zu wide, "
                 "the language model's %u", width, S->hidden);
            h3_weight_store_free(store);
            return 0;
        }
    }
    /* P-H3-Image's connector reads every layer; a prefix stops short of it. */
    int connector = shape.connector && layer_count == (int)shape.layers;
    h3_gpu *gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!gpu) {
        h3_weight_store_free(store);
        return 0;
    }
    h3_gpu_profile_set_label(gpu, "Qwen text encoder");
    load_context load = {store, gpu, {NULL}, 0, error, error_size, &shape};
    uint16_t *taps[CONNECTOR_TAPS] = {NULL};
    uint32_t tokens = (uint32_t)token_count;
    size_t hidden_count = token_count * S->hidden;
    size_t query_count = token_count * S->query_dim;
    size_t kv_count = token_count * S->kv_dim;
    size_t intermediate_count = token_count * S->intermediate;

    float *cosines = malloc(token_count * TEXT_ROPE_HALF * sizeof(*cosines));
    float *sines = malloc(token_count * TEXT_ROPE_HALF * sizeof(*sines));
    if (!cosines || !sines) {
        fail(error, error_size, "out of memory allocating Qwen RoPE tables");
        free(cosines);
        free(sines);
        h3_gpu_free(gpu);
        h3_weight_store_free(store);
        return 0;
    }
    float inverse_frequency[TEXT_ROPE_HALF];
    for (size_t index = 0; index < TEXT_ROPE_HALF; index++) {
        inverse_frequency[index] = 1.0f /
            powf(TEXT_ROPE_THETA,
                 (float)(index * 2) / (float)TEXT_HEAD_DIM);
    }
    for (size_t position = 0; position < token_count; position++) {
        for (size_t index = 0; index < TEXT_ROPE_HALF; index++) {
            size_t axis = 0;
            if (position_ids && index < 60 && index % 3 == 1) axis = 1;
            else if (position_ids && index < 60 && index % 3 == 2) axis = 2;
            float coordinate = position_ids ?
                (float)position_ids[axis * token_count + position] :
                (float)position;
            float angle = coordinate * inverse_frequency[index];
            float cosine = cosf(angle);
            float sine = sinf(angle);
            /* The explicit Qwen mRoPE path casts its tables to the BF16
             * embedding dtype. Keep F32 storage for the fused Metal kernel,
             * but pin the values to that same operation boundary. */
            cosines[position * TEXT_ROPE_HALF + index] =
                position_ids ? round_bf16(cosine) : cosine;
            sines[position * TEXT_ROPE_HALF + index] =
                position_ids ? round_bf16(sine) : sine;
        }
    }

    h3_gpu_tensor *ids = h3_gpu_tensor_from_u32(gpu, token_ids, token_count);
    h3_gpu_tensor *rope_cos = h3_gpu_tensor_from_f32(
        gpu, cosines, token_count * TEXT_ROPE_HALF);
    h3_gpu_tensor *rope_sin = h3_gpu_tensor_from_f32(
        gpu, sines, token_count * TEXT_ROPE_HALF);
    free(cosines);
    free(sines);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *norm = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *query = h3_gpu_tensor_new_bf16(gpu, query_count);
    h3_gpu_tensor *key = h3_gpu_tensor_new_bf16(gpu, kv_count);
    h3_gpu_tensor *value = h3_gpu_tensor_new_bf16(gpu, kv_count);
    h3_gpu_tensor *attention_heads = h3_gpu_tensor_new_bf16(gpu, query_count);
    h3_gpu_tensor *attention_output = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *gate = h3_gpu_tensor_new_bf16(gpu, intermediate_count);
    h3_gpu_tensor *up = h3_gpu_tensor_new_bf16(gpu, intermediate_count);
    h3_gpu_tensor *mlp_output = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *deepstack[3] = {NULL, NULL, NULL};
    if (span_count) {
        uint16_t *values = calloc(hidden_count, sizeof(*values));
        if (!values) {
            fail(error, error_size, "out of memory packing Qwen deepstack rows");
            goto early_cleanup;
        }
        for (size_t layer = 0; layer < 3; layer++) {
            memset(values, 0, hidden_count * sizeof(*values));
            for (size_t index = 0; index < span_count; index++) {
                const h3_text_vision_span *span = &spans[index];
                memcpy(values + span->start * S->hidden,
                       span->deepstack[layer],
                       span->tokens * S->hidden * sizeof(*values));
            }
            deepstack[layer] = h3_gpu_tensor_from_bf16(gpu, values,
                                                        hidden_count);
        }
        free(values);
    }
    h3_gpu_tensor *activations[] = {
        ids, rope_cos, rope_sin, hidden, norm, query, key, value,
        attention_heads, attention_output, gate, up, mlp_output,
        deepstack[0], deepstack[1], deepstack[2]
    };
    int ok = 1;
    for (size_t index = 0; index < sizeof(activations) / sizeof(*activations);
         index++) {
        if (!activations[index] && (index < 13 || span_count)) ok = 0;
    }
    if (!ok) {
        fail(error, error_size, "cannot allocate Qwen activations: %s",
             h3_gpu_error(gpu));
        goto cleanup;
    }

    h3_gpu_tensor *embedding_weight = load_2d(
        &load, "model.language_model.embed_tokens.weight", TEXT_VOCAB,
        S->hidden);
    if (!embedding_weight) goto cleanup;
    if (!gpu_operation(gpu, h3_gpu_begin(gpu), error, error_size,
                       "command stream begin", -1) ||
        !gpu_operation(gpu, h3_gpu_embedding_bf16(
                                 gpu, hidden, embedding_weight, ids, tokens,
                                 TEXT_VOCAB, S->hidden),
                       error, error_size, "embedding lookup", -1) ||
        !gpu_operation(gpu, h3_gpu_submit(gpu), error, error_size,
                       "embedding stream submit", -1)) {
        goto cleanup;
    }
    retire_deferred(&load);
    for (size_t index = 0; index < span_count; index++) {
        const h3_text_vision_span *span = &spans[index];
        if (!h3_gpu_tensor_write_bf16_range(
                hidden, span->start * S->hidden, span->embeddings,
                span->tokens * S->hidden)) {
            fail(error, error_size, "cannot splice Qwen vision embeddings");
            goto cleanup;
        }
    }

    int prefetch_threads = text_prefetch_threads();
    int prefetch_layers = prefetch_threads > 0 && layer_count > 1;
    int prefetch_depth = prefetch_layers ? text_prefetch_depth(gpu) : 0;
    text_prefetch_slot slots[6];
    memset(slots, 0, sizeof(slots));
    text_layer_weights weights;
    memset(&weights, 0, sizeof(weights));
    if (!layer_weights_load(&load, 0, &weights)) goto cleanup;
    int next_prefetch_layer = 1;
    for (int index = 0;
         index < prefetch_depth && next_prefetch_layer < layer_count;
         index++, next_prefetch_layer++) {
        if (!prefetch_slot_start(&slots[index], store, &shape, gpu,
                                 next_prefetch_layer, prefetch_threads,
                                 error, error_size)) {
            prefetch_slots_retire(slots, prefetch_depth);
            goto cleanup;
        }
    }
    for (int layer = 0; layer < layer_count; layer++) {
        /* A connector tap below the last layer is that layer's input. */
        for (int tap = 0; connector && tap < CONNECTOR_TAPS - 1; tap++) {
            if (connector_taps[tap] != layer) continue;
            taps[tap] = malloc(hidden_count * sizeof(*taps[tap]));
            if (!taps[tap] ||
                !h3_gpu_tensor_read_bf16(hidden, taps[tap], hidden_count)) {
                fail(error, error_size, "cannot read Qwen hidden state %d",
                     layer);
                prefetch_slots_retire(slots, prefetch_depth);
                goto cleanup;
            }
        }
        int layer_ok =
            gpu_operation(gpu, h3_gpu_begin(gpu), error, error_size,
                          "layer stream begin", layer) &&
            encode_layer(gpu, S, &weights, tokens, hidden, norm, query, key,
                         value, attention_heads, attention_output, gate, up,
                         mlp_output, rope_cos, rope_sin, layer,
                         error, error_size) &&
            (!(layer < 3 && span_count) ||
             gpu_operation(gpu, h3_gpu_add_bf16(
                 gpu, hidden, hidden, deepstack[layer],
                 tokens * S->hidden), error, error_size,
                 "deepstack residual", layer)) &&
            gpu_operation(gpu, h3_gpu_submit(gpu), error, error_size,
                          "layer stream submit", layer);

        retire_deferred(&load);
        if (!layer_ok) {
            prefetch_slots_retire(slots, prefetch_depth);
            goto cleanup;
        }
        debug_hash_hidden(hidden, tokens * S->hidden, layer);
        if (progress) progress(layer + 1, (int)shape.layers, progress_opaque);
        if (layer + 1 >= layer_count) continue;

        memset(&weights, 0, sizeof(weights));
        if (prefetch_layers) {
            text_prefetch_slot *next = NULL;
            for (int index = 0; index < prefetch_depth; index++) {
                if (slots[index].occupied &&
                    slots[index].layer == layer + 1) {
                    next = &slots[index];
                    break;
                }
            }
            if (!next || !prefetch_slot_take(next, &load, &weights,
                                              error, error_size)) {
                if (!next)
                    fail(error, error_size,
                         "Qwen layer %d is absent from the prefetch ring",
                         layer + 1);
                prefetch_slots_retire(slots, prefetch_depth);
                goto cleanup;
            }
            if (next_prefetch_layer < layer_count) {
                if (!prefetch_slot_start(
                        next, store, &shape, gpu, next_prefetch_layer,
                        prefetch_threads, error, error_size)) {
                    prefetch_slots_retire(slots, prefetch_depth);
                    goto cleanup;
                }
                next_prefetch_layer++;
            }
        } else if (!layer_weights_load(&load, layer + 1, &weights)) {
            goto cleanup;
        }
    }

    if (projection) {
        uint16_t *last = malloc(hidden_count * sizeof(*last));
        output->values = malloc(token_count * projection->output_width *
                                sizeof(*output->values));
        if (!last || !output->values ||
            !h3_gpu_tensor_read_bf16(hidden, last, hidden_count)) {
            free(last);
            h3_text_embedding_free(output);
            fail(error, error_size, "cannot read Qwen hidden state %d",
                 layer_count);
            goto cleanup;
        }
        int projected = h3_text_projection_apply(
            projection, last, token_count, output->values, error, error_size);
        free(last);
        if (!projected || !h3_gpu_get_stats(gpu, &output->gpu_stats)) {
            h3_text_embedding_free(output);
            if (error && error_size && !error[0])
                fail(error, error_size, "cannot read Qwen GPU statistics");
            goto cleanup;
        }
        output->width = projection->output_width;
    } else if (connector) {
        /* The last tap: after the final norm. */
        h3_gpu_tensor *final_norm = load_1d(
            &load, "model.language_model.norm.weight", S->hidden);
        int tap = CONNECTOR_TAPS - 1;
        taps[tap] = malloc(hidden_count * sizeof(*taps[tap]));
        if (!final_norm || !taps[tap] ||
            !gpu_operation(gpu, h3_gpu_begin(gpu), error, error_size,
                           "final norm begin", -1) ||
            !gpu_operation(gpu, h3_gpu_rms_norm_bf16(
                               gpu, norm, hidden, final_norm, tokens,
                               S->hidden, TEXT_RMS_EPSILON),
                           error, error_size, "final RMSNorm", -1) ||
            !gpu_operation(gpu, h3_gpu_submit(gpu), error, error_size,
                           "final norm submit", -1) ||
            !h3_gpu_tensor_read_bf16(norm, taps[tap], hidden_count)) {
            if (error && error_size && !error[0])
                fail(error, error_size, "cannot read the Qwen final norm");
            goto cleanup;
        }
        retire_deferred(&load);
        output->values = malloc(token_count * CONNECTOR_WIDTH *
                                sizeof(*output->values));
        if (!output->values) {
            fail(error, error_size, "out of memory for the text connector");
            goto cleanup;
        }
        if (!apply_connector(store, S, taps, token_count, output->values,
                             error, error_size) ||
            !h3_gpu_get_stats(gpu, &output->gpu_stats)) {
            h3_text_embedding_free(output);
            if (error && error_size && !error[0])
                fail(error, error_size, "cannot read Qwen GPU statistics");
            goto cleanup;
        }
        output->width = CONNECTOR_WIDTH;
    } else {
        output->values = malloc(hidden_count * sizeof(*output->values));
        if (!output->values) {
            fail(error, error_size, "out of memory reading Qwen output");
            goto cleanup;
        }
        if (!h3_gpu_tensor_read_bf16(hidden, output->values, hidden_count) ||
            !h3_gpu_get_stats(gpu, &output->gpu_stats)) {
            h3_text_embedding_free(output);
            fail(error, error_size, "cannot read completed Qwen output");
            goto cleanup;
        }
        output->width = S->hidden;
    }
    output->tokens = token_count;

    if (tags) {
        output->tags = malloc(token_count * sizeof(*output->tags));
        if (!output->tags) {
            h3_text_embedding_free(output);
            fail(error, error_size, "out of memory reading Qwen tags");
            goto cleanup;
        }
        memcpy(output->tags, tags, token_count * sizeof(*output->tags));
    }
    ok = 1;
    goto finished;

cleanup:
    ok = 0;
finished:
    retire_deferred(&load);
    for (int tap = 0; tap < CONNECTOR_TAPS; tap++) free(taps[tap]);
    for (size_t index = 0; index < sizeof(activations) / sizeof(*activations);
         index++) {
        h3_gpu_tensor_free(activations[index]);
    }
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    h3_memory_scratch_drain();
    return ok;

early_cleanup:
    h3_gpu_tensor_free(ids);
    h3_gpu_tensor_free(rope_cos);
    h3_gpu_tensor_free(rope_sin);
    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(norm);
    h3_gpu_tensor_free(query);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(attention_heads);
    h3_gpu_tensor_free(attention_output);
    h3_gpu_tensor_free(gate);
    h3_gpu_tensor_free(up);
    h3_gpu_tensor_free(mlp_output);
    for (size_t index = 0; index < 3; index++)
        h3_gpu_tensor_free(deepstack[index]);
    retire_deferred(&load);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    h3_memory_scratch_drain();
    return 0;
}

int h3_text_encode_bf16(const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size) {
    return text_encode_bf16_impl(
        weight_directory, shader_source_path, token_ids, token_count,
        NULL, 0, NULL, NULL, TEXT_MAX_LAYERS, NULL, progress, progress_opaque,
        output, error, error_size);
}

int h3_text_encode_projected_bf16(const char *weight_directory,
                                  const char *projection_path,
                                  const char *shader_source_path,
                                  const uint32_t *token_ids, size_t token_count,
                                  h3_text_progress progress,
                                  void *progress_opaque,
                                  h3_text_embedding *output,
                                  char *error, size_t error_size) {
    h3_text_projection projection;
    if (output) memset(output, 0, sizeof(*output));
    if (!h3_text_projection_load(projection_path, &projection, error,
                                 error_size))
        return 0;
    int ok = text_encode_bf16_impl(
        weight_directory, shader_source_path, token_ids, token_count,
        NULL, 0, NULL, NULL, TEXT_MAX_LAYERS, &projection, progress,
        progress_opaque, output, error, error_size);
    h3_text_projection_free(&projection);
    return ok;
}

int h3_text_encode_multimodal_bf16(
                        const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        const h3_text_vision_span *spans, size_t span_count,
                        const uint32_t *position_ids, const uint8_t *tags,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size) {
    if (!spans || !span_count || !position_ids || !tags) {
        if (output) memset(output, 0, sizeof(*output));
        fail(error, error_size, "multimodal Qwen presentation is incomplete");
        return 0;
    }
    return text_encode_bf16_impl(
        weight_directory, shader_source_path, token_ids, token_count,
        spans, span_count, position_ids, tags, TEXT_MAX_LAYERS, NULL,
        progress, progress_opaque,
        output, error, error_size);
}

int h3_text_encode_multimodal_layers_bf16(
                        const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        const h3_text_vision_span *spans, size_t span_count,
                        const uint32_t *position_ids, const uint8_t *tags,
                        int layer_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size) {
    if (!spans || !span_count || !position_ids || !tags) {
        if (output) memset(output, 0, sizeof(*output));
        fail(error, error_size, "multimodal Qwen presentation is incomplete");
        return 0;
    }
    return text_encode_bf16_impl(
        weight_directory, shader_source_path, token_ids, token_count,
        spans, span_count, position_ids, tags, layer_count, NULL,
        progress, progress_opaque, output, error, error_size);
}
