/* The DiT's block weights: loading them resident (plain and HyperFlow's factored blocks), streaming them from the SSD or from the int8 attention cache, and quantizing them. Split out of h3_dit.c. */
#include "h3_dit_private.h"
/* refiner/index identify the block for LoRA lookup (token-refiner blocks
 * are separate targets from the 50 DiT blocks). */
int h3_dit_load_block(h3_dit *dit, h3_dit_block *block, const char *prefix,
                      int refiner, unsigned index, char *error,
                      size_t error_size) {
    char name[160];
#define LOAD1(field, suffix, width) do {                                       \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = h3_dit_bf1(dit, name, width, error, error_size);                    \
    if (!block->field) return 0;                                                \
} while (0)
#define LOAD2_LORA(field, projection, suffix, rows, columns) do {             \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = h3_dit_bf2_patched(dit, name, rows, columns, refiner, index,       \
                               projection, error, error_size);                 \
    if (!block->field) return 0;                                                \
} while (0)
    LOAD1(norm1, "norm1.weight", HIDDEN);
    LOAD1(norm2, "norm2.weight", HIDDEN);
    LOAD2_LORA(qkv, H3_LORA_QKV, "attn.qkv_proj.weight", INNER * 3, HIDDEN);
    LOAD1(q_norm, "attn.q_norm.weight", HEAD_DIM);
    LOAD1(k_norm, "attn.k_norm.weight", HEAD_DIM);
    LOAD2_LORA(out, H3_LORA_OUT, "attn.out_proj.weight", HIDDEN, INNER);
    LOAD2_LORA(fc1, H3_LORA_FC1, "mlp.fc1.weight", FFN * 2, HIDDEN);
    LOAD2_LORA(fc2, H3_LORA_FC2, "mlp.fc2.weight", HIDDEN, FFN);
#undef LOAD1
#undef LOAD2_LORA
    return 1;
}

int h3_dit_load_block_norms(h3_dit *dit, h3_dit_block *block,
                            const char *prefix,
                            char *error, size_t error_size) {
    char name[160];
#define LOAD1(field, suffix, width) do {                                       \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = h3_dit_bf1(dit, name, width, error, error_size);                    \
    if (!block->field) return 0;                                                \
} while (0)
    LOAD1(norm1, "norm1.weight", HIDDEN);
    LOAD1(norm2, "norm2.weight", HIDDEN);
    LOAD1(q_norm, "attn.q_norm.weight", HEAD_DIM);
    LOAD1(k_norm, "attn.k_norm.weight", HEAD_DIM);
#undef LOAD1
    return 1;
}

/* Like h3_dit_load_block(), but for H3_ATTENTION_CACHE: QKV/attention-output come
 * from the streamed int8 cache instead (and get their LoRA delta as they
 * stream in), so only the norms and the resident MLP are loaded here. */
int h3_dit_load_block_norms_and_mlp(h3_dit *dit, h3_dit_block *block,
                                    const char *prefix, unsigned index,
                                    char *error, size_t error_size) {
    char name[160];
    const int refiner = 0;
#define LOAD1(field, suffix, width) do {                                       \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = h3_dit_bf1(dit, name, width, error, error_size);                    \
    if (!block->field) return 0;                                                \
} while (0)
#define LOAD2_LORA(field, projection, suffix, rows, columns) do {             \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = h3_dit_bf2_patched(dit, name, rows, columns, refiner, index,       \
                               projection, error, error_size);                 \
    if (!block->field) return 0;                                                \
} while (0)
    LOAD1(norm1, "norm1.weight", HIDDEN);
    LOAD1(norm2, "norm2.weight", HIDDEN);
    LOAD1(q_norm, "attn.q_norm.weight", HEAD_DIM);
    LOAD1(k_norm, "attn.k_norm.weight", HEAD_DIM);
    LOAD2_LORA(fc1, H3_LORA_FC1, "mlp.fc1.weight", FFN * 2, HIDDEN);
    LOAD2_LORA(fc2, H3_LORA_FC2, "mlp.fc2.weight", HIDDEN, FFN);
#undef LOAD1
#undef LOAD2_LORA
    return 1;
}

void h3_dit_free_block(h3_dit_block *block) {
    h3_dit_free_tensor(&block->norm1);
    h3_dit_free_tensor(&block->norm2);
    h3_dit_free_tensor(&block->qkv);
    h3_dit_free_tensor(&block->qkv_int8);
    h3_dit_free_tensor(&block->qkv_scales);
    h3_dit_free_tensor(&block->q_norm);
    h3_dit_free_tensor(&block->k_norm);
    h3_dit_free_tensor(&block->out);
    h3_dit_free_tensor(&block->out_int8);
    h3_dit_free_tensor(&block->out_scales);
    h3_dit_free_tensor(&block->fc1);
    h3_dit_free_tensor(&block->fc2);
    h3_dit_free_tensor(&block->fc1_int8);
    h3_dit_free_tensor(&block->fc1_scales);
    h3_dit_free_tensor(&block->fc2_int8);
    h3_dit_free_tensor(&block->fc2_scales);
    h3_dit_free_tensor(&block->q_down);
    h3_dit_free_tensor(&block->q_up);
    h3_dit_free_tensor(&block->k_down);
    h3_dit_free_tensor(&block->k_up);
    h3_dit_free_tensor(&block->v_down);
    h3_dit_free_tensor(&block->v_up);
    h3_dit_free_tensor(&block->out_down);
    h3_dit_free_tensor(&block->out_up);
    h3_dit_free_tensor(&block->fc1_down);
    h3_dit_free_tensor(&block->fc1_gate_up);
    h3_dit_free_tensor(&block->fc1_value_up);
    h3_dit_free_tensor(&block->fc1_gate);
    h3_dit_free_tensor(&block->fc1_value);
    h3_dit_free_tensor(&block->fc2_down);
    h3_dit_free_tensor(&block->fc2_up);
}

/* Splits a BF16 [2 * half_rows, columns] tensor into its two row halves
 * (FC1's SwiGLU gate rows first, then its up rows). */
static int split_rows_bf16(h3_dit *dit, h3_gpu_tensor *source,
                           uint64_t half_rows, uint64_t columns,
                           h3_gpu_tensor **first, h3_gpu_tensor **second,
                           const char *name, char *error,
                           size_t error_size) {
    size_t half = (size_t)half_rows * columns;
    *first = h3_gpu_tensor_new_bf16(dit->gpu, half);
    *second = h3_gpu_tensor_new_bf16(dit->gpu, half);
    if (!*first || !*second ||
        !h3_gpu_begin(dit->gpu) ||
        !h3_gpu_copy_bf16(dit->gpu, *first, 0, source, 0, half) ||
        !h3_gpu_copy_bf16(dit->gpu, *second, 0, source, half, half) ||
        !h3_gpu_submit(dit->gpu)) {
        h3_dit_fail(error, error_size, "cannot split %s: %s", name,
             h3_gpu_error(dit->gpu));
        h3_dit_free_tensor(first);
        h3_dit_free_tensor(second);
        return 0;
    }
    return 1;
}

/* A HyperFlow block (h3_hyperflow.h): the norms, then each linear dense or
 * as its down/up factors. No LoRA (load_dit refuses it for HyperFlow). */
int h3_dit_load_hyperflow_block(h3_dit *dit, unsigned index,
                                char *error, size_t error_size) {
    h3_dit_block *block = &dit->blocks[index];
    const uint32_t *rank = dit->hyperflow_info.rank[index];
    char prefix[64], name[160];
    snprintf(prefix, sizeof(prefix), "blocks.%u.", index);
    if (!h3_dit_load_block_norms(dit, block, prefix, error, error_size)) return 0;
    memcpy(block->rank, rank, sizeof(block->rank));
#define LOAD(field, suffix, rows, columns) do {                                \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = h3_dit_bf2(dit, name, rows, columns, error, error_size);           \
    if (!block->field) return 0;                                                \
} while (0)
    if (rank[H3_HF_Q]) {
        LOAD(q_down, "attn.q_proj.down.weight", rank[H3_HF_Q], HIDDEN);
        LOAD(q_up, "attn.q_proj.up.weight", INNER, rank[H3_HF_Q]);
        LOAD(k_down, "attn.k_proj.down.weight", rank[H3_HF_K], HIDDEN);
        LOAD(k_up, "attn.k_proj.up.weight", INNER, rank[H3_HF_K]);
        LOAD(v_down, "attn.v_proj.down.weight", rank[H3_HF_V], HIDDEN);
        LOAD(v_up, "attn.v_proj.up.weight", INNER, rank[H3_HF_V]);
    } else {
        LOAD(qkv, "attn.qkv_proj.weight", INNER * 3, HIDDEN);
    }
    if (rank[H3_HF_OUT]) {
        LOAD(out_down, "attn.out_proj.down.weight", rank[H3_HF_OUT], INNER);
        LOAD(out_up, "attn.out_proj.up.weight", HIDDEN, rank[H3_HF_OUT]);
    } else {
        LOAD(out, "attn.out_proj.weight", HIDDEN, INNER);
    }
    int factored_mlp = rank[H3_HF_FC1] || rank[H3_HF_FC2];
    h3_gpu_tensor *whole = NULL;
    if (rank[H3_HF_FC1]) {
        LOAD(fc1_down, "mlp.fc1.down.weight", rank[H3_HF_FC1], HIDDEN);
        snprintf(name, sizeof(name), "%smlp.fc1.up.weight", prefix);
        whole = h3_dit_bf2(dit, name, FFN * 2, rank[H3_HF_FC1], error, error_size);
        int ok = whole && split_rows_bf16(dit, whole, FFN, rank[H3_HF_FC1],
                                          &block->fc1_gate_up,
                                          &block->fc1_value_up, name,
                                          error, error_size);
        h3_dit_free_tensor(&whole);
        if (!ok) return 0;
    } else if (factored_mlp) {
        snprintf(name, sizeof(name), "%smlp.fc1.weight", prefix);
        whole = h3_dit_bf2(dit, name, FFN * 2, HIDDEN, error, error_size);
        int ok = whole && split_rows_bf16(dit, whole, FFN, HIDDEN,
                                          &block->fc1_gate, &block->fc1_value,
                                          name, error, error_size);
        h3_dit_free_tensor(&whole);
        if (!ok) return 0;
    } else {
        LOAD(fc1, "mlp.fc1.weight", FFN * 2, HIDDEN);
    }
    if (rank[H3_HF_FC2]) {
        LOAD(fc2_down, "mlp.fc2.down.weight", rank[H3_HF_FC2], FFN);
        LOAD(fc2_up, "mlp.fc2.up.weight", HIDDEN, rank[H3_HF_FC2]);
    } else {
        LOAD(fc2, "mlp.fc2.weight", HIDDEN, FFN);
    }
#undef LOAD
    return 1;
}

double h3_dit_stream_now(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int compare_stream_sources(const void *left, const void *right) {
    const h3_dit_stream_source *a = left;
    const h3_dit_stream_source *b = right;
    int path = strcmp(a->path, b->path);
    if (path) return path;
    if (a->file_offset < b->file_offset) return -1;
    return a->file_offset > b->file_offset;
}

/* rows x columns is the tensor's full shape; [row_begin, row_begin +
 * row_count) the rows this source streams (all of them, or one SwiGLU half
 * of an FC1 weight). */
static int prepare_stream_rows(h3_dit *dit, h3_dit_stream_source *source,
                               const char *name, uint64_t rows,
                               uint64_t columns, uint64_t row_begin,
                               uint64_t row_count, unsigned field,
                               char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(dit->weights, name, &header);
    if (!tensor) {
        h3_dit_fail(error, error_size, "required streaming weight is absent: %s",
             name);
        return 0;
    }
    if (h3_weight_is_comfy_int8(dit->weights, tensor)) {
        /* The int8 payload streams as it is - half a BF16 weight's bytes -
         * and expand_streamed restores it on the GPU. Each row carries its
         * own scale and rotation, so a row range (one SwiGLU half of FC1)
         * streams and restores like a whole weight. */
        uint32_t inner = 0, head_dim = 0, group = 0;
        if (!header || !row_count || row_begin > rows ||
            row_count > rows - row_begin ||
            tensor->shape[0] != rows || tensor->shape[1] != columns ||
            rows > UINT32_MAX || columns > UINT32_MAX ||
            tensor->data_end - tensor->data_begin != rows * columns) {
            h3_dit_fail(error, error_size,
                 "streaming weight has the wrong schema: %s", name);
            return 0;
        }
        h3_dit_free_tensor(&source->comfy_scales);
        if (!h3_weight_comfy_int8_scales(dit->weights, dit->gpu, tensor,
                                         row_begin, row_count, &group,
                                         &source->comfy_scales,
                                         error, error_size))
            return 0;
        if (group > 256) {
            h3_dit_fail(error, error_size,
                 "SSD streaming reads ConvRot groups up to 256 columns "
                 "(%s has %u)", name, group);
            return 0;
        }
        if (row_count == rows)
            h3_weight_qkv_reorder(dit->weights, name, tensor, &inner,
                                  &head_dim);
        source->path = header->path;
        source->file_offset = tensor->file_offset + row_begin * columns;
        source->elements = (size_t)(row_count * columns);
        source->field = field;
        source->quant = STREAM_COMFY_INT8;
        source->bytes = row_count * columns;
        source->qkv_inner = inner;
        source->qkv_head_dim = head_dim;
        source->comfy_group = group;
        source->comfy_rows = (uint32_t)row_count;
        dit->stream_quantized = 1;
        return 1;
    }
    h3_dit_free_tensor(&source->comfy_scales);
    h3_quant_type quant = (h3_quant_type)tensor->quant;
    int quantized = tensor->dtype != H3_DTYPE_BF16 && quant != H3_QUANT_NONE &&
                    quant != H3_QUANT_F32 && quant != H3_QUANT_BF16;
    uint32_t qkv_inner = 0, qkv_head_dim = 0;
    int whole = row_begin == 0 && row_count == rows;
    if (whole && h3_weight_qkv_reorder(dit->weights, name, tensor,
                                       &qkv_inner, &qkv_head_dim) &&
        !quantized && tensor->dtype == H3_DTYPE_BF16) {
        /* A BF16 [Q; K; V] weight streams through the staging buffer too,
         * so the expansion can reorder its rows. */
        quant = H3_QUANT_BF16;
        quantized = 1;
    }
    uint64_t bytes = quantized ? h3_quant_bytes(quant, rows * columns) :
                                 rows * columns * sizeof(uint16_t);
    uint64_t row_bytes = quantized ? h3_quant_bytes(quant, columns) :
                                     columns * sizeof(uint16_t);
    if (!header || (tensor->dtype != H3_DTYPE_BF16 && !quantized) ||
        tensor->ndim != 2 ||
        tensor->shape[0] != rows || tensor->shape[1] != columns ||
        rows > SIZE_MAX / columns || !bytes || !row_count ||
        row_begin + row_count > rows || row_bytes * rows != bytes ||
        bytes != tensor->data_end - tensor->data_begin) {
        h3_dit_fail(error, error_size, "streaming weight has the wrong schema: %s",
             name);
        return 0;
    }
    source->path = header->path;
    source->file_offset = tensor->file_offset + row_begin * row_bytes;
    source->elements = (size_t)(row_count * columns);
    bytes = row_count * row_bytes;
    source->field = field;
    source->quant = quantized ? (uint32_t)quant : 0;
    source->bytes = bytes;
    source->qkv_inner = qkv_inner;
    source->qkv_head_dim = qkv_head_dim;
    if (quantized) dit->stream_quantized = 1;
    return 1;
}

int h3_dit_prepare_stream_layer(h3_dit *dit, unsigned layer,
                                char *error, size_t error_size) {
    char name[160];
    h3_dit_stream_layer *stream = &dit->stream_layers[layer];
    static const uint32_t no_rank[H3_HF_LINEARS] = {0};
    const uint32_t *rank = dit->hyperflow ?
        dit->hyperflow_info.rank[layer] : no_rank;
    stream->count = 0;
#define ROWS(suffix_, rows_, columns_, begin_, count_, field_) do {          \
    snprintf(name, sizeof(name), "blocks.%u.%s", layer, suffix_);             \
    if (stream->count >= STREAM_MAX_SOURCES ||                                 \
        !prepare_stream_rows(dit, &stream->sources[stream->count], name,       \
                             rows_, columns_, begin_, count_, field_, error,   \
                             error_size))                                      \
        return 0;                                                               \
    stream->count++;                                                            \
} while (0)
#define SOURCE(suffix_, rows_, columns_, field_)                               \
    ROWS(suffix_, rows_, columns_, 0, rows_, field_)
    if (rank[H3_HF_Q]) {
        SOURCE("attn.q_proj.down.weight", rank[H3_HF_Q], HIDDEN, STREAM_Q_DOWN);
        SOURCE("attn.q_proj.up.weight", INNER, rank[H3_HF_Q], STREAM_Q_UP);
        SOURCE("attn.k_proj.down.weight", rank[H3_HF_K], HIDDEN, STREAM_K_DOWN);
        SOURCE("attn.k_proj.up.weight", INNER, rank[H3_HF_K], STREAM_K_UP);
        SOURCE("attn.v_proj.down.weight", rank[H3_HF_V], HIDDEN, STREAM_V_DOWN);
        SOURCE("attn.v_proj.up.weight", INNER, rank[H3_HF_V], STREAM_V_UP);
    } else {
        SOURCE("attn.qkv_proj.weight", INNER * 3, HIDDEN, STREAM_QKV);
    }
    if (rank[H3_HF_OUT]) {
        SOURCE("attn.out_proj.down.weight", rank[H3_HF_OUT], INNER,
               STREAM_OUT_DOWN);
        SOURCE("attn.out_proj.up.weight", HIDDEN, rank[H3_HF_OUT],
               STREAM_OUT_UP);
    } else {
        SOURCE("attn.out_proj.weight", HIDDEN, INNER, STREAM_OUT);
    }
    /* A factored MLP keeps FC1's SwiGLU halves apart (gate rows first);
     * each half is a contiguous row range of the checkpoint tensor. */
    if (rank[H3_HF_FC1]) {
        SOURCE("mlp.fc1.down.weight", rank[H3_HF_FC1], HIDDEN,
               STREAM_FC1_DOWN);
        ROWS("mlp.fc1.up.weight", FFN * 2, rank[H3_HF_FC1], 0, FFN,
             STREAM_FC1_GATE_UP);
        ROWS("mlp.fc1.up.weight", FFN * 2, rank[H3_HF_FC1], FFN, FFN,
             STREAM_FC1_VALUE_UP);
    } else if (rank[H3_HF_FC2]) {
        ROWS("mlp.fc1.weight", FFN * 2, HIDDEN, 0, FFN, STREAM_FC1_GATE);
        ROWS("mlp.fc1.weight", FFN * 2, HIDDEN, FFN, FFN, STREAM_FC1_VALUE);
    } else {
        SOURCE("mlp.fc1.weight", FFN * 2, HIDDEN, STREAM_FC1);
    }
    if (rank[H3_HF_FC2]) {
        SOURCE("mlp.fc2.down.weight", rank[H3_HF_FC2], FFN, STREAM_FC2_DOWN);
        SOURCE("mlp.fc2.up.weight", HIDDEN, rank[H3_HF_FC2], STREAM_FC2_UP);
    } else {
        SOURCE("mlp.fc2.weight", HIDDEN, FFN, STREAM_FC2);
    }
#undef SOURCE
#undef ROWS
    qsort(stream->sources, stream->count, sizeof(stream->sources[0]),
          compare_stream_sources);
    return 1;
}

/* The slot member a streamed field lands in. */
h3_gpu_tensor **h3_dit_stream_slot_field(h3_dit_block *slot, unsigned field) {
    switch (field) {
    case STREAM_QKV: return &slot->qkv;
    case STREAM_OUT: return &slot->out;
    case STREAM_FC1: return &slot->fc1;
    case STREAM_FC2: return &slot->fc2;
    case STREAM_Q_DOWN: return &slot->q_down;
    case STREAM_Q_UP: return &slot->q_up;
    case STREAM_K_DOWN: return &slot->k_down;
    case STREAM_K_UP: return &slot->k_up;
    case STREAM_V_DOWN: return &slot->v_down;
    case STREAM_V_UP: return &slot->v_up;
    case STREAM_OUT_DOWN: return &slot->out_down;
    case STREAM_OUT_UP: return &slot->out_up;
    case STREAM_FC1_DOWN: return &slot->fc1_down;
    case STREAM_FC1_GATE_UP: return &slot->fc1_gate_up;
    case STREAM_FC1_VALUE_UP: return &slot->fc1_value_up;
    case STREAM_FC1_GATE: return &slot->fc1_gate;
    case STREAM_FC1_VALUE: return &slot->fc1_value;
    case STREAM_FC2_DOWN: return &slot->fc2_down;
    case STREAM_FC2_UP: return &slot->fc2_up;
    default: return NULL;
    }
}

static size_t page_round(uint64_t bytes) {
    size_t page = h3_gpu_page_bytes();
    return (size_t)((bytes + page - 1) / page * page);
}

/* Two slots alternate between the block being computed and the one being
 * read. A block's matrices sit back to back (page-aligned) in its slot's
 * arenas, so each arena only has to hold the largest single block: HyperFlow
 * blocks mix dense and factored linears of different ranks, and separate
 * per-kind buffers would hold both kinds at once. Plain H3 blocks all have
 * the same four matrices. */
int h3_dit_allocate_stream_slots(h3_dit *dit, char *error,
                                 size_t error_size) {
    size_t arena = 0, raw_arena = 0;
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        if (!dit->block_active[block]) continue;
        const h3_dit_stream_layer *layer = &dit->stream_layers[block];
        size_t total = 0, raw_total = 0;
        for (unsigned index = 0; index < layer->count; index++) {
            const h3_dit_stream_source *source = &layer->sources[index];
            total += page_round((uint64_t)source->elements * sizeof(uint16_t));
            if (source->quant) raw_total += page_round(source->bytes);
        }
        if (total > arena) arena = total;
        if (raw_total > raw_arena) raw_arena = raw_total;
    }
    for (unsigned slot = 0; slot < 2; slot++) {
        dit->stream_arena[slot] =
            h3_gpu_tensor_new_bf16(dit->gpu, arena / sizeof(uint16_t));
        if (raw_arena)
            dit->stream_raw_arena[slot] = h3_gpu_tensor_new_i8(dit->gpu,
                                                               raw_arena);
        if (!dit->stream_arena[slot] ||
            (raw_arena && !dit->stream_raw_arena[slot])) {
            h3_dit_fail(error, error_size, "cannot allocate the SSD layer slots: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
        for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
            if (!dit->block_active[block]) continue;
            h3_dit_stream_layer *layer = &dit->stream_layers[block];
            const h3_dit_stream_layer *same = NULL;
            for (unsigned earlier = 0; earlier < block && !same; earlier++) {
                const h3_dit_stream_layer *other = &dit->stream_layers[earlier];
                if (!dit->block_active[earlier] || other->borrowed ||
                    other->count != layer->count) continue;
                int equal = 1;
                for (unsigned index = 0; index < layer->count; index++) {
                    const h3_dit_stream_source *a = &layer->sources[index];
                    const h3_dit_stream_source *b = &other->sources[index];
                    if (a->field != b->field || a->elements != b->elements ||
                        a->quant != b->quant || a->bytes != b->bytes)
                        equal = 0;
                }
                if (equal) same = other;
            }
            if (same) {
                memcpy(layer->target[slot], same->target[slot],
                       sizeof(layer->target[slot]));
                memcpy(layer->raw[slot], same->raw[slot],
                       sizeof(layer->raw[slot]));
                layer->borrowed = 1;
                continue;
            }
            size_t offset = 0, raw_offset = 0;
            for (unsigned index = 0; index < layer->count; index++) {
                const h3_dit_stream_source *source = &layer->sources[index];
                layer->target[slot][index] = h3_gpu_tensor_alias_bf16(
                    dit->stream_arena[slot], offset, source->elements);
                offset += page_round(
                    (uint64_t)source->elements * sizeof(uint16_t));
                if (source->quant) {
                    layer->raw[slot][index] = h3_gpu_tensor_alias_i8(
                        dit->stream_raw_arena[slot], raw_offset,
                        (size_t)source->bytes);
                    raw_offset += page_round(source->bytes);
                }
                if (!layer->target[slot][index] ||
                    (source->quant && !layer->raw[slot][index])) {
                    h3_dit_fail(error, error_size,
                         "cannot place block %u in the SSD slot", block);
                    return 0;
                }
            }
        }
    }
    fprintf(stderr, "h3: SSD stream slots 2 x (%.0f MiB BF16 + %.0f MiB raw)\n",
            (double)arena / 1048576.0, (double)raw_arena / 1048576.0);
    return 1;
}

void h3_dit_free_stream_slots(h3_dit *dit) {
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        h3_dit_stream_layer *layer = &dit->stream_layers[block];
        for (unsigned index = 0; index < STREAM_MAX_SOURCES; index++)
            h3_dit_free_tensor(&layer->sources[index].comfy_scales);
        for (unsigned slot = 0; slot < 2; slot++)
            for (unsigned index = 0; index < STREAM_MAX_SOURCES; index++) {
                if (layer->borrowed) {
                    layer->target[slot][index] = NULL;
                    layer->raw[slot][index] = NULL;
                } else {
                    h3_dit_free_tensor(&layer->target[slot][index]);
                    h3_dit_free_tensor(&layer->raw[slot][index]);
                }
            }
        layer->borrowed = 0;
    }
    for (unsigned slot = 0; slot < 2; slot++) {
        h3_dit_free_tensor(&dit->stream_arena[slot]);
        h3_dit_free_tensor(&dit->stream_raw_arena[slot]);
    }
}

typedef struct {
    const h3_dit_stream_source *source;
    h3_gpu_tensor *target;
    int ok;
    uint64_t bytes;
    char error[512];
} h3_dit_matrix_job;

static void *read_stream_matrix_thread(void *opaque) {
    h3_dit_matrix_job *job = opaque;
    job->error[0] = '\0';
    if (job->source->quant) {
        if (!job->target || !h3_gpu_tensor_stream_file_prefix_i8(
                job->target, job->source->path, job->source->file_offset,
                (size_t)job->source->bytes, job->error, sizeof(job->error))) {
            if (!job->error[0])
                snprintf(job->error, sizeof(job->error),
                         "invalid quantized streaming destination");
            job->ok = 0;
            return NULL;
        }
        job->ok = 1;
        job->bytes = job->source->bytes;
        return NULL;
    }
    if (!job->target || !h3_gpu_tensor_stream_file_prefix_bf16(
            job->target, job->source->path, job->source->file_offset,
            job->source->elements, job->error, sizeof(job->error))) {
        if (!job->error[0])
            snprintf(job->error, sizeof(job->error),
                     "invalid BF16 streaming destination");
        job->ok = 0;
        return NULL;
    }
    job->ok = 1;
    job->bytes = (uint64_t)job->source->elements * sizeof(uint16_t);
    return NULL;
}

/* The four matrices (QKV/OUT/FC1/FC2) target independent buffers and each
   open their own file descriptor, so they can be read concurrently. A
   single pread() per matrix only reaches queue depth 1, well under what
   Apple Silicon's NVMe controller can sustain - reading all four at once
   raises the outstanding request count instead. */
int h3_dit_read_stream_layer(h3_dit_stream_job *job) {
    h3_dit_stream_layer *layer = &job->dit->stream_layers[job->layer];
    double started = h3_dit_stream_now();

    h3_dit_matrix_job matrix_jobs[STREAM_MAX_SOURCES];
    pthread_t matrix_threads[STREAM_MAX_SOURCES];
    int started_thread[STREAM_MAX_SOURCES] = {0};

    for (unsigned index = 0; index < layer->count; index++) {
        matrix_jobs[index].source = &layer->sources[index];
        matrix_jobs[index].target = layer->sources[index].quant ?
            layer->raw[job->slot][index] : layer->target[job->slot][index];
        matrix_jobs[index].ok = 0;
        matrix_jobs[index].bytes = 0;
        if (pthread_create(&matrix_threads[index], NULL,
                            read_stream_matrix_thread,
                            &matrix_jobs[index]) == 0) {
            started_thread[index] = 1;
        } else {
            read_stream_matrix_thread(&matrix_jobs[index]);
        }
    }

    job->ok = 1;
    job->bytes = 0;
    job->error[0] = '\0';
    for (unsigned index = 0; index < layer->count; index++) {
        if (started_thread[index])
            pthread_join(matrix_threads[index], NULL);
        if (!matrix_jobs[index].ok) {
            job->ok = 0;
            if (!job->error[0])
                snprintf(job->error, sizeof(job->error), "%s",
                         matrix_jobs[index].error);
        }
        job->bytes += matrix_jobs[index].bytes;
    }
    job->seconds = h3_dit_stream_now() - started;
    return job->ok;
}

void *h3_dit_read_stream_layer_thread(void *opaque) {
    h3_dit_read_stream_layer(opaque);
    return NULL;
}

/* --- H3_ATTENTION_CACHE: streamed int8 QKV/OUT (+ optionally FC1/FC2) -- */

/* Every record holds all four matrices so the same cache file serves both
 * modes; attention-only mode simply never touches the FC1/FC2 portion. */
static uint64_t attention_cache_record_bytes(void) {
    return (uint64_t)INNER * 3 * HIDDEN            /* qkv_int8 */
         + (uint64_t)INNER * 3 * sizeof(float)      /* qkv_scales */
         + (uint64_t)HIDDEN * INNER                 /* out_int8 */
         + (uint64_t)HIDDEN * sizeof(float)          /* out_scales */
         + (uint64_t)FFN * 2 * HIDDEN                /* fc1_int8 */
         + (uint64_t)FFN * 2 * sizeof(float)          /* fc1_scales */
         + (uint64_t)HIDDEN * FFN                    /* fc2_int8 */
         + (uint64_t)HIDDEN * sizeof(float);          /* fc2_scales */
}

typedef struct {
    uint64_t qkv_int8, qkv_scales, out_int8, out_scales;
    uint64_t fc1_int8, fc1_scales, fc2_int8, fc2_scales;
} h3_attention_cache_offsets;

static void attention_cache_offsets(unsigned block,
                                    h3_attention_cache_offsets *off) {
    uint64_t base = (uint64_t)sizeof(h3_attention_cache_header) +
        (uint64_t)block * attention_cache_record_bytes();
    off->qkv_int8 = base;
    off->qkv_scales = off->qkv_int8 + (uint64_t)INNER * 3 * HIDDEN;
    off->out_int8 = off->qkv_scales + (uint64_t)INNER * 3 * sizeof(float);
    off->out_scales = off->out_int8 + (uint64_t)HIDDEN * INNER;
    off->fc1_int8 = off->out_scales + (uint64_t)HIDDEN * sizeof(float);
    off->fc1_scales = off->fc1_int8 + (uint64_t)FFN * 2 * HIDDEN;
    off->fc2_int8 = off->fc1_scales + (uint64_t)FFN * 2 * sizeof(float);
    off->fc2_scales = off->fc2_int8 + (uint64_t)HIDDEN * FFN;
}

/* Infers FL2VA vs Ref2VA from weight_directory's own last two path
 * components - h3.c's dit_path selection (h3.c: h3_path(ctx->model_dir,
 * ref2va ? "Ref2VA/transformer" : "FL2VA/transformer")) always ends the
 * directory this way, so this is reading back a convention h3.c's own
 * caller wrote, not guessing at one. Any other layout (test fixtures, a
 * directory pointed at directly with a different name) reports UNKNOWN,
 * which disables the model_kind check entirely rather than risk a false
 * mismatch. */
h3_cache_model_kind h3_dit_detect_model_kind(const char *weight_directory) {
    if (!weight_directory) return H3_CACHE_MODEL_UNKNOWN;
    size_t length = strlen(weight_directory);
    static const char fl2va_suffix[] = "FL2VA/transformer";
    static const char ref2va_suffix[] = "Ref2VA/transformer";
    if (length >= sizeof(ref2va_suffix) - 1 &&
        !strcmp(weight_directory + length - (sizeof(ref2va_suffix) - 1),
                ref2va_suffix))
        return H3_CACHE_MODEL_REF2VA;
    if (length >= sizeof(fl2va_suffix) - 1 &&
        !strcmp(weight_directory + length - (sizeof(fl2va_suffix) - 1),
                fl2va_suffix))
        return H3_CACHE_MODEL_FL2VA;
    return H3_CACHE_MODEL_UNKNOWN;
}

static const char *cache_model_kind_name(h3_cache_model_kind kind) {
    switch (kind) {
        case H3_CACHE_MODEL_FL2VA: return "FL2VA";
        case H3_CACHE_MODEL_REF2VA: return "Ref2VA";
        default: return "an unrecognized model layout";
    }
}

/* expected_kind is H3_CACHE_MODEL_UNKNOWN when the checkpoint directory
 * about to be loaded does not match the standard FL2VA/Ref2VA layout (see
 * h3_dit_detect_model_kind()) - the model_kind check is then skipped entirely,
 * same as a cache whose own model_kind is UNKNOWN (written by a tool that
 * predates this field, or given a non-standard transformer directory
 * itself). expected_model_id may be NULL to skip the fingerprint check;
 * legacy_model_id (may be NULL) is the path-based fingerprint caches were
 * written with before 2026-10-05 (h3_weight_store_fingerprint_legacy()),
 * also accepted so those caches keep validating without a rebuild while
 * the model stays at the path they were built from. */
int h3_dit_attention_cache_validate(const char *path, int need_mlp,
                                    h3_cache_model_kind expected_kind,
                                    const uint8_t *expected_model_id,
                                    const uint8_t *legacy_model_id,
                                    char *error, size_t error_size) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        h3_dit_fail(error, error_size, "cannot open attention cache %s: %s", path,
             strerror(errno));
        return 0;
    }
    h3_attention_cache_header header;
    int ok = fread(&header, sizeof(header), 1, file) == 1;
    if (ok) {
        ok = memcmp(header.magic, H3_ATTENTION_CACHE_MAGIC, 4) == 0 &&
             header.version == H3_ATTENTION_CACHE_VERSION &&
             header.block_count == H3_DIT_BLOCKS &&
             header.hidden == HIDDEN && header.inner == INNER &&
             header.ffn == FFN;
    }
    if (ok) {
        if (fseeko(file, 0, SEEK_END) != 0) {
            ok = 0;
        } else {
            off_t size = ftello(file);
            uint64_t wanted = (uint64_t)sizeof(header) +
                (uint64_t)H3_DIT_BLOCKS * attention_cache_record_bytes();
            ok = size >= 0 && (uint64_t)size >= wanted;
        }
    }
    fclose(file);
    (void)need_mlp; /* every v3 cache carries FC1/FC2 too; kept for clarity */
    if (!ok) {
        h3_dit_fail(error, error_size,
             "attention cache %s does not match this build (wrong model, "
             "quantization version, or a truncated file) - rebuild it with "
             "build_attention_cache", path);
        return 0;
    }
    if (expected_kind != H3_CACHE_MODEL_UNKNOWN &&
        header.model_kind != H3_CACHE_MODEL_UNKNOWN &&
        header.model_kind != (uint32_t)expected_kind) {
        h3_dit_fail(error, error_size,
             "%s generation cannot use a %s attention cache (%s) - rebuild "
             "it against the matching transformer directory",
             cache_model_kind_name(expected_kind),
             cache_model_kind_name((h3_cache_model_kind)header.model_kind),
             path);
        return 0;
    }
    if (expected_model_id &&
        memcmp(header.model_id, (const uint8_t[32]){0}, 32) != 0 &&
        memcmp(header.model_id, expected_model_id, 32) != 0 &&
        !(legacy_model_id &&
          memcmp(header.model_id, legacy_model_id, 32) == 0)) {
        /* A validation/benchmark run comparing against real captured
         * activations must not silently proceed on a stale cache - set
         * H3_ATTENTION_CACHE_STRICT=1 to turn this into a hard failure
         * instead of a warning (found the hard way: a stale cache here
         * once produced a garbage-looking kernel comparison that was
         * actually just wrong quantized weights, not a real bug). */
        if (h3_option("H3_ATTENTION_CACHE_STRICT")) {
            h3_dit_fail(error, error_size,
                 "attention cache %s's model fingerprint does not match "
                 "the loaded checkpoint (different weights, or weights "
                 "rewritten since) - rebuild it with build_attention_cache "
                 "(H3_ATTENTION_CACHE_STRICT is set, refusing to proceed)",
                 path);
            return 0;
        }
        fprintf(stderr,
                "h3: warning: attention cache %s's model fingerprint does "
                "not match the loaded checkpoint (different weights, or "
                "weights rewritten since) - results may be wrong; "
                "rebuild the cache if unsure\n", path);
    }
    return 1;
}

int h3_dit_allocate_attention_slot(h3_dit *dit, h3_dit_attention_slot *slot,
                                   char *error, size_t error_size) {
    slot->qkv_int8 = h3_gpu_tensor_new_i8(dit->gpu, (size_t)INNER * 3 * HIDDEN);
    slot->qkv_scales = h3_gpu_tensor_new_f32(dit->gpu, INNER * 3);
    slot->out_int8 = h3_gpu_tensor_new_i8(dit->gpu, (size_t)HIDDEN * INNER);
    slot->out_scales = h3_gpu_tensor_new_f32(dit->gpu, HIDDEN);
    int ok = slot->qkv_int8 && slot->qkv_scales && slot->out_int8 &&
             slot->out_scales;
    if (ok && dit->mlp_stream) {
        slot->fc1_int8 = h3_gpu_tensor_new_i8(dit->gpu, (size_t)FFN * 2 * HIDDEN);
        slot->fc1_scales = h3_gpu_tensor_new_f32(dit->gpu, FFN * 2);
        slot->fc2_int8 = h3_gpu_tensor_new_i8(dit->gpu, (size_t)HIDDEN * FFN);
        slot->fc2_scales = h3_gpu_tensor_new_f32(dit->gpu, HIDDEN);
        ok = slot->fc1_int8 && slot->fc1_scales && slot->fc2_int8 &&
             slot->fc2_scales;
    }
    if (!ok) {
        h3_dit_fail(error, error_size, "cannot allocate int8 attention slot: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

void h3_dit_free_attention_slot(h3_dit_attention_slot *slot) {
    h3_dit_free_tensor(&slot->qkv_int8);
    h3_dit_free_tensor(&slot->qkv_scales);
    h3_dit_free_tensor(&slot->out_int8);
    h3_dit_free_tensor(&slot->out_scales);
    h3_dit_free_tensor(&slot->fc1_int8);
    h3_dit_free_tensor(&slot->fc1_scales);
    h3_dit_free_tensor(&slot->fc2_int8);
    h3_dit_free_tensor(&slot->fc2_scales);
}

int h3_dit_read_attention_layer(h3_dit_attention_job *job) {
    h3_dit_attention_slot *slot = &job->dit->attention_slots[job->slot];
    const char *path = job->dit->attention_cache_path;
    h3_attention_cache_offsets off;
    attention_cache_offsets(job->layer, &off);
    double started = h3_dit_stream_now();
    job->ok =
        h3_gpu_tensor_stream_file_i8(slot->qkv_int8, path, off.qkv_int8,
                                     (size_t)INNER * 3 * HIDDEN,
                                     job->error, sizeof(job->error)) &&
        h3_gpu_tensor_stream_file_f32(slot->qkv_scales, path, off.qkv_scales,
                                      INNER * 3,
                                      job->error, sizeof(job->error)) &&
        h3_gpu_tensor_stream_file_i8(slot->out_int8, path, off.out_int8,
                                     (size_t)HIDDEN * INNER,
                                     job->error, sizeof(job->error)) &&
        h3_gpu_tensor_stream_file_f32(slot->out_scales, path, off.out_scales,
                                      HIDDEN,
                                      job->error, sizeof(job->error));
    uint64_t bytes = job->ok ?
        (uint64_t)INNER * 3 * HIDDEN + (uint64_t)INNER * 3 * sizeof(float) +
        (uint64_t)HIDDEN * INNER + (uint64_t)HIDDEN * sizeof(float) : 0;
    if (job->ok && job->dit->mlp_stream) {
        job->ok =
            h3_gpu_tensor_stream_file_i8(slot->fc1_int8, path, off.fc1_int8,
                                         (size_t)FFN * 2 * HIDDEN,
                                         job->error, sizeof(job->error)) &&
            h3_gpu_tensor_stream_file_f32(slot->fc1_scales, path,
                                          off.fc1_scales, FFN * 2,
                                          job->error, sizeof(job->error)) &&
            h3_gpu_tensor_stream_file_i8(slot->fc2_int8, path, off.fc2_int8,
                                         (size_t)HIDDEN * FFN,
                                         job->error, sizeof(job->error)) &&
            h3_gpu_tensor_stream_file_f32(slot->fc2_scales, path,
                                          off.fc2_scales, HIDDEN,
                                          job->error, sizeof(job->error));
        if (job->ok)
            bytes += (uint64_t)FFN * 2 * HIDDEN +
                (uint64_t)FFN * 2 * sizeof(float) +
                (uint64_t)HIDDEN * FFN + (uint64_t)HIDDEN * sizeof(float);
    }
    job->bytes = job->ok ? bytes : 0;
    job->seconds = h3_dit_stream_now() - started;
    return job->ok;
}

void *h3_dit_read_attention_layer_thread(void *opaque) {
    h3_dit_read_attention_layer(opaque);
    return NULL;
}

int h3_dit_quantize_block_mlp(h3_dit *dit, h3_dit_block *block,
                              char *error, size_t error_size) {
    block->fc1_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)FFN * 2 * HIDDEN);
    block->fc1_scales = h3_gpu_tensor_new_f32(dit->gpu, FFN * 2);
    block->fc2_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)HIDDEN * FFN);
    block->fc2_scales = h3_gpu_tensor_new_f32(dit->gpu, HIDDEN);
    int ok = block->fc1_int8 && block->fc1_scales &&
             block->fc2_int8 && block->fc2_scales &&
             h3_gpu_begin(dit->gpu) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->fc1_int8, block->fc1_scales, block->fc1,
                 FFN * 2, HIDDEN) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->fc2_int8, block->fc2_scales, block->fc2,
                 HIDDEN, FFN) &&
             h3_gpu_submit(dit->gpu);
    if (!ok) {
        h3_dit_fail(error, error_size, "cannot quantize DiT MLP weights: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!dit->keep_bf16_mlp) {
        h3_dit_free_tensor(&block->fc1);
        h3_dit_free_tensor(&block->fc2);
    }
    return 1;
}

int h3_dit_quantize_block_qkv(h3_dit *dit, h3_dit_block *block,
                              char *error, size_t error_size) {
    block->qkv_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)INNER * 3 * HIDDEN);
    block->qkv_scales = h3_gpu_tensor_new_f32(dit->gpu, INNER * 3);
    int ok = block->qkv_int8 && block->qkv_scales &&
             h3_gpu_begin(dit->gpu) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->qkv_int8, block->qkv_scales, block->qkv,
                 INNER * 3, HIDDEN) &&
             h3_gpu_submit(dit->gpu);
    if (!ok) {
        h3_dit_fail(error, error_size, "cannot quantize DiT QKV weight: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!dit->keep_bf16_qkv) h3_dit_free_tensor(&block->qkv);
    return 1;
}

int h3_dit_quantize_block_attention_out(h3_dit *dit, h3_dit_block *block,
                                        char *error, size_t error_size) {
    block->out_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)HIDDEN * INNER);
    block->out_scales = h3_gpu_tensor_new_f32(dit->gpu, HIDDEN);
    int ok = block->out_int8 && block->out_scales &&
             h3_gpu_begin(dit->gpu) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->out_int8, block->out_scales, block->out,
                 HIDDEN, INNER) &&
             h3_gpu_submit(dit->gpu);
    if (!ok) {
        h3_dit_fail(error, error_size,
             "cannot quantize DiT attention-output weight: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!dit->keep_bf16_attention_out) h3_dit_free_tensor(&block->out);
    return 1;
}
