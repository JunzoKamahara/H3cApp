/* One forward pass of the DiT: token reduction, a block (attention, MLP, AdaLN in their BF16, int8 and factored forms) and the pass over all blocks. Split out of h3_dit.c. */
#include "h3_dit_private.h"
static int enter_token_reduction(h3_dit *dit, char *error,
                                 size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    if (!h3_dit_gpu_op(dit, h3_gpu_token_pool_bf16(
            dit->gpu, dit->attention_output, dit->hidden, 0,
            original, dit->token_original_offset, dit->attention_output,
            dit->token_baseline_offset, dit->token_baseline_indices,
            dit->token_pool_pairs, dit->sequence, dit->reduced_sequence,
            dit->token_baseline_rows, HIDDEN),
            error, error_size, "snapshot and pool video tokens")) return 0;
    h3_gpu_tensor *swap = dit->hidden;
    dit->hidden = dit->attention_output;
    dit->attention_output = swap;
    dit->token_reduction_active = 1;
    return 1;
}

static int enter_token_reduction_adaln(h3_dit *dit, unsigned block,
                                       int step, char *error,
                                       size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    h3_dit_block *weight = &dit->blocks[block];
    const h3_gpu_tensor *modulation = h3_dit_schedule_block(
        dit->schedule, block);
    if (!h3_dit_gpu_op(dit, h3_gpu_token_pool_adaln_bf16(
            dit->gpu, dit->attention_output, dit->mod_attention,
            dit->hidden, 0, original, dit->token_original_offset,
            dit->attention_output, dit->token_baseline_offset,
            dit->token_baseline_indices, dit->token_pool_pairs,
            weight->norm1, modulation, dit->reduced_row_maps[step],
            dit->sequence, dit->reduced_sequence, dit->token_baseline_rows,
            HIDDEN, SLOTS, 0, 1, 1e-5f), error, error_size,
            "snapshot, pool, and apply attention AdaLN")) return 0;
    h3_gpu_tensor *swap = dit->hidden;
    dit->hidden = dit->attention_output;
    dit->attention_output = swap;
    dit->token_reduction_active = 1;
    return 1;
}

static int leave_token_reduction(h3_dit *dit, char *error,
                                 size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    if (!h3_dit_gpu_op(dit, h3_gpu_token_expand_delta_bf16(
            dit->gpu, dit->mod_attention, original,
            dit->token_original_offset, dit->hidden, dit->hidden,
            dit->token_baseline_offset, dit->token_baseline_indices,
            dit->token_expand_parents, dit->sequence,
            dit->reduced_sequence, dit->token_baseline_rows, HIDDEN,
            dit->video_target_start,
            dit->token_reduction_scale), error, error_size,
            "restore full video-token grid")) return 0;
    h3_gpu_tensor *swap = dit->hidden;
    dit->hidden = dit->mod_attention;
    dit->mod_attention = swap;
    dit->token_reduction_active = 0;
    return 1;
}

static int leave_token_reduction_adaln(h3_dit *dit, unsigned block,
                                       int step, char *error,
                                       size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    h3_dit_block *weight = &dit->blocks[block];
    const h3_gpu_tensor *modulation = h3_dit_schedule_block(
        dit->schedule, block);
    if (!h3_dit_gpu_op(dit, h3_gpu_token_expand_adaln_bf16(
            dit->gpu, dit->attention_output, dit->mod_attention,
            original, dit->token_original_offset, dit->hidden, dit->hidden,
            dit->token_baseline_offset, dit->token_baseline_indices,
            dit->token_expand_parents, weight->norm1, modulation,
            dit->row_maps[step], dit->sequence, dit->reduced_sequence,
            dit->token_baseline_rows, HIDDEN, dit->video_target_start,
            dit->token_reduction_scale, SLOTS, 0, 1, 1e-5f),
            error, error_size, "restore tokens and apply attention AdaLN"))
        return 0;
    h3_gpu_tensor *reduced = dit->hidden;
    dit->hidden = dit->attention_output;
    dit->attention_output = reduced;
    dit->token_reduction_active = 0;
    return 1;
}

/* A block-quantized stream (GGUF checkpoint) read raw bytes into the
 * slot's staging buffers; expand them into the slot's BF16 matrices ahead
 * of the block's own work, in the same command. */
static int expand_streamed(h3_dit *dit, unsigned block, unsigned slot_index,
                           char *error, size_t error_size) {
    if (!dit->stream_quantized) return 1;
    const h3_dit_stream_layer *layer = &dit->stream_layers[block];
    for (unsigned index = 0; index < layer->count; index++) {
        const h3_dit_stream_source *source = &layer->sources[index];
        if (!source->quant) continue;
        if (source->quant == STREAM_COMFY_INT8) {
            if (!h3_dit_gpu_op(dit, h3_gpu_comfy_int8_expand(
                    dit->gpu, layer->target[slot_index][index],
                    layer->raw[slot_index][index], source->comfy_scales,
                    source->comfy_rows,
                    (uint32_t)(source->elements / source->comfy_rows),
                    source->comfy_group, source->qkv_inner,
                    source->qkv_head_dim),
                    error, error_size, "expand int8 DiT block"))
                return 0;
            continue;
        }
        h3_quant_type quant = (h3_quant_type)source->quant;
        uint32_t per_block = h3_quant_block_elements(quant);
        if (!h3_dit_gpu_op(dit, h3_gpu_dequantize(
                dit->gpu, layer->target[slot_index][index],
                layer->raw[slot_index][index], source->quant,
                (uint32_t)(source->elements / per_block), per_block,
                h3_quant_block_bytes(quant),
                (uint32_t)(source->elements / (source->qkv_inner ?
                                              3u * source->qkv_inner : 1u)),
                source->qkv_inner, source->qkv_head_dim),
                error, error_size, "expand quantized DiT block"))
            return 0;
    }
    return 1;
}

/* Streamed slots are refilled straight from the checkpoint or int8 cache
 * every time a block streams in, so the LoRA delta is re-applied each time
 * (encoded ahead of the block's own work in the same command). */
static int patch_streamed_bf16(h3_dit *dit, unsigned block,
                               h3_dit_block *weight, char *error,
                               size_t error_size) {
    if (!dit->loras) return 1;
    h3_gpu_tensor *targets[H3_LORA_PROJECTIONS] = {
        weight->qkv, weight->out, weight->fc1, weight->fc2};
    for (int projection = 0; projection < H3_LORA_PROJECTIONS; projection++)
        if (!h3_dit_gpu_op(dit, h3_lora_set_apply_bf16(dit->loras, dit->gpu, 0,
                                                block, projection,
                                                targets[projection]),
                    error, error_size, "apply LoRA to streamed block"))
            return 0;
    return 1;
}

static int patch_streamed_int8(h3_dit *dit, unsigned block,
                               h3_dit_block *weight, char *error,
                               size_t error_size) {
    if (!dit->loras) return 1;
    struct { h3_gpu_tensor *weight, *scales; } targets[H3_LORA_PROJECTIONS] = {
        {weight->qkv_int8, weight->qkv_scales},
        {weight->out_int8, weight->out_scales},
        {dit->mlp_stream ? weight->fc1_int8 : NULL, weight->fc1_scales},
        {dit->mlp_stream ? weight->fc2_int8 : NULL, weight->fc2_scales},
    };
    for (int projection = 0; projection < H3_LORA_PROJECTIONS; projection++) {
        /* A resident MLP was already patched when it loaded. */
        if (!targets[projection].weight) continue;
        if (!h3_dit_gpu_op(dit, h3_lora_set_apply_int8(
                        dit->loras, dit->gpu, 0, block, projection,
                        targets[projection].weight,
                        targets[projection].scales),
                    error, error_size, "apply LoRA to streamed int8 block"))
            return 0;
    }
    return 1;
}

/* Debug-only: converts one BF16 element to FP16 (round-to-nearest, flush
 * subnormals to zero - values here are RMSNorm/RoPE-scaled attention
 * inputs, never subnormal in practice). Not used on any production path. */
static uint16_t debug_bf16_to_f16(uint16_t bf16_bits) {
    uint32_t bits = (uint32_t)bf16_bits << 16;
    float value;
    memcpy(&value, &bits, sizeof(value));
    uint32_t f32; memcpy(&f32, &value, 4);
    uint32_t sign = (f32 >> 16) & 0x8000u;
    int32_t exponent = (int32_t)((f32 >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = f32 & 0x7FFFFFu;
    if (exponent <= 0) return (uint16_t)sign;
    if (exponent >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    uint32_t rounded = mantissa + 0x00001000u;
    if (rounded & 0x00800000u) {
        rounded = 0;
        exponent++;
        if (exponent >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    }
    return (uint16_t)(sign | ((uint32_t)exponent << 10) | (rounded >> 13));
}

/* Debug-only: records shape/dtype and, critically, the SOURCE layout the
 * tensor actually had on the GPU versus the layout it was WRITTEN in
 * (input and output layouts can differ independently - see the head-major
 * bug this was added to catch, documented in plans/SPEEDUP_ROADMAP.md). Appends
 * one line per tensor to <prefix>.meta.txt so a later comparison never has
 * to assume a layout from context alone. */
static void debug_append_attention_meta(const char *prefix, int truncate,
                                        const char *tensor, uint32_t rows,
                                        const char *source_layout,
                                        const char *written_layout,
                                        unsigned index, int step) {
    char path[1024];
    snprintf(path, sizeof(path), "%s.meta.txt", prefix);
    FILE *file = fopen(path, truncate ? "w" : "a");
    if (!file) return;
    fprintf(file,
            "tensor=%s rows=%u heads=%d head_dim=%d dtype=fp16 "
            "source_layout=%s written_layout=%s block=%u step=%d\n",
            tensor, rows, HEADS, HEAD_DIM, source_layout, written_layout,
            index, step);
    fclose(file);
}

/* Debug-only: dumps this block's Q/K/V (right after QKV projection/RoPE,
 * the exact inputs the production SDPA call below consumes) as contiguous
 * FP16 [1, rows, HEADS, HEAD_DIM] - the raw format
 * bin/mfa/na_int8_sol_attention_bench.cpp in a liuliu/ccv checkout reads
 * for its `raw-prefix` argument, so a captured block can be replayed
 * through both the app's production SDPA and ccv's Sol Attention kernel
 * for a same-input comparison. Gated entirely behind three env vars,
 * off by default, never touches the return-value error path (a failed
 * dump only prints a warning, since it's not part of generation). */
static void debug_dump_attention_qkv(h3_dit *dit, unsigned index, int step,
                                     uint32_t rows) {
    const char *prefix = h3_option("H3_DUMP_ATTENTION_QKV");
    if (!prefix || !*prefix) return;
    const char *block_text = h3_option("H3_DUMP_ATTENTION_BLOCK");
    const char *step_text = h3_option("H3_DUMP_ATTENTION_STEP");
    unsigned want_block = block_text ? (unsigned)atoi(block_text) : 0;
    int want_step = step_text ? atoi(step_text) : 0;
    if (index != want_block || step != want_step) return;
    /* The preceding QKV projection may have left query/key/value in
     * [heads, rows, head_dim] layout for the SDPA call right after this
     * dump to consume (an internal producer/consumer optimization) -
     * read this now, before anything else touches it, so the capture can
     * be transposed to the [rows, heads, head_dim] layout raw-prefix
     * consumers (e.g. liuliu/ccv's benchmarks) expect. */
    int head_major = h3_gpu_head_major_sdpa_inputs(dit->gpu);
    if (!h3_dit_gpu_op(dit, h3_gpu_submit(dit->gpu), NULL, 0,
                "flush for attention QKV dump") ||
        !h3_dit_gpu_op(dit, h3_gpu_begin(dit->gpu), NULL, 0,
                "resume after attention QKV dump")) {
        fprintf(stderr, "h3: warning: could not flush GPU for "
                "H3_DUMP_ATTENTION_QKV\n");
        return;
    }
    size_t count = (size_t)rows * HEADS * HEAD_DIM;
    uint16_t *bf16 = malloc(count * sizeof(*bf16));
    uint16_t *f16 = malloc(count * sizeof(*f16));
    h3_gpu_tensor *tensors[3] = { dit->query, dit->key, dit->value };
    const char *suffix[3] = { "q", "k", "v" };
    if (!bf16 || !f16) {
        fprintf(stderr, "h3: warning: out of memory for "
                "H3_DUMP_ATTENTION_QKV\n");
        free(bf16); free(f16);
        return;
    }
    for (int which = 0; which < 3; which++) {
        if (!h3_gpu_tensor_read_bf16(tensors[which], bf16, count)) {
            fprintf(stderr, "h3: warning: could not read %s for "
                    "H3_DUMP_ATTENTION_QKV\n", suffix[which]);
            continue;
        }
        for (size_t row = 0; row < rows; row++) {
            for (int head = 0; head < HEADS; head++) {
                size_t dst_base = (row * HEADS + (size_t)head) * HEAD_DIM;
                size_t src_base = head_major
                    ? ((size_t)head * rows + row) * HEAD_DIM : dst_base;
                for (int d = 0; d < HEAD_DIM; d++)
                    f16[dst_base + (size_t)d] =
                        debug_bf16_to_f16(bf16[src_base + (size_t)d]);
            }
        }
        char path[1024];
        snprintf(path, sizeof(path), "%s.%s.bin", prefix, suffix[which]);
        FILE *file = fopen(path, "wb");
        if (!file) {
            fprintf(stderr, "h3: warning: cannot open %s for "
                    "H3_DUMP_ATTENTION_QKV\n", path);
            continue;
        }
        fwrite(f16, sizeof(*f16), count, file);
        fclose(file);
        debug_append_attention_meta(prefix, which == 0, suffix[which], rows,
                                    head_major ? "head_major" : "row_major",
                                    "row_major", index, step);
    }
    fprintf(stderr, "h3: dumped attention block %u step %d QKV "
            "(rows=%u heads=%d head_dim=%d, source layout %s) to "
            "%s.{q,k,v}.bin\n", index, step, rows, HEADS, HEAD_DIM,
            head_major ? "head-major" : "row-major", prefix);
    free(bf16); free(f16);
}

/* Debug-only: sibling of debug_dump_attention_qkv, called right after the
 * production SDPA call for the same block/step, dumping its real output
 * (dit->attention_heads) in the same [1, rows, HEADS, HEAD_DIM] FP16
 * format - lets an external kernel's output on the matching captured Q/K/V
 * be diffed directly against what this engine actually produced, not just
 * against that kernel's own internal reference. A head-major output is
 * reordered to row-major on the CPU, like the Q/K/V capture (previously
 * skipped here; H3_DISABLE_HEAD_MAJOR_ATTENTION_OUTPUT=1 still gives a
 * comparable capture). Gated behind the same env vars as the QKV dump. */
static void debug_dump_attention_output(h3_dit *dit, unsigned index, int step,
                                        uint32_t rows, int head_major) {
    const char *prefix = h3_option("H3_DUMP_ATTENTION_QKV");
    if (!prefix || !*prefix) return;
    const char *block_text = h3_option("H3_DUMP_ATTENTION_BLOCK");
    const char *step_text = h3_option("H3_DUMP_ATTENTION_STEP");
    unsigned want_block = block_text ? (unsigned)atoi(block_text) : 0;
    int want_step = step_text ? atoi(step_text) : 0;
    if (index != want_block || step != want_step) return;
    if (!h3_dit_gpu_op(dit, h3_gpu_submit(dit->gpu), NULL, 0,
                "flush for attention output dump") ||
        !h3_dit_gpu_op(dit, h3_gpu_begin(dit->gpu), NULL, 0,
                "resume after attention output dump")) {
        fprintf(stderr, "h3: warning: could not flush GPU for attention "
                "output dump\n");
        return;
    }
    size_t count = (size_t)rows * HEADS * HEAD_DIM;
    uint16_t *bf16 = malloc(count * sizeof(*bf16));
    uint16_t *f16 = malloc(count * sizeof(*f16));
    if (!bf16 || !f16) {
        fprintf(stderr, "h3: warning: out of memory for attention output "
                "dump\n");
        free(bf16); free(f16);
        return;
    }
    if (!h3_gpu_tensor_read_bf16(dit->attention_heads, bf16, count)) {
        fprintf(stderr, "h3: warning: could not read attention output for "
                "H3_DUMP_ATTENTION_QKV\n");
        free(bf16); free(f16);
        return;
    }
    for (size_t row = 0; row < rows; row++) {
        for (int head = 0; head < HEADS; head++) {
            size_t dst_base = (row * HEADS + (size_t)head) * HEAD_DIM;
            size_t src_base = head_major
                ? ((size_t)head * rows + row) * HEAD_DIM : dst_base;
            for (int d = 0; d < HEAD_DIM; d++)
                f16[dst_base + (size_t)d] =
                    debug_bf16_to_f16(bf16[src_base + (size_t)d]);
        }
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s.out.bin", prefix);
    FILE *file = fopen(path, "wb");
    if (file) {
        fwrite(f16, sizeof(*f16), count, file);
        fclose(file);
        debug_append_attention_meta(prefix, 0, "out", rows,
                                    head_major ? "head_major" : "row_major",
                                    "row_major", index, step);
        fprintf(stderr, "h3: dumped attention block %u step %d production "
                "output (rows=%u heads=%d head_dim=%d) to %s.out.bin\n",
                index, step, rows, HEADS, HEAD_DIM, prefix);
    } else {
        fprintf(stderr, "h3: warning: cannot open %s for attention output "
                "dump\n", path);
    }
    free(bf16); free(f16);
}

/* HyperFlow factored linears come in ~200 distinct (rank, width) shapes.
 * MPSGraph builds a graph per shape on first use (about 37 s at 512x512),
 * so on M5 they take the TensorOps kernel, which accepts any shape (ranks
 * are multiples of 256). Dense BF16 matrices (SSD streaming) keep the
 * default dispatch, where MPSGraph is faster for them. */
static int factored_linear(h3_dit *dit, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t input_dim, uint32_t output_dim) {
    if (dit->factored_tensor_ops) h3_gpu_set_nax_linear_generic(dit->gpu, 1);
    int ok = h3_gpu_linear_bf16(dit->gpu, output, input, weight, bias, rows,
                                input_dim, output_dim);
    if (dit->factored_tensor_ops) h3_gpu_set_nax_linear_generic(dit->gpu, 0);
    return ok;
}

static int dense_linear(h3_dit *dit, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight,
                        const h3_gpu_tensor *bias, uint32_t rows,
                        uint32_t input_dim, uint32_t output_dim) {
    return h3_gpu_linear_bf16(dit->gpu, output, input, weight, bias, rows,
                              input_dim, output_dim);
}

/* Writes the BF16 hidden state [sequence, HIDDEN] to <prefix>.blockNN.bf16,
 * waiting for the queued work first (H3_DUMP_BLOCK_HIDDEN). */
static int dump_block_hidden(h3_dit *dit, const char *prefix, unsigned block,
                             char *error, size_t error_size) {
    size_t count = (size_t)dit->sequence * HIDDEN;
    if (!h3_dit_gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                "submit before block dump")) return 0;
    uint16_t *values = malloc(count * sizeof(*values));
    char path[1024];
    snprintf(path, sizeof(path), "%s.block%02u.bf16", prefix, block);
    FILE *file = values ? fopen(path, "wb") : NULL;
    int ok = file && h3_gpu_tensor_read_bf16(dit->hidden, values, count) &&
        fwrite(values, sizeof(*values), count, file) == count;
    if (file && fclose(file) != 0) ok = 0;
    free(values);
    if (!ok) {
        h3_dit_fail(error, error_size, "cannot write block dump %s", path);
        return 0;
    }
    return h3_dit_gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                  "continue after block dump");
}

static int run_block(h3_dit *dit, unsigned index, int step,
                     h3_dit_block *weight,
                     int attention_adaln_ready,
                     int attention_input_quantized,
                     int fuse_next_attention, unsigned next_index,
                     int *next_attention_adaln_ready,
                     int *next_attention_input_quantized,
                     char *error, size_t error_size) {
    const h3_gpu_tensor *modulation = h3_dit_schedule_block(dit->schedule,
                                                            index);
    h3_gpu_tensor *row_map = dit->token_reduction_active ?
        dit->reduced_row_maps[step] : dit->row_maps[step];
    h3_gpu_tensor *rope_cos = dit->token_reduction_active ?
        dit->reduced_rope_cos : dit->rope_cos;
    h3_gpu_tensor *rope_sin = dit->token_reduction_active ?
        dit->reduced_rope_sin : dit->rope_sin;
    uint32_t rows = dit->token_reduction_active ?
        dit->reduced_sequence : dit->sequence;
#define OP(call, label) do {                                                    \
    if (!h3_dit_gpu_op(dit, (call), error, error_size, label)) return 0;               \
} while (0)
    int factored_qkv = weight->q_down != NULL;
    int factored_out = weight->out_down != NULL;
    int factored_mlp = weight->fc1_down || weight->fc2_down;
    if (!attention_adaln_ready)
        OP(h3_gpu_adaln_bf16(dit->gpu, dit->mod_attention, dit->hidden,
            weight->norm1, modulation, row_map, rows, HIDDEN, SLOTS,
            0, 1, 1e-5f), "DiT attention AdaLN");
    else if (factored_qkv && attention_input_quantized) {
        h3_dit_fail(error, error_size, "DiT block %u: factored QKV needs the BF16 "
             "AdaLN input, the previous block quantized it", index);
        return 0;
    }
    if (factored_qkv) {
        /* HyperFlow: Q, K, V each up(down(x)). The raw Q goes to the qkv
         * buffer and the raw K to the MLP's activated buffer (both free
         * until the attention output / MLP), V straight to value. */
        OP(factored_linear(dit, dit->lowrank_tmp, dit->mod_attention,
            weight->q_down, NULL, rows, HIDDEN, weight->rank[H3_HF_Q]),
           "HyperFlow Q down");
        OP(factored_linear(dit, dit->qkv, dit->lowrank_tmp,
            weight->q_up, NULL, rows, weight->rank[H3_HF_Q], INNER),
           "HyperFlow Q up");
        OP(factored_linear(dit, dit->lowrank_tmp, dit->mod_attention,
            weight->k_down, NULL, rows, HIDDEN, weight->rank[H3_HF_K]),
           "HyperFlow K down");
        OP(factored_linear(dit, dit->activated, dit->lowrank_tmp,
            weight->k_up, NULL, rows, weight->rank[H3_HF_K], INNER),
           "HyperFlow K up");
        OP(factored_linear(dit, dit->lowrank_tmp, dit->mod_attention,
            weight->v_down, NULL, rows, HIDDEN, weight->rank[H3_HF_V]),
           "HyperFlow V down");
        OP(factored_linear(dit, dit->value, dit->lowrank_tmp,
            weight->v_up, NULL, rows, weight->rank[H3_HF_V], INNER),
           "HyperFlow V up");
        OP(h3_gpu_qk_rope_split_bf16(dit->gpu, dit->query, dit->key, dit->qkv,
            dit->activated, weight->q_norm, weight->k_norm, rope_cos,
            rope_sin, rows, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f),
           "HyperFlow Q/K norm/RoPE");
    } else if (dit->int8_qkv && !h3_option("H3_DISABLE_INT8_QKV")) {
        OP(h3_gpu_grouped_qkv_linear_rope_int8(
            dit->gpu, dit->query, dit->key, dit->value,
            dit->int8_activation, dit->int8_activation_scales,
            dit->mod_attention, weight->qkv_int8, weight->qkv_scales,
            weight->q_norm, weight->k_norm, rope_cos, rope_sin,
            rows, HIDDEN, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f,
            attention_input_quantized,
            dit->use_slower_unfused_qkv_rope,
            dit->use_slower_scalar_qkv_rms,
            dit->use_slower_uncached_int8_scales),
           "DiT int8 QKV projection/norm/RoPE");
    } else {
        OP(h3_gpu_grouped_qkv_linear_rope_bf16(
            dit->gpu, dit->query, dit->key, dit->value, dit->qkv,
            dit->mod_attention, weight->qkv, weight->q_norm, weight->k_norm,
            rope_cos, rope_sin, rows, HIDDEN, HEADS, HEAD_DIM, ROPE_HALF,
            1e-5f), "DiT QKV projection/norm/RoPE");
    }
    debug_dump_attention_qkv(dit, index, step, rows);
    int int8_attention_output = dit->int8_attention_out && !factored_out &&
        !h3_option("H3_DISABLE_INT8_ATTENTION_OUT");
    int head_major_attention_output = int8_attention_output &&
        !dit->use_slower_row_major_attention_output &&
        !dit->use_slower_uncached_int8_scales &&
        !h3_option("H3_DISABLE_HEAD_MAJOR_ATTENTION_OUTPUT");
    /* Experimental, opt-in: replace this engine's own MPSGraph SDPA with
     * liuliu/ccv's Metal int8 NAX attention kernel for an A/B comparison
     * (see plans/SPEEDUP_ROADMAP.md item 5, tools/ccv_eval/README.md). Only
     * available in builds made with CCV_DIR set - h3_gpu_ccv_dense_
     * attention_bf16 fails cleanly with a clear error otherwise, it never
     * silently falls back to the default path, so a run that requests this
     * backend either really used it or aborted. Produces row-major output
     * (and so forces the following projection down the row-major path)
     * except in direct mode, which writes the head-major layout itself.
     * The mode is dit->attention_mode, fixed per generation by h3_generate
     * (h3_params.fast_attention, or H3_ATTENTION_BACKEND for diagnostics).
     * "ccv_fp16" is the same bridge with ccv's non-quantized FP16 kernel,
     * so the two differ only in int8 quantization of Q/K/V/P. */
    int ccv_direct = dit->attention_mode == H3_DIT_ATTENTION_CCV_INT8_DIRECT;
    int ccv_int8 = ccv_direct ||
                   dit->attention_mode == H3_DIT_ATTENTION_CCV_INT8;
    int ccv_fp16 = dit->attention_mode == H3_DIT_ATTENTION_CCV_FP16;
    if (ccv_int8 || ccv_fp16) {
        /* In: whether head-major output may be written (direct mode only,
         * and only when the projection below can take it). Out: what the
         * bridge actually wrote. */
        int ccv_head_major = ccv_direct && head_major_attention_output;
        OP(h3_gpu_ccv_dense_attention_bf16(
            dit->gpu, dit->attention_heads, dit->query, dit->key, dit->value,
            rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM), ccv_int8,
            &ccv_head_major),
           ccv_int8 ? "DiT ccv int8 attention" : "DiT ccv fp16 attention");
        head_major_attention_output = ccv_head_major;
    } else if (head_major_attention_output)
        OP(h3_gpu_sdpa_bf16_head_major_output(
            dit->gpu, dit->attention_heads, dit->query, dit->key, dit->value,
            rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
           "DiT head-major full attention");
    else
        OP(h3_gpu_sdpa_bf16(
            dit->gpu, dit->attention_heads, dit->query, dit->key, dit->value,
            rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
           "DiT full attention");
    debug_dump_attention_output(dit, index, step, rows,
                                head_major_attention_output);
    if (factored_out) {
        OP(factored_linear(dit, dit->lowrank_tmp,
            dit->attention_heads, weight->out_down, NULL, rows, INNER,
            weight->rank[H3_HF_OUT]), "HyperFlow attention output down");
        OP(factored_linear(dit, dit->attention_output,
            dit->lowrank_tmp, weight->out_up, NULL, rows,
            weight->rank[H3_HF_OUT], HIDDEN),
           "HyperFlow attention output up");
    } else if (int8_attention_output) {
        if (head_major_attention_output)
            OP(h3_gpu_linear_int8_head_major_bf16(
                dit->gpu, dit->attention_output, dit->int8_activation,
                dit->int8_activation_scales, dit->attention_heads,
                weight->out_int8, weight->out_scales, rows, HEADS, HEAD_DIM,
                HIDDEN), "DiT head-major int8 attention output");
        else
            OP(h3_gpu_linear_int8_bf16(
                dit->gpu, dit->attention_output, dit->int8_activation,
                dit->int8_activation_scales, dit->attention_heads,
                weight->out_int8, weight->out_scales, rows, INNER, HIDDEN,
                dit->use_slower_uncached_int8_scales),
               "DiT int8 attention output");
    } else {
        OP(h3_gpu_linear_bf16(dit->gpu, dit->attention_output,
            dit->attention_heads, weight->out, NULL, rows, INNER, HIDDEN),
           "DiT attention output");
    }
    int fused_int8_mlp_input = dit->int8_mlp && !factored_mlp &&
        !dit->use_slower_unfused_int8_inputs &&
        !h3_option("H3_DISABLE_FUSED_INT8_MLP_INPUT") &&
        !h3_option("H3_INT8_MLP_STAGE");
    if (fused_int8_mlp_input) {
        uint32_t padded_rows = (rows + 127u) & ~127u;
        OP(h3_gpu_gate_adaln_quantize_int8(
            dit->gpu, dit->hidden, dit->int8_activation,
            dit->int8_activation_scales, dit->hidden,
            dit->attention_output, weight->norm2, modulation, modulation,
            row_map, rows, padded_rows, HIDDEN, SLOTS, 2, 3, 4, 1e-5f),
           "DiT fused attention gate, MLP AdaLN and int8 quantization");
    } else if (!h3_option("H3_DISABLE_FUSED_GATE_ADALN")) {
        OP(h3_gpu_gate_adaln_bf16(
            dit->gpu, dit->hidden, dit->mod_mlp, dit->hidden,
            dit->attention_output, weight->norm2, modulation, modulation,
            row_map,
            rows, HIDDEN, SLOTS, 2, 3, 4, 1e-5f),
           "DiT fused attention gate and MLP AdaLN");
    } else {
        OP(h3_gpu_gate_bf16(dit->gpu, dit->hidden, dit->hidden,
            dit->attention_output, modulation, row_map, rows, HIDDEN,
            SLOTS, 2), "DiT attention gate");
        OP(h3_gpu_adaln_bf16(
            dit->gpu, dit->mod_mlp, dit->hidden, weight->norm2,
            modulation, row_map, rows, HIDDEN, SLOTS, 3, 4, 1e-5f),
           "DiT MLP AdaLN");
    }
    h3_gpu_tensor *mlp_output = dit->activation_aliases ?
        dit->attention_output : dit->mlp_output;
    if (factored_mlp) {
        /* HyperFlow: gate and up halves of FC1 separately, then SwiGLU in
         * place on the up half (in activated), then FC2. The gate half goes
         * to the qkv buffer, which holds mod_mlp when activations alias,
         * so its input is read first: through FC1's down factor, or copied
         * to lowrank_tmp for a dense FC1. */
        const h3_gpu_tensor *fc1_input = dit->mod_mlp;
        uint32_t fc1_width = HIDDEN;
        const h3_gpu_tensor *gate_weight = weight->fc1_gate;
        const h3_gpu_tensor *value_weight = weight->fc1_value;
        int (*fc1_linear)(h3_dit *, h3_gpu_tensor *, const h3_gpu_tensor *,
                          const h3_gpu_tensor *, const h3_gpu_tensor *,
                          uint32_t, uint32_t, uint32_t) =
            weight->fc1_down ? factored_linear : dense_linear;
        if (weight->fc1_down) {
            OP(factored_linear(dit, dit->lowrank_tmp, dit->mod_mlp,
                weight->fc1_down, NULL, rows, HIDDEN,
                weight->rank[H3_HF_FC1]), "HyperFlow FC1 down");
            fc1_width = weight->rank[H3_HF_FC1];
            gate_weight = weight->fc1_gate_up;
            value_weight = weight->fc1_value_up;
        } else {
            OP(h3_gpu_copy_bf16(dit->gpu, dit->lowrank_tmp, 0, dit->mod_mlp,
                0, (size_t)rows * HIDDEN), "HyperFlow FC1 input copy");
        }
        fc1_input = dit->lowrank_tmp;
        OP(fc1_linear(dit, dit->activated, fc1_input,
            value_weight, NULL, rows, fc1_width, FFN), "HyperFlow FC1 up half");
        OP(fc1_linear(dit, dit->qkv, fc1_input, gate_weight,
            NULL, rows, fc1_width, FFN), "HyperFlow FC1 gate half");
        OP(h3_gpu_swiglu_split_bf16(dit->gpu, dit->activated, dit->qkv,
            dit->activated, rows * FFN), "HyperFlow SwiGLU");
        if (weight->fc2_down) {
            OP(factored_linear(dit, dit->lowrank_tmp, dit->activated,
                weight->fc2_down, NULL, rows, FFN, weight->rank[H3_HF_FC2]),
               "HyperFlow FC2 down");
            OP(factored_linear(dit, mlp_output, dit->lowrank_tmp,
                weight->fc2_up, NULL, rows, weight->rank[H3_HF_FC2], HIDDEN),
               "HyperFlow FC2 up");
        } else {
            OP(h3_gpu_linear_bf16(dit->gpu, mlp_output, dit->activated,
                weight->fc2, NULL, rows, FFN, HIDDEN), "HyperFlow FC2");
        }
    } else if (dit->int8_mlp &&
        (!h3_option("H3_DISABLE_INT8_MLP") ||
         !weight->fc1 || !weight->fc2)) {
        OP(h3_gpu_mlp_int8_bf16(
            dit->gpu, mlp_output, dit->activated, dit->int8_activation,
            dit->int8_activation_scales, dit->mod_mlp,
            weight->fc1_int8, weight->fc1_scales,
            weight->fc2_int8, weight->fc2_scales,
            weight->fc1, weight->fc2,
            rows, HIDDEN, FFN, HIDDEN,
            dit->use_slower_grouped_quantizer,
            dit->use_slower_dynamic_fc1_k, dit->use_int8_row_fc2,
            fused_int8_mlp_input),
           "DiT int8 fused MLP");
    } else if (dit->nax_mlp && !h3_option("H3_DISABLE_NAX_MLP")) {
        OP(h3_gpu_mlp_nax_bf16(dit->gpu, mlp_output, dit->activated,
            dit->mod_mlp, weight->fc1, weight->fc2, rows, HIDDEN, FFN,
            HIDDEN), "DiT NAX fused MLP");
    } else if (dit->fused_mlp) {
        OP(h3_gpu_mlp_bf16(dit->gpu, mlp_output, dit->mod_mlp,
            weight->fc1, weight->fc2, rows, HIDDEN, FFN, HIDDEN),
           "DiT fused MLP");
    } else {
        OP(h3_gpu_linear_bf16(dit->gpu, dit->fc1, dit->mod_mlp, weight->fc1,
            NULL, rows, HIDDEN, FFN * 2), "DiT MLP input");
        OP(h3_gpu_swiglu_bf16(dit->gpu, dit->activated, dit->fc1, rows, FFN),
           "DiT SwiGLU");
        OP(h3_gpu_linear_bf16(dit->gpu, mlp_output, dit->activated,
            weight->fc2, NULL, rows, FFN, HIDDEN), "DiT MLP output");
    }
    if (fuse_next_attention) {
        h3_dit_block *next_weight = &dit->blocks[next_index];
        const h3_gpu_tensor *next_modulation = h3_dit_schedule_block(
            dit->schedule, next_index);
        int fuse_int8_qkv_input = dit->int8_qkv && !next_weight->q_down &&
            !dit->use_slower_unfused_int8_inputs &&
            !h3_option("H3_DISABLE_INT8_QKV") &&
            !h3_option("H3_DISABLE_FUSED_INT8_QKV_INPUT");
        if (fuse_int8_qkv_input) {
            uint32_t padded_rows = (rows + 127u) & ~127u;
            OP(h3_gpu_gate_adaln_quantize_int8(
                dit->gpu, dit->hidden, dit->int8_activation,
                dit->int8_activation_scales, dit->hidden, mlp_output,
                next_weight->norm1, modulation, next_modulation, row_map,
                rows, padded_rows, HIDDEN, SLOTS, 5, 0, 1, 1e-5f),
               "DiT fused MLP gate, next attention AdaLN and int8 quantization");
            *next_attention_input_quantized = 1;
        } else {
            OP(h3_gpu_gate_adaln_bf16(
                dit->gpu, dit->hidden, dit->mod_attention, dit->hidden,
                mlp_output, next_weight->norm1, modulation,
                next_modulation, row_map, rows, HIDDEN, SLOTS, 5, 0, 1,
                1e-5f), "DiT fused MLP gate and next attention AdaLN");
            *next_attention_input_quantized = 0;
        }
        *next_attention_adaln_ready = 1;
    } else {
        OP(h3_gpu_gate_bf16(
            dit->gpu, dit->hidden, dit->hidden, mlp_output,
            modulation, row_map, rows, HIDDEN, SLOTS, 5), "DiT MLP gate");
    }
#undef OP
    return 1;
}

int h3_dit_encode_forward(h3_dit *dit, int step, int begin, int submit,
                          int disable_command_split, char *error,
                          size_t error_size) {
#define OP(call, label) do {                                                    \
    if (!h3_dit_gpu_op(dit, (call), error, error_size, label)) return 0;               \
} while (0)
    if (begin) OP(h3_gpu_begin(dit->gpu), "begin DiT forward");
    size_t video_offset = 0;
    size_t audio_offset = 0;
    if (dit->fused_patch_pack) {
        if (dit->video_projection_map)
            OP(h3_gpu_patch_linear_bf16_map(
                dit->gpu, dit->hidden, dit->video_input, dit->video_patch_w,
                dit->video_patch_b, dit->video_projection_map, dit->sequence,
                dit->video_total_rows, VIDEO_PATCH, HIDDEN),
               "project mapped video sources");
        if (dit->audio_projection_map)
            OP(h3_gpu_patch_linear_bf16_map(
                dit->gpu, dit->hidden, dit->audio_input, dit->audio_patch_w,
                dit->audio_patch_b, dit->audio_projection_map, dit->sequence,
                dit->audio_total_rows, AUDIO_CHANNELS, HIDDEN),
               "project mapped audio sources");
        for (size_t index = 0; index < dit->layout.segment_count; index++) {
            const h3_segment *segment = &dit->layout.segments[index];
            size_t segment_rows = segment->stop - segment->start;
            size_t destination = segment->start * HIDDEN;
            if (segment->kind == H3_SEG_TEXT) {
                OP(h3_gpu_copy_bf16(dit->gpu, dit->hidden, destination,
                    dit->refined_text, 0, segment_rows * HIDDEN),
                   "pack refined text");
            } else if (segment->kind == H3_SEG_COND ||
                       segment->kind == H3_SEG_REF_IMAGE ||
                       segment->kind == H3_SEG_VIDEO) {
                if (!dit->video_projection_map)
                    OP(h3_gpu_patch_linear_bf16_offset(
                        dit->gpu, dit->hidden, destination, dit->video_input,
                        video_offset * VIDEO_PATCH, dit->video_patch_w,
                        dit->video_patch_b, (uint32_t)segment_rows,
                        VIDEO_PATCH, HIDDEN), "project packed video source");
                video_offset += segment_rows;
            } else {
                if (!dit->audio_projection_map)
                    OP(h3_gpu_patch_linear_bf16_offset(
                        dit->gpu, dit->hidden, destination, dit->audio_input,
                        audio_offset * AUDIO_CHANNELS, dit->audio_patch_w,
                        dit->audio_patch_b, (uint32_t)segment_rows,
                        AUDIO_CHANNELS, HIDDEN),
                       "project packed audio source");
                audio_offset += segment_rows;
            }
        }
    } else {
        if (dit->fused_patch_projection) {
            OP(h3_gpu_patch_linear_bf16(
                dit->gpu, dit->video_projected, dit->video_input,
                dit->video_patch_w, dit->video_patch_b,
                dit->video_total_rows, VIDEO_PATCH, HIDDEN),
               "fused video patch projection");
            OP(h3_gpu_patch_linear_bf16(
                dit->gpu, dit->audio_projected, dit->audio_input,
                dit->audio_patch_w, dit->audio_patch_b,
                dit->audio_total_rows, AUDIO_CHANNELS, HIDDEN),
               "fused audio patch projection");
        } else {
            OP(h3_gpu_linear_f32(
                dit->gpu, dit->video_projected_f32, dit->video_input,
                dit->video_patch_w, dit->video_patch_b,
                dit->video_total_rows, VIDEO_PATCH, HIDDEN),
               "video patch projection");
            OP(h3_gpu_linear_f32(
                dit->gpu, dit->audio_projected_f32, dit->audio_input,
                dit->audio_patch_w, dit->audio_patch_b,
                dit->audio_total_rows, AUDIO_CHANNELS, HIDDEN),
               "audio patch projection");
            OP(h3_gpu_cast_f32_to_bf16(
                dit->gpu, dit->video_projected, dit->video_projected_f32,
                dit->video_total_rows * HIDDEN), "video BF16 cast");
            OP(h3_gpu_cast_f32_to_bf16(
                dit->gpu, dit->audio_projected, dit->audio_projected_f32,
                dit->audio_total_rows * HIDDEN), "audio BF16 cast");
        }
        for (size_t index = 0; index < dit->layout.segment_count; index++) {
            const h3_segment *segment = &dit->layout.segments[index];
            size_t segment_rows = segment->stop - segment->start;
            size_t destination = segment->start * HIDDEN;
            if (segment->kind == H3_SEG_TEXT) {
                OP(h3_gpu_copy_bf16(dit->gpu, dit->hidden, destination,
                    dit->refined_text, 0, segment_rows * HIDDEN),
                   "pack refined text");
            } else if (segment->kind == H3_SEG_COND ||
                       segment->kind == H3_SEG_REF_IMAGE ||
                       segment->kind == H3_SEG_VIDEO) {
                OP(h3_gpu_copy_bf16(
                    dit->gpu, dit->hidden, destination, dit->video_projected,
                    video_offset * HIDDEN, segment_rows * HIDDEN),
                   "pack video source");
                video_offset += segment_rows;
            } else {
                OP(h3_gpu_copy_bf16(
                    dit->gpu, dit->hidden, destination, dit->audio_projected,
                    audio_offset * HIDDEN, segment_rows * HIDDEN),
                   "pack audio source");
                audio_offset += segment_rows;
            }
        }
    }
    if (video_offset != dit->video_total_rows ||
        audio_offset != dit->audio_total_rows) {
        h3_dit_fail(error, error_size, "DiT segment packing did not consume row sources");
        return 0;
    }
    int evaluate_core = dit->core_reuse_interval == 1 ||
        !dit->core_residual_ready ||
        dit->core_forward_count % dit->core_reuse_interval == 0 ||
        step == h3_dit_schedule_steps(dit->schedule) - 1;
    int use_token_reduction = evaluate_core && dit->token_reduction &&
        !h3_option("H3_DISABLE_TOKEN_REDUCTION");
    unsigned token_reduction_end =
        dit->token_reduction_early_steps &&
        (unsigned)step < dit->token_reduction_early_steps ?
            dit->token_reduction_early_end : dit->token_reduction_end;
    uint32_t hidden_elements = dit->sequence * HIDDEN;
    if (evaluate_core && dit->core_reuse_interval > 1)
        OP(h3_gpu_copy_bf16(dit->gpu, dit->core_input, 0, dit->hidden, 0,
                            hidden_elements), "save DiT core input");
    if (evaluate_core) {
        unsigned command_blocks = disable_command_split
            ? 0 : h3_dit_command_block_interval(dit);
        if (dit->ssd_streaming || dit->attention_stream) command_blocks = 0;
        unsigned completed_blocks = 0;
        /* Testing (tests/test_hyperflow_real.c): the hidden state after each
         * block of step 0 (resident or SSD-streamed weights). */
        const char *dump_prefix = h3_option("H3_DUMP_BLOCK_HIDDEN");
        if (dump_prefix && (!*dump_prefix || dit->attention_stream))
            dump_prefix = NULL;
        int carried_attention_adaln = 0;
        int carried_attention_input_quantized = 0;
        for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
            if (h3_dit_stop_if_cancelled(dit, error, error_size)) return 0;
            int fused_token_adaln = carried_attention_adaln;
            int fused_attention_input_quantized =
                carried_attention_input_quantized;
            carried_attention_adaln = 0;
            carried_attention_input_quantized = 0;
            if (use_token_reduction &&
                block == dit->token_reduction_begin) {
                fused_token_adaln = dit->block_active[block] &&
                    !h3_option("H3_DISABLE_FUSED_TOKEN_POOL_ADALN");
                if (fused_token_adaln) {
                    if (!enter_token_reduction_adaln(
                            dit, block, step, error, error_size)) return 0;
                    fused_attention_input_quantized = 0;
                } else if (!enter_token_reduction(
                               dit, error, error_size)) return 0;
            }
            if (use_token_reduction && block == token_reduction_end) {
                fused_token_adaln = dit->block_active[block] &&
                    !h3_option("H3_DISABLE_FUSED_TOKEN_ADALN");
                if (fused_token_adaln) {
                    if (!leave_token_reduction_adaln(
                            dit, block, step, error, error_size)) return 0;
                    fused_attention_input_quantized = 0;
                } else if (!leave_token_reduction(
                               dit, error, error_size)) return 0;
            }
            if (!dit->block_active[block]) continue;
            unsigned next_block = block + 1;
            int next_is_token_boundary = use_token_reduction &&
                (next_block == dit->token_reduction_begin ||
                 next_block == token_reduction_end);
            int fuse_next_attention =
                !h3_option("H3_DISABLE_FUSED_CROSS_BLOCK_ADALN") &&
                next_block < H3_DIT_BLOCKS &&
                dit->block_active[next_block] && !next_is_token_boundary;
            h3_dit_block streamed_weight;
            h3_dit_block *weight = &dit->blocks[block];
            h3_dit_stream_job stream_job;
            pthread_t stream_thread;
            int stream_started = 0;
            if (dit->ssd_streaming) {
                if (dit->stream_ready_layer != block ||
                    dit->stream_ready_slot > 1) {
                    h3_dit_fail(error, error_size,
                         "DiT SSD stream expected block %u, has block %u",
                         block, dit->stream_ready_layer);
                    return 0;
                }
                /* Norms and ranks from the resident block, each streamed
                 * matrix from the slot; fields this block does not stream
                 * stay NULL, which selects its dense/factored paths. */
                streamed_weight = dit->blocks[block];
                const h3_dit_stream_layer *layer =
                    &dit->stream_layers[block];
                for (unsigned index = 0; index < layer->count; index++) {
                    unsigned field = layer->sources[index].field;
                    *h3_dit_stream_slot_field(&streamed_weight, field) =
                        layer->target[dit->stream_ready_slot][index];
                }
                weight = &streamed_weight;
                if (!expand_streamed(dit, block, dit->stream_ready_slot,
                                     error, error_size) ||
                    !patch_streamed_bf16(dit, block, weight, error,
                                         error_size)) return 0;

                unsigned future = h3_dit_next_active_block(dit, block);
                if (future == H3_DIT_BLOCKS)
                    future = h3_dit_first_active_block(dit);
                stream_job = (h3_dit_stream_job){
                    .dit = dit,
                    .layer = future,
                    .slot = dit->stream_ready_slot ^ 1u
                };
                int thread_error = pthread_create(
                    &stream_thread, NULL, h3_dit_read_stream_layer_thread,
                    &stream_job);
                if (thread_error) {
                    h3_dit_fail(error, error_size,
                         "cannot start DiT SSD prefetch for block %u: %s",
                         future, strerror(thread_error));
                    return 0;
                }
                stream_started = 1;
            }
            h3_dit_attention_job attn_job;
            pthread_t attn_thread;
            int attn_started = 0;
            if (dit->attention_stream) {
                if (dit->attn_ready_layer != block ||
                    dit->attn_ready_slot > 1) {
                    h3_dit_fail(error, error_size,
                         "attention int8 stream expected block %u, has "
                         "block %u", block, dit->attn_ready_layer);
                    return 0;
                }
                h3_dit_attention_slot *slot =
                    &dit->attention_slots[dit->attn_ready_slot];
                streamed_weight = dit->blocks[block];
                streamed_weight.qkv_int8 = slot->qkv_int8;
                streamed_weight.qkv_scales = slot->qkv_scales;
                streamed_weight.out_int8 = slot->out_int8;
                streamed_weight.out_scales = slot->out_scales;
                if (dit->mlp_stream) {
                    streamed_weight.fc1_int8 = slot->fc1_int8;
                    streamed_weight.fc1_scales = slot->fc1_scales;
                    streamed_weight.fc2_int8 = slot->fc2_int8;
                    streamed_weight.fc2_scales = slot->fc2_scales;
                }
                weight = &streamed_weight;
                if (!patch_streamed_int8(dit, block, weight, error,
                                         error_size)) return 0;

                unsigned future = h3_dit_next_active_block(dit, block);
                if (future == H3_DIT_BLOCKS)
                    future = h3_dit_first_active_block(dit);
                attn_job = (h3_dit_attention_job){
                    .dit = dit,
                    .layer = future,
                    .slot = dit->attn_ready_slot ^ 1u
                };
                int thread_error = pthread_create(
                    &attn_thread, NULL, h3_dit_read_attention_layer_thread,
                    &attn_job);
                if (thread_error) {
                    h3_dit_fail(error, error_size,
                         "cannot start attention int8 prefetch for block "
                         "%u: %s", future, strerror(thread_error));
                    return 0;
                }
                attn_started = 1;
            }
            int block_ok = run_block(
                dit, block, step, weight, fused_token_adaln,
                fused_attention_input_quantized,
                fuse_next_attention, next_block,
                &carried_attention_adaln,
                &carried_attention_input_quantized,
                error, error_size);
            if (!block_ok) {
                if (stream_started) (void)pthread_join(stream_thread, NULL);
                if (attn_started) (void)pthread_join(attn_thread, NULL);
                return 0;
            }
            completed_blocks++;
            if (dump_prefix && step == 0 &&
                !dump_block_hidden(dit, dump_prefix, block, error,
                                   error_size)) return 0;
            if (command_blocks &&
                completed_blocks < dit->active_block_count &&
                completed_blocks % command_blocks == 0)
                OP(h3_gpu_continue(dit->gpu), "continue DiT command chain");
            if (stream_started) {
                int gpu_ok = h3_dit_gpu_op(dit, h3_gpu_submit(dit->gpu),
                                    error, error_size,
                                    "submit streamed DiT block");
                double wait_started = h3_dit_stream_now();
                int join_error = pthread_join(stream_thread, NULL);
                dit->stream_wait_seconds += h3_dit_stream_now() - wait_started;
                if (!gpu_ok) return 0;
                if (join_error) {
                    h3_dit_fail(error, error_size,
                         "cannot join DiT SSD prefetch: %s",
                         strerror(join_error));
                    return 0;
                }
                dit->stream_bytes += stream_job.bytes;
                dit->stream_read_seconds += stream_job.seconds;
                if (!stream_job.ok) {
                    h3_dit_fail(error, error_size,
                         "cannot stream DiT block %u: %s",
                         stream_job.layer, stream_job.error);
                    return 0;
                }
                dit->stream_ready_layer = stream_job.layer;
                dit->stream_ready_slot = stream_job.slot;
                OP(h3_gpu_begin(dit->gpu),
                   "continue after streamed DiT block");
            }
            if (attn_started) {
                int gpu_ok = h3_dit_gpu_op(dit, h3_gpu_submit(dit->gpu),
                                    error, error_size,
                                    "submit attention-streamed DiT block");
                double wait_started = h3_dit_stream_now();
                int join_error = pthread_join(attn_thread, NULL);
                dit->attn_stream_wait_seconds += h3_dit_stream_now() - wait_started;
                if (!gpu_ok) return 0;
                if (join_error) {
                    h3_dit_fail(error, error_size,
                         "cannot join attention int8 prefetch: %s",
                         strerror(join_error));
                    return 0;
                }
                dit->attn_stream_bytes += attn_job.bytes;
                dit->attn_stream_read_seconds += attn_job.seconds;
                if (!attn_job.ok) {
                    h3_dit_fail(error, error_size,
                         "cannot stream attention block %u: %s",
                         attn_job.layer, attn_job.error);
                    return 0;
                }
                dit->attn_ready_layer = attn_job.layer;
                dit->attn_ready_slot = attn_job.slot;
                OP(h3_gpu_begin(dit->gpu),
                   "continue after attention-streamed DiT block");
            }
        }
        if (use_token_reduction &&
            token_reduction_end == H3_DIT_BLOCKS &&
            !leave_token_reduction(dit, error, error_size)) return 0;
        if (dit->core_reuse_interval > 1) {
            OP(h3_gpu_sub_bf16(dit->gpu, dit->core_residual, dit->hidden,
                               dit->core_input, hidden_elements),
               "cache DiT core residual");
            dit->core_residual_ready = 1;
        }
    } else {
        OP(h3_gpu_add_bf16(dit->gpu, dit->hidden, dit->hidden,
                           dit->core_residual, hidden_elements),
           "reuse DiT core residual");
    }
    dit->core_forward_count++;
    const h3_gpu_tensor *final = h3_dit_schedule_final(dit->schedule);
    int fused_final_head = dit->bf16_final &&
        !h3_option("H3_DISABLE_FUSED_FINAL_HEAD") &&
        !h3_option("H3_DISABLE_FUSED_FINAL_SLICE");
    if (fused_final_head) {
        OP(h3_gpu_adaln_linear_bf16(
            dit->gpu, dit->audio_output_bf16, dit->final_audio_inverse,
            dit->hidden, (size_t)dit->audio_target_start * HIDDEN,
            dit->final_norm, final, dit->final_audio_maps[step],
            dit->final_audio_w, dit->final_audio_b, dit->audio_rows, HIDDEN,
            AUDIO_CHANNELS, FINAL_SLOTS, 0, 1, 1e-5f),
           "fused final audio AdaLN/head");
        OP(h3_gpu_adaln_linear_bf16(
            dit->gpu, dit->video_output_bf16, dit->final_video_inverse,
            dit->hidden, (size_t)dit->video_target_start * HIDDEN,
            dit->final_norm, final, dit->final_video_maps[step],
            dit->final_video_w, dit->final_video_b, dit->video_rows, HIDDEN,
            VIDEO_PATCH, FINAL_SLOTS, 0, 1, 1e-5f),
           "fused final video AdaLN/head");
    } else if (h3_option("H3_DISABLE_FUSED_FINAL_SLICE")) {
        OP(h3_gpu_copy_bf16(dit->gpu, dit->final_audio_input, 0, dit->hidden,
            (size_t)dit->audio_target_start * HIDDEN,
            (size_t)dit->audio_rows * HIDDEN), "slice final audio");
        OP(h3_gpu_copy_bf16(dit->gpu, dit->final_video_input, 0, dit->hidden,
            (size_t)dit->video_target_start * HIDDEN,
            (size_t)dit->video_rows * HIDDEN), "slice final video");
        OP(h3_gpu_adaln_bf16(dit->gpu, dit->final_audio_norm,
            dit->final_audio_input, dit->final_norm, final,
            dit->final_audio_maps[step], dit->audio_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "final audio AdaLN");
        OP(h3_gpu_adaln_bf16(dit->gpu, dit->final_video_norm,
            dit->final_video_input, dit->final_norm, final,
            dit->final_video_maps[step], dit->video_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "final video AdaLN");
    } else {
        OP(h3_gpu_adaln_bf16_offset(
            dit->gpu, dit->final_audio_norm, dit->hidden,
            (size_t)dit->audio_target_start * HIDDEN, dit->final_norm, final,
            dit->final_audio_maps[step], dit->audio_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "fused final audio slice/AdaLN");
        OP(h3_gpu_adaln_bf16_offset(
            dit->gpu, dit->final_video_norm, dit->hidden,
            (size_t)dit->video_target_start * HIDDEN, dit->final_norm, final,
            dit->final_video_maps[step], dit->video_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "fused final video slice/AdaLN");
    }
    if (dit->bf16_final && !fused_final_head) {
        OP(h3_gpu_linear_bf16(dit->gpu, dit->audio_output_bf16,
            dit->final_audio_norm, dit->final_audio_w, dit->final_audio_b,
            dit->audio_rows, HIDDEN, AUDIO_CHANNELS),
           "BF16 final audio head");
        OP(h3_gpu_linear_bf16(dit->gpu, dit->video_output_bf16,
            dit->final_video_norm, dit->final_video_w, dit->final_video_b,
            dit->video_rows, HIDDEN, VIDEO_PATCH),
           "BF16 final video head");
    } else if (!dit->bf16_final) {
        OP(h3_gpu_cast_bf16_to_f32(dit->gpu, dit->final_audio_f32,
            dit->final_audio_norm, dit->audio_rows * HIDDEN),
           "final audio F32 cast");
        OP(h3_gpu_cast_bf16_to_f32(dit->gpu, dit->final_video_f32,
            dit->final_video_norm, dit->video_rows * HIDDEN),
           "final video F32 cast");
        OP(h3_gpu_linear_f32(dit->gpu, dit->audio_output,
            dit->final_audio_f32, dit->final_audio_w, dit->final_audio_b,
            dit->audio_rows, HIDDEN, AUDIO_CHANNELS), "final audio head");
        OP(h3_gpu_linear_f32(dit->gpu, dit->video_output,
            dit->final_video_f32, dit->final_video_w, dit->final_video_b,
            dit->video_rows, HIDDEN, VIDEO_PATCH), "final video head");
        OP(h3_gpu_cast_f32_to_bf16(dit->gpu, dit->audio_output_bf16,
            dit->audio_output, dit->audio_rows * AUDIO_CHANNELS),
           "final audio output cast");
        OP(h3_gpu_cast_f32_to_bf16(dit->gpu, dit->video_output_bf16,
            dit->video_output, dit->video_rows * VIDEO_PATCH),
           "final video output cast");
    }
    if (submit) OP(h3_gpu_submit(dit->gpu), "submit DiT forward");
#undef OP
    return 1;
}
