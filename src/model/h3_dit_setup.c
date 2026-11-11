/* Building a DiT for one generation: the text refiner, RoPE and row maps, the core weights, activations, the prepared conditioning and h3_dit_load_t2va itself. Split out of h3_dit.c. */
#include "h3_dit_private.h"
static int run_refiner_block(h3_dit *dit, const h3_dit_block *weight,
                             h3_gpu_tensor *hidden, h3_gpu_tensor *norm,
                             h3_gpu_tensor *qkv, h3_gpu_tensor *query,
                             h3_gpu_tensor *key, h3_gpu_tensor *value,
                             h3_gpu_tensor *heads, h3_gpu_tensor *branch,
                             h3_gpu_tensor *fc1, h3_gpu_tensor *activated,
                             char *error, size_t error_size) {
    uint32_t rows = dit->text_rows;
#define OP(call, label) do {                                                    \
    if (!h3_dit_gpu_op(dit, (call), error, error_size, label)) return 0;               \
} while (0)
    OP(h3_gpu_rms_norm_bf16(dit->gpu, norm, hidden, weight->norm1, rows,
                             HIDDEN, 1e-5f), "refiner attention norm");
    OP(h3_gpu_linear_bf16(dit->gpu, qkv, norm, weight->qkv, NULL, rows,
                           HIDDEN, INNER * 3), "refiner QKV");
    OP(h3_gpu_grouped_qkv_rope_bf16(
                             dit->gpu, query, key, value, qkv, weight->q_norm,
                             weight->k_norm, weight->q_norm, weight->q_norm,
                             rows, HEADS, HEAD_DIM, 0, 1e-5f),
       "refiner QK norm");
    OP(h3_gpu_sdpa_bf16(dit->gpu, heads, query, key, value, rows, HEADS,
                         HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
       "refiner attention");
    OP(h3_gpu_linear_bf16(dit->gpu, branch, heads, weight->out, NULL, rows,
                           INNER, HIDDEN), "refiner attention output");
    OP(h3_gpu_add_bf16(dit->gpu, hidden, hidden, branch, rows * HIDDEN),
       "refiner attention residual");
    OP(h3_gpu_rms_norm_bf16(dit->gpu, norm, hidden, weight->norm2, rows,
                             HIDDEN, 1e-5f), "refiner MLP norm");
    OP(h3_gpu_linear_bf16(dit->gpu, fc1, norm, weight->fc1, NULL, rows,
                           HIDDEN, FFN * 2), "refiner MLP input");
    OP(h3_gpu_swiglu_bf16(dit->gpu, activated, fc1, rows, FFN),
       "refiner SwiGLU");
    OP(h3_gpu_linear_bf16(dit->gpu, branch, activated, weight->fc2, NULL,
                           rows, FFN, HIDDEN), "refiner MLP output");
    OP(h3_gpu_add_bf16(dit->gpu, hidden, hidden, branch, rows * HIDDEN),
       "refiner MLP residual");
#undef OP
    return 1;
}

static int refine_text(h3_dit *dit, const h3_text_embedding *text,
                       char *error, size_t error_size) {
    h3_gpu_tensor *source = h3_gpu_tensor_from_bf16(
        dit->gpu, text->values, text->tokens * TEXT_DIM);
    h3_gpu_tensor *condition_w = h3_dit_bf2(dit, "condition_proj.weight", HIDDEN,
                                     TEXT_DIM, error, error_size);
    h3_gpu_tensor *condition_b = h3_dit_bf1(dit, "condition_proj.bias", HIDDEN,
                                     error, error_size);
    h3_dit_block refiner[2];
    memset(refiner, 0, sizeof(refiner));
    h3_gpu_tensor *final_norm = NULL;
    h3_gpu_tensor *norm = NULL, *qkv = NULL, *query = NULL, *key = NULL;
    h3_gpu_tensor *value = NULL, *heads = NULL, *branch = NULL, *fc1 = NULL;
    h3_gpu_tensor *activated = NULL;
    int ok = source && condition_w && condition_b &&
             h3_dit_load_block(dit, &refiner[0], "token_refiner.blocks.0.", 1, 0,
                        error, error_size) &&
             h3_dit_load_block(dit, &refiner[1], "token_refiner.blocks.1.", 1, 1,
                        error, error_size);
    if (ok) final_norm = h3_dit_bf1(dit, "token_refiner.final_norm.weight", HIDDEN,
                             error, error_size);
    size_t rows = dit->text_rows;
    if (ok && final_norm) {
        dit->refined_text = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        norm = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        qkv = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER * 3);
        query = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        key = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        value = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        heads = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        branch = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        fc1 = h3_gpu_tensor_new_bf16(dit->gpu, rows * FFN * 2);
        activated = h3_gpu_tensor_new_bf16(dit->gpu, rows * FFN);
        ok = dit->refined_text && norm && qkv && query && key && value &&
             heads && branch && fc1 && activated;
    }
    if (!ok) {
        if (!error || !*error)
            h3_dit_fail(error, error_size, "cannot allocate token-refiner tensors: %s",
                 h3_gpu_error(dit->gpu));
        goto cleanup;
    }
    ok = h3_dit_gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                "begin token refinement") &&
         h3_dit_gpu_op(dit, h3_gpu_linear_bf16(
             dit->gpu, dit->refined_text, source, condition_w, condition_b,
             dit->text_rows, TEXT_DIM, HIDDEN), error, error_size,
             "condition projection") &&
         run_refiner_block(dit, &refiner[0], dit->refined_text, norm, qkv,
             query, key, value, heads, branch, fc1, activated,
             error, error_size) &&
         run_refiner_block(dit, &refiner[1], dit->refined_text, norm, qkv,
             query, key, value, heads, branch, fc1, activated,
             error, error_size) &&
         h3_dit_gpu_op(dit, h3_gpu_rms_norm_bf16(
             dit->gpu, dit->refined_text, dit->refined_text, final_norm,
             dit->text_rows, HIDDEN, 1e-5f), error, error_size,
             "refiner final norm") &&
         h3_dit_gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                "submit token refinement");
cleanup:
    h3_dit_free_tensor(&source);
    h3_dit_free_tensor(&condition_w);
    h3_dit_free_tensor(&condition_b);
    h3_dit_free_block(&refiner[0]);
    h3_dit_free_block(&refiner[1]);
    h3_dit_free_tensor(&final_norm);
    h3_dit_free_tensor(&norm);
    h3_dit_free_tensor(&qkv);
    h3_dit_free_tensor(&query);
    h3_dit_free_tensor(&key);
    h3_dit_free_tensor(&value);
    h3_dit_free_tensor(&heads);
    h3_dit_free_tensor(&branch);
    h3_dit_free_tensor(&fc1);
    h3_dit_free_tensor(&activated);
    return ok;
}

static int prepare_rope(h3_dit *dit, char *error, size_t error_size) {
    h3_gpu_tensor *inverse_tensor = h3_dit_f1(dit, "rope.inv_freq", ROPE_FREQS,
                                       error, error_size);
    float inverse[ROPE_FREQS];
    if (!inverse_tensor ||
        !h3_gpu_tensor_read_f32(inverse_tensor, inverse, ROPE_FREQS)) {
        h3_dit_free_tensor(&inverse_tensor);
        if (!error || !*error) h3_dit_fail(error, error_size, "cannot read RoPE frequencies");
        return 0;
    }
    h3_dit_free_tensor(&inverse_tensor);
    float spatial_scale = dit->spatial_rope_scale;
    size_t count = (size_t)dit->sequence * ROPE_HALF;
    size_t reduced_count = dit->token_reduction ?
        (size_t)dit->reduced_sequence * ROPE_HALF : 0;
    float *cosines = malloc(count * sizeof(*cosines));
    float *sines = malloc(count * sizeof(*sines));
    float *reduced_cosines = reduced_count ?
        malloc(reduced_count * sizeof(*reduced_cosines)) : NULL;
    float *reduced_sines = reduced_count ?
        malloc(reduced_count * sizeof(*reduced_sines)) : NULL;
    if (!cosines || !sines ||
        (reduced_count && (!reduced_cosines || !reduced_sines))) {
        free(cosines);
        free(sines);
        free(reduced_cosines);
        free(reduced_sines);
        h3_dit_fail(error, error_size, "out of memory allocating DiT RoPE tables");
        return 0;
    }
    for (uint32_t row = 0; row < dit->sequence; row++) {
        float axes[] = {(float)dit->layout.positions[row].t,
                        (float)dit->layout.positions[row].h * spatial_scale,
                        (float)dit->layout.positions[row].w * spatial_scale};
        for (uint32_t axis = 0; axis < 3; axis++) {
            for (uint32_t frequency = 0; frequency < ROPE_FREQS; frequency++) {
                size_t index = (size_t)row * ROPE_HALF +
                               axis * ROPE_FREQS + frequency;
                float angle = axes[axis] * inverse[frequency];
                cosines[index] = cosf(angle);
                sines[index] = sinf(angle);
            }
        }
    }
    for (uint32_t row = 0; row < dit->reduced_sequence; row++) {
        uint32_t first, second;
        h3_dit_token_pool_sources(dit, row, &first, &second);
        float axes[] = {
            (float)((dit->layout.positions[first].t +
                     dit->layout.positions[second].t) * 0.5),
            (float)((dit->layout.positions[first].h +
                     dit->layout.positions[second].h) * 0.5) * spatial_scale,
            (float)((dit->layout.positions[first].w +
                     dit->layout.positions[second].w) * 0.5) * spatial_scale
        };
        for (uint32_t axis = 0; axis < 3; axis++) {
            for (uint32_t frequency = 0; frequency < ROPE_FREQS; frequency++) {
                size_t index = (size_t)row * ROPE_HALF +
                               axis * ROPE_FREQS + frequency;
                float angle = axes[axis] * inverse[frequency];
                reduced_cosines[index] = cosf(angle);
                reduced_sines[index] = sinf(angle);
            }
        }
    }
    h3_gpu_tensor *cos_f32 = h3_gpu_tensor_from_f32(dit->gpu, cosines, count);
    h3_gpu_tensor *sin_f32 = h3_gpu_tensor_from_f32(dit->gpu, sines, count);
    h3_gpu_tensor *reduced_cos_f32 = reduced_count ?
        h3_gpu_tensor_from_f32(dit->gpu, reduced_cosines, reduced_count) : NULL;
    h3_gpu_tensor *reduced_sin_f32 = reduced_count ?
        h3_gpu_tensor_from_f32(dit->gpu, reduced_sines, reduced_count) : NULL;
    free(cosines);
    free(sines);
    free(reduced_cosines);
    free(reduced_sines);
    dit->rope_cos = h3_gpu_tensor_new_bf16(dit->gpu, count);
    dit->rope_sin = h3_gpu_tensor_new_bf16(dit->gpu, count);
    if (reduced_count) {
        dit->reduced_rope_cos = h3_gpu_tensor_new_bf16(
            dit->gpu, reduced_count);
        dit->reduced_rope_sin = h3_gpu_tensor_new_bf16(
            dit->gpu, reduced_count);
    }
    int ok = cos_f32 && sin_f32 && dit->rope_cos && dit->rope_sin &&
        (!reduced_count || (reduced_cos_f32 && reduced_sin_f32 &&
                            dit->reduced_rope_cos &&
                            dit->reduced_rope_sin));
    if (ok) {
        ok = h3_dit_gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                    "begin RoPE setup") &&
             h3_dit_gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                 dit->gpu, dit->rope_cos, cos_f32, (uint32_t)count),
                 error, error_size, "RoPE cosine cast") &&
             h3_dit_gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                 dit->gpu, dit->rope_sin, sin_f32, (uint32_t)count),
                 error, error_size, "RoPE sine cast");
        if (ok && reduced_count) {
            ok = h3_dit_gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                     dit->gpu, dit->reduced_rope_cos, reduced_cos_f32,
                     (uint32_t)reduced_count), error, error_size,
                     "reduced RoPE cosine cast") &&
                 h3_dit_gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                     dit->gpu, dit->reduced_rope_sin, reduced_sin_f32,
                     (uint32_t)reduced_count), error, error_size,
                     "reduced RoPE sine cast");
        }
        if (ok) ok =
             h3_dit_gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                    "submit RoPE setup");
    } else if (!error || !*error) {
        h3_dit_fail(error, error_size, "cannot allocate DiT RoPE buffers: %s",
             h3_gpu_error(dit->gpu));
    }
    h3_dit_free_tensor(&cos_f32);
    h3_dit_free_tensor(&sin_f32);
    h3_dit_free_tensor(&reduced_cos_f32);
    h3_dit_free_tensor(&reduced_sin_f32);
    return ok;
}

static int prepare_maps(h3_dit *dit, const h3_text_embedding *text,
                        char *error, size_t error_size) {
    int steps = h3_dit_schedule_steps(dit->schedule);
    dit->row_maps = calloc((size_t)steps, sizeof(*dit->row_maps));
    if (dit->token_reduction)
        dit->reduced_row_maps = calloc((size_t)steps,
                                       sizeof(*dit->reduced_row_maps));
    dit->final_audio_maps = calloc((size_t)steps,
                                   sizeof(*dit->final_audio_maps));
    dit->final_video_maps = calloc((size_t)steps,
                                   sizeof(*dit->final_video_maps));
    uint32_t *rows = malloc((size_t)dit->sequence * sizeof(*rows));
    uint32_t *reduced = dit->token_reduction ?
        malloc((size_t)dit->reduced_sequence * sizeof(*reduced)) : NULL;
    uint32_t *audio = malloc((size_t)dit->audio_rows * sizeof(*audio));
    uint32_t *video = malloc((size_t)dit->video_rows * sizeof(*video));
    if (!dit->row_maps || !dit->final_audio_maps || !dit->final_video_maps ||
        (dit->token_reduction && (!dit->reduced_row_maps || !reduced)) ||
        !rows || !audio || !video) {
        h3_dit_fail(error, error_size, "out of memory allocating modulation row maps");
        free(rows); free(reduced); free(audio); free(video);
        return 0;
    }
    for (int step = 0; step < steps; step++) {
        if (!h3_dit_schedule_row_map(dit->schedule, step, &dit->layout,
                                     text->tags, text->tokens, rows,
                                     dit->sequence)) {
            h3_dit_fail(error, error_size, "cannot construct modulation row map");
            free(rows); free(reduced); free(audio); free(video);
            return 0;
        }
        if (dit->token_reduction) {
            for (uint32_t row = 0; row < dit->reduced_sequence; row++) {
                uint32_t first, second;
                h3_dit_token_pool_sources(dit, row, &first, &second);
                (void)second;
                reduced[row] = rows[first];
            }
            dit->reduced_row_maps[step] = h3_gpu_tensor_from_u32(
                dit->gpu, reduced, dit->reduced_sequence);
        }
        uint32_t audio_row = h3_dit_schedule_audio_row(dit->schedule, step);
        uint32_t video_row = h3_dit_schedule_video_row(dit->schedule, step);
        for (uint32_t index = 0; index < dit->audio_rows; index++)
            audio[index] = audio_row;
        for (uint32_t index = 0; index < dit->video_rows; index++)
            video[index] = video_row;
        dit->row_maps[step] = h3_gpu_tensor_from_u32(
            dit->gpu, rows, dit->sequence);
        dit->final_audio_maps[step] = h3_gpu_tensor_from_u32(
            dit->gpu, audio, dit->audio_rows);
        dit->final_video_maps[step] = h3_gpu_tensor_from_u32(
            dit->gpu, video, dit->video_rows);
        if (!dit->row_maps[step] ||
            (dit->token_reduction && !dit->reduced_row_maps[step]) ||
            !dit->final_audio_maps[step] ||
            !dit->final_video_maps[step]) {
            h3_dit_fail(error, error_size, "cannot allocate modulation row maps: %s",
                 h3_gpu_error(dit->gpu));
            free(rows); free(reduced); free(audio); free(video);
            return 0;
        }
    }
    free(rows); free(reduced); free(audio); free(video);
    return 1;
}

static int prepare_projection_maps(h3_dit *dit, char *error,
                                   size_t error_size) {
    unsigned video_segments = 0, audio_segments = 0;
    for (size_t index = 0; index < dit->layout.segment_count; index++) {
        h3_segment_kind kind = dit->layout.segments[index].kind;
        if (kind == H3_SEG_COND || kind == H3_SEG_REF_IMAGE ||
            kind == H3_SEG_VIDEO)
            video_segments++;
        else if (kind != H3_SEG_TEXT)
            audio_segments++;
    }
    uint32_t *video = video_segments > 1 ?
        malloc((size_t)dit->video_total_rows * sizeof(*video)) : NULL;
    uint32_t *audio = audio_segments > 1 ?
        malloc((size_t)dit->audio_total_rows * sizeof(*audio)) : NULL;
    if ((video_segments > 1 && !video) || (audio_segments > 1 && !audio)) {
        free(video); free(audio);
        h3_dit_fail(error, error_size, "out of memory allocating projection maps");
        return 0;
    }
    size_t video_offset = 0, audio_offset = 0;
    for (size_t index = 0; index < dit->layout.segment_count; index++) {
        const h3_segment *segment = &dit->layout.segments[index];
        size_t rows = segment->stop - segment->start;
        if (segment->kind == H3_SEG_COND ||
            segment->kind == H3_SEG_REF_IMAGE ||
            segment->kind == H3_SEG_VIDEO) {
            for (size_t row = 0; video && row < rows; row++)
                video[video_offset + row] = (uint32_t)(segment->start + row);
            video_offset += rows;
        } else if (segment->kind != H3_SEG_TEXT) {
            for (size_t row = 0; audio && row < rows; row++)
                audio[audio_offset + row] = (uint32_t)(segment->start + row);
            audio_offset += rows;
        }
    }
    if (video_offset != dit->video_total_rows ||
        audio_offset != dit->audio_total_rows) {
        free(video); free(audio);
        h3_dit_fail(error, error_size, "projection map rows are inconsistent");
        return 0;
    }
    if (video)
        dit->video_projection_map = h3_gpu_tensor_from_u32(
            dit->gpu, video, dit->video_total_rows);
    if (audio)
        dit->audio_projection_map = h3_gpu_tensor_from_u32(
            dit->gpu, audio, dit->audio_total_rows);
    free(video); free(audio);
    if ((video_segments > 1 && !dit->video_projection_map) ||
        (audio_segments > 1 && !dit->audio_projection_map)) {
        h3_dit_fail(error, error_size, "cannot allocate projection map tensors: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static int prepare_token_reduction_maps(h3_dit *dit, char *error,
                                        size_t error_size) {
    if (!dit->token_reduction) return 1;
    size_t pair_count = (size_t)dit->reduced_sequence * 2;
    uint32_t *pairs = malloc(pair_count * sizeof(*pairs));
    uint32_t *baseline_indices = malloc(
        (size_t)dit->reduced_sequence * sizeof(*baseline_indices));
    uint32_t *parents = malloc((size_t)dit->sequence * sizeof(*parents));
    if (!pairs || !baseline_indices || !parents) {
        free(pairs);
        free(baseline_indices);
        free(parents);
        h3_dit_fail(error, error_size,
             "out of memory allocating token-reduction maps");
        return 0;
    }
    uint32_t baseline_row = 0;
    for (uint32_t row = 0; row < dit->reduced_sequence; row++) {
        h3_dit_token_pool_sources(dit, row, &pairs[(size_t)row * 2],
                           &pairs[(size_t)row * 2 + 1]);
        baseline_indices[row] =
            row >= dit->video_target_start &&
            pairs[(size_t)row * 2] != pairs[(size_t)row * 2 + 1] ?
                baseline_row++ : UINT32_MAX;
    }
    if (baseline_row != dit->token_baseline_rows) {
        free(pairs);
        free(baseline_indices);
        free(parents);
        h3_dit_fail(error, error_size, "token-reduction baseline map is inconsistent");
        return 0;
    }
    for (uint32_t row = 0; row < dit->sequence; row++)
        parents[row] = h3_dit_token_reduced_parent(dit, row);
    dit->token_pool_pairs = h3_gpu_tensor_from_u32(
        dit->gpu, pairs, pair_count);
    dit->token_baseline_indices = h3_gpu_tensor_from_u32(
        dit->gpu, baseline_indices, dit->reduced_sequence);
    dit->token_expand_parents = h3_gpu_tensor_from_u32(
        dit->gpu, parents, dit->sequence);
    free(pairs);
    free(baseline_indices);
    free(parents);
    if (!dit->token_pool_pairs || !dit->token_baseline_indices ||
        !dit->token_expand_parents) {
        h3_dit_fail(error, error_size,
             "cannot allocate token-reduction map tensors: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static void configure_active_blocks(h3_dit *dit, unsigned active) {
    memset(dit->block_active, 1, sizeof(dit->block_active));
    dit->active_block_count = active;
    unsigned skipped = H3_DIT_BLOCKS - active;
    for (unsigned index = 0; index < skipped; index++) {
        unsigned block = ((2 * index + 1) * H3_DIT_BLOCKS) / (2 * skipped);
        if (block == 0) block = 1;
        if (block >= H3_DIT_BLOCKS - 1) block = H3_DIT_BLOCKS - 2;
        dit->block_active[block] = 0;
    }
}

unsigned h3_dit_first_active_block(const h3_dit *dit) {
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        if (dit->block_active[block]) return block;
    return H3_DIT_BLOCKS;
}

unsigned h3_dit_next_active_block(const h3_dit *dit, unsigned current) {
    for (unsigned block = current + 1; block < H3_DIT_BLOCKS; block++)
        if (dit->block_active[block]) return block;
    return H3_DIT_BLOCKS;
}

static void configure_gate_ranked_blocks(h3_dit *dit) {
    const char *policy = h3_option("H3_DIT_LAYER_POLICY");
    if ((policy && !strcmp(policy, "uniform")) || dit->hyperflow ||
        dit->active_block_count == H3_DIT_BLOCKS) return;
    typedef struct { unsigned block; double score; } block_score;
    /* The first two and final blocks establish/close the residual stream.
     * Block 1 has a small gate but proved structurally essential in decoded
     * A/B renders, so magnitude ranking must not treat it as disposable. */
    block_score scores[H3_DIT_BLOCKS - 3];
    for (unsigned block = 2; block + 1 < H3_DIT_BLOCKS; block++) {
        double score = h3_dit_schedule_gate_score(dit->schedule, block);
        if (score < 0.0) return;
        scores[block - 2] = (block_score){block, score};
    }
    unsigned count = H3_DIT_BLOCKS - 3;
    for (unsigned left = 0; left < count; left++) {
        unsigned least = left;
        for (unsigned right = left + 1; right < count; right++)
            if (scores[right].score < scores[least].score) least = right;
        block_score temporary = scores[left];
        scores[left] = scores[least];
        scores[least] = temporary;
    }
    memset(dit->block_active, 1, sizeof(dit->block_active));
    unsigned skipped = H3_DIT_BLOCKS - dit->active_block_count;
    for (unsigned index = 0; index < skipped; index++)
        dit->block_active[scores[index].block] = 0;
    if (h3_option("H3_PROFILE")) {
        fprintf(stderr, "h3: gate-ranked DiT skips");
        for (unsigned index = 0; index < skipped; index++)
            fprintf(stderr, " %u(%.4g)", scores[index].block,
                    scores[index].score);
        fputc('\n', stderr);
    }
}

static int load_core(h3_dit *dit, h3_dit_progress progress, void *opaque,
                     char *error, size_t error_size) {
    for (unsigned index = 0; index < H3_DIT_BLOCKS; index++) {
        if (!dit->block_active[index]) {
            h3_dit_report(progress, opaque, "load transformer core", (int)index + 1,
                   H3_DIT_BLOCKS);
            continue;
        }
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "blocks.%u.", index);
        if (dit->hyperflow && !dit->ssd_streaming) {
            /* Resident. Dense parts are quantized like any block; factored
             * ones (and a dense half of a factored MLP) stay BF16. */
            h3_dit_block *block = &dit->blocks[index];
            if (!h3_dit_load_hyperflow_block(dit, index, error, error_size))
                return 0;
            if (dit->int8_mlp && block->fc1 && block->fc2 &&
                !h3_dit_quantize_block_mlp(dit, block, error, error_size)) return 0;
            if (dit->int8_qkv && block->qkv &&
                !h3_dit_quantize_block_qkv(dit, block, error, error_size)) return 0;
            if (dit->int8_attention_out && block->out &&
                !h3_dit_quantize_block_attention_out(dit, block, error, error_size))
                return 0;
        } else if (dit->ssd_streaming) {
            if (!h3_dit_load_block_norms(dit, &dit->blocks[index], prefix,
                                  error, error_size) ||
                !h3_dit_prepare_stream_layer(dit, index, error, error_size))
                return 0;
            if (dit->hyperflow)
                memcpy(dit->blocks[index].rank,
                       dit->hyperflow_info.rank[index],
                       sizeof(dit->blocks[index].rank));
        } else if (dit->attention_stream && dit->mlp_stream) {
            /* QKV/OUT/FC1/FC2 are all streamed; only norms stay resident. */
            if (!h3_dit_load_block_norms(dit, &dit->blocks[index], prefix,
                                  error, error_size)) return 0;
        } else if (dit->attention_stream) {
            if (!h3_dit_load_block_norms_and_mlp(dit, &dit->blocks[index], prefix,
                                         index, error, error_size))
                return 0;
            if (dit->int8_mlp &&
                !h3_dit_quantize_block_mlp(dit, &dit->blocks[index],
                                    error, error_size)) return 0;
        } else {
            if (!h3_dit_load_block(dit, &dit->blocks[index], prefix, 0, index,
                            error, error_size)) return 0;
            if (dit->int8_mlp &&
                !h3_dit_quantize_block_mlp(dit, &dit->blocks[index],
                                    error, error_size)) return 0;
            if (dit->int8_qkv &&
                !h3_dit_quantize_block_qkv(dit, &dit->blocks[index],
                                    error, error_size)) return 0;
            if (dit->int8_attention_out &&
                !h3_dit_quantize_block_attention_out(
                    dit, &dit->blocks[index], error, error_size)) return 0;
        }
        h3_dit_report(progress, opaque, "load transformer core", (int)index + 1,
               H3_DIT_BLOCKS);
    }
    if (dit->ssd_streaming) {
        unsigned first = h3_dit_first_active_block(dit);
        if (first == H3_DIT_BLOCKS) {
            h3_dit_fail(error, error_size, "SSD stream has no active DiT block");
            return 0;
        }
        if (!h3_dit_allocate_stream_slots(dit, error, error_size)) return 0;
        h3_dit_stream_job job = {
            .dit = dit, .layer = first, .slot = 0
        };
        if (!h3_dit_read_stream_layer(&job)) {
            h3_dit_fail(error, error_size, "cannot prime DiT SSD stream: %s",
                 job.error);
            return 0;
        }
        dit->stream_ready_layer = first;
        dit->stream_ready_slot = 0;
        dit->stream_bytes += job.bytes;
        dit->stream_read_seconds += job.seconds;
    }
    if (dit->attention_stream) {
        if (!h3_dit_allocate_attention_slot(dit, &dit->attention_slots[0],
                                     error, error_size) ||
            !h3_dit_allocate_attention_slot(dit, &dit->attention_slots[1],
                                     error, error_size)) return 0;
        unsigned first = h3_dit_first_active_block(dit);
        if (first == H3_DIT_BLOCKS) {
            h3_dit_fail(error, error_size,
                 "attention cache stream has no active DiT block");
            return 0;
        }
        h3_dit_attention_job job = {
            .dit = dit, .layer = first, .slot = 0
        };
        if (!h3_dit_read_attention_layer(&job)) {
            h3_dit_fail(error, error_size,
                 "cannot prime attention int8 stream: %s", job.error);
            return 0;
        }
        dit->attn_ready_layer = first;
        dit->attn_ready_slot = 0;
        dit->attn_stream_bytes += job.bytes;
        dit->attn_stream_read_seconds += job.seconds;
    }
    dit->video_patch_w = h3_dit_f2(dit, "video_patch_proj.weight", HIDDEN,
                            VIDEO_PATCH, error, error_size);
    dit->video_patch_b = h3_dit_f1(dit, "video_patch_proj.bias", HIDDEN,
                            error, error_size);
    dit->audio_patch_w = h3_dit_f2(dit, "audio_patch_proj.weight", HIDDEN,
                            AUDIO_CHANNELS, error, error_size);
    dit->audio_patch_b = h3_dit_f1(dit, "audio_patch_proj.bias", HIDDEN,
                            error, error_size);
    dit->final_norm = h3_dit_bf1(dit, "final_layer.norm.weight", HIDDEN,
                          error, error_size);
    dit->final_video_w = h3_dit_f2(dit, "final_layer.video_out.weight", VIDEO_PATCH,
                            HIDDEN, error, error_size);
    dit->final_video_b = h3_dit_f1(dit, "final_layer.video_out.bias", VIDEO_PATCH,
                            error, error_size);
    dit->final_audio_w = h3_dit_f2(dit, "final_layer.audio_out.weight", AUDIO_CHANNELS,
                            HIDDEN, error, error_size);
    dit->final_audio_b = h3_dit_f1(dit, "final_layer.audio_out.bias", AUDIO_CHANNELS,
                            error, error_size);
    if (dit->bf16_final && dit->final_video_w && dit->final_video_b &&
        dit->final_audio_w && dit->final_audio_b) {
        h3_gpu_tensor *source[4] = {
            dit->final_video_w, dit->final_video_b,
            dit->final_audio_w, dit->final_audio_b
        };
        size_t elements[4] = {
            (size_t)VIDEO_PATCH * HIDDEN, VIDEO_PATCH,
            (size_t)AUDIO_CHANNELS * HIDDEN, AUDIO_CHANNELS
        };
        h3_gpu_tensor *target[4] = {0};
        int ok = 1;
        for (unsigned index = 0; index < 4; index++) {
            target[index] = h3_gpu_tensor_new_bf16(dit->gpu,
                                                    elements[index]);
            if (!target[index]) ok = 0;
        }
        if (ok) ok = h3_gpu_begin(dit->gpu);
        for (unsigned index = 0; ok && index < 4; index++)
            ok = h3_gpu_cast_f32_to_bf16(dit->gpu, target[index],
                                         source[index],
                                         (uint32_t)elements[index]);
        if (ok) ok = h3_gpu_submit(dit->gpu);
        if (ok) {
            for (unsigned index = 0; index < 4; index++)
                h3_gpu_tensor_free(source[index]);
            dit->final_video_w = target[0];
            dit->final_video_b = target[1];
            dit->final_audio_w = target[2];
            dit->final_audio_b = target[3];
        } else {
            for (unsigned index = 0; index < 4; index++)
                h3_gpu_tensor_free(target[index]);
            h3_dit_fail(error, error_size, "cannot convert DiT final weights: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    return dit->video_patch_w && dit->video_patch_b && dit->audio_patch_w &&
           dit->audio_patch_b && dit->final_norm && dit->final_video_w &&
           dit->final_video_b && dit->final_audio_w && dit->final_audio_b;
}

static int allocate_activations(h3_dit *dit, char *error, size_t error_size) {
    size_t sequence = dit->sequence;
    size_t audio = dit->audio_rows;
    size_t video = dit->video_rows;
    size_t audio_total = dit->audio_total_rows;
    size_t video_total = dit->video_total_rows;
    dit->activation_aliases = !h3_option("H3_DISABLE_DIT_ACTIVATION_ALIAS");
    dit->fused_patch_projection =
        !h3_option("H3_DISABLE_FUSED_PATCH_CAST") && !h3_option("H3_SCALAR_PATCH");
    dit->fused_patch_pack = dit->fused_patch_projection &&
        !h3_option("H3_DISABLE_FUSED_PATCH_PACK");
#define BF(field, elements) (dit->field = h3_gpu_tensor_new_bf16(dit->gpu, (elements)))
#define F32(field, elements) (dit->field = h3_gpu_tensor_new_f32(dit->gpu, (elements)))
    h3_gpu_tensor *all[] = {
        F32(video_input, video_total * VIDEO_PATCH),
        F32(audio_input, audio_total * AUDIO_CHANNELS),
        BF(hidden, sequence * HIDDEN),
        BF(mod_attention, sequence * HIDDEN),
        BF(qkv, sequence * INNER * 3),
        BF(query, sequence * INNER),
        BF(key, sequence * INNER),
        BF(value, sequence * INNER),
        BF(attention_output, sequence * HIDDEN),
        F32(final_audio_inverse, audio),
        F32(final_video_inverse, video),
        BF(audio_output_bf16, audio * AUDIO_CHANNELS),
        BF(video_output_bf16, video * VIDEO_PATCH)
    };
#undef BF
#undef F32
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++) {
        if (!all[index]) {
            h3_dit_fail(error, error_size, "cannot allocate DiT activation arena: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->fused_patch_pack) {
        dit->video_projected = h3_gpu_tensor_new_bf16(
            dit->gpu, video_total * HIDDEN);
        dit->audio_projected = h3_gpu_tensor_new_bf16(
            dit->gpu, audio_total * HIDDEN);
        if (!dit->video_projected || !dit->audio_projected) {
            h3_dit_fail(error, error_size,
                 "cannot allocate packed patch projections: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->fused_patch_projection) {
        dit->video_projected_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, video_total * HIDDEN);
        dit->audio_projected_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, audio_total * HIDDEN);
        if (!dit->video_projected_f32 || !dit->audio_projected_f32) {
            h3_dit_fail(error, error_size,
                 "cannot allocate separate patch projections: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->activation_aliases) {
        dit->attention_heads = dit->qkv;
        dit->mod_mlp = dit->qkv;
        dit->mlp_output = NULL;
    } else {
        dit->attention_heads = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * INNER);
        dit->mod_mlp = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
        dit->mlp_output = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
    }
    if (!dit->attention_heads || !dit->mod_mlp ||
        (!dit->activation_aliases && !dit->mlp_output)) {
        h3_dit_fail(error, error_size,
             "cannot allocate DiT activation buffers: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (h3_option("H3_DISABLE_FUSED_FINAL_SLICE")) {
        dit->final_audio_input = h3_gpu_tensor_new_bf16(
            dit->gpu, audio * HIDDEN);
        dit->final_video_input = h3_gpu_tensor_new_bf16(
            dit->gpu, video * HIDDEN);
        if (!dit->final_audio_input || !dit->final_video_input) {
            h3_dit_fail(error, error_size,
                 "cannot allocate separate final DiT slices: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->bf16_final || h3_option("H3_DISABLE_FUSED_FINAL_HEAD") ||
        h3_option("H3_DISABLE_FUSED_FINAL_SLICE")) {
        dit->final_audio_norm = h3_gpu_tensor_new_bf16(
            dit->gpu, audio * HIDDEN);
        dit->final_video_norm = h3_gpu_tensor_new_bf16(
            dit->gpu, video * HIDDEN);
        if (!dit->final_audio_norm || !dit->final_video_norm) {
            h3_dit_fail(error, error_size,
                 "cannot allocate separate final DiT normalization: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->bf16_final) {
        dit->final_audio_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, audio * HIDDEN);
        dit->final_video_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, video * HIDDEN);
        dit->audio_output = h3_gpu_tensor_new_f32(
            dit->gpu, audio * AUDIO_CHANNELS);
        dit->video_output = h3_gpu_tensor_new_f32(
            dit->gpu, video * VIDEO_PATCH);
        if (!dit->final_audio_f32 || !dit->final_video_f32 ||
            !dit->audio_output || !dit->video_output) {
            h3_dit_fail(error, error_size,
                 "cannot allocate F32 DiT final activations: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->fused_mlp) {
        dit->fc1 = h3_gpu_tensor_new_bf16(dit->gpu, sequence * FFN * 2);
    }
    if (dit->hyperflow && dit->hyperflow_info.factored_blocks) {
        size_t width = dit->hyperflow_info.max_rank > HIDDEN ?
            dit->hyperflow_info.max_rank : HIDDEN;
        dit->lowrank_tmp = h3_gpu_tensor_new_bf16(dit->gpu, sequence * width);
        if (!dit->lowrank_tmp) {
            h3_dit_fail(error, error_size,
                 "cannot allocate the factored-linear buffer: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->fused_mlp || dit->nax_mlp || dit->int8_mlp ||
        dit->lowrank_tmp) {
        dit->activated = h3_gpu_tensor_new_bf16(dit->gpu, sequence * FFN);
        if ((!dit->fused_mlp && !dit->fc1) || !dit->activated) {
            h3_dit_fail(error, error_size,
                 "cannot allocate diagnostic DiT MLP tensors: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->int8_mlp || dit->int8_qkv || dit->int8_attention_out) {
        size_t padded_sequence = (sequence + 127) & ~(size_t)127;
        dit->int8_activation = h3_gpu_tensor_new_i8(
            dit->gpu, padded_sequence * FFN);
        dit->int8_activation_scales = h3_gpu_tensor_new_f32(
            dit->gpu, padded_sequence * (FFN / 1024));
        if (!dit->int8_activation || !dit->int8_activation_scales) {
            h3_dit_fail(error, error_size,
                 "cannot allocate int8 DiT activation arena: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->token_reduction) {
        size_t full_elements = sequence * HIDDEN;
        size_t qkv_capacity = sequence * INNER * 3;
        size_t qkv_used = (size_t)dit->reduced_sequence * INNER * 3;
        size_t baseline_elements =
            (size_t)dit->token_baseline_rows * HIDDEN;
        size_t attention_capacity = sequence * HIDDEN;
        size_t attention_used =
            (size_t)dit->reduced_sequence * HIDDEN;
        dit->token_original_in_qkv =
            qkv_used <= qkv_capacity &&
            full_elements <= qkv_capacity - qkv_used &&
            qkv_used <= UINT32_MAX &&
            full_elements <= UINT32_MAX - qkv_used;
        if (dit->token_original_in_qkv)
            dit->token_original_offset = qkv_used;
        else
            dit->token_original = h3_gpu_tensor_new_bf16(
                dit->gpu, full_elements);
        dit->token_baseline_offset = attention_used;
        if (attention_used > attention_capacity ||
            baseline_elements > attention_capacity - attention_used ||
            attention_used > UINT32_MAX ||
            baseline_elements > UINT32_MAX - attention_used ||
            (!dit->token_original_in_qkv && !dit->token_original)) {
            h3_dit_fail(error, error_size,
                 "cannot allocate token-reduction residual state: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->core_reuse_interval > 1) {
        dit->core_input = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
        dit->core_residual = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
        if (!dit->core_input || !dit->core_residual) {
            h3_dit_fail(error, error_size,
                 "cannot allocate DiT core residual cache: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    return 1;
}

/* Seed-independent preparation kept between loads (see h3_dit.h). Each part
 * is reused only when everything it was computed from matches exactly; the
 * seed only enters later (noise, condition augmentation), and every run
 * still gets fresh activations and step state. */
struct h3_dit_prep {
    int keep_refined;
    int keep_adaln;
    /* Token refiner: condition projection + 2 refiner blocks + final norm. */
    char *refined_directory;
    uint8_t refined_model[32];
    char *refined_loras;
    uint16_t *refined_input;
    size_t refined_input_count;
    uint32_t refined_rows;
    uint16_t *refined_output;
    /* AdaLN schedule, before gate-ranked pruning. */
    char *adaln_directory;
    uint8_t adaln_model[32];
    h3_sigma_schedule adaln_sigmas;
    int adaln_visual;
    int adaln_audio;
    h3_dit_schedule_host *adaln;
};

static void prep_clear_refined(h3_dit_prep *prep) {
    free(prep->refined_directory);
    free(prep->refined_loras);
    free(prep->refined_input);
    free(prep->refined_output);
    prep->refined_directory = NULL;
    prep->refined_loras = NULL;
    prep->refined_input = NULL;
    prep->refined_output = NULL;
    prep->refined_input_count = 0;
    prep->refined_rows = 0;
}

static void prep_clear_adaln(h3_dit_prep *prep) {
    free(prep->adaln_directory);
    prep->adaln_directory = NULL;
    h3_dit_schedule_host_free(prep->adaln);
    prep->adaln = NULL;
}

h3_dit_prep *h3_dit_prep_new(void) {
    return calloc(1, sizeof(h3_dit_prep));
}

void h3_dit_prep_free(h3_dit_prep *prep) {
    if (!prep) return;
    prep_clear_refined(prep);
    prep_clear_adaln(prep);
    free(prep);
}

void h3_dit_prep_clear(h3_dit_prep *prep) {
    if (!prep) return;
    prep_clear_refined(prep);
    prep_clear_adaln(prep);
}

void h3_dit_prep_set(h3_dit_prep *prep, int refined_text, int adaln) {
    if (!prep) return;
    prep->keep_refined = refined_text != 0;
    prep->keep_adaln = adaln != 0;
    if (!prep->keep_refined) prep_clear_refined(prep);
    if (!prep->keep_adaln) prep_clear_adaln(prep);
}

size_t h3_dit_prep_bytes(const h3_dit_prep *prep, int adaln) {
    if (!prep) return 0;
    if (adaln) return h3_dit_schedule_host_bytes(prep->adaln);
    if (!prep->refined_output) return 0;
    return ((size_t)prep->refined_rows * HIDDEN + prep->refined_input_count) *
           sizeof(uint16_t);
}

/* path, strength and the file's size and mtime for each adapter: a
 * rewritten adapter file at the same path is a different refiner. */
static char *lora_description(const h3_lora *loras, size_t count) {
    size_t capacity = 64 + count * (PATH_MAX + 96);
    char *text = malloc(capacity);
    if (!text) return NULL;
    size_t length = (size_t)snprintf(text, capacity, "%zu", count);
    for (size_t index = 0; index < count && length < capacity; index++) {
        struct stat status;
        long long size = -1, seconds = 0, nanoseconds = 0;
        if (loras[index].path && stat(loras[index].path, &status) == 0) {
            size = (long long)status.st_size;
            seconds = (long long)status.st_mtimespec.tv_sec;
            nanoseconds = (long long)status.st_mtimespec.tv_nsec;
        }
        length += (size_t)snprintf(text + length, capacity - length,
                                   "|%s@%.9g:%lld:%lld.%09lld",
                                   loras[index].path ? loras[index].path : "",
                                   (double)loras[index].strength, size,
                                   seconds, nanoseconds);
    }
    return text;
}

static int same_text(const char *left, const char *right) {
    return left && right && !strcmp(left, right);
}

/* refine_text(), or its stored result when the model, the input embedding,
 * the row count and the LoRA set (the refiner's weights can be patched)
 * all match. */
static int refine_text_reusing(h3_dit *dit, const h3_text_embedding *text,
                               h3_dit_prep *prep,
                               const char *weight_directory,
                               const h3_lora *loras, size_t lora_count,
                               char *error, size_t error_size) {
    if (!prep || !prep->keep_refined)
        return refine_text(dit, text, error, error_size);
    uint8_t model[32];
    h3_weight_store_fingerprint(dit->weights, model);
    char *lora_key = lora_description(loras, lora_count);
    size_t input_count = text->tokens * TEXT_DIM;
    size_t output_count = (size_t)dit->text_rows * HIDDEN;
    if (lora_key && prep->refined_output &&
        same_text(prep->refined_directory, weight_directory) &&
        !memcmp(prep->refined_model, model, sizeof(model)) &&
        same_text(prep->refined_loras, lora_key) &&
        prep->refined_rows == dit->text_rows &&
        prep->refined_input_count == input_count &&
        !memcmp(prep->refined_input, text->values,
                input_count * sizeof(uint16_t))) {
        free(lora_key);
        dit->refined_text = h3_gpu_tensor_from_bf16(
            dit->gpu, prep->refined_output, output_count);
        if (!dit->refined_text) {
            h3_dit_fail(error, error_size, "cannot restore refined text: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
        fprintf(stderr, "h3: refined text cache hit\n");
        return 1;
    }
    prep_clear_refined(prep);
    if (!refine_text(dit, text, error, error_size)) {
        free(lora_key);
        return 0;
    }
    prep->refined_directory = strdup(weight_directory);
    prep->refined_loras = lora_key;
    prep->refined_input = malloc(input_count * sizeof(uint16_t));
    prep->refined_output = malloc(output_count * sizeof(uint16_t));
    if (!prep->refined_directory || !prep->refined_loras ||
        !prep->refined_input || !prep->refined_output ||
        !h3_gpu_tensor_read_bf16(dit->refined_text, prep->refined_output,
                                 output_count)) {
        prep_clear_refined(prep);
        fprintf(stderr, "h3: warning: could not retain refined text\n");
        return 1;
    }
    memcpy(prep->refined_input, text->values, input_count * sizeof(uint16_t));
    memcpy(prep->refined_model, model, sizeof(model));
    prep->refined_input_count = input_count;
    prep->refined_rows = dit->text_rows;
    fprintf(stderr, "h3: refined text cache miss; stored %.1f MiB\n",
            (double)h3_dit_prep_bytes(prep, 0) / (1024.0 * 1024.0));
    return 1;
}

typedef struct {
    h3_dit_progress callback;
    void *opaque;
} schedule_progress;

static void schedule_report(int completed, int total, void *opaque) {
    schedule_progress *state = opaque;
    h3_dit_report(state->callback, state->opaque, "precompute AdaLN", completed, total);
}

static h3_dit *load_dit(const char *weight_directory,
                        const char *shader_source_path,
                        const h3_text_embedding *text,
                        const h3_layout *layout,
                        const h3_sigma_schedule *sigmas,
                        unsigned active_blocks,
                        unsigned core_reuse_interval,
                        int token_reduction,
                        int ssd_streaming,
                        float spatial_rope_scale,
                        int use_slower_bf16_mlp,
                        int use_slower_bf16_qkv,
                        int use_slower_bf16_attention_output,
                        int use_slower_row_major_attention_output,
                        int use_slower_unfused_int8_inputs,
                        int use_slower_unfused_qkv_rope,
                        int use_slower_scalar_qkv_rms,
                        int use_slower_uncached_int8_scales,
                        int use_slower_dynamic_fc1_k,
                        int use_slower_grouped_quantizer,
                        int use_int8_row_fc2,
                        const h3_lora *loras, size_t lora_count,
                        h3_dit_prep *prep,
                        const float *condition_video_rows,
                        size_t condition_video_elements,
                        const float *condition_audio_rows,
                        size_t condition_audio_elements,
                        h3_dit_progress progress, void *progress_opaque,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || !layout || !sigmas ||
        (ssd_streaming != 0 && ssd_streaming != 1) ||
        !isfinite(spatial_rope_scale) || spatial_rope_scale <= 0.0f ||
        active_blocks < H3_DIT_BLOCKS / 2 ||
        active_blocks > H3_DIT_BLOCKS || core_reuse_interval < 1 ||
        core_reuse_interval > 6) {
        h3_dit_fail(error, error_size, "invalid DiT load arguments");
        return NULL;
    }
    h3_dit *dit = calloc(1, sizeof(*dit));
    if (!dit) {
        h3_dit_fail(error, error_size, "out of memory creating DiT model");
        return NULL;
    }
    dit->fused_mlp = h3_option("H3_DISABLE_FUSED_MLP") == NULL;
    /* The released final heads are F32, but their inputs are already BF16.
     * Converting these small weights once selects the Iris-derived tiled
     * linear and eliminates two full-width casts plus the scalar F32 kernel.
     * Keep the old path available for close-reference diagnosis. */
    dit->bf16_final = h3_option("H3_DIT_F32_FINAL") == NULL;
    dit->core_reuse_interval = core_reuse_interval;
    dit->ssd_streaming = ssd_streaming;
    dit->spatial_rope_scale = spatial_rope_scale;
    configure_active_blocks(dit, active_blocks);
    if (!h3_dit_copy_layout(dit, layout, error, error_size) ||
        !h3_dit_validate_layout(dit, text, error, error_size) ||
        !h3_dit_configure_token_reduction(dit, token_reduction,
                                   error, error_size)) goto failed;
    size_t wanted_video_condition =
        (size_t)dit->video_condition_rows * VIDEO_PATCH;
    size_t wanted_audio_condition =
        (size_t)dit->audio_condition_rows * AUDIO_CHANNELS;
    if (condition_video_elements != wanted_video_condition ||
        condition_audio_elements != wanted_audio_condition ||
        (wanted_video_condition && !condition_video_rows) ||
        (wanted_audio_condition && !condition_audio_rows)) {
        h3_dit_fail(error, error_size,
             "condition row elements do not match the packed DiT layout");
        goto failed;
    }
    dit->sigmas = *sigmas;
    dit->weights = h3_weight_store_open(weight_directory, error, error_size);
    if (!dit->weights) goto failed;
    if (!h3_hyperflow_inspect(dit->weights, &dit->hyperflow_info, error,
                              error_size)) goto failed;
    dit->hyperflow = dit->hyperflow_info.two_time;
    /* What this family takes (h3_model_family.h); h3_generate has already
     * refused these with its own message. */
    dit->family = h3_model_family_of(dit->hyperflow);
    const unsigned can = dit->family->info.capabilities;
    if ((!(can & H3_CAN_LAYER_SKIP) && active_blocks != H3_DIT_BLOCKS) ||
        (!(can & H3_CAN_STEP_REUSE) &&
         (core_reuse_interval != 1 || token_reduction)) ||
        (!(can & H3_CAN_LORA) && lora_count)) {
        h3_dit_fail(error, error_size,
             "%s runs all of its blocks: layer skipping, "
             "core reuse, token reduction and LoRA are not available",
             dit->family->info.label);
        goto failed;
    }
    if (dit->hyperflow) {
        /* Its blocks are renumbered without gaps; the ones after
         * block_count do not exist. */
        memset(dit->block_active, 0, sizeof(dit->block_active));
        memset(dit->block_active, 1, dit->hyperflow_info.block_count);
        dit->active_block_count = dit->hyperflow_info.block_count;
        fprintf(stderr, "h3: HyperFlow H3 Flash: %u blocks (%u with factored "
                "linears, largest rank %u), two-time AdaLN (gate %g)\n",
                dit->hyperflow_info.block_count,
                dit->hyperflow_info.factored_blocks,
                dit->hyperflow_info.max_rank,
                (double)dit->hyperflow_info.gate);
    }
    dit->gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!dit->gpu) goto failed;
    /* factored_linear; H3_HYPERFLOW_MPSGRAPH=1 keeps MPSGraph. */
    dit->factored_tensor_ops = dit->hyperflow &&
        h3_gpu_has_tensor_ops(dit->gpu) && !h3_option("H3_HYPERFLOW_MPSGRAPH");
    if (lora_count) {
        dit->loras = h3_lora_set_load(dit->gpu, loras, lora_count, error,
                                      error_size);
        if (!dit->loras) goto failed;
    }
    dit->nax_mlp = dit->fused_mlp && h3_gpu_has_nax_mlp(dit->gpu);
    dit->int8_mlp = !dit->ssd_streaming && dit->fused_mlp &&
                    !use_slower_bf16_mlp &&
                    h3_gpu_has_int8_mlp(dit->gpu);
    dit->int8_qkv = !dit->ssd_streaming && !use_slower_bf16_qkv &&
                    dit->sequence >= 128 &&
                    h3_gpu_has_int8_mlp(dit->gpu);
    dit->int8_attention_out = !dit->ssd_streaming &&
                              !use_slower_bf16_attention_output &&
                              dit->sequence >= 128 &&
                              h3_gpu_has_int8_mlp(dit->gpu);
    const char *attention_cache_path = h3_option("H3_ATTENTION_CACHE");
    const char *attention_cache_dir = h3_option("H3_ATTENTION_CACHE_DIR");
    h3_cache_model_kind model_kind = h3_dit_detect_model_kind(weight_directory);
    char *selected_cache_path = NULL;
    if (attention_cache_path && *attention_cache_path &&
        attention_cache_dir && *attention_cache_dir) {
        h3_dit_fail(error, error_size,
             "set only one of H3_ATTENTION_CACHE or H3_ATTENTION_CACHE_DIR");
        goto failed;
    }
    if (attention_cache_dir && *attention_cache_dir) {
        if (model_kind == H3_CACHE_MODEL_UNKNOWN) {
            h3_dit_fail(error, error_size,
                 "H3_ATTENTION_CACHE_DIR needs the standard FL2VA/Ref2VA "
                 "transformer layout to auto-select a cache file - point "
                 "H3_ATTENTION_CACHE at a specific file instead for a "
                 "non-standard directory");
            goto failed;
        }
        const char *name = model_kind == H3_CACHE_MODEL_REF2VA ?
            "ref2va.cache" : "fl2va.cache";
        size_t length = strlen(attention_cache_dir) + strlen(name) + 2;
        selected_cache_path = malloc(length);
        if (!selected_cache_path) {
            h3_dit_fail(error, error_size, "out of memory building cache path");
            goto failed;
        }
        snprintf(selected_cache_path, length, "%s/%s", attention_cache_dir,
                name);
        attention_cache_path = selected_cache_path;
    }
    if (attention_cache_path && *attention_cache_path &&
        !(dit->family->info.capabilities & H3_CAN_ATTENTION_CACHE)) {
        /* Caches are built from the 50 dense H3 blocks. */
        fprintf(stderr, "h3: %s checkpoint; ignoring H3_ATTENTION_CACHE "
                        "(resident blocks instead)\n", dit->family->log_name);
        attention_cache_path = NULL;
        free(selected_cache_path);
        selected_cache_path = NULL;
    }
    if (attention_cache_path && *attention_cache_path && dit->ssd_streaming) {
        /* --ssd-streaming was passed explicitly - H3_ATTENTION_CACHE is
         * very likely just left set in the environment for other, longer
         * runs, so honor --ssd-streaming instead of forcing the user to
         * unset the env var by hand every time for a quick test. The
         * other conditions below that disable int8_qkv/int8_attention_out
         * (a short sequence, no int8 GPU support, --use-slower-bf16-qkv/
         * -attention-output) stay hard errors - those are configurations
         * that can't serve the cache at all, not an explicit alternate
         * mode the user chose. */
        fprintf(stderr,
                "h3: --ssd-streaming set; ignoring H3_ATTENTION_CACHE\n");
        attention_cache_path = NULL;
        free(selected_cache_path);
        selected_cache_path = NULL;
    }
    if (attention_cache_path && *attention_cache_path) {
        if (!dit->int8_qkv || !dit->int8_attention_out) {
            h3_dit_fail(error, error_size,
                 "H3_ATTENTION_CACHE needs the int8 QKV/attention-output "
                 "path (unavailable here: --ssd-streaming, "
                 "--use-slower-bf16-qkv/-attention-output, a short "
                 "sequence, or a GPU without the int8 path all disable it)");
            free(selected_cache_path);
            goto failed;
        }
        const char *stream_mlp = h3_option("H3_INT8_STREAM_MLP");
        int want_mlp_stream = stream_mlp && *stream_mlp && strcmp(stream_mlp, "0");
        if (want_mlp_stream && !dit->int8_mlp) {
            h3_dit_fail(error, error_size,
                 "H3_INT8_STREAM_MLP needs the int8 MLP path (unavailable "
                 "here: --ssd-streaming, --use-slower-bf16-mlp, or a GPU "
                 "without the int8 path disable it)");
            free(selected_cache_path);
            goto failed;
        }
        uint8_t expected_model_id[32], legacy_model_id[32];
        h3_weight_store_fingerprint(dit->weights, expected_model_id);
        h3_weight_store_fingerprint_legacy(dit->weights, legacy_model_id);
        if (!h3_dit_attention_cache_validate(attention_cache_path, want_mlp_stream,
                                      model_kind, expected_model_id,
                                      legacy_model_id, error, error_size)) {
            free(selected_cache_path);
            goto failed;
        }
        dit->attention_cache_path = selected_cache_path ? selected_cache_path :
            strdup(attention_cache_path);
        selected_cache_path = NULL;
        if (!dit->attention_cache_path) {
            h3_dit_fail(error, error_size, "out of memory copying cache path");
            goto failed;
        }
        dit->attention_stream = 1;
        dit->mlp_stream = want_mlp_stream;
    }
    free(selected_cache_path);
    dit->use_slower_row_major_attention_output =
        use_slower_row_major_attention_output;
    dit->use_slower_unfused_int8_inputs =
        use_slower_unfused_int8_inputs;
    dit->use_slower_unfused_qkv_rope =
        use_slower_unfused_qkv_rope;
    dit->use_slower_scalar_qkv_rms = use_slower_scalar_qkv_rms;
    dit->use_slower_uncached_int8_scales =
        use_slower_uncached_int8_scales;
    dit->use_slower_dynamic_fc1_k = use_slower_dynamic_fc1_k;
    dit->keep_bf16_attention_out = dit->int8_attention_out &&
        (h3_option("H3_INT8_KEEP_BF16_ATTENTION_OUT") ||
         h3_option("H3_BENCH_INT8_ATTENTION_OUT_AB"));
    dit->keep_bf16_qkv = dit->int8_qkv &&
        (h3_option("H3_INT8_KEEP_BF16_QKV") ||
         h3_option("H3_BENCH_INT8_QKV_AB"));
    dit->use_slower_grouped_quantizer = use_slower_grouped_quantizer;
    dit->use_int8_row_fc2 = dit->int8_mlp && use_int8_row_fc2;
    dit->keep_bf16_mlp = dit->int8_mlp &&
        (h3_option("H3_INT8_KEEP_BF16_MLP") ||
         h3_option("H3_BENCH_INT8_MLP_AB") ||
         h3_option("H3_INT8_MLP_STAGE"));
    h3_gpu_profile_set_label(dit->gpu, "H3 DiT");
    h3_dit_report(progress, progress_opaque, "refine text", 0, 1);
    if (!refine_text_reusing(dit, text, prep, weight_directory, loras,
                             lora_count, error, error_size)) goto failed;
    for (int projection = 0; projection < H3_LORA_PROJECTIONS; projection++)
        h3_lora_set_release(dit->loras, 1, projection);
    h3_dit_report(progress, progress_opaque, "refine text", 1, 1);
    schedule_progress schedule_state = {progress, progress_opaque};
    int visual_condition = dit->video_condition_rows != 0;
    int audio_condition = dit->audio_condition_rows != 0;
    uint8_t model[32];
    h3_weight_store_fingerprint(dit->weights, model);
    /* H3_ADALN_CURVE: the AdaLN schedule from a curve-form file
     * (h3_adaln_curve.h, built by build_adaln_curve) instead of the ~24 GiB
     * of dense projections. It must have been built from this transformer.
     * Its identity goes into the schedule-cache key (bytes 16..31, unused by
     * the weight fingerprint), so dense and curve schedules never mix. */
    const char *curve_path =
        dit->family->info.capabilities & H3_CAN_ADALN_CURVE ? h3_option("H3_ADALN_CURVE") : NULL;
    h3_adaln_curve *curve = NULL;
    if (dit->hyperflow) {
        /* Two-time AdaLN; no curve form applies. */
    } else if (h3_adaln_curve_in_store(dit->weights)) {
        /* A curve-form checkpoint has no dense AdaLN to fall back on. */
        curve_path = "the checkpoint itself";
        curve = h3_adaln_curve_from_store(dit->weights, error, error_size);
        if (!curve) goto failed;
        uint8_t curve_id[32];
        h3_adaln_curve_fingerprint(curve, curve_id);
        memcpy(model + 16, curve_id, 16);
    } else if (curve_path && *curve_path) {
        curve = h3_adaln_curve_open(curve_path, model, error, error_size);
        if (!curve) goto failed;
        uint8_t curve_id[32];
        h3_adaln_curve_fingerprint(curve, curve_id);
        memcpy(model + 16, curve_id, 16);
    }
    /* LoRA AdaLN curve branches change the schedule, so it is neither
     * taken from nor stored in the prepared cache then. */
    int adaln_patched = h3_lora_set_has_adaln(dit->loras);
    if (adaln_patched && !curve)
        fprintf(stderr, "h3: warning: a LoRA's AdaLN curve branches need a "
                        "curve-form checkpoint; ignored\n");
    if (!adaln_patched && prep && prep->keep_adaln && prep->adaln &&
        same_text(prep->adaln_directory, weight_directory) &&
        !memcmp(prep->adaln_model, model, sizeof(model)) &&
        !memcmp(&prep->adaln_sigmas, sigmas, sizeof(*sigmas)) &&
        prep->adaln_visual == visual_condition &&
        prep->adaln_audio == audio_condition) {
        /* Depends only on the weights, the sigma schedule and which
         * conditions are present - not on the prompt or the seed. */
        dit->schedule = h3_dit_schedule_import(prep->adaln, dit->gpu,
                                               error, error_size);
        if (dit->schedule) fprintf(stderr, "h3: AdaLN schedule cache hit\n");
    } else {
        if (prep) prep_clear_adaln(prep);
        if (dit->hyperflow) {
            dit->schedule = h3_dit_schedule_precompute_two_time(
                dit->weights, dit->gpu, sigmas, visual_condition,
                audio_condition, dit->hyperflow_info.gate,
                dit->hyperflow_info.time_dim, dit->hyperflow_info.block_count,
                schedule_report, &schedule_state, error, error_size);
        } else if (curve) {
            fprintf(stderr, "h3: AdaLN from curve %s (rank %u, grid %u)\n",
                    curve_path, h3_adaln_curve_rank(curve),
                    h3_adaln_curve_grid(curve));
            dit->schedule = h3_dit_schedule_precompute_curve(
                dit->loras, curve, dit->gpu, sigmas, visual_condition,
                audio_condition,
                schedule_report, &schedule_state, error, error_size);
        } else {
            dit->schedule = h3_dit_schedule_precompute(
                dit->weights, dit->gpu, sigmas, visual_condition,
                audio_condition, schedule_report, &schedule_state,
                error, error_size);
        }
        if (dit->schedule && prep && prep->keep_adaln && !adaln_patched) {
            prep->adaln = h3_dit_schedule_export(dit->schedule);
            prep->adaln_directory = strdup(weight_directory);
            if (!prep->adaln || !prep->adaln_directory) {
                prep_clear_adaln(prep);
                fprintf(stderr, "h3: warning: could not retain AdaLN schedule\n");
            } else {
                memcpy(prep->adaln_model, model, sizeof(model));
                prep->adaln_sigmas = *sigmas;
                prep->adaln_visual = visual_condition;
                prep->adaln_audio = audio_condition;
                fprintf(stderr, "h3: AdaLN schedule cache miss; stored %.1f MiB\n",
                        (double)h3_dit_prep_bytes(prep, 1) / (1024.0 * 1024.0));
            }
        }
    }
    h3_adaln_curve_free(curve);
    /* H3_ADALN_ULP_NOISE=<fraction> (diagnostic): a rounding-sized random
     * change to the schedule, as a yardstick for H3_ADALN_CURVE's. Applied
     * to this generation only (after the schedule cache stored its copy). */
    const char *ulp_noise = h3_option("H3_ADALN_ULP_NOISE");
    if (dit->schedule && ulp_noise && atof(ulp_noise) > 0.0) {
        fprintf(stderr, "h3: perturbing %.1f%% of AdaLN values by one BF16 step\n",
                100.0 * atof(ulp_noise));
        if (!h3_dit_schedule_perturb_ulps(dit->schedule, atof(ulp_noise), 1)) {
            h3_dit_fail(error, error_size, "cannot perturb the AdaLN schedule");
            goto failed;
        }
    }
    if (dit->schedule) {
        configure_gate_ranked_blocks(dit);
        h3_dit_schedule_prune(dit->schedule, dit->block_active,
                              H3_DIT_BLOCKS);
    }
    if (!dit->schedule || !prepare_rope(dit, error, error_size) ||
        !prepare_maps(dit, text, error, error_size) ||
        !prepare_projection_maps(dit, error, error_size) ||
        !prepare_token_reduction_maps(dit, error, error_size) ||
        !load_core(dit, progress, progress_opaque, error, error_size) ||
        !allocate_activations(dit, error, error_size)) goto failed;
    /* Keep LoRA factors only for projections that stream in again every
     * step; resident ones were patched once in load_core(). */
    for (int projection = 0; projection < H3_LORA_PROJECTIONS; projection++) {
        int mlp = projection == H3_LORA_FC1 || projection == H3_LORA_FC2;
        int streamed = dit->ssd_streaming ||
            (dit->attention_stream && (!mlp || dit->mlp_stream));
        if (!streamed) h3_lora_set_release(dit->loras, 0, projection);
    }
    if ((wanted_video_condition && !h3_gpu_tensor_write_f32_range(
             dit->video_input, 0, condition_video_rows,
             wanted_video_condition)) ||
        (wanted_audio_condition && !h3_gpu_tensor_write_f32_range(
             dit->audio_input, 0, condition_audio_rows,
             wanted_audio_condition))) {
        h3_dit_fail(error, error_size, "cannot write persistent DiT condition rows");
        goto failed;
    }
    h3_gpu_profile_mark(dit->gpu, "load");
    return dit;
failed:
    h3_dit_free(dit);
    return NULL;
}

h3_dit *h3_dit_load_t2va(const char *weight_directory,
                         const char *shader_source_path,
                         const h3_text_embedding *text,
                         const h3_layout *layout,
                         const h3_sigma_schedule *sigmas,
                         unsigned active_blocks,
                         unsigned core_reuse_interval,
                         int token_reduction,
                         int ssd_streaming,
                         float spatial_rope_scale,
                         int use_slower_bf16_mlp,
                         int use_slower_bf16_qkv,
                         int use_slower_bf16_attention_output,
                         int use_slower_row_major_attention_output,
                         int use_slower_unfused_int8_inputs,
                         int use_slower_unfused_qkv_rope,
                         int use_slower_scalar_qkv_rms,
                         int use_slower_uncached_int8_scales,
                         int use_slower_dynamic_fc1_k,
                         int use_slower_grouped_quantizer,
                         int use_int8_row_fc2,
                         const h3_lora *loras, size_t lora_count,
                         h3_dit_prep *prep,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size) {
    return load_dit(weight_directory, shader_source_path, text, layout, sigmas,
                    active_blocks, core_reuse_interval, token_reduction,
                    ssd_streaming,
                    spatial_rope_scale,
                    use_slower_bf16_mlp, use_slower_bf16_qkv,
                    use_slower_bf16_attention_output,
                    use_slower_row_major_attention_output,
                    use_slower_unfused_int8_inputs,
                    use_slower_unfused_qkv_rope,
                    use_slower_scalar_qkv_rms,
                    use_slower_uncached_int8_scales,
                    use_slower_dynamic_fc1_k,
                    use_slower_grouped_quantizer,
                    use_int8_row_fc2, loras, lora_count, prep,
                    NULL, 0, NULL, 0, progress, progress_opaque,
                    error, error_size);
}

h3_dit *h3_dit_load_conditioned(
                         const char *weight_directory,
                         const char *shader_source_path,
                         const h3_text_embedding *text,
                         const h3_layout *layout,
                         const h3_sigma_schedule *sigmas,
                         unsigned active_blocks,
                         unsigned core_reuse_interval,
                         int token_reduction,
                         int ssd_streaming,
                         float spatial_rope_scale,
                         int use_slower_bf16_mlp,
                         int use_slower_bf16_qkv,
                         int use_slower_bf16_attention_output,
                         int use_slower_row_major_attention_output,
                         int use_slower_unfused_int8_inputs,
                         int use_slower_unfused_qkv_rope,
                         int use_slower_scalar_qkv_rms,
                         int use_slower_uncached_int8_scales,
                         int use_slower_dynamic_fc1_k,
                         int use_slower_grouped_quantizer,
                         int use_int8_row_fc2,
                         const h3_lora *loras, size_t lora_count,
                         h3_dit_prep *prep,
                         const float *condition_video_rows,
                         size_t condition_video_elements,
                         const float *condition_audio_rows,
                         size_t condition_audio_elements,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size) {
    return load_dit(weight_directory, shader_source_path, text, layout, sigmas,
                    active_blocks, core_reuse_interval, token_reduction,
                    ssd_streaming,
                    spatial_rope_scale,
                    use_slower_bf16_mlp, use_slower_bf16_qkv,
                    use_slower_bf16_attention_output,
                    use_slower_row_major_attention_output,
                    use_slower_unfused_int8_inputs,
                    use_slower_unfused_qkv_rope,
                    use_slower_scalar_qkv_rms,
                    use_slower_uncached_int8_scales,
                    use_slower_dynamic_fc1_k,
                    use_slower_grouped_quantizer,
                    use_int8_row_fc2, loras, lora_count, prep,
                    condition_video_rows, condition_video_elements,
                    condition_audio_rows, condition_audio_elements,
                    progress, progress_opaque, error, error_size);
}
