#ifndef H3_HYPERFLOW_H
#define H3_HYPERFLOW_H

/* HyperFlow H3 Flash (Video Rebirth, a Model Derivative of MiniMax-H3
 * under the MiniMax H3 Community License): a 13.7B (FL2VA) / 13.9B (Ref2VA)
 * compression of the H3 transformer with its own 4/6/8-step sampler. Its
 * GGUF checkpoints use ComfyUI's H3 tensor names and differ in three ways:
 *
 *  - pruned blocks: the remaining blocks are renumbered without gaps
 *    (FL2VA keeps 47 of 50, Ref2VA 49), so blocks.N runs 0..count-1;
 *  - low-rank linears: some projections are blocks.N.<proj>.down.weight
 *    [rank, in] and .up.weight [out, rank] (y = up(down(x)), no bias), and
 *    a factored attention splits Q, K and V into attn.{q,k,v}_proj with
 *    ranks of their own instead of the fused attn.qkv_proj;
 *  - two-time AdaLN: time_embedder and endpoint_time_embedder (256 ->
 *    5376 -> time_dim, 16 in the released files) embed the step's time t
 *    and its endpoint r; temb = E_t(t) + gate * (E_r(r) - E_t(t)) with the
 *    F32 scalar hyperflow_gate, and the AdaLN projections are
 *    [outputs, time_dim].
 *
 * h3_hyperflow_inspect reads only tensor headers (and the two small F32
 * tensors hyperflow_gate / hyperflow_sigmas). */

#include "h3_weights.h"

#include <stddef.h>
#include <stdint.h>

#define H3_HYPERFLOW_MAX_BLOCKS 50u
#define H3_HYPERFLOW_MAX_SIGMAS 16u

typedef enum {
    H3_HF_Q = 0,
    H3_HF_K,
    H3_HF_V,
    H3_HF_OUT,
    H3_HF_FC1,
    H3_HF_FC2,
    H3_HF_LINEARS
} h3_hyperflow_linear;

typedef struct {
    int two_time;            /* 0: an ordinary H3 checkpoint (the rest is unset) */
    float gate;
    uint32_t time_dim;
    uint32_t block_count;
    /* rank[block][linear]; 0 = dense. Q, K and V are either all factored
     * or all dense (the fused qkv_proj). */
    uint32_t rank[H3_HYPERFLOW_MAX_BLOCKS][H3_HF_LINEARS];
    uint32_t max_rank;       /* the largest of all of them */
    uint32_t factored_blocks;/* blocks with at least one factored linear */
    uint32_t sigma_count;    /* hyperflow_sigmas (0 when absent) */
    float sigmas[H3_HYPERFLOW_MAX_SIGMAS];
} h3_hyperflow_info;

/* 1 on success: a valid HyperFlow checkpoint (two_time = 1) or any other
 * one (two_time = 0, block_count = 50). 0 with an error for a checkpoint
 * that has the two-time embedder but an inconsistent layout. */
int h3_hyperflow_inspect(const h3_weight_store *store, h3_hyperflow_info *info,
                         char *error, size_t error_size);

/* Linear name stem for messages and lookups ("attn.q_proj", ...). */
const char *h3_hyperflow_linear_name(h3_hyperflow_linear linear);

/* 1 when any of a block's linears in [first, last] is factored. */
int h3_hyperflow_block_factored(const h3_hyperflow_info *info, unsigned block,
                                h3_hyperflow_linear first,
                                h3_hyperflow_linear last);

#endif
