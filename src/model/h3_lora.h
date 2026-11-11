#ifndef H3_LORA_H
#define H3_LORA_H

/* LoRA adapters for the MiniMax-H3 DiT.
 *
 * Loading normalizes every published H3 LoRA layout onto the engine's own
 * weights. Those are the official checkpoint's, which differ from the
 * ComfyUI repack in one place: attn.qkv_proj rows are interleaved per head
 * ([q_h, k_h, v_h] for each of the 56 heads), while ComfyUI - and so every
 * LoRA trained or converted for it - stores them contiguously ([Q; K; V]).
 * diffusers/PEFT adapters split the projection into to_q/to_k/to_v (each
 * contiguous) and store ff.net.0.proj as [value; gate] where mlp.fc1 is
 * [gate; value]. out_proj and fc2 match everywhere.
 *
 * Adapters are applied by adding delta = sum_i s_i * B_i @ A_i to each
 * projection's weight on the GPU when the block is loaded (resident
 * weights) or every time it is streamed in (int8 attention cache, SSD
 * streaming). Stacked adapters are concatenated along the rank, so N
 * adapters cost one matmul per row group, not N. */

#include "h3.h"
#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

enum {
    H3_LORA_HIDDEN = 5376,
    H3_LORA_HEADS = 56,
    H3_LORA_HEAD_DIM = 128,
    H3_LORA_INNER = H3_LORA_HEADS * H3_LORA_HEAD_DIM,
    H3_LORA_FFN = 14336,
    H3_LORA_DIT_BLOCKS = 50,
    H3_LORA_REFINER_BLOCKS = 2,
    H3_MAX_LORAS = 8,
};

enum { H3_LORA_QKV = 0, H3_LORA_OUT = 1, H3_LORA_FC1 = 2, H3_LORA_FC2 = 3,
       H3_LORA_PROJECTIONS = 4,
       /* Not a block projection: the curve-form AdaLN projection
        * (blocks.N.adaln_proj.linear, [96768, k] on the k-dim curve), which
        * LMS Sharp's "curve branches" patch. Kept outside the projection
        * arrays; see h3_lora_set_adaln_delta. */
       H3_LORA_ADALN = 4 };

/* Which rows of the projection a tensor's lora_B covers, in its own file's
 * layout: the whole projection, one of the diffusers to_q/to_k/to_v parts,
 * or diffusers' [value; gate] FC1. */
typedef enum {
    H3_LORA_PART_ALL = 0,
    H3_LORA_PART_Q = 1,
    H3_LORA_PART_K = 2,
    H3_LORA_PART_V = 3,
    H3_LORA_PART_FC1_VALUE_GATE = 4,
} h3_lora_part;

typedef enum {
    H3_LORA_ROLE_A = 1,
    H3_LORA_ROLE_B = 2,
    H3_LORA_ROLE_ALPHA = 3,
} h3_lora_role;

typedef enum {
    H3_LORA_STYLE_NATIVE = 0,
    H3_LORA_STYLE_COMFYUI = 1,
    H3_LORA_STYLE_DIFFUSERS = 2,
    H3_LORA_STYLE_KOHYA = 3,
} h3_lora_style;

typedef struct {
    int refiner;          /* 0: DiT blocks, 1: token refiner */
    int block;
    int projection;       /* H3_LORA_QKV .. H3_LORA_FC2 */
    h3_lora_part part;
    h3_lora_role role;
    h3_lora_style style;
} h3_lora_key;

/* Classifies one adapter tensor name. Returns 1 for a factor or alpha of a
 * supported block projection, 0 for anything else. */
int h3_lora_parse_key(const char *name, h3_lora_key *key);

/* Engine row that row `row` of a projection row group lands on. QKV has
 * three groups (q, k, v; group_rows = INNER each) laid out per head in the
 * engine; every other projection is one identity group. */
uint32_t h3_lora_engine_row(int projection, int group, uint32_t row);
uint32_t h3_lora_group_count(int projection);
uint32_t h3_lora_group_rows(int projection);
uint32_t h3_lora_columns(int projection);
uint32_t h3_lora_rows(int projection);

typedef struct h3_lora_set h3_lora_set;

/* Reads and normalizes every adapter and uploads its factors to `gpu`.
 * Returns NULL with `error` set on unreadable or shape-incompatible files. */
h3_lora_set *h3_lora_set_load(h3_gpu *gpu, const h3_lora *loras,
                              size_t count, char *error, size_t error_size);
void h3_lora_set_free(h3_lora_set *set);

int h3_lora_set_covers(const h3_lora_set *set, int refiner, unsigned block,
                       int projection);

/* Frees one projection's factors in every block once they are no longer
 * needed - after resident weights were patched at load, nothing streams
 * that projection again. Later applies to it become no-ops. */
void h3_lora_set_release(h3_lora_set *set, int refiner, int projection);

/* Add the stacked delta to one projection's weight, encoding GPU work into
 * the current command (the caller owns h3_gpu_begin/submit). The BF16 form
 * is for resident and SSD-streamed weights; the int8 form adds to a per-row
 * int8 weight in place, keeping its scales. Both round stochastically: a
 * LoRA delta is far below either format's step, and round-to-nearest would
 * silently drop it. No-ops when nothing covers the projection. */
int h3_lora_set_apply_bf16(h3_lora_set *set, h3_gpu *gpu, int refiner,
                           unsigned block, int projection,
                           h3_gpu_tensor *weight);
int h3_lora_set_apply_int8(h3_lora_set *set, h3_gpu *gpu, int refiner,
                           unsigned block, int projection,
                           h3_gpu_tensor *weight, h3_gpu_tensor *scales);
/* The stacked AdaLN curve-branch delta of one block, [96768, rank] F32 on
 * the host (sum of strength * B @ A over the adapters), or NULL when no
 * adapter has one; *rank receives the curve dimension the factors expect.
 * The AdaLN schedule adds it to the checkpoint's curve projection weight
 * (h3_dit_schedule_precompute_curve). H3_LORA_NO_ADALN=1 ignores them. */
const float *h3_lora_set_adaln_delta(const h3_lora_set *set, unsigned block,
                                     uint32_t *rank);
int h3_lora_set_has_adaln(const h3_lora_set *set);

#endif
