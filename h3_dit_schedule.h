#ifndef H3_DIT_SCHEDULE_H
#define H3_DIT_SCHEDULE_H

#include "h3_gpu.h"
#include "h3_host.h"
#include "h3_weights.h"

#include <stddef.h>
#include <stdint.h>

#define H3_DIT_BLOCKS 50u
#define H3_DIT_HIDDEN 5376u
#define H3_DIT_TIME_DIM 2688u
#define H3_DIT_MODALITIES 3u
#define H3_DIT_ADALN_SLOTS 6u

typedef struct h3_dit_schedule h3_dit_schedule;

typedef void (*h3_dit_schedule_progress)(int completed_blocks,
                                         int total_blocks, void *opaque);

/* Materialize every per-step AdaLN value. This intentionally submits one
 * projection at a time, so a 498 MiB block projection is released before the
 * next is loaded. */
h3_dit_schedule *h3_dit_schedule_precompute(
    const h3_weight_store *weights, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, int visual_condition,
    int audio_condition,
    h3_dit_schedule_progress progress, void *progress_opaque,
    char *error, size_t error_size);
/* The same schedule from a curve-form AdaLN file (h3_adaln_curve.h)
 * instead of the dense projections: reads ~0.6 GiB instead of ~24 GiB, and
 * matches the dense values to well below their own BF16 rounding (not bit
 * for bit). */
typedef struct h3_adaln_curve h3_adaln_curve;
/* loras: an adapter set whose AdaLN curve-branch deltas (if any) are added
 * to the curve projections first; NULL for none. */
typedef struct h3_lora_set h3_lora_set;
h3_dit_schedule *h3_dit_schedule_precompute_curve(
    const h3_lora_set *loras,
    const h3_adaln_curve *curve, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, int visual_condition,
    int audio_condition,
    h3_dit_schedule_progress progress, void *progress_opaque,
    char *error, size_t error_size);
void h3_dit_schedule_free(h3_dit_schedule *schedule);

/* Diagnostic: moves a random `fraction` of the schedule's BF16 values one
 * BF16 step up or down (deterministic for a seed) - a rounding-sized
 * perturbation to compare other schedule changes against. */
int h3_dit_schedule_perturb_ulps(h3_dit_schedule *schedule, double fraction,
                                 uint64_t seed);

/* silu(time embedding(t)) exactly as the dense AdaLN path feeds it to the
 * projections (BF16, widened to F32 here), for count times; out is
 * [count, H3_DIT_TIME_DIM]. Used to build curve-form AdaLN files. */
int h3_dit_time_silu(const h3_weight_store *weights, h3_gpu *gpu,
                     const float *times, uint32_t count, float *out,
                     char *error, size_t error_size);

/* A host copy of a whole (unpruned) schedule, for rebuilding it on another
 * GPU context without the 50 AdaLN projections: the values depend only on
 * the weights, the sigma schedule and which conditions are present. */
typedef struct h3_dit_schedule_host h3_dit_schedule_host;
h3_dit_schedule_host *h3_dit_schedule_export(const h3_dit_schedule *schedule);
h3_dit_schedule *h3_dit_schedule_import(const h3_dit_schedule_host *host,
                                        h3_gpu *gpu, char *error,
                                        size_t error_size);
size_t h3_dit_schedule_host_bytes(const h3_dit_schedule_host *host);
void h3_dit_schedule_host_free(h3_dit_schedule_host *host);

int h3_dit_schedule_steps(const h3_dit_schedule *schedule);
uint32_t h3_dit_schedule_time_rows(const h3_dit_schedule *schedule);
uint32_t h3_dit_schedule_video_row(const h3_dit_schedule *schedule, int step);
uint32_t h3_dit_schedule_audio_row(const h3_dit_schedule *schedule, int step);
uint32_t h3_dit_schedule_visual_condition_row(
    const h3_dit_schedule *schedule, int step);
uint32_t h3_dit_schedule_audio_condition_row(
    const h3_dit_schedule *schedule, int step);
const h3_gpu_tensor *h3_dit_schedule_block(const h3_dit_schedule *schedule,
                                           unsigned block);
double h3_dit_schedule_gate_score(const h3_dit_schedule *schedule,
                                  unsigned block);
void h3_dit_schedule_prune(h3_dit_schedule *schedule,
                           const uint8_t *active_blocks, size_t count);
const h3_gpu_tensor *h3_dit_schedule_final(const h3_dit_schedule *schedule);

/* Build the row map consumed by the fused AdaLN/gate kernels. text_tags may be
 * NULL (all tag 1), or one tag per text row. Qwen vision presentation spans use
 * tag 0. Segment kinds select target/condition timesteps and modality tags. */
int h3_dit_schedule_row_map(const h3_dit_schedule *schedule, int step,
                            const h3_layout *layout,
                            const uint8_t *text_tags, size_t text_tag_count,
                            uint32_t *rows, size_t row_count);

#endif
