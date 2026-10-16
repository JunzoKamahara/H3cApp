#ifndef H3_ADALN_CURVE_H
#define H3_ADALN_CURVE_H

/* Curve-form AdaLN: the 50 per-block AdaLN projections (and the final
 * layer's) only ever see silu(time embedding(t)) for a scalar t in [0, 1],
 * a smooth curve in the 2688-wide time space. Projecting that curve onto a
 * small basis V (rank k) turns each [96768, 2688] projection W into W*V
 * ([96768, k]) plus a [grid, k] table of the curve's coordinates; a row for
 * any t is the linear interpolation of the two neighbouring table rows.
 * That shrinks the ~24 GiB of dense AdaLN weights to ~0.6 GiB at k=32.
 *
 * The file is a single safetensors file using the tensor names of
 * ComfyUI's curve-form ("pruned") MiniMax H3 checkpoints, so the same
 * reader can serve those later:
 *   adaln_t_table                              F32 [grid, k]
 *   blocks.N.adaln_proj.linear.weight          F32 [96768, k]
 *   blocks.N.adaln_proj.linear.bias            F32 [96768]
 *   final_layer.adaln_proj.linear.weight       F32 [10752, k]
 *   final_layer.adaln_proj.linear.bias         F32 [10752]
 * plus "__metadata__": format "h3-adaln-curve", version, grid, rank and
 * model_id (hex of h3_weight_store_fingerprint() of the transformer it was
 * built from), so a curve is refused for any other checkpoint. */

#include "h3.h"
#include "h3_gpu.h"
#include "h3_weights.h"

#include <stddef.h>
#include <stdint.h>

#define H3_ADALN_CURVE_DEFAULT_GRID 4097u
#define H3_ADALN_CURVE_DEFAULT_RANK 32u

typedef struct h3_adaln_curve h3_adaln_curve;

/* Opens and validates a curve file. When model_id is non-NULL the file must
 * have been built from a checkpoint with that fingerprint. */
h3_adaln_curve *h3_adaln_curve_open(const char *path, const uint8_t model_id[32],
                                    char *error, size_t error_size);
/* Curve-form checkpoints (ComfyUI's "pruned" files, unsloth's GGUFs) carry
 * the curve themselves - adaln_t_table and [outputs, k] projections, F32 or
 * F16 - instead of a time embedder and dense projections. in_store says
 * whether a transformer's weights are of that form; from_store reads the
 * curve from them (the store must outlive the curve). */
int h3_adaln_curve_in_store(const h3_weight_store *store);
h3_adaln_curve *h3_adaln_curve_from_store(const h3_weight_store *store,
                                          char *error, size_t error_size);
void h3_adaln_curve_free(h3_adaln_curve *curve);
uint32_t h3_adaln_curve_rank(const h3_adaln_curve *curve);
uint32_t h3_adaln_curve_grid(const h3_adaln_curve *curve);

/* The curve file's identity (name, size, mtime - like the weight store's
 * fingerprint), for keying cached schedules built from it. */
void h3_adaln_curve_fingerprint(const h3_adaln_curve *curve, uint8_t out[32]);

/* Interpolated coordinates for each time: out is [count, rank]. */
void h3_adaln_curve_coordinates(const h3_adaln_curve *curve,
                                const float *times, uint32_t count,
                                float *out);

/* ComfyUI's interpolation, exposed for tests: t is clamped to [0, 1],
 * pos = t * (grid - 1), the lower row is min(floor(pos), grid - 2), and the
 * result is lerp(lower, upper, pos - lower). out has rank values. */
void h3_adaln_curve_lerp(const float *table, uint32_t grid, uint32_t rank,
                         float t, float *out);

/* A projection's [rows, rank] weight or [rows] bias as an F32 GPU tensor.
 * prefix is "blocks.N.adaln_proj.linear" or "final_layer.adaln_proj.linear". */
h3_gpu_tensor *h3_adaln_curve_load_weight(const h3_adaln_curve *curve,
                                          h3_gpu *gpu, const char *prefix,
                                          uint32_t rows, char *error,
                                          size_t error_size);
h3_gpu_tensor *h3_adaln_curve_load_bias(const h3_adaln_curve *curve,
                                        h3_gpu *gpu, const char *prefix,
                                        uint32_t rows, char *error,
                                        size_t error_size);

#endif
