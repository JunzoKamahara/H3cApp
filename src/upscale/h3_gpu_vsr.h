/* GPU kernels of the SwiftVR upscaler (h3_gpu_vsr.m; shaders in h3_shaders_vsr.metal).
 * Part of the upscale module: the engine's own h3_gpu.h does not know these.
 * SeedVR2's VAE reuses the tap convolution and the grid helpers. */
#ifndef H3_GPU_VSR_H
#define H3_GPU_VSR_H

#include "h3_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One-step VSR helpers (UPSCALE_PLAN §6.5), channels-last F32 [B,T,H,W,C].
 * h3_gpu_vsr_pad_f32 zero-pads in space and prepends depth_front frames
 * taken from `previous` ([B,depth_front,H,W,C], the MemBlock state) or zero
 * when it is NULL. space_to_depth/depth_to_space are pixel (un)shuffle by 2
 * with PyTorch's channel order; `channels` is the input count for
 * space_to_depth and the output count for depth_to_space. split_frames
 * turns [B,T,H,W,2C] into [B,2T,H,W,C] (TGrow). The pixel converters take
 * 8-bit BGRA/RGB24 frames as I8 tensors; pixels_to_f32 resizes with torch's
 * bilinear (align_corners=False) and zero-pads to (out_w+pad_w, out_h+pad_h). */
/* One-step VSR: the Wan block on BF16 activations with F32 constants.
 * ln_mod: LayerNorm(no affine) [* weight + bias] [* (1 + scale) + shift],
 * NULL vectors are skipped. qkv_prepare (in place on [rows, segments*dim]):
 * + bias, RMSNorm across heads on segment 0 (q) and 1 (k) with weights,
 * then interleaved RoPE on q and k from (cos, sin) [rows, head_dim/2];
 * segments 1 handles a lone q. gather_rows: out[r] = in[index[r],
 * col_offset .. +width] with a U32 index (also the scatter, with the owner
 * map). bias_gate_add: out = residual + (y + bias) * gate, optional parts
 * NULL; out may be residual. patchify/unpatchify: F32 latent [T,h,w,C] <->
 * BF16 tokens [T*h/2*w/2, 4C] in the Wan (c,ph,pw) input and (ph,pw,c)
 * output orders; unpatchify writes z_lq - (v + bias). */
int h3_gpu_vsr_ln_mod_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, const h3_gpu_tensor *scale,
                           const h3_gpu_tensor *shift, const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows, uint32_t width,
                           float epsilon);
int h3_gpu_vsr_qkv_prepare_bf16(h3_gpu *gpu, h3_gpu_tensor *qkv,
                                const h3_gpu_tensor *bias,
                                const h3_gpu_tensor *norm_q_weight,
                                const h3_gpu_tensor *norm_k_weight,
                                const h3_gpu_tensor *rope_cos,
                                const h3_gpu_tensor *rope_sin, uint32_t rows,
                                uint32_t dim, uint32_t heads, uint32_t head_dim,
                                uint32_t segments, float epsilon);
int h3_gpu_vsr_gather_rows_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                const h3_gpu_tensor *input, const h3_gpu_tensor *index,
                                uint32_t rows, uint32_t width, uint32_t in_stride,
                                uint32_t col_offset);
/* out[index[r]] = in[r] for r < rows (the gather's inverse). */
int h3_gpu_vsr_scatter_rows_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *index, uint32_t rows, uint32_t width);
int h3_gpu_vsr_bias_gate_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *y, const h3_gpu_tensor *bias,
                                  const h3_gpu_tensor *gate, const h3_gpu_tensor *residual,
                                  uint32_t rows, uint32_t width);
int h3_gpu_vsr_gelu_bias_bf16(h3_gpu *gpu, h3_gpu_tensor *x, const h3_gpu_tensor *bias,
                              uint32_t rows, uint32_t width);
int h3_gpu_vsr_patchify_bf16(h3_gpu *gpu, h3_gpu_tensor *tokens,
                             const h3_gpu_tensor *latent, uint32_t frames,
                             uint32_t height, uint32_t width, uint32_t channels);
int h3_gpu_vsr_unpatchify_sub_f32(h3_gpu *gpu, h3_gpu_tensor *z_hq,
                                  const h3_gpu_tensor *z_lq, const h3_gpu_tensor *v,
                                  const h3_gpu_tensor *bias, uint32_t frames,
                                  uint32_t height, uint32_t width, uint32_t channels);
/* SDPA over a batch of equal windows ([batch, sequence, heads, head_dim]
 * row-major BF16), and cross-attention with kv_sequence keys/values. */
int h3_gpu_sdpa_batched_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                             const h3_gpu_tensor *value, uint32_t batch,
                             uint32_t sequence, uint32_t heads,
                             uint32_t head_dim, float scale);
/* Batched SDPA with separate query and key/value lengths: q [batch,
 * sequence, heads, head_dim], k/v [batch, kv_sequence, heads, head_dim]. */
int h3_gpu_sdpa_batched_kv_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                                const h3_gpu_tensor *value, uint32_t batch,
                                uint32_t sequence, uint32_t kv_sequence, uint32_t heads,
                                uint32_t head_dim, float scale);
int h3_gpu_sdpa_cross_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value, uint32_t sequence,
                           uint32_t kv_sequence, uint32_t heads,
                           uint32_t head_dim, float scale);

int h3_gpu_vsr_pad_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input, const h3_gpu_tensor *previous,
                       uint32_t batch, uint32_t depth, uint32_t height,
                       uint32_t width, uint32_t channels, uint32_t depth_front,
                       uint32_t pad_h, uint32_t pad_w);
int h3_gpu_vsr_upsample2_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input, uint32_t batch,
                             uint32_t depth, uint32_t height, uint32_t width,
                             uint32_t channels);
int h3_gpu_vsr_space_to_depth_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *input, uint32_t batch,
                                  uint32_t depth, uint32_t height, uint32_t width,
                                  uint32_t channels);
int h3_gpu_vsr_depth_to_space_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *input, uint32_t batch,
                                  uint32_t depth, uint32_t height, uint32_t width,
                                  uint32_t channels);
int h3_gpu_vsr_split_frames_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                const h3_gpu_tensor *input, uint32_t batch,
                                uint32_t depth, uint32_t height, uint32_t width,
                                uint32_t channels);
int h3_gpu_vsr_clamp_tanh_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                              const h3_gpu_tensor *input, uint32_t elements);
/* ReAE convolutions on M5 TensorOps: a "padded grid" of BF16 rows
 * [guard | T x (H+2) x (W+2) | guard] with channels padded to a multiple of
 * 64 (x_rows physical rows, guard >= W+3, trailing room >= guard + 128).
 * Up to 27 taps (3x3x3: a temporal tap is an offset of whole frames).
 * tapconv: out[r] = sum over taps of x[r + offset] . W_tap^T (x1/w1: a
 * second input accumulated the same way) + bias (+ residual) (ReLU). The
 * weights are [taps * out_ch, in_ch] BF16 row-major, tap-major. */
int h3_gpu_vsr_tapconv_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *x0, const h3_gpu_tensor *x1,
                            const h3_gpu_tensor *w0, const h3_gpu_tensor *w1,
                            const h3_gpu_tensor *bias, const h3_gpu_tensor *residual,
                            uint32_t rows, uint32_t x_rows, uint32_t guard,
                            uint32_t in_ch, uint32_t out_ch, uint32_t taps,
                            const int32_t *offsets, int relu);
/* The same convolution of x0 where the taps in `tap_mask` (bit t) read
 * `state` instead - a buffer of `state_rows` rows in x0's channel layout,
 * their offsets counted from the same output row (SeedVR2's causal state:
 * the frames before a video frame, kept apart from it instead of in slots
 * ahead of it). Rows [guard, guard + rows) of x0 are computed, so a frame
 * range of a grid is `guard` moved to its first row. */
int h3_gpu_vsr_tapconv_state_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *x0, const h3_gpu_tensor *state, uint32_t state_rows,
                                  const h3_gpu_tensor *w, const h3_gpu_tensor *bias, const h3_gpu_tensor *residual,
                                  uint32_t rows, uint32_t x_rows, uint32_t guard,
                                  uint32_t in_ch, uint32_t out_ch, uint32_t taps,
                                  const int32_t *offsets, uint32_t tap_mask, int relu);
int h3_gpu_vsr_zero_rows_bf16(h3_gpu *gpu, h3_gpu_tensor *x, uint32_t first,
                              uint32_t count, uint32_t width);
int h3_gpu_vsr_pack_grid_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                              uint32_t frames, uint32_t height, uint32_t width,
                              uint32_t channels, uint32_t padded_channels, uint32_t guard,
                              uint32_t x_rows, int zero_guards);
int h3_gpu_vsr_unpack_grid_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                               uint32_t frames, uint32_t height, uint32_t width,
                               uint32_t channels, uint32_t padded_channels, uint32_t guard,
                               uint32_t x_rows);
int h3_gpu_vsr_zero_border_bf16(h3_gpu *gpu, h3_gpu_tensor *x, uint32_t frames,
                                uint32_t height, uint32_t width, uint32_t padded_channels,
                                uint32_t guard, uint32_t x_rows);
/* The grid geometry the layer kernels below take: `channels` real
 * channels in `padded_channels` (a multiple of 64), frames x (height+2) x
 * (width+2) rows from `guard`, x_rows physical rows. Every kernel zeroes
 * its output's guard and border rows. */
typedef struct {
    uint32_t frames, height, width, channels, padded_channels, guard, x_rows;
} h3_vsr_grid;
/* F32 frames [T, 2h, 2w, 3] -> grid (12 channels): pixel_unshuffle(2). */
int h3_gpu_vsr_pack_grid_s2d_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_vsr_grid *out,
                                  const h3_gpu_tensor *frames);
/* F32 [T, h, w, C] -> grid with tanh(x / 3) * 3. */
int h3_gpu_vsr_pack_grid_clamp_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_vsr_grid *out,
                                    const h3_gpu_tensor *input);
/* grid (12 channels) -> F32 frames [T, 2h, 2w, 3] in [0, 1]: pixel_shuffle(2). */
int h3_gpu_vsr_unpack_grid_d2s_f32(h3_gpu *gpu, h3_gpu_tensor *frames, const h3_gpu_tensor *input,
                                   const h3_vsr_grid *in);
int h3_gpu_vsr_grid_upsample2_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_vsr_grid *out,
                                   const h3_gpu_tensor *input, const h3_vsr_grid *in);
/* grid [T, h, w, 2C] -> grid [2T, h, w, C]: frame 2t+k = channels k*C.. of frame t. */
int h3_gpu_vsr_grid_split_frames_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_vsr_grid *out,
                                      const h3_gpu_tensor *input, const h3_vsr_grid *in);
/* out(t, y, x) = in(t*stride_t, y*stride_y, x*stride_x). */
int h3_gpu_vsr_grid_select_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_vsr_grid *out,
                                const h3_gpu_tensor *input, const h3_vsr_grid *in,
                                uint32_t stride_t, uint32_t stride_y, uint32_t stride_x);
/* Per-channel means of an F32 [frames, height, width, 3] image over its
 * real (unpadded) top-left real_height x real_width area; begins, submits
 * and waits on its own. */
int h3_gpu_vsr_channel_means_f32(h3_gpu *gpu, const h3_gpu_tensor *input, uint32_t frames,
                                 uint32_t height, uint32_t width, uint32_t real_height,
                                 uint32_t real_width, float means[3]);
/* x[p, c] += offsets[c] for `pixels` RGB pixels (inside begin/submit). */
int h3_gpu_vsr_add_channel_offsets_f32(h3_gpu *gpu, h3_gpu_tensor *x, uint32_t pixels, const float offsets[3]);
/* 8-bit frames -> the tile (tile_x, tile_y, tile_w, tile_h) of their
 * bilinear resize to out_w x out_h as F32 RGB [frames, tile_h, tile_w, 3];
 * pixels at or past out_w / out_h are zero (the padding). */
int h3_gpu_vsr_pixels_to_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *pixels, uint32_t frames,
                             uint32_t in_w, uint32_t in_h, uint32_t in_stride,
                             int bgra, uint32_t out_w, uint32_t out_h,
                             uint32_t tile_x, uint32_t tile_y,
                             uint32_t tile_w, uint32_t tile_h);
/* F32 RGB frames first_frame .. first_frame + frames of [.., full_h, full_w, 3]
 * -> 8-bit frames cropped to crop_w x crop_h. */
int h3_gpu_vsr_f32_to_pixels(h3_gpu *gpu, h3_gpu_tensor *pixels,
                             const h3_gpu_tensor *input, uint32_t first_frame,
                             uint32_t frames, uint32_t full_w, uint32_t full_h,
                             uint32_t crop_w, uint32_t crop_h, uint32_t out_stride,
                             int bgra, int round_mode);
/* composite [frames, out_h, out_w, 3] += tile frames first_frame .. of
 * [.., tile_h, tile_w, 3] * weights [tile_h, tile_w], placed at (tile_x,
 * tile_y) and clipped to the composite (inside begin/submit). */
int h3_gpu_vsr_tile_accumulate_f32(h3_gpu *gpu, h3_gpu_tensor *composite,
                                   uint32_t out_w, uint32_t out_h,
                                   const h3_gpu_tensor *tile, uint32_t first_frame,
                                   uint32_t frames, uint32_t tile_w, uint32_t tile_h,
                                   uint32_t tile_x, uint32_t tile_y,
                                   const h3_gpu_tensor *weights);

/* gelu_tanh(x + bias) quantized per row to int8 (+ F32 scales) in one
 * pass; columns a multiple of 4, at most 8192. */
int h3_gpu_vsr_gelu_quantize_int8_rows(h3_gpu *gpu, h3_gpu_tensor *quantized, h3_gpu_tensor *scales,
                                       const h3_gpu_tensor *input, const h3_gpu_tensor *bias,
                                       uint32_t rows, uint32_t columns);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
