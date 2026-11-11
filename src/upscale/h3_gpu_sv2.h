/* GPU kernels of the SeedVR2 upscaler (h3_gpu_sv2.m; shaders in h3_shaders_seedvr2.metal).
 * Part of the upscale module. */
#ifndef H3_GPU_SV2_H
#define H3_GPU_SV2_H

#include "h3_gpu.h"
#include "h3_gpu_vsr.h"

#ifdef __cplusplus
extern "C" {
#endif

/* MLX affine 8-bit (group of `group` columns, F16 scales/biases as raw
 * bytes in I8 tensors, packed U32 [rows, columns / 4]) -> BF16 [rows, columns]. */
int h3_gpu_sv2_dequant_q8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                               const h3_gpu_tensor *packed, const h3_gpu_tensor *scales,
                               const h3_gpu_tensor *biases, uint32_t rows,
                               uint32_t columns, uint32_t group);
/* Per-row int8 (h3_gpu_quantize_rows_bf16_int8: I8 [rows, columns], F32
 * scales [rows]) -> BF16 [rows, columns]; columns a multiple of 4. */
int h3_gpu_sv2_dequant_rows_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *quantized,
                                      const h3_gpu_tensor *scales, uint32_t rows, uint32_t columns);
/* The reverse for `rows` rows starting at in_row0 of a BF16 row buffer,
 * into rows from out_row0 of the int8 one (and its scales). */
int h3_gpu_sv2_quantize_rows_bf16_int8(h3_gpu *gpu, h3_gpu_tensor *quantized, h3_gpu_tensor *scales,
                                       const h3_gpu_tensor *input, uint32_t in_row0, uint32_t out_row0,
                                       uint32_t rows, uint32_t columns);
/* 4-bit affine in groups of 64 columns: `packed` I8 [rows, columns / 2],
 * `params` F32 [rows, columns / 64, 2] (scale, minimum); columns a
 * multiple of 64. */
int h3_gpu_sv2_quantize_rows_bf16_int4(h3_gpu *gpu, h3_gpu_tensor *packed, h3_gpu_tensor *params,
                                       const h3_gpu_tensor *input, uint32_t in_row0, uint32_t out_row0,
                                       uint32_t rows, uint32_t columns);
int h3_gpu_sv2_dequant_rows_int4_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *packed,
                                      const h3_gpu_tensor *params, uint32_t rows, uint32_t columns);
/* out = x / rms(x) * scale + shift (F32 vectors, NULL = 1 / 0). */
int h3_gpu_sv2_rms_mod_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                            const h3_gpu_tensor *scale, const h3_gpu_tensor *shift,
                            uint32_t rows, uint32_t width, float epsilon);
/* The window gather with per-head RMSNorm of q/k and RoPE; index I32
 * [rows] (>= 0 video token, < 0 text token -1 - index), cos/sin F32
 * [rows, pairs], norms F32 [head_dim], outputs BF16 [rows, heads*head_dim]. */
int h3_gpu_sv2_qkv_window_bf16(h3_gpu *gpu, h3_gpu_tensor *q_out, h3_gpu_tensor *k_out,
                               h3_gpu_tensor *v_out, const h3_gpu_tensor *qkv_vid,
                               const h3_gpu_tensor *qkv_txt, const h3_gpu_tensor *index,
                               const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
                               const h3_gpu_tensor *norm_q_vid, const h3_gpu_tensor *norm_k_vid,
                               const h3_gpu_tensor *norm_q_txt, const h3_gpu_tensor *norm_k_txt,
                               uint32_t rows, uint32_t heads, uint32_t head_dim,
                               uint32_t pairs, float epsilon);
/* out[j] = mean_w in[base[w] + j] (base I32 [windows]). */
int h3_gpu_sv2_txt_mean_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                             const h3_gpu_tensor *base, uint32_t windows,
                             uint32_t txt_rows, uint32_t width);
/* Zero padding of F32 [H, W, C]. */
int h3_gpu_sv2_pad_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                       uint32_t height, uint32_t width, uint32_t channels,
                       uint32_t top, uint32_t bottom, uint32_t left, uint32_t right);
/* in [H, W, C*4*temporal] -> out [2H, 2W, C] (the first temporal slice). */
int h3_gpu_sv2_shuffle_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                           uint32_t height, uint32_t width, uint32_t channels,
                           uint32_t temporal);
/* The temporal Upsample3D: in [T, H, W, C*4*tf] -> out [T', 2H, 2W, C],
 * frame f*tf+z from slice z (b (x y z c) f h w -> b c (f z) (h x) (w y));
 * drop_head (a clip's first slice) leaves out frame 1 when tf > 1.
 * T' = h3_gpu_sv2_shuffle_frames(T, tf, drop_head). */
uint32_t h3_gpu_sv2_shuffle_frames(uint32_t frames, uint32_t temporal, int drop_head);
int h3_gpu_sv2_shuffle3d_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                             uint32_t frames, uint32_t height, uint32_t width, uint32_t channels,
                             uint32_t temporal, int drop_head);
/* The causal convolution's input: front frames then F32 [T, H, W, C]
 * zero-padded in space -> [front + T, H+top+bottom, W+left+right, C]. The
 * front frames are the last `front` of `previous` ([prev_frames, H, W, C],
 * the previous slice's state) or, when it is NULL, copies of frame 0. */
int h3_gpu_sv2_pad3d_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                         const h3_gpu_tensor *previous, uint32_t prev_frames,
                         uint32_t frames, uint32_t height, uint32_t width, uint32_t channels,
                         uint32_t front, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right);
/* GroupNorm (no SiLU) on F32 [frames, H, W, C]. */
int h3_gpu_sv2_group_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                              const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
                              uint32_t frames, uint32_t height, uint32_t width,
                              uint32_t channels, uint32_t groups, float epsilon);
/* 8-bit pixels (I8 tensor, in_stride bytes per row) resized with PIL's
 * bicubic to out_w x out_h inside an F32 [full_h, full_w, 3] frame in
 * [-1, 1]; outside out_w x out_h the value is pad_value. */
int h3_gpu_sv2_bicubic_pixels_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *pixels, uint32_t in_w, uint32_t in_h,
                                  uint32_t in_stride, int bgra, uint32_t out_w, uint32_t out_h,
                                  uint32_t full_w, uint32_t full_h, float pad_value);
/* F32 [full_h, full_w, 3] in [-1, 1] -> 8-bit, cropped, rounded. */
int h3_gpu_sv2_f32_to_pixels(h3_gpu *gpu, h3_gpu_tensor *pixels, const h3_gpu_tensor *input,
                             uint32_t full_w, uint32_t full_h, uint32_t crop_w,
                             uint32_t crop_h, uint32_t out_stride, int bgra);
int h3_gpu_sv2_wavelet_blur_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                                uint32_t height, uint32_t width, int radius);
int h3_gpu_sv2_rgb_to_lab_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                              uint32_t pixels);
/* luts F32 [3, bins]; channel c's bin b covers minimum[c] + b * step[c]. */
int h3_gpu_sv2_lab_match_to_rgb_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *lab,
                                    const h3_gpu_tensor *luts, uint32_t pixels, uint32_t bins,
                                    float l_weight, const float minimum[3], const float step[3]);

/* SeedVR2 VAE on the TensorOps grid (the ReAE's padded BF16 grid): GroupNorm
 * (+ SiLU) over one frame's interior pixels in three chunked passes (in
 * place allowed), the Upsample3D shuffle, and the stride-2 subsample. */
int h3_gpu_sv2_grid_group_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                                    const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
                                    uint32_t height, uint32_t width, uint32_t channels,
                                    uint32_t padded, uint32_t guard, uint32_t x_rows,
                                    uint32_t groups, float epsilon, int silu);
int h3_gpu_sv2_grid_shuffle_bf16(h3_gpu *gpu, h3_gpu_tensor *output, uint32_t out_padded,
                                 uint32_t out_guard, const h3_gpu_tensor *input, uint32_t in_height,
                                 uint32_t in_width, uint32_t in_padded, uint32_t in_guard,
                                 uint32_t channels, uint32_t temporal);
int h3_gpu_sv2_grid_select_bf16(h3_gpu *gpu, h3_gpu_tensor *output, uint32_t out_height,
                                uint32_t out_width, uint32_t out_guard, const h3_gpu_tensor *input,
                                uint32_t in_height, uint32_t in_width, uint32_t in_guard,
                                uint32_t padded, uint32_t phase_y, uint32_t phase_x);
/* The same on frames that follow each other from the guard: the temporal
 * shuffle (h3_gpu_sv2_shuffle3d_f32's rule) and out(t, y, x) =
 * in(t*stride_t + phase_t, 2y + phase_y, 2x + phase_x). */
int h3_gpu_sv2_grid_shuffle3d_bf16(h3_gpu *gpu, h3_gpu_tensor *output, uint32_t out_padded,
                                   uint32_t out_guard, const h3_gpu_tensor *input, uint32_t in_frames,
                                   uint32_t in_height, uint32_t in_width, uint32_t in_padded,
                                   uint32_t in_guard, uint32_t channels, uint32_t temporal, int drop_head);
int h3_gpu_sv2_grid_select3d_bf16(h3_gpu *gpu, h3_gpu_tensor *output, uint32_t out_frames,
                                  uint32_t out_height, uint32_t out_width, uint32_t out_guard,
                                  const h3_gpu_tensor *input, uint32_t in_frames, uint32_t in_height,
                                  uint32_t in_width, uint32_t in_guard, uint32_t padded,
                                  uint32_t stride_t, uint32_t phase_t, uint32_t phase_y, uint32_t phase_x);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
