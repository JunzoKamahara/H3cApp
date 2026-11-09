#ifndef H3_GPU_H
#define H3_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
/* Only h3_gpu_ccv_attention.mm (Objective-C++) includes this header from a
 * C++ translation unit - everything else in this codebase is C/Objective-C.
 * Without this, C++ would mangle these names and fail to find the plain C
 * symbols h3_gpu.m actually exports. */
extern "C" {
#endif

typedef struct h3_gpu h3_gpu;
typedef struct h3_gpu_tensor h3_gpu_tensor;

typedef enum {
    H3_GPU_F32 = 0,
    H3_GPU_BF16,
    H3_GPU_I8,
    H3_GPU_U32
} h3_gpu_dtype;

typedef struct {
    uint64_t allocated_bytes;
    uint64_t live_bytes;
    uint64_t peak_live_bytes;
    uint64_t tensor_allocations;
    uint64_t direct_dispatches;
    uint64_t mps_linear_dispatches;
    uint64_t mps_conv_dispatches;
    uint64_t mps_sdpa_dispatches;
    /* h3_gpu_ccv_dense_attention_bf16 calls that actually ran ccv's kernel
     * (see h3_gpu_ccv_attention.mm) - an opt-in, non-default attention
     * backend. Checking this after a run confirms the requested backend
     * was really used for every block, rather than silently falling back. */
    uint64_t ccv_attention_dispatches;
    /* Pieces of a chunked SDPA (h3_gpu_set_sdpa_mode): one full attention
     * call still counts once in mps_sdpa_dispatches, each piece once here. */
    uint64_t mps_sdpa_chunk_dispatches;
    uint64_t blit_copies;
    uint64_t submissions;
    double command_encode_seconds;
    double command_wait_seconds;
    /* Root MTLCommandBuffer timestamps; MPSGraph may schedule child buffers,
     * so command_wait_seconds is the complete turnaround measurement. */
    double gpu_seconds;
} h3_gpu_stats;

h3_gpu *h3_gpu_create(const char *shader_source_path,
                      char *error, size_t error_size);
void h3_gpu_free(h3_gpu *gpu);
int h3_gpu_is_m5(const h3_gpu *gpu);
int h3_gpu_has_nax_mlp(const h3_gpu *gpu);
int h3_gpu_has_int8_mlp(const h3_gpu *gpu);

h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values,
                                      size_t elements);
h3_gpu_tensor *h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values,
                                       size_t elements);
h3_gpu_tensor *h3_gpu_tensor_from_i8(h3_gpu *gpu, const int8_t *values,
                                     size_t elements);
h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values,
                                      size_t elements);
/* Allocate shared Metal storage and pread BF16 payload directly into it. */
h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *gpu, const char *path,
                                       uint64_t file_offset, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *gpu, const char *path,
                                      uint64_t file_offset, size_t elements);
/* Fill an existing shared BF16 buffer from a file. The tensor and its
 * accounting are unchanged, so this may run on an I/O thread while another
 * tensor is in flight on the GPU. */
int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size);
/* As above, but ask Darwin to avoid retaining a second copy in the file cache.
 * Intended for large sequential weight streams whose destination is the only
 * useful resident copy. */
int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size);
/* I8/F32 counterparts of the two functions above, for a pre-quantized
 * attention cache (int8 weights + f32 scales) rather than raw BF16 shards. */
int h3_gpu_tensor_read_file_i8(h3_gpu_tensor *tensor, const char *path,
                               uint64_t file_offset, size_t elements,
                               char *error, size_t error_size);
int h3_gpu_tensor_stream_file_i8(h3_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size);
/* A tensor over [byte_offset, byte_offset + elements * size) of `parent`'s
 * memory, no copy (Apple silicon shared memory). byte_offset must be a
 * multiple of h3_gpu_page_bytes() and the window, rounded up to a page, must
 * lie inside the parent; the parent must not itself be an alias. Free the
 * aliases before or after the parent - each keeps the memory alive. Writes
 * through one alias are not ordered against reads through an overlapping
 * one inside the same command buffer. */
size_t h3_gpu_page_bytes(void);
h3_gpu_tensor *h3_gpu_tensor_alias_bf16(h3_gpu_tensor *parent,
                                        size_t byte_offset, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_alias_i8(h3_gpu_tensor *parent,
                                      size_t byte_offset, size_t elements);
/* Streaming reads that fill only the leading `elements` of a larger tensor
 * (a slot sized for the largest of several matrices). */
int h3_gpu_tensor_stream_file_prefix_bf16(h3_gpu_tensor *tensor,
                                          const char *path,
                                          uint64_t file_offset,
                                          size_t elements, char *error,
                                          size_t error_size);
int h3_gpu_tensor_stream_file_prefix_i8(h3_gpu_tensor *tensor,
                                        const char *path,
                                        uint64_t file_offset,
                                        size_t elements, char *error,
                                        size_t error_size);
int h3_gpu_tensor_read_file_f32(h3_gpu_tensor *tensor, const char *path,
                                uint64_t file_offset, size_t elements,
                                char *error, size_t error_size);
int h3_gpu_tensor_stream_file_f32(h3_gpu_tensor *tensor, const char *path,
                                  uint64_t file_offset, size_t elements,
                                  char *error, size_t error_size);
void h3_gpu_tensor_free(h3_gpu_tensor *tensor);
size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor);
h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor);
int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values,
                           size_t elements);
int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements);
int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values,
                            size_t elements);
int h3_gpu_tensor_read_i8(const h3_gpu_tensor *tensor, int8_t *values,
                          size_t elements);
int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values,
                            size_t elements);
int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements);
int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements);
int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements);

int h3_gpu_begin(h3_gpu *gpu);
/* Commit the current command buffer without waiting, then continue encoding on
 * the same ordered queue. h3_gpu_submit() waits and validates the whole chain. */
int h3_gpu_continue(h3_gpu *gpu);
int h3_gpu_submit(h3_gpu *gpu);
const char *h3_gpu_error(const h3_gpu *gpu);
int h3_gpu_get_stats(const h3_gpu *gpu, h3_gpu_stats *stats);
/* Optional benchmark labels. With H3_PROFILE set, marks and context teardown
 * print wall time alongside command-buffer GPU time and allocation counters. */
void h3_gpu_profile_set_label(h3_gpu *gpu, const char *label);
void h3_gpu_profile_mark(h3_gpu *gpu, const char *phase);

int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim);
int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);
int h3_gpu_patch_linear_bf16_offset(
                             h3_gpu *gpu, h3_gpu_tensor *output,
                             size_t output_offset,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);
int h3_gpu_patch_linear_bf16_map(
                             h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias,
                             const h3_gpu_tensor *row_map,
                             uint32_t output_rows, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);
int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements);
int h3_gpu_cast_f32_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements);
/* Whether a command buffer is being encoded (between h3_gpu_begin and
 * h3_gpu_submit) - work issued now joins it instead of starting one. */
int h3_gpu_command_open(const h3_gpu *gpu);
/* Expands `blocks` blocks of a block-quantized (or F16/BF16/F32) payload -
 * raw bytes in an I8 tensor - into a BF16 or F32 output (its dtype
 * decides), with h3_quant.c's formulas. quant is an h3_quant_type.
 * qkv_inner != 0 also reorders the rows (of `columns` values) of a fused
 * QKV weight from [all Q; all K; all V] (qkv_inner rows each) to the
 * engine's per-head [q; k; v] (qkv_head_dim rows each). */
int h3_gpu_dequantize(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *raw, uint32_t quant,
                      uint32_t blocks, uint32_t block_elements,
                      uint32_t block_bytes, uint32_t columns,
                      uint32_t qkv_inner, uint32_t qkv_head_dim);
int h3_gpu_cast_bf16_to_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements);
/* output[r, c] = F32(input[r, c]) + bias[c]; the epilogue for int8 linears
 * that feed an F32 residual stream with a bias. */
int h3_gpu_cast_bf16_to_f32_bias(h3_gpu *gpu, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *bias, uint32_t rows,
                                 uint32_t width);
int h3_gpu_copy_bf16(h3_gpu *gpu, h3_gpu_tensor *destination,
                     size_t destination_offset,
                     const h3_gpu_tensor *source, size_t source_offset,
                     size_t elements);
int h3_gpu_copy_f32(h3_gpu *gpu, h3_gpu_tensor *destination,
                    size_t destination_offset,
                    const h3_gpu_tensor *source, size_t source_offset,
                    size_t elements);
int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon);
int h3_gpu_adaln_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon);
int h3_gpu_gate_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual,
                    const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot);
int h3_gpu_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon);
int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale);
int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width);
int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *scale, uint32_t rows,
                         uint32_t width);
int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon);
int h3_gpu_video_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon);

/* H3 AudioVAE uses time-major [batch,length,channels] activations and stores
 * Conv1d/ConvTranspose1d weights in PyTorch OIK/IOK order respectively. */
int h3_gpu_conv1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation);
int h3_gpu_conv1d_stride_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding,
                      uint32_t dilation);
int h3_gpu_conv_transpose1d_f32(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding);
int h3_gpu_weight_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude,
                           uint32_t outer, uint32_t inner);
int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left,
                          const h3_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements);
int h3_gpu_alias_free_snake_f32(
                          h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *alpha_log,
                          const h3_gpu_tensor *beta_log,
                          const h3_gpu_tensor *upsample_filter,
                          const h3_gpu_tensor *downsample_filter,
                          uint32_t batch, uint32_t length,
                          uint32_t channels);
int h3_gpu_snake1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *alpha, uint32_t batch,
                       uint32_t length, uint32_t channels);
int h3_gpu_audio_qkv_split_f32(h3_gpu *gpu,
                       h3_gpu_tensor *query, h3_gpu_tensor *key,
                       h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                       const h3_gpu_tensor *q_bias,
                       const h3_gpu_tensor *k_bias,
                       const h3_gpu_tensor *v_bias, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim);
int h3_gpu_sdpa_causal_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *query,
                       const h3_gpu_tensor *key,
                       const h3_gpu_tensor *value, uint32_t batch,
                       uint32_t sequence, uint32_t heads,
                       uint32_t head_dim, float scale);
int h3_gpu_audio_attention_pool_f32(h3_gpu *gpu,
                       h3_gpu_tensor *output,
                       const h3_gpu_tensor *attended, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim, uint32_t output_dim);
int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate,
                     const h3_gpu_tensor *linear, uint32_t elements);
int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum);


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
int h3_gpu_has_tensor_ops(const h3_gpu *gpu);
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
/* Fills `elements` floats with zero (a blit inside begin/submit). */
int h3_gpu_zero_f32(h3_gpu *gpu, h3_gpu_tensor *x, size_t elements);

/* Visual-VAE encoder tensors use channels-last [B,T,H,W,C] storage. Spatial
 * padding reflects pixels while temporal front padding is zero-filled. */
int h3_gpu_vae_encoder_pad_f32(
                    h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t batch,
                    uint32_t depth, uint32_t height, uint32_t width,
                    uint32_t channels, uint32_t depth_front,
                    uint32_t height_before, uint32_t height_after,
                    uint32_t width_before, uint32_t width_after);
int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t input_channels, uint32_t output_channels,
                      uint32_t kernel_depth, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride_depth,
                      uint32_t stride_height, uint32_t stride_width);
/* The same Conv3d as one 2D convolution per time tap, summed - about 4.6x
 * faster in MPSGraph, equal up to the order of the F32 sums. */
int h3_gpu_conv3d_as_2d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *bias, uint32_t batch,
                            uint32_t depth, uint32_t height, uint32_t width,
                            uint32_t input_channels, uint32_t output_channels,
                            uint32_t kernel_depth, uint32_t kernel_height,
                            uint32_t kernel_width, uint32_t stride_depth,
                            uint32_t stride_height, uint32_t stride_width);
/* h3_gpu_conv3d_f32 (as_2d 0) or h3_gpu_conv3d_as_2d_f32 (1) for one
 * output frame whose input starts with `zero_frames` frames of zeros, which
 * are not passed: `input` holds the kernel_depth - zero_frames frames after
 * them, and the taps that would meet the zeros are left out. On an M5 the
 * result matched the full convolution's bit for bit in every encoder layer
 * (still images of 256x256 to 1344x768, both forms; 2026-11-03). */
int h3_gpu_conv3d_zero_front_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *bias,
                                 uint32_t depth, uint32_t height, uint32_t width,
                                 uint32_t input_channels, uint32_t output_channels,
                                 uint32_t kernel_depth, uint32_t kernel_height,
                                 uint32_t kernel_width, uint32_t stride_depth,
                                 uint32_t stride_height, uint32_t stride_width,
                                 uint32_t zero_frames, int as_2d);
/* One 2D convolution with explicit zero padding in the graph: F32 NHWC
 * [height, width, C_in] -> [(height + pads - kh) / stride + 1, ..., C_out],
 * weight OIHW [C_out, C_in, kh, kw]. */
int h3_gpu_conv2d_padded_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
                             uint32_t height, uint32_t width, uint32_t input_channels,
                             uint32_t output_channels, uint32_t kernel_height, uint32_t kernel_width,
                             uint32_t stride_height, uint32_t stride_width,
                             uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right);
/* The same GroupNorm + SiLU in three passes that each read the frame in
 * order, much faster on large frames; equal up to the order of the F32
 * sums. Shapes it doesn't cover fall back to the one-pass kernel. */
int h3_gpu_vae_encoder_group_norm_silu_chunked_f32(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t groups, float epsilon);
int h3_gpu_vae_encoder_group_norm_silu_f32(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t groups, float epsilon);

/* LoRA weight patching (see h3_lora.h). delta[rows, columns] =
 * b[rows, rank] @ at[columns, rank]^T; the add forms apply one row
 * group of it to a projection weight, mapping group row r to the per-head
 * interleaved QKV row when `interleaved` is set, with stochastic rounding
 * seeded by `seed` so the tiny delta survives. */
int h3_gpu_lora_delta_bf16(h3_gpu *gpu, h3_gpu_tensor *delta,
                           const h3_gpu_tensor *b, const h3_gpu_tensor *at,
                           uint32_t rows, uint32_t rank, uint32_t columns);
int h3_gpu_lora_add_rows_bf16(h3_gpu *gpu, h3_gpu_tensor *weight,
                              const h3_gpu_tensor *delta, uint32_t rows,
                              uint32_t columns, uint32_t interleaved,
                              uint32_t group, uint32_t head_dim,
                              uint32_t seed);
/* Per-row int8 weight += stochastically rounded delta / scale; scales are
 * left unchanged (see h3_shaders.metal). */
int h3_gpu_lora_add_int8(h3_gpu *gpu, h3_gpu_tensor *weight,
                         const h3_gpu_tensor *scales,
                         const h3_gpu_tensor *delta, uint32_t rows,
                         uint32_t columns, uint32_t interleaved,
                         uint32_t group, uint32_t head_dim, uint32_t seed);

/* Portable BF16 storage path. Arithmetic accumulates in F32 and rounds at
 * operation boundaries, matching the released checkpoint's compute dtype. */
int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim);
int h3_gpu_mlp_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input,
                    const h3_gpu_tensor *fc1_weight,
                    const h3_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim);
/* Experimental M5 Metal 4 paired FC1/SwiGLU plus direct FC2 path. Available
 * only when the context was created with H3_NAX=mlp. */
int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                        h3_gpu_tensor *activated,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *fc1_weight,
                        const h3_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim);
/* Experimental M5 Metal 4 int8 MLP. Weights use one F32 scale per output
 * channel; activations are quantized dynamically with one F32 scale per row. */
int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t columns);
/* The int8 linear reduced in `splits` K slices (input_dim a multiple of
 * 128 * splits) into F32 `partial` [splits, padded rows, output_dim], then
 * summed to BF16: faster than one pass when K is long (14336). */
int h3_gpu_quantize_rows_bf16_int8(h3_gpu *gpu, h3_gpu_tensor *quantized, h3_gpu_tensor *scales,
                                   const h3_gpu_tensor *input, uint32_t rows, uint32_t columns);
int h3_gpu_linear_int8_bf16_split_k(h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *partial,
                                    h3_gpu_tensor *quantized_input, h3_gpu_tensor *input_scales,
                                    const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                                    const h3_gpu_tensor *weight_scales, uint32_t rows,
                                    uint32_t input_dim, uint32_t output_dim, uint32_t splits);
int h3_gpu_linear_int8_partial_f32(h3_gpu *gpu, h3_gpu_tensor *partial, uint32_t slot,
                                   h3_gpu_tensor *quantized_input, h3_gpu_tensor *input_scales,
                                   const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                                   const h3_gpu_tensor *weight_scales, uint32_t rows,
                                   uint32_t input_dim, uint32_t output_dim, int prequantized);
/* gelu_tanh(x + bias) quantized per row to int8 (+ F32 scales) in one
 * pass; columns a multiple of 4, at most 8192. */
int h3_gpu_vsr_gelu_quantize_int8_rows(h3_gpu *gpu, h3_gpu_tensor *quantized, h3_gpu_tensor *scales,
                                       const h3_gpu_tensor *input, const h3_gpu_tensor *bias,
                                       uint32_t rows, uint32_t columns);
int h3_gpu_sum_partials_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *partial,
                             uint32_t rows, uint32_t output_dim, uint32_t splits);
int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales);
/* Consume SDPA's native [head,row,dimension] BF16 layout without a full
 * BF16 transpose, gathering directly into the projection's row-major int8. */
int h3_gpu_linear_int8_head_major_bf16(
                            h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim, uint32_t output_dim);
int h3_gpu_mlp_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         h3_gpu_tensor *activated,
                         h3_gpu_tensor *quantized_activation,
                         h3_gpu_tensor *activation_scales,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *fc1_weight,
                         const h3_gpu_tensor *fc1_scales,
                         const h3_gpu_tensor *fc2_weight,
                         const h3_gpu_tensor *fc2_scales,
                         const h3_gpu_tensor *fc1_bf16,
                         const h3_gpu_tensor *fc2_bf16, uint32_t rows,
                         uint32_t input_dim, uint32_t hidden_dim,
                         uint32_t output_dim,
                         int use_slower_grouped_quantizer,
                         int use_slower_dynamic_fc1_k,
                         int use_int8_row_fc2,
                         int input_is_quantized);
int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements);
int h3_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon);
int h3_gpu_layer_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon);
int h3_gpu_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate);
int h3_gpu_vision_qkv_rope_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *query,
                     h3_gpu_tensor *key, h3_gpu_tensor *value,
                     const h3_gpu_tensor *qkv,
                     const h3_gpu_tensor *rope_cos,
                     const h3_gpu_tensor *rope_sin, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim,
                     uint32_t rope_half);
int h3_gpu_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon);
int h3_gpu_adaln_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon);
int h3_gpu_adaln_linear_bf16(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      h3_gpu_tensor *inverse,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t width, uint32_t output_dim, uint32_t slots,
                      uint32_t shift_slot, uint32_t scale_slot,
                      float epsilon);
int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot);
int h3_gpu_gate_adaln_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot,
                     uint32_t shift_slot, uint32_t scale_slot,
                     float epsilon);
int h3_gpu_gate_adaln_quantize_int8(
                     h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *quantized_output,
                     h3_gpu_tensor *quantized_scales,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t padded_rows, uint32_t width, uint32_t slots,
                     uint32_t gate_slot, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon);
int h3_gpu_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                         h3_gpu_tensor *key, h3_gpu_tensor *value,
                         const h3_gpu_tensor *qkv,
                         const h3_gpu_tensor *q_norm,
                         const h3_gpu_tensor *k_norm,
                         const h3_gpu_tensor *rope_cos,
                         const h3_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim,
                         uint32_t rope_half, float epsilon);
/* H3 checkpoint QKV rows are [head, q/k/v, dimension], unlike the
 * conventional [q/k/v, head, dimension] layout accepted above. */
int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key, h3_gpu_tensor *value,
                                 const h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon);
/* Project grouped H3 QKV and apply its exact Q/K norm/RoPE boundary. Metal 4
 * may route projections directly into the attention layout; other devices
 * retain the ordinary two calls. */
int h3_gpu_grouped_qkv_linear_rope_bf16(
                                 h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon);
int h3_gpu_grouped_qkv_linear_rope_int8(
                                 h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *quantized_input,
                                 h3_gpu_tensor *input_scales,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *weight_scales,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon,
                                 int input_is_quantized,
                                 int use_slower_unfused_qkv_rope,
                                 int use_slower_scalar_qkv_rms,
                                 int use_slower_uncached_int8_scales);
int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale);
/* Debug-only: true if the most recent QKV projection call left query/key/
 * value in [heads, rows, head_dim] layout for the next h3_gpu_sdpa_bf16
 * call to consume (an internal producer/consumer optimization, reset to 0
 * by that SDPA call), rather than the [rows, heads, head_dim] layout a raw
 * capture of those tensors would otherwise assume. */
int h3_gpu_head_major_sdpa_inputs(h3_gpu *gpu);

/* How the dense (non-causal, batch 1) SDPA calls below are executed.
 * FULL hands MPSGraph the whole [heads, sequence, sequence] problem in one
 * call. CHUNKED runs it in pieces - a group of heads, or one head and a
 * block of query rows - each a complete softmax over all keys (so the
 * result is the same computation) and none whose score matrix exceeds
 * `score_budget_bytes` (0 = 1 GiB). EXPLICIT is CHUNKED with the attention
 * written out of MPSGraph matmul/softmax ops instead of its fused SDPA op
 * (a diagnostic for machines where that op itself misbehaves). AUTO picks
 * FULL where it is validated (Apple10 GPU on macOS 26) and CHUNKED
 * elsewhere; the environment variable H3_SDPA_MODE (full/chunked/explicit)
 * and H3_SDPA_SCORE_BYTES override AUTO for diagnostics, H3_SDPA_LOG=1
 * prints each distinct call's device, OS, mode, shapes, layouts and buffer
 * sizes once. Background: issue #2 (M2 Max, macOS 15.8.2: SIGABRT inside
 * MPSGraph's SDPA encode on the DiT's first full attention). */
typedef enum {
    H3_GPU_SDPA_AUTO = 0,
    H3_GPU_SDPA_FULL = 1,
    H3_GPU_SDPA_CHUNKED = 2,
    H3_GPU_SDPA_EXPLICIT = 3
} h3_gpu_sdpa_mode;
void h3_gpu_set_sdpa_mode(h3_gpu *gpu, h3_gpu_sdpa_mode mode,
                          uint64_t score_budget_bytes);
/* What AUTO (or the override) resolves to on this GPU. */
h3_gpu_sdpa_mode h3_gpu_sdpa_effective_mode(h3_gpu *gpu);
const char *h3_gpu_sdpa_mode_name(h3_gpu_sdpa_mode mode);
/* The pieces a chunked call of this shape would run: heads per piece and
 * query rows per piece (rows == sequence unless one head alone exceeds the
 * budget). For logs and tests. */
void h3_gpu_sdpa_chunk_plan(h3_gpu *gpu, uint32_t sequence,
                            uint32_t kv_sequence, uint32_t heads,
                            h3_gpu_dtype dtype, uint32_t *chunk_heads,
                            uint32_t *chunk_rows);
/* Preserve SDPA's native [head,row,dimension] output for an immediately
 * following layout-aware projection. */
int h3_gpu_sdpa_bf16_head_major_output(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale);
int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width);
/* output = silu(gate) * up, element-wise over count values (output may be
 * up). HyperFlow's factored FC1 makes the two SwiGLU halves separately. */
int h3_gpu_swiglu_split_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *gate,
                             const h3_gpu_tensor *up, uint32_t count);
/* The Q/K per-head RMS norm + RoPE of h3_gpu_grouped_qkv_rope_bf16 for Q
 * and K that are already separate token-major [sequence, heads * head_dim]
 * buffers (HyperFlow's factored attention); writes query and key only.
 * head_dim must be 128 and heads a multiple of 4. */
int h3_gpu_qk_rope_split_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                              h3_gpu_tensor *key,
                              const h3_gpu_tensor *q_input,
                              const h3_gpu_tensor *k_input,
                              const h3_gpu_tensor *q_norm,
                              const h3_gpu_tensor *k_norm,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon);
int h3_gpu_embedding_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width);
int h3_gpu_text_qk_rope_bf16(h3_gpu *gpu,
                             h3_gpu_tensor *query_output,
                             h3_gpu_tensor *key_output,
                             const h3_gpu_tensor *query_input,
                             const h3_gpu_tensor *key_input,
                             const h3_gpu_tensor *q_norm,
                             const h3_gpu_tensor *k_norm,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin,
                             uint32_t sequence, uint32_t query_heads,
                             uint32_t kv_heads, uint32_t head_dim,
                             float epsilon);
int h3_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, float epsilon);
int h3_gpu_rope_text_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                          h3_gpu_tensor *key,
                          const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32,
                          uint32_t sequence, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim);
int h3_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value,
                           uint32_t sequence, uint32_t query_heads,
                           uint32_t kv_heads, uint32_t head_dim,
                           float scale);
/* The MPSGraph route h3_gpu_gqa_causal_bf16 takes past its kernel's
 * threadgroup ceiling (about 7,900 tokens), callable at any length for
 * tests: one call in FULL mode, otherwise pieces of one KV head's query
 * heads and a block of query rows under the h3_gpu_set_sdpa_mode budget.
 * h3_gpu_gqa_chunk_plan reports the query rows per piece (sequence in FULL
 * mode); there are kv_heads x ceil(sequence / rows) pieces. */
int h3_gpu_gqa_causal_graph_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *query,
                                 const h3_gpu_tensor *key,
                                 const h3_gpu_tensor *value,
                                 uint32_t sequence, uint32_t query_heads,
                                 uint32_t kv_heads, uint32_t head_dim,
                                 float scale);
void h3_gpu_gqa_chunk_plan(h3_gpu *gpu, uint32_t sequence,
                           uint32_t query_heads, uint32_t kv_heads,
                           uint32_t *chunk_rows);
int h3_gpu_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements);
int h3_gpu_sub_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements);
int h3_gpu_token_pool_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           size_t input_offset,
                           h3_gpu_tensor *original,
                           size_t original_offset,
                           h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width);
int h3_gpu_token_pool_adaln_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, size_t input_offset,
                           h3_gpu_tensor *original, size_t original_offset,
                           h3_gpu_tensor *baseline, size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t input_rows, uint32_t rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon);
int h3_gpu_token_expand_delta_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents, uint32_t rows,
                           uint32_t reduced_rows, uint32_t baseline_rows,
                           uint32_t width,
                           uint32_t exact_prefix_rows,
                           float update_scale);
int h3_gpu_token_expand_adaln_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t rows, uint32_t reduced_rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t exact_prefix_rows, float update_scale,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon);
/* Apply one Euler step to an F32 sample range from BF16 velocity caches:
 * sample += delta * (last + ratio * (last - previous)). */
int h3_gpu_euler_bf16(h3_gpu *gpu, h3_gpu_tensor *sample,
                      size_t sample_offset, const h3_gpu_tensor *last,
                      const h3_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio);
/* Apply one re-noise step to an F32 sample range from a BF16 velocity:
 * clean = sample + sigma * velocity; sample = clean + sigma_next *
 * (noise - clean), noise from h3_renoise_normal(element, keys). */
int h3_gpu_renoise_bf16(h3_gpu *gpu, h3_gpu_tensor *sample,
                        size_t sample_offset, const h3_gpu_tensor *velocity,
                        uint32_t elements, float sigma, float sigma_next,
                        const uint32_t keys[2]);
int h3_gpu_silu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate,
                         const h3_gpu_tensor *up, uint32_t elements);

/* --- Experimental, opt-in attention backend (h3_gpu_ccv_attention.mm) ---
 *
 * Bridges to liuliu/ccv's Metal int8 attention kernels for an A/B
 * comparison against this engine's own MPSGraph SDPA (see
 * SPEEDUP_ROADMAP.md item 5 and tools/ccv_eval/README.md). Only linked in
 * when the build sets CCV_DIR (see Makefile); h3_gpu_ccv_dense_attention_bf16
 * itself is always declared and always safe to call - it fails cleanly with
 * "ccv backend not compiled into this binary" when nothing registered a
 * real implementation, so the normal build and every other binary that
 * links libh3.a is unaffected whether or not the optional backend exists. */

/* Raw, unretained accessors for h3_gpu_ccv_attention.mm's Metal-cpp
 * bridging - it needs the actual MTLDevice/MTLCommandBuffer/MTLBuffer
 * objects, not the opaque h3_gpu/h3_gpu_tensor handles the rest of this
 * header deals in. Pointers are valid only as long as the h3_gpu/tensor
 * they came from is alive and, for the command buffer, only until the next
 * h3_gpu_submit(). */
void *h3_gpu_raw_device(h3_gpu *gpu);
void *h3_gpu_raw_command_buffer(h3_gpu *gpu);
void *h3_gpu_raw_buffer(const h3_gpu_tensor *tensor);
void h3_gpu_set_head_major_sdpa_inputs(h3_gpu *gpu, int value);
/* `direct`: the call used the BF16 head-major direct path. */
void h3_gpu_note_ccv_attention_dispatch(h3_gpu *gpu, int direct);
void h3_gpu_report_error(h3_gpu *gpu, const char *message);

/* Cast + (for Q/K/V only) layout-fix helpers, dispatched into the current
 * command buffer like any other h3_gpu op. `head_major` reflects
 * h3_gpu_head_major_sdpa_inputs() at the time query/key/value were
 * produced - output from ccv is always row-major, matching h3_gpu_sdpa_bf16
 * (non-head-major) semantics, by construction of the kernel used here. */
int h3_gpu_ccv_cast_bf16_to_f16(h3_gpu *gpu, void *output_f16_buffer,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t heads, uint32_t head_dim,
                                int head_major);
int h3_gpu_ccv_cast_f16_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                const void *input_f16_buffer,
                                uint32_t elements);

/* The actual entry point h3_dit.c calls. Forwards to whatever
 * h3_gpu_register_ccv_dense_attention() registered (see
 * h3_gpu_ccv_attention.mm), or fails cleanly if nothing did. `quantized`
 * selects ccv's int8 kernel (Q/K/V and the softmax weights P all int8);
 * zero selects its non-quantized FP16 kernel through the same bridge.
 * `head_major_output` may be NULL (row-major output required); otherwise
 * set it to 1 if a head-major [heads, rows, dim] output is acceptable, and
 * on return it says which layout `output` was actually written in. */
int h3_gpu_ccv_dense_attention_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                    const h3_gpu_tensor *query,
                                    const h3_gpu_tensor *key,
                                    const h3_gpu_tensor *value,
                                    uint32_t rows, uint32_t heads,
                                    uint32_t head_dim, float scale,
                                    int quantized, int *head_major_output);
typedef int (*h3_gpu_ccv_dense_attention_fn)(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *query,
    const h3_gpu_tensor *key, const h3_gpu_tensor *value, uint32_t rows,
    uint32_t heads, uint32_t head_dim, float scale, int quantized,
    int *head_major_output);
void h3_gpu_register_ccv_dense_attention(h3_gpu_ccv_dense_attention_fn fn);
/* Drops the ccv bridge's per-process scratch (its FP16 Q/K/V/out copies and
 * ccv's own int8 scratch; compiled pipelines are kept) once no command on
 * `gpu` is open or in flight. Returns 1 when released or nothing was held,
 * 0 (and changes nothing) while work that may use it is still pending. The
 * next ccv attention call reallocates. A no-op in builds without CCV_DIR. */
int h3_gpu_ccv_release_scratch(h3_gpu *gpu);
typedef void (*h3_gpu_ccv_release_scratch_fn)(h3_gpu *gpu);
void h3_gpu_register_ccv_release_scratch(h3_gpu_ccv_release_scratch_fn fn);
int h3_gpu_ccv_dense_attention_available(void);
/* 1 when the backend is linked in and this machine can run it (ccv
 * supports the device and it has neural matrix accelerators). */
int h3_gpu_ccv_attention_supported(void);
typedef int (*h3_gpu_ccv_attention_supported_fn)(void);
void h3_gpu_register_ccv_attention_supported(
    h3_gpu_ccv_attention_supported_fn fn);
/* Process-wide count of successful h3_gpu_ccv_dense_attention_bf16 calls -
 * use this after a run to confirm the backend was really used for every
 * block expected, not silently skipped - and how many of them took the
 * direct path. Defined in h3_gpu.m so they link in every build, with or
 * without CCV_DIR. */
uint64_t h3_gpu_ccv_attention_dispatch_count(void);
uint64_t h3_gpu_ccv_attention_direct_count(void);

/* --- SeedVR2 (h3_seedvr2.c): see the kernels' comments in h3_shaders.metal --- */
/* Lets h3_gpu_linear_bf16 take the TensorOps (NAX) BF16 kernel for any
 * shape that satisfies its constraints (in % 32, out % 64, rows >= 128, no
 * bias), not only the H3 projections. Per context; no effect without
 * tensor hardware. */
void h3_gpu_set_nax_linear_generic(h3_gpu *gpu, int on);
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
