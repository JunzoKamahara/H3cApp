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

#ifdef __cplusplus
} // extern "C"
#endif

#endif
