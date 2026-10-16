#ifndef H3_QUANT_H
#define H3_QUANT_H

/* Block-quantized weight formats from GGUF files (ggml's layouts) and
 * MLX's affine group quantization, with plain-C reference dequantizers.
 * The engine never computes on these formats directly: it expands a tensor
 * (or a streamed block) to BF16/F32 on the GPU and runs its usual kernels,
 * so these references are what the GPU kernels are tested against.
 *
 * Every ggml type stores consecutive blocks of block_elements values along
 * a row; a row of n values (n a multiple of block_elements) takes
 * n / block_elements * block_bytes bytes. */

#include <stddef.h>
#include <stdint.h>

typedef enum {
    H3_QUANT_NONE = 0,
    H3_QUANT_F32,
    H3_QUANT_F16,
    H3_QUANT_BF16,
    H3_QUANT_Q4_0,
    H3_QUANT_Q4_1,
    H3_QUANT_Q5_0,
    H3_QUANT_Q5_1,
    H3_QUANT_Q8_0,
    H3_QUANT_Q2_K,
    H3_QUANT_Q3_K,
    H3_QUANT_Q4_K,
    H3_QUANT_Q5_K,
    H3_QUANT_Q6_K,
} h3_quant_type;

/* ggml's type id (GGUF tensor info) to h3_quant_type; H3_QUANT_NONE for
 * types the engine does not read (IQ*, TQ*, MXFP4, ...). */
h3_quant_type h3_quant_from_ggml(uint32_t ggml_type);
const char *h3_quant_name(h3_quant_type type);
/* Values per block and bytes per block (1 and the element size for the
 * float types); 0 for H3_QUANT_NONE. */
uint32_t h3_quant_block_elements(h3_quant_type type);
uint32_t h3_quant_block_bytes(h3_quant_type type);
/* Bytes for count values, or 0 when count is not a whole number of blocks. */
uint64_t h3_quant_bytes(h3_quant_type type, uint64_t count);

/* Expands count values (a whole number of blocks) to F32. Returns 0 for an
 * unsupported type or a partial block. */
int h3_quant_dequantize(h3_quant_type type, const void *source, uint64_t count,
                        float *out);

/* Same, straight to BF16 (round to nearest even), block by block - no
 * F32 copy of the whole tensor. */
int h3_quant_dequantize_bf16(h3_quant_type type, const void *source,
                             uint64_t count, uint16_t *out);

/* MLX affine quantization: `bits` (2, 3, 4, 5, 6 or 8) packed
 * little-endian into uint32 words along each row, and per group of
 * group_size values a BF16 scale and bias: value = q * scale + bias. Expands
 * one row of `columns` values. */
int h3_mlx_dequantize_row(const uint32_t *packed, const uint16_t *scales,
                          const uint16_t *biases, uint32_t columns,
                          uint32_t bits, uint32_t group_size, float *out);

float h3_half_to_float(uint16_t half);

#endif
