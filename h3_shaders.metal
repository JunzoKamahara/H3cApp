#include <metal_stdlib>
#ifdef H3_METAL_HAS_TENSOR
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;
#ifdef H3_METAL_HAS_TENSOR
using namespace mpp::tensor_ops;
#endif

inline float h3_bf16_to_f32(ushort value) {
    return as_type<float>(uint(value) << 16);
}

inline ushort h3_f32_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return ushort(bits >> 16);
}

inline float4 h3_bf16x4_to_f32(ushort4 value) {
    return as_type<float4>(uint4(value) << 16);
}

inline ushort4 h3_f32x4_to_bf16(float4 value) {
    uint4 bits = as_type<uint4>(value);
    bits += uint4(0x7fffu) + ((bits >> 16) & uint4(1u));
    return ushort4(bits >> 16);
}

struct linear_args {
    uint rows;
    uint input_dim;
    uint output_dim;
    uint has_bias;
};

struct int8_quant_args {
    uint rows;
    uint columns;
    float clip;
};

struct int8_head_major_quant_args {
    uint rows;
    uint padded_rows;
    uint heads;
    uint head_dim;
    float clip;
};

struct int8_group_quant_args {
    uint rows;
    uint columns;
    uint group_size;
    uint groups;
};

kernel void h3_linear_f32(device const float *input [[buffer(0)]],
                          device const float *weight [[buffer(1)]],
                          device const float *bias [[buffer(2)]],
                          device float *output [[buffer(3)]],
                          constant linear_args &args [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.output_dim) return;
    float sum = args.has_bias ? bias[column] : 0.0f;
    device const float *x = input + row * args.input_dim;
    device const float *w = weight + column * args.input_dim;
    for (uint k = 0; k < args.input_dim; k++) sum = fma(x[k], w[k], sum);
    output[row * args.output_dim + column] = sum;
}

kernel void h3_linear_f32_tiled(device const float *input [[buffer(0)]],
                                device const float *weight [[buffer(1)]],
                                device const float *bias [[buffer(2)]],
                                device float *output [[buffer(3)]],
                                constant linear_args &args [[buffer(4)]],
                                uint2 tid [[thread_position_in_threadgroup]],
                                uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float input_tile[16][16];
    threadgroup float weight_tile[16][16];
    uint row = group.y * 16 + tid.y;
    uint column = group.x * 16 + tid.x;
    float sum = args.has_bias && column < args.output_dim ?
        bias[column] : 0.0f;
    uint tile_count = (args.input_dim + 15) / 16;
    for (uint tile = 0; tile < tile_count; tile++) {
        uint input_k = tile * 16 + tid.x;
        input_tile[tid.y][tid.x] =
            row < args.rows && input_k < args.input_dim ?
            input[row * args.input_dim + input_k] : 0.0f;
        uint weight_k = tile * 16 + tid.y;
        weight_tile[tid.y][tid.x] =
            column < args.output_dim && weight_k < args.input_dim ?
            weight[column * args.input_dim + weight_k] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; k++)
            sum = fma(input_tile[tid.y][k], weight_tile[k][tid.x], sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < args.rows && column < args.output_dim)
        output[row * args.output_dim + column] = sum;
}

kernel void h3_linear_f32_tiled_bf16(
                                device const float *input [[buffer(0)]],
                                device const float *weight [[buffer(1)]],
                                device const float *bias [[buffer(2)]],
                                device ushort *output [[buffer(3)]],
                                constant linear_args &args [[buffer(4)]],
                                uint2 tid [[thread_position_in_threadgroup]],
                                uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float input_tile[16][16];
    threadgroup float weight_tile[16][16];
    uint row = group.y * 16 + tid.y;
    uint column = group.x * 16 + tid.x;
    float sum = args.has_bias && column < args.output_dim ?
        bias[column] : 0.0f;
    uint tile_count = (args.input_dim + 15) / 16;
    for (uint tile = 0; tile < tile_count; tile++) {
        uint input_k = tile * 16 + tid.x;
        input_tile[tid.y][tid.x] =
            row < args.rows && input_k < args.input_dim ?
            input[row * args.input_dim + input_k] : 0.0f;
        uint weight_k = tile * 16 + tid.y;
        weight_tile[tid.y][tid.x] =
            column < args.output_dim && weight_k < args.input_dim ?
            weight[column * args.input_dim + weight_k] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; k++)
            sum = fma(input_tile[tid.y][k], weight_tile[k][tid.x], sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < args.rows && column < args.output_dim)
        output[row * args.output_dim + column] = h3_f32_to_bf16(sum);
}

kernel void h3_linear_f32_tiled_bf16_map(
                                device const float *input [[buffer(0)]],
                                device const float *weight [[buffer(1)]],
                                device const float *bias [[buffer(2)]],
                                device ushort *output [[buffer(3)]],
                                constant linear_args &args [[buffer(4)]],
                                device const uint *row_map [[buffer(5)]],
                                uint2 tid [[thread_position_in_threadgroup]],
                                uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float input_tile[16][16];
    threadgroup float weight_tile[16][16];
    threadgroup uint output_rows[16];
    uint row = group.y * 16 + tid.y;
    uint column = group.x * 16 + tid.x;
    if (tid.x == 0)
        output_rows[tid.y] = row < args.rows ? row_map[row] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = args.has_bias && column < args.output_dim ?
        bias[column] : 0.0f;
    uint tile_count = (args.input_dim + 15) / 16;
    for (uint tile = 0; tile < tile_count; tile++) {
        uint input_k = tile * 16 + tid.x;
        input_tile[tid.y][tid.x] =
            row < args.rows && input_k < args.input_dim ?
            input[row * args.input_dim + input_k] : 0.0f;
        uint weight_k = tile * 16 + tid.y;
        weight_tile[tid.y][tid.x] =
            column < args.output_dim && weight_k < args.input_dim ?
            weight[column * args.input_dim + weight_k] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; k++)
            sum = fma(input_tile[tid.y][k], weight_tile[k][tid.x], sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < args.rows && column < args.output_dim)
        output[output_rows[tid.y] * args.output_dim + column] =
            h3_f32_to_bf16(sum);
}

kernel void h3_silu_f32(device const float *input [[buffer(0)]],
                        device float *output [[buffer(1)]],
                        constant uint &count [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float value = input[gid];
    output[gid] = value / (1.0f + exp(-value));
}

kernel void h3_cast_f32_to_bf16(device const float *input [[buffer(0)]],
                                device ushort *output [[buffer(1)]],
                                constant uint &count [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid < count) output[gid] = h3_f32_to_bf16(input[gid]);
}

kernel void h3_cast_bf16_to_f32(device const ushort *input [[buffer(0)]],
                                device float *output [[buffer(1)]],
                                constant uint &count [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid < count) output[gid] = h3_bf16_to_f32(input[gid]);
}

/* Q/K/V for an experimental, opt-in attention backend (h3_gpu_ccv_attention)
 * that expects FP16 in [rows, heads, head_dim] regardless of this engine's
 * own [heads, rows, head_dim] producer/consumer optimization for the
 * default SDPA path - head_major selects which layout `input` is really
 * in, `output` is always written [rows, heads, head_dim]. */
struct cast_qkv_layout_args { uint rows; uint heads; uint head_dim; uint head_major; };

kernel void h3_cast_bf16_to_f16_qkv(device const ushort *input [[buffer(0)]],
                                    device half *output [[buffer(1)]],
                                    constant cast_qkv_layout_args &args [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]]) {
    uint total = args.rows * args.heads * args.head_dim;
    if (gid >= total) return;
    uint d = gid % args.head_dim;
    uint row_head = gid / args.head_dim;
    uint head = row_head % args.heads;
    uint row = row_head / args.heads;
    uint src_index = args.head_major
        ? (head * args.rows + row) * args.head_dim + d
        : gid;
    output[gid] = half(h3_bf16_to_f32(input[src_index]));
}

kernel void h3_cast_f16_to_bf16_flat(device const half *input [[buffer(0)]],
                                     device ushort *output [[buffer(1)]],
                                     constant uint &count [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]]) {
    if (gid < count) output[gid] = h3_f32_to_bf16(float(input[gid]));
}

struct cast_bias_args { uint rows; uint width; };

kernel void h3_cast_bf16_to_f32_bias(device const ushort *input [[buffer(0)]],
                                     device const float *bias [[buffer(1)]],
                                     device float *output [[buffer(2)]],
                                     constant cast_bias_args &args [[buffer(3)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint index = row * args.width + column;
    output[index] = h3_bf16_to_f32(input[index]) + bias[column];
}

/* LoRA deltas are computed per row group (see h3_lora.h); QKV groups land on
 * the per-head interleaved rows of the official checkpoint layout.
 *
 * A LoRA delta is typically ~400x smaller than an int8 quantization step and
 * below BF16 resolution too, so round-to-nearest erases it. Both kernels
 * round stochastically instead (unbiased: the expected patched weight is
 * exactly W + delta), with a hash of the element position as the random
 * source so every step patches identically. */
struct lora_rows_args {
    uint rows;
    uint columns;
    uint interleaved;
    uint group;
    uint head_dim;
    uint seed;
};

inline uint h3_lora_target_row(constant lora_rows_args &args, uint row) {
    if (!args.interleaved) return row;
    return (row / args.head_dim) * 3u * args.head_dim +
           args.group * args.head_dim + row % args.head_dim;
}

inline uint h3_lora_hash(uint value, uint seed) {
    uint h = value * 0x9E3779B1u ^ seed * 0x85EBCA77u;
    h ^= h >> 16; h *= 0x7FEB352Du;
    h ^= h >> 15; h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h;
}

kernel void h3_lora_add_rows_bf16(device ushort *weight [[buffer(0)]],
                                  device const ushort *delta [[buffer(1)]],
                                  constant lora_rows_args &args [[buffer(2)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.columns) return;
    uint target = h3_lora_target_row(args, row) * args.columns + column;
    float sum = h3_bf16_to_f32(weight[target]) +
                h3_bf16_to_f32(delta[row * args.columns + column]);
    uint bits = as_type<uint>(sum);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += h3_lora_hash(target, args.seed) & 0xffffu;
    weight[target] = ushort(bits >> 16);
}

/* Per-row int8 weights keep their scale: q += stochastic_round(delta /
 * scale), clamped to +-127. Statistically this matches quantizing W + delta
 * from full precision, without re-rounding the base weight. */
kernel void h3_lora_add_int8(device char *weight [[buffer(0)]],
                             device const float *scales [[buffer(1)]],
                             device const ushort *delta [[buffer(2)]],
                             constant lora_rows_args &args [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.columns) return;
    uint target_row = h3_lora_target_row(args, row);
    uint target = target_row * args.columns + column;
    float steps = h3_bf16_to_f32(delta[row * args.columns + column]) /
                  scales[target_row];
    float whole = floor(steps);
    float random = float(h3_lora_hash(target, args.seed) >> 8) * (1.0f / 16777216.0f);
    int change = int(whole) + (random < steps - whole ? 1 : 0);
    weight[target] = char(clamp(int(weight[target]) + change, -127, 127));
}

struct norm_args {
    uint rows;
    uint width;
    float epsilon;
};

struct qkv_args {
    uint sequence;
    uint heads;
    uint head_dim;
    uint rope_half;
    uint grouped;
    float epsilon;
};

kernel void h3_rms_norm_f32(device const float *input [[buffer(0)]],
                            device const float *weight [[buffer(1)]],
                            device float *output [[buffer(2)]],
                            constant norm_args &args [[buffer(3)]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint3 thread_position [[thread_position_in_threadgroup]],
                            uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    device const float *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint k = tid; k < args.width; k += threads)
        local_sum = fma(x[k], x[k], local_sum);
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    for (uint column = tid; column < args.width; column += threads)
        output[row * args.width + column] = x[column] * inverse * weight[column];
}

struct scale_add_args { uint rows; uint width; };

kernel void h3_scale_add_f32(device const float *residual [[buffer(0)]],
                             device const float *branch [[buffer(1)]],
                             device const float *scale [[buffer(2)]],
                             device float *output [[buffer(3)]],
                             constant scale_add_args &args [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint index = row * args.width + column;
    output[index] = residual[index] + branch[index] * scale[column];
}

kernel void h3_layer_norm_f32(device const float *input [[buffer(0)]],
                              device const float *weight [[buffer(1)]],
                              device const float *bias [[buffer(2)]],
                              device float *output [[buffer(3)]],
                              constant norm_args &args [[buffer(4)]],
                              uint3 group [[threadgroup_position_in_grid]],
                              uint3 thread_position [[thread_position_in_threadgroup]],
                              uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    device const float *x = input + row * args.width;
    float local = 0.0f;
    for (uint k = tid; k < args.width; k += threads) local += x[k];
    reductions[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = reductions[0] / float(args.width);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    local = 0.0f;
    for (uint k = tid; k < args.width; k += threads) {
        float centered = x[k] - mean;
        local = fma(centered, centered, local);
    }
    reductions[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    for (uint column = tid; column < args.width; column += threads)
        output[row * args.width + column] =
            (x[column] - mean) * inverse * weight[column] + bias[column];
}

kernel void h3_video_qkv_rope_f32(
                            device const float *qkv [[buffer(0)]],
                            device const float *rope_cos [[buffer(1)]],
                            device const float *rope_sin [[buffer(2)]],
                            device float *query [[buffer(3)]],
                            device float *key [[buffer(4)]],
                            device float *value [[buffer(5)]],
                            constant qkv_args &args [[buffer(6)]],
                            uint3 gid [[thread_position_in_grid]]) {
    uint dimension = gid.x;
    uint head = gid.y;
    uint row = gid.z;
    if (dimension >= args.head_dim || head >= args.heads || row >= args.sequence) return;
    uint base = (row * args.heads + head) * args.head_dim * 3;
    float q_sum = 0.0f, k_sum = 0.0f;
    for (uint d = 0; d < args.head_dim; d++) {
        float q = qkv[base + d];
        float k = qkv[base + args.head_dim + d];
        q_sum = fma(q, q, q_sum);
        k_sum = fma(k, k, k_sum);
    }
    float qi = rsqrt(q_sum / float(args.head_dim) + args.epsilon);
    float ki = rsqrt(k_sum / float(args.head_dim) + args.epsilon);
    float q0 = qkv[base + dimension] * qi;
    float k0 = qkv[base + args.head_dim + dimension] * ki;
    if (dimension < args.rope_half) {
        uint pair = dimension + args.rope_half;
        float q1 = qkv[base + pair] * qi;
        float k1 = qkv[base + args.head_dim + pair] * ki;
        float c = rope_cos[row * args.rope_half + dimension];
        float s = rope_sin[row * args.rope_half + dimension];
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2) {
        uint pair = dimension - args.rope_half;
        float q1 = qkv[base + pair] * qi;
        float k1 = qkv[base + args.head_dim + pair] * ki;
        float c = rope_cos[row * args.rope_half + pair];
        float s = rope_sin[row * args.rope_half + pair];
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint output_index = (row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = q0;
    key[output_index] = k0;
    value[output_index] = qkv[base + args.head_dim * 2 + dimension];
}

struct adaln_args {
    uint rows;
    uint width;
    uint slots;
    uint shift_slot;
    uint scale_slot;
    float epsilon;
};

kernel void h3_adaln_f32(device const float *input [[buffer(0)]],
                         device const float *weight [[buffer(1)]],
                         device const float *modulation [[buffer(2)]],
                         device const uint *row_map [[buffer(3)]],
                         device float *output [[buffer(4)]],
                         constant adaln_args &args [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    device const float *x = input + row * args.width;
    float sum = 0.0f;
    for (uint k = 0; k < args.width; k++) sum = fma(x[k], x[k], sum);
    float normalized = x[column] * rsqrt(sum / float(args.width) + args.epsilon) * weight[column];
    uint modulation_row = row_map[row];
    uint base = modulation_row * args.slots * args.width;
    float shift = modulation[base + args.shift_slot * args.width + column];
    float scale = modulation[base + args.scale_slot * args.width + column];
    output[row * args.width + column] = normalized * (1.0f + scale) + shift;
}

struct gate_args {
    uint rows;
    uint width;
    uint slots;
    uint gate_slot;
};

kernel void h3_gate_f32(device const float *residual [[buffer(0)]],
                        device const float *branch [[buffer(1)]],
                        device const float *modulation [[buffer(2)]],
                        device const uint *row_map [[buffer(3)]],
                        device float *output [[buffer(4)]],
                        constant gate_args &args [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint base = row_map[row] * args.slots * args.width;
    float gate = modulation[base + args.gate_slot * args.width + column];
    uint index = row * args.width + column;
    output[index] = residual[index] + branch[index] * gate;
}

kernel void h3_qkv_rope_f32(device const float *qkv [[buffer(0)]],
                            device const float *q_weight [[buffer(1)]],
                            device const float *k_weight [[buffer(2)]],
                            device const float *rope_cos [[buffer(3)]],
                            device const float *rope_sin [[buffer(4)]],
                            device float *query [[buffer(5)]],
                            device float *key [[buffer(6)]],
                            device float *value [[buffer(7)]],
                            constant qkv_args &args [[buffer(8)]],
                            uint3 gid [[thread_position_in_grid]]) {
    uint dimension = gid.x;
    uint head = gid.y;
    uint row = gid.z;
    if (dimension >= args.head_dim || head >= args.heads || row >= args.sequence) return;
    uint inner = args.heads * args.head_dim;
    uint row_base = row * inner * 3;
    uint q_base = row_base + head * args.head_dim;
    uint k_base = q_base + inner;
    uint v_base = q_base + inner * 2;
    if (args.grouped) {
        q_base = row_base + head * args.head_dim * 3;
        k_base = q_base + args.head_dim;
        v_base = k_base + args.head_dim;
    }
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint d = 0; d < args.head_dim; d++) {
        float q = qkv[q_base + d];
        float k = qkv[k_base + d];
        q_sum = fma(q, q, q_sum);
        k_sum = fma(k, k, k_sum);
    }
    float q_inverse = rsqrt(q_sum / float(args.head_dim) + args.epsilon);
    float k_inverse = rsqrt(k_sum / float(args.head_dim) + args.epsilon);
    float q0 = qkv[q_base + dimension] * q_inverse * q_weight[dimension];
    float k0 = qkv[k_base + dimension] * k_inverse * k_weight[dimension];
    if (dimension < args.rope_half) {
        uint pair = dimension + args.rope_half;
        float q1 = qkv[q_base + pair] * q_inverse * q_weight[pair];
        float k1 = qkv[k_base + pair] * k_inverse * k_weight[pair];
        float c = rope_cos[row * args.rope_half + dimension];
        float s = rope_sin[row * args.rope_half + dimension];
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2) {
        uint pair = dimension - args.rope_half;
        float q1 = qkv[q_base + pair] * q_inverse * q_weight[pair];
        float k1 = qkv[k_base + pair] * k_inverse * k_weight[pair];
        float c = rope_cos[row * args.rope_half + pair];
        float s = rope_sin[row * args.rope_half + pair];
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint output_index = (row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = q0;
    key[output_index] = k0;
    value[output_index] = qkv[v_base + dimension];
}

struct swiglu_args {
    uint rows;
    uint width;
};

kernel void h3_swiglu_f32(device const float *fused [[buffer(0)]],
                          device float *output [[buffer(1)]],
                          constant swiglu_args &args [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint base = row * args.width * 2;
    float gate = fused[base + column];
    float up = fused[base + args.width + column];
    output[row * args.width + column] = gate / (1.0f + exp(-gate)) * up;
}

struct vae_encoder_pad_args {
    uint batch;
    uint depth;
    uint height;
    uint width;
    uint channels;
    uint depth_front;
    uint height_before;
    uint height_after;
    uint width_before;
    uint width_after;
};

static int h3_reflect_coordinate(int coordinate, int length) {
    if (coordinate < 0) return -coordinate;
    if (coordinate >= length) return 2 * length - coordinate - 2;
    return coordinate;
}

kernel void h3_vae_encoder_pad_f32(
                            device const float *input [[buffer(0)]],
                            device float *output [[buffer(1)]],
                            constant vae_encoder_pad_args &args [[buffer(2)]],
                            uint3 gid [[thread_position_in_grid]]) {
    uint channel = gid.x;
    uint out_x = gid.y;
    uint out_height = args.height + args.height_before + args.height_after;
    uint out_width = args.width + args.width_before + args.width_after;
    uint out_depth = args.depth + args.depth_front;
    uint plane = gid.z;
    if (channel >= args.channels || out_x >= out_width ||
        plane >= args.batch * out_depth * out_height) return;
    uint out_y = plane % out_height;
    uint temporal_plane = plane / out_height;
    uint out_t = temporal_plane % out_depth;
    uint batch = temporal_plane / out_depth;
    size_t destination = ((((size_t)batch * out_depth + out_t) * out_height +
                           out_y) * out_width + out_x) * args.channels +
                         channel;
    if (out_t < args.depth_front) {
        output[destination] = 0.0f;
        return;
    }
    int source_y = h3_reflect_coordinate(
        int(out_y) - int(args.height_before), int(args.height));
    int source_x = h3_reflect_coordinate(
        int(out_x) - int(args.width_before), int(args.width));
    uint source_t = out_t - args.depth_front;
    size_t source = ((((size_t)batch * args.depth + source_t) * args.height +
                      uint(source_y)) * args.width + uint(source_x)) *
                    args.channels + channel;
    output[destination] = input[source];
}

struct vae_encoder_norm_args {
    uint batch;
    uint depth;
    uint height;
    uint width;
    uint channels;
    uint groups;
    float epsilon;
};

kernel void h3_vae_encoder_group_norm_silu_f32(
                            device const float *input [[buffer(0)]],
                            device const float *weight [[buffer(1)]],
                            device const float *bias [[buffer(2)]],
                            device float *output [[buffer(3)]],
                            constant vae_encoder_norm_args &args [[buffer(4)]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint3 thread_position [[thread_position_in_threadgroup]],
                            uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint rows = args.batch * args.depth * args.groups;
    if (row >= rows) return;
    uint channels_per_group = args.channels / args.groups;
    uint group_index = row % args.groups;
    uint temporal_plane = row / args.groups;
    uint elements = args.height * args.width * channels_per_group;
    threadgroup float reductions[256];
    float local = 0.0f;
    for (uint index = tid; index < elements; index += threadgroup_size.x) {
        uint spatial = index / channels_per_group;
        uint channel = group_index * channels_per_group +
                       index % channels_per_group;
        size_t source = ((size_t)temporal_plane * args.height * args.width +
                         spatial) * args.channels + channel;
        local += input[source];
    }
    reductions[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadgroup_size.x / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = reductions[0] / float(elements);
    /* Same hazard as in h3_gqa_causal_bf16: all threads read the mean
     * before reductions[] is reused for the variance. */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    local = 0.0f;
    for (uint index = tid; index < elements; index += threadgroup_size.x) {
        uint spatial = index / channels_per_group;
        uint channel = group_index * channels_per_group +
                       index % channels_per_group;
        size_t source = ((size_t)temporal_plane * args.height * args.width +
                         spatial) * args.channels + channel;
        float centered = input[source] - mean;
        local = fma(centered, centered, local);
    }
    reductions[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadgroup_size.x / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(elements) + args.epsilon);
    for (uint index = tid; index < elements; index += threadgroup_size.x) {
        uint spatial = index / channels_per_group;
        uint channel = group_index * channels_per_group +
                       index % channels_per_group;
        size_t destination =
            ((size_t)temporal_plane * args.height * args.width + spatial) *
            args.channels + channel;
        float value = (input[destination] - mean) * inverse * weight[channel] +
                      bias[channel];
        output[destination] = value / (1.0f + exp(-value));
    }
}

/* GroupNorm + SiLU for the fast video encoder, in three passes that each
 * read the frame in order: neighbouring threads read neighbouring channels
 * (h3_vae_encoder_group_norm_silu_f32 gives a whole threadgroup to one
 * group's 4-32 channels, then steps over the other groups' channels, so
 * most of every memory read is wasted). Each frame is cut into `chunks`
 * runs of `chunk_pixels` pixels, one threadgroup each: the first pass sums
 * each group's values in its run, the second the squares about the mean,
 * the third normalizes. The same two-pass mean and variance in F32; only
 * the order of the sums differs. 256 threads; channels / 4 divides 256 and
 * each group has a multiple of 4 channels. */
struct vae_encoder_norm_chunk_args {
    uint frames;
    uint pixels;
    uint channels;
    uint groups;
    uint chunks;
    uint chunk_pixels;
    float epsilon;
};

/* A run's per-group sum of `value(x)` into partial[frame][chunk][group]. */
static void h3_norm_chunk_sums(device const float4 *input,
                               device float *partial,
                               threadgroup float *shared,
                               constant vae_encoder_norm_chunk_args &args,
                               uint chunk, uint frame, uint tid,
                               threadgroup const float *mean) {
    uint vectors = args.channels / 4;
    uint lanes = 256 / vectors;
    uint vector = tid % vectors;
    uint first = chunk * args.chunk_pixels;
    uint last = min(first + args.chunk_pixels, args.pixels);
    uint group = vector * 4 / (args.channels / args.groups);
    float centre = mean ? mean[group] : 0.0f;
    float4 local = 0.0f;
    device const float4 *plane = input + (size_t)frame * args.pixels * vectors;
    for (uint pixel = first + tid / vectors; pixel < last; pixel += lanes) {
        float4 value = plane[(size_t)pixel * vectors + vector];
        if (mean) {
            value -= centre;
            local = fma(value, value, local);
        } else {
            local += value;
        }
    }
    shared[tid] = local.x + local.y + local.z + local.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < args.groups) {
        uint per_group = args.channels / args.groups / 4;
        float sum = 0.0f;
        for (uint lane = 0; lane < lanes; lane++)
            for (uint item = 0; item < per_group; item++)
                sum += shared[lane * vectors + tid * per_group + item];
        partial[((size_t)frame * args.chunks + chunk) * args.groups + tid] = sum;
    }
}

/* Each group's mean (and, with `squares`, inverse deviation) of `frame`. */
static void h3_norm_chunk_stats(device const float *sums,
                                device const float *squares,
                                threadgroup float *mean,
                                threadgroup float *inverse,
                                constant vae_encoder_norm_chunk_args &args,
                                uint frame, uint tid) {
    if (tid < args.groups) {
        float count = float(args.pixels) * float(args.channels / args.groups);
        float sum = 0.0f;
        for (uint chunk = 0; chunk < args.chunks; chunk++)
            sum += sums[((size_t)frame * args.chunks + chunk) * args.groups + tid];
        mean[tid] = sum / count;
        if (squares) {
            float square = 0.0f;
            for (uint chunk = 0; chunk < args.chunks; chunk++)
                square += squares[((size_t)frame * args.chunks + chunk) *
                                  args.groups + tid];
            inverse[tid] = rsqrt(square / count + args.epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void h3_vae_encoder_norm_chunk_sum_f32(
                            device const float4 *input [[buffer(0)]],
                            device float *sums [[buffer(1)]],
                            constant vae_encoder_norm_chunk_args &args [[buffer(2)]],
                            uint2 position [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[256];
    h3_norm_chunk_sums(input, sums, shared, args, position.x, position.y, tid,
                       nullptr);
}

kernel void h3_vae_encoder_norm_chunk_square_f32(
                            device const float4 *input [[buffer(0)]],
                            device const float *sums [[buffer(1)]],
                            device float *squares [[buffer(2)]],
                            constant vae_encoder_norm_chunk_args &args [[buffer(3)]],
                            uint2 position [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[256];
    threadgroup float mean[64];
    h3_norm_chunk_stats(sums, nullptr, mean, nullptr, args, position.y, tid);
    h3_norm_chunk_sums(input, squares, shared, args, position.x, position.y, tid,
                       mean);
}

kernel void h3_vae_encoder_norm_chunk_apply_silu_f32(
                            device const float4 *input [[buffer(0)]],
                            device const float *sums [[buffer(1)]],
                            device const float *squares [[buffer(2)]],
                            device const float4 *weight [[buffer(3)]],
                            device const float4 *bias [[buffer(4)]],
                            device float4 *output [[buffer(5)]],
                            constant vae_encoder_norm_chunk_args &args [[buffer(6)]],
                            uint2 position [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float mean[64];
    threadgroup float inverse[64];
    h3_norm_chunk_stats(sums, squares, mean, inverse, args, position.y, tid);
    uint vectors = args.channels / 4;
    uint lanes = 256 / vectors;
    uint vector = tid % vectors;
    uint group = vector * 4 / (args.channels / args.groups);
    float centre = mean[group];
    float4 scale = weight[vector] * inverse[group];
    float4 shift = bias[vector];
    uint first = position.x * args.chunk_pixels;
    uint last = min(first + args.chunk_pixels, args.pixels);
    size_t plane = (size_t)position.y * args.pixels * vectors;
    for (uint pixel = first + tid / vectors; pixel < last; pixel += lanes) {
        size_t at = plane + (size_t)pixel * vectors + vector;
        float4 value = fma(input[at] - centre, scale, shift);
        output[at] = value / (1.0f + exp(-value));
    }
}

struct weight_norm_args {
    uint outer;
    uint inner;
};

kernel void h3_weight_norm_f32(device const float *vector [[buffer(0)]],
                               device const float *magnitude [[buffer(1)]],
                               device float *output [[buffer(2)]],
                               constant weight_norm_args &args [[buffer(3)]],
                               uint row [[thread_position_in_grid]]) {
    if (row >= args.outer) return;
    uint base = row * args.inner;
    float square_sum = 0.0f;
    for (uint index = 0; index < args.inner; index++) {
        float value = vector[base + index];
        square_sum = fma(value, value, square_sum);
    }
    float scale = magnitude[row] * rsqrt(square_sum);
    for (uint index = 0; index < args.inner; index++)
        output[base + index] = vector[base + index] * scale;
}

struct add_scaled_args {
    uint elements;
    float left_scale;
    float right_scale;
};

kernel void h3_add_scaled_f32(device const float *left [[buffer(0)]],
                              device const float *right [[buffer(1)]],
                              device float *output [[buffer(2)]],
                              constant add_scaled_args &args [[buffer(3)]],
                              uint index [[thread_position_in_grid]]) {
    if (index < args.elements)
        output[index] = left[index] * args.left_scale +
                        right[index] * args.right_scale;
}

struct audio_activation_args {
    uint batch;
    uint length;
    uint channels;
};

/* BigVGAN's fixed 2x upsample -> SnakeBeta -> 2x low-pass downsample is
 * fused at the original rate. This avoids materializing the doubled waveform
 * for each of the decoder's 127 alias-free activations. */
kernel void h3_alias_free_snake_f32(
        device const float *input [[buffer(0)]],
        device const float *alpha_log [[buffer(1)]],
        device const float *beta_log [[buffer(2)]],
        device const float *upsample_filter [[buffer(3)]],
        device const float *downsample_filter [[buffer(4)]],
        device float *output [[buffer(5)]],
        constant audio_activation_args &args [[buffer(6)]],
        uint3 gid [[thread_position_in_grid]]) {
    uint channel = gid.x;
    uint time = gid.y;
    uint batch = gid.z;
    if (channel >= args.channels || time >= args.length ||
        batch >= args.batch) return;
    float alpha = exp(alpha_log[channel]);
    float beta = exp(beta_log[channel]);
    float result = 0.0f;
    for (int down_k = 0; down_k < 12; down_k++) {
        int up_time = int(time * 2) + down_k - 5;
        up_time = clamp(up_time, 0, int(args.length * 2) - 1);
        int raw_time = up_time + 15;
        float upsampled = 0.0f;
        for (int up_k = 0; up_k < 12; up_k++) {
            int numerator = raw_time - up_k;
            if (numerator < 0 || (numerator & 1)) continue;
            int padded_time = numerator / 2;
            int source_time = clamp(padded_time - 5, 0,
                                    int(args.length) - 1);
            uint source = (batch * args.length + uint(source_time)) *
                          args.channels + channel;
            upsampled = fma(input[source], 2.0f * upsample_filter[up_k],
                            upsampled);
        }
        float sine = sin(alpha * upsampled);
        float activated = upsampled + sine * sine / (beta + 1e-9f);
        result = fma(activated, downsample_filter[down_k], result);
    }
    uint destination = (batch * args.length + time) * args.channels + channel;
    output[destination] = result;
}

kernel void h3_snake1d_f32(device const float *input [[buffer(0)]],
                           device const float *alpha [[buffer(1)]],
                           device float *output [[buffer(2)]],
                           constant audio_activation_args &args [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    uint count = args.batch * args.length * args.channels;
    if (gid >= count) return;
    float a = alpha[gid % args.channels];
    float x = input[gid];
    float wave = sin(a * x);
    output[gid] = x + wave * wave / (a + 1e-9f);
}

struct audio_qkv_args {
    uint batch;
    uint length;
    uint heads;
    uint head_dim;
};

kernel void h3_audio_qkv_split_f32(
                           device const float *qkv [[buffer(0)]],
                           device const float *q_bias [[buffer(1)]],
                           device const float *k_bias [[buffer(2)]],
                           device const float *v_bias [[buffer(3)]],
                           device float *query [[buffer(4)]],
                           device float *key [[buffer(5)]],
                           device float *value [[buffer(6)]],
                           constant audio_qkv_args &args [[buffer(7)]],
                           uint gid [[thread_position_in_grid]]) {
    uint width = args.heads * args.head_dim;
    uint count = args.batch * args.length * width;
    if (gid >= count) return;
    uint column = gid % width;
    uint row = gid / width;
    uint base = row * width * 3;
    query[gid] = qkv[base + column] + q_bias[column];
    key[gid] = qkv[base + width + column] + k_bias[column];
    value[gid] = qkv[base + width * 2 + column] + v_bias[column];
}

struct audio_pool_args {
    uint batch;
    uint length;
    uint heads;
    uint head_dim;
    uint output_dim;
};

kernel void h3_audio_attention_pool_f32(
                           device const float *attended [[buffer(0)]],
                           device float *output [[buffer(1)]],
                           constant audio_pool_args &args [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    uint count = args.batch * args.length * args.output_dim;
    if (gid >= count) return;
    uint column = gid % args.output_dim;
    uint row = gid / args.output_dim;
    uint pool = args.head_dim / args.output_dim;
    float sum = 0.0f;
    for (uint head = 0; head < args.heads; head++) {
        uint base = (row * args.heads + head) * args.head_dim + column * pool;
        for (uint item = 0; item < pool; item++) sum += attended[base + item];
    }
    output[gid] = sum / float(args.heads * pool);
}

kernel void h3_geglu_f32(device const float *gate [[buffer(0)]],
                          device const float *linear [[buffer(1)]],
                          device float *output [[buffer(2)]],
                          constant uint &count [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float x = gate[gid];
    float cube = x * x * x;
    /* clamped: Metal's tanh can return NaN for large arguments, and it is
     * +-1 to float precision beyond |x| = 20 */
    float inner = clamp(0.7978845608028654f * (x + 0.044715f * cube), -20.0f, 20.0f);
    float gelu = 0.5f * x * (1.0f + tanh(inner));
    output[gid] = gelu * linear[gid];
}

struct clip_args {
    uint elements;
    float minimum;
    float maximum;
};

kernel void h3_clip_f32(device const float *input [[buffer(0)]],
                        device float *output [[buffer(1)]],
                        constant clip_args &args [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
    if (index < args.elements)
        output[index] = clamp(input[index], args.minimum, args.maximum);
}

kernel void h3_linear_bf16(device const ushort *input [[buffer(0)]],
                           device const ushort *weight [[buffer(1)]],
                           device const ushort *bias [[buffer(2)]],
                           device ushort *output [[buffer(3)]],
                           constant linear_args &args [[buffer(4)]],
                           uint2 tid [[thread_position_in_threadgroup]],
                           uint2 group [[threadgroup_position_in_grid]]) {
    /* Adapted from Iris's short-sequence Qwen path. Each 16x16 weight/input
     * tile is loaded once per threadgroup instead of once per output scalar. */
    threadgroup float input_tile[16][16];
    threadgroup float weight_tile[16][16];
    uint row = group.y * 16 + tid.y;
    uint column = group.x * 16 + tid.x;
    float sum = args.has_bias && column < args.output_dim ?
        h3_bf16_to_f32(bias[column]) : 0.0f;
    uint tile_count = (args.input_dim + 15) / 16;
    for (uint tile = 0; tile < tile_count; tile++) {
        uint input_k = tile * 16 + tid.x;
        input_tile[tid.y][tid.x] =
            row < args.rows && input_k < args.input_dim ?
            h3_bf16_to_f32(input[row * args.input_dim + input_k]) : 0.0f;
        uint weight_k = tile * 16 + tid.y;
        weight_tile[tid.y][tid.x] =
            column < args.output_dim && weight_k < args.input_dim ?
            h3_bf16_to_f32(weight[column * args.input_dim + weight_k]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; k++) {
            sum = fma(input_tile[tid.y][k], weight_tile[k][tid.x], sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < args.rows && column < args.output_dim) {
        output[row * args.output_dim + column] = h3_f32_to_bf16(sum);
    }
}

/* Draw Things/ccv-style dynamic symmetric row reduction. This helper is also
 * used by portable fused epilogues, so keep it outside the Metal 4 guard. */
inline float h3_int8_reduce_max(float value, threadgroup float *scratch,
                                ushort simdgroup, ushort lane) {
    value = max(value, simd_shuffle_xor(value, 16));
    value = max(value, simd_shuffle_xor(value, 8));
    value = max(value, simd_shuffle_xor(value, 4));
    value = max(value, simd_shuffle_xor(value, 2));
    value = max(value, simd_shuffle_xor(value, 1));
    if (lane == 0) scratch[simdgroup] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        value = lane < 8 ? scratch[lane] : 0.0f;
        value = max(value, simd_shuffle_xor(value, 16));
        value = max(value, simd_shuffle_xor(value, 8));
        value = max(value, simd_shuffle_xor(value, 4));
        value = max(value, simd_shuffle_xor(value, 2));
        value = max(value, simd_shuffle_xor(value, 1));
        if (lane == 0) scratch[0] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

#ifdef H3_METAL_HAS_TENSOR
/* Draw Things' Metal 4 matmul schedules neighboring row/column tiles in
 * Morton order. The decoder is adapted from ccv's BSD-3-Clause NAMatMul;
 * see THIRD_PARTY_NOTICES.md. Keep it local so the ordinary Metal path stays
 * buildable on pre-Metal-4 systems. */
inline uint h3_compact_morton_even_bits(uint value) {
    value &= 0x55555555u;
    value = (value | (value >> 1)) & 0x33333333u;
    value = (value | (value >> 2)) & 0x0f0f0f0fu;
    value = (value | (value >> 4)) & 0x00ff00ffu;
    value = (value | (value >> 8)) & 0x0000ffffu;
    return value;
}

inline uint h3_lower_bits_mask(uint bits) {
    return bits ? (1u << bits) - 1u : 0u;
}

inline uint2 h3_morton_decode_rectangular(uint code, uint x_bits,
                                          uint y_bits) {
    uint paired_bits = min(x_bits, y_bits);
    uint paired_code = code & h3_lower_bits_mask(paired_bits * 2);
    uint2 tile = uint2(h3_compact_morton_even_bits(paired_code),
                       h3_compact_morton_even_bits(paired_code >> 1));
    uint tail = code >> (paired_bits * 2);
    if (x_bits > paired_bits) {
        uint extra = x_bits - paired_bits;
        tile.x |= (tail & h3_lower_bits_mask(extra)) << paired_bits;
        tail >>= extra;
    }
    if (y_bits > paired_bits) tile.y |= tail << paired_bits;
    return tile;
}

/* Compact four-column Morton blocks. Full 4x4 row blocks use a true Morton
 * code; the final 1-3 row strip is packed without launching empty groups. */
inline uint2 h3_morton_decode_compact4(uint code, uint row_tiles) {
    uint groups_per_column_block = row_tiles * 4;
    uint column_block = code / groups_per_column_block;
    uint local = code - column_block * groups_per_column_block;
    uint full_rows = row_tiles & ~3u;
    uint full_groups = full_rows * 4;
    if (local < full_groups) {
        uint row_block = local >> 4;
        uint2 tile = uint2(h3_compact_morton_even_bits(local & 15u),
                           h3_compact_morton_even_bits((local & 15u) >> 1));
        return uint2(row_block * 4 + tile.x,
                     column_block * 4 + tile.y);
    }
    uint tail_rows = row_tiles - full_rows;
    uint tail = local - full_groups;
    return uint2(full_rows + tail % tail_rows,
                 column_block * 4 + tail / tail_rows);
}

/* Exact compact Morton walk for a final 1-3 column strip as well. */
inline uint2 h3_morton_decode_compact(uint code, uint row_tiles,
                                      uint column_tiles) {
    uint full_columns = column_tiles & ~3u;
    uint full_groups = row_tiles * full_columns;
    if (code < full_groups)
        return h3_morton_decode_compact4(code, row_tiles);
    uint tail_columns = column_tiles - full_columns;
    uint tail = code - full_groups;
    return uint2(tail / tail_columns,
                 full_columns + tail % tail_columns);
}

/*
 * M5 Metal 4/TensorOps path for the large H3 matrices.  This follows the
 * retained direct-RHS design in ds4.c: the contiguous dimension comes first
 * in Metal tensor extents, checkpoint weights are transposed logically rather
 * than copied, and TensorOps handles the final partial row tile.  Native
 * bfloat inputs avoid both a format conversion and a duplicate weight arena.
 */
kernel void h3_linear_bf16_nax_r128(
                           device bfloat *input [[buffer(0)]],
                           device bfloat *weight [[buffer(1)]],
                           device bfloat *output [[buffer(3)]],
                           constant linear_args &args [[buffer(4)]],
                           uint2 group [[threadgroup_position_in_grid]]) {
    auto x = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim, (int)args.rows));
    auto w = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        weight,
        dextents<int32_t, 2>((int)args.input_dim, (int)args.output_dim));
    auto y = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        output,
        dextents<int32_t, 2>((int)args.output_dim, (int)args.rows));
    auto mx = x.slice(0, (int)group.x * 128);
    auto mw = w.slice(0, (int)group.y * 64);
    auto my = y.slice((int)group.y * 64, (int)group.x * 128);
    matmul2d<matmul2d_descriptor(128, 64, dynamic_extent,
                                false, true, false),
              execution_simdgroups<4>> mm;
    mm.run(mx, mw, my);
}

kernel void h3_linear_bf16_nax_r128_morton(
                           device bfloat *input [[buffer(0)]],
                           device bfloat *weight [[buffer(1)]],
                           device bfloat *output [[buffer(3)]],
                           constant linear_args &args [[buffer(4)]],
                           uint code [[threadgroup_position_in_grid]]) {
    uint row_tiles = (args.rows + 127) / 128;
    uint column_tiles = (args.output_dim + 63) / 64;
    uint row_bits = row_tiles <= 1 ? 0 : 32 - clz(row_tiles - 1);
    uint column_bits = column_tiles <= 1 ? 0 : 32 - clz(column_tiles - 1);
    uint2 group = h3_morton_decode_rectangular(
        code, row_bits, column_bits);
    if (group.x >= row_tiles || group.y >= column_tiles) return;
    auto x = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim, (int)args.rows));
    auto w = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        weight,
        dextents<int32_t, 2>((int)args.input_dim, (int)args.output_dim));
    auto y = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        output,
        dextents<int32_t, 2>((int)args.output_dim, (int)args.rows));
    auto mx = x.slice(0, (int)group.x * 128);
    auto mw = w.slice(0, (int)group.y * 64);
    auto my = y.slice((int)group.y * 64, (int)group.x * 128);
    matmul2d<matmul2d_descriptor(128, 64, dynamic_extent,
                                false, true, false),
              execution_simdgroups<4>> mm;
    mm.run(mx, mw, my);
}

kernel void h3_linear_bf16_nax_r128_morton4(
                           device bfloat *input [[buffer(0)]],
                           device bfloat *weight [[buffer(1)]],
                           device bfloat *output [[buffer(3)]],
                           constant linear_args &args [[buffer(4)]],
                           uint code [[threadgroup_position_in_grid]]) {
    uint row_tiles = (args.rows + 127) / 128;
    uint2 group = h3_morton_decode_compact4(code, row_tiles);
    auto x = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim, (int)args.rows));
    auto w = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        weight,
        dextents<int32_t, 2>((int)args.input_dim, (int)args.output_dim));
    auto y = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        output,
        dextents<int32_t, 2>((int)args.output_dim, (int)args.rows));
    auto mx = x.slice(0, (int)group.x * 128);
    auto mw = w.slice(0, (int)group.y * 64);
    auto my = y.slice((int)group.y * 64, (int)group.x * 128);
    matmul2d<matmul2d_descriptor(128, 64, dynamic_extent,
                                false, true, false),
              execution_simdgroups<4>> mm;
    mm.run(mx, mw, my);
}

struct h3_qkv_project_rope_args {
    uint rows;
    uint input_dim;
    uint heads;
    uint head_dim;
    uint rope_half;
    uint head_major;
    float epsilon;
};

/* Preserve the successful direct 128x64 TensorOps matmul, but route each
 * grouped checkpoint tile directly to its Q, K, or V destination. The six
 * 64-column tiles of every head align exactly with Q0/Q1/K0/K1/V0/V1. */
kernel void h3_qkv_project_split_bf16_nax_r128_morton4(
                           device bfloat *input [[buffer(0)]],
                           device bfloat *weight [[buffer(1)]],
                           device bfloat *query [[buffer(2)]],
                           device bfloat *key [[buffer(3)]],
                           device bfloat *value [[buffer(4)]],
                           constant h3_qkv_project_rope_args &args
                               [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]]) {
    constexpr int ROW_TILE = 128;
    constexpr int COLUMN_TILE = 64;
    constexpr int HEAD_DIM = 128;
    constexpr int TILES_PER_HEAD = 6;
    uint row_tiles = (args.rows + ROW_TILE - 1) / ROW_TILE;
    uint2 group = h3_morton_decode_compact4(code, row_tiles);
    uint head = group.y / TILES_PER_HEAD;
    uint head_tile = group.y - head * TILES_PER_HEAD;
    uint stream = head_tile >> 1;
    uint tile_half = head_tile & 1;
    device bfloat *destination = stream == 0 ? query :
        stream == 1 ? key : value;
    if (args.head_major) destination += head * args.rows * HEAD_DIM;
    int output_width = args.head_major ? HEAD_DIM :
        (int)args.heads * HEAD_DIM;
    int output_column = (args.head_major ? 0 : (int)head * HEAD_DIM) +
        (int)tile_half * COLUMN_TILE;
    auto x = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim, (int)args.rows));
    auto w = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>(
            (int)args.input_dim, (int)args.heads * HEAD_DIM * 3));
    auto y = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        destination, dextents<int32_t, 2>(output_width, (int)args.rows));
    auto mx = x.slice(0, (int)group.x * ROW_TILE);
    auto mw = w.slice(0, (int)group.y * COLUMN_TILE);
    auto my = y.slice(output_column, (int)group.x * ROW_TILE);
    matmul2d<matmul2d_descriptor(ROW_TILE, COLUMN_TILE, dynamic_extent,
                                false, true, false),
              execution_simdgroups<4>> mm;
    mm.run(mx, mw, my);
}

/* Q and K already have the requested attention layout. Cache both 128-wide
 * heads, retain the oracle's lane-zero scalar RMS order, and overwrite them
 * with the rounded normalized/RoPE result. V needs no second pass. */
kernel void h3_qk_rope_bf16_nax_inplace(
                           device bfloat *query [[buffer(0)]],
                           device bfloat *key [[buffer(1)]],
                           device bfloat *q_weight [[buffer(2)]],
                           device bfloat *k_weight [[buffer(3)]],
                           device bfloat *rope_cos [[buffer(4)]],
                           device bfloat *rope_sin [[buffer(5)]],
                           constant h3_qkv_project_rope_args &args
                               [[buffer(6)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint HEAD_DIM = 128;
    constexpr uint HEADS_PER_GROUP = 4;
    uint head = group.x * HEADS_PER_GROUP + simdgroup;
    uint row = group.y;
    if (head >= args.heads || row >= args.rows) return;
    uint base = args.head_major ?
        (head * args.rows + row) * HEAD_DIM :
        (row * args.heads + head) * HEAD_DIM;
    threadgroup bfloat q_values[HEADS_PER_GROUP * HEAD_DIM];
    threadgroup bfloat k_values[HEADS_PER_GROUP * HEAD_DIM];
    uint cache_base = simdgroup * HEAD_DIM;
    for (uint dimension = lane; dimension < HEAD_DIM; dimension += 32) {
        q_values[cache_base + dimension] = query[base + dimension];
        k_values[cache_base + dimension] = key[base + dimension];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float q_inverse[HEADS_PER_GROUP];
    threadgroup float k_inverse[HEADS_PER_GROUP];
    if (lane == 0) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (uint dimension = 0; dimension < HEAD_DIM; dimension++) {
            float q_element = (float)q_values[cache_base + dimension];
            float k_element = (float)k_values[cache_base + dimension];
            q_sum = fma(q_element, q_element, q_sum);
            k_sum = fma(k_element, k_element, k_sum);
        }
        q_inverse[simdgroup] =
            rsqrt(q_sum / float(args.head_dim) + args.epsilon);
        k_inverse[simdgroup] =
            rsqrt(k_sum / float(args.head_dim) + args.epsilon);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dimension = lane; dimension < HEAD_DIM; dimension += 32) {
        float q0 = (float)q_values[cache_base + dimension] *
                   q_inverse[simdgroup] * (float)q_weight[dimension];
        float k0 = (float)k_values[cache_base + dimension] *
                   k_inverse[simdgroup] * (float)k_weight[dimension];
        if (dimension < args.rope_half) {
            uint pair = dimension + args.rope_half;
            float q1 = (float)q_values[cache_base + pair] *
                       q_inverse[simdgroup] * (float)q_weight[pair];
            float k1 = (float)k_values[cache_base + pair] *
                       k_inverse[simdgroup] * (float)k_weight[pair];
            float c = (float)rope_cos[row * args.rope_half + dimension];
            float s = (float)rope_sin[row * args.rope_half + dimension];
            q0 = q0 * c - q1 * s;
            k0 = k0 * c - k1 * s;
        } else if (dimension < args.rope_half * 2) {
            uint pair = dimension - args.rope_half;
            float q1 = (float)q_values[cache_base + pair] *
                       q_inverse[simdgroup] * (float)q_weight[pair];
            float k1 = (float)k_values[cache_base + pair] *
                       k_inverse[simdgroup] * (float)k_weight[pair];
            float c = (float)rope_cos[row * args.rope_half + pair];
            float s = (float)rope_sin[row * args.rope_half + pair];
            q0 = q0 * c + q1 * s;
            k0 = k0 * c + k1 * s;
        }
        query[base + dimension] = (bfloat)q0;
        key[base + dimension] = (bfloat)k0;
    }
}

/* Pair the two FC1 halves while the TensorOps accumulators are still local.
 * Only the post-SwiGLU [rows, hidden_dim] tensor reaches device memory. */
kernel void h3_fc1_swiglu_bf16_nax_r128(
                           device bfloat *input [[buffer(0)]],
                           device bfloat *weight [[buffer(1)]],
                           device bfloat *output [[buffer(2)]],
                           constant linear_args &args [[buffer(3)]],
                           threadgroup bfloat *tiles [[threadgroup(0)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int ROW_TILE = 128;
    constexpr int COLUMN_TILE = 64;
    auto x = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim, (int)args.rows));
    auto w = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        weight,
        dextents<int32_t, 2>((int)args.input_dim,
                             (int)args.output_dim * 2));
    auto mx = x.slice(0, (int)group.x * ROW_TILE);
    auto mw_gate = w.slice(0, (int)group.y * COLUMN_TILE);
    auto mw_up = w.slice(0, (int)args.output_dim +
                            (int)group.y * COLUMN_TILE);
    threadgroup bfloat *gate_tile = tiles;
    threadgroup bfloat *up_tile = tiles + ROW_TILE * COLUMN_TILE;
    auto tg_gate = tensor(gate_tile,
        dextents<int32_t, 2>(COLUMN_TILE, ROW_TILE));
    auto tg_up = tensor(up_tile,
        dextents<int32_t, 2>(COLUMN_TILE, ROW_TILE));
    matmul2d<matmul2d_descriptor(ROW_TILE, COLUMN_TILE, dynamic_extent,
                                false, true, false),
              execution_simdgroups<4>> mm;
    {
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(x), decltype(w), bfloat>();
        mm.run(mx, mw_gate, accum);
        accum.store(tg_gate);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(x), decltype(w), bfloat>();
        mm.run(mx, mw_up, accum);
        accum.store(tg_up);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row_base = group.x * ROW_TILE;
    uint column_base = group.y * COLUMN_TILE;
    for (uint index = tid; index < ROW_TILE * COLUMN_TILE; index += 128) {
        uint row = row_base + index / COLUMN_TILE;
        uint column = column_base + index % COLUMN_TILE;
        if (row < args.rows && column < args.output_dim) {
            float gate = (float)gate_tile[index];
            float up = (float)up_tile[index];
            float activated = gate / (1.0f + exp(-gate)) * up;
            output[row * args.output_dim + column] = (bfloat)activated;
        }
    }
}

kernel void h3_fc1_swiglu_bf16_nax_r128_morton(
                           device bfloat *input [[buffer(0)]],
                           device bfloat *weight [[buffer(1)]],
                           device bfloat *output [[buffer(2)]],
                           constant linear_args &args [[buffer(3)]],
                           threadgroup bfloat *tiles [[threadgroup(0)]],
                           uint code [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int ROW_TILE = 128;
    constexpr int COLUMN_TILE = 64;
    uint row_tiles = (args.rows + ROW_TILE - 1) / ROW_TILE;
    uint column_tiles = (args.output_dim + COLUMN_TILE - 1) / COLUMN_TILE;
    uint row_bits = row_tiles <= 1 ? 0 : 32 - clz(row_tiles - 1);
    uint column_bits = column_tiles <= 1 ? 0 : 32 - clz(column_tiles - 1);
    uint2 group = h3_morton_decode_rectangular(
        code, row_bits, column_bits);
    if (group.x >= row_tiles || group.y >= column_tiles) return;
    auto x = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim, (int)args.rows));
    auto w = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        weight,
        dextents<int32_t, 2>((int)args.input_dim,
                             (int)args.output_dim * 2));
    auto mx = x.slice(0, (int)group.x * ROW_TILE);
    auto mw_gate = w.slice(0, (int)group.y * COLUMN_TILE);
    auto mw_up = w.slice(0, (int)args.output_dim +
                            (int)group.y * COLUMN_TILE);
    threadgroup bfloat *gate_tile = tiles;
    threadgroup bfloat *up_tile = tiles + ROW_TILE * COLUMN_TILE;
    auto tg_gate = tensor(gate_tile,
        dextents<int32_t, 2>(COLUMN_TILE, ROW_TILE));
    auto tg_up = tensor(up_tile,
        dextents<int32_t, 2>(COLUMN_TILE, ROW_TILE));
    matmul2d<matmul2d_descriptor(ROW_TILE, COLUMN_TILE, dynamic_extent,
                                false, true, false),
              execution_simdgroups<4>> mm;
    {
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(x), decltype(w), bfloat>();
        mm.run(mx, mw_gate, accum);
        accum.store(tg_gate);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(x), decltype(w), bfloat>();
        mm.run(mx, mw_up, accum);
        accum.store(tg_up);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row_base = group.x * ROW_TILE;
    uint column_base = group.y * COLUMN_TILE;
    for (uint index = tid; index < ROW_TILE * COLUMN_TILE; index += 128) {
        uint row = row_base + index / COLUMN_TILE;
        uint column = column_base + index % COLUMN_TILE;
        if (row < args.rows && column < args.output_dim) {
            float gate = (float)gate_tile[index];
            float up = (float)up_tile[index];
            float activated = gate / (1.0f + exp(-gate)) * up;
            output[row * args.output_dim + column] = (bfloat)activated;
        }
    }
}

kernel void h3_fc1_swiglu_bf16_nax_r128_morton4(
                           device bfloat *input [[buffer(0)]],
                           device bfloat *weight [[buffer(1)]],
                           device bfloat *output [[buffer(2)]],
                           constant linear_args &args [[buffer(3)]],
                           threadgroup bfloat *tiles [[threadgroup(0)]],
                           uint code [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int ROW_TILE = 128;
    constexpr int COLUMN_TILE = 64;
    uint row_tiles = (args.rows + ROW_TILE - 1) / ROW_TILE;
    uint2 group = h3_morton_decode_compact4(code, row_tiles);
    auto x = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim, (int)args.rows));
    auto w = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        weight,
        dextents<int32_t, 2>((int)args.input_dim,
                             (int)args.output_dim * 2));
    auto mx = x.slice(0, (int)group.x * ROW_TILE);
    auto mw_gate = w.slice(0, (int)group.y * COLUMN_TILE);
    auto mw_up = w.slice(0, (int)args.output_dim +
                            (int)group.y * COLUMN_TILE);
    threadgroup bfloat *gate_tile = tiles;
    threadgroup bfloat *up_tile = tiles + ROW_TILE * COLUMN_TILE;
    auto tg_gate = tensor(gate_tile,
        dextents<int32_t, 2>(COLUMN_TILE, ROW_TILE));
    auto tg_up = tensor(up_tile,
        dextents<int32_t, 2>(COLUMN_TILE, ROW_TILE));
    matmul2d<matmul2d_descriptor(ROW_TILE, COLUMN_TILE, dynamic_extent,
                                false, true, false),
              execution_simdgroups<4>> mm;
    {
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(x), decltype(w), bfloat>();
        mm.run(mx, mw_gate, accum);
        accum.store(tg_gate);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(x), decltype(w), bfloat>();
        mm.run(mx, mw_up, accum);
        accum.store(tg_up);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row_base = group.x * ROW_TILE;
    uint column_base = group.y * COLUMN_TILE;
    for (uint index = tid; index < ROW_TILE * COLUMN_TILE; index += 128) {
        uint row = row_base + index / COLUMN_TILE;
        uint column = column_base + index % COLUMN_TILE;
        if (row < args.rows && column < args.output_dim) {
            float gate = (float)gate_tile[index];
            float up = (float)up_tile[index];
            float activated = gate / (1.0f + exp(-gate)) * up;
            output[row * args.output_dim + column] = (bfloat)activated;
        }
    }
}

/* Draw Things/ccv-style dynamic symmetric row quantization. Weight rows are
 * checkpoint output channels; activation rows include zero-padded M tails so
 * the int8 matmuls can keep a fully static 128x128x128 descriptor. */
kernel void h3_quantize_bf16_int8_rows(
                           device const bfloat *input [[buffer(0)]],
                           device int8_t *output [[buffer(1)]],
                           device float *scales [[buffer(2)]],
                           constant int8_quant_args &args [[buffer(3)]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort simdgroup
                               [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           uint row [[threadgroup_position_in_grid]]) {
    uint base = row * args.columns;
    if (row >= args.rows) {
        if ((args.columns & 3u) == 0) {
            device char4 *output4 =
                reinterpret_cast<device char4 *>(output);
            uint vector_base = row * (args.columns / 4);
            for (uint column = tid; column < args.columns / 4;
                 column += 256)
                output4[vector_base + column] = char4(0);
        } else {
            for (uint column = tid; column < args.columns; column += 256)
                output[base + column] = 0;
        }
        if (tid == 0) scales[row] = 1.0f;
        return;
    }
    threadgroup float scratch[8];
    float local_max = 0.0f;
    if ((args.columns & 3u) == 0) {
        device const bfloat4 *input4 =
            reinterpret_cast<device const bfloat4 *>(input);
        uint vectors_per_row = args.columns / 4;
        uint vector_base = row * vectors_per_row;
        for (uint column = tid; column < vectors_per_row; column += 256) {
            float4 value = float4(input4[vector_base + column]);
            local_max = max(local_max,
                max(max(fabs(value.x), fabs(value.y)),
                    max(fabs(value.z), fabs(value.w))));
        }
    } else {
        for (uint column = tid; column < args.columns; column += 256)
            local_max = max(local_max, fabs((float)input[base + column]));
    }
    float max_abs = h3_int8_reduce_max(
        local_max, scratch, simdgroup, lane);
    float clipped_max = max_abs * args.clip;
    float scale = clipped_max > 0.0f ? clipped_max / 127.0f : 1.0f / 127.0f;
    float inverse = clipped_max > 0.0f ? 127.0f / clipped_max : 127.0f;
    if (tid == 0) scales[row] = scale;
    if ((args.columns & 3u) == 0) {
        device const bfloat4 *input4 =
            reinterpret_cast<device const bfloat4 *>(input);
        device char4 *output4 =
            reinterpret_cast<device char4 *>(output);
        uint vectors_per_row = args.columns / 4;
        uint vector_base = row * vectors_per_row;
        for (uint column = tid; column < vectors_per_row; column += 256) {
            int4 quantized = int4(rint(
                float4(input4[vector_base + column]) * inverse));
            output4[vector_base + column] =
                char4(clamp(quantized, int4(-127), int4(127)));
        }
    } else {
        for (uint column = tid; column < args.columns; column += 256) {
            int quantized = (int)rint((float)input[base + column] * inverse);
            output[base + column] =
                (int8_t)clamp(quantized, -127, 127);
        }
    }
}

/* Retained only for crossed kernel measurements against the vec4 path. */
kernel void h3_quantize_bf16_int8_rows_scalar(
                           device const bfloat *input [[buffer(0)]],
                           device int8_t *output [[buffer(1)]],
                           device float *scales [[buffer(2)]],
                           constant int8_quant_args &args [[buffer(3)]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort simdgroup
                               [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           uint row [[threadgroup_position_in_grid]]) {
    uint base = row * args.columns;
    if (row >= args.rows) {
        for (uint column = tid; column < args.columns; column += 256)
            output[base + column] = 0;
        if (tid == 0) scales[row] = 1.0f;
        return;
    }
    threadgroup float scratch[8];
    float local_max = 0.0f;
    for (uint column = tid; column < args.columns; column += 256)
        local_max = max(local_max, fabs((float)input[base + column]));
    float max_abs = h3_int8_reduce_max(
        local_max, scratch, simdgroup, lane);
    float clipped_max = max_abs * args.clip;
    float scale = clipped_max > 0.0f ? clipped_max / 127.0f : 1.0f / 127.0f;
    float inverse = clipped_max > 0.0f ? 127.0f / clipped_max : 127.0f;
    if (tid == 0) scales[row] = scale;
    for (uint column = tid; column < args.columns; column += 256) {
        int quantized = (int)rint((float)input[base + column] * inverse);
        output[base + column] = (int8_t)clamp(quantized, -127, 127);
    }
}

/* Convert SDPA's head-major source directly to the row-major int8 layout used
 * by the projection. The gather replaces the removed BF16 transpose while
 * preserving fully coalesced int8 stores and the proven projection kernel. */
kernel void h3_quantize_bf16_int8_head_major_to_rows_cached(
                           device const bfloat4 *input [[buffer(0)]],
                           device char4 *output [[buffer(1)]],
                           device float *scales [[buffer(2)]],
                           constant int8_head_major_quant_args &args
                               [[buffer(3)]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort simdgroup
                               [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           uint row [[threadgroup_position_in_grid]]) {
    constexpr uint THREADS = 256;
    constexpr uint VECTORS = 7;
    constexpr uint VECTORS_PER_HEAD = 32;
    constexpr uint VECTORS_PER_ROW = 56 * VECTORS_PER_HEAD;
    if (row >= args.rows) {
        for (uint slot = 0; slot < VECTORS; slot++) {
            uint linear = tid + slot * THREADS;
            output[row * VECTORS_PER_ROW + linear] = char4(0);
        }
        if (tid == 0) scales[row] = 1.0f;
        return;
    }
    thread bfloat4 values[VECTORS];
    threadgroup float scratch[8];
    float local_max = 0.0f;
    #pragma clang loop unroll(full)
    for (uint slot = 0; slot < VECTORS; slot++) {
        uint linear = tid + slot * THREADS;
        uint head = linear >> 5;
        uint vector = linear & (VECTORS_PER_HEAD - 1);
        bfloat4 value = input[
            (head * args.rows + row) * VECTORS_PER_HEAD + vector];
        values[slot] = value;
        float4 f = float4(value);
        local_max = max(local_max,
            max(max(fabs(f.x), fabs(f.y)), max(fabs(f.z), fabs(f.w))));
    }
    float max_abs = h3_int8_reduce_max(
        local_max, scratch, simdgroup, lane);
    float clipped_max = max_abs * args.clip;
    float scale = clipped_max > 0.0f ? clipped_max / 127.0f : 1.0f / 127.0f;
    float inverse = clipped_max > 0.0f ? 127.0f / clipped_max : 127.0f;
    if (tid == 0) scales[row] = scale;
    #pragma clang loop unroll(full)
    for (uint slot = 0; slot < VECTORS; slot++) {
        uint linear = tid + slot * THREADS;
        int4 quantized = int4(rint(float4(values[slot]) * inverse));
        output[row * VECTORS_PER_ROW + linear] =
            char4(clamp(quantized, int4(-127), int4(127)));
    }
}

kernel void h3_quantize_bf16_int8_groups(
                           device const bfloat *input [[buffer(0)]],
                           device int8_t *output [[buffer(1)]],
                           device float *scales [[buffer(2)]],
                           constant int8_group_quant_args &args [[buffer(3)]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort simdgroup
                               [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           uint row [[threadgroup_position_in_grid]]) {
    uint vectors_per_row = args.columns / 4;
    uint vectors_per_group = args.group_size / 4;
    uint vector_base = row * vectors_per_row;
    device const bfloat4 *input4 =
        reinterpret_cast<device const bfloat4 *>(input);
    device char4 *output4 = reinterpret_cast<device char4 *>(output);
    if (row >= args.rows) {
        for (uint column = tid; column < vectors_per_row; column += 256)
            output4[vector_base + column] = char4(0);
        for (uint group = tid; group < args.groups; group += 256)
            scales[row * args.groups + group] = 1.0f;
        return;
    }
    threadgroup float scratch[8];
    for (uint group = 0; group < args.groups; group++) {
        uint start = vector_base + group * vectors_per_group;
        float local_max = 0.0f;
        for (uint local = tid; local < vectors_per_group; local += 256) {
            float4 value = float4(input4[start + local]);
            local_max = max(local_max,
                max(max(fabs(value.x), fabs(value.y)),
                    max(fabs(value.z), fabs(value.w))));
        }
        float max_abs = h3_int8_reduce_max(
            local_max, scratch, simdgroup, lane);
        float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
        float inverse = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
        if (tid == 0) scales[row * args.groups + group] = scale;
        for (uint local = tid; local < vectors_per_group; local += 256) {
            int4 quantized = int4(rint(float4(input4[start + local]) *
                                       inverse));
            output4[start + local] =
                char4(clamp(quantized, int4(-127), int4(127)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/* Retained only for crossed kernel measurements against the vec4 path. */
kernel void h3_quantize_bf16_int8_groups_scalar(
                           device const bfloat *input [[buffer(0)]],
                           device int8_t *output [[buffer(1)]],
                           device float *scales [[buffer(2)]],
                           constant int8_group_quant_args &args [[buffer(3)]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort simdgroup
                               [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           uint row [[threadgroup_position_in_grid]]) {
    uint row_base = row * args.columns;
    if (row >= args.rows) {
        for (uint column = tid; column < args.columns; column += 256)
            output[row_base + column] = 0;
        for (uint group = tid; group < args.groups; group += 256)
            scales[row * args.groups + group] = 1.0f;
        return;
    }
    threadgroup float scratch[8];
    for (uint group = 0; group < args.groups; group++) {
        uint start = group * args.group_size;
        float local_max = 0.0f;
        for (uint local = tid; local < args.group_size; local += 256)
            local_max = max(local_max,
                fabs((float)input[row_base + start + local]));
        float max_abs = h3_int8_reduce_max(
            local_max, scratch, simdgroup, lane);
        float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
        float inverse = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
        if (tid == 0) scales[row * args.groups + group] = scale;
        for (uint local = tid; local < args.group_size; local += 256) {
            int quantized = (int)rint(
                (float)input[row_base + start + local] * inverse);
            output[row_base + start + local] =
                (int8_t)clamp(quantized, -127, 127);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline float h3_int8_reduce_max4(float value, threadgroup float *scratch,
                                 ushort simdgroup, ushort lane) {
    value = max(value, simd_shuffle_xor(value, 16));
    value = max(value, simd_shuffle_xor(value, 8));
    value = max(value, simd_shuffle_xor(value, 4));
    value = max(value, simd_shuffle_xor(value, 2));
    value = max(value, simd_shuffle_xor(value, 1));
    if (lane == 0) scratch[simdgroup] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        value = lane < 4 ? scratch[lane] : 0.0f;
        value = max(value, simd_shuffle_xor(value, 16));
        value = max(value, simd_shuffle_xor(value, 8));
        value = max(value, simd_shuffle_xor(value, 4));
        value = max(value, simd_shuffle_xor(value, 2));
        value = max(value, simd_shuffle_xor(value, 1));
        if (lane == 0) scratch[0] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

/* Exact 128-thread form of the selected scalar grouped quantizer. */
kernel void h3_quantize_bf16_int8_groups_scalar128(
                           device const bfloat *input [[buffer(0)]],
                           device int8_t *output [[buffer(1)]],
                           device float *scales [[buffer(2)]],
                           constant int8_group_quant_args &args [[buffer(3)]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort simdgroup
                               [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           uint row [[threadgroup_position_in_grid]]) {
    constexpr uint THREADS = 128;
    uint row_base = row * args.columns;
    if (row >= args.rows) {
        for (uint column = tid; column < args.columns; column += THREADS)
            output[row_base + column] = 0;
        for (uint group = tid; group < args.groups; group += THREADS)
            scales[row * args.groups + group] = 1.0f;
        return;
    }
    threadgroup float scratch[4];
    for (uint group = 0; group < args.groups; group++) {
        uint start = group * args.group_size;
        float local_max = 0.0f;
        for (uint local = tid; local < args.group_size; local += THREADS)
            local_max = max(local_max,
                fabs((float)input[row_base + start + local]));
        float max_abs = h3_int8_reduce_max4(
            local_max, scratch, simdgroup, lane);
        float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
        float inverse = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
        if (tid == 0) scales[row * args.groups + group] = scale;
        for (uint local = tid; local < args.group_size; local += THREADS) {
            int quantized = (int)rint(
                (float)input[row_base + start + local] * inverse);
            output[row_base + start + local] =
                (int8_t)clamp(quantized, -127, 127);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/* The selected 1,024-wide grouped quantizer assigns exactly eight values to
 * each of 128 threads. Keep them private across max reduction and emission so
 * the quantization pass does not reread the BF16 activation row. */
kernel void h3_quantize_bf16_int8_groups_scalar128_cached(
                           device const bfloat *input [[buffer(0)]],
                           device int8_t *output [[buffer(1)]],
                           device float *scales [[buffer(2)]],
                           constant int8_group_quant_args &args [[buffer(3)]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort simdgroup
                               [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           uint row [[threadgroup_position_in_grid]]) {
    constexpr uint THREADS = 128;
    constexpr uint VALUES = 8;
    uint row_base = row * args.columns;
    if (row >= args.rows) {
        for (uint column = tid; column < args.columns; column += THREADS)
            output[row_base + column] = 0;
        for (uint group = tid; group < args.groups; group += THREADS)
            scales[row * args.groups + group] = 1.0f;
        return;
    }
    threadgroup float scratch[4];
    for (uint group = 0; group < args.groups; group++) {
        uint start = group * args.group_size;
        thread bfloat values[VALUES];
        float local_max = 0.0f;
        #pragma clang loop unroll(full)
        for (uint slot = 0; slot < VALUES; slot++) {
            uint local = tid + slot * THREADS;
            bfloat value = input[row_base + start + local];
            values[slot] = value;
            local_max = max(local_max, fabs((float)value));
        }
        float max_abs = h3_int8_reduce_max4(
            local_max, scratch, simdgroup, lane);
        float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
        float inverse = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
        if (tid == 0) scales[row * args.groups + group] = scale;
        #pragma clang loop unroll(full)
        for (uint slot = 0; slot < VALUES; slot++) {
            uint local = tid + slot * THREADS;
            int quantized = (int)rint((float)values[slot] * inverse);
            output[row_base + start + local] =
                (int8_t)clamp(quantized, -127, 127);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/* Int8 H3 QKV projection. Checkpoint columns are grouped as
 * [head, q/k/v, dimension], so each 128-column TensorOps tile writes one
 * complete stream/head directly in head-major SDPA layout. */
kernel void h3_qkv_project_split_int8_nax_r128_morton4(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *query [[buffer(4)]],
                           device bfloat *key [[buffer(5)]],
                           device bfloat *value [[buffer(6)]],
                           constant h3_qkv_project_rope_args &args
                               [[buffer(7)]],
                           uint code [[threadgroup_position_in_grid]]) {
    constexpr uint TILE = 128;
    constexpr uint STREAMS = 3;
    uint padded_rows = (args.rows + TILE - 1) & ~(TILE - 1);
    uint row_tiles = padded_rows / TILE;
    uint column_tiles = args.heads * STREAMS;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, column_tiles);
    uint row_start = group.x * TILE;
    uint checkpoint_column = group.y * TILE;
    uint head = group.y / STREAMS;
    uint stream = group.y - head * STREAMS;
    device bfloat *destination = stream == 0 ? query :
        stream == 1 ? key : value;
    destination += head * args.rows * TILE;
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>(
            (int)args.input_dim, (int)args.heads * (int)TILE * 3));
    constexpr auto descriptor = matmul2d_descriptor(
        TILE, TILE, TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    auto first_a = x.slice<TILE, TILE>(0, (int)row_start);
    auto first_b = w.slice<TILE, TILE>(0, (int)checkpoint_column);
    auto accum = mm.template get_destination_cooperative_tensor<
        decltype(first_a), decltype(first_b), int32_t>();
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++)
        if (accum.is_valid_element(element)) accum[element] = 0;
    for (uint k = 0; k < args.input_dim; k += TILE) {
        auto a = x.slice<TILE, TILE>((int)k, (int)row_start);
        auto b = w.slice<TILE, TILE>((int)k, (int)checkpoint_column);
        mm.run(a, b, accum);
    }
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++) {
        if (!accum.is_valid_element(element)) continue;
        auto index = accum.get_multidimensional_index(element);
        uint row = row_start + (uint)index[1];
        uint column = (uint)index[0];
        if (row < args.rows)
            destination[row * TILE + column] =
                (bfloat)((float)accum[element] * input_scales[row] *
                         weight_scales[checkpoint_column + column]);
    }
}

/* Keep each projected Q/K 128x128 tile resident in its owning threadgroup.
 * After the TensorOps epilogue has rounded to BF16, one thread computes each
 * row's exact scalar RMS order; all threads then normalize and apply RoPE in
 * disjoint dimension pairs. The temporary device tile is the final head-major
 * output itself, so this adds only 512 bytes of threadgroup storage and removes
 * the separate per-row/per-four-head Q/K kernel dispatch. */
template<uint K_TILE>
kernel void h3_qkv_project_split_int8_rope_nax_r128_morton4_impl(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *query [[buffer(4)]],
                           device bfloat *key [[buffer(5)]],
                           device bfloat *value [[buffer(6)]],
                           device const bfloat *q_weight [[buffer(7)]],
                           device const bfloat *k_weight [[buffer(8)]],
                           device const bfloat *rope_cos [[buffer(9)]],
                           device const bfloat *rope_sin [[buffer(10)]],
                           constant h3_qkv_project_rope_args &args
                               [[buffer(11)]],
                           uint code [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint TILE = 128;
    constexpr uint STREAMS = 3;
    constexpr uint ROPE_PAIRS = 48;
    constexpr uint NORMALIZED_UNITS = ROPE_PAIRS +
        (TILE - ROPE_PAIRS * 2);
    uint padded_rows = (args.rows + TILE - 1) & ~(TILE - 1);
    uint row_tiles = padded_rows / TILE;
    uint column_tiles = args.heads * STREAMS;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, column_tiles);
    uint row_start = group.x * TILE;
    uint checkpoint_column = group.y * TILE;
    uint head = group.y / STREAMS;
    uint stream = group.y - head * STREAMS;
    device bfloat *destination = stream == 0 ? query :
        stream == 1 ? key : value;
    destination += head * args.rows * TILE;
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>(
            (int)args.input_dim, (int)args.heads * (int)TILE * 3));
    constexpr auto descriptor = matmul2d_descriptor(
        TILE, TILE, K_TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    auto first_a = x.slice<TILE, K_TILE>(0, (int)row_start);
    auto first_b = w.slice<K_TILE, TILE>(0, (int)checkpoint_column);
    auto accum = mm.template get_destination_cooperative_tensor<
        decltype(first_a), decltype(first_b), int32_t>();
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++)
        if (accum.is_valid_element(element)) accum[element] = 0;
    for (uint k = 0; k < args.input_dim; k += K_TILE) {
        auto a = x.slice<TILE, K_TILE>((int)k, (int)row_start);
        auto b = w.slice<K_TILE, TILE>((int)k, (int)checkpoint_column);
        mm.run(a, b, accum);
    }
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++) {
        if (!accum.is_valid_element(element)) continue;
        auto index = accum.get_multidimensional_index(element);
        uint row = row_start + (uint)index[1];
        uint column = (uint)index[0];
        if (row < args.rows)
            destination[row * TILE + column] =
                (bfloat)((float)accum[element] * input_scales[row] *
                         weight_scales[checkpoint_column + column]);
    }
    if (stream < 2) {
        threadgroup float inverse[TILE];
        threadgroup_barrier(mem_flags::mem_device);
        uint row = row_start + tid;
        if (tid < TILE && row < args.rows) {
            float sum = 0.0f;
            if (args.head_major & 2u) {
                device const bfloat4 *destination4 =
                    reinterpret_cast<device const bfloat4 *>(destination);
                uint vector_base = row * (TILE / 4);
                for (uint vector = 0; vector < TILE / 4; vector++) {
                    float4 elements = float4(destination4[vector_base + vector]);
                    sum = fma(elements.x, elements.x, sum);
                    sum = fma(elements.y, elements.y, sum);
                    sum = fma(elements.z, elements.z, sum);
                    sum = fma(elements.w, elements.w, sum);
                }
            } else {
                for (uint dimension = 0; dimension < TILE; dimension++) {
                    float element =
                        (float)destination[row * TILE + dimension];
                    sum = fma(element, element, sum);
                }
            }
            inverse[tid] = rsqrt(sum / float(TILE) + args.epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const bfloat *norm_weight = stream == 0 ? q_weight : k_weight;
        if (args.head_major & 4u) {
            constexpr uint PACKED_UNITS = NORMALIZED_UNITS / 4;
            device ushort4 *destination4 =
                reinterpret_cast<device ushort4 *>(destination);
            device const ushort4 *norm_weight4 =
                reinterpret_cast<device const ushort4 *>(norm_weight);
            device const ushort4 *rope_cos4 =
                reinterpret_cast<device const ushort4 *>(rope_cos);
            device const ushort4 *rope_sin4 =
                reinterpret_cast<device const ushort4 *>(rope_sin);
            for (uint unit = tid; unit < TILE * PACKED_UNITS; unit += 256) {
                uint local_row = unit / PACKED_UNITS;
                uint part = unit - local_row * PACKED_UNITS;
                uint global_row = row_start + local_row;
                if (global_row >= args.rows) continue;
                uint base4 = global_row * (TILE / 4);
                float inv = inverse[local_row];
                if (part < ROPE_PAIRS / 4) {
                    uint upper = part + ROPE_PAIRS / 4;
                    ushort4 lower_bits = destination4[base4 + part];
                    ushort4 upper_bits = destination4[base4 + upper];
                    ushort4 lower_norm = norm_weight4[part];
                    ushort4 upper_norm = norm_weight4[upper];
                    float4 lower_value = h3_bf16x4_to_f32(lower_bits) * inv *
                        h3_bf16x4_to_f32(lower_norm);
                    float4 upper_value = h3_bf16x4_to_f32(upper_bits) * inv *
                        h3_bf16x4_to_f32(upper_norm);
                    uint rope_base = global_row * (args.rope_half / 4);
                    ushort4 cosine_bits = rope_cos4[rope_base + part];
                    ushort4 sine_bits = rope_sin4[rope_base + part];
                    float4 cosine = h3_bf16x4_to_f32(cosine_bits);
                    float4 sine = h3_bf16x4_to_f32(sine_bits);
                    float4 lower_output = lower_value * cosine -
                        upper_value * sine;
                    float4 upper_output = upper_value * cosine +
                        lower_value * sine;
                    destination4[base4 + part] =
                        h3_f32x4_to_bf16(lower_output);
                    destination4[base4 + upper] =
                        h3_f32x4_to_bf16(upper_output);
                } else {
                    uint dimension = part + ROPE_PAIRS / 4;
                    ushort4 value_bits = destination4[base4 + dimension];
                    ushort4 norm_bits = norm_weight4[dimension];
                    destination4[base4 + dimension] = h3_f32x4_to_bf16(
                        h3_bf16x4_to_f32(value_bits) * inv *
                        h3_bf16x4_to_f32(norm_bits));
                }
            }
        } else for (uint unit = tid; unit < TILE * NORMALIZED_UNITS;
                    unit += 256) {
            uint local_row = unit / NORMALIZED_UNITS;
            uint part = unit - local_row * NORMALIZED_UNITS;
            uint global_row = row_start + local_row;
            if (global_row >= args.rows) continue;
            uint base = global_row * TILE;
            float inv = inverse[local_row];
            if (part < ROPE_PAIRS) {
                uint upper = part + ROPE_PAIRS;
                float lower_value = (float)destination[base + part] * inv *
                                    (float)norm_weight[part];
                float upper_value = (float)destination[base + upper] * inv *
                                    (float)norm_weight[upper];
                float c = (float)rope_cos[
                    global_row * args.rope_half + part];
                float s = (float)rope_sin[
                    global_row * args.rope_half + part];
                destination[base + part] =
                    (bfloat)(lower_value * c - upper_value * s);
                destination[base + upper] =
                    (bfloat)(upper_value * c + lower_value * s);
            } else {
                uint dimension = part + ROPE_PAIRS;
                destination[base + dimension] = (bfloat)(
                    (float)destination[base + dimension] * inv *
                    (float)norm_weight[dimension]);
            }
        }
    }
}

typedef decltype(h3_qkv_project_split_int8_rope_nax_r128_morton4_impl<128>)
    h3_qkv_project_split_int8_rope_nax_r128_morton4_t;
template [[host_name("h3_qkv_project_split_int8_rope_nax_r128_morton4")]]
kernel h3_qkv_project_split_int8_rope_nax_r128_morton4_t
    h3_qkv_project_split_int8_rope_nax_r128_morton4_impl<128>;
template [[host_name("h3_qkv_project_split_int8_rope_nax_r128_k5376_morton4")]]
kernel h3_qkv_project_split_int8_rope_nax_r128_morton4_t
    h3_qkv_project_split_int8_rope_nax_r128_morton4_impl<5376>;

/* QKV variant that caches the row and checkpoint-column scale vectors. After
 * every cooperative fragment has consumed the scales, the first half is
 * recycled for inverse RMS and the second half for Q/K normalization weights. */
template<uint K_TILE>
kernel void h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4_impl(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *query [[buffer(4)]],
                           device bfloat *key [[buffer(5)]],
                           device bfloat *value [[buffer(6)]],
                           device const bfloat *q_weight [[buffer(7)]],
                           device const bfloat *k_weight [[buffer(8)]],
                           device const bfloat *rope_cos [[buffer(9)]],
                           device const bfloat *rope_sin [[buffer(10)]],
                           constant h3_qkv_project_rope_args &args
                               [[buffer(11)]],
                           uint code [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint TILE = 128;
    constexpr uint STREAMS = 3;
    constexpr uint ROPE_PAIRS = 48;
    constexpr uint NORMALIZED_UNITS = ROPE_PAIRS +
        (TILE - ROPE_PAIRS * 2);
    uint padded_rows = (args.rows + TILE - 1) & ~(TILE - 1);
    uint row_tiles = padded_rows / TILE;
    uint column_tiles = args.heads * STREAMS;
    uint2 group = h3_morton_decode_compact(code, row_tiles, column_tiles);
    uint row_start = group.x * TILE;
    uint checkpoint_column = group.y * TILE;
    uint head = group.y / STREAMS;
    uint stream = group.y - head * STREAMS;
    device bfloat *destination = stream == 0 ? query :
        stream == 1 ? key : value;
    destination += head * args.rows * TILE;
    threadgroup float scratch[TILE * 2];
    if (tid < TILE) {
        scratch[tid] = input_scales[row_start + tid];
        scratch[TILE + tid] = weight_scales[checkpoint_column + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>(
            (int)args.input_dim, (int)args.heads * (int)TILE * 3));
    constexpr auto descriptor = matmul2d_descriptor(
        TILE, TILE, K_TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    auto first_a = x.slice<TILE, K_TILE>(0, (int)row_start);
    auto first_b = w.slice<K_TILE, TILE>(0, (int)checkpoint_column);
    auto accum = mm.template get_destination_cooperative_tensor<
        decltype(first_a), decltype(first_b), int32_t>();
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++)
        if (accum.is_valid_element(element)) accum[element] = 0;
    for (uint k = 0; k < args.input_dim; k += K_TILE) {
        auto a = x.slice<TILE, K_TILE>((int)k, (int)row_start);
        auto b = w.slice<K_TILE, TILE>((int)k, (int)checkpoint_column);
        mm.run(a, b, accum);
    }
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++) {
        if (!accum.is_valid_element(element)) continue;
        auto index = accum.get_multidimensional_index(element);
        uint row = row_start + (uint)index[1];
        uint column = (uint)index[0];
        if (row < args.rows)
            destination[row * TILE + column] =
                (bfloat)((float)accum[element] * scratch[(uint)index[1]] *
                         scratch[TILE + column]);
    }
    if (stream < 2) {
        device const bfloat *norm_weight = stream == 0 ? q_weight : k_weight;
        threadgroup_barrier(mem_flags::mem_device |
                            mem_flags::mem_threadgroup);
        uint row = row_start + tid;
        if (tid < TILE) {
            scratch[TILE + tid] = (float)norm_weight[tid];
            if (row < args.rows) {
                float sum = 0.0f;
                if (args.head_major & 2u) {
                    device const bfloat4 *destination4 =
                        reinterpret_cast<device const bfloat4 *>(destination);
                    uint vector_base = row * (TILE / 4);
                    for (uint vector = 0; vector < TILE / 4; vector++) {
                        float4 elements =
                            float4(destination4[vector_base + vector]);
                        sum = fma(elements.x, elements.x, sum);
                        sum = fma(elements.y, elements.y, sum);
                        sum = fma(elements.z, elements.z, sum);
                        sum = fma(elements.w, elements.w, sum);
                    }
                } else {
                    for (uint dimension = 0; dimension < TILE; dimension++) {
                        float element =
                            (float)destination[row * TILE + dimension];
                        sum = fma(element, element, sum);
                    }
                }
                scratch[tid] = rsqrt(sum / float(TILE) + args.epsilon);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (args.head_major & 4u) {
            constexpr uint PACKED_UNITS = NORMALIZED_UNITS / 4;
            device ushort4 *destination4 =
                reinterpret_cast<device ushort4 *>(destination);
            device const ushort4 *rope_cos4 =
                reinterpret_cast<device const ushort4 *>(rope_cos);
            device const ushort4 *rope_sin4 =
                reinterpret_cast<device const ushort4 *>(rope_sin);
            threadgroup float4 *norm_weight4 =
                reinterpret_cast<threadgroup float4 *>(scratch + TILE);
            for (uint unit = tid; unit < TILE * PACKED_UNITS; unit += 256) {
                uint local_row = unit / PACKED_UNITS;
                uint part = unit - local_row * PACKED_UNITS;
                uint global_row = row_start + local_row;
                if (global_row >= args.rows) continue;
                uint base4 = global_row * (TILE / 4);
                float inv = scratch[local_row];
                if (part < ROPE_PAIRS / 4) {
                    uint upper = part + ROPE_PAIRS / 4;
                    ushort4 lower_bits = destination4[base4 + part];
                    ushort4 upper_bits = destination4[base4 + upper];
                    float4 lower_value = h3_bf16x4_to_f32(lower_bits) * inv *
                        norm_weight4[part];
                    float4 upper_value = h3_bf16x4_to_f32(upper_bits) * inv *
                        norm_weight4[upper];
                    uint rope_base = global_row * (args.rope_half / 4);
                    ushort4 cosine_bits = rope_cos4[rope_base + part];
                    ushort4 sine_bits = rope_sin4[rope_base + part];
                    float4 cosine = h3_bf16x4_to_f32(cosine_bits);
                    float4 sine = h3_bf16x4_to_f32(sine_bits);
                    float4 lower_output = lower_value * cosine -
                        upper_value * sine;
                    float4 upper_output = upper_value * cosine +
                        lower_value * sine;
                    destination4[base4 + part] =
                        h3_f32x4_to_bf16(lower_output);
                    destination4[base4 + upper] =
                        h3_f32x4_to_bf16(upper_output);
                } else {
                    uint dimension = part + ROPE_PAIRS / 4;
                    ushort4 value_bits = destination4[base4 + dimension];
                    destination4[base4 + dimension] = h3_f32x4_to_bf16(
                        h3_bf16x4_to_f32(value_bits) * inv *
                        norm_weight4[dimension]);
                }
            }
        } else for (uint unit = tid; unit < TILE * NORMALIZED_UNITS;
                    unit += 256) {
            uint local_row = unit / NORMALIZED_UNITS;
            uint part = unit - local_row * NORMALIZED_UNITS;
            uint global_row = row_start + local_row;
            if (global_row >= args.rows) continue;
            uint base = global_row * TILE;
            float inv = scratch[local_row];
            if (part < ROPE_PAIRS) {
                uint upper = part + ROPE_PAIRS;
                float lower_value = (float)destination[base + part] * inv *
                                    scratch[TILE + part];
                float upper_value = (float)destination[base + upper] * inv *
                                    scratch[TILE + upper];
                float c = (float)rope_cos[
                    global_row * args.rope_half + part];
                float s = (float)rope_sin[
                    global_row * args.rope_half + part];
                destination[base + part] =
                    (bfloat)(lower_value * c - upper_value * s);
                destination[base + upper] =
                    (bfloat)(upper_value * c + lower_value * s);
            } else {
                uint dimension = part + ROPE_PAIRS;
                destination[base + dimension] = (bfloat)(
                    (float)destination[base + dimension] * inv *
                    scratch[TILE + dimension]);
            }
        }
    }
}

typedef decltype(
    h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4_impl<128>)
    h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4_t;
template [[host_name(
    "h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4")]]
kernel h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4_t
    h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4_impl<128>;
template [[host_name(
    "h3_qkv_project_split_int8_rope_local_scales_nax_r128_k5376_morton4")]]
kernel h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4_t
    h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4_impl<5376>;

/* INPUT_DIM=5376 lets Metal schedule H3's fixed FC1 K loop without unrolling
 * its 42 TensorOps slices; zero retains the generic runtime-bound fallback. */
template<uint INPUT_DIM, uint K_TILE>
kernel void h3_fc1_swiglu_int8_nax_r128_impl(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]]) {
    constexpr uint TILE = 128;
    uint padded_rows = (args.rows + TILE - 1) & ~(TILE - 1);
    uint row_tiles = padded_rows / TILE;
    uint column_tiles = args.output_dim / TILE;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, column_tiles);
    uint row_start = group.x * TILE;
    uint column_start = group.y * TILE;
    uint input_dim = INPUT_DIM ? INPUT_DIM : args.input_dim;
    constexpr bool FULL_PRODUCT = INPUT_DIM != 0 && K_TILE == INPUT_DIM;
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)input_dim,
                                     (int)args.output_dim * 2));
    constexpr auto descriptor = matmul2d_descriptor(
        TILE, TILE, K_TILE, false, true, true,
        FULL_PRODUCT ? matmul2d_descriptor::mode::multiply :
                       matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    threadgroup bfloat gate_tile[TILE * TILE];
    {
        auto first_a = x.slice<TILE, K_TILE>(0, (int)row_start);
        auto first_b = w.slice<K_TILE, TILE>(0, (int)column_start);
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(first_a), decltype(first_b), int32_t>();
        if constexpr (!FULL_PRODUCT) {
            #pragma clang loop unroll(full)
            for (ushort element = 0; element < accum.get_capacity(); element++)
                if (accum.is_valid_element(element)) accum[element] = 0;
        }
        if (INPUT_DIM) {
            for (uint k = 0; k < INPUT_DIM; k += K_TILE) {
                auto a = x.slice<TILE, K_TILE>((int)k, (int)row_start);
                auto b = w.slice<K_TILE, TILE>((int)k, (int)column_start);
                mm.run(a, b, accum);
            }
        } else {
            for (uint k = 0; k < input_dim; k += K_TILE) {
                auto a = x.slice<TILE, K_TILE>((int)k, (int)row_start);
                auto b = w.slice<K_TILE, TILE>((int)k, (int)column_start);
                mm.run(a, b, accum);
            }
        }
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++) {
            if (!accum.is_valid_element(element)) continue;
            auto index = accum.get_multidimensional_index(element);
            uint row = row_start + (uint)index[1];
            uint column = column_start + (uint)index[0];
            float value = (float)accum[element] * input_scales[row] *
                          weight_scales[column];
            gate_tile[(uint)index[1] * TILE + (uint)index[0]] =
                (bfloat)value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        auto first_a = x.slice<TILE, K_TILE>(0, (int)row_start);
        auto first_b = w.slice<K_TILE, TILE>(
            0, (int)args.output_dim + (int)column_start);
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(first_a), decltype(first_b), int32_t>();
        if constexpr (!FULL_PRODUCT) {
            #pragma clang loop unroll(full)
            for (ushort element = 0; element < accum.get_capacity(); element++)
                if (accum.is_valid_element(element)) accum[element] = 0;
        }
        if (INPUT_DIM) {
            for (uint k = 0; k < INPUT_DIM; k += K_TILE) {
                auto a = x.slice<TILE, K_TILE>((int)k, (int)row_start);
                auto b = w.slice<K_TILE, TILE>(
                    (int)k, (int)args.output_dim + (int)column_start);
                mm.run(a, b, accum);
            }
        } else {
            for (uint k = 0; k < input_dim; k += K_TILE) {
                auto a = x.slice<TILE, K_TILE>((int)k, (int)row_start);
                auto b = w.slice<K_TILE, TILE>(
                    (int)k, (int)args.output_dim + (int)column_start);
                mm.run(a, b, accum);
            }
        }
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++) {
            if (!accum.is_valid_element(element)) continue;
            auto index = accum.get_multidimensional_index(element);
            uint local = (uint)index[1] * TILE + (uint)index[0];
            uint row = row_start + (uint)index[1];
            uint column = column_start + (uint)index[0];
            if (row >= args.rows) continue;
            float gate = (float)gate_tile[local];
            float up = (float)accum[element] * input_scales[row] *
                       weight_scales[args.output_dim + column];
            output[row * args.output_dim + column] =
                (bfloat)(gate / (1.0f + exp(-gate)) * up);
        }
    }
}

typedef decltype(h3_fc1_swiglu_int8_nax_r128_impl<0, 128>)
    h3_fc1_swiglu_int8_nax_r128_t;
template [[host_name("h3_fc1_swiglu_int8_nax_r128")]]
kernel h3_fc1_swiglu_int8_nax_r128_t
    h3_fc1_swiglu_int8_nax_r128_impl<0, 128>;
template [[host_name("h3_fc1_swiglu_int8_nax_r128_k5376")]]
kernel h3_fc1_swiglu_int8_nax_r128_t
    h3_fc1_swiglu_int8_nax_r128_impl<5376, 128>;
template [[host_name("h3_fc1_swiglu_int8_nax_r128_full_k5376")]]
kernel h3_fc1_swiglu_int8_nax_r128_t
    h3_fc1_swiglu_int8_nax_r128_impl<5376, 5376>;

/* Gate and up use the same cooperative-fragment mapping. Preserve the gate's
 * existing BF16 rounding point in thread-private storage, then consume it
 * after the up projection without a 32 KiB threadgroup tile or barrier. */
kernel void h3_fc1_swiglu_int8_local_nax_r128(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]]) {
    constexpr uint TILE = 128;
    constexpr uint FRAGMENT_CAPACITY = 64;
    uint padded_rows = (args.rows + TILE - 1) & ~(TILE - 1);
    uint row_tiles = padded_rows / TILE;
    uint column_tiles = args.output_dim / TILE;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, column_tiles);
    uint row_start = group.x * TILE;
    uint column_start = group.y * TILE;
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)args.input_dim,
                                     (int)args.output_dim * 2));
    constexpr auto descriptor = matmul2d_descriptor(
        TILE, TILE, TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    thread bfloat gate_values[FRAGMENT_CAPACITY];
    {
        auto first_a = x.slice<TILE, TILE>(0, (int)row_start);
        auto first_b = w.slice<TILE, TILE>(0, (int)column_start);
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(first_a), decltype(first_b), int32_t>();
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++)
            if (accum.is_valid_element(element)) accum[element] = 0;
        for (uint k = 0; k < args.input_dim; k += TILE) {
            auto a = x.slice<TILE, TILE>((int)k, (int)row_start);
            auto b = w.slice<TILE, TILE>((int)k, (int)column_start);
            mm.run(a, b, accum);
        }
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++) {
            if (!accum.is_valid_element(element)) continue;
            auto index = accum.get_multidimensional_index(element);
            uint row = row_start + (uint)index[1];
            uint column = column_start + (uint)index[0];
            float value = (float)accum[element] * input_scales[row] *
                          weight_scales[column];
            gate_values[element] = (bfloat)value;
        }
    }
    {
        auto first_a = x.slice<TILE, TILE>(0, (int)row_start);
        auto first_b = w.slice<TILE, TILE>(
            0, (int)args.output_dim + (int)column_start);
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(first_a), decltype(first_b), int32_t>();
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++)
            if (accum.is_valid_element(element)) accum[element] = 0;
        for (uint k = 0; k < args.input_dim; k += TILE) {
            auto a = x.slice<TILE, TILE>((int)k, (int)row_start);
            auto b = w.slice<TILE, TILE>(
                (int)k, (int)args.output_dim + (int)column_start);
            mm.run(a, b, accum);
        }
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++) {
            if (!accum.is_valid_element(element)) continue;
            auto index = accum.get_multidimensional_index(element);
            uint row = row_start + (uint)index[1];
            uint column = column_start + (uint)index[0];
            if (row >= args.rows) continue;
            float gate = (float)gate_values[element];
            float up = (float)accum[element] * input_scales[row] *
                       weight_scales[args.output_dim + column];
            output[row * args.output_dim + column] =
                (bfloat)(gate / (1.0f + exp(-gate)) * up);
        }
    }
}

kernel void h3_linear_int8_nax_r128(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]]) {
    constexpr uint TILE = 128;
    uint padded_rows = (args.rows + TILE - 1) & ~(TILE - 1);
    uint row_tiles = padded_rows / TILE;
    uint column_tiles = args.output_dim / TILE;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, column_tiles);
    uint row_start = group.x * TILE;
    uint column_start = group.y * TILE;
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)args.input_dim,
                                     (int)args.output_dim));
    constexpr auto descriptor = matmul2d_descriptor(
        TILE, TILE, TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    auto first_a = x.slice<TILE, TILE>(0, (int)row_start);
    auto first_b = w.slice<TILE, TILE>(0, (int)column_start);
    auto accum = mm.template get_destination_cooperative_tensor<
        decltype(first_a), decltype(first_b), int32_t>();
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++)
        if (accum.is_valid_element(element)) accum[element] = 0;
    for (uint k = 0; k < args.input_dim; k += TILE) {
        auto a = x.slice<TILE, TILE>((int)k, (int)row_start);
        auto b = w.slice<TILE, TILE>((int)k, (int)column_start);
        mm.run(a, b, accum);
    }
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++) {
        if (!accum.is_valid_element(element)) continue;
        auto index = accum.get_multidimensional_index(element);
        uint row = row_start + (uint)index[1];
        uint column = column_start + (uint)index[0];
        if (row < args.rows)
            output[row * args.output_dim + column] =
                (bfloat)((float)accum[element] * input_scales[row] *
                         weight_scales[column]);
    }
}

/* One-scale FC2 path. A static full-K product lets NAX own the
 * complete 14336-wide reduction; scale loads overlap that long operation. */
kernel void h3_linear_int8_nax_r128_full_k14336(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint TILE = 128;
    constexpr uint INPUT_DIM = 14336;
    constexpr uint OUTPUT_DIM = 5376;
    uint padded_rows = (args.rows + TILE - 1) & ~(TILE - 1);
    uint row_tiles = padded_rows / TILE;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, OUTPUT_DIM / TILE);
    uint row_start = group.x * TILE;
    uint column_start = group.y * TILE;
    threadgroup float local_input_scales[TILE];
    threadgroup float local_weight_scales[TILE];
    if (tid < TILE) {
        local_input_scales[tid] = input_scales[row_start + tid];
        local_weight_scales[tid] = weight_scales[column_start + tid];
    }
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)INPUT_DIM,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)INPUT_DIM,
                                     (int)OUTPUT_DIM));
    constexpr auto descriptor = matmul2d_descriptor(
        TILE, TILE, INPUT_DIM, false, true, true,
        matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    auto a = x.slice<TILE, INPUT_DIM>(0, (int)row_start);
    auto b = w.slice<INPUT_DIM, TILE>(0, (int)column_start);
    auto accum = mm.template get_destination_cooperative_tensor<
        decltype(a), decltype(b), int32_t>();
    mm.run(a, b, accum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++) {
        if (!accum.is_valid_element(element)) continue;
        auto index = accum.get_multidimensional_index(element);
        uint row = row_start + (uint)index[1];
        uint column = column_start + (uint)index[0];
        if (row < args.rows)
            output[row * OUTPUT_DIM + column] =
                (bfloat)((float)accum[element] *
                    local_input_scales[(uint)index[1]] *
                    local_weight_scales[(uint)index[0]]);
    }
}

/* The same full-K product for other static shapes (the SwiftVR FC2). */
template<uint INPUT_DIM, uint OUTPUT_DIM>
kernel void h3_linear_int8_nax_r128_full_k_impl(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint TILE = 128;
    uint padded_rows = (args.rows + TILE - 1) & ~(TILE - 1);
    uint row_tiles = padded_rows / TILE;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, OUTPUT_DIM / TILE);
    uint row_start = group.x * TILE;
    uint column_start = group.y * TILE;
    threadgroup float local_input_scales[TILE];
    threadgroup float local_weight_scales[TILE];
    if (tid < TILE) {
        local_input_scales[tid] = input_scales[row_start + tid];
        local_weight_scales[tid] = weight_scales[column_start + tid];
    }
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)INPUT_DIM,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)INPUT_DIM,
                                     (int)OUTPUT_DIM));
    constexpr auto descriptor = matmul2d_descriptor(
        TILE, TILE, INPUT_DIM, false, true, true,
        matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    auto a = x.slice<TILE, INPUT_DIM>(0, (int)row_start);
    auto b = w.slice<INPUT_DIM, TILE>(0, (int)column_start);
    auto accum = mm.template get_destination_cooperative_tensor<
        decltype(a), decltype(b), int32_t>();
    mm.run(a, b, accum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++) {
        if (!accum.is_valid_element(element)) continue;
        auto index = accum.get_multidimensional_index(element);
        uint row = row_start + (uint)index[1];
        uint column = column_start + (uint)index[0];
        if (row < args.rows)
            output[row * OUTPUT_DIM + column] =
                (bfloat)((float)accum[element] *
                    local_input_scales[(uint)index[1]] *
                    local_weight_scales[(uint)index[0]]);
    }
}

typedef decltype(h3_linear_int8_nax_r128_full_k_impl<14336, 3072>) h3_linear_int8_nax_r128_full_k_t;
template [[host_name("h3_linear_int8_nax_r128_full_k14336_n3072")]]
kernel h3_linear_int8_nax_r128_full_k_t h3_linear_int8_nax_r128_full_k_impl<14336, 3072>;

kernel void h3_linear_int8_nax_r128x256_full_k14336(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint ROW_TILE = 128;
    constexpr uint COLUMN_TILE = 256;
    constexpr uint INPUT_DIM = 14336;
    constexpr uint OUTPUT_DIM = 5376;
    uint padded_rows = (args.rows + ROW_TILE - 1) & ~(ROW_TILE - 1);
    uint row_tiles = padded_rows / ROW_TILE;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, OUTPUT_DIM / COLUMN_TILE);
    uint row_start = group.x * ROW_TILE;
    uint column_start = group.y * COLUMN_TILE;
    threadgroup float local_input_scales[ROW_TILE];
    threadgroup float local_weight_scales[COLUMN_TILE];
    if (tid < COLUMN_TILE) {
        if (tid < ROW_TILE)
            local_input_scales[tid] = input_scales[row_start + tid];
        local_weight_scales[tid] = weight_scales[column_start + tid];
    }
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)INPUT_DIM,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)INPUT_DIM,
                                     (int)OUTPUT_DIM));
    constexpr auto descriptor = matmul2d_descriptor(
        ROW_TILE, COLUMN_TILE, INPUT_DIM, false, true, true,
        matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor, execution_simdgroups<16>> mm;
    auto a = x.slice<ROW_TILE, INPUT_DIM>(0, (int)row_start);
    auto b = w.slice<INPUT_DIM, COLUMN_TILE>(0, (int)column_start);
    auto accum = mm.template get_destination_cooperative_tensor<
        decltype(a), decltype(b), int32_t>();
    mm.run(a, b, accum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++) {
        if (!accum.is_valid_element(element)) continue;
        auto index = accum.get_multidimensional_index(element);
        uint row = row_start + (uint)index[1];
        uint column = column_start + (uint)index[0];
        if (row < args.rows)
            output[row * OUTPUT_DIM + column] =
                (bfloat)((float)accum[element] *
                    local_input_scales[(uint)index[1]] *
                    local_weight_scales[(uint)index[0]]);
    }
}

/* The 128x256 full-K tile for the SwiftVR FC1/FC2 shapes: half the A
 * re-reads of the 128x128 tile, which matters once K is 14336. */
template<uint INPUT_DIM, uint OUTPUT_DIM>
kernel void h3_linear_int8_nax_r128x256_full_k_impl(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint ROW_TILE = 128;
    constexpr uint COLUMN_TILE = 256;
    uint padded_rows = (args.rows + ROW_TILE - 1) & ~(ROW_TILE - 1);
    uint row_tiles = padded_rows / ROW_TILE;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, OUTPUT_DIM / COLUMN_TILE);
    uint row_start = group.x * ROW_TILE;
    uint column_start = group.y * COLUMN_TILE;
    threadgroup float local_input_scales[ROW_TILE];
    threadgroup float local_weight_scales[COLUMN_TILE];
    if (tid < COLUMN_TILE) {
        if (tid < ROW_TILE)
            local_input_scales[tid] = input_scales[row_start + tid];
        local_weight_scales[tid] = weight_scales[column_start + tid];
    }
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)INPUT_DIM,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)INPUT_DIM,
                                     (int)OUTPUT_DIM));
    constexpr auto descriptor = matmul2d_descriptor(
        ROW_TILE, COLUMN_TILE, INPUT_DIM, false, true, true,
        matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor, execution_simdgroups<16>> mm;
    auto a = x.slice<ROW_TILE, INPUT_DIM>(0, (int)row_start);
    auto b = w.slice<INPUT_DIM, COLUMN_TILE>(0, (int)column_start);
    auto accum = mm.template get_destination_cooperative_tensor<
        decltype(a), decltype(b), int32_t>();
    mm.run(a, b, accum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++) {
        if (!accum.is_valid_element(element)) continue;
        auto index = accum.get_multidimensional_index(element);
        uint row = row_start + (uint)index[1];
        uint column = column_start + (uint)index[0];
        if (row < args.rows)
            output[row * OUTPUT_DIM + column] =
                (bfloat)((float)accum[element] *
                    local_input_scales[(uint)index[1]] *
                    local_weight_scales[(uint)index[0]]);
    }
}

typedef decltype(h3_linear_int8_nax_r128x256_full_k_impl<14336, 3072>) h3_linear_int8_nax_r128x256_full_k_t;
template [[host_name("h3_linear_int8_nax_r128x256_full_k14336_n3072")]]
kernel h3_linear_int8_nax_r128x256_full_k_t h3_linear_int8_nax_r128x256_full_k_impl<14336, 3072>;
template [[host_name("h3_linear_int8_nax_r128x256_full_k3072_n14336")]]
kernel h3_linear_int8_nax_r128x256_full_k_t h3_linear_int8_nax_r128x256_full_k_impl<3072, 14336>;
template [[host_name("h3_linear_int8_nax_r128x256_full_k3072_n9216")]]
kernel h3_linear_int8_nax_r128x256_full_k_t h3_linear_int8_nax_r128x256_full_k_impl<3072, 9216>;
template [[host_name("h3_linear_int8_nax_r128x256_full_k3072_n3072")]]
kernel h3_linear_int8_nax_r128x256_full_k_t h3_linear_int8_nax_r128x256_full_k_impl<3072, 3072>;

/* Cache the two 128-value dequantization vectors once per projection tile.
 * The cooperative output fragment otherwise rereads them for every element. */
template<uint INPUT_DIM, uint OUTPUT_DIM>
kernel void h3_linear_int8_local_scales_nax_r128_impl(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint TILE = 128;
    uint padded_rows = (args.rows + TILE - 1) & ~(TILE - 1);
    uint row_tiles = padded_rows / TILE;
    uint output_dim = OUTPUT_DIM ? OUTPUT_DIM : args.output_dim;
    uint column_tiles = output_dim / TILE;
    uint2 group = h3_morton_decode_compact(code, row_tiles, column_tiles);
    uint row_start = group.x * TILE;
    uint column_start = group.y * TILE;
    uint input_dim = INPUT_DIM ? INPUT_DIM : args.input_dim;
    threadgroup float local_input_scales[TILE];
    threadgroup float local_weight_scales[TILE];
    if (tid < TILE) {
        local_input_scales[tid] = input_scales[row_start + tid];
        local_weight_scales[tid] = weight_scales[column_start + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)input_dim,
                                     (int)output_dim));
    constexpr auto descriptor = matmul2d_descriptor(
        TILE, TILE, TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto first_descriptor = matmul2d_descriptor(
        TILE, TILE, TILE, false, true, true,
        matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    matmul2d<first_descriptor, execution_simdgroups<8>> first_mm;
    auto first_a = x.slice<TILE, TILE>(0, (int)row_start);
    auto first_b = w.slice<TILE, TILE>(0, (int)column_start);
    auto accum = mm.template get_destination_cooperative_tensor<
        decltype(first_a), decltype(first_b), int32_t>();
    first_mm.run(first_a, first_b, accum);
    if (INPUT_DIM) {
        for (uint k = TILE; k < INPUT_DIM; k += TILE) {
            auto a = x.slice<TILE, TILE>((int)k, (int)row_start);
            auto b = w.slice<TILE, TILE>((int)k, (int)column_start);
            mm.run(a, b, accum);
        }
    } else {
        for (uint k = TILE; k < input_dim; k += TILE) {
            auto a = x.slice<TILE, TILE>((int)k, (int)row_start);
            auto b = w.slice<TILE, TILE>((int)k, (int)column_start);
            mm.run(a, b, accum);
        }
    }
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++) {
        if (!accum.is_valid_element(element)) continue;
        auto index = accum.get_multidimensional_index(element);
        uint row = row_start + (uint)index[1];
        uint column = column_start + (uint)index[0];
        if (row < args.rows)
            output[row * output_dim + column] =
                (bfloat)((float)accum[element] *
                         local_input_scales[(uint)index[1]] *
                         local_weight_scales[(uint)index[0]]);
    }
}

/* Split-K int8 linear: threadgroup (x, y, z) reduces K columns
 * [z * K / splits, (z + 1) * K / splits) of the 128x128 output tile into
 * F32 partials [splits][padded_rows][N]; h3_linear_sum_partials_bf16 adds
 * them. A long K (the SwiftVR FC2's 14336) runs at half the throughput of
 * a short one in a single pass; two or four slices restore it. */
struct split_k_args { uint rows, input_dim, output_dim, splits; };

kernel void h3_linear_int8_split_k_nax_r128(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device float *partial [[buffer(4)]],
                           constant split_k_args &args [[buffer(5)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint TILE = 128;
    uint padded_rows = (args.rows + TILE - 1) & ~(TILE - 1);
    /* Morton order over the output tiles, like the other int8 linears:
     * neighbouring threadgroups share A row tiles and B column tiles. */
    uint2 tile = h3_morton_decode_compact(group.x, padded_rows / TILE, args.output_dim / TILE);
    uint row_start = tile.x * TILE;
    uint column_start = tile.y * TILE;
    uint k_count = args.input_dim / args.splits;
    uint k_begin = group.y * k_count;
    threadgroup float local_input_scales[TILE];
    threadgroup float local_weight_scales[TILE];
    if (tid < TILE) {
        local_input_scales[tid] = input_scales[row_start + tid];
        local_weight_scales[tid] = weight_scales[column_start + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim, (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)args.input_dim, (int)args.output_dim));
    constexpr auto descriptor = matmul2d_descriptor(
        TILE, TILE, TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto first_descriptor = matmul2d_descriptor(
        TILE, TILE, TILE, false, true, true,
        matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    matmul2d<first_descriptor, execution_simdgroups<8>> first_mm;
    auto first_a = x.slice<TILE, TILE>((int)k_begin, (int)row_start);
    auto first_b = w.slice<TILE, TILE>((int)k_begin, (int)column_start);
    auto accum = mm.template get_destination_cooperative_tensor<
        decltype(first_a), decltype(first_b), int32_t>();
    first_mm.run(first_a, first_b, accum);
    for (uint k = k_begin + TILE; k < k_begin + k_count; k += TILE) {
        auto a = x.slice<TILE, TILE>((int)k, (int)row_start);
        auto b = w.slice<TILE, TILE>((int)k, (int)column_start);
        mm.run(a, b, accum);
    }
    device float *out = partial + (size_t)group.y * padded_rows * args.output_dim;
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++) {
        if (!accum.is_valid_element(element)) continue;
        auto index = accum.get_multidimensional_index(element);
        uint row = row_start + (uint)index[1];
        uint column = column_start + (uint)index[0];
        if (row < args.rows)
            out[(size_t)row * args.output_dim + column] =
                (float)accum[element] * local_input_scales[(uint)index[1]] *
                local_weight_scales[(uint)index[0]];
    }
}

/* out[r, c] = sum over splits of partial[s][r][c], as BF16. Grid (N, rows). */
kernel void h3_linear_sum_partials_bf16(device const float *partial [[buffer(0)]],
                                        device bfloat *output [[buffer(1)]],
                                        constant split_k_args &args [[buffer(2)]],
                                        uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x, row = gid.y;
    if (column >= args.output_dim || row >= args.rows) return;
    uint padded_rows = (args.rows + 127) & ~127u;
    size_t stride = (size_t)padded_rows * args.output_dim;
    size_t at = (size_t)row * args.output_dim + column;
    float sum = 0.0f;
    for (uint s = 0; s < args.splits; s++) sum += partial[s * stride + at];
    output[at] = (bfloat)sum;
}

typedef decltype(h3_linear_int8_local_scales_nax_r128_impl<0, 0>)
    h3_linear_int8_local_scales_nax_r128_t;
template [[host_name("h3_linear_int8_local_scales_nax_r128")]]
kernel h3_linear_int8_local_scales_nax_r128_t
    h3_linear_int8_local_scales_nax_r128_impl<0, 0>;
template [[host_name("h3_linear_int8_local_scales_nax_r128_k7168")]]
kernel h3_linear_int8_local_scales_nax_r128_t
    h3_linear_int8_local_scales_nax_r128_impl<7168, 5376>;
/* The SwiftVR DiT's shapes (dim 3072, FFN 14336): static K loops. */
template [[host_name("h3_linear_int8_local_scales_nax_r128_k3072_n3072")]]
kernel h3_linear_int8_local_scales_nax_r128_t
    h3_linear_int8_local_scales_nax_r128_impl<3072, 3072>;
template [[host_name("h3_linear_int8_local_scales_nax_r128_k3072_n9216")]]
kernel h3_linear_int8_local_scales_nax_r128_t
    h3_linear_int8_local_scales_nax_r128_impl<3072, 9216>;
template [[host_name("h3_linear_int8_local_scales_nax_r128_k3072_n14336")]]
kernel h3_linear_int8_local_scales_nax_r128_t
    h3_linear_int8_local_scales_nax_r128_impl<3072, 14336>;
template [[host_name("h3_linear_int8_local_scales_nax_r128_k14336_n3072")]]
kernel h3_linear_int8_local_scales_nax_r128_t
    h3_linear_int8_local_scales_nax_r128_impl<14336, 3072>;

/* FC2 is more sensitive to a single scale spanning all 14336 activated
 * channels.  Retain one activation scale per 1024-wide K group, accumulate
 * each group's exact int32 product separately, then apply its scale before
 * adding it to the FP32 tile.  The 128x64 tile keeps that FP32 tile within
 * the M5 threadgroup-memory limit. */
kernel void h3_linear_int8_grouped_nax_r128x64(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint ROW_TILE = 128;
    constexpr uint COLUMN_TILE = 64;
    constexpr uint K_TILE = 128;
    constexpr uint SCALE_GROUP = 1024;
    constexpr uint K_TILES_PER_GROUP = SCALE_GROUP / K_TILE;
    uint padded_rows = (args.rows + ROW_TILE - 1) & ~(ROW_TILE - 1);
    uint row_tiles = padded_rows / ROW_TILE;
    uint column_tiles = args.output_dim / COLUMN_TILE;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, column_tiles);
    uint row_start = group.x * ROW_TILE;
    uint column_start = group.y * COLUMN_TILE;
    uint scale_groups = args.input_dim / SCALE_GROUP;
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)args.input_dim,
                                     (int)args.output_dim));
    constexpr auto descriptor = matmul2d_descriptor(
        ROW_TILE, COLUMN_TILE, K_TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> mm;
    threadgroup float totals[ROW_TILE * COLUMN_TILE];
    for (uint scale_group = 0; scale_group < scale_groups; scale_group++) {
        uint k_start = scale_group * SCALE_GROUP;
        auto first_a = x.slice<ROW_TILE, K_TILE>(
            (int)k_start, (int)row_start);
        auto first_b = w.slice<K_TILE, COLUMN_TILE>(
            (int)k_start, (int)column_start);
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(first_a), decltype(first_b), int32_t>();
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++)
            if (accum.is_valid_element(element)) accum[element] = 0;
        #pragma clang loop unroll(full)
        for (uint k_tile = 0; k_tile < K_TILES_PER_GROUP; k_tile++) {
            uint k = k_start + k_tile * K_TILE;
            auto a = x.slice<ROW_TILE, K_TILE>((int)k, (int)row_start);
            auto b = w.slice<K_TILE, COLUMN_TILE>(
                (int)k, (int)column_start);
            mm.run(a, b, accum);
        }
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++) {
            if (!accum.is_valid_element(element)) continue;
            auto index = accum.get_multidimensional_index(element);
            uint local = (uint)index[1] * COLUMN_TILE +
                         (uint)index[0];
            uint row = row_start + (uint)index[1];
            uint column = column_start + (uint)index[0];
            float value = (float)accum[element] *
                input_scales[row * scale_groups + scale_group] *
                weight_scales[column];
            if (scale_group == 0) totals[local] = value;
            else totals[local] += value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint local = tid; local < ROW_TILE * COLUMN_TILE; local += 128) {
        uint row = row_start + local / COLUMN_TILE;
        uint column = column_start + local % COLUMN_TILE;
        if (row < args.rows)
            output[row * args.output_dim + column] = (bfloat)totals[local];
    }
}

/* Same scaled K-group arithmetic as the threadgroup-tile version, but the
 * cooperative tensor assigns the same fragment element to the same thread
 * for every K slice. Keep those 64 FP32 totals private and issue two
 * 512-wide scale-group products before draining either fragment. */
kernel void h3_linear_int8_grouped_local_nax_r128x64(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint ROW_TILE = 128;
    constexpr uint COLUMN_TILE = 64;
    constexpr uint K_TILE = 512;
    constexpr uint SCALE_GROUP = 1024;
    constexpr uint K_TILES_PER_GROUP = SCALE_GROUP / K_TILE;
    constexpr uint SCALE_GROUPS = 14;
    constexpr uint FRAGMENT_CAPACITY = 64;
    uint padded_rows = (args.rows + ROW_TILE - 1) & ~(ROW_TILE - 1);
    uint row_tiles = padded_rows / ROW_TILE;
    uint column_tiles = args.output_dim / COLUMN_TILE;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, column_tiles);
    uint row_start = group.x * ROW_TILE;
    uint column_start = group.y * COLUMN_TILE;
    uint scale_groups = args.input_dim / SCALE_GROUP;
    threadgroup float local_input_scales[ROW_TILE * SCALE_GROUPS];
    threadgroup float local_weight_scales[COLUMN_TILE];
    for (uint local = tid; local < ROW_TILE * SCALE_GROUPS; local += 256) {
        uint row = row_start + local / SCALE_GROUPS;
        uint scale_group = local % SCALE_GROUPS;
        local_input_scales[local] =
            input_scales[row * SCALE_GROUPS + scale_group];
    }
    if (tid < COLUMN_TILE)
        local_weight_scales[tid] = weight_scales[column_start + tid];
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)args.input_dim,
                                     (int)args.output_dim));
    constexpr auto descriptor = matmul2d_descriptor(
        ROW_TILE, COLUMN_TILE, K_TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto first_descriptor = matmul2d_descriptor(
        ROW_TILE, COLUMN_TILE, K_TILE, false, true, true,
        matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    matmul2d<first_descriptor, execution_simdgroups<8>> first_mm;
    thread float totals[FRAGMENT_CAPACITY];
    for (uint scale_group = 0; scale_group < scale_groups;
         scale_group += 2) {
        uint first_k = scale_group * SCALE_GROUP;
        uint second_k = first_k + SCALE_GROUP;
        auto first_a = x.slice<ROW_TILE, K_TILE>(
            (int)first_k, (int)row_start);
        auto first_b = w.slice<K_TILE, COLUMN_TILE>(
            (int)first_k, (int)column_start);
        auto first_accum = mm.template get_destination_cooperative_tensor<
            decltype(first_a), decltype(first_b), int32_t>();
        first_mm.run(first_a, first_b, first_accum);
        #pragma clang loop unroll(full)
        for (uint k_tile = 1; k_tile < K_TILES_PER_GROUP; k_tile++) {
            uint k = first_k + k_tile * K_TILE;
            auto a = x.slice<ROW_TILE, K_TILE>((int)k, (int)row_start);
            auto b = w.slice<K_TILE, COLUMN_TILE>(
                (int)k, (int)column_start);
            mm.run(a, b, first_accum);
        }
        auto second_a = x.slice<ROW_TILE, K_TILE>(
            (int)second_k, (int)row_start);
        auto second_b = w.slice<K_TILE, COLUMN_TILE>(
            (int)second_k, (int)column_start);
        auto second_accum = mm.template get_destination_cooperative_tensor<
            decltype(second_a), decltype(second_b), int32_t>();
        first_mm.run(second_a, second_b, second_accum);
        #pragma clang loop unroll(full)
        for (uint k_tile = 1; k_tile < K_TILES_PER_GROUP; k_tile++) {
            uint k = second_k + k_tile * K_TILE;
            auto a = x.slice<ROW_TILE, K_TILE>((int)k, (int)row_start);
            auto b = w.slice<K_TILE, COLUMN_TILE>(
                (int)k, (int)column_start);
            mm.run(a, b, second_accum);
        }
        if (scale_group == 0)
            threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < first_accum.get_capacity();
             element++) {
            if (!first_accum.is_valid_element(element)) continue;
            auto index = first_accum.get_multidimensional_index(element);
            float value = (float)first_accum[element] *
                local_input_scales[(uint)index[1] * SCALE_GROUPS +
                                   scale_group] *
                local_weight_scales[(uint)index[0]];
            if (scale_group == 0) totals[element] = value;
            else totals[element] += value;
        }
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < second_accum.get_capacity();
             element++) {
            if (!second_accum.is_valid_element(element)) continue;
            auto index = second_accum.get_multidimensional_index(element);
            uint row = row_start + (uint)index[1];
            uint column = column_start + (uint)index[0];
            float value = (float)second_accum[element] *
                local_input_scales[(uint)index[1] * SCALE_GROUPS +
                                   scale_group + 1] *
                local_weight_scales[(uint)index[0]];
            totals[element] += value;
            if (scale_group + 2 == scale_groups && row < args.rows)
                output[row * args.output_dim + column] =
                    (bfloat)totals[element];
        }
    }
}

/* Wider-output form: the doubled tile uses eight SIMD groups, so each thread
 * still owns 64 output elements while the grid launches half as many FC2
 * workgroups. */
kernel void h3_linear_int8_grouped_local_nax_r128x128(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]]) {
    constexpr uint ROW_TILE = 128;
    constexpr uint COLUMN_TILE = 128;
    constexpr uint K_TILE = 128;
    constexpr uint SCALE_GROUP = 1024;
    constexpr uint K_TILES_PER_GROUP = SCALE_GROUP / K_TILE;
    constexpr uint FRAGMENT_CAPACITY = 64;
    uint padded_rows = (args.rows + ROW_TILE - 1) & ~(ROW_TILE - 1);
    uint row_tiles = padded_rows / ROW_TILE;
    uint column_tiles = args.output_dim / COLUMN_TILE;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, column_tiles);
    uint row_start = group.x * ROW_TILE;
    uint column_start = group.y * COLUMN_TILE;
    uint scale_groups = args.input_dim / SCALE_GROUP;
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)args.input_dim,
                                     (int)args.output_dim));
    constexpr auto descriptor = matmul2d_descriptor(
        ROW_TILE, COLUMN_TILE, K_TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    thread float totals[FRAGMENT_CAPACITY];
    for (uint scale_group = 0; scale_group < scale_groups; scale_group++) {
        uint k_start = scale_group * SCALE_GROUP;
        auto first_a = x.slice<ROW_TILE, K_TILE>(
            (int)k_start, (int)row_start);
        auto first_b = w.slice<K_TILE, COLUMN_TILE>(
            (int)k_start, (int)column_start);
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(first_a), decltype(first_b), int32_t>();
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++)
            if (accum.is_valid_element(element)) accum[element] = 0;
        #pragma clang loop unroll(full)
        for (uint k_tile = 0; k_tile < K_TILES_PER_GROUP; k_tile++) {
            uint k = k_start + k_tile * K_TILE;
            auto a = x.slice<ROW_TILE, K_TILE>((int)k, (int)row_start);
            auto b = w.slice<K_TILE, COLUMN_TILE>(
                (int)k, (int)column_start);
            mm.run(a, b, accum);
        }
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++) {
            if (!accum.is_valid_element(element)) continue;
            auto index = accum.get_multidimensional_index(element);
            uint row = row_start + (uint)index[1];
            uint column = column_start + (uint)index[0];
            float value = (float)accum[element] *
                input_scales[row * scale_groups + scale_group] *
                weight_scales[column];
            if (scale_group == 0) totals[element] = value;
            else totals[element] += value;
            if (scale_group + 1 == scale_groups && row < args.rows)
                output[row * args.output_dim + column] =
                    (bfloat)totals[element];
        }
    }
}
#endif

kernel void h3_silu_bf16(device const ushort *input [[buffer(0)]],
                         device ushort *output [[buffer(1)]],
                         constant uint &count [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float value = h3_bf16_to_f32(input[gid]);
    output[gid] = h3_f32_to_bf16(value / (1.0f + exp(-value)));
}

kernel void h3_rms_norm_bf16(device const ushort *input [[buffer(0)]],
                             device const ushort *weight [[buffer(1)]],
                             device ushort *output [[buffer(2)]],
                             constant norm_args &args [[buffer(3)]],
                             uint3 group [[threadgroup_position_in_grid]],
                             uint3 thread_position [[thread_position_in_threadgroup]],
                             uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    device const ushort *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint k = tid; k < args.width; k += threads) {
        float value = h3_bf16_to_f32(x[k]);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    for (uint column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_to_f32(x[column]) * inverse;
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * h3_bf16_to_f32(weight[column]));
    }
}

kernel void h3_layer_norm_bf16(device const ushort *input [[buffer(0)]],
                               device const ushort *weight [[buffer(1)]],
                               device const ushort *bias [[buffer(2)]],
                               device ushort *output [[buffer(3)]],
                               constant norm_args &args [[buffer(4)]],
                               uint3 group [[threadgroup_position_in_grid]],
                               uint3 thread_position [[thread_position_in_threadgroup]],
                               uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    device const ushort *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint column = tid; column < args.width; column += threads)
        local_sum += h3_bf16_to_f32(x[column]);
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = reductions[0] / float(args.width);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_square = 0.0f;
    for (uint column = tid; column < args.width; column += threads) {
        float centered = h3_bf16_to_f32(x[column]) - mean;
        local_square = fma(centered, centered, local_square);
    }
    reductions[tid] = local_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    for (uint column = tid; column < args.width; column += threads) {
        float normalized = (h3_bf16_to_f32(x[column]) - mean) * inverse;
        float value = fma(normalized, h3_bf16_to_f32(weight[column]),
                          h3_bf16_to_f32(bias[column]));
        output[row * args.width + column] = h3_f32_to_bf16(value);
    }
}

struct gelu_bf16_args {
    uint elements;
    uint approximate;
};

inline float h3_erf_approx(float value) {
    float sign = value < 0.0f ? -1.0f : 1.0f;
    float x = abs(value);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float polynomial = (((((1.061405429f * t - 1.453152027f) * t) +
                           1.421413741f) * t - 0.284496736f) * t +
                           0.254829592f) * t;
    return sign * (1.0f - polynomial * exp(-x * x));
}

kernel void h3_gelu_bf16(device const ushort *input [[buffer(0)]],
                         device ushort *output [[buffer(1)]],
                         constant gelu_bf16_args &args [[buffer(2)]],
                         uint index [[thread_position_in_grid]]) {
    if (index >= args.elements) return;
    float value = h3_bf16_to_f32(input[index]);
    float activated;
    if (args.approximate) {
        float inner = 0.7978845608028654f *
            (value + 0.044715f * value * value * value);
        /* Apple GPU tanh may return NaN for large finite arguments under the
         * runtime compiler's fast-math mode. Saturation is already exact at
         * BF16 precision outside this range. */
        activated = inner <= -10.0f ? 0.0f :
                    inner >= 10.0f ? value :
                    0.5f * value * (1.0f + tanh(inner));
    } else {
        activated = value <= -10.0f ? 0.0f :
                    value >= 10.0f ? value :
                    0.5f * value *
                    (1.0f + h3_erf_approx(value * 0.7071067811865475f));
    }
    output[index] = h3_f32_to_bf16(activated);
}

kernel void h3_vision_qkv_rope_bf16(
                            device const ushort *qkv [[buffer(0)]],
                            device const ushort *rope_cos [[buffer(1)]],
                            device const ushort *rope_sin [[buffer(2)]],
                            device ushort *query [[buffer(3)]],
                            device ushort *key [[buffer(4)]],
                            device ushort *value [[buffer(5)]],
                            constant qkv_args &args [[buffer(6)]],
                            uint3 gid [[thread_position_in_grid]]) {
    uint dimension = gid.x;
    uint head = gid.y;
    uint row = gid.z;
    if (dimension >= args.head_dim || head >= args.heads ||
        row >= args.sequence) return;
    uint inner = args.heads * args.head_dim;
    uint row_base = row * inner * 3;
    uint q_base = row_base + head * args.head_dim;
    uint k_base = row_base + inner + head * args.head_dim;
    uint v_base = row_base + inner * 2 + head * args.head_dim;
    uint half_dim = args.rope_half;
    uint rope_index = row * half_dim + dimension % half_dim;
    float c = h3_bf16_to_f32(rope_cos[rope_index]);
    float s = h3_bf16_to_f32(rope_sin[rope_index]);
    uint pair = dimension < half_dim ? dimension + half_dim :
                dimension - half_dim;
    float q0 = h3_bf16_to_f32(qkv[q_base + dimension]);
    float k0 = h3_bf16_to_f32(qkv[k_base + dimension]);
    float q1 = h3_bf16_to_f32(qkv[q_base + pair]);
    float k1 = h3_bf16_to_f32(qkv[k_base + pair]);
    float qr = dimension < half_dim ? q0 * c - q1 * s : q0 * c + q1 * s;
    float kr = dimension < half_dim ? k0 * c - k1 * s : k0 * c + k1 * s;
    uint output_index = (row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = h3_f32_to_bf16(qr);
    key[output_index] = h3_f32_to_bf16(kr);
    value[output_index] = qkv[v_base + dimension];
}

kernel void h3_adaln_bf16(device const ushort *input [[buffer(0)]],
                          device const ushort *weight [[buffer(1)]],
                          device const ushort *modulation [[buffer(2)]],
                          device const uint *row_map [[buffer(3)]],
                          device ushort *output [[buffer(4)]],
                          constant adaln_args &args [[buffer(5)]],
                          uint3 group [[threadgroup_position_in_grid]],
                          uint3 thread_position [[thread_position_in_threadgroup]],
                          uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    device const ushort *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint k = tid; k < args.width; k += threads) {
        float value = h3_bf16_to_f32(x[k]);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    uint base = row_map[row] * args.slots * args.width;
    for (uint column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_to_f32(x[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            modulation[base + args.scale_slot * args.width + column]);
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

/* Preserve the standalone AdaLN reduction tree while emitting only the one
 * F32 inverse RMS scalar needed by the fused final projection. */
kernel void h3_rms_inverse_bf16(
                          device const ushort *input [[buffer(0)]],
                          device float *inverse [[buffer(1)]],
                          constant norm_args &args [[buffer(2)]],
                          uint3 group [[threadgroup_position_in_grid]],
                          uint3 thread_position
                              [[thread_position_in_threadgroup]],
                          uint3 threadgroup_size
                              [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    device const ushort *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint k = tid; k < args.width; k += threads) {
        float value = h3_bf16_to_f32(x[k]);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0)
        inverse[row] = rsqrt(reductions[0] / float(args.width) +
                             args.epsilon);
}

struct h3_adaln_linear_args {
    uint rows;
    uint width;
    uint output_dim;
    uint slots;
    uint shift_slot;
    uint scale_slot;
    uint has_bias;
};

/* Apply final AdaLN while loading the existing 16x16 linear tiles. The
 * normalized value is rounded to BF16 before the dot product, retaining the
 * exact arithmetic boundary and accumulation order of AdaLN + linear. */
kernel void h3_adaln_linear_bf16(
                          device const ushort *input [[buffer(0)]],
                          device const float *inverse [[buffer(1)]],
                          device const ushort *norm_weight [[buffer(2)]],
                          device const ushort *modulation [[buffer(3)]],
                          device const uint *row_map [[buffer(4)]],
                          device const ushort *weight [[buffer(5)]],
                          device const ushort *bias [[buffer(6)]],
                          device ushort *output [[buffer(7)]],
                          constant h3_adaln_linear_args &args [[buffer(8)]],
                          uint2 tid [[thread_position_in_threadgroup]],
                          uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup ushort input_tile[16][16];
    threadgroup float weight_tile[16][16];
    uint row = group.y * 16 + tid.y;
    uint column = group.x * 16 + tid.x;
    float sum = args.has_bias && column < args.output_dim ?
        h3_bf16_to_f32(bias[column]) : 0.0f;
    uint base = row < args.rows ?
        row_map[row] * args.slots * args.width : 0;
    uint tile_count = (args.width + 15) / 16;
    for (uint tile = 0; tile < tile_count; tile++) {
        uint input_k = tile * 16 + tid.x;
        ushort normalized = 0;
        if (row < args.rows && input_k < args.width) {
            float value = h3_bf16_to_f32(
                input[row * args.width + input_k]);
            float shift = h3_bf16_to_f32(
                modulation[base + args.shift_slot * args.width + input_k]);
            float scale = h3_bf16_to_f32(
                modulation[base + args.scale_slot * args.width + input_k]);
            float normed = value * inverse[row] *
                h3_bf16_to_f32(norm_weight[input_k]);
            normalized = h3_f32_to_bf16(normed * (1.0f + scale) + shift);
        }
        input_tile[tid.y][tid.x] = normalized;
        uint weight_k = tile * 16 + tid.y;
        weight_tile[tid.y][tid.x] =
            column < args.output_dim && weight_k < args.width ?
            h3_bf16_to_f32(weight[column * args.width + weight_k]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; k++)
            sum = fma(h3_bf16_to_f32(input_tile[tid.y][k]),
                      weight_tile[k][tid.x], sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < args.rows && column < args.output_dim)
        output[row * args.output_dim + column] = h3_f32_to_bf16(sum);
}

kernel void h3_gate_bf16(device const ushort *residual [[buffer(0)]],
                         device const ushort *branch [[buffer(1)]],
                         device const ushort *modulation [[buffer(2)]],
                         device const uint *row_map [[buffer(3)]],
                         device ushort *output [[buffer(4)]],
                         constant gate_args &args [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint base = row_map[row] * args.slots * args.width;
    float gate = h3_bf16_to_f32(modulation[base + args.gate_slot * args.width + column]);
    uint index = row * args.width + column;
    float value = h3_bf16_to_f32(residual[index]) +
                  h3_bf16_to_f32(branch[index]) * gate;
    output[index] = h3_f32_to_bf16(value);
}

struct h3_gate_adaln_args {
    uint rows;
    uint width;
    uint slots;
    uint gate_slot;
    uint shift_slot;
    uint scale_slot;
    float epsilon;
};

/* Preserve the gate's BF16 rounding boundary while retaining that rounded row
 * in threadgroup memory for the immediately following AdaLN. */
kernel void h3_gate_adaln_bf16(
                         device const ushort *residual [[buffer(0)]],
                         device const ushort *branch [[buffer(1)]],
                         device const ushort *gate_modulation [[buffer(2)]],
                         device const uint *row_map [[buffer(3)]],
                         device const ushort *weight [[buffer(4)]],
                         device const ushort *norm_modulation [[buffer(5)]],
                         device ushort *gated_residual [[buffer(6)]],
                         device ushort *output [[buffer(7)]],
                         constant h3_gate_adaln_args &args [[buffer(8)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint3 thread_position
                             [[thread_position_in_threadgroup]],
                         uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    threadgroup ushort gated_values[5376];
    uint base = row_map[row] * args.slots * args.width;
    float local_sum = 0.0f;
    for (uint column = tid; column < args.width; column += threads) {
        uint index = row * args.width + column;
        float gate = h3_bf16_to_f32(
            gate_modulation[base + args.gate_slot * args.width + column]);
        ushort gated = h3_f32_to_bf16(
            h3_bf16_to_f32(residual[index]) +
            h3_bf16_to_f32(branch[index]) * gate);
        gated_residual[index] = gated;
        gated_values[column] = gated;
        float value = h3_bf16_to_f32(gated);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    for (uint column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_to_f32(gated_values[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            norm_modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            norm_modulation[base + args.scale_slot * args.width + column]);
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

/* Preserve the exact 256-way reduction tree, but perform its final five
 * stages with SIMD shuffles after the cross-SIMD stages have completed. */
kernel void h3_gate_adaln_bf16_exact_simd(
                         device const ushort *residual [[buffer(0)]],
                         device const ushort *branch [[buffer(1)]],
                         device const ushort *gate_modulation [[buffer(2)]],
                         device const uint *row_map [[buffer(3)]],
                         device const ushort *weight [[buffer(4)]],
                         device const ushort *norm_modulation [[buffer(5)]],
                         device ushort *gated_residual [[buffer(6)]],
                         device ushort *output [[buffer(7)]],
                         constant h3_gate_adaln_args &args [[buffer(8)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint3 thread_position
                             [[thread_position_in_threadgroup]],
                         uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    threadgroup ushort gated_values[5376];
    uint base = row_map[row] * args.slots * args.width;
    float local_sum = 0.0f;
    for (uint column = tid; column < args.width; column += threads) {
        uint index = row * args.width + column;
        float gate = h3_bf16_to_f32(
            gate_modulation[base + args.gate_slot * args.width + column]);
        ushort gated = h3_f32_to_bf16(
            h3_bf16_to_f32(residual[index]) +
            h3_bf16_to_f32(branch[index]) * gate);
        gated_residual[index] = gated;
        gated_values[column] = gated;
        float value = h3_bf16_to_f32(gated);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 32) {
        float sum = reductions[tid];
        for (uint stride = 16; stride; stride >>= 1) {
            float partner = simd_shuffle_down(sum, stride);
            if (tid < stride) sum += partner;
        }
        if (tid == 0)
            reductions[0] = rsqrt(sum / float(args.width) + args.epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverse = reductions[0];
    for (uint column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_to_f32(gated_values[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            norm_modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            norm_modulation[base + args.scale_slot * args.width + column]);
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

/* M5 int8 projections consume AdaLN through a dynamic per-row quantizer. Fold
 * that quantizer into the preceding gated-residual/AdaLN pass. The rounded
 * gated row is overwritten in-place by the rounded AdaLN row once its RMS is
 * known, so exact standalone-quantizer bytes need only one 10.5 KiB row in
 * threadgroup memory and never round-trip through device BF16 memory. */
kernel void h3_gate_adaln_quantize_int8_scalar(
                         device const ushort *residual [[buffer(0)]],
                         device const ushort *branch [[buffer(1)]],
                         device const ushort *gate_modulation [[buffer(2)]],
                         device const uint *row_map [[buffer(3)]],
                         device const ushort *weight [[buffer(4)]],
                         device const ushort *norm_modulation [[buffer(5)]],
                         device ushort *gated_residual [[buffer(6)]],
                         device int8_t *quantized [[buffer(7)]],
                         device float *quantized_scales [[buffer(8)]],
                         constant h3_gate_adaln_args &args [[buffer(9)]],
                         uint tid [[thread_index_in_threadgroup]],
                         ushort simdgroup [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]],
                         uint row [[threadgroup_position_in_grid]]) {
    if (row >= args.rows) {
        uint quantized_base = row * args.width;
        for (uint column = tid; column < args.width; column += 256)
            quantized[quantized_base + column] = 0;
        if (tid == 0) quantized_scales[row] = 1.0f;
        return;
    }
    threadgroup float reductions[256];
    threadgroup ushort gated_values[5376];
    threadgroup float max_scratch[8];
    uint base = row_map[row] * args.slots * args.width;
    uint row_base = row * args.width;
    float local_sum = 0.0f;
    for (uint column = tid; column < args.width; column += 256) {
        uint index = row_base + column;
        float gate = h3_bf16_to_f32(
            gate_modulation[base + args.gate_slot * args.width + column]);
        ushort gated = h3_f32_to_bf16(
            h3_bf16_to_f32(residual[index]) +
            h3_bf16_to_f32(branch[index]) * gate);
        gated_residual[index] = gated;
        gated_values[column] = gated;
        float value = h3_bf16_to_f32(gated);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride >= 32; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 32) {
        float sum = reductions[tid];
        for (uint stride = 16; stride; stride >>= 1) {
            float partner = simd_shuffle_down(sum, stride);
            if (tid < stride) sum += partner;
        }
        if (tid == 0)
            reductions[0] = rsqrt(sum / float(args.width) + args.epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverse = reductions[0];
    float local_max = 0.0f;
    for (uint column = tid; column < args.width; column += 256) {
        float normalized = h3_bf16_to_f32(gated_values[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            norm_modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            norm_modulation[base + args.scale_slot * args.width + column]);
        ushort value = h3_f32_to_bf16(
            normalized * (1.0f + scale) + shift);
        gated_values[column] = value;
        local_max = max(local_max, fabs(h3_bf16_to_f32(value)));
    }
    float max_abs = h3_int8_reduce_max(
        local_max, max_scratch, simdgroup, lane);
    float output_scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
    float quantize_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
    if (tid == 0) quantized_scales[row] = output_scale;
    for (uint column = tid; column < args.width; column += 256) {
        int value = (int)rint(
            h3_bf16_to_f32(gated_values[column]) * quantize_scale);
        quantized[row_base + column] = (int8_t)clamp(value, -127, 127);
    }
}

/* The full H3 width is a multiple of four. Load every row stream as four
 * adjacent BF16 values so each pass issues a quarter as many memory
 * instructions while retaining the same BF16 rounding boundaries. */
kernel void h3_gate_adaln_quantize_int8(
                         device const ushort *residual [[buffer(0)]],
                         device const ushort *branch [[buffer(1)]],
                         device const ushort *gate_modulation [[buffer(2)]],
                         device const uint *row_map [[buffer(3)]],
                         device const ushort *weight [[buffer(4)]],
                         device const ushort *norm_modulation [[buffer(5)]],
                         device ushort *gated_residual [[buffer(6)]],
                         device int8_t *quantized [[buffer(7)]],
                         device float *quantized_scales [[buffer(8)]],
                         constant h3_gate_adaln_args &args [[buffer(9)]],
                         uint tid [[thread_index_in_threadgroup]],
                         ushort simdgroup [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]],
                         uint row [[threadgroup_position_in_grid]]) {
    constexpr uint VECTOR_WIDTH = 5376 / 4;
    if (row >= args.rows) {
        device char4 *quantized4 =
            reinterpret_cast<device char4 *>(quantized);
        uint quantized_base = row * VECTOR_WIDTH;
        for (uint vector = tid; vector < VECTOR_WIDTH; vector += 256)
            quantized4[quantized_base + vector] = char4(0);
        if (tid == 0) quantized_scales[row] = 1.0f;
        return;
    }
    threadgroup float reductions[256];
    threadgroup ushort4 gated_values[VECTOR_WIDTH];
    threadgroup float max_scratch[8];
    uint base = row_map[row] * args.slots * args.width;
    uint vector_base = row * VECTOR_WIDTH;
    uint modulation_base = base / 4;
    device const ushort4 *residual4 =
        reinterpret_cast<device const ushort4 *>(residual);
    device const ushort4 *branch4 =
        reinterpret_cast<device const ushort4 *>(branch);
    device const ushort4 *gate4 =
        reinterpret_cast<device const ushort4 *>(gate_modulation);
    device ushort4 *gated_residual4 =
        reinterpret_cast<device ushort4 *>(gated_residual);
    for (uint vector = tid; vector < VECTOR_WIDTH; vector += 256) {
        ushort4 residual_bits = residual4[vector_base + vector];
        ushort4 branch_bits = branch4[vector_base + vector];
        ushort4 gate_bits = gate4[modulation_base +
            args.gate_slot * VECTOR_WIDTH + vector];
        float4 gated_float = h3_bf16x4_to_f32(residual_bits) +
            h3_bf16x4_to_f32(branch_bits) * h3_bf16x4_to_f32(gate_bits);
        ushort4 gated = h3_f32x4_to_bf16(gated_float);
        gated_residual4[vector_base + vector] = gated;
        gated_values[vector] = gated;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup ushort *gated_scalars =
        reinterpret_cast<threadgroup ushort *>(gated_values);
    float local_sum = 0.0f;
    for (uint column = tid; column < args.width; column += 256) {
        float value = h3_bf16_to_f32(gated_scalars[column]);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride >= 32; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 32) {
        float sum = reductions[tid];
        for (uint stride = 16; stride; stride >>= 1) {
            float partner = simd_shuffle_down(sum, stride);
            if (tid < stride) sum += partner;
        }
        if (tid == 0)
            reductions[0] = rsqrt(sum / float(args.width) + args.epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const ushort4 *weight4 =
        reinterpret_cast<device const ushort4 *>(weight);
    device const ushort4 *norm_modulation4 =
        reinterpret_cast<device const ushort4 *>(norm_modulation);
    float inverse = reductions[0];
    float local_max = 0.0f;
    for (uint vector = tid; vector < VECTOR_WIDTH; vector += 256) {
        ushort4 gated = gated_values[vector];
        ushort4 norm_bits = weight4[vector];
        ushort4 shift_bits = norm_modulation4[modulation_base +
            args.shift_slot * VECTOR_WIDTH + vector];
        ushort4 scale_bits = norm_modulation4[modulation_base +
            args.scale_slot * VECTOR_WIDTH + vector];
        float4 normalized = h3_bf16x4_to_f32(gated) * inverse *
            h3_bf16x4_to_f32(norm_bits);
        float4 value_float = normalized *
            (float4(1.0f) + h3_bf16x4_to_f32(scale_bits)) +
            h3_bf16x4_to_f32(shift_bits);
        ushort4 value = h3_f32x4_to_bf16(value_float);
        gated_values[vector] = value;
        float4 rounded = h3_bf16x4_to_f32(value);
        local_max = max(local_max,
            max(max(fabs(rounded.x), fabs(rounded.y)),
                max(fabs(rounded.z), fabs(rounded.w))));
    }
    float max_abs = h3_int8_reduce_max(
        local_max, max_scratch, simdgroup, lane);
    float output_scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
    float quantize_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
    if (tid == 0) quantized_scales[row] = output_scale;
    device char4 *quantized4 = reinterpret_cast<device char4 *>(quantized);
    for (uint vector = tid; vector < VECTOR_WIDTH; vector += 256) {
        ushort4 bits = gated_values[vector];
        int4 value = int4(rint(
            h3_bf16x4_to_f32(bits) * quantize_scale));
        quantized4[vector_base + vector] =
            char4(clamp(value, int4(-127), int4(127)));
    }
}

kernel void h3_qkv_rope_bf16(device const ushort *qkv [[buffer(0)]],
                             device const ushort *q_weight [[buffer(1)]],
                             device const ushort *k_weight [[buffer(2)]],
                             device const ushort *rope_cos [[buffer(3)]],
                             device const ushort *rope_sin [[buffer(4)]],
                             device ushort *query [[buffer(5)]],
                             device ushort *key [[buffer(6)]],
                             device ushort *value [[buffer(7)]],
                             constant qkv_args &args [[buffer(8)]],
                             uint3 gid [[thread_position_in_grid]]) {
    uint dimension = gid.x;
    uint head = gid.y;
    uint row = gid.z;
    if (dimension >= args.head_dim || head >= args.heads || row >= args.sequence) return;
    uint inner = args.heads * args.head_dim;
    uint row_base = row * inner * 3;
    uint q_base = row_base + head * args.head_dim;
    uint k_base = q_base + inner;
    uint v_base = q_base + inner * 2;
    if (args.grouped) {
        q_base = row_base + head * args.head_dim * 3;
        k_base = q_base + args.head_dim;
        v_base = k_base + args.head_dim;
    }
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint d = 0; d < args.head_dim; d++) {
        float q = h3_bf16_to_f32(qkv[q_base + d]);
        float k = h3_bf16_to_f32(qkv[k_base + d]);
        q_sum = fma(q, q, q_sum);
        k_sum = fma(k, k, k_sum);
    }
    float q_inverse = rsqrt(q_sum / float(args.head_dim) + args.epsilon);
    float k_inverse = rsqrt(k_sum / float(args.head_dim) + args.epsilon);
    float q0 = h3_bf16_to_f32(qkv[q_base + dimension]) * q_inverse *
               h3_bf16_to_f32(q_weight[dimension]);
    float k0 = h3_bf16_to_f32(qkv[k_base + dimension]) * k_inverse *
               h3_bf16_to_f32(k_weight[dimension]);
    if (dimension < args.rope_half) {
        uint pair = dimension + args.rope_half;
        float q1 = h3_bf16_to_f32(qkv[q_base + pair]) * q_inverse *
                   h3_bf16_to_f32(q_weight[pair]);
        float k1 = h3_bf16_to_f32(qkv[k_base + pair]) * k_inverse *
                   h3_bf16_to_f32(k_weight[pair]);
        float c = h3_bf16_to_f32(rope_cos[row * args.rope_half + dimension]);
        float s = h3_bf16_to_f32(rope_sin[row * args.rope_half + dimension]);
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2) {
        uint pair = dimension - args.rope_half;
        float q1 = h3_bf16_to_f32(qkv[q_base + pair]) * q_inverse *
                   h3_bf16_to_f32(q_weight[pair]);
        float k1 = h3_bf16_to_f32(qkv[k_base + pair]) * k_inverse *
                   h3_bf16_to_f32(k_weight[pair]);
        float c = h3_bf16_to_f32(rope_cos[row * args.rope_half + pair]);
        float s = h3_bf16_to_f32(rope_sin[row * args.rope_half + pair]);
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint output_index = (row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = h3_f32_to_bf16(q0);
    key[output_index] = h3_f32_to_bf16(k0);
    value[output_index] = qkv[v_base + dimension];
}

/* One SIMD group owns one (row, head), so a 128-thread threadgroup processes
 * four heads.  Lane zero forms its head's norm once in the oracle's original
 * scalar order.  Each lane then owns four dimensions; the +/-64 RoPE partner
 * stays in that same lane.  This removes duplicate reductions while keeping
 * every floating-point expression and rounding boundary unchanged. */
kernel void h3_qkv_rope_bf16_coop_uncached(
                             device const ushort *qkv [[buffer(0)]],
                             device const ushort *q_weight [[buffer(1)]],
                             device const ushort *k_weight [[buffer(2)]],
                             device const ushort *rope_cos [[buffer(3)]],
                             device const ushort *rope_sin [[buffer(4)]],
                             device ushort *query [[buffer(5)]],
                             device ushort *key [[buffer(6)]],
                             device ushort *value [[buffer(7)]],
                             constant qkv_args &args [[buffer(8)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint WIDTH = 128;
    constexpr uint HEADS_PER_GROUP = 4;
    uint head = group.x * HEADS_PER_GROUP + simdgroup;
    uint row = group.y;
    if (head >= args.heads || row >= args.sequence) return;
    uint inner = args.heads * args.head_dim;
    uint row_base = row * inner * 3;
    uint q_base = row_base + head * args.head_dim;
    uint k_base = q_base + inner;
    uint v_base = q_base + inner * 2;
    if (args.grouped) {
        q_base = row_base + head * args.head_dim * 3;
        k_base = q_base + args.head_dim;
        v_base = k_base + args.head_dim;
    }

    threadgroup float q_inverse[HEADS_PER_GROUP];
    threadgroup float k_inverse[HEADS_PER_GROUP];
    if (lane == 0) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (uint d = 0; d < WIDTH; d++) {
            float q_element = h3_bf16_to_f32(qkv[q_base + d]);
            float k_element = h3_bf16_to_f32(qkv[k_base + d]);
            q_sum = fma(q_element, q_element, q_sum);
            k_sum = fma(k_element, k_element, k_sum);
        }
        q_inverse[simdgroup] =
            rsqrt(q_sum / float(args.head_dim) + args.epsilon);
        k_inverse[simdgroup] =
            rsqrt(k_sum / float(args.head_dim) + args.epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dimension = lane; dimension < WIDTH; dimension += 32) {
        float q0 = h3_bf16_to_f32(qkv[q_base + dimension]) *
                   q_inverse[simdgroup] *
                   h3_bf16_to_f32(q_weight[dimension]);
        float k0 = h3_bf16_to_f32(qkv[k_base + dimension]) *
                   k_inverse[simdgroup] *
                   h3_bf16_to_f32(k_weight[dimension]);
        if (dimension < args.rope_half) {
            uint pair = dimension + args.rope_half;
            float q1 = h3_bf16_to_f32(qkv[q_base + pair]) *
                       q_inverse[simdgroup] *
                       h3_bf16_to_f32(q_weight[pair]);
            float k1 = h3_bf16_to_f32(qkv[k_base + pair]) *
                       k_inverse[simdgroup] *
                       h3_bf16_to_f32(k_weight[pair]);
            float c = h3_bf16_to_f32(
                rope_cos[row * args.rope_half + dimension]);
            float s = h3_bf16_to_f32(
                rope_sin[row * args.rope_half + dimension]);
            q0 = q0 * c - q1 * s;
            k0 = k0 * c - k1 * s;
        } else if (dimension < args.rope_half * 2) {
            uint pair = dimension - args.rope_half;
            float q1 = h3_bf16_to_f32(qkv[q_base + pair]) *
                       q_inverse[simdgroup] *
                       h3_bf16_to_f32(q_weight[pair]);
            float k1 = h3_bf16_to_f32(qkv[k_base + pair]) *
                       k_inverse[simdgroup] *
                       h3_bf16_to_f32(k_weight[pair]);
            float c = h3_bf16_to_f32(
                rope_cos[row * args.rope_half + pair]);
            float s = h3_bf16_to_f32(
                rope_sin[row * args.rope_half + pair]);
            q0 = q0 * c + q1 * s;
            k0 = k0 * c + k1 * s;
        }
        uint output_index =
            (row * args.heads + head) * args.head_dim + dimension;
        query[output_index] = h3_f32_to_bf16(q0);
        key[output_index] = h3_f32_to_bf16(k0);
        value[output_index] = qkv[v_base + dimension];
    }
}

/* Exact cooperative kernel with a per-threadgroup Q/K cache. Coalesced lanes
 * load every projected value once; lane zero retains the oracle's scalar RMS
 * order, and the RoPE epilogue reuses the cached values. Each SIMD group owns
 * disjoint storage, so only SIMD-local barriers are required. */
kernel void h3_qkv_rope_bf16_coop(
                             device const ushort *qkv [[buffer(0)]],
                             device const ushort *q_weight [[buffer(1)]],
                             device const ushort *k_weight [[buffer(2)]],
                             device const ushort *rope_cos [[buffer(3)]],
                             device const ushort *rope_sin [[buffer(4)]],
                             device ushort *query [[buffer(5)]],
                             device ushort *key [[buffer(6)]],
                             device ushort *value [[buffer(7)]],
                             constant qkv_args &args [[buffer(8)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint WIDTH = 128;
    constexpr uint HEADS_PER_GROUP = 4;
    uint head = group.x * HEADS_PER_GROUP + simdgroup;
    uint row = group.y;
    if (head >= args.heads || row >= args.sequence) return;
    uint inner = args.heads * args.head_dim;
    uint row_base = row * inner * 3;
    uint q_base = row_base + head * args.head_dim;
    uint k_base = q_base + inner;
    uint v_base = q_base + inner * 2;
    if (args.grouped) {
        q_base = row_base + head * args.head_dim * 3;
        k_base = q_base + args.head_dim;
        v_base = k_base + args.head_dim;
    }

    threadgroup ushort q_values[HEADS_PER_GROUP * WIDTH];
    threadgroup ushort k_values[HEADS_PER_GROUP * WIDTH];
    uint cache_base = simdgroup * WIDTH;
    for (uint dimension = lane; dimension < WIDTH; dimension += 32) {
        q_values[cache_base + dimension] = qkv[q_base + dimension];
        k_values[cache_base + dimension] = qkv[k_base + dimension];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float q_inverse[HEADS_PER_GROUP];
    threadgroup float k_inverse[HEADS_PER_GROUP];
    if (lane == 0) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (uint dimension = 0; dimension < WIDTH; dimension++) {
            float q_element =
                h3_bf16_to_f32(q_values[cache_base + dimension]);
            float k_element =
                h3_bf16_to_f32(k_values[cache_base + dimension]);
            q_sum = fma(q_element, q_element, q_sum);
            k_sum = fma(k_element, k_element, k_sum);
        }
        q_inverse[simdgroup] =
            rsqrt(q_sum / float(args.head_dim) + args.epsilon);
        k_inverse[simdgroup] =
            rsqrt(k_sum / float(args.head_dim) + args.epsilon);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dimension = lane; dimension < WIDTH; dimension += 32) {
        float q0 = h3_bf16_to_f32(q_values[cache_base + dimension]) *
                   q_inverse[simdgroup] *
                   h3_bf16_to_f32(q_weight[dimension]);
        float k0 = h3_bf16_to_f32(k_values[cache_base + dimension]) *
                   k_inverse[simdgroup] *
                   h3_bf16_to_f32(k_weight[dimension]);
        if (dimension < args.rope_half) {
            uint pair = dimension + args.rope_half;
            float q1 = h3_bf16_to_f32(q_values[cache_base + pair]) *
                       q_inverse[simdgroup] *
                       h3_bf16_to_f32(q_weight[pair]);
            float k1 = h3_bf16_to_f32(k_values[cache_base + pair]) *
                       k_inverse[simdgroup] *
                       h3_bf16_to_f32(k_weight[pair]);
            float c = h3_bf16_to_f32(
                rope_cos[row * args.rope_half + dimension]);
            float s = h3_bf16_to_f32(
                rope_sin[row * args.rope_half + dimension]);
            q0 = q0 * c - q1 * s;
            k0 = k0 * c - k1 * s;
        } else if (dimension < args.rope_half * 2) {
            uint pair = dimension - args.rope_half;
            float q1 = h3_bf16_to_f32(q_values[cache_base + pair]) *
                       q_inverse[simdgroup] *
                       h3_bf16_to_f32(q_weight[pair]);
            float k1 = h3_bf16_to_f32(k_values[cache_base + pair]) *
                       k_inverse[simdgroup] *
                       h3_bf16_to_f32(k_weight[pair]);
            float c = h3_bf16_to_f32(
                rope_cos[row * args.rope_half + pair]);
            float s = h3_bf16_to_f32(
                rope_sin[row * args.rope_half + pair]);
            q0 = q0 * c + q1 * s;
            k0 = k0 * c + k1 * s;
        }
        uint output_index =
            (row * args.heads + head) * args.head_dim + dimension;
        query[output_index] = h3_f32_to_bf16(q0);
        key[output_index] = h3_f32_to_bf16(k0);
        value[output_index] = qkv[v_base + dimension];
    }
}

kernel void h3_swiglu_bf16(device const ushort *fused [[buffer(0)]],
                           device ushort *output [[buffer(1)]],
                           constant swiglu_args &args [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint base = row * args.width * 2;
    float gate = h3_bf16_to_f32(fused[base + column]);
    float up = h3_bf16_to_f32(fused[base + args.width + column]);
    output[row * args.width + column] =
        h3_f32_to_bf16(gate / (1.0f + exp(-gate)) * up);
}

struct embedding_args {
    uint tokens;
    uint vocab_size;
    uint width;
};

kernel void h3_embedding_bf16(device const ushort *weight [[buffer(0)]],
                              device const uint *token_ids [[buffer(1)]],
                              device ushort *output [[buffer(2)]],
                              constant embedding_args &args [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint token = gid.y;
    if (token >= args.tokens || column >= args.width) return;
    uint identifier = token_ids[token];
    output[token * args.width + column] = identifier < args.vocab_size ?
        weight[identifier * args.width + column] : ushort(0);
}

struct text_rope_args {
    uint sequence;
    uint query_heads;
    uint kv_heads;
    uint head_dim;
    float epsilon;
};

kernel void h3_text_qk_rope_bf16(
        device const ushort *query_input [[buffer(0)]],
        device const ushort *key_input [[buffer(1)]],
        device const ushort *q_weight [[buffer(2)]],
        device const ushort *k_weight [[buffer(3)]],
        device const ushort *rope_cos [[buffer(4)]],
        device const ushort *rope_sin [[buffer(5)]],
        device ushort *query_output [[buffer(6)]],
        device ushort *key_output [[buffer(7)]],
        constant text_rope_args &args [[buffer(8)]],
        uint3 gid [[thread_position_in_grid]]) {
    uint dimension = gid.x;
    uint head = gid.y;
    uint row = gid.z;
    if (dimension >= args.head_dim || head >= args.query_heads || row >= args.sequence) return;
    uint half_dim = args.head_dim / 2;
    uint pair = dimension < half_dim ? dimension + half_dim : dimension - half_dim;
    float c = h3_bf16_to_f32(rope_cos[row * half_dim + (dimension % half_dim)]);
    float s = h3_bf16_to_f32(rope_sin[row * half_dim + (dimension % half_dim)]);

    uint q_base = (row * args.query_heads + head) * args.head_dim;
    float q_sum = 0.0f;
    for (uint d = 0; d < args.head_dim; d++) {
        float value = h3_bf16_to_f32(query_input[q_base + d]);
        q_sum = fma(value, value, q_sum);
    }
    float q_inverse = rsqrt(q_sum / float(args.head_dim) + args.epsilon);
    float q0 = h3_bf16_to_f32(query_input[q_base + dimension]) * q_inverse *
               h3_bf16_to_f32(q_weight[dimension]);
    float q1 = h3_bf16_to_f32(query_input[q_base + pair]) * q_inverse *
               h3_bf16_to_f32(q_weight[pair]);
    float q_rotated = dimension < half_dim ? q0 * c - q1 * s : q0 * c + q1 * s;
    query_output[q_base + dimension] = h3_f32_to_bf16(q_rotated);

    if (head < args.kv_heads) {
        uint k_base = (row * args.kv_heads + head) * args.head_dim;
        float k_sum = 0.0f;
        for (uint d = 0; d < args.head_dim; d++) {
            float value = h3_bf16_to_f32(key_input[k_base + d]);
            k_sum = fma(value, value, k_sum);
        }
        float k_inverse = rsqrt(k_sum / float(args.head_dim) + args.epsilon);
        float k0 = h3_bf16_to_f32(key_input[k_base + dimension]) * k_inverse *
                   h3_bf16_to_f32(k_weight[dimension]);
        float k1 = h3_bf16_to_f32(key_input[k_base + pair]) * k_inverse *
                   h3_bf16_to_f32(k_weight[pair]);
        float k_rotated = dimension < half_dim ? k0 * c - k1 * s : k0 * c + k1 * s;
        key_output[k_base + dimension] = h3_f32_to_bf16(k_rotated);
    }
}

struct gqa_args {
    uint sequence;
    uint query_heads;
    uint kv_heads;
    uint head_dim;
    float scale;
};

struct head_norm_args {
    uint sequence;
    uint heads;
    uint head_dim;
    float epsilon;
};

/* Adapted from Iris's Qwen3 BF16 per-head norm: a single thread owns a
 * complete head, making in-place normalization race-free. */
kernel void h3_head_rms_norm_bf16(
        device ushort *tensor [[buffer(0)]],
        device const ushort *weight [[buffer(1)]],
        constant head_norm_args &args [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.x;
    uint head = gid.y;
    if (row >= args.sequence || head >= args.heads) return;
    uint base = (row * args.heads + head) * args.head_dim;
    float sum = 0.0f;
    for (uint d = 0; d < args.head_dim; d++) {
        float value = h3_bf16_to_f32(tensor[base + d]);
        sum = fma(value, value, sum);
    }
    float inverse = rsqrt(sum / float(args.head_dim) + args.epsilon);
    for (uint d = 0; d < args.head_dim; d++) {
        float value = h3_bf16_to_f32(tensor[base + d]);
        tensor[base + d] = h3_f32_to_bf16(
            value * inverse * h3_bf16_to_f32(weight[d]));
    }
}

struct text_rope_inplace_args {
    uint sequence;
    uint query_heads;
    uint kv_heads;
    uint head_dim;
};

/* Iris-style text RoPE. F32 tables avoid compounding table quantization while
 * Q/K remain BF16 at the operation boundary. */
kernel void h3_rope_text_bf16(
        device ushort *query [[buffer(0)]],
        device ushort *key [[buffer(1)]],
        device const float *rope_cos [[buffer(2)]],
        device const float *rope_sin [[buffer(3)]],
        constant text_rope_inplace_args &args [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.x;
    uint head = gid.y;
    if (row >= args.sequence) return;
    uint half_dim = args.head_dim / 2;
    if (head < args.query_heads) {
        uint base = (row * args.query_heads + head) * args.head_dim;
        for (uint d = 0; d < half_dim; d++) {
            float first = h3_bf16_to_f32(query[base + d]);
            float second = h3_bf16_to_f32(query[base + half_dim + d]);
            float c = rope_cos[row * half_dim + d];
            float s = rope_sin[row * half_dim + d];
            query[base + d] = h3_f32_to_bf16(first * c - second * s);
            query[base + half_dim + d] = h3_f32_to_bf16(second * c + first * s);
        }
    }
    if (head < args.kv_heads) {
        uint base = (row * args.kv_heads + head) * args.head_dim;
        for (uint d = 0; d < half_dim; d++) {
            float first = h3_bf16_to_f32(key[base + d]);
            float second = h3_bf16_to_f32(key[base + half_dim + d]);
            float c = rope_cos[row * half_dim + d];
            float s = rope_sin[row * half_dim + d];
            key[base + d] = h3_f32_to_bf16(first * c - second * s);
            key[base + half_dim + d] = h3_f32_to_bf16(second * c + first * s);
        }
    }
}

kernel void h3_gqa_causal_bf16(
        device const ushort *query [[buffer(0)]],
        device const ushort *key [[buffer(1)]],
        device const ushort *value [[buffer(2)]],
        device ushort *output [[buffer(3)]],
        constant gqa_args &args [[buffer(4)]],
        threadgroup float *scores [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint3 thread_position [[thread_position_in_threadgroup]],
        uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    uint query_row = group.x;
    uint query_head = group.y;
    if (query_row >= args.sequence || query_head >= args.query_heads) return;
    uint kv_head = query_head / (args.query_heads / args.kv_heads);
    uint q_base = (query_row * args.query_heads + query_head) * args.head_dim;
    uint key_count = query_row + 1;
    threadgroup float reductions[128];
    threadgroup float shared_query[128];

    for (uint d = tid; d < args.head_dim; d += threads) {
        /* MLX's fused SDPA applies the scale to Q before the tiled QK
         * contraction. Matching that order matters at sharp late-layer
         * attention boundaries. */
        shared_query[d] = h3_bf16_to_f32(h3_f32_to_bf16(
            h3_bf16_to_f32(query[q_base + d]) * args.scale));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    for (uint key_row = tid; key_row < key_count; key_row += threads) {
        uint k_base = (key_row * args.kv_heads + kv_head) * args.head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < args.head_dim; d++) {
            dot = fma(shared_query[d], h3_bf16_to_f32(key[k_base + d]), dot);
        }
        float score = dot;
        scores[key_row] = score;
        local_max = max(local_max, score);
    }
    reductions[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] = max(reductions[tid], reductions[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maximum = reductions[0];
    /* Every thread must have read reductions[0] before any thread reuses
     * the array for the sum below - without this barrier a fast thread 0
     * overwrote the maximum with its partial sum while slower threads were
     * still reading it, so the same inputs intermittently gave different
     * attention output (seen as same-seed videos differing, most often
     * with the longer FL2VA/Ref2VA Qwen sequences). */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint key_row = tid; key_row < key_count; key_row += threads) {
        float probability = exp(scores[key_row] - maximum);
        scores[key_row] = probability;
        local_sum += probability;
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_sum = 1.0f / reductions[0];
    for (uint d = tid; d < args.head_dim; d += threads) {
        float sum = 0.0f;
        for (uint key_row = 0; key_row < key_count; key_row++) {
            uint v_index = (key_row * args.kv_heads + kv_head) * args.head_dim + d;
            sum = fma(scores[key_row] * inverse_sum,
                      h3_bf16_to_f32(value[v_index]), sum);
        }
        output[q_base + d] = h3_f32_to_bf16(sum);
    }
}

kernel void h3_add_bf16(device const ushort *left [[buffer(0)]],
                         device const ushort *right [[buffer(1)]],
                         device ushort *output [[buffer(2)]],
                         constant uint &count [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    output[gid] = h3_f32_to_bf16(h3_bf16_to_f32(left[gid]) +
                                  h3_bf16_to_f32(right[gid]));
}

kernel void h3_sub_bf16(device const ushort *left [[buffer(0)]],
                         device const ushort *right [[buffer(1)]],
                         device ushort *output [[buffer(2)]],
                         constant uint &count [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    output[gid] = h3_f32_to_bf16(h3_bf16_to_f32(left[gid]) -
                                  h3_bf16_to_f32(right[gid]));
}

struct h3_token_pool_args {
    uint input_offset;
    uint original_offset;
    uint baseline_offset;
    uint rows;
    uint width;
};

/* The pair table is [output row, {first, second}]. Singleton rows repeat their
 * source index so non-video prefixes remain bit exact. */
kernel void h3_token_pool_bf16(
                         device const ushort *input [[buffer(0)]],
                         device const uint2 *pairs [[buffer(1)]],
                         device ushort *output [[buffer(2)]],
                         device ushort *baseline [[buffer(3)]],
                         device const uint *baseline_indices [[buffer(4)]],
                         device ushort *original [[buffer(5)]],
                         constant h3_token_pool_args &args [[buffer(6)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint2 pair = pairs[row];
    ushort first = input[args.input_offset + pair.x * args.width + column];
    original[args.original_offset + pair.x * args.width + column] = first;
    ushort pooled = first;
    if (pair.x != pair.y) {
        ushort second = input[
            args.input_offset + pair.y * args.width + column];
        original[args.original_offset + pair.y * args.width + column] = second;
        float average = (h3_bf16_to_f32(first) +
                         h3_bf16_to_f32(second)) * 0.5f;
        pooled = h3_f32_to_bf16(average);
    }
    output[row * args.width + column] = pooled;
    uint baseline_index = baseline_indices[row];
    if (baseline_index != 0xffffffffu)
        baseline[args.baseline_offset + baseline_index * args.width + column] =
            pooled;
}

struct h3_token_pool_adaln_args {
    uint input_offset;
    uint original_offset;
    uint baseline_offset;
    uint rows;
    uint width;
    uint slots;
    uint shift_slot;
    uint scale_slot;
    float epsilon;
};

/* Pool and snapshot the full token grid while immediately producing the first
 * reduced block's attention AdaLN. Keeping the pooled BF16 row in threadgroup
 * memory avoids rereading the residual from global memory. */
kernel void h3_token_pool_adaln_bf16(
                         device const ushort *input [[buffer(0)]],
                         device const uint2 *pairs [[buffer(1)]],
                         device ushort *residual [[buffer(2)]],
                         device ushort *baseline [[buffer(3)]],
                         device const uint *baseline_indices [[buffer(4)]],
                         device ushort *original [[buffer(5)]],
                         device const ushort *weight [[buffer(6)]],
                         device const ushort *modulation [[buffer(7)]],
                         device const uint *row_map [[buffer(8)]],
                         device ushort *output [[buffer(9)]],
                         constant h3_token_pool_adaln_args &args
                             [[buffer(10)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint3 thread_position
                             [[thread_position_in_threadgroup]],
                         uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    threadgroup ushort pooled_values[5376];
    uint2 pair = pairs[row];
    uint baseline_index = baseline_indices[row];
    float local_sum = 0.0f;
    for (uint column = tid; column < args.width; column += threads) {
        uint first_index = args.input_offset +
            pair.x * args.width + column;
        ushort first = input[first_index];
        original[args.original_offset + pair.x * args.width + column] = first;
        ushort pooled = first;
        if (pair.x != pair.y) {
            ushort second = input[args.input_offset +
                pair.y * args.width + column];
            original[args.original_offset +
                     pair.y * args.width + column] = second;
            pooled = h3_f32_to_bf16((h3_bf16_to_f32(first) +
                                     h3_bf16_to_f32(second)) * 0.5f);
        }
        uint destination = row * args.width + column;
        residual[destination] = pooled;
        pooled_values[column] = pooled;
        if (baseline_index != 0xffffffffu)
            baseline[args.baseline_offset +
                     baseline_index * args.width + column] = pooled;
        float value = h3_bf16_to_f32(pooled);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    uint base = row_map[row] * args.slots * args.width;
    for (uint column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_to_f32(pooled_values[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            modulation[base + args.scale_slot * args.width + column]);
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

struct h3_token_expand_args {
    uint original_offset;
    uint baseline_offset;
    uint rows;
    uint width;
    uint exact_prefix_rows;
    float update_scale;
};

/* Restore the full grid while retaining the original within-pair detail. The
 * reduced stack contributes only its update relative to the pooled baseline. */
kernel void h3_token_expand_delta_bf16(
                         device const ushort *original [[buffer(0)]],
                         device const ushort *reduced [[buffer(1)]],
                         device const ushort *baseline [[buffer(2)]],
                         device const uint *baseline_indices [[buffer(3)]],
                         device const uint *parents [[buffer(4)]],
                         device ushort *output [[buffer(5)]],
                         constant h3_token_expand_args &args [[buffer(6)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint parent = parents[row];
    uint destination = row * args.width + column;
    uint reduced_index = parent * args.width + column;
    if (row < args.exact_prefix_rows) {
        output[destination] = reduced[reduced_index];
    } else {
        uint baseline_row = baseline_indices[parent];
        if (baseline_row == 0xffffffffu) {
            output[destination] = reduced[reduced_index];
            return;
        }
        uint baseline_index = args.baseline_offset +
            baseline_row * args.width + column;
        float update = h3_bf16_to_f32(reduced[reduced_index]) -
                       h3_bf16_to_f32(baseline[baseline_index]);
        output[destination] = h3_f32_to_bf16(
            h3_bf16_to_f32(original[args.original_offset + destination]) +
            args.update_scale * update);
    }
}

struct h3_token_expand_adaln_args {
    uint original_offset;
    uint baseline_offset;
    uint rows;
    uint width;
    uint exact_prefix_rows;
    uint slots;
    uint shift_slot;
    uint scale_slot;
    float update_scale;
    float epsilon;
};

/* H3's hidden width is 5376. Keep the restored BF16 row in 10.5 KiB of
 * threadgroup memory across the reduction instead of writing and then
 * rereading the residual globally. The host caps width at 5376. */
kernel void h3_token_expand_adaln_bf16(
                         device const ushort *original [[buffer(0)]],
                         device const ushort *reduced [[buffer(1)]],
                         device const ushort *baseline [[buffer(2)]],
                         device const uint *baseline_indices [[buffer(3)]],
                         device const uint *parents [[buffer(4)]],
                         device ushort *residual [[buffer(5)]],
                         device const ushort *weight [[buffer(6)]],
                         device const ushort *modulation [[buffer(7)]],
                         device const uint *row_map [[buffer(8)]],
                         device ushort *output [[buffer(9)]],
                         constant h3_token_expand_adaln_args &args
                             [[buffer(10)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint3 thread_position
                             [[thread_position_in_threadgroup]],
                         uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    threadgroup ushort restored_values[5376];
    uint parent = parents[row];
    uint baseline_row = baseline_indices[parent];
    bool direct = row < args.exact_prefix_rows ||
                  baseline_row == 0xffffffffu;
    float local_sum = 0.0f;
    for (uint column = tid; column < args.width; column += threads) {
        uint destination = row * args.width + column;
        uint reduced_index = parent * args.width + column;
        ushort restored = reduced[reduced_index];
        if (!direct) {
            uint baseline_index = args.baseline_offset +
                baseline_row * args.width + column;
            float update = h3_bf16_to_f32(restored) -
                           h3_bf16_to_f32(baseline[baseline_index]);
            restored = h3_f32_to_bf16(
                h3_bf16_to_f32(
                    original[args.original_offset + destination]) +
                args.update_scale * update);
        }
        restored_values[column] = restored;
        residual[destination] = restored;
        float value = h3_bf16_to_f32(restored);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    uint base = row_map[row] * args.slots * args.width;
    for (uint column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_to_f32(restored_values[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            modulation[base + args.scale_slot * args.width + column]);
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

struct h3_euler_args {
    uint sample_offset;
    uint elements;
    float delta;
    float ratio;
};

kernel void h3_euler_bf16(device float *sample [[buffer(0)]],
                           device const ushort *last [[buffer(1)]],
                           device const ushort *previous [[buffer(2)]],
                           constant h3_euler_args &args [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= args.elements) return;
    float last_value = h3_bf16_to_f32(last[gid]);
    float velocity = fma(args.ratio,
                         last_value - h3_bf16_to_f32(previous[gid]),
                         last_value);
    uint sample_index = args.sample_offset + gid;
    sample[sample_index] = fma(args.delta, velocity, sample[sample_index]);
}

/* Must match h3_pcg_hash / h3_renoise_normal in h3_host.c. */
inline uint h3_pcg_hash(uint value) {
    uint state = value * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct h3_renoise_args {
    uint sample_offset;
    uint elements;
    float sigma;
    float sigma_next;
    uint key0;
    uint key1;
};

kernel void h3_renoise_bf16(device float *sample [[buffer(0)]],
                            device const ushort *velocity [[buffer(1)]],
                            constant h3_renoise_args &args [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= args.elements) return;
    uint sample_index = args.sample_offset + gid;
    float clean = fma(args.sigma, h3_bf16_to_f32(velocity[gid]),
                      sample[sample_index]);
    uint a = h3_pcg_hash(h3_pcg_hash(gid) ^ args.key0);
    uint b = h3_pcg_hash(a ^ args.key1);
    float u1 = (float(a >> 8) + 1.0f) * (1.0f / 16777216.0f);
    float u2 = float(b >> 8) * (1.0f / 16777216.0f);
    float noise = precise::sqrt(-2.0f * precise::log(u1)) *
                  precise::cos(6.28318530717958647692f * u2);
    sample[sample_index] = fma(args.sigma_next, noise - clean, clean);
}

kernel void h3_silu_mul_bf16(device const ushort *gate [[buffer(0)]],
                              device const ushort *up [[buffer(1)]],
                              device ushort *output [[buffer(2)]],
                              constant uint &count [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float value = h3_bf16_to_f32(gate[gid]);
    float other = h3_bf16_to_f32(up[gid]);
    output[gid] = h3_f32_to_bf16(value / (1.0f + exp(-value)) * other);
}

// --- Block-quantized weights (GGUF/ggml layouts, see h3_quant.c) ---------
// One thread expands one block into BF16 (out_f32 == 0) or F32. The layouts
// and formulas mirror h3_quant.c's reference dequantizers, which the tests
// compare these against; `quant` is h3_quant.h's h3_quant_type.

struct h3_dequant_args {
    uint quant;
    uint blocks;
    uint out_f32;
    uint block_elements;
    // Row reorder for a fused QKV weight stored as [all Q; all K; all V]
    // (ComfyUI's layout) into the engine's per-head [q; k; v]: columns is
    // the row length, qkv_inner the Q (= K = V) row count, qkv_head_dim the
    // rows per head. qkv_inner == 0: no reorder.
    uint columns;
    uint qkv_inner;
    uint qkv_head_dim;
};

inline ulong h3_q_destination(constant h3_dequant_args &args, ulong element) {
    if (args.qkv_inner == 0) return element;
    ulong row = element / args.columns, column = element % args.columns;
    ulong part = row / args.qkv_inner, within = row % args.qkv_inner;
    ulong head = within / args.qkv_head_dim, d = within % args.qkv_head_dim;
    ulong destination = head * 3 * args.qkv_head_dim + part * args.qkv_head_dim + d;
    return destination * args.columns + column;
}

inline ushort h3_q_u16(device const uchar *b) {
    return ushort(b[0]) | (ushort(b[1]) << 8);
}

inline uint h3_q_u32(device const uchar *b) {
    return uint(b[0]) | (uint(b[1]) << 8) | (uint(b[2]) << 16) | (uint(b[3]) << 24);
}

inline float h3_q_half(device const uchar *b) {
    return float(as_type<half>(h3_q_u16(b)));
}

inline void h3_q_put(device ushort *bf16, device float *f32, bool to_f32,
                     ulong index, float value) {
    if (to_f32) f32[index] = value;
    else bf16[index] = h3_f32_to_bf16(value);
}

inline void h3_q_scale_min_k4(int j, device const uchar *q, thread uint &sc,
                              thread uint &m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

kernel void h3_dequantize(device const uchar *raw [[buffer(0)]],
                          device ushort *out_bf16 [[buffer(1)]],
                          device float *out_f32 [[buffer(2)]],
                          constant h3_dequant_args &args [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= args.blocks) return;
    bool to_f32 = args.out_f32 != 0;
    ulong block = gid;
    // Every block lies within one row (rows are whole blocks), so a row
    // reorder only moves where the block starts.
    ulong start = h3_q_destination(args, block * args.block_elements);
    switch (args.quant) {
    case 1: { // F32 (copied - only to reorder rows)
        device const uchar *b = raw + block * 4;
        h3_q_put(out_bf16, out_f32, to_f32, start, as_type<float>(h3_q_u32(b)));
        break;
    }
    case 2: { // F16
        h3_q_put(out_bf16, out_f32, to_f32, start, h3_q_half(raw + block * 2));
        break;
    }
    case 3: { // BF16 (copied - only to reorder rows)
        h3_q_put(out_bf16, out_f32, to_f32, start,
                 h3_bf16_to_f32(h3_q_u16(raw + block * 2)));
        break;
    }
    case 4: { // Q4_0
        device const uchar *b = raw + block * 18;
        float d = h3_q_half(b);
        ulong y = start;
        for (int j = 0; j < 16; j++) {
            h3_q_put(out_bf16, out_f32, to_f32, y + j, float(int(b[2 + j] & 0x0F) - 8) * d);
            h3_q_put(out_bf16, out_f32, to_f32, y + j + 16, float(int(b[2 + j] >> 4) - 8) * d);
        }
        break;
    }
    case 5: { // Q4_1
        device const uchar *b = raw + block * 20;
        float d = h3_q_half(b), m = h3_q_half(b + 2);
        ulong y = start;
        for (int j = 0; j < 16; j++) {
            h3_q_put(out_bf16, out_f32, to_f32, y + j, float(b[4 + j] & 0x0F) * d + m);
            h3_q_put(out_bf16, out_f32, to_f32, y + j + 16, float(b[4 + j] >> 4) * d + m);
        }
        break;
    }
    case 6: case 7: { // Q5_0, Q5_1
        bool one = args.quant == 7;
        device const uchar *b = raw + block * (one ? 24 : 22);
        float d = h3_q_half(b), m = one ? h3_q_half(b + 2) : 0.0f;
        uint qh = h3_q_u32(b + (one ? 4 : 2));
        device const uchar *qs = b + (one ? 8 : 6);
        ulong y = start;
        for (int j = 0; j < 16; j++) {
            uint h0 = ((qh >> uint(j)) << 4) & 0x10;
            uint h1 = (qh >> uint(j + 12)) & 0x10;
            int x0 = int((qs[j] & 0x0F) | h0), x1 = int((qs[j] >> 4) | h1);
            if (one) {
                h3_q_put(out_bf16, out_f32, to_f32, y + j, float(x0) * d + m);
                h3_q_put(out_bf16, out_f32, to_f32, y + j + 16, float(x1) * d + m);
            } else {
                h3_q_put(out_bf16, out_f32, to_f32, y + j, float(x0 - 16) * d);
                h3_q_put(out_bf16, out_f32, to_f32, y + j + 16, float(x1 - 16) * d);
            }
        }
        break;
    }
    case 8: { // Q8_0
        device const uchar *b = raw + block * 34;
        float d = h3_q_half(b);
        device const char *qs = (device const char *)(b + 2);
        ulong y = start;
        for (int j = 0; j < 32; j++)
            h3_q_put(out_bf16, out_f32, to_f32, y + j, float(qs[j]) * d);
        break;
    }
    case 9: { // Q2_K
        device const uchar *b = raw + block * 84;
        device const uchar *q = b + 16;
        float d = h3_q_half(b + 80), dmin = h3_q_half(b + 82);
        ulong y = start;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint sc = b[is++];
                float dl = d * float(sc & 0xF), ml = dmin * float(sc >> 4);
                for (int l = 0; l < 16; l++)
                    h3_q_put(out_bf16, out_f32, to_f32, y++, dl * float((q[l] >> shift) & 3) - ml);
                sc = b[is++];
                dl = d * float(sc & 0xF);
                ml = dmin * float(sc >> 4);
                for (int l = 0; l < 16; l++)
                    h3_q_put(out_bf16, out_f32, to_f32, y++, dl * float((q[l + 16] >> shift) & 3) - ml);
                shift += 2;
            }
            q += 32;
        }
        break;
    }
    case 10: { // Q3_K
        device const uchar *b = raw + block * 110;
        device const uchar *hm = b;
        device const uchar *q = b + 32;
        float d_all = h3_q_half(b + 108);
        uint aux0 = h3_q_u32(b + 96), aux1 = h3_q_u32(b + 100), tmp = h3_q_u32(b + 104);
        const uint kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
        uint a[4];
        a[2] = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        a[3] = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        a[0] = (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
        a[1] = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
        ulong y = start;
        int is = 0;
        uint m = 1;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int s = int(char((a[is / 4] >> (8 * (is % 4))) & 0xFF)); is++;
                float dl = d_all * float(s - 32);
                for (int l = 0; l < 16; l++)
                    h3_q_put(out_bf16, out_f32, to_f32, y++,
                             dl * float(int((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
                s = int(char((a[is / 4] >> (8 * (is % 4))) & 0xFF)); is++;
                dl = d_all * float(s - 32);
                for (int l = 0; l < 16; l++)
                    h3_q_put(out_bf16, out_f32, to_f32, y++,
                             dl * float(int((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4)));
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
        break;
    }
    case 11: case 12: { // Q4_K, Q5_K
        bool five = args.quant == 12;
        device const uchar *b = raw + block * (five ? 176 : 144);
        float d = h3_q_half(b), dmin = h3_q_half(b + 2);
        device const uchar *scales = b + 4;
        device const uchar *qh = b + 16;
        device const uchar *q = b + (five ? 48 : 16);
        ulong y = start;
        int is = 0;
        uint u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint sc, mn;
            h3_q_scale_min_k4(is + 0, scales, sc, mn);
            float d1 = d * float(sc), m1 = dmin * float(mn);
            h3_q_scale_min_k4(is + 1, scales, sc, mn);
            float d2 = d * float(sc), m2 = dmin * float(mn);
            for (int l = 0; l < 32; l++) {
                uint v = (q[l] & 0xF) + (five && (qh[l] & u1) ? 16 : 0);
                h3_q_put(out_bf16, out_f32, to_f32, y + l, d1 * float(v) - m1);
            }
            for (int l = 0; l < 32; l++) {
                uint v = (q[l] >> 4) + (five && (qh[l] & u2) ? 16 : 0);
                h3_q_put(out_bf16, out_f32, to_f32, y + 32 + l, d2 * float(v) - m2);
            }
            y += 64;
            q += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
        break;
    }
    case 13: { // Q6_K
        device const uchar *b = raw + block * 210;
        device const uchar *ql = b;
        device const uchar *qh = b + 128;
        device const char *sc = (device const char *)(b + 192);
        float d = h3_q_half(b + 208);
        ulong y = start;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                h3_q_put(out_bf16, out_f32, to_f32, y + l, d * float(sc[is + 0]) * float(q1));
                h3_q_put(out_bf16, out_f32, to_f32, y + l + 32, d * float(sc[is + 2]) * float(q2));
                h3_q_put(out_bf16, out_f32, to_f32, y + l + 64, d * float(sc[is + 4]) * float(q3));
                h3_q_put(out_bf16, out_f32, to_f32, y + l + 96, d * float(sc[is + 6]) * float(q4));
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
        break;
    }
    default:
        break;
    }
}

/* ------------------------------------------------------------------ */
/* One-step VSR (UPSCALE_PLAN §6.5): channels-last F32 helpers for the
 * ReAE autoencoder and its pixel I/O. Tensors are [B, T, H, W, C]. */

struct vsr_pad_args {
    uint batch, depth, height, width, channels;
    uint depth_front, pad_h, pad_w;
    uint has_previous;   /* the depth_front frames come from `previous` */
};

/* Zero padding in space; the depth_front frames in front are copied from
 * `previous` ([B, depth_front, H, W, C], also spatially padded) when
 * has_previous, else zero. Grid (C, out_w, B * out_depth * out_h). */
kernel void h3_vsr_pad_f32(device const float *input [[buffer(0)]],
                           device float *output [[buffer(1)]],
                           constant vsr_pad_args &args [[buffer(2)]],
                           device const float *previous [[buffer(3)]],
                           uint3 gid [[thread_position_in_grid]]) {
    uint channel = gid.x, out_x = gid.y;
    uint out_h = args.height + 2 * args.pad_h;
    uint out_w = args.width + 2 * args.pad_w;
    uint out_depth = args.depth + args.depth_front;
    uint plane = gid.z;
    if (channel >= args.channels || out_x >= out_w ||
        plane >= args.batch * out_depth * out_h) return;
    uint out_y = plane % out_h;
    uint temporal = plane / out_h;
    uint out_t = temporal % out_depth;
    uint batch = temporal / out_depth;
    size_t destination = ((((size_t)batch * out_depth + out_t) * out_h + out_y) * out_w +
                          out_x) * args.channels + channel;
    int y = int(out_y) - int(args.pad_h), x = int(out_x) - int(args.pad_w);
    if (y < 0 || x < 0 || y >= int(args.height) || x >= int(args.width)) {
        output[destination] = 0.0f;
        return;
    }
    if (out_t < args.depth_front) {
        if (!args.has_previous) {
            output[destination] = 0.0f;
            return;
        }
        size_t source = ((((size_t)batch * args.depth_front + out_t) * args.height + uint(y)) *
                         args.width + uint(x)) * args.channels + channel;
        output[destination] = previous[source];
        return;
    }
    uint t = out_t - args.depth_front;
    size_t source = ((((size_t)batch * args.depth + t) * args.height + uint(y)) * args.width +
                     uint(x)) * args.channels + channel;
    output[destination] = input[source];
}

struct vsr_shape_args {
    uint batch, depth, height, width, channels;
};

/* Nearest 2x in space: [B,T,H,W,C] -> [B,T,2H,2W,C]. Grid (C, 2W, B*T*2H). */
kernel void h3_vsr_upsample2_f32(device const float *input [[buffer(0)]],
                                 device float *output [[buffer(1)]],
                                 constant vsr_shape_args &args [[buffer(2)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    uint channel = gid.x, out_x = gid.y, plane = gid.z;
    uint out_h = args.height * 2, out_w = args.width * 2;
    if (channel >= args.channels || out_x >= out_w || plane >= args.batch * args.depth * out_h) return;
    uint out_y = plane % out_h;
    uint frame = plane / out_h;   /* batch * depth + t */
    size_t source = (((size_t)frame * args.height + out_y / 2) * args.width + out_x / 2) *
                    args.channels + channel;
    size_t destination = (((size_t)frame * out_h + out_y) * out_w + out_x) * args.channels + channel;
    output[destination] = input[source];
}

/* pixel_unshuffle(2) on channels-last: [B,T,H,W,C] -> [B,T,H/2,W/2,4C] with
 * output channel (c*2 + ph)*2 + pw (PyTorch's order). Grid (4C, W/2, B*T*H/2). */
kernel void h3_vsr_space_to_depth_f32(device const float *input [[buffer(0)]],
                                      device float *output [[buffer(1)]],
                                      constant vsr_shape_args &args [[buffer(2)]],
                                      uint3 gid [[thread_position_in_grid]]) {
    uint out_c = gid.x, out_x = gid.y, plane = gid.z;
    uint out_h = args.height / 2, out_w = args.width / 2, out_channels = args.channels * 4;
    if (out_c >= out_channels || out_x >= out_w || plane >= args.batch * args.depth * out_h) return;
    uint out_y = plane % out_h;
    uint frame = plane / out_h;
    uint c = out_c / 4, ph = (out_c / 2) % 2, pw = out_c % 2;
    size_t source = (((size_t)frame * args.height + out_y * 2 + ph) * args.width + out_x * 2 + pw) *
                    args.channels + c;
    size_t destination = (((size_t)frame * out_h + out_y) * out_w + out_x) * out_channels + out_c;
    output[destination] = input[source];
}

/* pixel_shuffle(2): [B,T,H,W,4C] -> [B,T,2H,2W,C]; `channels` is C (the
 * output count). Grid (C, 2W, B*T*2H). */
kernel void h3_vsr_depth_to_space_f32(device const float *input [[buffer(0)]],
                                      device float *output [[buffer(1)]],
                                      constant vsr_shape_args &args [[buffer(2)]],
                                      uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, out_x = gid.y, plane = gid.z;
    uint out_h = args.height * 2, out_w = args.width * 2, in_channels = args.channels * 4;
    if (c >= args.channels || out_x >= out_w || plane >= args.batch * args.depth * out_h) return;
    uint out_y = plane % out_h;
    uint frame = plane / out_h;
    uint in_c = (c * 2 + (out_y % 2)) * 2 + (out_x % 2);
    size_t source = (((size_t)frame * args.height + out_y / 2) * args.width + out_x / 2) *
                    in_channels + in_c;
    size_t destination = (((size_t)frame * out_h + out_y) * out_w + out_x) * args.channels + c;
    output[destination] = input[source];
}

/* TGrow(2) tail: [B,T,H,W,2C] -> [B,2T,H,W,C], frame 2t+k takes channels
 * k*C..k*C+C-1 of frame t. `channels` is C. Grid (C, W, B*2T*H). */
kernel void h3_vsr_split_frames_f32(device const float *input [[buffer(0)]],
                                    device float *output [[buffer(1)]],
                                    constant vsr_shape_args &args [[buffer(2)]],
                                    uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, x = gid.y, plane = gid.z;
    uint out_depth = args.depth * 2;
    if (c >= args.channels || x >= args.width || plane >= args.batch * out_depth * args.height) return;
    uint y = plane % args.height;
    uint temporal = plane / args.height;
    uint out_t = temporal % out_depth;
    uint batch = temporal / out_depth;
    uint t = out_t / 2, k = out_t % 2;
    size_t source = ((((size_t)batch * args.depth + t) * args.height + y) * args.width + x) *
                    (args.channels * 2) + k * args.channels + c;
    size_t destination = ((((size_t)batch * out_depth + out_t) * args.height + y) * args.width + x) *
                         args.channels + c;
    output[destination] = input[source];
}

struct vsr_count_args { uint elements; };

/* The ReAE decoder's Clamp: tanh(x / 3) * 3. */
kernel void h3_vsr_clamp_tanh_f32(device const float *input [[buffer(0)]],
                                  device float *output [[buffer(1)]],
                                  constant vsr_count_args &args [[buffer(2)]],
                                  uint index [[thread_position_in_grid]]) {
    /* tanh is +-1 to float precision beyond |x| = 20, and Metal's tanh can
     * return NaN for large arguments, so the argument is clamped. */
    if (index < args.elements) output[index] = tanh(clamp(input[index] / 3.0f, -20.0f, 20.0f)) * 3.0f;
}

struct vsr_pixels_in_args {
    uint frames, in_w, in_h, in_stride;   /* stride in bytes */
    uint bgra;                            /* 1: BGRA8, 0: RGB24 */
    uint out_w, out_h;                    /* the bilinear target */
    uint pad_w, pad_h;                    /* zero-filled columns/rows after it */
};

/* 8-bit frames -> F32 RGB [T, out_h + pad_h, out_w + pad_w, 3] in [0, 1]:
 * torch bilinear (align_corners=False, no antialias) from in_w x in_h to
 * out_w x out_h, zero beyond. Grid (out_w + pad_w, out_h + pad_h, T). */
kernel void h3_vsr_pixels_to_f32(device const uchar *input [[buffer(0)]],
                                 device float *output [[buffer(1)]],
                                 constant vsr_pixels_in_args &args [[buffer(2)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    uint x = gid.x, y = gid.y, t = gid.z;
    uint full_w = args.out_w + args.pad_w, full_h = args.out_h + args.pad_h;
    if (x >= full_w || y >= full_h || t >= args.frames) return;
    size_t destination = (((size_t)t * full_h + y) * full_w + x) * 3;
    if (x >= args.out_w || y >= args.out_h) {
        output[destination] = output[destination + 1] = output[destination + 2] = 0.0f;
        return;
    }
    float sx = (float(x) + 0.5f) * float(args.in_w) / float(args.out_w) - 0.5f;
    float sy = (float(y) + 0.5f) * float(args.in_h) / float(args.out_h) - 0.5f;
    sx = max(sx, 0.0f);
    sy = max(sy, 0.0f);
    uint x0 = min(uint(sx), args.in_w - 1), y0 = min(uint(sy), args.in_h - 1);
    uint x1 = min(x0 + 1, args.in_w - 1), y1 = min(y0 + 1, args.in_h - 1);
    float fx = sx - float(x0), fy = sy - float(y0);
    uint bpp = args.bgra ? 4 : 3;
    device const uchar *frame = input + (size_t)t * args.in_h * args.in_stride;
    for (uint c = 0; c < 3; c++) {
        uint channel = args.bgra ? 2 - c : c;   /* BGRA holds B, G, R */
        float p00 = frame[(size_t)y0 * args.in_stride + x0 * bpp + channel];
        float p01 = frame[(size_t)y0 * args.in_stride + x1 * bpp + channel];
        float p10 = frame[(size_t)y1 * args.in_stride + x0 * bpp + channel];
        float p11 = frame[(size_t)y1 * args.in_stride + x1 * bpp + channel];
        float top = p00 + (p01 - p00) * fx, bottom = p10 + (p11 - p10) * fx;
        output[destination + c] = (top + (bottom - top) * fy) / 255.0f;
    }
}

struct vsr_pixels_out_args {
    uint frames, full_w, full_h;      /* the padded F32 frame */
    uint crop_w, crop_h, out_stride;  /* the 8-bit frame, stride in bytes */
    uint bgra;
    uint round_mode;                  /* 0: truncate (reference), 1: round */
};

/* F32 RGB [T, full_h, full_w, 3] -> 8-bit frames cropped to crop_w x crop_h.
 * Grid (crop_w, crop_h, T). */
kernel void h3_vsr_f32_to_pixels(device const float *input [[buffer(0)]],
                                 device uchar *output [[buffer(1)]],
                                 constant vsr_pixels_out_args &args [[buffer(2)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    uint x = gid.x, y = gid.y, t = gid.z;
    if (x >= args.crop_w || y >= args.crop_h || t >= args.frames) return;
    size_t source = (((size_t)t * args.full_h + y) * args.full_w + x) * 3;
    uint bpp = args.bgra ? 4 : 3;
    device uchar *pixel = output + (size_t)t * args.crop_h * args.out_stride +
                          (size_t)y * args.out_stride + x * bpp;
    for (uint c = 0; c < 3; c++) {
        float v = clamp(input[source + c], 0.0f, 1.0f) * 255.0f;
        uint value = args.round_mode ? uint(v + 0.5f) : uint(v);
        pixel[args.bgra ? 2 - c : c] = uchar(min(value, 255u));
    }
    if (args.bgra) pixel[3] = 255;
}

/* ------------------------------------------------------------------ */
/* One-step VSR: the Wan DiT block on BF16 activations with F32 math and
 * F32 constants (UPSCALE_PLAN §6.5). */

struct vsr_ln_args {
    uint rows, width;
    float epsilon;
    uint has_scale, has_shift, has_weight, has_bias;
};

/* LayerNorm (no affine) then optionally * weight + bias (norm2) and / or
 * * (1 + scale) + shift (the modulated norm1 / norm3 / norm_out). */
kernel void h3_vsr_ln_mod_bf16(device const ushort *input [[buffer(0)]],
                               device ushort *output [[buffer(1)]],
                               device const float *scale [[buffer(2)]],
                               device const float *shift [[buffer(3)]],
                               device const float *weight [[buffer(4)]],
                               device const float *bias [[buffer(5)]],
                               constant vsr_ln_args &args [[buffer(6)]],
                               uint3 group [[threadgroup_position_in_grid]],
                               uint3 thread_position [[thread_position_in_threadgroup]],
                               uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x, tid = thread_position.x, threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    device const ushort *x = input + (size_t)row * args.width;
    float local_sum = 0.0f;
    for (uint column = tid; column < args.width; column += threads)
        local_sum += h3_bf16_to_f32(x[column]);
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = reductions[0] / float(args.width);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_square = 0.0f;
    for (uint column = tid; column < args.width; column += threads) {
        float centered = h3_bf16_to_f32(x[column]) - mean;
        local_square = fma(centered, centered, local_square);
    }
    reductions[tid] = local_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    device ushort *y = output + (size_t)row * args.width;
    for (uint column = tid; column < args.width; column += threads) {
        float value = (h3_bf16_to_f32(x[column]) - mean) * inverse;
        if (args.has_weight) value *= weight[column];
        if (args.has_bias) value += bias[column];
        if (args.has_scale) value *= 1.0f + scale[column];
        if (args.has_shift) value += shift[column];
        y[column] = h3_f32_to_bf16(value);
    }
}

struct vsr_qkv_args {
    uint rows, dim, heads, head_dim, segments;   /* segments: 3 for qkv, 1 for q alone */
    uint has_bias, norm_q, norm_k, rope;
    float epsilon;
};

/* In place on [rows, segments * dim]: add the F32 bias, RMSNorm across all
 * heads on segment 0 (q) and 1 (k) with their weights, then Wan's
 * interleaved RoPE on q and k: pair p of a head rotates (2p, 2p+1) by the
 * token's (cos, sin)[p]. One threadgroup per row. */
kernel void h3_vsr_qkv_prepare_bf16(device ushort *qkv [[buffer(0)]],
                                    device const float *bias [[buffer(1)]],
                                    device const float *norm_q_weight [[buffer(2)]],
                                    device const float *norm_k_weight [[buffer(3)]],
                                    device const float *rope_cos [[buffer(4)]],
                                    device const float *rope_sin [[buffer(5)]],
                                    constant vsr_qkv_args &args [[buffer(6)]],
                                    uint3 group [[threadgroup_position_in_grid]],
                                    uint3 thread_position [[thread_position_in_threadgroup]],
                                    uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x, tid = thread_position.x, threads = threadgroup_size.x;
    if (row >= args.rows) return;
    uint width = args.segments * args.dim;
    device ushort *x = qkv + (size_t)row * width;
    threadgroup float sum_q[256], sum_k[256];
    float local_q = 0.0f, local_k = 0.0f;
    for (uint column = tid; column < width; column += threads) {
        float v = h3_bf16_to_f32(x[column]) + (args.has_bias ? bias[column] : 0.0f);
        uint segment = column / args.dim;
        if (segment == 0) local_q = fma(v, v, local_q);
        else if (segment == 1) local_k = fma(v, v, local_k);
    }
    sum_q[tid] = local_q;
    sum_k[tid] = local_k;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) { sum_q[tid] += sum_q[tid + stride]; sum_k[tid] += sum_k[tid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_q = rsqrt(sum_q[0] / float(args.dim) + args.epsilon);
    float inverse_k = rsqrt(sum_k[0] / float(args.dim) + args.epsilon);
    uint pairs = args.head_dim / 2;
    device const float *c = rope_cos + (size_t)row * pairs;
    device const float *s = rope_sin + (size_t)row * pairs;
    for (uint pair = tid; pair < width / 2; pair += threads) {
        uint column = pair * 2;
        uint segment = column / args.dim;
        float v0 = h3_bf16_to_f32(x[column]) + (args.has_bias ? bias[column] : 0.0f);
        float v1 = h3_bf16_to_f32(x[column + 1]) + (args.has_bias ? bias[column + 1] : 0.0f);
        if (segment == 0 && args.norm_q) {
            v0 *= inverse_q * norm_q_weight[column];
            v1 *= inverse_q * norm_q_weight[column + 1];
        } else if (segment == 1 && args.norm_k) {
            v0 *= inverse_k * norm_k_weight[column - args.dim];
            v1 *= inverse_k * norm_k_weight[column + 1 - args.dim];
        }
        if (args.rope && segment < 2) {
            uint p = (column % args.head_dim) / 2;
            float cv = c[p], sv = s[p];
            float r0 = v0 * cv - v1 * sv, r1 = v0 * sv + v1 * cv;
            v0 = r0;
            v1 = r1;
        }
        x[column] = h3_f32_to_bf16(v0);
        x[column + 1] = h3_f32_to_bf16(v1);
    }
}

struct vsr_rows_args { uint rows, width, in_stride, col_offset; };

/* out[r, 0..width) = in[index[r], col_offset .. col_offset + width): the
 * window gather (col_offset selects q, k or v in a fused buffer) and, with
 * the owner map as index, the scatter back. Grid (width, rows). */
kernel void h3_vsr_gather_rows_bf16(device const ushort *input [[buffer(0)]],
                                    device const int *index [[buffer(1)]],
                                    device ushort *output [[buffer(2)]],
                                    constant vsr_rows_args &args [[buffer(3)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x, row = gid.y;
    if (column >= args.width || row >= args.rows) return;
    output[(size_t)row * args.width + column] =
        input[(size_t)index[row] * args.in_stride + args.col_offset + column];
}

/* out[index[r], 0..width) = in[r, 0..width): the owner-only attention's
 * write-back of each window's owned rows to their tokens. Grid (width, rows). */
kernel void h3_vsr_scatter_rows_bf16(device const ushort *input [[buffer(0)]],
                                     device const uint *index [[buffer(1)]],
                                     device ushort *output [[buffer(2)]],
                                     constant vsr_rows_args &args [[buffer(3)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x, row = gid.y;
    if (column >= args.width || row >= args.rows) return;
    output[(size_t)index[row] * args.width + column] = input[(size_t)row * args.width + column];
}

/* y = gelu_tanh(x + bias) quantized per row to int8 with its F32 scale,
 * in one pass over the row (the values stay in registers between the max
 * and the quantization): the FFN's activation feeding the int8 fc2.
 * One 256-thread threadgroup per row; columns a multiple of 4, at most
 * 8192; rows past `rows` are zeroed with scale 1 like the plain quantizer. */
struct vsr_gelu_quant_args { uint rows, columns; };

kernel void h3_vsr_gelu_quantize_int8_rows(device const bfloat *input [[buffer(0)]],
                                           device const float *bias [[buffer(1)]],
                                           device int8_t *output [[buffer(2)]],
                                           device float *scales [[buffer(3)]],
                                           constant vsr_gelu_quant_args &args [[buffer(4)]],
                                           uint tid [[thread_index_in_threadgroup]],
                                           ushort simdgroup [[simdgroup_index_in_threadgroup]],
                                           ushort lane [[thread_index_in_simdgroup]],
                                           uint row [[threadgroup_position_in_grid]]) {
    uint vectors = args.columns / 4;
    uint vector_base = row * vectors;
    device char4 *output4 = reinterpret_cast<device char4 *>(output);
    if (row >= args.rows) {
        for (uint column = tid; column < vectors; column += 256) output4[vector_base + column] = char4(0);
        if (tid == 0) scales[row] = 1.0f;
        return;
    }
    device const bfloat4 *input4 = reinterpret_cast<device const bfloat4 *>(input);
    device const float4 *bias4 = reinterpret_cast<device const float4 *>(bias);
    threadgroup float scratch[8];
    float4 kept[8];
    float local_max = 0.0f;
    const float k = 0.7978845608028654f;
    uint n = 0;
    for (uint column = tid; column < vectors && n < 8; column += 256, n++) {
        float4 v = float4(input4[vector_base + column]) + bias4[column];
        float4 t = clamp(k * (v + 0.044715f * v * v * v), -20.0f, 20.0f);
        float4 g = 0.5f * v * (1.0f + tanh(t));
        kept[n] = g;
        local_max = max(local_max, max(max(fabs(g.x), fabs(g.y)), max(fabs(g.z), fabs(g.w))));
    }
    float max_abs = h3_int8_reduce_max(local_max, scratch, simdgroup, lane);
    float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
    float inverse = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
    if (tid == 0) scales[row] = scale;
    n = 0;
    for (uint column = tid; column < vectors && n < 8; column += 256, n++) {
        int4 quantized = int4(rint(kept[n] * inverse));
        output4[vector_base + column] = char4(clamp(quantized, int4(-127), int4(127)));
    }
}

struct vsr_bias_gate_args { uint rows, width, has_residual, has_bias, has_gate; };

/* out = residual + (y + bias) * gate, with the optional parts dropped;
 * out may alias residual. Grid (width, rows). */
kernel void h3_vsr_bias_gate_add_bf16(device const ushort *y [[buffer(0)]],
                                      device const float *bias [[buffer(1)]],
                                      device const float *gate [[buffer(2)]],
                                      device const ushort *residual [[buffer(3)]],
                                      device ushort *output [[buffer(4)]],
                                      constant vsr_bias_gate_args &args [[buffer(5)]],
                                      uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x, row = gid.y;
    if (column >= args.width || row >= args.rows) return;
    size_t at = (size_t)row * args.width + column;
    float value = h3_bf16_to_f32(y[at]);
    if (args.has_bias) value += bias[column];
    if (args.has_gate) value *= gate[column];
    if (args.has_residual) value += h3_bf16_to_f32(residual[at]);
    output[at] = h3_f32_to_bf16(value);
}

/* x = gelu_tanh(x + bias) in place. Grid (width, rows). */
kernel void h3_vsr_gelu_bias_bf16(device ushort *x [[buffer(0)]],
                                  device const float *bias [[buffer(1)]],
                                  constant vsr_rows_args &args [[buffer(2)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x, row = gid.y;
    if (column >= args.width || row >= args.rows) return;
    size_t at = (size_t)row * args.width + column;
    float v = h3_bf16_to_f32(x[at]) + bias[column];
    const float k = 0.7978845608028654f;
    /* Metal's tanh overflows to NaN past |x| ~ 44 (exp(2x) / exp(2x)); it is
     * +-1 to float precision long before that, so the argument is clamped. */
    float t = clamp(k * (v + 0.044715f * v * v * v), -20.0f, 20.0f);
    float g = 0.5f * v * (1.0f + tanh(t));
    x[at] = h3_f32_to_bf16(g);
}

struct vsr_patch_args { uint frames, height, width, channels; };   /* latent cells */

/* F32 latent [T, h, w, C] -> BF16 tokens [T*(h/2)*(w/2), 4C] in the patch
 * embedding's (c, ph, pw) order. Grid (4C, w/2, T*h/2). */
kernel void h3_vsr_patchify_bf16(device const float *latent [[buffer(0)]],
                                 device ushort *tokens [[buffer(1)]],
                                 constant vsr_patch_args &args [[buffer(2)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x, ww = gid.y, plane = gid.z;
    uint h2 = args.height / 2, w2 = args.width / 2, c4 = args.channels * 4;
    if (k >= c4 || ww >= w2 || plane >= args.frames * h2) return;
    uint hh = plane % h2, t = plane / h2;
    uint c = k / 4, ph = (k / 2) % 2, pw = k % 2;
    size_t source = (((size_t)t * args.height + hh * 2 + ph) * args.width + ww * 2 + pw) *
                    args.channels + c;
    size_t token = ((size_t)t * h2 + hh) * w2 + ww;
    tokens[token * c4 + k] = h3_f32_to_bf16(latent[source]);
}

/* z_hq[T, h, w, C] = z_lq - (v + bias) where v is BF16 [tokens, 4C] in the
 * unpatchify (ph, pw, c) order. Grid (C, w, T*h). */
kernel void h3_vsr_unpatchify_sub_f32(device const float *z_lq [[buffer(0)]],
                                      device const ushort *v [[buffer(1)]],
                                      device const float *bias [[buffer(2)]],
                                      device float *z_hq [[buffer(3)]],
                                      constant vsr_patch_args &args [[buffer(4)]],
                                      uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, x = gid.y, plane = gid.z;
    if (c >= args.channels || x >= args.width || plane >= args.frames * args.height) return;
    uint y = plane % args.height, t = plane / args.height;
    uint h2 = args.height / 2, w2 = args.width / 2, c4 = args.channels * 4;
    size_t token = ((size_t)t * h2 + y / 2) * w2 + x / 2;
    uint k = ((y % 2) * 2 + (x % 2)) * args.channels + c;
    size_t at = (((size_t)t * args.height + y) * args.width + x) * args.channels + c;
    z_hq[at] = z_lq[at] - (h3_bf16_to_f32(v[token * c4 + k]) + bias[k]);
}

/* ------------------------------------------------------------------ */
/* VSR: 3x3 convolutions of the ReAE as tap-accumulated TensorOps matmuls
 * (M5). Activations live in a "padded grid": channels-last BF16 rows
 * [guard | T x (H+2) x (W+2) | guard], the spatial border rows zero, the
 * channel count padded to a multiple of 64. A 3x3 tap is then a constant
 * row offset, and out[r] = sum_taps x[r + off_tap] . W_tap^T (+ a second
 * input with its own weights for the MemBlock's cat[x, past]) + bias,
 * optionally + residual and ReLU. Rows past the grid and the border rows
 * come out as garbage and are zeroed by h3_vsr_zero_border_bf16. */

struct vsr_tapconv_args {
    uint rows;        /* grid rows: T * (H+2) * (W+2) */
    uint x_rows;      /* physical rows of each input and of the output */
    uint guard;       /* leading guard rows */
    uint in_ch, out_ch, taps, inputs, relu, has_bias, has_residual;
    int offsets[9];
};

kernel void h3_vsr_tapconv_bf16_nax(device bfloat *x0 [[buffer(0)]],
                                    device bfloat *x1 [[buffer(1)]],
                                    device bfloat *w0 [[buffer(2)]],
                                    device bfloat *w1 [[buffer(3)]],
                                    device const float *bias [[buffer(4)]],
                                    device const bfloat *residual [[buffer(5)]],
                                    device bfloat *output [[buffer(6)]],
                                    constant vsr_tapconv_args &args [[buffer(7)]],
                                    uint2 group [[threadgroup_position_in_grid]]) {
    constexpr int RT = 128, CT = 64, KT = 64;
    int row_start = (int)(group.x * RT), col_start = (int)(group.y * CT);
    auto xa = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        x0, dextents<int32_t, 2>((int)args.in_ch, (int)args.x_rows));
    auto xb = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        x1, dextents<int32_t, 2>((int)args.in_ch, (int)args.x_rows));
    auto wa = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        w0, dextents<int32_t, 2>((int)args.in_ch, (int)(args.taps * args.out_ch)));
    auto wb = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        w1, dextents<int32_t, 2>((int)args.in_ch, (int)(args.taps * args.out_ch)));
    constexpr auto descriptor = matmul2d_descriptor(
        RT, CT, KT, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> mm;
    auto first_a = xa.slice<KT, RT>(0, row_start);
    auto first_b = wa.slice<KT, CT>(0, col_start);
    auto accum = mm.template get_destination_cooperative_tensor<
        decltype(first_a), decltype(first_b), float>();
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++)
        if (accum.is_valid_element(element)) accum[element] = 0.0f;
    for (uint input = 0; input < args.inputs; input++) {
        for (uint t = 0; t < args.taps; t++) {
            int base = row_start + (int)args.guard + args.offsets[t];
            int col = (int)(t * args.out_ch) + col_start;
            for (int k = 0; k < (int)args.in_ch; k += KT) {
                if (input == 0) {
                    auto a = xa.slice<KT, RT>(k, base);
                    auto b = wa.slice<KT, CT>(k, col);
                    mm.run(a, b, accum);
                } else {
                    auto a = xb.slice<KT, RT>(k, base);
                    auto b = wb.slice<KT, CT>(k, col);
                    mm.run(a, b, accum);
                }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++) {
        if (!accum.is_valid_element(element)) continue;
        auto index = accum.get_multidimensional_index(element);
        int row = row_start + (int)index[1];
        int col = col_start + (int)index[0];
        if (row >= (int)args.rows) continue;
        float v = accum[element];
        if (args.has_bias) v += bias[col];
        size_t at = (size_t)(row + (int)args.guard) * args.out_ch + (size_t)col;
        if (args.has_residual) v += (float)residual[at];
        if (args.relu) v = max(v, 0.0f);
        output[at] = (bfloat)v;
    }
}

struct vsr_grid_args { uint frames, height, width, channels, padded_channels, guard; };

/* F32 [T, H, W, C] -> the padded grid (border rows and padded channels
 * zero). Grid (padded_channels, W+2, T*(H+2)). The guard rows are zeroed
 * separately (h3_vsr_zero_rows_bf16). */
kernel void h3_vsr_pack_grid_bf16(device const float *input [[buffer(0)]],
                                  device ushort *output [[buffer(1)]],
                                  constant vsr_grid_args &args [[buffer(2)]],
                                  uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, x = gid.y, ty = gid.z;
    uint hp = args.height + 2, wp = args.width + 2;
    if (c >= args.padded_channels || x >= wp || ty >= args.frames * hp) return;
    uint t = ty / hp, y = ty - t * hp;
    float v = 0.0f;
    if (c < args.channels && y >= 1 && y <= args.height && x >= 1 && x <= args.width)
        v = input[(((size_t)t * args.height + (y - 1)) * args.width + (x - 1)) * args.channels + c];
    size_t row = (size_t)args.guard + ((size_t)t * hp + y) * wp + x;
    output[row * args.padded_channels + c] = h3_f32_to_bf16(v);
}

/* The padded grid -> F32 [T, H, W, C]. Grid (C, W, T*H). */
kernel void h3_vsr_unpack_grid_f32(device const ushort *input [[buffer(0)]],
                                   device float *output [[buffer(1)]],
                                   constant vsr_grid_args &args [[buffer(2)]],
                                   uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, x = gid.y, ty = gid.z;
    if (c >= args.channels || x >= args.width || ty >= args.frames * args.height) return;
    uint t = ty / args.height, y = ty - t * args.height;
    uint hp = args.height + 2, wp = args.width + 2;
    size_t row = (size_t)args.guard + ((size_t)t * hp + (y + 1)) * wp + (x + 1);
    output[(((size_t)t * args.height + y) * args.width + x) * args.channels + c] =
        h3_bf16_to_f32(input[row * args.padded_channels + c]);
}

/* Zeroes the spatial border rows of every frame of a padded grid.
 * Grid (padded_channels, 2*(W+2) + 2*H, T). */
kernel void h3_vsr_zero_border_bf16(device ushort *x [[buffer(0)]],
                                    constant vsr_grid_args &args [[buffer(1)]],
                                    uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, i = gid.y, t = gid.z;
    uint hp = args.height + 2, wp = args.width + 2;
    if (c >= args.padded_channels || i >= 2 * wp + 2 * args.height || t >= args.frames) return;
    uint y, xx;
    if (i < wp) { y = 0; xx = i; }
    else if (i < 2 * wp) { y = hp - 1; xx = i - wp; }
    else { uint j = i - 2 * wp; y = 1 + j / 2; xx = (j & 1) ? wp - 1 : 0; }
    size_t row = (size_t)args.guard + ((size_t)t * hp + y) * wp + xx;
    x[row * args.padded_channels + c] = 0;
}

/* Zeroes rows [first, first + count) of a BF16 row-major buffer.
 * Grid (width, count). */
kernel void h3_vsr_zero_rows_bf16(device ushort *x [[buffer(0)]],
                                  constant vsr_rows_args &args [[buffer(1)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    x[((size_t)args.col_offset + gid.y) * args.width + gid.x] = 0;
}

/* ---- the rest of the ReAE on the padded grid ---------------------- */

/* Grid geometry shared by the kernels below: `channels` real channels in
 * `padded_channels`, frames x (height+2) x (width+2) rows after `guard`. */
struct vsr_grid2_args {
    uint frames, height, width, channels, padded_channels, guard;   /* output grid */
    uint in_height, in_width, in_channels, in_padded, in_guard;     /* input grid */
    uint stride_t, stride_y, stride_x, mode;
};

static inline size_t vsr_grid_row(uint guard, uint t, uint hp, uint wp, uint y, uint x) {
    return (size_t)guard + ((size_t)t * hp + y) * wp + x;
}

/* F32 frames [T, 2h, 2w, 3] -> grid [T, h, w, 12 of padded]: pixel_unshuffle(2),
 * channel c*4 + ph*2 + pw. Grid (padded, w+2, T*(h+2)); borders zero. */
kernel void h3_vsr_pack_grid_s2d_bf16(device const float *input [[buffer(0)]],
                                      device ushort *output [[buffer(1)]],
                                      constant vsr_grid2_args &args [[buffer(2)]],
                                      uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, x = gid.y, ty = gid.z;
    uint hp = args.height + 2, wp = args.width + 2;
    if (c >= args.padded_channels || x >= wp || ty >= args.frames * hp) return;
    uint t = ty / hp, y = ty - t * hp;
    float v = 0.0f;
    if (c < 12 && y >= 1 && y <= args.height && x >= 1 && x <= args.width) {
        uint pc = c / 4, ph = (c / 2) % 2, pw = c % 2;
        uint H = args.height * 2, W = args.width * 2;
        v = input[(((size_t)t * H + (y - 1) * 2 + ph) * W + (x - 1) * 2 + pw) * 3 + pc];
    }
    output[vsr_grid_row(args.guard, t, hp, wp, y, x) * args.padded_channels + c] = h3_f32_to_bf16(v);
}

/* Grid [T, h, w, 12 of padded] -> F32 frames [T, 2h, 2w, 3] clamped to
 * [0, 1]: pixel_shuffle(2). Grid (3, 2w, T*2h). */
kernel void h3_vsr_unpack_grid_d2s_f32(device const ushort *input [[buffer(0)]],
                                       device float *output [[buffer(1)]],
                                       constant vsr_grid2_args &args [[buffer(2)]],
                                       uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, ox = gid.y, toy = gid.z;
    uint H = args.in_height * 2, W = args.in_width * 2;
    if (c >= 3 || ox >= W || toy >= args.frames * H) return;
    uint t = toy / H, oy = toy - t * H;
    uint ic = (c * 2 + (oy % 2)) * 2 + (ox % 2);
    size_t row = vsr_grid_row(args.in_guard, t, args.in_height + 2, args.in_width + 2, oy / 2 + 1, ox / 2 + 1);
    float v = h3_bf16_to_f32(input[row * args.in_padded + ic]);
    output[(((size_t)t * H + oy) * W + ox) * 3 + c] = clamp(v, 0.0f, 1.0f);
}

/* Nearest 2x spatial upsample, grid -> grid (same channels).
 * Grid (padded, 2w+2, T*(2h+2)); borders zero. */
kernel void h3_vsr_grid_upsample2_bf16(device const ushort *input [[buffer(0)]],
                                       device ushort *output [[buffer(1)]],
                                       constant vsr_grid2_args &args [[buffer(2)]],
                                       uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, x = gid.y, ty = gid.z;
    uint hp = args.height + 2, wp = args.width + 2;
    if (c >= args.padded_channels || x >= wp || ty >= args.frames * hp) return;
    uint t = ty / hp, y = ty - t * hp;
    ushort v = 0;
    if (y >= 1 && y <= args.height && x >= 1 && x <= args.width) {
        size_t row = vsr_grid_row(args.in_guard, t, args.in_height + 2, args.in_width + 2, (y - 1) / 2 + 1, (x - 1) / 2 + 1);
        v = input[row * args.in_padded + c];
    }
    output[vsr_grid_row(args.guard, t, hp, wp, y, x) * args.padded_channels + c] = v;
}

/* TGrow(2) tail: grid [T, h, w, 2C of in_padded] -> grid [2T, h, w, C of
 * padded]; frame 2t+k takes channels k*C .. k*C+C-1 of frame t.
 * Grid (padded, w+2, 2T*(h+2)); borders and padded channels zero. */
kernel void h3_vsr_grid_split_frames_bf16(device const ushort *input [[buffer(0)]],
                                          device ushort *output [[buffer(1)]],
                                          constant vsr_grid2_args &args [[buffer(2)]],
                                          uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, x = gid.y, ty = gid.z;
    uint hp = args.height + 2, wp = args.width + 2;
    if (c >= args.padded_channels || x >= wp || ty >= args.frames * hp) return;
    uint t = ty / hp, y = ty - t * hp;
    ushort v = 0;
    if (c < args.channels && y >= 1 && y <= args.height && x >= 1 && x <= args.width) {
        size_t row = vsr_grid_row(args.in_guard, t / 2, hp, wp, y, x);
        v = input[row * args.in_padded + (t % 2) * args.channels + c];
    }
    output[vsr_grid_row(args.guard, t, hp, wp, y, x) * args.padded_channels + c] = v;
}

/* Subsampling, grid -> grid: out(t, y, x) = in(t*stride_t, y*stride_y,
 * x*stride_x) (the stride-2 conv's output, TPool's even frames).
 * Grid (padded, w+2, T*(h+2)); borders zero. */
kernel void h3_vsr_grid_select_bf16(device const ushort *input [[buffer(0)]],
                                    device ushort *output [[buffer(1)]],
                                    constant vsr_grid2_args &args [[buffer(2)]],
                                    uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, x = gid.y, ty = gid.z;
    uint hp = args.height + 2, wp = args.width + 2;
    if (c >= args.padded_channels || x >= wp || ty >= args.frames * hp) return;
    uint t = ty / hp, y = ty - t * hp;
    ushort v = 0;
    if (y >= 1 && y <= args.height && x >= 1 && x <= args.width) {
        size_t row = vsr_grid_row(args.in_guard, t * args.stride_t, args.in_height + 2, args.in_width + 2,
                                  (y - 1) * args.stride_y + 1, (x - 1) * args.stride_x + 1);
        v = input[row * args.in_padded + c];
    }
    output[vsr_grid_row(args.guard, t, hp, wp, y, x) * args.padded_channels + c] = v;
}

/* F32 [T, h, w, C] -> grid with tanh(x / 3) * 3 (the decoder's Clamp).
 * Grid (padded, w+2, T*(h+2)). */
kernel void h3_vsr_pack_grid_clamp_bf16(device const float *input [[buffer(0)]],
                                        device ushort *output [[buffer(1)]],
                                        constant vsr_grid_args &args [[buffer(2)]],
                                        uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, x = gid.y, ty = gid.z;
    uint hp = args.height + 2, wp = args.width + 2;
    if (c >= args.padded_channels || x >= wp || ty >= args.frames * hp) return;
    uint t = ty / hp, y = ty - t * hp;
    float v = 0.0f;
    if (c < args.channels && y >= 1 && y <= args.height && x >= 1 && x <= args.width) {
        v = input[(((size_t)t * args.height + (y - 1)) * args.width + (x - 1)) * args.channels + c];
        v = tanh(clamp(v / 3.0f, -20.0f, 20.0f)) * 3.0f;
    }
    output[vsr_grid_row(args.guard, t, hp, wp, y, x) * args.padded_channels + c] = h3_f32_to_bf16(v);
}

/* ---- VSR colour match: per-channel sums of an F32 [T, H, W, 3] image
 * over its real (unpadded) area, one partial per threadgroup, and the
 * per-channel offset that applies the match. ---- */
struct vsr_sum_args { uint frames, height, width, real_height, real_width, groups; };

kernel void h3_vsr_channel_sums_f32(device const float *input [[buffer(0)]],
                                    device float *partials [[buffer(1)]],
                                    constant vsr_sum_args &args [[buffer(2)]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint group [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[3][256];
    float3 sum = 0.0f;
    size_t pixels = (size_t)args.frames * args.real_height * args.real_width;
    for (size_t p = (size_t)group * 256 + tid; p < pixels; p += (size_t)args.groups * 256) {
        uint t = (uint)(p / ((size_t)args.real_height * args.real_width));
        uint rem = (uint)(p % ((size_t)args.real_height * args.real_width));
        uint y = rem / args.real_width, x = rem % args.real_width;
        size_t at = (((size_t)t * args.height + y) * args.width + x) * 3;
        sum += float3(input[at], input[at + 1], input[at + 2]);
    }
    scratch[0][tid] = sum.x; scratch[1][tid] = sum.y; scratch[2][tid] = sum.z;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride)
            for (uint c = 0; c < 3; c++) scratch[c][tid] += scratch[c][tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) for (uint c = 0; c < 3; c++) partials[(size_t)group * 3 + c] = scratch[c][0];
}

kernel void h3_vsr_add_channel_offsets_f32(device float *x [[buffer(0)]],
                                           constant float3 &offsets [[buffer(1)]],
                                           constant uint &pixels [[buffer(2)]],
                                           uint gid [[thread_position_in_grid]]) {
    if (gid >= pixels) return;
    size_t at = (size_t)gid * 3;
    x[at] += offsets.x; x[at + 1] += offsets.y; x[at + 2] += offsets.z;
}
