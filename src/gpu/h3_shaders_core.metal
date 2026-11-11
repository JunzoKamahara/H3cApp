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

