
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

/* HyperFlow factored attention (h3_hyperflow.h): Q and K come from their
 * own up-projections as token-major [sequence, heads * head_dim] buffers
 * (the layout the query/key outputs already have), so this is
 * h3_qkv_rope_bf16_coop's per-head RMS norm + RoPE with separate inputs and
 * the value left alone (its up-projection writes the value directly). */
kernel void h3_qk_rope_split_bf16_coop(
                             device const ushort *q_input [[buffer(0)]],
                             device const ushort *q_weight [[buffer(1)]],
                             device const ushort *k_weight [[buffer(2)]],
                             device const ushort *rope_cos [[buffer(3)]],
                             device const ushort *rope_sin [[buffer(4)]],
                             device ushort *query [[buffer(5)]],
                             device ushort *key [[buffer(6)]],
                             device const ushort *k_input [[buffer(7)]],
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
    uint q_base = row * inner + head * args.head_dim;
    uint k_base = q_base;

    threadgroup ushort q_values[HEADS_PER_GROUP * WIDTH];
    threadgroup ushort k_values[HEADS_PER_GROUP * WIDTH];
    uint cache_base = simdgroup * WIDTH;
    for (uint dimension = lane; dimension < WIDTH; dimension += 32) {
        q_values[cache_base + dimension] = q_input[q_base + dimension];
        k_values[cache_base + dimension] = k_input[k_base + dimension];
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
    }
}

/* SwiGLU from separate gate and up buffers (HyperFlow's factored FC1
 * writes the two halves on their own); output may be up itself. */
kernel void h3_swiglu_split_bf16(device const ushort *gate [[buffer(0)]],
                                 device const ushort *up [[buffer(1)]],
                                 device ushort *output [[buffer(2)]],
                                 constant uint &count [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float g = h3_bf16_to_f32(gate[gid]);
    float u = h3_bf16_to_f32(up[gid]);
    output[gid] = h3_f32_to_bf16(g / (1.0f + exp(-g)) * u);
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

/* h3_gqa_causal_bf16 with each score row in device memory instead of
 * threadgroup memory, for a batch of query rows starting at row_offset.
 * The arithmetic and its order are the same, so the output is bit-for-bit
 * the same; what changes is occupancy. A threadgroup holding a whole score
 * row of a long sequence (25 KiB at 6,400 tokens) leaves room for one per
 * GPU core, and the serial value sums below then wait on memory with
 * nothing to hide the latency. */
struct gqa_rows_args {
    uint sequence, query_heads, kv_heads, head_dim;
    float scale;
    uint row_offset, score_stride;
};

kernel void h3_gqa_causal_rows_bf16(
        device const ushort *query [[buffer(0)]],
        device const ushort *key [[buffer(1)]],
        device const ushort *value [[buffer(2)]],
        device ushort *output [[buffer(3)]],
        constant gqa_rows_args &args [[buffer(4)]],
        device float *score_rows [[buffer(5)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint3 thread_position [[thread_position_in_threadgroup]],
        uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    uint query_row = args.row_offset + group.x;
    uint query_head = group.y;
    if (query_row >= args.sequence || query_head >= args.query_heads) return;
    uint kv_head = query_head / (args.query_heads / args.kv_heads);
    uint q_base = (query_row * args.query_heads + query_head) * args.head_dim;
    uint key_count = query_row + 1;
    device float *scores = score_rows +
        ((ulong)group.x * args.query_heads + query_head) * args.score_stride;
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
    /* The value sums read score rows other threads wrote; in device memory
     * that needs a device barrier, not only the threadgroup ones above. */
    threadgroup_barrier(mem_flags::mem_device);
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

/* h3_gqa_causal_rows_bf16 for all the query heads that share one KV head
 * at once (exactly eight of them, so the per-head loops unroll into
 * registers): thread x does, for each of those heads, exactly what thread x of
 * that head does in h3_gqa_causal_bf16 - the same keys in the same order,
 * the same fma chains, the same 128-wide reductions - so the output is bit-
 * for-bit the same. A key or value element is then loaded and converted
 * once for up to eight heads instead of once per head, and the eight
 * independent chains in each thread hide each other's latency. */
#define H3_GQA_GROUP 8 /* Qwen3-VL-32B: 64 query heads over 8 KV heads */
kernel void h3_gqa_causal_group_bf16(
        device const ushort *query [[buffer(0)]],
        device const ushort *key [[buffer(1)]],
        device const ushort *value [[buffer(2)]],
        device ushort *output [[buffer(3)]],
        constant gqa_rows_args &args [[buffer(4)]],
        device float *score_rows [[buffer(5)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint3 thread_position [[thread_position_in_threadgroup]],
        uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    uint groups = args.query_heads / args.kv_heads;
    uint query_row = args.row_offset + group.x;
    uint kv_head = group.y;
    if (query_row >= args.sequence || groups != H3_GQA_GROUP) return;
    uint first_head = kv_head * groups;
    uint key_count = query_row + 1;
    device float *scores = score_rows +
        ((ulong)group.x * args.query_heads + first_head) * args.score_stride;
    threadgroup float reductions[H3_GQA_GROUP][128];
    threadgroup float shared_query[H3_GQA_GROUP][128];

    for (uint m = 0; m < H3_GQA_GROUP; m++) {
        uint q_base = (query_row * args.query_heads + first_head + m) * args.head_dim;
        for (uint d = tid; d < args.head_dim; d += threads)
            shared_query[m][d] = h3_bf16_to_f32(h3_f32_to_bf16(
                h3_bf16_to_f32(query[q_base + d]) * args.scale));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max[H3_GQA_GROUP];
    for (uint m = 0; m < H3_GQA_GROUP; m++) local_max[m] = -INFINITY;
    for (uint key_row = tid; key_row < key_count; key_row += threads) {
        uint k_base = (key_row * args.kv_heads + kv_head) * args.head_dim;
        float dot[H3_GQA_GROUP];
        for (uint m = 0; m < H3_GQA_GROUP; m++) dot[m] = 0.0f;
        for (uint d = 0; d < args.head_dim; d++) {
            float k = h3_bf16_to_f32(key[k_base + d]);
            for (uint m = 0; m < H3_GQA_GROUP; m++)
                dot[m] = fma(shared_query[m][d], k, dot[m]);
        }
        for (uint m = 0; m < H3_GQA_GROUP; m++) {
            scores[(ulong)m * args.score_stride + key_row] = dot[m];
            local_max[m] = max(local_max[m], dot[m]);
        }
    }
    for (uint m = 0; m < H3_GQA_GROUP; m++) reductions[m][tid] = local_max[m];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride)
            for (uint m = 0; m < H3_GQA_GROUP; m++)
                reductions[m][tid] = max(reductions[m][tid], reductions[m][tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maximum[H3_GQA_GROUP];
    for (uint m = 0; m < H3_GQA_GROUP; m++) maximum[m] = reductions[m][0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum[H3_GQA_GROUP];
    for (uint m = 0; m < H3_GQA_GROUP; m++) local_sum[m] = 0.0f;
    for (uint key_row = tid; key_row < key_count; key_row += threads)
        for (uint m = 0; m < H3_GQA_GROUP; m++) {
            device float *score = scores + (ulong)m * args.score_stride + key_row;
            float probability = exp(*score - maximum[m]);
            *score = probability;
            local_sum[m] += probability;
        }
    for (uint m = 0; m < H3_GQA_GROUP; m++) reductions[m][tid] = local_sum[m];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride)
            for (uint m = 0; m < H3_GQA_GROUP; m++)
                reductions[m][tid] += reductions[m][tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_sum[H3_GQA_GROUP];
    for (uint m = 0; m < H3_GQA_GROUP; m++) inverse_sum[m] = 1.0f / reductions[m][0];
    /* The value sums read score rows other threads wrote; in device memory
     * that needs a device barrier, not only the threadgroup ones above. */
    threadgroup_barrier(mem_flags::mem_device);
    for (uint d = tid; d < args.head_dim; d += threads) {
        float sum[H3_GQA_GROUP];
        for (uint m = 0; m < H3_GQA_GROUP; m++) sum[m] = 0.0f;
        for (uint key_row = 0; key_row < key_count; key_row++) {
            uint v_index = (key_row * args.kv_heads + kv_head) * args.head_dim + d;
            float v = h3_bf16_to_f32(value[v_index]);
            for (uint m = 0; m < H3_GQA_GROUP; m++)
                sum[m] = fma(scores[(ulong)m * args.score_stride + key_row] *
                             inverse_sum[m], v, sum[m]);
        }
        for (uint m = 0; m < H3_GQA_GROUP; m++)
            output[(query_row * args.query_heads + first_head + m) *
                   args.head_dim + d] = h3_f32_to_bf16(sum[m]);
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

// --- ComfyUI int8 (+ ConvRot) weights (h3_quant.c) ------------------------
// One thread restores one run of `chunk` columns of one row: the int8
// values, un-rotated by the Hadamard transform of the ConvRot group (chunk
// == group; every sum and halving of these small integers is exact), times
// the row's scale - h3_comfy_int8_dequantize_row's arithmetic, bit for bit.
// group == 0: no rotation, chunk is just the work unit. The QKV fields
// reorder rows as h3_dequantize does.

struct h3_comfy_int8_args {
    uint rows;
    uint columns;
    uint group;
    uint chunk;
    uint qkv_inner;
    uint qkv_head_dim;
};

kernel void h3_comfy_int8_expand(device const char *raw [[buffer(0)]],
                                 device const float *scales [[buffer(1)]],
                                 device ushort *output [[buffer(2)]],
                                 constant h3_comfy_int8_args &args [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    ulong first = ulong(gid.x) * args.chunk;
    if (gid.y >= args.rows || first >= args.columns) return;
    uint count = min(args.chunk, uint(args.columns - first));
    float v[256];
    device const char *source = raw + ulong(gid.y) * args.columns + first;
    for (uint i = 0; i < count; i++) v[i] = float(source[i]);
    for (uint stride = 1; stride < args.group; stride *= 4) {
        for (uint base = 0; base < count; base += 4 * stride) {
            for (uint i = base; i < base + stride; i++) {
                float a = v[i], b = v[i + stride];
                float c = v[i + 2 * stride], d = v[i + 3 * stride];
                float t = 0.5f * (a + b + c + d);
                v[i] = t - d;
                v[i + stride] = t - c;
                v[i + 2 * stride] = t - b;
                v[i + 3 * stride] = t - a;
            }
        }
    }
    ulong row = gid.y;
    if (args.qkv_inner != 0) {
        ulong part = row / args.qkv_inner, within = row % args.qkv_inner;
        row = (within / args.qkv_head_dim) * 3 * args.qkv_head_dim +
              part * args.qkv_head_dim + within % args.qkv_head_dim;
    }
    float scale = scales[gid.y];
    device ushort *destination = output + row * args.columns + first;
    for (uint i = 0; i < count; i++)
        destination[i] = h3_f32_to_bf16(v[i] * scale);
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

