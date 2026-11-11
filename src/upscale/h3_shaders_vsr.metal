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
    uint out_w, out_h;                    /* the bilinear target (the whole frame) */
    uint tile_x, tile_y, tile_w, tile_h;  /* the part of it produced: pixels at or
                                             past out_w / out_h are zero (padding) */
};

/* 8-bit frames -> F32 RGB [T, tile_h, tile_w, 3] in [0, 1]: the tile at
 * (tile_x, tile_y) of the torch bilinear (align_corners=False, no antialias)
 * resize of the in_w x in_h frames to
 * out_w x out_h, zero beyond. Grid (out_w + pad_w, out_h + pad_h, T). */
kernel void h3_vsr_pixels_to_f32(device const uchar *input [[buffer(0)]],
                                 device float *output [[buffer(1)]],
                                 constant vsr_pixels_in_args &args [[buffer(2)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    uint lx = gid.x, ly = gid.y, t = gid.z;
    if (lx >= args.tile_w || ly >= args.tile_h || t >= args.frames) return;
    size_t destination = (((size_t)t * args.tile_h + ly) * args.tile_w + lx) * 3;
    uint x = args.tile_x + lx, y = args.tile_y + ly;
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
    uint first_frame;                 /* the F32 frame output frame 0 comes from */
};

/* F32 RGB [.., full_h, full_w, 3] (from frame first_frame) -> 8-bit frames
 * cropped to crop_w x crop_h. Grid (crop_w, crop_h, T). */
kernel void h3_vsr_f32_to_pixels(device const float *input [[buffer(0)]],
                                 device uchar *output [[buffer(1)]],
                                 constant vsr_pixels_out_args &args [[buffer(2)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    uint x = gid.x, y = gid.y, t = gid.z;
    if (x >= args.crop_w || y >= args.crop_h || t >= args.frames) return;
    size_t source = (((size_t)(t + args.first_frame) * args.full_h + y) * args.full_w + x) * 3;
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

struct vsr_tile_args {
    uint frames, first_frame;         /* tile frames first_frame .. +frames */
    uint tile_w, tile_h, tile_x, tile_y;
    uint out_w, out_h;                /* the composite frame */
};

/* composite [T, out_h, out_w, 3] += tile [.., tile_h, tile_w, 3] (from frame
 * first_frame) * weights [tile_h, tile_w] over the tile's pixels inside the
 * out_w x out_h frame (the spatial tiles of h3_vsr.c). Grid (tile_w,
 * tile_h, T); one thread per composite pixel, so no two add to the same. */
kernel void h3_vsr_tile_accumulate_f32(device float *composite [[buffer(0)]],
                                       device const float *tile [[buffer(1)]],
                                       device const float *weights [[buffer(2)]],
                                       constant vsr_tile_args &args [[buffer(3)]],
                                       uint3 gid [[thread_position_in_grid]]) {
    uint x = gid.x, y = gid.y, t = gid.z;
    if (x >= args.tile_w || y >= args.tile_h || t >= args.frames) return;
    uint X = args.tile_x + x, Y = args.tile_y + y;
    if (X >= args.out_w || Y >= args.out_h) return;
    float w = weights[(size_t)y * args.tile_w + x];
    size_t source = (((size_t)(t + args.first_frame) * args.tile_h + y) * args.tile_w + x) * 3;
    size_t destination = (((size_t)t * args.out_h + Y) * args.out_w + X) * 3;
    for (uint c = 0; c < 3; c++) composite[destination + c] += tile[source + c] * w;
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
    int offsets[27];  /* 3x3 spatial, or 3x3x3 with the temporal taps a frame apart */
    uint x1_rows;     /* physical rows of x1 (x_rows when it is a second input) */
    uint tap_mask;    /* bit t: tap t reads x1 (with w0's tap weights) instead of x0 - the
                         causal state of a video frame, kept in its own buffer */
    uint order;       /* 0: threadgroup x = row tile, y = channel tile; 1: swapped, so the channel
                         tiles of one row tile run together and share its input rows in the cache */
};

/* Metal 4 tensor API: compiled only with H3_METAL_HAS_TENSOR, like the other
 * *_nax kernels, so the library still builds on Macs without TensorOps (and
 * for the no-TensorOps retry in h3_gpu.m). */
#ifdef H3_METAL_HAS_TENSOR
/* One output tile of RT rows x CT channels per threadgroup of SG simdgroups,
 * K in chunks of KT; instantiated below in several shapes (the host picks
 * one by name: h3_gpu.m vsr_tapconv_variants). */
template <int RT, int CT, int KT, int SG>
kernel void h3_vsr_tapconv_bf16_nax_t(device bfloat *x0 [[buffer(0)]],
                                      device bfloat *x1 [[buffer(1)]],
                                      device bfloat *w0 [[buffer(2)]],
                                      device bfloat *w1 [[buffer(3)]],
                                      device const float *bias [[buffer(4)]],
                                      device const bfloat *residual [[buffer(5)]],
                                      device bfloat *output [[buffer(6)]],
                                      constant vsr_tapconv_args &args [[buffer(7)]],
                                      uint2 group [[threadgroup_position_in_grid]]) {
    uint row_tile = args.order ? group.y : group.x, col_tile = args.order ? group.x : group.y;
    int row_start = (int)(row_tile * RT), col_start = (int)(col_tile * CT);
    auto xa = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        x0, dextents<int32_t, 2>((int)args.in_ch, (int)args.x_rows));
    auto xb = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        x1, dextents<int32_t, 2>((int)args.in_ch, (int)args.x1_rows));
    auto wa = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        w0, dextents<int32_t, 2>((int)args.in_ch, (int)(args.taps * args.out_ch)));
    auto wb = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        w1, dextents<int32_t, 2>((int)args.in_ch, (int)(args.taps * args.out_ch)));
    constexpr auto descriptor = matmul2d_descriptor(
        RT, CT, KT, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<SG>> mm;
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
            bool from_x1 = input == 0 && ((args.tap_mask >> t) & 1u);
            for (int k = 0; k < (int)args.in_ch; k += KT) {
                if (input == 0 && !from_x1) {
                    auto a = xa.slice<KT, RT>(k, base);
                    auto b = wa.slice<KT, CT>(k, col);
                    mm.run(a, b, accum);
                } else if (from_x1) {
                    auto a = xb.slice<KT, RT>(k, base);
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
template [[host_name("h3_vsr_tapconv_bf16_nax")]] kernel void h3_vsr_tapconv_bf16_nax_t<128, 64, 64, 4>(
    device bfloat *, device bfloat *, device bfloat *, device bfloat *, device const float *, device const bfloat *,
    device bfloat *, constant vsr_tapconv_args &, uint2);
template [[host_name("h3_vsr_tapconv_bf16_nax_r128c128")]] kernel void h3_vsr_tapconv_bf16_nax_t<128, 128, 64, 8>(
    device bfloat *, device bfloat *, device bfloat *, device bfloat *, device const float *, device const bfloat *,
    device bfloat *, constant vsr_tapconv_args &, uint2);
template [[host_name("h3_vsr_tapconv_bf16_nax_r256c64")]] kernel void h3_vsr_tapconv_bf16_nax_t<256, 64, 64, 8>(
    device bfloat *, device bfloat *, device bfloat *, device bfloat *, device const float *, device const bfloat *,
    device bfloat *, constant vsr_tapconv_args &, uint2);
template [[host_name("h3_vsr_tapconv_bf16_nax_r64c128")]] kernel void h3_vsr_tapconv_bf16_nax_t<64, 128, 64, 4>(
    device bfloat *, device bfloat *, device bfloat *, device bfloat *, device const float *, device const bfloat *,
    device bfloat *, constant vsr_tapconv_args &, uint2);
template [[host_name("h3_vsr_tapconv_bf16_nax_r128c64k128")]] kernel void h3_vsr_tapconv_bf16_nax_t<128, 64, 128, 4>(
    device bfloat *, device bfloat *, device bfloat *, device bfloat *, device const float *, device const bfloat *,
    device bfloat *, constant vsr_tapconv_args &, uint2);
#endif

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

