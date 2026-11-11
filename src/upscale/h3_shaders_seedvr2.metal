/* ------------------------------------------------------------------ */
/* SeedVR2 (h3_seedvr2.c): the one-step image restorer run frame by frame.
 * Transformer tensors are BF16 rows with F32 constants, the VAE runs on
 * the F32 channels-last conv path. UPSCALE_PLAN appendix A.7. */

struct sv2_dequant_args { uint rows, columns, group; };

/* MLX affine 8-bit weights (mx.quantize, bits 8, group 64): four values per
 * uint32, least significant first; w = scale[g] * q + bias[g] per group of
 * `group` columns, scales and biases F16. Grid (columns / 4, rows). */
kernel void h3_sv2_dequant_q8_bf16(device const uint *packed [[buffer(0)]],
                                   device const half *scales [[buffer(1)]],
                                   device const half *biases [[buffer(2)]],
                                   device ushort *output [[buffer(3)]],
                                   constant sv2_dequant_args &args [[buffer(4)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    uint word = gid.x, row = gid.y;
    uint words = args.columns / 4;
    if (word >= words || row >= args.rows) return;
    uint column = word * 4;
    uint groups = args.columns / args.group;
    size_t g = (size_t)row * groups + column / args.group;
    float scale = float(scales[g]), bias = float(biases[g]);
    uint value = packed[(size_t)row * words + word];
    device ushort *out = output + (size_t)row * args.columns + column;
    for (uint i = 0; i < 4; i++)
        out[i] = h3_f32_to_bf16(fma(scale, float((value >> (8 * i)) & 0xffu), bias));
}

struct sv2_dequant_rows_args { uint rows, columns; };
struct sv2_quant_rows_args { uint rows, columns, in_row0, out_row0; };

/* 4-bit affine in groups of 64 along the row (scale and minimum per group,
 * F32 pairs in `params`): q = round((x - min) / scale) in 0..15, two per
 * byte, the even column in the low nibble. rows x columns/64 threads,
 * one per group. */
kernel void h3_sv2_quantize_rows_bf16_int4(device const bfloat *input [[buffer(0)]],
                                           device uchar *output [[buffer(1)]],
                                           device float *params [[buffer(2)]],
                                           constant sv2_quant_rows_args &args [[buffer(3)]],
                                           uint2 gid [[thread_position_in_grid]]) {
    uint g = gid.x, row = gid.y, groups = args.columns / 64;
    if (g >= groups || row >= args.rows) return;
    size_t in_base = (size_t)(args.in_row0 + row) * args.columns + g * 64;
    float lo = INFINITY, hi = -INFINITY;
    for (uint i = 0; i < 64; i++) { float v = (float)input[in_base + i]; lo = min(lo, v); hi = max(hi, v); }
    float scale = (hi - lo) / 15.0f;
    if (!(scale > 0.0f)) scale = 1.0f;
    float inverse = 1.0f / scale;
    size_t p = ((size_t)(args.out_row0 + row) * groups + g) * 2;
    params[p] = scale; params[p + 1] = lo;
    device uchar *out = output + ((size_t)(args.out_row0 + row) * args.columns + g * 64) / 2;
    for (uint i = 0; i < 64; i += 2) {
        int a = clamp((int)rint(((float)input[in_base + i] - lo) * inverse), 0, 15);
        int b = clamp((int)rint(((float)input[in_base + i + 1] - lo) * inverse), 0, 15);
        out[i / 2] = (uchar)(a | (b << 4));
    }
}

/* The reverse: rows x columns/2 threads, one per byte. */
kernel void h3_sv2_dequant_rows_int4_bf16(device const uchar *input [[buffer(0)]],
                                          device const float *params [[buffer(1)]],
                                          device ushort *output [[buffer(2)]],
                                          constant sv2_dequant_rows_args &args [[buffer(3)]],
                                          uint2 gid [[thread_position_in_grid]]) {
    uint b = gid.x, row = gid.y, half_cols = args.columns / 2;
    if (b >= half_cols || row >= args.rows) return;
    uint column = b * 2, groups = args.columns / 64;
    size_t p = ((size_t)row * groups + column / 64) * 2;
    float scale = params[p], lo = params[p + 1];
    uchar q = input[(size_t)row * half_cols + b];
    device ushort *out = output + (size_t)row * args.columns + column;
    out[0] = h3_f32_to_bf16(fma(float(q & 15u), scale, lo));
    out[1] = h3_f32_to_bf16(fma(float(q >> 4), scale, lo));
}

/* Per-row symmetric int8 of rows [in_row0, in_row0 + rows) of a BF16 row
 * buffer into rows [out_row0, ...) of an int8 one (scales at out_row0 too):
 * one frame of a video window into its slot of the causal state. One
 * threadgroup of 256 per row. */
kernel void h3_sv2_quantize_rows_bf16_int8(device const bfloat *input [[buffer(0)]],
                                           device char *output [[buffer(1)]],
                                           device float *scales [[buffer(2)]],
                                           constant sv2_quant_rows_args &args [[buffer(3)]],
                                           uint tid [[thread_index_in_threadgroup]],
                                           uint row [[threadgroup_position_in_grid]]) {
    if (row >= args.rows) return;
    size_t in_base = (size_t)(args.in_row0 + row) * args.columns, out_base = (size_t)(args.out_row0 + row) * args.columns;
    threadgroup float scratch[256];
    float local_max = 0.0f;
    for (uint c = tid; c < args.columns; c += 256) local_max = max(local_max, fabs((float)input[in_base + c]));
    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_abs = scratch[0];
    float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
    float inverse = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
    if (tid == 0) scales[args.out_row0 + row] = scale;
    for (uint c = tid; c < args.columns; c += 256) {
        int q = (int)rint((float)input[in_base + c] * inverse);
        output[out_base + c] = (char)clamp(q, -127, 127);
    }
}

/* Per-row symmetric int8 (h3_quantize_bf16_int8_rows: one F32 scale per
 * row) -> BF16 rows: the causal state of a video window kept in int8.
 * Grid (columns / 4, rows). */
kernel void h3_sv2_dequant_rows_int8_bf16(device const char4 *input [[buffer(0)]],
                                          device const float *scales [[buffer(1)]],
                                          device ushort *output [[buffer(2)]],
                                          constant sv2_dequant_rows_args &args [[buffer(3)]],
                                          uint2 gid [[thread_position_in_grid]]) {
    uint vec = gid.x, row = gid.y, vectors = args.columns / 4;
    if (vec >= vectors || row >= args.rows) return;
    float s = scales[row];
    char4 q = input[(size_t)row * vectors + vec];
    device ushort *out = output + (size_t)row * args.columns + vec * 4;
    out[0] = h3_f32_to_bf16(s * float(q.x));
    out[1] = h3_f32_to_bf16(s * float(q.y));
    out[2] = h3_f32_to_bf16(s * float(q.z));
    out[3] = h3_f32_to_bf16(s * float(q.w));
}

struct sv2_rms_args { uint rows, width, has_scale, has_shift; float epsilon; };

/* out = x / rms(x) * scale + shift (no norm weight; scale and shift are
 * F32 vectors, 1 and 0 when absent). One threadgroup per row. */
kernel void h3_sv2_rms_mod_bf16(device const ushort *input [[buffer(0)]],
                                device ushort *output [[buffer(1)]],
                                device const float *scale [[buffer(2)]],
                                device const float *shift [[buffer(3)]],
                                constant sv2_rms_args &args [[buffer(4)]],
                                uint3 group [[threadgroup_position_in_grid]],
                                uint3 thread_position [[thread_position_in_threadgroup]],
                                uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x, tid = thread_position.x, threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    device const ushort *x = input + (size_t)row * args.width;
    float local = 0.0f;
    for (uint column = tid; column < args.width; column += threads) {
        float v = h3_bf16_to_f32(x[column]);
        local = fma(v, v, local);
    }
    reductions[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    device ushort *y = output + (size_t)row * args.width;
    for (uint column = tid; column < args.width; column += threads) {
        float v = h3_bf16_to_f32(x[column]) * inverse;
        if (args.has_scale) v *= scale[column];
        if (args.has_shift) v += shift[column];
        y[column] = h3_f32_to_bf16(v);
    }
}

inline float sv2_round_bf16(float x) { return h3_bf16_to_f32(h3_f32_to_bf16(x)); }

struct sv2_window_args { uint rows, heads, head_dim, dim, pairs; float epsilon; };

/* The window gather of SeedVR2's attention: row r of the output takes
 * token index[r] of the video stream (index >= 0) or text token
 * -1 - index[r], q and k RMS-normalised per head with the stream's weights,
 * then rotated by the row's (cos, sin)[pairs] on dimensions 0 .. 2*pairs
 * (interleaved pairs; the rest pass through). Q/K/V are written as
 * [rows, heads * head_dim]. Grid (heads, rows). */
kernel void h3_sv2_qkv_window_bf16(device const ushort *qkv_vid [[buffer(0)]],
                                   device const ushort *qkv_txt [[buffer(1)]],
                                   device const int *index [[buffer(2)]],
                                   device const float *rope_cos [[buffer(3)]],
                                   device const float *rope_sin [[buffer(4)]],
                                   device const float *norm_q_vid [[buffer(5)]],
                                   device const float *norm_k_vid [[buffer(6)]],
                                   device const float *norm_q_txt [[buffer(7)]],
                                   device const float *norm_k_txt [[buffer(8)]],
                                   device ushort *q_out [[buffer(9)]],
                                   device ushort *k_out [[buffer(10)]],
                                   device ushort *v_out [[buffer(11)]],
                                   constant sv2_window_args &args [[buffer(12)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    uint head = gid.x, row = gid.y;
    if (head >= args.heads || row >= args.rows) return;
    int token = index[row];
    device const ushort *source;
    device const float *wq, *wk;
    if (token >= 0) {
        source = qkv_vid + (size_t)token * 3 * args.dim;
        wq = norm_q_vid; wk = norm_k_vid;
    } else {
        source = qkv_txt + (size_t)(-1 - token) * 3 * args.dim;
        wq = norm_q_txt; wk = norm_k_txt;
    }
    uint hd = args.head_dim;
    float q[128], k[128];
    float sq = 0.0f, sk = 0.0f;
    for (uint d = 0; d < hd; d++) {
        q[d] = h3_bf16_to_f32(source[head * hd + d]);
        k[d] = h3_bf16_to_f32(source[args.dim + head * hd + d]);
        sq = fma(q[d], q[d], sq);
        sk = fma(k[d], k[d], sk);
    }
    float iq = rsqrt(sq / float(hd) + args.epsilon);
    float ik = rsqrt(sk / float(hd) + args.epsilon);
    for (uint d = 0; d < hd; d++) {
        q[d] = sv2_round_bf16(q[d] * iq * wq[d]);
        k[d] = sv2_round_bf16(k[d] * ik * wk[d]);
    }
    device const float *c = rope_cos + (size_t)row * args.pairs;
    device const float *s = rope_sin + (size_t)row * args.pairs;
    size_t out = (size_t)row * args.dim + head * hd;
    for (uint p = 0; p < args.pairs; p++) {
        float cp = c[p], sp = s[p];
        uint d = 2 * p;
        float q0 = q[d], q1 = q[d + 1], k0 = k[d], k1 = k[d + 1];
        q_out[out + d] = h3_f32_to_bf16(q0 * cp - q1 * sp);
        q_out[out + d + 1] = h3_f32_to_bf16(q1 * cp + q0 * sp);
        k_out[out + d] = h3_f32_to_bf16(k0 * cp - k1 * sp);
        k_out[out + d + 1] = h3_f32_to_bf16(k1 * cp + k0 * sp);
    }
    for (uint d = 2 * args.pairs; d < hd; d++) {
        q_out[out + d] = h3_f32_to_bf16(q[d]);
        k_out[out + d] = h3_f32_to_bf16(k[d]);
    }
    for (uint d = 0; d < hd; d++)
        v_out[out + d] = source[2 * args.dim + head * hd + d];
}

struct sv2_txt_mean_args { uint windows, txt_rows, width; };

/* out[j] = mean over windows w of in[base[w] + j]: the text tokens' outputs
 * averaged over the windows they were attached to. Grid (width, txt_rows). */
kernel void h3_sv2_txt_mean_bf16(device const ushort *input [[buffer(0)]],
                                 device const int *base [[buffer(1)]],
                                 device ushort *output [[buffer(2)]],
                                 constant sv2_txt_mean_args &args [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x, j = gid.y;
    if (column >= args.width || j >= args.txt_rows) return;
    float sum = 0.0f;
    for (uint w = 0; w < args.windows; w++)
        sum += h3_bf16_to_f32(input[((size_t)base[w] + j) * args.width + column]);
    output[(size_t)j * args.width + column] = h3_f32_to_bf16(sum / float(args.windows));
}

struct sv2_pad_args { uint height, width, channels, top, bottom, left, right; };

/* Zero padding of an F32 [H, W, C] plane by (top, bottom, left, right).
 * Grid (C, out_w, out_h). */
kernel void h3_sv2_pad_f32(device const float *input [[buffer(0)]],
                           device float *output [[buffer(1)]],
                           constant sv2_pad_args &args [[buffer(2)]],
                           uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, ox = gid.y, oy = gid.z;
    uint out_w = args.width + args.left + args.right;
    uint out_h = args.height + args.top + args.bottom;
    if (c >= args.channels || ox >= out_w || oy >= out_h) return;
    int x = int(ox) - int(args.left), y = int(oy) - int(args.top);
    float v = 0.0f;
    if (x >= 0 && y >= 0 && x < int(args.width) && y < int(args.height))
        v = input[((size_t)y * args.width + x) * args.channels + c];
    output[((size_t)oy * out_w + ox) * args.channels + c] = v;
}

struct sv2_shuffle_args { uint height, width, channels, temporal, frames, out_frames, drop_head; };

/* Output frame o of SeedVR2's temporal upsample: frame f*tf + z of the full
 * rearrange, or past the dropped frame 1 when drop_head (remove_head on a
 * clip's first slice) and tf > 1. */
static inline uint sv2_shuffle_source(uint o, uint temporal, uint drop_head) {
    return (drop_head && temporal > 1 && o > 0) ? o + 1 : o;
}

/* SeedVR2's Upsample3D: in [T, H, W, C * 4 * tf] -> out [T', 2H, 2W, C]
 * with the rearrange b (x y z c) f h w -> b c (f z) (h x) (w y):
 * out[f*tf+z, 2y+sh, 2x+sw, c] = in[f, y, x, ((sh*2+sw)*tf + z)*C + c],
 * T' = tf*T, or tf*T - 1 without frame 1 when drop_head (T = 1 keeps the
 * first temporal slice only). Grid (C, 2W, 2H * T'). */
kernel void h3_sv2_shuffle_f32(device const float *input [[buffer(0)]],
                               device float *output [[buffer(1)]],
                               constant sv2_shuffle_args &args [[buffer(2)]],
                               uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, ox = gid.y, oz = gid.z, oh = 2 * args.height;
    if (c >= args.channels || ox >= 2 * args.width || oz >= oh * args.out_frames) return;
    uint o = oz / oh, oy = oz - o * oh;
    uint full = sv2_shuffle_source(o, args.temporal, args.drop_head);
    uint f = full / args.temporal, z = full % args.temporal;
    uint x = ox / 2, sw = ox % 2, y = oy / 2, sh = oy % 2;
    uint in_channels = args.channels * 4 * args.temporal;
    uint channel = ((sh * 2 + sw) * args.temporal + z) * args.channels + c;
    output[(((size_t)o * oh + oy) * 2 * args.width + ox) * args.channels + c] =
        input[(((size_t)f * args.height + y) * args.width + x) * in_channels + channel];
}

struct sv2_pad3d_args {
    uint frames, height, width, channels;
    uint front, prev_frames, has_previous;
    uint top, bottom, left, right;
};

/* The causal convolution's input: `front` frames, then the clip [T, H, W, C]
 * zero-padded in space by (top, bottom, left, right). The front frames are
 * the last `front` of `previous` ([prev_frames, H, W, C], the state of the
 * clip's previous slice) or, without it, copies of frame 0 (extend_head).
 * Grid (C, out_w, out_h * (front + T)). */
kernel void h3_sv2_pad3d_f32(device const float *input [[buffer(0)]],
                             device const float *previous [[buffer(1)]],
                             device float *output [[buffer(2)]],
                             constant sv2_pad3d_args &args [[buffer(3)]],
                             uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, ox = gid.y, oz = gid.z;
    uint out_w = args.width + args.left + args.right;
    uint out_h = args.height + args.top + args.bottom;
    if (c >= args.channels || ox >= out_w || oz >= out_h * (args.front + args.frames)) return;
    uint t = oz / out_h, oy = oz - t * out_h;
    int x = int(ox) - int(args.left), y = int(oy) - int(args.top);
    float v = 0.0f;
    if (x >= 0 && y >= 0 && x < int(args.width) && y < int(args.height)) {
        size_t pixel = ((size_t)y * args.width + x) * args.channels + c;
        size_t plane = (size_t)args.height * args.width * args.channels;
        if (t >= args.front) v = input[(size_t)(t - args.front) * plane + pixel];
        else if (args.has_previous) v = previous[(size_t)(args.prev_frames - args.front + t) * plane + pixel];
        else v = input[pixel];
    }
    output[(((size_t)t * out_h + oy) * out_w + ox) * args.channels + c] = v;
}

/* GroupNorm without the SiLU (the VAE's attention input): the one-pass
 * kernel above, one threadgroup per (frame, group). */
kernel void h3_sv2_group_norm_f32(device const float *input [[buffer(0)]],
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
    uint plane = row / args.groups;
    uint elements = args.height * args.width * channels_per_group;
    threadgroup float reductions[256];
    float local = 0.0f;
    for (uint index = tid; index < elements; index += threadgroup_size.x) {
        uint spatial = index / channels_per_group;
        uint channel = group_index * channels_per_group + index % channels_per_group;
        local += input[((size_t)plane * args.height * args.width + spatial) * args.channels + channel];
    }
    reductions[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadgroup_size.x / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = reductions[0] / float(elements);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    local = 0.0f;
    for (uint index = tid; index < elements; index += threadgroup_size.x) {
        uint spatial = index / channels_per_group;
        uint channel = group_index * channels_per_group + index % channels_per_group;
        float centered = input[((size_t)plane * args.height * args.width + spatial) * args.channels + channel] - mean;
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
        uint channel = group_index * channels_per_group + index % channels_per_group;
        size_t at = ((size_t)plane * args.height * args.width + spatial) * args.channels + channel;
        output[at] = (input[at] - mean) * inverse * weight[channel] + bias[channel];
    }
}

struct sv2_bicubic_args {
    uint in_w, in_h, in_stride, bgra;
    uint out_w, out_h, full_w, full_h;   /* the resized image, inside the padded frame */
    float scale_x, scale_y;              /* input pixels per output pixel */
    float pad_value;
};

inline float sv2_bicubic(float x) {
    const float a = -0.5f;
    x = fabs(x);
    if (x < 1.0f) return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    if (x < 2.0f) return (((x - 5.0f) * x + 8.0f) * x - 4.0f) * a;
    return 0.0f;
}

/* PIL's BICUBIC resize (Catmull-Rom with a = -0.5, half-pixel centres, the
 * filter widened when shrinking), 8-bit BGRA/RGB in -> F32 [full_h, full_w,
 * 3] in [-1, 1], the area outside out_w x out_h filled with pad_value.
 * Grid (full_w, full_h). */
kernel void h3_sv2_bicubic_pixels_f32(device const uchar *pixels [[buffer(0)]],
                                      device float *output [[buffer(1)]],
                                      constant sv2_bicubic_args &args [[buffer(2)]],
                                      uint2 gid [[thread_position_in_grid]]) {
    uint x = gid.x, y = gid.y;
    if (x >= args.full_w || y >= args.full_h) return;
    device float *out = output + ((size_t)y * args.full_w + x) * 3;
    if (x >= args.out_w || y >= args.out_h) {
        out[0] = out[1] = out[2] = args.pad_value;
        return;
    }
    uint bpp = args.bgra ? 4 : 3;
    float fx = max(args.scale_x, 1.0f), fy = max(args.scale_y, 1.0f);
    float sx = 2.0f * fx, sy = 2.0f * fy;      /* support */
    float cx = (float(x) + 0.5f) * args.scale_x, cy = (float(y) + 0.5f) * args.scale_y;
    int x0 = int(cx - sx + 0.5f), x1 = int(cx + sx + 0.5f);
    int y0 = int(cy - sy + 0.5f), y1 = int(cy + sy + 0.5f);
    x0 = max(x0, 0); y0 = max(y0, 0);
    x1 = min(x1, int(args.in_w)); y1 = min(y1, int(args.in_h));
    float wx[16], wy[16];
    int nx = min(x1 - x0, 16), ny = min(y1 - y0, 16);
    float tx = 0.0f, ty = 0.0f;
    for (int i = 0; i < nx; i++) { wx[i] = sv2_bicubic((float(x0 + i) - cx + 0.5f) / fx); tx += wx[i]; }
    for (int j = 0; j < ny; j++) { wy[j] = sv2_bicubic((float(y0 + j) - cy + 0.5f) / fy); ty += wy[j]; }
    float acc[3] = {0.0f, 0.0f, 0.0f};
    for (int j = 0; j < ny; j++) {
        device const uchar *row = pixels + (size_t)(y0 + j) * args.in_stride;
        for (int i = 0; i < nx; i++) {
            float w = wx[i] * wy[j];
            device const uchar *p = row + (size_t)(x0 + i) * bpp;
            for (uint c = 0; c < 3; c++) acc[c] += w * float(p[args.bgra ? 2 - c : c]);
        }
    }
    float norm = tx * ty;
    for (uint c = 0; c < 3; c++)
        out[c] = clamp(acc[c] / norm, 0.0f, 255.0f) / 255.0f * 2.0f - 1.0f;
}

struct sv2_pixels_out_args { uint full_w, full_h, crop_w, crop_h, out_stride, bgra; };

/* F32 [full_h, full_w, 3] in [-1, 1] -> 8-bit BGRA/RGB cropped to
 * crop_w x crop_h, rounded to nearest. Grid (crop_w, crop_h). */
kernel void h3_sv2_f32_to_pixels(device const float *input [[buffer(0)]],
                                 device uchar *output [[buffer(1)]],
                                 constant sv2_pixels_out_args &args [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    uint x = gid.x, y = gid.y;
    if (x >= args.crop_w || y >= args.crop_h) return;
    device const float *src = input + ((size_t)y * args.full_w + x) * 3;
    uint bpp = args.bgra ? 4 : 3;
    device uchar *pixel = output + (size_t)y * args.out_stride + x * bpp;
    for (uint c = 0; c < 3; c++) {
        float v = clamp((src[c] + 1.0f) * 0.5f, 0.0f, 1.0f) * 255.0f;
        pixel[args.bgra ? 2 - c : c] = uchar(min(uint(v + 0.5f), 255u));
    }
    if (args.bgra) pixel[3] = 255;
}

struct sv2_blur_args { uint height, width; int radius; };

/* The colour transfer's wavelet blur: a 3x3 [1 2 1]/4 x [1 2 1]/4 kernel at
 * taps +-radius with edge clamping, on F32 [H, W, 3]. Grid (3, W, H). */
kernel void h3_sv2_wavelet_blur_f32(device const float *input [[buffer(0)]],
                                    device float *output [[buffer(1)]],
                                    constant sv2_blur_args &args [[buffer(2)]],
                                    uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, x = gid.y, y = gid.z;
    if (c >= 3 || x >= args.width || y >= args.height) return;
    const float k[3] = {0.25f, 0.5f, 0.25f};
    float sum = 0.0f;
    for (int j = -1; j <= 1; j++) {
        int yy = clamp(int(y) + j * args.radius, 0, int(args.height) - 1);
        for (int i = -1; i <= 1; i++) {
            int xx = clamp(int(x) + i * args.radius, 0, int(args.width) - 1);
            sum += k[j + 1] * k[i + 1] * input[((size_t)yy * args.width + xx) * 3 + c];
        }
    }
    output[((size_t)y * args.width + x) * 3 + c] = sum;
}

inline float sv2_srgb_to_linear(float x) {
    return x > 0.04045f ? pow((x + 0.055f) / 1.055f, 2.4f) : x / 12.92f;
}
inline float sv2_linear_to_srgb(float x) {
    return x > 0.0031308f ? 1.055f * pow(max(x, 0.0f), 1.0f / 2.4f) - 0.055f : 12.92f * x;
}
inline float sv2_lab_f(float t) {
    const float eps3 = (6.0f / 29.0f) * (6.0f / 29.0f) * (6.0f / 29.0f);
    const float kappa = (29.0f / 3.0f) * (29.0f / 3.0f) * (29.0f / 3.0f);
    return t > eps3 ? pow(t, 1.0f / 3.0f) : (kappa * t + 16.0f) / 116.0f;
}
inline float sv2_lab_finv(float f) {
    const float eps = 6.0f / 29.0f;
    const float kappa = (29.0f / 3.0f) * (29.0f / 3.0f) * (29.0f / 3.0f);
    return f > eps ? f * f * f : (116.0f * f - 16.0f) / kappa;
}

struct sv2_lab_args { uint pixels; };

/* RGB in [-1, 1] (clipped to [0, 1] after the shift) -> CIE L*a*b* (D65),
 * the same formulas as the reference. Grid (pixels). */
kernel void h3_sv2_rgb_to_lab_f32(device const float *input [[buffer(0)]],
                                  device float *output [[buffer(1)]],
                                  constant sv2_lab_args &args [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= args.pixels) return;
    device const float *p = input + (size_t)gid * 3;
    float r = sv2_srgb_to_linear(clamp((p[0] + 1.0f) * 0.5f, 0.0f, 1.0f));
    float g = sv2_srgb_to_linear(clamp((p[1] + 1.0f) * 0.5f, 0.0f, 1.0f));
    float b = sv2_srgb_to_linear(clamp((p[2] + 1.0f) * 0.5f, 0.0f, 1.0f));
    float X = (0.4124564f * r + 0.3575761f * g + 0.1804375f * b) / 0.95047f;
    float Y = 0.2126729f * r + 0.7151522f * g + 0.0721750f * b;
    float Z = (0.0193339f * r + 0.1191920f * g + 0.9503041f * b) / 1.08883f;
    float fx = sv2_lab_f(X), fy = sv2_lab_f(Y), fz = sv2_lab_f(Z);
    device float *o = output + (size_t)gid * 3;
    o[0] = 116.0f * fy - 16.0f;
    o[1] = 500.0f * (fx - fy);
    o[2] = 200.0f * (fy - fz);
}

struct sv2_lab_match_args {
    uint pixels, bins;
    float l_weight;
    float minimum[3], step[3];   /* bin b of channel c covers minimum + b * step */
};

/* L*a*b* -> RGB in [-1, 1] after the histogram match: a and b are replaced
 * by their matched values lut[c][bin(v)], L by l_weight * L + (1 - l_weight)
 * * lut[0][bin(L)]. Grid (pixels). */
kernel void h3_sv2_lab_match_to_rgb_f32(device const float *lab [[buffer(0)]],
                                        device const float *luts [[buffer(1)]],
                                        device float *output [[buffer(2)]],
                                        constant sv2_lab_match_args &args [[buffer(3)]],
                                        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.pixels) return;
    device const float *p = lab + (size_t)gid * 3;
    float v[3];
    for (uint c = 0; c < 3; c++) {
        float t = (p[c] - args.minimum[c]) / args.step[c];
        int bin = clamp(int(t), 0, int(args.bins) - 1);
        v[c] = luts[(size_t)c * args.bins + bin];
    }
    float L = args.l_weight * p[0] + (1.0f - args.l_weight) * v[0];
    float a = v[1], b = v[2];
    float fy = (L + 16.0f) / 116.0f, fx = a / 500.0f + fy, fz = fy - b / 200.0f;
    float X = sv2_lab_finv(fx) * 0.95047f, Y = sv2_lab_finv(fy), Z = sv2_lab_finv(fz) * 1.08883f;
    float r = 3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
    float g = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
    float bl = 0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;
    device float *o = output + (size_t)gid * 3;
    o[0] = clamp(sv2_linear_to_srgb(r), 0.0f, 1.0f) * 2.0f - 1.0f;
    o[1] = clamp(sv2_linear_to_srgb(g), 0.0f, 1.0f) * 2.0f - 1.0f;
    o[2] = clamp(sv2_linear_to_srgb(bl), 0.0f, 1.0f) * 2.0f - 1.0f;
}

/* --- SeedVR2 VAE on the TensorOps grid (h3_seedvr2.c, M5): the padded
 * BF16 grid of the ReAE ([guard | (H+2) x (W+2) | tail], channels padded
 * to 64), so h3_vsr_tapconv_bf16 does every convolution. --- */

struct sv2_ggn_args {
    uint height, width, channels, padded, groups, guard;
    uint chunks, chunk_pixels, silu;
    float epsilon;
};

/* GroupNorm statistics over the grid's interior pixels, one threadgroup per
 * chunk of pixels: 256 threads, `vectors` = padded / 4 channel vectors,
 * 256 / vectors lanes over the pixels. Pass 1 sums per (chunk, group). */
kernel void h3_sv2_grid_gn_sum_bf16(device const ushort4 *input [[buffer(0)]],
                                    device float *sums [[buffer(1)]],
                                    constant sv2_ggn_args &args [[buffer(2)]],
                                    uint chunk [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float4 partial[256];
    threadgroup float gsum[64];
    uint vectors = args.padded / 4, lanes = 256 / vectors;
    uint vector = tid % vectors, lane = tid / vectors;
    uint pixels = args.height * args.width, wp = args.width + 2;
    uint first = chunk * args.chunk_pixels, last = min(first + args.chunk_pixels, pixels);
    float4 acc = 0.0f;
    for (uint p = first + lane; p < last; p += lanes) {
        size_t row = (size_t)args.guard + (size_t)(1 + p / args.width) * wp + 1 + p % args.width;
        acc += h3_bf16x4_to_f32(input[row * vectors + vector]);
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < args.groups) gsum[tid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint cpg = args.channels / args.groups;
        for (uint v = 0; v < vectors && v * 4 < args.channels; v++) {
            float4 s = 0.0f;
            for (uint l = 0; l < lanes; l++) s += partial[l * vectors + v];
            gsum[(v * 4) / cpg] += s.x + s.y + s.z + s.w;
        }
        for (uint g = 0; g < args.groups; g++) sums[chunk * args.groups + g] = gsum[g];
    }
}

/* Pass 2: sums of squared deviations about each group's mean. */
kernel void h3_sv2_grid_gn_sq_bf16(device const ushort4 *input [[buffer(0)]],
                                   device const float *sums [[buffer(1)]],
                                   device float *squares [[buffer(2)]],
                                   constant sv2_ggn_args &args [[buffer(3)]],
                                   uint chunk [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float4 partial[256];
    threadgroup float mean[64];
    threadgroup float gsum[64];
    uint vectors = args.padded / 4, lanes = 256 / vectors;
    uint vector = tid % vectors, lane = tid / vectors;
    uint pixels = args.height * args.width, wp = args.width + 2;
    uint cpg = args.channels / args.groups;
    if (tid < args.groups) {
        float s = 0.0f;
        for (uint c = 0; c < args.chunks; c++) s += sums[c * args.groups + tid];
        mean[tid] = s / float(pixels * cpg);
        gsum[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint group = (vector * 4) / cpg;
    float centre = vector * 4 < args.channels ? mean[min(group, args.groups - 1)] : 0.0f;
    uint first = chunk * args.chunk_pixels, last = min(first + args.chunk_pixels, pixels);
    float4 acc = 0.0f;
    for (uint p = first + lane; p < last; p += lanes) {
        size_t row = (size_t)args.guard + (size_t)(1 + p / args.width) * wp + 1 + p % args.width;
        float4 d = h3_bf16x4_to_f32(input[row * vectors + vector]) - centre;
        acc = fma(d, d, acc);
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint v = 0; v < vectors && v * 4 < args.channels; v++) {
            float4 s = 0.0f;
            for (uint l = 0; l < lanes; l++) s += partial[l * vectors + v];
            gsum[(v * 4) / cpg] += s.x + s.y + s.z + s.w;
        }
        for (uint g = 0; g < args.groups; g++) squares[chunk * args.groups + g] = gsum[g];
    }
}

/* Pass 3: y = (x - mean) * rsqrt(var + eps) * w + b, SiLU when asked, on the
 * interior pixels; padded channels stay zero. In place is fine. */
kernel void h3_sv2_grid_gn_apply_bf16(device const ushort4 *input [[buffer(0)]],
                                      device const float *sums [[buffer(1)]],
                                      device const float *squares [[buffer(2)]],
                                      device const float *weight [[buffer(3)]],
                                      device const float *bias [[buffer(4)]],
                                      device ushort4 *output [[buffer(5)]],
                                      constant sv2_ggn_args &args [[buffer(6)]],
                                      uint chunk [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float mean[64];
    threadgroup float inverse[64];
    uint vectors = args.padded / 4, lanes = 256 / vectors;
    uint vector = tid % vectors, lane = tid / vectors;
    uint pixels = args.height * args.width, wp = args.width + 2;
    uint cpg = args.channels / args.groups;
    if (tid < args.groups) {
        float s = 0.0f, q = 0.0f;
        for (uint c = 0; c < args.chunks; c++) { s += sums[c * args.groups + tid]; q += squares[c * args.groups + tid]; }
        float n = float(pixels * cpg);
        mean[tid] = s / n;
        inverse[tid] = rsqrt(q / n + args.epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint c0 = vector * 4;
    bool real = c0 < args.channels;
    uint group = min(c0 / cpg, args.groups - 1);
    float centre = real ? mean[group] : 0.0f, scale = real ? inverse[group] : 0.0f;
    float4 w = real ? float4(weight[c0], weight[c0 + 1], weight[c0 + 2], weight[c0 + 3]) : float4(0.0f);
    float4 b = real ? float4(bias[c0], bias[c0 + 1], bias[c0 + 2], bias[c0 + 3]) : float4(0.0f);
    uint first = chunk * args.chunk_pixels, last = min(first + args.chunk_pixels, pixels);
    for (uint p = first + lane; p < last; p += lanes) {
        size_t row = (size_t)args.guard + (size_t)(1 + p / args.width) * wp + 1 + p % args.width;
        size_t at = row * vectors + vector;
        float4 v = fma(h3_bf16x4_to_f32(input[at]) - centre, scale * w, b);
        if (args.silu) v = v / (1.0f + exp(-v));
        if (!real) v = 0.0f;
        output[at] = ushort4(h3_f32_to_bf16(v.x), h3_f32_to_bf16(v.y), h3_f32_to_bf16(v.z), h3_f32_to_bf16(v.w));
    }
}

struct sv2_gshuffle_args {
    uint height, width;          /* the input grid's interior */
    uint channels, padded;       /* output channels C and its padding */
    uint in_padded, temporal, guard, in_guard;
    uint out_frames, drop_head;
};

/* The Upsample3D on the grid (h3_sv2_shuffle_f32's frame rule): in
 * [T, h, w, C*4*tf] -> out [T', 2h, 2w, C], out(f*tf+z, 2y+sh, 2x+sw, c) =
 * in(f, y, x, ((sh*2+sw)*tf + z)*C + c). Frames follow each other from the
 * guard. Grid (padded, 2w+2, (2h+2) * T'); borders and padded channels zero. */
kernel void h3_sv2_grid_shuffle_bf16(device const ushort *input [[buffer(0)]],
                                     device ushort *output [[buffer(1)]],
                                     constant sv2_gshuffle_args &args [[buffer(2)]],
                                     uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, x = gid.y, gz = gid.z;
    uint oh = 2 * args.height, ow = 2 * args.width;
    uint hp = oh + 2, wp = ow + 2;
    if (c >= args.padded || x >= wp || gz >= hp * args.out_frames) return;
    uint o = gz / hp, y = gz - o * hp;
    ushort v = 0;
    if (c < args.channels && y >= 1 && y <= oh && x >= 1 && x <= ow) {
        uint full = sv2_shuffle_source(o, args.temporal, args.drop_head);
        uint f = full / args.temporal, z = full % args.temporal;
        uint iy = (y - 1) / 2, sh = (y - 1) % 2, ix = (x - 1) / 2, sw = (x - 1) % 2;
        uint channel = ((sh * 2 + sw) * args.temporal + z) * args.channels + c;
        size_t row = (size_t)args.in_guard + (size_t)f * (args.height + 2) * (args.width + 2) +
                     (size_t)(iy + 1) * (args.width + 2) + ix + 1;
        v = input[row * args.in_padded + channel];
    }
    output[((size_t)args.guard + (size_t)o * hp * wp + (size_t)y * wp + x) * args.padded + c] = v;
}

struct sv2_gselect_args {
    uint height, width;          /* the output grid's interior */
    uint padded, guard, in_width, in_guard, phase_y, phase_x;
    uint frames, in_height, stride_t, phase_t;
};

/* out(t, y, x) = in(t*stride_t + phase_t, 2y + phase_y, 2x + phase_x) over
 * interior coordinates: the stride-2 convolution's output from the stride-1
 * one (PyTorch's padding (0, 1) means the odd phase; in time, the frames
 * whose window starts on the stride). Same channels. Grid (padded, w+2,
 * (h+2) * frames). */
kernel void h3_sv2_grid_select_bf16(device const ushort *input [[buffer(0)]],
                                    device ushort *output [[buffer(1)]],
                                    constant sv2_gselect_args &args [[buffer(2)]],
                                    uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, x = gid.y, gz = gid.z;
    uint hp = args.height + 2, wp = args.width + 2;
    if (c >= args.padded || x >= wp || gz >= hp * args.frames) return;
    uint t = gz / hp, y = gz - t * hp;
    ushort v = 0;
    if (y >= 1 && y <= args.height && x >= 1 && x <= args.width) {
        uint it = t * args.stride_t + args.phase_t;
        uint iy = 2 * (y - 1) + args.phase_y, ix = 2 * (x - 1) + args.phase_x;
        size_t row = (size_t)args.in_guard + (size_t)it * (args.in_height + 2) * (args.in_width + 2) +
                     (size_t)(iy + 1) * (args.in_width + 2) + ix + 1;
        v = input[row * args.padded + c];
    }
    output[((size_t)args.guard + (size_t)t * hp * wp + (size_t)y * wp + x) * args.padded + c] = v;
}
