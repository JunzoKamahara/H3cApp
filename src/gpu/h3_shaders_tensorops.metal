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
