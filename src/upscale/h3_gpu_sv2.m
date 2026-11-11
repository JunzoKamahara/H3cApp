/* GPU kernels of the SeedVR2 upscaler (declared in h3_gpu_sv2.h; shaders in h3_shaders_seedvr2.metal). Split out of h3_gpu.m:
 * it uses that file's internals through h3_gpu_internal.h and registers its
 * kernel names at load time, so a program that does not link the upscale
 * module builds no pipelines for it. */
#include "h3_gpu_internal.h"
#include "h3_gpu_sv2.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const char *const sv2_kernels[] = {
    "h3_sv2_dequant_q8_bf16",
    "h3_sv2_dequant_rows_int8_bf16",
    "h3_sv2_quantize_rows_bf16_int8",
    "h3_sv2_quantize_rows_bf16_int4",
    "h3_sv2_dequant_rows_int4_bf16",
    "h3_sv2_rms_mod_bf16",
    "h3_sv2_qkv_window_bf16",
    "h3_sv2_txt_mean_bf16",
    "h3_sv2_pad_f32",
    "h3_sv2_pad3d_f32",
    "h3_sv2_shuffle_f32",
    "h3_sv2_group_norm_f32",
    "h3_sv2_bicubic_pixels_f32",
    "h3_sv2_f32_to_pixels",
    "h3_sv2_wavelet_blur_f32",
    "h3_sv2_rgb_to_lab_f32",
    "h3_sv2_lab_match_to_rgb_f32",
    "h3_sv2_grid_gn_sum_bf16",
    "h3_sv2_grid_gn_sq_bf16",
    "h3_sv2_grid_gn_apply_bf16",
    "h3_sv2_grid_shuffle_bf16",
    "h3_sv2_grid_select_bf16",
};

__attribute__((constructor))
static void h3_gpu_sv2_register(void) {
    h3_gpu_register_kernels(sv2_kernels, sizeof(sv2_kernels) / sizeof(*sv2_kernels), 0);
}

/* --- SeedVR2 kernels (h3_seedvr2.c) --- */



typedef struct { uint32_t rows, columns, group; } sv2_dequant_args;
typedef struct { uint32_t rows, columns; } sv2_dequant_rows_args;
typedef struct { uint32_t rows, columns, in_row0, out_row0; } sv2_quant_rows_args;
typedef struct { uint32_t rows, width, has_scale, has_shift; float epsilon; } sv2_rms_args;
typedef struct { uint32_t rows, heads, head_dim, dim, pairs; float epsilon; } sv2_window_args;
typedef struct { uint32_t windows, txt_rows, width; } sv2_txt_mean_args;
typedef struct { uint32_t height, width, channels, top, bottom, left, right; } sv2_pad_args;
typedef struct { uint32_t height, width, channels, temporal, frames, out_frames, drop_head; } sv2_shuffle_args;
typedef struct { uint32_t frames, height, width, channels, front, prev_frames, has_previous, top, bottom, left, right; } sv2_pad3d_args;
typedef struct { uint32_t in_w, in_h, in_stride, bgra, out_w, out_h, full_w, full_h;
                 float scale_x, scale_y, pad_value; } sv2_bicubic_args;
typedef struct { uint32_t full_w, full_h, crop_w, crop_h, out_stride, bgra; } sv2_pixels_out_args;
typedef struct { uint32_t height, width; int32_t radius; } sv2_blur_args;
typedef struct { uint32_t pixels; } sv2_lab_args;
typedef struct { uint32_t pixels, bins; float l_weight; float minimum[3], step[3]; } sv2_lab_match_args;

static int sv2_check_f32(H3GPU *gpu, const h3_gpu_tensor *t, size_t count, NSString *what) {
    return h3_gpu_require_elements(gpu, t, count, what) && TENSOR(t).dtype == H3_GPU_F32;
}
static int sv2_check_bytes(H3GPU *gpu, const h3_gpu_tensor *t, size_t count, NSString *what) {
    return h3_gpu_require_elements(gpu, t, count, what) && TENSOR(t).dtype == H3_GPU_I8;
}
static int sv2_check_u32(H3GPU *gpu, const h3_gpu_tensor *t, size_t count, NSString *what) {
    return h3_gpu_require_elements(gpu, t, count, what) && TENSOR(t).dtype == H3_GPU_U32;
}

int h3_gpu_sv2_dequant_q8_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                               const h3_gpu_tensor *packed, const h3_gpu_tensor *scales,
                               const h3_gpu_tensor *biases, uint32_t rows,
                               uint32_t columns, uint32_t group) {
    H3GPU *gpu = GPU(opaque);
    if (columns % 4 || !group || columns % group) {
        h3_gpu_set_error(gpu, @"sv2 dequant: columns %u, group %u", columns, group);
        return 0;
    }
    size_t groups = (size_t)rows * (columns / group);
    if (!sv2_check_u32(gpu, packed, (size_t)rows * columns / 4, @"sv2 dequant packed") ||
        !sv2_check_bytes(gpu, scales, groups * 2, @"sv2 dequant scales") ||
        !sv2_check_bytes(gpu, biases, groups * 2, @"sv2 dequant biases") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * columns, @"sv2 dequant output")) return 0;
    sv2_dequant_args args = {rows, columns, group};
    return h3_gpu_dispatch_2d(gpu, @"h3_sv2_dequant_q8_bf16", columns / 4, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(packed).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(scales).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(biases).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

int h3_gpu_sv2_dequant_rows_int8_bf16(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *quantized,
                                      const h3_gpu_tensor *scales, uint32_t rows, uint32_t columns) {
    H3GPU *gpu = GPU(opaque);
    if (!rows || columns % 4) { h3_gpu_set_error(gpu, @"sv2 dequant rows: columns %u", columns); return 0; }
    if (!h3_gpu_require_i8(gpu, quantized, (size_t)rows * columns, @"sv2 dequant rows input") ||
        !h3_gpu_require_f32(gpu, scales, rows, @"sv2 dequant rows scales") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * columns, @"sv2 dequant rows output")) return 0;
    sv2_dequant_rows_args args = {rows, columns};
    return h3_gpu_dispatch_2d(gpu, @"h3_sv2_dequant_rows_int8_bf16", columns / 4, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(quantized).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(scales).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_sv2_quantize_rows_bf16_int8(h3_gpu *opaque, h3_gpu_tensor *quantized, h3_gpu_tensor *scales,
                                       const h3_gpu_tensor *input, uint32_t in_row0, uint32_t out_row0,
                                       uint32_t rows, uint32_t columns) {
    H3GPU *gpu = GPU(opaque);
    if (!rows || !columns) { h3_gpu_set_error(gpu, @"sv2 quantize rows: bad arguments"); return 0; }
    if (!h3_gpu_require_bf16(gpu, input, ((size_t)in_row0 + rows) * columns, @"sv2 quantize rows input") ||
        !h3_gpu_require_i8(gpu, quantized, ((size_t)out_row0 + rows) * columns, @"sv2 quantize rows output") ||
        !h3_gpu_require_f32(gpu, scales, (size_t)out_row0 + rows, @"sv2 quantize rows scales") ||
        !h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, @"h3_sv2_quantize_rows_bf16_int8");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu, @"device cannot dispatch the sv2 row quantizer");
        return 0;
    }
    sv2_quant_rows_args args = {rows, columns, in_row0, out_row0};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(quantized).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(scales).buffer offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_sv2_quantize_rows_bf16_int4(h3_gpu *opaque, h3_gpu_tensor *packed, h3_gpu_tensor *params,
                                       const h3_gpu_tensor *input, uint32_t in_row0, uint32_t out_row0,
                                       uint32_t rows, uint32_t columns) {
    H3GPU *gpu = GPU(opaque);
    if (!rows || !columns || columns % 64) { h3_gpu_set_error(gpu, @"sv2 quantize rows int4: columns %u", columns); return 0; }
    size_t groups = columns / 64;
    if (!h3_gpu_require_bf16(gpu, input, ((size_t)in_row0 + rows) * columns, @"sv2 int4 input") ||
        !h3_gpu_require_i8(gpu, packed, ((size_t)out_row0 + rows) * columns / 2, @"sv2 int4 packed") ||
        !h3_gpu_require_f32(gpu, params, ((size_t)out_row0 + rows) * groups * 2, @"sv2 int4 params")) return 0;
    sv2_quant_rows_args args = {rows, columns, in_row0, out_row0};
    return h3_gpu_dispatch_2d(gpu, @"h3_sv2_quantize_rows_bf16_int4", (uint32_t)groups, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(packed).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(params).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_sv2_dequant_rows_int4_bf16(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *packed,
                                      const h3_gpu_tensor *params, uint32_t rows, uint32_t columns) {
    H3GPU *gpu = GPU(opaque);
    if (!rows || !columns || columns % 64) { h3_gpu_set_error(gpu, @"sv2 dequant rows int4: columns %u", columns); return 0; }
    if (!h3_gpu_require_i8(gpu, packed, (size_t)rows * columns / 2, @"sv2 int4 packed") ||
        !h3_gpu_require_f32(gpu, params, (size_t)rows * (columns / 64) * 2, @"sv2 int4 params") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * columns, @"sv2 int4 output")) return 0;
    sv2_dequant_rows_args args = {rows, columns};
    return h3_gpu_dispatch_2d(gpu, @"h3_sv2_dequant_rows_int4_bf16", columns / 2, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(packed).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(params).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_sv2_rms_mod_bf16(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                            const h3_gpu_tensor *scale, const h3_gpu_tensor *shift,
                            uint32_t rows, uint32_t width, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, input, count, @"sv2 rms input") ||
        !h3_gpu_require_bf16(gpu, output, count, @"sv2 rms output") ||
        (scale && !sv2_check_f32(gpu, scale, width, @"sv2 rms scale")) ||
        (shift && !sv2_check_f32(gpu, shift, width, @"sv2 rms shift"))) return 0;
    sv2_rms_args args = {rows, width, scale != NULL, shift != NULL, epsilon};
    const h3_gpu_tensor *sc = scale ? scale : input, *sh = shift ? shift : input;
    return h3_gpu_dispatch_rows(gpu, @"h3_sv2_rms_mod_bf16", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(sc).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(sh).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

int h3_gpu_sv2_qkv_window_bf16(h3_gpu *opaque, h3_gpu_tensor *q_out, h3_gpu_tensor *k_out,
                               h3_gpu_tensor *v_out, const h3_gpu_tensor *qkv_vid,
                               const h3_gpu_tensor *qkv_txt, const h3_gpu_tensor *index,
                               const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
                               const h3_gpu_tensor *norm_q_vid, const h3_gpu_tensor *norm_k_vid,
                               const h3_gpu_tensor *norm_q_txt, const h3_gpu_tensor *norm_k_txt,
                               uint32_t rows, uint32_t heads, uint32_t head_dim,
                               uint32_t pairs, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    uint32_t dim = heads * head_dim;
    size_t count = (size_t)rows * dim;
    if (head_dim > 128 || head_dim % 2 || 2 * pairs > head_dim) {
        h3_gpu_set_error(gpu, @"sv2 window: head_dim %u, pairs %u", head_dim, pairs);
        return 0;
    }
    if (!h3_gpu_require_bf16(gpu, q_out, count, @"sv2 window q") ||
        !h3_gpu_require_bf16(gpu, k_out, count, @"sv2 window k") ||
        !h3_gpu_require_bf16(gpu, v_out, count, @"sv2 window v") ||
        !h3_gpu_require_elements(gpu, qkv_vid, 3 * (size_t)dim, @"sv2 window qkv_vid") ||
        !h3_gpu_require_elements(gpu, qkv_txt, 3 * (size_t)dim, @"sv2 window qkv_txt") ||
        !sv2_check_u32(gpu, index, rows, @"sv2 window index") ||
        !sv2_check_f32(gpu, rope_cos, (size_t)rows * pairs, @"sv2 window cos") ||
        !sv2_check_f32(gpu, rope_sin, (size_t)rows * pairs, @"sv2 window sin") ||
        !sv2_check_f32(gpu, norm_q_vid, head_dim, @"sv2 window norm_q_vid") ||
        !sv2_check_f32(gpu, norm_k_vid, head_dim, @"sv2 window norm_k_vid") ||
        !sv2_check_f32(gpu, norm_q_txt, head_dim, @"sv2 window norm_q_txt") ||
        !sv2_check_f32(gpu, norm_k_txt, head_dim, @"sv2 window norm_k_txt")) return 0;
    sv2_window_args args = {rows, heads, head_dim, dim, pairs, epsilon};
    return h3_gpu_dispatch_2d(gpu, @"h3_sv2_qkv_window_bf16", heads, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(qkv_vid).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(qkv_txt).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(index).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(norm_q_vid).buffer offset:0 atIndex:5];
            [encoder setBuffer:TENSOR(norm_k_vid).buffer offset:0 atIndex:6];
            [encoder setBuffer:TENSOR(norm_q_txt).buffer offset:0 atIndex:7];
            [encoder setBuffer:TENSOR(norm_k_txt).buffer offset:0 atIndex:8];
            [encoder setBuffer:TENSOR(q_out).buffer offset:0 atIndex:9];
            [encoder setBuffer:TENSOR(k_out).buffer offset:0 atIndex:10];
            [encoder setBuffer:TENSOR(v_out).buffer offset:0 atIndex:11];
            [encoder setBytes:&args length:sizeof(args) atIndex:12];
        });
}

int h3_gpu_sv2_txt_mean_bf16(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                             const h3_gpu_tensor *base, uint32_t windows,
                             uint32_t txt_rows, uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, output, (size_t)txt_rows * width, @"sv2 txt mean output") ||
        !h3_gpu_require_bf16(gpu, input, width, @"sv2 txt mean input") ||
        !sv2_check_u32(gpu, base, windows, @"sv2 txt mean base")) return 0;
    sv2_txt_mean_args args = {windows, txt_rows, width};
    return h3_gpu_dispatch_2d(gpu, @"h3_sv2_txt_mean_bf16", width, txt_rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(base).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_sv2_pad_f32(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                       uint32_t height, uint32_t width, uint32_t channels,
                       uint32_t top, uint32_t bottom, uint32_t left, uint32_t right) {
    H3GPU *gpu = GPU(opaque);
    uint32_t out_w = width + left + right, out_h = height + top + bottom;
    if (!sv2_check_f32(gpu, input, (size_t)height * width * channels, @"sv2 pad input") ||
        !sv2_check_f32(gpu, output, (size_t)out_h * out_w * channels, @"sv2 pad output")) return 0;
    sv2_pad_args args = {height, width, channels, top, bottom, left, right};
    return h3_gpu_dispatch_3d(gpu, @"h3_sv2_pad_f32", MTLSizeMake(channels, out_w, out_h),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

uint32_t h3_gpu_sv2_shuffle_frames(uint32_t frames, uint32_t temporal, int drop_head) {
    if (temporal <= 1) return frames;
    return drop_head && frames ? temporal * frames - 1 : temporal * frames;
}

int h3_gpu_sv2_shuffle_f32(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                           uint32_t height, uint32_t width, uint32_t channels,
                           uint32_t temporal) {
    return h3_gpu_sv2_shuffle3d_f32(opaque, output, input, 1, height, width, channels, temporal, 1);
}

int h3_gpu_sv2_shuffle3d_f32(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                             uint32_t frames, uint32_t height, uint32_t width, uint32_t channels,
                             uint32_t temporal, int drop_head) {
    H3GPU *gpu = GPU(opaque);
    uint32_t out_frames = h3_gpu_sv2_shuffle_frames(frames, temporal, drop_head);
    if (!frames || !temporal) { h3_gpu_set_error(gpu, @"sv2 shuffle: no frames"); return 0; }
    if (!sv2_check_f32(gpu, input, (size_t)frames * height * width * channels * 4 * temporal, @"sv2 shuffle input") ||
        !sv2_check_f32(gpu, output, (size_t)out_frames * height * width * 4 * channels, @"sv2 shuffle output")) return 0;
    sv2_shuffle_args args = {height, width, channels, temporal, frames, out_frames, drop_head != 0};
    return h3_gpu_dispatch_3d(gpu, @"h3_sv2_shuffle_f32", MTLSizeMake(channels, 2 * width, 2 * height * out_frames),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_sv2_pad3d_f32(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                         const h3_gpu_tensor *previous, uint32_t prev_frames,
                         uint32_t frames, uint32_t height, uint32_t width, uint32_t channels,
                         uint32_t front, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right) {
    H3GPU *gpu = GPU(opaque);
    uint32_t out_w = width + left + right, out_h = height + top + bottom;
    size_t plane = (size_t)height * width * channels;
    if (!frames || (previous && prev_frames < front)) { h3_gpu_set_error(gpu, @"sv2 pad3d: bad frames"); return 0; }
    if (!sv2_check_f32(gpu, input, (size_t)frames * plane, @"sv2 pad3d input") ||
        (previous && !sv2_check_f32(gpu, previous, (size_t)prev_frames * plane, @"sv2 pad3d previous")) ||
        !sv2_check_f32(gpu, output, (size_t)(front + frames) * out_h * out_w * channels, @"sv2 pad3d output")) return 0;
    sv2_pad3d_args args = {frames, height, width, channels, front, previous ? prev_frames : 0, previous != NULL,
                           top, bottom, left, right};
    const h3_gpu_tensor *prev = previous ? previous : input;
    return h3_gpu_dispatch_3d(gpu, @"h3_sv2_pad3d_f32", MTLSizeMake(channels, out_w, (size_t)out_h * (front + frames)),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(prev).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_sv2_group_norm_f32(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                              const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
                              uint32_t frames, uint32_t height, uint32_t width,
                              uint32_t channels, uint32_t groups, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)frames * height * width * channels;
    if (!groups || channels % groups) {
        h3_gpu_set_error(gpu, @"sv2 group norm: %u channels, %u groups", channels, groups);
        return 0;
    }
    if (!sv2_check_f32(gpu, input, count, @"sv2 group norm input") ||
        !sv2_check_f32(gpu, output, count, @"sv2 group norm output") ||
        !sv2_check_f32(gpu, weight, channels, @"sv2 group norm weight") ||
        !sv2_check_f32(gpu, bias, channels, @"sv2 group norm bias")) return 0;
    struct { uint32_t batch, depth, height, width, channels, groups; float epsilon; } args =
        {1, frames, height, width, channels, groups, epsilon};
    return h3_gpu_dispatch_rows(gpu, @"h3_sv2_group_norm_f32", frames * groups,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

int h3_gpu_sv2_bicubic_pixels_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *pixels, uint32_t in_w, uint32_t in_h,
                                  uint32_t in_stride, int bgra, uint32_t out_w, uint32_t out_h,
                                  uint32_t full_w, uint32_t full_h, float pad_value) {
    H3GPU *gpu = GPU(opaque);
    if (!sv2_check_bytes(gpu, pixels, (size_t)in_h * in_stride, @"sv2 bicubic pixels") ||
        !sv2_check_f32(gpu, output, (size_t)full_h * full_w * 3, @"sv2 bicubic output") ||
        out_w > full_w || out_h > full_h || !out_w || !out_h) return 0;
    sv2_bicubic_args args = {in_w, in_h, in_stride, bgra != 0, out_w, out_h, full_w, full_h,
                             (float)in_w / (float)out_w, (float)in_h / (float)out_h, pad_value};
    if (args.scale_x > 8.0f || args.scale_y > 8.0f) {
        h3_gpu_set_error(gpu, @"sv2 bicubic: shrinking by more than 8x is not supported");
        return 0;
    }
    return h3_gpu_dispatch_2d(gpu, @"h3_sv2_bicubic_pixels_f32", full_w, full_h,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(pixels).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_sv2_f32_to_pixels(h3_gpu *opaque, h3_gpu_tensor *pixels, const h3_gpu_tensor *input,
                             uint32_t full_w, uint32_t full_h, uint32_t crop_w,
                             uint32_t crop_h, uint32_t out_stride, int bgra) {
    H3GPU *gpu = GPU(opaque);
    if (!sv2_check_f32(gpu, input, (size_t)full_h * full_w * 3, @"sv2 pixels input") ||
        !sv2_check_bytes(gpu, pixels, (size_t)crop_h * out_stride, @"sv2 pixels output") ||
        crop_w > full_w || crop_h > full_h) return 0;
    sv2_pixels_out_args args = {full_w, full_h, crop_w, crop_h, out_stride, bgra != 0};
    return h3_gpu_dispatch_2d(gpu, @"h3_sv2_f32_to_pixels", crop_w, crop_h,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(pixels).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_sv2_wavelet_blur_f32(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                                uint32_t height, uint32_t width, int radius) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)height * width * 3;
    if (!sv2_check_f32(gpu, input, count, @"sv2 blur input") ||
        !sv2_check_f32(gpu, output, count, @"sv2 blur output")) return 0;
    sv2_blur_args args = {height, width, radius};
    return h3_gpu_dispatch_3d(gpu, @"h3_sv2_wavelet_blur_f32", MTLSizeMake(3, width, height),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_sv2_rgb_to_lab_f32(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                              uint32_t pixels) {
    H3GPU *gpu = GPU(opaque);
    if (!sv2_check_f32(gpu, input, (size_t)pixels * 3, @"sv2 lab input") ||
        !sv2_check_f32(gpu, output, (size_t)pixels * 3, @"sv2 lab output")) return 0;
    sv2_lab_args args = {pixels};
    return h3_gpu_dispatch_1d(gpu, @"h3_sv2_rgb_to_lab_f32", pixels,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_sv2_lab_match_to_rgb_f32(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *lab,
                                    const h3_gpu_tensor *luts, uint32_t pixels, uint32_t bins,
                                    float l_weight, const float minimum[3], const float step[3]) {
    H3GPU *gpu = GPU(opaque);
    if (!sv2_check_f32(gpu, lab, (size_t)pixels * 3, @"sv2 lab match input") ||
        !sv2_check_f32(gpu, output, (size_t)pixels * 3, @"sv2 lab match output") ||
        !sv2_check_f32(gpu, luts, (size_t)bins * 3, @"sv2 lab match luts")) return 0;
    sv2_lab_match_args args = {pixels, bins, l_weight,
                               {minimum[0], minimum[1], minimum[2]}, {step[0], step[1], step[2]}};
    return h3_gpu_dispatch_1d(gpu, @"h3_sv2_lab_match_to_rgb_f32", pixels,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(lab).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(luts).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}


/* --- SeedVR2 VAE grid kernels --- */

typedef struct { uint32_t height, width, channels, padded, groups, guard, chunks, chunk_pixels, silu; float epsilon; } sv2_ggn_args;
typedef struct { uint32_t height, width, channels, padded, in_padded, temporal, guard, in_guard, out_frames, drop_head; } sv2_gshuffle_args;
typedef struct { uint32_t height, width, padded, guard, in_width, in_guard, phase_y, phase_x, frames, in_height, stride_t, phase_t; } sv2_gselect_args;

int h3_gpu_sv2_grid_group_norm_bf16(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                                    const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
                                    uint32_t height, uint32_t width, uint32_t channels,
                                    uint32_t padded, uint32_t guard, uint32_t x_rows,
                                    uint32_t groups, float epsilon, int silu) {
    H3GPU *gpu = GPU(opaque);
    uint32_t vectors = padded / 4;
    if (!groups || groups > 64 || padded % 64 || channels > padded || channels % groups ||
        (channels / groups) % 4 || !vectors || vectors > 256 || 256 % vectors) {
        h3_gpu_set_error(gpu, @"sv2 grid group norm: %u channels in %u, %u groups", channels, padded, groups);
        return 0;
    }
    size_t count = (size_t)x_rows * padded;
    if (!h3_gpu_require_bf16(gpu, input, count, @"sv2 grid gn input") ||
        !h3_gpu_require_bf16(gpu, output, count, @"sv2 grid gn output") ||
        !sv2_check_f32(gpu, weight, channels, @"sv2 grid gn weight") ||
        !sv2_check_f32(gpu, bias, channels, @"sv2 grid gn bias") ||
        guard + (size_t)(height + 2) * (width + 2) > x_rows || !h3_gpu_require_command(gpu)) return 0;
    uint32_t pixels = height * width, lanes = 256 / vectors;
    uint32_t chunks = MIN((uint32_t)64, (pixels + lanes - 1) / lanes);
    uint32_t chunk_pixels = (pixels + chunks - 1) / chunks;
    chunks = (pixels + chunk_pixels - 1) / chunk_pixels;
    sv2_ggn_args args = {height, width, channels, padded, groups, guard, chunks, chunk_pixels, silu != 0, epsilon};
    size_t partial_bytes = (size_t)chunks * groups * sizeof(float);
    id<MTLBuffer> sums = [gpu.device newBufferWithLength:partial_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> squares = [gpu.device newBufferWithLength:partial_bytes options:MTLResourceStorageModePrivate];
    if (!sums || !squares) { h3_gpu_set_error(gpu, @"sv2 grid gn: cannot allocate the partial sums"); return 0; }
    NSString *names[3] = {@"h3_sv2_grid_gn_sum_bf16", @"h3_sv2_grid_gn_sq_bf16", @"h3_sv2_grid_gn_apply_bf16"};
    for (int pass = 0; pass < 3; pass++) {
        id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, names[pass]);
        if (!pipeline) return 0;
        if (pipeline.maxTotalThreadsPerThreadgroup < 256) {
            h3_gpu_set_error(gpu, @"%@ needs 256 threads", names[pass]);
            return 0;
        }
        @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            if (pass == 0) {
                [encoder setBuffer:sums offset:0 atIndex:1];
                [encoder setBytes:&args length:sizeof(args) atIndex:2];
            } else if (pass == 1) {
                [encoder setBuffer:sums offset:0 atIndex:1];
                [encoder setBuffer:squares offset:0 atIndex:2];
                [encoder setBytes:&args length:sizeof(args) atIndex:3];
            } else {
                [encoder setBuffer:sums offset:0 atIndex:1];
                [encoder setBuffer:squares offset:0 atIndex:2];
                [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:3];
                [encoder setBuffer:TENSOR(bias).buffer offset:0 atIndex:4];
                [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:5];
                [encoder setBytes:&args length:sizeof(args) atIndex:6];
            }
            [encoder dispatchThreadgroups:MTLSizeMake(chunks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
        }
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches += 3;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_sv2_grid_shuffle_bf16(h3_gpu *opaque, h3_gpu_tensor *output, uint32_t out_padded,
                                 uint32_t out_guard, const h3_gpu_tensor *input, uint32_t in_height,
                                 uint32_t in_width, uint32_t in_padded, uint32_t in_guard,
                                 uint32_t channels, uint32_t temporal) {
    return h3_gpu_sv2_grid_shuffle3d_bf16(opaque, output, out_padded, out_guard, input, 1, in_height, in_width,
                                          in_padded, in_guard, channels, temporal, 1);
}

int h3_gpu_sv2_grid_shuffle3d_bf16(h3_gpu *opaque, h3_gpu_tensor *output, uint32_t out_padded,
                                   uint32_t out_guard, const h3_gpu_tensor *input, uint32_t in_frames,
                                   uint32_t in_height, uint32_t in_width, uint32_t in_padded,
                                   uint32_t in_guard, uint32_t channels, uint32_t temporal, int drop_head) {
    H3GPU *gpu = GPU(opaque);
    if (!in_frames || !temporal || channels > out_padded || channels * 4 * temporal > in_padded) {
        h3_gpu_set_error(gpu, @"sv2 grid shuffle: bad channels or frames");
        return 0;
    }
    uint32_t out_frames = h3_gpu_sv2_shuffle_frames(in_frames, temporal, drop_head);
    uint32_t hp = 2 * in_height + 2, wp = 2 * in_width + 2;
    if (!h3_gpu_require_bf16(gpu, input, ((size_t)in_guard + (size_t)in_frames * (in_height + 2) * (in_width + 2)) * in_padded, @"sv2 grid shuffle input") ||
        !h3_gpu_require_bf16(gpu, output, ((size_t)out_guard + (size_t)out_frames * hp * wp) * out_padded, @"sv2 grid shuffle output")) return 0;
    sv2_gshuffle_args args = {in_height, in_width, channels, out_padded, in_padded, temporal, out_guard, in_guard,
                              out_frames, drop_head != 0};
    return h3_gpu_dispatch_3d(gpu, @"h3_sv2_grid_shuffle_bf16", MTLSizeMake(out_padded, wp, (size_t)hp * out_frames),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_sv2_grid_select_bf16(h3_gpu *opaque, h3_gpu_tensor *output, uint32_t out_height,
                                uint32_t out_width, uint32_t out_guard, const h3_gpu_tensor *input,
                                uint32_t in_height, uint32_t in_width, uint32_t in_guard,
                                uint32_t padded, uint32_t phase_y, uint32_t phase_x) {
    return h3_gpu_sv2_grid_select3d_bf16(opaque, output, 1, out_height, out_width, out_guard, input, 1,
                                         in_height, in_width, in_guard, padded, 1, 0, phase_y, phase_x);
}

int h3_gpu_sv2_grid_select3d_bf16(h3_gpu *opaque, h3_gpu_tensor *output, uint32_t out_frames,
                                  uint32_t out_height, uint32_t out_width, uint32_t out_guard,
                                  const h3_gpu_tensor *input, uint32_t in_frames, uint32_t in_height,
                                  uint32_t in_width, uint32_t in_guard, uint32_t padded,
                                  uint32_t stride_t, uint32_t phase_t, uint32_t phase_y, uint32_t phase_x) {
    H3GPU *gpu = GPU(opaque);
    if (!out_frames || !stride_t || (out_frames - 1) * stride_t + phase_t >= in_frames ||
        2 * (out_height - 1) + phase_y >= in_height || 2 * (out_width - 1) + phase_x >= in_width) {
        h3_gpu_set_error(gpu, @"sv2 grid select: the output reads outside the input");
        return 0;
    }
    uint32_t hp = out_height + 2, wp = out_width + 2;
    if (!h3_gpu_require_bf16(gpu, input, ((size_t)in_guard + (size_t)in_frames * (in_height + 2) * (in_width + 2)) * padded, @"sv2 grid select input") ||
        !h3_gpu_require_bf16(gpu, output, ((size_t)out_guard + (size_t)out_frames * hp * wp) * padded, @"sv2 grid select output")) return 0;
    sv2_gselect_args args = {out_height, out_width, padded, out_guard, in_width, in_guard, phase_y, phase_x,
                             out_frames, in_height, stride_t, phase_t};
    return h3_gpu_dispatch_3d(gpu, @"h3_sv2_grid_select_bf16", MTLSizeMake(padded, wp, (size_t)hp * out_frames),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}
