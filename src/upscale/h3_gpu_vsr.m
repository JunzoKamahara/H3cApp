/* GPU kernels of the SwiftVR upscaler (declared in h3_gpu_vsr.h; shaders in h3_shaders_vsr.metal). Split out of h3_gpu.m:
 * it uses that file's internals through h3_gpu_internal.h and registers its
 * kernel names at load time, so a program that does not link the upscale
 * module builds no pipelines for it. */
#include "h3_gpu_internal.h"
#include "h3_gpu_vsr.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* The tapconv kernel's tile shapes (h3_shaders.metal instantiations):
 * rows x channels per threadgroup, threads. H3_TAPCONV_KERNEL names one
 * (benchmarks); the default is the first. */
typedef struct { const char *name; uint32_t rt, ct, threads; } vsr_tapconv_variant;
static const vsr_tapconv_variant vsr_tapconv_variants[] = {
    {"h3_vsr_tapconv_bf16_nax", 128, 64, 128},
    {"h3_vsr_tapconv_bf16_nax_r128c128", 128, 128, 256},
    {"h3_vsr_tapconv_bf16_nax_r256c64", 256, 64, 256},
    {"h3_vsr_tapconv_bf16_nax_r64c128", 64, 128, 128},
    {"h3_vsr_tapconv_bf16_nax_r128c64k128", 128, 64, 128},
};

static const char *const vsr_kernels[] = {
    "h3_vsr_pad_f32",
    "h3_vsr_upsample2_f32",
    "h3_vsr_space_to_depth_f32",
    "h3_vsr_depth_to_space_f32",
    "h3_vsr_split_frames_f32",
    "h3_vsr_clamp_tanh_f32",
    "h3_vsr_pixels_to_f32",
    "h3_vsr_f32_to_pixels",
    "h3_vsr_tile_accumulate_f32",
    "h3_vsr_ln_mod_bf16",
    "h3_vsr_qkv_prepare_bf16",
    "h3_vsr_gather_rows_bf16",
    "h3_vsr_bias_gate_add_bf16",
    "h3_vsr_gelu_bias_bf16",
    "h3_vsr_patchify_bf16",
    "h3_vsr_pack_grid_bf16",
    "h3_vsr_unpack_grid_f32",
    "h3_vsr_zero_border_bf16",
    "h3_vsr_zero_rows_bf16",
    "h3_vsr_scatter_rows_bf16",
    "h3_vsr_gelu_quantize_int8_rows",
    "h3_vsr_channel_sums_f32",
    "h3_vsr_add_channel_offsets_f32",
    "h3_vsr_pack_grid_s2d_bf16",
    "h3_vsr_unpack_grid_d2s_f32",
    "h3_vsr_grid_upsample2_bf16",
    "h3_vsr_grid_split_frames_bf16",
    "h3_vsr_grid_select_bf16",
    "h3_vsr_pack_grid_clamp_bf16",
    "h3_vsr_unpatchify_sub_f32",
};

static const char *vsr_tensor_kernels[sizeof(vsr_tapconv_variants) / sizeof(*vsr_tapconv_variants)];

__attribute__((constructor))
static void h3_gpu_vsr_register(void) {
    for (size_t i = 0; i < sizeof(vsr_tapconv_variants) / sizeof(*vsr_tapconv_variants); i++)
        vsr_tensor_kernels[i] = vsr_tapconv_variants[i].name;
    h3_gpu_register_kernels(vsr_kernels, sizeof(vsr_kernels) / sizeof(*vsr_kernels), 0);
    h3_gpu_register_kernels(vsr_tensor_kernels, sizeof(vsr_tensor_kernels) / sizeof(*vsr_tensor_kernels), 1);
}

/* --- One-step VSR helpers (channels-last F32; h3_shaders.metal) --- */

typedef struct {
    uint32_t batch, depth, height, width, channels;
    uint32_t depth_front, pad_h, pad_w;
    uint32_t has_previous;
} vsr_pad_args;

typedef struct { uint32_t batch, depth, height, width, channels; } vsr_shape_args;
typedef struct { uint32_t elements; } vsr_count_args;

static int vsr_check_f32(H3GPU *gpu, const h3_gpu_tensor *tensor, size_t count,
                         NSString *what) {
    return h3_gpu_require_elements(gpu, tensor, count, what) &&
           TENSOR(tensor).dtype == H3_GPU_F32;
}

int h3_gpu_vsr_pad_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input, const h3_gpu_tensor *previous,
                       uint32_t batch, uint32_t depth, uint32_t height,
                       uint32_t width, uint32_t channels, uint32_t depth_front,
                       uint32_t pad_h, uint32_t pad_w) {
    H3GPU *gpu = GPU(opaque);
    if (!batch || !depth || !height || !width || !channels) return 0;
    uint32_t out_depth = depth + depth_front;
    uint32_t out_h = height + 2 * pad_h, out_w = width + 2 * pad_w;
    size_t input_count = (size_t)batch * depth * height * width * channels;
    size_t output_count = (size_t)batch * out_depth * out_h * out_w * channels;
    size_t previous_count = (size_t)batch * depth_front * height * width * channels;
    if (!vsr_check_f32(gpu, input, input_count, @"VSR pad input") ||
        !vsr_check_f32(gpu, output, output_count, @"VSR pad output") ||
        (previous && !vsr_check_f32(gpu, previous, previous_count, @"VSR pad previous")))
        return 0;
    vsr_pad_args args = {batch, depth, height, width, channels, depth_front,
                         pad_h, pad_w, previous != NULL};
    const h3_gpu_tensor *previous_or_input = previous ? previous : input;
    return h3_gpu_dispatch_3d(gpu, @"h3_vsr_pad_f32",
        MTLSizeMake(channels, out_w, (NSUInteger)batch * out_depth * out_h),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
            [encoder setBuffer:TENSOR(previous_or_input).buffer offset:0 atIndex:3];
        });
}

static int vsr_shape_dispatch(H3GPU *gpu, NSString *name, h3_gpu_tensor *output,
                              const h3_gpu_tensor *input, size_t input_count,
                              size_t output_count, vsr_shape_args args,
                              MTLSize grid) {
    if (!vsr_check_f32(gpu, input, input_count, name) ||
        !vsr_check_f32(gpu, output, output_count, name)) return 0;
    return h3_gpu_dispatch_3d(gpu, name, grid,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_vsr_upsample2_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input, uint32_t batch,
                             uint32_t depth, uint32_t height, uint32_t width,
                             uint32_t channels) {
    if (!batch || !depth || !height || !width || !channels) return 0;
    size_t count = (size_t)batch * depth * height * width * channels;
    vsr_shape_args args = {batch, depth, height, width, channels};
    return vsr_shape_dispatch(GPU(opaque), @"h3_vsr_upsample2_f32", output, input,
                              count, count * 4, args,
                              MTLSizeMake(channels, (NSUInteger)width * 2,
                                          (NSUInteger)batch * depth * height * 2));
}

int h3_gpu_vsr_space_to_depth_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *input, uint32_t batch,
                                  uint32_t depth, uint32_t height, uint32_t width,
                                  uint32_t channels) {
    if (!batch || !depth || height % 2 || width % 2 || !height || !width || !channels) return 0;
    size_t count = (size_t)batch * depth * height * width * channels;
    vsr_shape_args args = {batch, depth, height, width, channels};
    return vsr_shape_dispatch(GPU(opaque), @"h3_vsr_space_to_depth_f32", output, input,
                              count, count, args,
                              MTLSizeMake((NSUInteger)channels * 4, width / 2,
                                          (NSUInteger)batch * depth * (height / 2)));
}

int h3_gpu_vsr_depth_to_space_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *input, uint32_t batch,
                                  uint32_t depth, uint32_t height, uint32_t width,
                                  uint32_t channels) {
    if (!batch || !depth || !height || !width || !channels) return 0;
    size_t count = (size_t)batch * depth * height * width * channels * 4;
    vsr_shape_args args = {batch, depth, height, width, channels};
    return vsr_shape_dispatch(GPU(opaque), @"h3_vsr_depth_to_space_f32", output, input,
                              count, count, args,
                              MTLSizeMake(channels, (NSUInteger)width * 2,
                                          (NSUInteger)batch * depth * height * 2));
}

int h3_gpu_vsr_split_frames_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                                const h3_gpu_tensor *input, uint32_t batch,
                                uint32_t depth, uint32_t height, uint32_t width,
                                uint32_t channels) {
    if (!batch || !depth || !height || !width || !channels) return 0;
    size_t count = (size_t)batch * depth * height * width * channels * 2;
    vsr_shape_args args = {batch, depth, height, width, channels};
    return vsr_shape_dispatch(GPU(opaque), @"h3_vsr_split_frames_f32", output, input,
                              count, count, args,
                              MTLSizeMake(channels, width,
                                          (NSUInteger)batch * depth * 2 * height));
}

int h3_gpu_vsr_clamp_tanh_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                              const h3_gpu_tensor *input, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!vsr_check_f32(gpu, input, elements, @"VSR clamp input") ||
        !vsr_check_f32(gpu, output, elements, @"VSR clamp output")) return 0;
    vsr_count_args args = {elements};
    return h3_gpu_dispatch_1d(gpu, @"h3_vsr_clamp_tanh_f32", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

typedef struct {
    uint32_t frames, in_w, in_h, in_stride, bgra, out_w, out_h, tile_x, tile_y, tile_w, tile_h;
} vsr_pixels_in_args;

typedef struct {
    uint32_t frames, full_w, full_h, crop_w, crop_h, out_stride, bgra, round_mode, first_frame;
} vsr_pixels_out_args;

typedef struct {
    uint32_t frames, first_frame, tile_w, tile_h, tile_x, tile_y, out_w, out_h;
} vsr_tile_args;

int h3_gpu_vsr_pixels_to_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                             const h3_gpu_tensor *pixels, uint32_t frames,
                             uint32_t in_w, uint32_t in_h, uint32_t in_stride,
                             int bgra, uint32_t out_w, uint32_t out_h,
                             uint32_t tile_x, uint32_t tile_y,
                             uint32_t tile_w, uint32_t tile_h) {
    H3GPU *gpu = GPU(opaque);
    if (!frames || !in_w || !in_h || !out_w || !out_h || !tile_w || !tile_h ||
        in_stride < in_w * (bgra ? 4u : 3u)) return 0;
    size_t input_count = (size_t)frames * in_h * in_stride;
    size_t output_count = (size_t)frames * tile_h * tile_w * 3;
    if (!h3_gpu_require_elements(gpu, pixels, input_count, @"VSR pixels in") ||
        TENSOR(pixels).dtype != H3_GPU_I8 ||
        !vsr_check_f32(gpu, output, output_count, @"VSR pixels out")) return 0;
    vsr_pixels_in_args args = {frames, in_w, in_h, in_stride, bgra ? 1u : 0u,
                               out_w, out_h, tile_x, tile_y, tile_w, tile_h};
    return h3_gpu_dispatch_3d(gpu, @"h3_vsr_pixels_to_f32",
        MTLSizeMake(tile_w, tile_h, frames),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(pixels).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_vsr_f32_to_pixels(h3_gpu *opaque, h3_gpu_tensor *pixels,
                             const h3_gpu_tensor *input, uint32_t first_frame,
                             uint32_t frames, uint32_t full_w, uint32_t full_h,
                             uint32_t crop_w, uint32_t crop_h, uint32_t out_stride,
                             int bgra, int round_mode) {
    H3GPU *gpu = GPU(opaque);
    if (!frames || !full_w || !full_h || !crop_w || !crop_h || crop_w > full_w ||
        crop_h > full_h || out_stride < crop_w * (bgra ? 4u : 3u)) return 0;
    size_t input_count = ((size_t)first_frame + frames) * full_h * full_w * 3;
    size_t output_count = (size_t)frames * crop_h * out_stride;
    if (!vsr_check_f32(gpu, input, input_count, @"VSR frames in") ||
        !h3_gpu_require_elements(gpu, pixels, output_count, @"VSR pixels out") ||
        TENSOR(pixels).dtype != H3_GPU_I8) return 0;
    vsr_pixels_out_args args = {frames, full_w, full_h, crop_w, crop_h, out_stride,
                                bgra ? 1u : 0u, round_mode ? 1u : 0u, first_frame};
    return h3_gpu_dispatch_3d(gpu, @"h3_vsr_f32_to_pixels",
        MTLSizeMake(crop_w, crop_h, frames),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(pixels).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_vsr_tile_accumulate_f32(h3_gpu *opaque, h3_gpu_tensor *composite,
                                   uint32_t out_w, uint32_t out_h,
                                   const h3_gpu_tensor *tile, uint32_t first_frame,
                                   uint32_t frames, uint32_t tile_w, uint32_t tile_h,
                                   uint32_t tile_x, uint32_t tile_y,
                                   const h3_gpu_tensor *weights) {
    H3GPU *gpu = GPU(opaque);
    if (!frames || !out_w || !out_h || !tile_w || !tile_h) return 0;
    if (!vsr_check_f32(gpu, composite, (size_t)frames * out_h * out_w * 3, @"VSR composite") ||
        !vsr_check_f32(gpu, tile, ((size_t)first_frame + frames) * tile_h * tile_w * 3, @"VSR tile") ||
        !vsr_check_f32(gpu, weights, (size_t)tile_h * tile_w, @"VSR tile weights")) return 0;
    vsr_tile_args args = {frames, first_frame, tile_w, tile_h, tile_x, tile_y, out_w, out_h};
    return h3_gpu_dispatch_3d(gpu, @"h3_vsr_tile_accumulate_f32",
        MTLSizeMake(tile_w, tile_h, frames),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(composite).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(tile).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(weights).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_zero_f32(h3_gpu *opaque, h3_gpu_tensor *x, size_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!vsr_check_f32(gpu, x, elements, @"zero") || !h3_gpu_require_command(gpu)) return 0;
    @autoreleasepool {
        id<MTLBlitCommandEncoder> blit = [gpu.command blitCommandEncoder];
        [blit fillBuffer:TENSOR(x).buffer range:NSMakeRange(0, elements * sizeof(float)) value:0];
        [blit endEncoding];
    }
    return 1;
}


/* --- One-step VSR: Wan block kernels on BF16 (h3_shaders.metal) --- */

typedef struct { uint32_t rows, width; float epsilon; uint32_t has_scale, has_shift, has_weight, has_bias; } vsr_ln_args;
typedef struct { uint32_t rows, dim, heads, head_dim, segments, has_bias, norm_q, norm_k, rope; float epsilon; } vsr_qkv_args;
typedef struct { uint32_t rows, width, in_stride, col_offset; } vsr_rows_args;
typedef struct { uint32_t rows, width, has_residual, has_bias, has_gate; } vsr_bias_gate_args;
typedef struct { uint32_t frames, height, width, channels; } vsr_patch_args;

static int vsr_check_f32_vector(H3GPU *gpu, const h3_gpu_tensor *tensor, size_t count,
                                NSString *what) {
    return tensor == NULL || vsr_check_f32(gpu, tensor, count, what);
}

int h3_gpu_vsr_ln_mod_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, const h3_gpu_tensor *scale,
                           const h3_gpu_tensor *shift, const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows, uint32_t width,
                           float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, input, count, @"VSR LN input") ||
        !h3_gpu_require_bf16(gpu, output, count, @"VSR LN output") ||
        !vsr_check_f32_vector(gpu, scale, width, @"VSR LN scale") ||
        !vsr_check_f32_vector(gpu, shift, width, @"VSR LN shift") ||
        !vsr_check_f32_vector(gpu, weight, width, @"VSR LN weight") ||
        !vsr_check_f32_vector(gpu, bias, width, @"VSR LN bias")) return 0;
    vsr_ln_args args = {rows, width, epsilon, scale != NULL, shift != NULL,
                        weight != NULL, bias != NULL};
    const h3_gpu_tensor *s = scale ? scale : input, *sh = shift ? shift : input;
    const h3_gpu_tensor *w = weight ? weight : input, *b = bias ? bias : input;
    return h3_gpu_dispatch_rows(gpu, @"h3_vsr_ln_mod_bf16", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(s).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(sh).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(w).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(b).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
        });
}

int h3_gpu_vsr_qkv_prepare_bf16(h3_gpu *opaque, h3_gpu_tensor *qkv,
                                const h3_gpu_tensor *bias,
                                const h3_gpu_tensor *norm_q_weight,
                                const h3_gpu_tensor *norm_k_weight,
                                const h3_gpu_tensor *rope_cos,
                                const h3_gpu_tensor *rope_sin, uint32_t rows,
                                uint32_t dim, uint32_t heads, uint32_t head_dim,
                                uint32_t segments, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    if (!rows || !dim || !heads || !head_dim || head_dim % 2 || heads * head_dim != dim ||
        segments < 1 || segments > 3) return 0;
    size_t count = (size_t)rows * segments * dim;
    if (!h3_gpu_require_bf16(gpu, qkv, count, @"VSR qkv") ||
        !vsr_check_f32_vector(gpu, bias, (size_t)segments * dim, @"VSR qkv bias") ||
        !vsr_check_f32_vector(gpu, norm_q_weight, dim, @"VSR norm_q") ||
        !vsr_check_f32_vector(gpu, norm_k_weight, dim, @"VSR norm_k") ||
        !vsr_check_f32_vector(gpu, rope_cos, (size_t)rows * (head_dim / 2), @"VSR rope cos") ||
        !vsr_check_f32_vector(gpu, rope_sin, (size_t)rows * (head_dim / 2), @"VSR rope sin") ||
        (rope_cos == NULL) != (rope_sin == NULL)) return 0;
    vsr_qkv_args args = {rows, dim, heads, head_dim, segments, bias != NULL,
                         norm_q_weight != NULL, norm_k_weight != NULL && segments > 1,
                         rope_cos != NULL, epsilon};
    /* Unused slots are bound to an existing buffer; the kernel never reads them. */
    const h3_gpu_tensor *b = bias ? bias : qkv, *nq = norm_q_weight ? norm_q_weight : qkv;
    const h3_gpu_tensor *nk = norm_k_weight ? norm_k_weight : qkv;
    const h3_gpu_tensor *rc = rope_cos ? rope_cos : qkv, *rs = rope_sin ? rope_sin : qkv;
    return h3_gpu_dispatch_rows(gpu, @"h3_vsr_qkv_prepare_bf16", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(b).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(nq).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(nk).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(rc).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(rs).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
        });
}

int h3_gpu_vsr_gather_rows_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                                const h3_gpu_tensor *input, const h3_gpu_tensor *index,
                                uint32_t rows, uint32_t width, uint32_t in_stride,
                                uint32_t col_offset) {
    H3GPU *gpu = GPU(opaque);
    if (!rows || !width || in_stride < col_offset + width) return 0;
    if (!h3_gpu_require_bf16(gpu, output, (size_t)rows * width, @"VSR gather output") ||
        !h3_gpu_require_elements(gpu, index, rows, @"VSR gather index") ||
        TENSOR(index).dtype != H3_GPU_U32 ||
        !h3_gpu_require_elements(gpu, input, 1, @"VSR gather input") ||
        TENSOR(input).dtype != H3_GPU_BF16) return 0;
    vsr_rows_args args = {rows, width, in_stride, col_offset};
    return h3_gpu_dispatch_2d(gpu, @"h3_vsr_gather_rows_bf16", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(index).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_vsr_bias_gate_add_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *y, const h3_gpu_tensor *bias,
                                  const h3_gpu_tensor *gate, const h3_gpu_tensor *residual,
                                  uint32_t rows, uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, y, count, @"VSR gate y") ||
        !h3_gpu_require_bf16(gpu, output, count, @"VSR gate output") ||
        (residual && !h3_gpu_require_bf16(gpu, residual, count, @"VSR gate residual")) ||
        !vsr_check_f32_vector(gpu, bias, width, @"VSR gate bias") ||
        !vsr_check_f32_vector(gpu, gate, width, @"VSR gate gate")) return 0;
    vsr_bias_gate_args args = {rows, width, residual != NULL, bias != NULL, gate != NULL};
    const h3_gpu_tensor *b = bias ? bias : y, *g = gate ? gate : y, *r = residual ? residual : y;
    return h3_gpu_dispatch_2d(gpu, @"h3_vsr_bias_gate_add_bf16", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(y).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(b).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(g).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(r).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
            [encoder setBytes:&args length:sizeof(args) atIndex:5];
        });
}

int h3_gpu_vsr_gelu_bias_bf16(h3_gpu *opaque, h3_gpu_tensor *x, const h3_gpu_tensor *bias,
                              uint32_t rows, uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, x, (size_t)rows * width, @"VSR gelu x") ||
        !vsr_check_f32(gpu, bias, width, @"VSR gelu bias")) return 0;
    vsr_rows_args args = {rows, width, width, 0};
    return h3_gpu_dispatch_2d(gpu, @"h3_vsr_gelu_bias_bf16", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(x).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(bias).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_vsr_patchify_bf16(h3_gpu *opaque, h3_gpu_tensor *tokens,
                             const h3_gpu_tensor *latent, uint32_t frames,
                             uint32_t height, uint32_t width, uint32_t channels) {
    H3GPU *gpu = GPU(opaque);
    if (!frames || height % 2 || width % 2 || !height || !width || !channels) return 0;
    size_t count = (size_t)frames * height * width * channels;
    if (!vsr_check_f32(gpu, latent, count, @"VSR patchify latent") ||
        !h3_gpu_require_bf16(gpu, tokens, count, @"VSR patchify tokens")) return 0;
    vsr_patch_args args = {frames, height, width, channels};
    return h3_gpu_dispatch_3d(gpu, @"h3_vsr_patchify_bf16",
        MTLSizeMake((NSUInteger)channels * 4, width / 2, (NSUInteger)frames * (height / 2)),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(latent).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(tokens).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_vsr_unpatchify_sub_f32(h3_gpu *opaque, h3_gpu_tensor *z_hq,
                                  const h3_gpu_tensor *z_lq, const h3_gpu_tensor *v,
                                  const h3_gpu_tensor *bias, uint32_t frames,
                                  uint32_t height, uint32_t width, uint32_t channels) {
    H3GPU *gpu = GPU(opaque);
    if (!frames || height % 2 || width % 2 || !height || !width || !channels) return 0;
    size_t count = (size_t)frames * height * width * channels;
    if (!vsr_check_f32(gpu, z_lq, count, @"VSR unpatchify z_lq") ||
        !vsr_check_f32(gpu, z_hq, count, @"VSR unpatchify z_hq") ||
        !h3_gpu_require_bf16(gpu, v, count, @"VSR unpatchify v") ||
        !vsr_check_f32(gpu, bias, (size_t)channels * 4, @"VSR unpatchify bias")) return 0;
    vsr_patch_args args = {frames, height, width, channels};
    return h3_gpu_dispatch_3d(gpu, @"h3_vsr_unpatchify_sub_f32",
        MTLSizeMake(channels, width, (NSUInteger)frames * height),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(z_lq).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(v).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(z_hq).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

/* ---- VSR: ReAE convolutions on TensorOps (see h3_shaders.metal) ---- */

typedef struct {
    uint32_t rows, x_rows, guard, in_ch, out_ch, taps, inputs, relu, has_bias, has_residual;
    int32_t offsets[27];
    uint32_t x1_rows, tap_mask, order;
} vsr_tapconv_args;

/* A variant's row tile must also fit the rows after `guard + rows` (the
 * tail): a tile of rt rows reads up to rt + max_offset past its start. */
static const vsr_tapconv_variant *vsr_tapconv_variant_for(uint32_t in_ch, uint32_t out_ch, uint32_t rows,
                                                          uint32_t guard, uint32_t x_rows, int32_t max_offset) {
    /* The other shapes are within a few per cent of the first alone and
     * made no measurable difference inside a SeedVR2 window (bench_tapconv
     * and UPSCALE_PLAN appendix A.10, 2026-11-07): they stay for the
     * benchmark, H3_TAPCONV_KERNEL picks one. */
    const char *want = getenv("H3_TAPCONV_KERNEL");
    if (want)
        for (size_t i = 0; i < sizeof(vsr_tapconv_variants) / sizeof(*vsr_tapconv_variants); i++) {
            const vsr_tapconv_variant *v = &vsr_tapconv_variants[i];
            if (strcmp(want, v->name) || out_ch % v->ct || in_ch % (strstr(v->name, "k128") ? 128u : 64u)) continue;
            uint32_t last_tile_end = (rows + v->rt - 1) / v->rt * v->rt;
            if ((int64_t)guard + last_tile_end + (max_offset > 0 ? max_offset : 0) <= (int64_t)x_rows) return v;
        }
    return &vsr_tapconv_variants[0];
}
typedef struct { uint32_t frames, height, width, channels, padded_channels, guard; } vsr_grid_args;


/* x1 as a second input (w1, every tap; state_rows 0) or as the causal state
 * of the taps in tap_mask (w0's weights, state_rows its rows). */
static int vsr_tapconv(H3GPU *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *x0, const h3_gpu_tensor *x1, uint32_t state_rows,
                       const h3_gpu_tensor *w0, const h3_gpu_tensor *w1,
                       const h3_gpu_tensor *bias, const h3_gpu_tensor *residual,
                       uint32_t rows, uint32_t x_rows, uint32_t guard,
                       uint32_t in_ch, uint32_t out_ch, uint32_t taps,
                       const int32_t *offsets, uint32_t tap_mask, int relu) {
    if (!gpu.tensorOpsEnabled) { h3_gpu_set_error(gpu, @"VSR tapconv needs TensorOps"); return 0; }
    int state = state_rows != 0;
    if (!rows || !taps || taps > 27 || in_ch % 64 || out_ch % 64 || !offsets ||
        guard + rows + 128 > x_rows || (state && !x1) || (!state && tap_mask) ||
        (tap_mask >> taps) != 0) {   /* tiles over 128 rows are clamped to the tail below */
        h3_gpu_set_error(gpu, @"VSR tapconv: bad arguments");
        return 0;
    }
    for (uint32_t t = 0; t < taps; t++) {
        uint32_t limit = ((tap_mask >> t) & 1u) ? state_rows : x_rows;
        if (offsets[t] < -(int32_t)guard || (int64_t)guard + rows + 128 + offsets[t] > (int64_t)limit) {
            h3_gpu_set_error(gpu, @"VSR tapconv: tap offset outside the guard rows");
            return 0;
        }
    }
    size_t x_count = (size_t)x_rows * in_ch, y_count = (size_t)x_rows * out_ch;
    size_t x1_count = state ? (size_t)state_rows * in_ch : x_count;
    size_t w_count = (size_t)taps * out_ch * in_ch;
    if (!h3_gpu_require_bf16(gpu, x0, x_count, @"VSR tapconv x0") ||
        (x1 && !h3_gpu_require_bf16(gpu, x1, x1_count, @"VSR tapconv x1")) ||
        !h3_gpu_require_bf16(gpu, w0, w_count, @"VSR tapconv w0") ||
        (x1 && !state && !h3_gpu_require_bf16(gpu, w1, w_count, @"VSR tapconv w1")) ||
        !h3_gpu_require_bf16(gpu, output, y_count, @"VSR tapconv output") ||
        (residual && !h3_gpu_require_bf16(gpu, residual, y_count, @"VSR tapconv residual")) ||
        !vsr_check_f32_vector(gpu, bias, out_ch, @"VSR tapconv bias")) return 0;
    if (!h3_gpu_require_command(gpu)) return 0;
    int32_t max_offset = 0;
    for (uint32_t t = 0; t < taps; t++) if (offsets[t] > max_offset) max_offset = offsets[t];
    const vsr_tapconv_variant *variant = vsr_tapconv_variant_for(in_ch, out_ch, rows, guard, x_rows, max_offset);
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, [NSString stringWithUTF8String:variant->name]);
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < variant->threads) {
        h3_gpu_set_error(gpu, @"device cannot dispatch the VSR TensorOps conv");
        return 0;
    }
    /* row tiles fastest (as before): running a row tile's channel tiles
     * together (H3_TAPCONV_ORDER=1) gained nothing at 128-256 channels and
     * lost 1.5x at 512 channels x 27 taps (bench_tapconv, 2026-11-07) */
    const char *order_env = getenv("H3_TAPCONV_ORDER");
    uint32_t order = order_env ? (uint32_t)(atoi(order_env) != 0) : 0u;
    vsr_tapconv_args args = {rows, x_rows, guard, in_ch, out_ch, taps, (x1 && !state) ? 2u : 1u,
                             relu != 0, bias != NULL, residual != NULL, {0},
                             state ? state_rows : x_rows, tap_mask, order};
    memcpy(args.offsets, offsets, taps * sizeof(int32_t));
    const h3_gpu_tensor *xb = x1 ? x1 : x0, *wb = (x1 && !state) ? w1 : w0;
    const h3_gpu_tensor *b = bias ? bias : w0, *r = residual ? residual : output;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(x0).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(xb).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(w0).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(wb).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(b).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(r).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:6];
        [encoder setBytes:&args length:sizeof(args) atIndex:7];
        uint32_t row_tiles = (rows + variant->rt - 1) / variant->rt, col_tiles = out_ch / variant->ct;
        [encoder dispatchThreadgroups:order ? MTLSizeMake(col_tiles, row_tiles, 1) : MTLSizeMake(row_tiles, col_tiles, 1)
                 threadsPerThreadgroup:MTLSizeMake(variant->threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_vsr_tapconv_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                            const h3_gpu_tensor *x0, const h3_gpu_tensor *x1,
                            const h3_gpu_tensor *w0, const h3_gpu_tensor *w1,
                            const h3_gpu_tensor *bias, const h3_gpu_tensor *residual,
                            uint32_t rows, uint32_t x_rows, uint32_t guard,
                            uint32_t in_ch, uint32_t out_ch, uint32_t taps,
                            const int32_t *offsets, int relu) {
    return vsr_tapconv(GPU(opaque), output, x0, x1, 0, w0, w1, bias, residual, rows, x_rows, guard,
                       in_ch, out_ch, taps, offsets, 0, relu);
}

int h3_gpu_vsr_tapconv_state_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *x0, const h3_gpu_tensor *state, uint32_t state_rows,
                                  const h3_gpu_tensor *w, const h3_gpu_tensor *bias, const h3_gpu_tensor *residual,
                                  uint32_t rows, uint32_t x_rows, uint32_t guard,
                                  uint32_t in_ch, uint32_t out_ch, uint32_t taps,
                                  const int32_t *offsets, uint32_t tap_mask, int relu) {
    if (!state_rows) { h3_gpu_set_error(GPU(opaque), @"VSR tapconv: a state without rows"); return 0; }
    return vsr_tapconv(GPU(opaque), output, x0, state, state_rows, w, NULL, bias, residual, rows, x_rows, guard,
                       in_ch, out_ch, taps, offsets, tap_mask, relu);
}

int h3_gpu_vsr_zero_rows_bf16(h3_gpu *opaque, h3_gpu_tensor *x, uint32_t first,
                              uint32_t count, uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    if (!count) return 1;
    if (!h3_gpu_require_bf16(gpu, x, ((size_t)first + count) * width, @"VSR zero rows")) return 0;
    vsr_rows_args args = {count, width, width, first};
    return h3_gpu_dispatch_2d(gpu, @"h3_vsr_zero_rows_bf16", width, count,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(x).buffer offset:0 atIndex:0];
            [encoder setBytes:&args length:sizeof(args) atIndex:1];
        });
}

int h3_gpu_vsr_pack_grid_bf16(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                              uint32_t frames, uint32_t height, uint32_t width,
                              uint32_t channels, uint32_t padded_channels, uint32_t guard,
                              uint32_t x_rows, int zero_guards) {
    H3GPU *gpu = GPU(opaque);
    uint32_t rows = frames * (height + 2) * (width + 2);
    if (!frames || !height || !width || !channels || padded_channels < channels ||
        guard + rows > x_rows) return 0;
    if (!vsr_check_f32(gpu, input, (size_t)frames * height * width * channels, @"VSR pack input") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)x_rows * padded_channels, @"VSR pack output")) return 0;
    vsr_grid_args args = {frames, height, width, channels, padded_channels, guard};
    if (zero_guards &&
        (!h3_gpu_vsr_zero_rows_bf16(opaque, output, 0, guard, padded_channels) ||
         !h3_gpu_vsr_zero_rows_bf16(opaque, output, guard + rows, x_rows - guard - rows, padded_channels)))
        return 0;
    return h3_gpu_dispatch_3d(gpu, @"h3_vsr_pack_grid_bf16",
        MTLSizeMake(padded_channels, width + 2, (NSUInteger)frames * (height + 2)),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_vsr_unpack_grid_f32(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                               uint32_t frames, uint32_t height, uint32_t width,
                               uint32_t channels, uint32_t padded_channels, uint32_t guard,
                               uint32_t x_rows) {
    H3GPU *gpu = GPU(opaque);
    uint32_t rows = frames * (height + 2) * (width + 2);
    if (!frames || !height || !width || !channels || padded_channels < channels ||
        guard + rows > x_rows) return 0;
    if (!h3_gpu_require_bf16(gpu, input, (size_t)x_rows * padded_channels, @"VSR unpack input") ||
        !vsr_check_f32(gpu, output, (size_t)frames * height * width * channels, @"VSR unpack output")) return 0;
    vsr_grid_args args = {frames, height, width, channels, padded_channels, guard};
    return h3_gpu_dispatch_3d(gpu, @"h3_vsr_unpack_grid_f32",
        MTLSizeMake(channels, width, (NSUInteger)frames * height),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_vsr_zero_border_bf16(h3_gpu *opaque, h3_gpu_tensor *x, uint32_t frames,
                                uint32_t height, uint32_t width, uint32_t padded_channels,
                                uint32_t guard, uint32_t x_rows) {
    H3GPU *gpu = GPU(opaque);
    uint32_t rows = frames * (height + 2) * (width + 2);
    if (!frames || !height || !width || guard + rows > x_rows) return 0;
    if (!h3_gpu_require_bf16(gpu, x, (size_t)x_rows * padded_channels, @"VSR zero border")) return 0;
    vsr_grid_args args = {frames, height, width, padded_channels, padded_channels, guard};
    return h3_gpu_dispatch_3d(gpu, @"h3_vsr_zero_border_bf16",
        MTLSizeMake(padded_channels, 2 * (width + 2) + 2 * height, frames),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(x).buffer offset:0 atIndex:0];
            [encoder setBytes:&args length:sizeof(args) atIndex:1];
        });
}

typedef struct {
    uint32_t frames, height, width, channels, padded_channels, guard;
    uint32_t in_height, in_width, in_channels, in_padded, in_guard;
    uint32_t stride_t, stride_y, stride_x, mode;
} vsr_grid2_args;

/* A grid's row count and the check that it fits its buffer. */
static int vsr_grid_fits(H3GPU *gpu, const h3_gpu_tensor *t, const h3_vsr_grid *g, NSString *what) {
    uint64_t rows = (uint64_t)g->frames * (g->height + 2) * (g->width + 2);
    if (!g->frames || !g->height || !g->width || g->padded_channels < g->channels || g->padded_channels % 64 ||
        g->guard + rows > g->x_rows) {
        h3_gpu_set_error(gpu, @"%@: bad grid geometry", what);
        return 0;
    }
    return h3_gpu_require_bf16(gpu, t, (size_t)g->x_rows * g->padded_channels, what);
}

static int vsr_grid_zero_guards(h3_gpu *opaque, h3_gpu_tensor *t, const h3_vsr_grid *g) {
    uint32_t rows = g->frames * (g->height + 2) * (g->width + 2);
    return h3_gpu_vsr_zero_rows_bf16(opaque, t, 0, g->guard, g->padded_channels) &&
           h3_gpu_vsr_zero_rows_bf16(opaque, t, g->guard + rows, g->x_rows - g->guard - rows, g->padded_channels);
}

static vsr_grid2_args vsr_grid2(const h3_vsr_grid *out, const h3_vsr_grid *in) {
    vsr_grid2_args a = {out->frames, out->height, out->width, out->channels, out->padded_channels, out->guard,
                        in ? in->height : 0, in ? in->width : 0, in ? in->channels : 0, in ? in->padded_channels : 0,
                        in ? in->guard : 0, 1, 1, 1, 0};
    return a;
}

#define VSR_GRID_DISPATCH(name, out_grid, in_tensor, out_tensor, args_value) \
    h3_gpu_dispatch_3d(gpu, name, \
        MTLSizeMake((out_grid)->padded_channels, (out_grid)->width + 2, (NSUInteger)(out_grid)->frames * ((out_grid)->height + 2)), \
        ^(id<MTLComputeCommandEncoder> encoder) { \
            [encoder setBuffer:TENSOR(in_tensor).buffer offset:0 atIndex:0]; \
            [encoder setBuffer:TENSOR(out_tensor).buffer offset:0 atIndex:1]; \
            [encoder setBytes:&(args_value) length:sizeof(args_value) atIndex:2]; \
        })

int h3_gpu_vsr_pack_grid_s2d_bf16(h3_gpu *opaque, h3_gpu_tensor *output, const h3_vsr_grid *out,
                                  const h3_gpu_tensor *frames) {
    H3GPU *gpu = GPU(opaque);
    if (!vsr_grid_fits(gpu, output, out, @"VSR s2d pack") || out->channels != 12 ||
        !vsr_check_f32(gpu, frames, (size_t)out->frames * out->height * 2 * out->width * 2 * 3, @"VSR s2d frames"))
        return 0;
    if (!vsr_grid_zero_guards(opaque, output, out)) return 0;
    vsr_grid2_args args = vsr_grid2(out, NULL);
    return VSR_GRID_DISPATCH(@"h3_vsr_pack_grid_s2d_bf16", out, frames, output, args);
}

int h3_gpu_vsr_pack_grid_clamp_bf16(h3_gpu *opaque, h3_gpu_tensor *output, const h3_vsr_grid *out,
                                    const h3_gpu_tensor *input) {
    H3GPU *gpu = GPU(opaque);
    if (!vsr_grid_fits(gpu, output, out, @"VSR clamp pack") ||
        !vsr_check_f32(gpu, input, (size_t)out->frames * out->height * out->width * out->channels, @"VSR clamp input"))
        return 0;
    if (!vsr_grid_zero_guards(opaque, output, out)) return 0;
    vsr_grid_args args = {out->frames, out->height, out->width, out->channels, out->padded_channels, out->guard};
    return VSR_GRID_DISPATCH(@"h3_vsr_pack_grid_clamp_bf16", out, input, output, args);
}

int h3_gpu_vsr_unpack_grid_d2s_f32(h3_gpu *opaque, h3_gpu_tensor *frames, const h3_gpu_tensor *input,
                                   const h3_vsr_grid *in) {
    H3GPU *gpu = GPU(opaque);
    if (!vsr_grid_fits(gpu, input, in, @"VSR d2s unpack") || in->channels != 12 ||
        !vsr_check_f32(gpu, frames, (size_t)in->frames * in->height * 2 * in->width * 2 * 3, @"VSR d2s frames"))
        return 0;
    vsr_grid2_args args = vsr_grid2(in, in);
    return h3_gpu_dispatch_3d(gpu, @"h3_vsr_unpack_grid_d2s_f32",
        MTLSizeMake(3, in->width * 2, (NSUInteger)in->frames * in->height * 2),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(frames).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_vsr_grid_upsample2_bf16(h3_gpu *opaque, h3_gpu_tensor *output, const h3_vsr_grid *out,
                                   const h3_gpu_tensor *input, const h3_vsr_grid *in) {
    H3GPU *gpu = GPU(opaque);
    if (!vsr_grid_fits(gpu, output, out, @"VSR grid upsample out") || !vsr_grid_fits(gpu, input, in, @"VSR grid upsample in") ||
        out->height != in->height * 2 || out->width != in->width * 2 || out->frames != in->frames ||
        out->padded_channels != in->padded_channels) {
        h3_gpu_set_error(gpu, @"VSR grid upsample: shapes");
        return 0;
    }
    if (!vsr_grid_zero_guards(opaque, output, out)) return 0;
    vsr_grid2_args args = vsr_grid2(out, in);
    return VSR_GRID_DISPATCH(@"h3_vsr_grid_upsample2_bf16", out, input, output, args);
}

int h3_gpu_vsr_grid_split_frames_bf16(h3_gpu *opaque, h3_gpu_tensor *output, const h3_vsr_grid *out,
                                      const h3_gpu_tensor *input, const h3_vsr_grid *in) {
    H3GPU *gpu = GPU(opaque);
    if (!vsr_grid_fits(gpu, output, out, @"VSR split out") || !vsr_grid_fits(gpu, input, in, @"VSR split in") ||
        out->height != in->height || out->width != in->width || out->frames != in->frames * 2 ||
        in->channels != out->channels * 2) {
        h3_gpu_set_error(gpu, @"VSR grid split: shapes");
        return 0;
    }
    if (!vsr_grid_zero_guards(opaque, output, out)) return 0;
    vsr_grid2_args args = vsr_grid2(out, in);
    return VSR_GRID_DISPATCH(@"h3_vsr_grid_split_frames_bf16", out, input, output, args);
}

int h3_gpu_vsr_grid_select_bf16(h3_gpu *opaque, h3_gpu_tensor *output, const h3_vsr_grid *out,
                                const h3_gpu_tensor *input, const h3_vsr_grid *in,
                                uint32_t stride_t, uint32_t stride_y, uint32_t stride_x) {
    H3GPU *gpu = GPU(opaque);
    if (!vsr_grid_fits(gpu, output, out, @"VSR select out") || !vsr_grid_fits(gpu, input, in, @"VSR select in") ||
        !stride_t || !stride_y || !stride_x || out->padded_channels != in->padded_channels ||
        (out->frames - 1) * stride_t >= in->frames || (out->height - 1) * stride_y >= in->height ||
        (out->width - 1) * stride_x >= in->width) {
        h3_gpu_set_error(gpu, @"VSR grid select: shapes");
        return 0;
    }
    if (!vsr_grid_zero_guards(opaque, output, out)) return 0;
    vsr_grid2_args args = vsr_grid2(out, in);
    args.stride_t = stride_t; args.stride_y = stride_y; args.stride_x = stride_x;
    return VSR_GRID_DISPATCH(@"h3_vsr_grid_select_bf16", out, input, output, args);
}

int h3_gpu_vsr_scatter_rows_bf16(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *index, uint32_t rows, uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, input, (size_t)rows * width, @"VSR scatter input") ||
        !h3_gpu_require_elements(gpu, index, rows, @"VSR scatter index") || TENSOR(index).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, width, @"VSR scatter output")) return 0;
    vsr_rows_args args = {rows, width, width, 0};
    return h3_gpu_dispatch_2d(gpu, @"h3_vsr_scatter_rows_bf16", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(index).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

typedef struct { uint32_t rows, columns; } vsr_gelu_quant_args;

int h3_gpu_vsr_gelu_quantize_int8_rows(h3_gpu *opaque, h3_gpu_tensor *quantized, h3_gpu_tensor *scales,
                                       const h3_gpu_tensor *input, const h3_gpu_tensor *bias,
                                       uint32_t rows, uint32_t columns) {
    H3GPU *gpu = GPU(opaque);
    uint32_t padded_rows = (rows + 127u) & ~127u;
    if (!rows || columns % 4 || columns > 8192 ||
        !h3_gpu_require_bf16(gpu, input, (size_t)rows * columns, @"VSR gelu-quantize input") ||
        !vsr_check_f32(gpu, bias, columns, @"VSR gelu-quantize bias") ||
        !h3_gpu_require_i8(gpu, quantized, (size_t)padded_rows * columns, @"VSR gelu-quantize output") ||
        !h3_gpu_require_f32(gpu, scales, padded_rows, @"VSR gelu-quantize scales") ||
        !h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, @"h3_vsr_gelu_quantize_int8_rows");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu, @"VSR gelu-quantize is unavailable");
        return 0;
    }
    vsr_gelu_quant_args args = {rows, columns};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(bias).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(quantized).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(scales).buffer offset:0 atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(padded_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

/* ---- VSR colour match ---- */
typedef struct { uint32_t frames, height, width, real_height, real_width, groups; } vsr_sum_args;

int h3_gpu_vsr_channel_means_f32(h3_gpu *opaque, const h3_gpu_tensor *input, uint32_t frames,
                                 uint32_t height, uint32_t width, uint32_t real_height,
                                 uint32_t real_width, float means[3]) {
    H3GPU *gpu = GPU(opaque);
    if (!frames || !real_height || !real_width || real_height > height || real_width > width ||
        !vsr_check_f32(gpu, input, (size_t)frames * height * width * 3, @"VSR channel means")) return 0;
    const uint32_t groups = 512;
    h3_gpu_tensor *partials = h3_gpu_tensor_new_f32(opaque, (size_t)groups * 3);
    if (!partials) return 0;
    vsr_sum_args args = {frames, height, width, real_height, real_width, groups};
    int ok = h3_gpu_begin(opaque) && h3_gpu_require_command(gpu);
    if (ok) {
        id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, @"h3_vsr_channel_sums_f32");
        if (!pipeline) ok = 0;
        else @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(partials).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
        }
    }
    ok = ok && h3_gpu_submit(opaque);
    float *host = malloc((size_t)groups * 3 * sizeof(float));
    ok = ok && host && h3_gpu_tensor_read_f32(partials, host, (size_t)groups * 3);
    if (ok) {
        double sum[3] = {0, 0, 0};
        for (uint32_t g = 0; g < groups; g++) for (int c = 0; c < 3; c++) sum[c] += host[(size_t)g * 3 + c];
        double pixels = (double)frames * real_height * real_width;
        for (int c = 0; c < 3; c++) means[c] = (float)(sum[c] / pixels);
    }
    free(host);
    h3_gpu_tensor_free(partials);
    return ok;
}

int h3_gpu_vsr_add_channel_offsets_f32(h3_gpu *opaque, h3_gpu_tensor *x, uint32_t pixels, const float offsets[3]) {
    H3GPU *gpu = GPU(opaque);
    if (!vsr_check_f32(gpu, x, (size_t)pixels * 3, @"VSR channel offsets")) return 0;
    struct { float v[4]; } off = {{offsets[0], offsets[1], offsets[2], 0.0f}};
    uint32_t count = pixels;
    return h3_gpu_dispatch_1d(gpu, @"h3_vsr_add_channel_offsets_f32", pixels,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(x).buffer offset:0 atIndex:0];
            [encoder setBytes:&off length:sizeof(off) atIndex:1];
            [encoder setBytes:&count length:sizeof(count) atIndex:2];
        });
}
