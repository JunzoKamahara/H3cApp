/* What the files of the Metal layer share among themselves (h3_gpu.m and h3_gpu_*.m in this folder): the cached graph objects, kernel argument layouts and file-scope helpers. h3_gpu_internal.h is the smaller part that code in other modules may use; this one is for src/gpu only. */
#ifndef H3_GPU_PRIVATE_H
#define H3_GPU_PRIVATE_H

#include "h3_options.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "h3_gpu.h"
#include "h3_gpu_internal.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

@class H3GPU;

@interface H3SDPA : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *query;
@property(nonatomic, strong) MPSGraphTensor *key;
@property(nonatomic, strong) MPSGraphTensor *value;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *inputShape;
@property(nonatomic, strong) NSArray<NSNumber *> *outputShape;
@property (nonatomic, strong) NSArray<NSNumber *> *kvShape;
@end

@interface H3GQA : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *query;
@property(nonatomic, strong) MPSGraphTensor *key;
@property(nonatomic, strong) MPSGraphTensor *value;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *queryShape;
@property(nonatomic, strong) NSArray<NSNumber *> *kvShape;
@end

@interface H3Linear : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input;
@property(nonatomic, strong) MPSGraphTensor *weight;
@property(nonatomic, strong) MPSGraphTensor *bias;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *inputShape;
@property(nonatomic, strong) NSArray<NSNumber *> *weightShape;
@property(nonatomic, strong) NSArray<NSNumber *> *biasShape;
@property(nonatomic, strong) NSArray<NSNumber *> *outputShape;
@end

@interface H3MLP : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input;
@property(nonatomic, strong) MPSGraphTensor *fc1Weight;
@property(nonatomic, strong) MPSGraphTensor *fc2Weight;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *inputShape;
@property(nonatomic, strong) NSArray<NSNumber *> *fc1Shape;
@property(nonatomic, strong) NSArray<NSNumber *> *fc2Shape;
@property(nonatomic, strong) NSArray<NSNumber *> *outputShape;
@end

@interface H3Conv : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input;
@property(nonatomic, strong) MPSGraphTensor *weight;
@property(nonatomic, strong) MPSGraphTensor *bias;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *inputShape;
@property(nonatomic, strong) NSArray<NSNumber *> *weightShape;
@property(nonatomic, strong) NSArray<NSNumber *> *biasShape;
@property(nonatomic, strong) NSArray<NSNumber *> *outputShape;
@property(nonatomic, strong) MPSGraphTensor *accumulator;
@end

typedef struct { uint32_t rows, input_dim, output_dim, has_bias; } linear_args;

typedef struct { uint32_t rows, columns; float clip; } int8_quant_args;

typedef struct {
    uint32_t rows, padded_rows, heads, head_dim;
    float clip;
} int8_head_major_quant_args;

typedef struct { uint32_t rows, columns, group_size, groups; }
    int8_group_quant_args;

typedef struct { uint32_t rows, width; float epsilon; } norm_args;

typedef struct {
    uint32_t rows, width, slots, shift_slot, scale_slot;
    float epsilon;
} adaln_args;

typedef struct { uint32_t rows, width, slots, gate_slot; } gate_args;

typedef struct {
    uint32_t sequence, heads, head_dim, rope_half, grouped;
    float epsilon;
} qkv_args;

typedef struct { uint32_t outer, inner; } weight_norm_args;

typedef struct { uint32_t elements; float left_scale, right_scale; }
    add_scaled_args;

typedef struct { uint32_t batch, length, channels; } audio_activation_args;

typedef struct { uint32_t batch, length, heads, head_dim; } audio_qkv_args;

typedef struct {
    uint32_t batch, length, heads, head_dim, output_dim;
} audio_pool_args;

typedef struct { uint32_t elements; float minimum, maximum; } clip_args;

typedef struct {
    uint32_t batch, depth, height, width, channels, depth_front;
    uint32_t height_before, height_after, width_before, width_after;
} vae_encoder_pad_args;

typedef struct {
    uint32_t batch, depth, height, width, channels, groups;
    float epsilon;
} vae_encoder_norm_args;

typedef struct {
    uint32_t frames;
    uint32_t pixels;
    uint32_t channels;
    uint32_t groups;
    uint32_t chunks;
    uint32_t chunk_pixels;
    float epsilon;
} vae_encoder_norm_chunk_args;

typedef struct { uint32_t rows, width; } swiglu_args;

typedef struct { uint32_t elements, approximate; } gelu_bf16_args;

typedef struct { uint32_t tokens, vocab_size, width; } embedding_args;

typedef struct {
    uint32_t sequence, query_heads, kv_heads, head_dim;
    float epsilon;
} text_rope_args;

typedef struct { uint32_t sequence, heads, head_dim; float epsilon; } head_norm_args;

typedef struct { uint32_t sequence, query_heads, kv_heads, head_dim; } text_rope_inplace_args;

typedef struct {
    uint32_t sequence, query_heads, kv_heads, head_dim;
    float scale;
} gqa_args;

typedef struct {
    uint32_t sequence, query_heads, kv_heads, head_dim;
    float scale;
    uint32_t row_offset, score_stride;
} gqa_rows_args;

/* Shortest sequences the device-memory GQA kernels take; below them the
 * threadgroup-memory kernel is as fast (measured, see h3_gpu_gqa_rows). */
enum { H3_GQA_GROUP_MIN_SEQUENCE = 64, H3_GQA_ROWS_MIN_SEQUENCE = 2048 };

static const uint64_t H3_GQA_ROWS_SCRATCH_BYTES = 256ull << 20;

typedef struct { uint32_t sample_offset, elements; float delta, ratio; }
    euler_args;

typedef struct {
    uint32_t sample_offset, elements;
    float sigma, sigma_next;
    uint32_t key0, key1;
} renoise_args;

/* ---- split-K int8 linear (see h3_linear_int8_split_k_nax_r128) ---- */
typedef struct { uint32_t rows, input_dim, output_dim, splits; } split_k_args;

/* File-scope functions and state used by more than one of the files. */
BOOL h3_device_has_tensor_ops(id<MTLDevice> device);
MPSGraphTensorData *h3_gpu_graph_data(const h3_gpu_tensor *tensor,
                                             NSArray<NSNumber *> *shape,
                                             MPSDataType data_type,
                                             int stable);
MPSCommandBuffer *h3_gpu_mps_command(H3GPU *gpu);
int h3_gpu_linear_mps(H3GPU *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim,
                             MPSDataType dataType);
int h3_gpu_quantize_bf16_int8_rows(
                        h3_gpu *opaque, h3_gpu_tensor *output,
                        h3_gpu_tensor *scales,
                        const h3_gpu_tensor *input, uint32_t rows,
                        uint32_t dispatch_rows, uint32_t columns,
                        float clip, NSString *label);

#endif
