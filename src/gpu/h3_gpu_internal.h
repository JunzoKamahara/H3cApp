/* Internals of h3_gpu.m shared with GPU code that lives in other modules
 * (src/upscale/h3_gpu_vsr.m, h3_gpu_sv2.m). Objective-C only; not an API. */
#ifndef H3_GPU_INTERNAL_H
#define H3_GPU_INTERNAL_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "h3_gpu.h"

@class H3GPU, H3SDPA, H3GQA, H3Linear, H3MLP, H3Conv;

@interface H3Tensor : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic) size_t elements;
@property(nonatomic) size_t bytes;
@property(nonatomic) h3_gpu_dtype dtype;
@property(nonatomic, weak) H3GPU *owner;
@property(nonatomic, strong) MPSGraphTensorData *graphData;
@property(nonatomic, strong) NSArray<NSNumber *> *graphDataShape;
@property(nonatomic) MPSDataType graphDataType;
/* h3_gpu_tensor_alias_*: a window of `parent`'s memory, kept alive here. */
@property(nonatomic, strong) id<MTLBuffer> aliasParent;
@end

@interface H3GPU : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) id<MTLCommandBuffer> command;
@property(nonatomic, strong) NSMutableArray<id<MTLCommandBuffer>> *inflightCommands;
@property(nonatomic, strong) NSDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
@property(nonatomic, strong) NSMutableDictionary<NSString *, H3SDPA *> *sdpaCache;
@property(nonatomic, strong) NSMutableDictionary<NSString *, H3GQA *> *gqaCache;
@property(nonatomic, strong) NSMutableDictionary<NSString *, H3Linear *> *linearCache;
@property(nonatomic, strong) NSMutableDictionary<NSString *, H3MLP *> *mlpCache;
@property(nonatomic, strong) NSMutableDictionary<NSString *, H3Conv *> *convCache;
/* Per-chunk group sums of h3_gpu_vae_encoder_group_norm_silu_chunked_f32. */
@property(nonatomic, strong) id<MTLBuffer> normScratch;
/* The running sum of h3_gpu_conv3d_by_tap, alternating with the output. */
@property(nonatomic, strong) id<MTLBuffer> convScratch;
@property(nonatomic, strong) MPSCommandBuffer *mpsCommand;
@property(nonatomic) BOOL reuseMPSCommandDefault;
@property(nonatomic, copy) NSString *lastError;
@property(nonatomic) h3_gpu_stats stats;
@property(nonatomic, copy) NSString *profileLabel;
@property(nonatomic) BOOL tensorOpsEnabled;
@property(nonatomic) NSUInteger tensorOpsMode;
@property(nonatomic) BOOL naxLinearGeneric;   /* h3_gpu_set_nax_linear_generic */
@property(nonatomic) BOOL headMajorSDPAInputs;
/* SDPA execution policy (h3_gpu_set_sdpa_mode; issue #2). */
@property(nonatomic) NSUInteger sdpaMode;
@property(nonatomic) uint64_t sdpaBudget;
@property(nonatomic, strong) NSMutableSet<NSString *> *sdpaLogged;
/* Head-major copies of row-major Q/K/V and of the output for the chunked
 * path, grown to the largest shape seen; retired ones are kept until
 * teardown rather than freed under a command buffer that may still read
 * them. */
@property(nonatomic) h3_gpu_tensor *sdpaScratchQ;
@property(nonatomic) h3_gpu_tensor *sdpaScratchK;
@property(nonatomic) h3_gpu_tensor *sdpaScratchV;
@property(nonatomic) h3_gpu_tensor *sdpaScratchO;
/* Score rows of h3_gqa_causal_rows_bf16 (f32, one batch of query rows). */
@property(nonatomic) h3_gpu_tensor *gqaScores;
@property(nonatomic, strong) NSMutableArray<NSValue *> *sdpaRetired;
@property(nonatomic) h3_gpu_stats profileStartStats;
@property(nonatomic) h3_gpu_stats profileMarkStats;
@property(nonatomic) double profileStartWall;
@property(nonatomic) double profileMarkWall;
@property(nonatomic) double commandStartWall;
@end

static inline H3GPU *GPU(h3_gpu *gpu) {
    return (__bridge H3GPU *)gpu;
}

static inline H3Tensor *TENSOR(const h3_gpu_tensor *tensor) {
    return (__bridge H3Tensor *)(void *)tensor;
}

void h3_gpu_set_error(H3GPU *gpu, NSString *format, ...);
int h3_gpu_require_command(H3GPU *gpu);
int h3_gpu_require_elements(H3GPU *gpu, const h3_gpu_tensor *tensor,
                                   size_t elements, NSString *label);
id<MTLComputePipelineState> h3_gpu_pipeline(H3GPU *gpu, NSString *name);
int h3_gpu_dispatch_1d(H3GPU *gpu, NSString *name, uint32_t count,
                              void (^bindings)(id<MTLComputeCommandEncoder>));
int h3_gpu_dispatch_2d(H3GPU *gpu, NSString *name, uint32_t width,
                              uint32_t height,
                              void (^bindings)(id<MTLComputeCommandEncoder>));
int h3_gpu_dispatch_3d(H3GPU *gpu, NSString *name, MTLSize grid,
                              void (^bindings)(id<MTLComputeCommandEncoder>));
int h3_gpu_dispatch_rows(H3GPU *gpu, NSString *name, uint32_t rows,
                                void (^bindings)(id<MTLComputeCommandEncoder>));
int h3_gpu_require_bf16(H3GPU *gpu, const h3_gpu_tensor *tensor,
                               size_t elements, NSString *label);
int h3_gpu_require_i8(H3GPU *gpu, const h3_gpu_tensor *tensor,
                             size_t elements, NSString *label);
int h3_gpu_require_f32(H3GPU *gpu, const h3_gpu_tensor *tensor,
                              size_t elements, NSString *label);

/* A module's kernels: h3_gpu_create builds a pipeline for every name, those
 * of a set with needs_tensor_ops only when TensorOps are on. Call before
 * the first h3_gpu_create (a load-time constructor); `names` must stay valid. */
typedef struct { const char *const *names; size_t count; int needs_tensor_ops; } h3_gpu_kernel_set;
#define H3_GPU_MAX_KERNEL_SETS 16
extern h3_gpu_kernel_set h3_gpu_kernel_sets[H3_GPU_MAX_KERNEL_SETS];
extern size_t h3_gpu_kernel_set_count;
void h3_gpu_register_kernels(const char *const *names, size_t count, int needs_tensor_ops);

#endif
