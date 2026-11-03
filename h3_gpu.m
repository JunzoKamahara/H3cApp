#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "h3_gpu.h"

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
@interface H3Tensor : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic) size_t elements;
@property(nonatomic) size_t bytes;
@property(nonatomic) h3_gpu_dtype dtype;
@property(nonatomic, weak) H3GPU *owner;
@property(nonatomic, strong) MPSGraphTensorData *graphData;
@property(nonatomic, strong) NSArray<NSNumber *> *graphDataShape;
@property(nonatomic) MPSDataType graphDataType;
@end
@implementation H3Tensor
@end

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
@implementation H3SDPA
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
@implementation H3GQA
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
@implementation H3Linear
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
@implementation H3MLP
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
@implementation H3Conv
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
@implementation H3GPU
@end

static H3GPU *GPU(h3_gpu *gpu) {
    return (__bridge H3GPU *)gpu;
}

/* Whether this device actually exposes the Metal 4 hardware tensor/matmul
 * units (Neural Accelerators) that the MetalPerformancePrimitives-based
 * kernels in h3_shaders.metal require - not just whether its marketing name
 * happens to contain the substring "M5". MTLGPUFamilyApple10 is the GPU
 * family Apple introduced alongside that hardware (first shipped in M5), so
 * querying it via -supportsFamily: reflects the real capability and, unlike
 * a name match, keeps working for future chips in the same GPU family. */
static BOOL h3_device_has_tensor_ops(id<MTLDevice> device) {
    return device && [device supportsFamily:MTLGPUFamilyApple10];
}

static H3Tensor *TENSOR(const h3_gpu_tensor *tensor) {
    return (__bridge H3Tensor *)(void *)tensor;
}

static MPSGraphTensorData *h3_gpu_graph_data(const h3_gpu_tensor *tensor,
                                             NSArray<NSNumber *> *shape,
                                             MPSDataType data_type,
                                             int stable) {
    H3Tensor *object = TENSOR(tensor);
    if (!stable || getenv("H3_DISABLE_GRAPH_DATA_CACHE"))
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:object.buffer
            shape:shape dataType:data_type];
    if (object.graphData && object.graphDataShape == shape &&
        object.graphDataType == data_type) return object.graphData;
    MPSGraphTensorData *data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:object.buffer shape:shape dataType:data_type];
    if (!object.graphData) {
        object.graphData = data;
        object.graphDataShape = shape;
        object.graphDataType = data_type;
    }
    return data;
}

static MPSCommandBuffer *h3_gpu_mps_command(H3GPU *gpu) {
    const char *override = getenv("H3_REUSE_MPS_COMMAND");
    BOOL reuse = override ? (*override && strcmp(override, "0")) :
                            gpu.reuseMPSCommandDefault;
    if (!reuse)
        return [MPSCommandBuffer commandBufferWithCommandBuffer:gpu.command];
    if (!gpu.mpsCommand || gpu.mpsCommand.rootCommandBuffer != gpu.command)
        gpu.mpsCommand =
            [MPSCommandBuffer commandBufferWithCommandBuffer:gpu.command];
    return gpu.mpsCommand;
}

static double h3_gpu_now(void) {
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0) return 0.0;
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

static int h3_gpu_profile_enabled(void) {
    const char *value = getenv("H3_PROFILE");
    return value && *value && strcmp(value, "0");
}

static uint64_t h3_gpu_counter_delta(uint64_t value, uint64_t start) {
    return value >= start ? value - start : 0;
}

static void h3_gpu_profile_emit(H3GPU *gpu, NSString *phase,
                                h3_gpu_stats start, double wall_start) {
    if (!h3_gpu_profile_enabled()) return;
    h3_gpu_stats value = gpu.stats;
    double wall = h3_gpu_now() - wall_start;
    NSString *label = gpu.profileLabel ? gpu.profileLabel : @"Metal context";
    fprintf(stderr,
        "h3 profile: %-24s %-14s wall=%8.3fs encode=%7.3fs "
        "wait=%8.3fs root-gpu=%7.3fs "
        "peak=%7.3fGiB alloc=%7.3fGiB submissions=%llu "
        "direct=%llu linear=%llu conv=%llu attention=%llu\n",
        label.UTF8String, phase.UTF8String, wall,
        value.command_encode_seconds - start.command_encode_seconds,
        value.command_wait_seconds - start.command_wait_seconds,
        value.gpu_seconds - start.gpu_seconds,
        (double)value.peak_live_bytes / (1024.0 * 1024.0 * 1024.0),
        (double)h3_gpu_counter_delta(value.allocated_bytes,
                                     start.allocated_bytes) /
            (1024.0 * 1024.0 * 1024.0),
        (unsigned long long)h3_gpu_counter_delta(value.submissions,
                                                 start.submissions),
        (unsigned long long)h3_gpu_counter_delta(value.direct_dispatches,
                                                 start.direct_dispatches),
        (unsigned long long)h3_gpu_counter_delta(value.mps_linear_dispatches,
                                                 start.mps_linear_dispatches),
        (unsigned long long)h3_gpu_counter_delta(value.mps_conv_dispatches,
                                                 start.mps_conv_dispatches),
        (unsigned long long)h3_gpu_counter_delta(value.mps_sdpa_dispatches,
                                                 start.mps_sdpa_dispatches));
}

static void h3_gpu_set_error(H3GPU *gpu, NSString *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    gpu.lastError = [[NSString alloc] initWithFormat:format arguments:arguments];
    va_end(arguments);
}

static int h3_gpu_require_command(H3GPU *gpu) {
    if (!gpu || !gpu.command) {
        if (gpu) h3_gpu_set_error(gpu, @"h3_gpu_begin() was not called");
        return 0;
    }
    return 1;
}

static int h3_gpu_require_elements(H3GPU *gpu, const h3_gpu_tensor *tensor,
                                   size_t elements, NSString *label) {
    if (!tensor || TENSOR(tensor).elements < elements) {
        h3_gpu_set_error(gpu, @"%@ tensor is absent or too small", label);
        return 0;
    }
    return 1;
}

static id<MTLComputePipelineState> h3_gpu_pipeline(H3GPU *gpu, NSString *name) {
    id<MTLComputePipelineState> pipeline = gpu.pipelines[name];
    if (!pipeline) h3_gpu_set_error(gpu, @"missing Metal pipeline %@", name);
    return pipeline;
}

static int h3_gpu_dispatch_1d(H3GPU *gpu, NSString *name, uint32_t count,
                              void (^bindings)(id<MTLComputeCommandEncoder>)) {
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, name);
    if (!pipeline) return 0;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        bindings(encoder);
        NSUInteger width = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_dispatch_2d(H3GPU *gpu, NSString *name, uint32_t width,
                              uint32_t height,
                              void (^bindings)(id<MTLComputeCommandEncoder>)) {
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, name);
    if (!pipeline) return 0;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        bindings(encoder);
        NSUInteger x = 16;
        NSUInteger y = MIN((NSUInteger)16, pipeline.maxTotalThreadsPerThreadgroup / x);
        [encoder dispatchThreads:MTLSizeMake(width, height, 1)
           threadsPerThreadgroup:MTLSizeMake(x, y, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_dispatch_3d(H3GPU *gpu, NSString *name, MTLSize grid,
                              void (^bindings)(id<MTLComputeCommandEncoder>)) {
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, name);
    if (!pipeline) return 0;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        bindings(encoder);
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_dispatch_rows(H3GPU *gpu, NSString *name, uint32_t rows,
                                void (^bindings)(id<MTLComputeCommandEncoder>)) {
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, name);
    if (!pipeline) return 0;
    NSUInteger maximum = MIN((NSUInteger)256,
                             pipeline.maxTotalThreadsPerThreadgroup);
    NSUInteger threads = 1;
    while (threads * 2 <= maximum) threads *= 2;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        bindings(encoder);
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

h3_gpu *h3_gpu_create(const char *shader_source_path,
                      char *error, size_t error_size) {
    @autoreleasepool {
        H3GPU *gpu = [[H3GPU alloc] init];
        gpu.profileLabel = @"Metal context";
        gpu.profileStartWall = h3_gpu_now();
        gpu.profileMarkWall = gpu.profileStartWall;
        gpu.device = MTLCreateSystemDefaultDevice();
        gpu.queue = [gpu.device newCommandQueue];
        gpu.reuseMPSCommandDefault = YES;
        gpu.inflightCommands = [NSMutableArray array];
        gpu.sdpaCache = [NSMutableDictionary dictionary];
        gpu.gqaCache = [NSMutableDictionary dictionary];
        gpu.linearCache = [NSMutableDictionary dictionary];
        gpu.mlpCache = [NSMutableDictionary dictionary];
        gpu.convCache = [NSMutableDictionary dictionary];
        if (!gpu.device || !gpu.queue) {
            if (error && error_size) snprintf(error, error_size, "cannot initialize Metal");
            return NULL;
        }
        if (getenv("H3_DEBUG_GPU_MEMORY")) {
            fprintf(stderr, "h3: Metal live allocation at GPU startup: "
                    "%.3f GiB\n", (double)gpu.device.currentAllocatedSize /
                    (1024.0 * 1024.0 * 1024.0));
        }
        const char *source_path = shader_source_path ? shader_source_path :
                                                       "h3_shaders.metal";
        NSString *path = [NSString stringWithUTF8String:source_path];
        /* Callers pass the shader as a relative name. Inside an .app bundle
         * that name means the bundle's own Resources copy, and only that:
         * the working directory is not trusted there, because an open/save
         * panel can move it into a folder the user browsed to (picking
         * files in the GGUF import did), and whatever h3_shaders.metal sat
         * in that folder would be compiled and run on the GPU. Outside an
         * .app (the CLI and tests, run from the repo root) a relative path
         * is read from the working directory as before, falling back to the
         * same file next to the executable when it does not resolve. */
        if (!path.isAbsolutePath) {
            NSBundle *bundle = NSBundle.mainBundle;
            NSString *bundled = [bundle.resourcePath
                stringByAppendingPathComponent:path.lastPathComponent];
            BOOL inAppBundle = [bundle.bundlePath.pathExtension.lowercaseString
                isEqualToString:@"app"];
            if (inAppBundle) {
                /* No Resources path: fail to open rather than use the CWD. */
                path = bundled ? bundled : @"";
            } else if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
                if (bundled && [[NSFileManager defaultManager] fileExistsAtPath:bundled])
                    path = bundled;
            }
        }
        NSError *libraryError = nil;
        NSString *source = [NSString stringWithContentsOfFile:path
                                                     encoding:NSUTF8StringEncoding
                                                        error:&libraryError];
        if (source) {
            MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
            options.mathMode = MTLMathModeSafe;
            const char *nax = getenv("H3_NAX");
            BOOL hasTensorHardware = h3_device_has_tensor_ops(gpu.device);
            BOOL wantsTensorOps =
                hasTensorHardware && (!nax || !*nax || strcmp(nax, "0") != 0);
            if (wantsTensorOps) {
                options.preprocessorMacros = @{ @"H3_METAL_HAS_TENSOR": @"1" };
                /* metal_tensor and MetalPerformancePrimitives are Metal 4.0
                 * APIs; don't rely on the runtime's default language version
                 * to include them. */
                if (@available(macOS 26.0, *))
                    options.languageVersion = MTLLanguageVersion4_0;
            }
            gpu.library = [gpu.device newLibraryWithSource:source
                                                   options:options
                                                     error:&libraryError];
            gpu.tensorOpsEnabled = gpu.library && wantsTensorOps;
            if (gpu.tensorOpsEnabled) {
                const char *mode = nax && *nax ? nax : "qkv-attn";
                gpu.tensorOpsMode = !strcmp(mode, "attn") ? 2u :
                                    !strcmp(mode, "qkv-attn") ? 3u :
                                    !strcmp(mode, "qkv") ? 4u :
                                    !strcmp(mode, "mlp") ? 5u : 1u;
            }
            if (!gpu.library && wantsTensorOps) {
                /* TensorOps is optional: an older runtime must retain the
                 * ordinary MPSGraph/direct Metal implementation. */
                if (getenv("H3_NAX_DIAGNOSTIC"))
                    fprintf(stderr, "h3: TensorOps compile failed: %s\n",
                            libraryError.localizedDescription.UTF8String);
                options.preprocessorMacros = @{};
                libraryError = nil;
                gpu.library = [gpu.device newLibraryWithSource:source
                                                       options:options
                                                         error:&libraryError];
                gpu.tensorOpsEnabled = NO;
                gpu.tensorOpsMode = 0;
            }
        }
        if (!gpu.library) {
            if (error && error_size) {
                const char *description = libraryError.localizedDescription.UTF8String;
                snprintf(error, error_size, "cannot compile %s: %s",
                         source_path, description ? description : "unknown error");
            }
            return NULL;
        }
        NSMutableArray<NSString *> *names = [@[
            @"h3_linear_f32", @"h3_linear_f32_tiled",
            @"h3_linear_f32_tiled_bf16", @"h3_silu_f32",
            @"h3_linear_f32_tiled_bf16_map",
            @"h3_cast_f32_to_bf16",
            @"h3_cast_bf16_to_f32",
            @"h3_dequantize",
            @"h3_cast_bf16_to_f32_bias",
            @"h3_cast_bf16_to_f16_qkv", @"h3_cast_f16_to_bf16_flat",
            @"h3_lora_add_rows_bf16", @"h3_lora_add_int8",
            @"h3_rms_norm_f32",
            @"h3_scale_add_f32", @"h3_layer_norm_f32",
            @"h3_video_qkv_rope_f32",
            @"h3_adaln_f32", @"h3_gate_f32", @"h3_qkv_rope_f32",
            @"h3_swiglu_f32", @"h3_linear_bf16", @"h3_silu_bf16",
            @"h3_rms_norm_bf16", @"h3_adaln_bf16", @"h3_gate_bf16",
            @"h3_rms_inverse_bf16", @"h3_adaln_linear_bf16",
            @"h3_gate_adaln_bf16", @"h3_gate_adaln_bf16_exact_simd",
            @"h3_qkv_rope_bf16", @"h3_qkv_rope_bf16_coop",
            @"h3_qkv_rope_bf16_coop_uncached",
            @"h3_swiglu_bf16",
            @"h3_layer_norm_bf16", @"h3_gelu_bf16",
            @"h3_vision_qkv_rope_bf16",
            @"h3_embedding_bf16", @"h3_text_qk_rope_bf16",
            @"h3_head_rms_norm_bf16", @"h3_rope_text_bf16",
            @"h3_gqa_causal_bf16", @"h3_gqa_causal_rows_bf16", @"h3_gqa_causal_group_bf16", @"h3_add_bf16", @"h3_sub_bf16",
            @"h3_token_pool_bf16", @"h3_token_pool_adaln_bf16",
            @"h3_token_expand_delta_bf16",
            @"h3_token_expand_adaln_bf16",
            @"h3_euler_bf16", @"h3_renoise_bf16", @"h3_silu_mul_bf16",
            @"h3_weight_norm_f32", @"h3_add_scaled_f32",
            @"h3_alias_free_snake_f32", @"h3_snake1d_f32",
            @"h3_audio_qkv_split_f32", @"h3_audio_attention_pool_f32",
            @"h3_geglu_f32", @"h3_clip_f32",
            @"h3_vae_encoder_pad_f32",
            @"h3_vsr_pad_f32", @"h3_vsr_upsample2_f32",
            @"h3_vsr_space_to_depth_f32", @"h3_vsr_depth_to_space_f32",
            @"h3_vsr_split_frames_f32", @"h3_vsr_clamp_tanh_f32",
            @"h3_vsr_pixels_to_f32", @"h3_vsr_f32_to_pixels", @"h3_vsr_tile_accumulate_f32",
            @"h3_vsr_ln_mod_bf16", @"h3_vsr_qkv_prepare_bf16",
            @"h3_vsr_gather_rows_bf16", @"h3_vsr_bias_gate_add_bf16",
            @"h3_vsr_gelu_bias_bf16", @"h3_vsr_patchify_bf16",
            @"h3_vsr_pack_grid_bf16", @"h3_vsr_unpack_grid_f32",
            @"h3_vsr_zero_border_bf16", @"h3_vsr_zero_rows_bf16",
            @"h3_vsr_scatter_rows_bf16", @"h3_vsr_gelu_quantize_int8_rows",
            @"h3_vsr_channel_sums_f32", @"h3_vsr_add_channel_offsets_f32",
            @"h3_vsr_pack_grid_s2d_bf16", @"h3_vsr_unpack_grid_d2s_f32",
            @"h3_vsr_grid_upsample2_bf16", @"h3_vsr_grid_split_frames_bf16",
            @"h3_vsr_grid_select_bf16", @"h3_vsr_pack_grid_clamp_bf16",
            @"h3_vsr_unpatchify_sub_f32",
            @"h3_vae_encoder_group_norm_silu_f32",
            @"h3_vae_encoder_norm_chunk_sum_f32",
            @"h3_vae_encoder_norm_chunk_square_f32",
            @"h3_vae_encoder_norm_chunk_apply_silu_f32",
            @"h3_sv2_dequant_q8_bf16", @"h3_sv2_rms_mod_bf16", @"h3_sv2_qkv_window_bf16",
            @"h3_sv2_txt_mean_bf16", @"h3_sv2_pad_f32", @"h3_sv2_shuffle_f32",
            @"h3_sv2_group_norm_f32", @"h3_sv2_bicubic_pixels_f32", @"h3_sv2_f32_to_pixels",
            @"h3_sv2_wavelet_blur_f32", @"h3_sv2_rgb_to_lab_f32", @"h3_sv2_lab_match_to_rgb_f32",
            @"h3_sv2_grid_gn_sum_bf16", @"h3_sv2_grid_gn_sq_bf16", @"h3_sv2_grid_gn_apply_bf16",
            @"h3_sv2_grid_shuffle_bf16", @"h3_sv2_grid_select_bf16"
        ] mutableCopy];
        if (gpu.tensorOpsEnabled) {
            [names addObject:@"h3_linear_bf16_nax_r128"];
            [names addObject:@"h3_vsr_tapconv_bf16_nax"];
            [names addObject:@"h3_linear_bf16_nax_r128_morton"];
            [names addObject:@"h3_linear_bf16_nax_r128_morton4"];
            [names addObject:
                @"h3_qkv_project_split_bf16_nax_r128_morton4"];
            [names addObject:@"h3_qk_rope_bf16_nax_inplace"];
            [names addObject:@"h3_fc1_swiglu_bf16_nax_r128"];
            [names addObject:@"h3_fc1_swiglu_bf16_nax_r128_morton"];
            [names addObject:@"h3_fc1_swiglu_bf16_nax_r128_morton4"];
            [names addObject:@"h3_quantize_bf16_int8_rows"];
            [names addObject:@"h3_quantize_bf16_int8_rows_scalar"];
            [names addObject:
                @"h3_quantize_bf16_int8_head_major_to_rows_cached"];
            [names addObject:@"h3_quantize_bf16_int8_groups"];
            [names addObject:@"h3_quantize_bf16_int8_groups_scalar"];
            [names addObject:@"h3_quantize_bf16_int8_groups_scalar128"];
            [names addObject:
                @"h3_quantize_bf16_int8_groups_scalar128_cached"];
            [names addObject:
                @"h3_qkv_project_split_int8_nax_r128_morton4"];
            [names addObject:
                @"h3_qkv_project_split_int8_rope_nax_r128_morton4"];
            [names addObject:
                @"h3_qkv_project_split_int8_rope_nax_r128_k5376_morton4"];
            [names addObject:
                @"h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4"];
            [names addObject:
                @"h3_qkv_project_split_int8_rope_local_scales_nax_r128_k5376_morton4"];
            [names addObject:@"h3_fc1_swiglu_int8_nax_r128"];
            [names addObject:@"h3_fc1_swiglu_int8_nax_r128_k5376"];
            [names addObject:@"h3_fc1_swiglu_int8_nax_r128_full_k5376"];
            [names addObject:@"h3_fc1_swiglu_int8_local_nax_r128"];
            [names addObject:@"h3_linear_int8_nax_r128"];
            [names addObject:
                @"h3_linear_int8_nax_r128_full_k14336"];
            [names addObject:
                @"h3_linear_int8_nax_r128x256_full_k14336"];
            [names addObject:@"h3_linear_int8_local_scales_nax_r128"];
            [names addObject:@"h3_linear_int8_local_scales_nax_r128_k7168"];
            [names addObject:@"h3_linear_int8_local_scales_nax_r128_k3072_n3072"];
            [names addObject:@"h3_linear_int8_local_scales_nax_r128_k3072_n9216"];
            [names addObject:@"h3_linear_int8_local_scales_nax_r128_k3072_n14336"];
            [names addObject:@"h3_linear_int8_local_scales_nax_r128_k14336_n3072"];
            [names addObject:@"h3_linear_int8_nax_r128_full_k14336_n3072"];
            [names addObject:@"h3_linear_int8_nax_r128x256_full_k14336_n3072"];
            [names addObject:@"h3_linear_int8_nax_r128x256_full_k3072_n14336"];
            [names addObject:@"h3_linear_int8_nax_r128x256_full_k3072_n9216"];
            [names addObject:@"h3_linear_int8_nax_r128x256_full_k3072_n3072"];
            [names addObject:@"h3_linear_int8_split_k_nax_r128"];
            [names addObject:@"h3_linear_sum_partials_bf16"];
            [names addObject:@"h3_gate_adaln_quantize_int8"];
            [names addObject:@"h3_gate_adaln_quantize_int8_scalar"];
            [names addObject:@"h3_linear_int8_grouped_nax_r128x64"];
            [names addObject:
                @"h3_linear_int8_grouped_local_nax_r128x64"];
            [names addObject:
                @"h3_linear_int8_grouped_local_nax_r128x128"];
        }
        NSMutableDictionary *pipelines = [NSMutableDictionary dictionary];
        for (NSString *name in names) {
            id<MTLFunction> function = [gpu.library newFunctionWithName:name];
            NSError *pipelineError = nil;
            id<MTLComputePipelineState> pipeline =
                function ? [gpu.device newComputePipelineStateWithFunction:function
                                                                       error:&pipelineError] : nil;
            if (!pipeline) {
                if (error && error_size) {
                    const char *description = pipelineError.localizedDescription.UTF8String;
                    snprintf(error, error_size, "cannot build %s: %s", name.UTF8String,
                             description ? description : "function missing");
                }
                return NULL;
            }
            pipelines[name] = pipeline;
        }
        gpu.pipelines = pipelines;
        return (__bridge_retained h3_gpu *)gpu;
    }
}

void h3_gpu_free(h3_gpu *gpu) {
    if (!gpu) return;
    @autoreleasepool {
        H3GPU *object = CFBridgingRelease(gpu);
        h3_gpu_profile_emit(object, @"total", object.profileStartStats,
                            object.profileStartWall);
        id<MTLDevice> device = object.device;
        NSUInteger before = device.currentAllocatedSize;
        object.command = nil;
        object.inflightCommands = nil;
        if (object.sdpaScratchQ) h3_gpu_tensor_free(object.sdpaScratchQ);
        if (object.sdpaScratchK) h3_gpu_tensor_free(object.sdpaScratchK);
        if (object.sdpaScratchV) h3_gpu_tensor_free(object.sdpaScratchV);
        if (object.sdpaScratchO) h3_gpu_tensor_free(object.sdpaScratchO);
        if (object.gqaScores) h3_gpu_tensor_free(object.gqaScores);
        for (NSValue *retired in object.sdpaRetired)
            h3_gpu_tensor_free((h3_gpu_tensor *)retired.pointerValue);
        object.sdpaRetired = nil;
        object.sdpaCache = nil;
        object.linearCache = nil;
        object.pipelines = nil;
        object.library = nil;
        object.queue = nil;
        object.device = nil;
        if (getenv("H3_DEBUG_GPU_MEMORY")) {
            fprintf(stderr, "h3: Metal live allocation at GPU teardown: "
                    "%.3f GiB\n", (double)before /
                    (1024.0 * 1024.0 * 1024.0));
        }
    }
}

/* Despite the name (kept for compatibility with existing callers), this
 * reports whether the GPU has the Metal 4 hardware tensor units the int8
 * cache/attention path and other "M5" heuristics in this file actually
 * depend on - see h3_device_has_tensor_ops() - rather than matching the
 * device name string. */
int h3_gpu_is_m5(const h3_gpu *opaque) {
    if (!opaque) return 0;
    H3GPU *gpu = GPU((h3_gpu *)(void *)opaque);
    return h3_device_has_tensor_ops(gpu.device);
}

int h3_gpu_has_nax_mlp(const h3_gpu *opaque) {
    if (!opaque) return 0;
    H3GPU *gpu = GPU((h3_gpu *)(void *)opaque);
    return gpu.tensorOpsEnabled && gpu.tensorOpsMode == 5;
}

int h3_gpu_has_int8_mlp(const h3_gpu *opaque) {
    if (!opaque) return 0;
    H3GPU *gpu = GPU((h3_gpu *)(void *)opaque);
    return gpu.tensorOpsEnabled;
}

static h3_gpu_tensor *h3_gpu_tensor_new(h3_gpu *opaque, const void *values,
                                        size_t elements, size_t item_size,
                                        h3_gpu_dtype dtype) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || elements > SIZE_MAX / item_size) return NULL;
    size_t bytes = elements * item_size;
    H3Tensor *tensor = [[H3Tensor alloc] init];
    tensor.elements = elements;
    tensor.bytes = bytes;
    tensor.dtype = dtype;
    tensor.buffer = [gpu.device newBufferWithLength:MAX(bytes, (size_t)1)
                                            options:MTLResourceStorageModeShared];
    if (!tensor.buffer) {
        h3_gpu_set_error(gpu, @"cannot allocate %zu-byte Metal buffer", bytes);
        return NULL;
    }
    tensor.owner = gpu;
    if (values && bytes) memcpy(tensor.buffer.contents, values, bytes);
    h3_gpu_stats stats = gpu.stats;
    stats.allocated_bytes += bytes;
    stats.live_bytes += bytes;
    if (stats.live_bytes > stats.peak_live_bytes)
        stats.peak_live_bytes = stats.live_bytes;
    stats.tensor_allocations++;
    gpu.stats = stats;
    return (__bridge_retained h3_gpu_tensor *)tensor;
}

h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements) {
    return h3_gpu_tensor_new(gpu, NULL, elements, sizeof(float), H3_GPU_F32);
}

h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements) {
    return h3_gpu_tensor_new(gpu, NULL, elements, sizeof(uint16_t), H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements) {
    return h3_gpu_tensor_new(gpu, NULL, elements, sizeof(int8_t), H3_GPU_I8);
}

h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values,
                                      size_t elements) {
    return h3_gpu_tensor_new(gpu, values, elements, sizeof(float), H3_GPU_F32);
}

h3_gpu_tensor *h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values,
                                       size_t elements) {
    return h3_gpu_tensor_new(gpu, values, elements, sizeof(uint16_t), H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_from_i8(h3_gpu *gpu, const int8_t *values,
                                     size_t elements) {
    return h3_gpu_tensor_new(gpu, values, elements, sizeof(int8_t), H3_GPU_I8);
}

h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values,
                                      size_t elements) {
    return h3_gpu_tensor_new(gpu, values, elements, sizeof(uint32_t), H3_GPU_U32);
}

static h3_gpu_tensor *h3_gpu_tensor_load_file(h3_gpu *opaque, const char *path,
                                              uint64_t file_offset,
                                              size_t elements,
                                              size_t item_size,
                                              h3_gpu_dtype dtype,
                                              const char *label) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || !path || !*path || file_offset > INT64_MAX ||
        elements > SIZE_MAX / item_size) return NULL;
    size_t bytes = elements * item_size;
    if ((uint64_t)bytes > (uint64_t)INT64_MAX - file_offset) return NULL;
    const char *zero_copy = getenv("H3_ZERO_COPY_WEIGHTS");
    int transformer_weight = strstr(path, "/transformer/") != NULL;
    int m5 = h3_device_has_tensor_ops(gpu.device);
    int map_weight = bytes &&
        ((zero_copy && !strcmp(zero_copy, "1")) ||
         (transformer_weight &&
          ((zero_copy && !strcmp(zero_copy, "transformer")) ||
           (!zero_copy && m5))));
    if (map_weight) {
        int descriptor = open(path, O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) {
            h3_gpu_set_error(gpu, @"cannot open %s: %s", path,
                             strerror(errno));
            return NULL;
        }
        size_t page = (size_t)getpagesize();
        uint64_t aligned_offset = file_offset - file_offset % page;
        size_t delta = (size_t)(file_offset - aligned_offset);
        if (bytes > SIZE_MAX - delta) {
            close(descriptor);
            return NULL;
        }
        size_t map_bytes = bytes + delta;
        void *mapping = mmap(NULL, map_bytes, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE, descriptor, (off_t)aligned_offset);
        int map_error = errno;
        close(descriptor);
        if (mapping == MAP_FAILED) {
            h3_gpu_set_error(gpu, @"cannot map %s payload from %s: %s", label,
                             path, strerror(map_error));
            return NULL;
        }
        void *values = (unsigned char *)mapping + delta;
        id<MTLBuffer> buffer = [gpu.device
            newBufferWithBytesNoCopy:values length:MAX(bytes, (size_t)1)
            options:MTLResourceStorageModeShared
            deallocator:^(void *pointer, NSUInteger length) {
                (void)pointer;
                (void)length;
                munmap(mapping, map_bytes);
            }];
        if (!buffer) {
            munmap(mapping, map_bytes);
            h3_gpu_set_error(gpu, @"cannot map %zu-byte Metal buffer", bytes);
            return NULL;
        }
        H3Tensor *tensor = [[H3Tensor alloc] init];
        tensor.elements = elements;
        tensor.bytes = bytes;
        tensor.dtype = dtype;
        tensor.buffer = buffer;
        tensor.owner = gpu;
        h3_gpu_stats stats = gpu.stats;
        stats.allocated_bytes += bytes;
        stats.live_bytes += bytes;
        if (stats.live_bytes > stats.peak_live_bytes)
            stats.peak_live_bytes = stats.live_bytes;
        stats.tensor_allocations++;
        gpu.stats = stats;
        return (__bridge_retained h3_gpu_tensor *)tensor;
    }
    h3_gpu_tensor *opaque_tensor = h3_gpu_tensor_new(
        opaque, NULL, elements, item_size, dtype);
    if (!opaque_tensor) return NULL;
    H3Tensor *tensor = TENSOR(opaque_tensor);
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        h3_gpu_set_error(gpu, @"cannot open %s: %s", path, strerror(errno));
        h3_gpu_tensor_free(opaque_tensor);
        return NULL;
    }
    size_t remaining = bytes;
    size_t completed = 0;
    while (remaining) {
        size_t request = MIN(remaining, (size_t)SSIZE_MAX);
        ssize_t count = pread(descriptor,
                              (unsigned char *)tensor.buffer.contents + completed,
                              request, (off_t)(file_offset + completed));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            int detail = count < 0 ? errno : 0;
            h3_gpu_set_error(gpu, @"cannot read %s payload from %s: %s", label,
                             path, detail ? strerror(detail) :
                                            "unexpected end of file");
            close(descriptor);
            h3_gpu_tensor_free(opaque_tensor);
            return NULL;
        }
        completed += (size_t)count;
        remaining -= (size_t)count;
    }
    close(descriptor);
    return opaque_tensor;
}

h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *opaque, const char *path,
                                       uint64_t file_offset, size_t elements) {
    return h3_gpu_tensor_load_file(opaque, path, file_offset, elements,
                                   sizeof(uint16_t), H3_GPU_BF16, "BF16");
}

h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *opaque, const char *path,
                                      uint64_t file_offset, size_t elements) {
    return h3_gpu_tensor_load_file(opaque, path, file_offset, elements,
                                   sizeof(float), H3_GPU_F32, "F32");
}

/* Shared by the BF16/I8/F32 file-read and SSD-stream entry points below:
 * fills an EXISTING tensor's buffer from a file, unlike h3_gpu_tensor_load_file
 * which allocates a new one. item_size/dtype select which element width and
 * H3_GPU_* tag the request must match. */
static int h3_gpu_tensor_read_file_mode(
                                 h3_gpu_tensor *opaque, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 size_t item_size, h3_gpu_dtype dtype,
                                 const char *label, int uncached,
                                 char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!opaque || !path || !*path ||
        TENSOR(opaque).dtype != dtype ||
        elements != TENSOR(opaque).elements ||
        elements > SIZE_MAX / item_size || file_offset > INT64_MAX) {
        if (error && error_size)
            snprintf(error, error_size, "invalid %s file read request", label);
        return 0;
    }
    size_t bytes = elements * item_size;
    if ((uint64_t)bytes > (uint64_t)INT64_MAX - file_offset) {
        if (error && error_size)
            snprintf(error, error_size, "%s file read range overflows", label);
        return 0;
    }
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        if (error && error_size)
            snprintf(error, error_size, "cannot open %s: %s", path,
                     strerror(errno));
        return 0;
    }
#ifdef F_NOCACHE
    if (uncached) (void)fcntl(descriptor, F_NOCACHE, 1);
#else
    (void)uncached;
#endif
    unsigned char *destination = TENSOR(opaque).buffer.contents;
    size_t completed = 0;
    while (completed < bytes) {
        size_t request = MIN(bytes - completed, (size_t)SSIZE_MAX);
        ssize_t count = pread(descriptor, destination + completed, request,
                              (off_t)(file_offset + completed));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            int detail = count < 0 ? errno : 0;
            if (error && error_size) {
                snprintf(error, error_size, "cannot read %s payload from %s: %s",
                         label, path, detail ? strerror(detail) :
                                              "unexpected end of file");
            }
            close(descriptor);
            return 0;
        }
        completed += (size_t)count;
    }
    close(descriptor);
    return 1;
}

int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *opaque, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size) {
    return h3_gpu_tensor_read_file_mode(
        opaque, path, file_offset, elements, sizeof(uint16_t), H3_GPU_BF16,
        "BF16", 0, error, error_size);
}

int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *opaque, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size) {
    return h3_gpu_tensor_read_file_mode(
        opaque, path, file_offset, elements, sizeof(uint16_t), H3_GPU_BF16,
        "BF16", 1, error, error_size);
}

int h3_gpu_tensor_read_file_i8(h3_gpu_tensor *opaque, const char *path,
                               uint64_t file_offset, size_t elements,
                               char *error, size_t error_size) {
    return h3_gpu_tensor_read_file_mode(
        opaque, path, file_offset, elements, sizeof(int8_t), H3_GPU_I8,
        "I8", 0, error, error_size);
}

int h3_gpu_tensor_stream_file_i8(h3_gpu_tensor *opaque, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size) {
    return h3_gpu_tensor_read_file_mode(
        opaque, path, file_offset, elements, sizeof(int8_t), H3_GPU_I8,
        "I8", 1, error, error_size);
}

int h3_gpu_tensor_read_file_f32(h3_gpu_tensor *opaque, const char *path,
                                uint64_t file_offset, size_t elements,
                                char *error, size_t error_size) {
    return h3_gpu_tensor_read_file_mode(
        opaque, path, file_offset, elements, sizeof(float), H3_GPU_F32,
        "F32", 0, error, error_size);
}

int h3_gpu_tensor_stream_file_f32(h3_gpu_tensor *opaque, const char *path,
                                  uint64_t file_offset, size_t elements,
                                  char *error, size_t error_size) {
    return h3_gpu_tensor_read_file_mode(
        opaque, path, file_offset, elements, sizeof(float), H3_GPU_F32,
        "F32", 1, error, error_size);
}

void h3_gpu_tensor_free(h3_gpu_tensor *tensor) {
    if (!tensor) return;
    @autoreleasepool {
        H3Tensor *object = CFBridgingRelease(tensor);
        H3GPU *owner = object.owner;
        if (owner) {
            h3_gpu_stats stats = owner.stats;
            stats.live_bytes = stats.live_bytes >= object.bytes ?
                stats.live_bytes - object.bytes : 0;
            owner.stats = stats;
        }
        [object.buffer setPurgeableState:MTLPurgeableStateEmpty];
        object.buffer = nil;
    }
}

size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor) {
    return tensor ? TENSOR(tensor).elements : 0;
}

h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor) {
    return tensor ? TENSOR(tensor).dtype : H3_GPU_F32;
}

int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values,
                           size_t elements) {
    return h3_gpu_tensor_read_f32_range(tensor, 0, values, elements);
}

int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements) {
    if (!tensor || !values || TENSOR(tensor).dtype != H3_GPU_F32 ||
        source_offset > TENSOR(tensor).elements ||
        elements > TENSOR(tensor).elements - source_offset) return 0;
    const unsigned char *source = TENSOR(tensor).buffer.contents;
    memcpy(values, source + source_offset * sizeof(float),
           elements * sizeof(float));
    return 1;
}

int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values,
                            size_t elements) {
    if (!tensor || !values || TENSOR(tensor).dtype != H3_GPU_BF16 ||
        elements > TENSOR(tensor).elements) return 0;
    memcpy(values, TENSOR(tensor).buffer.contents, elements * sizeof(uint16_t));
    return 1;
}

int h3_gpu_tensor_read_i8(const h3_gpu_tensor *tensor, int8_t *values,
                          size_t elements) {
    if (!tensor || !values || TENSOR(tensor).dtype != H3_GPU_I8 ||
        elements > TENSOR(tensor).elements) return 0;
    memcpy(values, TENSOR(tensor).buffer.contents, elements * sizeof(int8_t));
    return 1;
}

int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values,
                            size_t elements) {
    return h3_gpu_tensor_write_f32_range(tensor, 0, values, elements);
}

int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements) {
    if (!tensor || !values || TENSOR(tensor).dtype != H3_GPU_F32 ||
        destination_offset > TENSOR(tensor).elements ||
        elements > TENSOR(tensor).elements - destination_offset) return 0;
    unsigned char *destination = TENSOR(tensor).buffer.contents;
    memcpy(destination + destination_offset * sizeof(float), values,
           elements * sizeof(float));
    return 1;
}

int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements) {
    return h3_gpu_tensor_write_bf16_range(tensor, 0, values, elements);
}

int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements) {
    if (!tensor || !values || TENSOR(tensor).dtype != H3_GPU_BF16 ||
        destination_offset > TENSOR(tensor).elements ||
        elements > TENSOR(tensor).elements - destination_offset) return 0;
    unsigned char *destination = TENSOR(tensor).buffer.contents;
    memcpy(destination + destination_offset * sizeof(uint16_t), values,
           elements * sizeof(uint16_t));
    return 1;
}

int h3_gpu_begin(h3_gpu *opaque) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || gpu.command || gpu.inflightCommands.count) return 0;
    gpu.lastError = nil;
    @autoreleasepool {
        gpu.command = [gpu.queue commandBuffer];
    }
    if (!gpu.command) {
        h3_gpu_set_error(gpu, @"cannot create Metal command buffer");
        return 0;
    }
    gpu.commandStartWall = h3_gpu_now();
    return 1;
}

int h3_gpu_continue(h3_gpu *opaque) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || !gpu.command) return 0;
    @autoreleasepool {
        id<MTLCommandBuffer> command = gpu.command;
        gpu.command = nil;
        gpu.mpsCommand = nil;
        double commit_time = h3_gpu_now();
        [command commit];
        [gpu.inflightCommands addObject:command];
        h3_gpu_stats stats = gpu.stats;
        stats.submissions++;
        stats.command_encode_seconds += commit_time - gpu.commandStartWall;
        gpu.stats = stats;
        gpu.command = [gpu.queue commandBuffer];
    }
    if (!gpu.command) {
        h3_gpu_set_error(gpu, @"cannot continue Metal command chain");
        return 0;
    }
    gpu.commandStartWall = h3_gpu_now();
    return 1;
}

int h3_gpu_submit(h3_gpu *opaque) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || !gpu.command) return 0;
    @autoreleasepool {
        id<MTLCommandBuffer> command = gpu.command;
        gpu.command = nil;
        gpu.mpsCommand = nil;
        double commit_time = h3_gpu_now();
        [command commit];
        [gpu.inflightCommands addObject:command];
        for (id<MTLCommandBuffer> pending in gpu.inflightCommands)
            [pending waitUntilCompleted];
        double complete_time = h3_gpu_now();
        for (id<MTLCommandBuffer> pending in gpu.inflightCommands) {
            if (pending.status == MTLCommandBufferStatusError) {
                h3_gpu_set_error(gpu, @"Metal command failed: %@",
                                 pending.error.localizedDescription);
                [gpu.inflightCommands removeAllObjects];
                return 0;
            }
        }
        h3_gpu_stats stats = gpu.stats;
        stats.submissions++;
        stats.command_encode_seconds += commit_time - gpu.commandStartWall;
        stats.command_wait_seconds += complete_time - commit_time;
        for (id<MTLCommandBuffer> pending in gpu.inflightCommands) {
            if (pending.GPUEndTime >= pending.GPUStartTime)
                stats.gpu_seconds += pending.GPUEndTime - pending.GPUStartTime;
        }
        gpu.stats = stats;
        [gpu.inflightCommands removeAllObjects];
    }
    return 1;
}

const char *h3_gpu_error(const h3_gpu *opaque) {
    H3GPU *gpu = GPU((h3_gpu *)(void *)opaque);
    const char *message = gpu.lastError.UTF8String;
    return message ? message : "unknown Metal error";
}

int h3_gpu_get_stats(const h3_gpu *opaque, h3_gpu_stats *stats) {
    if (!opaque || !stats) return 0;
    *stats = GPU((h3_gpu *)(void *)opaque).stats;
    return 1;
}

void h3_gpu_profile_set_label(h3_gpu *opaque, const char *label) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || !label || !*label) return;
    gpu.profileLabel = [NSString stringWithUTF8String:label];
}

void h3_gpu_profile_mark(h3_gpu *opaque, const char *phase) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || !phase || !*phase || !h3_gpu_profile_enabled()) return;
    h3_gpu_profile_emit(gpu, [NSString stringWithUTF8String:phase],
                        gpu.profileMarkStats, gpu.profileMarkWall);
    gpu.profileMarkStats = gpu.stats;
    gpu.profileMarkWall = h3_gpu_now();
}

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

static int h3_gpu_linear_mps(H3GPU *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim,
                             MPSDataType dataType);

int h3_gpu_linear_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!h3_gpu_require_elements(gpu, input, input_count, @"linear input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, weight_count, @"linear weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count, @"linear output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_dim, @"linear bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32))) return 0;
    if (!getenv("H3_SCALAR_PATCH") && rows >= 16 && output_dim == 5376 &&
        (input_dim == 32 || input_dim == 96)) {
        linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
        const h3_gpu_tensor *bias_buffer = bias ? bias : input;
        if (!h3_gpu_require_command(gpu)) return 0;
        id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
            gpu, @"h3_linear_f32_tiled");
        if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
            h3_gpu_set_error(gpu,
                             @"device cannot dispatch the F32 16x16 tile");
            return 0;
        }
        @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder =
                [gpu.command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake((output_dim + 15) / 16,
                                                      (rows + 15) / 16, 1)
                     threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [encoder endEncoding];
        }
        h3_gpu_stats stats = gpu.stats;
        stats.direct_dispatches++;
        gpu.stats = stats;
        return 1;
    }
    if (rows >= 32 && input_dim >= 256 && output_dim >= 256 &&
        h3_gpu_linear_mps(gpu, output, input, weight, bias, rows,
                          input_dim, output_dim, MPSDataTypeFloat32)) return 1;
    linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    return h3_gpu_dispatch_2d(gpu, @"h3_linear_f32", output_dim, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

int h3_gpu_patch_linear_bf16_offset(
                             h3_gpu *opaque, h3_gpu_tensor *output,
                             size_t output_offset,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (output_dim != 5376 || (input_dim != 32 && input_dim != 96)) {
        h3_gpu_set_error(gpu, @"unsupported fused patch projection shape");
        return 0;
    }
    if (input_offset > SIZE_MAX - input_count ||
        output_offset > SIZE_MAX - output_count ||
        input_offset > SIZE_MAX / sizeof(float) ||
        output_offset > SIZE_MAX / sizeof(uint16_t)) {
        h3_gpu_set_error(gpu, @"fused patch projection offset is out of range");
        return 0;
    }
    if (!h3_gpu_require_elements(gpu, input, input_offset + input_count,
                                 @"patch projection input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, weight_count,
                                 @"patch projection weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_offset + output_count,
                                 @"patch projection output") ||
        TENSOR(output).dtype != H3_GPU_BF16 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_dim,
                                           @"patch projection bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32)) ||
        !h3_gpu_require_command(gpu)) return 0;
    linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, @"h3_linear_f32_tiled_bf16");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu,
                         @"device cannot dispatch the fused F32/BF16 tile");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer
                       offset:input_offset * sizeof(float) atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(output).buffer
                       offset:output_offset * sizeof(uint16_t) atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake((output_dim + 15) / 16,
                                                  (rows + 15) / 16, 1)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_patch_linear_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    return h3_gpu_patch_linear_bf16_offset(
        opaque, output, 0, input, 0, weight, bias, rows, input_dim,
        output_dim);
}

int h3_gpu_patch_linear_bf16_map(
                             h3_gpu *opaque, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias,
                             const h3_gpu_tensor *row_map,
                             uint32_t output_rows, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)output_rows * output_dim;
    if (output_dim != 5376 || (input_dim != 32 && input_dim != 96)) {
        h3_gpu_set_error(gpu, @"unsupported mapped patch projection shape");
        return 0;
    }
    if (!h3_gpu_require_elements(gpu, input, input_count,
                                 @"mapped patch input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, weight_count,
                                 @"mapped patch weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count,
                                 @"mapped patch output") ||
        TENSOR(output).dtype != H3_GPU_BF16 ||
        !h3_gpu_require_elements(gpu, row_map, rows,
                                 @"mapped patch row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_dim,
                                           @"mapped patch bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32)) ||
        !h3_gpu_require_command(gpu)) return 0;
    linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, @"h3_linear_f32_tiled_bf16_map");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu,
                         @"device cannot dispatch the mapped patch tile");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake((output_dim + 15) / 16,
                                                  (rows + 15) / 16, 1)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_silu_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, input, elements, @"SiLU input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, @"SiLU output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_silu_f32", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        });
}

int h3_gpu_cast_f32_to_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, input, elements, @"cast input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, @"cast output") ||
        TENSOR(output).dtype != H3_GPU_BF16) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_cast_f32_to_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        });
}

int h3_gpu_command_open(const h3_gpu *opaque) {
    if (!opaque) return 0;
    H3GPU *gpu = GPU((h3_gpu *)(void *)opaque);
    return gpu.command != nil || gpu.inflightCommands.count != 0;
}

int h3_gpu_dequantize(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *raw, uint32_t quant,
                      uint32_t blocks, uint32_t block_elements,
                      uint32_t block_bytes, uint32_t columns,
                      uint32_t qkv_inner, uint32_t qkv_head_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t values = (size_t)blocks * block_elements;
    if (!h3_gpu_require_elements(gpu, raw, (size_t)blocks * block_bytes,
                                 @"dequantize input") ||
        TENSOR(raw).dtype != H3_GPU_I8 ||
        !h3_gpu_require_elements(gpu, output, values, @"dequantize output") ||
        (TENSOR(output).dtype != H3_GPU_BF16 &&
         TENSOR(output).dtype != H3_GPU_F32)) return 0;
    if (qkv_inner && (!columns || !qkv_head_dim || qkv_inner % qkv_head_dim ||
                      columns % block_elements ||
                      values != (size_t)3 * qkv_inner * columns)) {
        h3_gpu_set_error(gpu, @"invalid QKV row reorder");
        return 0;
    }
    struct {
        uint32_t quant, blocks, out_f32, block_elements;
        uint32_t columns, qkv_inner, qkv_head_dim;
    } args = {
        quant, blocks, TENSOR(output).dtype == H3_GPU_F32 ? 1u : 0u,
        block_elements, columns, qkv_inner, qkv_head_dim };
    return h3_gpu_dispatch_1d(gpu, @"h3_dequantize", blocks,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(raw).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_cast_bf16_to_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, input, elements, @"cast input") ||
        TENSOR(input).dtype != H3_GPU_BF16 ||
        !h3_gpu_require_elements(gpu, output, elements, @"cast output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_cast_bf16_to_f32", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        });
}

int h3_gpu_cast_bf16_to_f32_bias(h3_gpu *opaque, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *bias, uint32_t rows,
                                 uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, @"bias cast input") ||
        TENSOR(input).dtype != H3_GPU_BF16 ||
        !h3_gpu_require_elements(gpu, bias, width, @"bias cast bias") ||
        TENSOR(bias).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, @"bias cast output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    swiglu_args args = {rows, width};
    return h3_gpu_dispatch_2d(gpu, @"h3_cast_bf16_to_f32_bias", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(bias).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_copy_bf16(h3_gpu *opaque, h3_gpu_tensor *destination,
                     size_t destination_offset,
                     const h3_gpu_tensor *source, size_t source_offset,
                     size_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_command(gpu) || !destination || !source ||
        TENSOR(destination).dtype != H3_GPU_BF16 ||
        TENSOR(source).dtype != H3_GPU_BF16 ||
        source_offset > TENSOR(source).elements ||
        elements > TENSOR(source).elements - source_offset ||
        destination_offset > TENSOR(destination).elements ||
        elements > TENSOR(destination).elements - destination_offset ||
        elements > SIZE_MAX / sizeof(uint16_t)) {
        h3_gpu_set_error(gpu, @"invalid BF16 blit range");
        return 0;
    }
    @autoreleasepool {
        id<MTLBlitCommandEncoder> encoder = [gpu.command blitCommandEncoder];
        [encoder copyFromBuffer:TENSOR(source).buffer
                   sourceOffset:source_offset * sizeof(uint16_t)
                       toBuffer:TENSOR(destination).buffer
              destinationOffset:destination_offset * sizeof(uint16_t)
                           size:elements * sizeof(uint16_t)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.blit_copies++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_copy_f32(h3_gpu *opaque, h3_gpu_tensor *destination,
                    size_t destination_offset,
                    const h3_gpu_tensor *source, size_t source_offset,
                    size_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_command(gpu) || !destination || !source ||
        TENSOR(destination).dtype != H3_GPU_F32 ||
        TENSOR(source).dtype != H3_GPU_F32 ||
        source_offset > TENSOR(source).elements ||
        elements > TENSOR(source).elements - source_offset ||
        destination_offset > TENSOR(destination).elements ||
        elements > TENSOR(destination).elements - destination_offset ||
        elements > SIZE_MAX / sizeof(float)) {
        h3_gpu_set_error(gpu, @"invalid F32 blit range");
        return 0;
    }
    @autoreleasepool {
        id<MTLBlitCommandEncoder> encoder = [gpu.command blitCommandEncoder];
        [encoder copyFromBuffer:TENSOR(source).buffer
                   sourceOffset:source_offset * sizeof(float)
                       toBuffer:TENSOR(destination).buffer
              destinationOffset:destination_offset * sizeof(float)
                           size:elements * sizeof(float)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.blit_copies++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_rms_norm_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, @"RMSNorm input") ||
        !h3_gpu_require_elements(gpu, weight, width, @"RMSNorm weight") ||
        !h3_gpu_require_elements(gpu, output, count, @"RMSNorm output")) return 0;
    norm_args args = {rows, width, epsilon};
    return h3_gpu_dispatch_rows(gpu, @"h3_rms_norm_f32", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_adaln_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, @"AdaLN input") ||
        !h3_gpu_require_elements(gpu, norm_weight, width, @"AdaLN norm") ||
        !h3_gpu_require_elements(gpu, row_map, rows, @"AdaLN row map") ||
        !h3_gpu_require_elements(gpu, output, count, @"AdaLN output") ||
        !modulation || shift_slot >= slots || scale_slot >= slots) return 0;
    adaln_args args = {rows, width, slots, shift_slot, scale_slot, epsilon};
    return h3_gpu_dispatch_2d(gpu, @"h3_adaln_f32", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
            [encoder setBytes:&args length:sizeof(args) atIndex:5];
        });
}

int h3_gpu_gate_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual,
                    const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, residual, count, @"gate residual") ||
        !h3_gpu_require_elements(gpu, branch, count, @"gate branch") ||
        !h3_gpu_require_elements(gpu, row_map, rows, @"gate row map") ||
        !h3_gpu_require_elements(gpu, output, count, @"gate output") ||
        !modulation || gate_slot >= slots) return 0;
    gate_args args = {rows, width, slots, gate_slot};
    return h3_gpu_dispatch_2d(gpu, @"h3_gate_f32", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(branch).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
            [encoder setBytes:&args length:sizeof(args) atIndex:5];
        });
}

int h3_gpu_qkv_rope_f32(h3_gpu *opaque, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_gpu_require_elements(gpu, qkv, count * 3, @"QKV input") ||
        !h3_gpu_require_elements(gpu, q_norm, head_dim, @"Q norm") ||
        !h3_gpu_require_elements(gpu, k_norm, head_dim, @"K norm") ||
        !h3_gpu_require_elements(gpu, rope_cos, rope_count, @"RoPE cosine") ||
        !h3_gpu_require_elements(gpu, rope_sin, rope_count, @"RoPE sine") ||
        !h3_gpu_require_elements(gpu, query, count, @"query") ||
        !h3_gpu_require_elements(gpu, key, count, @"key") ||
        !h3_gpu_require_elements(gpu, value, count, @"value") ||
        rope_half * 2 > head_dim) return 0;
    qkv_args args = {sequence, heads, head_dim, rope_half, 0, epsilon};
    return h3_gpu_dispatch_3d(gpu, @"h3_qkv_rope_f32",
        MTLSizeMake(head_dim, heads, sequence),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:5];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:6];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:7];
            [encoder setBytes:&args length:sizeof(args) atIndex:8];
        });
}

static H3SDPA *h3_gpu_sdpa_graph(H3GPU *gpu, uint32_t batch,
                                 uint32_t sequence, uint32_t kv_sequence,
                                 uint32_t heads, uint32_t head_dim, float scale,
                                 MPSDataType dataType, int causal,
                                 int headMajor, int outputHeadMajor) {
    @autoreleasepool {
        NSString *cacheKey = [NSString stringWithFormat:
                              @"%u:%u:%u:%u:%u:%u:%.9g:%d:%d:%d",
                              (unsigned)dataType, batch, sequence, kv_sequence, heads,
                              head_dim, scale, causal, headMajor,
                              outputHeadMajor];
        H3SDPA *cached = gpu.sdpaCache[cacheKey];
        if (cached) return cached;
        MPSGraph *graph = [[MPSGraph alloc] init];
        NSArray<NSNumber *> *rowMajorShape =
            @[@(batch), @(sequence), @(heads), @(head_dim)];
        NSArray<NSNumber *> *headMajorShape =
            @[@(batch), @(heads), @(sequence), @(head_dim)];
        NSArray<NSNumber *> *outputShape = outputHeadMajor ?
            headMajorShape : rowMajorShape;
        NSArray<NSNumber *> *inputShape = headMajor ?
            headMajorShape : rowMajorShape;
        NSArray<NSNumber *> *kvShape = headMajor ?
            @[@(batch), @(heads), @(kv_sequence), @(head_dim)] :
            @[@(batch), @(kv_sequence), @(heads), @(head_dim)];
        MPSGraphTensor *q = [graph placeholderWithShape:inputShape
                                               dataType:dataType name:nil];
        MPSGraphTensor *k = [graph placeholderWithShape:kvShape
                                               dataType:dataType name:nil];
        MPSGraphTensor *v = [graph placeholderWithShape:kvShape
                                               dataType:dataType name:nil];
        MPSGraphTensor *qt = headMajor ? q :
            [graph transposeTensor:q dimension:1 withDimension:2 name:nil];
        MPSGraphTensor *kt = headMajor ? k :
            [graph transposeTensor:k dimension:1 withDimension:2 name:nil];
        MPSGraphTensor *vt = headMajor ? v :
            [graph transposeTensor:v dimension:1 withDimension:2 name:nil];
        if (![graph respondsToSelector:@selector(scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:scale:name:)]) {
            h3_gpu_set_error(gpu, @"native MPSGraph SDPA is unavailable");
            return nil;
        }
        MPSGraphTensor *attention;
        if (causal) {
            size_t mask_count = (size_t)sequence * sequence;
            float *mask_values = malloc(mask_count * sizeof(*mask_values));
            if (!mask_values) return nil;
            for (uint32_t row = 0; row < sequence; row++)
                for (uint32_t column = 0; column < sequence; column++)
                    mask_values[(size_t)row * sequence + column] =
                        column <= row ? 0.0f : -INFINITY;
            NSData *mask_data = [NSData dataWithBytesNoCopy:mask_values
                length:mask_count * sizeof(*mask_values) freeWhenDone:YES];
            MPSGraphTensor *mask = [graph constantWithData:mask_data
                shape:@[@1, @1, @(sequence), @(sequence)]
                dataType:MPSDataTypeFloat32];
            attention = [graph
                scaledDotProductAttentionWithQueryTensor:qt keyTensor:kt
                valueTensor:vt maskTensor:mask scale:scale name:nil];
        } else {
            attention = [graph scaledDotProductAttentionWithQueryTensor:qt
                keyTensor:kt valueTensor:vt scale:scale name:nil];
        }
        H3SDPA *result = [[H3SDPA alloc] init];
        result.graph = graph;
        result.query = q;
        result.key = k;
        result.value = v;
        result.output = outputHeadMajor ? attention :
            [graph transposeTensor:attention dimension:1 withDimension:2
             name:nil];
        result.inputShape = inputShape;
        result.kvShape = kvShape;
        result.outputShape = outputShape;
        gpu.sdpaCache[cacheKey] = result;
        return result;
    }
}

/* --- SDPA execution policy (issue #2) ---------------------------------
 *
 * h3_gpu_sdpa_graph() hands MPSGraph the whole problem in one call: for the
 * DiT that is bf16, 56 heads and the full token sequence (about 17k rows
 * for 5 s at 960x544, 37k at 1344x768), i.e. a [1, 56, S, S] score matrix
 * of tens of GB if it is ever materialised. On this project's development
 * machine (Apple10 GPU, macOS 26) the call runs at 37k rows. An M2 Max on
 * macOS 15.8.2 (issue #2) aborted with SIGABRT inside
 * GPU::ScaledDotProductAttentionOpHandler::encodeNDArrayOp on the DiT's
 * first full attention, after the text encoder's small bf16 SDPA calls
 * (h3_gpu_gqa_graph, a few hundred to a few thousand rows) had completed -
 * consistent with that MPSGraph build decomposing the op and tripping an
 * MPSNDArray size assertion, though without the full crash report the
 * assertion text is not known. So where the one-call path has not been
 * shown to work, the attention runs in bounded pieces: a group of heads at
 * a time, or one head and a block of query rows when a single head's score
 * matrix is still above the budget. Every piece is a complete softmax over
 * all keys (the pieces partition the output; nothing is renormalised), so
 * the computation is the same and no piece's score matrix exceeds the
 * budget. The pieces address contiguous head-major ranges of Q/K/V and of
 * the output through MPSNDArray buffer offsets; row-major inputs are
 * transposed into head-major scratch once per call (which the one-call
 * graph does internally anyway), and the head-major result is transposed
 * back when the caller wants rows. */

static const char *h3_gpu_family_name(id<MTLDevice> device) {
    if (!device) return "none";
    if ([device supportsFamily:MTLGPUFamilyApple10]) return "Apple10";
    if ([device supportsFamily:MTLGPUFamilyApple9]) return "Apple9";
    if ([device supportsFamily:MTLGPUFamilyApple8]) return "Apple8";
    if ([device supportsFamily:MTLGPUFamilyApple7]) return "Apple7";
    return "older";
}

static h3_gpu_sdpa_mode h3_gpu_sdpa_resolve_mode(H3GPU *gpu) {
    if (gpu.sdpaMode) return (h3_gpu_sdpa_mode)gpu.sdpaMode;
    const char *env = getenv("H3_SDPA_MODE");
    if (env && *env) {
        if (!strcmp(env, "full")) return H3_GPU_SDPA_FULL;
        if (!strcmp(env, "chunked")) return H3_GPU_SDPA_CHUNKED;
        if (!strcmp(env, "explicit")) return H3_GPU_SDPA_EXPLICIT;
        fprintf(stderr, "h3: warning: H3_SDPA_MODE=%s is not full, chunked "
                "or explicit; ignoring\n", env);
    }
    /* The one-call path is validated on the GPU family this project is
     * developed on, running the OS that ships with it. Everything else
     * (M1 to M4 on any macOS; the issue #2 machine among them) takes the
     * bounded path until shown otherwise. */
    BOOL validated = h3_device_has_tensor_ops(gpu.device);
    BOOL modernOS = NO;
    if (@available(macOS 26.0, *)) modernOS = YES;
    return validated && modernOS ? H3_GPU_SDPA_FULL : H3_GPU_SDPA_CHUNKED;
}

static uint64_t h3_gpu_sdpa_budget(H3GPU *gpu) {
    if (gpu.sdpaBudget) return gpu.sdpaBudget;
    const char *env = getenv("H3_SDPA_SCORE_BYTES");
    if (env && *env) {
        unsigned long long parsed = strtoull(env, NULL, 10);
        if (parsed >= (1ull << 20)) return parsed;
        fprintf(stderr, "h3: warning: H3_SDPA_SCORE_BYTES=%s is below 1 MiB; "
                "ignoring\n", env);
    }
    return 1ull << 30;
}

/* Bytes a piece's score matrix costs per element: the op's own dtype for
 * the fused SDPA, f32 scores plus the dtype-sized copies around the softmax
 * when it is written out explicitly. */
static uint64_t h3_gpu_sdpa_score_cost(h3_gpu_dtype dtype, int explicit) {
    uint64_t elem = dtype == H3_GPU_F32 ? 4 : 2;
    return explicit ? 4 + 2 * elem : elem;
}

static void h3_gpu_sdpa_plan(uint64_t budget, uint64_t cost, uint32_t sequence,
                             uint32_t kv_sequence, uint32_t heads,
                             uint32_t *chunk_heads, uint32_t *chunk_rows) {
    uint64_t per_head = (uint64_t)sequence * kv_sequence * cost;
    if (per_head <= budget) {
        uint64_t fit = budget / per_head;
        *chunk_heads = (uint32_t)(fit < heads ? fit : heads);
        *chunk_rows = sequence;
    } else {
        uint64_t rows = budget / ((uint64_t)kv_sequence * cost);
        *chunk_heads = 1;
        *chunk_rows = (uint32_t)(rows < 1 ? 1 : rows < sequence ? rows : sequence);
    }
}

void h3_gpu_set_sdpa_mode(h3_gpu *opaque, h3_gpu_sdpa_mode mode,
                          uint64_t score_budget_bytes) {
    H3GPU *gpu = GPU(opaque);
    gpu.sdpaMode = (NSUInteger)mode;
    gpu.sdpaBudget = score_budget_bytes;
}

h3_gpu_sdpa_mode h3_gpu_sdpa_effective_mode(h3_gpu *opaque) {
    return h3_gpu_sdpa_resolve_mode(GPU(opaque));
}

const char *h3_gpu_sdpa_mode_name(h3_gpu_sdpa_mode mode) {
    switch (mode) {
    case H3_GPU_SDPA_FULL: return "full";
    case H3_GPU_SDPA_CHUNKED: return "chunked";
    case H3_GPU_SDPA_EXPLICIT: return "explicit";
    default: return "auto";
    }
}

void h3_gpu_sdpa_chunk_plan(h3_gpu *opaque, uint32_t sequence,
                            uint32_t kv_sequence, uint32_t heads,
                            h3_gpu_dtype dtype, uint32_t *chunk_heads,
                            uint32_t *chunk_rows) {
    H3GPU *gpu = GPU(opaque);
    h3_gpu_sdpa_mode mode = h3_gpu_sdpa_resolve_mode(gpu);
    if (mode == H3_GPU_SDPA_FULL) {
        *chunk_heads = heads;
        *chunk_rows = sequence;
        return;
    }
    h3_gpu_sdpa_plan(h3_gpu_sdpa_budget(gpu),
                     h3_gpu_sdpa_score_cost(dtype, mode == H3_GPU_SDPA_EXPLICIT),
                     sequence, kv_sequence, heads, chunk_heads, chunk_rows);
}

/* A graph tensor over part of a buffer. MPSGraphTensorData itself has no
 * offset; an MPSNDArray over the buffer does (available since macOS 12.3). */
static MPSGraphTensorData *h3_gpu_graph_view(const h3_gpu_tensor *tensor,
                                             NSArray<NSNumber *> *shape,
                                             MPSDataType data_type,
                                             size_t byte_offset) {
    H3Tensor *object = TENSOR(tensor);
    if (!byte_offset)
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:object.buffer
            shape:shape dataType:data_type];
    MPSNDArrayDescriptor *descriptor =
        [MPSNDArrayDescriptor descriptorWithDataType:data_type shape:shape];
    MPSNDArray *array = [[MPSNDArray alloc] initWithBuffer:object.buffer
                                                     offset:byte_offset
                                                 descriptor:descriptor];
    return [[MPSGraphTensorData alloc] initWithMPSNDArray:array];
}

/* [1, rows, heads, head_dim] <-> [1, heads, rows, head_dim]. */
static H3SDPA *h3_gpu_sdpa_transpose_graph(H3GPU *gpu, uint32_t rows,
                                           uint32_t heads, uint32_t head_dim,
                                           MPSDataType dataType,
                                           int toHeadMajor) {
    @autoreleasepool {
        NSString *cacheKey = [NSString stringWithFormat:@"T:%u:%u:%u:%u:%d",
                              (unsigned)dataType, rows, heads, head_dim,
                              toHeadMajor];
        H3SDPA *cached = gpu.sdpaCache[cacheKey];
        if (cached) return cached;
        MPSGraph *graph = [[MPSGraph alloc] init];
        NSArray<NSNumber *> *rowMajor = @[@1, @(rows), @(heads), @(head_dim)];
        NSArray<NSNumber *> *headMajor = @[@1, @(heads), @(rows), @(head_dim)];
        MPSGraphTensor *input = [graph placeholderWithShape:
            toHeadMajor ? rowMajor : headMajor dataType:dataType name:nil];
        H3SDPA *result = [[H3SDPA alloc] init];
        result.graph = graph;
        result.query = input;
        result.output = [graph transposeTensor:input dimension:1
                                 withDimension:2 name:nil];
        result.inputShape = toHeadMajor ? rowMajor : headMajor;
        result.outputShape = toHeadMajor ? headMajor : rowMajor;
        gpu.sdpaCache[cacheKey] = result;
        return result;
    }
}

static int h3_gpu_sdpa_run_transpose(H3GPU *gpu, const h3_gpu_tensor *source,
                                     h3_gpu_tensor *destination, uint32_t rows,
                                     uint32_t heads, uint32_t head_dim,
                                     MPSDataType dataType, int toHeadMajor) {
    H3SDPA *graph = h3_gpu_sdpa_transpose_graph(gpu, rows, heads, head_dim,
                                                dataType, toHeadMajor);
    if (!graph) return 0;
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        NSDictionary *feeds = @{
            graph.query: h3_gpu_graph_data(source, graph.inputShape, dataType, 0)
        };
        NSDictionary *results = @{
            graph.output: h3_gpu_graph_data(destination, graph.outputShape,
                                            dataType, 0)
        };
        @try {
            [graph.graph encodeToCommandBuffer:command feeds:feeds
                              targetOperations:nil resultsDictionary:results
                           executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph SDPA transpose failed: %@",
                             exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    return 1;
}

/* One piece: head-major q [1, heads, rows, D], k/v [1, heads, kv_rows, D],
 * head-major output. */
static H3SDPA *h3_gpu_sdpa_chunk_graph(H3GPU *gpu, uint32_t heads,
                                       uint32_t rows, uint32_t kv_rows,
                                       uint32_t head_dim, float scale,
                                       MPSDataType dataType, int explicit) {
    @autoreleasepool {
        NSString *cacheKey = [NSString stringWithFormat:
                              @"C:%u:%u:%u:%u:%u:%.9g:%d", (unsigned)dataType,
                              heads, rows, kv_rows, head_dim, scale, explicit];
        H3SDPA *cached = gpu.sdpaCache[cacheKey];
        if (cached) return cached;
        if (!explicit &&
            ![[MPSGraph class] instancesRespondToSelector:@selector(scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:scale:name:)]) {
            h3_gpu_set_error(gpu, @"native MPSGraph SDPA is unavailable");
            return nil;
        }
        MPSGraph *graph = [[MPSGraph alloc] init];
        NSArray<NSNumber *> *qShape = @[@1, @(heads), @(rows), @(head_dim)];
        NSArray<NSNumber *> *kvShape = @[@1, @(heads), @(kv_rows), @(head_dim)];
        MPSGraphTensor *q = [graph placeholderWithShape:qShape dataType:dataType
                                                   name:nil];
        MPSGraphTensor *k = [graph placeholderWithShape:kvShape dataType:dataType
                                                   name:nil];
        MPSGraphTensor *v = [graph placeholderWithShape:kvShape dataType:dataType
                                                   name:nil];
        MPSGraphTensor *attention;
        if (explicit) {
            /* scores = (q k^T) * scale in f32, softmax over keys, back to the
             * op dtype for the value product. */
            MPSGraphTensor *kT = [graph transposeTensor:k dimension:2
                                          withDimension:3 name:nil];
            MPSGraphTensor *scores = [graph
                matrixMultiplicationWithPrimaryTensor:q secondaryTensor:kT
                name:nil];
            scores = [graph castTensor:scores toType:MPSDataTypeFloat32
                                  name:nil];
            MPSGraphTensor *factor = [graph constantWithScalar:scale
                                                      dataType:MPSDataTypeFloat32];
            scores = [graph multiplicationWithPrimaryTensor:scores
                                            secondaryTensor:factor name:nil];
            MPSGraphTensor *probabilities = [graph softMaxWithTensor:scores
                                                                axis:3 name:nil];
            probabilities = [graph castTensor:probabilities toType:dataType
                                         name:nil];
            attention = [graph matrixMultiplicationWithPrimaryTensor:probabilities
                                                     secondaryTensor:v name:nil];
        } else {
            attention = [graph scaledDotProductAttentionWithQueryTensor:q
                keyTensor:k valueTensor:v scale:scale name:nil];
        }
        H3SDPA *result = [[H3SDPA alloc] init];
        result.graph = graph;
        result.query = q;
        result.key = k;
        result.value = v;
        result.output = attention;
        result.inputShape = qShape;
        result.kvShape = kvShape;
        result.outputShape = qShape;
        gpu.sdpaCache[cacheKey] = result;
        return result;
    }
}

static h3_gpu_tensor *h3_gpu_sdpa_scratch(H3GPU *gpu, int which,
                                          size_t elements, h3_gpu_dtype dtype) {
    h3_gpu_tensor *current = which == 0 ? gpu.sdpaScratchQ :
                             which == 1 ? gpu.sdpaScratchK :
                             which == 2 ? gpu.sdpaScratchV : gpu.sdpaScratchO;
    if (current && TENSOR(current).dtype == dtype &&
        TENSOR(current).elements >= elements) return current;
    if (current) {
        if (!gpu.sdpaRetired) gpu.sdpaRetired = [NSMutableArray array];
        [gpu.sdpaRetired addObject:[NSValue valueWithPointer:current]];
    }
    h3_gpu *opaque = (__bridge h3_gpu *)gpu;
    h3_gpu_tensor *fresh = dtype == H3_GPU_F32 ?
        h3_gpu_tensor_new_f32(opaque, elements) :
        h3_gpu_tensor_new_bf16(opaque, elements);
    if (!fresh) {
        h3_gpu_set_error(gpu, @"cannot allocate SDPA scratch (%zu elements)",
                         elements);
        return NULL;
    }
    switch (which) {
    case 0: gpu.sdpaScratchQ = fresh; break;
    case 1: gpu.sdpaScratchK = fresh; break;
    case 2: gpu.sdpaScratchV = fresh; break;
    default: gpu.sdpaScratchO = fresh; break;
    }
    return fresh;
}

static void h3_gpu_sdpa_log(H3GPU *gpu, h3_gpu_sdpa_mode mode,
                            uint64_t budget, uint32_t chunk_heads,
                            uint32_t chunk_rows, MPSDataType dtype,
                            uint32_t batch, uint32_t sequence,
                            uint32_t kv_sequence, uint32_t heads,
                            uint32_t head_dim, int causal, int headMajor,
                            int outputHeadMajor, const h3_gpu_tensor *query,
                            const h3_gpu_tensor *key,
                            const h3_gpu_tensor *value,
                            const h3_gpu_tensor *output) {
    if (!getenv("H3_SDPA_LOG")) return;
    @autoreleasepool {
        NSString *key_text = [NSString stringWithFormat:
            @"%d:%u:%u:%u:%u:%u:%u:%d:%d:%d", (int)mode, (unsigned)dtype, batch,
            sequence, kv_sequence, heads, head_dim, causal, headMajor,
            outputHeadMajor];
        if (!gpu.sdpaLogged) gpu.sdpaLogged = [NSMutableSet set];
        if ([gpu.sdpaLogged containsObject:key_text]) return;
        [gpu.sdpaLogged addObject:key_text];
        size_t elem = dtype == MPSDataTypeFloat32 ? 4 : 2;
        double scores = (double)heads * (double)sequence *
                        (double)kv_sequence * (double)elem;
        fprintf(stderr,
                "h3: sdpa os=%s gpu=\"%s\" family=%s tensorops=%d mode=%s "
                "dtype=%s batch=%u seq=%u kv=%u heads=%u head_dim=%u "
                "causal=%d layout=%s->%s scores=%.2f GiB budget=%.2f GiB "
                "pieces=%u heads x %u rows buffers q=%lu k=%lu v=%lu out=%lu\n",
                [NSProcessInfo processInfo].operatingSystemVersionString.UTF8String,
                gpu.device.name.UTF8String, h3_gpu_family_name(gpu.device),
                gpu.tensorOpsEnabled ? 1 : 0, h3_gpu_sdpa_mode_name(mode),
                dtype == MPSDataTypeFloat32 ? "f32" :
                dtype == MPSDataTypeBFloat16 ? "bf16" : "other",
                batch, sequence, kv_sequence, heads, head_dim, causal,
                headMajor ? "head-major" : "row-major",
                outputHeadMajor ? "head-major" : "row-major",
                scores / (1024.0 * 1024.0 * 1024.0),
                (double)budget / (1024.0 * 1024.0 * 1024.0),
                chunk_heads, chunk_rows,
                (unsigned long)TENSOR(query).buffer.length,
                (unsigned long)TENSOR(key).buffer.length,
                (unsigned long)TENSOR(value).buffer.length,
                (unsigned long)TENSOR(output).buffer.length);
    }
}

static int h3_gpu_sdpa_chunked(H3GPU *gpu, h3_gpu_tensor *output,
                               const h3_gpu_tensor *query,
                               const h3_gpu_tensor *key,
                               const h3_gpu_tensor *value, uint32_t sequence,
                               uint32_t kv_sequence, uint32_t heads,
                               uint32_t head_dim, float scale,
                               h3_gpu_dtype dtype, MPSDataType mps_dtype,
                               int headMajorInputs, int outputHeadMajor,
                               int explicit, uint32_t chunk_heads,
                               uint32_t chunk_rows) {
    size_t elem = dtype == H3_GPU_F32 ? 4 : 2;
    if (((size_t)head_dim * elem) % 16) {
        h3_gpu_set_error(gpu, @"chunked SDPA needs head_dim * bytes to be a "
                         @"multiple of 16 (head_dim %u)", head_dim);
        return 0;
    }
    const h3_gpu_tensor *q = query, *k = key, *v = value;
    if (!headMajorInputs) {
        size_t q_elements = (size_t)sequence * heads * head_dim;
        size_t kv_elements = (size_t)kv_sequence * heads * head_dim;
        h3_gpu_tensor *sq = h3_gpu_sdpa_scratch(gpu, 0, q_elements, dtype);
        h3_gpu_tensor *sk = h3_gpu_sdpa_scratch(gpu, 1, kv_elements, dtype);
        h3_gpu_tensor *sv = h3_gpu_sdpa_scratch(gpu, 2, kv_elements, dtype);
        if (!sq || !sk || !sv) return 0;
        if (!h3_gpu_sdpa_run_transpose(gpu, query, sq, sequence, heads,
                                       head_dim, mps_dtype, 1) ||
            !h3_gpu_sdpa_run_transpose(gpu, key, sk, kv_sequence, heads,
                                       head_dim, mps_dtype, 1) ||
            !h3_gpu_sdpa_run_transpose(gpu, value, sv, kv_sequence, heads,
                                       head_dim, mps_dtype, 1)) return 0;
        q = sq; k = sk; v = sv;
    }
    h3_gpu_tensor *out = output;
    if (!outputHeadMajor) {
        out = h3_gpu_sdpa_scratch(gpu, 3, (size_t)sequence * heads * head_dim,
                                  dtype);
        if (!out) return 0;
    }
    uint64_t pieces = 0;
    for (uint32_t h0 = 0; h0 < heads; h0 += chunk_heads) {
        uint32_t hn = heads - h0 < chunk_heads ? heads - h0 : chunk_heads;
        for (uint32_t r0 = 0; r0 < sequence; r0 += chunk_rows) {
            uint32_t rn = sequence - r0 < chunk_rows ? sequence - r0 : chunk_rows;
            /* A piece of several heads must span all rows: only then is its
             * range of a head-major tensor contiguous. */
            if (hn > 1 && rn != sequence) {
                h3_gpu_set_error(gpu, @"chunked SDPA plan is inconsistent");
                return 0;
            }
            H3SDPA *piece = h3_gpu_sdpa_chunk_graph(gpu, hn, rn, kv_sequence,
                                                    head_dim, scale, mps_dtype,
                                                    explicit);
            if (!piece) return 0;
            size_t q_offset = ((size_t)h0 * sequence + r0) * head_dim * elem;
            size_t kv_offset = (size_t)h0 * kv_sequence * head_dim * elem;
            @autoreleasepool {
                MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
                NSDictionary *feeds = @{
                    piece.query: h3_gpu_graph_view(q, piece.inputShape,
                                                   mps_dtype, q_offset),
                    piece.key: h3_gpu_graph_view(k, piece.kvShape, mps_dtype,
                                                 kv_offset),
                    piece.value: h3_gpu_graph_view(v, piece.kvShape, mps_dtype,
                                                   kv_offset)
                };
                NSDictionary *results = @{
                    piece.output: h3_gpu_graph_view(out, piece.outputShape,
                                                    mps_dtype, q_offset)
                };
                @try {
                    [piece.graph encodeToCommandBuffer:command feeds:feeds
                                      targetOperations:nil
                                     resultsDictionary:results
                                   executionDescriptor:nil];
                } @catch (NSException *exception) {
                    h3_gpu_set_error(gpu, @"MPSGraph chunked SDPA failed "
                                     @"(heads %u..%u, rows %u..%u): %@", h0,
                                     h0 + hn, r0, r0 + rn, exception.reason);
                    return 0;
                }
                gpu.command = command.rootCommandBuffer;
            }
            pieces++;
        }
    }
    if (!outputHeadMajor &&
        !h3_gpu_sdpa_run_transpose(gpu, out, output, sequence, heads, head_dim,
                                   mps_dtype, 0)) return 0;
    h3_gpu_stats stats = gpu.stats;
    stats.mps_sdpa_dispatches++;
    stats.mps_sdpa_chunk_dispatches += pieces;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_sdpa_kv(h3_gpu *opaque, h3_gpu_tensor *output,
                       const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                       const h3_gpu_tensor *value, uint32_t batch,
                       uint32_t sequence, uint32_t kv_sequence,
                       uint32_t heads, uint32_t head_dim, float scale,
                       h3_gpu_dtype tensor_dtype, MPSDataType mps_dtype,
                       int causal, int outputHeadMajor) {
    H3GPU *gpu = GPU(opaque);
    int headMajor = gpu.headMajorSDPAInputs &&
        tensor_dtype == H3_GPU_BF16 && batch == 1 && !causal &&
        kv_sequence == sequence;
    gpu.headMajorSDPAInputs = NO;
    size_t count = (size_t)batch * sequence * heads * head_dim;
    size_t kv_count = (size_t)batch * kv_sequence * heads * head_dim;
    if (!batch || !sequence || !kv_sequence || !heads || !head_dim ||
        (causal && kv_sequence != sequence) ||
        !h3_gpu_require_command(gpu) ||
        !h3_gpu_require_elements(gpu, query, count, @"SDPA query") ||
        !h3_gpu_require_elements(gpu, key, kv_count, @"SDPA key") ||
        !h3_gpu_require_elements(gpu, value, kv_count, @"SDPA value") ||
        !h3_gpu_require_elements(gpu, output, count, @"SDPA output")) return 0;
    if (TENSOR(query).dtype != tensor_dtype || TENSOR(key).dtype != tensor_dtype ||
        TENSOR(value).dtype != tensor_dtype || TENSOR(output).dtype != tensor_dtype) {
        h3_gpu_set_error(gpu, @"SDPA tensor dtype mismatch");
        return 0;
    }
    h3_gpu_sdpa_mode mode = h3_gpu_sdpa_resolve_mode(gpu);
    /* Only the dense single-batch calls (the DiT, the refiner, SwiftVR's
     * cross attention) take the bounded path; the causal and batched ones
     * are small and keep the one-call graph. */
    int chunking = mode != H3_GPU_SDPA_FULL && batch == 1 && !causal;
    uint32_t chunk_heads = heads, chunk_rows = sequence;
    uint64_t budget = h3_gpu_sdpa_budget(gpu);
    if (chunking)
        h3_gpu_sdpa_plan(budget, h3_gpu_sdpa_score_cost(tensor_dtype,
                             mode == H3_GPU_SDPA_EXPLICIT),
                         sequence, kv_sequence, heads, &chunk_heads,
                         &chunk_rows);
    h3_gpu_sdpa_log(gpu, chunking ? mode : H3_GPU_SDPA_FULL, budget,
                    chunk_heads, chunk_rows, mps_dtype, batch, sequence,
                    kv_sequence, heads, head_dim, causal, headMajor,
                    outputHeadMajor, query, key, value, output);
    if (chunking)
        return h3_gpu_sdpa_chunked(gpu, output, query, key, value, sequence,
                                   kv_sequence, heads, head_dim, scale,
                                   tensor_dtype, mps_dtype, headMajor,
                                   outputHeadMajor,
                                   mode == H3_GPU_SDPA_EXPLICIT, chunk_heads,
                                   chunk_rows);
    H3SDPA *cache = h3_gpu_sdpa_graph(gpu, batch, sequence, kv_sequence, heads,
                                      head_dim, scale, mps_dtype, causal, headMajor,
                                      outputHeadMajor);
    if (!cache) return 0;
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        NSDictionary *feeds = @{
            cache.query: h3_gpu_graph_data(query, cache.inputShape, mps_dtype, 0),
            cache.key: h3_gpu_graph_data(key, cache.kvShape, mps_dtype, 0),
            cache.value: h3_gpu_graph_data(value, cache.kvShape, mps_dtype, 0)
        };
        MPSGraphTensorData *outputData = h3_gpu_graph_data(
            output, cache.outputShape, mps_dtype, 0);
        NSDictionary *results = @{cache.output: outputData};
        @try {
            [cache.graph encodeToCommandBuffer:command feeds:feeds targetOperations:nil
                             resultsDictionary:results executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph SDPA failed: %@", exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_sdpa_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_sdpa(h3_gpu *opaque, h3_gpu_tensor *output,
                       const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                       const h3_gpu_tensor *value, uint32_t batch,
                       uint32_t sequence,
                       uint32_t heads, uint32_t head_dim, float scale,
                       h3_gpu_dtype tensor_dtype, MPSDataType mps_dtype,
                       int causal, int outputHeadMajor) {
    return h3_gpu_sdpa_kv(opaque, output, query, key, value, batch, sequence,
                          sequence, heads, head_dim, scale, tensor_dtype,
                          mps_dtype, causal, outputHeadMajor);
}

int h3_gpu_sdpa_batched_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                             const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                             const h3_gpu_tensor *value, uint32_t batch,
                             uint32_t sequence, uint32_t heads,
                             uint32_t head_dim, float scale) {
    return h3_gpu_sdpa_kv(opaque, output, query, key, value, batch, sequence,
                          sequence, heads, head_dim, scale, H3_GPU_BF16,
                          MPSDataTypeBFloat16, 0, 0);
}

int h3_gpu_sdpa_batched_kv_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                                const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                                const h3_gpu_tensor *value, uint32_t batch,
                                uint32_t sequence, uint32_t kv_sequence, uint32_t heads,
                                uint32_t head_dim, float scale) {
    return h3_gpu_sdpa_kv(opaque, output, query, key, value, batch, sequence,
                          kv_sequence, heads, head_dim, scale, H3_GPU_BF16,
                          MPSDataTypeBFloat16, 0, 0);
}

int h3_gpu_sdpa_cross_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value, uint32_t sequence,
                           uint32_t kv_sequence, uint32_t heads,
                           uint32_t head_dim, float scale) {
    return h3_gpu_sdpa_kv(opaque, output, query, key, value, 1, sequence,
                          kv_sequence, heads, head_dim, scale, H3_GPU_BF16,
                          MPSDataTypeBFloat16, 0, 0);
}

int h3_gpu_sdpa_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale) {
    return h3_gpu_sdpa(opaque, output, query, key, value, 1, sequence, heads,
                       head_dim, scale, H3_GPU_F32, MPSDataTypeFloat32, 0, 0);
}

int h3_gpu_sdpa_causal_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t batch,
                    uint32_t sequence, uint32_t heads, uint32_t head_dim,
                    float scale) {
    return h3_gpu_sdpa(opaque, output, query, key, value, batch, sequence,
                       heads, head_dim, scale, H3_GPU_F32,
                       MPSDataTypeFloat32, 1, 0);
}

int h3_gpu_sdpa_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    return h3_gpu_sdpa(opaque, output, query, key, value, 1, sequence, heads,
                       head_dim, scale, H3_GPU_BF16, MPSDataTypeBFloat16, 0,
                       0);
}

int h3_gpu_head_major_sdpa_inputs(h3_gpu *opaque) {
    return GPU(opaque).headMajorSDPAInputs != 0;
}

void h3_gpu_set_head_major_sdpa_inputs(h3_gpu *opaque, int value) {
    GPU(opaque).headMajorSDPAInputs = value != 0;
}

static uint64_t h3_ccv_attention_dispatch_total = 0;
static uint64_t h3_ccv_attention_direct_total = 0;

void h3_gpu_note_ccv_attention_dispatch(h3_gpu *opaque, int direct) {
    H3GPU *gpu = GPU(opaque);
    h3_gpu_stats stats = gpu.stats;
    stats.ccv_attention_dispatches++;
    gpu.stats = stats;
    h3_ccv_attention_dispatch_total++;
    if (direct) h3_ccv_attention_direct_total++;
}

uint64_t h3_gpu_ccv_attention_dispatch_count(void) {
    return h3_ccv_attention_dispatch_total;
}

uint64_t h3_gpu_ccv_attention_direct_count(void) {
    return h3_ccv_attention_direct_total;
}

void h3_gpu_report_error(h3_gpu *opaque, const char *message) {
    h3_gpu_set_error(GPU(opaque), @"%s", message);
}

void *h3_gpu_raw_device(h3_gpu *opaque) {
    return (__bridge void *)GPU(opaque).device;
}

void *h3_gpu_raw_command_buffer(h3_gpu *opaque) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_command(gpu)) return NULL;
    return (__bridge void *)gpu.command;
}

void *h3_gpu_raw_buffer(const h3_gpu_tensor *tensor) {
    return (__bridge void *)TENSOR(tensor).buffer;
}

int h3_gpu_ccv_cast_bf16_to_f16(h3_gpu *opaque, void *output_f16_buffer,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t heads, uint32_t head_dim,
                                int head_major) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * heads * head_dim;
    if (!h3_gpu_require_elements(gpu, input, count, @"ccv cast input") ||
        !output_f16_buffer) return 0;
    id<MTLBuffer> output = (__bridge id<MTLBuffer>)output_f16_buffer;
    struct { uint32_t rows, heads, head_dim, head_major; } args = {
        rows, heads, head_dim, (uint32_t)(head_major != 0)
    };
    return h3_gpu_dispatch_1d(gpu, @"h3_cast_bf16_to_f16_qkv", (uint32_t)count,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:output offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_ccv_cast_f16_to_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                                const void *input_f16_buffer,
                                uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, output, elements, @"ccv cast output") ||
        !input_f16_buffer) return 0;
    id<MTLBuffer> input = (__bridge id<MTLBuffer>)input_f16_buffer;
    return h3_gpu_dispatch_1d(gpu, @"h3_cast_f16_to_bf16_flat", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:input offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        });
}

static h3_gpu_ccv_dense_attention_fn h3_gpu_ccv_dense_attention_impl = NULL;

void h3_gpu_register_ccv_dense_attention(h3_gpu_ccv_dense_attention_fn fn) {
    h3_gpu_ccv_dense_attention_impl = fn;
}

static h3_gpu_ccv_release_scratch_fn h3_gpu_ccv_release_scratch_impl = NULL;

void h3_gpu_register_ccv_release_scratch(h3_gpu_ccv_release_scratch_fn fn) {
    h3_gpu_ccv_release_scratch_impl = fn;
}

int h3_gpu_ccv_release_scratch(h3_gpu *opaque) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_ccv_release_scratch_impl) return 1;
    /* Completion, not the CPU call returning, is what makes the buffers
     * free to go: h3_gpu_submit waits for and clears inflightCommands, so
     * an empty list with no open command means the GPU is done with them. */
    if (!gpu || gpu.command || gpu.inflightCommands.count) return 0;
    h3_gpu_ccv_release_scratch_impl(opaque);
    return 1;
}

static h3_gpu_ccv_attention_supported_fn h3_gpu_ccv_attention_supported_impl =
    NULL;

void h3_gpu_register_ccv_attention_supported(
    h3_gpu_ccv_attention_supported_fn fn) {
    h3_gpu_ccv_attention_supported_impl = fn;
}

int h3_gpu_ccv_attention_supported(void) {
    return h3_gpu_ccv_attention_supported_impl &&
           h3_gpu_ccv_attention_supported_impl();
}

int h3_gpu_ccv_dense_attention_available(void) {
    return h3_gpu_ccv_dense_attention_impl != NULL;
}

int h3_gpu_ccv_dense_attention_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                                    const h3_gpu_tensor *query,
                                    const h3_gpu_tensor *key,
                                    const h3_gpu_tensor *value,
                                    uint32_t rows, uint32_t heads,
                                    uint32_t head_dim, float scale,
                                    int quantized, int *head_major_output) {
    if (!h3_gpu_ccv_dense_attention_impl) {
        h3_gpu_set_error(GPU(opaque), @"ccv backend not compiled into this "
                         "binary (build with CCV_DIR set - see "
                         "tools/ccv_eval/README.md)");
        return 0;
    }
    return h3_gpu_ccv_dense_attention_impl(opaque, output, query, key, value,
                                           rows, heads, head_dim, scale,
                                           quantized, head_major_output);
}

int h3_gpu_sdpa_bf16_head_major_output(
                     h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    return h3_gpu_sdpa(opaque, output, query, key, value, 1, sequence, heads,
                       head_dim, scale, H3_GPU_BF16, MPSDataTypeBFloat16, 0,
                       1);
}

int h3_gpu_swiglu_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, fused, (size_t)rows * width * 2, @"SwiGLU input") ||
        !h3_gpu_require_elements(gpu, output, (size_t)rows * width, @"SwiGLU output")) return 0;
    swiglu_args args = {rows, width};
    return h3_gpu_dispatch_2d(gpu, @"h3_swiglu_f32", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(fused).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_scale_add_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *scale, uint32_t rows,
                         uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, residual, count, @"scale-add residual") ||
        TENSOR(residual).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, branch, count, @"scale-add branch") ||
        TENSOR(branch).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, scale, width, @"scale-add scale") ||
        TENSOR(scale).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, @"scale-add output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    swiglu_args args = {rows, width};
    return h3_gpu_dispatch_2d(gpu, @"h3_scale_add_f32", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(branch).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(scale).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

int h3_gpu_layer_norm_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, @"LayerNorm input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, width, @"LayerNorm weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, bias, width, @"LayerNorm bias") ||
        TENSOR(bias).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, @"LayerNorm output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    norm_args args = {rows, width, epsilon};
    return h3_gpu_dispatch_rows(gpu, @"h3_layer_norm_f32", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

int h3_gpu_video_qkv_rope_f32(h3_gpu *opaque, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_gpu_require_elements(gpu, qkv, count * 3, @"video QKV") ||
        TENSOR(qkv).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, rope_cos, rope_count, @"video RoPE cosine") ||
        TENSOR(rope_cos).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, rope_sin, rope_count, @"video RoPE sine") ||
        TENSOR(rope_sin).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, query, count, @"video query") ||
        TENSOR(query).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, key, count, @"video key") ||
        TENSOR(key).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, value, count, @"video value") ||
        TENSOR(value).dtype != H3_GPU_F32 || rope_half * 2 > head_dim) return 0;
    qkv_args args = {sequence, heads, head_dim, rope_half, 0, epsilon};
    return h3_gpu_dispatch_3d(gpu, @"h3_video_qkv_rope_f32",
        MTLSizeMake(head_dim, heads, sequence),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
        });
}

static H3Conv *h3_gpu_conv_graph(H3GPU *gpu, uint32_t batch,
                                 uint32_t length, uint32_t input_channels,
                                 uint32_t output_channels, uint32_t kernel,
                                 uint32_t stride, uint32_t padding,
                                 uint32_t dilation, uint32_t output_length,
                                 int transpose, int has_bias) {
    @autoreleasepool {
        NSString *key = [NSString stringWithFormat:
            @"%d:%u:%u:%u:%u:%u:%u:%u:%u:%d", transpose, batch, length,
            input_channels, output_channels, kernel, stride, padding,
            dilation, has_bias];
        H3Conv *cached = gpu.convCache[key];
        if (cached) return cached;

        H3Conv *conv = [[H3Conv alloc] init];
        conv.graph = [[MPSGraph alloc] init];
        conv.inputShape = @[@(batch), @1, @(length), @(input_channels)];
        conv.weightShape = transpose ?
            @[@(input_channels), @(output_channels), @1, @(kernel)] :
            @[@(output_channels), @(input_channels), @1, @(kernel)];
        conv.biasShape = @[@1, @1, @1, @(output_channels)];
        conv.outputShape = @[@(batch), @1, @(output_length),
                             @(output_channels)];
        conv.input = [conv.graph placeholderWithShape:conv.inputShape
                                              dataType:MPSDataTypeFloat32
                                                  name:nil];
        conv.weight = [conv.graph placeholderWithShape:conv.weightShape
                                               dataType:MPSDataTypeFloat32
                                                   name:nil];
        MPSGraphConvolution2DOpDescriptor *descriptor =
            [MPSGraphConvolution2DOpDescriptor
                descriptorWithStrideInX:stride strideInY:1
                dilationRateInX:dilation dilationRateInY:1 groups:1
                paddingLeft:padding paddingRight:padding
                paddingTop:0 paddingBottom:0
                paddingStyle:MPSGraphPaddingStyleExplicit
                dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        MPSGraphTensor *result = transpose ?
            [conv.graph convolutionTranspose2DWithSourceTensor:conv.input
                 weightsTensor:conv.weight outputShape:conv.outputShape
                 descriptor:descriptor name:nil] :
            [conv.graph convolution2DWithSourceTensor:conv.input
                 weightsTensor:conv.weight descriptor:descriptor name:nil];
        if (has_bias) {
            conv.bias = [conv.graph placeholderWithShape:conv.biasShape
                                                dataType:MPSDataTypeFloat32
                                                    name:nil];
            result = [conv.graph additionWithPrimaryTensor:result
                                           secondaryTensor:conv.bias name:nil];
        }
        conv.output = result;
        gpu.convCache[key] = conv;
        return conv;
    }
}

static int h3_gpu_conv_mps(H3GPU *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t batch,
                           uint32_t length, uint32_t input_channels,
                           uint32_t output_channels, uint32_t kernel,
                           uint32_t stride, uint32_t padding,
                           uint32_t dilation, uint32_t output_length,
                           int transpose) {
    if (!h3_gpu_require_command(gpu)) return 0;
    H3Conv *conv = h3_gpu_conv_graph(
        gpu, batch, length, input_channels, output_channels, kernel, stride,
        padding, dilation, output_length, transpose, bias != NULL);
    if (!conv) return 0;
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        MPSGraphTensorData *input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(input).buffer shape:conv.inputShape
            dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *weight_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(weight).buffer shape:conv.weightShape
            dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(output).buffer shape:conv.outputShape
            dataType:MPSDataTypeFloat32];
        NSMutableDictionary *feeds = [@{conv.input: input_data,
                                         conv.weight: weight_data} mutableCopy];
        if (bias) {
            MPSGraphTensorData *bias_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:TENSOR(bias).buffer shape:conv.biasShape
                dataType:MPSDataTypeFloat32];
            feeds[conv.bias] = bias_data;
        }
        NSDictionary *results = @{conv.output: output_data};
        @try {
            [conv.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil resultsDictionary:results
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph Conv1d failed: %@",
                             exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_conv_dispatches++;
    gpu.stats = stats;
    return 1;
}

static H3Conv *h3_gpu_conv3d_graph(
        H3GPU *gpu, uint32_t batch, uint32_t depth, uint32_t height,
        uint32_t width, uint32_t input_channels, uint32_t output_channels,
        uint32_t kernel_depth, uint32_t kernel_height, uint32_t kernel_width,
        uint32_t stride_depth, uint32_t stride_height, uint32_t stride_width,
        uint32_t output_depth, uint32_t output_height, uint32_t output_width,
        int has_bias, int as_2d, uint32_t skipped_taps) {
    @autoreleasepool {
        NSString *key = [NSString stringWithFormat:
            @"3:%u:%u:%u:%u:%u:%u:%u:%u:%u:%u:%u:%u:%d:%d:%u", batch, depth,
            height, width, input_channels, output_channels, kernel_depth,
            kernel_height, kernel_width, stride_depth, stride_height,
            stride_width, has_bias, as_2d, skipped_taps];
        H3Conv *cached = gpu.convCache[key];
        if (cached) return cached;
        H3Conv *conv = [[H3Conv alloc] init];
        conv.graph = [[MPSGraph alloc] init];
        conv.inputShape = @[@(batch), @(depth), @(height), @(width),
                            @(input_channels)];
        conv.weightShape = @[@(output_channels), @(input_channels),
                             @(kernel_depth), @(kernel_height), @(kernel_width)];
        conv.biasShape = @[@1, @1, @1, @1, @(output_channels)];
        conv.outputShape = @[@(batch), @(output_depth), @(output_height),
                             @(output_width), @(output_channels)];
        conv.input = [conv.graph placeholderWithShape:conv.inputShape
                                              dataType:MPSDataTypeFloat32
                                                  name:nil];
        conv.weight = [conv.graph placeholderWithShape:conv.weightShape
                                               dataType:MPSDataTypeFloat32
                                                   name:nil];
        /* h3_gpu_conv3d_zero_front_f32: the taps that would meet the
         * skipped zero frames are left out of the kernel. */
        MPSGraphTensor *weights = conv.weight;
        if (skipped_taps) {
            weights = [conv.graph sliceTensor:weights dimension:2
                start:(NSInteger)skipped_taps
                length:(NSInteger)(kernel_depth - skipped_taps) name:nil];
            kernel_depth -= skipped_taps;
        }
        MPSGraphTensor *result = nil;
        if (as_2d) {
            /* The same convolution as one 2D convolution per time tap over
             * the frames (as a batch), summed: MPSGraph runs these about
             * 4.6x faster than its Conv3d on an M5, in the same F32 (only
             * the order of the sums differs). */
            MPSGraphConvolution2DOpDescriptor *descriptor2d =
                [MPSGraphConvolution2DOpDescriptor
                    descriptorWithStrideInX:stride_width strideInY:stride_height
                    dilationRateInX:1 dilationRateInY:1 groups:1
                    paddingLeft:0 paddingRight:0 paddingTop:0 paddingBottom:0
                    paddingStyle:MPSGraphPaddingStyleExplicit
                    dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                    weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
            MPSGraphTensor *source = conv.input;
            for (uint32_t tap = 0; tap < kernel_depth; tap++) {
                MPSGraphTensor *frames = [conv.graph sliceTensor:source
                    starts:@[@0, @(tap), @0, @0, @0]
                    ends:@[@(batch), @(tap + stride_depth * (output_depth - 1) + 1),
                           @(height), @(width), @(input_channels)]
                    strides:@[@1, @(stride_depth), @1, @1, @1] name:nil];
                frames = [conv.graph reshapeTensor:frames
                    withShape:@[@(batch * output_depth), @(height), @(width),
                                @(input_channels)] name:nil];
                MPSGraphTensor *tapWeight = [conv.graph sliceTensor:weights
                    dimension:2 start:(NSInteger)tap length:1 name:nil];
                tapWeight = [conv.graph reshapeTensor:tapWeight
                    withShape:@[@(output_channels), @(input_channels),
                                @(kernel_height), @(kernel_width)] name:nil];
                MPSGraphTensor *partial = [conv.graph
                    convolution2DWithSourceTensor:frames weightsTensor:tapWeight
                    descriptor:descriptor2d name:nil];
                result = result ? [conv.graph additionWithPrimaryTensor:result
                                                       secondaryTensor:partial name:nil]
                                : partial;
            }
            result = [conv.graph reshapeTensor:result withShape:conv.outputShape name:nil];
        } else {
            MPSGraphConvolution3DOpDescriptor *descriptor =
                [MPSGraphConvolution3DOpDescriptor
                    descriptorWithStrideInX:stride_width
                    strideInY:stride_height strideInZ:stride_depth
                    dilationRateInX:1 dilationRateInY:1 dilationRateInZ:1 groups:1
                    paddingLeft:0 paddingRight:0 paddingTop:0 paddingBottom:0
                    paddingFront:0 paddingBack:0
                    paddingStyle:MPSGraphPaddingStyleExplicit
                    dataLayout:MPSGraphTensorNamedDataLayoutNDHWC
                    weightsLayout:MPSGraphTensorNamedDataLayoutOIDHW];
            result = [conv.graph
                convolution3DWithSourceTensor:conv.input weightsTensor:weights
                descriptor:descriptor name:nil];
        }
        if (has_bias) {
            conv.bias = [conv.graph placeholderWithShape:conv.biasShape
                                                dataType:MPSDataTypeFloat32
                                                    name:nil];
            result = [conv.graph additionWithPrimaryTensor:result
                                           secondaryTensor:conv.bias name:nil];
        }
        conv.output = result;
        gpu.convCache[key] = conv;
        return conv;
    }
}

/* One time tap of a Conv3d as a 2D convolution over its frames (as a
 * batch): output = conv2d(frames, weight[:, :, tap]) (+ bias) (+ the sum
 * of the taps before). */
static H3Conv *h3_gpu_conv_tap_graph_padded(
        H3GPU *gpu, uint32_t output_depth, uint32_t height, uint32_t width,
        uint32_t input_channels, uint32_t output_channels,
        uint32_t kernel_depth, uint32_t kernel_height, uint32_t kernel_width,
        uint32_t stride_height, uint32_t stride_width, uint32_t output_height,
        uint32_t output_width, uint32_t tap, int has_bias, int accumulates,
        uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {
    @autoreleasepool {
        NSString *key = [NSString stringWithFormat:
            @"tap:%u:%u:%u:%u:%u:%u:%u:%u:%u:%u:%u:%d:%d:%u:%u:%u:%u", output_depth, height,
            width, input_channels, output_channels, kernel_depth, kernel_height,
            kernel_width, stride_height, stride_width, tap, has_bias, accumulates,
            pad_top, pad_bottom, pad_left, pad_right];
        H3Conv *cached = gpu.convCache[key];
        if (cached) return cached;
        H3Conv *conv = [[H3Conv alloc] init];
        conv.graph = [[MPSGraph alloc] init];
        conv.inputShape = @[@(output_depth), @(height), @(width), @(input_channels)];
        conv.weightShape = @[@(output_channels), @(input_channels),
                             @(kernel_depth), @(kernel_height), @(kernel_width)];
        conv.biasShape = @[@1, @1, @1, @(output_channels)];
        conv.outputShape = @[@(output_depth), @(output_height), @(output_width),
                             @(output_channels)];
        conv.input = [conv.graph placeholderWithShape:conv.inputShape
            dataType:MPSDataTypeFloat32 name:nil];
        conv.weight = [conv.graph placeholderWithShape:conv.weightShape
            dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphConvolution2DOpDescriptor *descriptor =
            [MPSGraphConvolution2DOpDescriptor
                descriptorWithStrideInX:stride_width strideInY:stride_height
                dilationRateInX:1 dilationRateInY:1 groups:1
                paddingLeft:pad_left paddingRight:pad_right paddingTop:pad_top paddingBottom:pad_bottom
                paddingStyle:MPSGraphPaddingStyleExplicit
                dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        MPSGraphTensor *weight = [conv.graph sliceTensor:conv.weight
            dimension:2 start:(NSInteger)tap length:1 name:nil];
        weight = [conv.graph reshapeTensor:weight
            withShape:@[@(output_channels), @(input_channels),
                        @(kernel_height), @(kernel_width)] name:nil];
        MPSGraphTensor *result = [conv.graph
            convolution2DWithSourceTensor:conv.input weightsTensor:weight
            descriptor:descriptor name:nil];
        if (has_bias) {
            conv.bias = [conv.graph placeholderWithShape:conv.biasShape
                dataType:MPSDataTypeFloat32 name:nil];
            result = [conv.graph additionWithPrimaryTensor:result
                                           secondaryTensor:conv.bias name:nil];
        }
        if (accumulates) {
            conv.accumulator = [conv.graph placeholderWithShape:conv.outputShape
                dataType:MPSDataTypeFloat32 name:nil];
            result = [conv.graph additionWithPrimaryTensor:conv.accumulator
                                           secondaryTensor:result name:nil];
        }
        conv.output = result;
        gpu.convCache[key] = conv;
        return conv;
    }
}

static H3Conv *h3_gpu_conv_tap_graph(
        H3GPU *gpu, uint32_t output_depth, uint32_t height, uint32_t width,
        uint32_t input_channels, uint32_t output_channels,
        uint32_t kernel_depth, uint32_t kernel_height, uint32_t kernel_width,
        uint32_t stride_height, uint32_t stride_width, uint32_t output_height,
        uint32_t output_width, uint32_t tap, int has_bias, int accumulates) {
    return h3_gpu_conv_tap_graph_padded(gpu, output_depth, height, width, input_channels,
                                        output_channels, kernel_depth, kernel_height, kernel_width,
                                        stride_height, stride_width, output_height, output_width,
                                        tap, has_bias, accumulates, 0, 0, 0, 0);
}

/* One frame, one tap, explicit zero padding inside the graph (no padded
 * copy of the input): F32 NHWC [height, width, C_in] -> [out_h, out_w,
 * C_out], weight OIHW [C_out, C_in, kh, kw] (the OIDHW of a 1-deep kernel). */
int h3_gpu_conv2d_padded_f32(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
                             uint32_t height, uint32_t width, uint32_t input_channels,
                             uint32_t output_channels, uint32_t kernel_height, uint32_t kernel_width,
                             uint32_t stride_height, uint32_t stride_width,
                             uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {
    H3GPU *gpu = GPU(opaque);
    uint32_t padded_h = height + pad_top + pad_bottom, padded_w = width + pad_left + pad_right;
    if (!height || !width || !input_channels || !output_channels || !kernel_height || !kernel_width ||
        !stride_height || !stride_width || padded_h < kernel_height || padded_w < kernel_width) return 0;
    uint32_t output_height = (padded_h - kernel_height) / stride_height + 1;
    uint32_t output_width = (padded_w - kernel_width) / stride_width + 1;
    if (!h3_gpu_require_elements(gpu, input, (size_t)height * width * input_channels, @"Conv2d input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, (size_t)output_channels * input_channels * kernel_height * kernel_width, @"Conv2d weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, (size_t)output_height * output_width * output_channels, @"Conv2d output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_channels, @"Conv2d bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32)) ||
        !h3_gpu_require_command(gpu)) return 0;
    H3Conv *conv = h3_gpu_conv_tap_graph_padded(
        gpu, 1, height, width, input_channels, output_channels, 1, kernel_height, kernel_width,
        stride_height, stride_width, output_height, output_width, 0, bias != NULL, 0,
        pad_top, pad_bottom, pad_left, pad_right);
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        MPSNDArrayDescriptor *frames = [MPSNDArrayDescriptor
            descriptorWithDataType:MPSDataTypeFloat32 shape:conv.inputShape];
        frames.preferPackedRows = YES;
        MPSNDArray *view = [[MPSNDArray alloc]
            initWithBuffer:TENSOR(input).buffer offset:0 descriptor:frames];
        NSMutableDictionary *feeds = [@{
            conv.input: [[MPSGraphTensorData alloc] initWithMPSNDArray:view],
            conv.weight: [[MPSGraphTensorData alloc]
                initWithMTLBuffer:TENSOR(weight).buffer shape:conv.weightShape
                dataType:MPSDataTypeFloat32]} mutableCopy];
        if (conv.bias)
            feeds[conv.bias] = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:TENSOR(bias).buffer shape:conv.biasShape
                dataType:MPSDataTypeFloat32];
        NSDictionary *results = @{conv.output: [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(output).buffer shape:conv.outputShape
            dataType:MPSDataTypeFloat32]};
        @try {
            [conv.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil resultsDictionary:results
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph Conv2d failed: %@", exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_conv_dispatches += 1;
    gpu.stats = stats;
    return 1;
}

/* h3_gpu_conv3d_as_2d_f32 for one batch with a time stride of 1, one tap
 * at a time: each tap reads its frames in place (a view into the input)
 * and adds into the sum so far, so only one tap's result is held at once -
 * about half the memory of the single graph, which keeps every tap's
 * frames and result alive together. */
static int h3_gpu_conv3d_by_tap(
        H3GPU *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
        const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
        uint32_t height, uint32_t width, uint32_t input_channels,
        uint32_t output_channels, uint32_t kernel_depth, uint32_t kernel_height,
        uint32_t kernel_width, uint32_t stride_height, uint32_t stride_width,
        uint32_t output_depth, uint32_t output_height, uint32_t output_width,
        uint32_t skipped_taps) {
    uint32_t taps = kernel_depth - skipped_taps;
    size_t output_bytes = (size_t)output_depth * output_height * output_width *
                          output_channels * sizeof(float);
    if (taps > 1 && gpu.convScratch.length < output_bytes) {
        gpu.convScratch = [gpu.device newBufferWithLength:output_bytes
                                                  options:MTLResourceStorageModePrivate];
        if (!gpu.convScratch) {
            h3_gpu_set_error(gpu, @"cannot allocate the Conv3d sum");
            return 0;
        }
    }
    size_t frame_bytes = (size_t)height * width * input_channels * sizeof(float);
    for (uint32_t run = 0; run < taps; run++) {
        uint32_t tap = skipped_taps + run;
        H3Conv *conv = h3_gpu_conv_tap_graph(
            gpu, output_depth, height, width, input_channels, output_channels,
            kernel_depth, kernel_height, kernel_width, stride_height, stride_width,
            output_height, output_width, tap, run == 0 && bias != NULL, run > 0);
        /* The last tap writes the output; the ones before alternate. */
        id<MTLBuffer> target = (taps - 1 - run) % 2 ? gpu.convScratch
                                                    : TENSOR(output).buffer;
        id<MTLBuffer> previous = (taps - run) % 2 ? gpu.convScratch
                                                  : TENSOR(output).buffer;
        @autoreleasepool {
            MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
            MPSNDArrayDescriptor *frames = [MPSNDArrayDescriptor
                descriptorWithDataType:MPSDataTypeFloat32 shape:conv.inputShape];
            /* As stored: no padding of the channel rows (3 at the input). */
            frames.preferPackedRows = YES;
            MPSNDArray *view = [[MPSNDArray alloc]
                initWithBuffer:TENSOR(input).buffer offset:run * frame_bytes
                descriptor:frames];
            NSMutableDictionary *feeds = [@{
                conv.input: [[MPSGraphTensorData alloc] initWithMPSNDArray:view],
                conv.weight: [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:TENSOR(weight).buffer shape:conv.weightShape
                    dataType:MPSDataTypeFloat32]} mutableCopy];
            if (conv.bias)
                feeds[conv.bias] = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:TENSOR(bias).buffer shape:conv.biasShape
                    dataType:MPSDataTypeFloat32];
            if (conv.accumulator)
                feeds[conv.accumulator] = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:previous shape:conv.outputShape
                    dataType:MPSDataTypeFloat32];
            NSDictionary *results = @{conv.output: [[MPSGraphTensorData alloc]
                initWithMTLBuffer:target shape:conv.outputShape
                dataType:MPSDataTypeFloat32]};
            @try {
                [conv.graph encodeToCommandBuffer:command feeds:feeds
                    targetOperations:nil resultsDictionary:results
                    executionDescriptor:nil];
            } @catch (NSException *exception) {
                h3_gpu_set_error(gpu, @"MPSGraph Conv3d tap failed: %@",
                                 exception.reason);
                return 0;
            }
            gpu.command = command.rootCommandBuffer;
        }
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_conv_dispatches += taps;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_conv3d_impl(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t input_channels, uint32_t output_channels,
                      uint32_t kernel_depth, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride_depth,
                      uint32_t stride_height, uint32_t stride_width, int as_2d,
                      uint32_t skipped_taps) {
    H3GPU *gpu = GPU(opaque);
    if (!batch || !depth || !height || !width || !input_channels ||
        !output_channels || !kernel_depth || !kernel_height || !kernel_width ||
        !stride_depth || !stride_height || !stride_width ||
        skipped_taps >= kernel_depth ||
        (skipped_taps && depth + skipped_taps != kernel_depth) ||
        depth + skipped_taps < kernel_depth || height < kernel_height ||
        width < kernel_width)
        return 0;
    uint32_t output_depth = (depth + skipped_taps - kernel_depth) / stride_depth + 1;
    uint32_t output_height = (height - kernel_height) / stride_height + 1;
    uint32_t output_width = (width - kernel_width) / stride_width + 1;
    size_t input_count = (size_t)batch * depth * height * width * input_channels;
    size_t weight_count = (size_t)output_channels * input_channels *
                          kernel_depth * kernel_height * kernel_width;
    size_t output_count = (size_t)batch * output_depth * output_height *
                          output_width * output_channels;
    if (!h3_gpu_require_elements(gpu, input, input_count, @"Conv3d input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, weight_count, @"Conv3d weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count, @"Conv3d output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_channels,
                                           @"Conv3d bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32)) ||
        !h3_gpu_require_command(gpu)) return 0;
    if (as_2d && batch == 1 && stride_depth == 1)
        return h3_gpu_conv3d_by_tap(
            gpu, output, input, weight, bias, height, width, input_channels,
            output_channels, kernel_depth, kernel_height, kernel_width,
            stride_height, stride_width, output_depth, output_height,
            output_width, skipped_taps);
    H3Conv *conv = h3_gpu_conv3d_graph(
        gpu, batch, depth, height, width, input_channels, output_channels,
        kernel_depth, kernel_height, kernel_width, stride_depth, stride_height,
        stride_width, output_depth, output_height, output_width, bias != NULL, as_2d,
        skipped_taps);
    if (!conv) return 0;
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        MPSGraphTensorData *input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(input).buffer shape:conv.inputShape
            dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *weight_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(weight).buffer shape:conv.weightShape
            dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(output).buffer shape:conv.outputShape
            dataType:MPSDataTypeFloat32];
        NSMutableDictionary *feeds = [@{conv.input: input_data,
                                         conv.weight: weight_data} mutableCopy];
        if (bias) {
            MPSGraphTensorData *bias_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:TENSOR(bias).buffer shape:conv.biasShape
                dataType:MPSDataTypeFloat32];
            feeds[conv.bias] = bias_data;
        }
        NSDictionary *results = @{conv.output: output_data};
        @try {
            [conv.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil resultsDictionary:results
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph Conv3d failed: %@",
                             exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_conv_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_conv3d_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t input_channels, uint32_t output_channels,
                      uint32_t kernel_depth, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride_depth,
                      uint32_t stride_height, uint32_t stride_width) {
    return h3_gpu_conv3d_impl(opaque, output, input, weight, bias, batch, depth,
                              height, width, input_channels, output_channels,
                              kernel_depth, kernel_height, kernel_width,
                              stride_depth, stride_height, stride_width, 0, 0);
}

int h3_gpu_conv3d_as_2d_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *bias, uint32_t batch,
                            uint32_t depth, uint32_t height, uint32_t width,
                            uint32_t input_channels, uint32_t output_channels,
                            uint32_t kernel_depth, uint32_t kernel_height,
                            uint32_t kernel_width, uint32_t stride_depth,
                            uint32_t stride_height, uint32_t stride_width) {
    return h3_gpu_conv3d_impl(opaque, output, input, weight, bias, batch, depth,
                              height, width, input_channels, output_channels,
                              kernel_depth, kernel_height, kernel_width,
                              stride_depth, stride_height, stride_width, 1, 0);
}

int h3_gpu_conv3d_zero_front_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *bias,
                                 uint32_t depth, uint32_t height, uint32_t width,
                                 uint32_t input_channels, uint32_t output_channels,
                                 uint32_t kernel_depth, uint32_t kernel_height,
                                 uint32_t kernel_width, uint32_t stride_depth,
                                 uint32_t stride_height, uint32_t stride_width,
                                 uint32_t zero_frames, int as_2d) {
    return h3_gpu_conv3d_impl(opaque, output, input, weight, bias, 1, depth,
                              height, width, input_channels, output_channels,
                              kernel_depth, kernel_height, kernel_width,
                              stride_depth, stride_height, stride_width, as_2d,
                              zero_frames);
}

int h3_gpu_conv1d_stride_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding,
                      uint32_t dilation) {
    H3GPU *gpu = GPU(opaque);
    uint64_t effective = (uint64_t)dilation * (kernel - 1) + 1;
    if (!batch || !length || !input_channels || !output_channels || !kernel ||
        !stride || !dilation || (uint64_t)length + 2 * padding < effective)
        return 0;
    uint32_t output_length = (uint32_t)(((uint64_t)length + 2 * padding -
                                         effective) / stride + 1);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)output_channels * input_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (!h3_gpu_require_elements(gpu, input, input_count, @"Conv1d input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, weight_count, @"Conv1d weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count, @"Conv1d output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_channels,
                                           @"Conv1d bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32))) return 0;
    return h3_gpu_conv_mps(gpu, output, input, weight, bias, batch, length,
        input_channels, output_channels, kernel, stride, padding, dilation,
        output_length, 0);
}

int h3_gpu_conv1d_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation) {
    return h3_gpu_conv1d_stride_f32(opaque, output, input, weight, bias,
        batch, length, input_channels, output_channels, kernel, 1, padding,
        dilation);
}

int h3_gpu_conv_transpose1d_f32(
                      h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding) {
    H3GPU *gpu = GPU(opaque);
    if (!batch || !length || !input_channels || !output_channels || !kernel ||
        !stride || (uint64_t)(length - 1) * stride + kernel < 2 * padding)
        return 0;
    uint32_t output_length = (uint32_t)((uint64_t)(length - 1) * stride +
                                        kernel - 2 * padding);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)input_channels * output_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (!h3_gpu_require_elements(gpu, input, input_count,
                                 @"ConvTranspose1d input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, weight_count,
                                 @"ConvTranspose1d weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count,
                                 @"ConvTranspose1d output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_channels,
                                           @"ConvTranspose1d bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32))) return 0;
    return h3_gpu_conv_mps(gpu, output, input, weight, bias, batch, length,
        input_channels, output_channels, kernel, stride, padding, 1,
        output_length, 1);
}

int h3_gpu_weight_norm_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude,
                           uint32_t outer, uint32_t inner) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)outer * inner;
    if (!outer || !inner ||
        !h3_gpu_require_elements(gpu, vector, count, @"weight-norm vector") ||
        TENSOR(vector).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, magnitude, outer,
                                 @"weight-norm magnitude") ||
        TENSOR(magnitude).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, @"weight-norm output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    weight_norm_args args = {outer, inner};
    return h3_gpu_dispatch_1d(gpu, @"h3_weight_norm_f32", outer,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(vector).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(magnitude).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_add_scaled_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left,
                          const h3_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, left, elements, @"scaled-add left") ||
        TENSOR(left).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, right, elements, @"scaled-add right") ||
        TENSOR(right).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, @"scaled-add output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    add_scaled_args args = {elements, left_scale, right_scale};
    return h3_gpu_dispatch_1d(gpu, @"h3_add_scaled_f32", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(left).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(right).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_alias_free_snake_f32(
                          h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *alpha_log,
                          const h3_gpu_tensor *beta_log,
                          const h3_gpu_tensor *upsample_filter,
                          const h3_gpu_tensor *downsample_filter,
                          uint32_t batch, uint32_t length,
                          uint32_t channels) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)batch * length * channels;
    if (!batch || !length || !channels ||
        !h3_gpu_require_elements(gpu, input, count, @"Snake input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, @"Snake output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, alpha_log, channels, @"Snake alpha") ||
        TENSOR(alpha_log).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, beta_log, channels, @"Snake beta") ||
        TENSOR(beta_log).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, upsample_filter, 12,
                                 @"Snake upsample filter") ||
        TENSOR(upsample_filter).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, downsample_filter, 12,
                                 @"Snake downsample filter") ||
        TENSOR(downsample_filter).dtype != H3_GPU_F32) return 0;
    audio_activation_args args = {batch, length, channels};
    return h3_gpu_dispatch_3d(gpu, @"h3_alias_free_snake_f32",
        MTLSizeMake(channels, length, batch),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(alpha_log).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(beta_log).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(upsample_filter).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(downsample_filter).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
        });
}

int h3_gpu_snake1d_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *alpha, uint32_t batch,
                       uint32_t length, uint32_t channels) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)batch * length * channels;
    if (!batch || !length || !channels || count > UINT32_MAX ||
        !h3_gpu_require_elements(gpu, input, count, @"Snake1d input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, alpha, channels, @"Snake1d alpha") ||
        TENSOR(alpha).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, @"Snake1d output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    audio_activation_args args = {batch, length, channels};
    return h3_gpu_dispatch_1d(gpu, @"h3_snake1d_f32", (uint32_t)count,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(alpha).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_audio_qkv_split_f32(h3_gpu *opaque,
                       h3_gpu_tensor *query, h3_gpu_tensor *key,
                       h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                       const h3_gpu_tensor *q_bias,
                       const h3_gpu_tensor *k_bias,
                       const h3_gpu_tensor *v_bias, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t width = (size_t)heads * head_dim;
    size_t count = (size_t)batch * length * width;
    if (!batch || !length || !heads || !head_dim || count > UINT32_MAX ||
        !h3_gpu_require_elements(gpu, qkv, count * 3, @"audio QKV") ||
        TENSOR(qkv).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, q_bias, width, @"audio Q bias") ||
        !h3_gpu_require_elements(gpu, k_bias, width, @"audio K bias") ||
        !h3_gpu_require_elements(gpu, v_bias, width, @"audio V bias") ||
        TENSOR(q_bias).dtype != H3_GPU_F32 ||
        TENSOR(k_bias).dtype != H3_GPU_F32 ||
        TENSOR(v_bias).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, query, count, @"audio query") ||
        !h3_gpu_require_elements(gpu, key, count, @"audio key") ||
        !h3_gpu_require_elements(gpu, value, count, @"audio value") ||
        TENSOR(query).dtype != H3_GPU_F32 || TENSOR(key).dtype != H3_GPU_F32 ||
        TENSOR(value).dtype != H3_GPU_F32) return 0;
    audio_qkv_args args = {batch, length, heads, head_dim};
    return h3_gpu_dispatch_1d(gpu, @"h3_audio_qkv_split_f32",
        (uint32_t)count, ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(q_bias).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(k_bias).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(v_bias).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:5];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:6];
            [encoder setBytes:&args length:sizeof(args) atIndex:7];
        });
}

int h3_gpu_audio_attention_pool_f32(h3_gpu *opaque,
                       h3_gpu_tensor *output,
                       const h3_gpu_tensor *attended, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim, uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t input_count = (size_t)batch * length * heads * head_dim;
    size_t output_count = (size_t)batch * length * output_dim;
    if (!batch || !length || !heads || !head_dim || !output_dim ||
        head_dim % output_dim || output_count > UINT32_MAX ||
        !h3_gpu_require_elements(gpu, attended, input_count,
                                 @"audio attended values") ||
        TENSOR(attended).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count,
                                 @"audio pooled values") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    audio_pool_args args = {batch, length, heads, head_dim, output_dim};
    return h3_gpu_dispatch_1d(gpu, @"h3_audio_attention_pool_f32",
        (uint32_t)output_count, ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(attended).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_geglu_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate,
                     const h3_gpu_tensor *linear, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, gate, elements, @"GeGLU gate") ||
        !h3_gpu_require_elements(gpu, linear, elements, @"GeGLU linear") ||
        !h3_gpu_require_elements(gpu, output, elements, @"GeGLU output") ||
        TENSOR(gate).dtype != H3_GPU_F32 ||
        TENSOR(linear).dtype != H3_GPU_F32 ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_geglu_f32", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(gate).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(linear).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:3];
        });
}

int h3_gpu_clip_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum) {
    H3GPU *gpu = GPU(opaque);
    if (!(minimum <= maximum) ||
        !h3_gpu_require_elements(gpu, input, elements, @"clip input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, @"clip output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    clip_args args = {elements, minimum, maximum};
    return h3_gpu_dispatch_1d(gpu, @"h3_clip_f32", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_vae_encoder_pad_f32(
                    h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t batch,
                    uint32_t depth, uint32_t height, uint32_t width,
                    uint32_t channels, uint32_t depth_front,
                    uint32_t height_before, uint32_t height_after,
                    uint32_t width_before, uint32_t width_after) {
    H3GPU *gpu = GPU(opaque);
    if (!batch || !depth || height < 2 || width < 2 || !channels ||
        height_before >= height || height_after >= height ||
        width_before >= width || width_after >= width) return 0;
    uint32_t output_depth = depth + depth_front;
    uint32_t output_height = height + height_before + height_after;
    uint32_t output_width = width + width_before + width_after;
    size_t input_count = (size_t)batch * depth * height * width * channels;
    size_t output_count = (size_t)batch * output_depth * output_height *
                          output_width * channels;
    if (!h3_gpu_require_elements(gpu, input, input_count,
                                 @"VAE encoder pad input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count,
                                 @"VAE encoder pad output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    vae_encoder_pad_args args = {
        batch, depth, height, width, channels, depth_front,
        height_before, height_after, width_before, width_after
    };
    return h3_gpu_dispatch_3d(gpu, @"h3_vae_encoder_pad_f32",
        MTLSizeMake(channels, output_width,
                    (NSUInteger)batch * output_depth * output_height),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
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

static int h3_gpu_require_bf16(H3GPU *gpu, const h3_gpu_tensor *tensor,
                               size_t elements, NSString *label);

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

int h3_gpu_vae_encoder_group_norm_silu_f32(
                      h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t groups, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)batch * depth * height * width * channels;
    if (!batch || !depth || !height || !width || !channels || !groups ||
        channels % groups || !(epsilon > 0.0f) ||
        !h3_gpu_require_elements(gpu, input, count,
                                 @"VAE encoder norm input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, channels,
                                 @"VAE encoder norm weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, bias, channels,
                                 @"VAE encoder norm bias") ||
        TENSOR(bias).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count,
                                 @"VAE encoder norm output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    vae_encoder_norm_args args = {
        batch, depth, height, width, channels, groups, epsilon
    };
    uint64_t rows = (uint64_t)batch * depth * groups;
    if (rows > UINT32_MAX) return 0;
    return h3_gpu_dispatch_rows(
        gpu, @"h3_vae_encoder_group_norm_silu_f32", (uint32_t)rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

int h3_gpu_vae_encoder_group_norm_silu_chunked_f32(
                      h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t groups, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    uint32_t vectors = channels / 4;
    /* Shapes the three passes don't cover take the one-pass kernel. */
    if (!groups || groups > 64 || channels % 4 || !vectors || vectors > 256 ||
        256 % vectors || channels % groups || (channels / groups) % 4)
        return h3_gpu_vae_encoder_group_norm_silu_f32(
            opaque, output, input, weight, bias, batch, depth, height, width,
            channels, groups, epsilon);
    size_t count = (size_t)batch * depth * height * width * channels;
    if (!batch || !depth || !height || !width || !(epsilon > 0.0f) ||
        !h3_gpu_require_elements(gpu, input, count, @"VAE encoder norm input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, channels, @"VAE encoder norm weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, bias, channels, @"VAE encoder norm bias") ||
        TENSOR(bias).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, @"VAE encoder norm output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        !h3_gpu_require_command(gpu)) return 0;
    uint32_t frames = batch * depth, pixels = height * width;
    /* Up to 64 runs a frame, each at least one pass of the threadgroup. */
    uint32_t lanes = 256 / vectors;
    uint32_t chunks = MIN((uint32_t)64, (pixels + lanes - 1) / lanes);
    uint32_t chunk_pixels = (pixels + chunks - 1) / chunks;
    chunks = (pixels + chunk_pixels - 1) / chunk_pixels;
    vae_encoder_norm_chunk_args args = {
        frames, pixels, channels, groups, chunks, chunk_pixels, epsilon
    };
    size_t partial_bytes = (size_t)frames * chunks * groups * sizeof(float);
    if (gpu.normScratch.length < 2 * partial_bytes) {
        gpu.normScratch = [gpu.device newBufferWithLength:2 * partial_bytes
                                                  options:MTLResourceStorageModePrivate];
        if (!gpu.normScratch) {
            h3_gpu_set_error(gpu, @"cannot allocate the GroupNorm sums");
            return 0;
        }
    }
    NSString *names[3] = {@"h3_vae_encoder_norm_chunk_sum_f32",
                          @"h3_vae_encoder_norm_chunk_square_f32",
                          @"h3_vae_encoder_norm_chunk_apply_silu_f32"};
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
            id<MTLBuffer> scratch = gpu.normScratch;
            int index = 0;
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:index++];
            [encoder setBuffer:scratch offset:0 atIndex:index++];
            if (pass > 0) [encoder setBuffer:scratch offset:partial_bytes atIndex:index++];
            if (pass == 2) {
                [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:index++];
                [encoder setBuffer:TENSOR(bias).buffer offset:0 atIndex:index++];
                [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:index++];
            }
            [encoder setBytes:&args length:sizeof(args) atIndex:index];
            [encoder dispatchThreadgroups:MTLSizeMake(chunks, frames, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
        }
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches += 3;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_require_bf16(H3GPU *gpu, const h3_gpu_tensor *tensor,
                               size_t elements, NSString *label) {
    if (!h3_gpu_require_elements(gpu, tensor, elements, label)) return 0;
    if (TENSOR(tensor).dtype != H3_GPU_BF16) {
        h3_gpu_set_error(gpu, @"%@ tensor is not BF16", label);
        return 0;
    }
    return 1;
}

static int h3_gpu_require_i8(H3GPU *gpu, const h3_gpu_tensor *tensor,
                             size_t elements, NSString *label) {
    if (!h3_gpu_require_elements(gpu, tensor, elements, label)) return 0;
    if (TENSOR(tensor).dtype != H3_GPU_I8) {
        h3_gpu_set_error(gpu, @"%@ tensor is not int8", label);
        return 0;
    }
    return 1;
}

static int h3_gpu_require_f32(H3GPU *gpu, const h3_gpu_tensor *tensor,
                              size_t elements, NSString *label) {
    if (!h3_gpu_require_elements(gpu, tensor, elements, label)) return 0;
    if (TENSOR(tensor).dtype != H3_GPU_F32) {
        h3_gpu_set_error(gpu, @"%@ tensor is not F32", label);
        return 0;
    }
    return 1;
}

static H3Linear *h3_gpu_linear_graph(H3GPU *gpu, uint32_t rows,
                                     uint32_t input_dim, uint32_t output_dim,
                                     int has_bias, MPSDataType dataType) {
    @autoreleasepool {
        NSString *key = [NSString stringWithFormat:@"%u:%u:%u:%d:%u", rows,
                         input_dim, output_dim, has_bias, (unsigned)dataType];
        H3Linear *cached = gpu.linearCache[key];
        if (cached) return cached;

        H3Linear *linear = [[H3Linear alloc] init];
        linear.graph = [[MPSGraph alloc] init];
        linear.inputShape = @[@1, @(rows), @(input_dim)];
        linear.weightShape = @[@1, @(output_dim), @(input_dim)];
        linear.biasShape = @[@1, @1, @(output_dim)];
        linear.outputShape = @[@1, @(rows), @(output_dim)];
        linear.input = [linear.graph placeholderWithShape:linear.inputShape
                                                 dataType:dataType name:nil];
        linear.weight = [linear.graph placeholderWithShape:linear.weightShape
                                                  dataType:dataType name:nil];
        MPSGraphTensor *transposed =
            [linear.graph transposeTensor:linear.weight dimension:1
                            withDimension:2 name:nil];
        MPSGraphTensor *output =
            [linear.graph matrixMultiplicationWithPrimaryTensor:linear.input
                                                secondaryTensor:transposed name:nil];
        if (has_bias) {
            linear.bias = [linear.graph placeholderWithShape:linear.biasShape
                                                    dataType:dataType name:nil];
            output = [linear.graph additionWithPrimaryTensor:output
                                             secondaryTensor:linear.bias name:nil];
        }
        linear.output = [linear.graph castTensor:output toType:dataType name:nil];
        gpu.linearCache[key] = linear;
        return linear;
    }
}

static int h3_gpu_linear_mps(H3GPU *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim,
                             MPSDataType dataType) {
    if (!h3_gpu_require_command(gpu)) return 0;
    H3Linear *linear = h3_gpu_linear_graph(gpu, rows, input_dim, output_dim,
                                           bias != NULL, dataType);
    if (!linear) return 0;
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        MPSGraphTensorData *input_data = h3_gpu_graph_data(
            input, linear.inputShape, dataType, 0);
        MPSGraphTensorData *weight_data = h3_gpu_graph_data(
            weight, linear.weightShape, dataType, 1);
        MPSGraphTensorData *output_data = h3_gpu_graph_data(
            output, linear.outputShape, dataType, 0);
        NSMutableDictionary *feeds = [@{linear.input: input_data,
                                         linear.weight: weight_data} mutableCopy];
        if (bias) {
            MPSGraphTensorData *bias_data = h3_gpu_graph_data(
                bias, linear.biasShape, dataType, 1);
            feeds[linear.bias] = bias_data;
        }
        NSDictionary *results = @{linear.output: output_data};
        @try {
            [linear.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil resultsDictionary:results
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph linear failed: %@", exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_linear_dispatches++;
    gpu.stats = stats;
    return 1;
}

typedef struct {
    uint32_t rows, columns, interleaved, group, head_dim, seed;
} lora_rows_args;

/* delta[rows, columns] = b[rows, rank] @ at[columns, rank]^T. Always the
 * MPSGraph matmul: LoRA ranks are usually below the inner-dimension bound
 * h3_gpu_linear_bf16 applies before choosing it. */
int h3_gpu_lora_delta_bf16(h3_gpu *opaque, h3_gpu_tensor *delta,
                           const h3_gpu_tensor *b, const h3_gpu_tensor *at,
                           uint32_t rows, uint32_t rank, uint32_t columns) {
    H3GPU *gpu = GPU(opaque);
    if (!rows || !rank || !columns ||
        !h3_gpu_require_bf16(gpu, b, (size_t)rows * rank, @"LoRA B") ||
        !h3_gpu_require_bf16(gpu, at, (size_t)columns * rank, @"LoRA A") ||
        !h3_gpu_require_bf16(gpu, delta, (size_t)rows * columns,
                             @"LoRA delta")) return 0;
    /* M5 TensorOps: the same BF16 tile kernel the DiT's own projections use
     * (h3_lora pads ranks to a multiple of 32 for it). MPSGraph's BF16
     * matmul measured ~4x slower at these skinny shapes. */
    if (gpu.tensorOpsEnabled && rows >= 128 && !(rank % 32) &&
        !(columns % 64) && !getenv("H3_LORA_MPS")) {
        if (!h3_gpu_require_command(gpu)) return 0;
        id<MTLComputePipelineState> pipeline =
            h3_gpu_pipeline(gpu, @"h3_linear_bf16_nax_r128");
        if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 128) {
            h3_gpu_set_error(gpu, @"device cannot dispatch M5 BF16 TensorOps");
            return 0;
        }
        linear_args args = {rows, rank, columns, 0};
        @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder =
                [gpu.command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:TENSOR(b).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(at).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(delta).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
            [encoder dispatchThreadgroups:
                MTLSizeMake((rows + 127) / 128, (columns + 63) / 64, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
        }
        h3_gpu_stats stats = gpu.stats;
        stats.direct_dispatches++;
        gpu.stats = stats;
        return 1;
    }
    return h3_gpu_linear_mps(gpu, delta, b, at, NULL, rows, rank, columns,
                             MPSDataTypeBFloat16);
}

int h3_gpu_lora_add_rows_bf16(h3_gpu *opaque, h3_gpu_tensor *weight,
                              const h3_gpu_tensor *delta, uint32_t rows,
                              uint32_t columns, uint32_t interleaved,
                              uint32_t group, uint32_t head_dim,
                              uint32_t seed) {
    H3GPU *gpu = GPU(opaque);
    size_t weight_rows = interleaved ? (size_t)rows * 3 : rows;
    if (!h3_gpu_require_bf16(gpu, weight, weight_rows * columns,
                             @"LoRA target weight") ||
        !h3_gpu_require_bf16(gpu, delta, (size_t)rows * columns,
                             @"LoRA delta")) return 0;
    lora_rows_args args = {rows, columns, interleaved, group, head_dim, seed};
    return h3_gpu_dispatch_2d(gpu, @"h3_lora_add_rows_bf16", columns, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(delta).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_lora_add_int8(h3_gpu *opaque, h3_gpu_tensor *weight,
                         const h3_gpu_tensor *scales,
                         const h3_gpu_tensor *delta, uint32_t rows,
                         uint32_t columns, uint32_t interleaved,
                         uint32_t group, uint32_t head_dim, uint32_t seed) {
    H3GPU *gpu = GPU(opaque);
    size_t weight_rows = interleaved ? (size_t)rows * 3 : rows;
    if (!h3_gpu_require_i8(gpu, weight, weight_rows * columns,
                           @"LoRA target int8 weight") ||
        !h3_gpu_require_f32(gpu, scales, weight_rows,
                            @"LoRA target int8 scales") ||
        !h3_gpu_require_bf16(gpu, delta, (size_t)rows * columns,
                             @"LoRA delta")) return 0;
    lora_rows_args args = {rows, columns, interleaved, group, head_dim, seed};
    return h3_gpu_dispatch_2d(gpu, @"h3_lora_add_int8", columns, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(scales).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(delta).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_linear_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!h3_gpu_require_bf16(gpu, input, input_count, @"linear input") ||
        !h3_gpu_require_bf16(gpu, weight, weight_count, @"linear weight") ||
        !h3_gpu_require_bf16(gpu, output, output_count, @"linear output") ||
        (bias && !h3_gpu_require_bf16(gpu, bias, output_dim, @"linear bias"))) return 0;
    const char *splitRowsValue = getenv("H3_NAX_SPLIT_ROWS");
    BOOL autoSplitRows = gpu.tensorOpsEnabled && rows > 2048 && rows <= 3072 &&
        (gpu.tensorOpsMode == 2 || gpu.tensorOpsMode == 3 ||
         gpu.tensorOpsMode == 4);
    BOOL splitRows = rows > 2048 &&
        (autoSplitRows || (splitRowsValue && *splitRowsValue));
    BOOL specializedRows = rows <= 2048 || splitRows;
    BOOL naxShape = gpu.tensorOpsMode == 1 || gpu.naxLinearGeneric ||
        (specializedRows && gpu.tensorOpsMode == 2 &&
         input_dim == 7168 && output_dim == 5376) ||
        (specializedRows && gpu.tensorOpsMode == 3 &&
         ((input_dim == 7168 && output_dim == 5376) ||
          (input_dim == 5376 && output_dim == 21504))) ||
        (specializedRows && gpu.tensorOpsMode == 4 &&
         input_dim == 5376 && output_dim == 21504);
    if (gpu.tensorOpsEnabled && naxShape && !bias && rows >= 128 &&
        !getenv("H3_DISABLE_NAX_LINEAR") &&
        (input_dim % 32) == 0 && (output_dim % 64) == 0) {
        if (!h3_gpu_require_command(gpu)) return 0;
        uint32_t column_tiles = (output_dim + 63) / 64;
        BOOL morton4 = (rows <= 2048 || splitRows) &&
                       !(column_tiles % 4) &&
                       getenv("H3_DISABLE_NAX_LINEAR_MORTON4") == NULL;
        id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
            gpu, morton4 ? @"h3_linear_bf16_nax_r128_morton4" :
                           @"h3_linear_bf16_nax_r128");
        if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 128) {
            h3_gpu_set_error(gpu, @"device cannot dispatch M5 BF16 TensorOps");
            return 0;
        }
        uint32_t splitAt = splitRows ? 2048 : rows;
        if (splitRows) {
            unsigned long requested = splitRowsValue ?
                strtoul(splitRowsValue, NULL, 10) : 0;
            if (requested >= 128 && requested <= 2048)
                splitAt = (uint32_t)requested;
        }
        uint32_t dispatches = 0;
        for (uint32_t rowOffset = 0; rowOffset < rows;
             rowOffset += splitAt) @autoreleasepool {
            uint32_t chunkRows = MIN(splitAt, rows - rowOffset);
            linear_args args = {chunkRows, input_dim, output_dim, 0};
            uint32_t row_tiles = (chunkRows + 127) / 128;
            id<MTLComputeCommandEncoder> encoder =
                [gpu.command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:TENSOR(input).buffer
                         offset:(NSUInteger)rowOffset * input_dim *
                                sizeof(uint16_t) atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer
                         offset:(NSUInteger)rowOffset * output_dim *
                                sizeof(uint16_t) atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
            if (morton4) {
                [encoder dispatchThreadgroups:
                    MTLSizeMake((NSUInteger)row_tiles * column_tiles, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                [encoder dispatchThreadgroups:
                    MTLSizeMake(row_tiles, column_tiles, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            [encoder endEncoding];
            dispatches++;
        }
        h3_gpu_stats stats = gpu.stats;
        stats.direct_dispatches += dispatches;
        gpu.stats = stats;
        return 1;
    }
    if (rows >= 32 && input_dim >= 256 && output_dim >= 256 &&
        h3_gpu_linear_mps(gpu, output, input, weight, bias, rows,
                          input_dim, output_dim,
                          MPSDataTypeBFloat16)) return 1;
    linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu,
                                                           @"h3_linear_bf16");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu, @"device cannot dispatch the 16x16 BF16 linear tile");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake((output_dim + 15) / 16,
                                                  (rows + 15) / 16, 1)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static H3MLP *h3_gpu_mlp_graph(H3GPU *gpu, uint32_t rows,
                               uint32_t input_dim, uint32_t hidden_dim,
                               uint32_t output_dim) {
    @autoreleasepool {
        NSString *key = [NSString stringWithFormat:@"%u:%u:%u:%u", rows,
                         input_dim, hidden_dim, output_dim];
        H3MLP *cached = gpu.mlpCache[key];
        if (cached) return cached;

        H3MLP *mlp = [[H3MLP alloc] init];
        mlp.graph = [[MPSGraph alloc] init];
        mlp.inputShape = @[@1, @(rows), @(input_dim)];
        mlp.fc1Shape = @[@1, @(hidden_dim * 2), @(input_dim)];
        mlp.fc2Shape = @[@1, @(output_dim), @(hidden_dim)];
        mlp.outputShape = @[@1, @(rows), @(output_dim)];
        mlp.input = [mlp.graph placeholderWithShape:mlp.inputShape
                                           dataType:MPSDataTypeBFloat16 name:nil];
        mlp.fc1Weight = [mlp.graph placeholderWithShape:mlp.fc1Shape
                                               dataType:MPSDataTypeBFloat16 name:nil];
        mlp.fc2Weight = [mlp.graph placeholderWithShape:mlp.fc2Shape
                                               dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *fc1Transposed =
            [mlp.graph transposeTensor:mlp.fc1Weight dimension:1
                         withDimension:2 name:nil];
        MPSGraphTensor *fused =
            [mlp.graph matrixMultiplicationWithPrimaryTensor:mlp.input
                                             secondaryTensor:fc1Transposed name:nil];
        NSArray<MPSGraphTensor *> *halves =
            [mlp.graph splitTensor:fused numSplits:2 axis:2 name:nil];
        MPSGraphTensor *sigmoid = [mlp.graph sigmoidWithTensor:halves[0]
                                                              name:nil];
        MPSGraphTensor *silu =
            [mlp.graph multiplicationWithPrimaryTensor:halves[0]
                                       secondaryTensor:sigmoid name:nil];
        MPSGraphTensor *activated =
            [mlp.graph multiplicationWithPrimaryTensor:silu
                                       secondaryTensor:halves[1] name:nil];
        MPSGraphTensor *fc2Transposed =
            [mlp.graph transposeTensor:mlp.fc2Weight dimension:1
                         withDimension:2 name:nil];
        MPSGraphTensor *result =
            [mlp.graph matrixMultiplicationWithPrimaryTensor:activated
                                             secondaryTensor:fc2Transposed name:nil];
        mlp.output = [mlp.graph castTensor:result
                                    toType:MPSDataTypeBFloat16 name:nil];
        gpu.mlpCache[key] = mlp;
        return mlp;
    }
}

int h3_gpu_mlp_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input,
                    const h3_gpu_tensor *fc1_weight,
                    const h3_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, input, (size_t)rows * input_dim,
                             @"MLP input") ||
        !h3_gpu_require_bf16(gpu, fc1_weight,
                             (size_t)hidden_dim * 2 * input_dim,
                             @"MLP fc1 weight") ||
        !h3_gpu_require_bf16(gpu, fc2_weight,
                             (size_t)output_dim * hidden_dim,
                             @"MLP fc2 weight") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * output_dim,
                             @"MLP output") ||
        !h3_gpu_require_command(gpu)) return 0;
    H3MLP *mlp = h3_gpu_mlp_graph(gpu, rows, input_dim, hidden_dim,
                                  output_dim);
    if (!mlp) return 0;
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        MPSGraphTensorData *inputData = h3_gpu_graph_data(
            input, mlp.inputShape, MPSDataTypeBFloat16, 0);
        MPSGraphTensorData *fc1Data = h3_gpu_graph_data(
            fc1_weight, mlp.fc1Shape, MPSDataTypeBFloat16, 1);
        MPSGraphTensorData *fc2Data = h3_gpu_graph_data(
            fc2_weight, mlp.fc2Shape, MPSDataTypeBFloat16, 1);
        MPSGraphTensorData *outputData = h3_gpu_graph_data(
            output, mlp.outputShape, MPSDataTypeBFloat16, 0);
        NSDictionary *feeds = @{mlp.input: inputData,
                                mlp.fc1Weight: fc1Data,
                                mlp.fc2Weight: fc2Data};
        NSDictionary *results = @{mlp.output: outputData};
        @try {
            [mlp.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil resultsDictionary:results
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph MLP failed: %@", exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_linear_dispatches += 2;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_fc1_swiglu_nax_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                                      const h3_gpu_tensor *input,
                                      const h3_gpu_tensor *weight,
                                      uint32_t rows, uint32_t input_dim,
                                      uint32_t hidden_dim) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu.tensorOpsEnabled ||
        (gpu.tensorOpsMode != 5 && !h3_gpu_has_int8_mlp(opaque)) ||
        !h3_gpu_require_bf16(gpu, input, (size_t)rows * input_dim,
                             @"NAX FC1 input") ||
        !h3_gpu_require_bf16(gpu, weight,
                             (size_t)hidden_dim * 2 * input_dim,
                             @"NAX FC1 weight") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * hidden_dim,
                             @"NAX FC1 output") ||
        !rows || (input_dim % 32) || (hidden_dim % 64) ||
        !h3_gpu_require_command(gpu)) return 0;
    BOOL morton = getenv("H3_DISABLE_NAX_MORTON") == NULL &&
                  getenv("H3_DISABLE_NAX_MORTON_FC1") == NULL;
    uint32_t row_tiles = (rows + 127) / 128;
    uint32_t column_tiles = (hidden_dim + 63) / 64;
    BOOL morton4 = morton && !(column_tiles % 4) &&
                   getenv("H3_DISABLE_NAX_MORTON4") == NULL &&
                   getenv("H3_DISABLE_NAX_MORTON4_FC1") == NULL;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, morton4 ? @"h3_fc1_swiglu_bf16_nax_r128_morton4" :
        morton ? @"h3_fc1_swiglu_bf16_nax_r128_morton" :
                 @"h3_fc1_swiglu_bf16_nax_r128");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 128) {
        h3_gpu_set_error(gpu, @"device cannot dispatch fused M5 FC1/SwiGLU");
        return 0;
    }
    linear_args args = {rows, input_dim, hidden_dim, 0};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        [encoder setThreadgroupMemoryLength:128 * 64 * 2 * sizeof(uint16_t)
                                   atIndex:0];
        if (morton4) {
            [encoder dispatchThreadgroups:
                MTLSizeMake((NSUInteger)row_tiles * column_tiles, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (morton) {
            uint32_t row_codes = 1, column_codes = 1;
            while (row_codes < row_tiles) row_codes <<= 1;
            while (column_codes < column_tiles) column_codes <<= 1;
            [encoder dispatchThreadgroups:
                MTLSizeMake((NSUInteger)row_codes * column_codes, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [encoder dispatchThreadgroups:
                MTLSizeMake(row_tiles, column_tiles, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_mlp_nax_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                        h3_gpu_tensor *activated,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *fc1_weight,
                        const h3_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu.tensorOpsEnabled || gpu.tensorOpsMode != 5 || rows < 128 ||
        (hidden_dim % 32) || (output_dim % 64) ||
        !h3_gpu_require_bf16(gpu, fc2_weight,
                             (size_t)output_dim * hidden_dim,
                             @"NAX FC2 weight") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * output_dim,
                             @"NAX FC2 output") ||
        !h3_gpu_fc1_swiglu_nax_bf16(opaque, activated, input, fc1_weight,
                                     rows, input_dim, hidden_dim)) return 0;
    BOOL morton = getenv("H3_DISABLE_NAX_MORTON") == NULL &&
                  getenv("H3_DISABLE_NAX_MORTON_FC2") == NULL;
    uint32_t row_tiles = (rows + 127) / 128;
    uint32_t column_tiles = (output_dim + 63) / 64;
    BOOL morton4 = morton && !(column_tiles % 4) &&
                   getenv("H3_DISABLE_NAX_MORTON4") == NULL &&
                   getenv("H3_DISABLE_NAX_MORTON4_FC2") == NULL;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, morton4 ? @"h3_linear_bf16_nax_r128_morton4" :
        morton ? @"h3_linear_bf16_nax_r128_morton" :
                 @"h3_linear_bf16_nax_r128");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 128) {
        h3_gpu_set_error(gpu, @"device cannot dispatch M5 BF16 TensorOps FC2");
        return 0;
    }
    linear_args args = {rows, hidden_dim, output_dim, 0};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(activated).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(fc2_weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        if (morton4) {
            [encoder dispatchThreadgroups:
                MTLSizeMake((NSUInteger)row_tiles * column_tiles, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (morton) {
            uint32_t row_codes = 1, column_codes = 1;
            while (row_codes < row_tiles) row_codes <<= 1;
            while (column_codes < column_tiles) column_codes <<= 1;
            [encoder dispatchThreadgroups:
                MTLSizeMake((NSUInteger)row_codes * column_codes, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [encoder dispatchThreadgroups:
                MTLSizeMake(row_tiles, column_tiles, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_quantize_bf16_int8_rows(
                        h3_gpu *opaque, h3_gpu_tensor *output,
                        h3_gpu_tensor *scales,
                        const h3_gpu_tensor *input, uint32_t rows,
                        uint32_t dispatch_rows, uint32_t columns,
                        float clip, NSString *label) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu.tensorOpsEnabled || !rows || dispatch_rows < rows || !columns ||
        !h3_gpu_require_bf16(gpu, input, (size_t)rows * columns, label) ||
        !h3_gpu_require_i8(gpu, output,
                           (size_t)dispatch_rows * columns,
                           @"int8 quantized output") ||
        !h3_gpu_require_f32(gpu, scales, dispatch_rows,
                            @"int8 quantization scales") ||
        !h3_gpu_require_command(gpu)) return 0;
    BOOL vector_quantizer =
        [label isEqualToString:@"int8 MLP FC2 input"] ||
        getenv("H3_INT8_VECTOR_QUANT");
    NSString *kernel = vector_quantizer ?
        @"h3_quantize_bf16_int8_rows" :
        @"h3_quantize_bf16_int8_rows_scalar";
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, kernel);
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu, @"device cannot dispatch M5 int8 quantizer");
        return 0;
    }
    int8_quant_args args = {rows, columns, clip};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(scales).buffer offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(dispatch_rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_quantize_bf16_int8_head_major_rows(
                        h3_gpu *opaque, h3_gpu_tensor *output,
                        h3_gpu_tensor *scales,
                        const h3_gpu_tensor *input, uint32_t rows,
                        uint32_t padded_rows, uint32_t heads,
                        uint32_t head_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t columns = (size_t)heads * head_dim;
    if (!gpu.tensorOpsEnabled || !rows || padded_rows < rows ||
        heads != 56 || head_dim != 128 ||
        !h3_gpu_require_bf16(gpu, input, (size_t)rows * columns,
                             @"head-major int8 linear input") ||
        !h3_gpu_require_i8(gpu, output, (size_t)padded_rows * columns,
                           @"head-major int8 quantized output") ||
        !h3_gpu_require_f32(gpu, scales, padded_rows,
                            @"head-major int8 quantization scales") ||
        !h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, @"h3_quantize_bf16_int8_head_major_to_rows_cached");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(
            gpu, @"device cannot dispatch M5 head-major int8 quantizer");
        return 0;
    }
    int8_head_major_quant_args args = {
        rows, padded_rows, heads, head_dim, 1.0f
    };
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(scales).buffer offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(padded_rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_quantize_bf16_int8_groups(
                        h3_gpu *opaque, h3_gpu_tensor *output,
                        h3_gpu_tensor *scales,
                        const h3_gpu_tensor *input, uint32_t rows,
                        uint32_t dispatch_rows, uint32_t columns,
                        uint32_t group_size,
                        int use_slower_grouped_quantizer) {
    H3GPU *gpu = GPU(opaque);
    if (!group_size || (group_size % 4)) return 0;
    uint32_t groups = (columns + group_size - 1) / group_size;
    if (!gpu.tensorOpsEnabled || !rows || dispatch_rows < rows || !columns ||
        (columns % group_size) ||
        !h3_gpu_require_bf16(gpu, input, (size_t)rows * columns,
                             @"grouped int8 input") ||
        !h3_gpu_require_i8(gpu, output,
                           (size_t)dispatch_rows * columns,
                           @"grouped int8 output") ||
        !h3_gpu_require_f32(gpu, scales,
                            (size_t)dispatch_rows * groups,
                            @"grouped int8 scales") ||
        !h3_gpu_require_command(gpu)) return 0;
    BOOL scalar128 = !use_slower_grouped_quantizer && rows <= 2048;
    const char *quantizer_override = getenv("H3_INT8_GROUP_QUANT_128");
    if (quantizer_override)
        scalar128 = *quantizer_override && strcmp(quantizer_override, "0");
    BOOL cached128 = scalar128 && group_size == 1024;
    const char *cache_override = getenv("H3_INT8_GROUP_QUANT_128_CACHE");
    if (cache_override)
        cached128 = cached128 && *cache_override &&
            strcmp(cache_override, "0");
    NSString *kernel = getenv("H3_INT8_VECTOR_QUANT") ?
        @"h3_quantize_bf16_int8_groups" : cached128 ?
        @"h3_quantize_bf16_int8_groups_scalar128_cached" : scalar128 ?
        @"h3_quantize_bf16_int8_groups_scalar128" :
        @"h3_quantize_bf16_int8_groups_scalar";
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, kernel);
    NSUInteger threads = scalar128 ? 128u : 256u;
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < threads) {
        h3_gpu_set_error(gpu,
                         @"device cannot dispatch grouped M5 int8 quantizer");
        return 0;
    }
    int8_group_quant_args args = {
        rows, columns, group_size, groups
    };
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(scales).buffer offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(dispatch_rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_quantize_weight_int8(h3_gpu *opaque, h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t columns) {
    return h3_gpu_quantize_bf16_int8_rows(
        opaque, output, scales, input, rows, rows, columns,
        1.0f, @"BF16 weight to quantize");
}

static int h3_gpu_linear_int8_bf16_layout(
                            h3_gpu *opaque, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales,
                            BOOL headMajorInput, uint32_t heads,
                            uint32_t headDim) {
    H3GPU *gpu = GPU(opaque);
    uint32_t padded_rows = (rows + 127u) & ~127u;
    if (!gpu.tensorOpsEnabled || rows < 128 || input_dim % 128 ||
        output_dim % 128 ||
        !h3_gpu_require_i8(gpu, weight, (size_t)output_dim * input_dim,
                           @"int8 linear weight") ||
        !h3_gpu_require_f32(gpu, weight_scales, output_dim,
                            @"int8 linear weight scales") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * output_dim,
                             @"int8 linear output") ||
        !h3_gpu_require_command(gpu)) return 0;
    if (headMajorInput) {
        if (use_slower_uncached_int8_scales || heads * headDim != input_dim ||
            !h3_gpu_quantize_bf16_int8_head_major_rows(
                opaque, quantized_input, input_scales, input, rows,
                padded_rows, heads, headDim)) return 0;
    } else if (!getenv("H3_BENCH_SKIP_QUANT") &&   /* diagnosis only: time the matmul alone */
               !h3_gpu_quantize_bf16_int8_rows(
                   opaque, quantized_input, input_scales, input, rows,
                   padded_rows, input_dim, 1.0f,
                   @"int8 linear input")) return 0;
    BOOL local_scales = !use_slower_uncached_int8_scales &&
        getenv("H3_DISABLE_INT8_LOCAL_SCALES") == NULL;
    BOOL known_linear = local_scales && rows <= 2048 && input_dim == 7168 &&
        output_dim == 5376 &&
        getenv("H3_DISABLE_INT8_LINEAR_KNOWN") == NULL;
    /* The SwiftVR DiT's shapes get static K loops (and FC2 a full-K
     * product unless H3_VSR_FC2_LOOP is set). */
    NSString *vsr_name = nil;
    BOOL wide = NO;
    if (local_scales && getenv("H3_DISABLE_INT8_LINEAR_KNOWN") == NULL) {
        /* 128x256 tiles only on request (H3_VSR_X256=1): measured slower
         * for FC2 (114 vs 77 ms) and equal elsewhere */
        BOOL x256 = getenv("H3_VSR_X256") != NULL && atoi(getenv("H3_VSR_X256")) != 0;
        if (input_dim == 3072 && output_dim == 3072)
            vsr_name = x256 ? @"h3_linear_int8_nax_r128x256_full_k3072_n3072" : @"h3_linear_int8_local_scales_nax_r128_k3072_n3072";
        else if (input_dim == 3072 && output_dim == 9216)
            vsr_name = x256 ? @"h3_linear_int8_nax_r128x256_full_k3072_n9216" : @"h3_linear_int8_local_scales_nax_r128_k3072_n9216";
        else if (input_dim == 3072 && output_dim == 14336)
            vsr_name = x256 ? @"h3_linear_int8_nax_r128x256_full_k3072_n14336" : @"h3_linear_int8_local_scales_nax_r128_k3072_n14336";
        else if (input_dim == 14336 && output_dim == 3072)
            vsr_name = x256 ? @"h3_linear_int8_nax_r128x256_full_k14336_n3072" :
                       getenv("H3_VSR_FC2_LOOP") ? @"h3_linear_int8_local_scales_nax_r128_k14336_n3072"
                                                 : @"h3_linear_int8_nax_r128_full_k14336_n3072";
        wide = vsr_name != nil && x256;
        if (getenv("H3_VSR_INT8_GENERIC")) { vsr_name = nil; wide = NO; }
    }
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, vsr_name ? vsr_name :
             known_linear ? @"h3_linear_int8_local_scales_nax_r128_k7168" :
             local_scales ? @"h3_linear_int8_local_scales_nax_r128" :
                            @"h3_linear_int8_nax_r128");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < (wide ? 512u : 256u)) {
        h3_gpu_set_error(gpu, @"int8 M5 linear projection is unavailable");
        return 0;
    }
    linear_args args = {rows, input_dim, output_dim, 0};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(quantized_input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(input_scales).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(weight_scales).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
        NSUInteger groups = (NSUInteger)(padded_rows / 128u) *
                            (output_dim / (wide ? 256u : 128u));
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(wide ? 512 : 256, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_linear_int8_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales) {
    return h3_gpu_linear_int8_bf16_layout(
        opaque, output, quantized_input, input_scales, input, weight,
        weight_scales, rows, input_dim, output_dim,
        use_slower_uncached_int8_scales, NO, 0, 0);
}

int h3_gpu_linear_int8_head_major_bf16(
                            h3_gpu *opaque, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim, uint32_t output_dim) {
    return h3_gpu_linear_int8_bf16_layout(
        opaque, output, quantized_input, input_scales, input, weight,
        weight_scales, rows, heads * head_dim, output_dim, 0, YES, heads,
        head_dim);
}

int h3_gpu_mlp_int8_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                         h3_gpu_tensor *activated,
                         h3_gpu_tensor *quantized_activation,
                         h3_gpu_tensor *activation_scales,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *fc1_weight,
                         const h3_gpu_tensor *fc1_scales,
                         const h3_gpu_tensor *fc2_weight,
                         const h3_gpu_tensor *fc2_scales,
                         const h3_gpu_tensor *fc1_bf16,
                         const h3_gpu_tensor *fc2_bf16, uint32_t rows,
                         uint32_t input_dim, uint32_t hidden_dim,
                         uint32_t output_dim,
                         int use_slower_grouped_quantizer,
                         int use_slower_dynamic_fc1_k,
                         int use_int8_row_fc2,
                         int input_is_quantized) {
    H3GPU *gpu = GPU(opaque);
    uint32_t padded_rows = (rows + 127u) & ~127u;
    uint32_t fc2_scale_groups = hidden_dim / 1024u;
    size_t activation_capacity = (size_t)padded_rows *
        MAX(input_dim, hidden_dim);
    const char *stage = getenv("H3_INT8_MLP_STAGE");
    BOOL int8_fc1 = !stage || (strcmp(stage, "fc2") &&
                               strcmp(stage, "bf16"));
    BOOL int8_fc2 = !stage || (strcmp(stage, "fc1") &&
                               strcmp(stage, "bf16"));
    if (!gpu.tensorOpsEnabled || rows < 128 || rows > UINT32_MAX - 127u ||
        (input_dim % 128) || (hidden_dim % 128) || (output_dim % 128) ||
        !h3_gpu_require_i8(gpu, quantized_activation, activation_capacity,
                           @"int8 MLP activation") ||
        !h3_gpu_require_f32(gpu, activation_scales,
                            (size_t)padded_rows * MAX(fc2_scale_groups, 1u),
                            @"int8 MLP activation scales") ||
        !h3_gpu_require_i8(gpu, fc1_weight,
                           (size_t)hidden_dim * 2 * input_dim,
                           @"int8 MLP FC1 weight") ||
        !h3_gpu_require_f32(gpu, fc1_scales, (size_t)hidden_dim * 2,
                            @"int8 MLP FC1 scales") ||
        !h3_gpu_require_i8(gpu, fc2_weight,
                           (size_t)output_dim * hidden_dim,
                           @"int8 MLP FC2 weight") ||
        !h3_gpu_require_f32(gpu, fc2_scales, output_dim,
                            @"int8 MLP FC2 scales") ||
        (!int8_fc1 && !h3_gpu_require_bf16(
            gpu, fc1_bf16, (size_t)hidden_dim * 2 * input_dim,
            @"diagnostic BF16 MLP FC1 weight")) ||
        (!int8_fc2 && !h3_gpu_require_bf16(
            gpu, fc2_bf16, (size_t)output_dim * hidden_dim,
            @"diagnostic BF16 MLP FC2 weight")) ||
        (!input_is_quantized &&
         !h3_gpu_require_bf16(gpu, input, (size_t)rows * input_dim,
                              @"int8 MLP input")) ||
        !h3_gpu_require_bf16(gpu, activated,
                             (size_t)rows * hidden_dim,
                             @"int8 MLP activated") ||
        !h3_gpu_require_bf16(gpu, output,
                             (size_t)rows * output_dim,
                             @"int8 MLP output") ||
        !h3_gpu_require_command(gpu)) return 0;

    id<MTLComputePipelineState> fc1 = h3_gpu_pipeline(
        gpu, @"h3_fc1_swiglu_int8_nax_r128");
    id<MTLComputePipelineState> fc1_known = h3_gpu_pipeline(
        gpu, @"h3_fc1_swiglu_int8_nax_r128_k5376");
    id<MTLComputePipelineState> fc1_full = h3_gpu_pipeline(
        gpu, @"h3_fc1_swiglu_int8_nax_r128_full_k5376");
    id<MTLComputePipelineState> fc1_local = h3_gpu_pipeline(
        gpu, @"h3_fc1_swiglu_int8_local_nax_r128");
    id<MTLComputePipelineState> fc2 = h3_gpu_pipeline(
        gpu, @"h3_linear_int8_nax_r128");
    id<MTLComputePipelineState> fc2_full = h3_gpu_pipeline(
        gpu, @"h3_linear_int8_nax_r128_full_k14336");
    id<MTLComputePipelineState> fc2_full_n256 = h3_gpu_pipeline(
        gpu, @"h3_linear_int8_nax_r128x256_full_k14336");
    id<MTLComputePipelineState> fc2_grouped = h3_gpu_pipeline(
        gpu, @"h3_linear_int8_grouped_nax_r128x64");
    id<MTLComputePipelineState> fc2_grouped_local = h3_gpu_pipeline(
        gpu, @"h3_linear_int8_grouped_local_nax_r128x64");
    id<MTLComputePipelineState> fc2_grouped_local128 = h3_gpu_pipeline(
        gpu, @"h3_linear_int8_grouped_local_nax_r128x128");
    if (!fc1 || !fc1_known || !fc1_full || !fc1_local || !fc2 ||
        !fc2_full || !fc2_full_n256 ||
        fc1.maxTotalThreadsPerThreadgroup < 256 ||
        fc1_known.maxTotalThreadsPerThreadgroup < 256 ||
        fc1_full.maxTotalThreadsPerThreadgroup < 256 ||
        fc1_local.maxTotalThreadsPerThreadgroup < 256 ||
        fc2.maxTotalThreadsPerThreadgroup < 256 ||
        fc2_full.maxTotalThreadsPerThreadgroup < 256 ||
        fc2_full_n256.maxTotalThreadsPerThreadgroup < 512 || !fc2_grouped ||
        fc2_grouped.maxTotalThreadsPerThreadgroup < 128 ||
        !fc2_grouped_local ||
        fc2_grouped_local.maxTotalThreadsPerThreadgroup < 256 ||
        !fc2_grouped_local128 ||
        fc2_grouped_local128.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu, @"device cannot dispatch M5 int8 MLP");
        return 0;
    }
    float activation_clip = 1.0f;
    const char *clip_value = getenv("H3_INT8_ACTIVATION_CLIP");
    if (clip_value) {
        float parsed = strtof(clip_value, NULL);
        if (parsed >= 0.1f && parsed <= 1.0f) activation_clip = parsed;
    }
    BOOL grouped_fc2 = int8_fc2 && !use_int8_row_fc2;
    BOOL row_fc2_n256 = use_int8_row_fc2 && rows <= 2048;
    BOOL grouped_fc2_local = grouped_fc2 &&
        getenv("H3_INT8_GROUP_FC2_THREADGROUP") == NULL;
    BOOL grouped_fc2_local128 = grouped_fc2 &&
        getenv("H3_INT8_GROUP_FC2_LOCAL128") != NULL;
    BOOL int8_fc1_local = int8_fc1 &&
        getenv("H3_INT8_FC1_LOCAL") != NULL;
    BOOL int8_fc1_known = int8_fc1 && input_dim == 5376 &&
        !use_slower_dynamic_fc1_k;
    const char *known_override = getenv("H3_INT8_FC1_KNOWN");
    if (known_override)
        int8_fc1_known = int8_fc1 && input_dim == 5376 &&
            strcmp(known_override, "0") != 0;
    BOOL int8_fc1_full = int8_fc1_known &&
        getenv("H3_DISABLE_FC1_FULL_K") == NULL;
    uint32_t row_tiles = padded_rows / 128;
    if (input_is_quantized && !int8_fc1) {
        h3_gpu_set_error(gpu,
            @"prequantized MLP input requires the int8 FC1 path");
        return 0;
    }
    if (int8_fc1 && !input_is_quantized &&
        !h3_gpu_quantize_bf16_int8_rows(
            opaque, quantized_activation, activation_scales, input, rows,
            padded_rows, input_dim, activation_clip,
            @"int8 MLP input")) return 0;
    if (int8_fc1) @autoreleasepool {
        linear_args fc1_args = {rows, input_dim, hidden_dim, 0};
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:int8_fc1_local ? fc1_local :
            int8_fc1_full ? fc1_full :
            int8_fc1_known ? fc1_known : fc1];
        [encoder setBuffer:TENSOR(quantized_activation).buffer
                         offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(fc1_weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(activation_scales).buffer
                         offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(fc1_scales).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(activated).buffer offset:0 atIndex:4];
        [encoder setBytes:&fc1_args length:sizeof(fc1_args) atIndex:5];
        [encoder dispatchThreadgroups:
            MTLSizeMake((NSUInteger)row_tiles * (hidden_dim / 128), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    } else if (!h3_gpu_fc1_swiglu_nax_bf16(
                   opaque, activated, input, fc1_bf16,
                   rows, input_dim, hidden_dim)) {
        return 0;
    }
    if (int8_fc1) {
        h3_gpu_stats stats = gpu.stats;
        stats.direct_dispatches++;
        gpu.stats = stats;
    }

    if (int8_fc2 && grouped_fc2 &&
        !h3_gpu_quantize_bf16_int8_groups(
            opaque, quantized_activation, activation_scales, activated, rows,
            padded_rows, hidden_dim, 1024,
            use_slower_grouped_quantizer)) return 0;
    if (int8_fc2 && !grouped_fc2 &&
        !h3_gpu_quantize_bf16_int8_rows(
            opaque, quantized_activation, activation_scales, activated, rows,
            padded_rows, hidden_dim, activation_clip,
            @"int8 MLP FC2 input")) return 0;
    if (int8_fc2) @autoreleasepool {
        linear_args fc2_args = {rows, hidden_dim, output_dim, 0};
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:grouped_fc2_local128 ?
            fc2_grouped_local128 : grouped_fc2_local ?
            fc2_grouped_local : grouped_fc2 ? fc2_grouped :
            row_fc2_n256 ? fc2_full_n256 :
            hidden_dim == 14336u && output_dim == 5376u ? fc2_full : fc2];
        [encoder setBuffer:TENSOR(quantized_activation).buffer
                         offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(fc2_weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(activation_scales).buffer
                         offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(fc2_scales).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
        [encoder setBytes:&fc2_args length:sizeof(fc2_args) atIndex:5];
        [encoder dispatchThreadgroups:
            MTLSizeMake((NSUInteger)row_tiles *
                        (output_dim /
                         (row_fc2_n256 ? 256u :
                          grouped_fc2 && !grouped_fc2_local128 ?
                          64u : 128u)), 1, 1)
                 threadsPerThreadgroup:
                    MTLSizeMake(row_fc2_n256 ? 512u :
                        grouped_fc2_local ? 256u :
                        grouped_fc2 ? 128u : 256u, 1, 1)];
        [encoder endEncoding];
    } else if (!h3_gpu_linear_bf16(
                   opaque, output, activated, fc2_bf16, NULL,
                   rows, hidden_dim, output_dim)) {
        return 0;
    }
    if (int8_fc2) {
        h3_gpu_stats stats = gpu.stats;
        stats.direct_dispatches++;
        gpu.stats = stats;
    }
    return 1;
}

int h3_gpu_silu_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, input, elements, @"SiLU input") ||
        !h3_gpu_require_bf16(gpu, output, elements, @"SiLU output")) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_silu_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        });
}

int h3_gpu_rms_norm_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, input, count, @"RMSNorm input") ||
        !h3_gpu_require_bf16(gpu, weight, width, @"RMSNorm weight") ||
        !h3_gpu_require_bf16(gpu, output, count, @"RMSNorm output")) return 0;
    norm_args args = {rows, width, epsilon};
    return h3_gpu_dispatch_rows(gpu, @"h3_rms_norm_bf16", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_layer_norm_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, input, count, @"LayerNorm input") ||
        !h3_gpu_require_bf16(gpu, weight, width, @"LayerNorm weight") ||
        !h3_gpu_require_bf16(gpu, bias, width, @"LayerNorm bias") ||
        !h3_gpu_require_bf16(gpu, output, count, @"LayerNorm output")) return 0;
    norm_args args = {rows, width, epsilon};
    return h3_gpu_dispatch_rows(gpu, @"h3_layer_norm_bf16", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

int h3_gpu_gelu_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, input, elements, @"GELU input") ||
        !h3_gpu_require_bf16(gpu, output, elements, @"GELU output")) return 0;
    gelu_bf16_args args = {elements, approximate ? 1u : 0u};
    return h3_gpu_dispatch_1d(gpu, @"h3_gelu_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_vision_qkv_rope_bf16(
                     h3_gpu *opaque, h3_gpu_tensor *query,
                     h3_gpu_tensor *key, h3_gpu_tensor *value,
                     const h3_gpu_tensor *qkv,
                     const h3_gpu_tensor *rope_cos,
                     const h3_gpu_tensor *rope_sin, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim,
                     uint32_t rope_half) {
    H3GPU *gpu = GPU(opaque);
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_gpu_require_bf16(gpu, qkv, count * 3, @"vision QKV") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count,
                              @"vision RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count,
                              @"vision RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query, count, @"vision query") ||
        !h3_gpu_require_bf16(gpu, key, count, @"vision key") ||
        !h3_gpu_require_bf16(gpu, value, count, @"vision value") ||
        rope_half * 2 != head_dim) return 0;
    qkv_args args = {sequence, heads, head_dim, rope_half, 0, 0.0f};
    return h3_gpu_dispatch_3d(gpu, @"h3_vision_qkv_rope_bf16",
        MTLSizeMake(head_dim, heads, sequence),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
        });
}

int h3_gpu_adaln_bf16_offset(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (input_offset > SIZE_MAX - count ||
        input_offset > SIZE_MAX / sizeof(uint16_t)) {
        h3_gpu_set_error(gpu, @"AdaLN input offset is out of range");
        return 0;
    }
    if (!h3_gpu_require_bf16(gpu, input, input_offset + count,
                             @"AdaLN input") ||
        !h3_gpu_require_bf16(gpu, norm_weight, width, @"AdaLN norm") ||
        !h3_gpu_require_bf16(gpu, modulation, 1, @"AdaLN modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows, @"AdaLN row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, count, @"AdaLN output") ||
        shift_slot >= slots || scale_slot >= slots) return 0;
    adaln_args args = {rows, width, slots, shift_slot, scale_slot, epsilon};
    return h3_gpu_dispatch_rows(gpu, @"h3_adaln_bf16", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer
                         offset:input_offset * sizeof(uint16_t) atIndex:0];
            [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
            [encoder setBytes:&args length:sizeof(args) atIndex:5];
        });
}

int h3_gpu_adaln_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    return h3_gpu_adaln_bf16_offset(
        opaque, output, input, 0, norm_weight, modulation, row_map, rows,
        width, slots, shift_slot, scale_slot, epsilon);
}

int h3_gpu_adaln_linear_bf16(
                      h3_gpu *opaque, h3_gpu_tensor *output,
                      h3_gpu_tensor *inverse,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t width, uint32_t output_dim, uint32_t slots,
                      uint32_t shift_slot, uint32_t scale_slot,
                      float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t input_count = (size_t)rows * width;
    size_t weight_count = (size_t)output_dim * width;
    size_t output_count = (size_t)rows * output_dim;
    if (input_offset > SIZE_MAX - input_count ||
        input_offset > SIZE_MAX / sizeof(uint16_t)) {
        h3_gpu_set_error(gpu, @"fused final head input offset is out of range");
        return 0;
    }
    if (!h3_gpu_require_bf16(gpu, input, input_offset + input_count,
                             @"fused final head input") ||
        !h3_gpu_require_elements(gpu, inverse, rows,
                                 @"fused final head inverse RMS") ||
        TENSOR(inverse).dtype != H3_GPU_F32 ||
        !h3_gpu_require_bf16(gpu, norm_weight, width,
                             @"fused final head norm") ||
        !h3_gpu_require_bf16(gpu, modulation, 1,
                             @"fused final head modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows,
                                 @"fused final head row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, weight, weight_count,
                             @"fused final head weight") ||
        !h3_gpu_require_bf16(gpu, output, output_count,
                             @"fused final head output") ||
        (bias && !h3_gpu_require_bf16(gpu, bias, output_dim,
                                      @"fused final head bias")) ||
        shift_slot >= slots || scale_slot >= slots) return 0;
    norm_args rms_args = {rows, width, epsilon};
    if (!h3_gpu_dispatch_rows(
            gpu, @"h3_rms_inverse_bf16", rows,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:TENSOR(input).buffer
                             offset:input_offset * sizeof(uint16_t) atIndex:0];
                [encoder setBuffer:TENSOR(inverse).buffer offset:0 atIndex:1];
                [encoder setBytes:&rms_args length:sizeof(rms_args) atIndex:2];
            })) return 0;
    typedef struct {
        uint32_t rows, width, output_dim, slots, shift_slot, scale_slot;
        uint32_t has_bias;
    } adaln_linear_args;
    adaln_linear_args args = {
        rows, width, output_dim, slots, shift_slot, scale_slot,
        bias ? 1u : 0u
    };
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, @"h3_adaln_linear_bf16");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu,
                         @"device cannot dispatch fused final BF16 head");
        return 0;
    }
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer
                     offset:input_offset * sizeof(uint16_t) atIndex:0];
        [encoder setBuffer:TENSOR(inverse).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:6];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:7];
        [encoder setBytes:&args length:sizeof(args) atIndex:8];
        [encoder dispatchThreadgroups:
            MTLSizeMake((output_dim + 15) / 16, (rows + 15) / 16, 1)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_gate_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, residual, count, @"gate residual") ||
        !h3_gpu_require_bf16(gpu, branch, count, @"gate branch") ||
        !h3_gpu_require_bf16(gpu, modulation, 1, @"gate modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows, @"gate row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, count, @"gate output") ||
        gate_slot >= slots) return 0;
    gate_args args = {rows, width, slots, gate_slot};
    return h3_gpu_dispatch_2d(gpu, @"h3_gate_bf16", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(branch).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
            [encoder setBytes:&args length:sizeof(args) atIndex:5];
        });
}

int h3_gpu_gate_adaln_bf16(
                     h3_gpu *opaque, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot,
                     uint32_t shift_slot, uint32_t scale_slot,
                     float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t elements = (size_t)rows * width;
    if (!rows || !width || width > 5376 || gate_slot >= slots ||
        shift_slot >= slots || scale_slot >= slots ||
        elements > UINT32_MAX ||
        !h3_gpu_require_bf16(gpu, residual, elements,
                             @"fused gate AdaLN residual") ||
        !h3_gpu_require_bf16(gpu, branch, elements,
                             @"fused gate AdaLN branch") ||
        !h3_gpu_require_bf16(gpu, norm_weight, width,
                             @"fused gate AdaLN norm") ||
        !h3_gpu_require_bf16(gpu, gate_modulation, 1,
                             @"fused gate AdaLN gate modulation") ||
        !h3_gpu_require_bf16(gpu, norm_modulation, 1,
                             @"fused gate AdaLN norm modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows,
                                 @"fused gate AdaLN row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, gated_residual, elements,
                             @"fused gate AdaLN gated residual") ||
        !h3_gpu_require_bf16(gpu, output, elements,
                             @"fused gate AdaLN output") ||
        !h3_gpu_require_command(gpu)) return 0;
    typedef struct {
        uint32_t rows, width, slots, gate_slot, shift_slot, scale_slot;
        float epsilon;
    } gate_adaln_args;
    gate_adaln_args args = {
        rows, width, slots, gate_slot, shift_slot, scale_slot, epsilon
    };
    NSString *pipeline_name = getenv("H3_DISABLE_EXACT_SIMD_GATE_ADALN") ?
        @"h3_gate_adaln_bf16" : @"h3_gate_adaln_bf16_exact_simd";
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, pipeline_name);
    if (!pipeline) return 0;
    const NSUInteger threads = 256;
    if (pipeline.maxTotalThreadsPerThreadgroup < threads) {
        h3_gpu_set_error(gpu,
            @"fused gate AdaLN needs a 256-thread threadgroup");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(branch).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(gate_modulation).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(norm_modulation).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(gated_residual).buffer offset:0 atIndex:6];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:7];
        [encoder setBytes:&args length:sizeof(args) atIndex:8];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_gate_adaln_quantize_int8(
                     h3_gpu *opaque, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *quantized_output,
                     h3_gpu_tensor *quantized_scales,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t padded_rows, uint32_t width, uint32_t slots,
                     uint32_t gate_slot, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t elements = (size_t)rows * width;
    size_t padded_elements = (size_t)padded_rows * width;
    if (!gpu.tensorOpsEnabled || !rows || padded_rows < rows || !width ||
        width > 5376 || gate_slot >= slots || shift_slot >= slots ||
        scale_slot >= slots || elements > UINT32_MAX ||
        !h3_gpu_require_bf16(gpu, residual, elements,
                             @"fused int8 gate AdaLN residual") ||
        !h3_gpu_require_bf16(gpu, branch, elements,
                             @"fused int8 gate AdaLN branch") ||
        !h3_gpu_require_bf16(gpu, norm_weight, width,
                             @"fused int8 gate AdaLN norm") ||
        !h3_gpu_require_bf16(gpu, gate_modulation, 1,
                             @"fused int8 gate modulation") ||
        !h3_gpu_require_bf16(gpu, norm_modulation, 1,
                             @"fused int8 norm modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows,
                                 @"fused int8 gate AdaLN row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, gated_residual, elements,
                             @"fused int8 gated residual") ||
        !h3_gpu_require_i8(gpu, quantized_output, padded_elements,
                           @"fused int8 AdaLN output") ||
        !h3_gpu_require_f32(gpu, quantized_scales, padded_rows,
                            @"fused int8 AdaLN scales") ||
        !h3_gpu_require_command(gpu)) return 0;
    typedef struct {
        uint32_t rows, width, slots, gate_slot, shift_slot, scale_slot;
        float epsilon;
    } gate_adaln_args;
    gate_adaln_args args = {
        rows, width, slots, gate_slot, shift_slot, scale_slot, epsilon
    };
    NSString *pipeline_name = width == 5376 &&
        getenv("H3_DISABLE_VECTOR_GATE_ADALN") == NULL ?
        @"h3_gate_adaln_quantize_int8" :
        @"h3_gate_adaln_quantize_int8_scalar";
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, pipeline_name);
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu,
            @"fused M5 int8 gate AdaLN needs a 256-thread threadgroup");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(branch).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(gate_modulation).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(norm_modulation).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(gated_residual).buffer offset:0 atIndex:6];
        [encoder setBuffer:TENSOR(quantized_output).buffer offset:0 atIndex:7];
        [encoder setBuffer:TENSOR(quantized_scales).buffer offset:0 atIndex:8];
        [encoder setBytes:&args length:sizeof(args) atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake(padded_rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_qkv_rope_bf16_layout(h3_gpu *opaque, h3_gpu_tensor *query,
                                       h3_gpu_tensor *key,
                                       h3_gpu_tensor *value,
                                       const h3_gpu_tensor *qkv,
                                       const h3_gpu_tensor *q_norm,
                                       const h3_gpu_tensor *k_norm,
                                       const h3_gpu_tensor *rope_cos,
                                       const h3_gpu_tensor *rope_sin,
                                       uint32_t sequence, uint32_t heads,
                                       uint32_t head_dim, uint32_t rope_half,
                                       uint32_t grouped, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_gpu_require_bf16(gpu, qkv, count * 3, @"QKV input") ||
        !h3_gpu_require_bf16(gpu, q_norm, head_dim, @"Q norm") ||
        !h3_gpu_require_bf16(gpu, k_norm, head_dim, @"K norm") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count, @"RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count, @"RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query, count, @"query") ||
        !h3_gpu_require_bf16(gpu, key, count, @"key") ||
        !h3_gpu_require_bf16(gpu, value, count, @"value") ||
        rope_half * 2 > head_dim) return 0;
    qkv_args args = {sequence, heads, head_dim, rope_half, grouped, epsilon};
    if (head_dim == 128 && !(heads % 4) &&
        !getenv("H3_DISABLE_COOP_QKV")) {
        if (!h3_gpu_require_command(gpu)) return 0;
        NSString *pipeline_name = getenv("H3_DISABLE_CACHED_QKV") ?
            @"h3_qkv_rope_bf16_coop_uncached" :
            @"h3_qkv_rope_bf16_coop";
        id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
            gpu, pipeline_name);
        if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < head_dim) {
            h3_gpu_set_error(gpu,
                @"cooperative QKV/RoPE needs a 128-thread threadgroup");
            return 0;
        }
        @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder =
                [gpu.command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:5];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:6];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:7];
            [encoder setBytes:&args length:sizeof(args) atIndex:8];
            [encoder dispatchThreadgroups:MTLSizeMake(heads / 4, sequence, 1)
                     threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
            [encoder endEncoding];
        }
        h3_gpu_stats stats = gpu.stats;
        stats.direct_dispatches++;
        gpu.stats = stats;
        return 1;
    }
    return h3_gpu_dispatch_3d(gpu, @"h3_qkv_rope_bf16",
        MTLSizeMake(head_dim, heads, sequence),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:5];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:6];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:7];
            [encoder setBytes:&args length:sizeof(args) atIndex:8];
        });
}

int h3_gpu_qkv_rope_bf16(h3_gpu *opaque, h3_gpu_tensor *query,
                         h3_gpu_tensor *key, h3_gpu_tensor *value,
                         const h3_gpu_tensor *qkv,
                         const h3_gpu_tensor *q_norm,
                         const h3_gpu_tensor *k_norm,
                         const h3_gpu_tensor *rope_cos,
                         const h3_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim,
                         uint32_t rope_half, float epsilon) {
    return h3_gpu_qkv_rope_bf16_layout(
        opaque, query, key, value, qkv, q_norm, k_norm, rope_cos, rope_sin,
        sequence, heads, head_dim, rope_half, 0, epsilon);
}

int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *opaque, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key, h3_gpu_tensor *value,
                                 const h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon) {
    return h3_gpu_qkv_rope_bf16_layout(
        opaque, query, key, value, qkv, q_norm, k_norm, rope_cos, rope_sin,
        sequence, heads, head_dim, rope_half, 1, epsilon);
}

int h3_gpu_grouped_qkv_linear_rope_bf16(
                                 h3_gpu *opaque,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    gpu.headMajorSDPAInputs = NO;
    uint32_t inner = heads * head_dim;
    BOOL candidate = gpu.tensorOpsEnabled && rows >= 64 && rows <= 2048 &&
        input_dim == 5376 && heads == 56 && head_dim == 128 &&
        rope_half == 48 &&
        (gpu.tensorOpsMode == 1 || gpu.tensorOpsMode == 3 ||
         gpu.tensorOpsMode == 4) &&
        !getenv("H3_DISABLE_NAX_LINEAR") &&
        !getenv("H3_DISABLE_SPLIT_NAX_QKV");
    if (!candidate) {
        return h3_gpu_linear_bf16(opaque, qkv, input, weight, NULL, rows,
                                  input_dim, inner * 3) &&
            h3_gpu_grouped_qkv_rope_bf16(
                opaque, query, key, value, qkv, q_norm, k_norm, rope_cos,
                rope_sin, rows, heads, head_dim, rope_half, epsilon);
    }
    size_t projected = (size_t)rows * inner;
    size_t rope_count = (size_t)rows * rope_half;
    if (!h3_gpu_require_bf16(gpu, input, (size_t)rows * input_dim,
                             @"split QKV projection input") ||
        !h3_gpu_require_bf16(gpu, weight,
                             (size_t)inner * 3 * input_dim,
                             @"split QKV projection weight") ||
        !h3_gpu_require_bf16(gpu, q_norm, head_dim,
                             @"split Q norm") ||
        !h3_gpu_require_bf16(gpu, k_norm, head_dim,
                             @"split K norm") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count,
                             @"split RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count,
                             @"split RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query, projected, @"split query") ||
        !h3_gpu_require_bf16(gpu, key, projected, @"split key") ||
        !h3_gpu_require_bf16(gpu, value, projected, @"split value") ||
        !h3_gpu_require_command(gpu)) return 0;
    uint32_t headMajor = getenv("H3_DISABLE_HEAD_MAJOR_SDPA") == NULL;
    typedef struct {
        uint32_t rows, input_dim, heads, head_dim, rope_half, head_major;
        float epsilon;
    } qkv_project_rope_args;
    qkv_project_rope_args args = {
        rows, input_dim, heads, head_dim, rope_half, headMajor, epsilon
    };
    id<MTLComputePipelineState> projection = h3_gpu_pipeline(
        gpu, @"h3_qkv_project_split_bf16_nax_r128_morton4");
    id<MTLComputePipelineState> rope = h3_gpu_pipeline(
        gpu, @"h3_qk_rope_bf16_nax_inplace");
    const NSUInteger threads = 128;
    if (!projection || !rope ||
        projection.maxTotalThreadsPerThreadgroup < threads ||
        rope.maxTotalThreadsPerThreadgroup < threads) {
        h3_gpu_set_error(gpu,
            @"split M5 QKV projection/RoPE needs 128 threads");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:projection];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
        NSUInteger groups = (NSUInteger)((rows + 127) / 128) * heads * 6;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:rope];
        [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:5];
        [encoder setBytes:&args length:sizeof(args) atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake(heads / 4, rows, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches += 2;
    gpu.stats = stats;
    gpu.headMajorSDPAInputs = headMajor != 0;
    return 1;
}

int h3_gpu_grouped_qkv_linear_rope_int8(
                                 h3_gpu *opaque,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *quantized_input,
                                 h3_gpu_tensor *input_scales,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *weight_scales,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon,
                                 int input_is_quantized,
                                 int use_slower_unfused_qkv_rope,
                                 int use_slower_scalar_qkv_rms,
                                 int use_slower_uncached_int8_scales) {
    H3GPU *gpu = GPU(opaque);
    gpu.headMajorSDPAInputs = NO;
    uint32_t inner = heads * head_dim;
    uint32_t padded_rows = (rows + 127u) & ~127u;
    size_t projected = (size_t)rows * inner;
    size_t rope_count = (size_t)rows * rope_half;
    if (!gpu.tensorOpsEnabled || rows < 128 || input_dim % 128 ||
        head_dim != 128 || heads % 4 || rope_half != 48 ||
        !h3_gpu_require_i8(gpu, weight,
                           (size_t)inner * 3 * input_dim,
                           @"int8 QKV projection weight") ||
        !h3_gpu_require_f32(gpu, weight_scales, inner * 3,
                            @"int8 QKV weight scales") ||
        !h3_gpu_require_i8(gpu, quantized_input,
                           (size_t)padded_rows * input_dim,
                           @"int8 QKV input") ||
        !h3_gpu_require_f32(gpu, input_scales, padded_rows,
                            @"int8 QKV input scales") ||
        (!input_is_quantized &&
         !h3_gpu_require_bf16(gpu, input, (size_t)rows * input_dim,
                              @"BF16 QKV input")) ||
        !h3_gpu_require_bf16(gpu, q_norm, head_dim, @"int8 Q norm") ||
        !h3_gpu_require_bf16(gpu, k_norm, head_dim, @"int8 K norm") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count,
                             @"int8 RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count,
                             @"int8 RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query, projected, @"int8 query") ||
        !h3_gpu_require_bf16(gpu, key, projected, @"int8 key") ||
        !h3_gpu_require_bf16(gpu, value, projected, @"int8 value") ||
        !h3_gpu_require_command(gpu)) return 0;
    if (!input_is_quantized && !h3_gpu_quantize_bf16_int8_rows(
            opaque, quantized_input, input_scales, input, rows, padded_rows,
            input_dim, 1.0f, @"int8 QKV input")) return 0;
    BOOL fused_rope = !use_slower_unfused_qkv_rope &&
        getenv("H3_DISABLE_FUSED_INT8_QKV_ROPE") == NULL;
    BOOL local_scales = fused_rope && rows > 2048 &&
        !use_slower_uncached_int8_scales &&
        getenv("H3_DISABLE_QKV_LOCAL_SCALES") == NULL;
    BOOL full_k = fused_rope && input_dim == 5376u &&
        getenv("H3_DISABLE_QKV_FULL_K") == NULL;
    id<MTLComputePipelineState> projection = h3_gpu_pipeline(
        gpu, local_scales && full_k ?
            @"h3_qkv_project_split_int8_rope_local_scales_nax_r128_k5376_morton4" :
            local_scales ?
            @"h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4" :
            full_k ?
            @"h3_qkv_project_split_int8_rope_nax_r128_k5376_morton4" :
            fused_rope ?
            @"h3_qkv_project_split_int8_rope_nax_r128_morton4" :
            @"h3_qkv_project_split_int8_nax_r128_morton4");
    id<MTLComputePipelineState> rope = fused_rope ? nil : h3_gpu_pipeline(
        gpu, @"h3_qk_rope_bf16_nax_inplace");
    if (!projection || projection.maxTotalThreadsPerThreadgroup < 256 ||
        (!fused_rope &&
         (!rope || rope.maxTotalThreadsPerThreadgroup < 128))) {
        h3_gpu_set_error(gpu, @"int8 M5 QKV projection is unavailable");
        return 0;
    }
    typedef struct {
        uint32_t rows, input_dim, heads, head_dim, rope_half, head_major;
        float epsilon;
    } qkv_project_rope_args;
    /* The historical layout field now carries fused-epilogue mode bits:
     * bit 1 selects ordered BF16x4 RMS loads, bit 2 packs norm/RoPE by four. */
    uint32_t rms_mode = 1u;
    if (fused_rope && rows <= 2048 && !use_slower_scalar_qkv_rms &&
        getenv("H3_DISABLE_VECTOR_QKV_RMS") == NULL)
        rms_mode |= 2u;
    if (fused_rope && getenv("H3_DISABLE_VECTOR_QKV_ROPE") == NULL)
        rms_mode |= 4u;
    qkv_project_rope_args args = {
        rows, input_dim, heads, head_dim, rope_half, rms_mode, epsilon
    };
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:projection];
        [encoder setBuffer:TENSOR(quantized_input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(input_scales).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(weight_scales).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:6];
        if (fused_rope) {
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:7];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:8];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:9];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:10];
            [encoder setBytes:&args length:sizeof(args) atIndex:11];
        } else {
            [encoder setBytes:&args length:sizeof(args) atIndex:7];
        }
        NSUInteger groups = (NSUInteger)(padded_rows / 128u) * heads * 3u;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        if (!fused_rope) {
            encoder = [gpu.command computeCommandEncoder];
            [encoder setComputePipelineState:rope];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake(heads / 4, rows, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
        }
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches += fused_rope ? 1 : 2;
    gpu.stats = stats;
    gpu.headMajorSDPAInputs = YES;
    return 1;
}

int h3_gpu_swiglu_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, fused, (size_t)rows * width * 2,
                              @"SwiGLU input") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * width,
                              @"SwiGLU output")) return 0;
    swiglu_args args = {rows, width};
    return h3_gpu_dispatch_2d(gpu, @"h3_swiglu_bf16", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(fused).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_embedding_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, weight, (size_t)vocab_size * width,
                              @"embedding weight") ||
        !h3_gpu_require_elements(gpu, token_ids, tokens, @"token IDs") ||
        TENSOR(token_ids).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, (size_t)tokens * width,
                              @"embedding output")) return 0;
    embedding_args args = {tokens, vocab_size, width};
    return h3_gpu_dispatch_2d(gpu, @"h3_embedding_bf16", width, tokens,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(token_ids).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_text_qk_rope_bf16(h3_gpu *opaque,
                             h3_gpu_tensor *query_output,
                             h3_gpu_tensor *key_output,
                             const h3_gpu_tensor *query_input,
                             const h3_gpu_tensor *key_input,
                             const h3_gpu_tensor *q_norm,
                             const h3_gpu_tensor *k_norm,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin,
                             uint32_t sequence, uint32_t query_heads,
                             uint32_t kv_heads, uint32_t head_dim,
                             float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t key_count = (size_t)sequence * kv_heads * head_dim;
    size_t rope_count = (size_t)sequence * (head_dim / 2);
    if (head_dim % 2 || !kv_heads || query_heads % kv_heads ||
        !h3_gpu_require_bf16(gpu, query_input, query_count, @"text query") ||
        !h3_gpu_require_bf16(gpu, key_input, key_count, @"text key") ||
        !h3_gpu_require_bf16(gpu, q_norm, head_dim, @"text Q norm") ||
        !h3_gpu_require_bf16(gpu, k_norm, head_dim, @"text K norm") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count, @"text RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count, @"text RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query_output, query_count, @"text query output") ||
        !h3_gpu_require_bf16(gpu, key_output, key_count, @"text key output")) return 0;
    text_rope_args args = {sequence, query_heads, kv_heads, head_dim, epsilon};
    return h3_gpu_dispatch_3d(gpu, @"h3_text_qk_rope_bf16",
        MTLSizeMake(head_dim, query_heads, sequence),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(query_input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(key_input).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:5];
            [encoder setBuffer:TENSOR(query_output).buffer offset:0 atIndex:6];
            [encoder setBuffer:TENSOR(key_output).buffer offset:0 atIndex:7];
            [encoder setBytes:&args length:sizeof(args) atIndex:8];
        });
}

int h3_gpu_head_rms_norm_bf16(h3_gpu *opaque, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)sequence * heads * head_dim;
    if (!h3_gpu_require_bf16(gpu, tensor, count, @"head norm tensor") ||
        !h3_gpu_require_bf16(gpu, weight, head_dim, @"head norm weight")) return 0;
    head_norm_args args = {sequence, heads, head_dim, epsilon};
    return h3_gpu_dispatch_2d(gpu, @"h3_head_rms_norm_bf16", sequence, heads,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(tensor).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_rope_text_bf16(h3_gpu *opaque, h3_gpu_tensor *query,
                          h3_gpu_tensor *key,
                          const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32,
                          uint32_t sequence, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t key_count = (size_t)sequence * kv_heads * head_dim;
    size_t rope_count = (size_t)sequence * (head_dim / 2);
    if (head_dim % 2 || !kv_heads || query_heads % kv_heads ||
        !h3_gpu_require_bf16(gpu, query, query_count, @"RoPE query") ||
        !h3_gpu_require_bf16(gpu, key, key_count, @"RoPE key") ||
        !h3_gpu_require_elements(gpu, rope_cos_f32, rope_count, @"RoPE cosine") ||
        TENSOR(rope_cos_f32).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, rope_sin_f32, rope_count, @"RoPE sine") ||
        TENSOR(rope_sin_f32).dtype != H3_GPU_F32) return 0;
    text_rope_inplace_args args = {sequence, query_heads, kv_heads, head_dim};
    uint32_t maximum_heads = query_heads > kv_heads ? query_heads : kv_heads;
    return h3_gpu_dispatch_2d(gpu, @"h3_rope_text_bf16", sequence, maximum_heads,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(rope_cos_f32).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(rope_sin_f32).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

static H3GQA *h3_gpu_gqa_graph(H3GPU *gpu, uint32_t sequence,
                               uint32_t query_heads, uint32_t kv_heads,
                               uint32_t head_dim, float scale) {
    @autoreleasepool {
        NSString *key = [NSString stringWithFormat:@"%u:%u:%u:%u:%.9g",
                         sequence, query_heads, kv_heads, head_dim, scale];
        H3GQA *cached = gpu.gqaCache[key];
        if (cached) return cached;
        if (!kv_heads || query_heads % kv_heads) return nil;
        MPSGraph *graph = [[MPSGraph alloc] init];
        MPSShape *query_shape = @[@1, @(sequence), @(query_heads), @(head_dim)];
        MPSShape *kv_shape = @[@1, @(sequence), @(kv_heads), @(head_dim)];
        MPSGraphTensor *query = [graph placeholderWithShape:query_shape
                                                   dataType:MPSDataTypeBFloat16
                                                       name:nil];
        MPSGraphTensor *key_tensor = [graph placeholderWithShape:kv_shape
                                                         dataType:MPSDataTypeBFloat16
                                                             name:nil];
        MPSGraphTensor *value = [graph placeholderWithShape:kv_shape
                                                   dataType:MPSDataTypeBFloat16
                                                       name:nil];
        MPSGraphTensor *qt = [graph transposeTensor:query dimension:1
                                      withDimension:2 name:nil];
        MPSGraphTensor *kt = [graph transposeTensor:key_tensor dimension:1
                                      withDimension:2 name:nil];
        MPSGraphTensor *vt = [graph transposeTensor:value dimension:1
                                      withDimension:2 name:nil];
        uint32_t groups = query_heads / kv_heads;
        MPSShape *split_shape = @[@1, @(kv_heads), @1, @(sequence), @(head_dim)];
        MPSShape *broadcast_shape = @[@1, @(kv_heads), @(groups),
                                      @(sequence), @(head_dim)];
        MPSShape *expanded_shape = @[@1, @(query_heads), @(sequence),
                                     @(head_dim)];
        kt = [graph reshapeTensor:kt withShape:split_shape name:nil];
        vt = [graph reshapeTensor:vt withShape:split_shape name:nil];
        kt = [graph broadcastTensor:kt toShape:broadcast_shape name:nil];
        vt = [graph broadcastTensor:vt toShape:broadcast_shape name:nil];
        kt = [graph reshapeTensor:kt withShape:expanded_shape name:nil];
        vt = [graph reshapeTensor:vt withShape:expanded_shape name:nil];
        size_t mask_count = (size_t)sequence * sequence;
        uint16_t *mask_values = malloc(mask_count * sizeof(*mask_values));
        if (!mask_values) return nil;
        for (uint32_t row = 0; row < sequence; row++)
            for (uint32_t column = 0; column < sequence; column++)
                mask_values[(size_t)row * sequence + column] =
                    column <= row ? 0u : UINT16_C(0xff80);
        NSData *mask_data = [NSData dataWithBytesNoCopy:mask_values
                                                 length:mask_count * sizeof(*mask_values)
                                           freeWhenDone:YES];
        MPSGraphTensor *mask = [graph constantWithData:mask_data
            shape:@[@1, @1, @(sequence), @(sequence)]
            dataType:MPSDataTypeBFloat16];
        MPSGraphTensor *attention = [graph
            scaledDotProductAttentionWithQueryTensor:qt keyTensor:kt
            valueTensor:vt maskTensor:mask scale:scale name:nil];
        H3GQA *result = [[H3GQA alloc] init];
        result.graph = graph;
        result.query = query;
        result.key = key_tensor;
        result.value = value;
        result.output = [graph transposeTensor:attention dimension:1
                                 withDimension:2 name:nil];
        result.queryShape = query_shape;
        result.kvShape = kv_shape;
        gpu.gqaCache[key] = result;
        return result;
    }
}

static int h3_gpu_gqa_mps(H3GPU *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *query,
                          const h3_gpu_tensor *key,
                          const h3_gpu_tensor *value,
                          uint32_t sequence, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim,
                          float scale) {
    if (!h3_gpu_require_command(gpu)) return 0;
    H3GQA *cache = h3_gpu_gqa_graph(gpu, sequence, query_heads, kv_heads,
                                    head_dim, scale);
    if (!cache) {
        h3_gpu_set_error(gpu, @"cannot build MPSGraph causal GQA");
        return 0;
    }
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        MPSGraphTensorData *query_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(query).buffer shape:cache.queryShape
            dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData *(^kv_data)(const h3_gpu_tensor *) =
            ^MPSGraphTensorData *(const h3_gpu_tensor *tensor) {
                return [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:TENSOR(tensor).buffer shape:cache.kvShape
                    dataType:MPSDataTypeBFloat16];
            };
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(output).buffer shape:cache.queryShape
            dataType:MPSDataTypeBFloat16];
        NSDictionary *feeds = @{cache.query: query_data,
                                cache.key: kv_data(key),
                                cache.value: kv_data(value)};
        NSDictionary *results = @{cache.output: output_data};
        @try {
            [cache.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil resultsDictionary:results
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph causal GQA failed: %@",
                             exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_sdpa_dispatches++;
    gpu.stats = stats;
    return 1;
}

/* --- Causal GQA in bounded pieces -------------------------------------
 *
 * Past the custom kernel's threadgroup ceiling, h3_gpu_gqa_mps() hands the
 * whole causal attention to MPSGraph's SDPA in one call: [1, 64, S, S]
 * scores for Qwen's 64 query heads, with S in the thousands once reference
 * videos add their vision tokens. That is the op issue #2 aborted in on an
 * M2 Max / macOS 15 (there in the DiT), so outside the validated FULL mode
 * this path follows the same policy as h3_gpu_sdpa_chunked(): pieces of one
 * KV head's group of query heads and a block of query rows, each under the
 * score budget. A piece over rows [r0, r0 + rn) reads keys [0, r0 + rn)
 * only and builds its causal mask in the graph from coordinates, so no
 * S x S constant is held. Q is viewed as [S, kv_heads, groups * D] and
 * transposed once to [kv_heads, S, groups * D], K and V to
 * [kv_heads, S, D], which makes every piece a contiguous range. */
static H3SDPA *h3_gpu_gqa_chunk_graph(H3GPU *gpu, uint32_t groups,
                                      uint32_t rows, uint32_t first_row,
                                      uint32_t head_dim, float scale,
                                      int explicit) {
    @autoreleasepool {
        uint32_t kv_rows = first_row + rows;
        NSString *cacheKey = [NSString stringWithFormat:
                              @"G:%u:%u:%u:%u:%.9g:%d", groups, rows,
                              first_row, head_dim, scale, explicit];
        H3SDPA *cached = gpu.sdpaCache[cacheKey];
        if (cached) return cached;
        if (!explicit &&
            ![[MPSGraph class] instancesRespondToSelector:@selector(scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:maskTensor:scale:name:)]) {
            h3_gpu_set_error(gpu, @"native MPSGraph SDPA is unavailable");
            return nil;
        }
        MPSGraph *graph = [[MPSGraph alloc] init];
        MPSDataType bf16 = MPSDataTypeBFloat16;
        NSArray<NSNumber *> *qShape = @[@1, @(rows), @(groups), @(head_dim)];
        NSArray<NSNumber *> *kvShape = @[@1, @1, @(kv_rows), @(head_dim)];
        MPSGraphTensor *q = [graph placeholderWithShape:qShape dataType:bf16
                                                   name:nil];
        MPSGraphTensor *k = [graph placeholderWithShape:kvShape dataType:bf16
                                                   name:nil];
        MPSGraphTensor *v = [graph placeholderWithShape:kvShape dataType:bf16
                                                   name:nil];
        MPSGraphTensor *qt = [graph transposeTensor:q dimension:1
                                      withDimension:2 name:nil];
        MPSShape *expanded = @[@1, @(groups), @(kv_rows), @(head_dim)];
        MPSGraphTensor *kb = [graph broadcastTensor:k toShape:expanded name:nil];
        MPSGraphTensor *vb = [graph broadcastTensor:v toShape:expanded name:nil];
        /* Key j is in the future of query row r0 + i when j > r0 + i. */
        MPSShape *maskShape = @[@1, @1, @(rows), @(kv_rows)];
        MPSGraphTensor *row = [graph coordinateAlongAxis:2 withShape:maskShape
                                                    name:nil];
        MPSGraphTensor *column = [graph coordinateAlongAxis:3
                                                  withShape:maskShape name:nil];
        row = [graph additionWithPrimaryTensor:row
            secondaryTensor:[graph constantWithScalar:(double)first_row
                                             dataType:MPSDataTypeInt32]
            name:nil];
        MPSGraphTensor *future = [graph greaterThanWithPrimaryTensor:column
                                                     secondaryTensor:row
                                                                name:nil];
        MPSDataType maskType = explicit ? MPSDataTypeFloat32 : bf16;
        MPSGraphTensor *mask = [graph selectWithPredicateTensor:future
            truePredicateTensor:[graph constantWithScalar:-INFINITY
                                                 dataType:maskType]
            falsePredicateTensor:[graph constantWithScalar:0.0
                                                  dataType:maskType]
            name:nil];
        MPSGraphTensor *attention;
        if (explicit) {
            MPSGraphTensor *kT = [graph transposeTensor:kb dimension:2
                                          withDimension:3 name:nil];
            MPSGraphTensor *scores = [graph
                matrixMultiplicationWithPrimaryTensor:qt secondaryTensor:kT
                name:nil];
            scores = [graph castTensor:scores toType:MPSDataTypeFloat32
                                  name:nil];
            scores = [graph multiplicationWithPrimaryTensor:scores
                secondaryTensor:[graph constantWithScalar:scale
                                                 dataType:MPSDataTypeFloat32]
                name:nil];
            scores = [graph additionWithPrimaryTensor:scores
                                      secondaryTensor:mask name:nil];
            MPSGraphTensor *probabilities = [graph softMaxWithTensor:scores
                                                                axis:3 name:nil];
            probabilities = [graph castTensor:probabilities toType:bf16
                                         name:nil];
            attention = [graph matrixMultiplicationWithPrimaryTensor:probabilities
                                                     secondaryTensor:vb name:nil];
        } else {
            attention = [graph scaledDotProductAttentionWithQueryTensor:qt
                keyTensor:kb valueTensor:vb maskTensor:mask scale:scale
                name:nil];
        }
        H3SDPA *result = [[H3SDPA alloc] init];
        result.graph = graph;
        result.query = q;
        result.key = k;
        result.value = v;
        result.output = [graph transposeTensor:attention dimension:1
                                 withDimension:2 name:nil];
        result.inputShape = qShape;
        result.kvShape = kvShape;
        result.outputShape = qShape;
        gpu.sdpaCache[cacheKey] = result;
        return result;
    }
}

/* Query rows per piece: a piece's scores are groups x rows x keys, and the
 * last row block sees all `sequence` keys. */
static uint32_t h3_gpu_gqa_plan_rows(uint64_t budget, uint64_t cost,
                                     uint32_t sequence, uint32_t groups) {
    uint64_t per_row = (uint64_t)groups * sequence * cost;
    uint64_t rows = per_row ? budget / per_row : sequence;
    return (uint32_t)(rows < 1 ? 1 : rows < sequence ? rows : sequence);
}

static int h3_gpu_gqa_chunked(H3GPU *gpu, h3_gpu_tensor *output,
                              const h3_gpu_tensor *query,
                              const h3_gpu_tensor *key,
                              const h3_gpu_tensor *value, uint32_t sequence,
                              uint32_t query_heads, uint32_t kv_heads,
                              uint32_t head_dim, float scale, int explicit,
                              uint32_t chunk_rows) {
    uint32_t groups = query_heads / kv_heads;
    size_t elem = 2;
    if (((size_t)head_dim * elem) % 16) {
        h3_gpu_set_error(gpu, @"chunked causal GQA needs head_dim * 2 to be "
                         @"a multiple of 16 (head_dim %u)", head_dim);
        return 0;
    }
    MPSDataType bf16 = MPSDataTypeBFloat16;
    size_t q_elements = (size_t)sequence * query_heads * head_dim;
    size_t kv_elements = (size_t)sequence * kv_heads * head_dim;
    h3_gpu_tensor *sq = h3_gpu_sdpa_scratch(gpu, 0, q_elements, H3_GPU_BF16);
    h3_gpu_tensor *sk = h3_gpu_sdpa_scratch(gpu, 1, kv_elements, H3_GPU_BF16);
    h3_gpu_tensor *sv = h3_gpu_sdpa_scratch(gpu, 2, kv_elements, H3_GPU_BF16);
    h3_gpu_tensor *so = h3_gpu_sdpa_scratch(gpu, 3, q_elements, H3_GPU_BF16);
    if (!sq || !sk || !sv || !so) return 0;
    if (!h3_gpu_sdpa_run_transpose(gpu, query, sq, sequence, kv_heads,
                                   groups * head_dim, bf16, 1) ||
        !h3_gpu_sdpa_run_transpose(gpu, key, sk, sequence, kv_heads,
                                   head_dim, bf16, 1) ||
        !h3_gpu_sdpa_run_transpose(gpu, value, sv, sequence, kv_heads,
                                   head_dim, bf16, 1)) return 0;
    uint64_t pieces = 0;
    for (uint32_t g = 0; g < kv_heads; g++) {
        for (uint32_t r0 = 0; r0 < sequence; r0 += chunk_rows) {
            uint32_t rn = sequence - r0 < chunk_rows ? sequence - r0 : chunk_rows;
            H3SDPA *piece = h3_gpu_gqa_chunk_graph(gpu, groups, rn, r0,
                                                   head_dim, scale, explicit);
            if (!piece) return 0;
            size_t q_offset = ((size_t)g * sequence + r0) * groups *
                              head_dim * elem;
            size_t kv_offset = (size_t)g * sequence * head_dim * elem;
            @autoreleasepool {
                MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
                NSDictionary *feeds = @{
                    piece.query: h3_gpu_graph_view(sq, piece.inputShape, bf16,
                                                   q_offset),
                    piece.key: h3_gpu_graph_view(sk, piece.kvShape, bf16,
                                                 kv_offset),
                    piece.value: h3_gpu_graph_view(sv, piece.kvShape, bf16,
                                                   kv_offset)
                };
                NSDictionary *results = @{
                    piece.output: h3_gpu_graph_view(so, piece.outputShape,
                                                    bf16, q_offset)
                };
                @try {
                    [piece.graph encodeToCommandBuffer:command feeds:feeds
                                      targetOperations:nil
                                     resultsDictionary:results
                                   executionDescriptor:nil];
                } @catch (NSException *exception) {
                    h3_gpu_set_error(gpu, @"MPSGraph chunked causal GQA "
                                     @"failed (KV head %u, rows %u..%u): %@",
                                     g, r0, r0 + rn, exception.reason);
                    return 0;
                }
                gpu.command = command.rootCommandBuffer;
            }
            pieces++;
        }
    }
    if (!h3_gpu_sdpa_run_transpose(gpu, so, output, sequence, kv_heads,
                                   groups * head_dim, bf16, 0)) return 0;
    h3_gpu_stats stats = gpu.stats;
    stats.mps_sdpa_dispatches++;
    stats.mps_sdpa_chunk_dispatches += pieces;
    gpu.stats = stats;
    return 1;
}

/* The MPSGraph route for causal GQA: one call in FULL mode, pieces
 * otherwise. */
static int h3_gpu_gqa_graph_route(H3GPU *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *query,
                                  const h3_gpu_tensor *key,
                                  const h3_gpu_tensor *value,
                                  uint32_t sequence, uint32_t query_heads,
                                  uint32_t kv_heads, uint32_t head_dim,
                                  float scale) {
    h3_gpu_sdpa_mode mode = h3_gpu_sdpa_resolve_mode(gpu);
    uint64_t budget = h3_gpu_sdpa_budget(gpu);
    uint32_t groups = query_heads / kv_heads;
    uint32_t chunk_rows = mode == H3_GPU_SDPA_FULL ? sequence :
        h3_gpu_gqa_plan_rows(budget, h3_gpu_sdpa_score_cost(
                                 H3_GPU_BF16, mode == H3_GPU_SDPA_EXPLICIT),
                             sequence, groups);
    h3_gpu_sdpa_log(gpu, mode, budget,
                    mode == H3_GPU_SDPA_FULL ? query_heads : groups,
                    chunk_rows, MPSDataTypeBFloat16, 1, sequence, sequence,
                    query_heads, head_dim, 1, 0, 0, query, key, value, output);
    if (mode == H3_GPU_SDPA_FULL)
        return h3_gpu_gqa_mps(gpu, output, query, key, value, sequence,
                              query_heads, kv_heads, head_dim, scale);
    return h3_gpu_gqa_chunked(gpu, output, query, key, value, sequence,
                              query_heads, kv_heads, head_dim, scale,
                              mode == H3_GPU_SDPA_EXPLICIT, chunk_rows);
}

void h3_gpu_gqa_chunk_plan(h3_gpu *opaque, uint32_t sequence,
                           uint32_t query_heads, uint32_t kv_heads,
                           uint32_t *chunk_rows) {
    H3GPU *gpu = GPU(opaque);
    h3_gpu_sdpa_mode mode = h3_gpu_sdpa_resolve_mode(gpu);
    *chunk_rows = mode == H3_GPU_SDPA_FULL || !kv_heads ? sequence :
        h3_gpu_gqa_plan_rows(h3_gpu_sdpa_budget(gpu), h3_gpu_sdpa_score_cost(
                                 H3_GPU_BF16, mode == H3_GPU_SDPA_EXPLICIT),
                             sequence, query_heads / kv_heads);
}

int h3_gpu_gqa_causal_graph_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *query,
                                 const h3_gpu_tensor *key,
                                 const h3_gpu_tensor *value,
                                 uint32_t sequence, uint32_t query_heads,
                                 uint32_t kv_heads, uint32_t head_dim,
                                 float scale) {
    H3GPU *gpu = GPU(opaque);
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t kv_count = (size_t)sequence * kv_heads * head_dim;
    if (!sequence || !query_heads || !kv_heads || !head_dim ||
        query_heads % kv_heads ||
        !h3_gpu_require_bf16(gpu, query, query_count, @"GQA query") ||
        !h3_gpu_require_bf16(gpu, key, kv_count, @"GQA key") ||
        !h3_gpu_require_bf16(gpu, value, kv_count, @"GQA value") ||
        !h3_gpu_require_bf16(gpu, output, query_count, @"GQA output") ||
        !h3_gpu_require_command(gpu)) return 0;
    return h3_gpu_gqa_graph_route(gpu, output, query, key, value, sequence,
                                  query_heads, kv_heads, head_dim, scale);
}

/* The same attention with its score rows in device memory, in batches of
 * query rows under a fixed scratch size. h3_gqa_causal_group_bf16 runs the
 * eight query heads of one KV head in one threadgroup (Qwen's layout),
 * h3_gqa_causal_rows_bf16 one head per threadgroup otherwise. Both keep
 * the arithmetic, its order and the thread count the reductions depend on
 * from h3_gqa_causal_bf16, so the output is bit-for-bit the same (checked
 * by tests/test_sdpa.c). Measured on the M5 at 64/8 heads: the grouped
 * kernel 2.1x faster at 64 tokens, 2.7x at 1,000, 4.0x at 3,428, 6.8x at
 * 6,416 and 8.8x at 7,900; the per-head one 1.5x at 3,428 and 2.5x at
 * 6,416 and no faster below about 2,000. Returns 1 when it ran, -1 on an
 * error, and 0 when this path does not apply (too short, H3_GQA_ROWS=0, or
 * a pipeline that cannot run the same thread count); the caller then runs
 * the original kernel. H3_GQA_ROWS=<n> sets the shortest sequence either
 * kernel takes, H3_GQA_GROUP=0 turns the grouped one off. */
static int h3_gpu_gqa_rows(H3GPU *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value, uint32_t sequence,
                           uint32_t query_heads, uint32_t kv_heads,
                           uint32_t head_dim, float scale,
                           NSUInteger threads) {
    uint32_t groups = query_heads / kv_heads;
    const char *group_env = getenv("H3_GQA_GROUP");
    int grouped = groups == 8 && !(group_env && !strcmp(group_env, "0"));
    id<MTLComputePipelineState> pipeline = grouped ?
        h3_gpu_pipeline(gpu, @"h3_gqa_causal_group_bf16") : nil;
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < threads) {
        grouped = 0;
        pipeline = h3_gpu_pipeline(gpu, @"h3_gqa_causal_rows_bf16");
    }
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < threads)
        return 0;
    const char *env = getenv("H3_GQA_ROWS");
    uint32_t minimum = env && *env ? (uint32_t)strtoul(env, NULL, 10) :
        grouped ? H3_GQA_GROUP_MIN_SEQUENCE : H3_GQA_ROWS_MIN_SEQUENCE;
    if (!minimum || sequence < minimum) return 0;
    uint64_t row_bytes = (uint64_t)query_heads * sequence * sizeof(float);
    uint64_t batch = H3_GQA_ROWS_SCRATCH_BYTES / row_bytes;
    if (batch < 1) batch = 1;
    if (batch > sequence) batch = sequence;
    size_t elements = (size_t)batch * query_heads * sequence;
    h3_gpu_tensor *scores = gpu.gqaScores;
    if (!scores || TENSOR(scores).elements < elements) {
        if (scores) {
            if (!gpu.sdpaRetired) gpu.sdpaRetired = [NSMutableArray array];
            [gpu.sdpaRetired addObject:[NSValue valueWithPointer:scores]];
        }
        scores = h3_gpu_tensor_new_f32((__bridge h3_gpu *)gpu, elements);
        gpu.gqaScores = scores;
        if (!scores) {
            h3_gpu_set_error(gpu, @"cannot allocate GQA score rows "
                             @"(%zu floats)", elements);
            return -1;
        }
    }
    @autoreleasepool {
        /* A serial encoder: each batch finishes before the next one
         * reuses the score rows. */
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(scores).buffer offset:0 atIndex:5];
        for (uint32_t r0 = 0; r0 < sequence; r0 += (uint32_t)batch) {
            uint32_t rows = sequence - r0 < batch ? sequence - r0 : (uint32_t)batch;
            gqa_rows_args args = {sequence, query_heads, kv_heads, head_dim,
                                  scale, r0, sequence};
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
            if (grouped)
                [encoder dispatchThreadgroups:MTLSizeMake(rows, kv_heads, 1)
                         threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            else
                [encoder dispatchThreadgroups:MTLSizeMake(rows, query_heads, 1)
                         threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_gqa_causal_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value,
                           uint32_t sequence, uint32_t query_heads,
                           uint32_t kv_heads, uint32_t head_dim,
                           float scale) {
    H3GPU *gpu = GPU(opaque);
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t kv_count = (size_t)sequence * kv_heads * head_dim;
    if (!sequence || !query_heads || !kv_heads || !head_dim ||
        query_heads % kv_heads || head_dim > 128 ||
        !h3_gpu_require_bf16(gpu, query, query_count, @"GQA query") ||
        !h3_gpu_require_bf16(gpu, key, kv_count, @"GQA key") ||
        !h3_gpu_require_bf16(gpu, value, kv_count, @"GQA value") ||
        !h3_gpu_require_bf16(gpu, output, query_count, @"GQA output") ||
        !h3_gpu_require_command(gpu)) return 0;
    if (getenv("H3_MPS_GQA") && h3_gpu_gqa_mps(
            gpu, output, query, key, value, sequence, query_heads,
            kv_heads, head_dim, scale)) return 1;
    size_t score_bytes = (size_t)sequence * sizeof(float);
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu,
                                                           @"h3_gqa_causal_bf16");
    if (!pipeline) return 0;
    if (score_bytes + pipeline.staticThreadgroupMemoryLength >
        gpu.device.maxThreadgroupMemoryLength) {
        /* The custom kernel keeps one score row per threadgroup in a fixed
         * allocation sized off the device's maxThreadgroupMemoryLength, so
         * it can't handle an arbitrarily long causal sequence - and
         * reference videos (unlike the short prompts and single reference
         * images this kernel was originally written against) can easily
         * produce enough vision tokens to exceed it. MPSGraph's
         * scaledDotProductAttention manages its own memory instead of a
         * fixed threadgroup buffer, so it has no such ceiling on its own -
         * but h3_gpu_gqa_graph() builds an explicit O(sequence^2) causal
         * mask and caches it (keyed by sequence) for the GPU's whole
         * lifetime, so an unbounded sequence here means an unbounded mask,
         * and every distinct sequence length seen stays resident.
         *
         * That's exactly what crashed this machine: a large reference
         * video pushed Qwen's causal sequence far beyond the custom
         * kernel's threadgroup ceiling, the fallback below built a series
         * of huge masks for it, and the process's resident memory hit
         * ~63 GiB (this Mac has 24 GiB) - severe enough thrashing that the
         * kernel's watchdog force-panicked the whole machine (confirmed
         * from the panic log's "LOW swap space" and the process snapshot's
         * residentMemoryBytes). Refuse cleanly past a size where the mask
         * itself is already unreasonable, instead of trading that hard
         * failure for a system crash. 16384 tokens = a 512 MiB mask; real
         * prompts/reference conditioning haven't been seen anywhere near
         * that once the fast kernel's own ~8000-token ceiling is
         * accounted for, so this only ever bites the pathological case.
         * Below the cap the fallback is h3_gpu_gqa_graph_route(): one
         * MPSGraph call where FULL is validated, bounded pieces elsewhere. */
        size_t mask_bytes = (size_t)sequence * (size_t)sequence * sizeof(uint16_t);
        size_t max_fallback_mask_bytes = (size_t)512 * 1024 * 1024;
        if (mask_bytes > max_fallback_mask_bytes) {
            h3_gpu_set_error(gpu,
                @"reference conditioning needs a %u-token causal sequence - "
                @"too long even for the MPSGraph fallback (would need a "
                @"%.1f GiB attention mask). Try a shorter or lower-"
                @"resolution reference video or image.",
                sequence, (double)mask_bytes / (1024.0 * 1024.0 * 1024.0));
            return 0;
        }
        return h3_gpu_gqa_graph_route(gpu, output, query, key, value,
                                      sequence, query_heads, kv_heads,
                                      head_dim, scale);
    }
    NSUInteger maximum_threads = MIN((NSUInteger)128,
                                     pipeline.maxTotalThreadsPerThreadgroup);
    NSUInteger threads = 1;
    while (threads * 2 <= maximum_threads) threads *= 2;
    int rows = h3_gpu_gqa_rows(gpu, output, query, key, value, sequence,
                               query_heads, kv_heads, head_dim, scale,
                               threads);
    if (rows) return rows > 0;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
        gqa_args args = {sequence, query_heads, kv_heads, head_dim, scale};
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        [encoder setThreadgroupMemoryLength:score_bytes atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(sequence, query_heads, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_add_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, left, elements, @"add left") ||
        !h3_gpu_require_bf16(gpu, right, elements, @"add right") ||
        !h3_gpu_require_bf16(gpu, output, elements, @"add output")) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_add_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(left).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(right).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:3];
        });
}

int h3_gpu_sub_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, left, elements, @"subtract left") ||
        !h3_gpu_require_bf16(gpu, right, elements, @"subtract right") ||
        !h3_gpu_require_bf16(gpu, output, elements, @"subtract output"))
        return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_sub_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(left).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(right).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:3];
        });
}

int h3_gpu_token_pool_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           size_t input_offset,
                           h3_gpu_tensor *original,
                           size_t original_offset,
                           h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    size_t elements = (size_t)rows * width;
    size_t input_elements = (size_t)input_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!input_rows || !rows || rows > input_rows ||
        baseline_rows > rows || !width ||
        elements > UINT32_MAX || input_offset > UINT32_MAX ||
        input_elements > UINT32_MAX - input_offset ||
        original_offset > UINT32_MAX ||
        input_elements > UINT32_MAX - original_offset ||
        baseline_offset > UINT32_MAX ||
        baseline_elements > UINT32_MAX - baseline_offset ||
        !input || TENSOR(input).dtype != H3_GPU_BF16 ||
        input_offset > TENSOR(input).elements ||
        input_elements > TENSOR(input).elements - input_offset ||
        !original || TENSOR(original).dtype != H3_GPU_BF16 ||
        original_offset > TENSOR(original).elements ||
        input_elements > TENSOR(original).elements - original_offset ||
        !h3_gpu_require_bf16(gpu, output, elements, @"token pool output") ||
        !baseline || TENSOR(baseline).dtype != H3_GPU_BF16 ||
        baseline_offset > TENSOR(baseline).elements ||
        baseline_elements > TENSOR(baseline).elements - baseline_offset ||
        !h3_gpu_require_elements(gpu, baseline_indices, rows,
                                 @"token pool baseline indices") ||
        TENSOR(baseline_indices).dtype != H3_GPU_U32 ||
        !h3_gpu_require_elements(gpu, pairs, (size_t)rows * 2,
                                 @"token pool pairs") ||
        TENSOR(pairs).dtype != H3_GPU_U32) return 0;
    typedef struct {
        uint32_t input_offset, original_offset, baseline_offset;
        uint32_t rows, width;
    } token_pool_args;
    token_pool_args args = {
        (uint32_t)input_offset, (uint32_t)original_offset,
        (uint32_t)baseline_offset, rows, width
    };
    return h3_gpu_dispatch_2d(gpu, @"h3_token_pool_bf16", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(pairs).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(baseline).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(baseline_indices).buffer
                          offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(original).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
        });
}

int h3_gpu_token_pool_adaln_bf16(
                           h3_gpu *opaque, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, size_t input_offset,
                           h3_gpu_tensor *original, size_t original_offset,
                           h3_gpu_tensor *baseline, size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t input_rows, uint32_t rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t elements = (size_t)rows * width;
    size_t input_elements = (size_t)input_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!input_rows || !rows || rows > input_rows || !width || width > 5376 ||
        baseline_rows > rows || shift_slot >= slots || scale_slot >= slots ||
        elements > UINT32_MAX || input_offset > UINT32_MAX ||
        input_elements > UINT32_MAX - input_offset ||
        original_offset > UINT32_MAX ||
        input_elements > UINT32_MAX - original_offset ||
        baseline_offset > UINT32_MAX ||
        baseline_elements > UINT32_MAX - baseline_offset ||
        !input || TENSOR(input).dtype != H3_GPU_BF16 ||
        input_offset > TENSOR(input).elements ||
        input_elements > TENSOR(input).elements - input_offset ||
        !original || TENSOR(original).dtype != H3_GPU_BF16 ||
        original_offset > TENSOR(original).elements ||
        input_elements > TENSOR(original).elements - original_offset ||
        !h3_gpu_require_bf16(gpu, residual, elements,
                             @"fused token pool residual") ||
        !baseline || TENSOR(baseline).dtype != H3_GPU_BF16 ||
        baseline_offset > TENSOR(baseline).elements ||
        baseline_elements > TENSOR(baseline).elements - baseline_offset ||
        !h3_gpu_require_elements(gpu, baseline_indices, rows,
                                 @"fused token pool baseline indices") ||
        TENSOR(baseline_indices).dtype != H3_GPU_U32 ||
        !h3_gpu_require_elements(gpu, pairs, (size_t)rows * 2,
                                 @"fused token pool pairs") ||
        TENSOR(pairs).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, norm_weight, width,
                             @"fused token pool norm") ||
        !h3_gpu_require_bf16(gpu, modulation, 1,
                             @"fused token pool modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows,
                                 @"fused token pool row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, elements,
                             @"fused token pool AdaLN output") ||
        !h3_gpu_require_command(gpu)) return 0;
    typedef struct {
        uint32_t input_offset, original_offset, baseline_offset;
        uint32_t rows, width, slots, shift_slot, scale_slot;
        float epsilon;
    } token_pool_adaln_args;
    token_pool_adaln_args args = {
        (uint32_t)input_offset, (uint32_t)original_offset,
        (uint32_t)baseline_offset, rows, width, slots, shift_slot,
        scale_slot, epsilon
    };
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, @"h3_token_pool_adaln_bf16");
    if (!pipeline) return 0;
    const NSUInteger threads = 256;
    if (pipeline.maxTotalThreadsPerThreadgroup < threads) {
        h3_gpu_set_error(gpu,
            @"fused token pool AdaLN needs a 256-thread threadgroup");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(pairs).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(baseline).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(baseline_indices).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(original).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:6];
        [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:7];
        [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:8];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:9];
        [encoder setBytes:&args length:sizeof(args) atIndex:10];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_token_expand_delta_bf16(
                           h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents, uint32_t rows,
                           uint32_t reduced_rows, uint32_t baseline_rows,
                           uint32_t width,
                           uint32_t exact_prefix_rows,
                           float update_scale) {
    H3GPU *gpu = GPU(opaque);
    size_t elements = (size_t)rows * width;
    size_t reduced_elements = (size_t)reduced_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!rows || !reduced_rows || reduced_rows > rows ||
        baseline_rows > reduced_rows || !width ||
        exact_prefix_rows > reduced_rows || elements > UINT32_MAX ||
        reduced_elements > UINT32_MAX || original_offset > UINT32_MAX ||
        elements > UINT32_MAX - original_offset ||
        baseline_offset > UINT32_MAX ||
        baseline_elements > UINT32_MAX - baseline_offset ||
        !original || TENSOR(original).dtype != H3_GPU_BF16 ||
        original_offset > TENSOR(original).elements ||
        elements > TENSOR(original).elements - original_offset ||
        !h3_gpu_require_bf16(gpu, output, elements,
                             @"token expand output") ||
        !h3_gpu_require_bf16(gpu, reduced, reduced_elements,
                             @"token expand reduced") ||
        !baseline || TENSOR(baseline).dtype != H3_GPU_BF16 ||
        baseline_offset > TENSOR(baseline).elements ||
        baseline_elements > TENSOR(baseline).elements - baseline_offset ||
        !h3_gpu_require_elements(gpu, baseline_indices, reduced_rows,
                                 @"token expand baseline indices") ||
        TENSOR(baseline_indices).dtype != H3_GPU_U32 ||
        !h3_gpu_require_elements(gpu, parents, rows,
                                 @"token expand parents") ||
        TENSOR(parents).dtype != H3_GPU_U32) return 0;
    typedef struct {
        uint32_t original_offset, baseline_offset;
        uint32_t rows, width, exact_prefix_rows;
        float update_scale;
    } token_expand_args;
    token_expand_args args = {
        (uint32_t)original_offset, (uint32_t)baseline_offset,
        rows, width, exact_prefix_rows, update_scale
    };
    return h3_gpu_dispatch_2d(gpu, @"h3_token_expand_delta_bf16", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(original).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(reduced).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(baseline).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(baseline_indices).buffer
                          offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(parents).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
        });
}

int h3_gpu_token_expand_adaln_bf16(
                           h3_gpu *opaque, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t rows, uint32_t reduced_rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t exact_prefix_rows, float update_scale,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t elements = (size_t)rows * width;
    size_t reduced_elements = (size_t)reduced_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!rows || !reduced_rows || reduced_rows > rows || !width ||
        width > 5376 || baseline_rows > reduced_rows ||
        exact_prefix_rows > reduced_rows || shift_slot >= slots ||
        scale_slot >= slots || elements > UINT32_MAX ||
        reduced_elements > UINT32_MAX || original_offset > UINT32_MAX ||
        elements > UINT32_MAX - original_offset ||
        baseline_offset > UINT32_MAX ||
        baseline_elements > UINT32_MAX - baseline_offset ||
        !original || TENSOR(original).dtype != H3_GPU_BF16 ||
        original_offset > TENSOR(original).elements ||
        elements > TENSOR(original).elements - original_offset ||
        !h3_gpu_require_bf16(gpu, reduced, reduced_elements,
                             @"fused token AdaLN reduced input") ||
        !baseline || TENSOR(baseline).dtype != H3_GPU_BF16 ||
        baseline_offset > TENSOR(baseline).elements ||
        baseline_elements > TENSOR(baseline).elements - baseline_offset ||
        !h3_gpu_require_elements(gpu, baseline_indices, reduced_rows,
                                 @"fused token AdaLN baseline indices") ||
        TENSOR(baseline_indices).dtype != H3_GPU_U32 ||
        !h3_gpu_require_elements(gpu, parents, rows,
                                 @"fused token AdaLN parents") ||
        TENSOR(parents).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, residual, elements,
                             @"fused token AdaLN residual") ||
        !h3_gpu_require_bf16(gpu, norm_weight, width,
                             @"fused token AdaLN norm") ||
        !h3_gpu_require_bf16(gpu, modulation, 1,
                             @"fused token AdaLN modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows,
                                 @"fused token AdaLN row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, elements,
                             @"fused token AdaLN output") ||
        !h3_gpu_require_command(gpu)) return 0;
    typedef struct {
        uint32_t original_offset, baseline_offset, rows, width;
        uint32_t exact_prefix_rows, slots, shift_slot, scale_slot;
        float update_scale, epsilon;
    } token_expand_adaln_args;
    token_expand_adaln_args args = {
        (uint32_t)original_offset, (uint32_t)baseline_offset, rows, width,
        exact_prefix_rows, slots, shift_slot, scale_slot,
        update_scale, epsilon
    };
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, @"h3_token_expand_adaln_bf16");
    if (!pipeline) return 0;
    const NSUInteger threads = 256;
    if (pipeline.maxTotalThreadsPerThreadgroup < threads) {
        h3_gpu_set_error(gpu,
            @"fused token AdaLN needs a 256-thread threadgroup");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(original).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(reduced).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(baseline).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(baseline_indices).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(parents).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:6];
        [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:7];
        [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:8];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:9];
        [encoder setBytes:&args length:sizeof(args) atIndex:10];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_euler_bf16(h3_gpu *opaque, h3_gpu_tensor *sample,
                      size_t sample_offset, const h3_gpu_tensor *last,
                      const h3_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio) {
    H3GPU *gpu = GPU(opaque);
    if (!sample || TENSOR(sample).dtype != H3_GPU_F32 ||
        sample_offset > TENSOR(sample).elements ||
        elements > TENSOR(sample).elements - sample_offset ||
        sample_offset > UINT32_MAX || elements > UINT32_MAX - sample_offset ||
        !h3_gpu_require_bf16(gpu, last, elements, @"Euler last velocity") ||
        !h3_gpu_require_bf16(gpu, previous, elements,
                             @"Euler previous velocity")) return 0;
    euler_args args = {(uint32_t)sample_offset, elements, delta, ratio};
    return h3_gpu_dispatch_1d(gpu, @"h3_euler_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(sample).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(last).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(previous).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_renoise_bf16(h3_gpu *opaque, h3_gpu_tensor *sample,
                        size_t sample_offset, const h3_gpu_tensor *velocity,
                        uint32_t elements, float sigma, float sigma_next,
                        const uint32_t keys[2]) {
    H3GPU *gpu = GPU(opaque);
    if (!sample || !keys || TENSOR(sample).dtype != H3_GPU_F32 ||
        sample_offset > TENSOR(sample).elements ||
        elements > TENSOR(sample).elements - sample_offset ||
        sample_offset > UINT32_MAX || elements > UINT32_MAX - sample_offset ||
        !h3_gpu_require_bf16(gpu, velocity, elements, @"re-noise velocity"))
        return 0;
    renoise_args args = {(uint32_t)sample_offset, elements, sigma, sigma_next,
                         keys[0], keys[1]};
    return h3_gpu_dispatch_1d(gpu, @"h3_renoise_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(sample).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(velocity).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_silu_mul_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate,
                         const h3_gpu_tensor *up, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, gate, elements, @"SiLU gate") ||
        !h3_gpu_require_bf16(gpu, up, elements, @"SiLU up") ||
        !h3_gpu_require_bf16(gpu, output, elements, @"SiLU product")) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_silu_mul_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(gate).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(up).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:3];
        });
}

/* ---- VSR: ReAE convolutions on TensorOps (see h3_shaders.metal) ---- */

typedef struct {
    uint32_t rows, x_rows, guard, in_ch, out_ch, taps, inputs, relu, has_bias, has_residual;
    int32_t offsets[9];
} vsr_tapconv_args;
typedef struct { uint32_t frames, height, width, channels, padded_channels, guard; } vsr_grid_args;

int h3_gpu_has_tensor_ops(const h3_gpu *opaque) {
    if (!opaque) return 0;
    H3GPU *gpu = GPU((h3_gpu *)(void *)opaque);
    return gpu.tensorOpsEnabled;
}

int h3_gpu_vsr_tapconv_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                            const h3_gpu_tensor *x0, const h3_gpu_tensor *x1,
                            const h3_gpu_tensor *w0, const h3_gpu_tensor *w1,
                            const h3_gpu_tensor *bias, const h3_gpu_tensor *residual,
                            uint32_t rows, uint32_t x_rows, uint32_t guard,
                            uint32_t in_ch, uint32_t out_ch, uint32_t taps,
                            const int32_t *offsets, int relu) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu.tensorOpsEnabled) { h3_gpu_set_error(gpu, @"VSR tapconv needs TensorOps"); return 0; }
    if (!rows || !taps || taps > 9 || in_ch % 64 || out_ch % 64 || !offsets ||
        guard + rows + 128 > x_rows) {
        h3_gpu_set_error(gpu, @"VSR tapconv: bad arguments");
        return 0;
    }
    for (uint32_t t = 0; t < taps; t++)
        if (offsets[t] < -(int32_t)guard || (int64_t)guard + rows + 128 + offsets[t] > (int64_t)x_rows) {
            h3_gpu_set_error(gpu, @"VSR tapconv: tap offset outside the guard rows");
            return 0;
        }
    size_t x_count = (size_t)x_rows * in_ch, y_count = (size_t)x_rows * out_ch;
    size_t w_count = (size_t)taps * out_ch * in_ch;
    if (!h3_gpu_require_bf16(gpu, x0, x_count, @"VSR tapconv x0") ||
        (x1 && !h3_gpu_require_bf16(gpu, x1, x_count, @"VSR tapconv x1")) ||
        !h3_gpu_require_bf16(gpu, w0, w_count, @"VSR tapconv w0") ||
        (x1 && !h3_gpu_require_bf16(gpu, w1, w_count, @"VSR tapconv w1")) ||
        !h3_gpu_require_bf16(gpu, output, y_count, @"VSR tapconv output") ||
        (residual && !h3_gpu_require_bf16(gpu, residual, y_count, @"VSR tapconv residual")) ||
        !vsr_check_f32_vector(gpu, bias, out_ch, @"VSR tapconv bias")) return 0;
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, @"h3_vsr_tapconv_bf16_nax");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 128) {
        h3_gpu_set_error(gpu, @"device cannot dispatch the VSR TensorOps conv");
        return 0;
    }
    vsr_tapconv_args args = {rows, x_rows, guard, in_ch, out_ch, taps, x1 ? 2u : 1u,
                             relu != 0, bias != NULL, residual != NULL, {0}};
    memcpy(args.offsets, offsets, taps * sizeof(int32_t));
    const h3_gpu_tensor *xb = x1 ? x1 : x0, *wb = x1 ? w1 : w0;
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
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 127) / 128, out_ch / 64, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
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

/* ---- split-K int8 linear (see h3_linear_int8_split_k_nax_r128) ---- */
typedef struct { uint32_t rows, input_dim, output_dim, splits; } split_k_args;

int h3_gpu_linear_int8_bf16_split_k(h3_gpu *opaque, h3_gpu_tensor *output, h3_gpu_tensor *partial,
                                    h3_gpu_tensor *quantized_input, h3_gpu_tensor *input_scales,
                                    const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                                    const h3_gpu_tensor *weight_scales, uint32_t rows,
                                    uint32_t input_dim, uint32_t output_dim, uint32_t splits) {
    H3GPU *gpu = GPU(opaque);
    uint32_t padded_rows = (rows + 127u) & ~127u;
    if (!gpu.tensorOpsEnabled || rows < 128 || !splits || input_dim % (128 * splits) || output_dim % 128 ||
        !h3_gpu_require_i8(gpu, weight, (size_t)output_dim * input_dim, @"split-K weight") ||
        !h3_gpu_require_f32(gpu, weight_scales, output_dim, @"split-K weight scales") ||
        !h3_gpu_require_f32(gpu, partial, (size_t)splits * padded_rows * output_dim, @"split-K partials") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * output_dim, @"split-K output") ||
        !h3_gpu_require_command(gpu)) return 0;
    if (!getenv("H3_BENCH_SKIP_QUANT") &&
        !h3_gpu_quantize_bf16_int8_rows(opaque, quantized_input, input_scales, input, rows, padded_rows,
                                        input_dim, 1.0f, @"split-K input")) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, @"h3_linear_int8_split_k_nax_r128");
    id<MTLComputePipelineState> sum = h3_gpu_pipeline(gpu, @"h3_linear_sum_partials_bf16");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256 || !sum) {
        h3_gpu_set_error(gpu, @"split-K int8 linear is unavailable");
        return 0;
    }
    split_k_args args = {rows, input_dim, output_dim, splits};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(quantized_input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(input_scales).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(weight_scales).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(partial).buffer offset:0 atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake((padded_rows / 128) * (output_dim / 128), splits, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        id<MTLComputeCommandEncoder> adder = [gpu.command computeCommandEncoder];
        [adder setComputePipelineState:sum];
        [adder setBuffer:TENSOR(partial).buffer offset:0 atIndex:0];
        [adder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
        [adder setBytes:&args length:sizeof(args) atIndex:2];
        NSUInteger x = 32, y = MIN((NSUInteger)8, sum.maxTotalThreadsPerThreadgroup / x);
        [adder dispatchThreads:MTLSizeMake(output_dim, rows, 1) threadsPerThreadgroup:MTLSizeMake(x, y, 1)];
        [adder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches += 2;
    gpu.stats = stats;
    return 1;
}

/* The row quantizer alone (benchmarks). */
int h3_gpu_quantize_rows_bf16_int8(h3_gpu *opaque, h3_gpu_tensor *quantized, h3_gpu_tensor *scales,
                                   const h3_gpu_tensor *input, uint32_t rows, uint32_t columns) {
    uint32_t padded_rows = (rows + 127u) & ~127u;
    return h3_gpu_quantize_bf16_int8_rows(opaque, quantized, scales, input, rows, padded_rows, columns, 1.0f,
                                          @"row quantizer");
}

/* One K-contiguous int8 product into F32 partial slot `slot` (quantizes
 * the input first); h3_gpu_sum_partials_bf16 adds the slots. Lets a long
 * reduction run as K-contiguous halves, which the TensorOps tile handles
 * better than one long-stride pass. */
int h3_gpu_linear_int8_partial_f32(h3_gpu *opaque, h3_gpu_tensor *partial, uint32_t slot,
                                   h3_gpu_tensor *quantized_input, h3_gpu_tensor *input_scales,
                                   const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                                   const h3_gpu_tensor *weight_scales, uint32_t rows,
                                   uint32_t input_dim, uint32_t output_dim, int prequantized) {
    H3GPU *gpu = GPU(opaque);
    uint32_t padded_rows = (rows + 127u) & ~127u;
    size_t slot_elements = (size_t)padded_rows * output_dim;
    if (!gpu.tensorOpsEnabled || rows < 128 || input_dim % 128 || output_dim % 128 ||
        !h3_gpu_require_i8(gpu, weight, (size_t)output_dim * input_dim, @"partial weight") ||
        !h3_gpu_require_f32(gpu, weight_scales, output_dim, @"partial weight scales") ||
        !h3_gpu_require_f32(gpu, partial, (size_t)(slot + 1) * slot_elements, @"partials") ||
        !h3_gpu_require_command(gpu)) return 0;
    if (prequantized) {
        if (!h3_gpu_require_i8(gpu, quantized_input, (size_t)padded_rows * input_dim, @"partial quantized input") ||
            !h3_gpu_require_f32(gpu, input_scales, padded_rows, @"partial input scales")) return 0;
    } else if (!h3_gpu_quantize_bf16_int8_rows(opaque, quantized_input, input_scales, input, rows, padded_rows,
                                               input_dim, 1.0f, @"partial input")) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, @"h3_linear_int8_split_k_nax_r128");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu, @"split-K int8 linear is unavailable");
        return 0;
    }
    split_k_args args = {rows, input_dim, output_dim, 1};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(quantized_input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(input_scales).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(weight_scales).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(partial).buffer offset:(NSUInteger)slot * slot_elements * sizeof(float) atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake((padded_rows / 128) * (output_dim / 128), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_sum_partials_bf16(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *partial,
                             uint32_t rows, uint32_t output_dim, uint32_t splits) {
    H3GPU *gpu = GPU(opaque);
    uint32_t padded_rows = (rows + 127u) & ~127u;
    if (!splits || !h3_gpu_require_f32(gpu, partial, (size_t)splits * padded_rows * output_dim, @"partials") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * output_dim, @"partials output")) return 0;
    split_k_args args = {rows, output_dim, output_dim, splits};
    args.input_dim = 0;
    return h3_gpu_dispatch_2d(gpu, @"h3_linear_sum_partials_bf16", output_dim, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(partial).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
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


/* --- SeedVR2 kernels (h3_seedvr2.c) --- */

void h3_gpu_set_nax_linear_generic(h3_gpu *opaque, int on) {
    H3GPU *gpu = GPU(opaque);
    gpu.naxLinearGeneric = on != 0;
}

typedef struct { uint32_t rows, columns, group; } sv2_dequant_args;
typedef struct { uint32_t rows, width, has_scale, has_shift; float epsilon; } sv2_rms_args;
typedef struct { uint32_t rows, heads, head_dim, dim, pairs; float epsilon; } sv2_window_args;
typedef struct { uint32_t windows, txt_rows, width; } sv2_txt_mean_args;
typedef struct { uint32_t height, width, channels, top, bottom, left, right; } sv2_pad_args;
typedef struct { uint32_t height, width, channels, temporal; } sv2_shuffle_args;
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

int h3_gpu_sv2_shuffle_f32(h3_gpu *opaque, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                           uint32_t height, uint32_t width, uint32_t channels,
                           uint32_t temporal) {
    H3GPU *gpu = GPU(opaque);
    if (!sv2_check_f32(gpu, input, (size_t)height * width * channels * 4 * temporal, @"sv2 shuffle input") ||
        !sv2_check_f32(gpu, output, (size_t)height * width * 4 * channels, @"sv2 shuffle output")) return 0;
    sv2_shuffle_args args = {height, width, channels, temporal};
    return h3_gpu_dispatch_3d(gpu, @"h3_sv2_shuffle_f32", MTLSizeMake(channels, 2 * width, 2 * height),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
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
typedef struct { uint32_t height, width, channels, padded, in_padded, temporal, guard, in_guard; } sv2_gshuffle_args;
typedef struct { uint32_t height, width, padded, guard, in_width, in_guard, phase_y, phase_x; } sv2_gselect_args;

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
    H3GPU *gpu = GPU(opaque);
    if (channels > out_padded || channels * 4 * temporal > in_padded) {
        h3_gpu_set_error(gpu, @"sv2 grid shuffle: bad channels");
        return 0;
    }
    uint32_t hp = 2 * in_height + 2, wp = 2 * in_width + 2;
    if (!h3_gpu_require_bf16(gpu, input, ((size_t)in_guard + (size_t)(in_height + 2) * (in_width + 2)) * in_padded, @"sv2 grid shuffle input") ||
        !h3_gpu_require_bf16(gpu, output, ((size_t)out_guard + (size_t)hp * wp) * out_padded, @"sv2 grid shuffle output")) return 0;
    sv2_gshuffle_args args = {in_height, in_width, channels, out_padded, in_padded, temporal, out_guard, in_guard};
    return h3_gpu_dispatch_3d(gpu, @"h3_sv2_grid_shuffle_bf16", MTLSizeMake(out_padded, wp, hp),
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
    H3GPU *gpu = GPU(opaque);
    if (2 * (out_height - 1) + phase_y >= in_height || 2 * (out_width - 1) + phase_x >= in_width) {
        h3_gpu_set_error(gpu, @"sv2 grid select: the output reads outside the input");
        return 0;
    }
    uint32_t hp = out_height + 2, wp = out_width + 2;
    if (!h3_gpu_require_bf16(gpu, input, ((size_t)in_guard + (size_t)(in_height + 2) * (in_width + 2)) * padded, @"sv2 grid select input") ||
        !h3_gpu_require_bf16(gpu, output, ((size_t)out_guard + (size_t)hp * wp) * padded, @"sv2 grid select output")) return 0;
    sv2_gselect_args args = {out_height, out_width, padded, out_guard, in_width, in_guard, phase_y, phase_x};
    return h3_gpu_dispatch_3d(gpu, @"h3_sv2_grid_select_bf16", MTLSizeMake(padded, wp, hp),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}
