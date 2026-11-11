/* The Metal layer: the device and shader library, tensors, command buffers and
 * profiling. The kernels are in h3_gpu_blocks.m, h3_gpu_linear.m,
 * h3_gpu_attention.m and h3_gpu_conv.m (and the upscale module's own files). */
#include "h3_gpu_private.h"
@implementation H3Tensor
@end

@implementation H3SDPA
@end

@implementation H3GQA
@end

@implementation H3Linear
@end

@implementation H3MLP
@end

@implementation H3Conv
@end

@implementation H3GPU
@end

/* Kernel names registered by code outside this file (the upscale module's
 * h3_gpu_vsr.m and h3_gpu_sv2.m, from load-time constructors as the ccv
 * backend does): h3_gpu_create builds their pipelines with its own. */
h3_gpu_kernel_set h3_gpu_kernel_sets[H3_GPU_MAX_KERNEL_SETS];
size_t h3_gpu_kernel_set_count;

void h3_gpu_register_kernels(const char *const *names, size_t count, int needs_tensor_ops) {
    if (h3_gpu_kernel_set_count >= H3_GPU_MAX_KERNEL_SETS) return;
    h3_gpu_kernel_sets[h3_gpu_kernel_set_count++] = (h3_gpu_kernel_set){names, count, needs_tensor_ops};
}

/* Whether this device actually exposes the Metal 4 hardware tensor/matmul
 * units (Neural Accelerators) that the MetalPerformancePrimitives-based
 * kernels in h3_shaders.metal require - not just whether its marketing name
 * happens to contain the substring "M5". MTLGPUFamilyApple10 is the GPU
 * family Apple introduced alongside that hardware (first shipped in M5), so
 * querying it via -supportsFamily: reflects the real capability and, unlike
 * a name match, keeps working for future chips in the same GPU family. */
BOOL h3_device_has_tensor_ops(id<MTLDevice> device) {
    return device && [device supportsFamily:MTLGPUFamilyApple10];
}

MPSGraphTensorData *h3_gpu_graph_data(const h3_gpu_tensor *tensor,
                                             NSArray<NSNumber *> *shape,
                                             MPSDataType data_type,
                                             int stable) {
    H3Tensor *object = TENSOR(tensor);
    if (!stable || h3_option("H3_DISABLE_GRAPH_DATA_CACHE"))
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

MPSCommandBuffer *h3_gpu_mps_command(H3GPU *gpu) {
    const char *override = h3_option("H3_REUSE_MPS_COMMAND");
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
    const char *value = h3_option("H3_PROFILE");
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

void h3_gpu_set_error(H3GPU *gpu, NSString *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    gpu.lastError = [[NSString alloc] initWithFormat:format arguments:arguments];
    va_end(arguments);
}

int h3_gpu_require_command(H3GPU *gpu) {
    if (!gpu || !gpu.command) {
        if (gpu) h3_gpu_set_error(gpu, @"h3_gpu_begin() was not called");
        return 0;
    }
    return 1;
}

int h3_gpu_require_elements(H3GPU *gpu, const h3_gpu_tensor *tensor,
                                   size_t elements, NSString *label) {
    if (!tensor || TENSOR(tensor).elements < elements) {
        h3_gpu_set_error(gpu, @"%@ tensor is absent or too small", label);
        return 0;
    }
    return 1;
}

id<MTLComputePipelineState> h3_gpu_pipeline(H3GPU *gpu, NSString *name) {
    id<MTLComputePipelineState> pipeline = gpu.pipelines[name];
    if (!pipeline) h3_gpu_set_error(gpu, @"missing Metal pipeline %@", name);
    return pipeline;
}

int h3_gpu_dispatch_1d(H3GPU *gpu, NSString *name, uint32_t count,
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

int h3_gpu_dispatch_2d(H3GPU *gpu, NSString *name, uint32_t width,
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

int h3_gpu_dispatch_3d(H3GPU *gpu, NSString *name, MTLSize grid,
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

int h3_gpu_dispatch_rows(H3GPU *gpu, NSString *name, uint32_t rows,
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
        if (h3_option("H3_DEBUG_GPU_MEMORY")) {
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
            const char *nax = h3_option("H3_NAX");
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
                if (h3_option("H3_NAX_DIAGNOSTIC"))
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
            @"h3_dequantize", @"h3_comfy_int8_expand",
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
            @"h3_qk_rope_split_bf16_coop", @"h3_swiglu_split_bf16",
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
            @"h3_vae_encoder_group_norm_silu_f32",
            @"h3_vae_encoder_norm_chunk_sum_f32",
            @"h3_vae_encoder_norm_chunk_square_f32",
            @"h3_vae_encoder_norm_chunk_apply_silu_f32"
        ] mutableCopy];
        if (gpu.tensorOpsEnabled) {
            [names addObject:@"h3_linear_bf16_nax_r128"];
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
        /* Kernels of modules outside this file (h3_gpu_register_kernels). */
        for (size_t set = 0; set < h3_gpu_kernel_set_count; set++) {
            if (h3_gpu_kernel_sets[set].needs_tensor_ops && !gpu.tensorOpsEnabled) continue;
            for (size_t i = 0; i < h3_gpu_kernel_sets[set].count; i++)
                [names addObject:[NSString stringWithUTF8String:h3_gpu_kernel_sets[set].names[i]]];
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
        if (h3_option("H3_DEBUG_GPU_MEMORY")) {
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

size_t h3_gpu_page_bytes(void) {
    return (size_t)getpagesize();
}

static h3_gpu_tensor *h3_gpu_tensor_alias(h3_gpu_tensor *parent_opaque,
                                          size_t byte_offset,
                                          size_t elements, size_t item_size,
                                          h3_gpu_dtype dtype) {
    if (!parent_opaque || !elements || elements > SIZE_MAX / item_size)
        return NULL;
    H3Tensor *parent = TENSOR(parent_opaque);
    H3GPU *gpu = parent.owner;
    size_t page = h3_gpu_page_bytes();
    size_t bytes = elements * item_size;
    size_t length = (bytes + page - 1) / page * page;
    id<MTLBuffer> base = parent.aliasParent ? nil : parent.buffer;
    if (!gpu || !base || byte_offset % page ||
        byte_offset > base.length || length > base.length - byte_offset)
        return NULL;
    H3Tensor *tensor = [[H3Tensor alloc] init];
    tensor.elements = elements;
    tensor.bytes = 0; /* the parent owns (and counts) the memory */
    tensor.dtype = dtype;
    tensor.buffer = [gpu.device
        newBufferWithBytesNoCopy:(char *)base.contents + byte_offset
                          length:length
                         options:MTLResourceStorageModeShared
                     deallocator:nil];
    if (!tensor.buffer) {
        h3_gpu_set_error(gpu, @"cannot alias %zu bytes of a Metal buffer",
                         length);
        return NULL;
    }
    tensor.aliasParent = base;
    tensor.owner = gpu;
    return (__bridge_retained h3_gpu_tensor *)tensor;
}

h3_gpu_tensor *h3_gpu_tensor_alias_bf16(h3_gpu_tensor *parent,
                                        size_t byte_offset, size_t elements) {
    return h3_gpu_tensor_alias(parent, byte_offset, elements,
                               sizeof(uint16_t), H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_alias_i8(h3_gpu_tensor *parent,
                                      size_t byte_offset, size_t elements) {
    return h3_gpu_tensor_alias(parent, byte_offset, elements, sizeof(int8_t),
                               H3_GPU_I8);
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
    const char *zero_copy = h3_option("H3_ZERO_COPY_WEIGHTS");
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
    /* uncached bit 1: the read may fill only the leading elements. */
    int prefix = (uncached & 2) != 0;
    uncached &= 1;
    if (error && error_size) error[0] = '\0';
    if (!opaque || !path || !*path ||
        TENSOR(opaque).dtype != dtype ||
        (prefix ? !elements || elements > TENSOR(opaque).elements :
                  elements != TENSOR(opaque).elements) ||
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

int h3_gpu_tensor_stream_file_prefix_bf16(h3_gpu_tensor *opaque,
                                          const char *path,
                                          uint64_t file_offset,
                                          size_t elements, char *error,
                                          size_t error_size) {
    return h3_gpu_tensor_read_file_mode(
        opaque, path, file_offset, elements, sizeof(uint16_t), H3_GPU_BF16,
        "BF16", 3, error, error_size);
}

int h3_gpu_tensor_stream_file_prefix_i8(h3_gpu_tensor *opaque,
                                        const char *path,
                                        uint64_t file_offset,
                                        size_t elements, char *error,
                                        size_t error_size) {
    return h3_gpu_tensor_read_file_mode(
        opaque, path, file_offset, elements, sizeof(int8_t), H3_GPU_I8,
        "I8", 3, error, error_size);
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
        if (!object.aliasParent)
            [object.buffer setPurgeableState:MTLPurgeableStateEmpty];
        object.buffer = nil;
        object.aliasParent = nil;
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

int h3_gpu_require_bf16(H3GPU *gpu, const h3_gpu_tensor *tensor,
                               size_t elements, NSString *label) {
    if (!h3_gpu_require_elements(gpu, tensor, elements, label)) return 0;
    if (TENSOR(tensor).dtype != H3_GPU_BF16) {
        h3_gpu_set_error(gpu, @"%@ tensor is not BF16", label);
        return 0;
    }
    return 1;
}

int h3_gpu_require_i8(H3GPU *gpu, const h3_gpu_tensor *tensor,
                             size_t elements, NSString *label) {
    if (!h3_gpu_require_elements(gpu, tensor, elements, label)) return 0;
    if (TENSOR(tensor).dtype != H3_GPU_I8) {
        h3_gpu_set_error(gpu, @"%@ tensor is not int8", label);
        return 0;
    }
    return 1;
}

int h3_gpu_require_f32(H3GPU *gpu, const h3_gpu_tensor *tensor,
                              size_t elements, NSString *label) {
    if (!h3_gpu_require_elements(gpu, tensor, elements, label)) return 0;
    if (TENSOR(tensor).dtype != H3_GPU_F32) {
        h3_gpu_set_error(gpu, @"%@ tensor is not F32", label);
        return 0;
    }
    return 1;
}

int h3_gpu_has_tensor_ops(const h3_gpu *opaque) {
    if (!opaque) return 0;
    H3GPU *gpu = GPU((h3_gpu *)(void *)opaque);
    return gpu.tensorOpsEnabled;
}

void h3_gpu_set_nax_linear_generic(h3_gpu *opaque, int on) {
    H3GPU *gpu = GPU(opaque);
    gpu.naxLinearGeneric = on != 0;
}
