// Experimental, opt-in attention backend bridging to liuliu/ccv's Metal
// int8 NAX attention kernel (NAInt8AttentionKernel, via the public
// ccv_nnc_mfa_encode_attention C API). See SPEEDUP_ROADMAP.md item 5 and
// tools/ccv_eval/README.md for the numbers this was validated against and
// how to reproduce them.
//
// Only built and linked when the Makefile has CCV_DIR set (see Makefile) -
// every other binary that links libh3.a is completely unaffected, since
// h3_gpu_ccv_dense_attention_bf16 (h3_gpu.h/h3_gpu.m) already fails cleanly
// on its own when nothing registers a real implementation. Linking this
// file causes that registration to happen automatically, via the
// constructor at the bottom of this file - no other code needs to change
// to enable or disable this backend, only the build (see Makefile,
// CCV_DIR).
//
// Bridging note: metal-cpp's MTL::Device/MTL::Buffer/MTL::CommandBuffer are
// thin wrappers holding exactly the same pointer value as the underlying
// Objective-C id<MTLDevice>/id<MTLBuffer>/id<MTLCommandBuffer> object - this
// is metal-cpp's documented interop mechanism, not a private assumption
// made here. h3_gpu_raw_device()/h3_gpu_raw_buffer()/
// h3_gpu_raw_command_buffer() (h3_gpu.m) hand out that same pointer value
// as void*, unretained (the underlying h3_gpu/h3_gpu_tensor keeps it
// alive); casting it straight to the matching mtl_*_t* here is exactly
// that interop, not a new bridge.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "h3_gpu.h"
#include "nnc/mfa/ccv_nnc_mfa.hpp"
#include "nnc/mfa/ccv_nnc_mfa_attention.hpp"

namespace {

struct ccv_attention_state {
    ccv_nnc_mfa_context_t *context = nullptr;
    void *device_key = nullptr;
    id<MTLBuffer> q_f16 = nil;
    id<MTLBuffer> k_f16 = nil;
    id<MTLBuffer> v_f16 = nil;
    id<MTLBuffer> out_f16 = nil;
    size_t capacity_elements = 0;
};

/* Single-generation, single-threaded assumption matches how h3_dit.c drives
 * one h3_gpu at a time - not safe to share across concurrent generations. */
ccv_attention_state g_state;

/* H3_CCV_MEMLOG=1: report the bridge's own FP16 scratch and ccv's internal
 * context scratch (int8 Q/K/V, scales, V means, L) whenever either grows,
 * with the device's total allocation for context. Diagnostics only. */
bool memlog_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char *value = getenv("H3_CCV_MEMLOG");
        enabled = value && *value == '1';
    }
    return enabled;
}

double gib(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

bool ensure_scratch(id<MTLDevice> device, size_t elements) {
    if (elements <= g_state.capacity_elements && g_state.q_f16) return true;
    size_t bytes = elements * sizeof(uint16_t);
    MTLResourceOptions options = MTLResourceStorageModePrivate;
    g_state.q_f16 = [device newBufferWithLength:bytes options:options];
    g_state.k_f16 = [device newBufferWithLength:bytes options:options];
    g_state.v_f16 = [device newBufferWithLength:bytes options:options];
    g_state.out_f16 = [device newBufferWithLength:bytes options:options];
    if (!g_state.q_f16 || !g_state.k_f16 || !g_state.v_f16 || !g_state.out_f16) {
        g_state.capacity_elements = 0;
        return false;
    }
    g_state.capacity_elements = elements;
    if (memlog_enabled())
        fprintf(stderr, "[ccv-mem] bridge fp16 scratch: 4 x %zu B = %.3f GiB "
                "(elements=%zu), device allocated %.3f GiB\n", bytes,
                gib(4 * bytes), elements, gib([device currentAllocatedSize]));
    return true;
}

void memlog_after_encode(id<MTLDevice> device, uint32_t rows,
                         uint32_t heads, uint32_t head_dim, int head_major,
                         int quantized, int direct) {
    if (!memlog_enabled()) return;
    static int logged_shape = 0;
    static size_t logged_ccv_scratch = 0;
    if (!logged_shape) {
        if (direct)
            fprintf(stderr, "[ccv-mem] rows=%u heads=%u head_dim=%u "
                    "input=bf16 head-major passed directly (batch=%u, "
                    "Hq=Hk=1), output bf16 head-major, int8 path\n", rows,
                    heads, head_dim, heads);
        else
            fprintf(stderr, "[ccv-mem] rows=%u heads=%u head_dim=%u "
                    "input=bf16 %s-major -> fp16 row-major, %s\n", rows,
                    heads, head_dim, head_major ? "head" : "row",
                    quantized ? "int8 path" : "fp16 path");
        logged_shape = 1;
    }
    auto *context = (ccv::nnc::mfa::context *)g_state.context;
    size_t ccv_scratch = context->scratch ? context->scratch->length() : 0;
    if (ccv_scratch != logged_ccv_scratch) {
        size_t int8_qkv = 3 * (size_t)rows * heads * head_dim;
        fprintf(stderr, "[ccv-mem] ccv context scratch: %zu B = %.3f GiB "
                "(int8 Q/K/V alone %.3f GiB), device allocated %.3f GiB\n",
                ccv_scratch, gib(ccv_scratch),
                quantized ? gib(int8_qkv) : 0.0,
                gib([device currentAllocatedSize]));
        logged_ccv_scratch = ccv_scratch;
    }
}

/* Whether this machine can run the backend at all: ccv's context must
 * support the device and it must have neural matrix accelerators (M5). */
int ccv_attention_supported(void) {
    static int supported = -1;
    if (supported < 0) {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        ccv_nnc_mfa_context_t *context = device ?
            ccv_nnc_init_mfa_context((mtl_device_t *)(__bridge void *)device) :
            nullptr;
        supported = context && ccv_nnc_mfa_context_supported(context) &&
                    ccv_nnc_mfa_has_neural_accelerators(context);
        if (context) ccv_nnc_deinit_mfa_context(context);
    }
    return supported;
}

int ccv_dense_attention_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *query,
                             const h3_gpu_tensor *key,
                             const h3_gpu_tensor *value, uint32_t rows,
                             uint32_t heads, uint32_t head_dim, float scale,
                             int quantized, int *head_major_output) {
    int want_head_major_output = head_major_output && *head_major_output;
    if (head_major_output) *head_major_output = 0;
    id<MTLDevice> device = (__bridge id<MTLDevice>)h3_gpu_raw_device(gpu);
    if (!device) {
        h3_gpu_report_error(gpu, "ccv backend: no Metal device");
        return 0;
    }
    /* Read this before anything else touches it - same requirement as the
     * debug capture hook in h3_dit.c, and for the same reason (this
     * engine's default int8 QKV path leaves query/key/value head-major for
     * the immediately-following attention call to consume). */
    int head_major = h3_gpu_head_major_sdpa_inputs(gpu);

    void *raw_command_buffer = h3_gpu_raw_command_buffer(gpu);
    if (!raw_command_buffer) return 0; // error already set

    if (g_state.device_key != (__bridge void *)device) {
        if (g_state.context) ccv_nnc_deinit_mfa_context(g_state.context);
        g_state.context = ccv_nnc_init_mfa_context(
            (mtl_device_t *)(__bridge void *)device);
        g_state.device_key = (__bridge void *)device;
        g_state.q_f16 = g_state.k_f16 = g_state.v_f16 = g_state.out_f16 = nil;
        g_state.capacity_elements = 0;
    }
    if (!g_state.context ||
        !ccv_nnc_mfa_context_supported(g_state.context) ||
        !ccv_nnc_mfa_has_neural_accelerators(g_state.context)) {
        h3_gpu_report_error(gpu, "ccv backend: context unsupported or this "
                            "device lacks neural matrix accelerators");
        return 0;
    }

    /* Direct path: when Q/K/V arrive head-major ([heads, rows, dim] BF16,
     * the default int8 QKV path's layout) and the caller asked for
     * head-major output, hand them to ccv's int8 kernel as-is - BF16 I/O,
     * described as batch = heads with Hq = Hk = 1 - instead of casting and
     * reordering into the FP16 row-major scratch. */
    if (quantized && head_major && want_head_major_output) {
        ccv_nnc_mfa_attention_params_t params = {};
        params.data_type = MTL::DataTypeBFloat;
        params.R = rows; params.C = rows; params.Hq = 1; params.Hk = 1;
        params.D = head_dim; params.output_rows = rows;
        params.K_trans = 1; // same as the validated row-major path below
        params.alpha = scale;
        params.use_neural_accelerators = 1;
        params.use_quantized_attention = 1;
        params.batched = 1;
        params.batch_dims_q[0] = heads;
        auto batch = ccv_nnc_start_command_batch_from_command_buffer(
            (mtl_command_buffer_t *)raw_command_buffer, /*commit_on_finish=*/0);
        mtl_buffer_t *tensors[10] = { // 10 slots: see the note below
            (mtl_buffer_t *)h3_gpu_raw_buffer(query),
            (mtl_buffer_t *)h3_gpu_raw_buffer(key),
            (mtl_buffer_t *)h3_gpu_raw_buffer(value),
            (mtl_buffer_t *)h3_gpu_raw_buffer(output),
        };
        size_t offsets[10] = {};
        ccv_nnc_mfa_encode_attention(g_state.context, params, batch, tensors,
                                    offsets);
        ccv_nnc_finish_command_batch(batch);
        memlog_after_encode(device, rows, heads, head_dim, head_major, 1, 1);
        *head_major_output = 1;
        h3_gpu_set_head_major_sdpa_inputs(gpu, 0);
        h3_gpu_note_ccv_attention_dispatch(gpu, 1);
        return 1;
    }

    size_t count = (size_t)rows * heads * head_dim;
    if (!ensure_scratch(device, count)) {
        h3_gpu_report_error(gpu, "ccv backend: scratch buffer allocation "
                            "failed");
        return 0;
    }

    if (!h3_gpu_ccv_cast_bf16_to_f16(gpu, (__bridge void *)g_state.q_f16,
                                    query, rows, heads, head_dim,
                                    head_major) ||
        !h3_gpu_ccv_cast_bf16_to_f16(gpu, (__bridge void *)g_state.k_f16,
                                    key, rows, heads, head_dim,
                                    head_major) ||
        !h3_gpu_ccv_cast_bf16_to_f16(gpu, (__bridge void *)g_state.v_f16,
                                    value, rows, heads, head_dim,
                                    head_major))
        return 0; // h3_gpu_ccv_cast_bf16_to_f16 already set the error

    ccv_nnc_mfa_attention_params_t params = {};
    params.data_type = MTL::DataTypeHalf;
    params.R = rows; params.C = rows; params.Hq = heads; params.Hk = heads;
    params.D = head_dim; params.output_rows = rows;
    /* K_trans=1 matches the exact configuration validated against this
     * engine's real production SDPA output in tools/ccv_eval/ (52-62 dB
     * PSNR across three real blocks) - do not change without re-running
     * that validation, since this parameter's effect on the quantized NAX
     * kernel path specifically was not independently confirmed (flipping
     * it made no measurable difference in that same validation, which is
     * itself worth re-checking before relying on it either way). */
    params.K_trans = 1;
    params.alpha = scale;
    params.use_neural_accelerators = 1;
    params.use_quantized_attention = quantized ? 1 : 0;
    /* FP32 softmax/accumulator intermediates instead of FP16. Does not
     * un-quantize the int8 kernel's P/V, so only meaningful with
     * quantized=0. */
    const char *upcast = getenv("H3_CCV_UPCAST");
    params.upcast = upcast && *upcast == '1';

    auto command_buffer = (mtl_command_buffer_t *)raw_command_buffer;
    auto batch = ccv_nnc_start_command_batch_from_command_buffer(
        command_buffer, /*commit_on_finish=*/0);
    /* ccv_nnc_mfa_encode_attention reads a fixed 10-slot tensor/offset
     * array internally (matching every call site in ccv's own bin/mfa
     * tools) - only the first 4 slots (Q, K, V, O) are meaningful for this
     * plain dense/no-mask/no-bias configuration, but passing a
     * shorter array is undefined behavior (a real SIGBUS was hit and
     * root-caused here before this comment was added: a 4-element array
     * reads past its end as soon as ccv's code touches slot 4+). */
    mtl_buffer_t *tensors[10] = {
        (mtl_buffer_t *)(__bridge void *)g_state.q_f16,
        (mtl_buffer_t *)(__bridge void *)g_state.k_f16,
        (mtl_buffer_t *)(__bridge void *)g_state.v_f16,
        (mtl_buffer_t *)(__bridge void *)g_state.out_f16,
    };
    size_t offsets[10] = {};
    ccv_nnc_mfa_encode_attention(g_state.context, params, batch, tensors,
                                offsets);
    ccv_nnc_finish_command_batch(batch);

    memlog_after_encode(device, rows, heads, head_dim, head_major, quantized,
                        0);

    if (!h3_gpu_ccv_cast_f16_to_bf16(gpu, output,
                                    (__bridge void *)g_state.out_f16,
                                    (uint32_t)count))
        return 0;

    h3_gpu_set_head_major_sdpa_inputs(gpu, 0);
    h3_gpu_note_ccv_attention_dispatch(gpu, 0);
    return 1;
}

void ccv_release_scratch(h3_gpu *gpu) {
    if (!g_state.q_f16 && !g_state.context) return;
    id<MTLDevice> device = (__bridge id<MTLDevice>)h3_gpu_raw_device(gpu);
    size_t before = device ? [device currentAllocatedSize] : 0;
    g_state.q_f16 = g_state.k_f16 = g_state.v_f16 = g_state.out_f16 = nil;
    g_state.capacity_elements = 0;
    if (g_state.context) {
        /* request_scratch() dereferences the current buffer, so shrink it
         * back to the 64 KiB ccv's context starts with instead of dropping
         * it; the context and its compiled kernels stay. */
        auto *context = (ccv::nnc::mfa::context *)g_state.context;
        if (context->scratch && context->scratch->length() > 65536) {
            auto *small = context->device->newBuffer(65536, MTL::ResourceStorageModePrivate);
            if (small) context->scratch = NS::TransferPtr(small);
        }
    }
    if (memlog_enabled() && device)
        fprintf(stderr, "[ccv-mem] released scratch after DiT: device "
                "allocated %.3f -> %.3f GiB\n", gib(before),
                gib([device currentAllocatedSize]));
}

} // namespace

/* Not a constructor: a static archive only links in a member that resolves
 * some other object's undefined symbol, so a self-registering constructor
 * with nothing else in this translation unit referenced would be invisible
 * to the linker and silently dropped. Instead, this file is kept OUT of
 * libh3.a entirely and linked as a loose object file whenever CCV_DIR is
 * set (see Makefile, Package.swift) - a loose .o on the link line is
 * always included, so this runs unconditionally in that build. */
__attribute__((constructor))
static void h3_gpu_ccv_attention_register(void) {
    h3_gpu_register_ccv_dense_attention(&ccv_dense_attention_bf16);
    h3_gpu_register_ccv_release_scratch(&ccv_release_scratch);
    h3_gpu_register_ccv_attention_supported(&ccv_attention_supported);
}
