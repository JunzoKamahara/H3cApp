/* The Metal layer: attention. Dense scaled dot-product attention (one MPSGraph call, chunked or explicit), the ccv backend's hooks, and Qwen's grouped-query causal attention. Split out of h3_gpu.m. */
#include "h3_gpu_private.h"
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
    const char *env = h3_option("H3_SDPA_MODE");
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
    const char *env = h3_option("H3_SDPA_SCORE_BYTES");
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
    if (!h3_option("H3_SDPA_LOG")) return;
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
    const char *group_env = h3_option("H3_GQA_GROUP");
    int grouped = groups == 8 && !(group_env && !strcmp(group_env, "0"));
    id<MTLComputePipelineState> pipeline = grouped ?
        h3_gpu_pipeline(gpu, @"h3_gqa_causal_group_bf16") : nil;
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < threads) {
        grouped = 0;
        pipeline = h3_gpu_pipeline(gpu, @"h3_gqa_causal_rows_bf16");
    }
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < threads)
        return 0;
    const char *env = h3_option("H3_GQA_ROWS");
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
    if (h3_option("H3_MPS_GQA") && h3_gpu_gqa_mps(
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
