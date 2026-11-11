/* The Metal layer: convolutions and the VAE and audio kernels built on them. Split out of h3_gpu.m. */
#include "h3_gpu_private.h"
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
