import Foundation
import Metal
import MetalPerformanceShadersGraph

/// Real-ESRGAN's SRVGGNetCompact (realesr-general-x4v3, realesr-animevideov3)
/// on MPSGraph, for one frame size: BGRA 8-bit in, the frame enlarged 4x out
/// (`scale` 4), or 2x (`scale` 2: each 2x2 block of the 4x averaged; Real-
/// ESRGAN does a smaller outscale by resizing its 4x).
///
/// The network: a 3x3 convolution to 64 features, num_conv more 3x3
/// convolutions, each followed by a per-channel PReLU, a last 3x3
/// convolution to 3 * 4 * 4 channels, PixelShuffle(4), plus the input
/// enlarged 4x by nearest neighbour. Weights: F16 safetensors made by
/// tools/convert_realesrgan.py (body.<n>.weight / .bias, the PReLU slopes
/// at the odd n). Real-ESRGAN: BSD-3-Clause, Copyright (c) 2021, Xintao Wang.
final class RealESRGANNetwork {
    enum NetworkError: LocalizedError {
        case unreadable(String)
        case noDevice
        case failed(String)
        var errorDescription: String? {
            switch self {
            case .unreadable(let detail): return String(localized: "AI超解像のモデルを読めませんでした（\(detail)）")
            case .noDevice: return String(localized: "GPUを使えません")
            case .failed(let detail): return String(localized: "AI超解像を実行できませんでした（\(detail)）")
            }
        }
    }

    let width: Int
    let height: Int
    let outputWidth: Int
    let outputHeight: Int
    let scale: Int
    /// The size the graph runs at: the frame, or a tile of it - a core of
    /// `tile` pixels with `context` more on each side, so each core sees
    /// everything its output depends on (each 3x3 convolution reaches one
    /// pixel further) and the tiles give the frame's own result.
    private let graphWidth: Int
    private let graphHeight: Int
    private let context: Int

    /// The largest core of a tile, in input pixels (H3_ESRGAN_TILE sets it,
    /// for measuring; 0 runs whole frames). An axis no longer than a core
    /// and its context runs whole; a longer one is cut into equal cores.
    /// 1024 keeps the usual sizes whole or in two (1620x1080: two windows,
    /// about 9% more work) and bounds the largest frames' memory.
    static var tile: Int {
        if let value = ProcessInfo.processInfo.environment["H3_ESRGAN_TILE"], let tile = Int(value), tile >= 0 {
            return tile
        }
        return 1024
    }

    /// The graph's length along an axis of `length`: the whole axis, or an
    /// equal share of it with `context` on each side.
    private static func window(length: Int, tile: Int, context: Int) -> Int {
        guard tile > 0, length > tile + 2 * context else { return length }
        let pieces = (length + tile - 1) / tile
        return min((length + pieces - 1) / pieces + 2 * context, length)
    }
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let graph: MPSGraph
    private let input: MPSGraphTensor
    private let output: MPSGraphTensor
    private let inputBuffer: MTLBuffer
    private let outputBuffer: MTLBuffer

    /// `weights`: the safetensors file; `width` x `height`: the frames;
    /// `scale`: 2 or 4.
    init(weights: URL, width: Int, height: Int, scale: Int = 2) throws {
        guard scale == 2 || scale == 4 else { throw NetworkError.failed("scale \(scale)") }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw NetworkError.noDevice
        }
        self.device = device
        self.queue = queue
        self.width = width
        self.height = height
        self.scale = scale
        outputWidth = width * scale
        outputHeight = height * scale
        let tensors = try Self.readSafetensors(weights)
        // One pixel of reach per convolution, and a little more.
        context = tensors.values.filter { $0.1.count == 4 }.count + 2
        let tile = Self.tile
        graphWidth = Self.window(length: width, tile: tile, context: context)
        graphHeight = Self.window(length: height, tile: tile, context: context)
        let width = graphWidth, height = graphHeight
        let graph = MPSGraph()

        let input = graph.placeholder(shape: [1, height, width, 4].map { NSNumber(value: $0) },
                                  dataType: .uInt8, name: nil)
        // BGRA bytes to RGB in 0-1, F16.
        let scaled = graph.division(graph.cast(input, to: .float16, name: nil),
                                    graph.constant(255, dataType: .float16), name: nil)
        let channels = graph.split(scaled, numSplits: 4, axis: 3, name: nil)
        let rgb = graph.concatTensors([channels[2], channels[1], channels[0]], dimension: 3, name: nil)

        let convolution = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1, groups: 1,
            paddingLeft: 1, paddingRight: 1, paddingTop: 1, paddingBottom: 1,
            paddingStyle: .explicit, dataLayout: .NHWC, weightsLayout: .OIHW)!
        func constant(_ name: String) throws -> MPSGraphTensor {
            guard let (data, shape) = tensors[name] else { throw NetworkError.unreadable(name) }
            return graph.constant(data, shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        }
        var index = 0
        var features = rgb
        while tensors["body.\(index).weight"] != nil {
            let weight = try constant("body.\(index).weight")
            let bias = try constant("body.\(index).bias")
            features = graph.convolution2D(features, weights: weight, descriptor: convolution, name: nil)
            features = graph.addition(features, graph.reshape(bias, shape: [1, 1, 1, -1], name: nil), name: nil)
            index += 1
            if let (_, shape) = tensors["body.\(index).weight"], shape.count == 1 {
                // PReLU: x where positive, slope * x elsewhere, a slope per channel.
                let slope = graph.reshape(try constant("body.\(index).weight"), shape: [1, 1, 1, -1], name: nil)
                let zero = graph.constant(0, dataType: .float16)
                features = graph.addition(graph.maximum(features, zero, name: nil),
                                          graph.multiplication(slope, graph.minimum(features, zero, name: nil), name: nil),
                                          name: nil)
                index += 1
            }
        }
        guard index > 2 else { throw NetworkError.unreadable("body") }
        let shuffled = graph.depth(toSpace2DTensor: features, widthAxis: 2, heightAxis: 1, depthAxis: 3,
                                            blockSize: 4, usePixelShuffleOrder: true, name: nil)
        let base = graph.resize(rgb, size: [NSNumber(value: height * 4), NSNumber(value: width * 4)],
                                mode: .nearest, centerResult: true, alignCorners: false, layout: .NHWC, name: nil)
        let enlarged = graph.clamp(graph.addition(shuffled, base, name: nil),
                                   min: graph.constant(0, dataType: .float16),
                                   max: graph.constant(1, dataType: .float16), name: nil)
        // 4x to 2x: the mean of each 2x2 block.
        let pooling = MPSGraphPooling2DOpDescriptor(kernelWidth: 2, kernelHeight: 2, strideInX: 2, strideInY: 2,
                                                    paddingStyle: .explicit, dataLayout: .NHWC)!
        let halved = scale == 2 ? graph.avgPooling2D(withSourceTensor: enlarged, descriptor: pooling, name: nil)
                                : enlarged
        // Back to BGRA bytes, alpha opaque.
        let bytes = graph.round(with: graph.multiplication(graph.cast(halved, to: .float32, name: nil),
                                                           graph.constant(255, dataType: .float32), name: nil), name: nil)
        let parts = graph.split(bytes, numSplits: 3, axis: 3, name: nil)
        let alpha = graph.constant(255, shape: [1, NSNumber(value: height * scale), NSNumber(value: width * scale), 1],
                                   dataType: .float32)
        let output = graph.cast(graph.concatTensors([parts[2], parts[1], parts[0], alpha], dimension: 3, name: nil),
                                to: .uInt8, name: nil)

        guard let inputBuffer = device.makeBuffer(length: width * height * 4, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: width * height * 4 * scale * scale,
                                                   options: .storageModeShared) else {
            throw NetworkError.noDevice
        }
        self.inputBuffer = inputBuffer
        self.outputBuffer = outputBuffer
        self.graph = graph
        self.input = input
        self.output = output
    }

    /// Where the graph's window sits along one axis of `length` for each
    /// core: (core start, core end, window start), the window kept inside
    /// the frame.
    private func windows(length: Int, window: Int) -> [(Int, Int, Int)] {
        guard window < length else { return [(0, length, 0)] }
        let core = window - 2 * context
        return stride(from: 0, to: length, by: core).map { start in
            (start, min(start + core, length), min(max(start - context, 0), length - window))
        }
    }

    /// One BGRA frame (`width` x `height`) to its enlargement (rows of any
    /// stride), tile by tile when the graph is smaller than the frame.
    func upscale(_ source: UnsafeRawPointer, sourceRowBytes: Int,
                 into destination: UnsafeMutableRawPointer, destinationRowBytes: Int) throws {
        let row = graphWidth * 4
        let graphOutputWidth = graphWidth * scale
        let feed = MPSGraphTensorData(inputBuffer, shape: [1, graphHeight, graphWidth, 4].map { NSNumber(value: $0) },
                                      dataType: .uInt8)
        let result = MPSGraphTensorData(outputBuffer,
                                        shape: [1, graphHeight * scale, graphOutputWidth, 4].map { NSNumber(value: $0) },
                                        dataType: .uInt8)
        for (top, bottom, windowY) in windows(length: height, window: graphHeight) {
            for (left, right, windowX) in windows(length: width, window: graphWidth) {
                for y in 0..<graphHeight {
                    inputBuffer.contents().advanced(by: y * row).copyMemory(
                        from: source.advanced(by: (windowY + y) * sourceRowBytes + windowX * 4), byteCount: row)
                }
                graph.run(with: queue, feeds: [input: feed], targetOperations: nil, resultsDictionary: [output: result])
                // The core's part of the window's output.
                let bytes = (right - left) * scale * 4
                for y in (top * scale)..<(bottom * scale) {
                    destination.advanced(by: y * destinationRowBytes + left * scale * 4).copyMemory(
                        from: outputBuffer.contents().advanced(by: ((y - windowY * scale) * graphOutputWidth
                                                                   + (left - windowX) * scale) * 4),
                        byteCount: bytes)
                }
            }
        }
    }

    /// name → (F16 bytes, shape), from a safetensors file of F16 tensors.
    private static func readSafetensors(_ url: URL) throws -> [String: (Data, [Int])] {
        guard let file = try? Data(contentsOf: url, options: .mappedIfSafe), file.count > 8 else {
            throw NetworkError.unreadable(url.lastPathComponent)
        }
        let headerLength = Int(file.prefix(8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian })
        guard headerLength > 0, 8 + headerLength <= file.count,
              let header = try? JSONSerialization.jsonObject(with: file.subdata(in: 8..<(8 + headerLength))) as? [String: Any] else {
            throw NetworkError.unreadable(url.lastPathComponent)
        }
        let base = 8 + headerLength
        var tensors: [String: (Data, [Int])] = [:]
        for (name, value) in header where name != "__metadata__" {
            guard let entry = value as? [String: Any], entry["dtype"] as? String == "F16",
                  let shape = entry["shape"] as? [Int], let offsets = entry["data_offsets"] as? [Int],
                  offsets.count == 2, base + offsets[1] <= file.count else {
                throw NetworkError.unreadable(name)
            }
            tensors[name] = (file.subdata(in: (base + offsets[0])..<(base + offsets[1])), shape)
        }
        return tensors
    }
}
