import CoreGraphics
import Foundation
import Metal
import MetalPerformanceShadersGraph

// The preview shown while a video is still denoising: madebyollin's TAEHV
// decoder with its MiniMax H3 weights (taeh3.safetensors, MIT, ~10M
// parameters), run with MPSGraph on the clean-latent estimate the engine
// hands over (H3LatentPreview). It turns a few latent time steps into RGB
// frames at the video's full size; no VAE runs during the DiT. Same
// network as taehv.py / ComfyUI's comfy/taesd/taehv.py: Clamp, conv, three
// MemBlocks per stage (each also sees the previous time step's input),
// nearest 2x upsampling, TGrow (one 1x1 conv making `stride` frames), and a
// final conv whose 12 channels are pixel-shuffled 2x.
//
// Kept free of app state so it can be checked on its own against the
// reference decoder.

enum TinyPreviewDecoderError: LocalizedError {
    case unreadable(String)
    case missingTensor(String)
    case noGPU

    var errorDescription: String? {
        switch self {
        case .unreadable(let detail): return String(localized: "プレビュー用のデコーダを読み込めません（\(detail)）")
        case .missingTensor(let name): return String(localized: "プレビュー用のデコーダに \(name) がありません")
        case .noGPU: return String(localized: "GPUを使えません")
        }
    }
}

final class TinyPreviewDecoder: @unchecked Sendable {
    /// Latent time steps decoded together: the MemBlocks need the steps
    /// before the one shown (three match a whole-video decode at ~47 dB).
    static let window = 3

    private struct Tensor {
        let shape: [Int]
        let data: Data   // F16
    }

    private let tensors: [String: Tensor]
    private let patch: Int
    private let device: MTLDevice
    private let graphDevice: MPSGraphDevice
    private let queue: MTLCommandQueue
    private let lock = NSLock()
    private var graphs: [String: MPSGraphExecutable] = [:]

    init(path: String) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw TinyPreviewDecoderError.noGPU }
        self.device = device
        graphDevice = MPSGraphDevice(mtlDevice: device)
        guard let queue = device.makeCommandQueue() else { throw TinyPreviewDecoderError.noGPU }
        self.queue = queue
        guard let file = FileHandle(forReadingAtPath: path) else {
            throw TinyPreviewDecoderError.unreadable(path)
        }
        defer { try? file.close() }
        let data = try file.readToEnd() ?? Data()
        guard data.count > 8 else { throw TinyPreviewDecoderError.unreadable("too short") }
        let headerBytes = data.prefix(8).withUnsafeBytes { Int($0.loadUnaligned(as: UInt64.self).littleEndian) }
        guard headerBytes > 0, 8 + headerBytes <= data.count,
              let header = try JSONSerialization.jsonObject(with: data.subdata(in: 8 ..< 8 + headerBytes)) as? [String: Any] else {
            throw TinyPreviewDecoderError.unreadable("header")
        }
        let base = 8 + headerBytes
        var tensors: [String: Tensor] = [:]
        for (name, value) in header where name.hasPrefix("decoder.") {
            guard let entry = value as? [String: Any], let dtype = entry["dtype"] as? String,
                  let shape = entry["shape"] as? [Int], let offsets = entry["data_offsets"] as? [Int],
                  offsets.count == 2, base + offsets[1] <= data.count else {
                throw TinyPreviewDecoderError.unreadable(name)
            }
            let raw = data.subdata(in: base + offsets[0] ..< base + offsets[1])
            switch dtype {
            case "F16":
                tensors[name] = Tensor(shape: shape, data: raw)
            case "F32":
                var halves = [Float16](repeating: 0, count: raw.count / 4)
                raw.withUnsafeBytes { bytes in
                    for i in halves.indices { halves[i] = Float16(bytes.loadUnaligned(fromByteOffset: i * 4, as: Float.self)) }
                }
                tensors[name] = Tensor(shape: shape, data: halves.withUnsafeBufferPointer { Data(buffer: $0) })
            default:
                throw TinyPreviewDecoderError.unreadable("\(name): \(dtype)")
            }
        }
        guard let last = tensors["decoder.22.weight"], last.shape.count == 4,
              tensors["decoder.1.weight"]?.shape == [256, 24, 3, 3] else {
            throw TinyPreviewDecoderError.missingTensor("decoder.1/22.weight")
        }
        self.tensors = tensors
        // 12 output channels: pixel shuffle 2 (full size); 3: half size.
        patch = last.shape[0] == 12 ? 2 : 1
    }

    /// Output frames per latent time step decoded.
    static let framesPerStep = 4

    /// Decodes `window` latent time steps ending at `lastStep` of a preview
    /// and returns the last frame of that step as an image.
    func image(from preview: H3LatentPreviewInput, lastStep: Int) throws -> CGImage? {
        let steps = min(Self.window, lastStep + 1)
        let first = lastStep - steps + 1
        let (h, w) = (preview.height, preview.width)
        // [24, T, H, W] -> [steps, 24, H, W]
        var input = [Float](repeating: 0, count: steps * 24 * h * w)
        let plane = h * w
        for t in 0 ..< steps {
            for c in 0 ..< 24 {
                let source = (c * preview.time + first + t) * plane
                let target = (t * 24 + c) * plane
                input[target ..< target + plane] = preview.latent[source ..< source + plane]
            }
        }
        let executable = try graph(steps: steps, height: h, width: w)
        let feed = input.withUnsafeBufferPointer { buffer in
            MPSGraphTensorData(device: graphDevice, data: Data(buffer: buffer),
                               shape: [steps, 24, h, w].map { NSNumber(value: $0) }, dataType: .float32)
        }
        let results = executable.run(with: queue, inputs: [feed], results: nil, executionDescriptor: nil)
        guard let result = results.first else { return nil }
        let outH = h * 8 * patch, outW = w * 8 * patch
        var rgb = [Float](repeating: 0, count: 3 * outH * outW)
        rgb.withUnsafeMutableBytes { result.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
        return Self.cgImage(rgb, width: outW, height: outH)
    }

    // MARK: Graph

    /// One compiled graph per window length and latent size (compiling
    /// takes a few seconds, once); the output is the last frame only,
    /// [3, H*16, W*16] F32 in 0...1. Compiled for the GPU alone: the
    /// default placement also tries the Neural Engine, which rejects F32.
    private func graph(steps: Int, height: Int, width: Int) throws -> MPSGraphExecutable {
        let key = "\(steps)x\(height)x\(width)"
        lock.lock(); defer { lock.unlock() }
        if let cached = graphs[key] { return cached }
        let g = MPSGraph()
        let input = g.placeholder(shape: [steps, 24, height, width].map { NSNumber(value: $0) },
                                  dataType: .float32, name: nil)
        var x = g.cast(input, to: .float16, name: nil)
        var frames = steps
        // Clamp: tanh(x / 3) * 3
        let three = g.constant(3, dataType: .float16)
        x = g.multiplication(g.tanh(with: g.division(x, three, name: nil), name: nil), three, name: nil)
        x = g.reLU(with: try conv(g, x, "decoder.1", bias: true), name: nil)
        for (blocks, grow, next) in [([3, 4, 5], 7, 8), ([9, 10, 11], 13, 14), ([15, 16, 17], 19, 20)] {
            for block in blocks { x = try memBlock(g, x, frames: frames, "decoder.\(block)") }
            x = upsample(g, x)
            let stride = try tensor("decoder.\(grow).conv.weight").shape[0] / channels(x)
            x = try conv(g, x, "decoder.\(grow).conv", bias: false)
            if stride > 1 {
                let s = x.shape!.map(\.intValue)
                x = g.reshape(x, shape: [s[0] * stride, s[1] / stride, s[2], s[3]].map { NSNumber(value: $0) }, name: nil)
                frames *= stride
            }
            x = try conv(g, x, "decoder.\(next)", bias: false)
        }
        x = g.reLU(with: x, name: nil)
        x = try conv(g, x, "decoder.22", bias: true)
        // Only the last frame is shown.
        x = g.sliceTensor(x, dimension: 0, start: frames - 1, length: 1, name: nil)
        if patch == 2 {
            let s = x.shape!.map(\.intValue)  // [1, 12, H, W]
            x = g.reshape(x, shape: [3, 2, 2, s[2], s[3]].map { NSNumber(value: $0) }, name: nil)
            x = g.transpose(x, permutation: [0, 3, 1, 4, 2], name: nil)  // [3, H, 2, W, 2]
            x = g.reshape(x, shape: [3, s[2] * 2, s[3] * 2].map { NSNumber(value: $0) }, name: nil)
        } else {
            // The 3-channel decoders output -1...1.
            let one = g.constant(1, dataType: .float16), half = g.constant(0.5, dataType: .float16)
            x = g.multiplication(g.addition(x, one, name: nil), half, name: nil)
            let s = x.shape!.map(\.intValue)
            x = g.reshape(x, shape: [3, s[2], s[3]].map { NSNumber(value: $0) }, name: nil)
        }
        x = g.clamp(x, min: g.constant(0, dataType: .float16), max: g.constant(1, dataType: .float16), name: nil)
        let output = g.cast(x, to: .float32, name: nil)
        let descriptor = MPSGraphCompilationDescriptor()
        descriptor.optimizationLevel = .level0
        let executable = g.compile(
            with: graphDevice,
            feeds: [input: MPSGraphShapedType(shape: input.shape, dataType: .float32)],
            targetTensors: [output], targetOperations: nil, compilationDescriptor: descriptor)
        graphs[key] = executable
        return executable
    }

    private func tensor(_ name: String) throws -> Tensor {
        guard let t = tensors[name] else { throw TinyPreviewDecoderError.missingTensor(name) }
        return t
    }

    private func channels(_ x: MPSGraphTensor) -> Int { x.shape![1].intValue }

    private func conv(_ g: MPSGraph, _ x: MPSGraphTensor, _ name: String, bias: Bool) throws -> MPSGraphTensor {
        let weight = try tensor(name + ".weight")
        let k = weight.shape[2]
        let w = g.constant(weight.data, shape: weight.shape.map { NSNumber(value: $0) }, dataType: .float16)
        let pad = k / 2
        let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1, groups: 1,
            paddingLeft: pad, paddingRight: pad, paddingTop: pad, paddingBottom: pad,
            paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW)!
        var y = g.convolution2D(x, weights: w, descriptor: descriptor, name: nil)
        if bias {
            let b = try tensor(name + ".bias")
            let bt = g.constant(b.data, shape: [1, b.shape[0], 1, 1].map { NSNumber(value: $0) }, dataType: .float16)
            y = g.addition(y, bt, name: nil)
        }
        return y
    }

    /// relu(conv(relu(conv(relu(conv([x, past]))))) + x), past = the
    /// previous time step's x (zeros for the first).
    private func memBlock(_ g: MPSGraph, _ x: MPSGraphTensor, frames: Int, _ name: String) throws -> MPSGraphTensor {
        let s = x.shape!.map(\.intValue)
        let zero = g.constant(0, shape: [1, s[1], s[2], s[3]].map { NSNumber(value: $0) }, dataType: .float16)
        let past = frames > 1
            ? g.concatTensors([zero, g.sliceTensor(x, dimension: 0, start: 0, length: frames - 1, name: nil)],
                              dimension: 0, name: nil)
            : zero
        var h = g.concatTensors([x, past], dimension: 1, name: nil)
        h = g.reLU(with: try conv(g, h, name + ".conv.0", bias: true), name: nil)
        h = g.reLU(with: try conv(g, h, name + ".conv.2", bias: true), name: nil)
        h = try conv(g, h, name + ".conv.4", bias: true)
        return g.reLU(with: g.addition(h, x, name: nil), name: nil)
    }

    /// Nearest-neighbour 2x, as broadcasting (exact, unlike resize).
    private func upsample(_ g: MPSGraph, _ x: MPSGraphTensor) -> MPSGraphTensor {
        let s = x.shape!.map(\.intValue)
        let a = g.reshape(x, shape: [s[0], s[1], s[2], 1, s[3], 1].map { NSNumber(value: $0) }, name: nil)
        let b = g.broadcast(a, shape: [s[0], s[1], s[2], 2, s[3], 2].map { NSNumber(value: $0) }, name: nil)
        return g.reshape(b, shape: [s[0], s[1], s[2] * 2, s[3] * 2].map { NSNumber(value: $0) }, name: nil)
    }

    private static func cgImage(_ rgb: [Float], width: Int, height: Int) -> CGImage? {
        let plane = width * height
        var pixels = [UInt8](repeating: 255, count: plane * 4)
        for i in 0 ..< plane {
            pixels[i * 4] = UInt8(max(0, min(255, rgb[i] * 255 + 0.5)))
            pixels[i * 4 + 1] = UInt8(max(0, min(255, rgb[plane + i] * 255 + 0.5)))
            pixels[i * 4 + 2] = UInt8(max(0, min(255, rgb[2 * plane + i] * 255 + 0.5)))
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}

/// What the decoder needs from a latent preview (H3LatentPreview, or a saved
/// .h3lt latent for checking).
struct H3LatentPreviewInput {
    let time: Int
    let height: Int
    let width: Int
    let latent: [Float]   // [24, time, height, width]
}
