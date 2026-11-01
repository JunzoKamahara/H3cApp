import CH3
import CoreVideo
import Foundation

// SeedVR2-3B through the engine (h3_seedvr2.h): the one-step image restorer
// run frame by frame, the app's slow high-detail enlarger. The weights are
// read straight from the downloaded safetensors (no cache). UPSCALE_PLAN
// appendix A.7.

public struct H3SeedVR2Info: Sendable {
    public let name: String
    public let variant: String
    public let dtype: String
    public let fileBytes: UInt64
    public let residentBytes: UInt64
    public let tensorOps: Bool
}

public struct H3SeedVR2Requirements: Sendable {
    public let supported: Bool
    public let reason: String
    public let gpuBytes: UInt64
    public let estimatedSecondsPerFrame: Double
    public let tokens: Int
}

private func cString<T>(_ tuple: T) -> String {
    withUnsafeBytes(of: tuple) { raw in
        String(cString: raw.bindMemory(to: CChar.self).baseAddress!)
    }
}

private final class SV2CancelFlag: @unchecked Sendable {
    let engineFlag: UnsafeMutablePointer<Int32> = {
        let flag = UnsafeMutablePointer<Int32>.allocate(capacity: 1)
        flag.initialize(to: 0)
        return flag
    }()
    var isCancelled: Bool { engineFlag.pointee != 0 }
    func cancel() { engineFlag.pointee = 1 }
    deinit {
        engineFlag.deinitialize(count: 1)
        engineFlag.deallocate()
    }
}

/// One frame's progress: the engine's phases ("vsr encode", "vsr dit" per
/// block, "vsr decode") as a share of the frame.
private final class SV2ProcessBridge: @unchecked Sendable {
    let progress: (Double) -> Void
    let cancelFlag: SV2CancelFlag
    init(progress: @escaping (Double) -> Void, cancelFlag: SV2CancelFlag) {
        self.progress = progress
        self.cancelFlag = cancelFlag
    }
}

private let h3SeedVR2ProcessTrampoline: h3_progress_callback = { phase, completed, total, opaque in
    guard let opaque else { return 0 }
    let bridge = Unmanaged<SV2ProcessBridge>.fromOpaque(opaque).takeUnretainedValue()
    if let phase {
        let share: Double?
        switch String(cString: phase) {
        case "vsr encode": share = completed > 0 ? 0.2 : 0
        case "vsr dit": share = 0.2 + 0.4 * Double(completed) / Double(max(total, 1))
        case "vsr decode": share = completed > 0 ? 1 : 0.6
        default: share = nil
        }
        if let share { bridge.progress(share) }
    }
    return bridge.cancelFlag.isCancelled ? 1 : 0
}

public final class H3SeedVR2Restorer: @unchecked Sendable {
    private let handle: OpaquePointer
    public let info: H3SeedVR2Info

    /// What a model directory holds, without loading it (config.json and
    /// the safetensors headers).
    public static func inspect(modelDirectory: String) throws -> H3SeedVR2Info {
        var info = h3_seedvr2_info()
        var error = [CChar](repeating: 0, count: 512)
        let ok = modelDirectory.withCString { dir in
            error.withUnsafeMutableBufferPointer { buffer in
                h3_seedvr2_inspect_dir(dir, &info, buffer.baseAddress, buffer.count)
            }
        }
        guard ok != 0 else { throw H3VSRError.failed(String(cString: error)) }
        return H3SeedVR2Info(name: cString(info.name), variant: cString(info.variant), dtype: cString(info.dtype),
                             fileBytes: info.file_bytes, residentBytes: info.resident_bytes, tensorOps: info.tensor_ops != 0)
    }

    /// Loads the model onto the GPU (about 7 GB resident; a few seconds).
    /// `colorTransfer` matches each frame's colours to its input with the
    /// reference's wavelet + L*a*b* transfer; `seed` fixes the noise (the
    /// same for every frame of a session).
    public init(modelDirectory: String, colorTransfer: Bool = true, seed: UInt64 = 42,
                shaderPath: String = "h3_shaders.metal") throws {
        info = try Self.inspect(modelDirectory: modelDirectory)
        var error = [CChar](repeating: 0, count: 1024)
        var params = h3_seedvr2_open_params()
        params.color_transfer = colorTransfer ? 1 : 0
        params.seed = seed
        params.nax_linear = -1
        let handle = modelDirectory.withCString { dir in
            shaderPath.withCString { shader -> OpaquePointer? in
                params.shader_source_path = shader
                return error.withUnsafeMutableBufferPointer { buffer in
                    h3_seedvr2_open(dir, &params, nil, nil, buffer.baseAddress, buffer.count)
                }
            }
        }
        guard let handle else { throw H3VSRError.failed(String(cString: error)) }
        self.handle = handle
    }

    deinit { h3_seedvr2_close(handle) }

    public func requirements(inputWidth: Int, inputHeight: Int, outputWidth: Int, outputHeight: Int) -> H3SeedVR2Requirements {
        var req = h3_seedvr2_requirements()
        _ = h3_seedvr2_check_requirements(handle, Int32(inputWidth), Int32(inputHeight), Int32(outputWidth), Int32(outputHeight), &req)
        return H3SeedVR2Requirements(supported: req.supported != 0, reason: cString(req.reason), gpuBytes: req.gpu_bytes,
                                     estimatedSecondsPerFrame: req.estimated_seconds_per_frame, tokens: Int(req.tokens))
    }

    /// GPU seconds spent so far and the peak live bytes.
    public var stats: (seconds: Double, peakBytes: UInt64) {
        var seconds = 0.0
        var peak: UInt64 = 0
        _ = h3_seedvr2_get_stats(handle, &seconds, &peak)
        return (seconds, peak)
    }

    /// One video: a fixed input and output size. Frames are 32BGRA
    /// CVPixelBuffers, one in and one out per call.
    public final class Session {
        private let handle: OpaquePointer
        private let restorer: H3SeedVR2Restorer   // keeps the model alive
        private let cancelFlag = SV2CancelFlag()
        public let outputWidth: Int
        public let outputHeight: Int

        fileprivate init(restorer: H3SeedVR2Restorer, handle: OpaquePointer, outputWidth: Int, outputHeight: Int) {
            self.restorer = restorer
            self.handle = handle
            self.outputWidth = outputWidth
            self.outputHeight = outputHeight
        }

        deinit { h3_seedvr2_session_close(handle) }

        /// Asks the engine to stop at the next block.
        public func cancel() { cancelFlag.cancel() }

        /// Restores one frame into a buffer from `pool`. `progress` gets the
        /// share of the frame done (0-1, from the engine's thread).
        public func process(_ frame: CVPixelBuffer, pool: CVPixelBufferPool,
                            progress: ((Double) -> Void)? = nil) throws -> CVPixelBuffer {
            let bridge = progress.map { SV2ProcessBridge(progress: $0, cancelFlag: cancelFlag) }
            let bridgeHandle = bridge.map { Unmanaged.passRetained($0) }
            defer { bridgeHandle?.release() }
            var output: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &output)
            guard let output else { throw H3VSRError.failed(String(localized: "出力フレームを確保できませんでした")) }
            CVPixelBufferLockBaseAddress(frame, .readOnly)
            CVPixelBufferLockBaseAddress(output, [])
            defer {
                CVPixelBufferUnlockBaseAddress(frame, .readOnly)
                CVPixelBufferUnlockBaseAddress(output, [])
            }
            var input = h3_vsr_image(pixels: CVPixelBufferGetBaseAddress(frame)!.assumingMemoryBound(to: UInt8.self),
                                     width: Int32(CVPixelBufferGetWidth(frame)), height: Int32(CVPixelBufferGetHeight(frame)),
                                     stride_bytes: Int32(CVPixelBufferGetBytesPerRow(frame)), format: H3_VSR_BGRA8)
            var out = h3_vsr_image_out(pixels: CVPixelBufferGetBaseAddress(output)!.assumingMemoryBound(to: UInt8.self),
                                       width: Int32(CVPixelBufferGetWidth(output)), height: Int32(CVPixelBufferGetHeight(output)),
                                       stride_bytes: Int32(CVPixelBufferGetBytesPerRow(output)), format: H3_VSR_BGRA8)
            var error = [CChar](repeating: 0, count: 1024)
            let ok = error.withUnsafeMutableBufferPointer { buffer in
                h3_seedvr2_session_process(handle, &input, &out,
                                           bridgeHandle == nil ? nil : h3SeedVR2ProcessTrampoline,
                                           bridgeHandle?.toOpaque(), cancelFlag.engineFlag,
                                           buffer.baseAddress, buffer.count)
            }
            guard ok != 0 else {
                throw cancelFlag.isCancelled ? H3VSRError.cancelled : H3VSRError.failed(String(cString: error))
            }
            return output
        }
    }

    public func openSession(inputWidth: Int, inputHeight: Int, outputWidth: Int, outputHeight: Int) throws -> Session {
        var error = [CChar](repeating: 0, count: 1024)
        let session = error.withUnsafeMutableBufferPointer { buffer in
            h3_seedvr2_session_open(handle, Int32(inputWidth), Int32(inputHeight), Int32(outputWidth), Int32(outputHeight),
                                    buffer.baseAddress, buffer.count)
        }
        guard let session else { throw H3VSRError.failed(String(cString: error)) }
        return Session(restorer: self, handle: session, outputWidth: outputWidth, outputHeight: outputHeight)
    }
}
