import CH3
import CoreVideo
import Foundation

// The one-step video super-resolution engine (h3_vsr.h): a loaded model
// (SwiftVR first), its weight cache, and a streaming session that upscales
// a video chunk by chunk. UPSCALE_PLAN.md §6.2 / §7.

public struct H3VSRInfo: Sendable {
    public let name: String
    public let licenseId: String
    public let commercialUseAllowed: Bool
    public let firstChunkFrames: Int
    public let chunkFrames: Int
    public let int8Supported: Bool
    public let weightBytesBF16: UInt64
    public let weightBytesInt8: UInt64
    public let minOutputSide: Int
    public let maxOutputSide: Int
}

public enum H3VSRError: LocalizedError {
    case failed(String)
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .failed(let message): return message
        case .cancelled: return String(localized: "拡大を中止しました")
        }
    }
}

public struct H3VSRRequirements: Sendable {
    public let supported: Bool
    public let reason: String
    public let gpuBytes: UInt64
    public let estimatedSecondsPerChunk: Double
    public let tokensPerChunk: Int
}

public struct H3VSRCacheProgress: Sendable {
    public let completed: Int
    public let total: Int
}

private func cString<T>(_ tuple: T) -> String {
    withUnsafeBytes(of: tuple) { raw in
        String(cString: raw.bindMemory(to: CChar.self).baseAddress!)
    }
}

private final class VSRCancelFlag: @unchecked Sendable {
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

private final class VSRProgressBridge: @unchecked Sendable {
    let continuation: AsyncThrowingStream<H3VSRCacheProgress, Error>.Continuation
    let cancelFlag: VSRCancelFlag
    init(continuation: AsyncThrowingStream<H3VSRCacheProgress, Error>.Continuation, cancelFlag: VSRCancelFlag) {
        self.continuation = continuation
        self.cancelFlag = cancelFlag
    }
}

/// One chunk's progress for Session.process: the engine's phases
/// ("vsr encode", "vsr dit" per block, "vsr decode") as a share of the
/// chunk, and the cancel flag read back at each.
private final class VSRProcessBridge: @unchecked Sendable {
    let progress: (Double) -> Void
    let cancelFlag: VSRCancelFlag
    init(progress: @escaping (Double) -> Void, cancelFlag: VSRCancelFlag) {
        self.progress = progress
        self.cancelFlag = cancelFlag
    }
}

private let h3VSRProcessTrampoline: h3_progress_callback = { phase, completed, total, opaque in
    guard let opaque else { return 0 }
    let bridge = Unmanaged<VSRProcessBridge>.fromOpaque(opaque).takeUnretainedValue()
    if let phase {
        let share: Double?
        switch String(cString: phase) {
        case "vsr encode": share = completed > 0 ? 0.05 : 0
        case "vsr dit": share = 0.05 + 0.85 * Double(completed) / Double(max(total, 1))
        case "vsr decode": share = completed > 0 ? 1 : 0.9
        default: share = nil
        }
        if let share { bridge.progress(share) }
    }
    return bridge.cancelFlag.isCancelled ? 1 : 0
}

private let h3VSRCacheProgressTrampoline: h3_progress_callback = { _, completed, total, opaque in
    guard let opaque else { return 0 }
    let bridge = Unmanaged<VSRProgressBridge>.fromOpaque(opaque).takeUnretainedValue()
    bridge.continuation.yield(H3VSRCacheProgress(completed: Int(completed), total: Int(total)))
    return bridge.cancelFlag.isCancelled ? 1 : 0
}

public final class H3VideoSuperResolver: @unchecked Sendable {
    private let handle: OpaquePointer
    public let info: H3VSRInfo
    public let isInt8: Bool

    /// Whether this GPU takes the int8 (tensor hardware) path.
    public static var int8Supported: Bool { h3_vsr_int8_supported() != 0 }

    /// What a model directory holds, without loading it.
    public static func inspect(modelDirectory: String) throws -> H3VSRInfo {
        var info = h3_vsr_info()
        var error = [CChar](repeating: 0, count: 512)
        let ok = modelDirectory.withCString { dir in
            error.withUnsafeMutableBufferPointer { buffer in
                h3_vsr_inspect_dir(dir, &info, buffer.baseAddress, buffer.count)
            }
        }
        guard ok != 0 else { throw H3VSRError.failed(String(cString: error)) }
        return H3VSRInfo(name: cString(info.name), licenseId: cString(info.license_id),
                         commercialUseAllowed: info.commercial_use_allowed != 0,
                         firstChunkFrames: Int(info.first_chunk_frames), chunkFrames: Int(info.chunk_frames),
                         int8Supported: info.int8_supported != 0,
                         weightBytesBF16: info.weight_bytes_bf16, weightBytesInt8: info.weight_bytes_int8,
                         minOutputSide: Int(info.min_output_side), maxOutputSide: Int(info.max_output_side))
    }

    /// Where the cache of `modelDirectory` goes for a precision.
    public static func cachePath(modelDirectory: String, int8: Bool) -> String {
        var path = [CChar](repeating: 0, count: 4096)
        _ = modelDirectory.withCString { dir in
            path.withUnsafeMutableBufferPointer { buffer in
                h3_vsr_cache_path(dir, int8 ? H3_VSR_PRECISION_INT8 : H3_VSR_PRECISION_BF16,
                                  buffer.baseAddress, buffer.count)
            }
        }
        return String(cString: path)
    }

    /// The cache this Mac would use: int8 on tensor hardware, else BF16;
    /// nil when neither exists or matches the weights.
    public static func usableCachePath(modelDirectory: String) -> String? {
        let int8 = (try? inspect(modelDirectory: modelDirectory))?.int8Supported ?? false
        for precision in (int8 ? [true, false] : [false]) {
            let path = cachePath(modelDirectory: modelDirectory, int8: precision)
            if FileManager.default.fileExists(atPath: path) && cacheMatches(path: path, modelDirectory: modelDirectory) {
                return path
            }
        }
        return nil
    }

    public static func cacheMatches(path: String, modelDirectory: String) -> Bool {
        path.withCString { p in modelDirectory.withCString { d in h3_vsr_cache_matches(p, d) != 0 } }
    }

    /// Builds the cache from the downloaded weights, reporting per block.
    public static func buildCache(modelDirectory: String, outputPath: String, int8: Bool,
                                 shaderPath: String = "h3_shaders.metal") -> AsyncThrowingStream<H3VSRCacheProgress, Error> {
        AsyncThrowingStream { continuation in
            let cancelFlag = VSRCancelFlag()
            let bridge = VSRProgressBridge(continuation: continuation, cancelFlag: cancelFlag)
            let bridgeHandle = Unmanaged.passRetained(bridge)
            continuation.onTermination = { termination in
                if case .cancelled = termination { cancelFlag.cancel() }
            }
            DispatchQueue.global(qos: .userInitiated).async {
                defer { bridgeHandle.release() }
                var error = [CChar](repeating: 0, count: 1024)
                let ok = modelDirectory.withCString { dir in
                    outputPath.withCString { out in
                        shaderPath.withCString { shader in
                            error.withUnsafeMutableBufferPointer { buffer in
                                h3_vsr_build_cache(dir, out, int8 ? H3_VSR_PRECISION_INT8 : H3_VSR_PRECISION_BF16,
                                                   shader, h3VSRCacheProgressTrampoline, bridgeHandle.toOpaque(),
                                                   buffer.baseAddress, buffer.count)
                            }
                        }
                    }
                }
                if ok != 0 {
                    continuation.finish()
                } else {
                    continuation.finish(throwing: cancelFlag.isCancelled
                        ? H3VSRError.cancelled : H3VSRError.failed(String(cString: error)))
                }
            }
        }
    }

    /// Loads the model: the DiT from `cachePath`, the ReAE from the directory.
    public init(modelDirectory: String, cachePath: String, fullAttention: Bool = false,
                shaderPath: String = "h3_shaders.metal") throws {
        info = try Self.inspect(modelDirectory: modelDirectory)
        var error = [CChar](repeating: 0, count: 1024)
        var handle: OpaquePointer?
        var params = h3_vsr_open_params()
        params.precision = H3_VSR_PRECISION_AUTO
        params.full_attention = fullAttention ? 1 : 0
        handle = modelDirectory.withCString { dir in
            cachePath.withCString { cache in
                shaderPath.withCString { shader in
                    params.cache_path = cache
                    params.shader_source_path = shader
                    return error.withUnsafeMutableBufferPointer { buffer in
                        h3_vsr_open(dir, &params, nil, nil, buffer.baseAddress, buffer.count)
                    }
                }
            }
        }
        guard let handle else { throw H3VSRError.failed(String(cString: error)) }
        self.handle = handle
        isInt8 = cachePath.hasSuffix("int8.h3vc")
    }

    deinit { h3_vsr_close(handle) }

    public func requirements(inputWidth: Int, inputHeight: Int, outputWidth: Int, outputHeight: Int) -> H3VSRRequirements {
        var req = h3_vsr_requirements()
        _ = h3_vsr_check_requirements(handle, Int32(inputWidth), Int32(inputHeight), Int32(outputWidth), Int32(outputHeight), &req)
        return H3VSRRequirements(supported: req.supported != 0, reason: cString(req.reason), gpuBytes: req.gpu_bytes,
                                 estimatedSecondsPerChunk: req.estimated_seconds_per_chunk,
                                 tokensPerChunk: Int(req.tokens_per_chunk))
    }

    /// GPU seconds spent so far and the peak live bytes.
    public var stats: (seconds: Double, peakBytes: UInt64) {
        var dit = 0.0, reae = 0.0
        var peak: UInt64 = 0
        _ = h3_vsr_get_stats(handle, &dit, &reae, &peak)
        return (dit + reae, peak)
    }

    /// One video. Frames are 32BGRA CVPixelBuffers; `process` writes into
    /// buffers it takes from `pool` (at most frames + 3 per call).
    public final class Session {
        private let handle: OpaquePointer
        private let resolver: H3VideoSuperResolver   // keeps the model alive
        private let cancelFlag = VSRCancelFlag()
        public let outputWidth: Int
        public let outputHeight: Int

        fileprivate init(resolver: H3VideoSuperResolver, handle: OpaquePointer, outputWidth: Int, outputHeight: Int) {
            self.resolver = resolver
            self.handle = handle
            self.outputWidth = outputWidth
            self.outputHeight = outputHeight
        }

        deinit { h3_vsr_session_close(handle) }

        /// Frames the next call should bring (28, then 24; 0 when finished).
        public var nextChunkFrames: Int { Int(h3_vsr_session_next_chunk_frames(handle)) }

        /// Asks the engine to stop at the next block.
        public func cancel() { cancelFlag.cancel() }

        /// `progress` gets this call's share done (0-1; from the engine's
        /// thread, once per DiT block) when given.
        public func process(_ frames: [CVPixelBuffer], isLast: Bool, pool: CVPixelBufferPool,
                            progress: ((Double) -> Void)? = nil) throws -> [CVPixelBuffer] {
            guard !frames.isEmpty else { return [] }
            let bridge = progress.map { VSRProcessBridge(progress: $0, cancelFlag: cancelFlag) }
            let bridgeHandle = bridge.map { Unmanaged.passRetained($0) }
            defer { bridgeHandle?.release() }
            // Output buffers for everything the engine may hand back: the frames
            // still inside it (a chunk decodes on its own GPU context while the next
            // one runs, so a call returns the previous chunk's frames) plus these.
            let capacity = Int(h3_vsr_session_pending_frames(handle)) + frames.count + 3
            var outputs: [CVPixelBuffer] = []
            for _ in 0 ..< capacity {
                var buffer: CVPixelBuffer?
                CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
                guard let buffer else { throw H3VSRError.failed(String(localized: "出力フレームを確保できませんでした")) }
                outputs.append(buffer)
            }
            for frame in frames { CVPixelBufferLockBaseAddress(frame, .readOnly) }
            for buffer in outputs { CVPixelBufferLockBaseAddress(buffer, []) }
            defer {
                for frame in frames { CVPixelBufferUnlockBaseAddress(frame, .readOnly) }
                for buffer in outputs { CVPixelBufferUnlockBaseAddress(buffer, []) }
            }
            var inputs = frames.map { frame -> h3_vsr_image in
                h3_vsr_image(pixels: CVPixelBufferGetBaseAddress(frame)!.assumingMemoryBound(to: UInt8.self),
                             width: Int32(CVPixelBufferGetWidth(frame)), height: Int32(CVPixelBufferGetHeight(frame)),
                             stride_bytes: Int32(CVPixelBufferGetBytesPerRow(frame)), format: H3_VSR_BGRA8)
            }
            var outs = outputs.map { buffer -> h3_vsr_image_out in
                h3_vsr_image_out(pixels: CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self),
                                 width: Int32(CVPixelBufferGetWidth(buffer)), height: Int32(CVPixelBufferGetHeight(buffer)),
                                 stride_bytes: Int32(CVPixelBufferGetBytesPerRow(buffer)), format: H3_VSR_BGRA8)
            }
            var produced: Int32 = 0
            var error = [CChar](repeating: 0, count: 1024)
            let ok = inputs.withUnsafeMutableBufferPointer { ins in
                outs.withUnsafeMutableBufferPointer { outPtr in
                    error.withUnsafeMutableBufferPointer { buffer in
                        h3_vsr_session_process(handle, ins.baseAddress, Int32(frames.count), isLast ? 1 : 0,
                                               outPtr.baseAddress, Int32(outputs.count), &produced,
                                               bridgeHandle == nil ? nil : h3VSRProcessTrampoline,
                                               bridgeHandle?.toOpaque(),
                                               cancelFlag.engineFlag, buffer.baseAddress, buffer.count)
                    }
                }
            }
            guard ok != 0 else {
                throw cancelFlag.isCancelled ? H3VSRError.cancelled : H3VSRError.failed(String(cString: error))
            }
            return Array(outputs.prefix(Int(produced)))
        }
    }

    public func openSession(inputWidth: Int, inputHeight: Int, outputWidth: Int, outputHeight: Int,
                            totalFrames: Int = 0) throws -> Session {
        var error = [CChar](repeating: 0, count: 1024)
        let session = error.withUnsafeMutableBufferPointer { buffer in
            h3_vsr_session_open(handle, Int32(inputWidth), Int32(inputHeight), Int32(outputWidth), Int32(outputHeight),
                                Int32(totalFrames), buffer.baseAddress, buffer.count)
        }
        guard let session else { throw H3VSRError.failed(String(cString: error)) }
        return Session(resolver: self, handle: session, outputWidth: outputWidth, outputHeight: outputHeight)
    }
}
