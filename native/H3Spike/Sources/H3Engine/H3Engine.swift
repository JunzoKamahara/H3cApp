// Async Swift wrapper around libh3.a (see h3.h). Turns the C callback-driven
// h3_generate into an AsyncThrowingStream so SwiftUI can consume progress
// and frame events directly, in-process - no subprocess, no HTTP polling,
// unlike gui/server.py's job model.
import CH3
import Foundation
import Metal

public enum H3GenerationEvent: Sendable {
    case progress(phase: String, completed: Int, total: Int)
    case frame(index: Int, count: Int, width: Int, height: Int)
    case preview(step: Int, steps: Int)
    case finished(H3GenerationResult)
}

public struct H3AttentionCacheProgress: Sendable {
    public let completedBlocks: Int
    public let totalBlocks: Int
}

public struct H3DeviceInfo: Sendable {
    public let name: String
    public let architecture: String
    public let unifiedMemory: Bool
    public let metal4: Bool
    /// Highest Apple GPU family the device reports supporting (h3_metal.m
    /// probes MTLGPUFamilyApple1...10 and keeps the highest match). 10 is the
    /// family Apple introduced the Metal 4 hardware tensor/matmul units
    /// (Neural Accelerators) with, first shipping in M5 - this is the real
    /// capability the int8 attention cache path needs, not the device name.
    public let appleGPUFamily: Int
    /// True when this device has the hardware tensor units the int8
    /// attention cache path relies on - mirrors h3_gpu.m's
    /// h3_device_has_tensor_ops(), which checks -supportsFamily: rather than
    /// matching "M5" in the device's marketing name.
    public var hasTensorHardware: Bool { appleGPUFamily >= 10 }
}

public struct H3GenerationResult: Sendable {
    public let frames: Int
    public let fps: Int
    public let seed: UInt64
    public let outputPath: String
    // The attention path that actually ran (h3_result): full-attention
    // calls served by ccv, and how many took the fast-mode direct path.
    // Both 0 on the standard path.
    public let ccvAttentionCalls: Int
    public let ccvAttentionDirectCalls: Int
}

public enum H3ReferenceKind: Sendable, Equatable {
    case image, video, audio, videoAudio

    var cValue: h3_reference_kind {
        switch self {
        case .image: return H3_REFERENCE_IMAGE
        case .video: return H3_REFERENCE_VIDEO
        case .audio: return H3_REFERENCE_AUDIO
        case .videoAudio: return H3_REFERENCE_VIDEO_AUDIO
        }
    }
}

public struct H3ReferenceInput: Sendable, Identifiable {
    public let id = UUID()
    public var kind: H3ReferenceKind
    public var path: String
    public var audioPath: String?
    public var includeEmbeddedAudio: Bool

    public init(kind: H3ReferenceKind = .image, path: String, audioPath: String? = nil,
                includeEmbeddedAudio: Bool = false) {
        self.kind = kind
        self.path = path
        self.audioPath = audioPath
        self.includeEmbeddedAudio = includeEmbeddedAudio
    }
}

public struct H3LoRAInput: Sendable, Equatable {
    public var path: String
    /// Multiplies the adapter's own alpha/rank scale (1 = as trained).
    public var strength: Float

    public init(path: String, strength: Float = 1) {
        self.path = path
        self.strength = strength
    }
}

/// What h3_lora_inspect found in an adapter file (see h3.h).
public struct H3LoRAInfo: Sendable, Equatable {
    public let blocks: Int
    public let refinerBlocks: Int
    public let projections: Int
    public let unsupported: Int
    public let rankMin: Int
    public let rankMax: Int
    public let format: String
    public let baseModel: String
    /// The sampling the adapter's metadata recommends, when it records one
    /// (DMAD: "... audio shift 2, re-noise step rule ..."): the re-noise
    /// step rule, and an audio sigma shift other than the engine's 3.
    public let recommendsRenoise: Bool
    public let recommendedAudioShift: Double?

    /// Reads only the safetensors header - cheap enough to call from the UI.
    public static func inspect(path: String) throws -> H3LoRAInfo {
        var info = h3_lora_info()
        var errorBuffer = [CChar](repeating: 0, count: 512)
        let ok = errorBuffer.withUnsafeMutableBufferPointer { errorBuf in
            h3_lora_inspect(path, &info, errorBuf.baseAddress, errorBuf.count)
        }
        guard ok != 0 else { throw H3EngineError.unusableLoRA(String(cString: errorBuffer)) }
        return H3LoRAInfo(
            blocks: Int(info.blocks), refinerBlocks: Int(info.refiner_blocks),
            projections: Int(info.projections), unsupported: Int(info.unsupported),
            rankMin: Int(info.rank_min), rankMax: Int(info.rank_max),
            format: fixedCString(info.format), baseModel: fixedCString(info.base_model),
            recommendsRenoise: info.sampler == Int32(H3_SAMPLER_RENOISE),
            recommendedAudioShift: info.audio_sigma_shift > 0 ? Double(info.audio_sigma_shift) : nil)
    }
}

public struct H3GenerationParams: Sendable {
    public var width: Int32 = 512
    public var height: Int32 = 512
    public var renderWidth: Int32 = 0
    public var renderHeight: Int32 = 0
    public var frames: Int32 = 56
    public var steps: Int32 = 20
    public var seed: UInt64 = 42
    public var ditLayers: Int32 = 50
    public var denoiseReuse: Int32 = 1
    public var coreReuse: Int32 = 1
    public var tokenReduction = false
    public var firstFrame: String?
    public var lastFrame: String?
    public var references: [H3ReferenceInput] = []
    // libh3.a reads this from the environment rather than h3_params (see
    // H3_ATTENTION_CACHE in h3_dit.c), so H3Engine sets it just before each
    // call instead of once at process startup - different generation modes
    // (default/Ref2VA) need different prebuilt caches.
    public var attentionCachePath: String?
    // H3_ADALN_CURVE (h3_adaln_curve.h): a ~0.6 GiB curve-form stand-in for
    // the ~24 GiB of dense AdaLN projections, read from the environment like
    // attentionCachePath. Set to a directory: generate() uses
    // adaln_fl2va.curve or adaln_ref2va.curve there - whichever transformer
    // this job runs (Ref2VA exactly when there are references) - building
    // it first when it is missing or was built from other weights (~7 s,
    // reported as phase "AdaLN curve"), and falls back to the dense
    // projections if that fails. Works in every compute mode. nil: the
    // dense projections, as before.
    public var adalnCurveDirectory: String?
    // h3_params.loras: adapters added to the DiT weights, stacked in order.
    // Any published H3 LoRA layout (diffusers, ComfyUI, kohya, native) works
    // with every compute mode - the engine patches resident weights once at
    // load and streamed ones (int8 cache, SSD) as each block streams in.
    public var loras: [H3LoRAInput] = []
    // h3_params.ssd_streaming: keep only two original BF16 DiT blocks
    // resident and read the next from the checkpoint while the GPU runs the
    // current one. It's an alternative to attentionCachePath, not an
    // addition: it runs on the unquantized BF16 weights, so the int8 cache
    // (and H3_INT8_STREAM_MLP) don't apply - generate() clears those
    // instead of passing along a combination the engine would ignore.
    public var ssdStreaming = false
    // h3_params.fast_attention: opt-in fast mode (experimental) - ccv's int8
    // attention on M5-class GPUs; see H3Engine.fastAttentionAvailable. The
    // same seed gives a different video than the standard mode. Separate
    // from the speed settings above and never turned on by them.
    public var fastAttention = false
    // h3_params.sampler / audio_sigma_shift: the re-noise step rule and the
    // audio schedule shift a few-step adapter was trained with (DMAD), taken
    // from its metadata; the defaults are the engine's Euler rule and 3.0.
    // Re-noise evaluates every step, so it needs denoiseReuse 1.
    public var renoise = false
    public var audioSigmaShift: Float = 0
    // h3_params.forced_audio_path / forced_audio_offset: lip sync. This
    // file's audio, from forcedAudioOffset seconds on, is held as the
    // soundtrack through every denoising step, so only the picture is
    // generated, and the output carries that audio itself.
    public var forcedAudioPath: String?
    public var forcedAudioOffset: Double = 0
    // h3_params.save_latent_path / continue_latent_path / continue_frames:
    // keep this clip's latents (before decoding) for a later continuation,
    // and continue an earlier clip from its latent file - its last
    // continueFrames frames are pinned at the start of this one.
    public var saveLatentPath: String?
    public var continueLatentPath: String?
    public var continueFrames: Int32 = 22
    public init() {}
}

/* Matches align_frames() in gui/server.py, itself matching h3_align_frame_count
 * (h3_host.c): the engine only accepts frame counts of the form 5 + 17*k, so a
 * caller offering a "seconds" control has to round up to one itself. Not
 * calling into libh3.a for this since h3_align_frame_count lives in the
 * internal h3_host.h, not the public h3.h surface CH3 exposes. */
public func h3AlignedFrameCount(seconds: Double) -> Int32 {
    var requested = max(5, Int((seconds * 24).rounded()))
    let remainder = (requested - 5) % 17
    if remainder != 0 {
        requested += 17 - remainder
    }
    return Int32(requested)
}

/* Calls `body` with a C string for `string`, or nil if `string` is nil. */
private func withOptionalCString<R>(_ string: String?, _ body: (UnsafePointer<CChar>?) -> R) -> R {
    if let string {
        return string.withCString(body)
    }
    return body(nil)
}

/* Same nesting trick as withReferenceArray below, for h3_lora paths. */
private func withLoRAArray<R>(_ loras: [H3LoRAInput], built: [h3_lora] = [],
                              _ body: (UnsafeBufferPointer<h3_lora>) -> R) -> R {
    guard let lora = loras.first else {
        return built.withUnsafeBufferPointer(body)
    }
    let rest = Array(loras.dropFirst())
    return lora.path.withCString { pathC in
        var next = built
        next.append(h3_lora(path: pathC, strength: lora.strength))
        return withLoRAArray(rest, built: next, body)
    }
}

/* Builds the h3_reference array by nesting one withCString/withOptionalCString
 * per path so all of them stay valid for the single call to `body`, without
 * resorting to manual strdup/free bookkeeping. */
private func withReferenceArray<R>(_ references: [H3ReferenceInput], built: [h3_reference] = [],
                                    _ body: (UnsafeBufferPointer<h3_reference>) -> R) -> R {
    guard let reference = references.first else {
        return built.withUnsafeBufferPointer(body)
    }
    let rest = Array(references.dropFirst())
    return reference.path.withCString { pathC in
        withOptionalCString(reference.audioPath) { audioC in
            var next = built
            next.append(h3_reference(kind: reference.kind.cValue, path: pathC, audio_path: audioC,
                                      include_embedded_audio: reference.includeEmbeddedAudio ? 1 : 0))
            return withReferenceArray(rest, built: next, body)
        }
    }
}

public enum H3EngineError: Error, LocalizedError {
    case loadFailed(String)
    case generationFailed(String)
    case cacheBuildFailed(String)
    case unusableLoRA(String)
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .loadFailed(let message): return "Failed to load model: \(message)"
        case .generationFailed(let message): return "Generation failed: \(message)"
        case .cacheBuildFailed(let message): return "Attention cache build failed: \(message)"
        case .unusableLoRA(let message): return message
        case .cancelled: return "Cancelled"
        }
    }
}

private func fixedCString<T>(_ tuple: T) -> String {
    withUnsafeBytes(of: tuple) { raw in
        String(cString: raw.bindMemory(to: CChar.self).baseAddress!)
    }
}

/* A plain polling flag: the generation thread checks it from inside the C
 * callbacks, the caller (main actor) sets it when the user cancels. */
private final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return value
    }
    func cancel() {
        lock.lock(); value = true; lock.unlock()
    }
}

/* h3_frame_callback/h3_progress_callback are plain C function pointers, so
 * they can't capture Swift state directly. This bundles the stream
 * continuation and the cancel flag behind the `callback_opaque` pointer
 * instead, recovered on the other side with Unmanaged. */
private final class GenerationBridge: @unchecked Sendable {
    let continuation: AsyncThrowingStream<H3GenerationEvent, Error>.Continuation
    let cancelFlag: CancelFlag
    init(continuation: AsyncThrowingStream<H3GenerationEvent, Error>.Continuation, cancelFlag: CancelFlag) {
        self.continuation = continuation
        self.cancelFlag = cancelFlag
    }
}

private let h3ProgressTrampoline: h3_progress_callback = { phase, completed, total, opaque in
    guard let opaque else { return 0 }
    let bridge = Unmanaged<GenerationBridge>.fromOpaque(opaque).takeUnretainedValue()
    let phaseName = phase.map { String(cString: $0) } ?? ""
    bridge.continuation.yield(.progress(phase: phaseName, completed: Int(completed), total: Int(total)))
    return bridge.cancelFlag.isCancelled ? 1 : 0
}

/* Same bridging idea as GenerationBridge, for h3_build_attention_cache's
 * single progress callback (no frame callback, no generation result). */
private final class AttentionCacheBridge: @unchecked Sendable {
    let continuation: AsyncThrowingStream<H3AttentionCacheProgress, Error>.Continuation
    let cancelFlag: CancelFlag
    init(continuation: AsyncThrowingStream<H3AttentionCacheProgress, Error>.Continuation, cancelFlag: CancelFlag) {
        self.continuation = continuation
        self.cancelFlag = cancelFlag
    }
}

private let h3AttentionCacheProgressTrampoline: h3_progress_callback = { _, completed, total, opaque in
    guard let opaque else { return 0 }
    let bridge = Unmanaged<AttentionCacheBridge>.fromOpaque(opaque).takeUnretainedValue()
    bridge.continuation.yield(H3AttentionCacheProgress(completedBlocks: Int(completed), totalBlocks: Int(total)))
    return bridge.cancelFlag.isCancelled ? 1 : 0
}

private let h3FrameTrampoline: h3_frame_callback = { framePtr, opaque in
    guard let framePtr, let opaque else { return 0 }
    let bridge = Unmanaged<GenerationBridge>.fromOpaque(opaque).takeUnretainedValue()
    let frame = framePtr.pointee
    if frame.denoise_step >= 0 {
        bridge.continuation.yield(.preview(step: Int(frame.denoise_step), steps: Int(frame.denoise_steps)))
    } else {
        bridge.continuation.yield(.frame(index: Int(frame.frame_index), count: Int(frame.frame_count),
                                          width: Int(frame.width), height: Int(frame.height)))
    }
    return bridge.cancelFlag.isCancelled ? 1 : 0
}

public final class H3Engine: @unchecked Sendable {
    /// Every generation and cache build in the process runs on this one
    /// serial queue, so two can never overlap - not from a batch starting
    /// its next video, not from two windows (each has its own H3Engine).
    /// The engine's process-wide state (H3_* environment variables, the
    /// ccv attention backend's single global state, GPU memory) assumes one
    /// generation at a time; UI-level "is generating" checks don't cover
    /// other entry points.
    private static let workQueue = DispatchQueue(label: "h3.engine.work", qos: .userInitiated)

    private let ctx: OpaquePointer
    private let modelDirectory: String
    private var currentCancelFlag: CancelFlag?
    private var currentCacheBuildCancelFlag: CancelFlag?

    public init(modelDirectory: String) throws {
        guard let ctx = h3_load_dir(modelDirectory) else {
            throw H3EngineError.loadFailed("h3_load_dir returned NULL")
        }
        self.ctx = ctx
        self.modelDirectory = modelDirectory
        h3_cache_set_targets(ctx, Self.cacheTargets(batch: false))
    }

    /// What the engine keeps between generations. Always the prompt/
    /// reference conditioning and the token refiner's output (a few MB): a
    /// seed-only rerun or the next batch item skips the text encoder, Qwen
    /// vision, the reference VAE encoder and the refiner. During a batch
    /// also the AdaLN schedule (~150-400 MB, ~8 s per item). The prepared
    /// DiT and VAE decoder would save a little more but hold ~5.8 GiB
    /// between runs and raise the peak by ~2.9 GiB (SPEEDUP_ROADMAP.md
    /// item 7), too much for a 24 GB Mac.
    private static func cacheTargets(batch: Bool) -> UInt32 {
        var targets = UInt32(H3_CACHE_CONDITIONING) | UInt32(H3_CACHE_REFINED_TEXT)
        if batch { targets |= UInt32(H3_CACHE_ADALN) }
        return targets
    }

    /// Call with true before the first video of a batch and false after the
    /// last (or a cancel). Queued behind any generation in flight.
    public func setBatchReuse(_ batch: Bool) {
        let targets = Self.cacheTargets(batch: batch)
        // Holds the engine (and so its context) until the block has run -
        // a job ahead of it can outlast the window that owns the engine.
        Self.workQueue.async { [self] in h3_cache_set_targets(self.ctx, targets) }
    }

    deinit {
        h3_free(ctx)
    }

    public var device: H3DeviceInfo? {
        guard let device = h3_device(ctx)?.pointee else { return nil }
        return H3DeviceInfo(
            name: fixedCString(device.name),
            architecture: fixedCString(device.architecture),
            unifiedMemory: device.unified_memory != 0,
            metal4: device.metal4 != 0,
            appleGPUFamily: Int(device.apple_gpu_family)
        )
    }

    public func cancelCurrentGeneration() {
        currentCancelFlag?.cancel()
    }

    /// Whether h3_params.fast_attention can be used here: the engine was
    /// built with the ccv backend and the GPU has the neural matrix
    /// accelerators it needs.
    public static var fastAttentionAvailable: Bool { h3_fast_attention_available() != 0 }

    public func generate(prompt: String, outputPath: String,
                          params: H3GenerationParams) -> AsyncThrowingStream<H3GenerationEvent, Error> {
        return AsyncThrowingStream { continuation in
            let cancelFlag = CancelFlag()
            self.currentCancelFlag = cancelFlag
            let bridge = GenerationBridge(continuation: continuation, cancelFlag: cancelFlag)
            let bridgeHandle = Unmanaged.passRetained(bridge)

            Self.workQueue.async { [self] in
                defer { bridgeHandle.release() }
                // Read through self, so the engine - and the context h3_free
                // releases in deinit - lives until this job has finished.
                let ctx = self.ctx

                // A GUI app that's occluded or in the background is eligible
                // for App Nap, which lowers CPU/disk I/O priority - measured
                // at ~1.5x slower on this pipeline (see taskpolicy -b trace).
                // Generation is user-requested work, so opt out for its
                // duration.
                let activity = ProcessInfo.processInfo.beginActivity(
                    options: [.userInitiated, .idleSystemSleepDisabled],
                    reason: "Generating video")
                defer { ProcessInfo.processInfo.endActivity(activity) }

                if params.ssdStreaming {
                    // Exclusive with the int8 cache - see
                    // H3GenerationParams.ssdStreaming.
                    unsetenv("H3_ATTENTION_CACHE")
                    unsetenv("H3_INT8_STREAM_MLP")
                } else if let attentionCachePath = params.attentionCachePath {
                    setenv("H3_ATTENTION_CACHE", attentionCachePath, 1)
                    // Same pairing gui/server.py always used with a cache:
                    // stream FC1/FC2 from it too instead of keeping ~10.8GiB
                    // of int8 MLP resident for every block, which on a 24GB
                    // machine turns into memory pressure on longer clips
                    // (and measured +10s of setup even on a short one).
                    setenv("H3_INT8_STREAM_MLP", "1", 1)
                } else {
                    unsetenv("H3_ATTENTION_CACHE")
                    unsetenv("H3_INT8_STREAM_MLP")
                }
                // gui/server.py also pins the Qwen text-encoder prefetch
                // depth to 1 (default is 3 on M5) to keep its footprint down.
                if getenv("H3_QWEN_PREFETCH_DEPTH") == nil {
                    setenv("H3_QWEN_PREFETCH_DEPTH", "1", 1)
                }

                if let curveDirectory = params.adalnCurveDirectory {
                    let ref2va = !params.references.isEmpty
                    let curvePath = curveDirectory +
                        (ref2va ? "/adaln_ref2va.curve" : "/adaln_fl2va.curve")
                    let transformerDirectory = self.modelDirectory +
                        (ref2va ? "/Ref2VA/transformer" : "/FL2VA/transformer")
                    if Self.prepareAdaLNCurve(path: curvePath,
                                              transformerDirectory: transformerDirectory,
                                              progressOpaque: bridgeHandle.toOpaque()) {
                        setenv("H3_ADALN_CURVE", curvePath, 1)
                    } else {
                        unsetenv("H3_ADALN_CURVE")
                    }
                    if cancelFlag.isCancelled {
                        continuation.finish(throwing: H3EngineError.cancelled)
                        return
                    }
                } else {
                    unsetenv("H3_ADALN_CURVE")
                }

                // The stream ends only after the pool below has drained, so a
                // caller that starts the next generation as soon as this one
                // ends (a batch) never overlaps its teardown.
                var endStream: () -> Void = { continuation.finish() }

                // One pool per generation, inside the background closure: a
                // system global queue sets up no per-item autorelease pool
                // (AutoreleaseFrequency.never), so without this the Metal
                // objects h3_generate autoreleases would outlive the run.
                // Covers the call and the result handling, and is left on
                // success, failure and cancel alike. It does not replace the
                // engine's own GPU-completion waits.
                autoreleasepool {
                var cParams = h3_params()
                cParams.width = params.width
                cParams.height = params.height
                cParams.render_width = params.renderWidth
                cParams.render_height = params.renderHeight
                cParams.frames = params.frames
                cParams.steps = params.steps
                cParams.seed = params.seed
                cParams.dit_layers = params.ditLayers
                cParams.denoise_reuse = params.denoiseReuse
                cParams.core_reuse = params.coreReuse
                cParams.token_reduction = params.tokenReduction ? 1 : 0
                cParams.ssd_streaming = params.ssdStreaming ? 1 : 0
                cParams.fast_attention = params.fastAttention ? 1 : 0
                cParams.sampler = params.renoise ? Int32(H3_SAMPLER_RENOISE)
                                                 : Int32(H3_SAMPLER_EULER)
                cParams.audio_sigma_shift = params.audioSigmaShift
                cParams.forced_audio_offset = params.forcedAudioOffset
                cParams.continue_frames = params.continueFrames
                // The values actually handed to the engine (not the form's
                // labels), for checking presets and defaults end to end.
                NSLog("h3: generation params - %dx%d, %d frames, %d steps, dit_layers %d, denoise_reuse %d, core_reuse %d, token_reduction %d, fast_attention %d, ssd_streaming %d, sampler %d, audio_shift %.3g, forced_audio %d (offset %.3f s), continue %d (%d frames), seed %llu",
                      cParams.width, cParams.height, cParams.frames, cParams.steps,
                      cParams.dit_layers, cParams.denoise_reuse, cParams.core_reuse,
                      cParams.token_reduction, cParams.fast_attention, cParams.ssd_streaming,
                      cParams.sampler, Double(cParams.audio_sigma_shift),
                      params.forcedAudioPath == nil ? 0 : 1, params.forcedAudioOffset,
                      params.continueLatentPath == nil ? 0 : 1, params.continueFrames,
                      cParams.seed)
                cParams.on_progress = h3ProgressTrampoline
                cParams.on_frame = h3FrameTrampoline
                cParams.callback_opaque = bridgeHandle.toOpaque()

                let result: UnsafeMutablePointer<h3_result>? = withLoRAArray(params.loras) { loraBuffer in
                    withReferenceArray(params.references) { refBuffer in
                        outputPath.withCString { outputPathC in
                            prompt.withCString { promptC in
                                withOptionalCString(params.firstFrame) { firstFrameC in
                                    withOptionalCString(params.lastFrame) { lastFrameC in
                                    withOptionalCString(params.forcedAudioPath) { forcedAudioC in
                                    withOptionalCString(params.saveLatentPath) { saveLatentC in
                                    withOptionalCString(params.continueLatentPath) { continueLatentC in
                                        cParams.output_path = outputPathC
                                        cParams.forced_audio_path = forcedAudioC
                                        cParams.save_latent_path = saveLatentC
                                        cParams.continue_latent_path = continueLatentC
                                        cParams.first_frame = firstFrameC
                                        cParams.last_frame = lastFrameC
                                        cParams.references = refBuffer.baseAddress
                                        cParams.reference_count = refBuffer.count
                                        cParams.loras = loraBuffer.baseAddress
                                        cParams.lora_count = loraBuffer.count
                                        return h3_generate(ctx, promptC, &cParams)
                                    }
                                    }
                                    }
                                    }
                                }
                            }
                        }
                    }
                }

                if let result {
                    let genResult = H3GenerationResult(
                        frames: Int(result.pointee.frames),
                        fps: Int(result.pointee.fps),
                        seed: result.pointee.seed,
                        outputPath: outputPath,
                        ccvAttentionCalls: Int(result.pointee.ccv_attention_calls),
                        ccvAttentionDirectCalls: Int(result.pointee.ccv_attention_direct_calls)
                    )
                    h3_result_free(result)
                    let path = genResult.ccvAttentionCalls == 0 ? "standard" :
                        genResult.ccvAttentionDirectCalls == genResult.ccvAttentionCalls ?
                        "fast (direct)" : "ccv (mixed/bridge)"
                    NSLog("h3: generation finished - fast attention requested: %@, attention path: %@ (%d ccv calls, %d direct)",
                          params.fastAttention ? "yes" : "no", path,
                          genResult.ccvAttentionCalls, genResult.ccvAttentionDirectCalls)
                    continuation.yield(.finished(genResult))
                } else {
                    let message = h3_last_error(ctx).map { String(cString: $0) } ?? "unknown error"
                    let error = cancelFlag.isCancelled
                        ? H3EngineError.cancelled : H3EngineError.generationFailed(message)
                    endStream = { continuation.finish(throwing: error) }
                }
                }
                // After the pool has drained: what is still allocated on the
                // GPU once this generation's objects are gone.
                if let device = MTLCreateSystemDefaultDevice() {
                    NSLog("h3: device allocated after generation: %.3f GiB",
                          Double(device.currentAllocatedSize) / 1_073_741_824)
                }
                endStream()
            }
        }
    }

    public func cancelCurrentCacheBuild() {
        currentCacheBuildCancelFlag?.cancel()
    }

    /// Makes sure `path` holds a curve-form AdaLN file built from
    /// `transformerDirectory`, building it if not. Progress goes to the
    /// generation's own stream (phase "AdaLN curve"), whose cancel flag
    /// also stops the build. Returns whether the file is usable; a failed
    /// build is logged, not fatal - the generation then uses the dense
    /// projections.
    private static func prepareAdaLNCurve(path: String, transformerDirectory: String,
                                          progressOpaque: UnsafeMutableRawPointer) -> Bool {
        if h3_adaln_curve_matches(path, transformerDirectory) != 0 { return true }
        try? FileManager.default.createDirectory(
            atPath: (path as NSString).deletingLastPathComponent,
            withIntermediateDirectories: true)
        var errorBuffer = [CChar](repeating: 0, count: 512)
        let ok = errorBuffer.withUnsafeMutableBufferPointer { errorBuf in
            h3_build_adaln_curve(transformerDirectory, path, "h3_shaders.metal", 0, 0,
                                 h3ProgressTrampoline, progressOpaque,
                                 errorBuf.baseAddress, errorBuf.count)
        }
        if ok == 0 {
            NSLog("h3: AdaLN curve not built, using the dense projections: %@",
                  String(cString: errorBuffer))
            return false
        }
        return true
    }

    /// Builds the int8 attention cache the app is missing (see
    /// GenerationViewModel.validationMessage) directly, in-process - no
    /// separate build_attention_cache subprocess. Not tied to this
    /// engine's own ctx (h3_build_attention_cache opens its own GPU
    /// device), but lives here since the app always has an H3Engine
    /// instance already by the time it would offer this.
    public func buildAttentionCache(transformerDirectory: String,
                                    outputPath: String) -> AsyncThrowingStream<H3AttentionCacheProgress, Error> {
        return AsyncThrowingStream { continuation in
            let cancelFlag = CancelFlag()
            self.currentCacheBuildCancelFlag = cancelFlag
            let bridge = AttentionCacheBridge(continuation: continuation, cancelFlag: cancelFlag)
            let bridgeHandle = Unmanaged.passRetained(bridge)

            Self.workQueue.async {
                defer { bridgeHandle.release() }
                let activity = ProcessInfo.processInfo.beginActivity(
                    options: [.userInitiated, .idleSystemSleepDisabled],
                    reason: "Building attention cache")
                defer { ProcessInfo.processInfo.endActivity(activity) }

                var errorBuffer = [CChar](repeating: 0, count: 512)
                let ok = transformerDirectory.withCString { transformerDirC in
                    outputPath.withCString { outputPathC in
                        "h3_shaders.metal".withCString { shaderPathC in
                            errorBuffer.withUnsafeMutableBufferPointer { errorBuf in
                                h3_build_attention_cache(
                                    transformerDirC, outputPathC, shaderPathC,
                                    h3AttentionCacheProgressTrampoline,
                                    bridgeHandle.toOpaque(),
                                    errorBuf.baseAddress, errorBuf.count)
                            }
                        }
                    }
                }

                if ok != 0 {
                    continuation.finish()
                } else {
                    let message = String(cString: errorBuffer)
                    continuation.finish(throwing: cancelFlag.isCancelled
                        ? H3EngineError.cancelled : H3EngineError.cacheBuildFailed(message))
                }
            }
        }
    }
}
