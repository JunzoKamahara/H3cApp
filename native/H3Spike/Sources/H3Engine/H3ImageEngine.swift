import CH3
import Foundation
import Metal

// Still images with MiniMax H3 Image (include/h3_image.h): the Swift side
// of the image engine, apart from H3Engine (the video one) as the C APIs
// are. It runs on H3Engine's serial work queue, so an image never overlaps
// a video generation, a cache build or a RefMod extraction.

/// A picture an image is edited from. The prompt calls the references
/// <Picture 1>, <Picture 2>, ... in the order they are given.
public struct H3ImageReference: Sendable, Equatable {
    public var path: String
    /// The image's own latent (the file the app keeps beside an image it
    /// generated); used instead of encoding the picture when its grid is
    /// the one the reference is encoded at.
    public var latentPath: String?

    public init(path: String, latentPath: String? = nil) {
        self.path = path
        self.latentPath = latentPath
    }
}

public struct H3ImageParams: Sendable {
    public var prompt: String
    /// Pixels, both multiples of 32.
    public var width: Int32
    public var height: Int32
    public var seed: UInt64
    /// Editing: up to H3ImageEngine.maxReferences pictures. Empty: text to
    /// image.
    public var references: [H3ImageReference]
    /// The longest short side a reference's latent is encoded at (0: the
    /// released pipeline's 2048).
    public var referenceMaxShortEdge: Int32
    /// Hand over the model's estimate of the finished latent after every
    /// step but the last (H3ImageEvent.latentPreview).
    public var latentPreview: Bool
    /// Redraw this picture (image to image, read at width x height) from
    /// the noise level `initStrength` instead of making one from noise.
    public var initImagePath: String?
    public var initStrength: Float

    public init(prompt: String, width: Int32, height: Int32, seed: UInt64,
                references: [H3ImageReference] = [], referenceMaxShortEdge: Int32 = 0,
                latentPreview: Bool = false, initImagePath: String? = nil, initStrength: Float = 0.7) {
        self.initImagePath = initImagePath
        self.initStrength = initStrength
        self.prompt = prompt
        self.width = width
        self.height = height
        self.seed = seed
        self.references = references
        self.referenceMaxShortEdge = referenceMaxShortEdge
        self.latentPreview = latentPreview
    }
}

/// How a reference was used by a generation.
public struct H3ImageReferenceUse: Sendable, Equatable {
    /// The size its latent was encoded at.
    public let width: Int
    public let height: Int
    /// Its own latent was taken instead of encoding the picture.
    public let fromLatent: Bool
}

public struct H3ImageResult: Sendable {
    public let width: Int
    public let height: Int
    /// Row-major interleaved RGB8.
    public let rgb: Data
    /// Channel-major F32 [24, latentHeight, latentWidth], normalized.
    public let latent: [Float]
    public let latentWidth: Int
    public let latentHeight: Int
    public let seed: UInt64
    public let promptTokens: Int
    public let steps: Int
    public let sigmas: [Float]
    public let textSeconds: Double
    public let denoiseSeconds: Double
    public let decodeSeconds: Double
    /// Editing: one entry per reference, in order; the time the vision
    /// tower and the VAE encoder took for them.
    public let references: [H3ImageReferenceUse]
    public let referenceSeconds: Double
}

public enum H3ImageEvent: Sendable {
    /// phase: "vision", "text encoder", "reference", "image model",
    /// "denoise" or "decode".
    case progress(phase: String, completed: Int, total: Int)
    /// The estimate of the finished image while it denoises: [24, height,
    /// width] normalized, time 1.
    case latentPreview(H3LatentPreview)
    case finished(H3ImageResult)
}

private final class ImageBridge {
    let continuation: AsyncThrowingStream<H3ImageEvent, Error>.Continuation
    init(continuation: AsyncThrowingStream<H3ImageEvent, Error>.Continuation) {
        self.continuation = continuation
    }
}

private func h3ImageProgressTrampoline(phase: UnsafePointer<CChar>?, completed: Int32,
                                       total: Int32, opaque: UnsafeMutableRawPointer?) {
    guard let opaque, let phase else { return }
    let bridge = Unmanaged<ImageBridge>.fromOpaque(opaque).takeUnretainedValue()
    bridge.continuation.yield(.progress(phase: String(cString: phase),
                                        completed: Int(completed), total: Int(total)))
}

private func h3ImageLatentPreviewTrampoline(latent: UnsafePointer<Float>?, width: Int32, height: Int32,
                                            step: Int32, steps: Int32, opaque: UnsafeMutableRawPointer?) {
    guard let opaque, let latent else { return }
    let bridge = Unmanaged<ImageBridge>.fromOpaque(opaque).takeUnretainedValue()
    let count = 24 * Int(width) * Int(height)
    bridge.continuation.yield(.latentPreview(H3LatentPreview(
        step: Int(step), steps: Int(steps), time: 1, height: Int(height), width: Int(width),
        latent: Array(UnsafeBufferPointer(start: latent, count: count)))))
}

public final class H3ImageEngine: @unchecked Sendable {
    /// As many references as an edit takes.
    public static let maxReferences = Int(H3_IMAGE_MAX_REFERENCES)
    /// A reference's longer side may be this many times its shorter one.
    public static let referenceMaxAspect = H3_IMAGE_REFERENCE_MAX_ASPECT

    /// The size a reference of this size is encoded at for the DiT; nil
    /// when its sides differ by more than `referenceMaxAspect`.
    /// `ph3`: P-H3-Image's rule (the picture's own size within 1 to 8
    /// megapixels; maxShortEdge is not used).
    public static func referenceCanvas(width: Int, height: Int, maxShortEdge: Int = 0,
                                       ph3: Bool = false) -> (width: Int, height: Int)? {
        var encodeW: Int32 = 0, encodeH: Int32 = 0, visionW: Int32 = 0, visionH: Int32 = 0
        let ok = ph3
            ? h3_image_reference_canvas_ph3(Int32(clamping: width), Int32(clamping: height),
                                            &encodeW, &encodeH, &visionW, &visionH)
            : h3_image_reference_canvas(Int32(clamping: width), Int32(clamping: height), Int32(clamping: maxShortEdge),
                                        &encodeW, &encodeH, &visionW, &visionH)
        guard ok != 0 else { return nil }
        return (Int(encodeW), Int(encodeH))
    }

    private let ctx: OpaquePointer
    public let modelDirectory: String
    private var currentCancelFlag: CancelFlag?
    /// Which image model the folder holds (h3_image_get_info).
    public let modelName: String
    /// Whether it is P-H3-Image (Pruna AI), not MiniMax H3 Image.
    public let isPH3: Bool
    public let steps: Int
    /// Whether it takes reference images (edits).
    public let canEdit: Bool
    /// As many references as this model's edits take (16, P-H3-Image 8).
    public let referenceLimit: Int

    /// Checks the folder (transformer/, vae/, text_encoder/, tokenizer/ and
    /// that the transformer is a still-image checkpoint); loads nothing yet.
    public init(modelDirectory: String) throws {
        var error = [CChar](repeating: 0, count: 512)
        guard let ctx = h3_image_load_dir(modelDirectory, "h3_shaders.metal", &error, error.count) else {
            throw H3EngineError.loadFailed(String(cString: error))
        }
        self.ctx = ctx
        self.modelDirectory = modelDirectory
        var info = h3_image_info()
        h3_image_get_info(ctx, &info)
        modelName = info.name.map { String(cString: $0) } ?? "MiniMax H3 Image"
        isPH3 = info.model == H3_IMAGE_MODEL_PH3
        steps = Int(info.steps)
        canEdit = info.can_edit != 0
        referenceLimit = info.max_references > 0 ? Int(info.max_references) : Self.maxReferences
    }

    deinit {
        h3_image_free(ctx)
    }

    /// The reference paths of `params` as the C structs a call takes; they
    /// are valid only inside `body`.
    private static func withReferences<T>(_ params: H3ImageParams,
                                          _ body: (UnsafePointer<h3_image_reference>?, Int32) -> T) -> T {
        let paths = params.references.map { strdup($0.path) }
        let latents: [UnsafeMutablePointer<CChar>?] = params.references.map { reference in
            reference.latentPath.flatMap { strdup($0) }
        }
        defer {
            paths.forEach { free($0) }
            latents.forEach { free($0) }
        }
        var references = (0 ..< params.references.count).map { index in
            h3_image_reference(path: UnsafePointer(paths[index]), latent_path: UnsafePointer(latents[index]))
        }
        return references.withUnsafeMutableBufferPointer { buffer in
            body(UnsafePointer(buffer.baseAddress), Int32(buffer.count))
        }
    }

    /// What generating `params` will take (h3_image_plan): the tokens of the
    /// text encoder and the DiT, and each stage's memory on this Mac. Reads
    /// the references' sizes and loads the tokenizer (a fraction of a
    /// second); call it off the main thread. nil when the request cannot be
    /// planned (a reference that cannot be read, say) - the generation then
    /// reports that itself. Safe while a generation runs.
    public func plan(_ params: H3ImageParams) -> H3GenerationPlan? {
        var cParams = h3_image_params()
        cParams.width = params.width
        cParams.height = params.height
        cParams.reference_max_short_edge = params.referenceMaxShortEdge
        // As generate() will set the environment (below).
        cParams.plan_text_prefetch_depth = getenv("H3_QWEN_PREFETCH_DEPTH") == nil ? 1 : 0
        var plan = h3_plan()
        let ctx = self.ctx
        let planned: Int32 = (params.prompt.isEmpty ? " " : params.prompt).withCString { prompt in
            cParams.prompt = prompt
            return Self.withReferences(params) { references, count in
                cParams.references = references
                cParams.reference_count = count
                return h3_image_plan(ctx, &cParams, &plan)
            }
        }
        return planned != 0 ? H3GenerationPlan(plan) : nil
    }

    /// One image. The stream ends after everything the generation loaded
    /// has been released, as the video engine's does.
    public func generate(_ params: H3ImageParams) -> AsyncThrowingStream<H3ImageEvent, Error> {
        AsyncThrowingStream { continuation in
            let cancelFlag = CancelFlag()
            self.currentCancelFlag = cancelFlag
            let bridge = ImageBridge(continuation: continuation)
            let bridgeHandle = Unmanaged.passRetained(bridge)

            H3Engine.workQueue.async { [self] in
                defer { bridgeHandle.release() }
                let ctx = self.ctx
                // As for a video: no App Nap while the user waits for it.
                let activity = ProcessInfo.processInfo.beginActivity(
                    options: [.userInitiated, .idleSystemSleepDisabled],
                    reason: "Generating image")
                defer { ProcessInfo.processInfo.endActivity(activity) }
                // What a video generation leaves in the environment names
                // the video model's files; none of it applies here.
                for name in ["H3_ATTENTION_CACHE", "H3_ATTENTION_CACHE_DIR", "H3_INT8_STREAM_MLP", "H3_ADALN_CURVE"] {
                    unsetenv(name)
                }
                // As for a video: the text encoder reads one layer ahead
                // (the default of three on an M5 holds 3.7 GiB more).
                if getenv("H3_QWEN_PREFETCH_DEPTH") == nil {
                    setenv("H3_QWEN_PREFETCH_DEPTH", "1", 1)
                }
                if cancelFlag.isCancelled {
                    continuation.finish(throwing: H3EngineError.cancelled)
                    return
                }
                var endStream: () -> Void = { continuation.finish() }
                autoreleasepool {
                    var cParams = h3_image_params()
                    cParams.width = params.width
                    cParams.height = params.height
                    cParams.seed = params.seed
                    cParams.cancel_requested = UnsafePointer(cancelFlag.engineFlag)
                    cParams.progress = h3ImageProgressTrampoline
                    cParams.progress_opaque = bridgeHandle.toOpaque()
                    if params.latentPreview { cParams.latent_preview = h3ImageLatentPreviewTrampoline }
                    var result = h3_image_result()
                    var error = [CChar](repeating: 0, count: 512)
                    NSLog("h3: image generation params - %dx%d, seed %llu, %d references", params.width, params.height,
                          params.seed, params.references.count)
                    cParams.reference_max_short_edge = params.referenceMaxShortEdge
                    cParams.init_strength = params.initStrength
                    let initPath = params.initImagePath.flatMap { strdup($0) }
                    defer { free(initPath) }
                    cParams.init_image_path = UnsafePointer(initPath)
                    let ok = params.prompt.withCString { prompt -> Bool in
                        cParams.prompt = prompt
                        return Self.withReferences(params) { references, count -> Bool in
                            cParams.references = references
                            cParams.reference_count = count
                            return h3_image_generate(ctx, &cParams, &result, &error, error.count) != 0
                        }
                    }
                    if ok {
                        let pixels = Int(result.width) * Int(result.height) * 3
                        let latentCount = 24 * Int(result.latent_width) * Int(result.latent_height)
                        let sigmas = withUnsafeBytes(of: &result.sigmas) { raw in
                            Array(raw.bindMemory(to: Float.self).prefix(Int(result.steps) + 1))
                        }
                        let widths = withUnsafeBytes(of: &result.reference_width) { Array($0.bindMemory(to: Int32.self)) }
                        let heights = withUnsafeBytes(of: &result.reference_height) { Array($0.bindMemory(to: Int32.self)) }
                        let fromLatent = withUnsafeBytes(of: &result.reference_from_latent) { Array($0.bindMemory(to: Int32.self)) }
                        let uses = (0 ..< Int(result.reference_count)).map {
                            H3ImageReferenceUse(width: Int(widths[$0]), height: Int(heights[$0]), fromLatent: fromLatent[$0] != 0)
                        }
                        let image = H3ImageResult(
                            width: Int(result.width), height: Int(result.height),
                            rgb: Data(bytes: result.rgb, count: pixels),
                            latent: Array(UnsafeBufferPointer(start: result.latent, count: latentCount)),
                            latentWidth: Int(result.latent_width), latentHeight: Int(result.latent_height),
                            seed: result.seed, promptTokens: Int(result.prompt_tokens),
                            steps: Int(result.steps), sigmas: sigmas,
                            textSeconds: result.text_seconds, denoiseSeconds: result.denoise_seconds,
                            decodeSeconds: result.decode_seconds,
                            references: uses, referenceSeconds: result.reference_seconds)
                        h3_image_result_free(&result)
                        NSLog("h3: image generation finished - references %.1f s, text %.1f s, denoise %.1f s, decode %.1f s",
                              image.referenceSeconds, image.textSeconds, image.denoiseSeconds, image.decodeSeconds)
                        continuation.yield(.finished(image))
                    } else {
                        let failure = cancelFlag.isCancelled
                            ? H3EngineError.cancelled : H3EngineError.generationFailed(String(cString: error))
                        endStream = { continuation.finish(throwing: failure) }
                    }
                }
                // After the pool has drained: what is still allocated on the
                // GPU once this generation's objects are gone.
                if let device = MTLCreateSystemDefaultDevice() {
                    NSLog("h3: device allocated after image generation: %.3f GiB",
                          Double(device.currentAllocatedSize) / 1_073_741_824)
                }
                endStream()
            }
        }
    }

    public func cancel() {
        currentCancelFlag?.cancel()
    }
}
