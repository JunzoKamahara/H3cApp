import CH3
import Foundation
import Metal

// Still images with MiniMax H3 Image (include/h3_image.h): the Swift side
// of the image engine, apart from H3Engine (the video one) as the C APIs
// are. It runs on H3Engine's serial work queue, so an image never overlaps
// a video generation, a cache build or a RefMod extraction.

public struct H3ImageParams: Sendable {
    public var prompt: String
    /// Pixels, both multiples of 32.
    public var width: Int32
    public var height: Int32
    public var seed: UInt64

    public init(prompt: String, width: Int32, height: Int32, seed: UInt64) {
        self.prompt = prompt
        self.width = width
        self.height = height
        self.seed = seed
    }
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
}

public enum H3ImageEvent: Sendable {
    /// phase: "text encoder", "image model", "denoise" or "decode".
    case progress(phase: String, completed: Int, total: Int)
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

public final class H3ImageEngine: @unchecked Sendable {
    private let ctx: OpaquePointer
    public let modelDirectory: String
    private var currentCancelFlag: CancelFlag?

    /// Checks the folder (transformer/, vae/, text_encoder/, tokenizer/ and
    /// that the transformer is a still-image checkpoint); loads nothing yet.
    public init(modelDirectory: String) throws {
        var error = [CChar](repeating: 0, count: 512)
        guard let ctx = h3_image_load_dir(modelDirectory, "h3_shaders.metal", &error, error.count) else {
            throw H3EngineError.loadFailed(String(cString: error))
        }
        self.ctx = ctx
        self.modelDirectory = modelDirectory
    }

    deinit {
        h3_image_free(ctx)
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
                    var result = h3_image_result()
                    var error = [CChar](repeating: 0, count: 512)
                    NSLog("h3: image generation params - %dx%d, seed %llu", params.width, params.height, params.seed)
                    let ok = params.prompt.withCString { prompt -> Bool in
                        cParams.prompt = prompt
                        return h3_image_generate(ctx, &cParams, &result, &error, error.count) != 0
                    }
                    if ok {
                        let pixels = Int(result.width) * Int(result.height) * 3
                        let latentCount = 24 * Int(result.latent_width) * Int(result.latent_height)
                        let sigmas = withUnsafeBytes(of: &result.sigmas) { raw in
                            Array(raw.bindMemory(to: Float.self).prefix(Int(result.steps) + 1))
                        }
                        let image = H3ImageResult(
                            width: Int(result.width), height: Int(result.height),
                            rgb: Data(bytes: result.rgb, count: pixels),
                            latent: Array(UnsafeBufferPointer(start: result.latent, count: latentCount)),
                            latentWidth: Int(result.latent_width), latentHeight: Int(result.latent_height),
                            seed: result.seed, promptTokens: Int(result.prompt_tokens),
                            steps: Int(result.steps), sigmas: sigmas,
                            textSeconds: result.text_seconds, denoiseSeconds: result.denoise_seconds,
                            decodeSeconds: result.decode_seconds)
                        h3_image_result_free(&result)
                        NSLog("h3: image generation finished - text %.1f s, denoise %.1f s, decode %.1f s",
                              image.textSeconds, image.denoiseSeconds, image.decodeSeconds)
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
