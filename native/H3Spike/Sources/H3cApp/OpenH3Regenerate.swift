import Foundation
import H3Engine

// "Open-H3 再生成": enlarging a generated clip by regenerating it with H3
// itself - its latent upscaled onto the larger grid (LBH's 3D-conv latent
// upscaler), re-noised by the strength chosen (OpenH3Strength) and denoised
// in a few steps with the Open-H3 refiner LoRA and LightX2V Turbo v1.2, then
// decoded (NVIDIA's Open-H3-Regenerate-2K recipe without the LMS Sharp parts;
// UPSCALE_PLAN appendices A.4 and A.8). The DiT draws the new detail on its
// own canvas, so the result is at most the released 1344x768.

/// How much of the clip the regeneration redraws: the noise it goes back to
/// and the steps down from there. The rawValue is the API's and the record's.
enum OpenH3Strength: String, CaseIterable, Identifiable {
    /// Keeps the composition and the characters (appendix A.8).
    case light
    /// The recipe's two-step schedule: keeps the composition, not always
    /// the expressions.
    case medium
    /// The recipe's three steps: the most new detail, and at every factor
    /// tried it came back as another picture (another face, an anime clip
    /// turned live action).
    case strong

    static let standard = OpenH3Strength.light

    var id: String { rawValue }

    var sigmas: [Float] {
        switch self {
        case .light: return [0.4, 0.2, 0]
        case .medium: return [0.7, 0.494118, 0]
        case .strong: return [0.909375, 0.857143, 0.666667, 0]
        }
    }

    var label: String {
        switch self {
        case .light: return String(localized: "弱い（σ 0.4）")
        case .medium: return String(localized: "中（σ 0.7）")
        case .strong: return String(localized: "強い（σ 0.909）")
        }
    }

    var detail: String {
        switch self {
        case .light: return String(localized: "人物と構図を保ち、細かな質感を描き直します。描き足しは控えめです。")
        case .medium: return String(localized: "構図は保ちますが、表情や顔つきが変わることがあります。")
        case .strong: return String(localized: "配布元の手順です。描き足しは最も多く、別の人物や別の絵になることがあります（アニメが実写になることも）。時間は弱い・中の約1.3倍です。")
        }
    }

    /// Time against the two-step strengths (appendix A.8: 672x384 to
    /// 1344x768, 56 frames, 5:29 for the three steps and 4:19 for two).
    var timeScale: Double { sigmas.count > 3 ? 329.0 / 259.0 : 1 }
}

/// The size the regeneration makes and the latent grid it runs on.
struct OpenH3Plan: Equatable {
    let width: Int
    let height: Int
    var latentWidth: Int { width / 16 }
    var latentHeight: Int { height / 16 }

    /// The released DiT canvas limit (h3.c H3_MAX_PIXELS).
    static let maxPixels = 768 * 1344

    /// `width` x `height` enlarged by `factor`: multiples of 32 at the
    /// nearest size, brought down along the aspect to the canvas limit.
    /// Nil when that is no enlargement (the source fills the canvas),
    /// except for 等倍, which redraws at the video's own size.
    static func plan(width: Int, height: Int, factor: UpscaleFactor) -> OpenH3Plan? {
        guard width > 0, height > 0 else { return nil }
        func round32(_ value: Double) -> Int { max(32, Int((value / 32).rounded()) * 32) }
        var scale = factor.rawValue
        var plan = OpenH3Plan(width: round32(Double(width) * scale), height: round32(Double(height) * scale))
        while plan.width * plan.height > maxPixels, scale > 1.001 {
            scale -= 0.005
            plan = OpenH3Plan(width: round32(Double(width) * scale), height: round32(Double(height) * scale))
        }
        guard plan.width * plan.height <= maxPixels, plan.width >= width, plan.height >= height,
              factor == .same || plan.width > width || plan.height > height else { return nil }
        return plan
    }

    /// About how long on an M5 (appendix A.8: 318 s for 56 frames at
    /// 1344x768 in int8-cache mode), scaled by frames, pixels and steps.
    func estimatedSeconds(frames: Int, strength: OpenH3Strength) -> Double {
        318 * Double(max(frames, 1)) / 56 * Double(width * height) / Double(Self.maxPixels) * strength.timeScale
    }
}

extension GenerationViewModel {
    /// Why `video` can't be regenerated with Open-H3 (nil when it can): it
    /// needs the clip's own latent at the clip's full length (a joined
    /// video keeps only its last clip's), the H3 model loaded, and a size
    /// below the canvas.
    func openH3Reason(for video: ProjectVideo, size: (width: Int, height: Int)? = nil,
                      factor: UpscaleFactor) -> String? {
        guard let record = video.record else {
            return String(localized: "Open-H3 再生成は、このアプリで生成した動画（記録と潜在ファイルのあるもの）にだけ使えます。")
        }
        guard record.upscaledFrom == nil || continuationLatent(of: video) != nil else {
            return String(localized: "拡大した動画は、その元の動画から再生成してください。")
        }
        guard let latentURL = continuationLatent(of: video) else {
            return String(localized: "この動画の潜在ファイル（.h3lt）がありません。Open-H3 再生成は、生成したときの潜在を描き直すので、潜在のない動画には使えません。")
        }
        guard let info = try? H3LatentFileInfo.read(path: latentURL.path) else {
            return String(localized: "潜在ファイルを読めませんでした。")
        }
        if info.frames != record.actualFrameCount {
            return String(localized: "潜在ファイルの長さ（\(info.frames)フレーム）が動画（\(record.actualFrameCount)フレーム）と違います。つないだ動画は、つなぐ前の各クリップを再生成してください。")
        }
        // The factor applies to the video as seen (a ×2 size's video is
        // twice its latent); the latent is upscaled onto whatever grid that
        // makes.
        let size = size ?? recordedSize(of: record) ?? (info.pixelWidth, info.pixelHeight)
        guard OpenH3Plan.plan(width: size.width, height: size.height, factor: factor) != nil else {
            return String(localized: "この動画（\("\(size.width)×\(size.height)")）は、Open-H3の上限（1344×768ピクセル）までもう大きくできません。")
        }
        guard engine != nil, case .ready = engineState else {
            return String(localized: "H3のモデルが読み込まれていません。Open-H3 再生成はH3本体で描き直します。")
        }
        return nil
    }

    /// The size a record says its video has (an enlarged copy's, else its
    /// size profile's output size).
    func recordedSize(of record: ProjectVideoRecord) -> (width: Int, height: Int)? {
        if let size = record.upscaledSize, size.count == 2 { return (size[0], size[1]) }
        guard let dimensions = SizeProfile(rawValue: record.draft.sizeProfile)?.dimensions else { return nil }
        return (Int(dimensions.width), Int(dimensions.height))
    }

    /// The regeneration of `video` into `destination`: the latent upscaled,
    /// then a short generation from it at `strength`. Returns the size made.
    func runOpenH3Regeneration(_ video: ProjectVideo, to destination: URL, factor: UpscaleFactor,
                               strength: OpenH3Strength) async throws -> (width: Int, height: Int) {
        let size = try await VideoUpscale.frameSize(of: video.url)
        guard let engine, let record = video.record, let latentURL = continuationLatent(of: video),
              let latentInfo = try? H3LatentFileInfo.read(path: latentURL.path),
              let plan = OpenH3Plan.plan(width: size.width, height: size.height, factor: factor) else {
            throw VideoUpscale.UpscaleError.unavailable(openH3Reason(for: video, size: size, factor: factor) ?? "Open-H3")
        }
        guard OpenH3Model.ready, let turboPath = OpenH3Model.turboPath else {
            throw VideoUpscale.UpscaleError.unavailable(VideoUpscaler.openH3.unavailableReason ?? "Open-H3")
        }
        upscalingVideoName = video.url.lastPathComponent
        upscalingFactor = factor
        upscaleProgress = 0
        upscaleCancel.reset()
        isUpscalingVideo = true
        defer { isUpscalingVideo = false }
        let cancel = upscaleCancel

        // 1. The latent onto the larger grid (a few seconds to half a minute);
        // at its own size (等倍) it goes straight to the steps below.
        let upscaledLatent = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("H3cApp_openh3_\(Int(Date().timeIntervalSince1970)).h3lt")
        defer { try? FileManager.default.removeItem(at: upscaledLatent) }
        let upsamplerDirectory = OpenH3Model.upsamplerDirectory
        let latentPath = latentURL.path, outPath = upscaledLatent.path
        let sameGrid = latentInfo.latentHeight == plan.latentHeight && latentInfo.latentWidth == plan.latentWidth
        if !sameGrid {
            try await Task.detached(priority: .userInitiated) {
                let upscaler = try H3LatentUpscaler(weightsDirectory: upsamplerDirectory)
                try upscaler.upscaleFile(latentPath, to: outPath, latentHeight: plan.latentHeight,
                                         latentWidth: plan.latentWidth) { done, total in
                    Task { @MainActor in self.upscaleProgress = 0.05 * Double(done) / Double(max(total, 1)) }
                    return !cancel.isRequested
                }
            }.value
        }
        if cancel.isRequested { throw VideoUpscale.UpscaleError.cancelled }

        // 2. The denoising steps from it, with the refiner and Turbo.
        var params = H3GenerationParams()
        params.width = Int32(plan.width)
        params.height = Int32(plan.height)
        params.frames = Int32(record.actualFrameCount)
        params.steps = Int32(strength.sigmas.count - 1)
        params.denoiseReuse = 1
        params.initialLatentPath = sameGrid ? latentURL.path : upscaledLatent.path
        params.videoSigmas = strength.sigmas
        params.loras = [H3LoRAInput(path: OpenH3Model.refinerPath, strength: 1),
                        H3LoRAInput(path: turboPath, strength: 1)]
        params.seed = UInt64(record.seed) ?? 1
        params.ssdStreaming = computeMode == .ssdStreaming
        // Always the FL2VA transformer: the latent carries the picture, so
        // no references go in, whatever made the clip.
        params.attentionCachePath = computeMode == .attentionCache ? attentionCachePath(usesReferences: false) : nil
        params.adalnCurveDirectory = preferences.useAdaLNCurve
            ? library.activeModel.map { modelCacheDirectory(path: $0.path, id: $0.id) } : nil
        params.fastAttention = useFastAttention
        params.videoHEVC = true
        params.videoQuality = Float(VideoQuality.saved.value)
        let prompt = record.draft.prompt
        let outputPath = destination.path
        var sawDenoise = false
        for try await event in engine.generate(prompt: prompt, outputPath: outputPath, params: params) {
            if cancel.isRequested { engine.cancelCurrentGeneration() }
            guard case .progress(let phase, let completed, let total) = event else { continue }
            // Setup 5-10%, the steps 10-85%, decoding 85-95%, the file 95-100%.
            let share: Double?
            switch phase {
            case "denoise", "denoise enqueue":
                sawDenoise = true
                share = 0.10 + 0.75 * Double(completed) / Double(max(total, 1))
            case "audio VAE", "video VAE load": share = 0.85 + 0.10 * Double(completed) / Double(max(total, 1))
            case "encode": share = 0.95 + 0.05 * Double(completed) / Double(max(total, 1))
            default: share = sawDenoise ? nil : 0.05 + 0.05 * Double(completed) / Double(max(total, 1))
            }
            if let share { upscaleProgress = max(upscaleProgress, min(share, 1)) }
        }
        if cancel.isRequested { throw VideoUpscale.UpscaleError.cancelled }
        return (plan.width, plan.height)
    }
}
