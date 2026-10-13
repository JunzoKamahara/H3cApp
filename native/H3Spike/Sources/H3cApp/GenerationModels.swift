import Foundation

// Size tier of the preset sizes (プロンプト欄の「大きさ」).
enum SizeTier: String, CaseIterable, Identifiable {
    case small, medium, large
    var id: String { rawValue }

    var label: String {
        switch self {
        case .small: return String(localized: "小")
        case .medium: return String(localized: "中")
        case .large: return String(localized: "大")
        }
    }
}

// What the DiT generates and what the video is delivered at: a preset
// (shape x 小/中/大 - the DiT canvas, doubled by the upscale) or a custom
// finished size from 詳細設定 (generated at half of it when upscaled),
// either one optionally upscaled 2x on the host after decoding (Accelerate's high-quality
// resampling in h3_resize_rgb24_high_quality). Stored as rawValue in
// drafts, presets and project records; the six names from before the
// 小/中/大 presets keep their meaning.
struct SizeProfile: Hashable, Identifiable {
    enum Base: Hashable {
        case preset(AspectShape, SizeTier)
        /// The finished (output) size.
        case custom(width: Int32, height: Int32)
    }

    var base: Base
    var upscaled: Bool

    var id: String { rawValue }

    // DiT canvases: the large ones stay within customMaxPixels like a custom
    // size must. 960x544, 768x768 and 1920x1088/1536x1536 upscaled were
    // measured on a 24 GB M5 (SPEEDUP_ROADMAP.md section 8).
    private static func presetCanvas(_ shape: AspectShape, _ tier: SizeTier) -> (width: Int32, height: Int32) {
        let landscape: (Int32, Int32)
        switch (shape, tier) {
        case (.square, .small): return (256, 256)
        case (.square, .medium): return (512, 512)
        case (.square, .large): return (768, 768)
        case (_, .small): landscape = (448, 256)
        case (_, .medium): landscape = (672, 384)
        case (_, .large): landscape = (960, 544)
        }
        return shape == .portrait ? (landscape.1, landscape.0) : landscape
    }

    /// The custom limit on the generated (DiT) canvas, before any upscale.
    static let customMaxPixels = 600_000
    /// Sides of the generated canvas.
    static let customCanvasSideRange: ClosedRange<Int32> = 256 ... 1344

    /// A custom finished size's sides: multiples of 32, or of 64 when
    /// upscaled (the generated canvas is half and must stay a multiple of 32).
    static func customSideStep(upscaled: Bool) -> Int32 { upscaled ? 64 : 32 }

    static func customSideRange(upscaled: Bool) -> ClosedRange<Int32> {
        let factor: Int32 = upscaled ? 2 : 1
        return customCanvasSideRange.lowerBound * factor ... customCanvasSideRange.upperBound * factor
    }

    /// Sides in range and on the step; the pixel limit is checked separately
    /// (isWithinCustomLimit) so an over-limit draft still loads and is flagged.
    static func isWellFormedCustom(width: Int32, height: Int32, upscaled: Bool) -> Bool {
        let range = customSideRange(upscaled: upscaled)
        let step = customSideStep(upscaled: upscaled)
        return range.contains(width) && range.contains(height)
            && width % step == 0 && height % step == 0
    }

    /// Whether a custom size is generated within customMaxPixels (presets
    /// always are).
    var isWithinCustomLimit: Bool { !isCustom || ditPixels <= Self.customMaxPixels }

    static let square = SizeProfile(base: .preset(.square, .medium), upscaled: false)

    static func preset(_ shape: AspectShape, _ tier: SizeTier, upscaled: Bool = false) -> SizeProfile {
        SizeProfile(base: .preset(shape, tier), upscaled: upscaled)
    }

    static func custom(width: Int32, height: Int32, upscaled: Bool) -> SizeProfile {
        SizeProfile(base: .custom(width: width, height: height), upscaled: upscaled)
    }

    /// Every preset (not custom sizes): the values the API lists.
    static var allCases: [SizeProfile] {
        AspectShape.allCases.flatMap { shape in
            SizeTier.allCases.flatMap { tier in
                [false, true].map { preset(shape, tier, upscaled: $0) }
            }
        }
    }

    var isCustom: Bool {
        if case .custom = base { return true }
        return false
    }

    var tier: SizeTier? {
        if case .preset(_, let tier) = base { return tier }
        return nil
    }

    /// The canvas the DiT runs at.
    var canvas: (width: Int32, height: Int32) {
        switch base {
        case .preset(let shape, let tier): return Self.presetCanvas(shape, tier)
        case .custom(let width, let height): return upscaled ? (width / 2, height / 2) : (width, height)
        }
    }

    /// The same size with the 2x upscale switched on or off. A preset keeps
    /// its canvas (so the output doubles); a custom size keeps its finished
    /// size, rounded to a multiple of 64 when switching the upscale on.
    func withUpscale(_ on: Bool) -> SizeProfile {
        guard case .custom(let width, let height) = base, on, !upscaled else {
            return SizeProfile(base: base, upscaled: on)
        }
        let range = Self.customSideRange(upscaled: true)
        func rounded(_ side: Int32) -> Int32 {
            min(max((side + 32) / 64 * 64, range.lowerBound), range.upperBound)
        }
        return .custom(width: rounded(width), height: rounded(height), upscaled: true)
    }

    var shape: AspectShape {
        switch base {
        case .preset(let shape, _): return shape
        case .custom(let width, let height):
            return width > height ? .landscape : width < height ? .portrait : .square
        }
    }

    /// Output size and, when upscaled, the lower internal render size
    /// (render 0 = generated directly at the output size).
    var dimensions: (width: Int32, height: Int32, renderWidth: Int32, renderHeight: Int32) {
        let canvas = canvas
        return upscaled ? (canvas.width * 2, canvas.height * 2, canvas.width, canvas.height)
                        : (canvas.width, canvas.height, 0, 0)
    }

    var ditPixels: Int { Int(canvas.width) * Int(canvas.height) }
    var outputPixels: Int { Int(dimensions.width) * Int(dimensions.height) }

    var resolutionLabel: String {
        let d = dimensions
        return "\(d.width)×\(d.height)"
    }

    var canvasLabel: String { "\(canvas.width)×\(canvas.height)" }

    var label: String {
        let name = isCustom ? String(localized: "カスタム") : shape.label
        let text = "\(name) \(resolutionLabel)"
        return upscaled ? String(localized: "\(text)（\(canvasLabel)から2倍）") : text
    }

    private static let legacyNames: [String: SizeProfile] = [
        "smallSquare": preset(.square, .small),
        "square": preset(.square, .medium),
        "landscapeNative": preset(.landscape, .medium),
        "landscapeUpscaled": preset(.landscape, .medium, upscaled: true),
        "portraitNative": preset(.portrait, .medium),
        "portraitUpscaled": preset(.portrait, .medium, upscaled: true),
    ]

    /// Legacy names for the sizes that had one, else
    /// "<shape>-<tier>[-x2]" or "custom-<W>x<H>[-x2]".
    var rawValue: String {
        if let legacy = Self.legacyNames.first(where: { $0.value == self }) { return legacy.key }
        let suffix = upscaled ? "-x2" : ""
        switch base {
        case .preset(let shape, let tier): return "\(shape.rawValue)-\(tier.rawValue)\(suffix)"
        case .custom(let width, let height): return "custom-\(width)x\(height)\(suffix)"
        }
    }

    init(base: Base, upscaled: Bool) {
        self.base = base
        self.upscaled = upscaled
    }

    init?(rawValue: String) {
        if let legacy = Self.legacyNames[rawValue] {
            self = legacy
            return
        }
        var parts = rawValue.split(separator: "-").map(String.init)
        let upscaled = parts.last == "x2"
        if upscaled { parts.removeLast() }
        if parts.count == 2, parts[0] == "custom" {
            let size = parts[1].split(separator: "x").compactMap { Int32($0) }
            guard size.count == 2,
                  Self.isWellFormedCustom(width: size[0], height: size[1], upscaled: upscaled) else { return nil }
            self = .custom(width: size[0], height: size[1], upscaled: upscaled)
        } else if parts.count == 2, let shape = AspectShape(rawValue: parts[0]),
                  let tier = SizeTier(rawValue: parts[1]) {
            self = .preset(shape, tier, upscaled: upscaled)
        } else {
            return nil
        }
    }
}

// "画面の形": the spec asks for a semantic ratio choice (横長/正方形/縦長)
// rather than raw dimensions at the top level - only the three shapes that
// SizeProfile actually offers, never a fabricated 16:9/9:16.
enum AspectShape: String, CaseIterable, Identifiable {
    case landscape, square, portrait
    var id: String { rawValue }

    var label: String {
        switch self {
        case .landscape: return String(localized: "横長")
        case .square: return String(localized: "正方形")
        case .portrait: return String(localized: "縦長")
        }
    }

    var systemImage: String {
        switch self {
        case .landscape: return "rectangle"
        case .square: return "square"
        case .portrait: return "rectangle.portrait"
        }
    }
}

// "作り方": text-to-video vs. starting from an image. A second choice
// (ImageInputMode) narrows which image-driven mode within "画像から".
enum CreationMethod: String, CaseIterable, Identifiable {
    case text, image
    var id: String { rawValue }

    var label: String {
        switch self {
        case .text: return String(localized: "文章から")
        case .image: return String(localized: "画像から")
        }
    }
}

// FL2VA (first/last frame) and Ref2VA (reference images/videos/audio) use
// different transformer checkpoints/caches and can't be mixed (see
// build_job() in gui/server.py). Ref2VA itself accepts still images,
// videos, and audio-only files as references (H3ReferenceKind in
// H3Engine.swift) - the case name predates video/audio support here and
// stays for compatibility.
enum ImageInputMode: String, CaseIterable, Identifiable {
    case firstLastFrame, referenceImage
    var id: String { rawValue }

    var label: String {
        switch self {
        case .firstLastFrame: return String(localized: "最初・最後の画像")
        case .referenceImage: return String(localized: "参照画像・動画・音声")
        }
    }
}

// How the DiT weights are served. The three are mutually exclusive in the
// engine, not just in the UI:
// - attentionCache: pre-quantized int8 cache streamed from disk (+ streamed
//   MLP). Fast, needs the cache file, an M5-class GPU (tensor ops gate the
//   int8 path, h3_gpu.m).
// - resident: every DiT block loaded fully into memory at once instead of
//   streamed - h3_dit.c's plain load_block() path, the same one
//   attentionCache falls back on when H3_ATTENTION_CACHE isn't set. Needs
//   no cache file (quantizes to int8 in place at load time on a tensor-
//   capable GPU, same ~28s cost as building a cache, just not saved to
//   disk; falls back to full BF16 residency on older GPUs), but needs enough RAM to hold that residency - meant for a Mac with
//   memory to spare, not the default choice.
// - ssdStreaming: the original BF16 checkpoint, two blocks resident at a
//   time. No cache or quantization needed and far less memory, but slower.
// LoRA stacks apply in all three (h3_lora.c patches whichever weights the
// mode loads or streams).
// Approximations the engine already implements and validated (h3.h):
// gate-ranked DiT block skipping (dit_layers), transformer-core reuse across
// steps (core_reuse) and horizontal token pairing in the middle blocks
// (token_reduction). See GenerationViewModel.speedSettings for the values
// and the measurements behind them.
enum SpeedMode: String, CaseIterable, Identifiable, Codable {
    case quality, fast, fastest
    var id: String { rawValue }

    var label: String {
        switch self {
        case .quality: return String(localized: "標準（高品質）")
        case .fast: return String(localized: "高速")
        case .fastest: return String(localized: "最速")
        }
    }

    var summaryLabel: String {
        switch self {
        case .quality: return String(localized: "標準")
        case .fast: return String(localized: "高速")
        case .fastest: return String(localized: "最速")
        }
    }
}

enum ComputeMode: String, CaseIterable, Identifiable, Codable {
    case attentionCache, resident, ssdStreaming
    var id: String { rawValue }

    var label: String {
        switch self {
        case .attentionCache: return String(localized: "高速（int8キャッシュ）")
        case .resident: return String(localized: "常駐（大容量メモリ向け、キャッシュ不要）")
        case .ssdStreaming: return String(localized: "省メモリ（SSDストリーミング）")
        }
    }

    var summaryLabel: String {
        switch self {
        case .attentionCache: return String(localized: "int8キャッシュ")
        case .resident: return String(localized: "常駐モード")
        case .ssdStreaming: return String(localized: "SSDストリーミング")
        }
    }
}

// Immutable snapshot of exactly what a completed generation used - kept
// separate from the live, still-editable draft in GenerationViewModel so
// showing "設定を見る" or reusing it never depends on (and is never
// clobbered by) whatever the user has since typed. See invariant #1/#10 in
// the design spec.
/// One adapter of the stack a generation actually used.
struct ResolvedLoRA: Equatable {
    let name: String
    let path: String
    let strength: Float
}

struct ResolvedResult {
    let prompt: String
    let creationMethod: CreationMethod
    let imageInputMode: ImageInputMode?
    let sizeProfile: SizeProfile
    let requestedSeconds: Int
    let requestedFrames: Int
    let actualFrameCount: Int
    let fps: Int
    var actualDurationSeconds: Double?
    let steps: Int
    let denoiseReuse: Int           // the form's setting
    let effectiveDenoiseReuse: Int  // what ran (1 under a speed preset)
    let ditLayers: Int
    let computeMode: ComputeMode
    let speedMode: SpeedMode
    // Requested (the checkbox), and what the engine reports actually ran.
    let fastAttention: Bool
    let ccvAttentionCalls: Int
    let ccvAttentionDirectCalls: Int
    let seed: UInt64
    let seedWasRandom: Bool
    let loras: [ResolvedLoRA]
    let sampling: GenerationViewModel.SamplingSettings
    /// File names of the images/videos/audio the mode used.
    let references: [String]
    let deviceLine: String
    let completedAt: Date
    // Wall-clock time from pressing generate to the finished file.
    let generationSeconds: Double

    var seedDecimalString: String { String(seed) }

    /// From the path that actually ran; a diagnostic environment override
    /// can route through ccv even with the checkbox off.
    var fastAttentionSummary: String {
        if ccvAttentionCalls == 0 {
            return fastAttention ? String(localized: "指定したが未使用") : String(localized: "使用しない")
        }
        let path = ccvAttentionDirectCalls == ccvAttentionCalls
            ? String(localized: "使用") : String(localized: "一部使用（ccv経路）")
        return fastAttention ? path : path + String(localized: "（診断用の環境変数による）")
    }

    var settingsSummaryText: String {
        let method = imageInputMode.map { String(localized: "\(creationMethod.label)（\($0.label)）") }
            ?? creationMethod.label
        let actual = actualDurationSeconds.map { String(format: String(localized: "%.1f秒"), $0) }
            ?? String(localized: "不明")
        var reuseLine = String(localized: "ノイズ除去の再利用（reuse）: \(effectiveDenoiseReuse)")
        if effectiveDenoiseReuse != denoiseReuse {
            reuseLine += String(localized: "（設定値 \(denoiseReuse)、速度プリセットのため1で計算）")
        }
        let seedLine = seedWasRandom
            ? String(localized: "シード: \(seedDecimalString)（毎回変える設定で決定）")
            : String(localized: "シード: \(seedDecimalString)（固定）")
        var lines = [
            String(localized: "動画の内容: \(prompt)"),
            String(localized: "作り方: \(method)"),
            String(localized: "画面の形: \(sizeProfile.label)"),
            String(localized: "長さ: 指定\(requestedSeconds)秒 / 実測\(actual)"),
            String(localized: "生成ステップ数: \(steps)"),
            reuseLine,
            String(localized: "使用する層数: \(ditLayers) / 50"),
            String(localized: "計算方式: \(computeMode.label)"),
            String(localized: "速度: \(speedMode.label)"),
            String(localized: "高速モード（試験的）: \(fastAttentionSummary)"),
            String(localized: "生成時間: \(formatElapsed(generationSeconds))"),
            seedLine,
        ]
        for lora in loras {
            let file = URL(fileURLWithPath: lora.path).lastPathComponent
            // As a String: a Float interpolated into a localized string
            // formats as %f ("0.800000").
            let strength = "\(lora.strength)"
            lines.append(lora.strength == 1
                ? String(localized: "追加モデル（LoRA）: \(file)")
                : String(localized: "追加モデル（LoRA）: \(file)（強さ \(strength)）"))
        }
        if sampling.renoise {
            lines.append(String(localized: "ステップの進め方: re-noise（LCM方式、追加モデルの設定）"))
        }
        if sampling.audioShift > 0 {
            // As a String, like the LoRA strength above.
            let shift = "\(sampling.audioShift)"
            lines.append(String(localized: "音声のシフト値: \(shift)（追加モデルの設定）"))
        }
        if !references.isEmpty {
            lines.append(String(localized: "参照: \(references.joined(separator: ", "))"))
        }
        lines.append(String(localized: "動作環境: \(deviceLine)"))
        return lines.joined(separator: "\n")
    }
}
