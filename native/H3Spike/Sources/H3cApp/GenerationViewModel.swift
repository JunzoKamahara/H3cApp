import AppKit
import AVFoundation
import Combine
import Foundation
import H3Engine

let secondsRange = 1 ... 15
let stepsRange = 3 ... 40
// h3.c rejects anything outside [1, 3] ("denoise reuse must be in [1, 3]");
// gui/server.py's looser 1-6 check just let the engine reject 4+ later.
let reuseRange = 1 ... 3
// The real, existing engine defaults (H3GenerationParams / H3_DEFAULT_STEPS)
// - used only to detect whether "詳細設定" has been changed from them, not
// as a claim that these are the only valid values.
private let defaultSteps = 20
// 2 = the validated fast path (h3.h), as the old web GUI defaulted to; 1 is
// the close-reference path. Also the HTTP API's default when "reuse" is
// omitted.
let defaultReuse = 2
// h3.h: H3_MIN_DIT_LAYERS ... H3_DEFAULT_DIT_LAYERS. Fewer than 50 drops the
// lowest-gate DiT blocks; 45 was found to break the audio (2026-10-01).
let ditLayersRange = 35 ... 50
let defaultDitLayers = 50
let defaultSizeProfile: SizeProfile = .square

extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}

// Mirrors DEFAULT_ATTENTION_CACHE / REF2VA_ATTENTION_CACHE in gui/server.py,
// but under this user's Application Support instead of a dev checkout path -
// these 19GB caches can't be bundled into a distributable .app, and a path
// under a specific developer's home directory would never resolve on any
// other machine. Each registered model gets its own cache subdirectory (see
// GenerationViewModel.attentionCacheDirectory(for:)) now that ModelLibrary
// allows more than one - validationMessage below just reports a cache
// missing until one is built with build_attention_cache, in-app, or dropped
// in by hand.
// Not private: ModelLibrary.swift (same module, different file) also needs
// this base path to build the default download destination it suggests to
// brand-new installs - see defaultH3ModelDownloadPath there.
let h3AppSupportDirectory: String = {
    let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first?.path
        ?? (NSHomeDirectory() + "/Library/Application Support")
    let current = base + "/H3cApp"
    migrateAppSupportDirectory(from: base + "/h3c-app", to: current)
    return current
}()

/// Up to v0.5 the folder was named after the repository's old name,
/// h3c-app. Renamed once to H3cApp (same disk, so instant), with the paths
/// saved under it - registered models, LoRAs, anything else in this app's
/// settings - rewritten to match. Runs before ModelLibrary reads them (see
/// H3cApp.init).
private func migrateAppSupportDirectory(from legacy: String, to current: String) {
    let fm = FileManager.default
    guard fm.fileExists(atPath: legacy), !fm.fileExists(atPath: current) else { return }
    do {
        try fm.moveItem(atPath: legacy, toPath: current)
    } catch {
        NSLog("H3cApp: could not rename %@ to %@: %@", legacy, current, error.localizedDescription)
        return
    }
    guard let domain = Bundle.main.bundleIdentifier,
          let settings = UserDefaults.standard.persistentDomain(forName: domain) else { return }
    func rewrite(_ value: Any) -> Any {
        switch value {
        case let text as String:
            return text == legacy ? current : text.hasPrefix(legacy + "/") ? current + text.dropFirst(legacy.count) : text
        case let list as [Any]: return list.map(rewrite)
        case let table as [String: Any]: return table.mapValues(rewrite)
        default: return value
        }
    }
    for (key, value) in settings {
        // Lists saved as JSON data (the model and LoRA libraries).
        if let data = value as? Data, let json = try? JSONSerialization.jsonObject(with: data),
           json is [Any] || json is [String: Any] {
            let updated = rewrite(json)
            if !(updated as AnyObject).isEqual(json),
               let encoded = try? JSONSerialization.data(withJSONObject: updated) {
                UserDefaults.standard.set(encoded, forKey: key)
            }
        } else if value is String || value is [Any] || value is [String: Any] {
            let updated = rewrite(value)
            if !(updated as AnyObject).isEqual(value) { UserDefaults.standard.set(updated, forKey: key) }
        }
    }
}
// Paired with legacyDefaultH3ModelPath (ModelLibrary.swift) - moved
// alongside it, from h3AppSupportDirectory + "/cache" to ~/models/cache, so
// this dev's ~36GB of caches live next to the model instead of in
// ~/Library.
private let legacyCacheDirectory = NSHomeDirectory() + "/models/cache"

/// Where a model's int8 caches and AdaLN curves live (see
/// GenerationViewModel.attentionCacheDirectory(for:)).
func modelCacheDirectory(path: String, id: UUID) -> String {
    if path == legacyDefaultH3ModelPath, FileManager.default.fileExists(atPath: legacyCacheDirectory) {
        return legacyCacheDirectory
    }
    return h3AppSupportDirectory + "/cache/" + id.uuidString
}

enum EngineState: Equatable {
    case loading
    case ready
    case failed(String)
}

@MainActor
final class GenerationViewModel: ObservableObject {
    // MARK: Engine readiness (separate from job state and results - design
    // spec invariant: "エンジン準備状態・生成ジョブ状態・前回結果は別々に保持する")
    @Published private(set) var engineState: EngineState = .loading
    @Published private(set) var deviceLine: String = String(localized: "モデルを準備しています…")
    @Published private(set) var modelDirectory: String = ""
    let library = ModelLibrary()

    // MARK: Draft - editable settings, never overwritten by a running job or
    // a past result (design spec invariant #1).
    // Starts empty: the composer shows a suggestion in grey that Tab turns
    // into real text.
    @Published var prompt: String = ""
    @Published var creationMethod: CreationMethod = .text
    @Published var imageInputMode: ImageInputMode = .firstLastFrame
    @Published var sizeProfile: SizeProfile = defaultSizeProfile
    @Published var seconds: Int = 5
    @Published var steps: Int = defaultSteps
    // The user's reuse setting. A speed preset runs with reuse 1 instead
    // (effectiveDenoiseReuse) without overwriting this, so going back to
    // 標準 uses it again.
    @Published var denoiseReuse: Int = defaultReuse
    @Published var ditLayers: Int = defaultDitLayers
    // See ComputeMode: cache, resident and SSD streaming are mutually
    // exclusive; LoRA works with all three.
    @Published var computeMode: ComputeMode = .attentionCache
    // The int8 path (and so the cache) needs the Metal 4 hardware tensor
    // units first shipped in M5 - h3_gpu.m enables tensor ops based on the
    // device's actual supportsFamily: capability (H3DeviceInfo.hasTensorHardware),
    // not its marketing name. Set from the real device in loadModel().
    @Published private(set) var supportsInt8Cache = true
    var defaultComputeMode: ComputeMode { supportsInt8Cache ? .attentionCache : .ssdStreaming }
    @Published var speedMode: SpeedMode = .quality
    /// Curve-form AdaLN (H3GenerationParams.adalnCurveDirectory): ~10 s
    /// faster setup per video in every compute mode, values as close to
    /// exact as the dense path but not bit-identical, so the same seed gives
    /// a slightly different video than the dense path. An app-wide
    /// preference, not part of a draft; on by default.
    @Published var useAdaLNCurve: Bool =
        UserDefaults.standard.object(forKey: "h3c-app.useAdaLNCurve") as? Bool ?? true {
        didSet { UserDefaults.standard.set(useAdaLNCurve, forKey: "h3c-app.useAdaLNCurve") }
    }
    /// 途中の様子: show what the video is turning into while it denoises
    /// (DenoisePreview.swift). An app-wide preference; off until turned on,
    /// since it needs the 22.7 MB decoder downloaded first.
    @Published var showDenoisePreview: Bool =
        UserDefaults.standard.bool(forKey: "h3c-app.showDenoisePreview") {
        didSet { UserDefaults.standard.set(showDenoisePreview, forKey: "h3c-app.showDenoisePreview") }
    }
    /// The latest preview picture of the running generation and its step.
    @Published private(set) var denoisePreviewImage: CGImage?
    @Published private(set) var denoisePreviewStep: Int = 0
    @Published private(set) var denoisePreviewSteps: Int = 0
    let denoisePreviewDownloader = DenoisePreviewDownloader()
    private let denoisePreviewRenderer = DenoisePreviewRenderer()
    /// On, and the decoder is there.
    var denoisePreviewActive: Bool { showDenoisePreview && DenoisePreviewWeights.isPresent }
    /// Reuse actually sent to the engine: the 高速/最速 presets rely on core
    /// reuse, which the engine won't combine with denoiser reuse, and were
    /// measured at reuse 1 - so they run at 1 whatever denoiseReuse holds.
    /// The re-noise step rule evaluates every step, so it runs at 1 too.
    var effectiveDenoiseReuse: Int {
        speedMode == .quality && !samplingSettings.renoise ? denoiseReuse.clamped(to: reuseRange) : 1
    }

    struct SamplingSettings: Equatable {
        var renoise = false
        /// 0 = the engine's default audio shift (3).
        var audioShift: Float = 0
    }

    /// The step rule and audio shift for the next generation: what the
    /// enabled LoRAs ask for - each entry's own setting in モデル管理, else
    /// what its file's metadata declares (DMAD records "re-noise step rule"
    /// and "audio shift 2") - else the engine's Euler rule and default
    /// shift. Re-noise wins if any enabled LoRA asks for it; the first
    /// audio shift in stack order wins. An API request can override either
    /// for one generation (A/B tests).
    var samplingSettings: SamplingSettings {
        var settings = SamplingSettings()
        for entry in library.enabledLoRAs {
            let wanted = library.sampling(for: entry)
            if wanted.renoise { settings.renoise = true }
            if settings.audioShift == 0, let shift = wanted.audioShift { settings.audioShift = shift }
        }
        if let override = samplingOverride {
            if let renoise = override.renoise { settings.renoise = renoise }
            if let shift = override.audioShift { settings.audioShift = shift }
        }
        return settings
    }

    /// Set by POST /api/generate's "sampler"/"audio_shift" and cleared once
    /// that generation's request is built; nil fields keep the LoRA's choice.
    struct SamplingOverride {
        var renoise: Bool?
        var audioShift: Float?
    }
    var samplingOverride: SamplingOverride?
    // Opt-in fast mode (experimental): ccv's int8 attention, M5 only. Off by
    // default, offered only where the engine reports it can run, and kept
    // separate from speedMode - no speed preset turns it on.
    @Published var fastAttention = false
    // The 詳細設定 dialog, opened from the composer and the app menu (⌘,).
    @Published var showingAdvancedSettings = false
    // Settings presets live in PresetStore (shared by all windows); this
    // window's active one, and one waiting for a name in the save dialog.
    @Published var activePresetID: UUID?
    @Published var presetPendingName: SettingsPreset?
    let fastAttentionAvailable = H3Engine.fastAttentionAvailable

    struct SpeedSettings {
        var ditLayers: Int32 = 50
        var coreReuse: Int32 = 1
        var tokenReduction = false
    }

    /// The engine options behind speedMode for the current draft. The
    /// presets keep all 50 DiT blocks: dropping to 45 (the earlier preset,
    /// judged on video only) turned the audio into loud broadband noise with
    /// tonal bands - on 2026-10-01 (5 s, seed 7) 45 layers alone reproduced
    /// it, while core reuse, token reduction and denoiser reuse 2 didn't add
    /// such noise. DiT-phase time there (512x512, 124 frames, 20 steps, not
    /// the whole generation): 557 s standard, 204 s with core reuse 4 (高速,
    /// ~2.7x), 189 s adding token reduction (最速, ~2.9x).
    /// Stacking the engine's most aggressive values (40 layers, core reuse
    /// 6, token reduction, int8 row FC2) visibly smeared the subject.
    ///
    /// Core reuse refreshes the transformer core only every N steps, so it's
    /// scaled to the step count (a 4-step Turbo run has nothing to reuse
    /// across) and dropped when the separate whole-velocity reuse is on -
    /// the engine rejects combining the two.
    var speedSettings: SpeedSettings {
        var settings = SpeedSettings()
        // The layer count is its own advanced setting, not part of a preset.
        settings.ditLayers = Int32(ditLayers.clamped(to: ditLayersRange))
        guard speedMode != .quality else { return settings }
        settings.coreReuse = Int32(max(1, min(4, steps.clamped(to: stepsRange) / 5)))
        settings.tokenReduction = speedMode == .fastest
        return settings
    }
    // Whether the seed is pinned is its own state, not "seedText is
    // non-empty": the field must survive being cleared while typing.
    @Published var seedFixed = false {
        didSet {
            // Switching to 固定する starts from the seed actually used last
            // (also when that one was random), not from 0.
            if seedFixed && !oldValue, let last = Self.lastUsedSeed {
                seedText = String(last)
            }
        }
    }
    @Published var seedText: String = GenerationViewModel.lastUsedSeed.map(String.init) ?? "" {
        didSet {
            let digitsOnly = seedText.filter(\.isNumber)
            if digitsOnly != seedText { seedText = digitsOnly }
        }
    }
    /// The seed the last generation actually ran with, kept across launches.
    static var lastUsedSeed: UInt64? {
        get {
            (UserDefaults.standard.object(forKey: "h3c-app.lastUsedSeed") as? String)
                .flatMap { UInt64($0) }
        }
        set {
            UserDefaults.standard.set(newValue.map(String.init), forKey: "h3c-app.lastUsedSeed")
        }
    }
    // Not cleared on mode switches - design spec section 6: "モード切替では
    // 画像を消さない...画像からに戻すと復元する。" generate() below decides
    // which of these actually reach the engine based on the *current* mode.
    @Published var firstFramePath: String?
    @Published var lastFramePath: String?
    @Published var referenceImages: [H3ReferenceInput] = []
    /// Lip sync (h3_params.forced_audio_path): this file's audio, from
    /// forcedAudioOffset seconds on, is the video's soundtrack, and only the
    /// picture is generated - to match it. Works with every creation method.
    @Published var forcedAudioPath: String?
    @Published var forcedAudioOffset: Double = 0
    /// この動画の続きを作る (Continuation.swift): the project video the next
    /// generation carries on from.
    @Published var continuation: ContinuationSource?
    @Published var isExportingJoined = false
    /// つなげて生成 (ChainGeneration.swift): a length of 15-90 s made by
    /// joining clips of chainClipLength.
    @Published var chained = false
    @Published var chainClipLength: ChainClipLength = defaultChainClipLength
    /// What each clip of a chain adds to the prompt (chainClipPrompt).
    @Published var clipPrompts: [String] = []
    /// Whether each clip of a chain uses the reference images (missing:
    /// yes) - off for a scene the people don't appear in.
    @Published var clipReferences: [Bool] = []
    /// カスタムクリップ (CustomClips.swift): while chainCustom, a chained
    /// length is made of customClips.
    @Published var chainCustom = false
    @Published var customClips: [ChainClipSpec] = []
    /// The pointer is over the floating composer: the preview's pinch and
    /// scroll-wheel zoom leave it alone (ContentView, ResultPlayerView).
    var isPointerOverComposer = false
    /// 動画ファイルの続きを作る is converting a video (ExternalContinuation).
    @Published var isImportingVideo = false
    var importingVideoName = ""
    /// A chosen video waiting for its size to be picked (ImportVideoSheet).
    @Published var pendingVideoImport: PendingVideoImport?
    /// The form before カスタムクリップ's editor opened (its キャンセル).
    var clipEditSnapshot: ClipEditSnapshot?
    /// The つなげて生成 running now: kept past a cancel, so the clips made so
    /// far are still joined.
    var chainRun: ChainRun?
    /// 続きを生成 on a video outside any project: continue it once
    /// プロジェクトとして保存 has moved it into the new project.
    var continueAfterSavingAsProject = false

    /// The LoRA stack is chosen from ModelLibrary's registered library
    /// (entries switched on there or in the form) and applies in every
    /// compute mode.
    var effectiveLoRAs: [ResolvedLoRA] {
        library.enabledLoRAs.map { ResolvedLoRA(name: $0.name, path: $0.path, strength: $0.strength) }
    }

    /// Where the active model's attention cache files live. Isolated per
    /// registered model (by id) so switching models never risks streaming
    /// one model's cache against another's weights. The one exception: the
    /// very first model this app ever used didn't have this isolation and
    /// already has real ~19 GiB caches built at the flat legacy path -
    /// reuse that path for exactly that entry so upgrading doesn't demand
    /// rebuilding them.
    private func attentionCacheDirectory(for model: H3ModelEntry) -> String {
        modelCacheDirectory(path: model.path, id: model.id)
    }

    /// The int8 cache this job would use - which one depends on whether it
    /// runs the Ref2VA (reference image) transformer, and on the active
    /// model's own cache directory.
    var currentAttentionCachePath: String {
        attentionCachePath(usesReferences: creationMethod == .image && imageInputMode == .referenceImage)
    }

    func attentionCachePath(usesReferences: Bool) -> String {
        guard let model = library.activeModel else { return "" }
        let directory = attentionCacheDirectory(for: model)
        return directory + (usesReferences ? "/dit_int8_v2_ref2va.cache" : "/dit_int8_v2.cache")
    }

    /// The transformer directory currentAttentionCachePath would be built
    /// from - same reference-mode check, matching build_attention_cache's
    /// own FL2VA/Ref2VA convention (h3_dit.c's detect_model_kind()).
    var currentTransformerDirectory: String {
        let usesReferences = creationMethod == .image && imageInputMode == .referenceImage
        return modelDirectory + (usesReferences ? "/Ref2VA/transformer" : "/FL2VA/transformer")
    }

    var attentionCacheMissing: Bool {
        !FileManager.default.fileExists(atPath: currentAttentionCachePath)
    }

    /// What a generation started now would actually request.
    var useFastAttention: Bool { fastAttention && fastAttentionAvailable }

    var hasAdvancedChanges: Bool {
        sizeProfile != defaultSizeProfile || steps != defaultSteps || denoiseReuse != defaultReuse
            || ditLayers != defaultDitLayers
            || computeMode != defaultComputeMode || speedMode != .quality || fastAttention
            || seedFixed
            || !library.enabledLoRAs.isEmpty
            || chainClipLength != defaultChainClipLength
    }

    /// Max-resolution + long duration + SSD streaming measured as
    /// impractically slow on a 24GB M5 Mac: the run pushed past physical
    /// memory into swap (SSD streaming's own disk reads then compete with
    /// OS paging on the same disk), and a single denoise step never
    /// finished in over two and a half hours at 1344x768/15s. Smaller
    /// canvases (e.g. 512x512-class) complete in the expected few minutes
    /// with no swapping, so this only warns at the largest profiles with a
    /// double-digit duration - not a precise model, just a conservative
    /// flag for the one combination actually observed to be this bad, now
    /// covering every size above the 512x512 DiT canvas or with an output
    /// as large as 1344x768.
    var isHeavySsdStreamingConfig: Bool {
        guard computeMode == .ssdStreaming else { return false }
        let isLarge = sizeProfile.ditPixels > 512 * 512 || sizeProfile.outputPixels >= 1344 * 768
        return isLarge && seconds >= 10
    }

    /// The whole generation as the progress estimator would predict it
    /// before starting, from this Mac's past runs (TimingCalibration).
    var draftEstimatedSeconds: Double? {
        if isChained { return chainEstimatedSeconds }
        let frames = Int(h3AlignedFrameCount(seconds: Double(seconds.clamped(to: secondsRange))))
        let calibration = TimingCalibration.load(for: computeMode, speed: speedMode,
                                                 fastAttention: useFastAttention,
                                                 ditLayers: ditLayers.clamped(to: ditLayersRange))
        let shape = progressShape(sizeProfile: sizeProfile, frames: frames,
                                  steps: steps.clamped(to: stepsRange), reuse: effectiveDenoiseReuse)
        let now = Date()
        return ProgressEstimator(shape: shape, calibration: calibration, start: now)
            .snapshot(now: now).remaining
    }

    /// "目安 約25分" for the summary under the prompt: how long the current
    /// settings should take on this Mac, once there is a prompt to run.
    var draftEstimateText: String? {
        guard !promptIsEmpty, !isGenerating, let seconds = draftEstimatedSeconds else { return nil }
        let minutes = max(Int((seconds / 60).rounded(.up)), 1)
        if minutes < 60 { return String(localized: "目安 約\(minutes)分") }
        let hours = minutes / 60, rest = minutes % 60
        return rest == 0 ? String(localized: "目安 約\(hours)時間")
                         : String(localized: "目安 約\(hours)時間\(rest)分")
    }

    /// Estimates past this are flagged in 詳細設定.
    static let longGenerationSeconds = 60.0 * 60

    func progressShape(sizeProfile: SizeProfile, frames: Int, steps: Int, reuse: Int) -> ProgressEstimator.Shape {
        // Both use the canvas the DiT runs at: an upscaled size is also
        // decoded at that canvas, and the 2x upscale per chunk (vImage) adds
        // ~0.1 s for a 5 s clip, so output pixels would overstate it 4x.
        ProgressEstimator.Shape(
            steps: steps,
            reuse: reuse,
            totalFrames: frames,
            ditUnits: Double(frames) * Double(sizeProfile.ditPixels),
            decodeUnits: Double(frames) * Double(sizeProfile.ditPixels))
    }

    /// Resident mode holds every DiT block in memory at once (h3_dit.c's
    /// load_block() path, no streaming) - on a tensor-capable GPU that's
    /// int8-quantized (comparable to the ~18 GiB attention-cache file, just
    /// not written to disk); on any other GPU it stays the full ~37 GiB
    /// BF16 checkpoint. Both are on top of the text encoder and VAEs this
    /// process already holds - a rough, unmeasured threshold (unlike
    /// isHeavySsdStreamingConfig's benchmarked one) just to steer a
    /// memory-constrained Mac back to attentionCache/ssdStreaming instead.
    var isLowMemoryForResident: Bool {
        guard computeMode == .resident else { return false }
        let physicalGiB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        return physicalGiB < (supportsInt8Cache ? 32.0 : 64.0)
    }

    // MARK: Attention cache build - separate from generation job state,
    // same invariant as engine/job/result: none of these overwrite each
    // other. Triggered from the "int8キャッシュが見つかりません" message
    // above instead of just telling the user to go run a CLI tool.
    @Published private(set) var isBuildingCache = false
    @Published private(set) var cacheBuildProgress: Double?
    @Published var cacheBuildError: String?
    private var cacheBuildTask: Task<Void, Never>?

    func buildMissingAttentionCache() {
        guard let engine, !isBuildingCache, let model = library.activeModel else { return }
        isBuildingCache = true
        cacheBuildProgress = nil
        cacheBuildError = nil
        try? FileManager.default.createDirectory(
            atPath: attentionCacheDirectory(for: model), withIntermediateDirectories: true)
        let transformerDirectory = currentTransformerDirectory
        let outputPath = currentAttentionCachePath
        cacheBuildTask = Task {
            do {
                for try await progress in engine.buildAttentionCache(
                    transformerDirectory: transformerDirectory, outputPath: outputPath) {
                    self.cacheBuildProgress = Double(progress.completedBlocks) /
                        Double(max(progress.totalBlocks, 1))
                }
            } catch is CancellationError {
                // user-initiated - no error text needed
            } catch {
                if case H3EngineError.cancelled = error {
                    // user-initiated - no error text needed
                } else {
                    self.cacheBuildError = String(localized: "キャッシュの作成に失敗しました。（詳細: \(error.localizedDescription)）")
                }
            }
            self.isBuildingCache = false
            self.cacheBuildProgress = nil
        }
    }

    func cancelCacheBuild() {
        engine?.cancelCurrentCacheBuild()
        cacheBuildTask?.cancel()
    }

    // MARK: Job state
    @Published private(set) var isGenerating = false
    @Published private(set) var isCancelling = false
    @Published var phase: String = ""
    @Published var errorMessage: String?

    // MARK: Timing / progress - all derived by ProgressEstimator from the
    // engine's real phase events plus previously measured timings.
    @Published private(set) var elapsedSeconds: Double = 0
    @Published private(set) var estimatedRemainingSeconds: Double?
    @Published private(set) var estimatedFinishDate: Date?
    @Published private(set) var progressBarFraction: Double?
    @Published private(set) var stageTitle: String = ""
    @Published private(set) var stageDetail: String = ""

    private var estimator: ProgressEstimator?
    private var elapsedTimerTask: Task<Void, Never>?

    // MARK: Result - immutable once set, independent of the live draft.
    // Set from Projects.swift too (showing a saved project video).
    @Published var resultURL: URL?
    @Published var resultAspectRatio: CGFloat = 1
    @Published var lastResult: ResolvedResult?

    // MARK: Project (see Projects.swift) - nil: results are temp files.
    @Published var project: OpenProject?
    @Published var projectVideos: [ProjectVideo] = []
    /// 本数: videos per press of generate. Several need a project to keep
    /// them in; without one, generating asks for it (BatchProjectSheet).
    @Published var batchCount: Int = 1
    @Published var batchProgress: BatchProgress?
    @Published private(set) var batchTiming: BatchTiming?
    @Published var projectMessage: String?
    // The new-project dialog and the video list, opened from the toolbar
    // menu and from the menu bar's プロジェクト menu.
    @Published var showingNewProject = false
    @Published var showingProjectVideos = false
    /// Asked for when generating several videos with no project open.
    @Published var showingBatchProject = false
    /// プロジェクトとして保存 (no project open): the form, and the temp video
    /// in the preview if there is one.
    @Published var showingSaveAsProject = false
    /// The record of a video finished with no project open (its `video` is
    /// the temp file's name), so saving as a project can keep it.
    var temporaryResultRecord: ProjectVideoRecord?
    var projectAutosave: AnyCancellable?
    var lastSavedProjectFile: ProjectFile?
    /// Outside references being copied into the project (startReferenceImport).
    var referenceImportTask: Task<Void, Never>?
    /// "project path\nfile path" of copies that failed (e.g. a source in a
    /// folder the app was denied): not retried automatically, since every
    /// autosave would otherwise retry and re-raise the error once a second.
    var failedReferenceImports: Set<String> = []
    private var batchState: BatchState?

    private(set) var engine: H3Engine?
    private var generationTask: Task<Void, Never>?

    // MARK: Local automation API (see GenerationViewModel+API.swift) -
    // replaces the old Python gui/server.py entirely: while this app runs,
    // the same job/state a person drives through the form is also reachable
    // over HTTP, with no separate process or dependency to install. One
    // server for the app (APIHost); this is what it says about this window,
    // set when a window shows this model (AppModels).
    @Published var apiServerStatus: String = String(localized: "起動しています…")

    init() {
        // Sweep anything a previous run left behind (crash, force quit) -
        // generated previews are meant to be throwaway unless the user
        // explicitly exports, which copies them out.
        Self.sweepStaleTempFiles()
        NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.saveProjectIfChanged()
                self?.deleteTemporaryPreview()
                APIHost.shared.stop()
            }
        }
        followTurboLoRASteps()
        loadWhenFirstModelArrives()
    }

    private var turboStepsSubscription: AnyCancellable?
    private var firstModelSubscription: AnyCancellable?

    /// With no model loaded (none was registered), the first one to appear
    /// - added in this window, another window, the import wizard or the
    /// API - is loaded right away, so the window does not keep saying that
    /// no model is registered while the list shows one in use.
    private func loadWhenFirstModelArrives() {
        firstModelSubscription = library.$activeModelID
            .combineLatest(library.$models)
            .dropFirst() // launch loads the model itself
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _, _ in
                guard let self, self.modelDirectory.isEmpty, !self.isGenerating,
                      self.library.activeModel != nil else { return }
                self.loadModel()
            }
    }

    /// A distilled Turbo LoRA only works at the step count it was trained
    /// for, and base-model quality collapses at that count once the LoRA is
    /// gone - so the draft's steps follow the stack: enabling one with
    /// recommendedSteps adopts them (the first such entry in list order
    /// wins), and disabling it restores the default if the user hadn't
    /// changed them. Also runs at launch, since steps aren't persisted but
    /// the stack is.
    private func followTurboLoRASteps() {
        turboStepsSubscription = library.$loras
            .map { loras -> Int? in
                loras.first { $0.enabled && $0.recommendedSteps != nil }?.recommendedSteps
            }
            .removeDuplicates()
            .scan((previous: Int?.none, current: Int?.none)) { ($0.current, $1) }
            .sink { [weak self] change in
                guard let self else { return }
                if let steps = change.current {
                    self.steps = steps
                } else if let previous = change.previous, self.steps == previous {
                    self.steps = defaultSteps
                }
            }
    }

    private static func sweepStaleTempFiles() {
        let directory = NSTemporaryDirectory()
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: directory) else { return }
        // (h3c-app_ is the name before v0.6.)
        for name in names where (name.hasPrefix("H3cApp_") || name.hasPrefix("h3c-app_")) && name.hasSuffix(".mp4") {
            try? FileManager.default.removeItem(atPath: directory + name)
        }
    }

    /// Deletes the shown result if it is a temp file; a video saved in a
    /// project stays until the user deletes it.
    func deleteTemporaryPreview() {
        if let resultURL, isTemporaryResult(resultURL) {
            try? FileManager.default.removeItem(at: resultURL)
            try? FileManager.default.removeItem(at: ProjectFiles.latentURL(for: resultURL))
        }
    }

    /// Reloads the engine against the active model - called at launch and
    /// whenever ModelManagerView changes which registered model is active.
    func loadModel() {
        guard let active = library.activeModel else {
            engine = nil // lets go of the removed model's files
            modelDirectory = ""
            deviceLine = String(localized: "モデルが登録されていません")
            engineState = .failed(String(localized: "「モデル管理」からMiniMax-H3のフォルダを追加してください。"))
            return
        }
        let modelDir = active.path
        modelDirectory = modelDir
        engineState = .loading
        deviceLine = String(localized: "モデルを準備しています…")
        do {
            let engine = try H3Engine(modelDirectory: modelDir)
            self.engine = engine
            if let device = engine.device {
                deviceLine = "\(device.name) · \(device.architecture)"
                supportsInt8Cache = device.hasTensorHardware
                if !supportsInt8Cache { computeMode = .ssdStreaming }
            } else {
                deviceLine = String(localized: "モデルは読み込めましたが、GPU情報が取得できませんでした")
            }
            engineState = .ready
        } catch {
            deviceLine = String(localized: "モデルの読み込みに失敗しました")
            engineState = .failed(error.localizedDescription)
        }
    }

    /// Switches ModelLibrary's active model (when `id` is non-nil) and
    /// reloads the engine against it - always reloads even if `id` is
    /// already active, since a caller may have just changed that same
    /// entry's path (e.g. ModelDownloadWizardView finishing a download into
    /// it). `id` is nil when ModelManagerView just removed the active model
    /// and needs a reload to reflect whatever (possibly nothing) is active
    /// now. Callers that want to skip a redundant reload for an unchanged
    /// selection (a plain row tap) should check that themselves first.
    func switchModel(to id: UUID?) {
        if let id {
            library.selectModel(id: id)
        }
        loadModel()
    }

    // MARK: Image pickers

    func pickFirstFrame() {
        if let path = chooseFile(allowedContentTypes: imageTypes) { firstFramePath = path }
    }

    func pickLastFrame() {
        if let path = chooseFile(allowedContentTypes: imageTypes) { lastFramePath = path }
    }

    func pickForcedAudio() {
        if let path = chooseFile(allowedContentTypes: forcedAudioTypes) {
            if path != forcedAudioPath { forcedAudioOffset = 0 }
            forcedAudioPath = path
        }
    }

    func addReferenceImages() {
        for path in chooseFiles(allowedContentTypes: referenceMediaTypes) {
            let kind: H3ReferenceKind = isVideoFile(path: path) ? .video : (isAudioFile(path: path) ? .audio : .image)
            referenceImages.append(H3ReferenceInput(kind: kind, path: path))
        }
    }

    func removeReferenceImages(at offsets: IndexSet) {
        referenceImages.remove(atOffsets: offsets)
    }

    func moveReferenceImages(from source: IndexSet, to destination: Int) {
        referenceImages.move(fromOffsets: source, toOffset: destination)
    }

    func resetAdvancedSettings() {
        sizeProfile = defaultSizeProfile
        steps = defaultSteps
        denoiseReuse = defaultReuse
        ditLayers = defaultDitLayers
        computeMode = defaultComputeMode
        speedMode = .quality
        fastAttention = false
        seedFixed = false
        chainClipLength = defaultChainClipLength
        library.setEnabledLoRAs([])
    }

    // MARK: Validation (UI-07: block generation with a locatable reason
    // instead of a generic disabled button)

    var promptIsEmpty: Bool {
        prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// The seed field's own error, shown next to it in 詳細設定 whatever
    /// else is wrong with the form.
    var seedValidationMessage: String? {
        guard seedFixed && UInt64(seedText) == nil else { return nil }
        return seedText.isEmpty ? String(localized: "シード値を入力してください")
                                : String(localized: "シード値は0〜18446744073709551615の整数にしてください")
    }

    var validationMessage: String? {
        guard case .ready = engineState else { return nil } // covered by engine status instead
        // The form is set to continue the video only once it is in.
        if isImportingVideo {
            return String(localized: "動画を取り込んでいます。終わると、その続きをつくれます")
        }
        if promptIsEmpty {
            return String(localized: "動画の内容を入力してください")
        }
        if let seedValidationMessage { return seedValidationMessage }
        if !sizeProfile.isWithinCustomLimit {
            return String(localized: "カスタムサイズの生成サイズが上限（\(SizeProfile.customMaxPixels)ピクセル）を超えています。詳細設定で小さくするか、2倍の拡大を使ってください")
        }
        if creationMethod == .image {
            switch imageInputMode {
            case .firstLastFrame:
                if firstFramePath == nil { return String(localized: "最初の画像を選んでください") }
            case .referenceImage:
                if referenceImages.isEmpty { return String(localized: "参照画像・動画・音声を選んでください") }
                // h3.c rejects this combination outright ("reference audio
                // requires an image or video reference") - an audio file
                // can season a visual reference but can't carry a
                // generation on its own.
                let hasVisualReference = referenceImages.contains { $0.kind == .image || $0.kind == .video }
                if !hasVisualReference {
                    return String(localized: "音声だけの参照はできません。画像か動画の参照も追加してください")
                }
            }
        }
        if let continuationValidationMessage { return continuationValidationMessage }
        if let customClipsValidationMessage { return customClipsValidationMessage }
        if let forcedAudioPath, !FileManager.default.fileExists(atPath: forcedAudioPath) {
            return String(localized: "合わせる音声のファイルが見つかりません: \(URL(fileURLWithPath: forcedAudioPath).lastPathComponent)")
        }
        for lora in library.enabledLoRAs {
            if !FileManager.default.fileExists(atPath: lora.path) {
                return String(localized: "追加モデル「\(lora.name)」のファイルが見つかりません。モデル管理で確認してください")
            }
            if case .failure? = library.loraInfo(for: lora) {
                return String(localized: "追加モデル「\(lora.name)」はMiniMax-H3用として読み込めません。モデル管理で確認してください")
            }
        }
        if computeMode == .attentionCache {
            if !supportsInt8Cache {
                return String(localized: "このGPUではint8キャッシュを使えません。詳細設定の「計算方式」でSSDストリーミングを選んでください")
            }
            if attentionCacheMissing {
                return String(localized: "int8キャッシュが見つかりません。詳細設定の「計算方式」から作成するか、SSDストリーミングに切り替えてください")
            }
        }
        return nil
    }

    var canGenerate: Bool {
        engineState == .ready && !isGenerating && !isBuildingCache && !isImportingVideo
            && validationMessage == nil
    }

    // MARK: Summary text shown above the primary button, and in "設定を見る"

    var draftSummaryText: String {
        var parts = [sizeProfile.label, durationLabel]
        if usesCustomClips {
            let total = String(format: "%.1f", customClipsTotalSeconds)
            parts.append(String(localized: "\(customClips.count)本をつなぐ（計\(total)秒）"))
        } else if isChained {
            parts.append(String(localized: "\(currentChainPlan.count)本をつなぐ（1本\(chainClipLength.label)）"))
        }
        if hasAdvancedChanges {
            // Clamped like the engine params, so this shows what will run.
            parts.append("Steps \(steps.clamped(to: stepsRange))")
            if speedMode == .quality && denoiseReuse != defaultReuse { parts.append("reuse \(denoiseReuse)") }
            if ditLayers != defaultDitLayers { parts.append(String(localized: "層 \(ditLayers)")) }
            if computeMode != defaultComputeMode { parts.append(computeMode.summaryLabel) }
            if speedMode != .quality { parts.append(speedMode.summaryLabel) }
            if fastAttention { parts.append(String(localized: "高速モード（試験的）")) }
            if seedFixed { parts.append(String(localized: "シード固定")) }
            let loras = library.enabledLoRAs
            if !loras.isEmpty {
                let names = loras.map(\.name).joined(separator: " + ")
                parts.append(String(localized: "追加モデル: \(names)"))
            }
        }
        return parts.joined(separator: summarySeparator)
    }

    // MARK: Generation

    /// Everything one generation needs, captured from the form once: a
    /// batch reuses it for every video, changing only the seed, so editing
    /// the form mid-batch doesn't change the rest of the batch.
    struct GenerationRequest {
        var params: H3GenerationParams
        var prompt: String
        var creationMethod: CreationMethod
        var imageInputMode: ImageInputMode?
        var sizeProfile: SizeProfile
        var requestedSeconds: Int
        var requestedFrames: Int
        var steps: Int
        var denoiseReuse: Int
        var effectiveDenoiseReuse: Int
        var ditLayers: Int
        var computeMode: ComputeMode
        var speedMode: SpeedMode
        var loras: [ResolvedLoRA]
        var sampling: SamplingSettings
        var referenceNames: [String]
        var deviceLine: String
        var shape: ProgressEstimator.Shape
        /// Where the videos go: the project open when the batch started.
        var projectURL: URL?
        /// The form for the video records (paths relative to the project).
        var draft: ProjectDraft
        /// Where in the project the video goes (a つなげて生成's clip folder),
        /// nil for the project's own folder.
        var storeFolder: URL? = nil
        /// How late the video's sound will run (ProjectVideoRecord.soundLateness).
        var soundLateness: Double = 0
    }

    private struct BatchState {
        var total: Int
        /// The fixed seed of the first video, or nil for a random one. The
        /// videos after the first always get a new random seed, so a pinned
        /// seed (e.g. from 設定を戻す) doesn't repeat or crowd one region.
        var firstSeed: UInt64?
        var request: GenerationRequest
        let startedAt = Date()
        /// Each video's estimate before the batch started (nil: no timings
        /// on this Mac yet), and how long the finished ones really took.
        var planned: [Double]?
        var actual: [Double] = []
    }

    /// Starts generating. With a project open, `count` videos (default: the
    /// form's 本数) are made one after another with the same request and
    /// different seeds; without one, a single video.
    func generate(count: Int? = nil) {
        guard engine != nil, canGenerate else { return }
        // A project first gets copies of the references and the form saved,
        // so the request points at files inside it. Copying runs in the
        // background; wait for it rather than block the window.
        saveProjectIfChanged()
        // The enabled LoRAs' headers are read in the background (see
        // ModelLibrary.loraInfo); 自動 sampling and the file check need them.
        let pendingLoRAs = library.enabledLoRAs.filter { library.isLoRAInfoPending($0) }
        let copying = referenceImportTask
        if copying != nil || !pendingLoRAs.isEmpty {
            isGenerating = true
            isCancelling = false
            phase = copying != nil ? String(localized: "参照ファイルをプロジェクトにコピーしています")
                                   : String(localized: "追加モデルのファイルを確認しています")
            Task {
                await copying?.value
                await self.library.loadLoRAInfo(for: pendingLoRAs)
                let cancelled = self.isCancelling
                self.isGenerating = false
                self.isCancelling = false
                self.phase = ""
                if cancelled { self.samplingOverride = nil } else { self.startGenerating(count: count) }
            }
            return
        }
        startGenerating(count: count)
    }

    private func startGenerating(count: Int?) {
        guard engine != nil, canGenerate else { return }
        saveProjectIfChanged(importingReferences: false)
        var total = project == nil ? 1 : (count ?? batchCount).clamped(to: batchCountRange)
        let request = makeRequest()
        chainRun = nil
        if isChained, let project {
            let run = ChainRun(plans: chainClipPlans(), projectURL: project.url,
                               fixedSeed: seedFixed ? UInt64(seedText) : nil, request: request)
            chainRun = run
            total = run.plan.count
        }
        var planned: [Double]?
        if let chain = chainRun {
            planned = chainClipEstimates(plan: chain.plan)
        } else if total > 1 {
            let now = Date()
            let calibration = TimingCalibration.load(for: request.computeMode, speed: request.speedMode,
                                                     fastAttention: request.params.fastAttention,
                                                     ditLayers: request.ditLayers)
            planned = ProgressEstimator(shape: request.shape, calibration: calibration, start: now)
                .snapshot(now: now).remaining.map { Array(repeating: $0, count: total) }
        }
        batchState = BatchState(total: total, firstSeed: seedFixed ? UInt64(seedText) : nil,
                                request: request, planned: planned)
        samplingOverride = nil
        // Items after the first reuse the AdaLN schedule (released below
        // when the batch ends).
        if total > 1 { engine?.setBatchReuse(true) }
        runBatchItem(index: 1)
    }

    private func makeRequest() -> GenerationRequest {
        let dimensions = sizeProfile.dimensions
        let requestedSeconds = seconds.clamped(to: currentSecondsRange)
        // A chain's clips set their own frame counts (chainClipRequest).
        let requestedFrames = Int(h3AlignedFrameCount(seconds: Double(min(requestedSeconds, secondsRange.upperBound))))

        // Design spec invariant #4: only the image state matching the
        // *current* mode reaches the engine - the rest stays in the draft,
        // untouched, for if the user switches back.
        let effectiveFirstFrame = (creationMethod == .image && imageInputMode == .firstLastFrame) ? firstFramePath : nil
        let effectiveLastFrame = (creationMethod == .image && imageInputMode == .firstLastFrame) ? lastFramePath : nil
        let effectiveReferences = (creationMethod == .image && imageInputMode == .referenceImage) ? referenceImages : []

        var params = H3GenerationParams()
        params.width = dimensions.width
        params.height = dimensions.height
        params.renderWidth = dimensions.renderWidth
        params.renderHeight = dimensions.renderHeight
        params.frames = Int32(requestedFrames)
        params.steps = Int32(steps.clamped(to: stepsRange))
        params.denoiseReuse = Int32(effectiveDenoiseReuse)
        params.firstFrame = effectiveFirstFrame
        params.lastFrame = effectiveLastFrame
        params.references = effectiveReferences
        params.ssdStreaming = computeMode == .ssdStreaming
        params.attentionCachePath = computeMode == .attentionCache ? currentAttentionCachePath : nil
        params.adalnCurveDirectory = useAdaLNCurve
            ? library.activeModel.map { attentionCacheDirectory(for: $0) } : nil
        let loras = effectiveLoRAs
        params.loras = loras.map { H3LoRAInput(path: $0.path, strength: $0.strength) }
        let speed = speedSettings
        params.ditLayers = speed.ditLayers
        params.coreReuse = speed.coreReuse
        params.tokenReduction = speed.tokenReduction
        // Decided here, once, for this generation: the value is copied into
        // params, so later toggles in the form can't affect a running job.
        params.fastAttention = useFastAttention
        let sampling = samplingSettings
        params.renoise = sampling.renoise
        params.audioSigmaShift = sampling.audioShift
        params.forcedAudioPath = forcedAudioPath
        params.forcedAudioOffset = max(0, forcedAudioOffset)
        params.continueLatentPath = continuation?.latentURL.path
        params.continueFrames = Int32(continuation?.frames ?? continuationFrames)
        params.latentPreviewEvery = denoisePreviewActive
            ? DenoisePreviewRenderer.interval(steps: Int(params.steps), reuse: effectiveDenoiseReuse) : 0

        var lateness = 0.0
        if let continuation {
            let recordURL = continuation.videoURL.deletingPathExtension().appendingPathExtension("json")
            let source = (try? Data(contentsOf: recordURL))
                .flatMap { try? ProjectFiles.decoder.decode(ProjectVideoRecord.self, from: $0) }
            lateness = continuationSoundLateness(source: source, latentURL: continuation.latentURL,
                                                 forced: forcedAudioPath != nil, overlap: continuation.frames)
        }

        var draft = currentDraft()
        if let projectURL = project?.url {
            draft.firstFrame = draft.firstFrame.map { ProjectFiles.stored($0, in: projectURL) }
            draft.lastFrame = draft.lastFrame.map { ProjectFiles.stored($0, in: projectURL) }
            draft.forcedAudio = draft.forcedAudio.map { ProjectFiles.stored($0, in: projectURL) }
            draft.continueFrom = draft.continueFrom.map { ProjectFiles.stored($0, in: projectURL) }
            draft.references = draft.references.map { .init(kind: $0.kind, path: ProjectFiles.stored($0.path, in: projectURL)) }
            draft = draft.mappingClipPaths { ProjectFiles.stored($0, in: projectURL) }
        }
        draft.fastAttention = params.fastAttention
        // Kept in the project for when the length is chained again, but not
        // in the record of a video they didn't shape.
        if !isChained {
            draft.clipPrompts = nil
            draft.clipReferences = nil
            draft.chainCustom = nil
            draft.customClips = nil
        } else if usesCustomClips {
            draft.clipPrompts = nil
            draft.clipReferences = nil
        } else {
            draft.chainCustom = nil
            draft.customClips = nil
        }

        return GenerationRequest(
            params: params,
            prompt: prompt,
            creationMethod: creationMethod,
            imageInputMode: creationMethod == .image ? imageInputMode : nil,
            sizeProfile: sizeProfile,
            requestedSeconds: requestedSeconds,
            requestedFrames: requestedFrames,
            steps: steps.clamped(to: stepsRange),
            denoiseReuse: denoiseReuse.clamped(to: reuseRange),
            effectiveDenoiseReuse: effectiveDenoiseReuse,
            ditLayers: ditLayers.clamped(to: ditLayersRange),
            computeMode: computeMode,
            speedMode: speedMode,
            loras: loras,
            sampling: sampling,
            referenceNames: referenceNames(draft),
            deviceLine: deviceLine,
            shape: progressShape(sizeProfile: sizeProfile, frames: requestedFrames,
                                 steps: steps.clamped(to: stepsRange), reuse: effectiveDenoiseReuse),
            projectURL: project?.url,
            draft: draft,
            soundLateness: lateness)
    }

    private func runBatchItem(index: Int) {
        guard let engine, let batch = batchState else { return }
        var request = batch.request
        let fixedSeed = index == 1 ? batch.firstSeed : nil
        var seed = fixedSeed ?? UInt64.random(in: UInt64.min ... UInt64.max)
        var seedWasRandom = fixedSeed == nil
        if let chain = chainRun {
            // A chain's clips: the base seed and the next ones after it, so
            // pinning the base seed makes the same chain again.
            request = chainClipRequest(chain, index: index - 1)
            seed = chain.baseSeed &+ UInt64(index - 1)
            seedWasRandom = chain.seedWasRandom
        }
        batchProgress = batch.total > 1
            ? BatchProgress(index: index, total: batch.total, chained: chainRun != nil) : nil

        deleteTemporaryPreview()
        isGenerating = true
        isCancelling = false
        errorMessage = nil
        resultURL = nil
        phase = ""
        denoisePreviewImage = nil

        let outputPath = NSTemporaryDirectory() + "H3cApp_\(Int(Date().timeIntervalSince1970))_\(index).mp4"
        estimator = ProgressEstimator(
            shape: request.shape,
            calibration: TimingCalibration.load(for: request.computeMode, speed: request.speedMode,
                                                fastAttention: request.params.fastAttention,
                                                ditLayers: request.ditLayers),
            start: Date())
        publishTiming()
        startElapsedTimer()

        var params = request.params
        params.seed = seed
        // Every video keeps its latents next to it, so it can be continued.
        let latentURL = ProjectFiles.latentURL(for: URL(fileURLWithPath: outputPath))
        params.saveLatentPath = latentURL.path
        let startedAt = Date()

        generationTask = Task {
            var succeeded = false
            do {
                for try await event in engine.generate(prompt: request.prompt, outputPath: outputPath, params: params) {
                    switch event {
                    case .progress(let phase, let completed, let total):
                        self.phase = phase
                        self.estimator?.handle(phase: phase, completed: completed, total: total, now: Date())
                        self.publishTiming()
                    case .frame:
                        break
                    case .preview:
                        break
                    case .latentPreview(let preview):
                        self.denoisePreviewRenderer.render(preview) { [weak self] image in
                            guard let self, self.isGenerating, let image else { return }
                            self.denoisePreviewImage = image
                            self.denoisePreviewStep = preview.step
                            self.denoisePreviewSteps = preview.steps
                        }
                    case .finished(let result):
                        self.finish(result, request: request, seedWasRandom: seedWasRandom, startedAt: startedAt)
                        succeeded = true
                    }
                }
            } catch is CancellationError {
                self.phase = String(localized: "生成を中止しました")
            } catch {
                if case H3EngineError.cancelled = error {
                    self.phase = String(localized: "生成を中止しました")
                } else {
                    self.errorMessage = Self.userFacingMessage(for: error)
                }
            }
            if !succeeded { try? FileManager.default.removeItem(at: latentURL) }
            if succeeded { self.batchState?.actual.append(Date().timeIntervalSince(startedAt)) }
            self.isGenerating = false
            self.isCancelling = false
            self.denoisePreviewImage = nil
            self.stopElapsedTimer()
            // Next video of the batch, unless this one failed or the batch
            // was cancelled.
            if succeeded, index < batch.total, self.batchState != nil, self.engineState == .ready {
                self.runBatchItem(index: index + 1)
            } else {
                if batch.total > 1 { engine.setBatchReuse(false) }
                self.batchState = nil
                self.batchProgress = nil
                self.batchTiming = nil
                // A つなげて生成 joins what it made, also when stopped early.
                if let chain = self.chainRun {
                    self.chainRun = nil
                    Task { await self.finishChain(chain) }
                }
            }
        }
    }

    private func finish(_ result: H3GenerationResult, request: GenerationRequest,
                        seedWasRandom: Bool, startedAt: Date) {
        Self.lastUsedSeed = result.seed
        phase = String(localized: "できあがりました")
        estimator?.finishedCalibration(now: Date())
            // Keyed by the path that actually ran, not the checkbox: a
            // diagnostic H3_ATTENTION_BACKEND can route through ccv even
            // with it off.
            .save(for: request.computeMode, speed: request.speedMode,
                  fastAttention: result.ccvAttentionCalls > 0,
                  ditLayers: request.ditLayers)
        let completedAt = Date()
        let generationSeconds = completedAt.timeIntervalSince(startedAt)
        var url = URL(fileURLWithPath: result.outputPath)
        var draft = request.draft
        draft.seed = String(result.seed)
        let record = ProjectVideoRecord(
            video: url.lastPathComponent, completedAt: completedAt, seed: String(result.seed),
            seedWasRandom: seedWasRandom, generationSeconds: generationSeconds,
            actualFrameCount: result.frames, fps: result.fps,
            effectiveDenoiseReuse: request.effectiveDenoiseReuse,
            ccvAttentionCalls: result.ccvAttentionCalls,
            ccvAttentionDirectCalls: result.ccvAttentionDirectCalls,
            deviceLine: request.deviceLine, appVersion: Self.appVersion, draft: draft,
            sampler: request.sampling.renoise ? "renoise" : nil,
            audioSigmaShift: request.sampling.audioShift > 0 ? Double(request.sampling.audioShift) : nil,
            soundLateness: request.soundLateness,
            latentSoundLateness: request.soundLateness)
        GenerationMetadata.tag(url, record: record)
        if let projectURL = request.projectURL {
            if let stored = storeInProject(videoAt: url, projectURL: projectURL, record: record,
                                           completedAt: completedAt, seed: result.seed,
                                           folder: request.storeFolder) {
                url = stored
                if request.storeFolder != nil, let chain = chainRun {
                    var storedRecord = record
                    storedRecord.video = stored.lastPathComponent
                    chain.clips.append((stored, storedRecord))
                }
            }
            temporaryResultRecord = nil
        } else {
            // Kept for プロジェクトとして保存, which can take this video along.
            temporaryResultRecord = record
        }
        let dimensions = request.sizeProfile.dimensions
        resultURL = url
        resultAspectRatio = CGFloat(dimensions.width) / CGFloat(dimensions.height)
        lastResult = ResolvedResult(
            prompt: request.prompt,
            creationMethod: request.creationMethod,
            imageInputMode: request.imageInputMode,
            sizeProfile: request.sizeProfile,
            requestedSeconds: request.requestedSeconds,
            requestedFrames: request.requestedFrames,
            actualFrameCount: result.frames,
            fps: result.fps,
            actualDurationSeconds: nil,
            steps: request.steps,
            denoiseReuse: request.denoiseReuse,
            effectiveDenoiseReuse: request.effectiveDenoiseReuse,
            ditLayers: request.ditLayers,
            computeMode: request.computeMode,
            speedMode: request.speedMode,
            fastAttention: request.params.fastAttention,
            ccvAttentionCalls: result.ccvAttentionCalls,
            ccvAttentionDirectCalls: result.ccvAttentionDirectCalls,
            seed: result.seed,
            seedWasRandom: seedWasRandom,
            loras: request.loras,
            sampling: request.sampling,
            references: request.referenceNames,
            deviceLine: request.deviceLine,
            completedAt: completedAt,
            generationSeconds: generationSeconds,
            forcedAudio: request.params.forcedAudioPath.map { URL(fileURLWithPath: $0).lastPathComponent },
            forcedAudioOffset: request.params.forcedAudioOffset,
            continuedFrom: request.draft.continueFrom.map { URL(fileURLWithPath: $0).lastPathComponent },
            continueFrames: request.draft.continueFrames ?? continuationFrames)
        loadActualDuration(for: url)
    }

    // 指定秒数と実際のメディア長は一致するとは限らない（design spec 7章:
    // 「長さの不一致」）ため、生成結果のフレーム数/fpsからの概算ではなく、
    // 実際に書き出されたファイルをAVFoundationで読んで確認する。
    func loadActualDuration(for url: URL) {
        Task {
            let asset = AVURLAsset(url: url)
            if let duration = try? await asset.load(.duration) {
                self.lastResult?.actualDurationSeconds = duration.seconds
            }
        }
    }

    private static func userFacingMessage(for error: Error) -> String {
        // Best-effort mapping of known engine failure text to the design
        // spec's Japanese error copy (section 8) - anything unrecognized
        // falls through to the generic message with the real detail kept
        // alongside it rather than guessing at a specific cause.
        let raw = error.localizedDescription
        if raw.contains("out of memory") || raw.contains("insufficient memory") {
            return String(localized: "この設定ではメモリが足りませんでした。もっと小さいサイズや短い長さをお試しください。（詳細: \(raw)）")
        }
        return String(localized: "動画をつくれませんでした。（詳細: \(raw)）")
    }

    func cancel() {
        guard isGenerating else { return }
        // Stops the rest of a batch too.
        batchState = nil
        isCancelling = true
        engine?.cancelCurrentGeneration()
    }

    // MARK: Timing

    private func startElapsedTimer() {
        elapsedTimerTask?.cancel()
        elapsedTimerTask = Task { [weak self] in
            while let self, !Task.isCancelled {
                self.publishTiming()
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
        }
    }

    private func stopElapsedTimer() {
        elapsedTimerTask?.cancel()
        elapsedTimerTask = nil
    }

    private func publishTiming() {
        guard let estimator else { return }
        let now = Date()
        let snapshot = estimator.snapshot(now: now)
        elapsedSeconds = snapshot.elapsed
        estimatedRemainingSeconds = snapshot.remaining
        estimatedFinishDate = snapshot.remaining.map { now.addingTimeInterval($0) }
        progressBarFraction = snapshot.fraction
        stageTitle = snapshot.stageTitle
        stageDetail = snapshot.detail
        batchTiming = batchTiming(current: snapshot, now: now)
    }

    /// The current video's remaining time plus the estimates of the ones
    /// after it. Those estimates come from this Mac's earlier timings and
    /// can be far off for a size it rarely makes, so they are scaled by the
    /// pace actually seen: the finished videos, and the current one's own
    /// projection once its steps have started.
    private func batchTiming(current: ProgressEstimator.Snapshot, now: Date) -> BatchTiming? {
        guard let batch = batchState, batch.total > 1, let progress = batchProgress else { return nil }
        let elapsed = now.timeIntervalSince(batch.startedAt)
        guard let planned = batch.planned, planned.count == batch.total,
              let currentRemaining = current.remaining else {
            return BatchTiming(elapsed: elapsed, remaining: nil, finishDate: nil, fraction: nil)
        }
        let index = min(progress.index - 1, planned.count - 1)
        let done = min(batch.actual.count, index)
        var seen = batch.actual[..<done].reduce(0, +)
        var expected = planned[..<done].reduce(0, +)
        if let stage = estimator?.stage, stage > .setup {
            seen += current.elapsed + currentRemaining
            expected += planned[index]
        }
        let ratio = seen > 0 && expected > 0 ? (seen / expected).clamped(to: 0.1 ... 10) : 1
        let later = planned[(index + 1)...].reduce(0, +) * ratio
        let remaining = currentRemaining + later
        return BatchTiming(elapsed: elapsed, remaining: remaining,
                           finishDate: now.addingTimeInterval(remaining),
                           fraction: elapsed / max(elapsed + remaining, 1))
    }

    // MARK: Reuse

    /// "この設定を使う": copies a past result's request back into the draft
    /// without starting generation, restoring the original seed *policy*
    /// (random stays random) rather than pinning the exact value.
    func applyDraft(from result: ResolvedResult) {
        prompt = result.prompt
        creationMethod = result.creationMethod
        if let imageInputMode = result.imageInputMode { self.imageInputMode = imageInputMode }
        sizeProfile = result.sizeProfile
        seconds = result.requestedSeconds
        speedMode = result.speedMode
        denoiseReuse = result.denoiseReuse
        ditLayers = result.ditLayers
        computeMode = result.computeMode
        fastAttention = result.fastAttention && fastAttentionAvailable
        // The used seed stays in the field either way, so switching to
        // 固定する later reproduces that run.
        seedText = result.seedDecimalString
        seedFixed = !result.seedWasRandom
        applyLoRAs(result.loras.map { ($0.path, $0.strength) })
        // After the LoRAs: enabling a Turbo LoRA moves steps to its
        // recommended count, and the past result's own value should win.
        steps = result.steps
    }

    /// The result / preset only kept the paths and strengths used, not
    /// library entry ids (which may since have been renamed or removed) -
    /// best-effort match them back to still-registered LoRAs by path.
    func applyLoRAs(_ used: [(path: String, strength: Float)]) {
        var ids = Set<UUID>()
        for item in used {
            guard let match = library.loraEntry(forPath: item.path) else { continue }
            ids.insert(match.id)
            library.setLoRAScale(id: match.id, scaleText: item.strength == 1 ? "" : "\(item.strength)")
        }
        library.setEnabledLoRAs(ids)
    }

    /// "同じシードを使う": pins the exact seed regardless of the original
    /// policy, for reproducing one specific past output.
    func useSameSeed(from result: ResolvedResult) {
        seedFixed = true
        seedText = result.seedDecimalString
    }
}
