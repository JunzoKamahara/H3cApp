import Foundation

/// 全体設定 > 生成: the generation preferences every window shares. One copy
/// here rather than one per window's model, so a change made in 全体設定 (or
/// through the API) reaches every open window at once.
@MainActor
final class AppPreferences: ObservableObject {
    static let shared = AppPreferences()

    /// Curve-form AdaLN (H3GenerationParams.adalnCurveDirectory): ~10 s
    /// faster setup per video in every compute mode, values as close to
    /// exact as the dense path but not bit-identical, so the same seed gives
    /// a slightly different video than the dense path. Not part of a draft;
    /// on by default.
    @Published var useAdaLNCurve: Bool =
        UserDefaults.standard.object(forKey: "h3c-app.useAdaLNCurve") as? Bool ?? true {
        didSet { UserDefaults.standard.set(useAdaLNCurve, forKey: "h3c-app.useAdaLNCurve") }
    }
    /// 途中の様子: show what the video is turning into while it denoises
    /// (DenoisePreview.swift). Off until turned on, since it needs the
    /// 22.7 MB decoder downloaded first.
    @Published private(set) var showDenoisePreview: Bool =
        UserDefaults.standard.bool(forKey: "h3c-app.showDenoisePreview") {
        didSet { UserDefaults.standard.set(showDenoisePreview, forKey: "h3c-app.showDenoisePreview") }
    }
    /// One download for the whole app, whichever window asked for it.
    let denoisePreviewDownloader = DenoisePreviewDownloader()

    /// Turns 途中の様子 on or off, fetching the decoder when it is turned on
    /// and not there yet.
    func setShowDenoisePreview(_ on: Bool) {
        showDenoisePreview = on
        if on && !DenoisePreviewWeights.isPresent { denoisePreviewDownloader.download() }
    }
}
