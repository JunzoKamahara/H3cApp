import Combine
import CoreVideo
import Foundation
import SwiftUI

// The upscalers as modules: each enlargement method is a type that says what
// it is, whether it can run here and how it makes frames (UpscalerMethod);
// the models some of them download are UpscalerModel. The sheet, the model
// manager and the API go through these and the registry, not through the
// names.

/// One enlargement of one video, as a method sees it.
struct UpscaleJob {
    let inputWidth: Int, inputHeight: Int
    /// The size asked for.
    let width: Int, height: Int
    let factor: UpscaleFactor
    let isCancelled: () -> Bool
}

protocol UpscalerMethod {
    /// Its name in the API and in saved records.
    var id: VideoUpscaler { get }
    var label: String { get }
    /// What it is and what it suits, for the sheet.
    var detail: String { get }
    /// Whether this Mac can run it now.
    var isAvailable: Bool { get }
    /// Why not, when it can't.
    var missingReason: String { get }
    /// The model it needs downloaded, if any.
    var model: (any UpscalerModel)? { get }
    /// Whether it wants the GPU to itself (so it waits for a generation).
    var needsIdleGPU: Bool { get }
    /// Whether it regenerates from a project clip's latent, not the frames.
    var regeneratesFromLatent: Bool { get }
    var factorFloor: Double { get }
    var factorCeiling: Double { get }
    /// Whether sizes (input width, height, output width, height) fit this
    /// Mac, when the method can tell without loading weights.
    var sizeCheck: (@Sendable (Int, Int, Int, Int) -> Bool)? { get }
    /// What GET /api/upscalers adds for this method beyond its model's fields.
    var apiFields: [String: Any] { get }
    /// How it cuts the video into windows, recorded with the result.
    var recordedWindow: SeedVR2VideoWindow? { get }
    /// The scaler for one video.
    func makeScaler(_ job: UpscaleJob) throws -> ChunkedFrameScaler
}

extension UpscalerMethod {
    var missingReason: String { String(localized: "このMacでは使えません。") }
    var model: (any UpscalerModel)? { nil }
    var needsIdleGPU: Bool { model != nil }
    var regeneratesFromLatent: Bool { false }
    /// 等倍 and up to the 4x the AI models were made for.
    var factorFloor: Double { UpscaleFactor.minimum }
    var factorCeiling: Double { 4 }
    var sizeCheck: (@Sendable (Int, Int, Int, Int) -> Bool)? { nil }
    var apiFields: [String: Any] { [:] }
    var recordedWindow: SeedVR2VideoWindow? { nil }
}

/// A model an upscaler needs downloaded and prepared: where it is, how far
/// its setup is, and the sheet that prepares it.
protocol UpscalerModel: AnyObject {
    /// Its name in the API paths (/api/upscalers/<id>/setup).
    var id: String { get }
    var title: String { get }
    var directory: String { get }
    /// Whether it can be used now.
    var isReady: Bool { get }
    @MainActor var isBusy: Bool { get }
    /// The setup's phase as the API spells it.
    @MainActor var phaseText: String { get }
    /// Fires after the setup's phase changed.
    @MainActor var phaseChanged: AnyPublisher<Void, Never> { get }
    /// The model manager's line about it.
    @MainActor var statusText: String { get }
    /// What GET /api/upscalers reports for the methods that use it.
    @MainActor var apiFields: [String: Any] { get }
    @MainActor func setupView(busy: Bool, close: @escaping () -> Void) -> AnyView
}

/// An UpscalerModel as a sheet's item.
struct UpscalerModelRef: Identifiable {
    let model: any UpscalerModel
    var id: String { model.id }
}

enum UpscalerRegistry {
    /// Every method, in the order of VideoUpscaler's cases.
    static let methods: [any UpscalerMethod] = [
        StandardUpscaler(), MetalFXUpscaler(),
        RealESRGANUpscaler(id: .realESRGANGeneral), RealESRGANUpscaler(id: .realESRGANAnime),
        SwiftVRUpscaler(id: .swiftVR), SwiftVRUpscaler(id: .swiftVRFourX),
        SeedVR2Upscaler(id: .seedVR2), SeedVR2Upscaler(id: .seedVR2Video),
        OpenH3Upscaler(),
    ]

    /// The downloadable models, in the model manager's order.
    static let models: [any UpscalerModel] = [SwiftVRKit.shared, SeedVR2Kit.shared, OpenH3Kit.shared]

    private static let byID: [VideoUpscaler: any UpscalerMethod] = {
        let table = Dictionary(uniqueKeysWithValues: methods.map { ($0.id, $0) })
        precondition(VideoUpscaler.allCases.allSatisfy { table[$0] != nil }, "an upscaler has no method")
        return table
    }()

    static func method(for upscaler: VideoUpscaler) -> any UpscalerMethod { byID[upscaler]! }

    /// Fires after any model's setup phase changed.
    @MainActor static var modelChanges: AnyPublisher<Void, Never> {
        Publishers.MergeMany(models.map(\.phaseChanged)).eraseToAnyPublisher()
    }
}
