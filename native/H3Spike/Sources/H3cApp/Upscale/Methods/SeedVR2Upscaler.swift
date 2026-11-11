import Combine
import CoreVideo
import Foundation
import H3Upscale
import Metal
import SwiftUI

/// VideoUpscaler.seedVR2 and .seedVR2Video: SeedVR2-3B through the engine
/// (h3_seedvr2.h), frame by frame or on windows of frames.
struct SeedVR2Upscaler: UpscalerMethod {
    let id: VideoUpscaler
    private var video: Bool { id == .seedVR2Video }
    var label: String {
        video ? String(localized: "SeedVR2 動画（ちらつきを抑える・非常に遅い・試験的）")
              : String(localized: "SeedVR2（AI超解像・細部重視、非常に遅い・試験的）")
    }
    var detail: String {
        video ? String(localized: "SeedVR2-3Bに、17フレームの窓で動画として描き起こさせます（時間方向のVAEと3Dウィンドウ注意）。細部の描き方は1フレームずつのSeedVR2と同じで、髪・肌・毛皮のちらつきが減ります（窓の継ぎ目は5フレームの重なりをクロスフェード。比較では、フレーム間の揺れが1枚ずつより35〜45%小さくなり、速い動きやアニメでは差が小さく、細部はわずかに柔らかくなりました）。時間は1枚ずつの約2倍で、5秒の動画をM5で672×384から2倍にすると約17分、GPUに約13GBを使います（24GBのMacでは1920×1088まで。1344×768より大きいときは状態を8ビット、1920×1088では4ビットで持つため出力がわずかに変わります。16GBのMacでは使えません）。重みは1枚ずつのSeedVR2と同じです。試験的な方式です。")
              : String(localized: "SeedVR2-3B（ByteDance Seed、Apache License 2.0）で1フレームずつ描き起こします。まつ毛や肌のきめなど細部の表現はSwiftVRより強く、そのぶん元の映像から離れることがあります（顔の印象が変わることも）。1フレームに数秒かかり、5秒の動画をM5で672×384から2倍にすると約8分（960×544からは約21分）、GPUに約10GBを使います。重みは同梱せず、配布元の記録と照合してダウンロードします。試験的な方式です。")
    }
    var model: (any UpscalerModel)? { SeedVR2Kit.shared }
    var isAvailable: Bool { SeedVR2Model.weightsPresent && MTLCreateSystemDefaultDevice() != nil }
    var missingReason: String {
        return String(localized: "SeedVR2のモデルが入っていません。「モデルを準備…」で重み（約4.7GB）をダウンロードすると選べます。")
    }
    var sizeCheck: (@Sendable (Int, Int, Int, Int) -> Bool)? {
        guard SeedVR2Model.weightsPresent else { return nil }
        if video {
            return H3SeedVR2Restorer.videoSizeCheck(modelDirectory: VideoUpscaler.seedVR2ModelDirectory,
                                                    windowFrames: SeedVR2VideoWindow.chosen.windowFrames)
        }
        return H3SeedVR2Restorer.sizeCheck(modelDirectory: VideoUpscaler.seedVR2ModelDirectory)
    }
    var recordedWindow: SeedVR2VideoWindow? { video ? SeedVR2VideoWindow.chosen : nil }
    var apiFields: [String: Any] {
        guard video else { return [:] }
        // How it cuts the video (appendix A.10's choice; fixed).
        let window = SeedVR2VideoWindow.chosen
        return ["window_frames": window.windowFrames, "overlap_frames": window.overlapFrames,
                "blend": window.blend ? "hann" : "cut"]
    }
    func makeScaler(_ job: UpscaleJob) throws -> ChunkedFrameScaler {
        guard SeedVR2Model.weightsPresent else { throw VideoUpscale.UpscaleError.unavailable(label) }
        let restorer: H3SeedVR2Restorer
        do {
            restorer = try H3SeedVR2Restorer(modelDirectory: VideoUpscaler.seedVR2ModelDirectory)
        } catch {
            throw VideoUpscale.UpscaleError.cannotWrite(error.localizedDescription)
        }
        if video {
            let window = SeedVR2VideoWindow.chosen
            let requirements = restorer.videoRequirements(inputWidth: job.inputWidth, inputHeight: job.inputHeight,
                                                          outputWidth: job.width, outputHeight: job.height,
                                                          windowFrames: window.windowFrames)
            guard requirements.supported else { throw VideoUpscale.UpscaleError.cannotWrite(requirements.reason) }
            do {
                // The length is left unknown (0): the frame count from the
                // duration can be off by one, and the session moves its
                // last window back onto the final frame on `isLast` anyway.
                return SeedVR2ChunkScaler(session: try restorer.openVideoSession(inputWidth: job.inputWidth, inputHeight: job.inputHeight,
                                                                                 outputWidth: job.width, outputHeight: job.height,
                                                                                 totalFrames: 0, settings: window.engineSettings),
                                          isCancelled: job.isCancelled)
            } catch {
                throw VideoUpscale.UpscaleError.cannotWrite(error.localizedDescription)
            }
        }
        let requirements = restorer.requirements(inputWidth: job.inputWidth, inputHeight: job.inputHeight,
                                                 outputWidth: job.width, outputHeight: job.height)
        guard requirements.supported else { throw VideoUpscale.UpscaleError.cannotWrite(requirements.reason) }
        do {
            return SeedVR2FrameScaler(session: try restorer.openSession(inputWidth: job.inputWidth, inputHeight: job.inputHeight,
                                                                        outputWidth: job.width, outputHeight: job.height),
                                      isCancelled: job.isCancelled)
        } catch {
            throw VideoUpscale.UpscaleError.cannotWrite(error.localizedDescription)
        }
    }
}

/// How SeedVR2 動画 cuts a video into windows: `windowFrames` (4n+1) per
/// window, `overlapFrames` of them shared with the next, and whether the
/// shared frames cross-fade (Hann) or switch over. Recorded with the video.
struct SeedVR2VideoWindow: Hashable, Codable {
    var windowFrames: Int
    var overlapFrames: Int
    var blend: Bool

    /// What the app runs: the configuration appendix A.10 chose (17 frames,
    /// 5 shared and cross-faded: a switch-over left a visible step at every
    /// window, 5-frame windows were worse than per frame, 29 frames don't
    /// fit 24 GB at 1344x768).
    static let standard = SeedVR2VideoWindow(windowFrames: 17, overlapFrames: 5, blend: true)

    /// `standard`, unless the developer environment variables
    /// H3_SV2_VIDEO_WINDOW / H3_SV2_VIDEO_OVERLAP / H3_SV2_VIDEO_BLEND (0/1)
    /// name another valid one (for comparisons; not a user setting).
    static var chosen: SeedVR2VideoWindow {
        let env = ProcessInfo.processInfo.environment
        guard env["H3_SV2_VIDEO_WINDOW"] != nil || env["H3_SV2_VIDEO_OVERLAP"] != nil || env["H3_SV2_VIDEO_BLEND"] != nil else {
            return standard
        }
        let other = SeedVR2VideoWindow(windowFrames: env["H3_SV2_VIDEO_WINDOW"].flatMap(Int.init) ?? standard.windowFrames,
                                       overlapFrames: env["H3_SV2_VIDEO_OVERLAP"].flatMap(Int.init) ?? standard.overlapFrames,
                                       blend: env["H3_SV2_VIDEO_BLEND"].map { $0 != "0" } ?? standard.blend)
        return other.engineSettings.isValid ? other : standard
    }

    /// The engine's form, with the per-frame path's seed.
    var engineSettings: H3SeedVR2VideoSettings {
        H3SeedVR2VideoSettings(windowFrames: windowFrames, overlapFrames: overlapFrames, blend: blend, seed: 42)
    }

    /// "17枚（重なり5、クロスフェード）", for the record.
    var label: String {
        blend ? String(localized: "\(windowFrames)枚（重なり\(overlapFrames)、クロスフェード）")
            : String(localized: "\(windowFrames)枚（重なり\(overlapFrames)、切り替え）")
    }
}

/// SeedVR2 on windows of frames through the engine's temporal path (the
/// same contract as SwiftVR's chunks): a window takes a minute or more, so
/// 中止 is read at every phase step.
final class SeedVR2ChunkScaler: ChunkedFrameScaler {
    let session: H3SeedVR2Restorer.VideoSession
    let isCancelled: () -> Bool
    init(session: H3SeedVR2Restorer.VideoSession, isCancelled: @escaping () -> Bool) {
        self.session = session
        self.isCancelled = isCancelled
    }
    var chunkFrames: Int { max(1, session.nextChunkFrames) }
    func process(_ frames: [CVPixelBuffer], isLast: Bool, pool: CVPixelBufferPool,
                 progress: @escaping (Double) -> Void) throws -> [CVPixelBuffer] {
        do {
            return try session.process(frames, isLast: isLast, pool: pool) { [session, isCancelled] share in
                if isCancelled() { session.cancel() }
                progress(share)
            }
        } catch H3VSRError.cancelled {
            throw VideoUpscale.UpscaleError.cancelled
        }
    }
    func cancel() { session.cancel() }
}

/// SeedVR2 through the engine: one frame at a time, each taking seconds, so
/// 中止 is read at every DiT block.
final class SeedVR2FrameScaler: ChunkedFrameScaler {
    let session: H3SeedVR2Restorer.Session
    let isCancelled: () -> Bool
    init(session: H3SeedVR2Restorer.Session, isCancelled: @escaping () -> Bool) {
        self.session = session
        self.isCancelled = isCancelled
    }
    var chunkFrames: Int { 1 }
    func process(_ frames: [CVPixelBuffer], isLast: Bool, pool: CVPixelBufferPool,
                 progress: @escaping (Double) -> Void) throws -> [CVPixelBuffer] {
        var outputs: [CVPixelBuffer] = []
        for (index, frame) in frames.enumerated() {
            do {
                outputs.append(try session.process(frame, pool: pool) { [session, isCancelled] share in
                    if isCancelled() { session.cancel() }
                    progress((Double(index) + share) / Double(frames.count))
                })
            } catch H3VSRError.cancelled {
                throw VideoUpscale.UpscaleError.cancelled
            }
        }
        return outputs
    }
    func cancel() { session.cancel() }
}
