import Combine
import CoreVideo
import Foundation
import H3Upscale
import Metal
import SwiftUI

/// VideoUpscaler.swiftVR and .swiftVRFourX: SwiftVR through the engine
/// (h3_vsr.h), directly or restored at 4x and brought down.
struct SwiftVRUpscaler: UpscalerMethod {
    let id: VideoUpscaler
    private var fourX: Bool { id == .swiftVRFourX }
    var label: String {
        fourX ? String(localized: "SwiftVR 4倍で生成してから縮小（質感が穏やか）")
              : String(localized: "SwiftVR（AI動画超解像）")
    }
    var detail: String {
        fourX ? String(localized: "SwiftVRに4倍で復元させてから、標準の拡大と同じ方法で選んだ大きさに縮めます。目や眉を描き直しにくく、質感が穏やかで元の動画に忠実になります。4倍の途中の画面がGPUのメモリに入らない大きさでは、重なりを持たせたタイルに分けて復元します。時間はSwiftVRの約5倍です（M5で5秒の544×960を2倍にするのに約4分）。")
              : String(localized: "動画全体を一度に見るAIモデル（SwiftVR、Wan2.2ベース）で、質感や細部を描き起こしながら拡大します。24フレームずつ処理し、Real-ESRGANより大幅に時間がかかります（M5で5秒の動画に1分前後）。重みは同梱せず、配布元でApache License 2.0と宣言されていた版をダウンロードし（SHA-256で照合）、アプリで変換して使います。")
    }
    var model: (any UpscalerModel)? { SwiftVRKit.shared }
    var isAvailable: Bool { VideoUpscaler.swiftVRCachePath != nil && MTLCreateSystemDefaultDevice() != nil }
    var missingReason: String {
        if SwiftVRModel.originalRemoved {
            return String(localized: "元の重みを削除したあとで、変換したキャッシュが使えなくなりました（アプリの更新でキャッシュの形式が変わったときなど）。「モデルを準備…」で重みをもう一度ダウンロードして変換すると選べます。")
        }
        if SwiftVRModel.weightsPresent {
            return String(localized: "SwiftVRの重みはありますが、このMac用に変換されていません。「モデルを準備…」で変換すると選べます。")
        }
        return String(localized: "SwiftVRのモデルが入っていません。「モデルを準備…」で重み（約20GB）をダウンロードして変換すると選べます。")
    }
    var sizeCheck: (@Sendable (Int, Int, Int, Int) -> Bool)? {
        guard let cachePath = VideoUpscaler.swiftVRCachePath else { return nil }
        return H3VideoSuperResolver.sizeCheck(modelDirectory: VideoUpscaler.swiftVRModelDirectory,
                                              int8: cachePath.hasSuffix("int8.h3vc"))
    }
    func makeScaler(_ job: UpscaleJob) throws -> ChunkedFrameScaler {
        guard let cachePath = VideoUpscaler.swiftVRCachePath else { throw VideoUpscale.UpscaleError.unavailable(label) }
        let resolver: H3VideoSuperResolver
        do {
            resolver = try H3VideoSuperResolver(modelDirectory: VideoUpscaler.swiftVRModelDirectory, cachePath: cachePath)
        } catch {
            throw VideoUpscale.UpscaleError.cannotWrite(error.localizedDescription)
        }
        // 4倍で生成してから縮小: restore at 4x (in tiles when the frame
        // is large), or at 3x or 2x where the engine or the writer's limit
        // stops that; the size asked for when none of them is larger.
        var engineWidth = job.width, engineHeight = job.height
        if fourX {
            for large in [UpscaleFactor.four, .three, .two] where large.rawValue > job.factor.rawValue {
                let (w, h) = large.size(width: job.inputWidth, height: job.inputHeight)
                if w <= VideoUpscale.maximumSide && h <= VideoUpscale.maximumSide &&
                    resolver.requirements(inputWidth: job.inputWidth, inputHeight: job.inputHeight,
                                          outputWidth: w, outputHeight: h).supported {
                    engineWidth = w; engineHeight = h
                    break
                }
            }
        }
        let requirements = resolver.requirements(inputWidth: job.inputWidth, inputHeight: job.inputHeight,
                                                 outputWidth: engineWidth, outputHeight: engineHeight)
        guard requirements.supported else { throw VideoUpscale.UpscaleError.cannotWrite(requirements.reason) }
        do {
            let direct = VSRChunkScaler(session: try resolver.openSession(inputWidth: job.inputWidth, inputHeight: job.inputHeight,
                                                                          outputWidth: engineWidth, outputHeight: engineHeight),
                                        isCancelled: job.isCancelled)
            return engineWidth == job.width ? direct
                : HalvingChunkScaler(inner: direct, largeWidth: engineWidth, largeHeight: engineHeight)
        } catch {
            throw VideoUpscale.UpscaleError.cannotWrite(error.localizedDescription)
        }
    }
}

/// SwiftVR through the engine: 28 frames, then 24 at a time. A chunk
/// takes seconds, so 中止 is read at each DiT block (`isCancelled`) and
/// stops the engine there.
final class VSRChunkScaler: ChunkedFrameScaler {
    let session: H3VideoSuperResolver.Session
    let isCancelled: () -> Bool
    init(session: H3VideoSuperResolver.Session, isCancelled: @escaping () -> Bool) {
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

/// SwiftVR's "4倍で生成してから縮小": the engine restores to 4x, and each frame
/// is brought down to the asked-for size with the standard resize. Gentler
/// than a direct 2x on soft sources (UPSCALE_PLAN appendix A.3), at about
/// four times the time. When the 4x frame does not fit the GPU's memory in
/// one piece the engine restores it as overlapping tiles and blends them.
final class HalvingChunkScaler: ChunkedFrameScaler {
    let inner: ChunkedFrameScaler
    private let largeWidth: Int, largeHeight: Int
    private var largePool: CVPixelBufferPool?
    private let resizer = VImageScaler()
    init(inner: ChunkedFrameScaler, largeWidth: Int, largeHeight: Int) {
        self.inner = inner
        self.largeWidth = largeWidth
        self.largeHeight = largeHeight
    }
    var chunkFrames: Int { inner.chunkFrames }
    func process(_ frames: [CVPixelBuffer], isLast: Bool, pool: CVPixelBufferPool,
                 progress: @escaping (Double) -> Void) throws -> [CVPixelBuffer] {
        if largePool == nil {
            let attributes: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: largeWidth,
                kCVPixelBufferHeightKey as String: largeHeight,
                kCVPixelBufferMetalCompatibilityKey as String: true,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            ]
            var made: CVPixelBufferPool?
            CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &made)
            largePool = made
        }
        guard let largePool else {
            throw VideoUpscale.UpscaleError.cannotWrite(String(localized: "フレームを拡大できませんでした"))
        }
        let large = try inner.process(frames, isLast: isLast, pool: largePool, progress: progress)
        var outputs: [CVPixelBuffer] = []
        for frame in large {
            var halved: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &halved)
            guard let halved else {
                throw VideoUpscale.UpscaleError.cannotWrite(String(localized: "フレームを拡大できませんでした"))
            }
            try resizer.scale(frame, into: halved)
            outputs.append(halved)
        }
        return outputs
    }
    func cancel() { inner.cancel() }
}
