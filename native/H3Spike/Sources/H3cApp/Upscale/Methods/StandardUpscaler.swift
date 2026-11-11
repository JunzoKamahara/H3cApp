import Accelerate
import CoreVideo
import Foundation

/// VideoUpscaler.standard: the same high-quality resize as the ×2 sizes.
struct StandardUpscaler: UpscalerMethod {
    let id = VideoUpscaler.standard
    var label: String { String(localized: "標準（高品質な拡大）") }
    var detail: String {
        String(localized: "大きさの「×2」と同じ拡大です。なめらかで、元の動画の見た目をそのまま大きくします。CPUで処理します。")
    }
    var isAvailable: Bool { true }
    /// A resizer at 等倍 would change nothing; it goes on as far as a side allows.
    var factorFloor: Double { 1.1 }
    var factorCeiling: Double { UpscaleFactor.maximum }
    func makeScaler(_ job: UpscaleJob) throws -> ChunkedFrameScaler { PerFrameChunkAdapter(VImageScaler()) }
}

/// VideoUpscaler.standard: vImage's Lanczos on the CPU.
struct VImageScaler: FrameScaler {
    func scale(_ source: CVPixelBuffer, into destination: CVPixelBuffer) throws {
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(destination, [])
        }
        guard let input = CVPixelBufferGetBaseAddress(source),
              let output = CVPixelBufferGetBaseAddress(destination) else {
            throw VideoUpscale.UpscaleError.cannotWrite(String(localized: "フレームを拡大できませんでした"))
        }
        var from = vImage_Buffer(data: input,
                                 height: vImagePixelCount(CVPixelBufferGetHeight(source)),
                                 width: vImagePixelCount(CVPixelBufferGetWidth(source)),
                                 rowBytes: CVPixelBufferGetBytesPerRow(source))
        var to = vImage_Buffer(data: output,
                               height: vImagePixelCount(CVPixelBufferGetHeight(destination)),
                               width: vImagePixelCount(CVPixelBufferGetWidth(destination)),
                               rowBytes: CVPixelBufferGetBytesPerRow(destination))
        // Four 8-bit channels in any order: BGRA scales as ARGB does.
        guard vImageScale_ARGB8888(&from, &to, nil,
                                   vImage_Flags(kvImageHighQualityResampling | kvImageEdgeExtend)) == kvImageNoError else {
            throw VideoUpscale.UpscaleError.cannotWrite(String(localized: "フレームを拡大できませんでした"))
        }
    }
}
