import Accelerate
import CoreVideo
import Foundation
import Metal

/// VideoUpscaler.realESRGANGeneral and .realESRGANAnime: Real-ESRGAN's
/// SRVGGNetCompact (RealESRGANNetwork) with the bundled weights.
struct RealESRGANUpscaler: UpscalerMethod {
    let id: VideoUpscaler
    private var anime: Bool { id == .realESRGANAnime }
    var label: String {
        anime ? String(localized: "Real-ESRGAN アニメ（AI超解像）")
              : String(localized: "Real-ESRGAN 汎用（AI超解像）")
    }
    var detail: String {
        anime ? String(localized: "AIで線や色の面をくっきりさせながら拡大します（Real-ESRGAN realesr-animevideov3）。アニメやイラスト調の映像向けで、汎用より速く処理します。")
              : String(localized: "AIで細部を補いながら拡大します（Real-ESRGAN realesr-general-x4v3）。実写や一般の映像向けです。細部を描き足すので、元にない模様が出ることがあります。標準より時間がかかります（GPUで、5秒の動画に数十秒〜数分）。")
    }
    /// The bundled weights (Resources/Upscalers).
    var weightsURL: URL? {
        Bundle.main.url(forResource: anime ? "realesr-animevideov3" : "realesr-general-x4v3",
                        withExtension: "safetensors", subdirectory: "Upscalers")
    }
    var isAvailable: Bool { weightsURL != nil && MTLCreateSystemDefaultDevice() != nil }
    func makeScaler(_ job: UpscaleJob) throws -> ChunkedFrameScaler {
        guard let weights = weightsURL else { throw VideoUpscale.UpscaleError.unavailable(label) }
        return PerFrameChunkAdapter(try RealESRGANScaler(network: RealESRGANNetwork(weights: weights, width: job.inputWidth,
                                                                                    height: job.inputHeight,
                                                                                    scale: job.factor == .two ? 2 : 4),
                                                         outputWidth: job.width, outputHeight: job.height))
    }
}

/// VideoUpscaler.realESRGAN*: RealESRGANNetwork on each frame - its 2x
/// straight into the output, or its 4x resized to the size asked for
/// (Lanczos, as Real-ESRGAN does an outscale other than its own).
final class RealESRGANScaler: FrameScaler {
    let network: RealESRGANNetwork
    private let outputWidth: Int
    private let outputHeight: Int
    /// The network's whole output when it isn't the output's size.
    private var full: UnsafeMutableRawPointer?

    init(network: RealESRGANNetwork, outputWidth: Int, outputHeight: Int) {
        self.network = network
        self.outputWidth = outputWidth
        self.outputHeight = outputHeight
        if network.outputWidth != outputWidth || network.outputHeight != outputHeight {
            full = UnsafeMutableRawPointer.allocate(byteCount: network.outputWidth * network.outputHeight * 4,
                                                    alignment: 64)
        }
    }

    deinit { full?.deallocate() }

    /// Whether `buffer` holds at least `width` x `height` BGRA pixels: the
    /// network and vImage copy that many rows of that many bytes, at the
    /// buffer's own row stride, with nothing else bounding them.
    private static func holds(_ buffer: CVPixelBuffer, width: Int, height: Int) -> Bool {
        CVPixelBufferGetWidth(buffer) >= width && CVPixelBufferGetHeight(buffer) >= height
            && CVPixelBufferGetBytesPerRow(buffer) >= width * 4
    }

    func scale(_ source: CVPixelBuffer, into destination: CVPixelBuffer) throws {
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(destination, [])
        }
        // The network was made for the track's size (naturalSize); a frame
        // decoded smaller would be read past its end.
        guard let input = CVPixelBufferGetBaseAddress(source),
              let output = CVPixelBufferGetBaseAddress(destination),
              Self.holds(source, width: network.width, height: network.height),
              Self.holds(destination, width: outputWidth, height: outputHeight) else {
            throw VideoUpscale.UpscaleError.cannotWrite(String(localized: "フレームを拡大できませんでした"))
        }
        guard let full else {
            try network.upscale(input, sourceRowBytes: CVPixelBufferGetBytesPerRow(source),
                                into: output, destinationRowBytes: CVPixelBufferGetBytesPerRow(destination))
            return
        }
        try network.upscale(input, sourceRowBytes: CVPixelBufferGetBytesPerRow(source),
                            into: full, destinationRowBytes: network.outputWidth * 4)
        var from = vImage_Buffer(data: full, height: vImagePixelCount(network.outputHeight),
                                 width: vImagePixelCount(network.outputWidth), rowBytes: network.outputWidth * 4)
        var to = vImage_Buffer(data: output, height: vImagePixelCount(outputHeight),
                               width: vImagePixelCount(outputWidth),
                               rowBytes: CVPixelBufferGetBytesPerRow(destination))
        guard vImageScale_ARGB8888(&from, &to, nil,
                                   vImage_Flags(kvImageHighQualityResampling | kvImageEdgeExtend)) == kvImageNoError else {
            throw VideoUpscale.UpscaleError.cannotWrite(String(localized: "フレームを拡大できませんでした"))
        }
    }
}
