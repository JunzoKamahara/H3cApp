import Accelerate
import AppKit
import CoreVideo
import Foundation
import H3Engine

// 高解像度化: a project image enlarged into a new project image. Two kinds:
//
// - 画像モデルで描き直す (redraw): the image model draws the picture again at
//   the larger size, starting from the picture itself mixed with noise
//   (h3_image_params.init_image_path) and running only the last steps, so
//   the composition stays and the model adds detail at the new size. Best
//   up to about 8 megapixels, where P-H3-Image does best.
// - The video enlargers on the picture as one frame (UpscalerMethod):
//   resizing, MetalFX, Real-ESRGAN, and SwiftVR / SeedVR2 when downloaded.

enum ImageUpscaleMethod: Hashable, Identifiable {
    case redraw
    case frame(VideoUpscaler)

    /// The enlargers that work on one picture (not Open-H3, which redraws a
    /// video from its latent, nor SeedVR2's window-of-frames path).
    static let frameUpscalers: [VideoUpscaler] = [
        .standard, .metalFX, .realESRGANGeneral, .realESRGANAnime, .swiftVR, .swiftVRFourX, .seedVR2,
    ]

    /// Every method this Mac can use now, the redraw first.
    static var available: [ImageUpscaleMethod] {
        [.redraw] + frameUpscalers.filter { $0.method.isAvailable }.map { .frame($0) }
    }

    /// The name in records and the API.
    var id: String {
        switch self {
        case .redraw: return "redraw"
        case .frame(let upscaler): return upscaler.rawValue
        }
    }

    init?(id: String) {
        if id == "redraw" { self = .redraw; return }
        guard let upscaler = VideoUpscaler(rawValue: id), Self.frameUpscalers.contains(upscaler) else { return nil }
        self = .frame(upscaler)
    }

    var label: String {
        switch self {
        case .redraw: return String(localized: "画像モデルで描き直す")
        case .frame(let upscaler): return upscaler.label
        }
    }

    var detail: String {
        switch self {
        case .redraw:
            return String(localized: "使用中の画像モデルが、大きくした画像をもう一度描きます。元の画像にノイズを混ぜたところから最後の数ステップだけを進めるので、構図や顔立ちは保ったまま、新しい大きさに合った細部（肌、髪、文字など）を描き足します。P-H3-Imageは約8メガピクセルで最も良くなります。元のプロンプトを使います。")
        case .frame(let upscaler):
            return upscaler.method.detail
        }
    }
}

/// How far a redraw departs from the picture: the noise level (sigma) it
/// starts from.
enum RedrawStrength: String, CaseIterable, Identifiable {
    case gentle, standard, strong

    var id: String { rawValue }
    var sigma: Float {
        switch self {
        case .gentle: return 0.6
        case .standard: return 0.7
        case .strong: return 0.85
        }
    }
    var label: String {
        switch self {
        case .gentle: return String(localized: "控えめ")
        case .standard: return String(localized: "標準")
        case .strong: return String(localized: "強め")
        }
    }
    var detail: String {
        switch self {
        case .gentle: return String(localized: "元の画像に近いまま整えます")
        case .standard: return String(localized: "細部をはっきり描き足します（おすすめ）")
        case .strong: return String(localized: "細部を大きく描き足しますが、元にない模様や形が出やすくなります")
        }
    }
}

enum ImageUpscale {
    /// The largest redraw: about P-H3-Image's 8 megapixels.
    static let redrawMaxPixels = 2912 * 2912

    /// The canvas a redraw by `factor` makes of a `width` x `height`
    /// picture: sides multiples of 32 within 256...4096, brought down to
    /// redrawMaxPixels if larger. nil when that is no larger than the
    /// picture (except 等倍).
    static func redrawSize(width: Int, height: Int, factor: UpscaleFactor) -> (width: Int, height: Int)? {
        guard width > 0, height > 0 else { return nil }
        var w = Double(width) * factor.rawValue, h = Double(height) * factor.rawValue
        let area = w * h
        if area > Double(redrawMaxPixels) {
            let shrink = (Double(redrawMaxPixels) / area).squareRoot()
            w *= shrink
            h *= shrink
        }
        func side(_ value: Double) -> Int {
            min(4096, max(256, Int((value / 32).rounded(.down)) * 32))
        }
        let made = (side(w), side(h))
        if factor != .same && made.0 * made.1 <= width * height { return nil }
        return made
    }

    /// The factors a method offers for a picture, with the size each makes
    /// (no two the same).
    static func choices(_ method: ImageUpscaleMethod, width: Int, height: Int)
        -> [(factor: UpscaleFactor, width: Int, height: Int)] {
        var result: [(factor: UpscaleFactor, width: Int, height: Int)] = []
        let factors = [UpscaleFactor.same, .oneAndHalf, .two, .three, .four]
        for factor in factors {
            let made: (width: Int, height: Int)?
            switch method {
            case .redraw:
                made = redrawSize(width: width, height: height, factor: factor)
            case .frame(let upscaler):
                made = factor == .same ? nil : UpscaleChoices(upscaler: upscaler, width: width, height: height).size(factor)
            }
            guard let made, !result.contains(where: { $0.width == made.width && $0.height == made.height }) else { continue }
            result.append((factor, made.width, made.height))
        }
        return result
    }

    /// `image` enlarged to `width` x `height` by one of the video enlargers,
    /// as a single frame.
    static func enlarge(_ image: CGImage, upscaler: VideoUpscaler, factor: UpscaleFactor,
                        width: Int, height: Int, progress: @escaping (Double) -> Void,
                        isCancelled: @escaping () -> Bool) throws -> CGImage {
        let scaler = try upscaler.method.makeScaler(UpscaleJob(
            inputWidth: image.width, inputHeight: image.height, width: width, height: height,
            factor: factor, isCancelled: isCancelled))
        guard let input = pixelBuffer(width: image.width, height: image.height) else {
            throw VideoUpscale.UpscaleError.cannotWrite(String(localized: "画像を読み込めませんでした"))
        }
        try draw(image, into: input)
        var poolOut: CVPixelBufferPool?
        CVPixelBufferPoolCreate(nil, nil, attributes(width: width, height: height) as CFDictionary, &poolOut)
        guard let pool = poolOut else {
            throw VideoUpscale.UpscaleError.cannotWrite(String(localized: "画像を拡大できませんでした"))
        }
        if isCancelled() { throw VideoUpscale.UpscaleError.cancelled }
        let outputs = try scaler.process([input], isLast: true, pool: pool, progress: progress)
        if isCancelled() { throw VideoUpscale.UpscaleError.cancelled }
        guard let output = outputs.last, let made = cgImage(from: output) else {
            throw VideoUpscale.UpscaleError.cannotWrite(String(localized: "画像を拡大できませんでした"))
        }
        return made
    }

    private static func attributes(width: Int, height: Int) -> [String: Any] {
        [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
    }

    private static func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                            attributes(width: width, height: height) as CFDictionary, &buffer)
        return buffer
    }

    /// BGRA, opaque, sRGB: the layout the frame enlargers read.
    private static let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue

    private static func draw(_ image: CGImage, into buffer: CVPixelBuffer) throws {
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: image.width,
                                      height: image.height, bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: bitmapInfo) else {
            throw VideoUpscale.UpscaleError.cannotWrite(String(localized: "画像を読み込めませんでした"))
        }
        context.interpolationQuality = .none
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    }

    private static func cgImage(from buffer: CVPixelBuffer) -> CGImage? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: CVPixelBufferGetWidth(buffer),
                                      height: CVPixelBufferGetHeight(buffer), bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: bitmapInfo) else {
            return nil
        }
        return context.makeImage()
    }
}
