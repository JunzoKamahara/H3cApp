import Accelerate
import CoreGraphics
import Foundation
import H3Engine

// 途中の様子 for a still image: while it denoises, the engine hands over
// its estimate of the finished latent after every step (H3ImageEngine's
// latentPreview), and a fixed linear map turns it into a picture - each
// latent pixel's 3x3 neighbourhood to the RGB patch the VAE would decode it
// to. One map per image model, fitted by least squares on pictures the
// engine generated (tools/image_preview/fit_preview.py), shipped in the app
// (Packaging/Previews, 333 KB each). Nothing to download, and a few
// milliseconds per picture, so it is on unless turned off.
//
// The picture is drawn at half the image's size (8x8 pixels per latent
// pixel, each the mean of the map's 2x2 pixels): the stage shows it at
// most 640 points wide.

final class ImagePreviewMap: @unchecked Sendable {
    static let neighbourhood = 3
    static let channels = 24
    /// Output pixels per latent pixel, per side.
    static let patch = 8

    /// [features, patch * patch * 3], features = 3 * 3 * 24 + 1 (bias last).
    private let weight: [Float]
    private let features: Int
    private let outputs: Int

    /// The map for an image model: "P-H3-Image" or "MiniMax H3 Image".
    static func load(isPH3: Bool) -> ImagePreviewMap? {
        let name = isPH3 ? "p-h3-image" : "minimax-h3-image"
        let candidates = [
            Bundle.main.resourceURL?.appendingPathComponent("Previews/\(name).safetensors").path,
            // swift run from the package: the file beside the sources.
            URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .appendingPathComponent("../../Packaging/Previews/\(name).safetensors").standardized.path,
        ].compactMap { $0 }
        for path in candidates where FileManager.default.fileExists(atPath: path) {
            if let map = try? ImagePreviewMap(path: path) { return map }
        }
        NSLog("H3cApp: no image preview map for %@", name)
        return nil
    }

    init(path: String) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard data.count > 8 else { throw CocoaError(.fileReadCorruptFile) }
        let headerBytes = data.prefix(8).withUnsafeBytes { Int($0.loadUnaligned(as: UInt64.self).littleEndian) }
        guard 8 + headerBytes <= data.count,
              let header = try JSONSerialization.jsonObject(with: data.subdata(in: 8 ..< 8 + headerBytes)) as? [String: Any],
              let entry = header["weight"] as? [String: Any], entry["dtype"] as? String == "F16",
              let shape = entry["shape"] as? [Int], shape.count == 2,
              let offsets = entry["data_offsets"] as? [Int], offsets.count == 2 else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let full = 16
        guard shape[0] == Self.neighbourhood * Self.neighbourhood * Self.channels + 1,
              shape[1] == full * full * 3,
              8 + headerBytes + offsets[1] <= data.count,
              offsets[1] - offsets[0] == shape[0] * shape[1] * 2 else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let base = 8 + headerBytes + offsets[0]
        var source = [Float](repeating: 0, count: shape[0] * shape[1])
        data.withUnsafeBytes { bytes in
            for index in source.indices {
                source[index] = Float(bytes.loadUnaligned(fromByteOffset: base + index * 2, as: Float16.self))
            }
        }
        // 16x16 -> 8x8: each output pixel the mean of a 2x2 block (a mean
        // of linear maps is the map of the mean).
        features = shape[0]
        let patch = Self.patch, scale = full / patch
        outputs = patch * patch * 3
        var weight = [Float](repeating: 0, count: features * outputs)
        for row in 0 ..< features {
            for py in 0 ..< full {
                for px in 0 ..< full {
                    for c in 0 ..< 3 {
                        let target = ((py / scale) * patch + px / scale) * 3 + c
                        weight[row * outputs + target] += source[row * shape[1] + (py * full + px) * 3 + c]
                            / Float(scale * scale)
                    }
                }
            }
        }
        self.weight = weight
    }

    /// The picture of a latent [24, height, width], half the image's size.
    func image(latent: [Float], height: Int, width: Int) -> CGImage? {
        guard latent.count >= Self.channels * height * width, height > 0, width > 0 else { return nil }
        let patch = Self.patch
        let outW = width * patch, outH = height * patch
        var pixels = [UInt8](repeating: 255, count: outW * outH * 4)
        var x = [Float](repeating: 0, count: width * features)
        var y = [Float](repeating: 0, count: width * outputs)
        let plane = height * width
        let r = Self.neighbourhood / 2
        for row in 0 ..< height {
            // One latent row at a time: its features, then one product.
            for column in 0 ..< width {
                var feature = column * features
                for dy in -r ... r {
                    let sy = min(max(row + dy, 0), height - 1)
                    for dx in -r ... r {
                        let sx = min(max(column + dx, 0), width - 1)
                        let at = sy * width + sx
                        for c in 0 ..< Self.channels {
                            x[feature] = latent[c * plane + at]
                            feature += 1
                        }
                    }
                }
                x[feature] = 1
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        Int32(width), Int32(outputs), Int32(features),
                        1, x, Int32(features), weight, Int32(outputs), 0, &y, Int32(outputs))
            for column in 0 ..< width {
                for py in 0 ..< patch {
                    let line = ((row * patch + py) * outW + column * patch) * 4
                    for px in 0 ..< patch {
                        let source = column * outputs + (py * patch + px) * 3
                        let target = line + px * 4
                        for c in 0 ..< 3 {
                            pixels[target + c] = UInt8(max(0, min(255, y[source + c] * 255 + 0.5)))
                        }
                    }
                }
            }
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        return CGImage(width: outW, height: outH, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: outW * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}

/// Draws image previews off the main thread, one at a time; one arriving
/// while the last is still drawn is dropped (the next step brings another).
final class ImageDenoisePreviewRenderer: @unchecked Sendable {
    private let queue = DispatchQueue(label: "h3c.image-preview", qos: .userInitiated)
    private let lock = NSLock()
    private var busy = false
    private var maps: [Bool: ImagePreviewMap] = [:]

    func render(_ preview: H3LatentPreview, isPH3: Bool, completion: @escaping @MainActor (CGImage?) -> Void) {
        lock.lock()
        if busy { lock.unlock(); return }
        busy = true
        lock.unlock()
        queue.async { [self] in
            if maps[isPH3] == nil { maps[isPH3] = ImagePreviewMap.load(isPH3: isPH3) }
            let image = maps[isPH3]?.image(latent: preview.latent, height: preview.height, width: preview.width)
            lock.lock(); busy = false; lock.unlock()
            Task { @MainActor in completion(image) }
        }
    }
}
