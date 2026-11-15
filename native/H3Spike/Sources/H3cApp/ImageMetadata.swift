import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

// PNG files of the image side: reading any still image the system can
// decode, and writing a PNG that says inside what made it - the still
// counterpart of GenerationMetadata (which writes the same fields into an
// MP4). The XMP goes into the PNG's iTXt chunk "XML:com.adobe.xmp", which
// exiftool and Adobe's tools read:
//   - Iptc4xmpExt:DigitalSourceType trainedAlgorithmicMedia, AISystemUsed,
//     AISystemVersionUsed, and with the prompt AIPromptInformation;
//   - dc:description "AI-generated with MiniMax H3 Image";
//   - xmp:CreatorTool, xmp:CreateDate.
// An image brought in from a file is not tagged: the app did not make it.
enum ImageMetadata {
    static let label = "AI-generated with MiniMax H3 Image"
    static let iptcExtNamespace = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
    static let trainedAlgorithmicMedia = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"

    /// What a generated image says about itself.
    struct Disclosure {
        var creatorTool: String
        var createdAt: Date
        /// The model family ("MiniMax H3 Image") and the weights' variant.
        var system: String
        var systemVersion: String?
        /// Left out when the prompt is not to be recorded.
        var prompt: String?
    }

    /// The first image of a file, upright (its EXIF orientation applied).
    static func loadImage(at url: URL) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int else { return nil }
        let orientation = properties[kCGImagePropertyOrientation] as? UInt32 ?? 1
        if orientation == 1 { return CGImageSourceCreateImageAtIndex(source, 0, nil) }
        // The "thumbnail" at full size is the image with its orientation
        // applied.
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(width, height),
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }

    /// An image file as the PNG a project keeps, with its pixel size. A PNG
    /// is taken as it is; anything else is converted, upright, with the
    /// metadata it carried (so what it says about its origin - made by a
    /// camera, or by an AI system - is not lost on the way in).
    /// The picture's size as it is shown (its orientation applied), read
    /// from the file's properties without decoding it.
    static func pixelSize(atPath path: String) -> (width: Int, height: Int)? {
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int, width > 0, height > 0 else { return nil }
        // Orientations 5 to 8 turn the picture a quarter.
        let orientation = properties[kCGImagePropertyOrientation] as? Int ?? 1
        return orientation >= 5 ? (height, width) : (width, height)
    }

    static func pngForImport(at url: URL) -> (png: Data, width: Int, height: Int)? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = loadImage(at: url) else { return nil }
        if let type = CGImageSourceGetType(source), UTType(type as String) == .png,
           let bytes = try? Data(contentsOf: url) {
            return (bytes, image.width, image.height)
        }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data, UTType.png.identifier as CFString, 1, nil) else { return nil }
        if let carried = CGImageSourceCopyMetadataAtIndex(source, 0, nil),
           let metadata = CGImageMetadataCreateMutableCopy(carried) {
            // The pixels are upright now: the old orientation no longer holds.
            CGImageMetadataRemoveTagWithPath(metadata, nil, "tiff:Orientation" as CFString)
            CGImageDestinationAddImageAndMetadata(destination, image, metadata, nil)
        } else {
            CGImageDestinationAddImage(destination, image, nil)
        }
        guard CGImageDestinationFinalize(destination) else { return nil }
        return (data as Data, image.width, image.height)
    }

    /// A copy no longer than `maxSide` on its longer side, upright.
    static func thumbnail(at url: URL, maxSide: Int) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxSide,
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }

    /// The image as PNG bytes; with `disclosure`, tagged as AI-generated.
    static func pngData(_ image: CGImage, disclosure: Disclosure? = nil) -> Data? {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data, UTType.png.identifier as CFString, 1, nil) else { return nil }
        if let disclosure, let metadata = xmp(disclosure) {
            CGImageDestinationAddImageAndMetadata(destination, image, metadata, nil)
        } else {
            CGImageDestinationAddImage(destination, image, nil)
        }
        return CGImageDestinationFinalize(destination) ? data as Data : nil
    }

    static func jpegData(_ image: CGImage, quality: Double) -> Data? {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data, UTType.jpeg.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(destination, image,
                                   [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        return CGImageDestinationFinalize(destination) ? data as Data : nil
    }

    private static func xmp(_ disclosure: Disclosure) -> CGImageMetadata? {
        let metadata = CGImageMetadataCreateMutable()
        var failure: Unmanaged<CFError>?
        guard CGImageMetadataRegisterNamespaceForPrefix(
            metadata, iptcExtNamespace as CFString, "Iptc4xmpExt" as CFString, &failure) else { return nil }
        func set(_ namespace: CFString, _ prefix: CFString, _ name: String, _ value: String) {
            guard let tag = CGImageMetadataTagCreate(namespace, prefix, name as CFString, .string,
                                                     value as CFString) else { return }
            CGImageMetadataSetTagWithPath(metadata, nil, "\(prefix):\(name)" as CFString, tag)
        }
        set(kCGImageMetadataNamespaceDublinCore, kCGImageMetadataPrefixDublinCore, "description", label)
        set(kCGImageMetadataNamespaceXMPBasic, kCGImageMetadataPrefixXMPBasic, "CreatorTool", disclosure.creatorTool)
        set(kCGImageMetadataNamespaceXMPBasic, kCGImageMetadataPrefixXMPBasic, "CreateDate",
            ISO8601DateFormatter().string(from: disclosure.createdAt))
        let iptc = iptcExtNamespace as CFString, prefix = "Iptc4xmpExt" as CFString
        set(iptc, prefix, "DigitalSourceType", trainedAlgorithmicMedia)
        set(iptc, prefix, "AISystemUsed", disclosure.system)
        if let version = disclosure.systemVersion { set(iptc, prefix, "AISystemVersionUsed", version) }
        if let prompt = disclosure.prompt { set(iptc, prefix, "AIPromptInformation", prompt) }
        return metadata
    }
}
