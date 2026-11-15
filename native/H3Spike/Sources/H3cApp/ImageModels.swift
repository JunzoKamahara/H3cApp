import CryptoKit
import Foundation

// Still images (plans/IMAGE_PLAN.ja.md, phase 1): what a window makes -
// videos or images - and the image side's own size, draft and record types.
// A project keeps its images next to its videos:
//
//   images/20261115-101500_seed42.png               each image
//   images/20261115-101500_seed42.json              its record (ProjectImageRecord)
//   images/20261115-101500_seed42.latent.safetensors  its latent, when one was kept
//   images/.thumbnails/                             small copies, made again when missing

/// What the composer makes. The video side's 作り方 (CreationMethod, "from
/// an image") is a different thing and keeps its name.
enum GenerationKind: String, CaseIterable, Identifiable, Codable {
    case video, image
    var id: String { rawValue }

    var label: String {
        switch self {
        case .video: return String(localized: "動画")
        case .image: return String(localized: "画像")
        }
    }

    var systemImage: String {
        switch self {
        case .video: return "video"
        case .image: return "photo"
        }
    }
}

/// The two preset classes of an image (plans/IMAGE_PLAN.ja.md, appendix A.2).
enum ImageSizeTier: String, CaseIterable, Identifiable {
    case standard, large
    var id: String { rawValue }

    var label: String {
        switch self {
        case .standard: return String(localized: "標準")
        case .large: return String(localized: "大")
        }
    }
}

// The size an image is generated at: a preset (shape x 標準/大) or a given
// canvas - the video canvases, for an image that starts a video. Apart from
// SizeProfile: no 2x enlargement and none of the video's pixel limit.
struct ImageSizeProfile: Hashable, Identifiable {
    enum Base: Hashable {
        case preset(AspectShape, ImageSizeTier)
        case canvas(width: Int32, height: Int32)
    }

    var base: Base

    var id: String { rawValue }

    static let `default` = ImageSizeProfile(base: .preset(.square, .standard))

    /// Sides the image model takes: multiples of 32.
    static let sideStep: Int32 = 32
    static let sideRange: ClosedRange<Int32> = 256 ... 2688

    static func preset(_ shape: AspectShape, _ tier: ImageSizeTier) -> ImageSizeProfile {
        ImageSizeProfile(base: .preset(shape, tier))
    }

    static func canvas(width: Int32, height: Int32) -> ImageSizeProfile {
        ImageSizeProfile(base: .canvas(width: width, height: height))
    }

    /// The video canvas of this shape (the largest preset): an image of this
    /// size goes into a video of the same size as it is.
    static func videoCanvas(_ shape: AspectShape) -> ImageSizeProfile {
        let canvas = SizeProfile.preset(shape, .large).canvas
        return .canvas(width: canvas.width, height: canvas.height)
    }

    static func isWellFormedCanvas(width: Int32, height: Int32) -> Bool {
        sideRange.contains(width) && sideRange.contains(height)
            && width % sideStep == 0 && height % sideStep == 0
    }

    var dimensions: (width: Int32, height: Int32) {
        switch base {
        case .preset(let shape, let tier):
            let landscape: (Int32, Int32)
            switch (shape, tier) {
            case (.square, .standard): return (1024, 1024)
            case (.square, .large): return (2048, 2048)
            case (_, .standard): landscape = (1344, 768)
            case (_, .large): landscape = (2688, 1536)
            }
            return shape == .portrait ? (landscape.1, landscape.0) : landscape
        case .canvas(let width, let height):
            return (width, height)
        }
    }

    var shape: AspectShape {
        switch base {
        case .preset(let shape, _): return shape
        case .canvas(let width, let height):
            return width > height ? .landscape : width < height ? .portrait : .square
        }
    }

    var tier: ImageSizeTier? {
        if case .preset(_, let tier) = base { return tier }
        return nil
    }

    /// Whether it is one of the video canvases (動画用).
    var isVideoCanvas: Bool { self == Self.videoCanvas(shape) }

    var pixels: Int { Int(dimensions.width) * Int(dimensions.height) }

    var resolutionLabel: String { "\(dimensions.width)×\(dimensions.height)" }

    /// "標準", "大", "動画用", or the size itself.
    var tierLabel: String {
        if let tier { return tier.label }
        return isVideoCanvas ? String(localized: "動画用") : resolutionLabel
    }

    var label: String { "\(shape.label) \(resolutionLabel)" }

    /// "<shape>-<tier>" or "canvas-<W>x<H>".
    var rawValue: String {
        switch base {
        case .preset(let shape, let tier): return "\(shape.rawValue)-\(tier.rawValue)"
        case .canvas(let width, let height): return "canvas-\(width)x\(height)"
        }
    }

    init(base: Base) {
        self.base = base
    }

    /// The size for an edit of a picture of this size: the picture's own
    /// when the image model takes it (a preset, if it is one), else its
    /// shape at about a megapixel, sides rounded to the step.
    static func matching(width: Int, height: Int) -> ImageSizeProfile {
        guard width > 0, height > 0 else { return .default }
        func wellFormed(_ w: Int, _ h: Int) -> ImageSizeProfile? {
            guard w <= Int(Int32.max), h <= Int(Int32.max),
                  isWellFormedCanvas(width: Int32(w), height: Int32(h)) else { return nil }
            for shape in AspectShape.allCases {
                for tier in ImageSizeTier.allCases {
                    let preset = ImageSizeProfile.preset(shape, tier)
                    if preset.dimensions == (Int32(w), Int32(h)) { return preset }
                }
            }
            return .canvas(width: Int32(w), height: Int32(h))
        }
        if let own = wellFormed(width, height) { return own }
        let scale = (1024.0 * 1024.0 / (Double(width) * Double(height))).squareRoot()
        let step = Double(sideStep)
        func side(_ value: Int) -> Int {
            let rounded = Int((Double(value) * scale / step).rounded() * step)
            return min(max(rounded, Int(sideRange.lowerBound)), Int(sideRange.upperBound))
        }
        return wellFormed(side(width), side(height)) ?? .default
    }

    init?(rawValue: String) {
        let parts = rawValue.split(separator: "-").map(String.init)
        guard parts.count == 2 else { return nil }
        if parts[0] == "canvas" {
            let size = parts[1].split(separator: "x").compactMap { Int32($0) }
            guard size.count == 2, Self.isWellFormedCanvas(width: size[0], height: size[1]) else { return nil }
            self = .canvas(width: size[0], height: size[1])
        } else if let shape = AspectShape(rawValue: parts[0]), let tier = ImageSizeTier(rawValue: parts[1]) {
            self = .preset(shape, tier)
        } else {
            return nil
        }
    }
}

let imageCountRange = 1 ... 20

/// The image side of the form, kept in project.json next to the video
/// draft (ProjectFile.imageDraft). Absent in projects from before images.
struct ImageDraft: Codable, Equatable {
    var prompt: String
    /// The image model's id in the library (H3ModelEntry.id); nil: the
    /// active one.
    var model: String? = nil
    var size: String
    var seedFixed: Bool
    var seed: String
    var count: Int
    /// Editing: the pictures the prompt calls <Picture 1>, <Picture 2>, ...
    /// in order - paths, relative when inside the project (as the video
    /// draft's inputs). Absent: an image from the prompt alone.
    var references: [String]? = nil

    /// The image side as a new window has it.
    static let blank = ImageDraft(prompt: "", model: nil, size: ImageSizeProfile.default.rawValue,
                                  seedFixed: false, seed: "", count: 1)
}

/// The .json next to each image in images/.
struct ProjectImageRecord: Codable, Equatable {
    /// The model that made it.
    struct Model: Codable, Equatable {
        var name: String
        /// Where its files came from (a repository) and at which revision.
        var source: String? = nil
        var revision: String? = nil
        /// Identifies the weights (a hash of their headers), whatever the
        /// folder is called.
        var fingerprint: String? = nil
    }

    /// The latent kept next to the image (<name>.latent.safetensors):
    /// enough to tell whether another model can take it as it is.
    struct Latent: Codable, Equatable {
        var file: String
        var width: Int
        var height: Int
        var channels: Int
        /// Which VAE's latent space, and how the values are normalized.
        var spaceID: String
        var normalization: String
        var dtype: String
        /// The order of the axes, e.g. "CHW".
        var layout: String
    }

    /// A picture the image was edited from.
    struct Reference: Codable, Equatable {
        /// The file, relative when inside the project (references/).
        var file: String
        /// SHA-256 of that file.
        var sha256: String? = nil
        /// The project image it is, or is a copy of (its assetID).
        var sourceImage: String? = nil
        /// The size its latent was encoded at.
        var width: Int
        var height: Int
        /// Its own latent was used instead of encoding the picture.
        var usedLatent: Bool? = nil
    }

    var version = 1
    /// Stays with the image wherever its files go: a video made from it
    /// names it (ProjectVideoRecord.sourceImage).
    var assetID: String
    var image: String
    /// SHA-256 of the PNG file.
    var sha256: String
    var completedAt: Date
    var width: Int
    var height: Int
    var appVersion: String
    /// The form as it ran, with this image's seed pinned. Absent for an
    /// image brought in from a file.
    var draft: ImageDraft? = nil
    var model: Model? = nil
    var seed: String? = nil
    var seedWasRandom: Bool? = nil
    var steps: Int? = nil
    var sigmas: [Double]? = nil
    var generationSeconds: Double? = nil
    var deviceLine: String? = nil
    var latent: Latent? = nil
    /// Why there is no latent although the image was generated.
    var latentMissingReason: String? = nil
    /// Brought in from this file rather than generated here.
    var importedFrom: String? = nil
    /// The pictures it was edited from, in the prompt's order (absent: made
    /// from the prompt alone), and the time they took to encode.
    var references: [Reference]? = nil
    var referenceSeconds: Double? = nil

    var prompt: String? { draft?.prompt }
    var isEdit: Bool { !(references ?? []).isEmpty }
    var isImported: Bool { importedFrom != nil }
}

struct ProjectImage: Identifiable, Equatable {
    var url: URL
    var record: ProjectImageRecord?
    var id: String { url.path }

    /// Its latent, if the file is there.
    var latentURL: URL? {
        let url = ProjectFiles.imageLatentURL(for: url)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }
}

enum ProjectImageError: LocalizedError {
    case unreadable(String)
    case cannotEncode

    var errorDescription: String? {
        switch self {
        case .unreadable(let name):
            return String(localized: "画像として読み込めません: \(name)")
        case .cannotEncode:
            return String(localized: "画像をPNGに変換できませんでした。")
        }
    }
}

extension ProjectFiles {
    static let imagesFolder = "images"
    static let thumbnailsFolder = ".thumbnails"
    static let imageLatentSuffix = ".latent.safetensors"

    static func imagesURL(in projectURL: URL) -> URL {
        projectURL.appendingPathComponent(imagesFolder, isDirectory: true)
    }

    static func imageRecordURL(for image: URL) -> URL {
        image.deletingPathExtension().appendingPathExtension("json")
    }

    static func imageLatentURL(for image: URL) -> URL {
        let base = image.deletingPathExtension()
        return base.deletingLastPathComponent().appendingPathComponent(base.lastPathComponent + imageLatentSuffix)
    }

    /// Where the small copy of an image is cached (made again when missing).
    static func imageThumbnailURL(for image: URL) -> URL {
        image.deletingLastPathComponent().appendingPathComponent(thumbnailsFolder, isDirectory: true)
            .appendingPathComponent(image.deletingPathExtension().lastPathComponent + ".jpg")
    }

    /// "20261115-101500_seed42" (or "…_imported" without a seed), unique
    /// within the folder.
    static func imageBaseName(completedAt: Date, seed: UInt64?, in folder: URL) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let base = formatter.string(from: completedAt) + (seed.map { "_seed\($0)" } ?? "_imported")
        var name = base
        var counter = 2
        while FileManager.default.fileExists(atPath: folder.appendingPathComponent(name + ".png").path) {
            name = "\(base)-\(counter)"
            counter += 1
        }
        return name
    }

    /// The project's images, newest first (the file names sort by time).
    static func images(in projectURL: URL) -> [ProjectImage] {
        let folder = imagesURL(in: projectURL)
        let names = (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []
        // Not hidden files: an image being written.
        let images = names.filter { !$0.hasPrefix(".") && $0.lowercased().hasSuffix(".png") }.map { name -> ProjectImage in
            let url = folder.appendingPathComponent(name)
            return ProjectImage(url: url, record: imageRecord(for: url))
        }
        return images.sorted { $0.url.lastPathComponent > $1.url.lastPathComponent }
    }

    static func imageRecord(for image: URL) -> ProjectImageRecord? {
        (try? Data(contentsOf: imageRecordURL(for: image)))
            .flatMap { try? decoder.decode(ProjectImageRecord.self, from: $0) }
    }

    /// Whether an image made (or brought in) in the project is in images/.
    static func containsImage(_ folder: URL) -> Bool {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: imagesURL(in: folder).path)) ?? []
        return names.contains { !$0.hasPrefix(".") && $0.lowercased().hasSuffix(".png") }
    }

    /// Whether the project holds anything it made: a video or an image.
    /// An untitled project without either has nothing to keep.
    static func containsOutput(_ folder: URL) -> Bool {
        containsImage(folder) || containsVideo(folder)
    }

    static func sha256(of data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Writes an image into the project's images/ with its record and, if
    /// there is one, its latent. The PNG and the record are written under
    /// hidden names and only then given their own, so the list never shows
    /// half of an image; a latent that cannot be kept leaves the image in
    /// place, with the reason in its record. `record` is given the image's
    /// file name and hash to fill in. Writes files: call it off the main actor.
    static func storeImage(png: Data, latent: (data: Data, description: ProjectImageRecord.Latent)?,
                           completedAt: Date, seed: UInt64?, into projectURL: URL,
                           record make: (_ imageName: String, _ sha256: String) -> ProjectImageRecord) throws -> ProjectImage {
        let folder = imagesURL(in: projectURL)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let base = imageBaseName(completedAt: completedAt, seed: seed, in: folder)
        let imageURL = folder.appendingPathComponent(base + ".png")
        let recordURL = imageRecordURL(for: imageURL)
        var record = make(imageURL.lastPathComponent, sha256(of: png))
        if let latent {
            let latentURL = imageLatentURL(for: imageURL)
            do {
                try latent.data.write(to: latentURL, options: .atomic)
                var description = latent.description
                description.file = latentURL.lastPathComponent
                record.latent = description
            } catch {
                record.latent = nil
                record.latentMissingReason = error.localizedDescription
            }
        }
        let pendingImage = folder.appendingPathComponent("." + base + ".png.writing")
        let pendingRecord = folder.appendingPathComponent("." + base + ".json.writing")
        do {
            try png.write(to: pendingImage)
            try encoder.encode(record).write(to: pendingRecord)
            try FileManager.default.moveItem(at: pendingRecord, to: recordURL)
            try FileManager.default.moveItem(at: pendingImage, to: imageURL)
        } catch {
            for url in [pendingImage, pendingRecord, recordURL, imageLatentURL(for: imageURL)] {
                try? FileManager.default.removeItem(at: url)
            }
            throw error
        }
        return ProjectImage(url: imageURL, record: record)
    }

    /// Moves an image with its record and latent into another project's
    /// images/, under its own name unless that is taken there (then with
    /// "-2", "-3"...). The thumbnail stays behind as nothing: it is made
    /// again. Returns where the image is now. Moves files: off the main actor.
    static func moveImage(_ image: URL, to destinationProject: URL) throws -> URL {
        let folder = imagesURL(in: destinationProject)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let stem = image.deletingPathExtension().lastPathComponent
        var base = stem
        var counter = 2
        while FileManager.default.fileExists(atPath: folder.appendingPathComponent(base + ".png").path) {
            base = "\(stem)-\(counter)"
            counter += 1
        }
        let destination = folder.appendingPathComponent(base + ".png")
        // The record first, rewritten for the new name; the image last, so
        // neither project ever lists an image without its record.
        var moved: [URL] = []
        do {
            let latent = imageLatentURL(for: image)
            let hasLatent = FileManager.default.fileExists(atPath: latent.path)
            if var record = imageRecord(for: image) {
                record.image = destination.lastPathComponent
                if hasLatent { record.latent?.file = imageLatentURL(for: destination).lastPathComponent }
                try encoder.encode(record).write(to: imageRecordURL(for: destination), options: .withoutOverwriting)
                moved.append(imageRecordURL(for: destination))
            }
            if hasLatent {
                try FileManager.default.moveItem(at: latent, to: imageLatentURL(for: destination))
                moved.append(imageLatentURL(for: destination))
            }
            try FileManager.default.moveItem(at: image, to: destination)
        } catch {
            // Back as it was: the latent returns, the new record goes.
            let latent = imageLatentURL(for: destination)
            if moved.contains(latent) { try? FileManager.default.moveItem(at: latent, to: imageLatentURL(for: image)) }
            try? FileManager.default.removeItem(at: imageRecordURL(for: destination))
            throw error
        }
        try? FileManager.default.removeItem(at: imageRecordURL(for: image))
        try? FileManager.default.removeItem(at: imageThumbnailURL(for: image))
        return destination
    }

    /// Next to a copy made by importReference: the image's record and
    /// latent, when `source` is an image of a project (it has a record
    /// naming it). The copy then still says which image it is and can be
    /// used as a latent. Regular files only, never through a link.
    static func importImageSidecars(of source: String, toCopy copy: String) {
        let sourceURL = URL(fileURLWithPath: source), copyURL = URL(fileURLWithPath: copy)
        guard sourceURL.standardizedFileURL != copyURL.standardizedFileURL,
              sourceURL.pathExtension.lowercased() == "png",
              let recordData = regularFileData(imageRecordURL(for: sourceURL), limit: 1 << 20),
              var record = try? decoder.decode(ProjectImageRecord.self, from: recordData),
              record.image == sourceURL.lastPathComponent else { return }
        let copiedRecord = imageRecordURL(for: copyURL)
        guard !FileManager.default.fileExists(atPath: copiedRecord.path) else { return }
        record.image = copyURL.lastPathComponent
        if record.latent != nil,
           let latent = regularFileData(imageLatentURL(for: sourceURL), limit: 256 << 20),
           (try? latent.write(to: imageLatentURL(for: copyURL), options: .withoutOverwriting)) != nil {
            record.latent?.file = imageLatentURL(for: copyURL).lastPathComponent
        } else {
            record.latent = nil
        }
        if let encoded = try? encoder.encode(record) {
            try? encoded.write(to: copiedRecord, options: .withoutOverwriting)
        }
    }

    /// The bytes of a regular file (not a link to one) of at most `limit`.
    private static func regularFileData(_ url: URL, limit: Int) -> Data? {
        let fd = open(url.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else { return nil }
        let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_size <= off_t(limit) else { return nil }
        return try? handle.readToEnd()
    }

    /// The latent to condition a clip of this canvas with, for a frame
    /// image that is a generated image (or a copy of one in references/)
    /// of exactly that size: the .latent.safetensors next to it, when its
    /// record describes a latent of this latent space and grid. nil
    /// otherwise - the image is then encoded.
    static func frameLatent(forImage path: String, canvasWidth: Int32, canvasHeight: Int32) -> String? {
        let url = URL(fileURLWithPath: path)
        guard url.pathExtension.lowercased() == "png",
              let data = regularFileData(imageRecordURL(for: url), limit: 1 << 20),
              let record = try? decoder.decode(ProjectImageRecord.self, from: data),
              record.image == url.lastPathComponent, let latent = record.latent,
              record.width == Int(canvasWidth), record.height == Int(canvasHeight),
              latent.width * 16 == record.width, latent.height * 16 == record.height,
              latent.channels == 24, latent.spaceID == imageLatentSpaceID,
              latent.normalization == imageLatentNormalization, latent.dtype == "F32" else { return nil }
        let file = imageLatentURL(for: url)
        return FileManager.default.fileExists(atPath: file.path) ? file.path : nil
    }

    /// The record next to an image file, when it is that file's.
    static func imageRecord(forImageAt path: String) -> ProjectImageRecord? {
        let url = URL(fileURLWithPath: path)
        guard let data = regularFileData(imageRecordURL(for: url), limit: 1 << 20),
              let record = try? decoder.decode(ProjectImageRecord.self, from: data),
              record.image == url.lastPathComponent else { return nil }
        return record
    }

    /// The image a frame file came from: the id in the record next to it
    /// (an image of a project, or a copy of one in references/).
    static func sourceImageID(forFrame path: String) -> String? {
        let url = URL(fileURLWithPath: path)
        guard url.pathExtension.lowercased() == "png",
              let data = regularFileData(imageRecordURL(for: url), limit: 1 << 20),
              let record = try? decoder.decode(ProjectImageRecord.self, from: data),
              record.image == url.lastPathComponent else { return nil }
        return record.assetID
    }
}
