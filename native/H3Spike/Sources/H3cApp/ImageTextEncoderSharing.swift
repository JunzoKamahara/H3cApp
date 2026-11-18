import Foundation
import H3Engine

/// The text encoder of an image model that keeps none of its own. MiniMax
/// H3 Image uses the video model's (the same Qwen3-VL language layers).
/// Where a copy would be an APFS clone - the image model on the video
/// model's volume - the app still makes one: it takes no space, and the
/// image model then does not depend on the video model. Anywhere else (an
/// image model on another disk) a copy would take the encoder's whole size
/// again, 18 to 27 GB, so the folder holds none and the engine is given the
/// video model's folder (h3_image_load_parts).
///
/// Which video model that is, is recorded per image model folder: the
/// registered video models can hold different encoders (a GGUF, the int8
/// one), and the pictures follow the encoder. The recorded one is used
/// while its files are there; without it - a disk not connected, the model
/// deleted - the first registered video model that has an encoder is.
enum SharedTextEncoder {
    static func folder(inVideoModel path: String) -> String { path + "/FL2VA/text_encoder" }

    static func holdsFiles(_ folder: String) -> Bool {
        !((try? FileManager.default.contentsOfDirectory(atPath: folder)) ?? []).isEmpty
    }

    /// The image model folder has a text encoder of its own.
    static func hasOwn(imageModel directory: String) -> Bool {
        holdsFiles(directory + "/" + H3ImageModel.textEncoderFolder)
    }

    // MARK: The recorded video model

    private static let sourcesKey = "h3c-app.imageModelTextEncoderSources"

    private static var sources: [String: String] {
        UserDefaults.standard.dictionary(forKey: sourcesKey) as? [String: String] ?? [:]
    }

    /// The video model folder recorded for the image model folder.
    static func recordedSource(forImageModel directory: String) -> String? {
        sources[directory]
    }

    static func record(videoModel path: String?, forImageModel directory: String) {
        var all = sources
        all[directory] = path
        if all.isEmpty {
            UserDefaults.standard.removeObject(forKey: sourcesKey)
        } else {
            UserDefaults.standard.set(all, forKey: sourcesKey)
        }
    }

    // MARK: Which one is read

    /// The registered video models that have a text encoder, in the order given.
    static func candidates(in videoModelPaths: [String]) -> [String] {
        var seen = Set<String>()
        return videoModelPaths.filter { seen.insert($0).inserted && holdsFiles(folder(inVideoModel: $0)) }
    }

    /// Whether the image model folder can read a video model's text
    /// encoder at all: not an install of P-H3-Image, whose own carries the
    /// text connector a video model's does not.
    static func applies(toImageModel directory: String) -> Bool {
        H3ImageModel.installed(at: directory)?.sharesTextEncoder ?? true
    }

    /// The video model whose text encoder the image model at `directory`
    /// reads: nil when it has its own, or when there is none to read.
    static func source(forImageModel directory: String, videoModelPaths: [String]) -> String? {
        if hasOwn(imageModel: directory) || !applies(toImageModel: directory) { return nil }
        if let recorded = recordedSource(forImageModel: directory), holdsFiles(folder(inVideoModel: recorded)) {
            return recorded
        }
        return candidates(in: videoModelPaths).first
    }

    /// What H3ImageEngine takes as textEncoderDirectory for the image model.
    static func directory(forImageModel directory: String, videoModelPaths: [String]) -> String? {
        source(forImageModel: directory, videoModelPaths: videoModelPaths).map(folder(inVideoModel:))
    }

    /// The image model has no text encoder to read: none of its own, and
    /// no video model with one.
    static func isMissing(forImageModel directory: String, videoModelPaths: [String]) -> Bool {
        !hasOwn(imageModel: directory) && applies(toImageModel: directory)
            && source(forImageModel: directory, videoModelPaths: videoModelPaths) == nil
    }

    // MARK: Deleting a video model

    /// An image model that reads a video model's text encoder in place.
    struct Dependent: Identifiable, Sendable {
        var id: String { directory }
        let name: String
        let directory: String
        /// The video model whose encoder it reads once that one's files
        /// are gone; nil when no other has one, and it can make nothing.
        let next: String?
    }

    /// The image models that read the text encoder of the video model at
    /// `path`, which deleting its files would move to another's or stop.
    static func dependents(onVideoModel path: String, imageModels: [(name: String, path: String)],
                           videoModelPaths: [String]) -> [Dependent] {
        let next = candidates(in: videoModelPaths.filter { $0 != path }).first
        return imageModels.compactMap { image in
            source(forImageModel: image.path, videoModelPaths: videoModelPaths) == path
                ? Dependent(name: image.name, directory: image.path, next: next) : nil
        }
    }

    // MARK: Copy or read in place

    /// Whether copying `source` into `destination` makes clones: both on
    /// one volume, and that volume clones files (APFS).
    static func copyIsClone(from source: String, to destination: String) -> Bool {
        let keys: Set<URLResourceKey> = [.volumeIdentifierKey, .volumeSupportsFileCloningKey]
        // A destination not made yet is on the volume of the folder it will be made in.
        var target = URL(fileURLWithPath: destination)
        while target.path != "/", !FileManager.default.fileExists(atPath: target.path) {
            target.deleteLastPathComponent()
        }
        guard let a = try? URL(fileURLWithPath: source).resourceValues(forKeys: keys),
              let b = try? target.resourceValues(forKeys: keys),
              let volumeA = a.volumeIdentifier, let volumeB = b.volumeIdentifier else { return false }
        return volumeA.isEqual(volumeB) && a.volumeSupportsFileCloning == true
    }

    /// The weight files of a text encoder folder with their sizes.
    private static func files(in folder: String) -> [String: Int64] {
        var result: [String: Int64] = [:]
        for name in (try? FileManager.default.contentsOfDirectory(atPath: folder)) ?? [] where !name.hasPrefix(".") {
            let path = folder + "/" + name
            guard let attributes = try? FileManager.default.attributesOfItem(atPath: (path as NSString).resolvingSymlinksInPath),
                  attributes[.type] as? FileAttributeType == .typeRegular else { continue }
            result[name] = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        }
        return result
    }

    /// What giving up the image model's own copy would do.
    struct Switch {
        /// The video model that holds the same encoder.
        let videoModel: String
        /// The space the copy holds alone (nothing when it is a clone).
        let freedBytes: Int64
    }

    /// The image model's own text encoder can go: a registered video model
    /// holds the same files (names and sizes), and the copy takes space of
    /// its own. Reads file attributes: off the main actor for a slow disk.
    static func possibleSwitch(forImageModel directory: String, videoModelPaths: [String]) -> Switch? {
        let own = directory + "/" + H3ImageModel.textEncoderFolder
        let ownFiles = files(in: own)
        guard !ownFiles.isEmpty else { return nil }
        let recorded = recordedSource(forImageModel: directory)
        let ordered = ([recorded].compactMap { $0 } + videoModelPaths)
        guard let match = candidates(in: ordered).first(where: { files(in: folder(inVideoModel: $0)) == ownFiles }) else {
            return nil
        }
        let freed = ownFiles.keys.reduce(Int64(0)) { total, name in
            total + (CloneStat.of(own + "/" + name)?.privateSize ?? 0)
        }
        // Under a megabyte is a clone's bookkeeping, not a copy.
        return freed >= 1_000_000 ? Switch(videoModel: match, freedBytes: freed) : nil
    }

    enum SwitchError: LocalizedError {
        case engine(String)
        case remove(String)

        var errorDescription: String? {
            switch self {
            case .engine(let detail):
                return String(localized: "動画モデルのテキストエンコーダでは画像モデルを開けませんでした。複製はそのままにしました（詳細: \(detail)）")
            case .remove(let detail):
                return String(localized: "複製したテキストエンコーダを削除できませんでした（詳細: \(detail)）")
            }
        }
    }

    /// Gives up the image model's own copy for the video model's: checks
    /// that the engine takes the model with that encoder, records the video
    /// model, then deletes the copy (not to the Trash: it is a copy, and
    /// the space is the point).
    static func switchToShared(imageModel directory: String, videoModel: String) throws {
        do {
            _ = try H3ImageEngine(modelDirectory: directory, textEncoderDirectory: folder(inVideoModel: videoModel))
        } catch {
            throw SwitchError.engine(error.localizedDescription)
        }
        // Renamed first, so a deletion that stops halfway does not leave a
        // text_encoder/ with some of its files for the engine to read.
        let own = directory + "/" + H3ImageModel.textEncoderFolder
        let leaving = own + ".removing"
        let previous = recordedSource(forImageModel: directory)
        record(videoModel: videoModel, forImageModel: directory)
        do {
            try? FileManager.default.removeItem(atPath: leaving)
            try FileManager.default.moveItem(atPath: own, toPath: leaving)
        } catch {
            record(videoModel: previous, forImageModel: directory)
            throw SwitchError.remove(error.localizedDescription)
        }
        do {
            try FileManager.default.removeItem(atPath: leaving)
        } catch {
            throw SwitchError.remove(error.localizedDescription)
        }
    }
}
