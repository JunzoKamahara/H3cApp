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
/// Which video model that is, and what its encoder is (each file's size and
/// the hash of its first mebibyte), is recorded in the image model's folder
/// (H3IMAGE_TEXT_ENCODER.json): the registered video models can hold
/// different encoders (a GGUF, the int8 one), and the pictures follow the
/// encoder. The recorded video model is read while it holds that encoder.
/// Without it - a disk not connected, the model moved or deleted - another
/// registered video model is read only when it holds the same encoder;
/// one that differs is never taken silently: the user chooses it.
enum SharedTextEncoder {
    static func folder(inVideoModel path: String) -> String { path + "/FL2VA/text_encoder" }

    static func holdsFiles(_ folder: String) -> Bool {
        !((try? FileManager.default.contentsOfDirectory(atPath: folder)) ?? []).isEmpty
    }

    static func ownFolder(ofImageModel directory: String) -> String {
        directory + "/" + H3ImageModel.textEncoderFolder
    }

    /// The image model folder has a text encoder of its own.
    static func hasOwn(imageModel directory: String) -> Bool {
        holdsFiles(ownFolder(ofImageModel: directory))
    }

    /// Whether the image model folder can read a video model's text
    /// encoder at all: not an install of P-H3-Image, whose own carries the
    /// text connector a video model's does not.
    static func applies(toImageModel directory: String) -> Bool {
        H3ImageModel.installed(at: directory)?.sharesTextEncoder ?? true
    }

    // MARK: What an encoder is

    /// One file of a text encoder folder.
    struct FileMark: Codable, Equatable, Sendable {
        var bytes: Int64
        /// SHA-256 of the first mebibyte (a GGUF's or safetensors' header:
        /// the tensors, their types and sizes).
        var head: String
        /// Of the whole file, once it has been read through (a copy given
        /// up for this one, compareContents).
        var sha256: String?
    }

    private static let headBytes = 1 << 20
    private static let markLock = NSLock()
    nonisolated(unsafe) private static var markCache: [String: (bytes: Int64, modified: Date?, head: String)] = [:]

    /// The regular files of a folder (a link to one counts), by name.
    private static func regularFiles(in folder: String) -> [String: (path: String, bytes: Int64, modified: Date?)] {
        var result: [String: (path: String, bytes: Int64, modified: Date?)] = [:]
        for name in (try? FileManager.default.contentsOfDirectory(atPath: folder)) ?? [] where !name.hasPrefix(".") {
            let path = ((folder + "/" + name) as NSString).resolvingSymlinksInPath
            guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
                  attributes[.type] as? FileAttributeType == .typeRegular else { continue }
            result[name] = (path, (attributes[.size] as? NSNumber)?.int64Value ?? 0,
                            attributes[.modificationDate] as? Date)
        }
        return result
    }

    /// What the text encoder in `folder` is; nil when there is none. The
    /// head of a file is read once, then kept while its size and date stay.
    static func fingerprint(of folder: String) -> [String: FileMark]? {
        let files = regularFiles(in: folder)
        guard !files.isEmpty else { return nil }
        var marks: [String: FileMark] = [:]
        for (name, file) in files {
            markLock.lock()
            let cached = markCache[file.path]
            markLock.unlock()
            if let cached, cached.bytes == file.bytes, cached.modified == file.modified {
                marks[name] = FileMark(bytes: file.bytes, head: cached.head)
                continue
            }
            guard let handle = FileHandle(forReadingAtPath: file.path) else { return nil }
            let data = (try? handle.read(upToCount: headBytes)) ?? Data()
            try? handle.close()
            let head = FileHash.sha256(of: data)
            markLock.lock()
            markCache[file.path] = (file.bytes, file.modified, head)
            markLock.unlock()
            marks[name] = FileMark(bytes: file.bytes, head: head)
        }
        return marks
    }

    /// The same files by name, size and head (and by the whole file's
    /// hash, where both sides know it).
    static func same(_ a: [String: FileMark]?, _ b: [String: FileMark]?) -> Bool {
        guard let a, let b, !a.isEmpty, a.count == b.count else { return false }
        return a.allSatisfy { name, mark in
            guard let other = b[name], other.bytes == mark.bytes, other.head == mark.head else { return false }
            if let one = mark.sha256, let two = other.sha256 { return one == two }
            return true
        }
    }

    // MARK: The record

    static let recordFile = "H3IMAGE_TEXT_ENCODER.json"

    struct Record: Codable, Equatable, Sendable {
        /// The video model folder chosen for the image model.
        var videoModel: String
        /// Its text encoder when it was chosen.
        var files: [String: FileMark]
        /// A copy being given up: the files of text_encoder.removing and
        /// their sizes, from before the first of them is deleted until the
        /// last is gone (finishInterruptedSwitch).
        var removing: [String: Int64]?
    }

    static func record(forImageModel directory: String) -> Record? {
        guard let data = FileManager.default.contents(atPath: directory + "/" + recordFile) else { return nil }
        return try? JSONDecoder().decode(Record.self, from: data)
    }

    static func write(_ record: Record?, forImageModel directory: String) throws {
        let path = directory + "/" + recordFile
        guard let record else {
            if FileManager.default.fileExists(atPath: path) { try FileManager.default.removeItem(atPath: path) }
            return
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(record).write(to: URL(fileURLWithPath: path), options: .atomic)
    }

    enum ChoiceError: LocalizedError {
        case noTextEncoder(String)

        var errorDescription: String? {
            switch self {
            case .noTextEncoder(let path):
                return String(localized: "「\(path)」にテキストエンコーダがありません。")
            }
        }
    }

    /// Records the video model as the one whose text encoder the image
    /// model reads, with what that encoder is now.
    static func choose(videoModel: String, forImageModel directory: String) throws {
        guard let files = fingerprint(of: folder(inVideoModel: videoModel)) else {
            throw ChoiceError.noTextEncoder(videoModel)
        }
        var record = Record(videoModel: videoModel, files: files, removing: nil)
        record.removing = self.record(forImageModel: directory)?.removing
        try write(record, forImageModel: directory)
    }

    // MARK: Which one is read

    enum Reading: Equatable, Sendable {
        /// The folder's own text_encoder/ (also for a model that never
        /// reads a video model's, whether or not its own is there).
        case own
        /// A video model's, in place.
        case shared(videoModel: String)
        /// The recorded one is not to be found and the registered video
        /// models hold a different encoder: one must be chosen.
        case needsChoice
        /// No video model has a text encoder.
        case missing
    }

    /// The registered video models that have a text encoder, in the order given.
    static func candidates(in videoModelPaths: [String]) -> [String] {
        var seen = Set<String>()
        return videoModelPaths.filter { seen.insert($0).inserted && holdsFiles(folder(inVideoModel: $0)) }
    }

    /// Where the image model at `directory` reads its text encoder.
    /// `without`: a video model taken as gone (what deleting it would do).
    static func reading(forImageModel directory: String, videoModelPaths: [String],
                        without gone: String? = nil) -> Reading {
        if hasOwn(imageModel: directory) || !applies(toImageModel: directory) { return .own }
        let candidates = candidates(in: videoModelPaths.filter { $0 != gone })
        guard let record = record(forImageModel: directory) else {
            // Nothing was chosen (a folder the user put together): the first.
            return candidates.first.map { .shared(videoModel: $0) } ?? .missing
        }
        if record.videoModel != gone,
           same(fingerprint(of: folder(inVideoModel: record.videoModel)), record.files) {
            return .shared(videoModel: record.videoModel)
        }
        if let twin = candidates.first(where: { same(fingerprint(of: folder(inVideoModel: $0)), record.files) }) {
            return .shared(videoModel: twin)
        }
        return candidates.isEmpty ? .missing : .needsChoice
    }

    /// The video model whose text encoder the image model reads in place.
    static func source(forImageModel directory: String, videoModelPaths: [String]) -> String? {
        if case .shared(let videoModel) = reading(forImageModel: directory, videoModelPaths: videoModelPaths) {
            return videoModel
        }
        return nil
    }

    /// What H3ImageEngine takes as textEncoderDirectory for the image model.
    static func directory(forImageModel directory: String, videoModelPaths: [String]) -> String? {
        source(forImageModel: directory, videoModelPaths: videoModelPaths).map(folder(inVideoModel:))
    }

    // MARK: Deleting a video model

    /// An image model that reads a video model's text encoder in place.
    struct Dependent: Identifiable, Sendable {
        var id: String { directory }
        let name: String
        let directory: String
        /// The video model whose encoder it reads once that one's files
        /// are gone; nil when it then reads none.
        let next: String?
        /// `next` holds the same encoder: the pictures stay the same.
        let sameEncoder: Bool
        /// Other video models have an encoder, but a different one: it
        /// makes nothing until one is chosen.
        let needsChoice: Bool
    }

    /// The image models that read the text encoder of the video model at
    /// `path`, and what deleting its files would do to each.
    static func dependents(onVideoModel path: String, imageModels: [(name: String, path: String)],
                           videoModelPaths: [String]) -> [Dependent] {
        let mark = fingerprint(of: folder(inVideoModel: path))
        return imageModels.compactMap { image in
            guard reading(forImageModel: image.path, videoModelPaths: videoModelPaths) == .shared(videoModel: path) else {
                return nil
            }
            switch reading(forImageModel: image.path, videoModelPaths: videoModelPaths, without: path) {
            case .shared(let next):
                return Dependent(name: image.name, directory: image.path, next: next,
                                 sameEncoder: same(fingerprint(of: folder(inVideoModel: next)), mark),
                                 needsChoice: false)
            case .needsChoice:
                return Dependent(name: image.name, directory: image.path, next: nil, sameEncoder: false, needsChoice: true)
            case .own, .missing:
                return Dependent(name: image.name, directory: image.path, next: nil, sameEncoder: false, needsChoice: false)
            }
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

    // MARK: Giving up a copy

    /// What giving up the image model's own copy would do.
    struct Switch: Sendable {
        /// The video model that seems to hold the same encoder.
        let videoModel: String
        /// The space the copy holds alone (nothing when it is a clone).
        let freedBytes: Int64
    }

    /// The image model's own text encoder could go: a registered video
    /// model holds files of the same names, sizes and heads, and the copy
    /// takes space of its own. Only an offer: the files are compared
    /// through before anything is deleted (compareContents).
    static func possibleSwitch(forImageModel directory: String, videoModelPaths: [String]) -> Switch? {
        let own = ownFolder(ofImageModel: directory)
        guard let mark = fingerprint(of: own) else { return nil }
        let ordered = [record(forImageModel: directory)?.videoModel].compactMap { $0 } + videoModelPaths
        guard let match = candidates(in: ordered).first(where: {
            same(fingerprint(of: folder(inVideoModel: $0)), mark)
        }) else { return nil }
        let freed = mark.keys.reduce(Int64(0)) { total, name in
            total + (CloneStat.of(own + "/" + name)?.privateSize ?? 0)
        }
        // Under a megabyte is a clone's bookkeeping, not a copy.
        return freed >= 1_000_000 ? Switch(videoModel: match, freedBytes: freed) : nil
    }

    enum SwitchError: LocalizedError {
        case nothingToGiveUp
        case differs(String)
        case engine(String)
        case remove(String)

        var errorDescription: String? {
            switch self {
            case .nothingToGiveUp:
                return String(localized: "共用にできる複製がありません（同じテキストエンコーダを持つ動画モデルが登録されていないか、複製が容量を占めていません）。")
            case .differs(let file):
                return String(localized: "複製の「\(file)」の内容が、動画モデルのものと一致しません（SHA-256が違います）。複製はそのままにしました。")
            case .engine(let detail):
                return String(localized: "動画モデルのテキストエンコーダでは画像モデルを開けませんでした。複製はそのままにしました（詳細: \(detail)）")
            case .remove(let detail):
                return String(localized: "複製したテキストエンコーダを削除できませんでした（詳細: \(detail)）")
            }
        }
    }

    /// Reads every file of the image model's own copy and of the video
    /// model's encoder through, and returns the video model's files with
    /// their SHA-256 when the two are the same, byte for byte. Throws
    /// SwitchError.differs otherwise, and CancellationError when the task
    /// is cancelled. Minutes for 18 to 27 GB on each side.
    static func compareContents(imageModel directory: String, videoModel: String,
                                progress: @escaping @Sendable (Int64, Int64) -> Void) throws -> [String: FileMark] {
        let own = regularFiles(in: ownFolder(ofImageModel: directory))
        let other = regularFiles(in: folder(inVideoModel: videoModel))
        guard var marks = fingerprint(of: folder(inVideoModel: videoModel)), !own.isEmpty,
              Set(own.keys) == Set(other.keys) else {
            throw SwitchError.nothingToGiveUp
        }
        let total = own.values.reduce(Int64(0)) { $0 + $1.bytes } + other.values.reduce(Int64(0)) { $0 + $1.bytes }
        var done: Int64 = 0
        for name in own.keys.sorted() {
            guard let copy = own[name], let original = other[name], copy.bytes == original.bytes else {
                throw SwitchError.differs(name)
            }
            let first = try FileHash.sha256(ofFile: copy.path) { bytes in
                done += bytes
                progress(done, total)
            }
            let second = try FileHash.sha256(ofFile: original.path) { bytes in
                done += bytes
                progress(done, total)
            }
            guard first == second else { throw SwitchError.differs(name) }
            marks[name]?.sha256 = second
        }
        return marks
    }

    /// Gives up the image model's own copy for the video model's, whose
    /// files `files` are (from compareContents: the same as the copy's).
    /// Checks that the engine takes the model with that encoder, records
    /// the video model, then deletes the copy (not to the Trash: it is a
    /// copy, and the space is the point).
    static func switchToShared(imageModel directory: String, videoModel: String, files: [String: FileMark]) throws {
        do {
            _ = try H3ImageEngine(modelDirectory: directory, textEncoderDirectory: folder(inVideoModel: videoModel))
        } catch {
            throw SwitchError.engine(error.localizedDescription)
        }
        // Renamed first, so a deletion that stops halfway does not leave a
        // text_encoder/ with some of its files for the engine to read; and
        // the record names what is being deleted before any of it goes, so
        // that a deletion cut short can be finished or undone later.
        let own = ownFolder(ofImageModel: directory)
        let leaving = own + ".removing"
        let previous = record(forImageModel: directory)
        let manifest = regularFiles(in: own).mapValues(\.bytes)
        do {
            if FileManager.default.fileExists(atPath: leaving) { try FileManager.default.removeItem(atPath: leaving) }
            try write(Record(videoModel: videoModel, files: files, removing: manifest), forImageModel: directory)
            try FileManager.default.moveItem(atPath: own, toPath: leaving)
        } catch {
            try? write(previous, forImageModel: directory)
            throw SwitchError.remove(error.localizedDescription)
        }
        do {
            try FileManager.default.removeItem(atPath: leaving)
            try write(Record(videoModel: videoModel, files: files, removing: nil), forImageModel: directory)
        } catch {
            throw SwitchError.remove(error.localizedDescription)
        }
    }

    // MARK: A switch cut short

    enum Leftover: Equatable, Sendable {
        /// Nothing was left behind.
        case none
        /// The rest of the copy was deleted.
        case removed
        /// The copy was whole and nothing else could be read: it is the
        /// model's text encoder again.
        case restored
        /// Still there (bytes): it could not be deleted, or it is not
        /// known to be whole and nothing else can be read.
        case left(Int64)
    }

    /// What is left of a copy that was being given up when the app was
    /// quit, crashed, or could not delete it. With an encoder to read (the
    /// video model's, or the folder's own) the rest of the copy goes. With
    /// none - the video model has gone since - the copy is put back when
    /// it still holds every file the record lists at its size, and left
    /// alone otherwise: the last copy is never deleted.
    @discardableResult
    static func finishInterruptedSwitch(imageModel directory: String, videoModelPaths: [String]) -> Leftover {
        let own = ownFolder(ofImageModel: directory)
        let leaving = own + ".removing"
        var record = record(forImageModel: directory)
        func forget() {
            guard record?.removing != nil else { return }
            record?.removing = nil
            try? write(record, forImageModel: directory)
        }
        guard FileManager.default.fileExists(atPath: leaving) else {
            forget()
            return .none
        }
        let held = regularFiles(in: leaving).mapValues(\.bytes)
        let bytes = held.values.reduce(0, +)
        var canRead = hasOwn(imageModel: directory)
        if !canRead, case .shared = reading(forImageModel: directory, videoModelPaths: videoModelPaths) { canRead = true }
        if canRead {
            guard (try? FileManager.default.removeItem(atPath: leaving)) != nil else { return .left(bytes) }
            forget()
            return .removed
        }
        guard let manifest = record?.removing, !manifest.isEmpty, manifest == held else { return .left(bytes) }
        do {
            // An emptied text_encoder/ may be in the way.
            if FileManager.default.fileExists(atPath: own) { try FileManager.default.removeItem(atPath: own) }
            try FileManager.default.moveItem(atPath: leaving, toPath: own)
        } catch {
            return .left(bytes)
        }
        forget()
        return .restored
    }

    /// At launch: the installs that can share a text encoder.
    static func finishInterruptedSwitches() {
        let videoModelPaths = ModelLibrary.storedVideoModelPaths()
        for model in H3ImageModel.all where model.sharesTextEncoder {
            finishInterruptedSwitch(imageModel: model.directory, videoModelPaths: videoModelPaths)
        }
    }

    /// The bytes of a copy left behind in the image model's folder, if any.
    static func leftoverBytes(inImageModel directory: String) -> Int64? {
        let leaving = ownFolder(ofImageModel: directory) + ".removing"
        guard FileManager.default.fileExists(atPath: leaving) else { return nil }
        return regularFiles(in: leaving).values.reduce(Int64(0)) { $0 + $1.bytes }
    }
}
