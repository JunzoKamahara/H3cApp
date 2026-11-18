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
/// Which video model that is, and what its encoder is (each file's size,
/// the hash of its first mebibyte and the SHA-256 of the whole file), is
/// recorded in the image model's folder (H3IMAGE_TEXT_ENCODER.json): the
/// registered video models can hold different encoders (a GGUF, the int8
/// one), and the pictures follow the encoder. The recorded video model is
/// read while its folder holds that encoder. Without it - a disk not
/// connected, the model moved or deleted - another registered video model
/// is read only when its files are known to have the same SHA-256; one
/// that differs, or whose files have not been read through yet, is never
/// taken silently: the user chooses it.
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
        /// Of the whole file, where it has been read through and has not
        /// changed since (knownHash).
        var sha256: String?
    }

    private struct File {
        let path: String
        let bytes: Int64
        let modified: Date?
    }

    private static let headBytes = 1 << 20
    private static let lock = NSLock()
    nonisolated(unsafe) private static var headCache: [String: (bytes: Int64, modified: Date?, head: String)] = [:]

    /// The regular files of a folder (a link to one counts), by name.
    private static func regularFiles(in folder: String) -> [String: File] {
        var result: [String: File] = [:]
        for name in (try? FileManager.default.contentsOfDirectory(atPath: folder)) ?? [] where !name.hasPrefix(".") {
            let path = ((folder + "/" + name) as NSString).resolvingSymlinksInPath
            guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
                  attributes[.type] as? FileAttributeType == .typeRegular else { continue }
            result[name] = File(path: path, bytes: (attributes[.size] as? NSNumber)?.int64Value ?? 0,
                                modified: attributes[.modificationDate] as? Date)
        }
        return result
    }

    // The SHA-256 of whole files that have been read through, kept in the
    // app's folder by path with the size and date the file had: reading 18
    // to 27 GB takes a minute or more, so it is done once per file.
    private struct KnownHash: Codable {
        var bytes: Int64
        var modified: Double
        var sha256: String
    }

    private static let knownHashesPath = h3AppSupportDirectory + "/text-encoder-hashes.json"
    nonisolated(unsafe) private static var knownHashes: [String: KnownHash]?

    /// Call with `lock` held.
    private static func loadedHashes() -> [String: KnownHash] {
        if let knownHashes { return knownHashes }
        let loaded = FileManager.default.contents(atPath: knownHashesPath)
            .flatMap { try? JSONDecoder().decode([String: KnownHash].self, from: $0) } ?? [:]
        knownHashes = loaded
        return loaded
    }

    private static func knownHash(of file: File) -> String? {
        guard let modified = file.modified else { return nil }
        lock.lock()
        defer { lock.unlock() }
        guard let known = loadedHashes()[file.path], known.bytes == file.bytes,
              known.modified == modified.timeIntervalSinceReferenceDate else { return nil }
        return known.sha256
    }

    private static func remember(_ sha256: String, of file: File) {
        guard let modified = file.modified else { return }
        lock.lock()
        defer { lock.unlock() }
        var all = loadedHashes()
        all[file.path] = KnownHash(bytes: file.bytes, modified: modified.timeIntervalSinceReferenceDate, sha256: sha256)
        knownHashes = all
        if let data = try? JSONEncoder().encode(all) {
            try? data.write(to: URL(fileURLWithPath: knownHashesPath), options: .atomic)
        }
    }

    private static func mark(of file: File) -> FileMark? {
        lock.lock()
        let cached = headCache[file.path]
        lock.unlock()
        var head: String
        if let cached, cached.bytes == file.bytes, cached.modified == file.modified {
            head = cached.head
        } else {
            guard let handle = FileHandle(forReadingAtPath: file.path) else { return nil }
            let data = (try? handle.read(upToCount: headBytes)) ?? Data()
            try? handle.close()
            head = FileHash.sha256(of: data)
            lock.lock()
            headCache[file.path] = (file.bytes, file.modified, head)
            lock.unlock()
        }
        return FileMark(bytes: file.bytes, head: head, sha256: knownHash(of: file))
    }

    /// What the text encoder in `folder` is; nil when there is none. Reads
    /// no more than the head of each file (once): the whole file's SHA-256
    /// is there only where it is known already.
    static func fingerprint(of folder: String) -> [String: FileMark]? {
        let files = regularFiles(in: folder)
        guard !files.isEmpty else { return nil }
        var marks: [String: FileMark] = [:]
        for (name, file) in files {
            guard let mark = mark(of: file) else { return nil }
            marks[name] = mark
        }
        return marks
    }

    /// What the text encoder in `folder` is, with the SHA-256 of every
    /// file: those not known yet are read through (and then known).
    /// `progress`: bytes read so far. Stops on task cancellation.
    static func fullFingerprint(of folder: String,
                                progress: @escaping (Int64) -> Void = { _ in }) throws -> [String: FileMark] {
        var marks: [String: FileMark] = [:]
        var done: Int64 = 0
        for (name, file) in regularFiles(in: folder).sorted(by: { $0.key < $1.key }) {
            guard var mark = mark(of: file) else { throw ChoiceError.noTextEncoder(folder) }
            if mark.sha256 == nil {
                let sha256 = try FileHash.sha256(ofFile: file.path) { bytes in
                    done += bytes
                    progress(done)
                }
                remember(sha256, of: file)
                mark.sha256 = sha256
            } else {
                done += file.bytes
                progress(done)
            }
            marks[name] = mark
        }
        guard !marks.isEmpty else { throw ChoiceError.noTextEncoder(folder) }
        return marks
    }

    /// The bytes fullFingerprint reports reading for `folder`.
    static func bytes(of folder: String) -> Int64 {
        regularFiles(in: folder).values.reduce(0) { $0 + $1.bytes }
    }

    /// Files of the same names, sizes and heads. Not proof that the rest
    /// of each file is the same: enough to offer a comparison, and to know
    /// a folder still holds what it held.
    static func looksSame(_ a: [String: FileMark]?, _ b: [String: FileMark]?) -> Bool {
        guard let a, let b, !a.isEmpty, a.count == b.count else { return false }
        return a.allSatisfy { name, mark in
            guard let other = b[name], other.bytes == mark.bytes, other.head == mark.head else { return false }
            if let one = mark.sha256, let two = other.sha256 { return one == two }
            return true
        }
    }

    /// The same files, byte for byte: every file's SHA-256 is known on
    /// both sides and equal.
    static func isSame(_ a: [String: FileMark]?, _ b: [String: FileMark]?) -> Bool {
        guard let a, let b, looksSame(a, b) else { return false }
        return a.allSatisfy { name, mark in mark.sha256 != nil && mark.sha256 == b[name]?.sha256 }
    }

    // MARK: The record

    static let recordFile = "H3IMAGE_TEXT_ENCODER.json"

    struct Record: Codable, Equatable, Sendable {
        /// The video model folder chosen for the image model.
        var videoModel: String
        /// Its text encoder when it was chosen, each file with its SHA-256.
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
    /// model reads. `files`: that encoder, from fullFingerprint.
    static func choose(videoModel: String, files: [String: FileMark], forImageModel directory: String) throws {
        try write(Record(videoModel: videoModel, files: files,
                         removing: record(forImageModel: directory)?.removing), forImageModel: directory)
    }

    // MARK: Which one is read

    enum Reading: Equatable, Sendable {
        /// The folder's own text_encoder/ (also for a model that never
        /// reads a video model's, whether or not its own is there).
        case own
        /// A video model's, in place.
        case shared(videoModel: String)
        /// The recorded one is not to be found, and no registered video
        /// model is known to hold the same encoder: one must be chosen.
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
        // The chosen video model, while its folder holds what it held.
        if record.videoModel != gone,
           looksSame(fingerprint(of: folder(inVideoModel: record.videoModel)), record.files) {
            return .shared(videoModel: record.videoModel)
        }
        // Another only when its files are known to be the same, whole.
        if let twin = candidates.first(where: { isSame(fingerprint(of: folder(inVideoModel: $0)), record.files) }) {
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
        /// `next` is known to hold the same encoder: the pictures stay.
        let sameEncoder: Bool
        /// Other video models have an encoder, but none known to be the
        /// same: it makes nothing until one is chosen.
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
                                 sameEncoder: isSame(fingerprint(of: folder(inVideoModel: next)), mark),
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
            looksSame(fingerprint(of: folder(inVideoModel: $0)), mark)
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
        case leftover(String)
        case generating

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
            case .leftover(let path):
                return String(localized: "前の切り替えで残ったフォルダ「\(path)」があります。切り替えの記録と一致しないので、自動では削除しません。中身を確かめて片付けてから、もう一度お試しください。複製はそのままにしました。")
            case .generating:
                return String(localized: "画像を生成している間は、共用に切り替えられません。複製はそのままにしました。")
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
        guard !own.isEmpty, Set(own.keys) == Set(other.keys) else { throw SwitchError.nothingToGiveUp }
        for (name, copy) in own where copy.bytes != other[name]?.bytes { throw SwitchError.differs(name) }
        let half = own.values.reduce(Int64(0)) { $0 + $1.bytes }
        let marks = try fullFingerprint(of: folder(inVideoModel: videoModel)) { progress($0, 2 * half) }
        var done = half
        for name in own.keys.sorted() {
            guard let copy = own[name] else { continue }
            let sha256 = try FileHash.sha256(ofFile: copy.path) { bytes in
                done += bytes
                progress(done, 2 * half)
            }
            guard sha256 == marks[name]?.sha256 else { throw SwitchError.differs(name) }
        }
        return marks
    }

    /// Gives up the image model's own copy for the video model's, whose
    /// files `files` are (from compareContents: the same as the copy's).
    /// Checks that the engine takes the model with that encoder, records
    /// the video model, then deletes the copy (not to the Trash: it is a
    /// copy, and the space is the point). Deletes nothing else: with a
    /// text_encoder.removing in the way (finishInterruptedSwitch first)
    /// it throws SwitchError.leftover.
    static func switchToShared(imageModel directory: String, videoModel: String, files: [String: FileMark]) throws {
        let own = ownFolder(ofImageModel: directory)
        let leaving = own + ".removing"
        guard !FileManager.default.fileExists(atPath: leaving) else { throw SwitchError.leftover(leaving) }
        do {
            _ = try H3ImageEngine(modelDirectory: directory, textEncoderDirectory: folder(inVideoModel: videoModel))
        } catch {
            throw SwitchError.engine(error.localizedDescription)
        }
        // Renamed first, so a deletion that stops halfway does not leave a
        // text_encoder/ with some of its files for the engine to read; and
        // the record names what is being deleted before any of it goes, so
        // that a deletion cut short can be finished or undone later.
        let previous = record(forImageModel: directory)
        let manifest = regularFiles(in: own).mapValues(\.bytes)
        do {
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
        /// What the record of the switch names is still there (bytes): it
        /// could not be deleted, or it is not whole and nothing else can
        /// be read.
        case left(Int64)
        /// A text_encoder.removing that no record of a switch accounts for
        /// (bytes): never touched.
        case unrecognized(Int64)
    }

    /// The files of text_encoder.removing when they are what the record of
    /// a switch names - each listed there at its size, nothing else in the
    /// folder - and nil when the folder is not accounted for.
    private static func recognizedLeftover(_ leaving: String, record: Record?) -> [String: Int64]? {
        guard let manifest = record?.removing, !manifest.isEmpty else { return nil }
        let held = regularFiles(in: leaving).mapValues(\.bytes)
        let entries = Set((try? FileManager.default.contentsOfDirectory(atPath: leaving)) ?? []).subtracting([".DS_Store"])
        guard entries == Set(held.keys), held.allSatisfy({ manifest[$0.key] == $0.value }) else { return nil }
        return held
    }

    /// What is left of a copy that was being given up when the app was
    /// quit, crashed, or could not delete it. Only a text_encoder.removing
    /// that the record of the switch accounts for is acted on (each file
    /// listed there, at its size, and nothing else in the folder); any
    /// other is left as it is. With an encoder to read - the folder's own,
    /// or a video model's whose files are known to have the recorded
    /// SHA-256 - the rest of the copy goes. With none, the copy is put back
    /// when it still holds every file listed, and left otherwise: the last
    /// copy is never deleted.
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
        guard let held = recognizedLeftover(leaving, record: record), let record else {
            return .unrecognized(bytes(of: leaving))
        }
        let bytes = held.values.reduce(0, +)
        var canRead = !regularFiles(in: own).isEmpty
        if !canRead, case .shared(let source) = reading(forImageModel: directory, videoModelPaths: videoModelPaths) {
            canRead = isSame(fingerprint(of: folder(inVideoModel: source)), record.files)
        }
        if canRead {
            guard (try? FileManager.default.removeItem(atPath: leaving)) != nil else { return .left(bytes) }
            forget()
            return .removed
        }
        guard held == record.removing else { return .left(bytes) }
        do {
            // An emptied text_encoder/ may be in the way (rmdir: only if empty).
            if FileManager.default.fileExists(atPath: own) {
                try? FileManager.default.removeItem(atPath: own + "/.DS_Store")
                guard rmdir(own) == 0 else { return .left(bytes) }
            }
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

    /// A text_encoder.removing in the image model's folder, if any: its
    /// bytes, and whether the record of a switch accounts for it (only
    /// then is it ever deleted or put back).
    static func leftover(inImageModel directory: String) -> (bytes: Int64, recognized: Bool)? {
        let leaving = ownFolder(ofImageModel: directory) + ".removing"
        guard FileManager.default.fileExists(atPath: leaving) else { return nil }
        return (bytes(of: leaving), recognizedLeftover(leaving, record: record(forImageModel: directory)) != nil)
    }
}
