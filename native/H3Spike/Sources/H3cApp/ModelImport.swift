import Foundation
import H3Engine
import SwiftUI
import UniformTypeIdentifiers

// Importing quantized (GGUF) checkpoints as a model folder the engine loads
// like a released one: FL2VA/ (and optionally Ref2VA/) with the DiT and text
// encoder from GGUF files and everything else - tokenizer, processor, VAEs,
// configs - cloned from a registered model. On APFS, FileManager.copyItem
// clones, so those copies take no extra space while the base model exists
// and stay valid if it is deleted later. The text encoder is rewritten
// with only the 50 layers the engine reads, and a text encoder without a
// vision tower gets the base model's (extracted, about 1.1 GB).

let ggufTypes: [UTType] = [UTType(filenameExtension: "gguf") ?? .data]

/// Where imported models go: beside the default model folder, outside the
/// folders macOS asks permission for.
let importedModelsDirectory = h3AppSupportDirectory + "/Imported"

/// Written into an imported model folder: where its parts came from.
struct ModelImportManifest: Codable {
    struct Part: Codable {
        var role: String
        var file: String
        var source: String
        var type: String
        var bytes: UInt64
    }
    var importedAt: Date
    var baseModel: String
    var parts: [Part]

    static let fileName = "h3c-import.json"

    static func load(modelPath: String) -> ModelImportManifest? {
        let url = URL(fileURLWithPath: modelPath).appendingPathComponent(fileName)
        guard let data = try? Data(contentsOf: url) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(ModelImportManifest.self, from: data)
    }

    func save(modelPath: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(self).write(
            to: URL(fileURLWithPath: modelPath).appendingPathComponent(Self.fileName))
    }

    /// e.g. "DiT Q8_0・文章 Q8_0" for the model list.
    var summary: String {
        parts.filter { $0.role != "vision" }.map { part in
            let label = part.role == "textEncoder" ? String(localized: "文章") :
                part.role == "ref2va" ? "Ref2VA" : "DiT"
            return "\(label) \(part.type)"
        }.joined(separator: itemSeparator)
    }
}

/// One picked file and what h3_gguf_inspect found in it.
struct ImportPick: Equatable, Identifiable {
    let path: String
    let info: H3GGUFInfo?
    let error: String?
    var id: String { path }

    init(path: String) {
        self.path = path
        do {
            info = try H3GGUFInfo.inspect(path: path)
            error = nil
        } catch {
            info = nil
            self.error = error.localizedDescription
        }
    }

    var fileName: String { (path as NSString).lastPathComponent }

    /// A DiT file's own guess at which transformer it is: the two have the
    /// same tensors, so only the file name or general.name tells them apart.
    var looksLikeRef2VA: Bool {
        let text = (fileName + " " + (info?.name ?? "")).lowercased()
        return text.contains("ref2v")
    }
}

/// Where the vision tower of an import comes from.
enum ImportVisionSource: Equatable {
    case inTextEncoder          // the text-encoder GGUF holds it
    case file(ImportPick)       // an mmproj GGUF
    case extractFromBase        // copied out of the base model's text encoder
    case cloneFromBase(String)  // the base model's own vision file, cloned
    case baseTextEncoder        // the whole base text encoder is used
    case none
}

/// A stop request the trim/extract C routines poll from their own thread.
final class ImportCancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }
    func set(_ newValue: Bool) { lock.lock(); value = newValue; lock.unlock() }
}

@MainActor
final class ModelImporter: ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var step: String = ""
    @Published private(set) var fraction: Double?
    @Published var errorMessage: String?
    private var task: Task<Void, Never>?
    private let cancelFlag = ImportCancelFlag()
    private var cancelled: Bool { cancelFlag.isSet }

    struct Request {
        var name: String
        var dit: ImportPick
        var ref2vaDit: ImportPick?
        var textEncoder: ImportPick?
        var vision: ImportVisionSource
        var base: H3ModelEntry
        /// The model folder to fill; default a new one under Imported. It
        /// may exist already holding only the downloads (see
        /// importDownloadFolderName), which are left in place.
        var outputPath: String? = nil
    }

    func start(_ request: Request, onFinished: @escaping (String) -> Void) {
        guard !isRunning else { return }
        isRunning = true
        cancelFlag.set(false)
        errorMessage = nil
        task = Task {
            do {
                let path = try await run(request)
                onFinished(path)
            } catch is CancellationError {
                // the user stopped it - the partial folder is already gone
            } catch {
                NSLog("H3cApp: GGUF import failed: %@", error.localizedDescription)
                errorMessage = String(localized: "取り込めませんでした（詳細: \(error.localizedDescription)）")
            }
            isRunning = false
            step = ""
            fraction = nil
        }
    }

    func cancel() {
        cancelFlag.set(true)
        task?.cancel()
    }

    /// The same import, awaited (for the automation API).
    func importNow(_ request: Request) async throws -> String {
        guard !isRunning else { throw CancellationError() }
        isRunning = true
        cancelFlag.set(false)
        defer { isRunning = false; step = ""; fraction = nil }
        return try await run(request)
    }

    private func run(_ request: Request) async throws -> String {
        let fm = FileManager.default
        let root = request.outputPath ?? Self.uniqueFolder(name: request.name)
        if let existing = try? fm.contentsOfDirectory(atPath: root),
           existing.contains(where: { $0 != importDownloadFolderName && $0 != ".DS_Store" }) {
            throw CocoaError(.fileWriteFileExists, userInfo: [NSFilePathErrorKey: root])
        }
        try fm.createDirectory(atPath: root, withIntermediateDirectories: true)
        do {
            try await fill(root: root, request)
            return root
        } catch {
            // Everything this import made goes; downloads kept in the
            // folder stay for another try.
            for name in (try? fm.contentsOfDirectory(atPath: root)) ?? [] where name != importDownloadFolderName {
                try? fm.removeItem(atPath: root + "/" + name)
            }
            if ((try? fm.contentsOfDirectory(atPath: root)) ?? []).isEmpty { try? fm.removeItem(atPath: root) }
            throw error
        }
    }

    /// A folder name under Imported not taken yet, for a default.
    static func suggestedFolder(name: String) -> String { uniqueFolder(name: name) }

    private func fill(root: String, _ request: Request) async throws {
        let fm = FileManager.default
        let base = request.base.path
        var parts: [ModelImportManifest.Part] = []
        // Polled from the C routines' own thread: no main-actor state here.
        let flag = cancelFlag
        let isCancelled: @Sendable () -> Bool = { flag.isSet }

        // FL2VA: everything but the DiT weights and the text encoder from the
        // base model (clones).
        step = String(localized: "基のモデルから共通の部品を複製しています")
        let fl2va = root + "/FL2VA"
        try fm.createDirectory(atPath: fl2va + "/transformer", withIntermediateDirectories: true)
        for item in ["tokenizer", "processor", "video_vae", "audio_vae", "model_index.json"] {
            try Self.cloneIfPresent(base + "/FL2VA/" + item, to: fl2va + "/" + item)
        }
        try fm.copyItem(atPath: base + "/FL2VA/transformer/config.json",
                        toPath: fl2va + "/transformer/config.json")
        try checkCancelled()

        step = String(localized: "DiTを置いています")
        try fm.copyItem(atPath: request.dit.path, toPath: fl2va + "/transformer/model.gguf")
        parts.append(.init(role: "dit", file: "FL2VA/transformer/model.gguf",
                           source: request.dit.fileName, type: request.dit.info?.mainType ?? "",
                           bytes: request.dit.info?.fileBytes ?? 0))
        try checkCancelled()

        // Text encoder: the GGUF trimmed to 50 layers, or the base model's.
        let textEncoder = fl2va + "/text_encoder"
        if let te = request.textEncoder, let info = te.info {
            try fm.createDirectory(atPath: textEncoder, withIntermediateDirectories: true)
            if info.textLayers > 50 {
                step = String(localized: "文章の部分を50層に切り詰めています")
                fraction = 0
                try await H3GGUFTools.trimTextEncoder(
                    source: te.path, output: textEncoder + "/model.gguf", layers: 50,
                    progress: { value in Task { @MainActor in self.fraction = value } },
                    isCancelled: isCancelled)
                fraction = nil
            } else {
                step = String(localized: "文章の部分を置いています")
                try fm.copyItem(atPath: te.path, toPath: textEncoder + "/model.gguf")
            }
            parts.append(.init(role: "textEncoder", file: "FL2VA/text_encoder/model.gguf",
                               source: te.fileName, type: info.mainType,
                               bytes: (try? fm.attributesOfItem(atPath: textEncoder + "/model.gguf")[.size]
                                       as? UInt64) ?? 0))
            switch request.vision {
            case .file(let vision):
                step = String(localized: "画像の部分を置いています")
                try fm.copyItem(atPath: vision.path, toPath: textEncoder + "/mmproj.gguf")
                parts.append(.init(role: "vision", file: "FL2VA/text_encoder/mmproj.gguf",
                                   source: vision.fileName, type: vision.info?.mainType ?? "",
                                   bytes: vision.info?.fileBytes ?? 0))
            case .extractFromBase:
                step = String(localized: "基のモデルから画像の部分を取り出しています")
                fraction = 0
                try await H3GGUFTools.extractTensors(
                    componentDirectory: base + "/FL2VA/text_encoder", prefix: "model.visual.",
                    output: textEncoder + "/vision.safetensors",
                    progress: { value in Task { @MainActor in self.fraction = value } },
                    isCancelled: isCancelled)
                fraction = nil
                parts.append(.init(role: "vision", file: "FL2VA/text_encoder/vision.safetensors",
                                   source: request.base.name, type: "BF16",
                                   bytes: (try? fm.attributesOfItem(atPath: textEncoder + "/vision.safetensors")[.size]
                                           as? UInt64) ?? 0))
            case .cloneFromBase(let source):
                step = String(localized: "画像の部分を置いています")
                let name = (source as NSString).lastPathComponent
                try fm.copyItem(atPath: source, toPath: textEncoder + "/" + name)
                parts.append(.init(role: "vision", file: "FL2VA/text_encoder/" + name,
                                   source: request.base.name,
                                   type: name.hasSuffix(".gguf") ? "GGUF" : "BF16",
                                   bytes: (try? fm.attributesOfItem(atPath: source)[.size] as? UInt64) ?? 0))
            case .inTextEncoder, .baseTextEncoder, .none:
                break
            }
        } else {
            step = String(localized: "基のモデルから共通の部品を複製しています")
            try fm.copyItem(atPath: base + "/FL2VA/text_encoder", toPath: textEncoder)
        }
        try checkCancelled()

        // Ref2VA: its own DiT, the rest shared with FL2VA (clones).
        if let ref = request.ref2vaDit {
            step = String(localized: "Ref2VAを置いています")
            let ref2va = root + "/Ref2VA"
            try fm.createDirectory(atPath: ref2va + "/transformer", withIntermediateDirectories: true)
            // The shared-parts download has only Ref2VA's two own files.
            func refSource(_ item: String) -> String {
                fm.fileExists(atPath: base + "/Ref2VA/" + item) ? base + "/Ref2VA/" + item : base + "/FL2VA/" + item
            }
            for item in ["tokenizer", "processor", "video_vae", "audio_vae", "model_index.json"] {
                try Self.cloneIfPresent(refSource(item), to: ref2va + "/" + item)
            }
            try fm.copyItem(atPath: refSource("transformer/config.json"),
                            toPath: ref2va + "/transformer/config.json")
            try fm.copyItem(atPath: ref.path, toPath: ref2va + "/transformer/model.gguf")
            try fm.copyItem(atPath: textEncoder, toPath: ref2va + "/text_encoder")
            parts.append(.init(role: "ref2va", file: "Ref2VA/transformer/model.gguf",
                               source: ref.fileName, type: ref.info?.mainType ?? "",
                               bytes: ref.info?.fileBytes ?? 0))
        }

        try ModelImportManifest(importedAt: Date(), baseModel: request.base.name, parts: parts)
            .save(modelPath: root)
    }

    private func checkCancelled() throws {
        if cancelled || Task.isCancelled { throw CancellationError() }
    }

    private static func cloneIfPresent(_ source: String, to destination: String) throws {
        guard FileManager.default.fileExists(atPath: source) else { return }
        try FileManager.default.copyItem(atPath: source, toPath: destination)
    }

    private static func uniqueFolder(name: String) -> String {
        let cleaned = name.replacingOccurrences(of: "/", with: "-")
            .trimmingCharacters(in: .whitespaces)
        let stem = importedModelsDirectory + "/" + (cleaned.isEmpty ? "MiniMax-H3 GGUF" : cleaned)
        var candidate = stem
        var number = 2
        while FileManager.default.fileExists(atPath: candidate) {
            candidate = stem + " \(number)"
            number += 1
        }
        return candidate
    }
}

/// The size of a base model's text encoder files (what an import without
/// a text-encoder GGUF clones from it).
func baseTextEncoderBytes(_ model: H3ModelEntry) -> Int64 {
    let directory = model.path + "/FL2VA/text_encoder"
    return ((try? FileManager.default.contentsOfDirectory(atPath: directory)) ?? []).reduce(0) { total, name in
        total + (((try? FileManager.default.attributesOfItem(atPath: directory + "/" + name))?[.size] as? NSNumber)?.int64Value ?? 0)
    }
}

/// The base model must have every part an import leaves out.
func isUsableImportBase(_ model: H3ModelEntry) -> Bool {
    let fm = FileManager.default
    let fl2va = model.path + "/FL2VA/"
    return ["tokenizer", "video_vae", "audio_vae", "transformer/config.json", "text_encoder"]
        .allSatisfy { fm.fileExists(atPath: fl2va + $0) }
}

/// Where the base model keeps its vision tower: a file of its own that can
/// be cloned (an imported model's mmproj.gguf or vision.safetensors), or
/// inside a released text encoder's shards (extracted), or nowhere.
func baseVisionSource(_ model: H3ModelEntry) -> ImportVisionSource {
    let directory = model.path + "/FL2VA/text_encoder"
    let fm = FileManager.default
    for name in ["mmproj.gguf", "vision.safetensors"] where fm.fileExists(atPath: directory + "/" + name) {
        return .cloneFromBase(directory + "/" + name)
    }
    guard let names = try? fm.contentsOfDirectory(atPath: directory) else { return .none }
    if ModelImportManifest.load(modelPath: model.path) == nil,
       names.contains(where: { $0.hasSuffix(".safetensors") }) {
        return .extractFromBase
    }
    return .none
}

/// Released models first: they are the ones every part can come from.
func preferredImportBases(_ models: [H3ModelEntry]) -> [H3ModelEntry] {
    let usable = models.filter(isUsableImportBase)
    return usable.filter { ModelImportManifest.load(modelPath: $0.path) == nil } +
        usable.filter { ModelImportManifest.load(modelPath: $0.path) != nil }
}

/// A folder the user picked as the base without registering it (removing
/// a model from the app leaves its folder, which still has the shared
/// parts). Nothing is searched for: only registered models and folders
/// picked this way are offered. The id is derived from the path so the same
/// folder keeps one id.
func unregisteredImportBase(path: String, registered: [H3ModelEntry]) -> H3ModelEntry? {
    let fm = FileManager.default
    var folder = path
    // A Hugging Face snapshot nests the model one level down
    // (MiniMax-H3/MiniMax-H3/FL2VA).
    if !fm.fileExists(atPath: folder + "/FL2VA"),
       let children = try? fm.contentsOfDirectory(atPath: folder),
       let nested = children.first(where: { fm.fileExists(atPath: folder + "/" + $0 + "/FL2VA") }) {
        folder += "/" + nested
    }
    if registered.contains(where: { $0.path == folder }) { return nil }
    let probe = H3ModelEntry(name: "", path: folder)
    guard isUsableImportBase(probe) else { return nil }
    var hasher = Hasher()
    hasher.combine(folder)
    let seed = UInt64(bitPattern: Int64(hasher.finalize()))
    let id = UUID(uuid: (UInt8(seed & 0xff), UInt8((seed >> 8) & 0xff), UInt8((seed >> 16) & 0xff),
                         UInt8((seed >> 24) & 0xff), UInt8((seed >> 32) & 0xff), UInt8((seed >> 40) & 0xff),
                         0x40, UInt8((seed >> 48) & 0xff), 0x80, UInt8((seed >> 56) & 0xff),
                         0, 0, 0, 0, 0, 1))
    let label = String(localized: "\((folder as NSString).lastPathComponent)（選んだフォルダ）")
    return H3ModelEntry(id: id, name: label, path: folder)
}

/// The GGUF files in a folder the user picked (and its subfolders, a few
/// levels down) that are MiniMax H3 parts, and the names of the ones that
/// are not (unreadable, other models, split shards).
func importableGGUFs(in folder: String) -> (picks: [ImportPick], skipped: [String]) {
    // Walked by hand to follow linked folders too (a model folder of
    // links, as made for trying a layout), a few levels down; each file
    // once, however many links lead to it.
    let fm = FileManager.default
    var paths: [String] = []
    var seenFiles = Set<String>(), seenFolders = Set<String>()
    func walk(_ dir: String, depth: Int) {
        let resolved = (dir as NSString).resolvingSymlinksInPath
        guard depth <= 3, seenFolders.insert(resolved).inserted,
              let names = try? fm.contentsOfDirectory(atPath: dir) else { return }
        for name in names.sorted() where !name.hasPrefix(".") {
            let path = dir + "/" + name
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: path, isDirectory: &isDir) else { continue }
            if isDir.boolValue {
                if !name.hasSuffix(".app") { walk(path, depth: depth + 1) }
            } else if name.lowercased().hasSuffix(".gguf"),
                      seenFiles.insert((path as NSString).resolvingSymlinksInPath).inserted {
                paths.append(path)
            }
        }
    }
    walk(folder, depth: 0)
    var picks: [ImportPick] = [], skipped: [String] = []
    for path in paths {
        let pick = ImportPick(path: path)
        if pick.info.map({ $0.role != .unknown }) == true {
            picks.append(pick)
        } else {
            skipped.append(String(path.dropFirst(min(path.count, folder.count + 1))))
        }
    }
    return (picks, skipped)
}

/// Free space on the volume a path is on (the import writes there).
func availableBytes(at path: String) -> Int64? {
    let url = URL(fileURLWithPath: FileManager.default.fileExists(atPath: path) ? path : NSHomeDirectory())
    return (try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
        .volumeAvailableCapacityForImportantUsage ?? nil
}

// MARK: - What the picked files add up to

/// Sorts picked files into a model's parts, says what is missing, and
/// points out files that cover the same part (one of them is used - the
/// one chosen in `chosen`, else the first).
struct ImportPlan {
    enum Status { case ok, info, warning, missing }

    /// The parts a file can fill.
    enum Part: String, CaseIterable {
        case dit, ref2va, textEncoder, vision
    }

    struct Item: Identifiable {
        let id: String
        let title: String
        let status: Status
        let detail: String
        /// Where to get the part, when it is missing.
        let hint: String?
        var hintURL: URL? = nil
        /// What the wizard can download for it.
        var download: ImportDownloadKind? = nil
        /// Files that could fill this part; more than one is a duplicate the
        /// wizard asks to resolve by choosing one.
        var candidates: [ImportPick] = []
        var part: Part?
        var chosenPath: String?
    }

    var dit: ImportPick?
    var ref2vaDit: ImportPick?
    var textEncoder: ImportPick?
    var vision: ImportVisionSource = .none
    /// Problems that stop the import.
    var problems: [String] = []
    /// Points worth knowing that do not stop it (unused files, overlaps).
    var notes: [String] = []
    /// Picked files the import will not use, with the reason (the file list
    /// marks them).
    var unused: [String: String] = [:]
    var items: [Item] = []
    /// New data the import writes (trimmed text encoder, extracted vision).
    var bytesWritten: UInt64 = 0

    var canImport: Bool { dit != nil && problems.isEmpty }
    /// Parts with no file, required ones first: what the file page lists
    /// with where to get them.
    var absent: [Item] {
        items.filter { $0.hint != nil && $0.status == .missing } +
            items.filter { $0.hint != nil && $0.status != .missing }
    }

    static func part(of file: ImportPick, ref2vaPaths: Set<String>) -> Part? {
        switch file.info?.role {
        case .dit?: return ref2vaPaths.contains(file.path) ? .ref2va : .dit
        case .textEncoder?: return .textEncoder
        case .vision?: return .vision
        default: return nil
        }
    }

    /// How a file is named in the review: its file name, with the folder
    /// it is in when another picked file has the same name.
    static func labels(for files: [ImportPick]) -> [String: String] {
        var counts: [String: Int] = [:]
        for file in files { counts[file.fileName, default: 0] += 1 }
        var labels: [String: String] = [:]
        for file in files {
            let folder = ((file.path as NSString).deletingLastPathComponent as NSString).lastPathComponent
            labels[file.path] = (counts[file.fileName] ?? 0) > 1 ? folder + "/" + file.fileName : file.fileName
        }
        return labels
    }

    init(files: [ImportPick], ref2vaPaths: Set<String>, base: H3ModelEntry?,
         chosen: [Part: String] = [:], availableBytes: Int64? = nil) {
        defer {
            // Clones cost nothing; what the import writes must fit, with
            // room to spare for the system.
            let margin: Int64 = 2 << 30
            if let availableBytes, Int64(bytesWritten) + margin > availableBytes {
                let need = ByteCountFormatter.string(fromByteCount: Int64(bytesWritten), countStyle: .file)
                let have = ByteCountFormatter.string(fromByteCount: availableBytes, countStyle: .file)
                problems.append(String(localized: "空き容量が足りません：新しく約\(need)を書き出しますが、空きは\(have)です（余裕として2GBを残します）。"))
            }
        }
        let labels = Self.labels(for: files)
        func label(_ file: ImportPick) -> String { labels[file.path] ?? file.fileName }
        var candidates: [Part: [ImportPick]] = [:]
        for file in files {
            guard file.info != nil else {
                notes.append(String(localized: "\(label(file)) はGGUFとして読めないので使いません"))
                unused[file.path] = String(localized: "GGUFとして読めないので使いません")
                continue
            }
            guard let part = Self.part(of: file, ref2vaPaths: ref2vaPaths) else {
                notes.append(String(localized: "\(label(file)) はMiniMax H3の部品として認識できないので使いません"))
                unused[file.path] = String(localized: "MiniMax H3の部品として認識できないので使いません")
                continue
            }
            candidates[part, default: []].append(file)
        }
        func pick(_ part: Part) -> ImportPick? {
            let list = candidates[part] ?? []
            if let file = list.first(where: { $0.path == chosen[part] }) { return file }
            // Of text encoders in the same format, one already cut to the
            // 50 layers the engine reads needs nothing written.
            if part == .textEncoder, let first = list.first,
               let ready = list.first(where: {
                   $0.info?.textLayers == 50 && $0.info?.mainType == first.info?.mainType }) {
                return ready
            }
            return list.first
        }
        dit = pick(.dit)
        ref2vaDit = pick(.ref2va)
        textEncoder = pick(.textEncoder)
        let chosenVision = pick(.vision)
        // The files that lost to another for the same part.
        for part in Part.allCases {
            guard let used = pick(part) else { continue }
            for file in candidates[part] ?? [] where file.path != used.path {
                unused[file.path] = String(localized: "同じ部品の \(label(used)) を使うので使いません")
            }
        }

        // The same file picked twice under different names (same size, type
        // and tensor count) is worth saying out loud.
        for list in candidates.values where list.count > 1 {
            for (i, a) in list.enumerated() {
                for b in list[(i + 1)...] {
                    let resolvedA = (a.path as NSString).resolvingSymlinksInPath
                    let resolvedB = (b.path as NSString).resolvingSymlinksInPath
                    if resolvedA == resolvedB {
                        notes.append(String(localized: "\(label(a)) と \(label(b)) は同じファイルです（一方がもう一方へのリンク）"))
                    } else if a.info?.fileBytes == b.info?.fileBytes &&
                                a.info?.mainType == b.info?.mainType {
                        notes.append(String(localized: "\(label(a)) と \(label(b)) は同じ内容の可能性があります（大きさと形式が同じ）"))
                    }
                }
            }
        }

        if let info = textEncoder?.info, info.textLayers < 50 {
            problems.append(String(localized: "\(label(textEncoder!))：文章の部分の層が足りません（50層が必要です）"))
        }
        if base == nil {
            problems.append(String(localized: "複製元にできるモデルがありません。公式のモデルか取り込み済みのモデルのフォルダを登録するか、共通部品をダウンロードしてください。"))
        }

        func size(_ bytes: UInt64) -> String {
            ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
        }
        func describe(_ file: ImportPick) -> String {
            "\(label(file))\(itemSeparator)\(file.info?.mainType ?? "")\(itemSeparator)\(size(file.info?.fileBytes ?? 0))"
        }
        /// The detail line, prefixed with the duplicate notice when several
        /// files compete for the part.
        func duplicateText(_ part: Part) -> String? {
            let list = candidates[part] ?? []
            guard list.count > 1 else { return nil }
            return String(localized: "同じ部品のファイルが\(list.count)つあります（\(list.map(label).joined(separator: "、"))）。使うものを1つ選んでください。")
        }
        let q2kWarning = String(localized: "Q2_KのDiTは画質が大きく崩れます（人物が溶ける・背景が格子状になる・音声がうなる）。Q3_K_M以上をおすすめします。")
        func item(_ id: String, _ title: String, _ status: Status, _ detail: String,
                  hint: String? = nil, url: String? = nil, download: ImportDownloadKind? = nil,
                  part: Part? = nil, chosen file: ImportPick? = nil) -> Item {
            var status = status, detail = detail
            if let part, let duplicate = duplicateText(part) {
                status = status == .missing ? .missing : .warning
                detail = duplicate + "\n" + detail
            }
            return Item(id: id, title: title, status: status, detail: detail, hint: hint,
                        hintURL: url.flatMap(URL.init(string:)), download: download,
                        candidates: part.map { candidates[$0] ?? [] } ?? [], part: part,
                        chosenPath: file?.path)
        }

        // DiT (FL2VA)
        if let dit, dit.info != nil {
            let q2k = isPlainQ2KDiT(mainType: dit.info?.mainType ?? "", fileName: dit.fileName)
            items.append(item("dit", String(localized: "DiT（FL2VA）"), q2k ? .warning : .ok,
                              describe(dit) + (q2k ? "\n" + q2kWarning : ""),
                              part: .dit, chosen: dit))
        } else {
            items.append(item("dit", String(localized: "DiT（FL2VA）"), .missing,
                              String(localized: "必須です。文章からの生成と最初の画像からの生成に使います。"),
                              hint: String(localized: "例：Hugging Face の unsloth/MiniMax-H3-GGUF にある minimax_h3_fl2va_pruned-Q8_0.gguf（動作を確認済み）"),
                              url: "https://huggingface.co/unsloth/MiniMax-H3-GGUF", download: .dit))
        }

        // Text encoder
        if let te = textEncoder, let info = te.info {
            let trimmed = info.textLayers > 50
            let written = trimmed ? info.fileBytes / UInt64(info.textLayers) * 50 : 0
            bytesWritten += written
            let detail = describe(te) + itemSeparator + (trimmed
                ? String(localized: "\(info.textLayers)層から50層に切り詰めて書き出します（約\(size(written))）")
                : String(localized: "\(info.textLayers)層をそのまま使います"))
            items.append(item("text", String(localized: "文章の部分（テキストエンコーダー）"), .ok, detail,
                              part: .textEncoder, chosen: te))
        } else if let base, !baseHasTextEncoderWeights(base) {
            problems.append(String(localized: "文章の部分のGGUFが必要です（「\(base.name)」には文章の部分の重みがありません）"))
            items.append(item("text", String(localized: "文章の部分（テキストエンコーダー）"), .missing,
                              String(localized: "必須です。複製元に文章の部分の重みがないので、GGUFで用意してください。"),
                              hint: String(localized: "例：Hugging Face の Qwen/Qwen3-VL-32B-Instruct-GGUF にある Qwen3VL-32B-Instruct-Q8_0.gguf（動作を確認済み。H3の文章の部分はQwen3-VL-32B-Instructと同じ重みです）"),
                              url: "https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-GGUF", download: .textEncoder))
        } else if base == nil {
            items.append(item("text", String(localized: "文章の部分（テキストエンコーダー）"), .warning,
                              String(localized: "複製元が決まっていません。公式のモデルを複製元にするならなくても取り込めますが、共通部品だけをダウンロードして使う場合は必要です。"),
                              hint: String(localized: "例：Hugging Face の Qwen/Qwen3-VL-32B-Instruct-GGUF にある Qwen3VL-32B-Instruct-Q8_0.gguf（動作を確認済み。H3の文章の部分はQwen3-VL-32B-Instructと同じ重みです）"),
                              url: "https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-GGUF", download: .textEncoder))
        } else {
            let baseText = size(UInt64(max(base.map(baseTextEncoderBytes) ?? 0, 0)))
            items.append(item("text", String(localized: "文章の部分（テキストエンコーダー）"), .info,
                              String(localized: "GGUFがないので、複製元のモデルのもの（約\(baseText)）を使います。同じディスクなら容量は増えませんが、複製元を消すとその分の容量が残ります。"),
                              hint: String(localized: "例：Hugging Face の Qwen/Qwen3-VL-32B-Instruct-GGUF にある Qwen3VL-32B-Instruct-Q8_0.gguf（動作を確認済み。H3の文章の部分はQwen3-VL-32B-Instructと同じ重みです）"),
                              url: "https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-GGUF", download: .textEncoder))
        }

        // Vision
        let visionTitle = String(localized: "画像の部分")
        if let te = textEncoder, let info = te.info {
            if info.hasVision {
                vision = .inTextEncoder
                if let file = chosenVision {
                    notes.append(String(localized: "画像の部分は \(label(te)) にも入っているので、\(label(file)) は使いません"))
                    for other in candidates[.vision] ?? [] {
                        unused[other.path] = String(localized: "画像の部分は \(label(te)) にも入っているので使いません")
                    }
                }
                items.append(item("vision", visionTitle, .ok,
                                  String(localized: "文章の部分のファイル（\(label(te))）に含まれています")))
            } else if let file = chosenVision {
                vision = .file(file)
                items.append(item("vision", visionTitle, .ok, describe(file), part: .vision, chosen: file))
            } else if let base, case .cloneFromBase(let source) = baseVisionSource(base) {
                vision = .cloneFromBase(source)
                items.append(item("vision", visionTitle, .info,
                                  String(localized: "GGUFがないので、複製元のモデルのものを複製します（容量は増えません）")))
            } else if let base, baseVisionSource(base) == .extractFromBase {
                vision = .extractFromBase
                bytesWritten += 1_190_572_912
                items.append(item("vision", visionTitle, .info,
                                  String(localized: "GGUFがないので、複製元のモデルから取り出します（約1.1GB）"),
                                  hint: String(localized: "GGUFで用意するなら：Qwen/Qwen3-VL-32B-Instruct-GGUF にある mmproj-Qwen3VL-32B-Instruct-F16.gguf"),
                                  url: "https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-GGUF", download: .vision))
            } else {
                vision = .none
                items.append(item("vision", visionTitle, .warning,
                                  String(localized: "ありません。最初の画像や参照画像を使う生成はできません。"),
                                  hint: String(localized: "例：Qwen/Qwen3-VL-32B-Instruct-GGUF にある mmproj-Qwen3VL-32B-Instruct-F16.gguf"),
                                  url: "https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-GGUF", download: .vision))
            }
        } else if base.map(baseHasTextEncoderWeights) != true {
            // No text encoder yet: which vision tower is used depends on it.
            vision = .none
            items.append(item("vision", visionTitle, .info,
                              String(localized: "文章の部分が決まると決まります。画像の部分を含まない文章の部分のGGUFなら、mmprojのGGUFも必要です。"),
                              hint: chosenVision == nil
                                  ? String(localized: "例：Qwen/Qwen3-VL-32B-Instruct-GGUF にある mmproj-Qwen3VL-32B-Instruct-F16.gguf") : nil,
                              url: chosenVision == nil ? "https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-GGUF" : nil,
                              download: chosenVision == nil ? .vision : nil,
                              part: .vision, chosen: chosenVision))
        } else {
            vision = .baseTextEncoder
            if let file = chosenVision {
                notes.append(String(localized: "文章の部分を複製元から使うので、\(label(file)) は使いません"))
                for other in candidates[.vision] ?? [] {
                    unused[other.path] = String(localized: "文章の部分を複製元から使うので使いません（文章の部分のGGUFを追加すると使います）")
                }
            }
            items.append(item("vision", visionTitle, .ok,
                              String(localized: "複製元のモデルの文章の部分に含まれています")))
        }

        // Ref2VA
        if let ref = ref2vaDit, let info = ref.info {
            let q2k = isPlainQ2KDiT(mainType: info.mainType, fileName: ref.fileName)
            items.append(item("ref2va", String(localized: "DiT（Ref2VA）"),
                              info.curveAdaLN && !q2k ? .ok : .warning,
                              describe(ref) + (q2k ? "\n" + q2kWarning : ""), part: .ref2va, chosen: ref))
        } else {
            items.append(item("ref2va", String(localized: "DiT（Ref2VA）"), .info,
                              String(localized: "ありません。参照画像・動画・音声を使う生成はできません（文章から・最初の画像からは使えます）。"),
                              hint: String(localized: "例：unsloth/MiniMax-H3-GGUF にある minimax_h3_ref2va_pruned-Q8_0.gguf（DiTのファイルは、名前に ref2va を含むものをRef2VAとみなします）"),
                              url: "https://huggingface.co/unsloth/MiniMax-H3-GGUF", download: .ref2va))
        }

        // Shared parts
        if let base {
            items.append(item("shared", String(localized: "トークナイザー・VAE・設定"), .ok,
                              String(localized: "「\(base.name)」から複製します（容量は増えません）")))
        } else {
            items.append(item("shared", String(localized: "トークナイザー・VAE・設定"), .missing,
                              String(localized: "GGUFには入っていません（トークナイザーとVAEはGGUFの形では配布されていません）。モデル管理に登録したモデルか、ダウンロードした共通部品から複製します。"),
                              hint: String(localized: "公式のモデルか取り込み済みのモデルがあれば、そのフォルダを登録してください。ない場合は、公式のモデルから共通部品だけ（約11GB）をダウンロードできます（GGUFの形のものはありません）。"),
                              url: "https://huggingface.co/MiniMaxAI/MiniMax-H3", download: .shared))
        }
    }
}

// MARK: - The wizard

struct ModelImportView: View {
    @ObservedObject var library: ModelLibrary
    var onImported: (_ path: String, _ use: Bool) -> Void
    var onCancelled: () -> Void

    @StateObject private var importer = ModelImporter()
    @Environment(\.colorScheme) private var colorScheme
    @State private var page = 0
    @State private var files: [ImportPick] = []
    @State private var ref2vaPaths: Set<String> = []
    @State private var chosen: [ImportPlan.Part: String] = [:]
    @State private var name = "MiniMax-H3 GGUF"
    @State private var baseID: UUID?
    @State private var importedPath: String?
    @State private var useImported = true
    /// Base folders picked with Register Folder that are not models of
    /// their own (no DiT weights), so were not registered.
    @State private var extraBases: [H3ModelEntry] = []
    @State private var baseFolderError: String?
    @State private var baseFolderMessage: String?
    @State private var downloading: ImportDownloadKind?
    /// Downloaded GGUF files the finished import used (the import keeps its
    /// own copies, so these can go).
    @State private var downloadedSources: [String] = []
    /// What the last Add Folder found.
    @State private var folderMessage: String?

    private var palette: H3Palette { H3Palette(colorScheme) }
    private var bases: [H3ModelEntry] {
        preferredImportBases(library.models) + preferredImportBases(extraBases) +
            (downloadedImportBase().map { [$0] } ?? [])
    }
    private var base: H3ModelEntry? { bases.first { $0.id == baseID } ?? bases.first }
    private var plan: ImportPlan {
        ImportPlan(files: files, ref2vaPaths: ref2vaPaths, base: base, chosen: chosen,
                   availableBytes: availableBytes(at: importedModelsDirectory))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            header
            Divider()
            switch page {
            case 0: filesPage
            case 1: reviewPage
            default: importPage
            }
            Spacer(minLength: 0)
            footer
        }
        .padding(H3Spacing.lg)
        .frame(width: 640, height: 640)
        .background(palette.canvas)
        .sheet(item: $downloading) { kind in
            ImportDownloadView(kind: kind, onFinished: { path in
                downloading = nil
                downloaded(kind, path: path)
            }, onCancelled: { downloading = nil })
        }
    }

    /// A finished download: a GGUF joins the files, the shared parts become
    /// the base.
    private func downloaded(_ kind: ImportDownloadKind, path: String) {
        if kind == .shared {
            if path == importBaseDirectory {
                baseID = downloadedImportBase()?.id
            } else if let entry = unregisteredImportBase(path: path, registered: library.models + extraBases) {
                extraBases.append(entry)
                baseID = entry.id
            }
            return
        }
        guard !files.contains(where: { $0.path == path }) else { return }
        files.append(ImportPick(path: path))
        if kind == .ref2va { ref2vaPaths.insert(path) }
    }

    /// Registers a model folder (official or imported) in Model Management
    /// and makes it the base. A folder with the shared parts but no DiT is
    /// only used as the base.
    private func registerBaseFolder() {
        guard let path = chooseDirectory() else { return }
        baseFolderError = nil
        baseFolderMessage = nil
        if let entry = (library.models + extraBases).first(where: {
            $0.path == path || $0.path.hasPrefix(path + "/") }), isUsableImportBase(entry) {
            baseID = entry.id
            return
        }
        guard let entry = unregisteredImportBase(path: path, registered: library.models) else {
            baseFolderError = String(localized: "このフォルダにはFL2VAのトークナイザー・VAE・設定がそろっていません。")
            return
        }
        let transformer = (try? FileManager.default.contentsOfDirectory(atPath: entry.path + "/FL2VA/transformer")) ?? []
        if transformer.contains(where: { $0.hasSuffix(".safetensors") || $0.hasSuffix(".gguf") }) {
            library.addModel(path: entry.path)
            baseID = library.models.first { $0.path == entry.path }?.id
            baseFolderMessage = String(localized: "「\((entry.path as NSString).lastPathComponent)」をモデル管理に登録しました。")
        } else {
            extraBases.append(entry)
            baseID = entry.id
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("GGUFを取り込む").font(.system(size: 16, weight: .semibold))
            HStack(spacing: 6) {
                ForEach(Array([String(localized: "ファイルを選ぶ"), String(localized: "内容の確認"),
                               String(localized: "取り込み")].enumerated()), id: \.offset) { index, title in
                    Text("\(index + 1). \(title)")
                        .font(.caption.weight(index == page ? .semibold : .regular))
                        .foregroundStyle(index == page ? palette.accent : palette.textSecondary)
                    if index < 2 {
                        Image(systemName: "chevron.right").font(.caption2)
                            .foregroundStyle(palette.textSecondary)
                    }
                }
            }
        }
    }

    // Page 1: pick files; each is identified from its tensor table.
    private var filesPage: some View {
        VStack(alignment: .leading, spacing: H3Spacing.sm) {
            Text("量子化されたモデル（GGUF）のファイルをまとめて選ぶか、GGUFを置いたフォルダを選んでください。1つ1つの中身（DiT・文章の部分・画像の部分）を調べ、そろっている部品と足りない部品を示します。GGUFにない部品は、登録済みのモデルから複製します。")
                .font(.caption)
                .foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            if files.isEmpty {
                Text("まだファイルがありません。")
                    .font(.caption).foregroundStyle(palette.textSecondary)
                    .frame(maxWidth: .infinity, minHeight: 120)
            } else {
                let unused = plan.unused
                List {
                    ForEach(files) { file in fileRow(file, unused: unused[file.path]) }
                }
                .listStyle(.inset)
                .frame(minHeight: 200)
                if !unused.isEmpty {
                    Text("使われないファイルがあります。そのままでも取り込めますが、不要なら外してください。")
                        .font(.caption).foregroundStyle(.orange)
                }
            }
            HStack {
                Button("GGUFファイルを追加…") { addFiles() }
                Button("フォルダから追加…") { addFolder() }
                Spacer()
            }
            if let folderMessage {
                Text(folderMessage).font(.caption).foregroundStyle(palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            absentParts
        }
    }

    /// What the picked files leave out and where to get it.
    @ViewBuilder private var absentParts: some View {
        let plan = self.plan
        let absent = plan.absent.filter { $0.id != "shared" || $0.status == .missing }
        if !absent.isEmpty {
            Divider()
            Text(files.isEmpty ? String(localized: "必要な部品") : String(localized: "足りない部品"))
                .font(.callout.weight(.semibold))
            ScrollView {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(absent) { item in
                        HStack(alignment: .top, spacing: H3Spacing.sm) {
                            Image(systemName: icon(item.status)).foregroundStyle(color(item.status))
                                .frame(width: 18)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(item.status == .missing ? String(localized: "\(item.title)（必須）")
                                                             : String(localized: "\(item.title)（なくても取り込めます）"))
                                    .font(.caption.weight(.semibold))
                                hintText(item)
                            }
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    @ViewBuilder private func hintText(_ item: ImportPlan.Item) -> some View {
        if let hint = item.hint {
            Text(hint).font(.caption2).foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
        if let url = item.hintURL {
            Link(url.absoluteString.replacingOccurrences(of: "https://", with: ""), destination: url)
                .font(.caption2)
        }
        let needsBase = item.id == "shared" && item.status == .missing
        if item.download != nil || needsBase {
            HStack {
                if let kind = item.download {
                    Button(kind == .shared ? String(localized: "共通部品をダウンロード…")
                                           : String(localized: "ダウンロード…")) { downloading = kind }
                }
                if needsBase {
                    Button("フォルダを登録…") { registerBaseFolder() }
                }
            }
            .controlSize(.small)
            if needsBase {
                baseFolderStatus
            }
        }
    }

    @ViewBuilder private var baseFolderStatus: some View {
        if let baseFolderMessage {
            Text(baseFolderMessage).font(.caption).foregroundStyle(palette.textSecondary)
        }
        if let baseFolderError {
            Text(baseFolderError).font(.caption).foregroundStyle(palette.errorColor)
        }
    }

    private func fileRow(_ file: ImportPick, unused: String? = nil) -> some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                Text(ImportPlan.labels(for: files)[file.path] ?? file.fileName)
                    .font(.callout).lineLimit(1).truncationMode(.middle)
                    .help(file.path)
                Text(fileDescription(file)).font(.caption)
                    .foregroundStyle(file.info == nil || file.info?.role == .unknown
                                     ? palette.errorColor : palette.textSecondary)
                if let unused {
                    Label(String(localized: "使われません：\(unused)"),
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }
            Spacer()
            if file.info?.role == .dit {
                Picker("", selection: Binding(
                    get: { ref2vaPaths.contains(file.path) },
                    set: { isRef in
                        if isRef { ref2vaPaths.insert(file.path) } else { ref2vaPaths.remove(file.path) }
                    })) {
                    Text("FL2VA").tag(false)
                    Text("Ref2VA").tag(true)
                }
                .labelsHidden()
                .frame(width: 96)
            }
            Button {
                remove(file)
            } label: { Image(systemName: "xmark.circle") }
            .buttonStyle(.borderless)
            .help("このファイルを外す")
        }
        .padding(.vertical, 2)
    }

    private func fileDescription(_ file: ImportPick) -> String {
        guard let info = file.info else {
            return String(localized: "GGUFとして読めません")
        }
        let size = ByteCountFormatter.string(fromByteCount: Int64(info.fileBytes), countStyle: .file)
        var parts: [String]
        switch info.role {
        case .dit:
            parts = [String(localized: "DiT"), info.mainType, size,
                     info.curveAdaLN ? String(localized: "AdaLNは曲線形") : String(localized: "AdaLNは密な形")]
        case .textEncoder:
            parts = [String(localized: "文章の部分"), info.mainType, size,
                     String(localized: "\(info.textLayers)層")]
            parts.append(info.hasVision ? String(localized: "画像の部分を含む")
                                        : String(localized: "画像の部分は含まない"))
        case .vision:
            parts = [String(localized: "画像の部分（mmproj）"), info.mainType, size]
        case .unknown:
            return String(localized: "MiniMax H3の部品として認識できません")
        }
        return parts.joined(separator: summarySeparator)
    }

    private func addFiles() {
        for path in chooseFiles(allowedContentTypes: ggufTypes)
            where !files.contains(where: { $0.path == path }) {
            let file = ImportPick(path: path)
            files.append(file)
            if file.info?.role == .dit && file.looksLikeRef2VA { ref2vaPaths.insert(path) }
        }
    }

    /// Adds every MiniMax H3 part among the GGUF files in a folder.
    private func addFolder() {
        guard let folder = chooseDirectory() else { return }
        let found = importableGGUFs(in: folder)
        var added = 0
        for file in found.picks where !files.contains(where: { $0.path == file.path }) {
            files.append(file)
            if file.info?.role == .dit && file.looksLikeRef2VA { ref2vaPaths.insert(file.path) }
            added += 1
        }
        let name = (folder as NSString).lastPathComponent
        var message = found.picks.isEmpty && found.skipped.isEmpty
            ? String(localized: "「\(name)」にGGUFファイルはありませんでした。")
            : added == 1 ? String(localized: "「\(name)」から1個のファイルを追加しました。")
                         : String(localized: "「\(name)」から\(added)個のファイルを追加しました。")
        if !found.skipped.isEmpty {
            message += String(localized: "MiniMax H3の部品ではないので除いたもの：\(found.skipped.joined(separator: "、"))")
        }
        folderMessage = message
    }

    private func remove(_ file: ImportPick) {
        files.removeAll { $0.path == file.path }
        ref2vaPaths.remove(file.path)
        for (part, path) in chosen where path == file.path { chosen[part] = nil }
    }

    // Page 2: the picked files, what they add up to, what is missing or
    // doubled, and where to get the rest. Files can still be added here.
    private var reviewPage: some View {
        let plan = self.plan
        return ScrollView {
            VStack(alignment: .leading, spacing: H3Spacing.sm) {
                DisclosureGroup {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(files) { file in fileRow(file, unused: plan.unused[file.path]) }
                    }
                } label: {
                    Text("選んだファイル（\(files.count)）").font(.callout.weight(.semibold))
                }
                HStack {
                    Button("GGUFファイルを追加…") { addFiles() }
                    Button("フォルダから追加…") { addFolder() }
                    Spacer()
                }
                if let folderMessage {
                    Text(folderMessage).font(.caption).foregroundStyle(palette.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Divider()
                Text("部品").font(.callout.weight(.semibold))
                ForEach(plan.items) { item in reviewRow(item) }
                ForEach(plan.problems, id: \.self) { problem in
                    Label(problem, systemImage: "xmark.octagon.fill")
                        .font(.caption).foregroundStyle(palette.errorColor)
                }
                ForEach(plan.notes, id: \.self) { note in
                    Label(note, systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.orange)
                }
                Divider()
                HStack {
                    Text("部品の複製元").font(.caption)
                    Picker("", selection: Binding(get: { base?.id }, set: { baseID = $0 })) {
                        ForEach(bases) { model in Text(model.name).tag(Optional(model.id)) }
                    }
                    .labelsHidden()
                    .frame(maxWidth: 300)
                    Button("フォルダを登録…") { registerBaseFolder() }
                }
                if let base {
                    Text(base.path).font(.caption2).foregroundStyle(palette.textSecondary)
                        .lineLimit(1).truncationMode(.middle)
                }
                baseFolderStatus
                HStack {
                    Text("名前").font(.caption)
                    TextField("名前", text: $name).textFieldStyle(.roundedBorder).frame(maxWidth: 260)
                }
                Text(spaceText(plan)).font(.caption).foregroundStyle(palette.textSecondary)
            }
        }
    }

    private func reviewRow(_ item: ImportPlan.Item) -> some View {
        HStack(alignment: .top, spacing: H3Spacing.sm) {
            Image(systemName: icon(item.status))
                .foregroundStyle(color(item.status))
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 2) {
                Text(item.title).font(.body)
                Text(item.detail).font(.caption).foregroundStyle(palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                if let part = item.part, item.candidates.count > 1 {
                    Picker("使うファイル", selection: Binding(
                        get: { item.chosenPath ?? item.candidates.first?.path ?? "" },
                        set: { chosen[part] = $0 })) {
                        ForEach(item.candidates) { file in
                            Text("\(ImportPlan.labels(for: files)[file.path] ?? file.fileName)（\(file.info?.mainType ?? "")）")
                                .tag(file.path)
                        }
                    }
                    .pickerStyle(.radioGroup)
                    .font(.caption)
                }
                hintText(item)
            }
        }
    }

    private func icon(_ status: ImportPlan.Status) -> String {
        switch status {
        case .ok: return "checkmark.circle.fill"
        case .info: return "info.circle.fill"
        case .warning: return "exclamationmark.triangle.fill"
        case .missing: return "xmark.circle.fill"
        }
    }

    private func color(_ status: ImportPlan.Status) -> Color {
        switch status {
        case .ok: return .green
        case .info: return palette.accent
        case .warning: return .orange
        case .missing: return palette.errorColor
        }
    }

    private func spaceText(_ plan: ImportPlan) -> String {
        let free = (try? URL(fileURLWithPath: NSHomeDirectory())
            .resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            .volumeAvailableCapacityForImportantUsage) ?? nil
        let written = ByteCountFormatter.string(fromByteCount: Int64(plan.bytesWritten), countStyle: .file)
        let freeText = free.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) } ?? "?"
        if plan.bytesWritten == 0 {
            return String(localized: "新しく書き出すものはありません（すべて複製なので、同じディスクなら容量は増えません）。空き容量：\(freeText)")
        }
        return String(localized: "新しく書き出す量：約\(written)（ほかは複製なので、同じディスクなら容量は増えません）。空き容量：\(freeText)")
    }

    // Page 3: progress, then the result.
    private var importPage: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            if importer.isRunning {
                HStack {
                    if let fraction = importer.fraction {
                        ProgressView(value: fraction).frame(maxWidth: 240)
                    } else {
                        ProgressView().controlSize(.small)
                    }
                    Text(importer.step).font(.caption).foregroundStyle(palette.textSecondary)
                }
            } else if let path = importedPath {
                Label("取り込みました", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Text(path).font(.caption).foregroundStyle(palette.textSecondary)
                    .lineLimit(2).truncationMode(.middle)
                Toggle("このモデルに切り替える", isOn: $useImported).toggleStyle(.checkbox)
                if !downloadedSources.isEmpty {
                    HStack {
                        Text("ダウンロードした元のGGUF（\(downloadedSources.count)個）は、取り込んだモデルとは別に残っています。")
                            .font(.caption).foregroundStyle(palette.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                        Button("ゴミ箱に入れる") { trashDownloadedSources() }.controlSize(.small)
                    }
                }
                Text("M5の高速モードで使うときは、切り替えたあと詳細設定の「キャッシュを作成…」でint8キャッシュを作ってください。")
                    .font(.caption).foregroundStyle(palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let message = importer.errorMessage {
                Text(message).font(.caption).foregroundStyle(palette.errorColor)
            }
        }
    }

    private var footer: some View {
        HStack {
            if page == 1 && !importer.isRunning {
                Button("戻る") { page = 0 }
            }
            Spacer()
            if importer.isRunning {
                Button("中止") { importer.cancel() }
            } else if importedPath != nil {
                Button("完了") { onImported(importedPath!, useImported) }
                    .keyboardShortcut(.defaultAction)
            } else {
                Button("キャンセル") { onCancelled() }
                switch page {
                case 0:
                    Button("次へ") { page = 1 }
                        .keyboardShortcut(.defaultAction)
                        .disabled(files.isEmpty)
                case 1:
                    Button("取り込む") { startImport() }
                        .keyboardShortcut(.defaultAction)
                        .disabled(!plan.canImport)
                default:
                    Button("戻る") { page = 1; importer.errorMessage = nil }
                }
            }
        }
    }

    private func trashDownloadedSources() {
        for path in downloadedSources {
            try? FileManager.default.trashItem(at: URL(fileURLWithPath: path), resultingItemURL: nil)
        }
        downloadedSources = []
    }

    private func startImport() {
        let plan = self.plan
        guard let dit = plan.dit, let base else { return }
        page = 2
        importer.start(.init(name: name, dit: dit, ref2vaDit: plan.ref2vaDit,
                             textEncoder: plan.textEncoder, vision: plan.vision, base: base)) { path in
            importedPath = path
            downloadedSources = [plan.dit, plan.ref2vaDit, plan.textEncoder, {
                if case .file(let file) = plan.vision { return file } else { return nil }
            }()].compactMap { $0?.path }.filter { $0.hasPrefix(importDownloadsDirectory + "/") }
        }
    }
}
