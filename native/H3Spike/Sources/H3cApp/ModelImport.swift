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

    /// e.g. "DiT Q8_0・文章 Q8_0" for the model list.
    var summary: String {
        parts.filter { $0.role != "vision" }.map { part in
            let label = part.role == "textEncoder" ? String(localized: "文章") :
                part.role == "ref2va" ? "Ref2VA" : "DiT"
            return "\(label) \(part.type)"
        }.joined(separator: "・")
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
        let root = Self.uniqueFolder(name: request.name)
        try fm.createDirectory(atPath: root, withIntermediateDirectories: true)
        do {
            try await fill(root: root, request)
            return root
        } catch {
            try? fm.removeItem(atPath: root)
            throw error
        }
    }

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
            let refBase = fm.fileExists(atPath: base + "/Ref2VA") ? base + "/Ref2VA" : base + "/FL2VA"
            for item in ["tokenizer", "processor", "video_vae", "audio_vae", "model_index.json"] {
                try Self.cloneIfPresent(refBase + "/" + item, to: ref2va + "/" + item)
            }
            try fm.copyItem(atPath: refBase + "/transformer/config.json",
                            toPath: ref2va + "/transformer/config.json")
            try fm.copyItem(atPath: ref.path, toPath: ref2va + "/transformer/model.gguf")
            try fm.copyItem(atPath: textEncoder, toPath: ref2va + "/text_encoder")
            parts.append(.init(role: "ref2va", file: "Ref2VA/transformer/model.gguf",
                               source: ref.fileName, type: ref.info?.mainType ?? "",
                               bytes: ref.info?.fileBytes ?? 0))
        }

        let manifest = ModelImportManifest(importedAt: Date(), baseModel: request.base.name,
                                           parts: parts)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(manifest).write(
            to: URL(fileURLWithPath: root).appendingPathComponent(ModelImportManifest.fileName))
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
         chosen: [Part: String] = [:]) {
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
            return list.first { $0.path == chosen[part] } ?? list.first
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
            problems.append(String(localized: "複製元にできるモデル（FL2VAのトークナイザーとVAEを含むもの）が登録されていません。"))
        }

        func size(_ bytes: UInt64) -> String {
            ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
        }
        func describe(_ file: ImportPick) -> String {
            "\(label(file))・\(file.info?.mainType ?? "")・\(size(file.info?.fileBytes ?? 0))"
        }
        /// The detail line, prefixed with the duplicate notice when several
        /// files compete for the part.
        func duplicateText(_ part: Part) -> String? {
            let list = candidates[part] ?? []
            guard list.count > 1 else { return nil }
            return String(localized: "同じ部品のファイルが\(list.count)つあります（\(list.map(label).joined(separator: "、"))）。使うものを1つ選んでください。")
        }
        func item(_ id: String, _ title: String, _ status: Status, _ detail: String,
                  hint: String? = nil, part: Part? = nil, chosen file: ImportPick? = nil) -> Item {
            var status = status, detail = detail
            if let part, let duplicate = duplicateText(part) {
                status = status == .missing ? .missing : .warning
                detail = duplicate + "\n" + detail
            }
            return Item(id: id, title: title, status: status, detail: detail, hint: hint,
                        candidates: part.map { candidates[$0] ?? [] } ?? [], part: part,
                        chosenPath: file?.path)
        }

        // DiT (FL2VA)
        if let dit, let info = dit.info {
            var detail = describe(dit)
            if !info.curveAdaLN {
                detail += String(localized: "（AdaLNが密な形。この形式での動作はまだ確認していません）")
            }
            items.append(item("dit", String(localized: "DiT（FL2VA）"),
                              info.curveAdaLN ? .ok : .warning, detail, part: .dit, chosen: dit))
        } else {
            items.append(item("dit", String(localized: "DiT（FL2VA）"), .missing,
                              String(localized: "必須です。文章からの生成と最初の画像からの生成に使います。"),
                              hint: String(localized: "例：Hugging Face の unsloth/MiniMax-H3-GGUF にある minimax_h3_fl2va_pruned-Q8_0.gguf（動作を確認済み）")))
        }

        // Text encoder
        if let te = textEncoder, let info = te.info {
            let trimmed = info.textLayers > 50
            let written = trimmed ? info.fileBytes / UInt64(info.textLayers) * 50 : 0
            bytesWritten += written
            let detail = describe(te) + "・" + (trimmed
                ? String(localized: "\(info.textLayers)層から50層に切り詰めて書き出します（約\(size(written))）")
                : String(localized: "\(info.textLayers)層をそのまま使います"))
            items.append(item("text", String(localized: "文章の部分（テキストエンコーダー）"), .ok, detail,
                              part: .textEncoder, chosen: te))
        } else {
            items.append(item("text", String(localized: "文章の部分（テキストエンコーダー）"), .info,
                              String(localized: "GGUFがないので、複製元のモデルのもの（約62GB）を使います。同じディスクなら容量は増えませんが、複製元を消すとその分の容量が残ります。"),
                              hint: String(localized: "例：Hugging Face の Qwen/Qwen3-VL-32B-Instruct-GGUF にある Qwen3VL-32B-Instruct-Q8_0.gguf（動作を確認済み。H3の文章の部分はQwen3-VL-32B-Instructと同じ重みです）")))
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
                                  hint: String(localized: "GGUFで用意するなら：Qwen/Qwen3-VL-32B-Instruct-GGUF にある mmproj-Qwen3VL-32B-Instruct-F16.gguf")))
            } else {
                vision = .none
                items.append(item("vision", visionTitle, .warning,
                                  String(localized: "ありません。最初の画像や参照画像を使う生成はできません。"),
                                  hint: String(localized: "例：Qwen/Qwen3-VL-32B-Instruct-GGUF にある mmproj-Qwen3VL-32B-Instruct-F16.gguf")))
            }
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
            items.append(item("ref2va", String(localized: "DiT（Ref2VA）"),
                              info.curveAdaLN ? .ok : .warning, describe(ref), part: .ref2va, chosen: ref))
        } else {
            items.append(item("ref2va", String(localized: "DiT（Ref2VA）"), .info,
                              String(localized: "ありません。参照画像・動画・音声を使う生成はできません（文章から・最初の画像からは使えます）。"),
                              hint: String(localized: "例：unsloth/MiniMax-H3-GGUF にある minimax_h3_ref2va_pruned-Q8_0.gguf（DiTのファイルは、名前に ref2va を含むものをRef2VAとみなします）")))
        }

        // Shared parts
        if let base {
            items.append(item("shared", String(localized: "トークナイザー・VAE・設定"), .ok,
                              String(localized: "「\(base.name)」から複製します（容量は増えません）")))
        } else {
            items.append(item("shared", String(localized: "トークナイザー・VAE・設定"), .missing,
                              String(localized: "GGUFには入っていません。公式のモデル（FL2VA）を一度ダウンロードして登録してください。")))
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

    private var palette: H3Palette { H3Palette(colorScheme) }
    private var bases: [H3ModelEntry] { preferredImportBases(library.models) }
    private var base: H3ModelEntry? { bases.first { $0.id == baseID } ?? bases.first }
    private var plan: ImportPlan {
        ImportPlan(files: files, ref2vaPaths: ref2vaPaths, base: base, chosen: chosen)
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
            Text("量子化されたモデル（GGUF）のファイルをまとめて選んでください。1つ1つの中身（DiT・文章の部分・画像の部分）を調べ、次の画面で、そろっている部品と足りない部品を示します。GGUFにない部品は、登録済みのモデルから複製します。")
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
                .frame(minHeight: 240)
                if !unused.isEmpty {
                    Text("使われないファイルがあります。そのままでも取り込めますが、不要なら外してください。")
                        .font(.caption).foregroundStyle(.orange)
                }
            }
            HStack {
                Button("GGUFファイルを追加…") { addFiles() }
                Spacer()
            }
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
                    Spacer()
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
                    .frame(maxWidth: 260)
                }
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
                if let hint = item.hint {
                    Text(hint).font(.caption2).foregroundStyle(palette.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
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

    private func startImport() {
        let plan = self.plan
        guard let dit = plan.dit, let base else { return }
        page = 2
        importer.start(.init(name: name, dit: dit, ref2vaDit: plan.ref2vaDit,
                             textEncoder: plan.textEncoder, vision: plan.vision, base: base)) { path in
            importedPath = path
        }
    }
}
