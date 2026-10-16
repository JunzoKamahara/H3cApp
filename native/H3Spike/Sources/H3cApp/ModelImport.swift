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
// with only the 50 layers the engine reads.

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
struct ImportPick: Equatable {
    let path: String
    let info: H3GGUFInfo?
    let error: String?

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
}

@MainActor
final class ModelImporter: ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var step: String = ""
    @Published private(set) var fraction: Double?
    @Published var errorMessage: String?
    private var task: Task<Void, Never>?
    private var cancelled = false

    struct Request {
        var name: String
        var dit: ImportPick
        var ref2vaDit: ImportPick?
        var textEncoder: ImportPick?
        var vision: ImportPick?
        var base: H3ModelEntry
    }

    func start(_ request: Request, onFinished: @escaping (String) -> Void) {
        guard !isRunning else { return }
        isRunning = true
        cancelled = false
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
        cancelled = true
        task?.cancel()
    }

    /// The same import, awaited (for the automation API).
    func importNow(_ request: Request) async throws -> String {
        guard !isRunning else { throw CancellationError() }
        isRunning = true
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
                    isCancelled: { [weak self] in
                        MainActor.assumeIsolated { self?.cancelled ?? true }
                    })
                fraction = nil
            } else {
                step = String(localized: "文章の部分を置いています")
                try fm.copyItem(atPath: te.path, toPath: textEncoder + "/model.gguf")
            }
            parts.append(.init(role: "textEncoder", file: "FL2VA/text_encoder/model.gguf",
                               source: te.fileName, type: info.mainType,
                               bytes: (try? fm.attributesOfItem(atPath: textEncoder + "/model.gguf")[.size]
                                       as? UInt64) ?? 0))
            if let vision = request.vision {
                try fm.copyItem(atPath: vision.path, toPath: textEncoder + "/mmproj.gguf")
                parts.append(.init(role: "vision", file: "FL2VA/text_encoder/mmproj.gguf",
                                   source: vision.fileName, type: vision.info?.mainType ?? "",
                                   bytes: vision.info?.fileBytes ?? 0))
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

struct ModelImportView: View {
    @ObservedObject var library: ModelLibrary
    var onImported: (String) -> Void
    var onCancelled: () -> Void

    @StateObject private var importer = ModelImporter()
    @Environment(\.colorScheme) private var colorScheme
    @State private var name = "MiniMax-H3 GGUF"
    @State private var dit: ImportPick?
    @State private var ref2vaDit: ImportPick?
    @State private var textEncoder: ImportPick?
    @State private var vision: ImportPick?
    @State private var baseID: UUID?

    private var palette: H3Palette { H3Palette(colorScheme) }
    private var bases: [H3ModelEntry] { library.models.filter(isUsableImportBase) }
    private var base: H3ModelEntry? {
        bases.first { $0.id == baseID } ?? bases.first
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("GGUFを取り込む").font(.system(size: 16, weight: .semibold))
            Text("量子化されたモデル（GGUF）から、容量の小さいモデルを作ります。トークナイザーやVAEなど、GGUFに入っていない部品は登録済みのモデルから複製します（同じディスクなら容量は増えません）。")
                .font(.caption)
                .foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)

            slot(String(localized: "DiT（FL2VA、必須）"), pick: $dit, role: .dit)
            slot(String(localized: "DiT（Ref2VA、参照画像を使う場合）"), pick: $ref2vaDit, role: .dit)
            slot(String(localized: "文章の部分（テキストエンコーダー）"), pick: $textEncoder, role: .textEncoder)
            slot(String(localized: "画像の部分（mmproj）"), pick: $vision, role: .vision)

            HStack {
                Text("残りの部品の複製元").font(.caption)
                Picker("", selection: Binding(get: { base?.id }, set: { baseID = $0 })) {
                    ForEach(bases) { model in Text(model.name).tag(Optional(model.id)) }
                }
                .labelsHidden()
                .frame(maxWidth: 260)
            }
            if bases.isEmpty {
                Text("複製元にできるモデル（FL2VAのトークナイザーとVAEを含むもの）が登録されていません。")
                    .font(.caption)
                    .foregroundStyle(palette.errorColor)
            }
            HStack {
                Text("名前").font(.caption)
                TextField("名前", text: $name).textFieldStyle(.roundedBorder).frame(maxWidth: 260)
            }
            ForEach(notes, id: \.self) { note in
                Text(note).font(.caption).foregroundStyle(palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if importer.isRunning {
                HStack {
                    if let fraction = importer.fraction {
                        ProgressView(value: fraction).frame(maxWidth: 200)
                    } else {
                        ProgressView().controlSize(.small)
                    }
                    Text(importer.step).font(.caption).foregroundStyle(palette.textSecondary)
                }
            }
            if let message = importer.errorMessage {
                Text(message).font(.caption).foregroundStyle(palette.errorColor)
            }
            Spacer(minLength: 0)
            HStack {
                Spacer()
                Button(importer.isRunning ? "中止" : "キャンセル") {
                    if importer.isRunning { importer.cancel() } else { onCancelled() }
                }
                Button("取り込む") { startImport() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!canImport)
            }
        }
        .padding(H3Spacing.lg)
        .frame(width: 560, height: 560)
        .background(palette.canvas)
    }

    private func slot(_ title: String, pick: Binding<ImportPick?>, role: H3GGUFInfo.Role) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title).font(.caption)
                Spacer()
                if pick.wrappedValue != nil {
                    Button("外す") { pick.wrappedValue = nil }.disabled(importer.isRunning)
                }
                Button("ファイルを選ぶ…") {
                    if let path = chooseFile(allowedContentTypes: ggufTypes) {
                        pick.wrappedValue = ImportPick(path: path)
                    }
                }
                .disabled(importer.isRunning)
            }
            if let picked = pick.wrappedValue {
                Text(describe(picked, expected: role))
                    .font(.caption2)
                    .foregroundStyle(problem(picked, expected: role) == nil
                                     ? palette.textSecondary : palette.errorColor)
                    .lineLimit(2)
            }
        }
    }

    private func describe(_ pick: ImportPick, expected: H3GGUFInfo.Role) -> String {
        if let problem = problem(pick, expected: expected) { return pick.fileName + " — " + problem }
        guard let info = pick.info else { return pick.fileName }
        let size = ByteCountFormatter.string(fromByteCount: Int64(info.fileBytes), countStyle: .file)
        var parts = [pick.fileName, info.mainType, size]
        if info.role == .textEncoder {
            parts.append(String(localized: "\(info.textLayers)層"))
            if info.hasVision { parts.append(String(localized: "画像の部分を含む")) }
        }
        return parts.joined(separator: summarySeparator)
    }

    private func problem(_ pick: ImportPick, expected: H3GGUFInfo.Role) -> String? {
        guard let info = pick.info else {
            return pick.error ?? String(localized: "GGUFとして読めません")
        }
        if info.role != expected {
            switch info.role {
            case .dit: return String(localized: "これはDiTのファイルです")
            case .textEncoder: return String(localized: "これは文章の部分（テキストエンコーダー）のファイルです")
            case .vision: return String(localized: "これは画像の部分（mmproj）のファイルです")
            case .unknown: return String(localized: "MiniMax H3の部品として認識できません")
            }
        }
        if info.role == .textEncoder && info.textLayers < 50 {
            return String(localized: "層が足りません（50層が必要です）")
        }
        return nil
    }

    private var notes: [String] {
        var notes: [String] = []
        if let info = dit?.info, info.role == .dit, !info.curveAdaLN {
            notes.append(String(localized: "このDiTはAdaLNが密な形です。この形式での動作はまだ確認していません。"))
        }
        if textEncoder == nil {
            notes.append(String(localized: "文章の部分を選ばない場合は、複製元のモデルのもの（約62GB）を使います。"))
        } else if textEncoder?.info?.hasVision == false && vision == nil {
            notes.append(String(localized: "画像の部分（mmproj）がないと、最初の画像や参照画像を使う生成はできません。"))
        }
        if ref2vaDit == nil {
            notes.append(String(localized: "Ref2VAのDiTを選ばない場合、参照画像・動画・音声を使う生成はできません。"))
        }
        return notes
    }

    private var canImport: Bool {
        guard !importer.isRunning, base != nil, let dit, problem(dit, expected: .dit) == nil else { return false }
        for (pick, role) in [(ref2vaDit, H3GGUFInfo.Role.dit), (textEncoder, .textEncoder), (vision, .vision)] {
            if let pick, problem(pick, expected: role) != nil { return false }
        }
        return true
    }

    private func startImport() {
        guard let dit, let base else { return }
        importer.start(.init(name: name, dit: dit, ref2vaDit: ref2vaDit, textEncoder: textEncoder,
                             vision: vision, base: base)) { path in
            onImported(path)
        }
    }
}
