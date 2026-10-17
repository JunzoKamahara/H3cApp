import Foundation
import SwiftUI

// Downloads for the GGUF import wizard: a missing part straight from
// Hugging Face (the GGUF files), or - when no model is registered - only
// the official model's shared parts (tokenizer, configs, VAEs, about 11 GB)
// instead of the whole ~130 GB model. Tokenizers and VAEs have no GGUF form
// (a tokenizer is a few MB of JSON, not weights), so those come from the
// official repository as they are.

/// Where the wizard puts downloaded GGUF files.
let importDownloadsDirectory = h3AppSupportDirectory + "/Downloads"
/// Where the wizard puts the official model's shared parts. Not a model of
/// its own (no DiT, no text encoder weights), so it is not registered; the
/// wizard offers it as the base once the download has finished.
let importBaseDirectory = h3AppSupportDirectory + "/ImportBase/MiniMax-H3"
private let importBaseCompleteMarker = ".h3c-download-complete"

/// The shared-parts folder, when its download has finished.
func downloadedImportBase() -> H3ModelEntry? {
    guard FileManager.default.fileExists(atPath: importBaseDirectory + "/" + importBaseCompleteMarker) else {
        return nil
    }
    let entry = H3ModelEntry(id: UUID(uuidString: "6A1C0B5E-0000-4000-8000-00000000BA5E")!,
                             name: String(localized: "公式モデルの共通部品（ダウンロード済み）"),
                             path: importBaseDirectory)
    return isUsableImportBase(entry) ? entry : nil
}

/// Whether a base model's text encoder folder has weights (the shared-parts
/// download has only its configs).
func baseHasTextEncoderWeights(_ model: H3ModelEntry) -> Bool {
    let names = (try? FileManager.default.contentsOfDirectory(atPath: model.path + "/FL2VA/text_encoder")) ?? []
    return names.contains { $0.hasSuffix(".safetensors") || $0.hasSuffix(".gguf") }
}

/// What the wizard can download.
enum ImportDownloadKind: String, Identifiable, CaseIterable {
    case dit, ref2va, textEncoder, vision, shared
    var id: String { rawValue }

    var repo: String {
        switch self {
        case .dit, .ref2va: return "unsloth/MiniMax-H3-GGUF"
        case .textEncoder, .vision: return "Qwen/Qwen3-VL-32B-Instruct-GGUF"
        case .shared: return officialModelRepo
        }
    }

    var title: String {
        switch self {
        case .dit: return String(localized: "DiT（FL2VA）")
        case .ref2va: return String(localized: "DiT（Ref2VA）")
        case .textEncoder: return String(localized: "文章の部分（テキストエンコーダー）")
        case .vision: return String(localized: "画像の部分")
        case .shared: return String(localized: "トークナイザー・VAE・設定")
        }
    }

    var destination: String { self == .shared ? importBaseDirectory : importDownloadsDirectory }
}

/// One thing to download: a GGUF file, or the set of shared parts.
struct ImportDownloadChoice: Identifiable, Hashable {
    let id: String
    let title: String
    let entries: [HFFileEntry]
    /// Checked end to end with this app (the others are read by the same
    /// code, but their quality has not been looked at).
    let verified: Bool
    /// A text encoder file that also holds the vision tower.
    var includesVision = false
    /// What the import writes on top of the download (a 64-layer text
    /// encoder is cut to the 50 layers used).
    var importWrites: Int64 = 0
    var bytes: Int64 { entries.reduce(0) { $0 + $1.size } }

    static func == (a: Self, b: Self) -> Bool { a.id == b.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }

    /// The repository's current files for a part, largest (best) first.
    static func list(_ kind: ImportDownloadKind) async throws -> [ImportDownloadChoice] {
        func files(prefix: String, verifiedType: String, repo: String? = nil,
                   label: String = "") async throws -> [ImportDownloadChoice] {
            try await ModelDownloader.fetchEntries(repo: repo ?? kind.repo, subpath: "")
                // Not split files, nor unsloth's "UD" mixes (their tensor
                // types have not been checked against what the engine reads).
                .filter {
                    $0.path.hasPrefix(prefix) && $0.path.hasSuffix(".gguf") && !$0.path.contains("-split-") &&
                        !$0.path.contains("-UD-")
                }
                .sorted { $0.size > $1.size }
                .map { entry in
                    let type = String(entry.path.dropFirst(prefix.count).dropLast(".gguf".count))
                    var choice = ImportDownloadChoice(id: entry.path, title: label + type, entries: [entry],
                                                      verified: type == verifiedType)
                    if kind == .textEncoder {
                        // unsloth's H3 files are cut already and carry the
                        // vision tower; Qwen's have all 64 layers, no vision.
                        choice.includesVision = repo != nil
                        choice.importWrites = repo != nil ? 0 : entry.size / 64 * 50
                    }
                    return choice
                }
        }
        switch kind {
        case .dit: return try await files(prefix: "minimax_h3_fl2va_pruned-", verifiedType: "Q8_0")
        case .ref2va: return try await files(prefix: "minimax_h3_ref2va_pruned-", verifiedType: "")
        case .textEncoder:
            // unsloth's H3 text encoders are already cut to the 50 layers
            // and carry the vision tower (headers and sample tensors checked
            // against the released weights; generation quality not yet).
            let h3 = try await files(prefix: "qwen3vl_32b_minimax_h3-", verifiedType: "",
                                     repo: "unsloth/MiniMax-H3-GGUF",
                                     label: String(localized: "H3専用・50層・画像の部分入り "))
            return try await files(prefix: "Qwen3VL-32B-Instruct-", verifiedType: "Q8_0") + h3
        case .vision: return try await files(prefix: "mmproj-Qwen3VL-32B-Instruct-", verifiedType: "F16")
        case .shared:
            // FL2VA without the DiT and text encoder weights (and the
            // Python sources and shard indexes nothing here reads), plus
            // the two Ref2VA files that differ from FL2VA's.
            let fl2va = try await ModelDownloader.fetchEntries(repo: kind.repo, subpath: "FL2VA").filter {
                !$0.path.contains("/transformer/model") && !$0.path.contains("/text_encoder/model") &&
                    !$0.path.hasSuffix(".py")
            }
            let ref2va = try await ModelDownloader.fetchEntries(repo: kind.repo, subpath: "Ref2VA").filter {
                $0.path == "Ref2VA/model_index.json" || $0.path == "Ref2VA/transformer/config.json"
            }
            // The same parts with unsloth's VAE files (video in FP16, half
            // the size; audio with its weight normalization folded), put
            // where the official ones go.
            let vaeFiles = ["FL2VA/video_vae/source/model.safetensors": "vae/minimax_h3_video_vae_fp16.safetensors",
                            "FL2VA/audio_vae/model.safetensors": "vae/minimax_h3_audio_vae_fp32.safetensors"]
            let unsloth = try await ModelDownloader.fetchEntries(repo: "unsloth/MiniMax-H3-GGUF", subpath: "vae")
            var replaced = fl2va.filter { vaeFiles[$0.path] == nil }
            for (local, remote) in vaeFiles {
                guard var entry = unsloth.first(where: { $0.path == remote }) else { return [] }
                entry.localPath = local
                replaced.append(entry)
            }
            return [ImportDownloadChoice(id: "shared", title: String(localized: "公式のVAE（映像はF32）"),
                                         entries: fl2va + ref2va, verified: true),
                    ImportDownloadChoice(id: "shared-unsloth", title: String(localized: "unslothのVAE（映像はFP16）"),
                                         entries: replaced + ref2va, verified: false)]
        }
    }
}

/// The download sheet the wizard opens for one part.
struct ImportDownloadView: View {
    let kind: ImportDownloadKind
    /// The downloaded GGUF file, or the shared-parts folder.
    var onFinished: (String) -> Void
    var onCancelled: () -> Void

    @StateObject private var downloader = ModelDownloader()
    @Environment(\.colorScheme) private var colorScheme
    @State private var choices: [ImportDownloadChoice] = []
    @State private var choiceID: String?
    @State private var listError: String?
    @State private var parent: String
    @State private var folderName: String

    init(kind: ImportDownloadKind, onFinished: @escaping (String) -> Void, onCancelled: @escaping () -> Void) {
        self.kind = kind
        self.onFinished = onFinished
        self.onCancelled = onCancelled
        _parent = State(initialValue: (kind.destination as NSString).deletingLastPathComponent)
        _folderName = State(initialValue: (kind.destination as NSString).lastPathComponent)
    }

    private var destination: Result<String, DestinationError> {
        DownloadDestinationFields.path(parent: parent, name: folderName)
    }

    private var space: (text: String, fits: Bool)? {
        guard let choice, case .success(let path) = destination else { return nil }
        return downloadSpaceCheck(needed: choice.bytes - ModelDownloader.presentBytes(choice.entries, at: path),
                                  destination: path)
    }

    private var palette: H3Palette { H3Palette(colorScheme) }
    private var choice: ImportDownloadChoice? {
        choices.first { $0.id == choiceID } ?? choices.first
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("\(kind.title)をダウンロード").font(.system(size: 16, weight: .semibold))
            Text("Hugging Face：\(choice?.entries.first?.repo ?? kind.repo)").font(.caption).foregroundStyle(palette.textSecondary)
                .textSelection(.enabled)
            Divider()
            if let listError {
                Text(listError).font(.caption).foregroundStyle(palette.errorColor)
            } else if choices.isEmpty {
                HStack {
                    ProgressView().controlSize(.small)
                    Text("ファイル一覧を確認しています…").font(.caption).foregroundStyle(palette.textSecondary)
                }
            } else if kind == .shared {
                Picker("VAE", selection: Binding(get: { choice?.id ?? "" }, set: { choiceID = $0 })) {
                    ForEach(choices) { item in
                        Text(verbatim: "\(item.title)・\(gigabytes(item.bytes))").tag(item.id)
                    }
                }
                .pickerStyle(.radioGroup)
                .disabled(downloader.state == .downloading)
                Text("公式のモデルから、DiTと文章の部分の重みを除いた部品（トークナイザー・VAE・設定）だけをダウンロードします。これらはGGUFの形では配布されていません。VAEは、unslothが配っている小さいもの（映像のVAEをFP16にしたもの）も選べます。DiTと文章の部分はGGUFで用意してください。")
                    .font(.caption).foregroundStyle(palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Picker("ファイル", selection: Binding(get: { choice?.id ?? "" }, set: { choiceID = $0 })) {
                    ForEach(choices) { item in
                        Text(verbatim: "\(item.title)・\(gigabytes(item.bytes))" +
                             (item.verified ? String(localized: "（動作確認済み）") : ""))
                            .tag(item.id)
                    }
                }
                .pickerStyle(.radioGroup)
                .disabled(downloader.state == .downloading)
                Text(kindNote).font(.caption).foregroundStyle(palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            DownloadDestinationFields(parent: $parent, name: $folderName,
                                      disabled: downloader.state == .downloading)
            if case .failure(let error) = destination {
                Text(error.message).font(.caption).foregroundStyle(palette.errorColor)
            } else if let space {
                Text(space.text).font(.caption)
                    .foregroundStyle(space.fits ? palette.textSecondary : palette.errorColor)
                    .fixedSize(horizontal: false, vertical: true)
            }
            status
            Spacer(minLength: 0)
            footer
        }
        .padding(H3Spacing.lg)
        .frame(width: 560, height: 520)
        .background(palette.canvas)
        .task { await loadChoices() }
        .onDisappear { downloader.cancel() }
        .onChange(of: downloader.state) { _, state in
            if state == .completed { finish() }
        }
    }

    private var kindNote: String {
        switch kind {
        case .dit:
            return String(localized: "小さいものほど容量は減りますが、画質が落ちます。Q8_0以外は、読み込めることは確認していますが、画質はまだ確かめていません。")
        case .ref2va:
            return String(localized: "Ref2VAのGGUFは、まだこのアプリで試していません（FL2VAのものと同じ作りです）。小さいものほど容量は減りますが、画質が落ちます。")
        case .textEncoder:
            return String(localized: "Qwen公式のものは、取り込むときに使う50層だけに切り詰めて書き出します（Q8_0で約27GB）。H3専用のもの（unsloth）は切り詰め済みで画像の部分も入っているので、書き出しもmmprojも要りません（名前と形は公式の重みと一致を確認済み。画質は未確認）。")
        case .vision:
            return String(localized: "最初の画像や参照画像を使う生成に必要です。")
        case .shared:
            return ""
        }
    }

    @ViewBuilder private var status: some View {
        switch downloader.state {
        case .downloading:
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: Double(downloader.downloadedBytes), total: Double(max(downloader.totalBytes, 1)))
                Text(verbatim: "\(gigabytes(downloader.downloadedBytes)) / \(gigabytes(downloader.totalBytes))")
                    .font(.caption).foregroundStyle(palette.textSecondary)
                Text("中断しても、次回は続きから再開します。")
                    .font(.caption).foregroundStyle(palette.textSecondary)
            }
        case .failed(let message):
            Text(message).font(.caption).foregroundStyle(palette.errorColor)
                .fixedSize(horizontal: false, vertical: true)
        default:
            EmptyView()
        }
    }

    private var footer: some View {
        HStack {
            Button("閉じる") { downloader.cancel(); onCancelled() }
            Spacer()
            switch downloader.state {
            case .downloading:
                Button("中止") { downloader.cancel() }
            case .failed:
                Button("再開") { start() }.buttonStyle(.borderedProminent)
                    .disabled(space?.fits == false)
            default:
                Button("ダウンロード") { start() }
                    .buttonStyle(.borderedProminent)
                    .disabled(choice == nil || space?.fits != true)
            }
        }
    }

    private func loadChoices() async {
        do {
            choices = try await ImportDownloadChoice.list(kind)
            if choices.isEmpty { listError = String(localized: "ダウンロードできるファイルが見つかりませんでした。") }
        } catch {
            listError = ModelDownloadError.listingFailed(error.localizedDescription).errorDescription
        }
    }

    private func start() {
        guard let choice, case .success(let path) = destination else { return }
        if kind == .shared {
            try? FileManager.default.removeItem(atPath: path + "/" + importBaseCompleteMarker)
        }
        downloader.use(choice.entries)
        downloader.startDownload(destination: path)
    }

    private func finish() {
        guard let choice, case .success(let path) = destination else { return }
        if kind == .shared {
            FileManager.default.createFile(atPath: path + "/" + importBaseCompleteMarker, contents: Data())
            onFinished(path)
        } else {
            onFinished(path + "/" + (choice.entries.first?.local ?? ""))
        }
    }

    private func gigabytes(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}

/// Inside a model folder being made by the download wizard: the downloaded
/// GGUF files and shared parts, on the same disk as the model so the import
/// clones them instead of copying.
let importDownloadFolderName = ".downloads"

/// "保存先": where a download goes - a folder chosen with 参照… (or typed),
/// and the name of the folder made in it.
struct DownloadDestinationFields: View {
    @Binding var parent: String
    @Binding var name: String
    var disabled = false
    @Environment(\.colorScheme) private var colorScheme

    private var palette: H3Palette { H3Palette(colorScheme) }

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: H3Spacing.sm, verticalSpacing: 6) {
            GridRow {
                Text("保存する場所").font(.caption).foregroundStyle(palette.textSecondary)
                HStack {
                    TextField("保存する場所", text: $parent)
                        .textFieldStyle(.roundedBorder)
                        .font(.caption)
                        .labelsHidden()
                    Button("参照…") {
                        if let path = chooseDirectory() { parent = path }
                    }
                }
            }
            GridRow {
                Text("フォルダ名").font(.caption).foregroundStyle(palette.textSecondary)
                TextField("フォルダ名", text: $name)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
                    .labelsHidden()
                    .frame(maxWidth: 260)
            }
        }
        .disabled(disabled)
    }

    /// The folder the download makes, or nil with the reason it can't be
    /// used.
    static func path(parent: String, name: String) -> Result<String, DestinationError> {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        let place = (parent as NSString).expandingTildeInPath
        guard !trimmed.isEmpty, !trimmed.contains("/"), trimmed != ".", trimmed != ".." else {
            return .failure(.badName)
        }
        guard place.hasPrefix("/") else { return .failure(.badParent) }
        return .success(place + "/" + trimmed)
    }
}

enum DestinationError: Error {
    case badName, badParent

    var message: String {
        switch self {
        case .badName: return String(localized: "フォルダ名を入れてください（「/」は使えません）。")
        case .badParent: return String(localized: "保存する場所を「参照…」で選ぶか、「/」で始まるパスで入れてください。")
        }
    }
}

/// "必要な容量 … ・空き容量 …" for a download into `destination`, and
/// whether it fits (with the downloader's margin).
@MainActor func downloadSpaceCheck(needed: Int64, destination: String) -> (text: String, fits: Bool) {
    let free = ModelDownloader.freeDiskSpace(near: destination)
    let fits = free.map { $0 >= needed + ModelDownloader.spaceMargin } ?? true
    let format = { (bytes: Int64) in ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file) }
    var text = String(localized: "必要な容量：約\(format(max(needed, 0)))・空き容量：\(free.map(format) ?? "?")")
    if !fits {
        text += "\n" + String(localized: "保存先のディスクの空き容量が足りません（余裕として2GBを残します）。保存する場所を変えるか、ディスクを空けてください。")
    }
    return (text, fits)
}
