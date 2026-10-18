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

    /// Where a part comes from unless the user named another repository.
    var defaultRepo: String {
        switch self {
        case .dit, .ref2va, .shared: return "unsloth/MiniMax-H3-GGUF"
        case .textEncoder, .vision: return "Qwen/Qwen3-VL-32B-Instruct-GGUF"
        }
    }

    /// Publishers known to carry this part (offered next to the name field).
    var knownRepos: [String] {
        switch self {
        case .dit, .ref2va:
            return ["unsloth/MiniMax-H3-GGUF", "Abiray/MiniMax-H3-Pruned-GGUF", "Abiray/MiniMax-H3-GGUF",
                    "leejet/MiniMax-H3-GGUF", "molbal/MiniMax-H3-GGUF", "realrebelai/MiniMax-H3_GGUFs",
                    "joeygambino/MiniMax-H3-x-Z-Image-GGUF"]
        case .textEncoder:
            return ["Qwen/Qwen3-VL-32B-Instruct-GGUF", "unsloth/MiniMax-H3-GGUF", "leejet/MiniMax-H3-GGUF",
                    "Abiray/MiniMax-H3-GGUF", "realrebelai/MiniMax-H3_GGUFs",
                    "nif0/Qwen3-VL-32B-Instruct-MiniMax-H3-GGUF"]
        case .vision:
            return ["Qwen/Qwen3-VL-32B-Instruct-GGUF", "nif0/Qwen3-VL-32B-Instruct-MiniMax-H3-GGUF"]
        case .shared:
            return ["unsloth/MiniMax-H3-GGUF", "Comfy-Org/MiniMax-H3"]
        }
    }

    /// The repository last used for this part (kept between runs).
    var savedRepo: String {
        get { UserDefaults.standard.string(forKey: "H3cApp.downloadRepo." + rawValue) ?? defaultRepo }
        nonmutating set { UserDefaults.standard.set(newValue, forKey: "H3cApp.downloadRepo." + rawValue) }
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
    /// Why it can't be used (shown, not selectable).
    var unsupported: String?
    var bytes: Int64 { entries.reduce(0) { $0 + $1.size } }

    /// The picker's line: title, and whether it is tested or unusable.
    var label: String {
        if let unsupported { return title + String(localized: "（未対応：\(unsupported)）") }
        return title + (verified ? String(localized: "（動作確認済み）") : "")
    }

    static func == (a: Self, b: Self) -> Bool { a.id == b.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }

    /// Files the app has generated with end to end.
    private static let verifiedFiles: Set<String> = [
        "unsloth/MiniMax-H3-GGUF/minimax_h3_fl2va_pruned-Q8_0.gguf",
        "Qwen/Qwen3-VL-32B-Instruct-GGUF/Qwen3VL-32B-Instruct-Q8_0.gguf",
        "Qwen/Qwen3-VL-32B-Instruct-GGUF/mmproj-Qwen3VL-32B-Instruct-F16.gguf",
        "unsloth/MiniMax-H3-GGUF/vae/minimax_h3_video_vae_fp16.safetensors",
    ]

    private static func size(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    /// A part's files in `repo`, sorted out by reading each file's header:
    /// usable ones largest (best) first, then the ones the app can't read.
    static func list(_ kind: ImportDownloadKind, repo: String,
                     progress: @escaping @Sendable (Int, Int) -> Void = { _, _ in }) async throws -> [ImportDownloadChoice] {
        let repo = repo.trimmingCharacters(in: .whitespaces)
        if kind == .shared { return try await sharedChoices(vaeRepo: repo, progress: progress) }
        let files = try await HFRepoProbe.probe(repo: repo, progress: progress)
        let matches: [HFProbedFile] = files.filter { file in
            switch kind {
            case .dit: return file.kind == .dit && !file.isRef2VA
            case .ref2va: return file.kind == .dit && file.isRef2VA
            case .textEncoder: return file.kind == .textEncoder
            case .vision: return file.kind == .vision
            case .shared: return false
            }
        }
        return matches.map { choice(from: $0, kind: kind) }
            .sorted { ($0.unsupported == nil ? 0 : 1, -$0.bytes) < ($1.unsupported == nil ? 0 : 1, -$1.bytes) }
    }

    private static func choice(from file: HFProbedFile, kind: ImportDownloadKind) -> ImportDownloadChoice {
        var notes = [file.mainType, size(file.size)].filter { !$0.isEmpty }
        var unsupported = file.unsupported
        var writes: Int64 = 0
        switch kind {
        case .dit, .ref2va:
            notes.append(file.curveAdaLN ? String(localized: "AdaLNは曲線形")
                         : String(localized: "AdaLNは密な形（動作未確認）"))
        case .textEncoder:
            if file.textLayers < 50 {
                unsupported = unsupported ?? String(localized: "層が足りません（\(file.textLayers)層、50層が必要）")
            } else if file.textLayers > 50 {
                notes.append(String(localized: "\(file.textLayers)層（50層に切り詰めて書き出し）"))
                writes = file.size / Int64(file.textLayers) * 50
            } else {
                notes.append(String(localized: "50層"))
            }
            if file.hasVision { notes.append(String(localized: "画像の部分入り")) }
        default:
            break
        }
        let entry = HFFileEntry(path: file.path, size: file.size, repo: file.repo)
        return ImportDownloadChoice(id: file.repo + "/" + file.path,
                                    title: ([file.fileName] + notes).joined(separator: itemSeparator),
                                    entries: [entry],
                                    verified: verifiedFiles.contains(file.repo + "/" + file.path),
                                    includesVision: file.hasVision, importWrites: writes,
                                    unsupported: unsupported)
    }

    /// The official model's shared parts (FL2VA without the DiT and text
    /// encoder weights, the Python sources and shard indexes; plus the two
    /// Ref2VA files that differ from FL2VA's), with the official VAEs or
    /// the video and audio VAEs found in `vaeRepo`, placed where the
    /// official ones go.
    private static func sharedChoices(vaeRepo: String, progress: @escaping @Sendable (Int, Int) -> Void)
        async throws -> [ImportDownloadChoice] {
        let fl2va = try await ModelDownloader.fetchEntries(repo: officialModelRepo, subpath: "FL2VA").filter {
            !$0.path.contains("/transformer/model") && !$0.path.contains("/text_encoder/model") &&
                !$0.path.hasSuffix(".py")
        }
        let ref2va = try await ModelDownloader.fetchEntries(repo: officialModelRepo, subpath: "Ref2VA").filter {
            $0.path == "Ref2VA/model_index.json" || $0.path == "Ref2VA/transformer/config.json"
        }
        let videoLocal = "FL2VA/video_vae/source/model.safetensors"
        let audioLocal = "FL2VA/audio_vae/model.safetensors"
        var choices = [ImportDownloadChoice(
            id: "shared", title: String(localized: "公式のVAE（MiniMaxAI/MiniMax-H3、映像はF32）"),
            entries: fl2va + ref2va, verified: true)]
        guard !vaeRepo.isEmpty, vaeRepo != officialModelRepo else { return choices }

        let files = try await HFRepoProbe.probe(repo: vaeRepo, progress: progress)
        // The same file is sometimes published twice (unsloth's vae/ and
        // vae/official/); one of each is enough.
        var seen = Set<String>()
        let unique = files.filter { seen.insert("\($0.kind)|\($0.fileName)|\($0.size)").inserted }
        let audio = unique.first { $0.kind == .audioVAE && $0.unsupported == nil }
        let officialAudio = fl2va.first { $0.path == audioLocal }
        for video in unique where video.kind == .videoVAE {
            var entries = fl2va.filter { $0.path != videoLocal && $0.path != audioLocal } + ref2va
            var videoEntry = HFFileEntry(path: video.path, size: video.size, repo: vaeRepo)
            videoEntry.localPath = videoLocal
            entries.append(videoEntry)
            var audioNote = String(localized: "音声は公式")
            if let audio {
                var audioEntry = HFFileEntry(path: audio.path, size: audio.size, repo: vaeRepo)
                audioEntry.localPath = audioLocal
                entries.append(audioEntry)
                audioNote = String(localized: "音声 \(audio.mainType)")
            } else if let officialAudio {
                entries.append(officialAudio)
            }
            let title = String(localized: "\(vaeRepo)のVAE（映像 \(video.mainType)・\(audioNote)）")
            choices.append(ImportDownloadChoice(
                id: "shared|" + vaeRepo + "/" + video.path, title: title, entries: entries,
                verified: verifiedFiles.contains(vaeRepo + "/" + video.path), unsupported: video.unsupported))
        }
        return choices
    }
}

/// "配布元": the Hugging Face repository a part comes from - typed, or
/// picked from the publishers known to carry it - and the button that
/// fetches its file list.
struct RepoSourceField: View {
    let kind: ImportDownloadKind
    @Binding var repo: String
    var busy: Bool
    var onFetch: () -> Void

    var body: some View {
        HStack(spacing: 4) {
            TextField("配布元（Hugging Faceのリポジトリ名）", text: $repo)
                .textFieldStyle(.roundedBorder)
                .font(.caption)
                .onSubmit { onFetch() }
            Menu {
                ForEach(kind.knownRepos, id: \.self) { name in
                    Button(name) {
                        repo = name
                        onFetch()
                    }
                }
            } label: {
                Image(systemName: "list.bullet")
            }
            .menuStyle(.borderlessButton)
            .frame(width: 28)
            .help("この部品を配布している既知のリポジトリから選びます")
            Button {
                onFetch()
            } label: {
                Image(systemName: "arrow.down.circle")
            }
            .buttonStyle(.borderless)
            .help("Hugging Faceからこの配布元のファイル一覧を取得し、各ファイルの先頭を読んで部品ごとに分けます")
            .disabled(busy || repo.trimmingCharacters(in: .whitespaces).isEmpty)
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
    @State private var repo: String
    @State private var loading = false
    @State private var probed: (done: Int, total: Int) = (0, 0)

    init(kind: ImportDownloadKind, onFinished: @escaping (String) -> Void, onCancelled: @escaping () -> Void) {
        self.kind = kind
        self.onFinished = onFinished
        self.onCancelled = onCancelled
        _parent = State(initialValue: (kind.destination as NSString).deletingLastPathComponent)
        _folderName = State(initialValue: (kind.destination as NSString).lastPathComponent)
        _repo = State(initialValue: kind.savedRepo)
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
    /// The picked file; unusable ones are never picked.
    private var choice: ImportDownloadChoice? {
        let usable = choices.filter { $0.unsupported == nil }
        return usable.first { $0.id == choiceID } ?? usable.first { $0.verified } ?? usable.first
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("\(kind.title)をダウンロード").font(.system(size: 16, weight: .semibold))
            Text(kind == .shared ? String(localized: "VAEの配布元（Hugging Face）") : String(localized: "配布元（Hugging Face）"))
                .font(.caption).foregroundStyle(palette.textSecondary)
            RepoSourceField(kind: kind, repo: $repo, busy: loading || downloader.state == .downloading) {
                Task { await loadChoices() }
            }
            Divider()
            if loading {
                HStack {
                    ProgressView().controlSize(.small)
                    Text(probed.total > 0
                         ? String(localized: "ファイルの先頭を読んでいます（\(probed.done)/\(probed.total)）…")
                         : String(localized: "ファイル一覧を確認しています…"))
                        .font(.caption).foregroundStyle(palette.textSecondary)
                }
            } else if let listError {
                Text(listError).font(.caption).foregroundStyle(palette.errorColor)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                ScrollView {
                    Picker(kind == .shared ? "VAE" : String(localized: "ファイル"),
                           selection: Binding(get: { choice?.id ?? "" }, set: { id in
                               // An unusable file stays listed but can't be picked.
                               if choices.first(where: { $0.id == id })?.unsupported == nil { choiceID = id }
                           })) {
                        ForEach(choices) { item in
                            Text(verbatim: kind == .shared ? "\(item.label)\(itemSeparator)\(gigabytes(item.bytes))" : item.label)
                                .foregroundStyle(item.unsupported == nil ? palette.textPrimary : palette.textSecondary)
                                .tag(item.id)
                        }
                    }
                    .pickerStyle(.radioGroup)
                    .disabled(downloader.state == .downloading)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 160)
                Text(kindNote).font(.caption).foregroundStyle(palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ModelLicenseNotice(includesQwen: kind == .textEncoder || kind == .vision)
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
        .frame(width: 640, height: 640)
        .background(palette.canvas)
        .task { await loadChoices() }
        .onDisappear { downloader.cancel() }
        .onChange(of: downloader.state) { _, state in
            if state == .completed { finish() }
        }
    }

    private var kindNote: String {
        switch kind {
        case .dit, .ref2va:
            return String(localized: "各ファイルの先頭を読んで、DiTのファイルだけを並べています。小さい形式ほど容量は減り、画質は落ちます。「動作確認済み」以外は、読み込めることは確認していますが、画質はまだ確かめていません。")
        case .textEncoder:
            return String(localized: "50層より多いものは、取り込むときに使う50層だけに切り詰めて書き出します。画像の部分入りのものなら、mmprojは要りません。")
        case .vision:
            return String(localized: "最初の画像や参照画像を使う生成に必要です。")
        case .shared:
            return String(localized: "公式のモデルから、DiTと文章の部分の重みを除いた部品（トークナイザー・VAE・設定）だけをダウンロードします。これらはGGUFの形では配布されていません。VAEは、上の配布元にあるもの（unslothの映像FP16など）も選べます。")
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
                Button("ダウンロードする") { start() }
                    .buttonStyle(.borderedProminent)
                    .disabled(choice == nil || space?.fits != true)
            }
        }
    }

    /// Lists the part's files in the named repository (reading each
    /// file's header), and remembers the repository once that worked.
    private func loadChoices() async {
        loading = true
        listError = nil
        choices = []
        probed = (0, 0)
        defer { loading = false }
        let name = repo.trimmingCharacters(in: .whitespaces)
        do {
            choices = try await ImportDownloadChoice.list(kind, repo: name) { done, total in
                Task { @MainActor in probed = (done, total) }
            }
            kind.savedRepo = name
            if choices.isEmpty {
                listError = String(localized: "「\(name)」には、この部品のファイルが見つかりませんでした。")
            } else if choices.allSatisfy({ $0.unsupported != nil }) {
                listError = String(localized: "「\(name)」のこの部品のファイルは、どれもこのアプリでは使えません。")
            }
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

/// The license note shown wherever model files are downloaded: MiniMax H3's
/// own license covers the official model and the GGUF conversions of its
/// DiT (and unsloth's H3 text encoders); Qwen's files are Apache 2.0.
struct ModelLicenseNotice: View {
    var includesQwen = false
    @Environment(\.colorScheme) private var colorScheme

    private var palette: H3Palette { H3Palette(colorScheme) }

    var body: some View {
        HStack(alignment: .top, spacing: H3Spacing.sm) {
            Image(systemName: "doc.text.magnifyingglass").foregroundStyle(palette.accent)
            VStack(alignment: .leading, spacing: 4) {
                Text(includesQwen
                     ? "MiniMax H3のモデル（公式のもの、unslothのGGUF）には「MiniMax H3 Community License Agreement」が、Qwenの文章の部分・画像の部分には「Apache License 2.0」が適用されます。ダウンロードと利用の前に各ライセンスを読み、その条件に従ってください。"
                     : "MiniMax H3のモデルには「MiniMax H3 Community License Agreement」が適用されます。ダウンロードと利用の前にライセンスを読み、その条件に従ってください。")
                    .font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: H3Spacing.md) {
                    Link("MiniMax H3のライセンス",
                         destination: URL(string: "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE")!)
                    if includesQwen {
                        Link("Qwen3-VLのライセンス（Apache 2.0）",
                             destination: URL(string: "https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-GGUF")!)
                    }
                }
                .font(.caption)
            }
        }
        .padding(H3Spacing.sm)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(palette.accent.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
    }
}
