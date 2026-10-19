import SwiftUI

// "Ref2VAを追加…" in Model Manager: a registered model that has only FL2VA
// gets its Ref2VA later. Ref2VA differs from FL2VA only in its DiT weights
// and model_index.json (every other file of the official release is the
// same bytes), so only those are fetched; the rest is cloned from the
// model's own FL2VA (APFS clones: no extra space on the same disk).
// - A model made from GGUF files gets a Ref2VA DiT GGUF, downloaded from
//   Hugging Face or picked from disk.
// - An official model gets the official Ref2VA DiT (about 66 GB instead of
//   the whole 144 GB folder).
// Everything is put together in a hidden folder inside the model and only
// renamed to Ref2VA once complete; an interrupted download resumes there.

let ref2vaStagingFolderName = ".Ref2VA-partial"

/// The model is made from GGUF files (an import record, or a GGUF DiT).
func isGGUFModel(_ model: H3ModelEntry) -> Bool {
    ModelImportManifest.load(modelPath: model.path) != nil ||
        FileManager.default.fileExists(atPath: model.path + "/FL2VA/transformer/model.gguf")
}

func modelHasRef2VA(_ model: H3ModelEntry) -> Bool {
    FileManager.default.fileExists(atPath: model.path + "/Ref2VA")
}

/// What adding Ref2VA fetches and what it clones from the model's FL2VA.
struct Ref2VAAddPlan {
    /// Placed under the model folder (inside the staging folder).
    var downloads: [HFFileEntry] = []
    /// Absolute paths: a file or folder of FL2VA, and where it goes.
    var clones: [(from: String, to: String)] = []
    var cloneBytes: Int64 = 0

    var downloadBytes: Int64 { downloads.reduce(0) { $0 + $1.size } }

    private static func size(_ path: String) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? NSNumber)?.int64Value ?? -1
    }

    /// The official Ref2VA: files whose bytes FL2VA already has (same git
    /// object id, and the model's copy is complete) are cloned, the rest
    /// downloaded.
    static func official(modelPath: String) async throws -> Ref2VAAddPlan {
        let ref = try await ModelDownloader.fetchEntries(repo: officialModelRepo, subpath: "Ref2VA")
        let fl2va = try await ModelDownloader.fetchEntries(repo: officialModelRepo, subpath: "FL2VA")
        var byPath: [String: HFFileEntry] = [:]
        for entry in fl2va { byPath[String(entry.path.dropFirst("FL2VA/".count))] = entry }
        let staging = modelPath + "/" + ref2vaStagingFolderName
        var plan = Ref2VAAddPlan()
        for entry in ref {
            let relative = String(entry.path.dropFirst("Ref2VA/".count))
            // A Hugging Face snapshot keeps links to its blobs: the file itself.
            let local = (modelPath + "/FL2VA/" + relative as NSString).resolvingSymlinksInPath
            if let same = byPath[relative], same.oid != nil, same.oid == entry.oid,
               same.size == entry.size, size(local) == entry.size {
                plan.clones.append((local, staging + "/" + relative))
                plan.cloneBytes += entry.size
            } else {
                var download = entry
                download.localPath = ref2vaStagingFolderName + "/" + relative
                plan.downloads.append(download)
            }
        }
        return plan
    }

    /// A GGUF model's Ref2VA: everything but the DiT and model_index.json
    /// cloned from its FL2VA. Ref2VA's model_index.json comes from the
    /// downloaded shared parts when they are still there, else from the
    /// official repository (a few hundred bytes).
    static func gguf(modelPath: String) async throws -> Ref2VAAddPlan {
        let fm = FileManager.default
        let staging = modelPath + "/" + ref2vaStagingFolderName
        var plan = Ref2VAAddPlan()
        for item in ["tokenizer", "processor", "video_vae", "audio_vae", "text_encoder", "transformer/config.json"] {
            let source = modelPath + "/FL2VA/" + item
            if fm.fileExists(atPath: source) { plan.clones.append((source, staging + "/" + item)) }
        }
        let sharedIndex = importBaseDirectory + "/Ref2VA/model_index.json"
        if fm.fileExists(atPath: sharedIndex) {
            plan.clones.append((sharedIndex, staging + "/model_index.json"))
        } else {
            let entries = try await ModelDownloader.fetchEntries(repo: officialModelRepo, subpath: "Ref2VA")
            guard var index = entries.first(where: { $0.path == "Ref2VA/model_index.json" }) else {
                throw ModelDownloadError.listingFailed("Ref2VA/model_index.json")
            }
            index.localPath = ref2vaStagingFolderName + "/model_index.json"
            plan.downloads.append(index)
        }
        return plan
    }

    /// A Ref2VA DiT GGUF picked from disk: where it is cloned to, and how
    /// the import record names it. Nil when it is not a DiT.
    static func localDiT(_ pick: ImportPick, modelPath: String)
        -> (clone: (from: String, to: String), part: ModelImportManifest.Part)? {
        guard let info = pick.info, info.role == .dit else { return nil }
        return ((pick.path, modelPath + "/" + ref2vaStagingFolderName + "/transformer/model.gguf"),
                .init(role: "ref2va", file: "Ref2VA/transformer/model.gguf", source: pick.fileName,
                      type: info.mainType, bytes: info.fileBytes))
    }

    /// Small files (model_index.json) fetched in one go, for the API.
    static func downloadSmall(_ entries: [HFFileEntry], modelPath: String) async throws {
        for entry in entries {
            guard let path = entry.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed),
                  let url = URL(string: "https://huggingface.co/\(entry.repo)/resolve/\(hfRevisionComponent(entry.revision))/\(path)") else {
                throw ModelDownloadError.httpStatus(path: entry.path, status: -1)
            }
            let (data, response) = try await URLSession.shared.data(from: url)
            let status = (response as? HTTPURLResponse)?.statusCode ?? -1
            guard (200 ..< 300).contains(status) else { throw ModelDownloadError.httpStatus(path: entry.path, status: status) }
            guard Int64(data.count) == entry.size else { throw ModelDownloadError.sizeMismatch(path: entry.path) }
            let target = modelPath + "/" + entry.local
            try FileManager.default.createDirectory(atPath: (target as NSString).deletingLastPathComponent,
                                                    withIntermediateDirectories: true)
            try data.write(to: URL(fileURLWithPath: target))
        }
    }

    /// Clones what is planned (with a picked DiT file among it), turns the
    /// staging folder into Ref2VA, and records a GGUF DiT in the import
    /// record. Off the main thread: a picked file on another disk is copied.
    static func finish(modelPath: String, clones: [(from: String, to: String)],
                       manifestPart: ModelImportManifest.Part?) throws {
        let fm = FileManager.default
        let target = modelPath + "/Ref2VA"
        guard !fm.fileExists(atPath: target) else {
            throw CocoaError(.fileWriteFileExists, userInfo: [NSFilePathErrorKey: target])
        }
        for clone in clones {
            try fm.createDirectory(atPath: (clone.to as NSString).deletingLastPathComponent,
                                   withIntermediateDirectories: true)
            if fm.fileExists(atPath: clone.to) { try fm.removeItem(atPath: clone.to) }
            try fm.copyItem(atPath: clone.from, toPath: clone.to)
        }
        try fm.moveItem(atPath: modelPath + "/" + ref2vaStagingFolderName, toPath: target)
        if let manifestPart, var manifest = ModelImportManifest.load(modelPath: modelPath) {
            manifest.parts.removeAll { $0.role == "ref2va" }
            manifest.parts.append(manifestPart)
            try manifest.save(modelPath: modelPath)
        }
    }
}

/// Whether two paths are on the same volume (a copy between them is a clone).
private func sameVolume(_ a: String, _ b: String) -> Bool {
    let key = URLResourceKey.volumeIdentifierKey
    func volume(_ path: String) -> NSObject? {
        try? URL(fileURLWithPath: path).resourceValues(forKeys: [key]).allValues[key] as? NSObject
    }
    guard let first = volume(a), let second = volume(b) else { return false }
    return first.isEqual(second)
}

struct AddRef2VAView: View {
    let model: H3ModelEntry
    var onFinished: () -> Void
    var onCancelled: () -> Void

    private enum GGUFSource: Hashable { case download, file }

    @StateObject private var downloader = ModelDownloader()
    @Environment(\.colorScheme) private var colorScheme
    @State private var plan: Ref2VAAddPlan?
    @State private var planError: String?
    // GGUF models: where the Ref2VA DiT comes from.
    @State private var ggufSource: GGUFSource = .download
    @State private var repo = ImportDownloadKind.ref2va.savedRepo
    @State private var choices: [ImportDownloadChoice] = []
    @State private var choiceID: String?
    @State private var listing: (done: Int, total: Int)?
    @State private var listError: String?
    @State private var localPick: ImportPick?
    @State private var assembling = false
    @State private var finished = false
    @State private var errorMessage: String?

    private var palette: H3Palette { H3Palette(colorScheme) }
    private var isGGUF: Bool { isGGUFModel(model) }
    private var staging: String { model.path + "/" + ref2vaStagingFolderName }
    private var isBusy: Bool { downloader.state == .downloading || assembling }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("「\(model.name)」にRef2VAを追加").font(.system(size: 16, weight: .semibold))
            Text(isGGUF
                 ? String(localized: "参照画像・動画・音声を使う生成に必要なRef2VAを、あとから加えます。Ref2VAのDiT（GGUF）を用意し、ほかの部品（文章の部分・VAEなど）はこのモデルのFL2VAから複製します。同じディスクなので、複製で容量は増えません。")
                 : String(localized: "参照画像・動画・音声を使う生成に必要なRef2VAを、あとから加えます。公式のモデル（MiniMaxAI/MiniMax-H3）から、FL2VAと違うDiTの重みと設定だけをダウンロードし、残り（文章の部分・VAEなど）はこのモデルのFL2VAから複製します。同じディスクなので、複製で容量は増えません。"))
                .font(.caption).foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: H3Spacing.md) {
                    if isGGUF { ggufSection } else { officialSection }
                    ModelLicenseNotice()
                    statusSection
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            footer
        }
        .padding(H3Spacing.lg)
        .frame(width: 640, height: 600)
        .background(palette.canvas)
        .task { await prepare() }
        .onDisappear { downloader.cancel() }
        .onChange(of: downloader.state) { _, state in
            if state == .completed { assemble() }
        }
    }

    // MARK: Official model

    @ViewBuilder private var officialSection: some View {
        if let plan {
            Text("ダウンロード：約\(size(plan.downloadBytes))（\(plan.downloads.count)ファイル）・FL2VAから複製：約\(size(plan.cloneBytes))（\(plan.clones.count)ファイル）")
                .font(.caption)
                .fixedSize(horizontal: false, vertical: true)
            spaceLine
        } else if let planError {
            Text(planError).font(.caption).foregroundStyle(palette.errorColor)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            HStack {
                ProgressView().controlSize(.small)
                Text("ファイル一覧を確認しています…").font(.caption).foregroundStyle(palette.textSecondary)
            }
        }
    }

    // MARK: GGUF model

    @ViewBuilder private var ggufSection: some View {
        Picker("Ref2VAのDiT", selection: $ggufSource) {
            Text("Hugging Faceからダウンロード").tag(GGUFSource.download)
            Text("手元のファイルを使う").tag(GGUFSource.file)
        }
        .pickerStyle(.segmented)
        .disabled(isBusy)
        switch ggufSource {
        case .download:
            Text("配布元（Hugging Face）").font(.caption).foregroundStyle(palette.textSecondary)
            RepoSourceField(kind: .ref2va, repo: $repo, busy: isBusy || listing != nil) {
                Task { await loadChoices() }
            }
            if let listing {
                HStack {
                    ProgressView().controlSize(.small)
                    Text(listing.total > 0
                         ? String(localized: "ファイルの先頭を読んでいます（\(listing.done)/\(listing.total)）…")
                         : String(localized: "ファイル一覧を確認しています…"))
                        .font(.caption).foregroundStyle(palette.textSecondary)
                }
            } else if let listError {
                Text(listError).font(.caption).foregroundStyle(palette.errorColor)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Picker(String(localized: "ファイル"), selection: Binding(get: { choice?.id ?? "" }, set: { id in
                    // An unusable file stays listed but can't be picked.
                    if choices.first(where: { $0.id == id })?.unsupported == nil { choiceID = id }
                })) {
                    ForEach(choices) { item in
                        Text(verbatim: item.label)
                            .foregroundStyle(item.unsupported == nil ? palette.textPrimary : palette.textSecondary)
                            .tag(item.id)
                    }
                }
                .pickerStyle(.radioGroup)
                .disabled(isBusy)
                Text("DiTのファイルのうち、名前に ref2va を含むものを並べています。このモデルのFL2VAと同じ形式（量子化）を選ぶと、画質がそろいます。")
                    .font(.caption).foregroundStyle(palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        case .file:
            HStack {
                Button("GGUFファイルを選ぶ…") { pickFile() }.disabled(isBusy)
                if let localPick {
                    Text(verbatim: localPick.fileName).font(.caption)
                        .lineLimit(1).truncationMode(.middle)
                }
            }
            if let note = localPickNote {
                Text(note.text).font(.caption)
                    .foregroundStyle(note.isError ? palette.errorColor : palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        if plan != nil { spaceLine }
        if let planError {
            Text(planError).font(.caption).foregroundStyle(palette.errorColor)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// The picked Ref2VA download; unusable files are never picked.
    private var choice: ImportDownloadChoice? {
        let usable = choices.filter { $0.unsupported == nil }
        return usable.first { $0.id == choiceID } ?? usable.first { $0.verified } ?? usable.first
    }

    /// What the picked local file is, and whether it can be used.
    private var localPickNote: (text: String, isError: Bool)? {
        guard let localPick else { return nil }
        guard let info = localPick.info else {
            return (String(localized: "読み込めません（\(localPick.error ?? "")）"), true)
        }
        guard info.role == .dit else { return (String(localized: "DiTのファイルではありません"), true) }
        var parts = [info.mainType, size(Int64(info.fileBytes)),
                     info.curveAdaLN ? String(localized: "AdaLNは曲線形") : String(localized: "AdaLNは密な形")]
        if isPlainQ2KDiT(mainType: info.mainType, fileName: localPick.fileName) {
            parts.append(String(localized: "画質が大きく崩れます（Q2_K）"))
        }
        var text = parts.joined(separator: itemSeparator)
        if !localPick.looksLikeRef2VA {
            text += "\n" + String(localized: "名前に ref2va を含みません。FL2VAとRef2VAのDiTは中身の形が同じなので、取り違えても読み込めてしまいます。Ref2VAのファイルか確かめてください。")
        }
        return (text, false)
    }

    private var usableLocalPick: ImportPick? {
        guard let localPick, localPick.info?.role == .dit else { return nil }
        return localPick
    }

    private func pickFile() {
        guard let path = chooseFile(allowedContentTypes: ggufTypes) else { return }
        localPick = ImportPick(path: path)
    }

    private func loadChoices() async {
        let name = repo.trimmingCharacters(in: .whitespaces)
        listing = (0, 0)
        listError = nil
        defer { listing = nil }
        do {
            choices = try await ImportDownloadChoice.list(.ref2va, repo: name) { done, total in
                Task { @MainActor in if listing != nil { listing = (done, total) } }
            }
            choiceID = nil
            ImportDownloadKind.ref2va.savedRepo = name
            if choices.isEmpty {
                listError = String(localized: "「\(name)」には、この部品のファイルが見つかりませんでした。")
            } else if choices.allSatisfy({ $0.unsupported != nil }) {
                listError = String(localized: "「\(name)」のこの部品のファイルは、どれもこのアプリでは使えません。")
            }
        } catch {
            choices = []
            listError = ModelDownloadError.listingFailed(error.localizedDescription).errorDescription
        }
    }

    // MARK: Space and status

    /// Everything to download: the plan's files and, for a GGUF model, the
    /// Ref2VA DiT chosen on Hugging Face.
    private var downloads: [HFFileEntry] {
        var entries = plan?.downloads ?? []
        if isGGUF, ggufSource == .download, let entry = choice?.entries.first {
            var placed = entry
            placed.localPath = ref2vaStagingFolderName + "/transformer/model.gguf"
            entries.append(placed)
        }
        return entries
    }

    /// Bytes still to download, plus a picked file copied from another disk.
    private var neededBytes: Int64 {
        var bytes = downloads.reduce(0) { $0 + $1.size } - ModelDownloader.presentBytes(downloads, at: model.path)
        if isGGUF, ggufSource == .file, let pick = usableLocalPick, !sameVolume(pick.path, model.path) {
            bytes += Int64(pick.info?.fileBytes ?? 0)
        }
        return bytes
    }

    private var space: (text: String, fits: Bool) {
        downloadSpaceCheck(needed: neededBytes, destination: model.path)
    }

    private var spaceLine: some View {
        Text(space.text).font(.caption)
            .foregroundStyle(space.fits ? palette.textSecondary : palette.errorColor)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var ready: Bool {
        guard plan != nil, space.fits else { return false }
        guard isGGUF else { return true }
        switch ggufSource {
        case .download: return choice != nil && listing == nil
        case .file: return usableLocalPick != nil
        }
    }

    @ViewBuilder private var statusSection: some View {
        switch downloader.state {
        case .downloading:
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: Double(downloader.downloadedBytes), total: Double(max(downloader.totalBytes, 1)))
                Text(verbatim: "\(size(downloader.downloadedBytes)) / \(size(downloader.totalBytes))")
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
        if assembling {
            HStack {
                ProgressView().controlSize(.small)
                Text("Ref2VAを置いています…").font(.caption).foregroundStyle(palette.textSecondary)
            }
        } else if finished {
            Label("Ref2VAを追加しました", systemImage: "checkmark.circle.fill")
                .font(.caption).foregroundStyle(.green)
            Text("M5の高速モードで参照画像・動画・音声を使うときは、詳細設定の「キャッシュを作成…」でRef2VA用のint8キャッシュを作ってください。")
                .font(.caption).foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        if let errorMessage {
            Text(errorMessage).font(.caption).foregroundStyle(palette.errorColor)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var footer: some View {
        HStack {
            Button("閉じる") { downloader.cancel(); onCancelled() }
                .disabled(assembling)
            Spacer()
            if finished {
                Button("完了") { onFinished() }.buttonStyle(.borderedProminent)
            } else if downloader.state == .downloading {
                Button("中止") { downloader.cancel() }
            } else if case .failed = downloader.state {
                Button("続きからダウンロード") { start() }
                    .buttonStyle(.borderedProminent)
                    .disabled(!ready)
            } else {
                Button(neededBytes > 0
                       ? String(localized: "追加する（約\(size(neededBytes))）")
                       : String(localized: "追加する")) { start() }
                    .buttonStyle(.borderedProminent)
                    .disabled(!ready || assembling)
            }
        }
        .controlSize(.large)
    }

    // MARK: Flow

    private func prepare() async {
        do {
            plan = isGGUF ? try await .gguf(modelPath: model.path) : try await .official(modelPath: model.path)
        } catch {
            planError = ModelDownloadError.listingFailed(error.localizedDescription).errorDescription
        }
        if isGGUF { await loadChoices() }
    }

    private func start() {
        errorMessage = nil
        do {
            try FileManager.default.createDirectory(atPath: staging, withIntermediateDirectories: true)
        } catch {
            errorMessage = String(localized: "Ref2VAを追加できませんでした（詳細: \(error.localizedDescription)）")
            return
        }
        let entries = downloads
        if entries.isEmpty {
            assemble()
        } else {
            downloader.use(entries)
            downloader.startDownload(destination: model.path)
        }
    }

    /// Downloads are in place: clone the rest and make it Ref2VA.
    private func assemble() {
        guard let plan, !assembling else { return }
        var clones = plan.clones
        var part: ModelImportManifest.Part?
        if isGGUF {
            let target = staging + "/transformer/model.gguf"
            switch ggufSource {
            case .file:
                guard let pick = usableLocalPick,
                      let local = Ref2VAAddPlan.localDiT(pick, modelPath: model.path) else { return }
                clones.append(local.clone)
                part = local.part
            case .download:
                guard let entry = choice?.entries.first else { return }
                let picked = ImportPick(path: target)
                part = .init(role: "ref2va", file: "Ref2VA/transformer/model.gguf",
                             source: (entry.path as NSString).lastPathComponent,
                             type: picked.info?.mainType ?? "", bytes: UInt64(entry.size))
            }
        }
        assembling = true
        let path = model.path
        let manifestPart = part
        Task {
            let result = await Task.detached(priority: .userInitiated) {
                Result { try Ref2VAAddPlan.finish(modelPath: path, clones: clones, manifestPart: manifestPart) }
            }.value
            assembling = false
            switch result {
            case .success: finished = true
            case .failure(let error):
                NSLog("H3cApp: adding Ref2VA failed: %@", error.localizedDescription)
                errorMessage = String(localized: "Ref2VAを追加できませんでした（詳細: \(error.localizedDescription)）")
            }
        }
    }

    private func size(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}
