import SwiftUI

/// First-run (and "モデル管理" > "ダウンロードして追加…") setup flow. The user
/// picks what to download first:
/// - GGUF: quantized DiT and text encoder plus only the official model's
///   shared parts (tokenizer, configs, VAEs), downloaded together and then
///   imported as a model folder (ModelImporter) - a fraction of the size.
/// - The official model: MiniMax-H3's real file list from Hugging Face,
///   FL2VA (and optionally Ref2VA), downloaded with ModelDownloader.
/// Sizes are fetched as soon as a choice is shown, so the one button reads
/// what it will do and how much. An official download hands its folder back
/// via `onCompleted` (the caller decides whether that updates an entry or
/// adds one); a GGUF import hands back the imported folder via `onImported`.
struct ModelDownloadWizardView: View {
    enum Source: Hashable { case gguf, official }

    let onCompleted: (String) -> Void
    let onCancelled: () -> Void
    /// Needed for the GGUF route (the import registers nothing itself; the
    /// caller does, through onImported).
    var library: ModelLibrary? = nil
    var onImported: ((_ path: String, _ use: Bool) -> Void)? = nil

    @StateObject private var downloader = ModelDownloader()
    @StateObject private var importer = ModelImporter()
    @Environment(\.colorScheme) private var colorScheme
    @State private var destination: String
    @State private var includeRef2VA = false
    @State private var source: Source
    @State private var showingImport = false

    // GGUF route
    @State private var choices: [ImportDownloadKind: [ImportDownloadChoice]] = [:]
    @State private var picked: [ImportDownloadKind: String] = [:]
    @State private var listError: String?
    @State private var trashDownloads = true
    @State private var importError: String?
    @State private var importedPath: String?

    init(suggestedDestination: String, onCompleted: @escaping (String) -> Void, onCancelled: @escaping () -> Void,
         library: ModelLibrary? = nil, onImported: ((_ path: String, _ use: Bool) -> Void)? = nil) {
        self.onCompleted = onCompleted
        self.onCancelled = onCancelled
        self.library = library
        self.onImported = onImported
        _destination = State(initialValue: suggestedDestination)
        _source = State(initialValue: library != nil ? .gguf : .official)
    }

    private var palette: H3Palette { H3Palette(colorScheme) }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("MiniMax-H3 モデルの用意")
                .font(.system(size: 18, weight: .semibold))

            existingFolderRow

            Divider()

            if library != nil {
                Picker("ダウンロードするもの", selection: $source) {
                    Text("GGUF（量子化）と必要な部品").tag(Source.gguf)
                    Text("公式のモデル（BF16）").tag(Source.official)
                }
                .pickerStyle(.segmented)
                .disabled(isBusy)
            }

            ScrollView {
                VStack(alignment: .leading, spacing: H3Spacing.md) {
                    switch source {
                    case .gguf: ggufSection
                    case .official: officialSection
                    }
                    statusSection
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            footer
        }
        .padding(H3Spacing.xl)
        .frame(width: 580, height: 600)
        .background(palette.canvas)
        .onDisappear { downloader.cancel(); importer.cancel() }
        .task(id: source) { await prepare() }
        .onChange(of: includeRef2VA) { _, _ in
            if source == .official { downloader.listFiles(includeRef2VA: includeRef2VA) }
        }
        .onChange(of: downloader.state) { _, state in
            if state == .completed && source == .gguf { importDownloaded() }
        }
        .sheet(isPresented: $showingImport) {
            if let library {
                ModelImportView(library: library, onImported: { path, use in
                    showingImport = false
                    onImported?(path, use)
                }, onCancelled: { showingImport = false })
            }
        }
    }

    private var isBusy: Bool {
        downloader.state == .downloading || downloader.state == .listing || importer.isRunning
    }

    /// Lets a user who already has the model on disk (copied from another
    /// Mac, downloaded outside this app, or GGUF files at hand) register it
    /// without downloading again.
    private var existingFolderRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("FL2VA（と、あればRef2VA）を含むフォルダが既にある場合は、ダウンロードせずに登録できます。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                Spacer()
                Button("フォルダを追加…") { addExistingFolder() }
                    .disabled(isBusy)
            }
            if library != nil {
                HStack {
                    Text("GGUFファイルが既に手元にある場合は、取り込みウィザードで取り込めます。")
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                    Spacer()
                    Button("GGUFを取り込む…") { showingImport = true }
                        .disabled(isBusy)
                }
            }
        }
    }

    private func addExistingFolder() {
        guard let path = chooseDirectory() else { return }
        onCompleted(path)
    }

    // MARK: Official model

    private var officialSection: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("Hugging Face（MiniMaxAI/MiniMax-H3）から公式のモデルをそのままダウンロードします。FL2VAだけで約130GB、Ref2VAも含めると約260GBと非常に大きいので、時間と空き容量に余裕がある回線・ディスクで実行してください。")
                .font(.caption)
                .foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            VStack(alignment: .leading, spacing: 4) {
                Text("保存先").font(.caption).foregroundStyle(palette.textSecondary)
                HStack {
                    Text(destination)
                        .font(.caption)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Button("変更…") {
                        if let path = chooseDirectory() { destination = path + "/MiniMax-H3" }
                    }
                    .disabled(isBusy)
                }
            }
            Toggle("Ref2VA も含める（参照画像・動画機能。おおよそ倍のサイズになります）", isOn: $includeRef2VA)
                .disabled(isBusy)
        }
    }

    // MARK: GGUF

    private func choice(_ kind: ImportDownloadKind) -> ImportDownloadChoice? {
        let list = choices[kind] ?? []
        return list.first { $0.id == picked[kind] } ?? list.first { $0.verified } ?? list.first
    }

    /// The text encoder lacks the vision tower: the mmproj file is added.
    private var needsVision: Bool { choice(.textEncoder)?.includesVision == false }

    private var ggufDownloads: [(ImportDownloadKind, ImportDownloadChoice)] {
        var kinds: [ImportDownloadKind] = [.dit, .textEncoder, .shared]
        if needsVision { kinds.append(.vision) }
        if includeRef2VA { kinds.append(.ref2va) }
        return kinds.compactMap { kind in choice(kind).map { (kind, $0) } }
    }

    private var ggufBytes: Int64 { ggufDownloads.reduce(0) { $0 + $1.1.bytes } }
    private var ggufWrites: Int64 { ggufDownloads.reduce(0) { $0 + $1.1.importWrites } }

    @ViewBuilder private var ggufSection: some View {
        Text("量子化されたDiTと文章の部分（GGUF）に、公式のモデルからトークナイザー・VAE・設定だけを加えてダウンロードし、モデルとして取り込みます。公式のモデルより大幅に小さくなります。")
            .font(.caption)
            .foregroundStyle(palette.textSecondary)
            .fixedSize(horizontal: false, vertical: true)
        if let listError {
            Text(listError).font(.caption).foregroundStyle(palette.errorColor)
        } else if choices.count < ImportDownloadKind.allCases.count {
            HStack {
                ProgressView().controlSize(.small)
                Text("ファイル一覧を確認しています…").font(.caption).foregroundStyle(palette.textSecondary)
            }
        } else {
            Grid(alignment: .leading, horizontalSpacing: H3Spacing.md, verticalSpacing: 8) {
                choiceRow(.dit, String(localized: "DiT（FL2VA）"))
                choiceRow(.textEncoder, String(localized: "文章の部分"))
                GridRow {
                    Text("画像の部分").font(.caption)
                    Text(needsVision
                         ? String(localized: "mmproj-Qwen3VL-32B-Instruct-F16（\(size(choice(.vision)?.bytes ?? 0))）を加えます")
                         : String(localized: "文章の部分のファイルに含まれています"))
                        .font(.caption).foregroundStyle(palette.textSecondary)
                }
                choiceRow(.shared, String(localized: "トークナイザー・VAE・設定"))
                GridRow {
                    Toggle("Ref2VA も含める", isOn: $includeRef2VA)
                        .font(.caption)
                        .disabled(isBusy)
                    if includeRef2VA {
                        picker(.ref2va)
                    } else {
                        Text("参照画像・動画・音声を使う生成に必要です")
                            .font(.caption).foregroundStyle(palette.textSecondary)
                    }
                }
            }
            Text(ggufNote).font(.caption2).foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            Toggle("取り込んだあと、ダウンロードしたGGUFをゴミ箱に入れる（取り込んだモデルは自分のコピーを持っています）",
                   isOn: $trashDownloads)
                .font(.caption)
                .disabled(isBusy)
            Text(spaceLine).font(.caption)
                .foregroundStyle(hasRoom ? palette.textSecondary : palette.errorColor)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func choiceRow(_ kind: ImportDownloadKind, _ title: String) -> some View {
        GridRow {
            Text(title).font(.caption)
            picker(kind)
        }
    }

    private func picker(_ kind: ImportDownloadKind) -> some View {
        Picker("", selection: Binding(get: { choice(kind)?.id ?? "" }, set: { picked[kind] = $0 })) {
            ForEach(choices[kind] ?? []) { item in
                Text(verbatim: "\(item.title)・\(size(item.bytes))" +
                     (item.verified ? String(localized: "（動作確認済み）") : ""))
                    .tag(item.id)
            }
        }
        .labelsHidden()
        .frame(maxWidth: 360, alignment: .leading)
        .disabled(isBusy)
    }

    private var ggufNote: String {
        String(localized: "「動作確認済み」以外も読み込めることは確かめていますが、画質はまだ確かめていません。小さい形式ほど容量は減り、画質は落ちます。Qwen公式の文章の部分は、取り込むときに使う50層だけに切り詰めて書き出します。")
    }

    private var freeBytes: Int64 { availableBytes(at: h3AppSupportDirectory) ?? 0 }
    private var neededBytes: Int64 { ggufBytes + ggufWrites + (2 << 30) }
    private var hasRoom: Bool { freeBytes >= neededBytes }

    private var spaceLine: String {
        var text = String(localized: "ダウンロード：約\(size(ggufBytes))")
        if ggufWrites > 0 {
            text += String(localized: "・取り込みで書き出す量：約\(size(ggufWrites))")
        }
        text += String(localized: "・空き容量：\(size(freeBytes))")
        if !hasRoom {
            text += "\n" + String(localized: "空き容量が足りません（余裕として2GBを残します）。小さい形式を選ぶか、ディスクを空けてください。")
        }
        return text
    }

    private func size(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    // MARK: Flow

    private func prepare() async {
        switch source {
        case .official:
            if downloader.state == .idle || downloader.state == .ready || downloader.state == .completed {
                downloader.listFiles(includeRef2VA: includeRef2VA)
            }
        case .gguf:
            if downloader.state != .downloading { downloader.reset() }
            guard choices.count < ImportDownloadKind.allCases.count else { return }
            listError = nil
            do {
                for kind in ImportDownloadKind.allCases where choices[kind] == nil {
                    choices[kind] = try await ImportDownloadChoice.list(kind)
                }
            } catch {
                listError = ModelDownloadError.listingFailed(error.localizedDescription).errorDescription
            }
        }
    }

    private func startGGUFDownload() {
        importError = nil
        try? FileManager.default.removeItem(atPath: importBaseDirectory + "/.h3c-download-complete")
        let entries = ggufDownloads.flatMap { kind, choice in
            choice.entries.map {
                $0.placed(under: kind == .shared ? "ImportBase/MiniMax-H3" : "Downloads")
            }
        }
        downloader.use(entries)
        downloader.startDownload(destination: h3AppSupportDirectory)
    }

    /// The downloads are in place: mark the shared parts complete and import.
    private func importDownloaded() {
        FileManager.default.createFile(atPath: importBaseDirectory + "/.h3c-download-complete", contents: Data())
        guard let base = downloadedImportBase() else {
            importError = String(localized: "ダウンロードした共通部品がそろっていません。")
            return
        }
        func file(_ kind: ImportDownloadKind) -> String? {
            choice(kind)?.entries.first.map { importDownloadsDirectory + "/" + $0.local }
        }
        var paths = [file(.dit), file(.textEncoder)].compactMap { $0 }
        if needsVision, let vision = file(.vision) { paths.append(vision) }
        var ref2va: Set<String> = []
        if includeRef2VA, let ref = file(.ref2va) {
            paths.append(ref)
            ref2va.insert(ref)
        }
        let plan = ImportPlan(files: paths.map(ImportPick.init(path:)), ref2vaPaths: ref2va, base: base,
                              availableBytes: availableBytes(at: importedModelsDirectory))
        guard let dit = plan.dit, plan.canImport else {
            importError = (plan.problems + plan.items.filter { $0.status == .missing }.map(\.title))
                .joined(separator: "\n")
            NSLog("H3cApp: downloaded GGUFs not imported: %@", importError ?? "")
            return
        }
        let trash = trashDownloads
        importer.start(.init(name: "MiniMax-H3 GGUF", dit: dit, ref2vaDit: plan.ref2vaDit,
                             textEncoder: plan.textEncoder, vision: plan.vision, base: base)) { path in
            if trash {
                for source in paths {
                    try? FileManager.default.trashItem(at: URL(fileURLWithPath: source), resultingItemURL: nil)
                }
            }
            importedPath = path
        }
    }

    @ViewBuilder
    private var statusSection: some View {
        switch downloader.state {
        case .idle:
            EmptyView()

        case .listing:
            HStack {
                ProgressView().controlSize(.small)
                Text("ファイル一覧を確認しています…").font(.caption).foregroundStyle(palette.textSecondary)
            }

        case .ready:
            if source == .official {
                Text("合計 \(formatGB(downloader.totalBytes))（\(downloader.totalFiles)ファイル）。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            }

        case .downloading:
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: Double(downloader.downloadedBytes), total: Double(max(downloader.totalBytes, 1)))
                Text([
                    "\(formatGB(downloader.downloadedBytes)) / \(formatGB(downloader.totalBytes))",
                    String(localized: "\(downloader.filesCompleted)/\(downloader.totalFiles)ファイル"),
                    formatSpeed(downloader.bytesPerSecond),
                ].joined(separator: summarySeparator))
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                Text("中断しても、次回は続きから再開します。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            }

        case .failed(let message):
            Text(message).font(.caption).foregroundStyle(palette.errorColor)
                .fixedSize(horizontal: false, vertical: true)

        case .completed:
            if source == .official {
                Text("ダウンロードが完了しました。").font(.caption).foregroundStyle(palette.textSecondary)
            }
        }
        if source == .gguf {
            if importer.isRunning {
                HStack {
                    if let fraction = importer.fraction {
                        ProgressView(value: fraction).frame(maxWidth: 200)
                    } else {
                        ProgressView().controlSize(.small)
                    }
                    Text(importer.step).font(.caption).foregroundStyle(palette.textSecondary)
                }
            } else if importedPath != nil {
                Label("ダウンロードして取り込みました", systemImage: "checkmark.circle.fill")
                    .font(.caption).foregroundStyle(.green)
                Text("M5の高速モードで使うときは、詳細設定の「キャッシュを作成…」でint8キャッシュを作ってください。")
                    .font(.caption).foregroundStyle(palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let message = importError ?? importer.errorMessage {
                Text(message).font(.caption).foregroundStyle(palette.errorColor)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var footer: some View {
        HStack {
            Button("あとで") { onCancelled() }
            Spacer()
            if downloader.state == .downloading {
                Button("中止") { downloader.cancel() }
            } else if importer.isRunning {
                Button("中止") { importer.cancel() }
            } else {
                switch source {
                case .official: officialButton
                case .gguf: ggufButton
                }
            }
        }
        .controlSize(.large)
    }

    @ViewBuilder private var officialButton: some View {
        switch downloader.state {
        case .completed:
            Button("完了") { onCompleted(destination) }
                .buttonStyle(.borderedProminent)
        case .failed:
            Button("続きからダウンロード") { downloader.startDownload(destination: destination) }
                .buttonStyle(.borderedProminent)
        default:
            Button(downloader.state == .ready
                   ? String(localized: "ダウンロード開始（\(formatGB(downloader.totalBytes))）")
                   : String(localized: "ダウンロード開始")) {
                downloader.startDownload(destination: destination)
            }
            .buttonStyle(.borderedProminent)
            .disabled(downloader.state != .ready)
        }
    }

    @ViewBuilder private var ggufButton: some View {
        if let importedPath {
            Button("完了") { onImported?(importedPath, true) }
                .buttonStyle(.borderedProminent)
        } else if downloader.state == .completed {
            Button("取り込みをやり直す") { importDownloaded() }
                .buttonStyle(.borderedProminent)
        } else {
            Button(isFailed
                   ? String(localized: "続きからダウンロード")
                   : String(localized: "ダウンロードして取り込む（\(size(ggufBytes))）")) {
                startGGUFDownload()
            }
            .buttonStyle(.borderedProminent)
            .disabled(choices.count < ImportDownloadKind.allCases.count || !hasRoom)
        }
    }

    private var isFailed: Bool {
        if case .failed = downloader.state { return true }
        return false
    }

    private func formatGB(_ bytes: Int64) -> String {
        String(format: "%.1fGB", Double(bytes) / 1_073_741_824)
    }

    private func formatSpeed(_ bytesPerSecond: Double) -> String {
        guard bytesPerSecond > 0 else { return String(localized: "計測中…") }
        return String(format: "%.1fMB/s", bytesPerSecond / 1_048_576)
    }
}
