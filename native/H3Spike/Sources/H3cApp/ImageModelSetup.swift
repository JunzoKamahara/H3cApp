import AppKit
import Combine
import H3Engine
import SwiftUI

/// The MiniMax H3 Image model as the app installs it (plans/IMAGE_PLAN.ja.md,
/// phase 3): the ComfyUI int8 + ConvRot DiT and the fp16 VAE from Comfy-Org's
/// re-publication, the VAE's config.json and the license text from MiniMax's
/// own repository - each pinned to a revision and checked by SHA-256 - and
/// the text encoder and tokenizer of a video model already on this Mac (the
/// image model uses the same ones; they are copied, which on one volume
/// takes no extra space).
///
/// Nothing is bundled. The user downloads the weights here after reading
/// what the MiniMax-H3-Image Community License Agreement asks - it is not
/// the video model's license: it shuts out whole territories, and its
/// conditions reach the pictures made with the model.
enum H3ImageModel {
    static let displayName = "MiniMax H3 Image"
    static let weightsRepo = "Comfy-Org/MiniMax-H3-Image"
    static let weightsRevision = "892de7cb61df49753bf15145a711154803a355cb"
    static let weightsRevisionDate = "2026-11-10"
    static let officialRepo = "MiniMaxAI/MiniMax-H3-Image"
    /// The revision the VAE config and the tokenizer are read from.
    static let officialRevision = "492fd8c4eca666f1085f57e30738c1e0d86b3439"
    /// The revision of the license text (it last changed on 2026-11-11).
    static let licenseRevision = "5f90473023975e6271b0ed1dd84c4e88e4bbc6d3"
    static let licenseName = "MiniMax-H3-Image Community License Agreement"
    static let licenseFile = "LICENSE"
    static let licenseSHA256 = "0ec3bb5e6d9e885fa2aaf15e84689e50fd51f42b53fa24d3a64ed9d8c76d423b"
    static let licenseURL = URL(string: "https://huggingface.co/\(officialRepo)/blob/\(licenseRevision)/LICENSE")!
    static let licenseRawURL = URL(string: "https://huggingface.co/\(officialRepo)/resolve/\(licenseRevision)/LICENSE")!
    static let officialURL = URL(string: "https://huggingface.co/\(officialRepo)")!
    static let weightsURL = URL(string: "https://huggingface.co/\(weightsRepo)/tree/\(weightsRevision)")!
    static let authorizationURL = URL(string: "https://platform.minimax.io/console/license")!

    /// A weight file: where it is in the repository, where the engine
    /// looks for it, and what it must be.
    struct Weight {
        let repoPath: String
        let localPath: String
        let bytes: Int64
        let sha256: String
    }

    static let weights = [
        Weight(repoPath: "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
               localPath: "transformer/minimax_h3_image_pruned_int8_convrot.safetensors",
               bytes: 20_979_314_802,
               sha256: "83eb140f1ff95c068f8a6886e6d3f3d0d4ab2e07bb2fe6d2ef992a51146ee622"),
        Weight(repoPath: "vae/minimax_h3_image_vae_fp16.safetensors",
               localPath: "vae/minimax_h3_image_vae_fp16.safetensors",
               bytes: 1_168_505_312,
               sha256: "d80e9c2e5cfc9844869961ff5a58c9a9e52182df6bfb30b26018dd7f348721af"),
    ]
    /// Small files of the official repository, fetched whole.
    static let vaeConfig = (repoPath: "vae/config.json", localPath: "vae/config.json",
                            sha256: "7fcc21bec549804ff0a04d8246e7f90e679d4f8d4b0a27a252bbd18c30e3e4bb")
    static let tokenizer = (repoPath: "tokenizer/tokenizer.json", localPath: "tokenizer/tokenizer.json",
                            sha256: "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7")
    static let textEncoderFolder = "text_encoder"
    static var downloadBytes: Int64 { weights.reduce(0) { $0 + $1.bytes } }

    static var directory: String { h3AppSupportDirectory + "/ImageModels/MiniMax-H3-Image" }

    static func path(_ relative: String) -> String { directory + "/" + relative }

    static func fileSize(_ relative: String) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: path(relative))[.size]) as? Int64
    }

    /// The downloaded weights are here, each at its full size.
    static var weightsPresent: Bool {
        weights.allSatisfy { fileSize($0.localPath) == $0.bytes }
    }

    /// Everything the engine needs: the folder is an image model.
    static var complete: Bool {
        weightsPresent && fileSize(vaeConfig.localPath) != nil && fileSize(tokenizer.localPath) != nil
            && !((try? FileManager.default.contentsOfDirectory(atPath: path(textEncoderFolder))) ?? []).isEmpty
    }

    private static let licenseAcceptedKey = "h3c-app.imageModelLicenseAcceptedAt"

    static var licenseAcceptedAt: Date? {
        get { UserDefaults.standard.object(forKey: licenseAcceptedKey) as? Date }
        set { UserDefaults.standard.set(newValue, forKey: licenseAcceptedKey) }
    }

    // MARK: Provenance record (the form the upscaler models use)

    static let provenanceFile = "H3IMAGE_PROVENANCE.json"

    static var provenance: SwiftVRModel.Provenance? {
        guard let data = FileManager.default.contents(atPath: path(provenanceFile)) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(SwiftVRModel.Provenance.self, from: data)
    }

    /// A record for these revisions with the expected hashes, the weights
    /// still at their recorded sizes.
    static var verified: Bool {
        guard let record = provenance, record.revision == weightsRevision else { return false }
        return weights.allSatisfy { weight in
            record.files[weight.localPath]?.sha256 == weight.sha256 && fileSize(weight.localPath) == weight.bytes
        }
    }

    enum SetupError: LocalizedError {
        case listing(file: String)
        case mismatch(file: String)
        case smallFile(String, String)
        case noTextEncoder
        case copy(String)

        var errorDescription: String? {
            switch self {
            case .listing(let file):
                return String(localized: "配布元の一覧にある「\(file)」が、記録してある版のもの（大きさ・SHA-256）と一致しません。ダウンロードは始めていません。")
            case .mismatch(let file):
                return String(localized: "「\(file)」のSHA-256が、記録してある版のものと一致しません。このファイルは削除しました。もう一度ダウンロードしてください。")
            case .smallFile(let file, let detail):
                return String(localized: "「\(file)」を取得できませんでした（詳細: \(detail)）")
            case .noTextEncoder:
                return String(localized: "画像モデルは、動画モデルと同じテキストエンコーダを使います。先に動画モデル（MiniMax-H3）を導入してから、もう一度お試しください。")
            case .copy(let detail):
                return String(localized: "テキストエンコーダを複製できませんでした（詳細: \(detail)）")
            }
        }
    }

    /// A small file of the official repository at its pinned revision,
    /// written only when its bytes are the recorded ones.
    static func fetchSmallFile(from url: URL, sha256: String, to relative: String) async throws {
        if let data = FileManager.default.contents(atPath: path(relative)), FileHash.sha256(of: data) == sha256 { return }
        let data: Data
        do {
            let (body, response) = try await URLSession.shared.data(from: url)
            guard let http = response as? HTTPURLResponse, (200 ..< 300).contains(http.statusCode) else {
                throw SetupError.smallFile(relative, "HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)")
            }
            data = body
        } catch let error as SetupError {
            throw error
        } catch {
            throw SetupError.smallFile(relative, error.localizedDescription)
        }
        guard FileHash.sha256(of: data) == sha256 else {
            throw SetupError.smallFile(relative, String(localized: "取得した内容のSHA-256が記録と一致しません"))
        }
        let destination = URL(fileURLWithPath: path(relative))
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: destination, options: .atomic)
    }

    static func officialFileURL(_ repoPath: String) -> URL {
        URL(string: "https://huggingface.co/\(officialRepo)/resolve/\(officialRevision)/\(repoPath)")!
    }

    /// The video model whose text encoder the image model can share: the
    /// first of these folders that has one.
    static func textEncoderSource(in modelPaths: [String]) -> String? {
        modelPaths.map { $0 + "/FL2VA/text_encoder" }.first { folder in
            !((try? FileManager.default.contentsOfDirectory(atPath: folder)) ?? []).isEmpty
        }
    }

    /// Copies a video model's text encoder (a clone where the volume makes
    /// one) unless one is here already. Copies files: off the main actor.
    static func shareTextEncoder(from source: String) throws {
        let destination = path(textEncoderFolder)
        if !((try? FileManager.default.contentsOfDirectory(atPath: destination)) ?? []).isEmpty { return }
        try? FileManager.default.removeItem(atPath: destination)
        let staging = destination + ".copying"
        try? FileManager.default.removeItem(atPath: staging)
        do {
            try FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
            try FileManager.default.copyItem(atPath: source, toPath: staging)
            try FileManager.default.moveItem(atPath: staging, toPath: destination)
        } catch {
            try? FileManager.default.removeItem(atPath: staging)
            throw SetupError.copy(error.localizedDescription)
        }
    }

    /// Hashes the weights against the pinned revision and writes the
    /// record. A file whose hash differs is deleted and named in the error.
    static func verify(textEncoderSource: String?, progress: @escaping @Sendable (Int64, Int64) -> Void) async throws {
        let total = downloadBytes
        var done: Int64 = 0
        var record: [String: SwiftVRModel.ProvenanceEntry] = [:]
        for weight in weights {
            let file = path(weight.localPath)
            let base = done
            let actual = try await Task.detached(priority: .userInitiated) {
                var read: Int64 = 0
                return try FileHash.sha256(ofFile: file) { bytes in
                    read += bytes
                    progress(base + read, total)
                }
            }.value
            done += weight.bytes
            progress(done, total)
            guard actual == weight.sha256 else {
                try? FileManager.default.removeItem(atPath: file)
                throw SetupError.mismatch(file: weight.localPath)
            }
            record[weight.localPath] = .init(sha256: actual, bytes: weight.bytes)
        }
        for small in [vaeConfig.localPath, tokenizer.localPath, licenseFile] {
            if let data = FileManager.default.contents(atPath: path(small)) {
                record[small] = .init(sha256: FileHash.sha256(of: data), bytes: Int64(data.count))
            }
        }
        var note = "The DiT (ComfyUI int8 + ConvRot) and the fp16 VAE are Comfy-Org's re-publication of MiniMax H3 Image; vae/config.json and tokenizer/tokenizer.json are from \(officialRepo) at \(officialRevision); LICENSE is that repository's at \(licenseRevision)."
        if let textEncoderSource {
            note += " text_encoder/ is a copy of the video model's at \(textEncoderSource) (the same Qwen3-VL language layers)."
        }
        let provenance = SwiftVRModel.Provenance(
            repo: weightsRepo, revision: weightsRevision, revisionDate: weightsRevisionDate,
            declaredLicense: licenseName,
            declaration: "Model card of this revision: license_name minimax-h3-image-community-license-agreement",
            licenseText: licenseFile, files: record, verifiedAt: Date(), note: note)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(provenance).write(to: URL(fileURLWithPath: path(provenanceFile)), options: .atomic)
    }

    static func bytesOnDisk() -> Int64 {
        guard let enumerator = FileManager.default.enumerator(atPath: directory) else { return 0 }
        var total: Int64 = 0
        for case let item as String in enumerator {
            total += (try? FileManager.default.attributesOfItem(atPath: directory + "/" + item)[.size] as? Int64) ?? 0
        }
        return total
    }

    static func removeAll() throws {
        try FileManager.default.removeItem(atPath: directory)
    }
}

/// Downloads and checks the image model: one for the whole app, so a
/// download keeps going when the sheet showing it closes.
@MainActor
final class H3ImageModelSetup: ObservableObject {
    static let shared = H3ImageModelSetup()

    enum Phase: Equatable {
        case idle
        case listing
        case downloading
        case sharing
        case verifying(done: Int64, total: Int64)
        case failed(String)
        case done

        var apiText: String {
            switch self {
            case .idle: return "idle"
            case .listing: return "listing"
            case .downloading: return "downloading"
            case .sharing: return "copying the text encoder"
            case .verifying: return "verifying"
            case .failed(let message): return "failed: \(message)"
            case .done: return "done"
            }
        }
    }

    @Published private(set) var phase: Phase = .idle
    let downloader = ModelDownloader()
    private var task: Task<Void, Never>?

    var isBusy: Bool {
        switch phase {
        case .listing, .downloading, .sharing, .verifying: return true
        default: return false
        }
    }

    /// `videoModelPaths`: the registered video models, the one in use
    /// first - where a text encoder to share is looked for.
    func start(videoModelPaths: [String]) {
        guard !isBusy else { return }
        H3ImageModel.licenseAcceptedAt = H3ImageModel.licenseAcceptedAt ?? Date()
        phase = .listing
        task = Task { await run(videoModelPaths: videoModelPaths) }
    }

    func cancel() {
        task?.cancel()
        downloader.cancel()
        if isBusy { phase = .idle }
    }

    private func run(videoModelPaths: [String]) async {
        do {
            // Before 22 GB come down: what the model cannot do without.
            let hasTextEncoder = !((try? FileManager.default.contentsOfDirectory(
                atPath: H3ImageModel.path(H3ImageModel.textEncoderFolder))) ?? []).isEmpty
            let source = H3ImageModel.textEncoderSource(in: videoModelPaths)
            guard hasTextEncoder || source != nil else { throw H3ImageModel.SetupError.noTextEncoder }
            var fetched = false
            if !H3ImageModel.weightsPresent { fetched = try await download() }
            try Task.checkCancellation()
            try await H3ImageModel.fetchSmallFile(
                from: H3ImageModel.officialFileURL(H3ImageModel.vaeConfig.repoPath),
                sha256: H3ImageModel.vaeConfig.sha256, to: H3ImageModel.vaeConfig.localPath)
            try await H3ImageModel.fetchSmallFile(
                from: H3ImageModel.officialFileURL(H3ImageModel.tokenizer.repoPath),
                sha256: H3ImageModel.tokenizer.sha256, to: H3ImageModel.tokenizer.localPath)
            try await H3ImageModel.fetchSmallFile(
                from: H3ImageModel.licenseRawURL, sha256: H3ImageModel.licenseSHA256, to: H3ImageModel.licenseFile)
            try Task.checkCancellation()
            if !hasTextEncoder, let source {
                phase = .sharing
                try await Task.detached(priority: .userInitiated) {
                    try H3ImageModel.shareTextEncoder(from: source)
                }.value
            }
            try Task.checkCancellation()
            if fetched || !H3ImageModel.verified {
                phase = .verifying(done: 0, total: 0)
                try await H3ImageModel.verify(textEncoderSource: source) { done, total in
                    Task { @MainActor in
                        if case .verifying = self.phase { self.phase = .verifying(done: done, total: total) }
                    }
                }
            }
            try Task.checkCancellation()
            // Listed for every window, whether or not the sheet is open.
            if H3ImageModel.complete {
                ModelLibrary.registerImageModel(path: H3ImageModel.directory, name: H3ImageModel.displayName)
            }
            phase = .done
        } catch is CancellationError {
            phase = .idle
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    /// Returns whether any file was fetched or resumed.
    private func download() async throws -> Bool {
        var entries: [HFFileEntry] = []
        for folder in Set(H3ImageModel.weights.map { ($0.repoPath as NSString).deletingLastPathComponent }) {
            entries += try await ModelDownloader.fetchEntries(
                repo: H3ImageModel.weightsRepo, subpath: folder, revision: H3ImageModel.weightsRevision)
        }
        var wanted: [HFFileEntry] = []
        for weight in H3ImageModel.weights {
            guard var entry = entries.first(where: { $0.path == weight.repoPath }),
                  entry.size == weight.bytes, entry.sha256 == nil || entry.sha256 == weight.sha256 else {
                throw H3ImageModel.SetupError.listing(file: weight.repoPath)
            }
            entry.localPath = weight.localPath
            wanted.append(entry)
        }
        try Task.checkCancellation()
        let fetching = wanted.filter { H3ImageModel.fileSize($0.local) != $0.size }
        try SwiftVRModel.forgetProvenance(of: fetching.map(\.local),
                                          recordPath: H3ImageModel.path(H3ImageModel.provenanceFile))
        downloader.use(wanted)
        downloader.startDownload(destination: H3ImageModel.directory)
        phase = .downloading
        for await state in downloader.$state.values {
            switch state {
            case .completed: return !fetching.isEmpty
            case .failed(let message): throw ModelDownloadError.listingFailed(message)
            case .ready, .idle:
                if phase == .downloading { throw CancellationError() }
                return !fetching.isEmpty
            default: continue
            }
        }
        return !fetching.isEmpty
    }
}

/// 画像モデルを準備: what the license asks, the download, the check against
/// the pinned revisions, and what is installed.
struct H3ImageModelSetupView: View {
    @ObservedObject var library: ModelLibrary
    /// An image is being generated: the model's files stay.
    var busy = false
    var close: () -> Void

    @ObservedObject private var setup = H3ImageModelSetup.shared
    @ObservedObject private var downloader = H3ImageModelSetup.shared.downloader
    @Environment(\.colorScheme) private var colorScheme
    @State private var agreed = H3ImageModel.licenseAcceptedAt != nil
    @State private var confirmingRemoval = false
    @State private var removalError: String?
    @State private var version = 0

    private var palette: H3Palette { H3Palette(colorScheme) }
    private var installed: Bool { _ = version; _ = setup.phase; return H3ImageModel.complete }
    private var verified: Bool { _ = version; _ = setup.phase; return H3ImageModel.verified }
    private var registered: Bool {
        library.imageModels.contains { $0.path == H3ImageModel.directory }
    }
    /// The video models a text encoder can be shared from, the active first.
    private var videoModelPaths: [String] {
        ([library.activeModel].compactMap { $0 } + library.models).map(\.path)
    }

    private func gigabytes(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("画像モデルを準備").font(.headline)
            Text("静止画をつくるモデル「MiniMax H3 Image」の重みを、Hugging Faceの\(H3ImageModel.weightsRepo)（ComfyUI用のint8版、revision \(String(H3ImageModel.weightsRevision.prefix(8)))、\(H3ImageModel.weightsRevisionDate)）からダウンロードし、各ファイルのSHA-256をその版の記録と照合します。約\(gigabytes(H3ImageModel.downloadBytes))が「\(H3ImageModel.directory)」に置かれます。テキストエンコーダは、導入済みの動画モデルのものを複製して使います（同じディスクなら容量は増えません）。")
                .font(.caption).foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            licenseNotice
            Toggle(isOn: $agreed) {
                Text("上の説明とライセンスの本文を読みました。対象外の地域では使わず、その条件に従って使います")
                    .font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .disabled(setup.isBusy)
            Divider()
            statusRows
            progressView
            if let removalError {
                Text(removalError).font(.caption).foregroundStyle(palette.errorColor)
            }
            Spacer(minLength: 0)
            footer
        }
        .padding(H3Spacing.lg)
        .frame(width: 640, height: 640)
        .background(palette.canvas)
        .onChange(of: setup.phase) { _, _ in version += 1 }
        .alert("画像モデルを削除", isPresented: $confirmingRemoval) {
            Button("削除", role: .destructive) { remove() }
            Button("キャンセル", role: .cancel) {}
        } message: {
            Text("ダウンロードした画像モデル（計\(gigabytes(H3ImageModel.bytesOnDisk()))）を削除し、一覧からも外します。また使うには、もう一度ダウンロードが必要です。動画モデルは変わりません。")
        }
    }

    private var licenseNotice: some View {
        HStack(alignment: .top, spacing: H3Spacing.sm) {
            Image(systemName: "doc.text.magnifyingglass").foregroundStyle(palette.accent)
            VStack(alignment: .leading, spacing: 4) {
                Text("MiniMax H3 Imageには「\(H3ImageModel.licenseName)」が適用されます。動画モデル（MiniMax H3）とは別の契約です。おもな条件は次のとおりです。")
                    .font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
                VStack(alignment: .leading, spacing: 2) {
                    Text("・EU、英国、韓国、米国では、モデルも、モデルでつくった画像も使えません（日本は対象地域に含まれます）。")
                    Text("・年間売上が100万米ドルを超える場合と、第三者にホスティングサービスとして提供する場合は、MiniMaxの事前の書面許諾が必要です。")
                    Text("・商用の製品やサービスで使う場合は、画面などに「MiniMax-H3-Image」を表示します。")
                    Text("・つくった画像を公開するときは、機械が生成したものであることを明示します。")
                    Text("・ほかのAIモデルの学習や改良には使えません。禁止されている用途（付属書A）があります。")
                }
                .font(.caption)
                .fixedSize(horizontal: false, vertical: true)
                Text("これは要約で、契約の全文ではありません。あなたの利用にどの条件が当てはまるかは、本文を読んで、利用者自身（必要なら専門家）が判断してください。アプリは本文を重みの隣に保存します。")
                    .font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: H3Spacing.md) {
                    Link("ライセンスの本文（その版）", destination: H3ImageModel.licenseURL)
                    Link("許諾の申請先（MiniMax）", destination: H3ImageModel.authorizationURL)
                }
                .font(.caption)
                HStack(spacing: H3Spacing.md) {
                    Link("元のモデル（MiniMax）", destination: H3ImageModel.officialURL)
                    Link("配布元（Hugging Face）", destination: H3ImageModel.weightsURL)
                }
                .font(.caption)
            }
        }
        .padding(H3Spacing.sm)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(palette.accent.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
    }

    private var verificationText: String {
        if verified, let record = H3ImageModel.provenance {
            return String(localized: "SHA-256がその版の記録と一致（\(record.verifiedAt.formatted(date: .numeric, time: .shortened))に確認）")
        }
        if H3ImageModel.weightsPresent {
            return String(localized: "まだ照合していません（「取得元と照合する」で確かめられます）")
        }
        return String(localized: "まだありません")
    }

    private var statusRows: some View {
        Grid(alignment: .leading, horizontalSpacing: H3Spacing.md, verticalSpacing: 6) {
            GridRow {
                Text("重み").font(.caption).foregroundStyle(palette.textSecondary)
                Text(H3ImageModel.weightsPresent ? String(localized: "ダウンロード済み") : String(localized: "まだありません")).font(.caption)
            }
            GridRow {
                Text("照合").font(.caption).foregroundStyle(palette.textSecondary)
                Text(verificationText).font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
            GridRow {
                Text("テキストエンコーダ").font(.caption).foregroundStyle(palette.textSecondary)
                Text(textEncoderText).font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
            GridRow {
                Text("使用容量").font(.caption).foregroundStyle(palette.textSecondary)
                Text("\(gigabytes({ _ = version; return H3ImageModel.bytesOnDisk() }()))（複製したテキストエンコーダを含みます。同じディスクなら、実際に増えるのはダウンロードした分だけです）")
                    .font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let accepted = H3ImageModel.licenseAcceptedAt {
                GridRow {
                    Text("ライセンス").font(.caption).foregroundStyle(palette.textSecondary)
                    Text("\(accepted.formatted(date: .numeric, time: .shortened))に説明を確認").font(.caption)
                }
            }
        }
    }

    private var textEncoderText: String {
        _ = version; _ = setup.phase
        if !((try? FileManager.default.contentsOfDirectory(atPath: H3ImageModel.path(H3ImageModel.textEncoderFolder))) ?? []).isEmpty {
            return String(localized: "複製済み")
        }
        if H3ImageModel.textEncoderSource(in: videoModelPaths) != nil {
            return String(localized: "導入済みの動画モデルから複製します")
        }
        return String(localized: "動画モデルがありません。先に動画モデル（MiniMax-H3）を導入してください")
    }

    @ViewBuilder private var progressView: some View {
        switch setup.phase {
        case .listing:
            HStack {
                ProgressView().controlSize(.small)
                Text("ファイル一覧を確認しています…").font(.caption).foregroundStyle(palette.textSecondary)
            }
        case .downloading:
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: Double(downloader.downloadedBytes), total: Double(max(downloader.totalBytes, 1)))
                Text(verbatim: "\(gigabytes(downloader.downloadedBytes)) / \(gigabytes(downloader.totalBytes))"
                     + (downloader.bytesPerSecond > 0 ? "  \(gigabytes(Int64(downloader.bytesPerSecond)))/s" : ""))
                    .font(.caption).foregroundStyle(palette.textSecondary).monospacedDigit()
                Text("ダウンロードはこの画面を閉じても続きます。中断しても、次回は続きから再開します。")
                    .font(.caption).foregroundStyle(palette.textSecondary)
            }
        case .sharing:
            HStack {
                ProgressView().controlSize(.small)
                Text("テキストエンコーダを複製しています…").font(.caption).foregroundStyle(palette.textSecondary)
            }
        case .verifying(let done, let total):
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: Double(done), total: Double(max(total, 1)))
                Text("ファイルのSHA-256を照合しています（\(gigabytes(done)) / \(gigabytes(total))）…")
                    .font(.caption).foregroundStyle(palette.textSecondary).monospacedDigit()
            }
        case .failed(let message):
            Text(message).font(.caption).foregroundStyle(palette.errorColor)
                .fixedSize(horizontal: false, vertical: true)
        case .done:
            Text("画像モデルを使えるようになりました。パネルの「動画／画像」を画像に切り替えて生成できます。")
                .font(.caption).foregroundStyle(palette.textSecondary)
        case .idle:
            if !H3ImageModel.weightsPresent {
                let space = downloadSpaceCheck(needed: H3ImageModel.downloadBytes, destination: H3ImageModel.directory)
                Text(space.text).font(.caption)
                    .foregroundStyle(space.fits ? palette.textSecondary : palette.errorColor)
                    .fixedSize(horizontal: false, vertical: true)
            } else if installed && !registered {
                Text("ダウンロード済みです。「一覧に追加」で使えるようになります。")
                    .font(.caption).foregroundStyle(palette.textSecondary)
            }
        }
    }

    private var footer: some View {
        HStack {
            Button("Finderで表示") {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: H3ImageModel.directory)])
            }
            .disabled(!FileManager.default.fileExists(atPath: H3ImageModel.directory))
            Button("すべて削除…") { confirmingRemoval = true }
                .disabled(setup.isBusy || busy
                          || !FileManager.default.fileExists(atPath: H3ImageModel.directory))
            Spacer()
            Button("閉じる") { close() }
                .keyboardShortcut(.cancelAction)
            if setup.isBusy {
                Button("中止") { setup.cancel() }
            } else if !installed {
                Button(H3ImageModel.weightsPresent ? String(localized: "残りを準備する") : String(localized: "ダウンロードする")) {
                    setup.start(videoModelPaths: videoModelPaths)
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(!agreed)
            } else if !verified {
                Button("取得元と照合する") { setup.start(videoModelPaths: videoModelPaths) }
                    .buttonStyle(.borderedProminent)
                    .disabled(!agreed)
                    .help("ここにあるファイルのSHA-256を、その版の記録と照合し、結果をモデルのフォルダに記録します")
            } else if !registered {
                Button("一覧に追加") { registerIfComplete() }
                    .buttonStyle(.borderedProminent)
                    .disabled(!agreed)
            }
        }
    }

    private func registerIfComplete() {
        guard H3ImageModel.complete else { return }
        ModelLibrary.registerImageModel(path: H3ImageModel.directory, name: H3ImageModel.displayName)
        version += 1
    }

    private func remove() {
        do {
            for model in library.imageModels where model.path == H3ImageModel.directory {
                library.removeImageModel(id: model.id)
            }
            try H3ImageModel.removeAll()
            removalError = nil
        } catch {
            removalError = String(localized: "削除できませんでした（詳細: \(error.localizedDescription)）")
        }
        version += 1
    }
}
