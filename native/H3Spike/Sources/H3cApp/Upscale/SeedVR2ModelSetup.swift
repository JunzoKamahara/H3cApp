import AppKit
import H3Engine
import SwiftUI

/// The SeedVR2-3B model on this Mac (VideoUpscaler.seedVR2ModelDirectory):
/// the MLX 8-bit export the engine reads directly (no conversion), the
/// revision it is pinned to, and whether the files were checked against it
/// (UPSCALE_PLAN appendix A.7).
///
/// Nothing is bundled: the user downloads the weights here after reading
/// the license notice. The weights are ByteDance Seed's SeedVR2-3B
/// (Apache License 2.0), converted to MLX's 8-bit format by the
/// mlx-community export (its model card declares apache-2.0); the app
/// records the revision and each file's SHA-256 beside the weights
/// (H3VSR_PROVENANCE.json) and saves the upstream Apache License 2.0 text.
enum SeedVR2Model {
    static let repo = "mlx-community/SeedVR2-3B-mlx-int8"
    static let revision = "9dbcb304916d94b4a9769b588ea28439054ce3bb"
    static var revisionShort: String { String(revision.prefix(8)) }
    static let revisionDate = "2026-06-06"
    static let declaredLicense = "Apache-2.0"
    /// The original model and its license.
    static let upstream = "ByteDance-Seed/SeedVR2-3B"
    static let upstreamURL = URL(string: "https://huggingface.co/ByteDance-Seed/SeedVR2-3B")!
    static let repoURL = URL(string: "https://huggingface.co/mlx-community/SeedVR2-3B-mlx-int8/tree/\(revision)")!
    /// The model card of the pinned revision: `license: apache-2.0` and the
    /// provenance chain (ByteDance Seed -> fp16 safetensors -> MLX).
    static let declarationURL = URL(string: "https://huggingface.co/mlx-community/SeedVR2-3B-mlx-int8/blob/\(revision)/README.md")!
    static let apacheLicenseURL = URL(string: "https://www.apache.org/licenses/LICENSE-2.0")!
    /// The upstream repository's LICENSE (Apache License 2.0), pinned to the
    /// commit that added it; saved beside the weights after a hash check.
    static let licenseTextURL = URL(string: "https://raw.githubusercontent.com/ByteDance-Seed/SeedVR/790ea601a9798eb3ff3b5292dc80ace62fa17cb5/LICENSE")!
    static let licenseTextSHA256 = "5967a0c153e731109b76c5d83b5e0fae2382b6aaf7b898d08ff03702eea59430"
    static let licenseFile = "LICENSE"

    /// What the engine reads (h3_seedvr2_inspect_dir / h3_seedvr2_open).
    static let requiredFiles = ["config.json", "transformer.safetensors", "vae.safetensors", "pos_emb.safetensors"]
    static let downloadedFiles = requiredFiles + ["README.md"]
    /// SHA-256 of each file at the pinned revision: the LFS object ids
    /// Hugging Face lists for the weights, computed for the two small files.
    static let expectedSHA256: [String: String] = [
        "config.json": "20000eddfa8854c83d211c3c55d2ea3e36da538cd51191a2bcb9cd0ba42e29fd",
        "transformer.safetensors": "4f0d2b3786440d50b94cf7566cd244c4626f3f418d903f75e8422bfd48023dec",
        "vae.safetensors": "a2efbfb38a82a99a9cd16432ca5d564510678d228ce6209692d528aa32285cfb",
        "pos_emb.safetensors": "d2e06cadb10f1f149b5aa781185b250eb60f0e64327f12efd17886cecc9c2984",
        "README.md": "9cd5b962fe8dabb9bd92ca50ea44f02551606ef114d441eff5ba796c019feaf7",
    ]
    static let approximateDownloadBytes: Int64 = 4_719_450_000
    /// Resident on the GPU while enlarging (BF16 transformer + F32 VAE).
    static let approximateResidentBytes: Int64 = 7_000_000_000

    static var directory: String { VideoUpscaler.seedVR2ModelDirectory }

    static var weightsPresent: Bool {
        requiredFiles.allSatisfy { FileManager.default.fileExists(atPath: directory + "/" + $0) }
    }

    static func fileSize(_ relativePath: String) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: directory + "/" + relativePath)[.size]) as? Int64
    }

    private static let licenseAcceptedKey = "h3c-app.seedVR2LicenseAcceptedAt"

    static var licenseAcceptedAt: Date? {
        get { UserDefaults.standard.object(forKey: licenseAcceptedKey) as? Date }
        set { UserDefaults.standard.set(newValue, forKey: licenseAcceptedKey) }
    }

    // MARK: Provenance record (the same form as SwiftVRModel.Provenance)

    static let provenanceFile = "H3VSR_PROVENANCE.json"

    static var provenance: SwiftVRModel.Provenance? {
        guard let data = FileManager.default.contents(atPath: directory + "/" + provenanceFile) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(SwiftVRModel.Provenance.self, from: data)
    }

    /// A record for this revision with the expected hashes, every required
    /// file still at its recorded size.
    static var verified: Bool {
        guard let record = provenance, record.revision == revision else { return false }
        for name in requiredFiles {
            guard let entry = record.files[name], entry.sha256 == expectedSHA256[name],
                  let size = fileSize(name), size == entry.bytes else { return false }
        }
        return true
    }

    static func checkListing(_ entries: [HFFileEntry]) throws {
        for entry in entries {
            guard let listed = entry.sha256, let expected = expectedSHA256[entry.path] else { continue }
            if listed != expected {
                throw SwiftVRModel.VerificationError.listing(file: entry.path, listed: listed, expected: expected)
            }
        }
    }

    /// Hashes the downloaded files against the pinned revision, fetches the
    /// license text when it is not here, writes the record. A file whose
    /// hash differs is deleted and named in the error.
    static func verify(progress: @escaping @Sendable (Int64, Int64) -> Void) async throws {
        let directory = self.directory
        let files = downloadedFiles.filter { FileManager.default.fileExists(atPath: directory + "/" + $0) }
        let total = files.reduce(Int64(0)) { $0 + (fileSize($1) ?? 0) }
        var done: Int64 = 0
        var record: [String: SwiftVRModel.ProvenanceEntry] = [:]
        for name in files {
            let path = directory + "/" + name
            let expected = expectedSHA256[name] ?? ""
            let base = done
            let actual = try await Task.detached(priority: .userInitiated) {
                var read: Int64 = 0
                return try FileHash.sha256(ofFile: path) { bytes in
                    read += bytes
                    progress(base + read, total)
                }
            }.value
            done += fileSize(name) ?? 0
            progress(done, total)
            guard actual == expected else {
                try? FileManager.default.removeItem(atPath: path)
                throw SwiftVRModel.VerificationError.mismatch(file: name, expected: expected, actual: actual)
            }
            record[name] = SwiftVRModel.ProvenanceEntry(sha256: actual, bytes: fileSize(name) ?? 0)
        }
        try await fetchLicenseText()
        let provenance = SwiftVRModel.Provenance(
            repo: repo, revision: revision, revisionDate: revisionDate, declaredLicense: declaredLicense,
            declaration: "README.md of this revision: front matter `license: apache-2.0`, section Provenance & license",
            licenseText: licenseFile, files: record, verifiedAt: Date(),
            note: "SeedVR2-3B by ByteDance Seed (ByteDance-Seed/SeedVR, Apache License 2.0), converted to MLX affine 8-bit (group 64) by the mlx-community export; a format conversion of the same weights. LICENSE is the upstream repository's Apache License 2.0 text.")
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(provenance).write(to: URL(fileURLWithPath: directory + "/" + provenanceFile), options: .atomic)
    }

    private static func fetchLicenseText() async throws {
        let path = directory + "/" + licenseFile
        if let data = FileManager.default.contents(atPath: path), FileHash.sha256(of: data) == licenseTextSHA256 { return }
        do {
            let (data, response) = try await URLSession.shared.data(from: licenseTextURL)
            guard let http = response as? HTTPURLResponse, (200 ..< 300).contains(http.statusCode) else {
                throw SwiftVRModel.VerificationError.licenseText("HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)")
            }
            guard FileHash.sha256(of: data) == licenseTextSHA256 else {
                throw SwiftVRModel.VerificationError.licenseText(String(localized: "取得した本文のSHA-256が記録と一致しません"))
            }
            try data.write(to: URL(fileURLWithPath: path), options: .atomic)
        } catch let error as SwiftVRModel.VerificationError {
            throw error
        } catch {
            throw SwiftVRModel.VerificationError.licenseText(error.localizedDescription)
        }
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

/// Downloads and checks the SeedVR2 model: one for the whole app, so a
/// download keeps going when the sheet showing it closes. No conversion:
/// the engine reads the downloaded files.
@MainActor
final class SeedVR2Setup: ObservableObject {
    static let shared = SeedVR2Setup()

    enum Phase: Equatable {
        case idle
        case listing
        case downloading
        case verifying(done: Int64, total: Int64)
        case failed(String)
        case done

        var apiText: String {
            switch self {
            case .idle: return "idle"
            case .listing: return "listing"
            case .downloading: return "downloading"
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
        case .listing, .downloading, .verifying: return true
        default: return false
        }
    }

    func start() {
        guard !isBusy else { return }
        SeedVR2Model.licenseAcceptedAt = SeedVR2Model.licenseAcceptedAt ?? Date()
        phase = .listing
        task = Task { await run() }
    }

    func cancel() {
        task?.cancel()
        downloader.cancel()
        if isBusy { phase = .idle }
    }

    private func run() async {
        do {
            if !SeedVR2Model.weightsPresent || !Self.allFilesComplete() {
                try await download()
            }
            guard !Task.isCancelled else { return }
            if !SeedVR2Model.verified && SeedVR2Model.weightsPresent {
                phase = .verifying(done: 0, total: 0)
                try await SeedVR2Model.verify { done, total in
                    Task { @MainActor in
                        if case .verifying = self.phase { self.phase = .verifying(done: done, total: total) }
                    }
                }
            }
            guard !Task.isCancelled else { return }
            phase = .done
        } catch is CancellationError {
            phase = .idle
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    private static var listedEntries: [HFFileEntry] = []
    private static func allFilesComplete() -> Bool {
        guard !listedEntries.isEmpty else { return false }
        return listedEntries.allSatisfy { entry in
            let size = (try? FileManager.default.attributesOfItem(atPath: SeedVR2Model.directory + "/" + entry.local)[.size] as? Int64) ?? -1
            return size == entry.size
        }
    }

    private func download() async throws {
        let wanted = Set(SeedVR2Model.downloadedFiles)
        let listed = try await ModelDownloader.fetchEntries(repo: SeedVR2Model.repo, subpath: "", revision: SeedVR2Model.revision)
        let entries = listed.filter { wanted.contains($0.path) }
        let missing = SeedVR2Model.requiredFiles.filter { name in !entries.contains { $0.path == name } }
        guard missing.isEmpty else {
            throw ModelDownloadError.listingFailed(
                String(localized: "\(SeedVR2Model.repo)（revision \(SeedVR2Model.revisionShort)）に、必要なファイル（\(missing.joined(separator: ", "))）が見つかりませんでした"))
        }
        try SeedVR2Model.checkListing(entries)
        Self.listedEntries = entries
        try Task.checkCancellation()
        downloader.use(entries)
        downloader.startDownload(destination: SeedVR2Model.directory)
        phase = .downloading
        for await state in downloader.$state.values {
            switch state {
            case .completed: return
            case .failed(let message): throw ModelDownloadError.listingFailed(message)
            case .ready, .idle:
                if phase == .downloading { throw CancellationError() }
                return
            default: continue
            }
        }
    }
}

/// モデルを準備 for SeedVR2: the license notice, the download, the check
/// against the pinned revision, and what is installed.
struct SeedVR2SetupView: View {
    var busy = false
    var close: () -> Void

    @ObservedObject private var setup = SeedVR2Setup.shared
    @ObservedObject private var downloader = SeedVR2Setup.shared.downloader
    @Environment(\.colorScheme) private var colorScheme
    @State private var agreed = SeedVR2Model.licenseAcceptedAt != nil
    @State private var confirmingRemoval = false
    @State private var removalError: String?
    @State private var version = 0

    private var palette: H3Palette { H3Palette(colorScheme) }
    private var installed: Bool { _ = version; _ = setup.phase; return SeedVR2Model.weightsPresent }
    private var verified: Bool { _ = version; _ = setup.phase; return SeedVR2Model.verified }

    private func gigabytes(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("SeedVR2のモデルを準備").font(.headline)
            Text("SeedVR2-3B（ByteDance Seedの一段拡散超解像モデル）の重みを、Hugging Faceの\(SeedVR2Model.repo)（MLX用の8ビット版、revision \(SeedVR2Model.revisionShort)、\(SeedVR2Model.revisionDate)）からダウンロードし、各ファイルのSHA-256をその版の記録と照合します。変換は不要で、約\(gigabytes(SeedVR2Model.approximateDownloadBytes))が「\(SeedVR2Model.directory)」に置かれます。拡大のあいだGPUに約\(gigabytes(SeedVR2Model.approximateResidentBytes))を使います。")
                .font(.caption).foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            licenseNotice
            Toggle(isOn: $agreed) {
                Text("上の説明を読み、Apache License 2.0の条件に従って使います")
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
        .frame(width: 620, height: 560)
        .background(palette.canvas)
        .alert("SeedVR2のモデルを削除", isPresented: $confirmingRemoval) {
            Button("削除", role: .destructive) { remove() }
            Button("キャンセル", role: .cancel) {}
        } message: {
            Text("ダウンロードした重み（計\(gigabytes(SeedVR2Model.bytesOnDisk()))）を削除します。また使うには、もう一度ダウンロードが必要です。")
        }
    }

    private var licenseNotice: some View {
        HStack(alignment: .top, spacing: H3Spacing.sm) {
            Image(systemName: "doc.text.magnifyingglass").foregroundStyle(palette.accent)
            VStack(alignment: .leading, spacing: 4) {
                Text("SeedVR2-3BはByteDance Seedが Apache License 2.0 で公開したモデルです。ダウンロードする版は、その重みをMLXの8ビット形式に変換したもの（mlx-community、READMEの宣言: apache-2.0）で、形式の変換であり別のモデルではありません。アプリはこの版だけを取得し、ファイルが記録と同一であることをSHA-256で確かめ、元のリポジトリのApache License 2.0本文を重みの隣に保存します。Apache License 2.0には、著作権表示とライセンス文の保持、変更した場合の明示などの条件があります。あなたの利用にどの条件が当てはまるかは、利用者自身（必要なら専門家）が判断してください。")
                    .font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: H3Spacing.md) {
                    Link("宣言のあるREADME（その版）", destination: SeedVR2Model.declarationURL)
                    Link("Apache License 2.0の本文", destination: SeedVR2Model.apacheLicenseURL)
                }
                .font(.caption)
                HStack(spacing: H3Spacing.md) {
                    Link("元のモデル（ByteDance Seed）", destination: SeedVR2Model.upstreamURL)
                    Link("配布元（Hugging Face）", destination: SeedVR2Model.repoURL)
                }
                .font(.caption)
            }
        }
        .padding(H3Spacing.sm)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(palette.accent.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
    }

    private var verificationText: String {
        if verified, let record = SeedVR2Model.provenance {
            return String(localized: "SHA-256がその版の記録と一致（\(record.verifiedAt.formatted(date: .numeric, time: .shortened))に確認）")
        }
        if installed {
            return String(localized: "まだ照合していません（「取得元と照合する」で確かめられます）")
        }
        return String(localized: "まだありません")
    }

    private var statusRows: some View {
        Grid(alignment: .leading, horizontalSpacing: H3Spacing.md, verticalSpacing: 6) {
            GridRow {
                Text("取得元").font(.caption).foregroundStyle(palette.textSecondary)
                Text("\(SeedVR2Model.repo) revision \(SeedVR2Model.revisionShort)（\(SeedVR2Model.revisionDate)、READMEの宣言: Apache License 2.0）").font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
            GridRow {
                Text("重み").font(.caption).foregroundStyle(palette.textSecondary)
                Text(installed ? String(localized: "ダウンロード済み") : String(localized: "まだありません")).font(.caption)
            }
            GridRow {
                Text("照合").font(.caption).foregroundStyle(palette.textSecondary)
                Text(verificationText).font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
            GridRow {
                Text("使用容量").font(.caption).foregroundStyle(palette.textSecondary)
                Text(verbatim: gigabytes({ _ = version; return SeedVR2Model.bytesOnDisk() }())).font(.caption)
            }
            if let accepted = SeedVR2Model.licenseAcceptedAt {
                GridRow {
                    Text("ライセンス").font(.caption).foregroundStyle(palette.textSecondary)
                    Text("\(accepted.formatted(date: .numeric, time: .shortened))に説明を確認").font(.caption)
                }
            }
        }
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
            Text("SeedVR2を使えるようになりました。「動画を拡大」で選べます。")
                .font(.caption).foregroundStyle(palette.textSecondary)
        case .idle:
            if !installed {
                let space = downloadSpaceCheck(needed: SeedVR2Model.approximateDownloadBytes, destination: SeedVR2Model.directory)
                Text(space.text).font(.caption)
                    .foregroundStyle(space.fits ? palette.textSecondary : palette.errorColor)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if busy {
                Text("生成中はダウンロードできますが、拡大は生成が終わってからです。")
                    .font(.caption).foregroundStyle(palette.textSecondary)
            }
        }
    }

    private var footer: some View {
        HStack {
            Button("Finderで表示") {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: SeedVR2Model.directory)])
            }
            .disabled(!FileManager.default.fileExists(atPath: SeedVR2Model.directory))
            Button("すべて削除…") { confirmingRemoval = true }
                .disabled(setup.isBusy || !FileManager.default.fileExists(atPath: SeedVR2Model.directory))
            Spacer()
            Button("閉じる") { close() }
                .keyboardShortcut(.cancelAction)
            if setup.isBusy {
                Button("中止") { setup.cancel() }
            } else if !installed {
                Button("ダウンロードする") { setup.start() }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
                    .disabled(!agreed)
            } else if !verified {
                Button("取得元と照合する") { setup.start() }
                    .buttonStyle(.borderedProminent)
                    .disabled(!agreed)
                    .help("ここにあるファイルのSHA-256を、その版の記録と照合し、結果をモデルのフォルダに記録します")
            }
        }
    }

    private func remove() {
        do {
            try SeedVR2Model.removeAll()
            removalError = nil
        } catch {
            removalError = String(localized: "削除できませんでした（詳細: \(error.localizedDescription)）")
        }
        version += 1
    }
}
