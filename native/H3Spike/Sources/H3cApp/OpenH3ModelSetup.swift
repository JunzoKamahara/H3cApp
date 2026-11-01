import CH3
import H3Engine
import SwiftUI

/// The files the "Open-H3 再生成" enlargement needs, pinned to one revision of
/// NVIDIA's Open-H3-Regenerate-2K (UPSCALE_PLAN appendix A.4/A.5): the
/// merged FP32 LoRA (refiner ranks 0-383 | LightX2V Turbo 384-511 | LMS
/// Sharp 512-575), from which the refiner part alone is extracted on this
/// Mac as BF16 (the merged file is then deleted), LBH-123-AI's latent
/// upscaler as that repository ships it, and LightX2V's Turbo v1.2 (its own
/// file, not the merged ranks), taken from the app's LoRA folder when a copy
/// of the recorded revision is there and downloaded here otherwise.
enum OpenH3Model {
    static let repo = "Efficient-Large-Model/Open-H3-Regenerate-2K"
    static let revision = "5809c33730c18b6b1c827b40974855a4288641b3"
    static var revisionShort: String { String(revision.prefix(8)) }
    static let revisionDate = "2026-10-28"
    /// The model card: `license: other`, `license_name: minimax-h3-community`.
    static let declaredLicense = "MiniMax H3 Community License"
    static let repoURL = URL(string: "https://huggingface.co/\(repo)/tree/\(revision)")!
    static let readmeURL = URL(string: "https://huggingface.co/\(repo)/blob/\(revision)/README.md")!
    /// The license the card links (MiniMax-H3 at the revision the bundle mirrors).
    static let h3LicenseURL = URL(string: "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/6818f6c32d12b210915e44ad56a4228c2608f160/LICENSE")!
    static let upsamplerSourceURL = URL(string: "https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler")!
    static let turboRepo = "lightx2v/Minimax-h3-Turbo"
    static let turboRevision = "3ec17a324ced54151364f24f8b5fb6bf7e26414f"
    static let turboRepoURL = URL(string: "https://huggingface.co/\(turboRepo)/tree/\(turboRevision)")!

    // Repository paths (also the paths under `directory`).
    static let mergedFile = "lora/open_h3_regenerate_2k_lora.safetensors"
    static let upsamplerFile = "latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
    static let textFiles = ["base_model/LICENSE", "latent_upsampler/LICENSE", "NOTICE", "README.md",
                            "lora/lora_config.json", "lora/MERGE_VERIFY.json"]
    static let downloadedFiles = [mergedFile, upsamplerFile] + textFiles
    /// What the extraction writes (ranks 0-383 of the merge, BF16).
    static let refinerFile = "open_h3_refiner_r384_bf16.safetensors"
    static let refinerRanks = 0 ..< 384
    /// Turbo v1.2's own file: here when the LoRA folder has no copy.
    static let turboFileName = "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors"
    static let turboFile = "turbo/" + turboFileName
    static let turboSHA256 = "c3d4a2cf618efea71b9e21a4baaa12d412f1eb6c2b6f86efacaf0ebb6814b689"
    static let turboBytes: Int64 = 1_383_677_808

    /// SHA-256 of the files at the pinned revision (the safetensors from the
    /// listing's LFS ids, the text files hashed on 2026-11-01).
    static let expectedSHA256: [String: String] = [
        mergedFile: "8e36a0912ab72f7f3b00f41e3c7c97c7dfdec21e66c88ab22f08b9b0c681bb04",
        upsamplerFile: "4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6",
        "base_model/LICENSE": "59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44",
        "latent_upsampler/LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        "NOTICE": "154fe7fbdf198395d53c57d6786b2fbf7e63a083ff70843c5ef78c0c7303d91a",
        "README.md": "1d23b3f360cd4f8ba896c04c6d9bc7b9d81a8fc87c52a51e72b854dacf27727b",
        "lora/lora_config.json": "8c0b54b67cfa8be88814fabdebd1ec04d7cf72239d0b7ddd9756fe62e8cbd878",
        "lora/MERGE_VERIFY.json": "333dff709460e629d91ed0d9a116d83ca1c50571285494aa5d2fe841fbeedf70",
        turboFile: turboSHA256,
    ]
    static let mergedBytes: Int64 = 12_452_449_080
    static let upsamplerBytes: Int64 = 690_592_992
    /// The extracted refiner: 384/576 of the merge's rank rows and columns, F32 -> BF16.
    static let approximateRefinerBytes: Int64 = 4_150_000_000
    /// The merged file plus the upsampler; the extraction needs room for the
    /// refiner on top while the merge is still there.
    static var approximateDownloadBytes: Int64 { mergedBytes + upsamplerBytes + (turboNeedsDownload ? turboBytes : 0) }
    static var approximateSetupBytes: Int64 { approximateDownloadBytes + approximateRefinerBytes }

    static var directory: String { VideoUpscaler.openH3ModelDirectory }
    static var upsamplerDirectory: String { directory + "/latent_upsampler" }
    static var refinerPath: String { directory + "/" + refinerFile }
    static var mergedPath: String { directory + "/" + mergedFile }

    /// Turbo v1.2 in the app's LoRA folder (by name and size; hashed when
    /// the record is written), else the copy downloaded here.
    static var turboPath: String? {
        let own = directory + "/" + turboFile
        if fileSize(ofPath: own) == turboBytes { return own }
        let inFolder = defaultLoRADirectory + "/" + turboFileName
        if fileSize(ofPath: inFolder) == turboBytes { return inFolder }
        return nil
    }
    static var turboNeedsDownload: Bool { turboPath == nil }

    static func fileSize(ofPath path: String) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: path)[.size]) as? Int64
    }
    static func fileSize(_ relativePath: String) -> Int64? { fileSize(ofPath: directory + "/" + relativePath) }

    static var upsamplerPresent: Bool { fileSize(upsamplerFile) == upsamplerBytes }
    static var refinerPresent: Bool { (fileSize(refinerFile) ?? 0) > 0 }
    static var mergedPresent: Bool { fileSize(mergedFile) != nil }
    /// Everything the regeneration runs with is here.
    static var ready: Bool { upsamplerPresent && refinerPresent && turboPath != nil }

    private static let licenseAcceptedKey = "h3c-app.openH3LicenseAcceptedAt"
    static var licenseAcceptedAt: Date? {
        get { UserDefaults.standard.object(forKey: licenseAcceptedKey) as? Date }
        set { UserDefaults.standard.set(newValue, forKey: licenseAcceptedKey) }
    }

    // MARK: Provenance record (the same form as SwiftVRModel.Provenance)

    static let provenanceFile = "H3VSR_PROVENANCE.json"
    /// Keys in the record besides the repository paths.
    static let refinerKey = "extracted:" + refinerFile
    static let turboKey = "turbo:" + turboFileName

    static var provenance: SwiftVRModel.Provenance? {
        guard let data = FileManager.default.contents(atPath: directory + "/" + provenanceFile) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(SwiftVRModel.Provenance.self, from: data)
    }

    /// A record for this revision: the merge's hash matched before the
    /// extraction, the upsampler and the refiner still at their recorded
    /// sizes, Turbo's hash matched.
    static var verified: Bool {
        guard let record = provenance, record.revision == revision,
              record.files[mergedFile]?.sha256 == expectedSHA256[mergedFile],
              let up = record.files[upsamplerFile], up.sha256 == expectedSHA256[upsamplerFile],
              fileSize(upsamplerFile) == up.bytes,
              let refiner = record.files[refinerKey], fileSize(refinerFile) == refiner.bytes,
              let turbo = record.files[turboKey], turbo.sha256 == turboSHA256,
              let turboPath, fileSize(ofPath: turboPath) == turbo.bytes else { return false }
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

    enum SetupError: LocalizedError {
        case mismatch(file: String, expected: String, actual: String)
        case extracted(String)
        case noTurbo

        var errorDescription: String? {
            switch self {
            case .mismatch(let file, let expected, let actual):
                return String(localized: "「\(file)」のSHA-256（\(String(actual.prefix(16)))…）が、その版の記録（\(String(expected.prefix(16)))…）と一致しません。このファイルは削除しました。もう一度ダウンロードしてください。")
            case .extracted(let message):
                return String(localized: "精緻化LoRAの取り出しに失敗しました（詳細: \(message)）")
            case .noTurbo:
                return String(localized: "LightX2V Turbo v1.2のファイルが見つかりません。")
            }
        }
    }

    /// Hashes a downloaded file against the pinned revision; a mismatch
    /// deletes it and throws.
    private static func check(_ relativePath: String, path: String? = nil, expected: String,
                              progress: @escaping @Sendable (Int64) -> Void) async throws -> SwiftVRModel.ProvenanceEntry {
        let path = path ?? directory + "/" + relativePath
        let actual = try await Task.detached(priority: .userInitiated) {
            var read: Int64 = 0
            return try FileHash.sha256(ofFile: path) { bytes in
                read += bytes
                progress(read)
            }
        }.value
        guard actual == expected else {
            try? FileManager.default.removeItem(atPath: path)
            throw SetupError.mismatch(file: relativePath, expected: expected, actual: actual)
        }
        return SwiftVRModel.ProvenanceEntry(sha256: actual, bytes: fileSize(ofPath: path) ?? 0)
    }

    /// After the download: hashes the merge, the upsampler, the text files
    /// and Turbo against the records, extracts the refiner (ranks 0-383,
    /// BF16) from the merge, deletes the merge, writes the record.
    /// `progress` gets (phase, done, total) with bytes for the hashing and
    /// tensors for the extraction.
    static func verifyAndExtract(progress: @escaping @Sendable (String, Int64, Int64) -> Void) async throws {
        var record = provenance?.revision == revision ? provenance?.files ?? [:] : [:]
        guard let turboPath else { throw SetupError.noTurbo }
        // Hashing: whatever is here that the record does not cover yet.
        var toHash: [(key: String, path: String, expected: String)] = []
        if mergedPresent { toHash.append((mergedFile, mergedPath, expectedSHA256[mergedFile]!)) }
        for name in [upsamplerFile] + textFiles where fileSize(name) != nil {
            toHash.append((name, directory + "/" + name, expectedSHA256[name]!))
        }
        toHash.append((turboKey, turboPath, turboSHA256))
        let total = toHash.reduce(Int64(0)) { $0 + (fileSize(ofPath: $1.path) ?? 0) }
        var done: Int64 = 0
        for item in toHash {
            try Task.checkCancellation()
            let base = done
            let entry = try await check(item.key, path: item.path, expected: item.expected) { read in
                progress("verifying", base + read, total)
            }
            done += entry.bytes
            progress("verifying", done, total)
            record[item.key] = entry
        }
        // The refiner: extracted from the merge when the merge is here;
        // otherwise it must already be here from an earlier extraction.
        if mergedPresent {
            try Task.checkCancellation()
            let partial = refinerPath
            try await Task.detached(priority: .userInitiated) {
                try H3LoRAExtract.extractRanks(
                    from: mergedPath, to: partial, ranks: refinerRanks, bf16: true,
                    note: "Open-H3 refiner: ranks 0-383 of \(repo)@\(revision) \(mergedFile) (sha256 \(expectedSHA256[mergedFile]!)), extracted by H3cApp, BF16") { done, total in
                        progress("extracting", Int64(done), Int64(total))
                        return !Task.isCancelled
                    }
            }.value
            try Task.checkCancellation()
            let sha = try await Task.detached(priority: .userInitiated) { try FileHash.sha256(ofFile: partial) }.value
            record[refinerKey] = SwiftVRModel.ProvenanceEntry(sha256: sha, bytes: fileSize(ofPath: partial) ?? 0)
            try? FileManager.default.removeItem(atPath: mergedPath)
        } else if let size = fileSize(refinerFile), size > 0, record[refinerKey] == nil {
            let sha = try await Task.detached(priority: .userInitiated) { try FileHash.sha256(ofFile: refinerPath) }.value
            record[refinerKey] = SwiftVRModel.ProvenanceEntry(sha256: sha, bytes: size)
        }
        guard refinerPresent else { throw SetupError.extracted(String(localized: "取り出したファイルがありません")) }
        let provenance = SwiftVRModel.Provenance(
            repo: repo, revision: revision, revisionDate: revisionDate, declaredLicense: declaredLicense,
            declaration: "Model card of this revision: `license: other`, `license_name: minimax-h3-community`; README section License: the refiner LoRA follows the MiniMax H3 Community License, the latent upscaler and the merged Turbo and LMS Sharp parts their own licenses",
            licenseText: "base_model/LICENSE", files: record, verifiedAt: Date(),
            note: "The refiner (\(refinerFile)) is ranks \(refinerRanks.lowerBound)-\(refinerRanks.upperBound - 1) of the merged FP32 LoRA \(mergedFile) (rank 576 = refiner 384 | LightX2V Turbo v1.2 128 | LMS Sharp 64), cut on this Mac and stored as BF16; the merged file was deleted after the extraction. Turbo v1.2 (\(turboFileName), \(turboRepo)@\(turboRevision)) is applied from its own file. The LMS Sharp part is not used.")
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(provenance).write(to: URL(fileURLWithPath: directory + "/" + provenanceFile), options: .atomic)
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

/// Downloads, checks and extracts the Open-H3 files: one for the whole app,
/// so a download keeps going when the sheet showing it closes.
@MainActor
final class OpenH3Setup: ObservableObject {
    static let shared = OpenH3Setup()

    enum Phase: Equatable {
        case idle
        case listing
        case downloading
        case verifying(done: Int64, total: Int64)
        case extracting(done: Int64, total: Int64)
        case failed(String)
        case done

        var apiText: String {
            switch self {
            case .idle: return "idle"
            case .listing: return "listing"
            case .downloading: return "downloading"
            case .verifying: return "verifying"
            case .extracting: return "extracting"
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
        case .listing, .downloading, .verifying, .extracting: return true
        default: return false
        }
    }

    func start() {
        guard !isBusy else { return }
        OpenH3Model.licenseAcceptedAt = OpenH3Model.licenseAcceptedAt ?? Date()
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
            if !OpenH3Model.ready && !(OpenH3Model.mergedPresent && OpenH3Model.upsamplerPresent && !OpenH3Model.turboNeedsDownload) {
                try await download()
            }
            guard !Task.isCancelled else { return }
            if !OpenH3Model.verified {
                phase = .verifying(done: 0, total: 0)
                try await OpenH3Model.verifyAndExtract { stage, done, total in
                    Task { @MainActor in
                        guard self.isBusy else { return }
                        self.phase = stage == "extracting" ? .extracting(done: done, total: total)
                                                          : .verifying(done: done, total: total)
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

    private func download() async throws {
        // What is still needed: the merge only until the refiner exists.
        var wanted = Set(OpenH3Model.textFiles)
        if !OpenH3Model.upsamplerPresent { wanted.insert(OpenH3Model.upsamplerFile) }
        if !OpenH3Model.refinerPresent { wanted.insert(OpenH3Model.mergedFile) }
        let listed = try await ModelDownloader.fetchEntries(repo: OpenH3Model.repo, subpath: "", revision: OpenH3Model.revision)
        var entries = listed.filter { wanted.contains($0.path) }
        let missing = wanted.filter { name in !entries.contains { $0.path == name } }
        guard missing.isEmpty else {
            throw ModelDownloadError.listingFailed(
                String(localized: "\(OpenH3Model.repo)（revision \(OpenH3Model.revisionShort)）に、必要なファイル（\(missing.sorted().joined(separator: ", "))）が見つかりませんでした"))
        }
        try OpenH3Model.checkListing(entries)
        if OpenH3Model.turboNeedsDownload {
            let turbo = try await ModelDownloader.fetchEntries(repo: OpenH3Model.turboRepo, subpath: "", revision: OpenH3Model.turboRevision)
            guard var entry = turbo.first(where: { $0.path == OpenH3Model.turboFileName }) else {
                throw ModelDownloadError.listingFailed(
                    String(localized: "\(OpenH3Model.turboRepo)（revision \(String(OpenH3Model.turboRevision.prefix(8)))）に \(OpenH3Model.turboFileName) が見つかりませんでした"))
            }
            if let listed = entry.sha256, listed != OpenH3Model.turboSHA256 {
                throw SwiftVRModel.VerificationError.listing(file: entry.path, listed: listed, expected: OpenH3Model.turboSHA256)
            }
            entry.localPath = OpenH3Model.turboFile
            entries.append(entry)
        }
        try Task.checkCancellation()
        downloader.use(entries)
        downloader.startDownload(destination: OpenH3Model.directory)
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

/// モデルを準備 for the Open-H3 regeneration: the license notice, the
/// download, the check against the pinned revision, the extraction, and
/// what is installed.
struct OpenH3SetupView: View {
    var busy = false
    var close: () -> Void

    @ObservedObject private var setup = OpenH3Setup.shared
    @ObservedObject private var downloader = OpenH3Setup.shared.downloader
    @Environment(\.colorScheme) private var colorScheme
    @State private var agreed = OpenH3Model.licenseAcceptedAt != nil
    @State private var confirmingRemoval = false
    @State private var removalError: String?
    @State private var version = 0

    private var palette: H3Palette { H3Palette(colorScheme) }
    private var installed: Bool { _ = version; _ = setup.phase; return OpenH3Model.ready }
    private var verified: Bool { _ = version; _ = setup.phase; return OpenH3Model.verified }

    private func gigabytes(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("Open-H3 再生成のモデルを準備").font(.headline)
            Text("NVIDIAのOpen-H3-Regenerate-2K（Hugging Faceの\(OpenH3Model.repo)、revision \(OpenH3Model.revisionShort)、\(OpenH3Model.revisionDate)）から、結合されたLoRA（FP32・\(gigabytes(OpenH3Model.mergedBytes))）と潜在アップスケーラ（\(gigabytes(OpenH3Model.upsamplerBytes))）をダウンロードし、SHA-256をその版の記録と照合します。そのあと、このMac上で精緻化LoRAの部分（ランク0〜383）だけをBF16（約\(gigabytes(OpenH3Model.approximateRefinerBytes))）に取り出し、結合ファイルは削除します。LightX2V Turbo v1.2は、LoRAフォルダに同じ版があればそれを使い、なければ\(gigabytes(OpenH3Model.turboBytes))をダウンロードします。保存先は「\(OpenH3Model.directory)」です。再生成はH3本体の生成と同じメモリ（1344×768で約15GB）を使います。")
                .font(.caption).foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            licenseNotice
            Toggle(isOn: $agreed) {
                Text("上の説明を読み、各ライセンスの条件に従って使います")
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
        .alert("Open-H3のモデルを削除", isPresented: $confirmingRemoval) {
            Button("削除", role: .destructive) { remove() }
            Button("キャンセル", role: .cancel) {}
        } message: {
            Text("ダウンロードと取り出しで置いたファイル（計\(gigabytes(OpenH3Model.bytesOnDisk()))）を削除します。また使うには、もう一度ダウンロードが必要です。LoRAフォルダのTurboは残します。")
        }
    }

    private var licenseNotice: some View {
        HStack(alignment: .top, spacing: H3Spacing.sm) {
            Image(systemName: "doc.text.magnifyingglass").foregroundStyle(palette.accent)
            VStack(alignment: .leading, spacing: 4) {
                Text("Open-H3の精緻化LoRAには、ベースモデルと同じ「MiniMax H3 Community License Agreement」が適用されます（地域・用途・再配布の条件を含みます）。配布されているのは、この精緻化LoRAにLightX2V Turbo v1.2とLMS Sharpを結合した1つのファイルで、アプリはその版だけを取得して記録と照合し、精緻化の部分（ランク0〜383）だけを取り出して使います（取り出しは許諾を生みません。Sharpの部分は使いません）。潜在アップスケーラ（LBH-123-AI）は、同梱のLICENSE（Apache License 2.0）とともに置きます。Turbo v1.2は、Apache License 2.0と宣言されていた版（revision \(String(OpenH3Model.turboRevision.prefix(8)))）を使います。あなたの利用にどの条件が当てはまるかは、利用者自身（必要なら専門家）が判断してください。")
                    .font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: H3Spacing.md) {
                    Link("MiniMax H3のライセンス", destination: OpenH3Model.h3LicenseURL)
                    Link("READMEのライセンスの節（その版）", destination: OpenH3Model.readmeURL)
                }
                .font(.caption)
                HStack(spacing: H3Spacing.md) {
                    Link("配布元（Hugging Face）", destination: OpenH3Model.repoURL)
                    Link("潜在アップスケーラ（LBH-123-AI）", destination: OpenH3Model.upsamplerSourceURL)
                    Link("Turbo v1.2（その版）", destination: OpenH3Model.turboRepoURL)
                }
                .font(.caption)
            }
        }
        .padding(H3Spacing.sm)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(palette.accent.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
    }

    private var verificationText: String {
        if verified, let record = OpenH3Model.provenance {
            return String(localized: "SHA-256がその版の記録と一致（\(record.verifiedAt.formatted(date: .numeric, time: .shortened))に確認）")
        }
        if installed {
            return String(localized: "まだ照合していません（「取得元と照合する」で確かめられます）")
        }
        return String(localized: "まだありません")
    }

    private var filesText: String {
        _ = version; _ = setup.phase
        var parts: [String] = []
        parts.append(OpenH3Model.upsamplerPresent ? String(localized: "潜在アップスケーラ: あり") : String(localized: "潜在アップスケーラ: なし"))
        parts.append(OpenH3Model.refinerPresent ? String(localized: "精緻化LoRA（取り出し済み）: あり") : String(localized: "精緻化LoRA（取り出し済み）: なし"))
        if OpenH3Model.mergedPresent { parts.append(String(localized: "結合ファイル: あり（取り出し後に削除）")) }
        if let turbo = OpenH3Model.turboPath {
            parts.append(turbo.hasPrefix(OpenH3Model.directory)
                         ? String(localized: "Turbo v1.2: ここにダウンロード済み")
                         : String(localized: "Turbo v1.2: LoRAフォルダのものを使います"))
        } else {
            parts.append(String(localized: "Turbo v1.2: なし"))
        }
        return parts.joined(separator: "、")
    }

    private var statusRows: some View {
        Grid(alignment: .leading, horizontalSpacing: H3Spacing.md, verticalSpacing: 6) {
            GridRow {
                Text("取得元").font(.caption).foregroundStyle(palette.textSecondary)
                Text("\(OpenH3Model.repo) revision \(OpenH3Model.revisionShort)（\(OpenH3Model.revisionDate)、モデルカードの宣言: \(OpenH3Model.declaredLicense)）").font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
            GridRow {
                Text("ファイル").font(.caption).foregroundStyle(palette.textSecondary)
                Text(filesText).font(.caption).fixedSize(horizontal: false, vertical: true)
            }
            GridRow {
                Text("照合").font(.caption).foregroundStyle(palette.textSecondary)
                Text(verificationText).font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
            GridRow {
                Text("使用容量").font(.caption).foregroundStyle(palette.textSecondary)
                Text(verbatim: gigabytes({ _ = version; _ = setup.phase; return OpenH3Model.bytesOnDisk() }())).font(.caption)
            }
            if let accepted = OpenH3Model.licenseAcceptedAt {
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
        case .extracting(let done, let total):
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: Double(done), total: Double(max(total, 1)))
                Text("精緻化LoRAを取り出しています（\(done) / \(total) テンソル）…")
                    .font(.caption).foregroundStyle(palette.textSecondary).monospacedDigit()
            }
        case .failed(let message):
            Text(message).font(.caption).foregroundStyle(palette.errorColor)
                .fixedSize(horizontal: false, vertical: true)
        case .done:
            Text("Open-H3 再生成を使えるようになりました。「動画を拡大」で選べます。")
                .font(.caption).foregroundStyle(palette.textSecondary)
        case .idle:
            if !installed {
                let space = downloadSpaceCheck(needed: OpenH3Model.approximateSetupBytes, destination: OpenH3Model.directory)
                Text(space.text).font(.caption)
                    .foregroundStyle(space.fits ? palette.textSecondary : palette.errorColor)
                    .fixedSize(horizontal: false, vertical: true)
                Text("取り出しが終わって結合ファイルを削除すると、残るのは約\(gigabytes(OpenH3Model.approximateRefinerBytes + OpenH3Model.upsamplerBytes + (OpenH3Model.turboNeedsDownload ? OpenH3Model.turboBytes : 0)))です。")
                    .font(.caption).foregroundStyle(palette.textSecondary)
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
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: OpenH3Model.directory)])
            }
            .disabled(!FileManager.default.fileExists(atPath: OpenH3Model.directory))
            Button("すべて削除…") { confirmingRemoval = true }
                .disabled(setup.isBusy || !FileManager.default.fileExists(atPath: OpenH3Model.directory))
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
            try OpenH3Model.removeAll()
            removalError = nil
        } catch {
            removalError = String(localized: "削除できませんでした（詳細: \(error.localizedDescription)）")
        }
        version += 1
    }
}
