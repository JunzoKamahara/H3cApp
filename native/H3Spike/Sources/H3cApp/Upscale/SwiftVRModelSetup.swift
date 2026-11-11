import Combine
import AppKit
import H3Engine
import H3Upscale
import SwiftUI

/// The SwiftVR model on this Mac (VideoUpscaler.swiftVRModelDirectory):
/// the files Hugging Face serves at the pinned revision, which of them are
/// here, whether they were checked against that revision, and the .h3vc
/// cache the engine runs from (UPSCALE_PLAN §6.3, §7, appendix A.5).
///
/// None of the weights are bundled: the user downloads them here after
/// reading the license notice, and the app converts them for this Mac's
/// GPU. The download is pinned to the last revision of H-oliday/SwiftVR
/// whose README declared the weights under the Apache License 2.0
/// (2026-09-24). The repository moved new versions to the "SwiftVR Research
/// License" on 2026-10-10; that license's section 14 ("Historical
/// Releases") leaves versions previously released under the Apache License
/// 2.0 under it, and the three weight files are the same LFS objects at
/// both revisions. After the download the SHA-256 of every file is compared
/// with that revision's object ids and the result is written beside the
/// weights (H3VSR_PROVENANCE.json). The app shows the facts and the
/// license texts; which conditions apply to a use is the user's judgement.
enum SwiftVRModel {
    static let repo = "H-oliday/SwiftVR"
    /// The pinned revision (see above) and what its README declares.
    static let revision = "5e89ae51d17b564b40cb14029002b48a83c2ff50"
    static var revisionShort: String { String(revision.prefix(8)) }
    static let revisionDate = "2026-09-24"
    static let declaredLicense = "Apache-2.0"
    /// When the repository switched new versions to the Research License.
    static let researchLicenseDate = "2026-10-10"

    static let repoURL = URL(string: "https://huggingface.co/H-oliday/SwiftVR/tree/\(revision)")!
    /// The README of the pinned revision: `license: apache-2.0` in its
    /// front matter and "SwiftVR is released under the Apache License 2.0"
    /// in its License section.
    static let declarationURL = URL(string: "https://huggingface.co/H-oliday/SwiftVR/blob/\(revision)/README.md")!
    static let apacheLicenseURL = URL(string: "https://www.apache.org/licenses/LICENSE-2.0")!
    /// The current repository's license (the Research License, whose
    /// section 14 covers the earlier Apache-2.0 releases).
    static let currentLicenseURL = URL(string: "https://huggingface.co/H-oliday/SwiftVR/blob/main/LICENSE")!
    /// The Apache License 2.0 text that README points to (the GitHub
    /// repository's LICENSE, pinned to its commit); the Hugging Face
    /// revision has no LICENSE file of its own. Saved beside the weights
    /// as LICENSE, after its hash is checked.
    static let licenseTextURL = URL(string: "https://raw.githubusercontent.com/H-oliday/SwiftVR/6f54009b0f901ef72d03312b1de39eb2ac1f4939/LICENSE")!
    static let licenseTextSHA256 = "3c4b06673cfcd41f8967e42630b86457fabceedb2d8521f81b9396430af349f1"
    static let licenseFile = "LICENSE"

    /// The 20 GB F32 transformer weights: only converting reads them, so
    /// they can be removed once converted (h3_vsr_remove_original_weights).
    static let transformerWeights = "transformer/diffusion_pytorch_model.safetensors"
    /// What converting reads (h3_vsr_inspect_dir / h3_vsr_build_cache).
    static let requiredFiles = ["transformer/config.json", transformerWeights,
                                "reae.safetensors", "prompt_embedding.safetensors"]
    /// Kept beside them: the model card of that revision, which carries the
    /// license declaration.
    static let downloadedFiles = requiredFiles + ["README.md"]
    /// SHA-256 of each downloaded file at the pinned revision: the LFS
    /// object ids Hugging Face lists for the weights, computed for the two
    /// small files kept in git (UPSCALE_PLAN appendix A.5). Checked against
    /// the listing before the download and against the files after it.
    static let expectedSHA256: [String: String] = [
        "transformer/config.json": "dc00d9866e72cf77db6b531aaa33be4dc7148fef9338442bd0ae9181f7075e9b",
        transformerWeights: "f7ade5b8f7f4ff8b4e26a581772ebe5bcfb6a619ece2dd3483c5395c2d7e1a31",
        "reae.safetensors": "c915205d1833677b6887e2fdf675499d3fc781af0c644c99330f7d22fd855514",
        "prompt_embedding.safetensors": "cc4cf7b9aa9def4026bb5952b8aaec846ffc83eee43cafff0d3796b7e9fdf922",
        "README.md": "e27e571c31a3370a22b9abf797a93f1783cdf044b928a1f3a03460d9381c91b0",
    ]
    /// The listing of that revision (fixed: the revision is).
    static let approximateDownloadBytes: Int64 = 20_167_245_000
    /// The converted cache: int8 row-quantized on tensor hardware (M5),
    /// else BF16 (UPSCALE_PLAN §6.3).
    static let approximateCacheBytesInt8: Int64 = 4_600_000_000
    static let approximateCacheBytesBF16: Int64 = 8_900_000_000

    static var directory: String { VideoUpscaler.swiftVRModelDirectory }

    static var weightsPresent: Bool {
        requiredFiles.allSatisfy { FileManager.default.fileExists(atPath: directory + "/" + $0) }
    }

    /// The transformer weights were removed after conversion (the engine's
    /// record of them is here, the file is not).
    static var originalRemoved: Bool {
        !FileManager.default.fileExists(atPath: directory + "/" + transformerWeights)
            && FileManager.default.fileExists(atPath: directory + "/transformer/H3VSR_REMOVED")
    }

    static func fileSize(_ relativePath: String) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: directory + "/" + relativePath)[.size]) as? Int64
    }

    /// Bytes of the transformer weights when they are here.
    static var originalBytes: Int64? { fileSize(transformerWeights) }

    /// Whether the original weights can be removed now: here, and a
    /// converted cache runs without them.
    static var originalRemovable: Bool { weightsPresent && cachePath != nil }

    /// The setup's option: remove the original weights once converted.
    static let removeOriginalAfterConversionKey = "h3c-app.swiftVRRemoveOriginalAfterConversion"

    /// Deletes the transformer weights, keeping the converted cache usable.
    /// Returns the bytes freed.
    @discardableResult
    static func removeOriginalWeights() throws -> Int64 {
        Int64(try H3VideoSuperResolver.removeOriginalWeights(modelDirectory: directory))
    }

    /// The cache this Mac can run from, when the weights are converted.
    static var cachePath: String? { VideoUpscaler.swiftVRCachePath }

    /// Whether this GPU takes the int8 path (tensor hardware), from the
    /// installed model when there is one, else from Metal directly.
    static var usesInt8: Bool {
        if let info = try? H3VideoSuperResolver.inspect(modelDirectory: directory) { return info.int8Supported }
        return H3VideoSuperResolver.int8Supported
    }

    /// Where the cache for this Mac goes.
    static var targetCachePath: String {
        H3VideoSuperResolver.cachePath(modelDirectory: directory, int8: usesInt8)
    }

    static var approximateCacheBytes: Int64 { usesInt8 ? approximateCacheBytesInt8 : approximateCacheBytesBF16 }

    private static let licenseAcceptedKey = "h3c-app.swiftVRLicenseAcceptedAt"

    /// When the user confirmed the license notice here (nil: not yet).
    static var licenseAcceptedAt: Date? {
        get { UserDefaults.standard.object(forKey: licenseAcceptedKey) as? Date }
        set { UserDefaults.standard.set(newValue, forKey: licenseAcceptedKey) }
    }

    // MARK: Provenance record

    /// Written beside the weights once their hashes were checked: which
    /// repository revision they are, what it declared, and each file's
    /// SHA-256 and size. Read back by the setup sheet and GET /api/upscalers;
    /// the engine never reads it.
    static let provenanceFile = "H3VSR_PROVENANCE.json"

    struct ProvenanceEntry: Codable, Equatable {
        var sha256: String
        var bytes: Int64
    }

    struct Provenance: Codable, Equatable {
        var repo: String
        var revision: String
        var revisionDate: String
        var declaredLicense: String
        var declaration: String
        var licenseText: String
        var files: [String: ProvenanceEntry]
        var verifiedAt: Date
        var note: String
    }

    static var provenance: Provenance? {
        guard let data = FileManager.default.contents(atPath: directory + "/" + provenanceFile) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(Provenance.self, from: data)
    }

    /// Whether the files here are recorded as checked against the pinned
    /// revision: a record for this revision with the expected hashes, and
    /// every required file still at its recorded size (or the transformer
    /// weights gone after conversion, with the record kept).
    static var verified: Bool {
        guard let record = provenance, record.revision == revision else { return false }
        for name in requiredFiles {
            guard let entry = record.files[name], entry.sha256 == expectedSHA256[name] else { return false }
            if let size = fileSize(name) {
                if size != entry.bytes { return false }
            } else if !(name == transformerWeights && originalRemoved) {
                return false
            }
        }
        return true
    }

    enum VerificationError: LocalizedError {
        case listing(file: String, listed: String, expected: String)
        case mismatch(file: String, expected: String, actual: String)
        case licenseText(String)

        var errorDescription: String? {
            switch self {
            case .listing(let file, let listed, let expected):
                return String(localized: "配布元の一覧にある「\(file)」のSHA-256（\(String(listed.prefix(16)))…）が、Apache License 2.0と宣言されていた版の記録（\(String(expected.prefix(16)))…）と一致しません。ダウンロードは始めていません。")
            case .mismatch(let file, let expected, let actual):
                return String(localized: "「\(file)」のSHA-256（\(String(actual.prefix(16)))…）が、Apache License 2.0と宣言されていた版の記録（\(String(expected.prefix(16)))…）と一致しません。このファイルは削除しました。もう一度ダウンロードしてください。")
            case .licenseText(let message):
                return String(localized: "ライセンス本文（Apache License 2.0）を取得できませんでした（詳細: \(message)）")
            }
        }
    }

    /// Checks the listing's object ids against the pinned ones before any
    /// byte is downloaded (a typo here, or a repository whose revision no
    /// longer serves these objects, stops the setup at once).
    static func checkListing(_ entries: [HFFileEntry]) throws {
        for entry in entries {
            guard let listed = entry.sha256, let expected = expectedSHA256[entry.path] else { continue }
            if listed != expected { throw VerificationError.listing(file: entry.path, listed: listed, expected: expected) }
        }
    }

    /// Hashes every downloaded file and compares it with the pinned
    /// revision's; fetches the Apache License 2.0 text when it is not here
    /// yet; then writes the provenance record. A file whose hash differs is
    /// deleted (the resumable download treats a file at the expected size
    /// as complete, so it could never be mended otherwise) and named in the
    /// error. `progress` gets (bytes hashed, bytes to hash).
    static func verify(progress: @escaping @Sendable (Int64, Int64) -> Void) async throws {
        let directory = self.directory
        let files = downloadedFiles.filter { FileManager.default.fileExists(atPath: directory + "/" + $0) }
        let total = files.reduce(Int64(0)) { $0 + (fileSize($1) ?? 0) }
        var done: Int64 = 0
        var record: [String: ProvenanceEntry] = [:]
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
                throw VerificationError.mismatch(file: name, expected: expected, actual: actual)
            }
            record[name] = ProvenanceEntry(sha256: actual, bytes: fileSize(name) ?? 0)
        }
        try await fetchLicenseText()
        let provenance = Provenance(
            repo: repo, revision: revision, revisionDate: revisionDate, declaredLicense: declaredLicense,
            declaration: "README.md of this revision: front matter `license: apache-2.0`, section License",
            licenseText: licenseFile, files: record, verifiedAt: Date(),
            note: "SwiftVR Research License 1.0 (effective \(researchLicenseDate)), section 14 Historical Releases: model versions previously released under the Apache License 2.0 remain subject to it. The weight files are the same LFS objects at that revision and at the current one.")
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(provenance).write(to: URL(fileURLWithPath: directory + "/" + provenanceFile), options: .atomic)
    }

    /// The Apache License 2.0 text beside the weights, checked by hash;
    /// replaces a LICENSE of another version (an earlier setup saved the
    /// current repository's Research License there).
    private static func fetchLicenseText() async throws {
        let path = directory + "/" + licenseFile
        if let data = FileManager.default.contents(atPath: path), FileHash.sha256(of: data) == licenseTextSHA256 { return }
        do {
            let (data, response) = try await URLSession.shared.data(from: licenseTextURL)
            guard let http = response as? HTTPURLResponse, (200 ..< 300).contains(http.statusCode) else {
                throw VerificationError.licenseText("HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)")
            }
            guard FileHash.sha256(of: data) == licenseTextSHA256 else {
                throw VerificationError.licenseText(String(localized: "取得した本文のSHA-256が記録と一致しません"))
            }
            try data.write(to: URL(fileURLWithPath: path), options: .atomic)
        } catch let error as VerificationError {
            throw error
        } catch {
            throw VerificationError.licenseText(error.localizedDescription)
        }
    }

    /// Bytes of the files under the model folder (weights and caches).
    static func bytesOnDisk() -> Int64 {
        guard let enumerator = FileManager.default.enumerator(atPath: directory) else { return 0 }
        var total: Int64 = 0
        for case let item as String in enumerator {
            total += (try? FileManager.default.attributesOfItem(atPath: directory + "/" + item)[.size] as? Int64) ?? 0
        }
        return total
    }

    /// Removes the whole model folder (weights, license text and caches).
    static func removeAll() throws {
        try FileManager.default.removeItem(atPath: directory)
    }
}

/// Downloads, checks and converts the SwiftVR model: one for the whole app,
/// so a download keeps going when the sheet showing it closes. The weights
/// go to SwiftVRModel.directory through ModelDownloader (resumable, pinned
/// to SwiftVRModel.revision), their hashes are compared with that
/// revision's (SwiftVRModel.verify), then the engine writes the .h3vc cache
/// there (H3VideoSuperResolver.buildCache).
@MainActor
final class SwiftVRSetup: ObservableObject {
    static let shared = SwiftVRSetup()

    enum Phase: Equatable {
        case idle
        case listing
        case downloading
        case verifying(done: Int64, total: Int64)
        case converting(done: Int, total: Int)
        case failed(String)
        case done

        /// For GET /api/upscalers.
        var apiText: String {
            switch self {
            case .idle: return "idle"
            case .listing: return "listing"
            case .downloading: return "downloading"
            case .verifying: return "verifying"
            case .converting(let done, let total): return "converting \(done)/\(total)"
            case .failed(let message): return "failed: \(message)"
            case .done: return "done"
            }
        }
    }

    @Published private(set) var phase: Phase = .idle
    /// Why removing the original weights after the conversion failed (the
    /// conversion itself succeeded).
    @Published private(set) var removalNote: String?
    let downloader = ModelDownloader()
    private var task: Task<Void, Never>?

    var isBusy: Bool {
        switch phase {
        case .listing, .downloading, .verifying, .converting: return true
        default: return false
        }
    }

    /// Lists the pinned revision, downloads what is missing, checks the
    /// files against it and converts the weights for this Mac; each step
    /// is skipped when already done. Records the license confirmation first.
    func start() {
        guard !isBusy else { return }
        SwiftVRModel.licenseAcceptedAt = SwiftVRModel.licenseAcceptedAt ?? Date()
        phase = .listing
        task = Task { await run() }
    }

    /// Stops the download (it resumes next time), the check or the
    /// conversion (its partial cache is removed).
    func cancel() {
        task?.cancel()
        downloader.cancel()
        if isBusy { phase = .idle }
    }

    private func run() async {
        do {
            if !SwiftVRModel.weightsPresent || !Self.allFilesComplete() {
                try await download()
            }
            guard !Task.isCancelled else { return }
            if !SwiftVRModel.verified && SwiftVRModel.weightsPresent {
                phase = .verifying(done: 0, total: 0)
                try await SwiftVRModel.verify { done, total in
                    Task { @MainActor in
                        if case .verifying = self.phase { self.phase = .verifying(done: done, total: total) }
                    }
                }
            }
            guard !Task.isCancelled else { return }
            let converted = try await convert()
            guard !Task.isCancelled else { return }
            removalNote = nil
            if converted, UserDefaults.standard.bool(forKey: SwiftVRModel.removeOriginalAfterConversionKey),
               SwiftVRModel.originalRemovable {
                do {
                    try SwiftVRModel.removeOriginalWeights()
                } catch {
                    removalNote = String(localized: "変換は終わりましたが、元の重みを削除できませんでした（詳細: \(error.localizedDescription)）")
                }
            }
            phase = .done
        } catch is CancellationError {
            phase = .idle
        } catch H3VSRError.cancelled {
            phase = .idle
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    /// Whether every downloaded file is here at its listed size (nil when
    /// the listing is not known yet: false is assumed then).
    private static var listedEntries: [HFFileEntry] = []
    private static func allFilesComplete() -> Bool {
        guard !listedEntries.isEmpty else { return false }
        return listedEntries.allSatisfy { entry in
            let size = (try? FileManager.default.attributesOfItem(atPath: SwiftVRModel.directory + "/" + entry.local)[.size] as? Int64) ?? -1
            return size == entry.size
        }
    }

    private func download() async throws {
        let wanted = Set(SwiftVRModel.downloadedFiles)
        let listed = try await ModelDownloader.fetchEntries(repo: SwiftVRModel.repo, subpath: "", revision: SwiftVRModel.revision)
        let entries = listed.filter { wanted.contains($0.path) }
        let missing = SwiftVRModel.requiredFiles.filter { name in !entries.contains { $0.path == name } }
        guard missing.isEmpty else {
            throw ModelDownloadError.listingFailed(
                String(localized: "\(SwiftVRModel.repo)（revision \(SwiftVRModel.revisionShort)）に、必要なファイル（\(missing.joined(separator: ", "))）が見つかりませんでした"))
        }
        try SwiftVRModel.checkListing(entries)
        Self.listedEntries = entries
        try Task.checkCancellation()
        downloader.use(entries)
        downloader.startDownload(destination: SwiftVRModel.directory)
        phase = .downloading
        // The downloader runs its own task; wait for it to settle.
        for await state in downloader.$state.values {
            switch state {
            case .completed: return
            case .failed(let message): throw ModelDownloadError.listingFailed(message)
            case .ready, .idle:
                // Stopped (cancel()) before it finished.
                if phase == .downloading { throw CancellationError() }
                return
            default: continue
            }
        }
    }

    /// Returns whether a conversion ran (false: a matching cache was here).
    private func convert() async throws -> Bool {
        let output = SwiftVRModel.targetCachePath
        if FileManager.default.fileExists(atPath: output),
           H3VideoSuperResolver.cacheMatches(path: output, modelDirectory: SwiftVRModel.directory) {
            return false
        }
        phase = .converting(done: 0, total: 0)
        for try await progress in H3VideoSuperResolver.buildCache(modelDirectory: SwiftVRModel.directory,
                                                                 outputPath: output, int8: SwiftVRModel.usesInt8) {
            phase = .converting(done: progress.completed, total: progress.total)
        }
        return true
    }
}

/// モデルを準備: the license notice, the download, the check against the
/// pinned revision, the conversion, and what is installed. Shown by the
/// 拡大 sheet and the model manager.
struct SwiftVRSetupView: View {
    /// A generation (or cache build) is running: the conversion waits.
    var busy = false
    var close: () -> Void

    @ObservedObject private var setup = SwiftVRSetup.shared
    @ObservedObject private var downloader = SwiftVRSetup.shared.downloader
    @Environment(\.colorScheme) private var colorScheme
    @State private var agreed = SwiftVRModel.licenseAcceptedAt != nil
    @State private var confirmingRemoval = false
    @State private var confirmingOriginalRemoval = false
    @AppStorage(SwiftVRModel.removeOriginalAfterConversionKey) private var removeOriginalAfterConversion = false
    @State private var removalError: String?
    /// Bumped after a removal so the installed state is read again.
    @State private var version = 0

    private var palette: H3Palette { H3Palette(colorScheme) }

    private var installed: Bool { _ = version; return SwiftVRModel.weightsPresent }
    private var originalRemoved: Bool { _ = version; return SwiftVRModel.originalRemoved }
    private var converted: Bool { _ = version; return SwiftVRModel.cachePath != nil }
    private var verified: Bool { _ = version; _ = setup.phase; return SwiftVRModel.verified }

    private func gigabytes(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("SwiftVRのモデルを準備").font(.headline)
            Text("SwiftVR（Wan2.2ベースの一段動画超解像モデル）の重みを、Hugging Faceの配布元（\(SwiftVRModel.repo)）のうち、READMEでApache License 2.0と宣言されていた版（revision \(SwiftVRModel.revisionShort)、\(SwiftVRModel.revisionDate)）からダウンロードし、各ファイルのSHA-256をその版の記録と照合してから、このMacのGPU用に変換します。重みは約\(gigabytes(SwiftVRModel.approximateDownloadBytes))、変換後のキャッシュは約\(gigabytes(SwiftVRModel.approximateCacheBytes))（\(SwiftVRModel.usesInt8 ? "int8" : "BF16")）で、どちらも「\(SwiftVRModel.directory)」に置かれます。")
                .font(.caption).foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            licenseNotice
            Toggle(isOn: $agreed) {
                Text("上の説明を読み、取得する版に適用されるApache License 2.0の条件に従って使います")
                    .font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .disabled(setup.isBusy)
            if !converted {
                Toggle(isOn: $removeOriginalAfterConversion) {
                    Text("変換が終わったら、元の重み（約\(gigabytes(SwiftVRModel.approximateDownloadBytes))）を削除する")
                        .font(.caption)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .disabled(setup.isBusy)
            }
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
        .frame(width: 620, height: 640)
        .background(palette.canvas)
        .alert("SwiftVRのモデルを削除", isPresented: $confirmingRemoval) {
            Button("削除", role: .destructive) { remove() }
            Button("キャンセル", role: .cancel) {}
        } message: {
            Text("ダウンロードした重みと変換したキャッシュ（計\(gigabytes(SwiftVRModel.bytesOnDisk()))）を削除します。また使うには、もう一度ダウンロードと変換が必要です。")
        }
        .alert("元の重みを削除", isPresented: $confirmingOriginalRemoval) {
            Button("削除", role: .destructive) { removeOriginal() }
            Button("キャンセル", role: .cancel) {}
        } message: {
            Text("ダウンロードした元の重み（\(gigabytes(SwiftVRModel.originalBytes ?? SwiftVRModel.approximateDownloadBytes))）を削除し、このMac用に変換したキャッシュだけを残します。SwiftVRはこのまま使えます。アプリの更新でキャッシュの形式が変わったときなど、変換し直すときは、もう一度ダウンロードが必要です。")
        }
    }

    private var licenseNotice: some View {
        HStack(alignment: .top, spacing: H3Spacing.sm) {
            Image(systemName: "doc.text.magnifyingglass").foregroundStyle(palette.accent)
            VStack(alignment: .leading, spacing: 4) {
                Text("ダウンロードする版のREADMEは、SwiftVRの重みをApache License 2.0で公開すると宣言しています（コードもApache License 2.0）。配布元は2026年10月10日に、新しい版のライセンスを「SwiftVR Research License」（研究・非商用）に切り替えましたが、その第14条は、以前にApache License 2.0で公開された版は引き続きApache License 2.0に従うと定めています。このアプリはその以前の版だけを取得し、ファイルが同一であることをSHA-256で確かめます。Apache License 2.0には、著作権表示とライセンス文の保持、変更した場合の明示などの条件があります。あなたの利用にどの条件が当てはまるかは、利用者自身（必要なら専門家）が判断してください。")
                    .font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
                // Two rows: four links do not fit one line of the sheet.
                HStack(spacing: H3Spacing.md) {
                    Link("宣言のあるREADME（その版）", destination: SwiftVRModel.declarationURL)
                    Link("Apache License 2.0の本文", destination: SwiftVRModel.apacheLicenseURL)
                }
                .font(.caption)
                HStack(spacing: H3Spacing.md) {
                    Link("現在のSwiftVR Research License（第14条）", destination: SwiftVRModel.currentLicenseURL)
                    Link("配布元（Hugging Face）", destination: SwiftVRModel.repoURL)
                }
                .font(.caption)
            }
        }
        .padding(H3Spacing.sm)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(palette.accent.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
    }

    private var verificationText: String {
        if verified, let record = SwiftVRModel.provenance {
            return String(localized: "SHA-256がその版の記録と一致（\(record.verifiedAt.formatted(date: .numeric, time: .shortened))に確認）")
        }
        if originalRemoved && !installed {
            return String(localized: "照合できません（元の重みは削除済みで、照合の記録がありません）")
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
                Text("\(SwiftVRModel.repo) revision \(SwiftVRModel.revisionShort)（\(SwiftVRModel.revisionDate)、READMEの宣言: Apache License 2.0）").font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
            GridRow {
                Text("重み").font(.caption).foregroundStyle(palette.textSecondary)
                Text(installed ? String(localized: "ダウンロード済み")
                     : originalRemoved ? String(localized: "変換したあとで削除しました（変換し直すときは、もう一度ダウンロードします）")
                     : String(localized: "まだありません")).font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
            GridRow {
                Text("照合").font(.caption).foregroundStyle(palette.textSecondary)
                Text(verificationText).font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
            GridRow {
                Text("変換").font(.caption).foregroundStyle(palette.textSecondary)
                Text(converted
                     ? String(localized: "このMac用に変換済み（\(SwiftVRModel.usesInt8 ? "int8" : "BF16")）")
                     : String(localized: "まだ変換していません")).font(.caption)
            }
            GridRow {
                Text("使用容量").font(.caption).foregroundStyle(palette.textSecondary)
                Text(verbatim: gigabytes({ _ = version; return SwiftVRModel.bytesOnDisk() }())).font(.caption)
            }
            if let accepted = SwiftVRModel.licenseAcceptedAt {
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
        case .converting(let done, let total):
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: Double(done), total: Double(max(total, 1)))
                Text(total > 0
                     ? String(localized: "このMac用に変換しています（\(done)/\(total)）…")
                     : String(localized: "このMac用に変換しています…"))
                    .font(.caption).foregroundStyle(palette.textSecondary)
            }
        case .failed(let message):
            Text(message).font(.caption).foregroundStyle(palette.errorColor)
                .fixedSize(horizontal: false, vertical: true)
        case .done:
            Text("SwiftVRを使えるようになりました。「動画を拡大」で選べます。")
                .font(.caption).foregroundStyle(palette.textSecondary)
            if let note = setup.removalNote {
                Text(note).font(.caption).foregroundStyle(palette.errorColor)
                    .fixedSize(horizontal: false, vertical: true)
            }
        case .idle:
            if !converted {
                let needed = (installed ? 0 : SwiftVRModel.approximateDownloadBytes) + (converted ? 0 : SwiftVRModel.approximateCacheBytes)
                let space = downloadSpaceCheck(needed: needed, destination: SwiftVRModel.directory)
                Text(space.text).font(.caption)
                    .foregroundStyle(space.fits ? palette.textSecondary : palette.errorColor)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if busy {
                Text("生成中は変換できません。生成が終わってから始めてください。")
                    .font(.caption).foregroundStyle(.orange)
            }
        }
    }

    private var footer: some View {
        HStack {
            Button("Finderで表示") {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: SwiftVRModel.directory)])
            }
            .disabled(!FileManager.default.fileExists(atPath: SwiftVRModel.directory))
            if converted && installed {
                Button("元の重みを削除…") { confirmingOriginalRemoval = true }
                    .disabled(setup.isBusy)
                    .help("変換したキャッシュだけを残して、ダウンロードした約20GBの重みを削除します")
            }
            Button("すべて削除…") { confirmingRemoval = true }
                .disabled(setup.isBusy || !FileManager.default.fileExists(atPath: SwiftVRModel.directory))
            Spacer()
            Button("閉じる") { close() }
                .keyboardShortcut(.cancelAction)
            if setup.isBusy {
                Button("中止") { setup.cancel() }
            } else if !converted {
                Button(installed ? String(localized: "変換する") : String(localized: "ダウンロードして変換する")) {
                    setup.start()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(!agreed || busy)
            } else if installed && !verified {
                // Converted before the download was pinned: the same chain
                // fetches the pinned revision's model card, hashes the
                // files that are here and writes the record (nothing is
                // converted again; the cache already matches).
                Button("取得元と照合する") { setup.start() }
                    .buttonStyle(.borderedProminent)
                    .disabled(!agreed)
                    .help("ここにあるファイルのSHA-256を、Apache License 2.0と宣言されていた版の記録と照合し、結果をモデルのフォルダに記録します")
            }
        }
    }

    private func removeOriginal() {
        do {
            try SwiftVRModel.removeOriginalWeights()
            removalError = nil
        } catch {
            removalError = String(localized: "元の重みを削除できませんでした（詳細: \(error.localizedDescription)）")
        }
        version += 1
    }

    private func remove() {
        do {
            try SwiftVRModel.removeAll()
            removalError = nil
        } catch {
            removalError = String(localized: "削除できませんでした（詳細: \(error.localizedDescription)）")
        }
        version += 1
    }
}

/// SwiftVR as the upscalers see its model (UpscalerModel): the model manager's
/// row, the sheet's button and GET /api/upscalers read it here.
final class SwiftVRKit: UpscalerModel {
    static let shared = SwiftVRKit()
    let id = "swiftVR"
    var title: String { VideoUpscaler.swiftVR.label }
    var directory: String { SwiftVRModel.directory }
    var isReady: Bool { VideoUpscaler.swiftVR.isAvailable }
    @MainActor var isBusy: Bool { SwiftVRSetup.shared.isBusy }
    @MainActor var phaseText: String { SwiftVRSetup.shared.phase.apiText }
    @MainActor var phaseChanged: AnyPublisher<Void, Never> {
        SwiftVRSetup.shared.$phase.dropFirst().map { _ in () }.receive(on: DispatchQueue.main).eraseToAnyPublisher()
    }

    @MainActor var statusText: String {
        if SwiftVRSetup.shared.isBusy {
            return String(localized: "準備中です（\(SwiftVRSetup.shared.phase.apiText)）")
        }
        if VideoUpscaler.swiftVR.isAvailable {
            let size = ByteCountFormatter.string(fromByteCount: SwiftVRModel.bytesOnDisk(), countStyle: .file)
            let check = SwiftVRModel.verified ? "" : String(localized: "。取得元との照合はまだです（「管理…」から）")
            if SwiftVRModel.originalRemoved {
                return String(localized: "使えます（\(SwiftVRModel.usesInt8 ? "int8" : "BF16")で変換済み、元の重みは削除済み、\(size)）") + check
            }
            return String(localized: "使えます（\(SwiftVRModel.usesInt8 ? "int8" : "BF16")で変換済み、\(size)）") + check
        }
        if SwiftVRModel.originalRemoved {
            return String(localized: "元の重みを削除したあとで、キャッシュが使えなくなりました。もう一度ダウンロードと変換が必要です。")
        }
        if SwiftVRModel.weightsPresent { return String(localized: "重みはありますが、このMac用に変換されていません。") }
        return String(localized: "まだありません。約20GBのダウンロードと変換が必要です。")
    }


    @MainActor var apiFields: [String: Any] {
        var item: [String: Any] = [:]
            item["model_directory"] = SwiftVRModel.directory
            item["model_installed"] = SwiftVRModel.weightsPresent
            item["original_removed"] = SwiftVRModel.originalRemoved
            item["cache"] = SwiftVRModel.cachePath as Any
            item["license_accepted"] = SwiftVRModel.licenseAcceptedAt != nil
            item["setup"] = SwiftVRSetup.shared.phase.apiText
            // The pinned revision, what it declared, and whether the
            // files here were checked against it (appendix A.5).
            item["revision"] = SwiftVRModel.revision
            item["declared_license"] = SwiftVRModel.declaredLicense
            item["verified"] = SwiftVRModel.verified
            item["verified_at"] = SwiftVRModel.verified
                ? SwiftVRModel.provenance.map { ISO8601DateFormatter().string(from: $0.verifiedAt) } as Any
                : NSNull()
        return item
    }

    @MainActor func setupView(busy: Bool, close: @escaping () -> Void) -> AnyView {
        AnyView(SwiftVRSetupView(busy: busy, close: close))
    }
}
