import AppKit
import H3Engine
import SwiftUI

/// The SwiftVR model on this Mac (VideoUpscaler.swiftVRModelDirectory):
/// the files Hugging Face serves, which of them are here, and the .h3vc
/// cache the engine runs from (UPSCALE_PLAN §6.3, §7). The weights are
/// under the SwiftVR Research License (research and other non-commercial
/// use), so none are bundled: the user downloads them here after reading
/// the license, and the app converts them for this Mac's GPU.
enum SwiftVRModel {
    static let repo = "H-oliday/SwiftVR"
    static let repoURL = URL(string: "https://huggingface.co/H-oliday/SwiftVR")!
    static let licenseURL = URL(string: "https://huggingface.co/H-oliday/SwiftVR/blob/main/LICENSE")!
    /// What the engine reads (h3_vsr_inspect_dir / h3_vsr_open).
    static let requiredFiles = ["transformer/config.json", "transformer/diffusion_pytorch_model.safetensors",
                                "reae.safetensors", "prompt_embedding.safetensors"]
    /// Kept beside them: the license text and the model card.
    static let downloadedFiles = requiredFiles + ["LICENSE", "README.md"]
    /// The listing of 2026-10-29 (the live listing decides what is fetched).
    static let approximateDownloadBytes: Int64 = 20_167_245_000
    /// The converted cache: int8 row-quantized on tensor hardware (M5),
    /// else BF16 (UPSCALE_PLAN §6.3).
    static let approximateCacheBytesInt8: Int64 = 4_600_000_000
    static let approximateCacheBytesBF16: Int64 = 8_900_000_000

    static var directory: String { VideoUpscaler.swiftVRModelDirectory }

    static var weightsPresent: Bool {
        requiredFiles.allSatisfy { FileManager.default.fileExists(atPath: directory + "/" + $0) }
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

    /// When the user agreed to the license here (nil: not yet).
    static var licenseAcceptedAt: Date? {
        get { UserDefaults.standard.object(forKey: licenseAcceptedKey) as? Date }
        set { UserDefaults.standard.set(newValue, forKey: licenseAcceptedKey) }
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

/// Downloads and converts the SwiftVR model: one for the whole app, so a
/// download keeps going when the sheet showing it closes. The weights go
/// to SwiftVRModel.directory through ModelDownloader (resumable), then
/// the engine writes the .h3vc cache there (H3VideoSuperResolver.buildCache).
@MainActor
final class SwiftVRSetup: ObservableObject {
    static let shared = SwiftVRSetup()

    enum Phase: Equatable {
        case idle
        case listing
        case downloading
        case converting(done: Int, total: Int)
        case failed(String)
        case done

        /// For GET /api/upscalers.
        var apiText: String {
            switch self {
            case .idle: return "idle"
            case .listing: return "listing"
            case .downloading: return "downloading"
            case .converting(let done, let total): return "converting \(done)/\(total)"
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
        case .listing, .downloading, .converting: return true
        default: return false
        }
    }

    /// Lists the repository, downloads what is missing and converts the
    /// weights for this Mac. Records the license agreement first.
    func start() {
        guard !isBusy else { return }
        SwiftVRModel.licenseAcceptedAt = SwiftVRModel.licenseAcceptedAt ?? Date()
        phase = .listing
        task = Task { await run() }
    }

    /// Stops the download (it resumes next time) or the conversion (its
    /// partial cache is removed).
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
            try await convert()
            guard !Task.isCancelled else { return }
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
        let listed = try await ModelDownloader.fetchEntries(repo: SwiftVRModel.repo, subpath: "")
        let entries = listed.filter { wanted.contains($0.path) }
        let missing = SwiftVRModel.requiredFiles.filter { name in !entries.contains { $0.path == name } }
        guard missing.isEmpty else {
            throw ModelDownloadError.listingFailed(
                String(localized: "\(SwiftVRModel.repo) に、必要なファイル（\(missing.joined(separator: ", "))）が見つかりませんでした"))
        }
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

    private func convert() async throws {
        let output = SwiftVRModel.targetCachePath
        if FileManager.default.fileExists(atPath: output),
           H3VideoSuperResolver.cacheMatches(path: output, modelDirectory: SwiftVRModel.directory) {
            return
        }
        phase = .converting(done: 0, total: 0)
        for try await progress in H3VideoSuperResolver.buildCache(modelDirectory: SwiftVRModel.directory,
                                                                 outputPath: output, int8: SwiftVRModel.usesInt8) {
            phase = .converting(done: progress.completed, total: progress.total)
        }
    }
}

/// モデルを準備: the SwiftVR license, the download and the conversion, and
/// what is installed. Shown by the 拡大 sheet and the model manager.
struct SwiftVRSetupView: View {
    /// A generation (or cache build) is running: the conversion waits.
    var busy = false
    var close: () -> Void

    @ObservedObject private var setup = SwiftVRSetup.shared
    @ObservedObject private var downloader = SwiftVRSetup.shared.downloader
    @Environment(\.colorScheme) private var colorScheme
    @State private var agreed = SwiftVRModel.licenseAcceptedAt != nil
    @State private var confirmingRemoval = false
    @State private var removalError: String?
    /// Bumped after a removal so the installed state is read again.
    @State private var version = 0

    private var palette: H3Palette { H3Palette(colorScheme) }

    private var installed: Bool { _ = version; return SwiftVRModel.weightsPresent }
    private var converted: Bool { _ = version; return SwiftVRModel.cachePath != nil }

    private func gigabytes(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("SwiftVRのモデルを準備").font(.headline)
            Text("SwiftVR（Wan2.2ベースの一段動画超解像モデル）の重みをHugging Faceからダウンロードし、このMacのGPU用に変換します。重みは約\(gigabytes(SwiftVRModel.approximateDownloadBytes))、変換後のキャッシュは約\(gigabytes(SwiftVRModel.approximateCacheBytes))（\(SwiftVRModel.usesInt8 ? "int8" : "BF16")）で、どちらも「\(SwiftVRModel.directory)」に置かれます。")
                .font(.caption).foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            licenseNotice
            Toggle(isOn: $agreed) {
                Text("SwiftVR Research Licenseを読み、研究・個人利用などの非商用の範囲で使うことに同意します")
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
        .frame(width: 560, height: 520)
        .background(palette.canvas)
        .alert("SwiftVRのモデルを削除", isPresented: $confirmingRemoval) {
            Button("削除", role: .destructive) { remove() }
            Button("キャンセル", role: .cancel) {}
        } message: {
            Text("ダウンロードした重みと変換したキャッシュ（計\(gigabytes(SwiftVRModel.bytesOnDisk()))）を削除します。また使うには、もう一度ダウンロードと変換が必要です。")
        }
    }

    private var licenseNotice: some View {
        HStack(alignment: .top, spacing: H3Spacing.sm) {
            Image(systemName: "doc.text.magnifyingglass").foregroundStyle(palette.accent)
            VStack(alignment: .leading, spacing: 4) {
                Text("SwiftVRの重みには「SwiftVR Research License」が適用されます。研究などの非商用の利用に限られ、商用に使うには権利者の書面による許諾が必要です。ダウンロードと利用の前にライセンスを読み、その条件に従ってください。SwiftVRのコードはApache License 2.0です。")
                    .font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: H3Spacing.md) {
                    Link("SwiftVRのライセンス", destination: SwiftVRModel.licenseURL)
                    Link("配布元（Hugging Face）", destination: SwiftVRModel.repoURL)
                }
                .font(.caption)
            }
        }
        .padding(H3Spacing.sm)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(palette.accent.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
    }

    private var statusRows: some View {
        Grid(alignment: .leading, horizontalSpacing: H3Spacing.md, verticalSpacing: 6) {
            GridRow {
                Text("重み").font(.caption).foregroundStyle(palette.textSecondary)
                Text(installed ? String(localized: "ダウンロード済み") : String(localized: "まだありません")).font(.caption)
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
                    Text("\(accepted.formatted(date: .numeric, time: .shortened))に同意").font(.caption)
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
        case .idle:
            if !installed || !converted {
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
            Button("削除…") { confirmingRemoval = true }
                .disabled(setup.isBusy || !FileManager.default.fileExists(atPath: SwiftVRModel.directory))
            Spacer()
            Button("閉じる") { close() }
                .keyboardShortcut(.cancelAction)
            if setup.isBusy {
                Button("中止") { setup.cancel() }
            } else if !(installed && converted) {
                Button(installed ? String(localized: "変換する") : String(localized: "ダウンロードして変換する")) {
                    setup.start()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(!agreed || busy)
            }
        }
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
