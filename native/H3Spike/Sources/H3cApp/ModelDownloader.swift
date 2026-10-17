import Foundation

/// One file inside the MiniMax-H3 Hugging Face repository, as listed by its
/// tree API - a path relative to the repo root (e.g.
/// "FL2VA/transformer/model-00001-of-00030.safetensors") and its exact byte
/// size, which doubles as the completion check for resuming (see
/// ModelDownloader.attemptDownload).
struct HFFileEntry: Equatable {
    let path: String
    let size: Int64
    /// The repository it is in (the GGUF import downloads from others).
    var repo: String = officialModelRepo
    /// Where it goes under the destination, when not at its repo path
    /// (e.g. unsloth's VAE files put where the engine looks for them).
    var localPath: String? = nil
    var local: String { localPath ?? path }

    /// The same file placed under a subfolder of the destination.
    func placed(under folder: String) -> HFFileEntry {
        var entry = self
        entry.localPath = folder + "/" + local
        return entry
    }
}

let officialModelRepo = "MiniMaxAI/MiniMax-H3"

enum ModelDownloadError: LocalizedError {
    case listingFailed(String)
    case insufficientDiskSpace(neededGB: Double, availableGB: Double)
    case httpStatus(path: String, status: Int)
    case sizeMismatch(path: String)

    var errorDescription: String? {
        switch self {
        case .listingFailed(let detail):
            return String(localized: "ファイル一覧の取得に失敗しました。（詳細: \(detail)）")
        case .insufficientDiskSpace(let needed, let available):
            return String(format: String(localized: "保存先の空き容量が足りません。あと約%.1fGB必要です（空き: 約%.1fGB）。"),
                           needed, available)
        case .httpStatus(let path, let status):
            return String(localized: "\(path) のダウンロードに失敗しました。（HTTP \(status)）")
        case .sizeMismatch(let path):
            return String(localized: "\(path) のダウンロード結果のサイズが一致しませんでした。")
        }
    }
}

/// Downloads MiniMax-H3 (FL2VA, optionally Ref2VA) directly from Hugging
/// Face with no external dependency (no huggingface_hub/Python, unlike the
/// dl.py/dl_ref2va.py scripts this mirrors) - plain URLSession, so it ships
/// inside the app itself. Resumable per-file via HTTP Range (matched
/// against the file's already-known final size, not a separate .part file)
/// and retried per-file with backoff, which is what dl.py's own 20-attempt
/// outer retry loop was really compensating for: Hugging Face's large LFS
/// downloads fail intermittently often enough that naive single-shot
/// downloads of a ~70GB checkpoint are not viable.
@MainActor
final class ModelDownloader: ObservableObject {
    enum State: Equatable {
        case idle
        case listing
        case ready
        case downloading
        case failed(String)
        case completed
    }

    @Published private(set) var state: State = .idle
    @Published private(set) var downloadedBytes: Int64 = 0
    @Published private(set) var totalBytes: Int64 = 0
    @Published private(set) var filesCompleted: Int = 0
    @Published private(set) var totalFiles: Int = 0
    @Published private(set) var bytesPerSecond: Double = 0

    private var files: [HFFileEntry] = []
    private var destination: String = ""
    private var task: Task<Void, Never>?
    private var speedTask: Task<Void, Never>?
    private var lastSpeedSample: (bytes: Int64, date: Date)?

    private static let maxConcurrentDownloads = 3
    private static let maxAttemptsPerFile = 5
    /// Headroom beyond the exact byte total - safetensors/index files aside,
    /// the destination volume also needs room for whatever else is on it.
    private static let diskSpaceMarginBytes: Int64 = 2 * 1024 * 1024 * 1024

    // MARK: Listing

    func listFiles(includeRef2VA: Bool) {
        state = .listing
        task = Task {
            do {
                var entries = try await Self.fetchEntries(repo: officialModelRepo, subpath: "FL2VA")
                if includeRef2VA {
                    entries += try await Self.fetchEntries(repo: officialModelRepo, subpath: "Ref2VA")
                }
                guard !Task.isCancelled else { return }
                self.files = entries
                self.totalBytes = entries.reduce(0) { $0 + $1.size }
                self.totalFiles = entries.count
                self.state = .ready
            } catch {
                self.state = .failed(ModelDownloadError.listingFailed(error.localizedDescription).errorDescription!)
            }
        }
    }

    /// Back to nothing listed (switching what to download).
    func reset() {
        task?.cancel()
        files = []
        totalBytes = 0
        totalFiles = 0
        downloadedBytes = 0
        filesCompleted = 0
        state = .idle
    }

    /// Downloads exactly these files (already listed by the caller).
    func use(_ entries: [HFFileEntry]) {
        files = entries
        totalBytes = entries.reduce(0) { $0 + $1.size }
        totalFiles = entries.count
        downloadedBytes = 0
        filesCompleted = 0
        state = .ready
    }

    /// A repository's files under subpath ("" for all of them).
    static func fetchEntries(repo: String, subpath: String) async throws -> [HFFileEntry] {
        let tree = subpath.isEmpty ? "" : "/" + subpath
        guard let url = URL(string: "https://huggingface.co/api/models/\(repo)/tree/main\(tree)?recursive=true") else {
            throw ModelDownloadError.listingFailed("invalid URL")
        }
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse, (200 ..< 300).contains(http.statusCode) else {
            throw ModelDownloadError.listingFailed("HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)")
        }
        struct RawEntry: Decodable { let type: String; let path: String; let size: Int64 }
        let raw = try JSONDecoder().decode([RawEntry].self, from: data)
        return raw.filter { $0.type == "file" }.map { HFFileEntry(path: $0.path, size: $0.size, repo: repo) }
    }

    // MARK: Downloading

    /// Bytes already present at `destination` for the listed files - used
    /// both to preload the progress bar and to size the disk-space check
    /// against only what's actually still missing.
    private func alreadyDownloadedBytes() -> Int64 {
        Self.presentBytes(files, at: destination)
    }

    nonisolated static func presentBytes(_ entries: [HFFileEntry], at destination: String) -> Int64 {
        entries.reduce(into: Int64(0)) { total, entry in
            let size = (try? FileManager.default.attributesOfItem(atPath: destination + "/" + entry.local)[.size] as? Int64) ?? 0
            total += min(size, entry.size)
        }
    }

    /// What is still to download into `destination` (a resumed download
    /// counts only the rest), for checking the space before starting.
    func remainingBytes(at destination: String) -> Int64 {
        totalBytes - Self.presentBytes(files, at: destination)
    }

    /// Headroom the space checks keep free.
    nonisolated static var spaceMargin: Int64 { 2 * 1024 * 1024 * 1024 }

    func startDownload(destination: String) {
        guard state == .ready || isFailedOrIdleWithFiles else { return }
        self.destination = destination
        let already = alreadyDownloadedBytes()
        let remaining = totalBytes - already
        if let free = Self.freeDiskSpace(near: destination), free < remaining + Self.diskSpaceMarginBytes {
            let neededGB = Double(remaining + Self.diskSpaceMarginBytes) / 1_073_741_824
            let availableGB = Double(free) / 1_073_741_824
            state = .failed(ModelDownloadError.insufficientDiskSpace(neededGB: neededGB, availableGB: availableGB).errorDescription!)
            return
        }
        downloadedBytes = already
        filesCompleted = files.filter { entry in
            let size = (try? FileManager.default.attributesOfItem(atPath: destination + "/" + entry.local)[.size] as? Int64) ?? 0
            return size == entry.size
        }.count
        state = .downloading
        startSpeedTracking()

        task = Task {
            do {
                try await self.runDownloads()
                if !Task.isCancelled {
                    self.state = .completed
                }
            } catch is CancellationError {
                // cancel() already set state back to .ready
            } catch {
                self.state = .failed(error.localizedDescription)
            }
            self.stopSpeedTracking()
        }
    }

    private var isFailedOrIdleWithFiles: Bool {
        guard !files.isEmpty else { return false }
        if case .failed = state { return true }
        return false
    }

    func cancel() {
        task?.cancel()
        stopSpeedTracking()
        if case .downloading = state {
            state = .ready
        }
    }

    private func runDownloads() async throws {
        let entries = files
        // Captured once, up front, on the main actor - addNext() below runs
        // inside the task-group closure, which the compiler does not treat
        // as isolated to self even though this method is.
        let destination = self.destination
        try await withThrowingTaskGroup(of: Void.self) { group in
            var iterator = entries.makeIterator()
            var active = 0
            func addNext() {
                guard let entry = iterator.next() else { return }
                active += 1
                group.addTask { try await Self.downloadFile(entry, destination: destination, onProgress: { delta in
                    Task { @MainActor in self.downloadedBytes += delta }
                }) }
            }
            for _ in 0 ..< Self.maxConcurrentDownloads { addNext() }
            while active > 0 {
                try await group.next()
                active -= 1
                await MainActor.run { self.filesCompleted += 1 }
                if Task.isCancelled { throw CancellationError() }
                addNext()
            }
        }
    }

    private static func downloadFile(_ entry: HFFileEntry, destination: String, onProgress: @escaping (Int64) -> Void) async throws {
        let finalPath = destination + "/" + entry.local
        let directory = (finalPath as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)

        var attempt = 0
        while true {
            attempt += 1
            do {
                try await attemptDownload(entry, finalPath: finalPath, onProgress: onProgress)
                return
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                if attempt >= maxAttemptsPerFile {
                    throw error
                }
                try await Task.sleep(nanoseconds: UInt64(min(30.0, pow(2.0, Double(attempt))) * 1_000_000_000))
            }
        }
    }

    /// Resumes in place against `finalPath` itself (no separate .part file):
    /// a short/missing file is grown via an HTTP Range request from its
    /// current length, and a file already at the expected size is treated
    /// as complete and skipped entirely - safe across app relaunches and
    /// download retries alike.
    private static func attemptDownload(_ entry: HFFileEntry, finalPath: String, onProgress: @escaping (Int64) -> Void) async throws {
        var existingSize = (try? FileManager.default.attributesOfItem(atPath: finalPath)[.size] as? Int64) ?? 0
        if existingSize == entry.size { return }
        if existingSize > entry.size {
            try? FileManager.default.removeItem(atPath: finalPath)
            existingSize = 0
        }
        guard let encodedPath = entry.path.addingPercentEncoding(withAllowedCharacters: .urlHFPath) else {
            throw ModelDownloadError.httpStatus(path: entry.path, status: -1)
        }
        guard let url = URL(string: "https://huggingface.co/\(entry.repo)/resolve/main/\(encodedPath)") else {
            throw ModelDownloadError.httpStatus(path: entry.path, status: -1)
        }
        var request = URLRequest(url: url)
        let requestedRange = existingSize > 0
        if requestedRange {
            request.setValue("bytes=\(existingSize)-", forHTTPHeaderField: "Range")
        }

        if !FileManager.default.fileExists(atPath: finalPath) {
            FileManager.default.createFile(atPath: finalPath, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: finalPath))
        try handle.seekToEnd()
        defer { try? handle.close() }

        let delegate = DownloadStreamDelegate(
            path: entry.path, fileHandle: handle, expectedRangeResponse: requestedRange, onProgress: onProgress)
        let session = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)
        defer { session.invalidateAndCancel() }

        do {
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                    delegate.continuation = continuation
                    session.dataTask(with: request).resume()
                }
            } onCancel: {
                session.invalidateAndCancel()
            }
        } catch ModelDownloadStreamError.serverIgnoredRange {
            // The CDN answered 200 instead of 206 - our on-disk bytes and
            // its response are both "from the start", so keep neither:
            // wipe and let the next attempt fetch the whole file cleanly.
            try? FileManager.default.removeItem(atPath: finalPath)
            throw ModelDownloadError.httpStatus(path: entry.path, status: 200)
        }

        let finalSize = (try? FileManager.default.attributesOfItem(atPath: finalPath)[.size] as? Int64) ?? -1
        guard finalSize == entry.size else {
            throw ModelDownloadError.sizeMismatch(path: entry.path)
        }
    }

    /// Free space on the volume `path` would be on (its nearest existing
    /// parent folder).
    nonisolated static func freeDiskSpace(near path: String) -> Int64? {
        var candidate = path
        let fm = FileManager.default
        while !fm.fileExists(atPath: candidate) {
            let parent = (candidate as NSString).deletingLastPathComponent
            if parent.isEmpty || parent == candidate { break }
            candidate = parent
        }
        guard let values = try? URL(fileURLWithPath: candidate)
            .resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]) else { return nil }
        return values.volumeAvailableCapacityForImportantUsage
    }

    // MARK: Speed tracking

    private func startSpeedTracking() {
        lastSpeedSample = (downloadedBytes, Date())
        speedTask = Task { [weak self] in
            while let self, !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                guard let last = self.lastSpeedSample else { continue }
                let now = Date()
                let elapsed = now.timeIntervalSince(last.date)
                if elapsed > 0 {
                    self.bytesPerSecond = Double(self.downloadedBytes - last.bytes) / elapsed
                }
                self.lastSpeedSample = (self.downloadedBytes, now)
            }
        }
    }

    private func stopSpeedTracking() {
        speedTask?.cancel()
        speedTask = nil
        bytesPerSecond = 0
    }
}

private enum ModelDownloadStreamError: Error {
    case serverIgnoredRange
}

/// Streams a single URLSessionDataTask's response body straight to disk in
/// the chunks URLSession delivers them, instead of buffering the whole
/// response in memory (unusable here - individual shards run into the
/// gigabytes) or iterating AsyncBytes one UInt8 at a time (correct but far
/// too slow at this volume).
private final class DownloadStreamDelegate: NSObject, URLSessionDataDelegate {
    private let path: String
    private let fileHandle: FileHandle
    private let expectedRangeResponse: Bool
    private let onProgress: (Int64) -> Void
    var continuation: CheckedContinuation<Void, Error>?

    init(path: String, fileHandle: FileHandle, expectedRangeResponse: Bool, onProgress: @escaping (Int64) -> Void) {
        self.path = path
        self.fileHandle = fileHandle
        self.expectedRangeResponse = expectedRangeResponse
        self.onProgress = onProgress
    }

    func urlSession(
        _ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        if expectedRangeResponse && status == 200 {
            completionHandler(.cancel)
            continuation?.resume(throwing: ModelDownloadStreamError.serverIgnoredRange)
            continuation = nil
            return
        }
        guard (200 ..< 300).contains(status) else {
            completionHandler(.cancel)
            continuation?.resume(throwing: ModelDownloadError.httpStatus(path: path, status: status))
            continuation = nil
            return
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        fileHandle.write(data)
        onProgress(Int64(data.count))
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let continuation else { return }
        self.continuation = nil
        if let error {
            continuation.resume(throwing: error)
        } else {
            continuation.resume()
        }
    }

    func urlSession(_ session: URLSession, didBecomeInvalidWithError error: Error?) {
        guard let continuation else { return }
        self.continuation = nil
        continuation.resume(throwing: error ?? CancellationError())
    }
}

private extension CharacterSet {
    /// Hugging Face repo paths use plain "/"-separated segments; encode
    /// everything else that needs it while leaving the slashes themselves
    /// alone so the path structure survives.
    static let urlHFPath: CharacterSet = {
        var set = CharacterSet.urlPathAllowed
        set.insert(charactersIn: "/")
        return set
    }()
}
