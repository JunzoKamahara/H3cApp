import Foundation
import H3Engine

// What the files of a Hugging Face repository are, read from their first
// bytes rather than guessed from names (each publisher names files its own
// way): a GGUF's header (key/values and tensor table) and a safetensors
// file's JSON header are fetched with HTTP ranges and inspected - the GGUF
// with the engine's own h3_gguf_inspect on a sparse local stand-in. Results
// are kept on disk, keyed by repository, path and size.

/// One file of a repository, as far as the import is concerned.
struct HFProbedFile: Codable, Sendable, Equatable {
    enum Kind: String, Codable, Sendable { case dit, textEncoder, vision, videoVAE, audioVAE, other }

    let repo: String
    let path: String
    let size: Int64
    var kind: Kind = .other
    var format = "GGUF"
    var mainType = ""
    var curveAdaLN = false
    var denseAdaLN = false
    var textLayers = 0
    var hasVision = false
    /// Why the app can't use it; nil when it can. Kept as a reason, not
    /// text, so the cached result reads in whatever language the app is in.
    var problem: Problem?
    var unsupported: String? { problem?.message }

    enum Problem: Codable, Sendable, Equatable {
        case unreadable(String)
        case split(Int)
        case quantizations([String])
        case numberFormats([String])
        case safetensorsPart
        case unknownPart

        var message: String {
            switch self {
            case .unreadable(let detail): return String(localized: "先頭を読めませんでした（\(detail)）")
            case .split(let count): return String(localized: "分割されたGGUF（\(count)ファイル）は未対応です")
            case .quantizations(let names):
                return String(localized: "未対応の量子化形式（\(names.joined(separator: itemSeparator))）を含みます")
            case .numberFormats(let names):
                return String(localized: "未対応の数値形式（\(names.joined(separator: itemSeparator))）を含みます")
            case .safetensorsPart: return String(localized: "safetensors形式です（取り込めるのはGGUF）")
            case .unknownPart: return String(localized: "MiniMax H3の部品として認識できません")
            }
        }
    }

    var fileName: String { (path as NSString).lastPathComponent }
    /// DiT files of FL2VA and Ref2VA have the same tensors; only the name
    /// tells them apart.
    var isRef2VA: Bool { path.lowercased().contains("ref2v") }
}

enum HFRepoProbe {
    /// The repository's model files, each inspected (cached ones are not
    /// fetched again). `progress` gets (done, total).
    static func probe(repo: String, progress: @escaping @Sendable (Int, Int) -> Void = { _, _ in })
        async throws -> [HFProbedFile] {
        let entries = try await ModelDownloader.fetchEntries(repo: repo, subpath: "")
            .filter { $0.path.hasSuffix(".gguf") || $0.path.hasSuffix(".safetensors") }
        let cache = ProbeCache.shared
        var results: [HFProbedFile?] = Array(repeating: nil, count: entries.count)
        var pending: [Int] = []
        for (index, entry) in entries.enumerated() {
            if let hit = await cache.get(repo: repo, path: entry.path, size: entry.size) {
                results[index] = hit
            } else {
                pending.append(index)
            }
        }
        var done = entries.count - pending.count
        progress(done, entries.count)
        // A few at a time: each is one or two small range requests.
        try await withThrowingTaskGroup(of: (Int, HFProbedFile).self) { group in
            var queue = pending.makeIterator()
            func next() {
                guard let index = queue.next() else { return }
                let entry = entries[index]
                group.addTask { (index, await probeFile(repo: repo, entry: entry)) }
            }
            for _ in 0 ..< 4 { next() }
            while let (index, file) = try await group.next() {
                results[index] = file
                await cache.put(file)
                done += 1
                progress(done, entries.count)
                next()
            }
        }
        await cache.save()
        return results.compactMap { $0 }
    }

    private static func probeFile(repo: String, entry: HFFileEntry) async -> HFProbedFile {
        var file = HFProbedFile(repo: repo, path: entry.path, size: entry.size)
        do {
            if entry.path.hasSuffix(".gguf") {
                try await probeGGUF(&file)
            } else {
                try await probeSafetensors(&file)
            }
        } catch {
            file.problem = .unreadable(error.localizedDescription)
        }
        return file
    }

    // MARK: GGUF

    private static func probeGGUF(_ file: inout HFProbedFile) async throws {
        // Grow the fetched prefix until the whole header is in it (a
        // llama.cpp text encoder carries its tokenizer, several MB).
        var length: Int64 = 1 << 20
        var data = Data()
        var scan: GGUFScan?
        while scan == nil {
            data = try await fetchRange(repo: file.repo, path: file.path, from: 0, count: min(length, file.size))
            scan = GGUFScan(data)
            if scan == nil {
                guard length < 128 << 20, length < file.size else { throw ProbeError.headerTooLarge }
                length *= 8
            }
        }
        guard let scan else { return }
        let unsupportedTypes = scan.types.subtracting(supportedGGMLTypes)
        if let split = scan.splitCount, split > 1 {
            file.problem = .split(split)
        } else if !unsupportedTypes.isEmpty {
            file.problem = .quantizations(unsupportedTypes.sorted().map(ggmlTypeName))
        }
        // The engine's own inspection, on a sparse file of the real size
        // holding just the header.
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("H3cApp-probe-\(UUID().uuidString).gguf")
        defer { try? FileManager.default.removeItem(at: temp) }
        FileManager.default.createFile(atPath: temp.path, contents: data.prefix(Int(scan.headerBytes)))
        let handle = try FileHandle(forWritingTo: temp)
        try handle.truncate(atOffset: UInt64(file.size))
        try handle.close()
        let info = try H3GGUFInfo.inspect(path: temp.path)
        switch info.role {
        case .dit: file.kind = .dit
        case .textEncoder: file.kind = .textEncoder
        case .vision: file.kind = .vision
        case .unknown: file.kind = .other
        }
        file.mainType = info.mainType.isEmpty ? (unsupportedTypes.sorted().first.map(ggmlTypeName) ?? "") : info.mainType
        file.curveAdaLN = info.curveAdaLN
        file.denseAdaLN = info.denseAdaLN
        file.textLayers = info.textLayers
        file.hasVision = info.hasVision
    }

    /// ggml type ids the engine expands (see h3_quant_from_ggml).
    private static let supportedGGMLTypes: Set<UInt32> = [0, 1, 30, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14]

    static func ggmlTypeName(_ id: UInt32) -> String {
        let names: [UInt32: String] = [
            0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
            10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
            17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
            23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
            34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
        ]
        return names[id] ?? "type \(id)" // kept in the cache: not localized
    }

    // MARK: safetensors

    private static func probeSafetensors(_ file: inout HFProbedFile) async throws {
        file.format = "safetensors"
        let prefix = try await fetchRange(repo: file.repo, path: file.path, from: 0, count: 8)
        guard prefix.count == 8 else { throw ProbeError.short }
        let length = prefix.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        guard length > 0, length < 256 << 20 else { throw ProbeError.headerTooLarge }
        let json = try await fetchRange(repo: file.repo, path: file.path, from: 8, count: Int64(length))
        guard let header = (try? JSONSerialization.jsonObject(with: json)) as? [String: Any] else {
            throw ProbeError.short
        }
        var names: [String] = []
        var dtypeBytes: [String: Int64] = [:]
        for (name, value) in header where name != "__metadata__" {
            names.append(name)
            if let tensor = value as? [String: Any], let dtype = tensor["dtype"] as? String,
               let offsets = tensor["data_offsets"] as? [NSNumber], offsets.count == 2 {
                dtypeBytes[dtype, default: 0] += offsets[1].int64Value - offsets[0].int64Value
            }
        }
        let has = { (part: String) in names.contains { $0.contains(part) } }
        if has("decoder.conv_pre") || has("resblocks.0.convs1") {
            file.kind = .audioVAE
        } else if has("decoder.transformer_blocks") || names.contains("latents_mean") || has("encoder.down_blocks") {
            file.kind = .videoVAE
        } else if has("video_patch_proj") || has("adaln_proj") {
            file.kind = .dit
        } else if has("language_model.layers.") || has("model.layers.") {
            file.kind = .textEncoder
        } else if has("visual.") {
            file.kind = .vision
        }
        file.mainType = dtypeBytes.max { $0.value < $1.value }?.key ?? ""
        let readable: Set<String> = ["F32", "F16", "BF16"]
        switch file.kind {
        case .videoVAE, .audioVAE:
            let others = Set(dtypeBytes.keys).subtracting(readable)
            if !others.isEmpty {
                file.problem = .numberFormats(others.sorted())
            }
        case .dit, .textEncoder, .vision:
            file.problem = .safetensorsPart
        case .other:
            file.problem = .unknownPart
        }
    }

    // MARK: HTTP

    private static func fetchRange(repo: String, path: String, from: Int64, count: Int64) async throws -> Data {
        guard let encoded = path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed),
              let url = URL(string: "https://huggingface.co/\(repo)/resolve/main/\(encoded)") else {
            throw ProbeError.badURL
        }
        var request = URLRequest(url: url)
        request.setValue("bytes=\(from)-\(from + count - 1)", forHTTPHeaderField: "Range")
        // Streamed and cut off at the bytes wanted, so a server ignoring the
        // range (200 with the whole file) does not download gigabytes.
        let (stream, response) = try await URLSession.shared.bytes(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 206 || http.statusCode == 200 else {
            throw ProbeError.http((response as? HTTPURLResponse)?.statusCode ?? -1)
        }
        let skip = http.statusCode == 200 ? Int(from) : 0
        let wanted = skip + Int(count)
        var data = Data()
        data.reserveCapacity(wanted)
        for try await byte in stream {
            data.append(byte)
            if data.count >= wanted { break }
        }
        stream.task.cancel()
        return skip > 0 ? data.subdata(in: min(skip, data.count) ..< data.count) : data
    }

    enum ProbeError: LocalizedError {
        case headerTooLarge, short, badURL, http(Int)
        var errorDescription: String? {
            switch self {
            case .headerTooLarge: return String(localized: "先頭部分が大きすぎます")
            case .short: return String(localized: "先頭部分が読めません")
            case .badURL: return String(localized: "URLを作れません")
            case .http(let status): return "HTTP \(status)"
            }
        }
    }
}

/// Just enough of a GGUF header to know where it ends and which tensor
/// types the file uses; nil when `data` stops before the header does.
private struct GGUFScan {
    var headerBytes: Int64 = 0
    var types: Set<UInt32> = []
    var splitCount: Int?

    init?(_ data: Data) {
        var reader = Reader(data: data)
        guard let magic = reader.bytes(4), magic == Data("GGUF".utf8),
              let version = reader.u32(), version >= 2,
              let tensorCount = reader.u64(), let kvCount = reader.u64() else { return nil }
        var alignment: UInt64 = 32
        for _ in 0 ..< kvCount {
            guard let key = reader.string(), let type = reader.u32() else { return nil }
            if key == "general.alignment", type == 4, let value = reader.u32() {
                alignment = UInt64(max(value, 1))
                continue
            }
            if key == "split.count", type == 2, let value = reader.u16() {
                splitCount = Int(value)
                continue
            }
            guard reader.skipValue(type: type) else { return nil }
        }
        for _ in 0 ..< tensorCount {
            guard reader.string() != nil, let dims = reader.u32(), dims <= 8,
                  reader.bytes(Int(dims) * 8) != nil, let type = reader.u32(), reader.u64() != nil else { return nil }
            types.insert(type)
        }
        headerBytes = Int64((UInt64(reader.position) + alignment - 1) / alignment * alignment)
    }

    private struct Reader {
        let data: Data
        var position = 0

        mutating func bytes(_ count: Int) -> Data? {
            guard count >= 0, position + count <= data.count else { return nil }
            defer { position += count }
            return data.subdata(in: data.startIndex + position ..< data.startIndex + position + count)
        }
        mutating func u16() -> UInt16? { bytes(2).map { $0.withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) }.littleEndian } }
        mutating func u32() -> UInt32? { bytes(4).map { $0.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian } }
        mutating func u64() -> UInt64? { bytes(8).map { $0.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian } }
        mutating func string() -> String? {
            guard let length = u64(), length < 1 << 24, let raw = bytes(Int(length)) else { return nil }
            return String(decoding: raw, as: UTF8.self)
        }
        /// GGUF value types: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32,
        /// 7 bool, 8 string, 9 array, 10 u64, 11 i64, 12 f64.
        mutating func skipValue(type: UInt32) -> Bool {
            switch type {
            case 0, 1, 7: return bytes(1) != nil
            case 2, 3: return bytes(2) != nil
            case 4, 5, 6: return bytes(4) != nil
            case 10, 11, 12: return bytes(8) != nil
            case 8: return string() != nil
            case 9:
                guard let elementType = u32(), let count = u64(), count < 1 << 26 else { return false }
                let fixed: [UInt32: Int] = [0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8]
                if let size = fixed[elementType] { return bytes(Int(count) * size) != nil }
                for _ in 0 ..< count {
                    guard skipValue(type: elementType) else { return false }
                }
                return true
            default: return false
            }
        }
    }
}

/// Probed files kept between runs (Application Support/H3cApp/hf-probe-cache-v2.json).
private actor ProbeCache {
    static let shared = ProbeCache()
    private var files: [String: HFProbedFile] = [:]
    private var loaded = false
    private var dirty = false
    // v2: reasons kept as codes (v1 kept localized text).
    private let path = h3AppSupportDirectory + "/hf-probe-cache-v2.json"

    private static func key(_ repo: String, _ path: String, _ size: Int64) -> String { "\(repo)/\(path)@\(size)" }

    private func load() {
        guard !loaded else { return }
        loaded = true
        if let data = FileManager.default.contents(atPath: path),
           let decoded = try? JSONDecoder().decode([String: HFProbedFile].self, from: data) {
            files = decoded
        }
    }

    func get(repo: String, path: String, size: Int64) -> HFProbedFile? {
        load()
        return files[Self.key(repo, path, size)]
    }

    func put(_ file: HFProbedFile) {
        load()
        // A failed read is not kept: the next listing tries again.
        if case .unreadable = file.problem { return }
        files[Self.key(file.repo, file.path, file.size)] = file
        dirty = true
    }

    func save() {
        guard dirty, let data = try? JSONEncoder().encode(files) else { return }
        try? FileManager.default.createDirectory(atPath: h3AppSupportDirectory, withIntermediateDirectories: true)
        try? data.write(to: URL(fileURLWithPath: path))
        dirty = false
    }
}
