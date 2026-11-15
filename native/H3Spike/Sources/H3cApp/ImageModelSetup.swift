import AppKit
import Combine
import H3Engine
import SwiftUI

/// An image model as the app installs it, each file pinned to a revision
/// and checked by SHA-256:
///
/// - MiniMax H3 Image (plans/IMAGE_PLAN.ja.md, phase 3): the ComfyUI int8 +
///   ConvRot DiT and the fp16 VAE from Comfy-Org's re-publication, the VAE's
///   config.json and the license text from MiniMax's own repository, and the
///   text encoder and tokenizer of a video model already on this Mac (the
///   image model uses the same ones; they are copied, which on one volume
///   takes no extra space).
/// - P-H3-Image (Pruna AI; phase 8): the int8 + ConvRot DiT and the bf16
///   text encoder (Qwen3-VL-4B with its text connector) and VAE from
///   PrunaAI/P-H3-Image-Comfy, the VAE config and the tokenizer from the
///   diffusers repository PrunaAI/P-H3-Image. It needs no video model.
///
/// Nothing is bundled. The user downloads the weights here after reading
/// what the license asks: MiniMax H3 Image's is not the video model's (it
/// shuts out whole territories, and its conditions reach the pictures made
/// with the model); P-H3-Image's is the video model's own.
struct H3ImageModel: Identifiable, Hashable {
    enum Kind: String { case h3, ph3 }

    /// A weight file: where it is in the repository, where the engine
    /// looks for it, and what it must be.
    struct Weight: Hashable {
        let repoPath: String
        let localPath: String
        let bytes: Int64
        let sha256: String
    }

    /// A small file fetched whole and checked before it is written.
    struct SmallFile: Hashable {
        let url: URL
        let localPath: String
        let sha256: String
    }

    let kind: Kind
    /// The install: "h3", "ph3" (P-H3-Image's own DiT, all int8) or
    /// "ph3-lora" (a video model's FL2VA DiT with P-H3-Image's LoRA).
    let id: String
    let displayName: String
    let folderName: String
    let weightsRepo: String
    let weightsRevision: String
    let weightsRevisionDate: String
    let licenseName: String
    let licenseURL: URL
    let officialURL: URL
    let authorizationURL: URL?
    let weights: [Weight]
    let smallFiles: [SmallFile]
    /// It takes the text encoder of a video model already installed.
    let sharesTextEncoder: Bool
    let licenseAcceptedKey: String
    let provenanceNote: String
    let declaration: String
    /// It takes the FL2VA DiT of a video model already installed (copied
    /// into transformer/) and applies the LoRA in lora/ to it.
    var usesVideoTransformer = false

    static func == (a: H3ImageModel, b: H3ImageModel) -> Bool { a.id == b.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }

    static let licenseFile = "LICENSE"
    static let textEncoderFolder = "text_encoder"
    static let transformerFolder = "transformer"
    static let provenanceFile = "H3IMAGE_PROVENANCE.json"

    // MARK: MiniMax H3 Image

    private static let h3Repo = "Comfy-Org/MiniMax-H3-Image"
    private static let h3Revision = "892de7cb61df49753bf15145a711154803a355cb"
    private static let h3OfficialRepo = "MiniMaxAI/MiniMax-H3-Image"
    /// The revision the VAE config and the tokenizer are read from.
    private static let h3OfficialRevision = "492fd8c4eca666f1085f57e30738c1e0d86b3439"
    /// The revision of the license text (it last changed on 2026-11-11).
    private static let h3LicenseRevision = "5f90473023975e6271b0ed1dd84c4e88e4bbc6d3"

    private static func h3Official(_ repoPath: String) -> URL {
        URL(string: "https://huggingface.co/\(h3OfficialRepo)/resolve/\(h3OfficialRevision)/\(repoPath)")!
    }

    static let h3 = H3ImageModel(
        kind: .h3, id: "h3", displayName: "MiniMax H3 Image", folderName: "MiniMax-H3-Image",
        weightsRepo: h3Repo, weightsRevision: h3Revision, weightsRevisionDate: "2026-11-10",
        licenseName: "MiniMax-H3-Image Community License Agreement",
        licenseURL: URL(string: "https://huggingface.co/\(h3OfficialRepo)/blob/\(h3LicenseRevision)/LICENSE")!,
        officialURL: URL(string: "https://huggingface.co/\(h3OfficialRepo)")!,
        authorizationURL: URL(string: "https://platform.minimax.io/console/license")!,
        weights: [
            Weight(repoPath: "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
                   localPath: "transformer/minimax_h3_image_pruned_int8_convrot.safetensors",
                   bytes: 20_979_314_802,
                   sha256: "83eb140f1ff95c068f8a6886e6d3f3d0d4ab2e07bb2fe6d2ef992a51146ee622"),
            Weight(repoPath: "vae/minimax_h3_image_vae_fp16.safetensors",
                   localPath: "vae/minimax_h3_image_vae_fp16.safetensors",
                   bytes: 1_168_505_312,
                   sha256: "d80e9c2e5cfc9844869961ff5a58c9a9e52182df6bfb30b26018dd7f348721af"),
        ],
        smallFiles: [
            SmallFile(url: h3Official("vae/config.json"), localPath: "vae/config.json",
                      sha256: "7fcc21bec549804ff0a04d8246e7f90e679d4f8d4b0a27a252bbd18c30e3e4bb"),
            SmallFile(url: h3Official("tokenizer/tokenizer.json"), localPath: "tokenizer/tokenizer.json",
                      sha256: "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"),
            SmallFile(url: URL(string: "https://huggingface.co/\(h3OfficialRepo)/resolve/\(h3LicenseRevision)/LICENSE")!,
                      localPath: licenseFile,
                      sha256: "0ec3bb5e6d9e885fa2aaf15e84689e50fd51f42b53fa24d3a64ed9d8c76d423b"),
        ],
        sharesTextEncoder: true,
        licenseAcceptedKey: "h3c-app.imageModelLicenseAcceptedAt",
        provenanceNote: "The DiT (ComfyUI int8 + ConvRot) and the fp16 VAE are Comfy-Org's re-publication of MiniMax H3 Image; vae/config.json and tokenizer/tokenizer.json are from \(h3OfficialRepo) at \(h3OfficialRevision); LICENSE is that repository's at \(h3LicenseRevision).",
        declaration: "Model card of this revision: license_name minimax-h3-image-community-license-agreement")

    // MARK: P-H3-Image

    private static let ph3Repo = "PrunaAI/P-H3-Image-Comfy"
    private static let ph3Revision = "44a0fb73f299da1bb715bdb5ff59ce3fd3c6f543"
    /// The diffusers repository, where the VAE config and the tokenizer are.
    private static let ph3DiffusersRepo = "PrunaAI/P-H3-Image"
    private static let ph3DiffusersRevision = "7f98acef503cf2bcfba821a78cb1cf1a9928b2df"

    private static func ph3File(_ repoPath: String) -> URL {
        URL(string: "https://huggingface.co/\(ph3Repo)/resolve/\(ph3Revision)/\(repoPath)")!
    }

    private static func ph3Diffusers(_ repoPath: String) -> URL {
        URL(string: "https://huggingface.co/\(ph3DiffusersRepo)/resolve/\(ph3DiffusersRevision)/\(repoPath)")!
    }

    private static let ph3TextEncoder = Weight(
        repoPath: "text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors",
        localPath: "text_encoder/p-h3-image_text_encoder_int8_convrot.safetensors",
        bytes: 5_120_870_504,
        sha256: "fb42c9ff0597e61729475d0b48b69b591ffe06d0b442a79dde22b14fadbe1c0f")
    private static let ph3VAE = Weight(
        repoPath: "vae/p-h3-image_vae_int8_convrot.safetensors",
        localPath: "vae/p-h3-image_vae_int8_convrot.safetensors",
        bytes: 3_146_996_440,
        sha256: "583e0d43354b6b833aa9a0815a5fac831b1ef6d0c19a7eaa0616980a3dab5ffe")

    /// P-H3-Image's own DiT, text encoder and VAE, all int8 + ConvRot
    /// (about 29 GB; the bf16 text encoder and VAE, 6.5 GB more, gave the
    /// same pictures - plans/IMAGE_PLAN.ja.md, appendix J).
    static let ph3 = H3ImageModel(
        kind: .ph3, id: "ph3", displayName: "P-H3-Image", folderName: "P-H3-Image",
        weightsRepo: ph3Repo, weightsRevision: ph3Revision, weightsRevisionDate: "2026-11-03",
        licenseName: "MiniMax H3 Community License Agreement",
        licenseURL: URL(string: "https://huggingface.co/\(ph3Repo)/blob/\(ph3Revision)/LICENSE")!,
        officialURL: URL(string: "https://huggingface.co/\(ph3DiffusersRepo)")!,
        authorizationURL: nil,
        weights: [
            Weight(repoPath: "diffusion_models/p-h3-image_int8_convrot.safetensors",
                   localPath: "transformer/p-h3-image_int8_convrot.safetensors",
                   bytes: 21_213_068_840,
                   sha256: "b156fd4fe9d169e58cb2b07f9594ec740ff8f0e052d62b009e6dd034de5865e7"),
            ph3TextEncoder, ph3VAE,
        ],
        smallFiles: ph3SmallFiles,
        sharesTextEncoder: false,
        licenseAcceptedKey: "h3c-app.ph3ImageModelLicenseAcceptedAt",
        provenanceNote: "P-H3-Image by Pruna AI: the DiT, the text encoder (Qwen3-VL-4B with its text connector) and the VAE, all int8 + ConvRot, are \(ph3Repo)'s files, with its LICENSE, LICENSE-APACHE and NOTICE; vae/config.json and tokenizer/tokenizer.json are from \(ph3DiffusersRepo) at \(ph3DiffusersRevision).",
        declaration: ph3Declaration)

    /// P-H3-Image on a video model's own FL2VA DiT: Pruna AI's LoRA, which
    /// P-H3-Image's DiT is FL2VA with merged, and the int8 text encoder and
    /// VAE (about 10 GB). Slower: the DiT runs as the video model's does.
    static let ph3LoRA = H3ImageModel(
        kind: .ph3, id: "ph3-lora", displayName: "P-H3-Image LoRA", folderName: "P-H3-Image-LoRA",
        weightsRepo: ph3Repo, weightsRevision: ph3Revision, weightsRevisionDate: "2026-11-03",
        licenseName: "MiniMax H3 Community License Agreement",
        licenseURL: URL(string: "https://huggingface.co/\(ph3Repo)/blob/\(ph3Revision)/LICENSE")!,
        officialURL: URL(string: "https://huggingface.co/\(ph3DiffusersRepo)")!,
        authorizationURL: nil,
        weights: [
            Weight(repoPath: "loras/p-h3-image_lora_bf16.safetensors",
                   localPath: "lora/p-h3-image_lora_bf16.safetensors",
                   bytes: 1_956_195_360,
                   sha256: "0e9013ae2873e3415dcb7bb59a9677293bdbdb6f663e138a79bb9bdced00ea61"),
            ph3TextEncoder, ph3VAE,
        ],
        smallFiles: ph3SmallFiles,
        sharesTextEncoder: false,
        licenseAcceptedKey: "h3c-app.ph3ImageModelLicenseAcceptedAt",
        provenanceNote: "P-H3-Image by Pruna AI as its LoRA on a video model's FL2VA DiT: lora/, the text encoder (Qwen3-VL-4B with its text connector) and the VAE, int8 + ConvRot, are \(ph3Repo)'s files, with its LICENSE, LICENSE-APACHE and NOTICE; vae/config.json and tokenizer/tokenizer.json are from \(ph3DiffusersRepo) at \(ph3DiffusersRevision).",
        declaration: ph3Declaration,
        usesVideoTransformer: true)

    private static let ph3Declaration =
        "Model card of this revision: license_name minimax-h3-community-license-agreement (Qwen3-VL-4B parts: Apache-2.0)"

    private static let ph3SmallFiles = [
            SmallFile(url: ph3Diffusers("vae/config.json"), localPath: "vae/config.json",
                      sha256: "63d82611cd65279518d52b42b38eef80580cb7bac0c04b3afbba41b37d597ca7"),
            SmallFile(url: ph3Diffusers("tokenizer/tokenizer.json"), localPath: "tokenizer/tokenizer.json",
                      sha256: "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"),
            SmallFile(url: ph3File("LICENSE"), localPath: licenseFile,
                      sha256: "59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44"),
            SmallFile(url: ph3File("LICENSE-APACHE"), localPath: "LICENSE-APACHE",
                      sha256: "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"),
            SmallFile(url: ph3File("NOTICE"), localPath: "NOTICE",
                      sha256: "f518e346f0c20d4ea2c70c0c953add326d5594df7e0bcbcb647e21dec1f8b89f"),
    ]

    static let all = [h3, ph3, ph3LoRA]

    /// The model the app installed at this folder, if it is one.
    static func installed(at path: String) -> H3ImageModel? {
        all.first { $0.directory == path }
    }

    var weightsURL: URL { URL(string: "https://huggingface.co/\(weightsRepo)/tree/\(weightsRevision)")! }
    var downloadBytes: Int64 { weights.reduce(0) { $0 + $1.bytes } }

    var directory: String { h3AppSupportDirectory + "/ImageModels/" + folderName }

    func path(_ relative: String) -> String { directory + "/" + relative }

    func fileSize(_ relative: String) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: path(relative))[.size]) as? Int64
    }

    /// The downloaded weights are here, each at its full size.
    var weightsPresent: Bool {
        weights.allSatisfy { fileSize($0.localPath) == $0.bytes }
    }

    var hasTextEncoder: Bool {
        !((try? FileManager.default.contentsOfDirectory(atPath: path(Self.textEncoderFolder))) ?? []).isEmpty
    }

    var hasTransformer: Bool {
        !((try? FileManager.default.contentsOfDirectory(atPath: path(Self.transformerFolder))) ?? []).isEmpty
    }

    /// Everything the engine needs: the folder is an image model.
    var complete: Bool {
        weightsPresent && smallFiles.allSatisfy { fileSize($0.localPath) != nil } && hasTextEncoder
            && (!usesVideoTransformer || hasTransformer)
    }

    var licenseAcceptedAt: Date? {
        get { UserDefaults.standard.object(forKey: licenseAcceptedKey) as? Date }
        nonmutating set { UserDefaults.standard.set(newValue, forKey: licenseAcceptedKey) }
    }

    // MARK: Provenance record (the form the upscaler models use)

    var provenance: SwiftVRModel.Provenance? {
        guard let data = FileManager.default.contents(atPath: path(Self.provenanceFile)) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(SwiftVRModel.Provenance.self, from: data)
    }

    /// A record for these revisions with the expected hashes, the weights
    /// still at their recorded sizes.
    var verified: Bool {
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
        case noVideoTransformer
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
            case .noVideoTransformer:
                return String(localized: "この組み合わせは、導入済みの動画モデル（MiniMax-H3、FL2VA）のDiTを使います。先に動画モデルを導入してから、もう一度お試しください。")
            case .copy(let detail):
                return String(localized: "テキストエンコーダを複製できませんでした（詳細: \(detail)）")
            }
        }
    }

    /// A small file at its pinned revision, written only when its bytes are
    /// the recorded ones.
    func fetchSmallFile(_ file: SmallFile) async throws {
        let relative = file.localPath
        if let data = FileManager.default.contents(atPath: path(relative)), FileHash.sha256(of: data) == file.sha256 { return }
        let data: Data
        do {
            let (body, response) = try await URLSession.shared.data(from: file.url)
            guard let http = response as? HTTPURLResponse, (200 ..< 300).contains(http.statusCode) else {
                throw SetupError.smallFile(relative, "HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)")
            }
            data = body
        } catch let error as SetupError {
            throw error
        } catch {
            throw SetupError.smallFile(relative, error.localizedDescription)
        }
        guard FileHash.sha256(of: data) == file.sha256 else {
            throw SetupError.smallFile(relative, String(localized: "取得した内容のSHA-256が記録と一致しません"))
        }
        let destination = URL(fileURLWithPath: path(relative))
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: destination, options: .atomic)
    }

    /// The video model whose text encoder the image model can share: the
    /// first of these folders that has one.
    static func textEncoderSource(in modelPaths: [String]) -> String? {
        modelPaths.map { $0 + "/FL2VA/text_encoder" }.first { folder in
            !((try? FileManager.default.contentsOfDirectory(atPath: folder)) ?? []).isEmpty
        }
    }

    /// The video model whose FL2VA DiT the LoRA install uses.
    static func transformerSource(in modelPaths: [String]) -> String? {
        modelPaths.map { $0 + "/FL2VA/transformer" }.first { folder in
            !((try? FileManager.default.contentsOfDirectory(atPath: folder)) ?? []).isEmpty
        }
    }

    /// Copies a video model's FL2VA DiT (a clone where the volume makes
    /// one), as shareTextEncoder copies the text encoder.
    func shareTransformer(from source: String) throws {
        if hasTransformer { return }
        try shareFolder(from: source, to: Self.transformerFolder)
    }

    /// Copies a video model's text encoder (a clone where the volume makes
    /// one) unless one is here already. Copies files: off the main actor.
    func shareTextEncoder(from source: String) throws {
        if hasTextEncoder { return }
        try shareFolder(from: source, to: Self.textEncoderFolder)
    }

    private func shareFolder(from source: String, to folder: String) throws {
        let destination = path(folder)
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
    func verify(textEncoderSource: String?, transformerSource: String? = nil,
                progress: @escaping @Sendable (Int64, Int64) -> Void) async throws {
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
        for small in smallFiles.map(\.localPath) {
            if let data = FileManager.default.contents(atPath: path(small)) {
                record[small] = .init(sha256: FileHash.sha256(of: data), bytes: Int64(data.count))
            }
        }
        var note = provenanceNote
        if let textEncoderSource {
            note += " text_encoder/ is a copy of the video model's at \(textEncoderSource) (the same Qwen3-VL language layers)."
        }
        if let transformerSource {
            note += " transformer/ is a copy of the video model's FL2VA DiT at \(transformerSource), to which lora/ is applied at strength 1."
        }
        let provenance = SwiftVRModel.Provenance(
            repo: weightsRepo, revision: weightsRevision, revisionDate: weightsRevisionDate,
            declaredLicense: licenseName, declaration: declaration,
            licenseText: Self.licenseFile, files: record, verifiedAt: Date(), note: note)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(provenance).write(to: URL(fileURLWithPath: path(Self.provenanceFile)), options: .atomic)
    }

    func bytesOnDisk() -> Int64 {
        guard let enumerator = FileManager.default.enumerator(atPath: directory) else { return 0 }
        var total: Int64 = 0
        for case let item as String in enumerator {
            total += (try? FileManager.default.attributesOfItem(atPath: directory + "/" + item)[.size] as? Int64) ?? 0
        }
        return total
    }

    func removeAll() throws {
        try FileManager.default.removeItem(atPath: directory)
    }
}

/// Downloads and checks an image model: one per model for the whole app,
/// so a download keeps going when the sheet showing it closes.
@MainActor
final class H3ImageModelSetup: ObservableObject {
    static let shared = H3ImageModelSetup(model: .h3)
    private static let others = [H3ImageModel.ph3, H3ImageModel.ph3LoRA].map(H3ImageModelSetup.init(model:))

    static func of(_ model: H3ImageModel) -> H3ImageModelSetup {
        others.first { $0.model == model } ?? shared
    }

    let model: H3ImageModel

    private init(model: H3ImageModel) {
        self.model = model
    }

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
        model.licenseAcceptedAt = model.licenseAcceptedAt ?? Date()
        phase = .listing
        task = Task { await run(videoModelPaths: videoModelPaths) }
    }

    func cancel() {
        task?.cancel()
        downloader.cancel()
        if isBusy { phase = .idle }
    }

    private func run(videoModelPaths: [String]) async {
        let model = self.model
        do {
            // Before the weights come down: what the model cannot do without.
            let hasTextEncoder = model.hasTextEncoder
            let source = model.sharesTextEncoder ? H3ImageModel.textEncoderSource(in: videoModelPaths) : nil
            guard !model.sharesTextEncoder || hasTextEncoder || source != nil else {
                throw H3ImageModel.SetupError.noTextEncoder
            }
            let hasTransformer = model.hasTransformer
            let transformerSource = model.usesVideoTransformer
                ? H3ImageModel.transformerSource(in: videoModelPaths) : nil
            guard !model.usesVideoTransformer || hasTransformer || transformerSource != nil else {
                throw H3ImageModel.SetupError.noVideoTransformer
            }
            var fetched = false
            if !model.weightsPresent { fetched = try await download() }
            try Task.checkCancellation()
            for file in model.smallFiles { try await model.fetchSmallFile(file) }
            try Task.checkCancellation()
            if !hasTextEncoder, let source {
                phase = .sharing
                try await Task.detached(priority: .userInitiated) {
                    try model.shareTextEncoder(from: source)
                }.value
            }
            if !hasTransformer, let transformerSource {
                phase = .sharing
                try await Task.detached(priority: .userInitiated) {
                    try model.shareTransformer(from: transformerSource)
                }.value
            }
            try Task.checkCancellation()
            if fetched || !model.verified {
                phase = .verifying(done: 0, total: 0)
                try await model.verify(textEncoderSource: source, transformerSource: transformerSource) { done, total in
                    Task { @MainActor in
                        if case .verifying = self.phase { self.phase = .verifying(done: done, total: total) }
                    }
                }
            }
            try Task.checkCancellation()
            // Listed for every window, whether or not the sheet is open.
            if model.complete {
                ModelLibrary.registerImageModel(path: model.directory, name: model.displayName)
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
        for folder in Set(model.weights.map { ($0.repoPath as NSString).deletingLastPathComponent }) {
            entries += try await ModelDownloader.fetchEntries(
                repo: model.weightsRepo, subpath: folder, revision: model.weightsRevision)
        }
        var wanted: [HFFileEntry] = []
        for weight in model.weights {
            guard var entry = entries.first(where: { $0.path == weight.repoPath }),
                  entry.size == weight.bytes, entry.sha256 == nil || entry.sha256 == weight.sha256 else {
                throw H3ImageModel.SetupError.listing(file: weight.repoPath)
            }
            entry.localPath = weight.localPath
            wanted.append(entry)
        }
        try Task.checkCancellation()
        let fetching = wanted.filter { model.fileSize($0.local) != $0.size }
        try SwiftVRModel.forgetProvenance(of: fetching.map(\.local),
                                          recordPath: model.path(H3ImageModel.provenanceFile))
        downloader.use(wanted)
        downloader.startDownload(destination: model.directory)
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
    let model: H3ImageModel
    /// An image is being generated: the model's files stay.
    var busy = false
    var close: () -> Void

    @ObservedObject private var setup: H3ImageModelSetup
    @ObservedObject private var downloader: ModelDownloader
    @Environment(\.colorScheme) private var colorScheme
    @State private var agreed: Bool
    @State private var confirmingRemoval = false
    @State private var removalError: String?
    @State private var version = 0

    init(library: ModelLibrary, model: H3ImageModel = .h3, busy: Bool = false, close: @escaping () -> Void) {
        self.library = library
        self.model = model
        self.busy = busy
        self.close = close
        let setup = H3ImageModelSetup.of(model)
        _setup = ObservedObject(wrappedValue: setup)
        _downloader = ObservedObject(wrappedValue: setup.downloader)
        _agreed = State(initialValue: model.licenseAcceptedAt != nil)
    }

    private var palette: H3Palette { H3Palette(colorScheme) }
    private var installed: Bool { _ = version; _ = setup.phase; return model.complete }
    private var verified: Bool { _ = version; _ = setup.phase; return model.verified }
    private var registered: Bool {
        library.imageModels.contains { $0.path == model.directory }
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
            HStack(alignment: .firstTextBaseline) {
                Text("画像モデルを準備").font(.headline)
                Text(verbatim: model.displayName).font(.headline).foregroundStyle(palette.textSecondary)
            }
            Group {
                if model.usesVideoTransformer {
                    Text("静止画をつくるモデル「P-H3-Image」（Pruna AI）を、導入済みの動画モデルのDiT（FL2VA）に、付属のLoRAを当てて使います。LoRAとテキストエンコーダ・VAE（int8版）を、Hugging Faceの\(model.weightsRepo)（revision \(String(model.weightsRevision.prefix(8)))、\(model.weightsRevisionDate)）からダウンロードし、各ファイルのSHA-256をその版の記録と照合します。約\(gigabytes(model.downloadBytes))が「\(model.directory)」に置かれます。動画モデルのDiTは複製して使います（同じディスクなら容量は増えません）。生成には、P-H3-Image本体のおよそ2倍の時間がかかります。")
                } else if model.kind == .ph3 {
                    Text("静止画をつくるモデル「P-H3-Image」（Pruna AI、MiniMax H3を基にした画像用のモデル）の重み（DiT・テキストエンコーダ・VAE、どれもint8版）を、Hugging Faceの\(model.weightsRepo)（ComfyUI用、revision \(String(model.weightsRevision.prefix(8)))、\(model.weightsRevisionDate)）からダウンロードし、各ファイルのSHA-256をその版の記録と照合します。約\(gigabytes(model.downloadBytes))が「\(model.directory)」に置かれます。動画モデルは要りません。文章からの生成と、参照画像（8枚まで）を使った編集ができます。")
                } else {
                    Text("静止画をつくるモデル「MiniMax H3 Image」の重みを、Hugging Faceの\(model.weightsRepo)（ComfyUI用のint8版、revision \(String(model.weightsRevision.prefix(8)))、\(model.weightsRevisionDate)）からダウンロードし、各ファイルのSHA-256をその版の記録と照合します。約\(gigabytes(model.downloadBytes))が「\(model.directory)」に置かれます。テキストエンコーダは、導入済みの動画モデルのものを複製して使います（同じディスクなら容量は増えません）。")
                }
            }
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
            Text("ダウンロードした画像モデル（計\(gigabytes(model.bytesOnDisk()))）を削除し、一覧からも外します。また使うには、もう一度ダウンロードが必要です。動画モデルは変わりません。")
        }
    }

    private var licenseNotice: some View {
        HStack(alignment: .top, spacing: H3Spacing.sm) {
            Image(systemName: "doc.text.magnifyingglass").foregroundStyle(palette.accent)
            VStack(alignment: .leading, spacing: 4) {
                if model.kind == .ph3 {
                    Text("P-H3-Imageには、動画モデル（MiniMax H3）と同じ「\(model.licenseName)」が適用されます。テキストエンコーダのうちQwen3-VL-4Bに由来する部分はApache License 2.0です。おもな条件は次のとおりです。")
                        .font(.caption)
                        .fixedSize(horizontal: false, vertical: true)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("・EU、英国、韓国、米国では使えません（日本は対象地域に含まれます）。")
                        Text("・商用の製品やサービスの年間売上が2,000万米ドルを超える場合は、MiniMaxの事前の書面許諾が必要です。")
                        Text("・このモデルを使った製品やサービスには「Powered by MiniMax H3」と表示します。")
                        Text("・つくった画像を公開するときは、機械が生成したものであることを明示します。")
                        Text("・ほかのAIモデルの改良には使えません。禁止されている用途（付属書A）があります。")
                    }
                    .font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
                } else {
                    Text("MiniMax H3 Imageには「\(model.licenseName)」が適用されます。動画モデル（MiniMax H3）とは別の契約です。おもな条件は次のとおりです。")
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
                }
                Text("これは要約で、契約の全文ではありません。あなたの利用にどの条件が当てはまるかは、本文を読んで、利用者自身（必要なら専門家）が判断してください。アプリは本文を重みの隣に保存します。")
                    .font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: H3Spacing.md) {
                    Link("ライセンスの本文（その版）", destination: model.licenseURL)
                    if let authorizationURL = model.authorizationURL {
                        Link("許諾の申請先（MiniMax）", destination: authorizationURL)
                    }
                }
                .font(.caption)
                HStack(spacing: H3Spacing.md) {
                    if model.kind == .ph3 {
                        Link("元のモデル（Pruna AI）", destination: model.officialURL)
                    } else {
                        Link("元のモデル（MiniMax）", destination: model.officialURL)
                    }
                    Link("配布元（Hugging Face）", destination: model.weightsURL)
                }
                .font(.caption)
            }
        }
        .padding(H3Spacing.sm)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(palette.accent.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
    }

    private var verificationText: String {
        if verified, let record = model.provenance {
            return String(localized: "SHA-256がその版の記録と一致（\(record.verifiedAt.formatted(date: .numeric, time: .shortened))に確認）")
        }
        if model.weightsPresent {
            return String(localized: "まだ照合していません（「取得元と照合する」で確かめられます）")
        }
        return String(localized: "まだありません")
    }

    private var statusRows: some View {
        Grid(alignment: .leading, horizontalSpacing: H3Spacing.md, verticalSpacing: 6) {
            GridRow {
                Text("重み").font(.caption).foregroundStyle(palette.textSecondary)
                Text(model.weightsPresent ? String(localized: "ダウンロード済み") : String(localized: "まだありません")).font(.caption)
            }
            GridRow {
                Text("照合").font(.caption).foregroundStyle(palette.textSecondary)
                Text(verificationText).font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if model.usesVideoTransformer {
                GridRow {
                    Text("DiT").font(.caption).foregroundStyle(palette.textSecondary)
                    Text(transformerText).font(.caption)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            if model.sharesTextEncoder {
                GridRow {
                    Text("テキストエンコーダ").font(.caption).foregroundStyle(palette.textSecondary)
                    Text(textEncoderText).font(.caption)
                        .fixedSize(horizontal: false, vertical: true)
                }
                GridRow {
                    Text("使用容量").font(.caption).foregroundStyle(palette.textSecondary)
                    Text("\(gigabytes({ _ = version; return model.bytesOnDisk() }()))（複製したテキストエンコーダを含みます。同じディスクなら、実際に増えるのはダウンロードした分だけです）")
                        .font(.caption)
                        .fixedSize(horizontal: false, vertical: true)
                }
            } else {
                GridRow {
                    Text("使用容量").font(.caption).foregroundStyle(palette.textSecondary)
                    Text(verbatim: gigabytes({ _ = version; return model.bytesOnDisk() }())).font(.caption)
                }
            }
            if let accepted = model.licenseAcceptedAt {
                GridRow {
                    Text("ライセンス").font(.caption).foregroundStyle(palette.textSecondary)
                    Text("\(accepted.formatted(date: .numeric, time: .shortened))に説明を確認").font(.caption)
                }
            }
        }
    }

    private var transformerText: String {
        _ = version; _ = setup.phase
        if model.hasTransformer { return String(localized: "動画モデルのDiTを複製済み") }
        if H3ImageModel.transformerSource(in: videoModelPaths) != nil {
            return String(localized: "導入済みの動画モデル（FL2VA）から複製します")
        }
        return String(localized: "動画モデルがありません。先に動画モデル（MiniMax-H3）を導入してください")
    }

    private var textEncoderText: String {
        _ = version; _ = setup.phase
        if model.hasTextEncoder {
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
                if model.usesVideoTransformer {
                    Text("動画モデルのDiTを複製しています…").font(.caption).foregroundStyle(palette.textSecondary)
                } else {
                    Text("テキストエンコーダを複製しています…").font(.caption).foregroundStyle(palette.textSecondary)
                }
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
            if !model.weightsPresent {
                let space = downloadSpaceCheck(needed: model.downloadBytes, destination: model.directory)
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
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.directory)])
            }
            .disabled(!FileManager.default.fileExists(atPath: model.directory))
            Button("すべて削除…") { confirmingRemoval = true }
                .disabled(setup.isBusy || busy
                          || !FileManager.default.fileExists(atPath: model.directory))
            Spacer()
            Button("閉じる") { close() }
                .keyboardShortcut(.cancelAction)
            if setup.isBusy {
                Button("中止") { setup.cancel() }
            } else if !installed {
                Button(model.weightsPresent ? String(localized: "残りを準備する") : String(localized: "ダウンロードする")) {
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
        guard model.complete else { return }
        ModelLibrary.registerImageModel(path: model.directory, name: model.displayName)
        version += 1
    }

    private func remove() {
        do {
            for entry in library.imageModels where entry.path == model.directory {
                library.removeImageModel(id: entry.id)
            }
            try model.removeAll()
            removalError = nil
        } catch {
            removalError = String(localized: "削除できませんでした（詳細: \(error.localizedDescription)）")
        }
        version += 1
    }
}
