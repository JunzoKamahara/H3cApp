import Foundation
import H3Engine

/// This developer's own hand-placed model checkout - not inside Application
/// Support like a fresh download would be, so it needs its own literal path
/// here to be recognized. Originally at .../h3c-analysis/MiniMax-H3 (the
/// app's old name), then moved to ~/models/MiniMax-H3 to get the ~270GB
/// model and its caches out of ~/Library. Kept as an exact-path match so
/// ModelLibrary can seed an entry for it (if it exists) and
/// GenerationViewModel can reuse its already-built, multi-GB attention
/// caches instead of demanding a rebuild under a new per-model cache
/// directory. Never used to suggest a destination for a new download - see
/// defaultH3ModelDownloadPath for that.
let legacyDefaultH3ModelPath = NSHomeDirectory() + "/models/MiniMax-H3"

/// Suggested destination for a model a brand-new install downloads, under
/// this app's own current name - unlike legacyDefaultH3ModelPath above,
/// which only ever refers to a specific pre-existing folder from before the
/// app was renamed.
let defaultH3ModelDownloadPath = h3AppSupportDirectory + "/MiniMax-H3"

/// Where LoRA files are kept by default: beside the default model folder,
/// outside the folders macOS asks permission for (protectedFolderName).
let defaultLoRADirectory = h3AppSupportDirectory + "/LoRA"

/// Where RefMod files (saved reference latents, H3RefModInfo) are kept:
/// the ones made here are written there, and files put there by hand are
/// picked up on the next look (ModelLibrary.syncRefModFolder).
let defaultRefModDirectory = h3AppSupportDirectory + "/RefMods"

/// A RefMod file the library knows: its name is the user's label here (the
/// file's own metadata name is shown beside it).
struct RefModEntry: Identifiable, Codable, Equatable {
    let id: UUID
    var name: String
    var path: String

    init(id: UUID = UUID(), name: String, path: String) {
        self.id = id
        self.name = name
        self.path = path
    }

    var isInRefModFolder: Bool {
        (path as NSString).deletingLastPathComponent == defaultRefModDirectory
    }
}

/// The folder macOS guards a path with, for the message - Desktop,
/// Documents, Downloads, iCloud Drive or another volume - or nil. An app
/// reading a file there (outside the open panel that picked it) is asked
/// for permission, as a registered LoRA is at every launch.
func protectedFolderName(for path: String) -> String? {
    let resolved = (path as NSString).resolvingSymlinksInPath
    let home = (NSHomeDirectory() as NSString).resolvingSymlinksInPath
    let folders = [
        ("/Desktop/", String(localized: "デスクトップ")),
        ("/Documents/", String(localized: "書類")),
        ("/Downloads/", String(localized: "ダウンロード")),
        ("/Library/Mobile Documents/", "iCloud Drive"),
    ]
    for (folder, name) in folders where resolved.hasPrefix(home + folder) { return name }
    if resolved.hasPrefix("/Volumes/") { return String(localized: "外部のディスク") }
    return nil
}

struct H3ModelEntry: Identifiable, Codable, Equatable {
    let id: UUID
    var name: String
    var path: String

    init(id: UUID = UUID(), name: String, path: String) {
        self.id = id
        self.name = name
        self.path = path
    }
}

/// The denoising step rule a LoRA runs with. auto follows what the file's
/// metadata declares (DMAD writes "re-noise step rule"); nothing is guessed
/// from names. re-noise is ComfyUI's "lcm" sampler rule, which LCM/DMD-style
/// few-step distillations usually recommend.
enum LoRASampler: String, Codable, CaseIterable {
    case auto, euler, renoise
}

struct LoRAEntry: Identifiable, Codable, Equatable {
    let id: UUID
    var name: String
    var path: String
    /// Multiplier on the adapter's own alpha/rank scale; empty means 1.0
    /// (as trained). Named for what it held before strengths became
    /// relative - kept so saved libraries still decode.
    var scaleText: String
    /// Step count a distilled (Turbo) LoRA was trained for; enabling the
    /// LoRA switches the draft to it. nil for ordinary style LoRAs. Optional
    /// so entries saved before this field existed still decode.
    var recommendedSteps: Int?
    /// Part of the stack applied to the next generation. Any number of
    /// entries (up to maxStackedLoRAs) can be on at once; they add up.
    var enabled: Bool
    /// Set by hand for adapters whose files don't declare their sampling;
    /// auto / empty text follow the metadata (see ModelLibrary.sampling(for:)).
    var sampler: LoRASampler
    var audioShiftText: String
    /// Where the file was before being moved into the LoRA folder, so
    /// projects, videos and presets that name the old path still find it.
    var formerPaths: [String] = []
    /// The file's SHA-256 and the size it had when hashed, computed only
    /// for files whose size matches a recorded origin
    /// (ModelLibrary.knownLoRAProvenance); nil otherwise.
    var sha256: String? = nil
    var sha256Bytes: Int64? = nil

    init(id: UUID = UUID(), name: String, path: String, scaleText: String = "",
         recommendedSteps: Int? = nil, enabled: Bool = false,
         sampler: LoRASampler = .auto, audioShiftText: String = "") {
        self.id = id
        self.name = name
        self.path = path
        self.scaleText = scaleText
        self.recommendedSteps = recommendedSteps
        self.enabled = enabled
        self.sampler = sampler
        self.audioShiftText = audioShiftText
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        path = try container.decode(String.self, forKey: .path)
        scaleText = try container.decodeIfPresent(String.self, forKey: .scaleText) ?? ""
        recommendedSteps = try container.decodeIfPresent(Int.self, forKey: .recommendedSteps)
        enabled = try container.decodeIfPresent(Bool.self, forKey: .enabled) ?? false
        sampler = try container.decodeIfPresent(LoRASampler.self, forKey: .sampler) ?? .auto
        audioShiftText = try container.decodeIfPresent(String.self, forKey: .audioShiftText) ?? ""
        formerPaths = try container.decodeIfPresent([String].self, forKey: .formerPaths) ?? []
        sha256 = try container.decodeIfPresent(String.self, forKey: .sha256)
        sha256Bytes = try container.decodeIfPresent(Int64.self, forKey: .sha256Bytes)
    }

    var strength: Float {
        guard let value = Float(scaleText), value.isFinite else { return 1 }
        return value
    }

    /// The hand-set audio shift, or nil (empty or out of the engine's range).
    var audioShift: Float? {
        guard let value = Float(audioShiftText), value > 0, value <= 100 else { return nil }
        return value
    }

    /// "…_turbo_4step_…" / "8-step" style names, as the lightx2v and
    /// FastVideo distillations are published.
    static func detectedSteps(fromFileName name: String) -> Int? {
        guard let match = name.lowercased().firstMatch(of: #/(\d{1,2})[-_ ]?steps?/#),
              let steps = Int(match.1), (1 ... 50).contains(steps) else { return nil }
        return steps
    }
}

/// H3_MAX_LORAS in h3_lora.h - the engine rejects longer stacks.
let maxStackedLoRAs = 8

/// A LoRA file the app recognises by SHA-256: the repository revision it
/// is the object of and the license that revision's model card declared
/// (checked against the primary sources on the date in UPSCALE_PLAN
/// appendix A.5). Shown under the entry; never a judgement of a use.
struct LoRAProvenance: Equatable {
    let repo: String
    let revision: String
    let revisionDate: String
    let declaredLicense: String
    /// The file's size, so only files that could match are hashed.
    let bytes: Int64
    /// What changed since (localized, shown after the declaration).
    let note: String

    var revisionShort: String { String(revision.prefix(8)) }
}

/// Registered H3 checkpoint directories and LoRA files, operated on through
/// ModelManagerView. Persisted in UserDefaults - this only stores small
/// JSON (names/paths/ids), never the multi-GB model or cache files
/// themselves, which stay wherever the user pointed the library at.
@MainActor
final class ModelLibrary: ObservableObject {
    @Published private(set) var models: [H3ModelEntry] = []
    @Published private(set) var activeModelID: UUID?
    @Published private(set) var loras: [LoRAEntry] = []
    @Published private(set) var refmods: [RefModEntry] = []
    /// RefMod headers, read in the background like LoRA headers.
    @Published private var refmodInfoCache: [String: Result<H3RefModInfo, Error>] = [:]
    private var refmodInfoTasks: [String: Task<Result<H3RefModInfo, Error>, Never>] = [:]
    /// h3_lora_inspect results by path (header reads only), filled on a
    /// background task: opening a file can wait indefinitely - e.g. on
    /// macOS's 書類-folder permission prompt - and must never hold up the
    /// main thread. Failures are kept until retryFailedLoRAInfo() so views
    /// don't re-read a bad file on every redraw.
    @Published private var loraInfoCache: [String: Result<H3LoRAInfo, Error>] = [:]
    private var loraInfoTasks: [String: Task<Result<H3LoRAInfo, Error>, Never>] = [:]

    /// Files whose origin is recorded, by SHA-256 (UPSCALE_PLAN appendix A.5).
    static let knownLoRAProvenance: [String: LoRAProvenance] = [
        // lightx2v Turbo v1.2 (minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors),
        // uploaded 2026-09-04; the README declared apache-2.0 from this
        // revision (the one Open-H3-Regenerate-2K pins) up to a29e57d195
        // (2026-10-16), and declares nothing from 9bc1f3c040 (2026-10-19).
        "c3d4a2cf618efea71b9e21a4baaa12d412f1eb6c2b6f86efacaf0ebb6814b689": LoRAProvenance(
            repo: "lightx2v/Minimax-h3-Turbo", revision: "3ec17a324ced54151364f24f8b5fb6bf7e26414f",
            revisionDate: "2026-09-10", declaredLicense: "Apache-2.0", bytes: 1_383_677_808,
            note: String(localized: "配布元の現在の版（2026年10月19日以降）はライセンスを宣言していません。")),
    ]
    private var loraHashTasks: Set<String> = []

    private let defaults = UserDefaults.standard
    private var listsObserver: NSObjectProtocol?
    private static let modelsKey = "H3ModelLibrary.models"
    private static let activeModelKey = "H3ModelLibrary.activeModelID"
    private static let lorasKey = "H3ModelLibrary.loras"
    private static let refmodsKey = "H3ModelLibrary.refmods"
    /// Before stacking: the one selected LoRA. Read once to migrate.
    private static let legacyActiveLoRAKey = "H3ModelLibrary.activeLoRAID"

    var activeModel: H3ModelEntry? { models.first { $0.id == activeModelID } }
    /// The stack the next generation applies, in list order.
    var enabledLoRAs: [LoRAEntry] { loras.filter(\.enabled) }

    init() {
        models = Self.decodeArray(H3ModelEntry.self, key: Self.modelsKey)
        loras = Self.decodeArray(LoRAEntry.self, key: Self.lorasKey)
        refmods = Self.decodeArray(RefModEntry.self, key: Self.refmodsKey)
        syncRefModFolder()
        activeModelID = UUID(uuidString: defaults.string(forKey: Self.activeModelKey) ?? "")
        listsObserver = NotificationCenter.default.addObserver(
            forName: Self.listsChanged, object: nil, queue: .main
        ) { [weak self] note in
            MainActor.assumeIsolated {
                guard let self, (note.object as AnyObject?) !== self else { return }
                self.reloadLists()
            }
        }
        // Read the registered LoRAs' headers in the background right away,
        // so their info is usually ready before anything asks for it.
        for entry in loras { requestLoRAInfo(path: entry.path) }
        for entry in loras { requestLoRAHash(entry) }

        // Upgrading from before this feature existed: seed the one path the
        // app used to hardcode, so an existing setup (and any already-built
        // attention caches - see GenerationViewModel.attentionCacheDirectory(for:))
        // keeps working without the user having to re-add it by hand. Only
        // done when that exact folder is actually there - a brand-new
        // install (e.g. from the distributed .dmg, on another Mac) has
        // nothing at that path and should see an empty library instead,
        // so the download wizard offers defaultH3ModelDownloadPath rather
        // than this app's old name.
        if models.isEmpty, FileManager.default.fileExists(atPath: legacyDefaultH3ModelPath) {
            let seeded = H3ModelEntry(name: "MiniMax-H3", path: legacyDefaultH3ModelPath)
            models = [seeded]
            activeModelID = seeded.id
            persistModels()
            persistActiveModel()
        }

        // LoRAs registered before recommendedSteps existed decode as nil;
        // fill them in from the file name once, so a later deliberate clear
        // isn't undone on every launch.
        if !defaults.bool(forKey: Self.lorasStepsMigratedKey) {
            for index in loras.indices where loras[index].recommendedSteps == nil {
                loras[index].recommendedSteps = LoRAEntry.detectedSteps(
                    fromFileName: (loras[index].path as NSString).lastPathComponent)
            }
            persistLoRAs()
            defaults.set(true, forKey: Self.lorasStepsMigratedKey)
        }

        // Single selection -> stack: the selected LoRA becomes the one
        // enabled entry. Its strength used to replace the adapter's
        // alpha/rank scale outright and now multiplies it, so an old
        // absolute value would be wrong by that factor - reset to 1.0.
        if !defaults.bool(forKey: Self.lorasStackMigratedKey) {
            let legacy = defaults.string(forKey: Self.legacyActiveLoRAKey)
            for index in loras.indices {
                loras[index].enabled = loras[index].id.uuidString == legacy
                loras[index].scaleText = ""
            }
            persistLoRAs()
            defaults.removeObject(forKey: Self.legacyActiveLoRAKey)
            defaults.set(true, forKey: Self.lorasStackMigratedKey)
        }
    }

    private static let lorasStepsMigratedKey = "H3ModelLibrary.lorasStepsMigrated"
    private static let lorasStackMigratedKey = "H3ModelLibrary.lorasStackMigrated"

    // MARK: Models

    func addModel(path: String, name: String? = nil) {
        let entry = H3ModelEntry(name: name ?? (path as NSString).lastPathComponent, path: path)
        models.append(entry)
        if activeModelID == nil {
            activeModelID = entry.id
            persistActiveModel()
        }
        persistModels()
    }

    func renameModel(id: UUID, name: String) {
        guard let index = models.firstIndex(where: { $0.id == id }), !name.isEmpty else { return }
        models[index].name = name
        persistModels()
    }

    /// Used after ModelDownloadWizardView finishes downloading into a
    /// possibly-different folder than the entry originally pointed at.
    func setModelPath(id: UUID, path: String) {
        guard let index = models.firstIndex(where: { $0.id == id }) else { return }
        models[index].path = path
        persistModels()
    }

    func removeModel(id: UUID) {
        models.removeAll { $0.id == id }
        if activeModelID == id {
            activeModelID = models.first?.id
            persistActiveModel()
        }
        persistModels()
    }

    func selectModel(id: UUID) {
        guard models.contains(where: { $0.id == id }) else { return }
        activeModelID = id
        persistActiveModel()
    }

    // MARK: LoRA

    func addLoRA(path: String, name: String? = nil) {
        let fileName = (path as NSString).lastPathComponent
        let entry = LoRAEntry(name: name ?? fileName, path: path,
                              recommendedSteps: LoRAEntry.detectedSteps(fromFileName: fileName))
        loras.append(entry)
        persistLoRAs()
        requestLoRAInfo(path: path)
        requestLoRAHash(entry)
    }

    /// The recorded origin of an entry's file, when its SHA-256 is known
    /// and the file is still the size it was hashed at.
    func provenance(for entry: LoRAEntry) -> LoRAProvenance? {
        guard let sha = entry.sha256, let provenance = Self.knownLoRAProvenance[sha] else { return nil }
        let size = (try? FileManager.default.attributesOfItem(atPath: entry.path)[.size] as? Int64) ?? -1
        return size == entry.sha256Bytes ? provenance : nil
    }

    /// Hashes the file in the background when its size matches a recorded
    /// origin and it was not hashed at that size yet (a 1.4 GB file takes
    /// a second or two, once). Other files are never read whole.
    private func requestLoRAHash(_ entry: LoRAEntry) {
        guard let size = (try? FileManager.default.attributesOfItem(atPath: entry.path)[.size] as? Int64),
              Self.knownLoRAProvenance.values.contains(where: { $0.bytes == size }),
              entry.sha256Bytes != size || entry.sha256 == nil,
              !loraHashTasks.contains(entry.path) else { return }
        loraHashTasks.insert(entry.path)
        let path = entry.path
        Task.detached(priority: .utility) { [weak self] in
            let sha = try? FileHash.sha256(ofFile: path)
            await self?.recordLoRAHash(path: path, size: size, sha: sha)
        }
    }

    private func recordLoRAHash(path: String, size: Int64, sha: String?) {
        loraHashTasks.remove(path)
        guard let sha, let index = loras.firstIndex(where: { $0.path == path }) else { return }
        loras[index].sha256 = sha
        loras[index].sha256Bytes = size
        persistLoRAs()
    }

    /// Empty text clears it (an ordinary, non-distilled LoRA).
    func setLoRARecommendedSteps(id: UUID, text: String) {
        guard let index = loras.firstIndex(where: { $0.id == id }) else { return }
        let steps = Int(text.filter(\.isNumber))
        loras[index].recommendedSteps = steps.map { $0.clamped(to: stepsRange) }
        persistLoRAs()
    }

    func setLoRASampler(id: UUID, _ sampler: LoRASampler) {
        guard let index = loras.firstIndex(where: { $0.id == id }) else { return }
        loras[index].sampler = sampler
        persistLoRAs()
    }

    /// Empty text goes back to the metadata's value (or the engine's 3).
    func setLoRAAudioShift(id: UUID, text: String) {
        guard let index = loras.firstIndex(where: { $0.id == id }) else { return }
        loras[index].audioShiftText = text.filter { $0.isNumber || $0 == "." }
        persistLoRAs()
    }

    func renameLoRA(id: UUID, name: String) {
        guard let index = loras.firstIndex(where: { $0.id == id }), !name.isEmpty else { return }
        loras[index].name = name
        persistLoRAs()
    }

    func setLoRAScale(id: UUID, scaleText: String) {
        guard let index = loras.firstIndex(where: { $0.id == id }) else { return }
        let filtered = scaleText.filter { $0.isNumber || $0 == "." || $0 == "-" }
        loras[index].scaleText = filtered
        persistLoRAs()
    }

    /// The registered LoRA a saved path means: its current file, or one it
    /// was moved from.
    func loraEntry(forPath path: String) -> LoRAEntry? {
        loras.first { $0.path == path } ?? loras.first { $0.formerPaths.contains(path) }
    }

    /// Points an entry at the file's new place after moving it.
    func relocateLoRA(id: UUID, to newPath: String) {
        guard let index = loras.firstIndex(where: { $0.id == id }), loras[index].path != newPath else { return }
        let old = loras[index].path
        if !loras[index].formerPaths.contains(old) { loras[index].formerPaths.append(old) }
        loras[index].formerPaths.removeAll { $0 == newPath }
        loras[index].path = newPath
        if let info = loraInfoCache[old] { loraInfoCache[newPath] = info }
        persistLoRAs()
        requestLoRAInfo(path: newPath)
    }

    /// Moves (or copies) a LoRA file into defaultLoRADirectory, under a
    /// free name, and returns its new path. Off the main actor: a copy to
    /// another volume takes a while for a 1.4 GB file.
    nonisolated static func moveLoRAFileToFolder(_ path: String, copy: Bool) throws -> String {
        try moveFileToFolder(path, directory: defaultLoRADirectory, copy: copy)
    }

    /// The same for a RefMod file and the RefMods folder.
    nonisolated static func moveRefModFileToFolder(_ path: String, copy: Bool) throws -> String {
        try moveFileToFolder(path, directory: defaultRefModDirectory, copy: copy)
    }

    /// A path in `directory` that no file has yet, for a file named after
    /// `stem` (with "name 2", "name 3" when taken). The stem is a file name,
    /// never a path: "/" and NUL become "-", and an empty, "." or ".." stem
    /// becomes "-", so the file always lands directly in `directory`.
    nonisolated static func freePath(in directory: String, stem: String, extension ext: String) -> String {
        let fm = FileManager.default
        // Scalar by scalar: a "/" followed by a combining mark is a single
        // Character, which a String-level replace would not match.
        var scalars = String.UnicodeScalarView()
        for scalar in stem.unicodeScalars {
            scalars.append(scalar == "/" || scalar == "\u{0}" ? "-" : scalar)
        }
        var name = String(scalars)
        if name.isEmpty || name == "." || name == ".." { name = "-" }
        var target = URL(fileURLWithPath: directory).appendingPathComponent(name).appendingPathExtension(ext)
        var number = 2
        while fm.fileExists(atPath: target.path) {
            target = URL(fileURLWithPath: directory).appendingPathComponent("\(name) \(number)").appendingPathExtension(ext)
            number += 1
        }
        return target.path
    }

    nonisolated private static func moveFileToFolder(_ path: String, directory: String, copy: Bool) throws -> String {
        let fm = FileManager.default
        try fm.createDirectory(atPath: directory, withIntermediateDirectories: true)
        let source = URL(fileURLWithPath: path)
        let target = URL(fileURLWithPath: freePath(in: directory, stem: source.deletingPathExtension().lastPathComponent,
                                                   extension: source.pathExtension))
        if copy { try fm.copyItem(at: source, to: target) } else { try fm.moveItem(at: source, to: target) }
        return target.path
    }

    // MARK: RefMods

    /// Registers a RefMod file (already where it should stay). The name
    /// defaults to the file's stem; a path registered twice keeps its entry.
    @discardableResult
    func addRefMod(path: String, name: String? = nil) -> RefModEntry {
        if let existing = refmods.first(where: { $0.path == path }) { return existing }
        let entry = RefModEntry(name: name ?? URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent,
                                path: path)
        refmods.append(entry)
        persistRefMods()
        requestRefModInfo(path: path)
        return entry
    }

    func removeRefMod(id: UUID) {
        refmods.removeAll { $0.id == id }
        persistRefMods()
    }

    func renameRefMod(id: UUID, name: String) {
        guard let index = refmods.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        refmods[index].name = trimmed
        persistRefMods()
    }

    func moveRefMods(fromOffsets source: IndexSet, toOffset destination: Int) {
        refmods.move(fromOffsets: source, toOffset: destination)
        persistRefMods()
    }

    /// Registers .safetensors files in the RefMods folder that aren't
    /// listed yet (made here, or put there by hand), and forgets entries
    /// whose file in that folder is gone. Called when the lists load and
    /// when モデル管理 opens.
    func syncRefModFolder() {
        let fm = FileManager.default
        let names = (try? fm.contentsOfDirectory(atPath: defaultRefModDirectory)) ?? []
        var changed = false
        for name in names.sorted() where name.lowercased().hasSuffix(".safetensors") {
            let path = defaultRefModDirectory + "/" + name
            if refmods.contains(where: { $0.path == path }) { continue }
            refmods.append(RefModEntry(name: URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent,
                                       path: path))
            requestRefModInfo(path: path)
            changed = true
        }
        let before = refmods.count
        refmods.removeAll { $0.isInRefModFolder && !fm.fileExists(atPath: $0.path) }
        if changed || refmods.count != before { persistRefMods() }
    }

    /// The file's header, or nil while it is still being read (the read is
    /// started here; views refresh when it lands).
    func refmodInfo(path: String) -> Result<H3RefModInfo, Error>? {
        if let cached = refmodInfoCache[path] { return cached }
        requestRefModInfo(path: path)
        return nil
    }

    func refmodInfo(for entry: RefModEntry) -> Result<H3RefModInfo, Error>? { refmodInfo(path: entry.path) }

    /// Waits - without blocking the main thread - until this file's header
    /// has been read.
    func loadRefModInfo(path: String) async -> Result<H3RefModInfo, Error> {
        if let cached = refmodInfoCache[path] { return cached }
        let result = await requestRefModInfo(path: path).value
        refmodInfoCache[path] = result
        return result
    }

    func retryFailedRefModInfo() {
        for (path, result) in refmodInfoCache {
            if case .failure = result { refmodInfoCache[path] = nil }
        }
    }

    /// Forgets a file's header (after it was rewritten in place).
    func forgetRefModInfo(path: String) {
        refmodInfoCache[path] = nil
    }

    @discardableResult
    private func requestRefModInfo(path: String) -> Task<Result<H3RefModInfo, Error>, Never> {
        if let running = refmodInfoTasks[path] { return running }
        let task = Task.detached(priority: .userInitiated) {
            Result { try H3RefModInfo.inspect(path: path) }
        }
        refmodInfoTasks[path] = task
        Task { [weak self] in
            let result = await task.value
            guard let self else { return }
            self.refmodInfoTasks[path] = nil
            self.refmodInfoCache[path] = result
        }
        return task
    }

    func removeLoRA(id: UUID) {
        loras.removeAll { $0.id == id }
        persistLoRAs()
    }

    /// Enabling past maxStackedLoRAs is refused (returns false).
    @discardableResult
    func setLoRAEnabled(id: UUID, _ enabled: Bool) -> Bool {
        guard let index = loras.firstIndex(where: { $0.id == id }) else { return false }
        if enabled && !loras[index].enabled && enabledLoRAs.count >= maxStackedLoRAs { return false }
        loras[index].enabled = enabled
        persistLoRAs()
        return true
    }

    /// Replaces the whole stack: exactly `ids` end up enabled.
    func setEnabledLoRAs(_ ids: Set<UUID>) {
        for index in loras.indices { loras[index].enabled = ids.contains(loras[index].id) }
        persistLoRAs()
    }

    /// Stack order is list order; it decides which Turbo LoRA's step count
    /// wins when more than one is on.
    func moveLoRAs(fromOffsets source: IndexSet, toOffset destination: Int) {
        loras.move(fromOffsets: source, toOffset: destination)
        persistLoRAs()
    }

    /// The file's header info, or nil while it is still being read (the
    /// read is started here; views refresh when it lands).
    func loraInfo(for entry: LoRAEntry) -> Result<H3LoRAInfo, Error>? {
        if let cached = loraInfoCache[entry.path] { return cached }
        requestLoRAInfo(path: entry.path)
        return nil
    }

    func isLoRAInfoPending(_ entry: LoRAEntry) -> Bool { loraInfoCache[entry.path] == nil }

    /// Waits - without blocking the main thread - until these entries'
    /// headers have been read. Generating calls this first, so 自動
    /// sampling and the file check use the real metadata.
    func loadLoRAInfo(for entries: [LoRAEntry]) async {
        for entry in entries where loraInfoCache[entry.path] == nil {
            loraInfoCache[entry.path] = await requestLoRAInfo(path: entry.path).value
        }
    }

    /// Forgets failed reads, so a file that was put back (or a permission
    /// granted since) is picked up. Called when モデル管理 opens.
    func retryFailedLoRAInfo() {
        for (path, result) in loraInfoCache {
            if case .failure = result { loraInfoCache[path] = nil }
        }
    }

    @discardableResult
    private func requestLoRAInfo(path: String) -> Task<Result<H3LoRAInfo, Error>, Never> {
        if let running = loraInfoTasks[path] { return running }
        let task = Task.detached(priority: .userInitiated) {
            Result { try H3LoRAInfo.inspect(path: path) }
        }
        loraInfoTasks[path] = task
        Task { [weak self] in
            let result = await task.value
            guard let self else { return }
            self.loraInfoTasks[path] = nil
            self.loraInfoCache[path] = result
        }
        return task
    }

    /// What one entry asks of the step rule and the audio shift: its own
    /// setting, else what its file's metadata declares. Names are never
    /// used to guess, and nothing here is specific to one adapter.
    func sampling(for entry: LoRAEntry) -> (renoise: Bool, audioShift: Float?) {
        let info = try? loraInfo(for: entry)?.get()
        let renoise: Bool
        switch entry.sampler {
        case .euler: renoise = false
        case .renoise: renoise = true
        case .auto: renoise = info?.recommendsRenoise ?? false
        }
        return (renoise, entry.audioShift ?? info?.recommendedAudioShift.map { Float($0) })
    }

    // MARK: Persistence

    /// Every window has its own library (each runs its own model), but the
    /// lists are one set in UserDefaults: after writing them, tell the other
    /// libraries to reload, so none writes back a list it read earlier and
    /// undoes another window's add or remove.
    static let listsChanged = Notification.Name("H3ModelLibrary.listsChanged")

    private func persistModels() {
        Self.encodeArray(models, key: Self.modelsKey)
        NotificationCenter.default.post(name: Self.listsChanged, object: self)
    }

    private func persistLoRAs() {
        Self.encodeArray(loras, key: Self.lorasKey)
        NotificationCenter.default.post(name: Self.listsChanged, object: self)
    }

    private func persistRefMods() {
        Self.encodeArray(refmods, key: Self.refmodsKey)
        NotificationCenter.default.post(name: Self.listsChanged, object: self)
    }

    private func reloadLists() {
        models = Self.decodeArray(H3ModelEntry.self, key: Self.modelsKey)
        loras = Self.decodeArray(LoRAEntry.self, key: Self.lorasKey)
        refmods = Self.decodeArray(RefModEntry.self, key: Self.refmodsKey)
        // Each window keeps its own selection, unless it has none (or its
        // model was removed elsewhere): then it takes the one last chosen.
        if activeModel == nil {
            let stored = UUID(uuidString: defaults.string(forKey: Self.activeModelKey) ?? "")
            activeModelID = models.first { $0.id == stored }?.id ?? models.first?.id
        }
        for entry in loras where loraInfoCache[entry.path] == nil { requestLoRAInfo(path: entry.path) }
    }

    private func persistActiveModel() {
        defaults.set(activeModelID?.uuidString, forKey: Self.activeModelKey)
    }

    private static func decodeArray<T: Decodable>(_ type: T.Type, key: String) -> [T] {
        guard let data = UserDefaults.standard.data(forKey: key),
              let value = try? JSONDecoder().decode([T].self, from: data) else { return [] }
        return value
    }

    private static func encodeArray<T: Encodable>(_ value: [T], key: String) {
        guard let data = try? JSONEncoder().encode(value) else { return }
        UserDefaults.standard.set(data, forKey: key)
    }
}


/// One line describing a RefMod member: "画像 32×32（256トークン、Full）",
/// "動画 2.0秒 ..." or "音声 3.0秒（240トークン）", with its own name first
/// inside a bundle.
func refmodMemberSummary(_ member: H3RefModMember, inBundle: Bool) -> String {
    let kind: String
    switch member.kind {
    case .image: kind = String(localized: "画像")
    case .video: kind = String(localized: "動画")
    case .audio: kind = String(localized: "音声")
    }
    var parts: [String] = []
    if inBundle && !member.name.isEmpty { parts.append(member.name + ": ") }
    if member.isStack {
        // Still images stacked one per latent frame, not a clip.
        parts.append(String(localized: "画像\(member.latentT)枚"))
        parts.append(" \(member.latentW)×\(member.latentH)")
        let mode = member.mode == .compressed ? String(localized: "、圧縮") : String(localized: "、Full")
        parts.append(String(localized: "（\(member.tokens)トークン\(mode)）"))
        return parts.joined()
    }
    parts.append(kind)
    if member.kind == .audio {
        parts.append(String(localized: " \(String(format: "%.1f", member.seconds ?? 0))秒"))
    } else {
        parts.append(" \(member.latentW)×\(member.latentH)")
        if member.latentT > 1 { parts.append(String(localized: "×\(member.latentT)コマ")) }
    }
    let mode = member.kind == .audio ? "" :
        (member.mode == .compressed ? String(localized: "、圧縮") : String(localized: "、Full"))
    parts.append(String(localized: "（\(member.tokens)トークン\(mode)）"))
    return parts.joined()
}
