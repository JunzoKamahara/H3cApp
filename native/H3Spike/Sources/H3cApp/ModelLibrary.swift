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

/// Registered H3 checkpoint directories and LoRA files, operated on through
/// ModelManagerView. Persisted in UserDefaults - this only stores small
/// JSON (names/paths/ids), never the multi-GB model or cache files
/// themselves, which stay wherever the user pointed the library at.
@MainActor
final class ModelLibrary: ObservableObject {
    @Published private(set) var models: [H3ModelEntry] = []
    @Published private(set) var activeModelID: UUID?
    @Published private(set) var loras: [LoRAEntry] = []
    /// h3_lora_inspect results by path (header reads only), filled on a
    /// background task: opening a file can wait indefinitely - e.g. on
    /// macOS's 書類-folder permission prompt - and must never hold up the
    /// main thread. Failures are kept until retryFailedLoRAInfo() so views
    /// don't re-read a bad file on every redraw.
    @Published private var loraInfoCache: [String: Result<H3LoRAInfo, Error>] = [:]
    private var loraInfoTasks: [String: Task<Result<H3LoRAInfo, Error>, Never>] = [:]

    private let defaults = UserDefaults.standard
    private var listsObserver: NSObjectProtocol?
    private static let modelsKey = "H3ModelLibrary.models"
    private static let activeModelKey = "H3ModelLibrary.activeModelID"
    private static let lorasKey = "H3ModelLibrary.loras"
    /// Before stacking: the one selected LoRA. Read once to migrate.
    private static let legacyActiveLoRAKey = "H3ModelLibrary.activeLoRAID"

    var activeModel: H3ModelEntry? { models.first { $0.id == activeModelID } }
    /// The stack the next generation applies, in list order.
    var enabledLoRAs: [LoRAEntry] { loras.filter(\.enabled) }

    init() {
        models = Self.decodeArray(H3ModelEntry.self, key: Self.modelsKey)
        loras = Self.decodeArray(LoRAEntry.self, key: Self.lorasKey)
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
        let fm = FileManager.default
        try fm.createDirectory(atPath: defaultLoRADirectory, withIntermediateDirectories: true)
        let source = URL(fileURLWithPath: path)
        let base = source.deletingPathExtension().lastPathComponent
        let ext = source.pathExtension
        var target = URL(fileURLWithPath: defaultLoRADirectory).appendingPathComponent(source.lastPathComponent)
        var number = 2
        while fm.fileExists(atPath: target.path) {
            target = URL(fileURLWithPath: defaultLoRADirectory)
                .appendingPathComponent("\(base) \(number)").appendingPathExtension(ext)
            number += 1
        }
        if copy { try fm.copyItem(at: source, to: target) } else { try fm.moveItem(at: source, to: target) }
        return target.path
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

    private func reloadLists() {
        models = Self.decodeArray(H3ModelEntry.self, key: Self.modelsKey)
        loras = Self.decodeArray(LoRAEntry.self, key: Self.lorasKey)
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
