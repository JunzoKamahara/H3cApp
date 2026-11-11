import Foundation
import H3Engine

/// A named set of generation settings: the composer's shape / size / length
/// and everything in 詳細設定 (LoRAs included). The prompt and the creation
/// method are not part of it. Stored in UserDefaults.
struct SettingsPreset: Codable, Identifiable, Equatable {
    struct LoRA: Codable, Equatable {
        var path: String
        var strength: Float
    }

    var id = UUID()
    var name: String
    var sizeProfile: String
    var seconds: Int
    var steps: Int
    var denoiseReuse: Int
    var computeMode: String
    var speedMode: String
    var fastAttention: Bool
    var ditLayers: Int
    var seedFixed: Bool
    var seed: String
    var loras: [LoRA]
    /// つなげて生成 and its clip length; nil for the defaults (one
    /// generation, ChainClipLength.seven), so presets saved before 0.5 still
    /// match a form that uses them.
    var chained: Bool? = nil
    var chainClipFrames: Int? = nil
}

/// Presets are shared by every window: saving in one shows up in the
/// others. Also remembers the preset used last, and the default preset:
/// the one new windows without a project start from, and the settings
/// "変更あり" is measured against.
@MainActor
final class PresetStore: ObservableObject {
    static let shared = PresetStore()

    private static let presetsKey = "h3c-app.settingsPresets"
    private static let lastUsedKey = "h3c-app.lastUsedPresetID"
    /// An empty string means "no default" (the app's own defaults).
    private static let defaultKey = "h3c-app.defaultPresetID"

    @Published private(set) var presets: [SettingsPreset]
    @Published private(set) var lastUsedID: UUID?
    @Published private(set) var defaultID: UUID?

    private init() {
        if let data = UserDefaults.standard.data(forKey: Self.presetsKey),
           let decoded = try? JSONDecoder().decode([SettingsPreset].self, from: data) {
            presets = decoded
        } else {
            presets = []
        }
        lastUsedID = UserDefaults.standard.string(forKey: Self.lastUsedKey).flatMap(UUID.init)
        if let stored = UserDefaults.standard.string(forKey: Self.defaultKey) {
            defaultID = UUID(uuidString: stored)
        } else {
            // Before there was a default, new windows took the preset used
            // last; that one becomes the default.
            defaultID = lastUsedID
            UserDefaults.standard.set(lastUsedID?.uuidString ?? "", forKey: Self.defaultKey)
        }
    }

    var lastUsed: SettingsPreset? { presets.first { $0.id == lastUsedID } }
    var defaultPreset: SettingsPreset? { presets.first { $0.id == defaultID } }

    func setDefault(id: UUID?) {
        defaultID = id
        UserDefaults.standard.set(id?.uuidString ?? "", forKey: Self.defaultKey)
    }

    /// Saves under the preset's name, replacing one of the same name, and
    /// returns what was stored.
    @discardableResult
    func save(_ preset: SettingsPreset) -> SettingsPreset {
        var stored = preset
        if let index = presets.firstIndex(where: { $0.name == preset.name }) {
            stored.id = presets[index].id
            presets[index] = stored
        } else {
            presets.append(stored)
        }
        persist()
        markUsed(stored)
        return stored
    }

    func delete(id: UUID) {
        presets.removeAll { $0.id == id }
        if lastUsedID == id { lastUsedID = nil; UserDefaults.standard.removeObject(forKey: Self.lastUsedKey) }
        if defaultID == id { setDefault(id: nil) }
        persist()
    }

    func markUsed(_ preset: SettingsPreset) {
        lastUsedID = preset.id
        UserDefaults.standard.set(preset.id.uuidString, forKey: Self.lastUsedKey)
    }

    private func persist() {
        if let data = try? JSONEncoder().encode(presets) {
            UserDefaults.standard.set(data, forKey: Self.presetsKey)
        }
    }
}

extension SettingsPreset {
    /// Same settings, regardless of id and name.
    func hasSameSettings(as other: SettingsPreset) -> Bool {
        var a = self, b = other
        a.id = b.id; a.name = b.name
        return a == b
    }
}

extension GenerationViewModel {
    var presets: [SettingsPreset] { PresetStore.shared.presets }

    /// The preset this window last applied or saved, and whether the form
    /// still matches it.
    var activePreset: (preset: SettingsPreset, modified: Bool)? {
        guard let id = activePresetID,
              let preset = PresetStore.shared.presets.first(where: { $0.id == id }) else { return nil }
        return (preset, !preset.hasSameSettings(as: presetFromCurrentSettings()))
    }

    /// The form's current settings, ready to be named and saved.
    func presetFromCurrentSettings() -> SettingsPreset {
        SettingsPreset(
            name: "", sizeProfile: sizeProfile.rawValue, seconds: seconds,
            steps: steps, denoiseReuse: denoiseReuse, computeMode: computeMode.rawValue,
            speedMode: speedMode.rawValue, fastAttention: fastAttention, ditLayers: ditLayers,
            seedFixed: seedFixed, seed: seedText,
            loras: effectiveLoRAs.map { .init(path: $0.path, strength: $0.strength) },
            chained: isChained ? true : nil,
            chainClipFrames: chainClipLength == defaultChainClipLength ? nil : chainClipLength.frames)
    }

    /// The settings a finished generation actually used.
    func presetFromResult(_ result: ResolvedResult) -> SettingsPreset {
        SettingsPreset(
            name: "", sizeProfile: result.sizeProfile.rawValue, seconds: result.requestedSeconds,
            steps: result.steps, denoiseReuse: result.denoiseReuse,
            computeMode: result.computeMode.rawValue, speedMode: result.speedMode.rawValue,
            fastAttention: result.fastAttention, ditLayers: result.ditLayers,
            seedFixed: !result.seedWasRandom, seed: result.seedDecimalString,
            loras: result.loras.map { .init(path: $0.path, strength: $0.strength) },
            chained: result.chained ? true : nil,
            chainClipFrames: result.chainClipFrames.flatMap { $0 == defaultChainClipLength.frames ? nil : $0 })
    }

    /// Saves under the given name, replacing a preset of the same name;
    /// it becomes this window's active preset and the one new windows use.
    func savePreset(_ preset: SettingsPreset, named name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        var named = preset
        named.name = trimmed
        activePresetID = PresetStore.shared.save(named).id
    }

    func deletePreset(id: UUID) {
        PresetStore.shared.delete(id: id)
        if activePresetID == id { activePresetID = nil }
    }

    /// A new window without a project (and the first one at launch when
    /// no project reopens): no prompt or input files, and the default
    /// settings, so nothing reads "変更あり".
    func startBlankDraft() {
        var draft = currentDraft()
        draft.prompt = ""
        draft.creationMethod = CreationMethod.text.rawValue
        draft.imageInputMode = ImageInputMode.firstLastFrame.rawValue
        draft.firstFrame = nil
        draft.lastFrame = nil
        draft.references = []
        draft.forcedAudio = nil
        draft.forcedAudioOffset = nil
        draft.continueFrom = nil
        draft.continueFrames = nil
        draft.chained = nil
        draft.clipPrompts = nil
        draft.clipReferences = nil
        draft.chainCustom = nil
        draft.customClips = nil
        applyDraft(draft, projectURL: nil)
        batchCount = 1
        applyDefaultSettings()
    }

    /// The default preset, or the app's own defaults when none is set.
    func applyDefaultSettings() {
        if let preset = PresetStore.shared.defaultPreset {
            applyPreset(preset)
        } else {
            activePresetID = nil
            chained = false
            resetAdvancedSettings()
            seconds = 5
        }
    }

    /// Whether the settings differ from the default preset (or, with none,
    /// the app's defaults) - the "変更あり" on 詳細設定. Length and
    /// つなげて生成 live in the composer and don't count; nor does the seed
    /// text while the seed isn't fixed.
    var hasSettingsChanges: Bool {
        guard let preset = PresetStore.shared.defaultPreset else { return hasAdvancedChanges }
        var current = presetFromCurrentSettings()
        current.seconds = preset.seconds
        current.chained = preset.chained
        if !current.seedFixed && !preset.seedFixed { current.seed = preset.seed }
        return !preset.hasSameSettings(as: current)
    }

    /// Puts a preset's settings into the form (the prompt is left alone).
    func applyPreset(_ preset: SettingsPreset) {
        activePresetID = preset.id
        PresetStore.shared.markUsed(preset)
        if let profile = SizeProfile(rawValue: preset.sizeProfile) { sizeProfile = profile }
        chained = preset.chained ?? false
        chainClipLength = preset.chainClipFrames.flatMap(ChainClipLength.init(rawValue:)) ?? defaultChainClipLength
        seconds = preset.seconds.clamped(to: currentSecondsRange)
        if let speed = SpeedMode(rawValue: preset.speedMode) { speedMode = speed }
        denoiseReuse = preset.denoiseReuse.clamped(to: reuseRange)
        ditLayers = preset.ditLayers.clamped(to: ditLayersRange)
        if let mode = ComputeMode(rawValue: preset.computeMode),
           mode != .attentionCache || supportsInt8Cache {
            computeMode = mode
        }
        fastAttention = preset.fastAttention && fastAttentionAvailable
        seedText = preset.seed
        seedFixed = preset.seedFixed
        applyLoRAs(preset.loras.map { ($0.path, $0.strength) })
        // After the LoRAs: enabling a Turbo LoRA moves steps to its
        // recommended count, and the preset's own value should win - unless
        // a HyperFlow model is active, which only takes 4/6/8.
        steps = preset.steps.clamped(to: stepsRange)
        if let choices = fixedSteps, !choices.contains(steps) {
            steps = modelFamily.defaultSteps
        }
    }
}
