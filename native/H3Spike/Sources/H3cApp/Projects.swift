import AppKit
import Foundation
import H3Engine

// A project is a folder (by default ~/Movies/H3cApp/<name>) that keeps
// everything needed to make its videos again:
//
//   project.json                       the form: prompt, settings, references
//   references/                        copies of the images/videos/audio used
//   20261004-153012_seed123.mp4        each generated video, kept until the
//   20261004-153012_seed123.json       user deletes it, with the full request
//
// Paths inside the folder are stored relative to it, so a project can be
// moved or copied as a whole. Without an open project the app behaves as
// before: the result is a temp file that the next generation replaces.

/// Everything the form holds that affects the output - enough to make the
/// same video again (with the seed pinned).
struct ProjectDraft: Codable, Equatable {
    struct Reference: Codable, Equatable {
        var kind: String
        var path: String
    }

    var prompt: String
    var creationMethod: String
    var imageInputMode: String
    var firstFrame: String?
    var lastFrame: String?
    var references: [Reference]
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
    var loras: [SettingsPreset.LoRA]
    /// Lip sync: the audio file held as the soundtrack, and where in it
    /// the video starts (seconds). Absent in drafts from before 0.5.
    var forcedAudio: String? = nil
    var forcedAudioOffset: Double? = nil
    /// The project video this one continues, and how many of its last
    /// frames were carried over. Absent in drafts from before 0.5.
    var continueFrom: String? = nil
    var continueFrames: Int? = nil
    /// つなげて生成: the length is made by joining clips of
    /// chainClipFrames frames (ChainGeneration.swift).
    var chained: Bool? = nil
    var chainClipFrames: Int? = nil
    /// つなげて生成: what each clip adds to the prompt, in plan order (an
    /// empty one makes that clip from the prompt alone). Absent without any.
    var clipPrompts: [String]? = nil
    /// つなげて生成 with reference images: whether each clip uses them, in
    /// plan order (missing ones do). Absent while every clip does.
    var clipReferences: [Bool]? = nil
    /// カスタムクリップ (CustomClips.swift): the length is made of these
    /// clips (while chainCustom); kept when switching back to a length.
    var chainCustom: Bool? = nil
    var customClips: [ChainClipSpec]? = nil

    /// The draft with every path in customClips passed through `map`.
    func mappingClipPaths(_ map: (String) -> String) -> ProjectDraft {
        var draft = self
        draft.customClips = customClips?.map { $0.mappingPaths(map) }
        return draft
    }
}

/// project.json
struct ProjectFile: Codable {
    var version = 1
    var name: String
    var createdAt: Date
    var updatedAt: Date
    var batchCount: Int
    var draft: ProjectDraft
}

/// The .json next to each generated video: the request exactly as it ran.
struct ProjectVideoRecord: Codable {
    var version = 1
    var video: String
    var completedAt: Date
    var seed: String
    var seedWasRandom: Bool
    var generationSeconds: Double
    var actualFrameCount: Int
    var fps: Int
    var effectiveDenoiseReuse: Int
    var ccvAttentionCalls: Int
    var ccvAttentionDirectCalls: Int
    var deviceLine: String
    var appVersion: String
    /// The form as it ran, with this video's seed pinned.
    var draft: ProjectDraft
    /// Set only when they differ from the engine defaults (Euler, audio
    /// shift 3): "renoise" and the audio shift a LoRA's metadata asked for.
    var sampler: String? = nil
    var audioSigmaShift: Double? = nil
    /// How many seconds this video's sound runs behind its picture: a
    /// continuation that carried the earlier clip's sound over inherits that
    /// clip's lateness and adds the engine's snapping to the audio grid
    /// (JoinedVideoExporter.soundDelay). Joining moves each clip's sound
    /// earlier by it. latentSoundLateness is the lateness of the sound in the
    /// video's .h3lt, which a joined video takes from its last clip.
    var soundLateness: Double? = nil
    var latentSoundLateness: Double? = nil
    /// A joined video: its clips (paths in the project, oldest first), and
    /// how many a つなげて生成 planned (more than joined if it was stopped).
    var joinedClips: [String]? = nil
    var chainPlannedClips: Int? = nil
    /// A video brought in to be continued (動画ファイルの続きを作る): the
    /// file it came from. Not generated - its settings are the form's then.
    var importedFrom: String? = nil
    /// An enlarged copy (2倍に拡大した動画をつくる): the project video it was
    /// made from and the upscaler (VideoUpscaler). The rest is that video's.
    var upscaledFrom: String? = nil
    var upscaler: String? = nil
    /// Its frame size then, [width, height] (the draft keeps the generated one).
    var upscaledSize: [Int]? = nil
}

struct OpenProject: Equatable {
    var url: URL
    var name: String
    var createdAt: Date
}

struct ProjectVideo: Identifiable, Equatable {
    var url: URL
    var record: ProjectVideoRecord?
    var id: String { url.path }

    static func == (a: ProjectVideo, b: ProjectVideo) -> Bool {
        a.url == b.url && a.record?.completedAt == b.record?.completedAt
    }
}

struct BatchProgress: Equatable {
    var index: Int
    var total: Int
    /// The clips of a つなげて生成, rather than separate videos.
    var chained = false
}

/// The whole batch's (or chain's) time while it runs: the current video's
/// live estimate plus the ones still to come.
struct BatchTiming: Equatable {
    var elapsed: Double
    var remaining: Double?
    var finishDate: Date?
    var fraction: Double?
}

let batchCountRange = 1 ... 20

enum ProjectError: LocalizedError {
    case notAProject(String)
    case alreadyOpen(String)
    case alreadyExists(String)
    case invalidName

    var errorDescription: String? {
        switch self {
        case .notAProject(let path):
            return String(localized: "プロジェクトのフォルダではありません（project.jsonがありません）: \(path)")
        case .alreadyOpen(let name):
            return String(localized: "「\(name)」は別のウィンドウで開いています。")
        case .alreadyExists(let path):
            return String(localized: "同じ名前のフォルダがすでにあります: \(path)")
        case .invalidName:
            return String(localized: "プロジェクト名を入力してください。")
        }
    }
}

/// Recently opened projects and the one to reopen at launch, shared by all
/// windows; also which projects are open, so two windows never write the
/// same project.json.
@MainActor
final class ProjectStore: ObservableObject {
    static let shared = ProjectStore()

    private static let recentsKey = "h3c-app.recentProjects"
    private static let lastOpenKey = "h3c-app.lastProjectPath"
    private static let maxRecents = 10

    @Published private(set) var recents: [URL]
    /// Open project folders and the window (view model) holding each. Weak,
    /// so a window that went away without closing its project never keeps
    /// the project locked.
    private var openOwners: [String: WeakOwner] = [:]
    /// Only the first window reopens the last project at launch.
    var didReopenAtLaunch = false
    /// A project action chosen in the menu bar while no window was open; the
    /// window opened for it performs it (see ContentView).
    var pendingAction: PendingProjectAction?

    private struct WeakOwner {
        weak var owner: GenerationViewModel?
    }

    private init() {
        recents = (UserDefaults.standard.stringArray(forKey: Self.recentsKey) ?? [])
            .map { URL(fileURLWithPath: $0, isDirectory: true) }
    }

    /// ~/Movies/H3cApp
    static var defaultParentDirectory: URL {
        let movies = FileManager.default.urls(for: .moviesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory() + "/Movies", isDirectory: true)
        return movies.appendingPathComponent("H3cApp", isDirectory: true)
    }

    var lastOpened: URL? {
        UserDefaults.standard.string(forKey: Self.lastOpenKey).map { URL(fileURLWithPath: $0, isDirectory: true) }
    }

    /// Recent projects whose project.json still exists.
    var existingRecents: [URL] {
        recents.filter { FileManager.default.fileExists(atPath: $0.appendingPathComponent("project.json").path) }
    }

    func noteOpened(_ url: URL, by owner: GenerationViewModel) {
        openOwners[url.standardizedFileURL.path] = WeakOwner(owner: owner)
        recents.removeAll { $0.standardizedFileURL == url.standardizedFileURL }
        recents.insert(url, at: 0)
        recents = Array(recents.prefix(Self.maxRecents))
        UserDefaults.standard.set(recents.map(\.path), forKey: Self.recentsKey)
        UserDefaults.standard.set(url.path, forKey: Self.lastOpenKey)
    }

    func noteClosed(_ url: URL, explicitly: Bool) {
        openOwners[url.standardizedFileURL.path] = nil
        if explicitly { UserDefaults.standard.removeObject(forKey: Self.lastOpenKey) }
    }

    /// The live model that has this project open, if any.
    func owner(of url: URL) -> GenerationViewModel? {
        let path = url.standardizedFileURL.path
        guard let owner = openOwners[path]?.owner,
              owner.project?.url.standardizedFileURL.path == path else {
            openOwners[path] = nil
            return nil
        }
        return owner
    }

    /// Whether a window shows this project. A project opened over the API
    /// while no window shows the primary model doesn't count.
    func isOpen(_ url: URL) -> Bool {
        owner(of: url).map { AppModels.shared.isShown($0) } ?? false
    }
}

enum PendingProjectAction {
    case new
    case choose
    case open(URL)
}

// MARK: - Files

enum ProjectFiles {
    static let projectFileName = "project.json"
    static let referencesFolder = "references"

    static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    static func read(_ projectURL: URL) throws -> ProjectFile {
        let url = projectURL.appendingPathComponent(projectFileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ProjectError.notAProject(projectURL.path)
        }
        return try decoder.decode(ProjectFile.self, from: Data(contentsOf: url))
    }

    static func write(_ file: ProjectFile, to projectURL: URL) throws {
        try encoder.encode(file).write(to: projectURL.appendingPathComponent(projectFileName), options: .atomic)
    }

    /// A folder name from a project name: no path separators or colons.
    static func folderName(for name: String) -> String {
        name.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: ":", with: "-")
    }

    /// Stored form of a path: relative when it is inside the project.
    static func stored(_ path: String, in projectURL: URL) -> String {
        let root = projectURL.standardizedFileURL.path + "/"
        let standardized = URL(fileURLWithPath: path).standardizedFileURL.path
        return standardized.hasPrefix(root) ? String(standardized.dropFirst(root.count)) : path
    }

    static func resolved(_ stored: String, in projectURL: URL) -> String {
        stored.hasPrefix("/") ? stored : projectURL.appendingPathComponent(stored).path
    }

    /// Whether a path is already in the project's references folder.
    static func isImported(_ path: String, in projectURL: URL) -> Bool {
        let folder = projectURL.appendingPathComponent(referencesFolder, isDirectory: true)
            .standardizedFileURL.path + "/"
        return URL(fileURLWithPath: path).standardizedFileURL.path.hasPrefix(folder)
    }

    /// Copies a file into the project's references folder (once: an identical
    /// file already there is reused) and returns the copy's path. Only files
    /// already in references/ stay where they are - a generated video used as
    /// a reference is copied too, so deleting the video doesn't break it (on
    /// APFS the copy is a clone). Reads and copies files: call it off the
    /// main actor.
    static func importReference(_ path: String, into projectURL: URL) throws -> String {
        if isImported(path, in: projectURL) { return path }
        let folder = projectURL.appendingPathComponent(referencesFolder, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let source = URL(fileURLWithPath: path)
        let stem = source.deletingPathExtension().lastPathComponent
        let ext = source.pathExtension
        var destination = folder.appendingPathComponent(source.lastPathComponent)
        var counter = 2
        while FileManager.default.fileExists(atPath: destination.path) {
            if FileManager.default.contentsEqual(atPath: destination.path, andPath: path) {
                return destination.path
            }
            destination = folder.appendingPathComponent(ext.isEmpty ? "\(stem) \(counter)" : "\(stem) \(counter).\(ext)")
            counter += 1
        }
        try FileManager.default.copyItem(at: source, to: destination)
        return destination.path
    }

    /// "20261004-153012_seed123", unique within the folder.
    static func videoBaseName(completedAt: Date, seed: UInt64, in folder: URL) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let base = "\(formatter.string(from: completedAt))_seed\(seed)"
        var name = base
        var counter = 2
        while FileManager.default.fileExists(atPath: folder.appendingPathComponent(name + ".mp4").path) {
            name = "\(base)-\(counter)"
            counter += 1
        }
        return name
    }

    static func videos(in projectURL: URL) -> [ProjectVideo] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: projectURL.path)) ?? []
        let videos = names.filter { $0.lowercased().hasSuffix(".mp4") }.map { name -> ProjectVideo in
            let url = projectURL.appendingPathComponent(name)
            let recordURL = url.deletingPathExtension().appendingPathExtension("json")
            let record = (try? Data(contentsOf: recordURL)).flatMap { try? decoder.decode(ProjectVideoRecord.self, from: $0) }
            return ProjectVideo(url: url, record: record)
        }
        // Newest first; the file names sort by completion time too.
        return videos.sorted { $0.url.lastPathComponent > $1.url.lastPathComponent }
    }
}

extension H3ReferenceKind {
    var storedName: String {
        switch self {
        case .image: return "image"
        case .video: return "video"
        case .audio: return "audio"
        case .videoAudio: return "videoAudio"
        }
    }

    init?(storedName: String) {
        switch storedName {
        case "image": self = .image
        case "video": self = .video
        case "audio": self = .audio
        case "videoAudio": self = .videoAudio
        default: return nil
        }
    }
}

// MARK: - The window's project

extension GenerationViewModel {
    /// The form as a draft. Paths stay absolute here; ProjectFiles.stored
    /// makes them relative when written.
    func currentDraft() -> ProjectDraft {
        ProjectDraft(
            prompt: prompt,
            creationMethod: creationMethod.rawValue,
            imageInputMode: imageInputMode.rawValue,
            firstFrame: firstFramePath,
            lastFrame: lastFramePath,
            references: referenceImages.map { .init(kind: $0.kind.storedName, path: $0.path) },
            sizeProfile: sizeProfile.rawValue,
            seconds: seconds,
            steps: steps,
            denoiseReuse: denoiseReuse,
            computeMode: computeMode.rawValue,
            speedMode: speedMode.rawValue,
            fastAttention: fastAttention,
            ditLayers: ditLayers,
            seedFixed: seedFixed,
            seed: seedText,
            loras: effectiveLoRAs.map { .init(path: $0.path, strength: $0.strength) },
            forcedAudio: forcedAudioPath,
            forcedAudioOffset: forcedAudioPath == nil ? nil : forcedAudioOffset,
            continueFrom: continuation?.videoURL.path,
            continueFrames: continuation?.frames,
            chained: isChained ? true : nil,
            chainClipFrames: chainClipLength == defaultChainClipLength ? nil : chainClipLength.frames,
            clipPrompts: storedClipPrompts,
            clipReferences: storedClipReferences,
            chainCustom: chainCustom ? true : nil,
            customClips: customClips.isEmpty ? nil : customClips)
    }

    /// Puts a draft into the form. Relative paths resolve against
    /// `projectURL`. Returns the LoRA files that aren't registered (they
    /// can't be switched on).
    @discardableResult
    func applyDraft(_ draft: ProjectDraft, projectURL: URL?) -> [String] {
        func resolve(_ path: String) -> String {
            projectURL.map { ProjectFiles.resolved(path, in: $0) } ?? path
        }
        prompt = draft.prompt
        if let method = CreationMethod(rawValue: draft.creationMethod) { creationMethod = method }
        if let mode = ImageInputMode(rawValue: draft.imageInputMode) { imageInputMode = mode }
        firstFramePath = draft.firstFrame.map(resolve)
        lastFramePath = draft.lastFrame.map(resolve)
        forcedAudioPath = draft.forcedAudio.map(resolve)
        forcedAudioOffset = draft.forcedAudioOffset ?? 0
        continuation = draft.continueFrom.map { URL(fileURLWithPath: resolve($0)) }.flatMap { video in
            let latent = ProjectFiles.latentURL(for: video)
            guard FileManager.default.fileExists(atPath: latent.path) else { return nil }
            return ContinuationSource(videoURL: video, latentURL: latent, sizeProfileRaw: draft.sizeProfile,
                                      frames: draft.continueFrames ?? continuationFrames)
        }
        referenceImages = draft.references.compactMap { reference in
            H3ReferenceKind(storedName: reference.kind).map { H3ReferenceInput(kind: $0, path: resolve(reference.path)) }
        }
        if let profile = SizeProfile(rawValue: draft.sizeProfile) { sizeProfile = profile }
        chained = draft.chained ?? false
        chainClipLength = draft.chainClipFrames.flatMap(ChainClipLength.init(rawValue:)) ?? defaultChainClipLength
        clipPrompts = draft.clipPrompts ?? []
        clipReferences = draft.clipReferences ?? []
        chainCustom = draft.chainCustom ?? false
        customClips = (draft.customClips ?? []).map { $0.mappingPaths(resolve) }
        seconds = draft.seconds.clamped(to: currentSecondsRange)
        if let speed = SpeedMode(rawValue: draft.speedMode) { speedMode = speed }
        denoiseReuse = draft.denoiseReuse.clamped(to: reuseRange)
        ditLayers = draft.ditLayers.clamped(to: ditLayersRange)
        if let mode = ComputeMode(rawValue: draft.computeMode), mode != .attentionCache || supportsInt8Cache {
            computeMode = mode
        }
        fastAttention = draft.fastAttention && fastAttentionAvailable
        seedText = draft.seed
        seedFixed = draft.seedFixed
        let missing = draft.loras.filter { library.loraEntry(forPath: $0.path) == nil }.map(\.path)
        applyLoRAs(draft.loras.map { ($0.path, $0.strength) })
        // After the LoRAs: a Turbo LoRA moves steps to its own count, and
        // the draft's value should win.
        steps = draft.steps.clamped(to: stepsRange)
        activePresetID = nil
        return missing
    }

    // MARK: Open / create / close

    func createProject(name: String, parent: URL) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { projectMessage = ProjectError.invalidName.localizedDescription; return }
        let url = parent.appendingPathComponent(ProjectFiles.folderName(for: trimmed), isDirectory: true)
        do {
            if FileManager.default.fileExists(atPath: url.path) { throw ProjectError.alreadyExists(url.path) }
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            let now = Date()
            let file = ProjectFile(name: trimmed, createdAt: now, updatedAt: now, batchCount: 1,
                                   draft: currentDraft())
            try ProjectFiles.write(file, to: url)
            // The current form becomes the project's; its references are
            // copied in on the first save below.
            try attachProject(url: url, file: file, applyingDraft: false)
            saveProjectIfChanged(force: true)
        } catch {
            projectMessage = error.localizedDescription
        }
    }

    func openProject(at url: URL) {
        do {
            let file = try ProjectFiles.read(url)
            try attachProject(url: url, file: file, applyingDraft: true)
        } catch {
            projectMessage = error.localizedDescription
        }
    }

    // MARK: Several videos without a project

    /// The composer's generate action. Several videos are kept in a
    /// project, so with none open it asks for one first (BatchProjectSheet),
    /// which generates once the project is set - or not at all if cancelled.
    func requestGenerate() {
        guard canGenerate else { return }
        if project == nil && (batchCount > 1 || isChained) {
            projectMessage = nil
            showingBatchProject = true
        } else {
            generate()
        }
    }

    /// BatchProjectSheet: a new project from the current form, then the batch.
    func createProjectAndGenerate(name: String, parent: URL) -> Bool {
        let count = batchCount
        createProject(name: name, parent: parent)
        return generateBatchInAttachedProject(count: count)
    }

    /// Whether an existing project's saved prompt and settings differ from
    /// the form's, so choosing it would overwrite them. Input files compare
    /// by name: one from outside the project is copied in under its name.
    func projectDraftDiffers(at url: URL) -> Bool {
        guard let file = try? ProjectFiles.read(url) else { return false }
        func byName(_ draft: ProjectDraft) -> ProjectDraft {
            var draft = draft
            let name = { (path: String) in URL(fileURLWithPath: path).lastPathComponent }
            draft.firstFrame = draft.firstFrame.map(name)
            draft.lastFrame = draft.lastFrame.map(name)
            draft.forcedAudio = draft.forcedAudio.map(name)
            draft.continueFrom = draft.continueFrom.map(name)
            draft.references = draft.references.map { .init(kind: $0.kind, path: name($0.path)) }
            return draft
        }
        return byName(currentDraft()) != byName(file.draft)
    }

    /// BatchProjectSheet: an existing project, keeping the current form (it
    /// replaces the project's saved one, as a new project takes the form)
    /// rather than loading the project's, then the batch.
    func openProjectAndGenerate(at url: URL) -> Bool {
        let count = batchCount
        do {
            let file = try ProjectFiles.read(url)
            try attachProject(url: url, file: file, applyingDraft: false)
        } catch {
            projectMessage = error.localizedDescription
            return false
        }
        return generateBatchInAttachedProject(count: count)
    }

    private func generateBatchInAttachedProject(count: Int) -> Bool {
        guard project != nil else { return false }
        // Attaching loads the project's own count; the one chosen stays.
        batchCount = count
        saveProjectIfChanged(force: true)
        generate()
        return true
    }

    // MARK: Save as project

    /// The temp video in the preview with its record, if saving as a project
    /// can take it along.
    var savableTemporaryResult: (url: URL, record: ProjectVideoRecord)? {
        guard !isGenerating, let resultURL, isTemporaryResult(resultURL),
              let record = temporaryResultRecord, record.video == resultURL.lastPathComponent,
              FileManager.default.fileExists(atPath: resultURL.path) else { return nil }
        return (resultURL, record)
    }

    /// プロジェクトとして保存: a new project from the current form (as
    /// 新規プロジェクト), also moving in the temp video shown in the preview
    /// with its record.
    func saveAsProject(name: String, parent: URL) {
        let pending = savableTemporaryResult
        createProject(name: name, parent: parent)
        guard let project, let pending else { return }
        guard let stored = storeInProject(videoAt: pending.url, projectURL: project.url,
                                          record: pending.record,
                                          completedAt: pending.record.completedAt,
                                          seed: UInt64(pending.record.seed) ?? 0) else { return }
        temporaryResultRecord = nil
        resultURL = stored
        importRecordReferences(
            at: stored.deletingPathExtension().appendingPathExtension("json"), projectURL: project.url)
        if continueAfterSavingAsProject {
            continueAfterSavingAsProject = false
            reloadProjectVideos()
            if let video = projectVideos.first(where: { $0.url == stored }) { continueFromProjectVideo(video) }
        }
    }

    /// A record made outside a project points at its input files where they
    /// were: copy them into the project (an identical file already there is
    /// reused) off the main actor, and point the record at the copies.
    private func importRecordReferences(at recordURL: URL, projectURL: URL) {
        Task { [weak self] in
            let changed = await Task.detached(priority: .utility) { () -> Bool in
                guard let data = try? Data(contentsOf: recordURL),
                      var record = try? ProjectFiles.decoder.decode(ProjectVideoRecord.self, from: data)
                else { return false }
                func imported(_ path: String) -> String {
                    guard path.hasPrefix("/"), FileManager.default.fileExists(atPath: path),
                          let copy = try? ProjectFiles.importReference(path, into: projectURL)
                    else { return path }
                    return ProjectFiles.stored(copy, in: projectURL)
                }
                let before = record.draft
                record.draft.firstFrame = record.draft.firstFrame.map(imported)
                record.draft.lastFrame = record.draft.lastFrame.map(imported)
                record.draft.references = record.draft.references.map {
                    .init(kind: $0.kind, path: imported($0.path))
                }
                guard record.draft != before,
                      let encoded = try? ProjectFiles.encoder.encode(record) else { return false }
                return (try? encoded.write(to: recordURL, options: .atomic)) != nil
            }.value
            guard let self, changed, self.project?.url == projectURL else { return }
            self.reloadProjectVideos()
        }
    }

    func chooseAndOpenProject() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = ProjectStore.defaultParentDirectory
        if panel.runModal() == .OK, let url = panel.url { openProject(at: url) }
    }

    /// At launch, the first window reopens the project that was open when
    /// the app quit. Returns whether it did.
    func reopenLastProjectIfAny() -> Bool {
        let store = ProjectStore.shared
        guard !store.didReopenAtLaunch else { return false }
        store.didReopenAtLaunch = true
        guard let url = store.lastOpened,
              FileManager.default.fileExists(atPath: url.appendingPathComponent(ProjectFiles.projectFileName).path)
        else { return false }
        openProject(at: url)
        return project != nil
    }

    private func attachProject(url: URL, file: ProjectFile, applyingDraft: Bool) throws {
        if project?.url.standardizedFileURL == url.standardizedFileURL { return }
        if let owner = ProjectStore.shared.owner(of: url), owner !== self {
            guard !AppModels.shared.isShown(owner) else { throw ProjectError.alreadyOpen(file.name) }
            // Opened over the API with no window showing the primary
            // model: hand it over.
            owner.closeProject(explicitly: false)
        }
        closeProject(explicitly: false)
        if applyingDraft {
            let missing = applyDraft(file.draft, projectURL: url)
            if !missing.isEmpty {
                let names = missing.map { URL(fileURLWithPath: $0).lastPathComponent }.joined(separator: ", ")
                projectMessage = String(localized: "登録されていない追加モデル（LoRA）は使えません: \(names)")
            }
        }
        batchCount = file.batchCount.clamped(to: batchCountRange)
        project = OpenProject(url: url, name: file.name, createdAt: file.createdAt)
        lastSavedProjectFile = file
        ProjectStore.shared.noteOpened(url, by: self)
        reloadProjectVideos()
        startProjectAutosave()
    }

    /// Saves the form first. `explicitly` (the user closed it, rather than
    /// opening another) also stops it reopening at the next launch.
    func closeProject(explicitly: Bool = true) {
        guard let project else { return }
        saveProjectIfChanged()
        projectAutosave = nil
        ProjectStore.shared.noteClosed(project.url, explicitly: explicitly)
        // A video of the closed project leaves the preview with it, so the
        // next window (or the form without a project) starts empty. A temp
        // result from before the project, or one still being made, stays.
        let folder = project.url.standardizedFileURL.path + "/"
        if !isGenerating, let resultURL,
           resultURL.standardizedFileURL.path.hasPrefix(folder) {
            self.resultURL = nil
            lastResult = nil
        }
        self.project = nil
        projectVideos = []
        lastSavedProjectFile = nil
    }

    func reloadProjectVideos() {
        projectVideos = project.map { ProjectFiles.videos(in: $0.url) } ?? []
    }

    // MARK: Saving

    private func startProjectAutosave() {
        projectAutosave = objectWillChange.merge(with: library.objectWillChange)
            .debounce(for: .seconds(1), scheduler: RunLoop.main)
            .sink { [weak self] in
                MainActor.assumeIsolated { self?.saveProjectIfChanged() }
            }
    }

    /// Writes project.json when something changed. References from outside
    /// the project are copied into it in the background (startReferenceImport),
    /// which saves again with the copies once done.
    func saveProjectIfChanged(force: Bool = false, importingReferences: Bool = true) {
        guard let project, var file = lastSavedProjectFile else { return }
        if importingReferences { startReferenceImport() }
        var draft = currentDraft()
        draft.firstFrame = draft.firstFrame.map { ProjectFiles.stored($0, in: project.url) }
        draft.lastFrame = draft.lastFrame.map { ProjectFiles.stored($0, in: project.url) }
        draft.forcedAudio = draft.forcedAudio.map { ProjectFiles.stored($0, in: project.url) }
        draft.continueFrom = draft.continueFrom.map { ProjectFiles.stored($0, in: project.url) }
        draft.references = draft.references.map { .init(kind: $0.kind, path: ProjectFiles.stored($0.path, in: project.url)) }
        draft = draft.mappingClipPaths { ProjectFiles.stored($0, in: project.url) }
        guard force || draft != file.draft || batchCount != file.batchCount else { return }
        file.draft = draft
        file.batchCount = batchCount
        file.updatedAt = Date()
        do {
            try ProjectFiles.write(file, to: project.url)
            lastSavedProjectFile = file
        } catch {
            projectMessage = String(localized: "プロジェクトを保存できませんでした（詳細: \(error.localizedDescription)）")
        }
    }

    /// The form's input files that still have to be copied into the project.
    private func referencesToImport(for projectURL: URL) -> [String] {
        var paths = [firstFramePath, lastFramePath, forcedAudioPath].compactMap { $0 } + referenceImages.map(\.path)
            + customClips.flatMap(\.paths)
        paths = paths.filter {
            !ProjectFiles.isImported($0, in: projectURL) && FileManager.default.fileExists(atPath: $0)
                && !failedReferenceImports.contains(projectURL.path + "\n" + $0)
        }
        return Array(Set(paths))
    }

    /// Copies outside references into the project off the main actor (large
    /// videos would otherwise stall every window), then points the form at
    /// the copies and saves. One copy job at a time; generate() waits for it.
    func startReferenceImport() {
        guard referenceImportTask == nil, let project else { return }
        let projectURL = project.url
        let paths = referencesToImport(for: projectURL)
        guard !paths.isEmpty else { return }
        referenceImportTask = Task { [weak self] in
            let results = await Task.detached(priority: .userInitiated) { () -> [String: Result<String, Error>] in
                var results: [String: Result<String, Error>] = [:]
                for path in paths {
                    results[path] = Result { try ProjectFiles.importReference(path, into: projectURL) }
                }
                return results
            }.value
            guard let self else { return }
            self.referenceImportTask = nil
            // Opened another project meanwhile: its own save handles it.
            guard self.project?.url == projectURL else { return }
            var failure: Error?
            for case let (path, .failure) in results {
                self.failedReferenceImports.insert(projectURL.path + "\n" + path)
            }
            func copy(of path: String) -> String {
                switch results[path] {
                case .success(let copy)?: return copy
                case .failure(let error)?: failure = error; return path
                case nil: return path
                }
            }
            if let path = self.firstFramePath, case let new = copy(of: path), new != path { self.firstFramePath = new }
            if let path = self.lastFramePath, case let new = copy(of: path), new != path { self.lastFramePath = new }
            if let path = self.forcedAudioPath, case let new = copy(of: path), new != path { self.forcedAudioPath = new }
            for index in self.referenceImages.indices {
                let path = self.referenceImages[index].path
                let new = copy(of: path)
                if new != path { self.referenceImages[index].path = new }
            }
            let clips = self.customClips.map { $0.mappingPaths { copy(of: $0) } }
            if clips != self.customClips { self.customClips = clips }
            if let failure {
                self.projectMessage = String(localized: "参照ファイルをプロジェクトにコピーできませんでした（詳細: \(failure.localizedDescription)）")
            }
            // Not importing again: a failed copy would just be retried (and
            // later autosaves skip it - failedReferenceImports).
            self.saveProjectIfChanged(importingReferences: false)
        }
    }

    /// Moves a finished temp video into the project under its date/seed
    /// name and writes its record next to it. Returns the new location.
    func storeInProject(videoAt tempURL: URL, projectURL: URL, record: ProjectVideoRecord,
                        completedAt: Date, seed: UInt64, folder: URL? = nil) -> URL? {
        // A つなげて生成's clips go in their own folder; the joined video is
        // the one in the project's list.
        let folder = folder ?? projectURL
        let base = ProjectFiles.videoBaseName(completedAt: completedAt, seed: seed, in: folder)
        let destination = folder.appendingPathComponent(base + ".mp4")
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try FileManager.default.moveItem(at: tempURL, to: destination)
        } catch {
            projectMessage = String(localized: "動画をプロジェクトに保存できませんでした（詳細: \(error.localizedDescription)）")
            return nil
        }
        // Its latents, for continuing it; a video without them just can't be.
        let latent = ProjectFiles.latentURL(for: tempURL)
        if FileManager.default.fileExists(atPath: latent.path) {
            try? FileManager.default.moveItem(at: latent, to: ProjectFiles.latentURL(for: destination))
        }
        do {
            var stored = record
            stored.video = destination.lastPathComponent
            try ProjectFiles.encoder.encode(stored)
                .write(to: folder.appendingPathComponent(base + ".json"), options: .atomic)
        } catch {
            projectMessage = String(localized: "動画をプロジェクトに保存できませんでした（詳細: \(error.localizedDescription)）")
            // Put the video back where the preview expects it; if even that
            // fails, keep it in the project as a video without a record.
            if (try? FileManager.default.moveItem(at: destination, to: tempURL)) != nil { return nil }
        }
        if project?.url == projectURL { reloadProjectVideos() }
        return destination
    }

    // MARK: Project videos

    /// Shows a saved video in the preview.
    func showProjectVideo(_ video: ProjectVideo) {
        guard !isGenerating else { return }
        deleteTemporaryPreview()
        resultURL = video.url
        lastResult = video.record.map { ResolvedResult(record: $0) }
        if let profile = video.record.flatMap({ SizeProfile(rawValue: $0.draft.sizeProfile) }) {
            resultAspectRatio = CGFloat(profile.dimensions.width) / CGFloat(profile.dimensions.height)
        }
        if let url = resultURL { loadActualDuration(for: url) }
    }

    /// Puts a saved video's request back into the form with its seed
    /// pinned, so generating makes the same video.
    func restoreProjectVideo(_ video: ProjectVideo) {
        guard let record = video.record, let project else { return }
        var draft = record.draft
        draft.seedFixed = true
        draft.seed = record.seed
        let missing = applyDraft(draft, projectURL: project.url)
        if !missing.isEmpty {
            let names = missing.map { URL(fileURLWithPath: $0).lastPathComponent }.joined(separator: ", ")
            projectMessage = String(localized: "登録されていない追加モデル（LoRA）は使えません: \(names)")
        }
    }

    /// Moves the video and its record to the Trash (recoverable).
    func deleteProjectVideo(_ video: ProjectVideo) {
        let record = video.url.deletingPathExtension().appendingPathExtension("json")
        let clips = chainFolder(of: video)
        do {
            if let clips { try FileManager.default.trashItem(at: clips, resultingItemURL: nil) }
            try FileManager.default.trashItem(at: video.url, resultingItemURL: nil)
            if FileManager.default.fileExists(atPath: record.path) {
                try FileManager.default.trashItem(at: record, resultingItemURL: nil)
            }
            let latent = ProjectFiles.latentURL(for: video.url)
            if FileManager.default.fileExists(atPath: latent.path) {
                try FileManager.default.trashItem(at: latent, resultingItemURL: nil)
            }
            if continuation?.videoURL == video.url { continuation = nil }
            if resultURL == video.url { resultURL = nil; lastResult = nil }
        } catch {
            projectMessage = String(localized: "動画を削除できませんでした（詳細: \(error.localizedDescription)）")
        }
        reloadProjectVideos()
    }

    /// Adds a video as a reference: switches the form to 参照画像・動画・音声.
    /// A temp result (no project open) is copied first, since the next
    /// generation would delete it.
    func useVideoAsReference(_ url: URL) {
        var path = url.path
        if project == nil && isTemporaryResult(url) {
            let folder = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("H3cApp-references", isDirectory: true)
            try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let copy = folder.appendingPathComponent(url.lastPathComponent)
            try? FileManager.default.removeItem(at: copy)
            if (try? FileManager.default.copyItem(at: url, to: copy)) != nil { path = copy.path }
        }
        creationMethod = .image
        imageInputMode = .referenceImage
        if !referenceImages.contains(where: { $0.path == path }) {
            referenceImages.append(H3ReferenceInput(kind: .video, path: path))
        }
    }

    func isTemporaryResult(_ url: URL) -> Bool {
        (url.lastPathComponent.hasPrefix("H3cApp_") || url.lastPathComponent.hasPrefix("h3c-app_"))
            && url.standardizedFileURL.path.hasPrefix(URL(fileURLWithPath: NSTemporaryDirectory()).standardizedFileURL.path)
    }

    static var appVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "dev"
    }
}

extension ResolvedResult {
    /// A saved video's record as a result, for the preview and 使用した設定.
    init(record: ProjectVideoRecord) {
        let draft = record.draft
        let profile = SizeProfile(rawValue: draft.sizeProfile) ?? defaultSizeProfile
        let method = CreationMethod(rawValue: draft.creationMethod) ?? .text
        self.init(
            prompt: draft.prompt,
            creationMethod: method,
            imageInputMode: method == .image ? ImageInputMode(rawValue: draft.imageInputMode) : nil,
            sizeProfile: profile,
            requestedSeconds: draft.seconds,
            requestedFrames: Int(h3AlignedFrameCount(seconds: Double(draft.seconds))),
            actualFrameCount: record.actualFrameCount,
            fps: record.fps,
            actualDurationSeconds: nil,
            steps: draft.steps,
            denoiseReuse: draft.denoiseReuse,
            effectiveDenoiseReuse: record.effectiveDenoiseReuse,
            ditLayers: draft.ditLayers,
            computeMode: ComputeMode(rawValue: draft.computeMode) ?? .attentionCache,
            speedMode: SpeedMode(rawValue: draft.speedMode) ?? .quality,
            fastAttention: draft.fastAttention,
            ccvAttentionCalls: record.ccvAttentionCalls,
            ccvAttentionDirectCalls: record.ccvAttentionDirectCalls,
            seed: UInt64(record.seed) ?? 0,
            seedWasRandom: record.seedWasRandom,
            loras: draft.loras.map {
                ResolvedLoRA(name: URL(fileURLWithPath: $0.path).deletingPathExtension().lastPathComponent,
                             path: $0.path, strength: $0.strength)
            },
            sampling: .init(renoise: record.sampler == "renoise",
                            audioShift: Float(record.audioSigmaShift ?? 0)),
            references: referenceNames(draft),
            deviceLine: record.deviceLine,
            completedAt: record.completedAt,
            generationSeconds: record.generationSeconds,
            forcedAudio: draft.forcedAudio.map { URL(fileURLWithPath: $0).lastPathComponent },
            forcedAudioOffset: draft.forcedAudioOffset ?? 0,
            continuedFrom: draft.continueFrom.map { URL(fileURLWithPath: $0).lastPathComponent },
            continueFrames: draft.continueFrames ?? continuationFrames,
            chained: draft.chained ?? false,
            joinedClipCount: record.joinedClips?.count,
            importedFrom: record.importedFrom,
            upscaledFrom: record.upscaledFrom,
            upscaler: record.upscaler.flatMap(VideoUpscaler.init(rawValue:)),
            plannedClipCount: record.chainPlannedClips,
            chainClipFrames: (draft.chained ?? false) ? (draft.chainClipFrames ?? defaultChainClipLength.frames) : nil,
            clipPrompts: (draft.chained ?? false) ? draft.clipPrompts : nil,
            clipReferences: (draft.chained ?? false) ? draft.clipReferences : nil,
            customClips: (draft.chained ?? false) && (draft.chainCustom ?? false) ? draft.customClips : nil)
    }
}

/// File names of the inputs a draft's mode actually uses.
func referenceNames(_ draft: ProjectDraft) -> [String] {
    guard draft.creationMethod == CreationMethod.image.rawValue else { return [] }
    if draft.imageInputMode == ImageInputMode.firstLastFrame.rawValue {
        return [draft.firstFrame, draft.lastFrame].compactMap { $0 }.map { URL(fileURLWithPath: $0).lastPathComponent }
    }
    return draft.references.map { URL(fileURLWithPath: $0.path).lastPathComponent }
}
