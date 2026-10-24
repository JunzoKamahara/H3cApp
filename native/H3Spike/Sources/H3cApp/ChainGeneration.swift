import AVFoundation
import Foundation

/// つなげて生成: lengths past what one generation can make (15 s) come from
/// several clips, each continuing the one before (Continuation.swift), then
/// joined into one video. Measured on a 24 GB M5 at 672x384, 15 s this way
/// took about 23% less time than one 362-frame generation
/// (long-vs-chain/REPORT.md), so 15 s is offered chained as well.
let chainedSecondsRange = 15 ... 90

/// How long each clip of a chain is (詳細設定). Both lengths leave
/// frames past the carried-over 22 that fill whole 40 Hz audio steps, so the
/// carried sound isn't snapped off the picture (JoinedVideoExporter.soundDelay).
enum ChainClipLength: Int, CaseIterable, Identifiable {
    case five = 124   // 5.2 s
    case seven = 175  // 7.3 s

    var id: Int { rawValue }
    var frames: Int { rawValue }

    var label: String {
        switch self {
        case .five: return String(localized: "約5秒")
        case .seven: return String(localized: "約7秒")
        }
    }
}

let defaultChainClipLength = ChainClipLength.seven

/// The engine's frame counts: 5 + 17k, up to 362.
func alignedFrames(atLeast frames: Int) -> Int {
    let needed = max(5, frames)
    return 5 + (needed - 5 + 16) / 17 * 17
}

/// The clips a chain of `seconds` is made of, in frames: full-length clips,
/// each carrying over `overlap` frames of the one before, and a shorter last
/// one for what is left (skipped when under half a second). The total is the
/// first clip plus each later clip minus the overlap.
func chainPlan(seconds: Int, clip: ChainClipLength, overlap: Int = continuationFrames) -> [Int] {
    let target = seconds * 24
    var plan = [clip.frames]
    var total = clip.frames
    let step = clip.frames - overlap
    while target - total >= step {
        plan.append(clip.frames)
        total += step
    }
    let left = target - total
    if left >= 12 {
        // A continuation must be longer than what it carries over.
        plan.append(min(362, alignedFrames(atLeast: max(overlap + left, overlap + 17))))
    }
    return plan
}

func chainTotalFrames(_ plan: [Int], overlap: Int = continuationFrames) -> Int {
    plan.enumerated().reduce(0) { $0 + ($1.offset == 0 ? $1.element : $1.element - overlap) }
}

/// A clip's prompt in a つなげて生成 with クリップごとの指示: the form's
/// prompt (what stays the same: the people, their clothes, the style), then
/// what happens in this clip. Without a clip's text it is the prompt alone,
/// so every clip repeats it as before.
func chainClipPrompt(common: String, clip: String?) -> String {
    let clip = (clip ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    // Unchanged, so a chain without them is made exactly as before.
    if clip.isEmpty { return common }
    let common = common.trimmingCharacters(in: .whitespacesAndNewlines)
    return common.isEmpty ? clip : common + "\n\n" + clip
}

extension GenerationViewModel {
    /// Whether the form's length is made by joining clips.
    var isChained: Bool {
        chained && (chainCustom ? !customClips.isEmpty : seconds >= chainedSecondsRange.lowerBound)
    }

    /// The seconds the form may hold: 1-15 in one generation, 15-90 chained.
    var currentSecondsRange: ClosedRange<Int> { chained ? chainedSecondsRange : secondsRange }

    var currentChainPlan: [Int] {
        usesCustomClips ? customClips.map(\.frames)
            : chainPlan(seconds: seconds.clamped(to: chainedSecondsRange), clip: chainClipLength)
    }

    /// The クリップごとの指示 as saved: up to the last one with text, none
    /// when all are empty.
    var storedClipPrompts: [String]? {
        guard let last = clipPrompts.lastIndex(where: {
            !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }) else { return nil }
        return Array(clipPrompts[...last])
    }

    /// The per-clip reference switches as saved: up to the last one that is
    /// off, none while every clip uses them.
    var storedClipReferences: [Bool]? {
        guard let last = clipReferences.lastIndex(of: false) else { return nil }
        return Array(clipReferences[...last])
    }

    /// Whether clip `index` of a chain uses the reference images.
    func chainClipUsesReferences(_ flags: [Bool]?, index: Int) -> Bool {
        guard let flags, index < flags.count else { return true }
        return flags[index]
    }

    /// Picks a length from the duration menu (カスタムクリップ is
    /// useCustomClips).
    func setDuration(seconds value: Int, chained wantsChain: Bool) {
        let chain = wantsChain || value > secondsRange.upperBound
        chained = chain
        chainCustom = false
        seconds = value.clamped(to: chain ? chainedSecondsRange : secondsRange)
    }

    /// The composer chip's text: "5秒", "つないで15秒", "30秒・つなぐ",
    /// "カスタム・3本".
    var durationLabel: String {
        guard isChained else { return String(localized: "\(seconds)秒") }
        if usesCustomClips { return String(localized: "カスタム・\(customClips.count)本") }
        return seconds == chainedSecondsRange.lowerBound
            ? String(localized: "つないで15秒") : String(localized: "\(seconds)秒・つなぐ")
    }

    /// The whole chain's estimate: each clip from this Mac's timings, a
    /// continuation about 20% longer for its carried-over frames
    /// (continue-test/REPORT.md).
    var chainEstimatedSeconds: Double? {
        chainClipEstimates(plan: currentChainPlan)?.reduce(0, +)
    }

    /// Each clip's estimate, in plan order (see chainEstimatedSeconds).
    func chainClipEstimates(plan: [Int]) -> [Double]? {
        let calibration = TimingCalibration.load(for: computeMode, speed: speedMode,
                                                 fastAttention: useFastAttention,
                                                 ditLayers: ditLayers.clamped(to: ditLayersRange))
        let now = Date()
        var estimates: [Double] = []
        for (index, frames) in plan.enumerated() {
            let shape = progressShape(sizeProfile: sizeProfile, frames: frames,
                                      steps: steps.clamped(to: stepsRange), reuse: effectiveDenoiseReuse)
            guard let seconds = ProgressEstimator(shape: shape, calibration: calibration, start: now)
                .snapshot(now: now).remaining else { return nil }
            estimates.append(index == 0 ? seconds : seconds * 1.2)
        }
        return estimates
    }
}

/// One つなげて生成 while it runs: its plan, where its clips go, and the
/// clips made so far.
final class ChainRun {
    /// Each clip's frames (from clipPlans).
    let plan: [Int]
    let clipPlans: [ChainClipPlan]
    let folder: URL
    let baseName: String
    let baseSeed: UInt64
    let seedWasRandom: Bool
    let startedAt = Date()
    /// The whole chain's request (the first clip's settings).
    let request: GenerationViewModel.GenerationRequest
    var clips: [(url: URL, record: ProjectVideoRecord)] = []

    init(plans: [ChainClipPlan], projectURL: URL, fixedSeed: UInt64?,
         request: GenerationViewModel.GenerationRequest) {
        clipPlans = plans
        plan = plans.map(\.frames)
        baseSeed = fixedSeed ?? UInt64.random(in: 0 ... UInt64.max / 2)
        seedWasRandom = fixedSeed == nil
        let chains = projectURL.appendingPathComponent("chains", isDirectory: true)
        baseName = ProjectFiles.videoBaseName(completedAt: Date(), seed: baseSeed, in: projectURL)
        folder = chains.appendingPathComponent(baseName, isDirectory: true)
        self.request = request
    }
}

extension ProjectFiles {
    /// The frame count in a latent file's header (h3_latent_file_write).
    static func latentFrames(_ url: URL) -> Int? {
        guard let handle = try? FileHandle(forReadingFrom: url),
              let data = try? handle.read(upToCount: 36), data.count == 36 else { return nil }
        try? handle.close()
        let words = data.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        return words[0] == 0x544c3348 ? Int(words[8]) : nil
    }
}

extension GenerationViewModel {
    /// How late a continuation's sound will run (ProjectVideoRecord
    /// .soundLateness): the source's lateness plus the engine's snapping.
    /// None with forced audio, whose sound is the file's own.
    func continuationSoundLateness(source: ProjectVideoRecord?, latentURL: URL, forced: Bool,
                                   overlap: Int = continuationFrames) -> Double {
        guard !forced else { return 0 }
        let inherited = source?.latentSoundLateness ?? source?.soundLateness ?? 0
        let frames = ProjectFiles.latentFrames(latentURL) ?? source?.actualFrameCount ?? 0
        return inherited + JoinedVideoExporter.soundDelay(previousFrames: frames,
                                                          overlapFrames: overlap, fps: 24)
    }

    /// Clip `index` of a chain (its ChainClipPlan): the chain's request at
    /// the clip's length with the clip's own prompt (chainClipPrompt),
    /// images and lip sync; after the first, continuing the clip before
    /// unless it starts from a first frame of its own. The form's reference
    /// images go to every clip unless the clip says otherwise, so the people
    /// keep looking the same even after leaving the frame; the form's first
    /// and last frame only shape the first clip.
    func chainClipRequest(_ chain: ChainRun, index: Int) -> GenerationRequest {
        var request = chain.request
        let clip = chain.clipPlans[index]
        let frames = clip.frames
        request.params.frames = Int32(frames)
        request.requestedFrames = frames
        request.requestedSeconds = Int((Double(frames) / 24).rounded())
        request.storeFolder = chain.folder
        request.shape = progressShape(sizeProfile: request.sizeProfile, frames: frames,
                                      steps: request.steps, reuse: request.effectiveDenoiseReuse)
        var draft = chain.request.draft
        draft.chained = nil
        draft.chainClipFrames = nil
        draft.chainCustom = nil
        draft.customClips = nil
        draft.seconds = request.requestedSeconds
        let projectURL = request.projectURL
        func stored(_ path: String) -> String { projectURL.map { ProjectFiles.stored(path, in: $0) } ?? path }

        // Images.
        if index > 0 {
            request.params.firstFrame = nil
            request.params.lastFrame = nil
            draft.firstFrame = nil
            draft.lastFrame = nil
        }
        switch clip.images {
        case .form:
            break
        case .none:
            request.params.references = []
            draft.references = []
        case .references(let references):
            request.params.references = references
            draft.references = references.map { .init(kind: $0.kind.storedName, path: stored($0.path)) }
        case .frames(let first, let last):
            request.params.references = []
            request.params.firstFrame = first
            request.params.lastFrame = last
            draft.references = []
            draft.firstFrame = first.map(stored)
            draft.lastFrame = last.map(stored)
        }
        let usesFrames = request.params.firstFrame != nil || request.params.lastFrame != nil
        let usesReferences = !request.params.references.isEmpty
        if usesReferences || usesFrames {
            request.creationMethod = .image
            request.imageInputMode = usesReferences ? .referenceImage : .firstLastFrame
            draft.creationMethod = CreationMethod.image.rawValue
            draft.imageInputMode = (usesReferences ? ImageInputMode.referenceImage : .firstLastFrame).rawValue
        } else {
            request.creationMethod = .text
            request.imageInputMode = nil
            draft.creationMethod = CreationMethod.text.rawValue
        }
        if request.params.attentionCachePath != nil {
            request.params.attentionCachePath = attentionCachePath(usesReferences: usesReferences)
        }

        // Lip sync.
        request.params.forcedAudioPath = clip.audioPath
        request.params.forcedAudioOffset = clip.audioPath == nil ? 0 : clip.audioOffset
        draft.forcedAudio = clip.audioPath.map(stored)
        draft.forcedAudioOffset = clip.audioPath == nil ? nil : clip.audioOffset

        // Continuing the clip before.
        if index > 0 {
            request.params.continueLatentPath = nil
            draft.continueFrom = nil
            draft.continueFrames = nil
            request.soundLateness = 0
            if clip.continues, let previous = chain.clips.last {
                let latent = ProjectFiles.latentURL(for: previous.url)
                request.params.continueLatentPath = latent.path
                request.params.continueFrames = Int32(continuationFrames)
                request.soundLateness = continuationSoundLateness(
                    source: previous.record, latentURL: latent, forced: clip.audioPath != nil)
                draft.continueFrom = stored(previous.url.path)
                draft.continueFrames = continuationFrames
            }
        }

        request.prompt = chainClipPrompt(common: chain.request.prompt, clip: clip.text)
        request.referenceNames = referenceNames(draft)
        // The clip's record holds what it was made from; the joined video's
        // keeps the form's prompt and the clips' own settings.
        draft.prompt = request.prompt
        draft.clipPrompts = nil
        draft.clipReferences = nil
        request.draft = draft
        return request
    }

    /// The end of a つなげて生成 (also a stopped one): two or more clips are
    /// joined into the project's own folder as one video with a record; a
    /// single clip just moves there.
    func finishChain(_ chain: ChainRun) async {
        guard let projectURL = chain.request.projectURL else { return }
        let clips = chain.clips
        defer { if project?.url == projectURL { reloadProjectVideos() } }
        guard clips.count > 1, let last = clips.last else {
            if let clip = clips.first {
                let destination = projectURL.appendingPathComponent(clip.url.lastPathComponent)
                for ext in ["mp4", "json", "h3lt"] {
                    try? FileManager.default.moveItem(
                        at: clip.url.deletingPathExtension().appendingPathExtension(ext),
                        to: destination.deletingPathExtension().appendingPathExtension(ext))
                }
                if resultURL == clip.url { resultURL = destination }
            }
            try? FileManager.default.removeItem(at: chain.folder)
            return
        }
        phase = String(localized: "\(clips.count)本の動画をつないでいます")
        var draft = chain.request.draft
        draft.seed = String(chain.baseSeed)
        let completedAt = Date()
        let record = ProjectVideoRecord(
            video: "", completedAt: completedAt, seed: String(chain.baseSeed),
            seedWasRandom: chain.seedWasRandom,
            generationSeconds: completedAt.timeIntervalSince(chain.startedAt),
            actualFrameCount: joinedFrameCount(clips.map(\.record)),
            fps: last.record.fps,
            effectiveDenoiseReuse: last.record.effectiveDenoiseReuse,
            ccvAttentionCalls: clips.reduce(0) { $0 + $1.record.ccvAttentionCalls },
            ccvAttentionDirectCalls: clips.reduce(0) { $0 + $1.record.ccvAttentionDirectCalls },
            deviceLine: last.record.deviceLine, appVersion: Self.appVersion, draft: draft,
            sampler: clips[0].record.sampler, audioSigmaShift: clips[0].record.audioSigmaShift,
            soundLateness: 0, latentSoundLateness: last.record.soundLateness ?? 0,
            joinedClips: clips.map { ProjectFiles.stored($0.url.path, in: projectURL) },
            chainPlannedClips: chain.plan.count)
        switch await writeJoinedVideo(clips: clips.map { ($0.url, $0.record) }, record: record,
                                      baseName: chain.baseName, projectURL: projectURL) {
        case .success(let url):
            deleteTemporaryPreview()
            resultURL = url
            var stored = record
            stored.video = url.lastPathComponent
            lastResult = ResolvedResult(record: stored)
            resultAspectRatio = CGFloat(chain.request.sizeProfile.dimensions.width)
                / CGFloat(chain.request.sizeProfile.dimensions.height)
            loadActualDuration(for: url)
            phase = clips.count < chain.plan.count
                ? String(localized: "途中までの\(clips.count)本をつなぎました")
                : String(localized: "できあがりました")
        case .failure(let error):
            errorMessage = String(localized: "つないだ動画を書き出せませんでした（詳細: \(error.localizedDescription)）")
        }
    }

    /// Joins `clips` into <project>/<baseName>_joined.mp4 with `record` next
    /// to it and the last clip's latents (so the joined video can be
    /// continued). Each clip's sound moves earlier by its own lateness.
    func writeJoinedVideo(clips: [(url: URL, record: ProjectVideoRecord?)], record: ProjectVideoRecord,
                          baseName: String, projectURL: URL) async -> Result<URL, Error> {
        var output = projectURL.appendingPathComponent(baseName + "_joined.mp4")
        var counter = 2
        while FileManager.default.fileExists(atPath: output.path) {
            output = projectURL.appendingPathComponent("\(baseName)_joined-\(counter).mp4")
            counter += 1
        }
        isExportingJoined = true
        defer { isExportingJoined = false }
        do {
            let leads = soundLeads(clips)
            try await JoinedVideoExporter.export(clips.enumerated().map { index, clip in
                JoinedVideoExporter.Clip(url: clip.url,
                                         overlapFrames: joinOverlap(clip.record, index: index),
                                         soundLead: leads[index])
            }, to: output)
            var stored = record
            stored.video = output.lastPathComponent
            try ProjectFiles.encoder.encode(stored)
                .write(to: output.deletingPathExtension().appendingPathExtension("json"), options: .atomic)
            GenerationMetadata.tag(output, record: stored, clips: clips.compactMap(\.record))
            if let last = clips.last {
                try? FileManager.default.copyItem(at: ProjectFiles.latentURL(for: last.url),
                                                  to: ProjectFiles.latentURL(for: output))
            }
            return .success(output)
        } catch {
            try? FileManager.default.removeItem(at: output)
            return .failure(error)
        }
    }

    /// How many frames a clip overlaps the one before when joined: what it
    /// carried over if it continues it, none at a cut.
    func joinOverlap(_ record: ProjectVideoRecord?, index: Int) -> Int {
        guard index > 0 else { return 0 }
        if let record, record.draft.continueFrom == nil { return 0 }
        return record?.draft.continueFrames ?? continuationFrames
    }

    /// The joined length of a chain's clips, in frames.
    func joinedFrameCount(_ records: [ProjectVideoRecord]) -> Int {
        records.enumerated().reduce(0) { $0 + $1.element.actualFrameCount - joinOverlap($1.element, index: $1.offset) }
    }

    /// How much earlier each clip's sound goes when joining: its recorded
    /// lateness, or for records from before it was recorded, the snapping
    /// added up along the chain.
    func soundLeads(_ clips: [(url: URL, record: ProjectVideoRecord?)]) -> [Double] {
        var running = 0.0
        return clips.enumerated().map { index, clip in
            if let lateness = clip.record?.soundLateness {
                running = lateness
                return lateness
            }
            guard index > 0, clip.record?.draft.forcedAudio == nil else {
                running = 0
                return 0
            }
            let previous = clips[index - 1]
            let frames = ProjectFiles.latentFrames(ProjectFiles.latentURL(for: previous.url))
                ?? previous.record?.actualFrameCount ?? 0
            running += JoinedVideoExporter.soundDelay(previousFrames: frames,
                                                      overlapFrames: continuationFrames, fps: 24)
            return running
        }
    }
}
