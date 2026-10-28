import AVFoundation
import AppKit
import Foundation

/// The project video a generation continues (h3_params.continue_latent_path):
/// its last `frames` frames are pinned at the start of the new clip, so the
/// new clip's first frames repeat them and the two join by cross-fading.
struct ContinuationSource: Equatable {
    var videoURL: URL
    var latentURL: URL
    /// The source's size; a continuation must be made at the same one.
    var sizeProfileRaw: String
    var frames: Int = continuationFrames
    /// Set when continuing an enlarged copy (拡大した動画をつくる): the new
    /// clip is made at the original's size from the latents copied with
    /// the copy, then enlarged the same way, to join the enlarged copy.
    var enlargement: Enlargement? = nil

    var name: String { videoURL.lastPathComponent }

    struct Enlargement: Equatable {
        var upscaler: VideoUpscaler
        var factor: UpscaleFactor
        /// What the enlarged copy was made from: the clip itself continues
        /// that, the latents it carries being the original's.
        var originalURL: URL
    }
}

/// Frames a continuation carries over (about 0.9 s) - the ComfyUI H3 Motion
/// Context node's recommended length, and the one the engine was tested
/// with (continue-test/REPORT.md).
let continuationFrames = 22

/// The lengths the engine can carry over (h3.h continue_frames).
let continuationFrameCounts: Set<Int> = [5, 22, 39, 56]

extension ProjectFiles {
    /// The latents saved next to a generated video (h3_params.save_latent_path).
    static func latentURL(for videoURL: URL) -> URL {
        videoURL.deletingPathExtension().appendingPathExtension("h3lt")
    }

    /// The record saved next to a video, if any.
    static func record(for videoURL: URL) -> ProjectVideoRecord? {
        let recordURL = videoURL.deletingPathExtension().appendingPathExtension("json")
        return (try? Data(contentsOf: recordURL)).flatMap { try? decoder.decode(ProjectVideoRecord.self, from: $0) }
    }
}

extension GenerationViewModel {
    /// この動画の続きを作る: the video's prompt and settings go back into the
    /// panel (seed not pinned) and the next generation continues it. With
    /// lip sync, the audio moves on by what the video added beyond the
    /// carried-over frames.
    func continueFromProjectVideo(_ video: ProjectVideo) {
        guard let record = video.record, let project else { return }
        guard let latent = continuationLatent(of: video) else {
            projectMessage = String(localized: "この動画には続きを作るためのデータがありません（このバージョンより前に作った動画です）")
            return
        }
        var draft = record.draft
        draft.seedFixed = false
        // An imported video has no lip sync of its own: keep the form's.
        if record.importedFrom != nil {
            draft.forcedAudio = forcedAudioPath.map { ProjectFiles.stored($0, in: project.url) }
            draft.forcedAudioOffset = forcedAudioPath == nil ? nil : forcedAudioOffset
        }
        let missing = applyDraft(draft, projectURL: project.url)
        if !missing.isEmpty {
            let names = missing.map { URL(fileURLWithPath: $0).lastPathComponent }.joined(separator: ", ")
            projectMessage = String(localized: "登録されていない追加モデル（LoRA）は使えません: \(names)")
        }
        if !keepsReferencesWhenContinuing { creationMethod = .text }
        // An imported video's lip sync starts where the form had it.
        if let audio = forcedAudioPath, record.importedFrom == nil {
            let length = audioFileSeconds(audio)
            if let offset = continuationAudioOffset(after: video), offset < (length ?? .infinity) - 0.1 {
                forcedAudioOffset = offset
            } else {
                // Past the end of the audio: the continuation makes its own
                // sound, as a つなげて生成's later clips do.
                forcedAudioPath = nil
                forcedAudioOffset = 0
                projectMessage = String(localized: "音声の終わりを過ぎるので、続きはリップシンクなしで作ります")
            }
        }
        // An imported video's latent file is just its end, as long as it
        // was made to carry over.
        let frames = record.importedFrom != nil
            ? (ProjectFiles.latentFrames(latent) ?? continuationFrames) : continuationFrames
        continuation = ContinuationSource(videoURL: video.url, latentURL: latent,
                                          sizeProfileRaw: draft.sizeProfile, frames: frames,
                                          enlargement: enlargement(of: video))
    }

    /// How an enlarged copy was enlarged, for continuing it the same way.
    func enlargement(of video: ProjectVideo) -> ContinuationSource.Enlargement? {
        guard let record = video.record, let original = record.upscaledFrom else { return nil }
        return .init(upscaler: record.upscaler.flatMap(VideoUpscaler.init(rawValue:)) ?? .standard,
                     factor: record.upscaleFactor.flatMap(UpscaleFactor.init(rawValue:)) ?? .two,
                     originalURL: video.url.deletingLastPathComponent().appendingPathComponent(original))
    }

    /// The latents a continuation of `video` carries over from: its own, or
    /// for a copy enlarged before they were copied along, its original's
    /// (when that isn't an enlarged copy itself).
    func continuationLatent(of video: ProjectVideo) -> URL? {
        let own = ProjectFiles.latentURL(for: video.url)
        if FileManager.default.fileExists(atPath: own.path) { return own }
        guard let original = enlargement(of: video)?.originalURL,
              ProjectFiles.record(for: original).map({ $0.upscaledFrom == nil }) == true else { return nil }
        let latent = ProjectFiles.latentURL(for: original)
        return FileManager.default.fileExists(atPath: latent.path) ? latent : nil
    }

    /// Where lip sync carries on after `video`: where its last clip's audio
    /// started plus what that clip added beyond the frames the continuation
    /// carries over. A joined video's record has its whole length but
    /// either its last clip's settings (つないで書き出す) or its first
    /// clip's (つなげて生成), so the last clip's own record is used. Nil
    /// when that clip had no lip sync (its audio had already ended).
    func continuationAudioOffset(after video: ProjectVideo) -> Double? {
        guard var record = video.record else { return nil }
        // A joined video can end in another joined video; a few levels.
        for _ in 0 ..< 8 {
            guard let last = record.joinedClips?.last,
                  let clip = projectVideo(atStoredPath: last)?.record else { break }
            record = clip
        }
        guard record.draft.forcedAudio != nil else { return nil }
        return (record.draft.forcedAudioOffset ?? 0)
            + Double(record.actualFrameCount - continuationFrames) / Double(max(record.fps, 1))
    }

    /// Whether the panel offers 続きを生成 for the video in the preview: a
    /// project video with latents, or a temp result (saved as a project
    /// first).
    var canContinueDisplayedVideo: Bool {
        guard !isGenerating, continuation == nil, let resultURL else { return false }
        if continuableDisplayedVideo != nil { return true }
        return savableTemporaryResult != nil
            && FileManager.default.fileExists(atPath: ProjectFiles.latentURL(for: resultURL).path)
    }

    /// 続きを生成 (the panel): continue the video in the preview. One outside
    /// any project goes into a new one first (プロジェクトとして保存), and is
    /// continued once it's there.
    func continueDisplayedVideo() {
        if let video = continuableDisplayedVideo {
            continueFromProjectVideo(video)
        } else if savableTemporaryResult != nil {
            continueAfterSavingAsProject = true
            showingSaveAsProject = true
        }
    }

    /// The video in the preview, when it continues another and so can be
    /// joined with what it continues (つないで書き出す in the panel).
    var joinableDisplayedVideo: ProjectVideo? {
        guard !isGenerating, !isExportingJoined, let resultURL,
              let video = projectVideos.first(where: { $0.url == resultURL }),
              video.record?.joinedClips == nil, video.record?.draft.continueFrom != nil else { return nil }
        return video
    }

    /// A clip or other video by its path in the project's records.
    func projectVideo(atStoredPath path: String) -> ProjectVideo? {
        guard let project else { return nil }
        let url = URL(fileURLWithPath: ProjectFiles.resolved(path, in: project.url))
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return ProjectVideo(url: url, record: ProjectFiles.record(for: url))
    }

    /// The folder a つなげて生成's clips live in (chains/<name>), if all of a
    /// joined video's clips are in one - deleting the video takes it along.
    func chainFolder(of video: ProjectVideo) -> URL? {
        guard let project, let clips = video.record?.joinedClips, !clips.isEmpty else { return nil }
        let folders = Set(clips.map { URL(fileURLWithPath: ProjectFiles.resolved($0, in: project.url))
            .deletingLastPathComponent().standardizedFileURL })
        guard folders.count == 1, let folder = folders.first,
              folder.deletingLastPathComponent().lastPathComponent == "chains" else { return nil }
        // The record can name any path (absolute, "../", through a symbolic
        // link): only a folder directly in this project's chains folder, once
        // links are resolved, goes to the Trash with the video.
        func realPath(_ url: URL) -> (path: String?, missing: Bool) {
            url.path.withCString { raw in
                guard let real = realpath(raw, nil) else { return (nil, errno == ENOENT) }
                defer { free(real) }
                return (String(cString: real), false)
            }
        }
        let real = realPath(folder)
        if let path = real.path {
            guard let projectPath = realPath(project.url).path,
                  let chainsPath = realPath(project.url.appendingPathComponent("chains", isDirectory: true)).path,
                  (chainsPath as NSString).deletingLastPathComponent == projectPath,
                  (path as NSString).deletingLastPathComponent == chainsPath else { return nil }
            return folder
        }
        // Gone (say, taken along with another joined video): kept, so the
        // delete is refused as before, only when nothing, not even a link, is
        // there and its path is literally <project>/chains/<name>.
        guard real.missing, (try? FileManager.default.attributesOfItem(atPath: folder.path)) == nil,
              folder.path == project.url.appendingPathComponent("chains", isDirectory: true)
                .appendingPathComponent(folder.lastPathComponent).standardizedFileURL.path
        else { return nil }
        return folder
    }

    /// The project video in the preview, if it can be continued.
    var continuableDisplayedVideo: ProjectVideo? {
        guard let resultURL, let video = projectVideos.first(where: { $0.url == resultURL }),
              canContinue(video) else { return nil }
        return video
    }

    func canContinue(_ video: ProjectVideo) -> Bool {
        video.record != nil && continuationLatent(of: video) != nil
    }

    /// Why the continuation can't be generated as the form stands, if so.
    /// Whether a continuation keeps the form's reference images (and so
    /// its 画像から mode): only references, not a first or last frame.
    var keepsReferencesWhenContinuing: Bool {
        creationMethod == .image && imageInputMode == .referenceImage && !referenceImages.isEmpty
    }

    var continuationValidationMessage: String? {
        guard let continuation else { return nil }
        if !FileManager.default.fileExists(atPath: continuation.latentURL.path) {
            return String(localized: "続きの元の動画のデータが見つかりません: \(continuation.name)")
        }
        // Reference images are fine - they keep the people looking the same
        // while the carried frames fix where the clip starts - but a first
        // or last frame would fight those frames.
        if creationMethod == .image && imageInputMode == .firstLastFrame {
            return String(localized: "続きを作るときは、最初・最後の画像は使えません（参照画像は使えます）")
        }
        if continuation.enlargement != nil && isChained {
            return String(localized: "拡大した動画の続きは、つないだ長さでは作れません。1本の長さにしてください")
        }
        if sizeProfile.rawValue != continuation.sizeProfileRaw {
            let label = SizeProfile(rawValue: continuation.sizeProfileRaw)?.label ?? continuation.sizeProfileRaw
            return String(localized: "続きを作るには、元の動画と同じ大きさ（\(label)）にしてください")
        }
        return nil
    }

    /// The videos a continued video was made from, oldest first, ending
    /// with it - following each record's continuation back to a video that
    /// continues nothing. Nil unless there are at least two.
    func continuationChain(endingAt video: ProjectVideo) -> [ProjectVideo]? {
        guard let project else { return nil }
        var chain = [video]
        var seen: Set<String> = [video.url.path]
        while let previous = chain.first?.record?.draft.continueFrom {
            let url = URL(fileURLWithPath: ProjectFiles.resolved(previous, in: project.url))
            guard !seen.contains(url.path), FileManager.default.fileExists(atPath: url.path) else { return nil }
            seen.insert(url.path)
            let recordURL = url.deletingPathExtension().appendingPathExtension("json")
            let record = (try? Data(contentsOf: recordURL))
                .flatMap { try? ProjectFiles.decoder.decode(ProjectVideoRecord.self, from: $0) }
            chain.insert(ProjectVideo(url: url, record: record), at: 0)
        }
        return chain.count > 1 ? chain : nil
    }

    /// ここまでをつないで書き出す / つないで書き出す: joins the chain ending at
    /// `video` into one video in the project, cross-fading each
    /// continuation's carried-over frames and sound, and shows it in the
    /// preview.
    func exportJoined(endingAt video: ProjectVideo) {
        Task {
            guard case .success(let url) = await exportJoinedVideo(endingAt: video, reveal: false),
                  let joined = projectVideos.first(where: { $0.url == url }) else { return }
            showProjectVideo(joined)
        }
    }

    func exportJoinedVideo(endingAt video: ProjectVideo, reveal: Bool) async -> Result<URL, Error> {
        guard let project, let chain = continuationChain(endingAt: video), let lastRecord = video.record else {
            let error = JoinError.noChain
            projectMessage = error.localizedDescription
            return .failure(error)
        }
        projectMessage = String(localized: "\(chain.count)本の動画をつないで書き出しています…")
        // The joined video joins the list as a video of its own; its record
        // is the last clip's settings with the whole chain's length.
        var record = lastRecord
        record.completedAt = Date()
        record.generationSeconds = chain.reduce(0) { $0 + ($1.record?.generationSeconds ?? 0) }
        record.actualFrameCount = joinedFrameCount(chain.compactMap(\.record))
        record.draft.continueFrom = chain[0].record?.draft.continueFrom
        record.draft.seconds = Int((Double(record.actualFrameCount) / Double(max(record.fps, 1))).rounded())
        record.soundLateness = 0
        record.latentSoundLateness = lastRecord.soundLateness ?? soundLeads(chain.map { ($0.url, $0.record) }).last ?? 0
        record.joinedClips = chain.map { ProjectFiles.stored($0.url.path, in: project.url) }
        record.chainPlannedClips = nil
        let result = await writeJoinedVideo(clips: chain.map { ($0.url, $0.record) }, record: record,
                                            baseName: video.url.deletingPathExtension().lastPathComponent,
                                            projectURL: project.url)
        switch result {
        case .success(let output):
            projectMessage = String(localized: "つないだ動画を書き出しました: \(output.lastPathComponent)")
            reloadProjectVideos()
            if reveal { NSWorkspace.shared.activateFileViewerSelecting([output]) }
        case .failure(let error):
            projectMessage = String(localized: "つないだ動画を書き出せませんでした（詳細: \(error.localizedDescription)）")
        }
        return result
    }
}

enum JoinError: LocalizedError {
    case noChain
    case missingTrack(String)
    case exportFailed

    var errorDescription: String? {
        switch self {
        case .noChain:
            return String(localized: "この動画は、ほかの動画の続きとして作ったものではありません")
        case .missingTrack(let name):
            return String(localized: "動画か音声のトラックがありません: \(name)")
        case .exportFailed:
            return String(localized: "書き出しに失敗しました")
        }
    }
}

/// Joins continued clips: each later clip starts `overlapFrames` before the
/// earlier one ends, and over those frames the picture and the sound
/// cross-fade from the earlier clip to the later one (two alternating
/// tracks of each, opacity and volume ramps).
enum JoinedVideoExporter {
    struct Clip {
        var url: URL
        var overlapFrames: Int
        /// How much earlier the clip's sound goes: the lateness it was made
        /// with (ProjectVideoRecord.soundLateness).
        var soundLead = 0.0
    }

    /// How late a continuation's sound runs: the engine snaps the carried
    /// sound to the new clip's 40 Hz audio grid (h3_continuation_plan), so
    /// when the earlier clip's frames past the overlap don't fill whole
    /// audio steps it sits up to a third of a step (8.3 ms) off - 141 or
    /// 158 frames, but not 124 or 175.
    static func soundDelay(previousFrames: Int, overlapFrames: Int, fps: Double) -> Double {
        let steps = 40.0 / fps
        let audioT = (Double(previousFrames) * steps).rounded()
        let end = audioT - steps * Double(previousFrames - overlapFrames)
        return (end.rounded() - end) / 40.0
    }

    static func export(_ clips: [Clip], to output: URL) async throws {
        let composition = AVMutableComposition()
        let videoTracks = (0 ..< 2).compactMap { _ in
            composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid)
        }
        let audioTracks = (0 ..< 2).compactMap { _ in
            composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)
        }
        guard videoTracks.count == 2, audioTracks.count == 2 else { throw JoinError.exportFailed }
        let audioParameters = audioTracks.map { AVMutableAudioMixInputParameters(track: $0) }
        audioParameters.forEach { $0.setVolume(1, at: .zero) }

        // Where each clip lies on the joined timeline.
        var ranges: [CMTimeRange] = []
        var renderSize = CGSize.zero
        var frameRate: Float = 24
        for (index, clip) in clips.enumerated() {
            let asset = AVURLAsset(url: clip.url)
            guard let video = try await asset.loadTracks(withMediaType: .video).first,
                  let audio = try await asset.loadTracks(withMediaType: .audio).first else {
                throw JoinError.missingTrack(clip.url.lastPathComponent)
            }
            let videoRange = try await video.load(.timeRange)
            let audioRange = try await audio.load(.timeRange)
            if index == 0 {
                renderSize = try await video.load(.naturalSize)
                frameRate = try await video.load(.nominalFrameRate)
            }
            // Whole frames on a 1/fps grid: the tracks run a few ticks past
            // their last frame (165334/32000 s for 124 frames), and a clip
            // placed that far off the grid would be sampled a frame late.
            let fps = CMTimeScale(frameRate.rounded())
            let frames = CMTimeValue((videoRange.duration.seconds * Double(fps)).rounded())
            let overlap = CMTime(value: CMTimeValue(clip.overlapFrames), timescale: fps)
            let start = ranges.last.map { CMTimeSubtract($0.end, overlap) } ?? .zero
            let duration = CMTime(value: frames, timescale: fps)
            try videoTracks[index % 2].insertTimeRange(CMTimeRange(start: videoRange.start, duration: duration),
                                                       of: video, at: start)
            // The sound runs a few milliseconds past the last frame; the
            // joined file keeps each clip's sound only as long as its picture,
            // and a continuation's starts early by however late it runs.
            // A late sound starts that much further into its track (an
            // early one, that much later on the timeline), so nothing lands
            // before the clip.
            let lead = CMTime(seconds: abs(clip.soundLead), preferredTimescale: 48000)
            let soundSource = clip.soundLead >= 0 ? CMTimeAdd(audioRange.start, lead) : audioRange.start
            let soundAt = clip.soundLead >= 0 ? start : CMTimeAdd(start, lead)
            let available = CMTimeSubtract(audioRange.end, soundSource)
            let soundLength = CMTimeMinimum(CMTimeSubtract(CMTimeAdd(start, duration), soundAt), available)
            try audioTracks[index % 2].insertTimeRange(
                CMTimeRange(start: soundSource, duration: soundLength), of: audio, at: soundAt)
            if index > 0 && clip.overlapFrames > 0 {
                let fade = CMTimeRange(start: start, end: ranges[index - 1].end)
                audioParameters[(index - 1) % 2].setVolumeRamp(fromStartVolume: 1, toEndVolume: 0, timeRange: fade)
                audioParameters[index % 2].setVolumeRamp(fromStartVolume: 0, toEndVolume: 1, timeRange: fade)
            } else if index > 0 {
                // A cut (a clip that starts from its own first frame): its
                // track may have been faded out at an earlier cross-fade.
                audioParameters[index % 2].setVolume(1, at: start)
            }
            ranges.append(CMTimeRange(start: start, duration: duration))
        }

        // One instruction per stretch: a clip alone, or two cross-fading.
        var instructions: [AVMutableVideoCompositionInstruction] = []
        func instruction(_ range: CMTimeRange, _ layers: [AVMutableVideoCompositionLayerInstruction]) {
            guard range.duration > .zero else { return }
            let item = AVMutableVideoCompositionInstruction()
            item.timeRange = range
            item.layerInstructions = layers
            instructions.append(item)
        }
        for index in clips.indices {
            let soloStart = index == 0 ? ranges[0].start : ranges[index - 1].end
            let soloEnd = index + 1 < clips.count ? ranges[index + 1].start : ranges[index].end
            instruction(CMTimeRange(start: soloStart, end: soloEnd),
                        [AVMutableVideoCompositionLayerInstruction(assetTrack: videoTracks[index % 2])])
            if index + 1 < clips.count {
                let fade = CMTimeRange(start: ranges[index + 1].start, end: ranges[index].end)
                let incoming = AVMutableVideoCompositionLayerInstruction(assetTrack: videoTracks[(index + 1) % 2])
                incoming.setOpacityRamp(fromStartOpacity: 0, toEndOpacity: 1, timeRange: fade)
                let outgoing = AVMutableVideoCompositionLayerInstruction(assetTrack: videoTracks[index % 2])
                instruction(fade, [incoming, outgoing])
            }
        }
        let videoComposition = AVMutableVideoComposition()
        videoComposition.instructions = instructions
        videoComposition.renderSize = renderSize
        videoComposition.frameDuration = CMTime(value: 1, timescale: CMTimeScale(frameRate.rounded()))
        let audioMix = AVMutableAudioMix()
        audioMix.inputParameters = audioParameters

        try await VideoRewriter.write(composition, to: output, codec: .hevc, quality: VideoQuality.saved.value,
                                      composition: videoComposition, audioMix: audioMix,
                                      metadata: AIDisclosure.metadata())
    }
}
