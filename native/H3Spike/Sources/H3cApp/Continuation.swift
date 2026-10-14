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

    var name: String { videoURL.lastPathComponent }
}

/// Frames a continuation carries over (about 0.9 s) - the ComfyUI H3 Motion
/// Context node's recommended length, and the one the engine was tested
/// with (continue-test/REPORT.md).
let continuationFrames = 22

extension ProjectFiles {
    /// The latents saved next to a generated video (h3_params.save_latent_path).
    static func latentURL(for videoURL: URL) -> URL {
        videoURL.deletingPathExtension().appendingPathExtension("h3lt")
    }
}

extension GenerationViewModel {
    /// この動画の続きを作る: the video's prompt and settings go back into the
    /// panel (seed not pinned) and the next generation continues it. With
    /// lip sync, the audio moves on by what the video added beyond the
    /// carried-over frames.
    func continueFromProjectVideo(_ video: ProjectVideo) {
        guard let record = video.record, let project else { return }
        let latent = ProjectFiles.latentURL(for: video.url)
        guard FileManager.default.fileExists(atPath: latent.path) else {
            projectMessage = String(localized: "この動画には続きを作るためのデータがありません（このバージョンより前に作った動画です）")
            return
        }
        var draft = record.draft
        draft.seedFixed = false
        let missing = applyDraft(draft, projectURL: project.url)
        if !missing.isEmpty {
            let names = missing.map { URL(fileURLWithPath: $0).lastPathComponent }.joined(separator: ", ")
            projectMessage = String(localized: "登録されていない追加モデル（LoRA）は使えません: \(names)")
        }
        creationMethod = .text
        if forcedAudioPath != nil {
            let added = Double(record.actualFrameCount - continuationFrames) / Double(record.fps)
            forcedAudioOffset = (draft.forcedAudioOffset ?? 0) + added
        }
        continuation = ContinuationSource(videoURL: video.url, latentURL: latent,
                                          sizeProfileRaw: draft.sizeProfile)
    }

    /// The project video in the preview, if it can be continued.
    var continuableDisplayedVideo: ProjectVideo? {
        guard let resultURL, let video = projectVideos.first(where: { $0.url == resultURL }),
              canContinue(video) else { return nil }
        return video
    }

    func canContinue(_ video: ProjectVideo) -> Bool {
        video.record != nil && FileManager.default.fileExists(atPath: ProjectFiles.latentURL(for: video.url).path)
    }

    /// Why the continuation can't be generated as the form stands, if so.
    var continuationValidationMessage: String? {
        guard let continuation else { return nil }
        if !FileManager.default.fileExists(atPath: continuation.latentURL.path) {
            return String(localized: "続きの元の動画のデータが見つかりません: \(continuation.name)")
        }
        if creationMethod == .image {
            return String(localized: "続きを作るときは「文章から」でつくります（最初の画像や参照は使えません）")
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

    /// ここまでをつないで書き出す: joins the chain ending at `video` into
    /// one file in the project's "joined" folder, cross-fading each
    /// continuation's carried-over frames and sound, then shows it in Finder.
    func exportJoined(endingAt video: ProjectVideo) {
        Task { _ = await exportJoinedVideo(endingAt: video, reveal: true) }
    }

    func exportJoinedVideo(endingAt video: ProjectVideo, reveal: Bool) async -> Result<URL, Error> {
        guard let project, let chain = continuationChain(endingAt: video) else {
            let error = JoinError.noChain
            projectMessage = error.localizedDescription
            return .failure(error)
        }
        let folder = project.url.appendingPathComponent("joined", isDirectory: true)
        let base = video.url.deletingPathExtension().lastPathComponent + "_joined"
        var output = folder.appendingPathComponent(base + ".mp4")
        var counter = 2
        while FileManager.default.fileExists(atPath: output.path) {
            output = folder.appendingPathComponent("\(base)-\(counter).mp4")
            counter += 1
        }
        isExportingJoined = true
        projectMessage = String(localized: "\(chain.count)本の動画をつないで書き出しています…")
        defer { isExportingJoined = false }
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let clips = chain.enumerated().map { index, video in
                JoinedVideoExporter.Clip(url: video.url,
                                         overlapFrames: index == 0 ? 0 : (video.record?.draft.continueFrames ?? continuationFrames))
            }
            try await JoinedVideoExporter.export(clips, to: output)
            projectMessage = String(localized: "つないだ動画を書き出しました: joined/\(output.lastPathComponent)")
            if reveal { NSWorkspace.shared.activateFileViewerSelecting([output]) }
            return .success(output)
        } catch {
            projectMessage = String(localized: "つないだ動画を書き出せませんでした（詳細: \(error.localizedDescription)）")
            return .failure(error)
        }
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
            // joined file keeps each clip's sound only as long as its picture.
            try audioTracks[index % 2].insertTimeRange(
                CMTimeRange(start: audioRange.start, duration: CMTimeMinimum(duration, audioRange.duration)),
                of: audio, at: start)
            if index > 0 {
                let fade = CMTimeRange(start: start, end: ranges[index - 1].end)
                audioParameters[(index - 1) % 2].setVolumeRamp(fromStartVolume: 1, toEndVolume: 0, timeRange: fade)
                audioParameters[index % 2].setVolumeRamp(fromStartVolume: 0, toEndVolume: 1, timeRange: fade)
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

        guard let session = AVAssetExportSession(asset: composition,
                                                 presetName: AVAssetExportPresetHighestQuality) else {
            throw JoinError.exportFailed
        }
        session.videoComposition = videoComposition
        session.audioMix = audioMix
        try await session.export(to: output, as: .mp4)
    }
}
