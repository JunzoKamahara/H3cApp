import AVFoundation
import Foundation
import UniformTypeIdentifiers

// 動画ファイルの続きを作る: continuing a video that wasn't made here. It
// comes into the project as a video of its own - converted to the form's
// size and 24 fps, so continuing and joining it work as for a generated
// one - with a latent file made from its end by the engine
// (h3_continuation_latent_from_video: the last 22 frames through the video
// VAE, the sound under them through the audio VAE). Then the form continues
// it like any project video.
extension GenerationViewModel {
    enum ImportError: LocalizedError {
        case noVideoTrack
        case tooShort
        case cannotExport
        var errorDescription: String? {
            switch self {
            case .noVideoTrack: return String(localized: "映像のトラックがありません")
            case .tooShort: return String(localized: "動画が短すぎます（1秒以上必要です）")
            case .cannotExport: return String(localized: "動画を変換できませんでした")
            }
        }
    }

    /// What a continuation of an imported video can carry over, in frames.
    static let importCarryFrames = [22, 39, 56]

    /// A video about to be imported, for the sheet that picks the size to
    /// make its continuation at (ImportVideoSheet).
    struct PendingVideoImport: Identifiable {
        let id = UUID()
        let url: URL
        let width: Int
        let height: Int
        let fps: Double
        let seconds: Double
        /// The size closest to the video's own (SizeProfile.matching).
        let suggested: SizeProfile
    }

    /// The video's shown size (rotation applied), frame rate and length.
    static func videoInfo(_ url: URL) async throws -> (width: Int, height: Int, fps: Double, seconds: Double) {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw ImportError.noVideoTrack
        }
        let (natural, preferred, fps) = try await track.load(.naturalSize, .preferredTransform, .nominalFrameRate)
        let shown = CGRect(origin: .zero, size: natural).applying(preferred)
        let seconds = try await asset.load(.duration).seconds
        return (Int(abs(shown.width).rounded()), Int(abs(shown.height).rounded()), Double(fps), seconds)
    }

    /// 動画ファイルの続きを作る…: picks a video and asks which size to
    /// bring it in at (the video's own first).
    func pickVideoToContinue() {
        guard let path = chooseFile(allowedContentTypes: [.movie]) else { return }
        let url = URL(fileURLWithPath: path)
        Task {
            do {
                let info = try await Self.videoInfo(url)
                pendingVideoImport = PendingVideoImport(
                    url: url, width: info.width, height: info.height, fps: info.fps, seconds: info.seconds,
                    suggested: .matching(videoWidth: info.width, height: info.height))
            } catch {
                projectMessage = String(localized: "動画を取り込めませんでした（詳細: \(error.localizedDescription)）")
            }
        }
    }

    /// The name the imported copy of `source` gets in the project (for the
    /// API, which answers before the import is done).
    func importedVideoName(for source: URL) -> String? {
        guard let project else { return nil }
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let base = "\(formatter.string(from: Date()))_imported"
        var name = base
        var counter = 2
        while FileManager.default.fileExists(atPath: project.url.appendingPathComponent(name + ".mp4").path) {
            name = "\(base)-\(counter)"
            counter += 1
        }
        return name + ".mp4"
    }

    /// Imports `source` into the project at the form's size and sets the
    /// form to continue it. Returns whether it worked (projectMessage says
    /// what happened either way).
    /// With `size`, the form is set to it first (a continuation is made at
    /// its source's size); without, to the size closest to the video's own.
    /// `carryFrames` (22, 39 or 56) is how much of its end the continuation
    /// carries over: longer costs time. `cropPosition` places the part kept
    /// when the shapes differ, 0-1 across what is cut off (0.5 centred).
    @discardableResult
    func importVideoToContinue(_ source: URL, as name: String? = nil, size: SizeProfile? = nil,
                               carryFrames: Int = continuationFrames,
                               cropPosition: CGPoint = CGPoint(x: 0.5, y: 0.5)) async -> Bool {
        guard let project, let engine, !isGenerating, !isImportingVideo,
              let name = name ?? importedVideoName(for: source) else {
            NSLog("H3cApp: cannot import %@ now (project %d, engine %d, generating %d, importing %d)",
                  source.path, project != nil, self.engine != nil, isGenerating, isImportingVideo)
            return false
        }
        importingVideoName = source.lastPathComponent
        isImportingVideo = true
        defer { isImportingVideo = false }
        if let size {
            sizeProfile = size
        } else if let info = try? await Self.videoInfo(source) {
            sizeProfile = .matching(videoWidth: info.width, height: info.height)
        }
        let dimensions = sizeProfile.dimensions
        let destination = project.url.appendingPathComponent(name)
        let latent = ProjectFiles.latentURL(for: destination)
        do {
            let seconds = try await Self.convertForContinuing(
                source, to: destination,
                size: CGSize(width: CGFloat(dimensions.width), height: CGFloat(dimensions.height)),
                position: cropPosition)
            // The size the DiT works at: 0 means the finished size (no x2).
            let renderWidth = dimensions.renderWidth > 0 ? dimensions.renderWidth : dimensions.width
            let renderHeight = dimensions.renderHeight > 0 ? dimensions.renderHeight : dimensions.height
            try await engine.makeContinuationLatent(
                videoPath: destination.path, renderWidth: Int(renderWidth),
                renderHeight: Int(renderHeight), carryFrames: carryFrames, latentPath: latent.path)
            var draft = currentDraft()
            draft.firstFrame = draft.firstFrame.map { ProjectFiles.stored($0, in: project.url) }
            draft.lastFrame = draft.lastFrame.map { ProjectFiles.stored($0, in: project.url) }
            draft.references = draft.references.map { .init(kind: $0.kind, path: ProjectFiles.stored($0.path, in: project.url)) }
            draft.seconds = Int(seconds.rounded())
            draft.forcedAudio = nil
            draft.forcedAudioOffset = nil
            draft.continueFrom = nil
            draft.continueFrames = nil
            draft.chained = nil
            draft.chainClipFrames = nil
            draft.clipPrompts = nil
            draft.clipReferences = nil
            draft.chainCustom = nil
            draft.customClips = nil
            let record = ProjectVideoRecord(
                video: name, completedAt: Date(), seed: "0", seedWasRandom: false,
                generationSeconds: 0, actualFrameCount: Int((seconds * 24).rounded(.down)), fps: 24,
                effectiveDenoiseReuse: effectiveDenoiseReuse, ccvAttentionCalls: 0,
                ccvAttentionDirectCalls: 0, deviceLine: deviceLine, appVersion: Self.appVersion,
                draft: draft, soundLateness: 0, latentSoundLateness: 0,
                importedFrom: source.lastPathComponent)
            try ProjectFiles.encoder.encode(record)
                .write(to: destination.deletingPathExtension().appendingPathExtension("json"), options: .atomic)
        } catch {
            for url in [destination, latent, destination.deletingPathExtension().appendingPathExtension("json")] {
                try? FileManager.default.removeItem(at: url)
            }
            NSLog("H3cApp: importing %@ failed: %@", source.path, String(describing: error))
            projectMessage = String(localized: "動画を取り込めませんでした（詳細: \(error.localizedDescription)）")
            return false
        }
        reloadProjectVideos()
        guard let video = projectVideos.first(where: { $0.url.lastPathComponent == name }) else { return false }
        showProjectVideo(video)
        continueFromProjectVideo(video)
        projectMessage = String(localized: "\(source.lastPathComponent)を取り込みました。続きの内容を書いて生成してください")
        return true
    }

    /// `source` at `size` (scaled to fill it; `position` places it, 0-1
    /// across what is cut off, 0.5 centred) and 24 fps, with its sound.
    /// Returns its length in seconds.
    private static func convertForContinuing(_ source: URL, to destination: URL, size: CGSize,
                                             position: CGPoint) async throws -> Double {
        let asset = AVURLAsset(url: source)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw ImportError.noVideoTrack
        }
        let duration = try await asset.load(.duration)
        guard duration.seconds >= 1 else { throw ImportError.tooShort }
        let natural = try await track.load(.naturalSize)
        let preferred = try await track.load(.preferredTransform)
        let shown = CGRect(origin: .zero, size: natural).applying(preferred)
        let scale = max(size.width / shown.width, size.height / shown.height)
        let transform = preferred
            .concatenating(CGAffineTransform(translationX: -shown.minX, y: -shown.minY))
            .concatenating(CGAffineTransform(scaleX: scale, y: scale))
            .concatenating(CGAffineTransform(
                translationX: (size.width - shown.width * scale) * min(max(position.x, 0), 1),
                y: (size.height - shown.height * scale) * min(max(position.y, 0), 1)))
        let layer = AVMutableVideoCompositionLayerInstruction(assetTrack: track)
        layer.setTransform(transform, at: .zero)
        let instruction = AVMutableVideoCompositionInstruction()
        instruction.timeRange = CMTimeRange(start: .zero, duration: duration)
        instruction.layerInstructions = [layer]
        let composition = AVMutableVideoComposition()
        composition.renderSize = size
        composition.frameDuration = CMTime(value: 1, timescale: 24)
        composition.instructions = [instruction]
        guard let session = AVAssetExportSession(asset: asset, presetName: VideoCodec.masterExportPreset) else {
            throw ImportError.cannotExport
        }
        session.videoComposition = composition
        try? FileManager.default.removeItem(at: destination)
        try await session.export(to: destination, as: .mp4)
        return duration.seconds
    }
}
