import Accelerate
import AVFoundation
import Foundation

// 2倍に拡大した動画をつくる: a finished video, enlarged after the fact into a
// new project video next to it. It needs nothing from the generation - the
// "×2" sizes enlarge the decoded frames the same way - so it works on any
// project video, its sound copied as it is.

/// How the frames are enlarged. One for now; this is where other
/// upscalers go.
enum VideoUpscaler: String, CaseIterable, Identifiable {
    /// vImage's high-quality (Lanczos) resampling, as the engine uses for
    /// the ×2 sizes (h3_resize_rgb24_high_quality).
    case standard

    var id: String { rawValue }

    var label: String {
        switch self {
        case .standard: return String(localized: "標準（高品質な拡大）")
        }
    }
}

enum VideoUpscale {
    enum UpscaleError: LocalizedError {
        case noVideoTrack
        case tooLarge(width: Int, height: Int)
        case cannotRead
        case cannotWrite(String)
        var errorDescription: String? {
            switch self {
            case .noVideoTrack: return String(localized: "映像のトラックがありません")
            case .tooLarge(let width, let height):
                return String(localized: "拡大すると\(width)×\(height)になり、書き出せる大きさ（一辺4096ピクセル）を超えます")
            case .cannotRead: return String(localized: "動画を読み込めませんでした")
            case .cannotWrite(let detail): return String(localized: "拡大した動画を書き出せませんでした（\(detail)）")
            }
        }
    }

    /// The largest side H.264 is written at here.
    static let maximumSide = 4096

    /// The size `factor` makes of the video at `url` (its stored frame size).
    static func enlargedSize(of url: URL, factor: Int = 2) async throws -> (width: Int, height: Int) {
        guard let track = try await AVURLAsset(url: url).loadTracks(withMediaType: .video).first else {
            throw UpscaleError.noVideoTrack
        }
        let size = try await track.load(.naturalSize)
        return (Int(size.width.rounded()) * factor, Int(size.height.rounded()) * factor)
    }

    /// Writes `source` enlarged `factor` times to `destination` (H.264 at
    /// the engine's bitrate rule, the sound and metadata as they are).
    static func upscale(_ source: URL, to destination: URL, factor: Int = 2,
                        upscaler: VideoUpscaler = .standard) async throws {
        let asset = AVURLAsset(url: source)
        guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
            throw UpscaleError.noVideoTrack
        }
        let audioTrack = try await asset.loadTracks(withMediaType: .audio).first
        let (natural, transform) = try await videoTrack.load(.naturalSize, .preferredTransform)
        let width = Int(natural.width.rounded()) * factor
        let height = Int(natural.height.rounded()) * factor
        guard width <= maximumSide, height <= maximumSide else {
            throw UpscaleError.tooLarge(width: width, height: height)
        }

        let reader = try AVAssetReader(asset: asset)
        let videoOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        videoOutput.alwaysCopiesSampleData = false
        reader.add(videoOutput)
        var audioOutput: AVAssetReaderTrackOutput?
        if let audioTrack {
            let output = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: nil)
            reader.add(output)
            audioOutput = output
        }

        try? FileManager.default.removeItem(at: destination)
        let writer = try AVAssetWriter(outputURL: destination, fileType: .mp4)
        writer.metadata = (try? await asset.load(.metadata)) ?? []
        // The engine's settings (h3_av_writer.m): 5 bits a pixel, at least
        // 9.5 Mbps, High profile, a key frame every 30 frames.
        let bitrate = max(width * height * 5, 9_500_000)
        let macroblocks = ((width + 15) / 16) * ((height + 15) / 16)
        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitrate,
                AVVideoProfileLevelKey: macroblocks > 8192 ? AVVideoProfileLevelH264HighAutoLevel
                                                           : AVVideoProfileLevelH264High41,
                AVVideoMaxKeyFrameIntervalKey: 30,
                AVVideoAllowFrameReorderingKey: true,
            ],
        ])
        videoInput.expectsMediaDataInRealTime = false
        // A rotation keeps its meaning at the larger size.
        videoInput.transform = CGAffineTransform(a: transform.a, b: transform.b, c: transform.c, d: transform.d,
                                                 tx: transform.tx * CGFloat(factor), ty: transform.ty * CGFloat(factor))
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoInput, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ])
        writer.add(videoInput)
        var audioInput: AVAssetWriterInput?
        if let audioTrack {
            let hint = try await audioTrack.load(.formatDescriptions).first
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: nil, sourceFormatHint: hint)
            input.expectsMediaDataInRealTime = false
            writer.add(input)
            audioInput = input
        }

        guard reader.startReading() else { throw UpscaleError.cannotRead }
        guard writer.startWriting() else {
            throw UpscaleError.cannotWrite(writer.error?.localizedDescription ?? "")
        }
        writer.startSession(atSourceTime: .zero)

        let failure = Failure()
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                await pump(videoInput, on: DispatchQueue(label: "h3c.upscale.video")) {
                    guard let sample = videoOutput.copyNextSampleBuffer() else { return false }
                    guard let frame = CMSampleBufferGetImageBuffer(sample),
                          let pool = adaptor.pixelBufferPool else { return true }
                    var enlarged: CVPixelBuffer?
                    CVPixelBufferPoolCreatePixelBuffer(nil, pool, &enlarged)
                    guard let enlarged, scale(frame, into: enlarged, upscaler: upscaler) else {
                        failure.set(UpscaleError.cannotWrite(String(localized: "フレームを拡大できませんでした")))
                        return false
                    }
                    if !adaptor.append(enlarged, withPresentationTime: CMSampleBufferGetPresentationTimeStamp(sample)) {
                        failure.set(UpscaleError.cannotWrite(writer.error?.localizedDescription ?? ""))
                        return false
                    }
                    return true
                }
            }
            if let audioInput, let audioOutput {
                group.addTask {
                    await pump(audioInput, on: DispatchQueue(label: "h3c.upscale.audio")) {
                        guard let sample = audioOutput.copyNextSampleBuffer() else { return false }
                        if !audioInput.append(sample) {
                            failure.set(UpscaleError.cannotWrite(writer.error?.localizedDescription ?? ""))
                            return false
                        }
                        return true
                    }
                }
            }
        }
        if let error = failure.error ?? (reader.status == .failed ? reader.error : nil) {
            reader.cancelReading()
            writer.cancelWriting()
            try? FileManager.default.removeItem(at: destination)
            throw error
        }
        await writer.finishWriting()
        if writer.status != .completed {
            try? FileManager.default.removeItem(at: destination)
            throw UpscaleError.cannotWrite(writer.error?.localizedDescription ?? "")
        }
    }

    /// Feeds `input` from `next` (false: no more) whenever it wants data.
    private static func pump(_ input: AVAssetWriterInput, on queue: DispatchQueue,
                             _ next: @escaping () -> Bool) async {
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            var finished = false
            input.requestMediaDataWhenReady(on: queue) {
                while input.isReadyForMoreMediaData && !finished {
                    if !next() {
                        finished = true
                        input.markAsFinished()
                        done.resume()
                    }
                }
            }
        }
    }

    private static func scale(_ source: CVPixelBuffer, into destination: CVPixelBuffer,
                              upscaler: VideoUpscaler) -> Bool {
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(destination, [])
        }
        guard let input = CVPixelBufferGetBaseAddress(source),
              let output = CVPixelBufferGetBaseAddress(destination) else { return false }
        var from = vImage_Buffer(data: input,
                                 height: vImagePixelCount(CVPixelBufferGetHeight(source)),
                                 width: vImagePixelCount(CVPixelBufferGetWidth(source)),
                                 rowBytes: CVPixelBufferGetBytesPerRow(source))
        var to = vImage_Buffer(data: output,
                               height: vImagePixelCount(CVPixelBufferGetHeight(destination)),
                               width: vImagePixelCount(CVPixelBufferGetWidth(destination)),
                               rowBytes: CVPixelBufferGetBytesPerRow(destination))
        switch upscaler {
        case .standard:
            // Four 8-bit channels in any order: BGRA scales as ARGB does.
            return vImageScale_ARGB8888(&from, &to, nil,
                                        vImage_Flags(kvImageHighQualityResampling | kvImageEdgeExtend)) == kvImageNoError
        }
    }

    /// The first error from either track's queue.
    private final class Failure: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var error: Error?
        func set(_ error: Error) {
            lock.lock()
            if self.error == nil { self.error = error }
            lock.unlock()
        }
    }
}

extension GenerationViewModel {
    /// Whether `video` can be enlarged 2x (within the 4096-pixel limit):
    /// its size is an enlarged copy's own, or else the size it was made at.
    func canUpscale(_ video: ProjectVideo) -> Bool {
        guard !isUpscalingVideo, let record = video.record else { return false }
        var size = record.upscaledSize
        if size == nil, let dimensions = SizeProfile(rawValue: record.draft.sizeProfile)?.dimensions {
            size = [Int(dimensions.width), Int(dimensions.height)]
        }
        guard let size, size.count == 2 else { return true }
        return size[0] * 2 <= VideoUpscale.maximumSide && size[1] * 2 <= VideoUpscale.maximumSide
    }

    /// The name the enlarged copy of `video` gets: "<name>_x2.mp4", numbered
    /// if that is taken.
    func upscaledVideoName(for video: ProjectVideo) -> String? {
        guard let project else { return nil }
        let base = video.url.deletingPathExtension().lastPathComponent + "_x2"
        var name = base
        var counter = 2
        while FileManager.default.fileExists(atPath: project.url.appendingPathComponent(name + ".mp4").path) {
            name = "\(base)-\(counter)"
            counter += 1
        }
        return name + ".mp4"
    }

    /// 2倍に拡大した動画をつくる: `video` enlarged 2x into a new project
    /// video with its record (marked as enlarged; it can't be continued, as
    /// its latents are the original's size). Returns whether it worked;
    /// projectMessage says what happened either way.
    @discardableResult
    func upscaleProjectVideo(_ video: ProjectVideo, as name: String? = nil,
                             upscaler: VideoUpscaler = .standard) async -> Bool {
        guard let project, !isUpscalingVideo, let name = name ?? upscaledVideoName(for: video) else {
            return false
        }
        upscalingVideoName = video.url.lastPathComponent
        isUpscalingVideo = true
        defer { isUpscalingVideo = false }
        let destination = project.url.appendingPathComponent(name)
        let started = Date()
        do {
            try await VideoUpscale.upscale(video.url, to: destination, upscaler: upscaler)
            if var record = video.record {
                record.video = name
                record.completedAt = Date()
                record.upscaledFrom = video.url.lastPathComponent
                record.upscaler = upscaler.rawValue
                let enlarged = try await VideoUpscale.enlargedSize(of: video.url, factor: 2)
                record.upscaledSize = [enlarged.width, enlarged.height]
                // Not continuable or joinable as itself: its latents (not
                // copied) are the original's size.
                record.joinedClips = nil
                record.chainPlannedClips = nil
                record.draft.continueFrom = nil
                record.draft.continueFrames = nil
                try ProjectFiles.encoder.encode(record)
                    .write(to: destination.deletingPathExtension().appendingPathExtension("json"), options: .atomic)
            }
        } catch {
            try? FileManager.default.removeItem(at: destination)
            NSLog("H3cApp: upscaling %@ failed: %@", video.url.path, String(describing: error))
            projectMessage = String(localized: "動画を拡大できませんでした（詳細: \(error.localizedDescription)）")
            return false
        }
        reloadProjectVideos()
        let seconds = Date().timeIntervalSince(started)
        projectMessage = String(localized: "\(video.url.lastPathComponent)を2倍に拡大して、\(name)をつくりました（\(formatElapsed(seconds))）")
        return true
    }
}
