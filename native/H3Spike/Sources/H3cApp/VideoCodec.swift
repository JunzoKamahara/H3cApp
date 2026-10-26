import AVFoundation
import Foundation

/// How the app compresses video. Everything kept in a project (generated,
/// joined, enlarged and imported videos) is HEVC at a constant quality
/// (VideoQuality.saved): H.264 at an average bitrate spent a fifth of it on
/// dark scenes, which then broke into 8-pixel blocks. 動画を書き出す… can
/// turn a copy into H.264 for players without HEVC; quality lost in a
/// master could not be won back there, the other way round.
enum VideoCodec {
    case hevc
    case h264

    /// AVAssetWriter's settings at `quality` (0-1), as the engine writes its
    /// videos (h3_av_writer.m), with a key frame every 30 frames. H.264 at a
    /// constant quality too: its export preset gave a dark 1920x1088 clip
    /// 1.7 Mbps (47.1 dB), quality 0.85 3.4 Mbps (47.8 dB).
    func writerSettings(width: Int, height: Int, quality: Double) -> [String: Any] {
        var compression: [String: Any] = [
            AVVideoQualityKey: quality,
            AVVideoMaxKeyFrameIntervalKey: 30,
            AVVideoAllowFrameReorderingKey: true,
        ]
        if self == .h264 { compression[AVVideoProfileLevelKey] = AVVideoProfileLevelH264HighAutoLevel }
        return [
            AVVideoCodecKey: self == .hevc ? AVVideoCodecType.hevc : AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            // HD's colours (BT.709), said in the file, as the engine does.
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
            ],
            AVVideoCompressionPropertiesKey: compression,
        ]
    }

    /// The codec of `track`'s video, if it is one of these.
    static func of(_ track: AVAssetTrack) async -> VideoCodec? {
        guard let format = try? await track.load(.formatDescriptions).first else { return nil }
        switch CMFormatDescriptionGetMediaSubType(format) {
        case kCMVideoCodecType_HEVC: return .hevc
        case kCMVideoCodecType_H264: return .h264
        default: return nil
        }
    }
}

/// The constant quality videos are compressed at: 保存する動画の画質
/// (詳細設定) for what the project keeps, 書き出しの画質 for an export that
/// is re-encoded. Measured on HEVC (Mbps for a 768x768 skin close-up / a
/// dark 1920x1088 clip / a bright 1600x1088 one): 0.85 2.2/2.4/5.7 - the
/// close-up's 16-pixel blocks showed; 0.90 6.4/6.5/14.9; 0.95 17.8 on the
/// close-up, its blocks nearly gone.
enum VideoQuality: String, CaseIterable, Identifiable {
    case standard
    case high
    case highest

    var id: String { rawValue }

    var value: Double {
        switch self {
        case .standard: return 0.85
        case .high: return 0.90
        case .highest: return 0.95
        }
    }

    var label: String {
        switch self {
        case .standard: return String(localized: "標準")
        case .high: return String(localized: "高画質")
        case .highest: return String(localized: "最高画質")
        }
    }

    private static func stored(_ key: String) -> VideoQuality {
        UserDefaults.standard.string(forKey: key).flatMap(VideoQuality.init(rawValue:)) ?? .high
    }

    /// 保存する動画の画質, kept across launches.
    static var saved: VideoQuality {
        get { stored("h3c-app.videoQuality") }
        set { UserDefaults.standard.set(newValue.rawValue, forKey: "h3c-app.videoQuality") }
    }

    /// 書き出しの画質, kept across launches.
    static var export: VideoQuality {
        get { stored("h3c-app.exportVideoQuality") }
        set { UserDefaults.standard.set(newValue.rawValue, forKey: "h3c-app.exportVideoQuality") }
    }
}

/// Re-encodes a video (an export with the label or in H.264, a join, an
/// import) at a constant quality - AVAssetExportSession's presets pick their
/// own bitrate - through an AVAssetReader into an AVAssetWriter.
enum VideoRewriter {
    enum RewriteError: LocalizedError {
        case noVideoTrack
        case failed(String)
        var errorDescription: String? {
            switch self {
            case .noVideoTrack: return String(localized: "映像のトラックがありません")
            case .failed(let detail): return String(localized: "書き出しに失敗しました（\(detail)）")
            }
        }
    }

    /// `asset` as `codec` at `quality` into `destination`: through
    /// `composition` when given (in BT.709, as the export session renders
    /// it; left unsaid, the colours came out shifted), else its first video
    /// track as it is. The sound is copied when it is AAC and not mixed,
    /// else mixed by `audioMix` and encoded as AAC.
    static func write(_ asset: AVAsset, to destination: URL, codec: VideoCodec, quality: Double,
                      composition: AVVideoComposition? = nil, audioMix: AVAudioMix? = nil,
                      metadata: [AVMetadataItem]) async throws {
        let videoTracks = try await asset.loadTracks(withMediaType: .video)
        guard let firstVideo = videoTracks.first else { throw RewriteError.noVideoTrack }
        let reader = try AVAssetReader(asset: asset)
        let pixels: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String:
                                        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange]
        let videoOutput: AVAssetReaderOutput
        let size: CGSize
        var transform = CGAffineTransform.identity
        if let composition {
            let composition = composition.mutableCopy() as! AVMutableVideoComposition
            composition.colorPrimaries = AVVideoColorPrimaries_ITU_R_709_2
            composition.colorTransferFunction = AVVideoTransferFunction_ITU_R_709_2
            composition.colorYCbCrMatrix = AVVideoYCbCrMatrix_ITU_R_709_2
            let output = AVAssetReaderVideoCompositionOutput(videoTracks: videoTracks, videoSettings: pixels)
            output.videoComposition = composition
            videoOutput = output
            size = composition.renderSize
        } else {
            videoOutput = AVAssetReaderTrackOutput(track: firstVideo, outputSettings: pixels)
            (size, transform) = try await firstVideo.load(.naturalSize, .preferredTransform)
        }
        reader.add(videoOutput)

        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        var audioOutput: AVAssetReaderOutput?
        var audioInput: AVAssetWriterInput?
        if let firstAudio = audioTracks.first {
            let format = try await firstAudio.load(.formatDescriptions).first
            let isAAC = format.map { CMFormatDescriptionGetMediaSubType($0) == kAudioFormatMPEG4AAC } ?? false
            if audioMix == nil && audioTracks.count == 1 && isAAC {
                audioOutput = AVAssetReaderTrackOutput(track: firstAudio, outputSettings: nil)
                audioInput = AVAssetWriterInput(mediaType: .audio, outputSettings: nil, sourceFormatHint: format)
            } else {
                let source = format.flatMap { CMAudioFormatDescriptionGetStreamBasicDescription($0)?.pointee }
                let rate = source.map { $0.mSampleRate > 0 ? $0.mSampleRate : 48_000 } ?? 48_000
                let channels = min(max(Int(source?.mChannelsPerFrame ?? 2), 1), 2)
                let output = AVAssetReaderAudioMixOutput(audioTracks: audioTracks, audioSettings: [
                    AVFormatIDKey: kAudioFormatLinearPCM,
                    AVSampleRateKey: rate,
                    AVNumberOfChannelsKey: channels,
                    AVLinearPCMBitDepthKey: 32,
                    AVLinearPCMIsFloatKey: true,
                    AVLinearPCMIsNonInterleaved: false,
                    AVLinearPCMIsBigEndianKey: false,
                ])
                output.audioMix = audioMix
                audioOutput = output
                // As the engine encodes its sound: up to 192 kbps.
                audioInput = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                    AVFormatIDKey: kAudioFormatMPEG4AAC,
                    AVSampleRateKey: rate,
                    AVNumberOfChannelsKey: channels,
                    AVEncoderBitRateKey: channels > 1 ? 192_000 : 128_000,
                ])
            }
            reader.add(audioOutput!)
        }

        let writer = try AVAssetWriter(outputURL: destination, fileType: .mp4)
        writer.metadata = metadata
        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: codec.writerSettings(
            width: Int(size.width.rounded()), height: Int(size.height.rounded()), quality: quality))
        videoInput.expectsMediaDataInRealTime = false
        videoInput.transform = transform
        writer.add(videoInput)
        if let audioInput {
            audioInput.expectsMediaDataInRealTime = false
            writer.add(audioInput)
        }
        guard reader.startReading(), writer.startWriting() else {
            throw RewriteError.failed(writer.error?.localizedDescription ?? reader.error?.localizedDescription ?? "")
        }
        writer.startSession(atSourceTime: .zero)

        let failure = VideoUpscale.Failure()
        func copy(_ output: AVAssetReaderOutput, into input: AVAssetWriterInput, label: String) async {
            await VideoUpscale.pump(input, on: DispatchQueue(label: label)) {
                guard failure.error == nil, let sample = output.copyNextSampleBuffer() else { return false }
                if !input.append(sample) {
                    failure.set(RewriteError.failed(writer.error?.localizedDescription ?? ""))
                    return false
                }
                return true
            }
        }
        await withTaskGroup(of: Void.self) { group in
            group.addTask { await copy(videoOutput, into: videoInput, label: "h3c.rewrite.video") }
            if let audioInput, let audioOutput {
                group.addTask { await copy(audioOutput, into: audioInput, label: "h3c.rewrite.audio") }
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
            throw RewriteError.failed(writer.error?.localizedDescription ?? "")
        }
    }
}
