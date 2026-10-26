import AVFoundation
import AppKit
import CoreImage

// Saying that a video is AI-generated. MiniMax H3's license encourages an
// AI-generation identifier on generated files (§III.3(b)) and requires a
// clear, prominent disclosure when one is published (Acceptable Use Policy,
// item 12); MiniMax's own answers say a platform's AI label plus a credit in
// the description is enough and nothing has to be burned in. So every MP4
// the app writes carries it in its metadata (the engine's own files too,
// h3_av_writer.m), and 動画を書き出す… can also put a small label in the
// picture's lower right, on by default. Only on export: a label inside the
// project's videos would end up in the next generation when one is used as
// a reference.
enum AIDisclosure {
    static let label = "AI-generated with MiniMax H3"
    static let labelSettingKey = "h3c-app.exportAILabel"

    /// 動画を書き出す… adds the label unless the user turned it off.
    static var addsLabel: Bool {
        get { UserDefaults.standard.object(forKey: labelSettingKey) as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: labelSettingKey) }
    }

    /// The description and the software, for a file's metadata: in the
    /// common keys (MP4 user data, which AVFoundation reads) and the iTunes
    /// ones (description, comment, encoder - what ffmpeg, exiftool and most
    /// players show).
    static func metadata() -> [AVMetadataItem] {
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "dev"
        let software = "H3cApp \(version) (MiniMax H3)"
        func item(_ space: AVMetadataKeySpace, _ key: AVMetadataKey, _ value: String) -> AVMetadataItem {
            let item = AVMutableMetadataItem()
            item.keySpace = space
            item.key = key.rawValue as NSString
            item.value = value as NSString
            return item
        }
        return [
            item(.common, .commonKeyDescription, label),
            item(.common, .commonKeySoftware, software),
            item(.iTunes, .iTunesMetadataKeyDescription, label),
            item(.iTunes, .iTunesMetadataKeyUserComment, label),
            item(.iTunes, .iTunesMetadataKeyEncodingTool, software),
        ]
    }

    static let h264SettingKey = "h3c-app.exportH264"

    /// 動画を書き出す… makes H.264 when the user turned that on.
    static var exportsH264: Bool {
        get { UserDefaults.standard.bool(forKey: h264SettingKey) }
        set { UserDefaults.standard.set(newValue, forKey: h264SettingKey) }
    }

    enum ExportError: LocalizedError {
        case cannotExport
        case cannotWrite(String)
        var errorDescription: String? {
            switch self {
            case .cannotExport: return String(localized: "書き出しの準備ができませんでした")
            case .cannotWrite(let detail): return String(localized: "書き出しに失敗しました（\(detail)）")
            }
        }
    }

    /// Writes `source` to `destination` with the metadata, with the label in
    /// the lower right when `withLabel`, and as H.264 when `h264` (for
    /// players without HEVC). Re-encoded only for those: otherwise the
    /// tracks are copied as they are (an H.264 video from before HEVC stays
    /// H.264 either way).
    static func export(_ source: URL, to destination: URL, withLabel: Bool, h264: Bool) async throws {
        let asset = AVURLAsset(url: source)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw ExportError.cannotExport
        }
        let codec = await VideoCodec.of(track)
        let toH264 = h264 && (withLabel || codec != .h264)
        try? FileManager.default.removeItem(at: destination)
        if toH264 {
            try await writeH264(asset, video: track, to: destination,
                                composition: withLabel ? try await labelComposition(for: asset) : nil)
            return
        }
        let preset = withLabel ? VideoCodec.masterExportPreset : AVAssetExportPresetPassthrough
        guard let session = AVAssetExportSession(asset: asset, presetName: preset) else {
            throw ExportError.cannotExport
        }
        session.metadata = metadata()
        if withLabel {
            session.videoComposition = try await labelComposition(for: asset)
        }
        try await session.export(to: destination, as: .mp4)
    }

    /// `asset` re-encoded as H.264 at a constant quality (VideoCodec), through
    /// `composition` when given; the sound is copied as it is.
    private static func writeH264(_ asset: AVAsset, video track: AVAssetTrack, to destination: URL,
                                  composition: AVVideoComposition?) async throws {
        let reader = try AVAssetReader(asset: asset)
        let pixels: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String:
                                        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange]
        let videoOutput: AVAssetReaderOutput
        let size: CGSize
        var transform = CGAffineTransform.identity
        if let composition {
            // In HD's colours (BT.709), as the export session renders it;
            // left unsaid, the frames came out of the composition shifted.
            let composition = composition.mutableCopy() as! AVMutableVideoComposition
            composition.colorPrimaries = AVVideoColorPrimaries_ITU_R_709_2
            composition.colorTransferFunction = AVVideoTransferFunction_ITU_R_709_2
            composition.colorYCbCrMatrix = AVVideoYCbCrMatrix_ITU_R_709_2
            let output = AVAssetReaderVideoCompositionOutput(videoTracks: [track], videoSettings: pixels)
            output.videoComposition = composition
            videoOutput = output
            size = composition.renderSize
        } else {
            videoOutput = AVAssetReaderTrackOutput(track: track, outputSettings: pixels)
            (size, transform) = try await track.load(.naturalSize, .preferredTransform)
        }
        reader.add(videoOutput)
        var audioOutput: AVAssetReaderTrackOutput?
        let audioTrack = try await asset.loadTracks(withMediaType: .audio).first
        if let audioTrack {
            let output = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: nil)
            reader.add(output)
            audioOutput = output
        }

        let writer = try AVAssetWriter(outputURL: destination, fileType: .mp4)
        writer.metadata = metadata()
        var settings = VideoCodec.h264.writerSettings(width: Int(size.width.rounded()), height: Int(size.height.rounded()))
        settings[AVVideoColorPropertiesKey] = [
            AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
            AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
            AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
        ]
        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        videoInput.expectsMediaDataInRealTime = false
        videoInput.transform = transform
        writer.add(videoInput)
        var audioInput: AVAssetWriterInput?
        if let audioTrack {
            let hint = try await audioTrack.load(.formatDescriptions).first
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: nil, sourceFormatHint: hint)
            input.expectsMediaDataInRealTime = false
            writer.add(input)
            audioInput = input
        }
        guard reader.startReading(), writer.startWriting() else {
            throw ExportError.cannotWrite(writer.error?.localizedDescription ?? reader.error?.localizedDescription ?? "")
        }
        writer.startSession(atSourceTime: .zero)

        let failure = VideoUpscale.Failure()
        func copy(_ output: AVAssetReaderOutput, into input: AVAssetWriterInput, label: String) async {
            await VideoUpscale.pump(input, on: DispatchQueue(label: label)) {
                guard failure.error == nil, let sample = output.copyNextSampleBuffer() else { return false }
                if !input.append(sample) {
                    failure.set(ExportError.cannotWrite(writer.error?.localizedDescription ?? ""))
                    return false
                }
                return true
            }
        }
        await withTaskGroup(of: Void.self) { group in
            group.addTask { await copy(videoOutput, into: videoInput, label: "h3c.export.video") }
            if let audioInput, let audioOutput {
                group.addTask { await copy(audioOutput, into: audioInput, label: "h3c.export.audio") }
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
            throw ExportError.cannotWrite(writer.error?.localizedDescription ?? "")
        }
    }

    /// The video as it is, with the label drawn over its lower right: the
    /// text is drawn once into an image and laid over every frame by Core
    /// Image (a CATextLayer through the export's animation tool came out
    /// blank).
    private static func labelComposition(for asset: AVAsset) async throws -> AVVideoComposition {
        let size = try await AVMutableVideoComposition.videoComposition(withPropertiesOf: asset).renderSize
        let overlay = labelImage(height: size.height).map { image in
            let margin = max(6, (size.height * 0.025).rounded())
            return CIImage(cgImage: image).transformed(by: CGAffineTransform(
                translationX: size.width - margin - CGFloat(image.width), y: margin))
        }
        return try await AVMutableVideoComposition.videoComposition(with: asset) { request in
            let frame = request.sourceImage.clampedToExtent()
            let output = overlay.map { $0.composited(over: frame) } ?? frame
            request.finish(with: output.cropped(to: request.sourceImage.extent), context: nil)
        }
    }

    /// The label as an image for a picture `height` pixels high: about 3% of
    /// it (at least 11 px), white with a soft shadow so it reads on light and
    /// dark pictures alike.
    private static func labelImage(height: CGFloat) -> CGImage? {
        let fontSize = max(11, (height * 0.03).rounded())
        let shadow = NSShadow()
        shadow.shadowColor = NSColor.black.withAlphaComponent(0.7)
        shadow.shadowBlurRadius = max(1.5, fontSize * 0.15)
        shadow.shadowOffset = NSSize(width: 0, height: -1)
        let text = NSAttributedString(string: label, attributes: [
            .font: NSFont.systemFont(ofSize: fontSize, weight: .medium),
            .foregroundColor: NSColor.white.withAlphaComponent(0.9),
            .shadow: shadow,
        ])
        let pad = ceil(shadow.shadowBlurRadius * 2)
        let textSize = text.size()
        let width = Int(ceil(textSize.width + pad * 2)), imageHeight = Int(ceil(textSize.height + pad * 2))
        guard let context = CGContext(data: nil, width: width, height: imageHeight, bitsPerComponent: 8,
                                      bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(cgContext: context, flipped: false)
        text.draw(at: NSPoint(x: pad, y: pad))
        NSGraphicsContext.restoreGraphicsState()
        return context.makeImage()
    }
}
