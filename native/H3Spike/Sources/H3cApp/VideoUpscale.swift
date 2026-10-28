import Accelerate
import AppKit
import AVFoundation
import CoreVideo
import Foundation
import H3Engine
import Metal
import MetalFX

// 拡大した動画をつくる / 動画ファイルを拡大: a finished video, enlarged
// after the fact (1.5x to 4x). It needs nothing from the generation - the "×2"
// sizes enlarge the decoded frames the same way - so it works on any video,
// its sound copied as it is.

/// How the frames are enlarged (the choice in UpscaleSheet). Each runs in
/// the app, frame by frame; this is where other upscalers go.
enum VideoUpscaler: String, CaseIterable, Identifiable {
    /// vImage's high-quality (Lanczos) resampling, as the engine uses for
    /// the ×2 sizes (h3_resize_rgb24_high_quality). On the CPU.
    case standard
    /// MetalFX's spatial scaler: edge-aware enlargement with sharpening,
    /// made for games, on the GPU. Not AI super-resolution either.
    case metalFX
    /// Real-ESRGAN's SRVGGNetCompact (RealESRGANNetwork), AI
    /// super-resolution on the GPU: realesr-general-x4v3 (at the default
    /// denoise strength 0.5) and realesr-animevideov3, bundled.
    case realESRGANGeneral
    case realESRGANAnime
    /// SwiftVR (Wan2.2-based one-step video restoration), run by the
    /// engine (h3_vsr.h) from weights the user downloaded: AI video
    /// super-resolution with generated detail. Research license.
    case swiftVR

    var id: String { rawValue }

    var label: String {
        switch self {
        case .standard: return String(localized: "標準（高品質な拡大）")
        case .metalFX: return String(localized: "MetalFX（輪郭を保つ拡大）")
        case .realESRGANGeneral: return String(localized: "Real-ESRGAN 汎用（AI超解像）")
        case .realESRGANAnime: return String(localized: "Real-ESRGAN アニメ（AI超解像）")
        case .swiftVR: return String(localized: "SwiftVR（AI動画超解像・研究用）")
        }
    }

    /// Where the SwiftVR model lives (reae.safetensors, prompt_embedding.
    /// safetensors, transformer/) and its .h3vc cache next to it;
    /// H3_VSR_MODEL_DIR points elsewhere while developing.
    static var swiftVRModelDirectory: String {
        ProcessInfo.processInfo.environment["H3_VSR_MODEL_DIR"] ?? (h3AppSupportDirectory + "/Upscalers/SwiftVR")
    }

    /// The cache this Mac can run SwiftVR from, when the model is installed.
    static var swiftVRCachePath: String? {
        H3VideoSuperResolver.usableCachePath(modelDirectory: swiftVRModelDirectory)
    }

    /// The bundled weights of an AI upscaler (Resources/Upscalers).
    var weightsURL: URL? {
        switch self {
        case .standard, .metalFX: return nil
        case .realESRGANGeneral:
            return Bundle.main.url(forResource: "realesr-general-x4v3", withExtension: "safetensors", subdirectory: "Upscalers")
        case .realESRGANAnime:
            return Bundle.main.url(forResource: "realesr-animevideov3", withExtension: "safetensors", subdirectory: "Upscalers")
        case .swiftVR: return nil
        }
    }

    /// What it is and what it suits, for the sheet.
    var detail: String {
        switch self {
        case .standard:
            return String(localized: "大きさの「×2」と同じ拡大です。なめらかで、元の動画の見た目をそのまま大きくします。CPUで処理します。")
        case .metalFX:
            return String(localized: "AppleのMetalFXで、輪郭を見ながら拡大し、少しくっきりさせます。GPUで処理します。AIによる高画質化ではありません。")
        case .realESRGANGeneral:
            return String(localized: "AIで細部を補いながら拡大します（Real-ESRGAN realesr-general-x4v3）。実写や一般の映像向けです。細部を描き足すので、元にない模様が出ることがあります。標準より時間がかかります（GPUで、5秒の動画に数十秒〜数分）。")
        case .realESRGANAnime:
            return String(localized: "AIで線や色の面をくっきりさせながら拡大します（Real-ESRGAN realesr-animevideov3）。アニメやイラスト調の映像向けで、汎用より速く処理します。")
        case .swiftVR:
            return String(localized: "動画全体を一度に見るAIモデル（SwiftVR、Wan2.2ベース）で、質感や細部を描き起こしながら拡大します。24フレームずつ処理し、Real-ESRGANより大幅に時間がかかります（M5で5秒の動画に1分前後）。重みはSwiftVR Research License（研究・非商用）のため同梱せず、ダウンロードした重みをアプリで変換して使います。")
        }
    }

    /// Whether this Mac can run it.
    var isAvailable: Bool {
        switch self {
        case .standard: return true
        case .metalFX:
            guard let device = MTLCreateSystemDefaultDevice() else { return false }
            return MTLFXSpatialScalerDescriptor.supportsDevice(device)
        case .realESRGANGeneral, .realESRGANAnime:
            return weightsURL != nil && MTLCreateSystemDefaultDevice() != nil
        case .swiftVR:
            return Self.swiftVRCachePath != nil && MTLCreateSystemDefaultDevice() != nil
        }
    }

    /// Why an unavailable upscaler can't be chosen, for the sheet.
    var unavailableReason: String? {
        if isAvailable { return nil }
        switch self {
        case .swiftVR:
            return String(localized: "SwiftVRのモデルが入っていません。重みを「\(Self.swiftVRModelDirectory)」に置き、build_vsr_cache で変換すると選べます。")
        default:
            return String(localized: "このMacでは使えません。")
        }
    }
}

/// How much bigger (the sheet's 倍率). Every upscaler takes each of them.
enum UpscaleFactor: Double, CaseIterable, Identifiable {
    case oneAndHalf = 1.5
    case two = 2
    case three = 3
    case four = 4

    var id: Double { rawValue }

    /// "1.5倍", "2倍".
    var label: String {
        rawValue == rawValue.rounded()
            ? String(localized: "\(Int(rawValue))倍")
            : String(localized: "\(String(format: "%.1f", rawValue))倍")
    }

    /// "x1.5", "x2" (file names).
    var suffix: String {
        rawValue == rawValue.rounded() ? "x\(Int(rawValue))" : "x" + String(format: "%.1f", rawValue)
    }

    /// The size `width` x `height` becomes, rounded to even numbers (H.264).
    func size(width: Int, height: Int) -> (width: Int, height: Int) {
        (Int((Double(width) * rawValue / 2).rounded()) * 2, Int((Double(height) * rawValue / 2).rounded()) * 2)
    }

    /// Whether the result stays within VideoUpscale.maximumSide.
    func fits(width: Int, height: Int) -> Bool {
        let result = size(width: width, height: height)
        return result.width <= VideoUpscale.maximumSide && result.height <= VideoUpscale.maximumSide
    }
}

/// One frame in, the same frame enlarged out (both 32BGRA).
private protocol FrameScaler {
    func scale(_ source: CVPixelBuffer, into destination: CVPixelBuffer) throws
}

/// Frames in groups, enlarged frames out of a pool, in order and one output
/// per input over the whole video - possibly behind by a few frames, and
/// flushed by `isLast` (UPSCALE_PLAN §6.6). `chunkFrames` is how many the
/// next call should bring.
private protocol ChunkedFrameScaler {
    var chunkFrames: Int { get }
    func process(_ frames: [CVPixelBuffer], isLast: Bool, pool: CVPixelBufferPool) throws -> [CVPixelBuffer]
    /// Stops work inside a chunk when the user cancels.
    func cancel()
}

/// Any per-frame scaler as a chunk of one: the same operations, so the
/// output is unchanged byte for byte.
private final class PerFrameChunkAdapter: ChunkedFrameScaler {
    let scaler: FrameScaler
    init(_ scaler: FrameScaler) { self.scaler = scaler }
    var chunkFrames: Int { 1 }
    func process(_ frames: [CVPixelBuffer], isLast: Bool, pool: CVPixelBufferPool) throws -> [CVPixelBuffer] {
        var outputs: [CVPixelBuffer] = []
        for frame in frames {
            var enlarged: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &enlarged)
            guard let enlarged else {
                throw VideoUpscale.UpscaleError.cannotWrite(String(localized: "フレームを拡大できませんでした"))
            }
            try scaler.scale(frame, into: enlarged)
            outputs.append(enlarged)
        }
        return outputs
    }
    func cancel() {}
}

/// SwiftVR through the engine: 28 frames, then 24 at a time.
private final class VSRChunkScaler: ChunkedFrameScaler {
    let session: H3VideoSuperResolver.Session
    init(session: H3VideoSuperResolver.Session) { self.session = session }
    var chunkFrames: Int { max(1, session.nextChunkFrames) }
    func process(_ frames: [CVPixelBuffer], isLast: Bool, pool: CVPixelBufferPool) throws -> [CVPixelBuffer] {
        try session.process(frames, isLast: isLast, pool: pool)
    }
    func cancel() { session.cancel() }
}

enum VideoUpscale {
    enum UpscaleError: LocalizedError {
        case noVideoTrack
        case tooLarge(width: Int, height: Int)
        case cannotRead
        case cannotWrite(String)
        case unavailable(String)
        case cancelled
        case overSource
        var errorDescription: String? {
            switch self {
            case .noVideoTrack: return String(localized: "映像のトラックがありません")
            case .tooLarge(let width, let height):
                return String(localized: "拡大すると\(width)×\(height)になり、書き出せる大きさ（一辺4096ピクセル）を超えます")
            case .cannotRead: return String(localized: "動画を読み込めませんでした")
            case .cannotWrite(let detail): return String(localized: "拡大した動画を書き出せませんでした（\(detail)）")
            case .unavailable(let name): return String(localized: "このMacでは\(name)を使えません")
            case .cancelled: return String(localized: "拡大を中止しました")
            case .overSource: return String(localized: "拡大した動画を、元の動画と同じファイルには保存できません")
            }
        }
    }

    /// The largest side H.264 is written at here.
    static let maximumSide = 4096

    /// The video's stored frame size (before any rotation).
    static func frameSize(of url: URL) async throws -> (width: Int, height: Int) {
        guard let track = try await AVURLAsset(url: url).loadTracks(withMediaType: .video).first else {
            throw UpscaleError.noVideoTrack
        }
        let size = try await track.load(.naturalSize)
        return (Int(size.width.rounded()), Int(size.height.rounded()))
    }

    /// `url` with its symbolic links resolved - for a file not made yet,
    /// its folder's, with the name after it.
    private static func resolved(_ url: URL) -> URL {
        let standard = url.standardizedFileURL
        if FileManager.default.fileExists(atPath: standard.path) { return standard.resolvingSymlinksInPath() }
        return standard.deletingLastPathComponent().resolvingSymlinksInPath()
            .appendingPathComponent(standard.lastPathComponent)
    }

    /// Whether writing `destination` would write `source` itself: the same
    /// path once "..", symbolic links and the like are resolved, or the same
    /// file under another name (a hard link).
    static func isSameFile(_ source: URL, _ destination: URL) -> Bool {
        let from = resolved(source), to = resolved(destination)
        if from.path == to.path { return true }
        let keys: Set<URLResourceKey> = [.fileResourceIdentifierKey, .volumeIdentifierKey]
        guard let a = try? from.resourceValues(forKeys: keys), let b = try? to.resourceValues(forKeys: keys),
              let fileA = a.fileResourceIdentifier, let fileB = b.fileResourceIdentifier,
              let volumeA = a.volumeIdentifier, let volumeB = b.volumeIdentifier else { return false }
        return volumeA.isEqual(volumeB) && fileA.isEqual(fileB)
    }

    /// Writes `source` enlarged by `factor` to `destination` (H.264 at
    /// the engine's bitrate rule, the sound and metadata as they are).
    /// `progress` gets the share of frames done (on the main queue);
    /// `isCancelled` is asked before each frame.
    ///
    /// The video is written to a hidden file next to `destination` and put
    /// in its place only once it is complete, so a file already there stays
    /// as it was if this fails or is cancelled. Writing over `source` itself
    /// is refused.
    static func upscale(_ source: URL, to destination: URL, factor: UpscaleFactor = .two,
                        upscaler: VideoUpscaler = .standard,
                        progress: @escaping (Double) -> Void = { _ in },
                        isCancelled: @escaping () -> Bool = { false }) async throws {
        guard !isSameFile(source, destination) else { throw UpscaleError.overSource }
        // Same folder, so the finished file moves into place, not copies.
        let staging = destination.deletingLastPathComponent()
            .appendingPathComponent(".h3c-upscale-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: staging) }
        let asset = AVURLAsset(url: source)
        guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
            throw UpscaleError.noVideoTrack
        }
        let audioTrack = try await asset.loadTracks(withMediaType: .audio).first
        let (natural, transform, frameRate, timeRange) = try await videoTrack.load(
            .naturalSize, .preferredTransform, .nominalFrameRate, .timeRange)
        let inputWidth = Int(natural.width.rounded()), inputHeight = Int(natural.height.rounded())
        let (width, height) = factor.size(width: inputWidth, height: inputHeight)
        guard width <= maximumSide, height <= maximumSide else {
            throw UpscaleError.tooLarge(width: width, height: height)
        }
        let scaler: ChunkedFrameScaler
        switch upscaler {
        case .standard: scaler = PerFrameChunkAdapter(VImageScaler())
        case .metalFX:
            guard let made = MetalFXScaler(inputWidth: inputWidth, inputHeight: inputHeight,
                                           outputWidth: width, outputHeight: height) else {
                throw UpscaleError.unavailable(upscaler.label)
            }
            scaler = PerFrameChunkAdapter(made)
        case .realESRGANGeneral, .realESRGANAnime:
            guard let weights = upscaler.weightsURL else { throw UpscaleError.unavailable(upscaler.label) }
            scaler = PerFrameChunkAdapter(try RealESRGANScaler(network: RealESRGANNetwork(weights: weights, width: inputWidth,
                                                                                          height: inputHeight,
                                                                                          scale: factor == .two ? 2 : 4),
                                                               outputWidth: width, outputHeight: height))
        case .swiftVR:
            guard let cachePath = VideoUpscaler.swiftVRCachePath else { throw UpscaleError.unavailable(upscaler.label) }
            let resolver: H3VideoSuperResolver
            do {
                resolver = try H3VideoSuperResolver(modelDirectory: VideoUpscaler.swiftVRModelDirectory, cachePath: cachePath)
            } catch {
                throw UpscaleError.cannotWrite(error.localizedDescription)
            }
            let requirements = resolver.requirements(inputWidth: inputWidth, inputHeight: inputHeight,
                                                     outputWidth: width, outputHeight: height)
            guard requirements.supported else { throw UpscaleError.cannotWrite(requirements.reason) }
            do {
                scaler = VSRChunkScaler(session: try resolver.openSession(inputWidth: inputWidth, inputHeight: inputHeight,
                                                                          outputWidth: width, outputHeight: height))
            } catch {
                throw UpscaleError.cannotWrite(error.localizedDescription)
            }
        }
        let expectedFrames = max(1, Int((timeRange.duration.seconds * Double(frameRate)).rounded()))

        let reader = try AVAssetReader(asset: asset)
        let videoOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ])
        videoOutput.alwaysCopiesSampleData = false
        reader.add(videoOutput)
        var audioOutput: AVAssetReaderTrackOutput?
        if let audioTrack {
            let output = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: nil)
            reader.add(output)
            audioOutput = output
        }

        let writer = try AVAssetWriter(outputURL: staging, fileType: .mp4)
        writer.metadata = (try? await asset.load(.metadata)) ?? []
        // The engine's settings (h3_av_writer.m).
        let videoInput = AVAssetWriterInput(mediaType: .video,
                                            outputSettings: VideoCodec.hevc.writerSettings(width: width, height: height,
                                                                                           quality: VideoQuality.saved.value))
        videoInput.expectsMediaDataInRealTime = false
        // A rotation keeps its meaning at the larger size.
        // (Its offsets are the frame's width or height.)
        func scaled(_ offset: CGFloat) -> CGFloat {
            if abs(offset - CGFloat(inputWidth)) < 0.5 { return CGFloat(width) }
            if abs(offset - CGFloat(inputHeight)) < 0.5 { return CGFloat(height) }
            return offset * CGFloat(factor.rawValue)
        }
        videoInput.transform = CGAffineTransform(a: transform.a, b: transform.b, c: transform.c, d: transform.d,
                                                 tx: scaled(transform.tx), ty: scaled(transform.ty))
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoInput, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
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
                // Reader -> chunked scaler -> writer, in one serial pump
                // (UPSCALE_PLAN §6.6): frames are collected until the scaler
                // has its chunk, enlarged together, and appended with the
                // input timestamps in order (one output per input overall).
                var done = 0
                var pending: [CVPixelBuffer] = []
                var pendingTimes: [CMTime] = []
                var ready: [CVPixelBuffer] = []
                var readyTimes: [CMTime] = []
                var finished = false
                await pump(videoInput, on: DispatchQueue(label: "h3c.upscale.video")) {
                    if failure.error != nil { return false }
                    if isCancelled() {
                        scaler.cancel()
                        failure.set(UpscaleError.cancelled)
                        return false
                    }
                    while ready.isEmpty {
                        if finished { return false }
                        guard let pool = adaptor.pixelBufferPool else { return false }
                        var isLast = false
                        while pending.count < scaler.chunkFrames {
                            guard let sample = videoOutput.copyNextSampleBuffer() else { isLast = true; break }
                            guard let frame = CMSampleBufferGetImageBuffer(sample) else { continue }
                            pending.append(frame)
                            pendingTimes.append(CMSampleBufferGetPresentationTimeStamp(sample))
                        }
                        if pending.isEmpty { finished = true; return false }
                        do {
                            let outputs = try scaler.process(pending, isLast: isLast, pool: pool)
                            ready.append(contentsOf: outputs)
                            readyTimes.append(contentsOf: pendingTimes.prefix(outputs.count))
                            pendingTimes.removeFirst(min(outputs.count, pendingTimes.count))
                            // Times of inputs still inside the scaler stay queued for its later outputs.
                            pending.removeAll()
                        } catch {
                            failure.set(error)
                            return false
                        }
                        if isLast { finished = true }
                    }
                    let enlarged = ready.removeFirst()
                    let time = readyTimes.removeFirst()
                    if !adaptor.append(enlarged, withPresentationTime: time) {
                        failure.set(UpscaleError.cannotWrite(writer.error?.localizedDescription ?? ""))
                        return false
                    }
                    done += 1
                    if done % 4 == 0 {
                        let share = min(Double(done) / Double(expectedFrames), 1)
                        DispatchQueue.main.async { progress(share) }
                    }
                    return true
                }
            }
            if let audioInput, let audioOutput {
                group.addTask {
                    await pump(audioInput, on: DispatchQueue(label: "h3c.upscale.audio")) {
                        if failure.error != nil { return false }
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
            throw error
        }
        await writer.finishWriting()
        if writer.status != .completed {
            throw UpscaleError.cannotWrite(writer.error?.localizedDescription ?? "")
        }
        // Checked again: the names could have changed while it was written.
        guard !isSameFile(source, destination) else { throw UpscaleError.overSource }
        let manager = FileManager.default
        if manager.fileExists(atPath: destination.path) {
            _ = try manager.replaceItemAt(destination, withItemAt: staging)
        } else {
            try manager.moveItem(at: staging, to: destination)
        }
        DispatchQueue.main.async { progress(1) }
    }

    /// Feeds `input` from `next` (false: no more) whenever it wants data.
    static func pump(_ input: AVAssetWriterInput, on queue: DispatchQueue,
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

    /// The first error from either track's queue.
    final class Failure: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: Error?
        var error: Error? {
            lock.lock()
            defer { lock.unlock() }
            return stored
        }
        func set(_ error: Error) {
            lock.lock()
            if stored == nil { stored = error }
            lock.unlock()
        }
    }
}

/// VideoUpscaler.standard: vImage's Lanczos on the CPU.
private struct VImageScaler: FrameScaler {
    func scale(_ source: CVPixelBuffer, into destination: CVPixelBuffer) throws {
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(destination, [])
        }
        guard let input = CVPixelBufferGetBaseAddress(source),
              let output = CVPixelBufferGetBaseAddress(destination) else {
            throw VideoUpscale.UpscaleError.cannotWrite(String(localized: "フレームを拡大できませんでした"))
        }
        var from = vImage_Buffer(data: input,
                                 height: vImagePixelCount(CVPixelBufferGetHeight(source)),
                                 width: vImagePixelCount(CVPixelBufferGetWidth(source)),
                                 rowBytes: CVPixelBufferGetBytesPerRow(source))
        var to = vImage_Buffer(data: output,
                               height: vImagePixelCount(CVPixelBufferGetHeight(destination)),
                               width: vImagePixelCount(CVPixelBufferGetWidth(destination)),
                               rowBytes: CVPixelBufferGetBytesPerRow(destination))
        // Four 8-bit channels in any order: BGRA scales as ARGB does.
        guard vImageScale_ARGB8888(&from, &to, nil,
                                   vImage_Flags(kvImageHighQualityResampling | kvImageEdgeExtend)) == kvImageNoError else {
            throw VideoUpscale.UpscaleError.cannotWrite(String(localized: "フレームを拡大できませんでした"))
        }
    }
}

/// VideoUpscaler.realESRGAN*: RealESRGANNetwork on each frame - its 2x
/// straight into the output, or its 4x resized to the size asked for
/// (Lanczos, as Real-ESRGAN does an outscale other than its own).
private final class RealESRGANScaler: FrameScaler {
    let network: RealESRGANNetwork
    private let outputWidth: Int
    private let outputHeight: Int
    /// The network's whole output when it isn't the output's size.
    private var full: UnsafeMutableRawPointer?

    init(network: RealESRGANNetwork, outputWidth: Int, outputHeight: Int) {
        self.network = network
        self.outputWidth = outputWidth
        self.outputHeight = outputHeight
        if network.outputWidth != outputWidth || network.outputHeight != outputHeight {
            full = UnsafeMutableRawPointer.allocate(byteCount: network.outputWidth * network.outputHeight * 4,
                                                    alignment: 64)
        }
    }

    deinit { full?.deallocate() }

    /// Whether `buffer` holds at least `width` x `height` BGRA pixels: the
    /// network and vImage copy that many rows of that many bytes, at the
    /// buffer's own row stride, with nothing else bounding them.
    private static func holds(_ buffer: CVPixelBuffer, width: Int, height: Int) -> Bool {
        CVPixelBufferGetWidth(buffer) >= width && CVPixelBufferGetHeight(buffer) >= height
            && CVPixelBufferGetBytesPerRow(buffer) >= width * 4
    }

    func scale(_ source: CVPixelBuffer, into destination: CVPixelBuffer) throws {
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(destination, [])
        }
        // The network was made for the track's size (naturalSize); a frame
        // decoded smaller would be read past its end.
        guard let input = CVPixelBufferGetBaseAddress(source),
              let output = CVPixelBufferGetBaseAddress(destination),
              Self.holds(source, width: network.width, height: network.height),
              Self.holds(destination, width: outputWidth, height: outputHeight) else {
            throw VideoUpscale.UpscaleError.cannotWrite(String(localized: "フレームを拡大できませんでした"))
        }
        guard let full else {
            try network.upscale(input, sourceRowBytes: CVPixelBufferGetBytesPerRow(source),
                                into: output, destinationRowBytes: CVPixelBufferGetBytesPerRow(destination))
            return
        }
        try network.upscale(input, sourceRowBytes: CVPixelBufferGetBytesPerRow(source),
                            into: full, destinationRowBytes: network.outputWidth * 4)
        var from = vImage_Buffer(data: full, height: vImagePixelCount(network.outputHeight),
                                 width: vImagePixelCount(network.outputWidth), rowBytes: network.outputWidth * 4)
        var to = vImage_Buffer(data: output, height: vImagePixelCount(outputHeight),
                               width: vImagePixelCount(outputWidth),
                               rowBytes: CVPixelBufferGetBytesPerRow(destination))
        guard vImageScale_ARGB8888(&from, &to, nil,
                                   vImage_Flags(kvImageHighQualityResampling | kvImageEdgeExtend)) == kvImageNoError else {
            throw VideoUpscale.UpscaleError.cannotWrite(String(localized: "フレームを拡大できませんでした"))
        }
    }
}

/// VideoUpscaler.metalFX: MetalFX's spatial scaler on the GPU, reading and
/// writing the frames' own memory (CVMetalTextureCache). The scaler writes
/// a private texture, copied into the output frame.
private final class MetalFXScaler: FrameScaler {
    private let queue: MTLCommandQueue
    private let scaler: MTLFXSpatialScaler
    private let cache: CVMetalTextureCache
    private let scaled: MTLTexture

    init?(inputWidth: Int, inputHeight: Int, outputWidth: Int, outputHeight: Int) {
        guard let device = MTLCreateSystemDefaultDevice(),
              MTLFXSpatialScalerDescriptor.supportsDevice(device),
              let queue = device.makeCommandQueue() else { return nil }
        let descriptor = MTLFXSpatialScalerDescriptor()
        descriptor.inputWidth = inputWidth
        descriptor.inputHeight = inputHeight
        descriptor.outputWidth = outputWidth
        descriptor.outputHeight = outputHeight
        descriptor.colorTextureFormat = .bgra8Unorm
        descriptor.outputTextureFormat = .bgra8Unorm
        // Video is gamma-encoded, not linear light.
        descriptor.colorProcessingMode = .perceptual
        guard let scaler = descriptor.makeSpatialScaler(device: device) else { return nil }
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: outputWidth, height: outputHeight, mipmapped: false)
        textureDescriptor.usage = scaler.outputTextureUsage
        textureDescriptor.storageMode = .private
        var cache: CVMetalTextureCache?
        guard let scaled = device.makeTexture(descriptor: textureDescriptor),
              CVMetalTextureCacheCreate(nil, nil, device, nil, &cache) == kCVReturnSuccess,
              let cache else { return nil }
        self.queue = queue
        self.scaler = scaler
        self.cache = cache
        self.scaled = scaled
    }

    private func texture(_ buffer: CVPixelBuffer) throws -> (CVMetalTexture, MTLTexture) {
        var made: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            nil, cache, buffer, nil, .bgra8Unorm,
            CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer), 0, &made)
        guard status == kCVReturnSuccess, let made, let texture = CVMetalTextureGetTexture(made) else {
            throw VideoUpscale.UpscaleError.cannotWrite(String(localized: "フレームを拡大できませんでした"))
        }
        return (made, texture)
    }

    func scale(_ source: CVPixelBuffer, into destination: CVPixelBuffer) throws {
        // The CVMetalTextures are held until the GPU is done with them.
        let (sourceRef, input) = try texture(source)
        let (destinationRef, output) = try texture(destination)
        guard let command = queue.makeCommandBuffer() else {
            throw VideoUpscale.UpscaleError.cannotWrite(String(localized: "フレームを拡大できませんでした"))
        }
        scaler.colorTexture = input
        scaler.outputTexture = scaled
        scaler.encode(commandBuffer: command)
        guard let blit = command.makeBlitCommandEncoder() else {
            throw VideoUpscale.UpscaleError.cannotWrite(String(localized: "フレームを拡大できませんでした"))
        }
        blit.copy(from: scaled, to: output)
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        withExtendedLifetime((sourceRef, destinationRef)) {}
        if command.status != .completed {
            throw VideoUpscale.UpscaleError.cannotWrite(command.error?.localizedDescription ?? "")
        }
    }
}

extension GenerationViewModel {
    /// A video waiting for its upscaler to be picked (UpscaleSheet): a
    /// project video, or a file to enlarge into a file of the user's choosing.
    struct PendingUpscale: Identifiable {
        let id = UUID()
        let url: URL
        let projectVideo: ProjectVideo?
        let width: Int
        let height: Int
    }

    /// Whether `video` can be enlarged by `factor` (within the 4096-pixel
    /// limit): its size is an enlarged copy's own, or else the size it was
    /// made at.
    func canUpscale(_ video: ProjectVideo, by factor: UpscaleFactor = .oneAndHalf) -> Bool {
        guard !isUpscalingVideo, let record = video.record else { return false }
        var size = record.upscaledSize
        if size == nil, let dimensions = SizeProfile(rawValue: record.draft.sizeProfile)?.dimensions {
            size = [Int(dimensions.width), Int(dimensions.height)]
        }
        guard let size, size.count == 2 else { return true }
        return factor.fits(width: size[0], height: size[1])
    }

    /// 拡大した動画をつくる…: `video` for the upscaler sheet (shown by
    /// the project video list itself, being a sheet already).
    func upscaleRequest(for video: ProjectVideo) async -> PendingUpscale? {
        do {
            let size = try await VideoUpscale.frameSize(of: video.url)
            return PendingUpscale(url: video.url, projectVideo: video, width: size.width, height: size.height)
        } catch {
            projectMessage = String(localized: "動画を拡大できませんでした（詳細: \(error.localizedDescription)）")
            return nil
        }
    }

    /// The sheet's 拡大する: a project video into the project, a file into a
    /// file chosen now.
    func startUpscale(_ pending: PendingUpscale, upscaler: VideoUpscaler, factor: UpscaleFactor) {
        if let video = pending.projectVideo {
            Task { await upscaleProjectVideo(video, upscaler: upscaler, factor: factor) }
            return
        }
        let panel = NSSavePanel()
        // A temporary result's own folder is the temp folder: Downloads instead.
        let folder = project?.url ?? (isTemporaryResult(pending.url)
            ? FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask)[0]
            : pending.url.deletingLastPathComponent())
        panel.directoryURL = folder
        panel.nameFieldStringValue = Self.freeName(for: pending.url, in: folder, factor: factor)
        panel.allowedContentTypes = [.mpeg4Movie]
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        Task { await upscaleVideoFile(pending.url, to: destination, upscaler: upscaler, factor: factor) }
    }

    /// 動画ファイルを拡大…: picks any video and asks which upscaler.
    func pickVideoToUpscale() {
        guard let path = chooseFile(allowedContentTypes: [.movie]) else { return }
        askForUpscale(URL(fileURLWithPath: path), projectVideo: nil)
    }

    /// 表示中の動画を拡大…: the video in the preview - a project video
    /// (with its record) into the project, any other into a file.
    func upscaleDisplayedVideo() {
        guard canUpscaleDisplayedVideo, let resultURL else { return }
        askForUpscale(resultURL, projectVideo: displayedProjectVideoWithRecord)
    }

    var canUpscaleDisplayedVideo: Bool {
        guard !isUpscalingVideo, resultURL != nil else { return false }
        if let video = displayedProjectVideoWithRecord { return canUpscale(video) }
        return true
    }

    private var displayedProjectVideoWithRecord: ProjectVideo? {
        projectVideos.first { $0.url == resultURL && $0.record != nil }
    }

    /// Opens the upscaler sheet for `url`.
    private func askForUpscale(_ url: URL, projectVideo: ProjectVideo?) {
        Task {
            do {
                let size = try await VideoUpscale.frameSize(of: url)
                pendingUpscale = PendingUpscale(url: url, projectVideo: projectVideo,
                                                width: size.width, height: size.height)
            } catch {
                projectMessage = String(localized: "動画を拡大できませんでした（詳細: \(error.localizedDescription)）")
            }
        }
    }

    /// The name the enlarged copy of `video` gets: "<name>_x2.mp4" (by the
    /// factor), numbered if that is taken.
    func upscaledVideoName(for video: ProjectVideo, factor: UpscaleFactor = .two) -> String? {
        guard let project else { return nil }
        return Self.freeName(for: video.url, in: project.url, factor: factor)
    }

    /// "<name>_x2.mp4" next to the others in `folder`, numbered if taken.
    static func freeName(for source: URL, in folder: URL, factor: UpscaleFactor) -> String {
        let base = source.deletingPathExtension().lastPathComponent + "_" + factor.suffix
        var name = base
        var counter = 2
        while FileManager.default.fileExists(atPath: folder.appendingPathComponent(name + ".mp4").path) {
            name = "\(base)-\(counter)"
            counter += 1
        }
        return name + ".mp4"
    }

    /// 中止 on the progress banner (and POST /api/upscale/cancel).
    func cancelUpscale() {
        upscaleCancel.request()
    }

    /// Runs one upscale with the banner's progress and 中止.
    private func runUpscale(_ source: URL, to destination: URL, upscaler: VideoUpscaler,
                            factor: UpscaleFactor) async throws {
        upscalingVideoName = source.lastPathComponent
        upscalingFactor = factor
        upscaleProgress = 0
        upscaleCancel.reset()
        isUpscalingVideo = true
        defer { isUpscalingVideo = false }
        let cancel = upscaleCancel
        try await VideoUpscale.upscale(source, to: destination, factor: factor, upscaler: upscaler,
                                       progress: { [weak self] share in self?.upscaleProgress = share },
                                       isCancelled: { cancel.isRequested })
    }

    /// 拡大した動画をつくる: `video` enlarged by `factor` into a new project
    /// video with its record, marked as enlarged. The original's latents
    /// come along (unless it is an enlarged copy itself), so it can be
    /// continued: at the original's size, then enlarged the same way
    /// (ContinuationSource.enlargement). `previous` is the enlarged copy
    /// that continues (stored path), when `video` is such a continuation.
    /// Returns whether it worked; projectMessage says what happened either
    /// way.
    @discardableResult
    func upscaleProjectVideo(_ video: ProjectVideo, as name: String? = nil,
                             upscaler: VideoUpscaler = .standard, factor: UpscaleFactor = .two,
                             continuing previous: String? = nil) async -> Bool {
        guard let project, !isUpscalingVideo,
              let name = name ?? upscaledVideoName(for: video, factor: factor) else {
            return false
        }
        let destination = project.url.appendingPathComponent(name)
        let started = Date()
        do {
            try await runUpscale(video.url, to: destination, upscaler: upscaler, factor: factor)
            if var record = video.record {
                record.upscaleFactor = factor.rawValue
                record.video = name
                record.completedAt = Date()
                record.upscaledFrom = video.url.lastPathComponent
                record.upscaler = upscaler.rawValue
                let size = try await VideoUpscale.frameSize(of: destination)
                record.upscaledSize = [size.width, size.height]
                // Joins only other enlarged copies: the one it continues.
                record.joinedClips = nil
                record.chainPlannedClips = nil
                record.draft.continueFrom = previous
                if previous == nil { record.draft.continueFrames = nil }
                try ProjectFiles.encoder.encode(record)
                    .write(to: destination.deletingPathExtension().appendingPathExtension("json"), options: .atomic)
                if video.record?.upscaledFrom == nil {
                    try? FileManager.default.copyItem(at: ProjectFiles.latentURL(for: video.url),
                                                      to: ProjectFiles.latentURL(for: destination))
                }
            }
        } catch {
            try? FileManager.default.removeItem(at: destination)
            reportUpscaleFailure(error, source: video.url)
            return false
        }
        reloadProjectVideos()
        projectMessage = String(localized: "\(video.url.lastPathComponent)を\(factor.label)に拡大して、\(name)をつくりました（\(upscaler.label)、\(formatElapsed(Date().timeIntervalSince(started)))）")
        projectMessageFile = destination
        return true
    }

    /// A continuation of an enlarged copy, to enlarge the same way.
    struct PendingEnlargement {
        let video: URL
        /// The enlarged copy it continues (stored path).
        let previous: String?
        let enlargement: ContinuationSource.Enlargement
    }

    /// Enlarges the continuations of enlarged copies a run made, each into
    /// an enlarged copy that continues the one its clip continued, and
    /// shows the last.
    func enlargeContinuations(_ pending: [PendingEnlargement]) async {
        var made: [String] = []
        for item in pending {
            guard let video = projectVideos.first(where: { $0.url == item.video }),
                  let name = upscaledVideoName(for: video, factor: item.enlargement.factor),
                  await upscaleProjectVideo(video, as: name, upscaler: item.enlargement.upscaler,
                                            factor: item.enlargement.factor, continuing: item.previous) else {
                return
            }
            made.append(name)
        }
        guard let last = made.last, let video = projectVideos.first(where: { $0.url.lastPathComponent == last }) else { return }
        showProjectVideo(video)
        let factor = pending[0].enlargement.factor
        projectMessage = String(localized: "続きを\(factor.label)に拡大して、\(made.joined(separator: "、"))をつくりました。「つないで書き出す」で、拡大した動画どうしをつなげます")
        projectMessageFile = video.url
    }

    /// 動画ファイルを拡大: `source` enlarged by `factor` into `destination`.
    @discardableResult
    func upscaleVideoFile(_ source: URL, to destination: URL, upscaler: VideoUpscaler = .standard,
                          factor: UpscaleFactor = .two) async -> Bool {
        guard !isUpscalingVideo else { return false }
        let started = Date()
        do {
            try await runUpscale(source, to: destination, upscaler: upscaler, factor: factor)
        } catch {
            reportUpscaleFailure(error, source: source)
            return false
        }
        if let project, destination.deletingLastPathComponent().standardizedFileURL == project.url.standardizedFileURL {
            reloadProjectVideos()
        }
        projectMessage = String(localized: "\(source.lastPathComponent)を\(factor.label)に拡大して、\(destination.lastPathComponent)に保存しました（\(upscaler.label)、\(formatElapsed(Date().timeIntervalSince(started)))）")
        projectMessageFile = destination
        return true
    }

    private func reportUpscaleFailure(_ error: Error, source: URL) {
        if case VideoUpscale.UpscaleError.cancelled = error {
            projectMessage = String(localized: "拡大を中止しました")
            return
        }
        NSLog("H3cApp: upscaling %@ failed: %@", source.path, String(describing: error))
        projectMessage = String(localized: "動画を拡大できませんでした（詳細: \(error.localizedDescription)）")
    }
}

/// The 中止 request for the upscale running now, read from the video queue.
final class UpscaleCancel: @unchecked Sendable {
    private let lock = NSLock()
    private var requested = false
    var isRequested: Bool {
        lock.lock()
        defer { lock.unlock() }
        return requested
    }
    func request() {
        lock.lock()
        requested = true
        lock.unlock()
    }
    func reset() {
        lock.lock()
        requested = false
        lock.unlock()
    }
}
