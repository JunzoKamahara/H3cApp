import Accelerate
import AppKit
import AVFoundation
import CoreVideo
import Foundation
import H3Engine
import H3Upscale
import Metal
import MetalFX

// 拡大した動画をつくる / 動画ファイルを拡大: a finished video, enlarged
// after the fact (1.5x to 4x). It needs nothing from the generation - the "×2"
// sizes enlarge the decoded frames the same way - so it works on any video,
// its sound copied as it is.

/// How the frames are enlarged (the choice in UpscaleSheet): the names the
/// API and the saved records use. What each one does is its UpscalerMethod
/// (Upscale/Methods); a new upscaler is a case here, a type there and a line
/// in UpscalerRegistry.
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
    /// The same model restoring at 4x (in tiles when the frame is large)
    /// and the frames brought down to the chosen size with the standard
    /// resize: gentler texture, truer to the source, about five times the
    /// time of a direct 2x (UPSCALE_PLAN appendix A.3).
    case swiftVRFourX
    /// SeedVR2-3B (ByteDance Seed, Apache-2.0), run frame by frame by the
    /// engine (h3_seedvr2.h) from the downloaded MLX 8-bit weights: the
    /// most detailed enlargement and by far the slowest (UPSCALE_PLAN
    /// appendix A.7). Experimental.
    case seedVR2
    /// The same model on windows of frames through its temporal path (the
    /// causal video VAE and 3D window attention, h3_seedvr2_video.c): the
    /// per-frame detail with less flicker, at about 2.5x the time and
    /// 24 GB Macs' limit of memory (UPSCALE_PLAN appendix A.10, which chose
    /// SeedVR2VideoWindow.standard). Experimental.
    case seedVR2Video
    /// H3 itself regenerating the clip on a larger canvas from its latent:
    /// LBH's latent upscaler, then two denoising steps with NVIDIA's
    /// Open-H3 refiner LoRA and LightX2V Turbo v1.2 (OpenH3Regenerate.swift,
    /// UPSCALE_PLAN appendix A.4). Only for the app's own clips, up to the
    /// released 1344x768 canvas. Experimental.
    case openH3

    var id: String { rawValue }

    /// What this method is and does: its own type (Upscale/Methods), found
    /// in UpscalerRegistry. Everything below asks it.
    var method: any UpscalerMethod { UpscalerRegistry.method(for: self) }

    var label: String { method.label }

    /// What it is and what it suits, for the sheet.
    var detail: String { method.detail }

    /// The model it needs downloaded (a モデルを準備 button), if any.
    var model: (any UpscalerModel)? { method.model }

    /// Whether it regenerates with H3 itself (a project clip's latent, not
    /// the frames): only project videos, and sizes of its own.
    var regeneratesFromLatent: Bool { method.regeneratesFromLatent }

    /// Whether this Mac can run it.
    var isAvailable: Bool { method.isAvailable }

    /// Why an unavailable upscaler can't be chosen, for the sheet.
    var unavailableReason: String? { isAvailable ? nil : method.missingReason }

    /// Whether it wants the GPU to itself (so it waits for a generation).
    var needsIdleGPU: Bool { method.needsIdleGPU }

    /// The smallest factor it takes: 等倍 for the ones that redraw the
    /// frames, while a resizer at that size would change nothing.
    var factorFloor: Double { method.factorFloor }

    /// The largest factor it takes, whatever the video: the resizers go on
    /// as far as a side allows, the AI models stop at the 4x they were
    /// made for.
    var factorCeiling: Double { method.factorCeiling }

    /// The engine's own answer on whether sizes (input width, height,
    /// output width, height) fit this Mac, for the models that have one and
    /// are installed. It reads no weights: the sheet asks it before loading
    /// anything.
    var sizeCheck: (@Sendable (Int, Int, Int, Int) -> Bool)? { method.sizeCheck }

    /// Where the Open-H3 files live (the latent upscaler, the extracted
    /// refiner, Turbo when downloaded here); H3_OPENH3_MODEL_DIR points
    /// elsewhere while developing.
    static var openH3ModelDirectory: String {
        ProcessInfo.processInfo.environment["H3_OPENH3_MODEL_DIR"] ?? (h3AppSupportDirectory + "/Upscalers/OpenH3")
    }

    /// Where the SeedVR2 model lives (config.json and the three safetensors
    /// of the MLX 8-bit export); H3_SEEDVR2_MODEL_DIR points elsewhere while
    /// developing.
    static var seedVR2ModelDirectory: String {
        ProcessInfo.processInfo.environment["H3_SEEDVR2_MODEL_DIR"] ?? (h3AppSupportDirectory + "/Upscalers/SeedVR2")
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

    /// The method chosen last in the sheet, which it opens on next time.
    static var lastChosen: VideoUpscaler? {
        get { UserDefaults.standard.string(forKey: lastChosenKey).flatMap(VideoUpscaler.init(rawValue:)) }
        set { UserDefaults.standard.set(newValue?.rawValue, forKey: lastChosenKey) }
    }
    private static let lastChosenKey = "h3c-app.lastUpscaler"
}


/// How much bigger (the sheet's 倍率): one of the usual factors or any
/// other in hundredths. What each upscaler can make of a video is
/// UpscaleChoices.
struct UpscaleFactor: Hashable, Identifiable {
    let rawValue: Double

    /// 等倍: no larger, only redrawn, for the upscalers that add detail.
    static let minimum = 1.0
    static let maximum = 8.0

    /// Rounded to hundredths; nil outside minimum...maximum.
    init?(rawValue: Double) {
        let rounded = (rawValue * 100).rounded() / 100
        guard rounded >= Self.minimum, rounded <= Self.maximum else { return nil }
        self.rawValue = rounded
    }

    private init(usual rawValue: Double) { self.rawValue = rawValue }

    static let oneAndHalf = UpscaleFactor(usual: 1.5)
    static let two = UpscaleFactor(usual: 2)
    static let three = UpscaleFactor(usual: 3)
    static let four = UpscaleFactor(usual: 4)
    static let same = UpscaleFactor(usual: minimum)
    /// The factors offered without typing one.
    static let usual = [same, oneAndHalf, two, three, four]

    var id: Double { rawValue }

    /// "1.5", "2", "2.25".
    var text: String { Self.text(rawValue) }

    static func text(_ value: Double) -> String {
        guard value != value.rounded() else { return String(Int(value)) }
        var digits = String(format: "%.2f", value)
        while digits.hasSuffix("0") { digits.removeLast() }
        return digits
    }

    /// "等倍", "1.5倍", "2倍".
    var label: String {
        if self == .same { return String(localized: "等倍") }
        return rawValue == rawValue.rounded()
            ? String(localized: "\(Int(rawValue))倍")
            : String(localized: "\(text)倍")
    }

    /// "x1.5", "x2" (file names).
    var suffix: String { "x" + text }

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

/// What one upscaler can make of a `width` x `height` video: the factors
/// the sheet offers, the range a typed factor may take, and the size each
/// makes. Every upscaler takes any factor in its range: the output size is
/// free for the resizers and SwiftVR and SeedVR2, the AI networks of fixed
/// scale resize their 4x, and Open-H3 rounds to its canvas.
struct UpscaleChoices {
    let upscaler: VideoUpscaler
    let width: Int
    let height: Int
    /// The usual factors that make a size of their own.
    private(set) var usual: [UpscaleFactor] = []
    /// Lowest and highest factor in hundredths; nil when this video can't
    /// be enlarged this way at all.
    private(set) var range: ClosedRange<Double>?
    private(set) var limit = Limit.method
    private let sizeCheck: (@Sendable (Int, Int, Int, Int) -> Bool)?

    /// What stops the factor at the top of `range`.
    enum Limit: Equatable {
        case method, canvas, side, memory
    }

    init(upscaler: VideoUpscaler, width: Int, height: Int) {
        self.upscaler = upscaler
        self.width = width
        self.height = height
        sizeCheck = upscaler.sizeCheck
        let ceiling = Int((upscaler.factorCeiling * 100).rounded())
        let floor = Int((upscaler.factorFloor * 100).rounded())
        func made(_ hundredths: Int) -> (width: Int, height: Int)? {
            UpscaleFactor(rawValue: Double(hundredths) / 100).flatMap { size($0) }
        }
        /// The first in `span` that is true, those before it being false.
        func first(in span: ClosedRange<Int>, where test: (Int) -> Bool) -> Int {
            var low = span.lowerBound, high = span.upperBound
            while low < high {
                let middle = (low + high) / 2
                if test(middle) { high = middle } else { low = middle + 1 }
            }
            return low
        }
        // One size that works, by tenths, then the two ends around it. Too
        // small and too large both fail, so neither end can be searched
        // from the other.
        guard let found = stride(from: floor, through: ceiling, by: 10).first(where: { made($0) != nil }) else { return }
        let lowest = first(in: max(floor, found - 9)...found) { made($0) != nil }
        let highest: Int
        if upscaler.regeneratesFromLatent {
            // Past the canvas every factor makes the same size: the first
            // of those. A video that fills the canvas only has 等倍.
            highest = made(ceiling).map { top in
                first(in: found...ceiling) { hundredths in
                    made(hundredths).map { $0 == top } ?? false
                }
            } ?? found
            limit = .canvas
        } else {
            highest = first(in: found...(ceiling + 1)) { $0 > ceiling || made($0) == nil } - 1
            if highest < ceiling {
                limit = UpscaleFactor(rawValue: Double(highest + 1) / 100)?.fits(width: width, height: height) == false ? .side : .memory
            }
        }
        range = Double(lowest) / 100...Double(highest) / 100
        for factor in UpscaleFactor.usual {
            guard let made = size(factor) else { continue }
            let repeated = usual.contains { smaller in size(smaller).map { $0 == made } ?? false }
            if !repeated { usual.append(factor) }
        }
    }

    /// The size `factor` makes; nil when it can't.
    func size(_ factor: UpscaleFactor) -> (width: Int, height: Int)? {
        guard factor.rawValue >= upscaler.factorFloor, factor.rawValue <= upscaler.factorCeiling else { return nil }
        if upscaler.regeneratesFromLatent {
            return OpenH3Plan.plan(width: width, height: height, factor: factor).map { ($0.width, $0.height) }
        }
        let made = factor.size(width: width, height: height)
        guard factor.fits(width: width, height: height),
              factor == .same || made.width > width || made.height > height else { return nil }
        if let sizeCheck, !sizeCheck(width, height, made.width, made.height) { return nil }
        return made
    }

    /// Whether a typed factor is in range (the usual ones need not be: on
    /// Open-H3's canvas 2倍 can make what 1.4倍 already does).
    func allows(_ factor: UpscaleFactor) -> Bool {
        usual.contains(factor) || (range?.contains(factor.rawValue) ?? false)
    }

    /// Says why the range ends where it does.
    var limitText: String {
        switch limit {
        case .method: return String(localized: "この方法は\(UpscaleFactor.text(upscaler.factorCeiling))倍までです。")
        case .canvas: return String(localized: "Open-H3はH3の上限の1344×768までです。")
        case .side: return String(localized: "一辺が4096ピクセルを超えない倍率までです。")
        case .memory: return String(localized: "このMacのGPUメモリに入る大きさまでです。")
        }
    }
}

/// One frame in, the same frame enlarged out (both 32BGRA).
protocol FrameScaler {
    func scale(_ source: CVPixelBuffer, into destination: CVPixelBuffer) throws
}

/// Frames in groups, enlarged frames out of a pool, in order and one output
/// per input over the whole video - possibly behind by a few frames, and
/// flushed by `isLast` (UPSCALE_PLAN §6.6). `chunkFrames` is how many the
/// next call should bring.
/// `progress` may be told the share of the call done so far (0-1) while
/// a chunk takes long.
protocol ChunkedFrameScaler {
    var chunkFrames: Int { get }
    func process(_ frames: [CVPixelBuffer], isLast: Bool, pool: CVPixelBufferPool,
                 progress: @escaping (Double) -> Void) throws -> [CVPixelBuffer]
    /// Stops work inside a chunk when the user cancels.
    func cancel()
}

/// Any per-frame scaler as a chunk of one: the same operations, so the
/// output is unchanged byte for byte.
final class PerFrameChunkAdapter: ChunkedFrameScaler {
    let scaler: FrameScaler
    init(_ scaler: FrameScaler) { self.scaler = scaler }
    var chunkFrames: Int { 1 }
    func process(_ frames: [CVPixelBuffer], isLast: Bool, pool: CVPixelBufferPool,
                 progress: @escaping (Double) -> Void) throws -> [CVPixelBuffer] {
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
        let scaler = try upscaler.method.makeScaler(UpscaleJob(inputWidth: inputWidth, inputHeight: inputHeight,
                                                               width: width, height: height, factor: factor,
                                                               isCancelled: isCancelled))
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
                // The frame after a full chunk, read to know whether that
                // chunk is the last: a scaler that holds frames back only
                // flushes them on `isLast`, and a video can end exactly on
                // a chunk (124 frames are 28 + 4 x 24).
                var lookahead: (frame: CVPixelBuffer, time: CMTime)?
                var ready: [CVPixelBuffer] = []
                var readyTimes: [CMTime] = []
                var finished = false
                var reported = 0.0
                func report(_ share: Double) {
                    let clamped = min(max(share, reported), 1)
                    guard clamped > reported else { return }
                    reported = clamped
                    DispatchQueue.main.async { progress(clamped) }
                }
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
                        func nextFrame() -> (frame: CVPixelBuffer, time: CMTime)? {
                            while let sample = videoOutput.copyNextSampleBuffer() {
                                if let frame = CMSampleBufferGetImageBuffer(sample) {
                                    return (frame, CMSampleBufferGetPresentationTimeStamp(sample))
                                }
                            }
                            return nil
                        }
                        while pending.count < scaler.chunkFrames {
                            guard let next = lookahead ?? nextFrame() else { isLast = true; break }
                            lookahead = nil
                            pending.append(next.frame)
                            pendingTimes.append(next.time)
                        }
                        if !isLast {
                            lookahead = nextFrame()
                            isLast = lookahead == nil
                        }
                        if pending.isEmpty { finished = true; return false }
                        do {
                            // Frames inside the scaler count by the share of
                            // the chunk it reports done.
                            let base = done, inFlight = pending.count
                            let outputs = try scaler.process(pending, isLast: isLast, pool: pool) { share in
                                report((Double(base) + share * Double(inFlight)) / Double(expectedFrames))
                            }
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
                    if done % 4 == 0 { report(Double(done) / Double(expectedFrames)) }
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




extension GenerationViewModel {
    /// A video waiting for its upscaler to be picked (UpscaleSheet): a
    /// project video, or a file to enlarge into a file of the user's choosing.
    struct PendingUpscale: Identifiable {
        let id = UUID()
        let url: URL
        /// The video with its record, when it has one: a project video, or
        /// the preview's temporary result (generated with no project open),
        /// whose record and latent let Open-H3 regenerate it.
        let projectVideo: ProjectVideo?
        let width: Int
        let height: Int
        /// Whether the enlarged copy joins the open project (else it goes
        /// to a file chosen next).
        var inProject = false
    }

    /// Whether `video` can be enlarged by `factor` (within the 4096-pixel
    /// limit): its size is an enlarged copy's own, or else the size it was
    /// made at.
    func canUpscale(_ video: ProjectVideo, by factor: UpscaleFactor = .same) -> Bool {
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
            return PendingUpscale(url: video.url, projectVideo: video, width: size.width, height: size.height,
                                  inProject: true)
        } catch {
            projectMessage = String(localized: "動画を拡大できませんでした（詳細: \(error.localizedDescription)）")
            return nil
        }
    }

    /// The sheet's 拡大する: a project video into the project, a file into a
    /// file chosen now.
    func startUpscale(_ pending: PendingUpscale, upscaler: VideoUpscaler, factor: UpscaleFactor,
                      strength: OpenH3Strength = .standard) {
        if let video = pending.projectVideo, pending.inProject {
            Task { await upscaleProjectVideo(video, upscaler: upscaler, factor: factor, strength: strength) }
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
        Task {
            await upscaleVideoFile(pending.url, to: destination, upscaler: upscaler, factor: factor,
                                   strength: strength, video: pending.projectVideo)
        }
    }

    /// 動画ファイルを拡大…: picks any video and asks which upscaler.
    func pickVideoToUpscale() {
        guard let path = chooseFile(allowedContentTypes: [.movie]) else { return }
        askForUpscale(URL(fileURLWithPath: path), projectVideo: nil)
    }

    /// 表示中の動画を拡大…: the video in the preview - a project video
    /// (with its record) into the project, the temporary result of a
    /// generation with no project open (with its record, so Open-H3 can
    /// regenerate it too) or any other video into a file.
    func upscaleDisplayedVideo() {
        guard canUpscaleDisplayedVideo, let resultURL else { return }
        let projectVideo = displayedProjectVideoWithRecord
        askForUpscale(resultURL, projectVideo: projectVideo ?? displayedTemporaryVideo,
                      inProject: projectVideo != nil)
    }

    var canUpscaleDisplayedVideo: Bool {
        guard !isUpscalingVideo, resultURL != nil else { return false }
        if let video = displayedProjectVideoWithRecord ?? displayedTemporaryVideo { return canUpscale(video) }
        return true
    }

    private var displayedProjectVideoWithRecord: ProjectVideo? {
        projectVideos.first { $0.url == resultURL && $0.record != nil }
    }

    /// The preview's temporary result with its record: its latent lies
    /// beside it in the temp folder until the next generation replaces it.
    private var displayedTemporaryVideo: ProjectVideo? {
        savableTemporaryResult.map { ProjectVideo(url: $0.url, record: $0.record) }
    }

    /// Opens the upscaler sheet for `url`.
    private func askForUpscale(_ url: URL, projectVideo: ProjectVideo?, inProject: Bool = false) {
        Task {
            do {
                let size = try await VideoUpscale.frameSize(of: url)
                pendingUpscale = PendingUpscale(url: url, projectVideo: projectVideo,
                                                width: size.width, height: size.height, inProject: inProject)
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

    /// Why `upscaler` can't start now, if it can't: one that wants the GPU
    /// to itself (SwiftVR) waits for a generation, a cache build or an
    /// import, as those wait for an enlargement (validationMessage).
    func upscaleBlockedReason(_ upscaler: VideoUpscaler) -> String? {
        guard upscaler.needsIdleGPU else { return nil }
        if isGenerating || isBuildingCache {
            return String(localized: "生成中は\(upscaler.label)で拡大できません。生成が終わってから拡大してください")
        }
        if isImportingVideo {
            return String(localized: "動画の取り込み中は\(upscaler.label)で拡大できません。取り込みが終わってから拡大してください")
        }
        return nil
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
                             strength: OpenH3Strength = .standard,
                             continuing previous: String? = nil) async -> Bool {
        guard let project, !isUpscalingVideo,
              let name = name ?? upscaledVideoName(for: video, factor: factor) else {
            return false
        }
        if let reason = upscaleBlockedReason(upscaler) {
            projectMessage = reason
            return false
        }
        if upscaler.regeneratesFromLatent, let reason = openH3Reason(for: video, factor: factor) {
            projectMessage = reason
            return false
        }
        let destination = project.url.appendingPathComponent(name)
        let started = Date()
        do {
            if upscaler.regeneratesFromLatent {
                _ = try await runOpenH3Regeneration(video, to: destination, factor: factor, strength: strength)
            } else {
                try await runUpscale(video.url, to: destination, upscaler: upscaler, factor: factor)
            }
            if var record = video.record {
                record.upscaleFactor = factor.rawValue
                record.video = name
                record.completedAt = Date()
                record.upscaledFrom = video.url.lastPathComponent
                record.upscaler = upscaler.rawValue
                record.upscaleStrength = upscaler.regeneratesFromLatent ? strength.rawValue : nil
                let window = upscaler.method.recordedWindow
                record.upscaleWindowFrames = window?.windowFrames
                record.upscaleOverlapFrames = window?.overlapFrames
                record.upscaleBlend = window?.blend
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
            // Nothing of a failed copy stays (its latent copy is made after
            // the video, so this is for a failure in between).
            try? FileManager.default.removeItem(at: ProjectFiles.latentURL(for: destination))
            reportUpscaleFailure(error, source: video.url)
            return false
        }
        reloadProjectVideos()
        let made = (try? await VideoUpscale.frameSize(of: destination)).map { "\($0.width)×\($0.height)" } ?? factor.label
        projectMessage = upscaler.regeneratesFromLatent
            ? String(localized: "\(video.url.lastPathComponent)を\(made)に再生成して、\(name)をつくりました（\(upscaler.label)、\(formatElapsed(Date().timeIntervalSince(started)))）")
            : String(localized: "\(video.url.lastPathComponent)を\(factor.label)に拡大して、\(name)をつくりました（\(upscaler.label)、\(formatElapsed(Date().timeIntervalSince(started)))）")
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
                                            factor: item.enlargement.factor, strength: item.enlargement.strength,
                                            continuing: item.previous) else {
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
    /// Open-H3 needs `video`, the source with its record and latent (the
    /// preview's temporary result); the file it makes carries that record
    /// as its label, the way an enlarged project copy's .json does.
    @discardableResult
    func upscaleVideoFile(_ source: URL, to destination: URL, upscaler: VideoUpscaler = .standard,
                          factor: UpscaleFactor = .two, strength: OpenH3Strength = .standard,
                          video: ProjectVideo? = nil) async -> Bool {
        guard !isUpscalingVideo else { return false }
        let regenerated = upscaler.regeneratesFromLatent ? video : nil
        if upscaler.regeneratesFromLatent, regenerated?.record == nil {
            projectMessage = String(localized: "Open-H3 再生成は、このアプリで生成した動画（記録と潜在ファイルのあるもの）にだけ使えます。")
            return false
        }
        if let reason = upscaleBlockedReason(upscaler) {
            projectMessage = reason
            return false
        }
        if let regenerated, let reason = openH3Reason(for: regenerated, factor: factor) {
            projectMessage = reason
            return false
        }
        let started = Date()
        var made: String?
        do {
            if let regenerated, var record = regenerated.record {
                let size = try await runOpenH3Regeneration(regenerated, to: destination, factor: factor, strength: strength)
                made = "\(size.width)×\(size.height)"
                record.upscaleFactor = factor.rawValue
                record.video = destination.lastPathComponent
                record.completedAt = Date()
                record.upscaledFrom = source.lastPathComponent
                record.upscaler = upscaler.rawValue
                record.upscaleStrength = strength.rawValue
                record.upscaleWindowFrames = nil
                record.upscaleOverlapFrames = nil
                record.upscaleBlend = nil
                record.upscaledSize = [size.width, size.height]
                GenerationMetadata.tag(destination, record: record)
            } else {
                try await runUpscale(source, to: destination, upscaler: upscaler, factor: factor)
            }
        } catch {
            if regenerated != nil { try? FileManager.default.removeItem(at: destination) }
            reportUpscaleFailure(error, source: source)
            return false
        }
        if let project, destination.deletingLastPathComponent().standardizedFileURL == project.url.standardizedFileURL {
            reloadProjectVideos()
        }
        let elapsed = formatElapsed(Date().timeIntervalSince(started))
        projectMessage = made.map { made in
            String(localized: "\(source.lastPathComponent)を\(made)に再生成して、\(destination.lastPathComponent)に保存しました（\(upscaler.label)、\(elapsed)）")
        } ?? String(localized: "\(source.lastPathComponent)を\(factor.label)に拡大して、\(destination.lastPathComponent)に保存しました（\(upscaler.label)、\(elapsed)）")
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
