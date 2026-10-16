import Foundation

// Stages of one generation, in the order libh3.a actually runs them. The
// mapping from event names to stages comes from a timestamped trace of a
// real run (see h3_progress_emit / report() in h3.c and h3_dit.c):
//
//   [AdaLN curve, first use only], tokenizer, text encoder, refine text,
//   precompute AdaLN, load transformer core -> setup
//   denoise enqueue k/N, denoise N/N      -> denoise (k = steps finished)
//   audio VAE 0..7, video VAE load 1..36  -> decode. NOT a usable progress
//                                            counter: for a small clip it is
//                                            the decode's own 36 blocks, but
//                                            for a large (chunked/tiled) clip
//                                            it is only the ~2s weight load,
//                                            followed by a silent decode that
//                                            measured 109s for 124 frames at
//                                            512x512. Decode time scales with
//                                            frames x pixels either way, so
//                                            it's modelled by time.
//   [all frames delivered at once]
//   encode 0/N ... N/N                    -> encode
//
// Only ever moves forward.
enum JobStage: Int, Comparable {
    case setup = 0
    case denoise
    case decode
    case encode

    static func < (lhs: JobStage, rhs: JobStage) -> Bool { lhs.rawValue < rhs.rawValue }

    var title: String {
        switch self {
        case .setup: return String(localized: "準備中")
        case .denoise: return String(localized: "ノイズ除去")
        case .decode: return String(localized: "映像への変換")
        case .encode: return String(localized: "動画ファイルの書き出し")
        }
    }

    static func stage(forPhase phase: String) -> JobStage? {
        switch phase {
        case "denoise", "denoise enqueue":
            return .denoise
        case "audio VAE", "video VAE load":
            return .decode
        case "encode":
            return .encode
        case "AdaLN curve", "tokenizer", "text encoder", "refine text", "precompute AdaLN",
             "load transformer core",
             "audio VAE encoder", "video VAE encoder":
            return .setup
        default:
            return nil
        }
    }
}

private let phaseLabels: [String: String] = [
    "AdaLN curve": String(localized: "計算の準備（初回のみ）"),
    "tokenizer": String(localized: "文章を解析"),
    "text encoder": String(localized: "文章を理解"),
    "refine text": String(localized: "文章を調整"),
    "precompute AdaLN": String(localized: "条件を計算"),
    "load transformer core": String(localized: "モデルを読み込み"),
    "audio VAE encoder": String(localized: "音声を解析"),
    "video VAE encoder": String(localized: "画像を解析"),
]

struct TimingSample: Codable {
    var units: Double
    var seconds: Double
}

// Measured timings from previous jobs, kept across launches so the first
// estimate of a new session isn't blind. Each list is the last few samples.
struct TimingCalibration: Codable {
    // The initial values are timings measured on this project's dev machine
    // (M5, 24GB, int8 cache) from real runs - 22 frames @256x256, 124 frames
    // @512x512, and 39 frames (the app's 1 s) at 512x512, 960x544 and
    // 768x768 (SPEEDUP_ROADMAP.md section 8) - so the very first job has an
    // estimate at the larger sizes too. Real jobs then append their own
    // samples and the seeds age out after `keep` runs; live measurement
    // inside a job overrides them as soon as a step / chunk completes.
    var setup: [Double] = [17.2]             // seconds: start -> first denoise event
    var denoisePerStep: [TimingSample] = [   // units = frames x DiT pixels, seconds = per evaluated step
        TimingSample(units: 22.0 * 256 * 256, seconds: 3.64),
        TimingSample(units: 39.0 * 512 * 512, seconds: 6.96),
        TimingSample(units: 39.0 * 960 * 544, seconds: 14.8),
        TimingSample(units: 39.0 * 768 * 768, seconds: 17.1),
        TimingSample(units: 124.0 * 512 * 512, seconds: 31.3),
    ]
    // The 124-frame 512x512 decode was 112 s before the int8 video VAE;
    // 38.4 s since.
    var decode: [TimingSample] = [           // units = frames x DiT pixels, seconds = audio+video VAE decode
        TimingSample(units: 22.0 * 256 * 256, seconds: 5.5),
        TimingSample(units: 39.0 * 512 * 512, seconds: 14.1),
        TimingSample(units: 39.0 * 960 * 544, seconds: 27.0),
        TimingSample(units: 39.0 * 768 * 768, seconds: 29.3),
        TimingSample(units: 124.0 * 512 * 512, seconds: 38.4),
    ]
    var encode: [TimingSample] = [           // units = frames x DiT pixels, seconds = encode stage
        TimingSample(units: 22.0 * 256 * 256, seconds: 0.11),
        TimingSample(units: 124.0 * 512 * 512, seconds: 0.27),
    ]

    // One calibration per compute mode: SSD streaming reads the whole BF16
    // checkpoint every step, so its setup and per-step times have nothing
    // in common with the int8-cache path's. (v2: stage model changed.)
    // Faster speed modes also get their own history - their per-step times
    // would otherwise drag the standard mode's estimates down. The standard
    // mode keeps the original key so existing calibration carries over.
    // Fast mode (ccv attention) likewise keeps separate history.
    // So does a reduced layer count.
    private static func defaultsKey(_ mode: ComputeMode, _ speed: SpeedMode,
                                    _ fastAttention: Bool, _ ditLayers: Int) -> String {
        let base = mode == .attentionCache ? "h3c-app.timingCalibration.v2"
                                           : "h3c-app.timingCalibration.v2.\(mode.rawValue)"
        var key = speed == .quality ? base : base + ".speed.\(speed.rawValue)"
        if fastAttention { key += ".fastAttention" }
        if ditLayers != 50 { key += ".layers\(ditLayers)" }
        return key
    }
    private static let keep = 12

    /// Fresh calibration for a mode, seeded with timings measured on the dev
    /// machine (same two sizes as the cache seeds). Setup is the same in both
    /// modes; only the denoise step differs - SSD streaming re-reads the BF16
    /// checkpoint each step: 6.24s (22 frames @256x256) and 43.0s (124
    /// frames @512x512) vs 3.64s / 31.3s with the int8 cache. VAE decode and
    /// encode are the same code either way, so they share the seeds.
    static func initial(for mode: ComputeMode) -> TimingCalibration {
        var value = TimingCalibration()
        if mode == .ssdStreaming {
            value.denoisePerStep = [
                TimingSample(units: 22.0 * 256 * 256, seconds: 6.24),
                TimingSample(units: 124.0 * 512 * 512, seconds: 43.0),
            ]
        }
        return value
    }

    static func load(for mode: ComputeMode, speed: SpeedMode = .quality,
                     fastAttention: Bool = false, ditLayers: Int = 50) -> TimingCalibration {
        guard let data = UserDefaults.standard.data(forKey: defaultsKey(mode, speed, fastAttention, ditLayers)),
              let value = try? JSONDecoder().decode(TimingCalibration.self, from: data) else {
            return initial(for: mode)
        }
        return value
    }

    func save(for mode: ComputeMode, speed: SpeedMode = .quality, fastAttention: Bool = false,
              ditLayers: Int = 50) {
        if let data = try? JSONEncoder().encode(self) {
            UserDefaults.standard.set(data, forKey: Self.defaultsKey(mode, speed, fastAttention, ditLayers))
        }
    }

    mutating func trim() {
        setup = Array(setup.suffix(Self.keep))
        denoisePerStep = Array(denoisePerStep.suffix(Self.keep))
        decode = Array(decode.suffix(Self.keep))
        encode = Array(encode.suffix(Self.keep))
    }
}

private func median(_ values: [Double]) -> Double? {
    guard !values.isEmpty else { return nil }
    let sorted = values.sorted()
    let mid = sorted.count / 2
    return sorted.count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

// Least-squares seconds = a + b * units. A fixed cost matters for denoise:
// with the streamed int8 cache every step re-reads the weights from disk, so
// per-step time depends only weakly on image size. With <2 distinct sizes
// seen (or a nonsensical fit) it falls back to the mean instead of
// extrapolating.
private func predictAffine(_ samples: [TimingSample], units: Double) -> Double? {
    guard !samples.isEmpty else { return nil }
    let n = Double(samples.count)
    let meanX = samples.reduce(0) { $0 + $1.units } / n
    let meanY = samples.reduce(0) { $0 + $1.seconds } / n
    let sxx = samples.reduce(0) { $0 + ($1.units - meanX) * ($1.units - meanX) }
    // All samples one size (the seeds aged out after many runs of the same
    // setting): no slope to fit, so scale with size rather than predict
    // that size's time for every other - a 256x256 clip after a run of
    // 672x384 ones was otherwise estimated about 6x too long.
    guard sxx > 1e-9 * max(meanX * meanX, 1) else { return meanX > 0 ? meanY / meanX * units : meanY }
    let sxy = samples.reduce(0) { $0 + ($1.units - meanX) * ($1.seconds - meanY) }
    let slope = sxy / sxx
    guard slope >= 0 else { return meanY }
    let value = meanY + slope * (units - meanX)
    return value > 0 ? value : meanY
}

// VAE decode and encode do work proportional to frames x pixels, so there's
// no meaningful fixed cost to fit. Scale the seconds-per-unit rate of the
// earlier job closest in size (ties: most recent) - a single job of a
// different size stays useful, and an old outlier can't drag a repeat of a
// size we've already measured.
private func predictProportional(_ samples: [TimingSample], units: Double) -> Double? {
    guard units > 0 else { return nil }
    var best: TimingSample?
    var bestDistance = Double.infinity
    for sample in samples.reversed() where sample.units > 0 {
        let distance = abs(log(units / sample.units))
        if distance < bestDistance { best = sample; bestDistance = distance }
    }
    guard let best else { return nil }
    return best.seconds / best.units * units
}

/// Mirrors h3_dit_reuse_schedule() in h3_dit.c: with denoise reuse N the model
/// only runs on step 0, the last step, and every Nth step in between; the
/// other steps just extrapolate from earlier results and cost ~nothing. So
/// denoise time is (evaluated steps) x (time per evaluated step), not
/// steps x time per step.
func isEvaluatedStep(_ step: Int, steps: Int, reuse: Int) -> Bool {
    reuse <= 1 || step == 0 || step == steps - 1 || step % reuse == 0
}

func evaluatedStepCount(steps: Int, reuse: Int, upTo limit: Int? = nil) -> Int {
    let end = min(limit ?? steps, steps)
    guard end > 0 else { return 0 }
    return (0 ..< end).filter { isEvaluatedStep($0, steps: steps, reuse: reuse) }.count
}

/// Turns the engine's raw phase events into a stage-aware elapsed/remaining
/// estimate and a progress bar value. Everything it reports comes from real
/// events or from previously *measured* timings - it never invents a stage.
final class ProgressEstimator {
    struct Shape {
        let steps: Int
        let reuse: Int           // denoise_reuse: only every Nth step runs the model
        let totalFrames: Int
        let ditUnits: Double     // frames x pixels the DiT actually runs at
        let decodeUnits: Double  // frames x output pixels
    }

    struct Snapshot {
        var elapsed: Double
        var remaining: Double?
        var fraction: Double?
        var stageTitle: String
        var detail: String
    }

    private(set) var stage: JobStage = .setup
    private let shape: Shape
    private let calibration: TimingCalibration
    private let start: Date

    private var currentPhase = ""
    private var currentCompleted = 0
    private var currentTotal = 0

    private var denoiseStart: Date?
    private var lastDenoiseEventAt: Date?
    private var denoiseDone = 0

    private var decodeStart: Date?
    private var decodeEnd: Date?

    private var encodeStart: Date?
    private var lastEncodeEmitAt: Date?
    private var encodeEmitted = 0

    private var displayedFraction = 0.0

    init(shape: Shape, calibration: TimingCalibration, start: Date) {
        self.shape = shape
        self.calibration = calibration
        self.start = start
    }

    func handle(phase: String, completed: Int, total: Int, now: Date) {
        currentPhase = phase
        currentCompleted = completed
        currentTotal = total
        guard let newStage = JobStage.stage(forPhase: phase) else { return }
        if newStage > stage { stage = newStage }
        switch newStage {
        case .setup:
            break
        case .denoise:
            if denoiseStart == nil { denoiseStart = now }
            // Both "denoise enqueue k/N" (the pipelined GPU Euler path - one
            // event per finished step, ~real time) and "denoise k/N" (other
            // paths) count finished steps.
            if completed > denoiseDone {
                denoiseDone = completed
                lastDenoiseEventAt = now
            }
        case .decode:
            if decodeStart == nil { decodeStart = now }
        case .encode:
            if decodeStart != nil { decodeEnd = decodeEnd ?? now }
            if encodeStart == nil { encodeStart = now }
            if completed > encodeEmitted {
                encodeEmitted = completed
                lastEncodeEmitAt = now
            }
        }
    }

    func snapshot(now: Date) -> Snapshot {
        let elapsed = now.timeIntervalSince(start)
        var remaining = 0.0
        var known = true

        // Setup: only a prior is available (its sub-phases don't add up to a
        // fraction), counted down by this job's own elapsed time.
        if stage == .setup {
            if let typical = median(calibration.setup) {
                remaining += max(typical - elapsed, 2)
            } else {
                known = false
            }
        }

        // Denoise: time per *evaluated* step measured live from this job's
        // own steps as soon as one has finished; before that, predicted from
        // earlier jobs (which are recorded per evaluated step too, so jobs
        // with different reuse settings stay comparable).
        let totalEvals = Double(evaluatedStepCount(steps: shape.steps, reuse: shape.reuse))
        switch stage {
        case .setup:
            if let perEval = predictAffine(calibration.denoisePerStep, units: shape.ditUnits) {
                remaining += perEval * totalEvals
            } else {
                known = false
            }
        case .denoise:
            let evalsDone = Double(evaluatedStepCount(steps: shape.steps, reuse: shape.reuse, upTo: denoiseDone))
            if denoiseDone >= 1, evalsDone >= 1, let first = denoiseStart, let last = lastDenoiseEventAt {
                let perEval = last.timeIntervalSince(first) / evalsDone
                let left = perEval * (totalEvals - evalsDone) - now.timeIntervalSince(last)
                remaining += max(left, 0.5)
            } else if let perEval = predictAffine(calibration.denoisePerStep, units: shape.ditUnits),
                      let first = denoiseStart {
                remaining += max(perEval * totalEvals - now.timeIntervalSince(first), perEval)
            } else {
                known = false
            }
        case .decode, .encode:
            break
        }

        // Decode: no usable live counter (see header), so predicted from
        // frames x output size and counted down by time in the stage.
        let decodeUnits = shape.decodeUnits
        switch stage {
        case .setup, .denoise:
            if let seconds = predictProportional(calibration.decode, units: decodeUnits) {
                remaining += seconds
            } else {
                known = false
            }
        case .decode:
            if let began = decodeStart,
               let seconds = predictProportional(calibration.decode, units: decodeUnits) {
                remaining += max(seconds - now.timeIntervalSince(began), 0.5)
            } else {
                known = false
            }
        case .encode:
            break
        }

        // Encode: predicted from frames x output size, replaced by
        // the live rate if the engine reports per-chunk progress.
        switch stage {
        case .setup, .denoise, .decode:
            if let seconds = predictProportional(calibration.encode, units: decodeUnits) {
                remaining += seconds
            } else {
                known = false
            }
        case .encode:
            let total = Double(shape.totalFrames)
            if encodeEmitted > 0, encodeEmitted < shape.totalFrames,
               let began = encodeStart, let last = lastEncodeEmitAt {
                let perFrame = last.timeIntervalSince(began) / Double(encodeEmitted)
                let left = perFrame * (total - Double(encodeEmitted)) - now.timeIntervalSince(last)
                remaining += max(left, 0.3)
            } else if let began = encodeStart,
                      let seconds = predictProportional(calibration.encode, units: decodeUnits) {
                remaining += max(seconds - now.timeIntervalSince(began), 0.3)
            } else {
                remaining += 0.3
            }
        }

        let fraction: Double?
        if known {
            // Time-weighted overall progress, kept from ever moving backwards
            // (a revised estimate can only slow it, not rewind it). A
            // transient underestimate must not pin it near 100% while real
            // work remains, so leave headroom until the final stages.
            let ceiling: Double
            switch stage {
            case .setup, .denoise: ceiling = 0.85
            case .decode: ceiling = 0.95
            case .encode: ceiling = 0.99
            }
            let raw = min(elapsed / (elapsed + remaining), ceiling)
            displayedFraction = max(displayedFraction, raw)
            fraction = displayedFraction
        } else {
            fraction = stageLocalFraction(now: now)
        }

        return Snapshot(
            elapsed: elapsed,
            remaining: known ? remaining : nil,
            fraction: fraction,
            stageTitle: stage.title,
            detail: detailText()
        )
    }

    // Used only while there's no calibration yet (first ever run): real
    // per-stage progress, smoothed between events so the bar keeps moving.
    private func stageLocalFraction(now: Date) -> Double? {
        switch stage {
        case .denoise:
            var done = Double(denoiseDone)
            if denoiseDone >= 1, let first = denoiseStart, let last = lastDenoiseEventAt {
                let perStep = last.timeIntervalSince(first) / Double(denoiseDone)
                if perStep > 0 { done += min(0.95, now.timeIntervalSince(last) / perStep) }
            }
            return min(done / Double(max(shape.steps, 1)), 0.99)
        case .decode:
            return nil
        case .encode:
            return min(Double(encodeEmitted) / Double(max(shape.totalFrames, 1)), 0.99)
        case .setup:
            return nil
        }
    }

    private func detailText() -> String {
        switch stage {
        case .setup:
            let label = phaseLabels[currentPhase] ?? currentPhase
            return currentTotal > 1 ? "\(label) \(currentCompleted)/\(currentTotal)" : label
        case .denoise:
            return String(localized: "\(denoiseDone)/\(shape.steps) ステップ")
        case .decode:
            return String(localized: "音声と映像をデコードしています")
        case .encode:
            return String(localized: "\(encodeEmitted)/\(shape.totalFrames) フレーム")
        }
    }

    /// Calibration including this finished job's measured timings.
    func finishedCalibration(now: Date) -> TimingCalibration {
        var updated = calibration
        if let first = denoiseStart {
            updated.setup.append(first.timeIntervalSince(start))
        }
        // Denoise ends where the decode stage begins; steps are counted
        // from "denoise enqueue"/"denoise" events, so only record a full run.
        if let first = denoiseStart, denoiseDone >= shape.steps, shape.steps > 0,
           let end = decodeStart ?? encodeStart {
            let evals = max(evaluatedStepCount(steps: shape.steps, reuse: shape.reuse), 1)
            updated.denoisePerStep.append(TimingSample(
                units: shape.ditUnits,
                seconds: end.timeIntervalSince(first) / Double(evals)))
        }
        if let began = decodeStart, let end = decodeEnd ?? encodeStart {
            updated.decode.append(TimingSample(units: shape.decodeUnits, seconds: end.timeIntervalSince(began)))
        }
        if let began = encodeStart {
            updated.encode.append(TimingSample(units: shape.decodeUnits, seconds: now.timeIntervalSince(began)))
        }
        updated.trim()
        return updated
    }
}
