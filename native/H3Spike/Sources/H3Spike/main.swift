// Minimal spike: link libh3.a directly into a Swift executable and drive
// h3_load_dir -> h3_generate in-process, receiving progress/frames through
// the C callbacks instead of parsing subprocess stdout (as gui/server.py
// and the CLI's terminal front end currently do).
import CH3
import Foundation

/* h3_device_info's `name`/`architecture` fields are fixed-size C char arrays,
 * imported into Swift as anonymous tuples. Reading them as raw bytes avoids
 * having to spell out the tuple's exact arity. */
func fixedCString<T>(_ tuple: T) -> String {
    withUnsafeBytes(of: tuple) { raw in
        String(cString: raw.bindMemory(to: CChar.self).baseAddress!)
    }
}

final class GenerationContext {
    var framesReceived = 0
    var previewsReceived = 0
    // H3SPIKE_DUMP=<path>: append every final frame as raw packed RGB24, so
    // two runs (e.g. H3_VAE_INT8=0 vs default) can be compared numerically
    // without the lossy mp4 encode in between.
    var dump: FileHandle?
}

let traceStart = Date()
setvbuf(stdout, nil, _IOLBF, 0)

let progressCallback: h3_progress_callback = { phase, completed, total, _ in
    let phaseName = phase.map { String(cString: $0) } ?? "?"
    print(String(format: "[%7.2fs] [progress] ", Date().timeIntervalSince(traceStart)) + "\(phaseName) \(completed)/\(total)")
    return 0 // return non-zero from here to cancel generation
}

let frameCallback: h3_frame_callback = { framePtr, opaque in
    guard let framePtr, let opaque else { return 0 }
    let frame = framePtr.pointee
    let context = Unmanaged<GenerationContext>.fromOpaque(opaque).takeUnretainedValue()
    if frame.denoise_step >= 0 {
        context.previewsReceived += 1
        print("[preview] denoise step \(frame.denoise_step + 1)/\(frame.denoise_steps)")
    } else {
        context.framesReceived += 1
        print(String(format: "[%7.2fs] [frame] ", Date().timeIntervalSince(traceStart)) + "\(frame.frame_index + 1)/\(frame.frame_count)")
        if let dump = context.dump, let rgb = frame.rgb {
            let rowBytes = Int(frame.width) * 3
            for row in 0 ..< Int(frame.height) {
                dump.write(Data(bytes: rgb + row * Int(frame.stride), count: rowBytes))
            }
        }
    }
    return 0
}

let arguments = CommandLine.arguments
// This spike target isn't part of the distributed .app (see Package.swift),
// so its fallbacks just point at this dev machine's model and cache under
// ~/models - the same pinned location H3cApp's ModelLibrary recognizes.
let modelDir = arguments.count > 1 ? arguments[1]
    : (NSHomeDirectory() + "/models/MiniMax-H3")
let outputPath = arguments.count > 2 ? arguments[2] : "/tmp/h3spike_output.mp4"
let prompt = arguments.count > 3 ? arguments[3] : "A cat playing with a ball of yarn."

print("== H3cApp native spike ==")

// Reproduces a real, unexplained failure: this exact call (creates its own
// GPU, no model needed) succeeds from a plain C/Objective-C++ process but
// consistently fails with a garbled Metal shader compile error when made
// from this Swift binary - see SPEEDUP_ROADMAP.md item 5's "Swift runtime
// blocks the live H3_ATTENTION_BACKEND=ccv_dense path" note. Kept here as
// a minimal, fast repro for whoever investigates this further; not part
// of normal operation.
if ProcessInfo.processInfo.environment["H3_CCV_WARMUP_DIAG"] != nil {
    var err = [CChar](repeating: 0, count: 4096)
    let ok = h3_debug_ccv_warmup("h3_shaders.metal", &err, err.count)
    print("H3_CCV_WARMUP_DIAG result=\(ok) err=\(String(cString: err))")
    exit(ok != 0 ? 0 : 1)
}

print("Model dir: \(modelDir)")

guard let ctx = h3_load_dir(modelDir) else {
    print("h3_load_dir returned NULL")
    exit(1)
}
defer { h3_free(ctx) }

if let device = h3_device(ctx)?.pointee {
    print("Device: \(fixedCString(device.name)) (\(fixedCString(device.architecture)))")
    print("  unified memory: \(device.unified_memory != 0), metal4: \(device.metal4 != 0)")
}

if let model = h3_model(ctx)?.pointee {
    print("FL2VA transformer: \(model.fl2va_transformer.bytes) bytes, \(model.fl2va_transformer.tensors) tensors")
}

// Point at the existing validated attention cache so this spike runs the
// same fast int8 path the app uses by default, rather than the slower
// close-reference BF16 path.
let useSSD = ProcessInfo.processInfo.environment["H3SPIKE_SSD"] == "1"
if getenv("H3_ATTENTION_CACHE") == nil && !useSSD {
    setenv("H3_ATTENTION_CACHE", NSHomeDirectory() + "/models/cache/dit_int8_v2.cache", 1)
}

let context = GenerationContext()
if let dumpPath = ProcessInfo.processInfo.environment["H3SPIKE_DUMP"] {
    FileManager.default.createFile(atPath: dumpPath, contents: nil)
    context.dump = FileHandle(forWritingAtPath: dumpPath)
}
let opaque = Unmanaged.passUnretained(context).toOpaque()

func envInt(_ name: String, _ fallback: Int32) -> Int32 {
    Int32(ProcessInfo.processInfo.environment[name] ?? "") ?? fallback
}

var params = h3_params()
let sizeArg = envInt("H3SPIKE_SIZE", 256)
params.width = sizeArg
params.height = sizeArg
params.frames = envInt("H3SPIKE_FRAMES", 9)
params.steps = envInt("H3SPIKE_STEPS", 4)
params.seed = UInt64(envInt("H3SPIKE_SEED", 42))
params.dit_layers = envInt("H3SPIKE_LAYERS", 50)
params.ssd_streaming = useSSD ? 1 : 0
params.denoise_reuse = envInt("H3SPIKE_REUSE", 1)
params.core_reuse = envInt("H3SPIKE_CORE_REUSE", 1)
params.token_reduction = envInt("H3SPIKE_TOKEN_REDUCTION", 0)
params.fast_attention = envInt("H3SPIKE_FAST_ATTENTION", 0)
print("Fast attention available: \(h3_fast_attention_available() != 0)")
params.use_int8_row_fc2 = envInt("H3SPIKE_INT8_ROW_FC2", 0)
params.reference_image_size = H3_REFERENCE_IMAGE_MATCH
params.on_progress = progressCallback
params.on_frame = frameCallback
params.callback_opaque = opaque

// H3SPIKE_LORA="path[@strength],path2[@strength]" stacks adapters.
let loraSpecs = (ProcessInfo.processInfo.environment["H3SPIKE_LORA"] ?? "")
    .split(separator: ",").map { spec -> (String, Float) in
        let parts = spec.split(separator: "@", maxSplits: 1)
        return (String(parts[0]), parts.count > 1 ? Float(parts[1]) ?? 1 : 1)
    }
let loraPaths = loraSpecs.map { strdup($0.0) }
defer { loraPaths.forEach { free($0) } }
var loras = zip(loraPaths, loraSpecs).map { h3_lora(path: $0.0, strength: $0.1.1) }
for (path, strength) in loraSpecs { print("LoRA: \(path) @ \(strength)") }

// H3SPIKE_REPEAT=<n>: run the same generation n times in this process
// (outputs <name>_run<k>.<ext> after the first), to exercise state that
// survives between generations - e.g. attention-backend scratch released
// after one DiT and reallocated by the next. H3SPIKE_CACHE=1 also enables
// the in-process model cache, so later runs reuse the prepared DiT.
let repeatCount = max(1, envInt("H3SPIKE_REPEAT", 1))
if ProcessInfo.processInfo.environment["H3SPIKE_CACHE"] == "1" {
    h3_cache_set_enabled(ctx, 1)
}
for run in 1...repeatCount {
// Without a per-run pool, Objective-C objects autoreleased inside
// h3_generate (e.g. per-run Metal buffers) live until process exit here.
autoreleasepool {
let runOutputPath = run == 1 ? outputPath : {
    let url = URL(fileURLWithPath: outputPath)
    let ext = url.pathExtension
    let base = url.deletingPathExtension().path
    return ext.isEmpty ? "\(base)_run\(run)" : "\(base)_run\(run).\(ext)"
}()
print("Generating \(params.width)x\(params.height), \(params.frames) frames, \(params.steps) steps... (run \(run)/\(repeatCount))")
let start = Date()

let result: UnsafeMutablePointer<h3_result>? = runOutputPath.withCString { outputPathC in
    prompt.withCString { promptC in
        loras.withUnsafeBufferPointer { loraBuffer in
            params.output_path = outputPathC
            params.loras = loraBuffer.baseAddress
            params.lora_count = loraBuffer.count
            return h3_generate(ctx, promptC, &params)
        }
    }
}

guard let result else {
    let error = h3_last_error(ctx).map { String(cString: $0) } ?? "unknown error"
    print("h3_generate failed: \(error)")
    exit(1)
}

let elapsed = Date().timeIntervalSince(start)
print(String(format: "Done in %.1fs", elapsed))
print("Result: \(result.pointee.frames) frames @ \(result.pointee.fps)fps, seed=\(result.pointee.seed)")
print("Attention: \(result.pointee.ccv_attention_calls) ccv calls (\(result.pointee.ccv_attention_direct_calls) direct)")
print("Frames delivered via on_frame: \(context.framesReceived), previews: \(context.previewsReceived)")
print("Output written to: \(runOutputPath)")
h3_result_free(result)
}
}
