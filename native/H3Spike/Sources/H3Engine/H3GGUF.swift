import CH3
import Foundation

/// What a GGUF file holds, as the importer sorts files into a model's parts
/// (h3_gguf_inspect in h3.h - reads only the tensor table). A ".safetensors"
/// path is read too when it is a ComfyUI int8 DiT; any other safetensors
/// file throws.
public struct H3GGUFInfo: Sendable, Equatable {
    public enum Role: Sendable { case unknown, dit, textEncoder, vision }
    public let role: Role
    /// DiT: carries its own curve-form AdaLN (ComfyUI-derived, e.g. unsloth).
    public let curveAdaLN: Bool
    public let denseAdaLN: Bool
    /// Text encoder: language layers in the file (Qwen3-VL-32B has 64; the
    /// engine reads the first 50).
    public let textLayers: Int
    /// Text encoder: the vision tower is in the same file.
    public let hasVision: Bool
    public let fileBytes: UInt64
    /// Format of most of the weight bytes, e.g. "Q8_0".
    public let mainType: String
    public let architecture: String
    public let name: String
    /// DiT: HyperFlow H3 Flash (a distilled 4/6/8-step H3 derivative with
    /// factored linears and pruned blocks), its block count and how many
    /// blocks have factored linears.
    public var hyperflow = false
    public var ditBlocks = 0
    public var factoredBlocks = 0
    /// DiT in a single ComfyUI safetensors file with int8 linears (the one
    /// safetensors layout `inspect` reads); `convRot` when they are also
    /// rotated (mainType "INT8 ConvRot").
    public var comfyInt8 = false
    public var convRot = false
    /// DiT: the file says it is a merge serving as both FL2VA and Ref2VA.
    public var bothTransformers = false

    public static func inspect(path: String) throws -> H3GGUFInfo {
        var info = h3_gguf_info()
        var errorBuffer = [CChar](repeating: 0, count: 512)
        let ok = errorBuffer.withUnsafeMutableBufferPointer { errorBuf in
            h3_gguf_inspect(path, &info, errorBuf.baseAddress, errorBuf.count)
        }
        guard ok != 0 else { throw H3EngineError.loadFailed(String(cString: errorBuffer)) }
        let role: Role
        switch info.role {
        case H3_GGUF_DIT: role = .dit
        case H3_GGUF_TEXT_ENCODER: role = .textEncoder
        case H3_GGUF_VISION: role = .vision
        default: role = .unknown
        }
        func text<T>(_ field: T) -> String {
            withUnsafeBytes(of: field) { raw in
                String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
            }
        }
        var result = H3GGUFInfo(
            role: role, curveAdaLN: info.curve_adaln != 0, denseAdaLN: info.dense_adaln != 0,
            textLayers: Int(info.text_layers), hasVision: info.has_vision != 0,
            fileBytes: info.file_bytes, mainType: text(info.main_type),
            architecture: text(info.architecture), name: text(info.name))
        result.hyperflow = info.hyperflow != 0
        result.ditBlocks = Int(info.dit_blocks)
        result.factoredBlocks = Int(info.factored_blocks)
        result.comfyInt8 = info.comfy_int8 != 0
        result.convRot = info.comfy_int8 == 2
        result.bothTransformers = info.both_transformers != 0
        return result
    }
}

/// Which kind of transformer a model folder runs. HyperFlow H3 Flash only
/// samples in 4, 6 or 8 steps with its own schedule, and takes none of the
/// H3 speed-ups, LoRAs, the re-noise sampler or the int8 attention cache
/// (h3.c refuses them).
public enum H3ModelFamily: String, Codable, Sendable {
    case h3
    case hyperflow

    /// What a family can be asked for (the engine's H3_CAN_* flags).
    public struct Capabilities: OptionSet, Sendable {
        public let rawValue: UInt32
        public init(rawValue: UInt32) { self.rawValue = rawValue }
        public static let lora = Capabilities(rawValue: UInt32(H3_CAN_LORA))
        public static let renoise = Capabilities(rawValue: UInt32(H3_CAN_RENOISE))
        /// Denoiser reuse, core reuse, token reduction and the speed presets built on them.
        public static let stepReuse = Capabilities(rawValue: UInt32(H3_CAN_STEP_REUSE))
        public static let layerSkip = Capabilities(rawValue: UInt32(H3_CAN_LAYER_SKIP))
        public static let customSigmas = Capabilities(rawValue: UInt32(H3_CAN_CUSTOM_SIGMAS))
        public static let audioShift = Capabilities(rawValue: UInt32(H3_CAN_AUDIO_SHIFT))
        public static let attentionCache = Capabilities(rawValue: UInt32(H3_CAN_ATTENTION_CACHE))
        public static let adalnCurve = Capabilities(rawValue: UInt32(H3_CAN_ADALN_CURVE))
    }

    /// This family's row in the engine's table (h3_model_family), found by
    /// name: the app asks it what applies instead of keeping a copy.
    private var info: h3_model_family_info {
        for id in 0 ..< h3_model_family_count() {
            if let row = h3_model_family(id), String(cString: row.pointee.name) == rawValue { return row.pointee }
        }
        preconditionFailure("the engine has no model family named \(rawValue)")
    }

    public var label: String { String(cString: info.label) }

    /// The step counts it samples in; empty when it takes any.
    public var stepChoices: [Int] {
        let row = info
        return withUnsafeBytes(of: row.step_choices) { bytes in
            Array(bytes.bindMemory(to: Int32.self).prefix(Int(row.step_choice_count))).map(Int.init)
        }
    }

    /// `stepChoices` as words: "4, 6 or 8".
    public var stepChoicesText: String {
        let numbers = stepChoices.map(String.init)
        guard let last = numbers.last, numbers.count > 1 else { return numbers.first ?? "" }
        return numbers.dropLast().joined(separator: ", ") + " or " + last
    }

    public var defaultSteps: Int { Int(info.default_steps) }

    public func can(_ capability: Capabilities) -> Bool {
        Capabilities(rawValue: UInt32(info.capabilities)).contains(capability)
    }

    /// From the FL2VA transformer's header: a GGUF, or a single ComfyUI
    /// int8 safetensors file (HyperFlow ships both; the released shards of
    /// an official model are not inspected and are plain H3). Reads only
    /// the tensor table.
    public static func detect(modelDirectory: String) -> H3ModelFamily {
        let transformer = (modelDirectory as NSString)
            .appendingPathComponent("FL2VA/transformer")
        let files = ((try? FileManager.default.contentsOfDirectory(atPath: transformer)) ?? []).sorted()
        let single = files.filter { $0.lowercased().hasSuffix(".safetensors") }
        let candidates = files.filter { $0.lowercased().hasSuffix(".gguf") } + (single.count == 1 ? single : [])
        for file in candidates {
            let path = (transformer as NSString).appendingPathComponent(file)
            if let info = try? H3GGUFInfo.inspect(path: path), info.role == .dit {
                return info.hyperflow ? .hyperflow : .h3
            }
        }
        return .h3
    }
}

/// Holds the progress closure and cancel flag for h3_gguf_trim_text_encoder's
/// C callback.
private final class TrimBridge: @unchecked Sendable {
    let progress: @Sendable (Double) -> Void
    let isCancelled: @Sendable () -> Bool
    init(progress: @escaping @Sendable (Double) -> Void,
         isCancelled: @escaping @Sendable () -> Bool) {
        self.progress = progress
        self.isCancelled = isCancelled
    }
}

private let h3TrimProgressTrampoline: h3_progress_callback = { _, completed, total, opaque in
    guard let opaque else { return 0 }
    let bridge = Unmanaged<TrimBridge>.fromOpaque(opaque).takeUnretainedValue()
    bridge.progress(total > 0 ? Double(completed) / Double(total) : 0)
    return bridge.isCancelled() ? 1 : 0
}

public enum H3GGUFTools {
    /// Copies the tensors named `prefix`* out of a component directory into
    /// one safetensors file (h3_extract_tensors) - e.g. the vision tower
    /// ("model.visual.") of a released text encoder - off the caller's thread.
    public static func extractTensors(componentDirectory: String, prefix: String, output: String,
                                      progress: @escaping @Sendable (Double) -> Void,
                                      isCancelled: @escaping @Sendable () -> Bool) async throws {
        try await Task.detached(priority: .userInitiated) {
            let bridge = TrimBridge(progress: progress, isCancelled: isCancelled)
            let handle = Unmanaged.passRetained(bridge)
            defer { handle.release() }
            var errorBuffer = [CChar](repeating: 0, count: 512)
            let ok = errorBuffer.withUnsafeMutableBufferPointer { errorBuf in
                h3_extract_tensors(componentDirectory, prefix, output,
                                   h3TrimProgressTrampoline, handle.toOpaque(),
                                   errorBuf.baseAddress, errorBuf.count)
            }
            if ok == 0 {
                if isCancelled() { throw CancellationError() }
                throw H3EngineError.loadFailed(String(cString: errorBuffer))
            }
        }.value
    }

    /// Writes a copy of a text-encoder GGUF holding only its first `layers`
    /// language layers (and no output head), off the caller's thread.
    public static func trimTextEncoder(source: String, output: String, layers: Int = 50,
                                       progress: @escaping @Sendable (Double) -> Void,
                                       isCancelled: @escaping @Sendable () -> Bool) async throws {
        try await Task.detached(priority: .userInitiated) {
            let bridge = TrimBridge(progress: progress, isCancelled: isCancelled)
            let handle = Unmanaged.passRetained(bridge)
            defer { handle.release() }
            var errorBuffer = [CChar](repeating: 0, count: 512)
            let ok = errorBuffer.withUnsafeMutableBufferPointer { errorBuf in
                h3_gguf_trim_text_encoder(source, output, Int32(layers),
                                          h3TrimProgressTrampoline, handle.toOpaque(),
                                          errorBuf.baseAddress, errorBuf.count)
            }
            if ok == 0 {
                if isCancelled() { throw CancellationError() }
                throw H3EngineError.loadFailed(String(cString: errorBuffer))
            }
        }.value
    }
}

/// The engine's compute modes (h3_compute_mode), by the names the app
/// records ("attentionCache", "resident", "ssdStreaming").
public enum H3ComputeModes {
    private static func info(named name: String) -> h3_compute_mode_info {
        for id in 0 ..< h3_compute_mode_count() {
            if let row = h3_compute_mode(id), String(cString: row.pointee.name) == name { return row.pointee }
        }
        preconditionFailure("the engine has no compute mode named \(name)")
    }

    public static func needsTensorOps(named name: String) -> Bool { info(named: name).needs_tensor_ops != 0 }
    public static func needsCacheFile(named name: String) -> Bool { info(named: name).needs_cache_file != 0 }
    public static func streamsBlocks(named name: String) -> Bool { info(named: name).streams_blocks != 0 }
}
