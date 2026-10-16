import CH3
import Foundation

/// What a GGUF file holds, as the importer sorts files into a model's parts
/// (h3_gguf_inspect in h3.h - reads only the tensor table).
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
        return H3GGUFInfo(
            role: role, curveAdaLN: info.curve_adaln != 0, denseAdaLN: info.dense_adaln != 0,
            textLayers: Int(info.text_layers), hasVision: info.has_vision != 0,
            fileBytes: info.file_bytes, mainType: text(info.main_type),
            architecture: text(info.architecture), name: text(info.name))
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
