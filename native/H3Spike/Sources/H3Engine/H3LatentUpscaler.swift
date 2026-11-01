import CH3
import Foundation

// The pieces of the Open-H3 regeneration that are not a generation: the
// clip latent file's shape, LBH's 3D-conv latent upscaler
// (h3_latent_upscale.h), and the rank-range extraction of a merged LoRA
// (h3_lora_extract.h). UPSCALE_PLAN appendix A.4/A.5.

/// What a clip latent file (.h3lt, next to every generated video) holds.
public struct H3LatentFileInfo: Sendable, Equatable {
    /// The video latent's grid, in latent cells (16 pixels each).
    public let latentFrames: Int
    public let latentHeight: Int
    public let latentWidth: Int
    /// The clip's frame count.
    public let frames: Int

    public var pixelWidth: Int { latentWidth * 16 }
    public var pixelHeight: Int { latentHeight * 16 }

    public static func read(path: String) throws -> H3LatentFileInfo {
        var t: Int32 = 0, h: Int32 = 0, w: Int32 = 0, frames: Int32 = 0
        var error = [CChar](repeating: 0, count: 512)
        let ok = path.withCString { pathC in
            error.withUnsafeMutableBufferPointer { buffer in
                h3_latent_file_info(pathC, &t, &h, &w, &frames, buffer.baseAddress, buffer.count)
            }
        }
        guard ok != 0 else { throw H3VSRError.failed(String(cString: error)) }
        return H3LatentFileInfo(latentFrames: Int(t), latentHeight: Int(h), latentWidth: Int(w), frames: Int(frames))
    }
}

private final class LatentProgressBridge: @unchecked Sendable {
    let progress: (Int, Int) -> Bool   // returns true to go on
    init(progress: @escaping (Int, Int) -> Bool) { self.progress = progress }
}

private let h3LatentProgressTrampoline: h3_progress_callback = { _, completed, total, opaque in
    guard let opaque else { return 0 }
    let bridge = Unmanaged<LatentProgressBridge>.fromOpaque(opaque).takeUnretainedValue()
    return bridge.progress(Int(completed), Int(total)) ? 0 : 1
}

/// LBH-123-AI's 3D-convolution latent upscaler: a clip latent onto a larger
/// spatial grid, the first step of the Open-H3 regeneration.
public final class H3LatentUpscaler: @unchecked Sendable {
    private let handle: OpaquePointer

    /// `weightsDirectory` holds the one BF16 safetensors file.
    public init(weightsDirectory: String, shaderPath: String = "h3_shaders.metal") throws {
        var error = [CChar](repeating: 0, count: 1024)
        let handle = weightsDirectory.withCString { dir in
            shaderPath.withCString { shader in
                error.withUnsafeMutableBufferPointer { buffer in
                    h3_latent_upscaler_open(dir, shader, buffer.baseAddress, buffer.count)
                }
            }
        }
        guard let handle else { throw H3VSRError.failed(String(cString: error)) }
        self.handle = handle
    }

    deinit { h3_latent_upscaler_free(handle) }

    /// `input` (.h3lt) onto a `latentHeight` x `latentWidth` grid, written
    /// to `output` with the audio latent and frame count copied. `progress`
    /// gets (done, total) and returns false to stop (the call then throws
    /// H3VSRError.cancelled).
    public func upscaleFile(_ input: String, to output: String, latentHeight: Int, latentWidth: Int,
                            progress: @escaping (Int, Int) -> Bool) throws {
        let bridge = LatentProgressBridge(progress: progress)
        let opaque = Unmanaged.passRetained(bridge)
        defer { opaque.release() }
        var error = [CChar](repeating: 0, count: 1024)
        let ok = input.withCString { inC in
            output.withCString { outC in
                error.withUnsafeMutableBufferPointer { buffer in
                    h3_latent_upscale_file(handle, inC, outC, Int32(latentHeight), Int32(latentWidth),
                                           h3LatentProgressTrampoline, opaque.toOpaque(),
                                           buffer.baseAddress, buffer.count)
                }
            }
        }
        guard ok != 0 else {
            let message = String(cString: error)
            throw message.contains("cancel") ? H3VSRError.cancelled : H3VSRError.failed(message)
        }
    }
}

/// A rank range of a merged LoRA into its own file (lora_A rows and lora_B
/// columns cut, other tensors copied), BF16: how the Open-H3 refiner is
/// taken out of NVIDIA's merged file on this Mac.
public enum H3LoRAExtract {
    public static func extractRanks(from input: String, to output: String, ranks: Range<Int>, bf16: Bool = true,
                                    note: String? = nil, progress: @escaping (Int, Int) -> Bool) throws {
        let bridge = LatentProgressBridge(progress: progress)
        let opaque = Unmanaged.passRetained(bridge)
        defer { opaque.release() }
        var error = [CChar](repeating: 0, count: 1024)
        let ok = input.withCString { inC in
            output.withCString { outC in
                (note ?? "").withCString { noteC in
                    error.withUnsafeMutableBufferPointer { buffer in
                        h3_lora_extract_ranks(inC, outC, Int32(ranks.lowerBound), Int32(ranks.upperBound), bf16 ? 1 : 0,
                                              noteC, h3LatentProgressTrampoline, opaque.toOpaque(),
                                              buffer.baseAddress, buffer.count)
                    }
                }
            }
        }
        guard ok != 0 else {
            let message = String(cString: error)
            throw message.contains("cancel") ? H3VSRError.cancelled : H3VSRError.failed(message)
        }
    }
}
