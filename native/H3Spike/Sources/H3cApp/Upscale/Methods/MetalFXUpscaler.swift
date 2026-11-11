import CoreVideo
import Foundation
import Metal
import MetalFX

/// VideoUpscaler.metalFX: MetalFX's spatial scaler.
struct MetalFXUpscaler: UpscalerMethod {
    let id = VideoUpscaler.metalFX
    var label: String { String(localized: "MetalFX（輪郭を保つ拡大）") }
    var detail: String {
        String(localized: "AppleのMetalFXで、輪郭を見ながら拡大し、少しくっきりさせます。GPUで処理します。AIによる高画質化ではありません。")
    }
    var isAvailable: Bool {
        guard let device = MTLCreateSystemDefaultDevice() else { return false }
        return MTLFXSpatialScalerDescriptor.supportsDevice(device)
    }
    var factorFloor: Double { 1.1 }
    var factorCeiling: Double { UpscaleFactor.maximum }
    func makeScaler(_ job: UpscaleJob) throws -> ChunkedFrameScaler {
        guard let made = MetalFXScaler(inputWidth: job.inputWidth, inputHeight: job.inputHeight,
                                       outputWidth: job.width, outputHeight: job.height) else {
            throw VideoUpscale.UpscaleError.unavailable(label)
        }
        return PerFrameChunkAdapter(made)
    }
}

/// VideoUpscaler.metalFX: MetalFX's spatial scaler on the GPU, reading and
/// writing the frames' own memory (CVMetalTextureCache). The scaler writes
/// a private texture, copied into the output frame.
final class MetalFXScaler: FrameScaler {
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
