import Combine
import Foundation
import Metal
import SwiftUI

/// VideoUpscaler.openH3: H3 itself regenerating the clip from its latent
/// (OpenH3Regenerate.swift). Not a frame scaler.
struct OpenH3Upscaler: UpscalerMethod {
    let id = VideoUpscaler.openH3
    var label: String { String(localized: "Open-H3 再生成（H3で描き直し・最大1344×768、非常に遅い・試験的）") }
    var detail: String {
        String(localized: "H3自身に描き直させます。生成したときの潜在をOpen-H3の潜在アップスケーラで大きな格子に広げ、選んだ強さまでノイズを戻して、NVIDIAのOpen-H3精緻化LoRAとLightX2V Turbo v1.2で2〜3ステップだけ生成し直します。模様や質感がH3の生成そのものの質で描かれ、生成時の設定（文章・seed）を引き継ぎます。大きさはH3の上限（1344×768ピクセル）まで、このアプリで生成した動画（潜在ファイルのあるもの）だけに使えます。1344×768・5秒でM5で約12分、生成と同じメモリ（約15GB）を使います。重みは同梱せず、配布元の記録と照合してダウンロードし、精緻化LoRAはこのMacで取り出します。試験的な方式です。")
    }
    var model: (any UpscalerModel)? { OpenH3Kit.shared }
    var regeneratesFromLatent: Bool { true }
    var isAvailable: Bool { OpenH3Model.ready && MTLCreateSystemDefaultDevice() != nil }
    var missingReason: String {
        if OpenH3Model.upsamplerPresent && OpenH3Model.refinerPresent {
            return String(localized: "LightX2V Turbo v1.2のファイルがありません。「モデルを準備…」でダウンロードすると選べます。")
        }
        return String(localized: "Open-H3のモデルが入っていません。「モデルを準備…」で結合LoRA（約12.5GB）と潜在アップスケーラをダウンロードし、精緻化LoRAを取り出すと選べます。")
    }
    func makeScaler(_ job: UpscaleJob) throws -> ChunkedFrameScaler {
        // Not a frame scaler: GenerationViewModel.runOpenH3Regeneration.
        throw VideoUpscale.UpscaleError.unavailable(label)
    }
}
