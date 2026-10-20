import CoreGraphics
import CryptoKit
import Foundation
import SwiftUI
import H3Engine

// 途中の様子: while a video denoises, the engine hands over its estimate of
// the finished latent every few model evaluations (H3LatentPreview) and
// TinyPreviewDecoder turns the middle of it into a picture. The decoder's
// weights (madebyollin/taehv's taeh3, MIT, 22.7 MB) are downloaded the
// first time the preview is turned on, not shipped in the app.

/// Where the preview decoder's weights live and how they are fetched.
enum DenoisePreviewWeights {
    static let path = h3AppSupportDirectory + "/Preview/taeh3.safetensors"
    /// Pinned to the commit that added it, checked by size and SHA-256.
    static let url = URL(string: "https://raw.githubusercontent.com/madebyollin/taehv/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taeh3.safetensors")!
    static let bytes: Int64 = 22_709_752
    static let sha256 = "4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13"
    static let sourcePage = URL(string: "https://github.com/madebyollin/taehv")!

    static var isPresent: Bool {
        ((try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? NSNumber)?.int64Value == bytes
    }
}

@MainActor
final class DenoisePreviewDownloader: ObservableObject {
    enum State: Equatable { case idle, downloading, failed(String) }
    @Published private(set) var state: State = .idle

    func download(then done: @escaping @MainActor () -> Void = {}) {
        guard state != .downloading else { return }
        state = .downloading
        Task {
            do {
                let (temp, response) = try await URLSession.shared.download(from: DenoisePreviewWeights.url)
                defer { try? FileManager.default.removeItem(at: temp) }
                let status = (response as? HTTPURLResponse)?.statusCode ?? -1
                guard (200 ..< 300).contains(status) else {
                    throw ModelDownloadError.httpStatus(path: "taeh3.safetensors", status: status)
                }
                let data = try Data(contentsOf: temp)
                let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                guard Int64(data.count) == DenoisePreviewWeights.bytes, digest == DenoisePreviewWeights.sha256 else {
                    throw ModelDownloadError.sizeMismatch(path: "taeh3.safetensors")
                }
                let target = URL(fileURLWithPath: DenoisePreviewWeights.path)
                try FileManager.default.createDirectory(at: target.deletingLastPathComponent(),
                                                        withIntermediateDirectories: true)
                try? FileManager.default.removeItem(at: target)
                try data.write(to: target, options: .atomic)
                state = .idle
                done()
            } catch {
                NSLog("H3cApp: preview decoder download failed: %@", error.localizedDescription)
                state = .failed(error.localizedDescription)
            }
        }
    }
}

/// Decodes latent previews off the main thread, one at a time: a preview
/// arriving while the previous one is still being decoded is dropped (the
/// next one is a few model evaluations away).
final class DenoisePreviewRenderer: @unchecked Sendable {
    private let queue = DispatchQueue(label: "h3c.denoise-preview", qos: .userInitiated)
    private let lock = NSLock()
    private var busy = false
    private var decoder: TinyPreviewDecoder?

    /// The decoded picture (nil if it could not be made), on the main actor.
    func render(_ preview: H3LatentPreview, completion: @escaping @MainActor (CGImage?) -> Void) {
        lock.lock()
        if busy { lock.unlock(); return }
        busy = true
        lock.unlock()
        queue.async { [self] in
            var image: CGImage?
            do {
                if decoder == nil { decoder = try TinyPreviewDecoder(path: DenoisePreviewWeights.path) }
                let input = H3LatentPreviewInput(time: preview.time, height: preview.height,
                                                 width: preview.width, latent: preview.latent)
                image = try decoder?.image(from: input, lastStep: preview.time / 2)
            } catch {
                NSLog("H3cApp: denoise preview failed: %@", error.localizedDescription)
            }
            lock.lock(); busy = false; lock.unlock()
            let result = image
            Task { @MainActor in completion(result) }
        }
    }

    /// How often to ask the engine: about five pictures per video.
    static func interval(steps: Int, reuse: Int) -> Int32 {
        let evaluations = max(1, (steps + max(1, reuse) - 1) / max(1, reuse))
        return Int32(max(1, Int((Double(evaluations) / 5).rounded())))
    }
}

/// Advanced Settings' 途中の様子 row: the switch, what it does, and the
/// decoder download it needs.
struct DenoisePreviewSettingRow: View {
    @ObservedObject var viewModel: GenerationViewModel
    @ObservedObject var downloader: DenoisePreviewDownloader
    @Environment(\.colorScheme) private var colorScheme
    private var palette: H3Palette { H3Palette(colorScheme) }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Toggle("生成中に途中の様子を表示する", isOn: Binding(
                get: { viewModel.showDenoisePreview },
                set: { on in
                    viewModel.showDenoisePreview = on
                    if on && !DenoisePreviewWeights.isPresent { downloader.download() }
                }))
                .toggleStyle(.checkbox)
            Text("ノイズを取り除いている途中で、できあがりつつある動画の1コマを数回表示します。思っていた動画と違えば、早めに止められます。生成時間はほとんど変わりません。小さなデコーダ（madebyollin/taehv の taeh3、MITライセンス、22.7MB）を使い、初めてオンにしたときにダウンロードします。")
                .font(.caption)
                .foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            if viewModel.showDenoisePreview && !DenoisePreviewWeights.isPresent {
                switch downloader.state {
                case .downloading:
                    HStack {
                        ProgressView().controlSize(.small)
                        Text("デコーダをダウンロードしています…").font(.caption).foregroundStyle(palette.textSecondary)
                    }
                case .failed(let message):
                    HStack {
                        Text("ダウンロードできませんでした（\(message)）").font(.caption).foregroundStyle(palette.errorColor)
                        Button("再試行") { downloader.download() }.controlSize(.small)
                    }
                case .idle:
                    Button("デコーダをダウンロード（22.7MB）") { downloader.download() }.controlSize(.small)
                }
            }
        }
    }
}
