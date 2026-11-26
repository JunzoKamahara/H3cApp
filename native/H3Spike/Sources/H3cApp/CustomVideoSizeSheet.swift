import SwiftUI

/// 大きさ > カスタム…: a video's finished width and height, typed in, and
/// whether to generate it at half and upscale 2x. The sides are rounded to
/// their step (32, or 64 with the upscale) within their range; the size
/// generated, its pixels against the limit and the time on this Mac are
/// shown as the numbers change, and nothing changes until it is applied.
struct CustomVideoSizeSheet: View {
    @ObservedObject var viewModel: GenerationViewModel
    let close: () -> Void
    @State private var width: Int
    @State private var height: Int
    @State private var upscaled: Bool

    init(viewModel: GenerationViewModel, close: @escaping () -> Void) {
        self.viewModel = viewModel
        self.close = close
        let current = viewModel.sizeProfile
        let output = current.dimensions
        _width = State(initialValue: Int(output.width))
        _height = State(initialValue: Int(output.height))
        _upscaled = State(initialValue: current.upscaled)
    }

    private func rounded(_ side: Int) -> Int32 {
        let range = SizeProfile.customSideRange(upscaled: upscaled)
        let step = Int(SizeProfile.customSideStep(upscaled: upscaled))
        let value = Int32(clamping: ((max(side, 0) + step / 2) / step) * step)
        return min(max(value, range.lowerBound), range.upperBound)
    }

    private var profile: SizeProfile {
        .custom(width: rounded(width), height: rounded(height), upscaled: upscaled)
    }

    var body: some View {
        let used = profile
        let output = used.dimensions
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("動画の大きさ").font(.headline)
            HStack(alignment: .firstTextBaseline, spacing: H3Spacing.sm) {
                TextField("幅", value: $width, format: .number.grouping(.never))
                    .textFieldStyle(.roundedBorder)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 80)
                Text(verbatim: "×")
                TextField("高さ", value: $height, format: .number.grouping(.never))
                    .textFieldStyle(.roundedBorder)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 80)
                Text("ピクセル")
                Button {
                    swap(&width, &height)
                } label: {
                    Label("縦横を入れ替える", systemImage: "arrow.left.arrow.right")
                }
                .help("幅と高さを入れ替えます")
            }
            Toggle("半分のサイズで生成して2倍に拡大", isOn: $upscaled)
            VStack(alignment: .leading, spacing: 4) {
                Text("完成 \(used.resolutionLabel) ・ 生成 \(used.canvasLabel)（\(used.ditPixels.formatted())ピクセル / 上限 \(SizeProfile.customMaxPixels.formatted())）")
                    .foregroundStyle(used.isWithinCustomLimit ? Color.primary : Color.red)
                if Int(output.width) != width || Int(output.height) != height {
                    Text(upscaled
                         ? "2倍に拡大するときは、幅と高さを64の倍数・512〜2688ピクセルに直して使います。"
                         : "幅と高さは32の倍数・256〜1344ピクセルです。近い大きさに直して使います。")
                        .foregroundStyle(.secondary)
                }
                if !used.isWithinCustomLimit {
                    Text(upscaled
                         ? "生成するサイズが上限を超えています。幅か高さを小さくしてください。"
                         : "生成するサイズが上限を超えています。小さくするか、「半分のサイズで生成して2倍に拡大」をオンにしてください。")
                        .foregroundStyle(.red)
                } else if let estimate = viewModel.draftEstimatedSeconds(for: used) {
                    let minutes = max(Int((estimate / 60).rounded(.up)), 1)
                    Text("このMacでの目安: 約\(minutes)分（いまの長さと設定・これまでの生成実績から）")
                        .foregroundStyle(.secondary)
                }
                Text("拡大は生成後に行うため、時間はほとんど増えません。メニューで小・中・大を選ぶとプリセットに戻ります。")
                    .foregroundStyle(.secondary)
            }
            .font(.caption)
            .fixedSize(horizontal: false, vertical: true)
            HStack {
                Spacer()
                Button("キャンセル", role: .cancel, action: close)
                    .keyboardShortcut(.cancelAction)
                Button("この大きさにする") {
                    viewModel.sizeProfile = used
                    close()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!used.isWithinCustomLimit)
            }
        }
        .padding(H3Spacing.lg)
        .frame(width: 460)
    }
}
