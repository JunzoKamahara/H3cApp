import SwiftUI

/// 大きさ > カスタム…: an image's width and height, typed in. The sides are
/// rounded to the image model's step (32) and kept within its range
/// (256-4096); what will be used, its megapixels and its time against
/// 標準 are shown as the numbers change.
struct CustomImageSizeSheet: View {
    @ObservedObject var viewModel: GenerationViewModel
    let close: () -> Void
    @State private var width: Int
    @State private var height: Int

    init(viewModel: GenerationViewModel, close: @escaping () -> Void) {
        self.viewModel = viewModel
        self.close = close
        let current = viewModel.imageSizeProfile.dimensions
        _width = State(initialValue: Int(current.width))
        _height = State(initialValue: Int(current.height))
    }

    private var size: (width: Int32, height: Int32) {
        (ImageSizeProfile.roundedSide(width), ImageSizeProfile.roundedSide(height))
    }

    private var megapixels: Double { Double(size.width) * Double(size.height) / (1024 * 1024) }

    var body: some View {
        let used = size
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("画像の大きさ").font(.headline)
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
            VStack(alignment: .leading, spacing: 4) {
                Text("つくる大きさ: \(String(used.width))×\(String(used.height))（\(String(format: "%.1f", megapixels))メガピクセル）")
                if Int(used.width) != width || Int(used.height) != height {
                    Text("幅と高さは32の倍数で、256〜4096ピクセルです。近い大きさに直して使います。")
                        .foregroundStyle(.secondary)
                }
                Text("時間の目安: 標準（1024×1024）の約\(String(format: "%.1f", ImageSizeProfile.relativeTime(width: used.width, height: used.height)))倍（参照画像なしのとき）")
                    .foregroundStyle(.secondary)
                if megapixels > 8.5 {
                    Text("約8メガピクセルを超えると、時間とメモリが大きく増え、このMacのメモリでは生成できないことがあります。")
                        .foregroundStyle(.orange)
                } else if viewModel.imageModelIsPH3 {
                    Text("P-H3-Imageは約8メガピクセル（特大）で最も良くなります。")
                        .foregroundStyle(.secondary)
                }
            }
            .font(.caption)
            .fixedSize(horizontal: false, vertical: true)
            HStack {
                Spacer()
                Button("キャンセル", role: .cancel, action: close)
                    .keyboardShortcut(.cancelAction)
                Button("この大きさにする") {
                    viewModel.imageSizeProfile = .custom(width: used.width, height: used.height)
                    close()
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(H3Spacing.lg)
        .frame(width: 460)
    }
}
