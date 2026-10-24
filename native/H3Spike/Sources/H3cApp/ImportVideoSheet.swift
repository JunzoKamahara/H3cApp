import SwiftUI

/// 動画ファイルの続きを作る: what the chosen video is, and the size to bring
/// it in (and make its continuation) at - the video's own by default.
struct ImportVideoSheet: View {
    @ObservedObject var viewModel: GenerationViewModel
    let pending: GenerationViewModel.PendingVideoImport
    @State private var choice: SizeProfile
    @State private var carryFrames = continuationFrames
    @Environment(\.colorScheme) private var colorScheme
    private var palette: H3Palette { H3Palette(colorScheme) }

    init(viewModel: GenerationViewModel, pending: GenerationViewModel.PendingVideoImport) {
        self.viewModel = viewModel
        self.pending = pending
        _choice = State(initialValue: pending.suggested)
    }

    /// The video's size first, then the form's (if different), then the
    /// presets.
    private var choices: [SizeProfile] {
        var list = [pending.suggested]
        if viewModel.sizeProfile != pending.suggested { list.append(viewModel.sizeProfile) }
        for shape in AspectShape.allCases {
            for tier in SizeTier.allCases {
                let preset = SizeProfile.preset(shape, tier)
                if !list.contains(preset) { list.append(preset) }
            }
        }
        return list
    }

    private func carryTitle(_ frames: Int) -> String {
        let seconds = String(format: "%.1f", Double(frames) / 24)
        return frames == continuationFrames
            ? String(localized: "約\(seconds)秒（\(frames)コマ、標準）")
            : String(localized: "約\(seconds)秒（\(frames)コマ）")
    }

    private func title(_ profile: SizeProfile) -> String {
        if profile == pending.suggested { return String(localized: "動画に合わせる：\(profile.label)") }
        if profile == viewModel.sizeProfile { return String(localized: "今の設定：\(profile.label)") }
        return profile.label
    }

    /// The video and the chosen size have different shapes (cropped).
    private var crops: Bool {
        let video = Double(pending.width) / Double(max(pending.height, 1))
        let size = Double(choice.dimensions.width) / Double(max(choice.dimensions.height, 1))
        return abs(video - size) / video > 0.02
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("動画を取り込む").font(.headline)
            Text("\(pending.url.lastPathComponent)：\(pending.width)×\(pending.height)、\(String(format: "%.0f", pending.fps))fps、\(String(format: "%.1f", pending.seconds))秒")
                .font(.callout)
            Picker("生成する大きさ", selection: $choice) {
                ForEach(choices, id: \.self) { profile in
                    Text(title(profile)).tag(profile)
                }
            }
            .fixedSize()
            Picker("引き継ぐ長さ", selection: $carryFrames) {
                ForEach(GenerationViewModel.importCarryFrames, id: \.self) { frames in
                    Text(carryTitle(frames)).tag(frames)
                }
            }
            .fixedSize()
            VStack(alignment: .leading, spacing: 4) {
                Text("この大きさと24fpsに変換して取り込み、その続きも同じ大きさでつくります。パネルの大きさの設定も、この大きさになります。")
                if choice.upscaled {
                    Text("大きい動画なので、半分の大きさで生成して2倍に拡大します。")
                }
                if crops {
                    Text("動画と縦横比が違うので、中央を切り抜いて取り込みます。")
                }
                Text("引き継ぐ長さを長くすると、取り込みと続きの生成に時間がかかり、続きで新しくつくれる部分が短くなります。")
            }
            .font(.caption)
            .foregroundStyle(palette.textSecondary)
            .fixedSize(horizontal: false, vertical: true)
            HStack {
                Spacer()
                Button("キャンセル") { viewModel.pendingVideoImport = nil }
                    .keyboardShortcut(.cancelAction)
                Button("取り込む") {
                    let size = choice
                    let url = pending.url
                    let carry = carryFrames
                    viewModel.pendingVideoImport = nil
                    Task { await viewModel.importVideoToContinue(url, size: size, carryFrames: carry) }
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(H3Spacing.lg)
        .frame(width: 520)
    }
}
