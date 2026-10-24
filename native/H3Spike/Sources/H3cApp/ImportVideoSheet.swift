import AVFoundation
import SwiftUI

/// 動画ファイルの続きを作る: what the chosen video is, and the size to bring
/// it in (and make its continuation) at - the video's own by default.
struct ImportVideoSheet: View {
    @ObservedObject var viewModel: GenerationViewModel
    let pending: GenerationViewModel.PendingVideoImport
    @State private var choice: SizeProfile
    @State private var carryFrames = continuationFrames
    /// The video's last frame: where the continuation starts.
    @State private var lastFrame: CGImage?
    @State private var lastFrameFailed = false
    /// Where the kept part sits, 0-1 across what is cut off (0.5 centred).
    @State private var cropPosition = CGPoint(x: 0.5, y: 0.5)
    @State private var dragStart: CGPoint?
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

    /// The part of the frame the chosen size keeps (scaled to fill it, at
    /// cropPosition - as convertForContinuing does), in 0-1 coordinates.
    private var keptRect: CGRect {
        let video = Double(pending.width) / Double(max(pending.height, 1))
        let size = Double(choice.dimensions.width) / Double(max(choice.dimensions.height, 1))
        if size < video {
            let width = size / video
            return CGRect(x: (1 - width) * cropPosition.x, y: 0, width: width, height: 1)
        }
        let height = video / size
        return CGRect(x: 0, y: (1 - height) * cropPosition.y, width: 1, height: height)
    }

    /// Moves the kept part with the pointer, across what is cut off.
    private func drag(_ translation: CGSize, shown: CGSize) {
        let start = dragStart ?? cropPosition
        dragStart = start
        let kept = keptRect
        let slack = CGSize(width: (1 - kept.width) * shown.width, height: (1 - kept.height) * shown.height)
        cropPosition = CGPoint(
            x: slack.width > 0.5 ? min(max(start.x + translation.width / slack.width, 0), 1) : 0.5,
            y: slack.height > 0.5 ? min(max(start.y + translation.height / slack.height, 0), 1) : 0.5)
    }

    /// The last frame with what the chosen size keeps outlined and the rest
    /// darkened.
    @ViewBuilder private var preview: some View {
        let frame = CGSize(width: max(pending.width, 1), height: max(pending.height, 1))
        let scale = min(488 / frame.width, 240 / frame.height)
        let shown = CGSize(width: frame.width * scale, height: frame.height * scale)
        let kept = keptRect
        let keptShown = CGRect(x: kept.minX * shown.width, y: kept.minY * shown.height,
                               width: kept.width * shown.width, height: kept.height * shown.height)
        VStack(alignment: .leading, spacing: H3Spacing.xs) {
            ZStack(alignment: .topLeading) {
                if let lastFrame {
                    Image(decorative: lastFrame, scale: 1)
                        .resizable()
                        .frame(width: shown.width, height: shown.height)
                } else {
                    Rectangle().fill(palette.textSecondary.opacity(0.15))
                        .frame(width: shown.width, height: shown.height)
                        .overlay {
                            if lastFrameFailed {
                                Text("プレビューを表示できません").font(.caption)
                                    .foregroundStyle(palette.textSecondary)
                            } else {
                                ProgressView().controlSize(.small)
                            }
                        }
                }
                Path { path in
                    path.addRect(CGRect(origin: .zero, size: shown))
                    path.addRect(keptShown)
                }
                .fill(Color.black.opacity(0.6), style: FillStyle(eoFill: true))
                Rectangle()
                    .stroke(Color.white, lineWidth: 2)
                    .frame(width: keptShown.width, height: keptShown.height)
                    .offset(x: keptShown.minX, y: keptShown.minY)
            }
            .frame(width: shown.width, height: shown.height)
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .animation(.easeInOut(duration: 0.15), value: choice)
            .contentShape(Rectangle())
            .pointerStyle(crops ? (dragStart == nil ? .grabIdle : .grabActive) : nil)
            .gesture(DragGesture(minimumDistance: 1)
                .onChanged { drag($0.translation, shown: shown) }
                .onEnded { _ in dragStart = nil })
            .onTapGesture(count: 2) { cropPosition = CGPoint(x: 0.5, y: 0.5) }
            (crops ? Text("取り込まれる範囲（動画の最後のフレーム）。枠はドラッグで動かせます（ダブルクリックで中央）")
                   : Text("取り込まれる範囲（動画の最後のフレーム）"))
                .font(.caption)
                .foregroundStyle(palette.textSecondary)
        }
        .frame(maxWidth: .infinity)
    }

    /// The last frame of the video track (the sound may run longer, so
    /// not the file's own end).
    private static func lastFrame(of url: URL) async -> CGImage? {
        let asset = AVURLAsset(url: url)
        guard let track = try? await asset.loadTracks(withMediaType: .video).first,
              let range = try? await track.load(.timeRange) else { return nil }
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 976, height: 976)
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.5, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = .zero
        let end = range.end.seconds - 0.05
        if let image = try? await generator.image(at: CMTime(seconds: max(end, 0), preferredTimescale: 600)).image {
            return image
        }
        // Whatever frame is nearest a little before the end.
        generator.requestedTimeToleranceBefore = .positiveInfinity
        generator.requestedTimeToleranceAfter = .positiveInfinity
        return try? await generator.image(at: CMTime(seconds: max(end - 0.5, 0), preferredTimescale: 600)).image
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("動画を取り込む").font(.headline)
            Text("\(pending.url.lastPathComponent)：\(pending.width)×\(pending.height)、\(String(format: "%.0f", pending.fps))fps、\(String(format: "%.1f", pending.seconds))秒")
                .font(.callout)
            preview
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
                    Text("動画と縦横比が違うので、枠の範囲を切り抜いて取り込みます。")
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
                    let position = cropPosition
                    viewModel.pendingVideoImport = nil
                    Task { await viewModel.importVideoToContinue(url, size: size, carryFrames: carry,
                                                                 cropPosition: position) }
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(H3Spacing.lg)
        .frame(width: 520)
        .task {
            lastFrame = await Self.lastFrame(of: pending.url)
            lastFrameFailed = lastFrame == nil
        }
    }
}
