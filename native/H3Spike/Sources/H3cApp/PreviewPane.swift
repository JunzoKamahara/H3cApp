import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct PreviewPane: View {
    @ObservedObject var viewModel: GenerationViewModel
    // Room the floating composer takes at the bottom: the empty and
    // in-progress states centre their text above it. The finished video uses
    // the whole stage and the composer overlays it (its transport bar sits
    // on top for that reason).
    var bottomInset: CGFloat = 0
    @Environment(\.colorScheme) private var colorScheme
    private var palette: H3Palette { H3Palette(colorScheme) }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            header
            centerContent
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .padding(.horizontal, H3Spacing.xl)
        .padding(.bottom, H3Spacing.xl)
        .padding(.top, H3Spacing.sm)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(palette.canvas)
    }

    private var header: some View {
        // The result actions sit up here: the bottom of the stage is where
        // the floating composer lives.
        HStack(alignment: .top) {
            Text("プレビュー").font(.system(size: 14, weight: .semibold)).foregroundStyle(palette.textPrimary)
            Text(statusLabel).font(.caption).foregroundStyle(palette.textSecondary)
            Spacer()
            if viewModel.resultURL != nil, let result = viewModel.lastResult {
                ResultActionsView(viewModel: viewModel, result: result)
                    .fixedSize()
            }
        }
    }

    private var statusLabel: String {
        if viewModel.isCancelling { return String(localized: "中止しています") }
        if viewModel.isGenerating {
            if let batch = viewModel.batchProgress {
                return batch.chained
                    ? String(localized: "つなげて生成しています（\(batch.index)/\(batch.total)本目）")
                    : String(localized: "動画を生成しています（\(batch.index)/\(batch.total)本目）")
            }
            return String(localized: "動画を生成しています")
        }
        if let url = viewModel.resultURL {
            var label = String(localized: "できあがりました")
            if let seconds = viewModel.lastResult?.generationSeconds {
                label = String(localized: "できあがりました ・ 生成時間 \(formatElapsed(seconds))")
            }
            if !viewModel.isTemporaryResult(url) {
                label += summarySeparator + url.lastPathComponent
            }
            return label
        }
        if let message = viewModel.errorMessage { return message }
        return ""
    }

    @ViewBuilder
    private var centerContent: some View {
        if let url = viewModel.resultURL {
            ResultPlayerView(url: url, aspectRatio: viewModel.resultAspectRatio,
                             nativeSize: viewModel.lastResult.map {
                                 CGSize(width: CGFloat($0.sizeProfile.dimensions.width),
                                        height: CGFloat($0.sizeProfile.dimensions.height))
                             } ?? .zero)
                // The player is created once per view; a new identity per
                // video makes showing another saved video (or a new result)
                // load it instead of keeping the previous one.
                .id(url)
        } else if viewModel.isGenerating {
            generatingView
        } else {
            emptyStateView
        }
    }

    private var emptyStateView: some View {
        VStack(spacing: H3Spacing.md) {
            Image(systemName: "video")
                .font(.system(size: 32))
                .foregroundStyle(palette.textSecondary)
            Text("ここに動画が表示されます").font(.body).foregroundStyle(palette.textSecondary)
            Text("下のパネルで内容を決めて、動画をつくりましょう。")
                .font(.caption)
                .foregroundStyle(palette.textSecondary)
            if let error = viewModel.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(palette.errorColor)
                    .multilineTextAlignment(.center)
                    .padding(.top, H3Spacing.sm)
            }
        }
        .padding(.bottom, bottomInset)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(palette.stage)
        .clipShape(RoundedRectangle(cornerRadius: H3Radius.stage))
    }

    private var generatingView: some View {
        GeometryReader { geometry in
            generatingStack(height: geometry.size.height)
                .frame(width: geometry.size.width, height: geometry.size.height)
        }
        .background(palette.stage)
        .clipShape(RoundedRectangle(cornerRadius: H3Radius.stage))
    }

    /// The preview picture gets what the progress text below it leaves of
    /// the stage (above the composer), so that text always stays in view.
    private func denoisePreviewHeight(stage height: CGFloat) -> CGFloat {
        let text: CGFloat = viewModel.batchProgress == nil ? 190 : 290
        return min(360, max(60, height - bottomInset - text))
    }

    private func generatingStack(height: CGFloat) -> some View {
        VStack(spacing: H3Spacing.md) {
            if let image = viewModel.denoisePreviewImage {
                denoisePreview(image, maxHeight: denoisePreviewHeight(stage: height))
            } else if viewModel.denoisePreviewActive {
                Text("途中の様子は、ノイズ除去が少し進むと表示されます")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            }
            if let batch = viewModel.batchProgress, let timing = viewModel.batchTiming {
                batchTimingView(batch, timing)
            }
            ProgressView(value: viewModel.progressBarFraction)
                .frame(maxWidth: 260)

            VStack(spacing: 2) {
                Text(viewModel.stageTitle).font(.callout)
                Text(viewModel.stageDetail)
                    .font(.caption)
                    .monospacedDigit()
                    .foregroundStyle(palette.textSecondary)
            }

            VStack(spacing: 2) {
                Text(viewModel.batchProgress == nil
                     ? String(localized: "経過 \(formatElapsed(viewModel.elapsedSeconds))")
                     : String(localized: "この1本の経過 \(formatElapsed(viewModel.elapsedSeconds))"))
                    .font(.caption)
                    .monospacedDigit()
                if let remaining = viewModel.estimatedRemainingSeconds,
                   let finishDate = viewModel.estimatedFinishDate {
                    Text(viewModel.batchProgress == nil
                         ? String(localized: "残り約 \(formatElapsed(remaining))（完了予定 \(formatClockTime(finishDate))）")
                         : String(localized: "この1本の残り約 \(formatElapsed(remaining))（完了予定 \(formatClockTime(finishDate))）"))
                        .font(.caption)
                        .monospacedDigit()
                        .foregroundStyle(palette.textSecondary)
                } else {
                    Text("残り時間を見積り中…")
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                }
            }
        }
        .padding(.bottom, bottomInset)
    }

    /// 途中の様子: the picture the video is turning into, from the model's
    /// estimate of the finished latent (Advanced Settings).
    private func denoisePreview(_ image: CGImage, maxHeight: CGFloat) -> some View {
        VStack(spacing: 4) {
            Image(decorative: image, scale: 1)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxWidth: 640, maxHeight: maxHeight)
                .clipShape(RoundedRectangle(cornerRadius: H3Radius.control))
            Text("途中の様子（ステップ \(viewModel.denoisePreviewStep)/\(viewModel.denoisePreviewSteps)）。できあがる動画とは細部が変わります。")
                .font(.caption)
                .foregroundStyle(palette.textSecondary)
        }
        .padding(.horizontal, H3Spacing.lg)
    }

    /// Several videos or a つなげて生成: which one this is and when the whole
    /// run should end, above the current video's own progress.
    private func batchTimingView(_ batch: BatchProgress, _ timing: BatchTiming) -> some View {
        VStack(spacing: 4) {
            Text(batch.chained
                 ? String(localized: "つなげて生成：全\(batch.total)本中\(batch.index)本目を生成しています")
                 : String(localized: "全\(batch.total)本中\(batch.index)本目を生成しています"))
                .font(.callout.weight(.semibold))
            ProgressView(value: timing.fraction)
                .frame(maxWidth: 260)
            Text("全体の経過 \(formatElapsed(timing.elapsed))")
                .font(.caption)
                .monospacedDigit()
            if let remaining = timing.remaining, let finishDate = timing.finishDate {
                Text("全体の残り約 \(formatElapsed(remaining))（全体の完了予定 \(formatClockTime(finishDate))）")
                    .font(.caption)
                    .monospacedDigit()
            } else {
                Text("全体の残り時間を見積り中…")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            }
            Divider().frame(maxWidth: 260).padding(.top, H3Spacing.sm)
        }
    }
}

private struct ResultActionsView: View {
    @ObservedObject var viewModel: GenerationViewModel
    let result: ResolvedResult
    @State private var showingSettings = false
    @State private var exportConfirmation: String?
    @State private var lastExportedURL: URL?

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.xs) {
            HStack(spacing: H3Spacing.sm) {
                Button("動画を書き出す…") { exportVideo() }
                    .keyboardShortcut("s", modifiers: .command)

                Button("使用した設定…") { showingSettings = true }
                    .popover(isPresented: $showingSettings) {
                        SettingsDetailView(viewModel: viewModel, result: result) {
                            showingSettings = false
                        }
                    }

                if let url = viewModel.resultURL {
                    Button("参照に使う") { viewModel.useVideoAsReference(url) }
                        .help("この動画を参照動画としてパネルに追加します")
                    if !viewModel.isTemporaryResult(url) {
                        Button("Finderで表示") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                    }
                }

                Spacer()

                if let lastExportedURL {
                    Button("Finderで表示") {
                        NSWorkspace.shared.activateFileViewerSelecting([lastExportedURL])
                    }
                    .font(.caption)
                }
            }
            if let exportConfirmation {
                Text(exportConfirmation).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private static let exportDirectoryKey = "h3c-app.lastExportDirectory"

    /// The folder the last export went to, or ~/Downloads the first time
    /// (and whenever that folder no longer exists).
    private static var initialExportDirectory: URL? {
        var isDirectory: ObjCBool = false
        if let path = UserDefaults.standard.string(forKey: exportDirectoryKey),
           FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
           isDirectory.boolValue {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        return FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first
    }

    private func exportVideo() {
        guard let sourceURL = viewModel.resultURLForExport else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = sourceURL.lastPathComponent
        panel.allowedContentTypes = [.mpeg4Movie]
        panel.directoryURL = Self.initialExportDirectory
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        try? FileManager.default.removeItem(at: destination)
        do {
            try FileManager.default.copyItem(at: sourceURL, to: destination)
            // Remembered only once something was actually saved there.
            UserDefaults.standard.set(destination.deletingLastPathComponent().path,
                                      forKey: Self.exportDirectoryKey)
            lastExportedURL = destination
            exportConfirmation = String(localized: "保存しました: \(destination.lastPathComponent)")
        } catch {
            exportConfirmation = String(localized: "選んだ場所に保存できませんでした（詳細: \(error.localizedDescription)）")
        }
    }
}

private extension GenerationViewModel {
    // Export needs the still-live temp file, which is intentionally not
    // otherwise exposed as a stable public URL (design spec: distinguish
    // the engine's temp output from a user-chosen save location).
    var resultURLForExport: URL? { resultURL }
}

private struct SettingsDetailView: View {
    @ObservedObject var viewModel: GenerationViewModel
    let result: ResolvedResult
    var close: () -> Void = {}

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.sm) {
            Text("使用した設定").font(.headline)
            ScrollView {
                Text(result.settingsSummaryText)
                    .font(.caption)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 260)

            HStack {
                Button("フォームに戻す") {
                    viewModel.applyDraft(from: result)
                    close()
                }
                .help("この動画のプロンプトと設定をパネルに戻します")
                Button("プリセットとして保存…") {
                    viewModel.presetPendingName = viewModel.presetFromResult(result)
                    close()
                }
                .help("この動画の設定（プロンプト以外）に名前を付けて保存します")
                Spacer()
                Menu("その他") {
                    Button("設定をテキストでコピー") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(result.settingsSummaryText, forType: .string)
                    }
                    Button("同じシードを使う") { viewModel.useSameSeed(from: result) }
                }
                .fixedSize()
            }
        }
        .padding(H3Spacing.lg)
        .frame(minWidth: 360)
    }
}
