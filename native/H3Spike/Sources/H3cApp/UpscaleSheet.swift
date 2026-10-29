import SwiftUI

/// 拡大: the video, the factor (with the size it makes), and the upscaler.
struct UpscaleSheet: View {
    @ObservedObject var viewModel: GenerationViewModel
    let pending: GenerationViewModel.PendingUpscale
    var close: () -> Void
    @State private var upscaler: VideoUpscaler = .standard
    @State private var factor: UpscaleFactor
    @AppStorage(VideoUpscale.swiftVRFourXKey) private var swiftVRFourX = false
    @State private var showingSwiftVRSetup = false
    @Environment(\.colorScheme) private var colorScheme
    private var palette: H3Palette { H3Palette(colorScheme) }

    init(viewModel: GenerationViewModel, pending: GenerationViewModel.PendingUpscale, close: @escaping () -> Void) {
        self.viewModel = viewModel
        self.pending = pending
        self.close = close
        // 2x when it fits, else the largest that does.
        let fitting = UpscaleFactor.allCases.filter { $0.fits(width: pending.width, height: pending.height) }
        _factor = State(initialValue: fitting.contains(.two) ? .two : (fitting.last ?? .two))
    }

    private func fits(_ factor: UpscaleFactor) -> Bool {
        factor.fits(width: pending.width, height: pending.height)
    }

    private func sizeText(_ factor: UpscaleFactor) -> String {
        let size = factor.size(width: pending.width, height: pending.height)
        return "\(size.width)×\(size.height)"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("動画を拡大").font(.headline)
            Text("\(pending.url.lastPathComponent)：\("\(pending.width)×\(pending.height)") → \(sizeText(factor))")
                .font(.callout)
                .monospacedDigit()
            Picker("倍率", selection: $factor) {
                ForEach(UpscaleFactor.allCases) { option in
                    (fits(option) ? Text("\(option.label)（\(sizeText(option))）")
                                  : Text("\(option.label)（\(sizeText(option))、4096ピクセルを超えるため選べません）"))
                        .tag(option)
                        .disabled(!fits(option))
                }
            }
            .pickerStyle(.radioGroup)
            Picker("拡大の方法", selection: $upscaler) {
                ForEach(VideoUpscaler.allCases) { option in
                    Text(option.label).tag(option)
                        .disabled(!option.isAvailable)
                }
            }
            .pickerStyle(.radioGroup)
            VStack(alignment: .leading, spacing: 4) {
                Text(upscaler.detail)
                if let reason = upscaler.unavailableReason {
                    Text(reason).foregroundStyle(.orange)
                } else if let reason = viewModel.upscaleBlockedReason(upscaler) {
                    Text(reason).foregroundStyle(.orange)
                }
                if upscaler == .swiftVR && factor == .two
                    && UpscaleFactor.four.fits(width: pending.width, height: pending.height) {
                    Toggle("4倍で生成してから縮小する（質感が穏やかになり、時間は約4倍）", isOn: $swiftVRFourX)
                        .font(.callout)
                }
                if upscaler == .swiftVR {
                    HStack {
                        Button(upscaler.isAvailable ? "モデルを管理…" : "モデルを準備…") { showingSwiftVRSetup = true }
                            .controlSize(.small)
                        if SwiftVRSetup.shared.isBusy {
                            Text("準備中です（\(SwiftVRSetup.shared.phase.apiText)）").foregroundStyle(.orange)
                        }
                    }
                }
                Text(pending.projectVideo != nil
                     ? String(localized: "拡大した動画は、プロジェクトに新しく加わります（名前の最後に「_\(factor.suffix)」）。音声とAIで生成したという記録は、そのままです。")
                     : String(localized: "拡大した動画の保存先は、次に選びます。音声とファイルの情報は、そのままです。"))
                if !fits(factor) {
                    Text("一辺が4096ピクセルを超えるので、この倍率にはできません。")
                        .foregroundStyle(.red)
                }
            }
            .font(.caption)
            .foregroundStyle(palette.textSecondary)
            .fixedSize(horizontal: false, vertical: true)
            HStack {
                Spacer()
                Button("キャンセル") { close() }
                    .keyboardShortcut(.cancelAction)
                Button("拡大する") {
                    let chosen = upscaler
                    let scale = factor
                    close()
                    viewModel.startUpscale(pending, upscaler: chosen, factor: scale)
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!fits(factor) || !upscaler.isAvailable || viewModel.isUpscalingVideo
                          || viewModel.upscaleBlockedReason(upscaler) != nil)
            }
        }
        .padding(H3Spacing.lg)
        .frame(width: 500)
        .sheet(isPresented: $showingSwiftVRSetup) {
            SwiftVRSetupView(busy: viewModel.isGenerating || viewModel.isBuildingCache) { showingSwiftVRSetup = false }
        }
    }
}
