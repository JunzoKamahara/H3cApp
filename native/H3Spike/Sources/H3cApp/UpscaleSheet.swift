import SwiftUI

/// 拡大: the video, the factor (with the size it makes), and the upscaler.
struct UpscaleSheet: View {
    @ObservedObject var viewModel: GenerationViewModel
    let pending: GenerationViewModel.PendingUpscale
    var close: () -> Void
    @State private var upscaler: VideoUpscaler = .standard
    @State private var factor: UpscaleFactor
    @State private var showingSwiftVRSetup = false
    @State private var showingSeedVR2Setup = false
    @State private var showingOpenH3Setup = false
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

    /// Open-H3 regenerates on H3's canvas: its own sizes (OpenH3Plan).
    private func openH3Plan(_ factor: UpscaleFactor) -> OpenH3Plan? {
        OpenH3Plan.plan(width: pending.width, height: pending.height, factor: factor)
    }

    /// For Open-H3, a factor whose size a smaller factor already makes
    /// (the canvas limit) is not offered twice.
    private func fits(_ factor: UpscaleFactor) -> Bool {
        guard upscaler.regeneratesFromLatent else {
            return factor.fits(width: pending.width, height: pending.height)
        }
        guard let plan = openH3Plan(factor) else { return false }
        return !UpscaleFactor.allCases.contains { $0.rawValue < factor.rawValue && openH3Plan($0) == plan }
    }

    private func sizeText(_ factor: UpscaleFactor) -> String {
        if upscaler.regeneratesFromLatent {
            guard let plan = openH3Plan(factor) else { return String(localized: "拡大できません") }
            return "\(plan.width)×\(plan.height)"
        }
        let size = factor.size(width: pending.width, height: pending.height)
        return "\(size.width)×\(size.height)"
    }

    /// Why the Open-H3 choice can't run on this video (nil when it can).
    private var openH3Reason: String? {
        guard upscaler.regeneratesFromLatent else { return nil }
        guard let video = pending.projectVideo else {
            return String(localized: "Open-H3 再生成は、プロジェクトの動画（潜在ファイルのあるもの）にだけ使えます。")
        }
        return viewModel.openH3Reason(for: video, size: (pending.width, pending.height), factor: factor)
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
                     : upscaler.regeneratesFromLatent
                        ? (openH3Plan(option) == nil ? Text("\(option.label)（Open-H3の上限1344×768を超えるため選べません）")
                                                     : Text("\(option.label)（\(sizeText(option))、上限のため下の倍率と同じ大きさ）"))
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
                } else if let reason = openH3Reason {
                    Text(reason).foregroundStyle(.orange)
                }
                if upscaler.regeneratesFromLatent, openH3Reason == nil, let plan = openH3Plan(factor),
                   let frames = pending.projectVideo?.record?.actualFrameCount {
                    Text("見込み：\(plan.width)×\(plan.height)・\(frames)フレームで、M5で約\(formatElapsed(plan.estimatedSeconds(frames: frames)))（生成の設定による）。")
                }
                if upscaler.isSwiftVR {
                    HStack {
                        Button(upscaler.isAvailable ? "モデルを管理…" : "モデルを準備…") { showingSwiftVRSetup = true }
                            .controlSize(.small)
                        if SwiftVRSetup.shared.isBusy {
                            Text("準備中です（\(SwiftVRSetup.shared.phase.apiText)）").foregroundStyle(.orange)
                        }
                    }
                }
                if upscaler == .seedVR2 {
                    HStack {
                        Button(upscaler.isAvailable ? "モデルを管理…" : "モデルを準備…") { showingSeedVR2Setup = true }
                            .controlSize(.small)
                        if SeedVR2Setup.shared.isBusy {
                            Text("準備中です（\(SeedVR2Setup.shared.phase.apiText)）").foregroundStyle(.orange)
                        }
                    }
                }
                if upscaler == .openH3 {
                    HStack {
                        Button(upscaler.isAvailable ? "モデルを管理…" : "モデルを準備…") { showingOpenH3Setup = true }
                            .controlSize(.small)
                        if OpenH3Setup.shared.isBusy {
                            Text("準備中です（\(OpenH3Setup.shared.phase.apiText)）").foregroundStyle(.orange)
                        }
                    }
                }
                Text(pending.projectVideo != nil
                     ? String(localized: "拡大した動画は、プロジェクトに新しく加わります（名前の最後に「_\(factor.suffix)」）。音声とAIで生成したという記録は、そのままです。")
                     : String(localized: "拡大した動画の保存先は、次に選びます。音声とファイルの情報は、そのままです。"))
                if !fits(factor) {
                    (upscaler.regeneratesFromLatent
                        ? Text("この倍率はOpen-H3では選べません。")
                        : Text("一辺が4096ピクセルを超えるので、この倍率にはできません。"))
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
                          || viewModel.upscaleBlockedReason(upscaler) != nil || openH3Reason != nil)
            }
        }
        .padding(H3Spacing.lg)
        .frame(width: 500)
        .sheet(isPresented: $showingSwiftVRSetup) {
            SwiftVRSetupView(busy: viewModel.isGenerating || viewModel.isBuildingCache) { showingSwiftVRSetup = false }
        }
        .sheet(isPresented: $showingSeedVR2Setup) {
            SeedVR2SetupView(busy: viewModel.isGenerating || viewModel.isBuildingCache) { showingSeedVR2Setup = false }
        }
        .sheet(isPresented: $showingOpenH3Setup) {
            OpenH3SetupView(busy: viewModel.isGenerating || viewModel.isBuildingCache) { showingOpenH3Setup = false }
        }
    }
}
