import SwiftUI

/// 拡大: the video, then the upscaler, then the factors that upscaler can
/// make of this video (UpscaleChoices) with the size each makes, or a
/// factor of one's own.
struct UpscaleSheet: View {
    @ObservedObject var viewModel: GenerationViewModel
    let pending: GenerationViewModel.PendingUpscale
    var close: () -> Void
    @State private var upscaler: VideoUpscaler = .standard
    @State private var choices: UpscaleChoices
    @State private var selection: Selection
    /// カスタム's field, as typed.
    @State private var customText: String
    /// Open-H3 再生成's strength.
    @State private var strength: OpenH3Strength = .standard
    @State private var showingSwiftVRSetup = false
    @State private var showingSeedVR2Setup = false
    @State private var showingOpenH3Setup = false
    /// Whether the chosen upscaler's model is here, as last checked. A
    /// setup finishing changes nothing else the sheet holds for Open-H3, so
    /// this is what redraws it.
    @State private var modelReady = true
    /// What Open-H3's factor applies to: the latent's own size, the canvas
    /// the clip was generated on (GenerationViewModel.openH3Base) - half of
    /// a ×2 size's video. Nil when the video has no latent.
    private let latentBase: (width: Int, height: Int)?
    @ObservedObject private var swiftVRSetup = SwiftVRSetup.shared
    @ObservedObject private var seedVR2Setup = SeedVR2Setup.shared
    @ObservedObject private var openH3Setup = OpenH3Setup.shared
    @Environment(\.colorScheme) private var colorScheme
    private var palette: H3Palette { H3Palette(colorScheme) }

    private enum Selection: Hashable {
        case usual(UpscaleFactor)
        case custom
    }

    init(viewModel: GenerationViewModel, pending: GenerationViewModel.PendingUpscale, close: @escaping () -> Void) {
        self.viewModel = viewModel
        self.pending = pending
        self.close = close
        latentBase = pending.projectVideo.flatMap { viewModel.openH3Base(for: $0) }
        let choices = UpscaleChoices(upscaler: .standard, width: pending.width, height: pending.height)
        _choices = State(initialValue: choices)
        // 2x when it fits, else the largest that does.
        let usual = choices.usual.contains(.two) ? UpscaleFactor.two : choices.usual.last
        _selection = State(initialValue: usual.map(Selection.usual) ?? .custom)
        _customText = State(initialValue: choices.range.map { UpscaleFactor.text(min($0.upperBound, 2.5)) } ?? "2.5")
    }

    /// What カスタム's field holds, as a factor (in range or not).
    private var typedFactor: UpscaleFactor? {
        let normal = (customText.applyingTransform(.fullwidthToHalfwidth, reverse: false) ?? customText)
            .replacingOccurrences(of: ",", with: ".")
            .trimmingCharacters(in: .whitespaces)
        return Double(normal).flatMap(UpscaleFactor.init(rawValue:))
    }

    /// The factor chosen, when it is one this upscaler can make.
    private var factor: UpscaleFactor? {
        switch selection {
        case .usual(let factor): return choices.usual.contains(factor) ? factor : nil
        case .custom:
            guard let typed = typedFactor, choices.range?.contains(typed.rawValue) == true else { return nil }
            return typed
        }
    }

    private func sizeText(_ factor: UpscaleFactor) -> String {
        guard let size = choices.size(factor) else { return String(localized: "拡大できません") }
        return "\(size.width)×\(size.height)"
    }

    /// The upscaler changed (or its model did): its own factors, keeping
    /// the one chosen when it can, else the next below it.
    private func updateChoices() {
        let old = factor
        modelReady = upscaler.isAvailable
        // Open-H3 enlarges the latent, so its factors are of the latent's size.
        let own = (width: pending.width, height: pending.height)
        let base = upscaler.regeneratesFromLatent ? latentBase ?? own : own
        choices = UpscaleChoices(upscaler: upscaler, width: base.width, height: base.height)
        switch selection {
        case .usual(let chosen):
            if !choices.usual.contains(chosen) {
                let below = choices.usual.last { $0.rawValue <= (old ?? chosen).rawValue } ?? choices.usual.first
                selection = below.map(Selection.usual) ?? .custom
            }
        case .custom:
            if let typed = typedFactor, let range = choices.range, !range.contains(typed.rawValue) {
                customText = UpscaleFactor.text(min(max(typed.rawValue, range.lowerBound), range.upperBound))
            }
        }
    }

    private func stepCustom(by step: Double) {
        guard let range = choices.range else { return }
        let value = ((typedFactor?.rawValue ?? range.lowerBound) + step)
        customText = UpscaleFactor.text(((min(max(value, range.lowerBound), range.upperBound)) * 100).rounded() / 100)
    }

    /// Why the Open-H3 choice can't run on this video (nil when it can).
    private var openH3Reason: String? {
        guard upscaler.regeneratesFromLatent else { return nil }
        guard let video = pending.projectVideo else {
            return String(localized: "Open-H3 再生成は、このアプリで生成した動画（記録と潜在ファイルのあるもの）にだけ使えます。")
        }
        return viewModel.openH3Reason(for: video, factor: factor ?? .same)
    }

    /// Whether Open-H3 starts from another size than the video's (a ×2
    /// size's video is twice its latent), which the sheet then says.
    private var regeneratesFromOtherSize: Bool {
        guard upscaler.regeneratesFromLatent, let latentBase else { return false }
        return latentBase.width != pending.width || latentBase.height != pending.height
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("動画を拡大").font(.headline)
            Text("\(pending.url.lastPathComponent)：\("\(pending.width)×\(pending.height)") → \(factor.map(sizeText) ?? String(localized: "拡大できません"))")
                .font(.callout)
                .monospacedDigit()
            Picker("拡大の方法", selection: $upscaler) {
                ForEach(VideoUpscaler.allCases) { option in
                    Text(option.label).tag(option)
                }
            }
            .pickerStyle(.menu)
            methodNotes
            factorChoice
            HStack {
                Spacer()
                Button("キャンセル") { close() }
                    .keyboardShortcut(.cancelAction)
                Button("拡大する") {
                    guard let scale = factor else { return }
                    let chosen = upscaler
                    close()
                    viewModel.startUpscale(pending, upscaler: chosen, factor: scale, strength: strength)
                }
                .keyboardShortcut(.defaultAction)
                .disabled(factor == nil || !modelReady || viewModel.isUpscalingVideo
                          || viewModel.upscaleBlockedReason(upscaler) != nil || openH3Reason != nil)
            }
        }
        .padding(H3Spacing.lg)
        .frame(width: 500)
        .onAppear {
            if let last = VideoUpscaler.lastChosen { upscaler = last }
            updateChoices()
        }
        .onChange(of: upscaler) { _, chosen in
            VideoUpscaler.lastChosen = chosen
            updateChoices()
        }
        .onChange(of: swiftVRSetup.phase) { _, _ in updateChoices() }
        .onChange(of: seedVR2Setup.phase) { _, _ in updateChoices() }
        .onChange(of: openH3Setup.phase) { _, _ in updateChoices() }
        .sheet(isPresented: $showingSwiftVRSetup, onDismiss: updateChoices) {
            SwiftVRSetupView(busy: viewModel.isGenerating || viewModel.isBuildingCache) { showingSwiftVRSetup = false }
        }
        .sheet(isPresented: $showingSeedVR2Setup, onDismiss: updateChoices) {
            SeedVR2SetupView(busy: viewModel.isGenerating || viewModel.isBuildingCache) { showingSeedVR2Setup = false }
        }
        .sheet(isPresented: $showingOpenH3Setup, onDismiss: updateChoices) {
            OpenH3SetupView(busy: viewModel.isGenerating || viewModel.isBuildingCache) { showingOpenH3Setup = false }
        }
    }

    /// About the upscaler chosen: what it does, why it can't run, its model.
    private var methodNotes: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(upscaler.detail)
            if let reason = upscaler.unavailableReason {
                Text(reason).foregroundStyle(.orange)
            } else if let reason = viewModel.upscaleBlockedReason(upscaler) {
                Text(reason).foregroundStyle(.orange)
            }
            // About the video itself, so said even while the model is missing.
            if let reason = openH3Reason {
                Text(reason).foregroundStyle(.orange)
            }
            if upscaler.isSwiftVR {
                HStack {
                    Button(modelReady ? "モデルを管理…" : "モデルを準備…") { showingSwiftVRSetup = true }
                        .controlSize(.small)
                    if swiftVRSetup.isBusy {
                        Text("準備中です（\(swiftVRSetup.phase.apiText)）").foregroundStyle(.orange)
                    }
                }
            }
            if upscaler == .seedVR2 {
                HStack {
                    Button(modelReady ? "モデルを管理…" : "モデルを準備…") { showingSeedVR2Setup = true }
                        .controlSize(.small)
                    if seedVR2Setup.isBusy {
                        Text("準備中です（\(seedVR2Setup.phase.apiText)）").foregroundStyle(.orange)
                    }
                }
            }
            if upscaler == .openH3 {
                HStack {
                    Button(modelReady ? "モデルを管理…" : "モデルを準備…") { showingOpenH3Setup = true }
                        .controlSize(.small)
                    if openH3Setup.isBusy {
                        Text("準備中です（\(openH3Setup.phase.apiText)）").foregroundStyle(.orange)
                    }
                }
            }
        }
        .font(.caption)
        .foregroundStyle(palette.textSecondary)
        .fixedSize(horizontal: false, vertical: true)
    }

    /// The factors of the upscaler chosen, カスタム among them.
    @ViewBuilder
    private var factorChoice: some View {
        if let range = choices.range {
            let low = UpscaleFactor.text(range.lowerBound), high = UpscaleFactor.text(range.upperBound)
            Picker("倍率", selection: $selection) {
                ForEach(choices.usual) { option in
                    Text("\(option.label)（\(sizeText(option))）").tag(Selection.usual(option))
                }
                if range.lowerBound < range.upperBound {
                    Text("カスタム（\(low)〜\(high)倍）").tag(Selection.custom)
                }
            }
            .pickerStyle(.radioGroup)
            if selection == .custom {
                HStack(spacing: 6) {
                    TextField("倍率", text: $customText)
                        .labelsHidden()
                        .textFieldStyle(.roundedBorder)
                        .multilineTextAlignment(.trailing)
                        .monospacedDigit()
                        .frame(width: 64)
                    Stepper("倍率", onIncrement: { stepCustom(by: 0.1) }, onDecrement: { stepCustom(by: -0.1) })
                        .labelsHidden()
                    Text("倍")
                    if let factor {
                        Text("拡大後：\(sizeText(factor))").monospacedDigit()
                    }
                }
            }
            if upscaler.regeneratesFromLatent {
                Picker("描き直しの強さ", selection: $strength) {
                    ForEach(OpenH3Strength.allCases) { option in
                        Text(option.label).tag(option)
                    }
                }
                .pickerStyle(.segmented)
            }
        }
        VStack(alignment: .leading, spacing: 4) {
            if regeneratesFromOtherSize, let latentBase {
                Text("Open-H3は、生成したときの大きさ（\("\(latentBase.width)×\(latentBase.height)")、潜在の格子）を基準に描き直します。倍率はその大きさに対するものです。")
            }
            if choices.range == nil {
                Text("この動画は、この方法では拡大できません。").foregroundStyle(.orange)
            } else {
                Text(choices.limitText)
                if factor == .same, !regeneratesFromOtherSize {
                    Text("等倍では大きさを変えず、細部を描き直します。")
                }
                if selection == .custom, factor == nil, let range = choices.range {
                    Text("\(UpscaleFactor.text(range.lowerBound))〜\(UpscaleFactor.text(range.upperBound))倍の範囲で、小数第2位まで指定できます。")
                        .foregroundStyle(.red)
                }
            }
            if upscaler.regeneratesFromLatent, openH3Reason == nil, let factor,
               let plan = OpenH3Plan.plan(width: choices.width, height: choices.height, factor: factor),
               let frames = pending.projectVideo?.record?.actualFrameCount {
                Text(strength.detail)
                Text("見込み：\("\(plan.width)×\(plan.height)")・\(frames)フレームで、M5で約\(formatElapsed(plan.estimatedSeconds(frames: frames, strength: strength)))（生成の設定による）。")
            }
            if !pending.inProject {
                Text("拡大した動画の保存先は、次に選びます。音声とファイルの情報は、そのままです。")
            } else if let factor {
                Text("拡大した動画は、プロジェクトに新しく加わります（名前の最後に「_\(factor.suffix)」）。音声とAIで生成したという記録は、そのままです。")
            }
        }
        .font(.caption)
        .foregroundStyle(palette.textSecondary)
        .fixedSize(horizontal: false, vertical: true)
    }
}
