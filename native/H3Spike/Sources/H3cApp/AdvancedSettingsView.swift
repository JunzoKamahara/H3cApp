import H3Engine
import SwiftUI

/// 詳細設定 as its own dialog, opened from the composer's button. Everything here applies from the next generation.
struct AdvancedSettingsView: View {
    @ObservedObject var viewModel: GenerationViewModel
    @ObservedObject var library: ModelLibrary
    @Binding var showingModelManager: Bool
    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.dismiss) private var dismiss

    private var palette: H3Palette { H3Palette(colorScheme) }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("詳細設定").font(.system(size: 16, weight: .semibold))
                if viewModel.hasAdvancedChanges {
                    Text("変更あり").font(.caption).foregroundStyle(palette.accent)
                }
                Spacer()
            }
            .padding(H3Spacing.lg)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: H3Spacing.lg) {
                    sizeSection
                    VStack(alignment: .leading, spacing: 4) {
                        Text("生成ステップ数").font(.caption).foregroundStyle(palette.textSecondary)
                        Picker("生成ステップ数", selection: $viewModel.steps) {
                            ForEach(stepsRange, id: \.self) { value in
                                Text("\(value)").tag(value)
                            }
                        }
                        .pickerStyle(.menu)
                        .labelsHidden()
                        .frame(maxWidth: 120)
                    }
                    computeModeSection
                    speedSection
                    reuseSection
                    layersSection
                    chainSection
                    seedSection
                    loraSection
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(H3Spacing.lg)
            }

            Divider()

            HStack {
                Button("既定に戻す") { viewModel.resetAdvancedSettings() }
                Spacer()
                if viewModel.isGenerating {
                    Text("変更は次の生成から適用されます").font(.caption).foregroundStyle(palette.textSecondary)
                }
                Button("閉じる") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
            .padding(H3Spacing.lg)
        }
        .frame(width: 540, height: 640)
        .background(palette.canvas)
    }

    // MARK: サイズ

    /// Custom finished size; プロンプト欄の「大きさ」 picks a preset again.
    private var sizeSection: some View {
        let profile = viewModel.sizeProfile
        return VStack(alignment: .leading, spacing: 4) {
            Text("サイズ").font(.caption).foregroundStyle(palette.textSecondary)
            Toggle("カスタムサイズを使う", isOn: Binding(
                get: { profile.isCustom },
                set: { custom in
                    // Start from the preset's finished size and its 2x setting.
                    let output = viewModel.sizeProfile.dimensions
                    viewModel.sizeProfile = custom
                        ? .custom(width: output.width, height: output.height, upscaled: profile.upscaled)
                        : .preset(profile.shape, .medium, upscaled: profile.upscaled)
                }
            ))
            if profile.isCustom {
                HStack(spacing: H3Spacing.md) {
                    sideField("幅", value: customSide(width: true))
                    Text(verbatim: "×").foregroundStyle(palette.textSecondary)
                    sideField("高さ", value: customSide(width: false))
                }
                Toggle("半分のサイズで生成して2倍に拡大", isOn: Binding(
                    get: { viewModel.sizeProfile.upscaled },
                    set: { viewModel.sizeProfile = viewModel.sizeProfile.withUpscale($0) }
                ))
                let pixels = profile.ditPixels
                Text("完成 \(profile.resolutionLabel) ・ 生成 \(profile.canvasLabel)（\(pixels.formatted())ピクセル / 上限 \(SizeProfile.customMaxPixels.formatted())）")
                    .font(.caption)
                    .foregroundStyle(profile.isWithinCustomLimit ? palette.textSecondary : palette.errorColor)
                if !profile.isWithinCustomLimit && !profile.upscaled {
                    Text("このままでは生成できません。サイズを小さくするか、「半分のサイズで生成して2倍に拡大」をオンにしてください。")
                        .font(.caption)
                        .foregroundStyle(palette.errorColor)
                }
                Text("幅・高さは完成する動画のサイズです（32の倍数、2倍の拡大では64の倍数）。生成するサイズ（2倍の拡大ではその半分）が\(SizeProfile.customMaxPixels.formatted())ピクセル以下になるようにしてください。拡大は生成後に行うため、時間はほとんど増えません。プロンプト欄の「大きさ」を選ぶとプリセットに戻ります。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            } else {
                Text("現在: \(profile.label)。プリセットはプロンプト欄の「画面の形」「大きさ」で選びます。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            }
            if let estimate = viewModel.draftEstimatedSeconds {
                let minutes = max(Int((estimate / 60).rounded(.up)), 1)
                Text("このMacでの目安: 約\(minutes)分（いまの設定・これまでの生成実績から）")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                if estimate > GenerationViewModel.longGenerationSeconds {
                    Text("完成まで1時間以上かかる見込みです。サイズ・長さ・ステップ数を下げることを検討してください。")
                        .font(.caption)
                        .foregroundStyle(palette.errorColor)
                }
            }
        }
    }

    /// One side of the custom finished size, rounded to its step within
    /// its range (both depend on the 2x upscale).
    private func customSide(width isWidth: Bool) -> Binding<Int> {
        Binding(
            get: {
                let output = viewModel.sizeProfile.dimensions
                return Int(isWidth ? output.width : output.height)
            },
            set: { newValue in
                let upscaled = viewModel.sizeProfile.upscaled
                let range = SizeProfile.customSideRange(upscaled: upscaled)
                let step = Int(SizeProfile.customSideStep(upscaled: upscaled))
                let rounded = Int32(clamping: ((newValue + step / 2) / step) * step)
                let value = min(max(rounded, range.lowerBound), range.upperBound)
                let output = viewModel.sizeProfile.dimensions
                viewModel.sizeProfile = .custom(width: isWidth ? value : output.width,
                                                height: isWidth ? output.height : value,
                                                upscaled: upscaled)
            })
    }

    private func sideField(_ title: LocalizedStringKey, value: Binding<Int>) -> some View {
        let upscaled = viewModel.sizeProfile.upscaled
        let range = SizeProfile.customSideRange(upscaled: upscaled)
        return HStack(spacing: 4) {
            Text(title).font(.caption).foregroundStyle(palette.textSecondary)
            TextField(title, value: value, format: .number.grouping(.never))
                .labelsHidden()
                .frame(width: 64)
            Stepper(title, value: value, in: Int(range.lowerBound) ... Int(range.upperBound),
                    step: Int(SizeProfile.customSideStep(upscaled: upscaled)))
                .labelsHidden()
        }
    }

    private var computeModeSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("計算方式").font(.caption).foregroundStyle(palette.textSecondary)
            Picker("計算方式", selection: $viewModel.computeMode) {
                ForEach(ComputeMode.allCases) { mode in
                    Text(mode.label).tag(mode)
                        .disabled(mode == .attentionCache && !viewModel.supportsInt8Cache)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .frame(maxWidth: 260)
            switch viewModel.computeMode {
            case .attentionCache:
                Text("事前に作ったint8キャッシュを読みながら計算します。速く動作します。キャッシュと、Tensor演算ユニットを搭載したGPU（M5以降）が必要です。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            case .resident:
                Text("キャッシュファイルを作らず、起動のたびにモデル全体をメモリ上に展開して計算します。Tensor演算ユニット搭載GPU（M5以降）ではint8に量子化して常駐、それ以外ではBF16のまま常駐するため、大容量メモリのMac向けです。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            case .ssdStreaming:
                Text("元のBF16モデルを、必要なブロックだけSSDから読みながら計算します。メモリは少なくて済みますが遅くなります。キャッシュは使いません。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            }
            if viewModel.supportsInt8Cache && viewModel.computeMode == .attentionCache {
                attentionCacheBuildSection
            }
            if viewModel.isHeavySsdStreamingConfig {
                Text("最大解像度・長い秒数・SSDストリーミングの組み合わせは、メモリ不足でスワップが発生し非常に遅くなることがあります（数時間かかる場合も）。解像度か秒数を下げることをおすすめします。")
                    .font(.caption)
                    .foregroundStyle(palette.errorColor)
            }
            if viewModel.isLowMemoryForResident {
                Text("このMacの物理メモリでは常駐モードがスワップを起こす可能性があります。int8キャッシュ方式かSSDストリーミングをおすすめします。")
                    .font(.caption)
                    .foregroundStyle(palette.errorColor)
            }
        }
    }

    private var speedSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("速度").font(.caption).foregroundStyle(palette.textSecondary)
            Picker("速度", selection: $viewModel.speedMode) {
                ForEach(SpeedMode.allCases) { mode in
                    Text(mode.label).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(maxWidth: 300)
            Text(speedDescription)
                .font(.caption)
                .foregroundStyle(palette.textSecondary)
            if viewModel.fastAttentionAvailable {
                Toggle("高速モード（試験的）", isOn: $viewModel.fastAttention)
                    .toggleStyle(.checkbox)
                    .padding(.top, 6)
                Text("M5のニューラルアクセラレータでAttentionをint8計算します。長い動画ほど効果が大きく（15秒・20ステップで約1.4倍速、メモリ約1.3GB増）、短い動画では効果が小さく、条件によっては遅くなることがあります。同じシードでも標準とは映像が変わります。上の速度設定やreuseとも併用できますが、その組み合わせでの効果と画質は十分に検証していません。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                if viewModel.isGenerating {
                    Text("生成中の変更は次の生成から適用されます。")
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                }
            }
        }
    }

    private var speedDescription: String {
        switch viewModel.speedMode {
        case .quality:
            return String(localized: "省略なしで計算します。")
        case .fast:
            return String(localized: "ステップ間でTransformerの計算結果を使い回します（5秒・20ステップのDiT部分で約2.7倍速）。画質は保たれますが、同じシードでも構図は標準と変わります。")
        case .fastest:
            return String(localized: "高速に加えて、一部のトークンをまとめて計算します（5秒・20ステップのDiT部分で約2.9倍速）。細部がわずかに甘くなることがあります。")
        }
    }

    /// Shown under the compute-mode picker when the current mode/reference
    /// selection needs a cache that doesn't exist yet on disk - lets the
    /// user build it in-app instead of just being told it's missing.
    private var attentionCacheBuildSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            if viewModel.isBuildingCache {
                HStack {
                    ProgressView(value: viewModel.cacheBuildProgress)
                        .frame(maxWidth: 200)
                    Text(viewModel.cacheBuildProgress.map { String(localized: "\(Int($0 * 50))/50 ブロック") } ?? String(localized: "準備中…"))
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                    Spacer()
                    Button("中止") { viewModel.cancelCacheBuild() }
                        .font(.caption)
                }
                Text("キャッシュを作成しています。モデルの重みを読み込んで量子化するため、数十秒〜1分程度かかります。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            } else if viewModel.attentionCacheMissing {
                HStack {
                    Text("int8キャッシュがまだありません。")
                        .font(.caption)
                        .foregroundStyle(palette.errorColor)
                    Spacer()
                    Button("キャッシュを作成…") { viewModel.buildMissingAttentionCache() }
                        .font(.caption)
                }
            }
            if let cacheBuildError = viewModel.cacheBuildError {
                Text(cacheBuildError)
                    .font(.caption)
                    .foregroundStyle(palette.errorColor)
            }
        }
    }

    private var reuseSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("ノイズ除去の再利用（reuse）").font(.caption).foregroundStyle(palette.textSecondary)
            Picker("ノイズ除去の再利用", selection: $viewModel.denoiseReuse) {
                ForEach(reuseRange, id: \.self) { value in
                    Text(reuseLabel(value)).tag(value)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .frame(maxWidth: 200)
            .disabled(viewModel.speedMode != .quality)
            Text("N回に1回だけモデルを計算し、残りは前回までの結果から補って高速化します。大きいほど速く、品質は下がる可能性があります。")
                .font(.caption)
                .foregroundStyle(palette.textSecondary)
            if viewModel.speedMode != .quality {
                Text("速度が「\(viewModel.speedMode.label)」の間は、Transformerの使い回しと併用できないためreuse 1で計算します（設定値は保持され、「標準」に戻すと使われます）。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            }
            if viewModel.effectiveDenoiseReuse > 1 && viewModel.steps <= 7 {
                Text("ステップ数が少ないと、実際にモデルを計算する回数が極端に少なくなります。")
                    .font(.caption)
                    .foregroundStyle(palette.errorColor)
            }
        }
    }

    private var layersSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("使用する層数（DiTブロック）").font(.caption).foregroundStyle(palette.textSecondary)
            Picker("使用する層数", selection: $viewModel.ditLayers) {
                ForEach(Array(ditLayersRange.reversed()), id: \.self) { value in
                    Text(value == defaultDitLayers ? String(localized: "\(value)（全層・標準）") : "\(value)").tag(value)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .frame(maxWidth: 200)
            Text("50未満にすると、影響の小さいブロックから省いて高速化します。速度設定とは別に効きます。")
                .font(.caption)
                .foregroundStyle(palette.textSecondary)
            if viewModel.ditLayers < defaultDitLayers {
                Text("層を減らすと音声が壊れることがあります（45層で強い雑音に変わることを確認）。映像も変わります。")
                    .font(.caption)
                    .foregroundStyle(palette.errorColor)
            }
        }
    }

    private var chainSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("つなげて生成の1本の長さ").font(.caption).foregroundStyle(palette.textSecondary)
            Picker("つなげて生成の1本の長さ", selection: $viewModel.chainClipLength) {
                ForEach(ChainClipLength.allCases) { length in
                    Text(length == defaultChainClipLength
                         ? String(localized: "\(length.label)（標準）") : length.label).tag(length)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(maxWidth: 220)
            Text("15秒を超える長さ（と「つないで15秒」）は、この長さの動画を前の動画の続きとして順につくり、重なり（約0.9秒）でつないで1本にします。長いほど、つなぎ目が少なくなります。")
                .font(.caption)
                .foregroundStyle(palette.textSecondary)
        }
    }

    private func reuseLabel(_ value: Int) -> String {
        switch value {
        case 1: return String(localized: "1（標準）")
        case 2: return String(localized: "2（高速）")
        default: return String(localized: "\(value)（さらに高速）")
        }
    }

    private var seedSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("ランダムさ").font(.caption).foregroundStyle(palette.textSecondary)
            Picker("ランダムさ", selection: Binding(
                get: { !viewModel.seedFixed },
                set: { useRandom in viewModel.seedFixed = !useRandom }
            )) {
                Text("毎回変える").tag(true)
                Text("固定する").tag(false)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(maxWidth: 240)

            if viewModel.seedFixed {
                TextField("シード値", text: $viewModel.seedText)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 200)
                if let message = viewModel.seedValidationMessage {
                    Text(message).font(.caption).foregroundStyle(palette.errorColor)
                }
            }
        }
    }

    private var loraSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("追加モデル（LoRA）").font(.caption).foregroundStyle(palette.textSecondary)
                Spacer()
                Button("管理…") {
                    // One sheet at a time: close this dialog, then open the
                    // model manager from the main window.
                    dismiss()
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { showingModelManager = true }
                }
                    .font(.caption)
            }
            if library.loras.isEmpty {
                Text("登録された追加モデルはありません。「管理…」から追加できます。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            } else {
                ForEach(library.loras) { entry in
                    loraStackRow(entry)
                }
                Text("オンにした追加モデルは重ねて適用されます。強さは学習時の効き具合に対する倍率です（空欄=1）。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            }
            let enabled = library.enabledLoRAs
            let turbo = enabled.filter { $0.recommendedSteps != nil }
            if let first = turbo.first, let turboSteps = first.recommendedSteps {
                Text("「\(first.name)」はTurbo用の追加モデルです。生成ステップ数を\(turboSteps)に合わせました（\(turboSteps)以外では品質が落ちます）。")
                    .font(.caption)
                    .foregroundStyle(viewModel.steps == turboSteps ? palette.textSecondary : palette.errorColor)
            }
            if let renoiseEntry = enabled.first(where: { library.sampling(for: $0).renoise }) {
                Text("「\(renoiseEntry.name)」の設定に合わせて、ステップの進め方をre-noise（LCM方式）にします（ノイズ除去の再利用は1になります）。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            }
            if viewModel.samplingSettings.audioShift > 0 {
                // As a String: a Float interpolated into a localized string
                // formats as %f.
                let shift = "\(viewModel.samplingSettings.audioShift)"
                Text("追加モデルの設定に合わせて、音声のシフト値を\(shift)にします（標準は3）。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            }
            if turbo.count > 1 {
                Text("Turbo用の追加モデルを複数重ねると効果が強くなりすぎます。1つにすることをおすすめします。")
                    .font(.caption)
                    .foregroundStyle(palette.errorColor)
            }
            if !enabled.isEmpty {
                Text("読み込むモデル（最初/最後の画像・参照画像・動画）に対応したファイルか、事前に確認できません。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            }
        }
    }

    private func loraStackRow(_ entry: LoRAEntry) -> some View {
        HStack(spacing: H3Spacing.sm) {
            Toggle(isOn: Binding(
                get: { entry.enabled },
                set: { library.setLoRAEnabled(id: entry.id, $0) }
            )) {
                Text(entry.name).lineLimit(1).truncationMode(.middle)
            }
            .toggleStyle(.checkbox)
            .disabled(!entry.enabled && library.enabledLoRAs.count >= maxStackedLoRAs)
            Spacer()
            if entry.enabled {
                Text("強さ").font(.caption).foregroundStyle(palette.textSecondary)
                TextField("1", text: Binding(
                    get: { entry.scaleText },
                    set: { library.setLoRAScale(id: entry.id, scaleText: $0) }
                ))
                .textFieldStyle(.roundedBorder)
                .frame(width: 56)
            }
        }
    }

}
