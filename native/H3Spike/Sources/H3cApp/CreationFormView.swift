import H3Engine
import QuickLook
import SwiftUI

// Shown in grey while the prompt is empty; Tab turns it into real text.
private let promptSuggestion = "A cat playing with a ball of yarn."

struct CreationFormView: View {
    @ObservedObject var viewModel: GenerationViewModel
    @ObservedObject var library: ModelLibrary
    @Binding var showingModelManager: Bool
    @ObservedObject private var presetStore = PresetStore.shared
    // Floating-panel placement, owned by ContentView: offset from the
    // bottom-centre home position, clamped to `bounds` (the window).
    @Binding var offset: CGSize
    @Binding var isCollapsed: Bool
    var bounds: CGSize
    @Environment(\.colorScheme) private var colorScheme
    @State private var dragBase: CGSize?
    /// The panel's own height, so moving it (and opening it once moved)
    /// keeps all of it inside the window.
    @State private var panelHeight: CGFloat = 0
    /// The input file shown in Quick Look (clicking its name).
    @State private var quickLookURL: URL?
    @State private var showingClipEditor = false

    private var palette: H3Palette { H3Palette(colorScheme) }

    static let panelWidth: CGFloat = 640

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            dragHandle
            if !isCollapsed {
                topRow
                if viewModel.creationMethod == .image {
                    imageInputSection
                }
                if viewModel.continuation != nil {
                    continuationSection
                }
                lipSyncSection
                if viewModel.isChained {
                    ClipPromptsSection(viewModel: viewModel, showingEditor: $showingClipEditor)
                }
                promptSection
            }
            bottomBar
        }
        .padding(.horizontal, H3Spacing.lg)
        .padding(.bottom, H3Spacing.lg)
        .padding(.top, 6)
        .frame(width: Self.panelWidth)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(palette.border, lineWidth: 1))
        .shadow(color: .black.opacity(0.25), radius: 18, y: 6)
        .background(GeometryReader { geometry in
            Color.clear
                .onAppear { panelHeight = geometry.size.height }
                .onChange(of: geometry.size.height) { _, height in panelHeight = height }
        })
        // Opened (taller) or the window made smaller: back inside.
        .onChange(of: panelHeight) { _, _ in offset = clamped(offset) }
        .onChange(of: bounds) { _, _ in offset = clamped(offset) }
        .quickLookPreview($quickLookURL)
        .sheet(isPresented: $showingClipEditor) {
            CustomClipsSheet(viewModel: viewModel, isPresented: $showingClipEditor)
        }
    }

    // MARK: 移動・折りたたみ

    /// Drag to move the panel anywhere over the preview; double-click to
    /// put it back at the bottom centre.
    private var dragHandle: some View {
        ZStack {
            Capsule()
                .fill(palette.textSecondary.opacity(0.45))
                .frame(width: 44, height: 5)
            // Collapsed, the bottom row is the (much larger) open target.
            if !isCollapsed {
                HStack {
                    Spacer()
                    Button {
                        withAnimation(.easeInOut(duration: 0.15)) { isCollapsed = true }
                    } label: {
                        Image(systemName: "chevron.down")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(palette.textSecondary)
                            .frame(width: 40, height: 22)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help("パネルを折りたたむ")
                }
            }
        }
        .frame(maxWidth: .infinity)
        .frame(height: 22)
        .contentShape(Rectangle())
        .gesture(
            // Measured in the window: in the handle's own space the
            // translation shrinks as the panel moves under the pointer,
            // and the panel falls behind it.
            DragGesture(minimumDistance: 2, coordinateSpace: .global)
                .onChanged { value in
                    let base = dragBase ?? offset
                    dragBase = base
                    offset = clamped(CGSize(width: base.width + value.translation.width,
                                            height: base.height + value.translation.height))
                }
                .onEnded { _ in dragBase = nil }
        )
        .onTapGesture(count: 2) { withAnimation { offset = .zero } }
        .help("ドラッグで移動、ダブルクリックで元の位置へ")
    }

    /// The panel sits 20 pt above the window's bottom edge (ContentView);
    /// it may go up until its top is 8 pt below the window's top.
    private func clamped(_ proposed: CGSize) -> CGSize {
        let maxX = max(0, (bounds.width - Self.panelWidth) / 2 - 8)
        let height = panelHeight > 0 ? panelHeight : 140
        let maxUp = max(0, bounds.height - 20 - height - 8)
        return CGSize(width: min(max(proposed.width, -maxX), maxX),
                      height: min(max(proposed.height, -maxUp), 0))
    }

    // MARK: 作り方・プリセット・詳細設定

    private var presetMenu: some View {
        Menu {
            if presetStore.presets.isEmpty {
                Text("保存したプリセットはありません")
            } else {
                ForEach(presetStore.presets) { preset in
                    Button {
                        viewModel.applyPreset(preset)
                    } label: {
                        if preset.id == viewModel.activePresetID {
                            Label(preset.name, systemImage: "checkmark")
                        } else {
                            Text(preset.name)
                        }
                    }
                }
            }
            Divider()
            Button("現在の設定をプリセットとして保存…") {
                viewModel.presetPendingName = viewModel.presetFromCurrentSettings()
            }
            if !presetStore.presets.isEmpty {
                Menu("削除") {
                    ForEach(presetStore.presets) { preset in
                        Button(preset.name) { viewModel.deletePreset(id: preset.id) }
                    }
                }
            }
        } label: {
            // The preset in use, marked when the form has moved off it.
            if let active = viewModel.activePreset {
                Label(active.modified ? String(localized: "\(active.preset.name)・変更あり") : active.preset.name,
                      systemImage: "square.stack")
            } else {
                Label("プリセット", systemImage: "square.stack")
            }
        }
        .fixedSize()
        .help("形・大きさ・長さと詳細設定をまとめて保存・適用します（プロンプトは含みません）")
    }

    private var topRow: some View {
        HStack(spacing: H3Spacing.sm) {
            Picker("作り方", selection: $viewModel.creationMethod) {
                ForEach(CreationMethod.allCases) { method in
                    Text(method.label).tag(method)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(width: 200)

            Spacer(minLength: 0)

            presetMenu

            Button {
                viewModel.showingAdvancedSettings = true
            } label: {
                Label(viewModel.hasAdvancedChanges ? String(localized: "詳細設定・変更あり") : String(localized: "詳細設定"),
                      systemImage: "slider.horizontal.3")
            }
            .help("詳細設定を開く（⌘,）")
        }
    }

    // MARK: 画像入力

    private var imageInputSection: some View {
        VStack(alignment: .leading, spacing: H3Spacing.sm) {
            // Here rather than next to 文章から・画像から: both segmented
            // pickers plus プリセット and 詳細設定 were wider than the panel,
            // and the fixed-size row pushed out past both edges.
            Picker("画像の使い方", selection: $viewModel.imageInputMode) {
                ForEach(ImageInputMode.allCases) { mode in
                    Text(mode.label).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            switch viewModel.imageInputMode {
            case .firstLastFrame:
                Text("この画像から動き始めます。").font(.caption).foregroundStyle(palette.textSecondary)
                ImagePickerRow(label: String(localized: "最初の画像"), path: $viewModel.firstFramePath,
                               choose: viewModel.pickFirstFrame) { quickLookURL = $0 }
                if viewModel.lastFramePath != nil {
                    ImagePickerRow(label: String(localized: "最後の画像"), path: $viewModel.lastFramePath,
                                   choose: viewModel.pickLastFrame) { quickLookURL = $0 }
                } else {
                    Button("最後の画像も指定…") { viewModel.pickLastFrame() }
                        .font(.caption)
                }

            case .referenceImage:
                Text("画像・動画・音声の特徴を参考にして動画をつくります。最初のフレームが同じになるとは限りません。音声だけを参照にすることはできないので、画像か動画と組み合わせて追加してください。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                HStack {
                    Text("参照画像・動画・音声").font(.caption).foregroundStyle(palette.textSecondary)
                    Spacer()
                    Button("ファイルを選ぶ…") { viewModel.addReferenceImages() }
                }
                if !viewModel.referenceImages.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(Array(viewModel.referenceImages.enumerated()), id: \.element.id) { index, reference in
                            HStack {
                                Image(systemName: referenceIcon(for: reference.kind))
                                    .foregroundStyle(palette.textSecondary)
                                Text(verbatim: "\(index + 1).").font(.caption)
                                FileLink(path: reference.path) { quickLookURL = $0 }
                                Spacer()
                                Button("取り除く") {
                                    viewModel.removeReferenceImages(at: IndexSet(integer: index))
                                }
                                .font(.caption)
                            }
                        }
                    }
                }
            }
        }
        .padding(H3Spacing.md)
        .background(palette.surfaceMuted)
        .clipShape(RoundedRectangle(cornerRadius: H3Radius.control))
    }

    // MARK: 続きを作る (continuation)

    @ViewBuilder
    private var continuationSection: some View {
        if let continuation = viewModel.continuation {
            VStack(alignment: .leading, spacing: H3Spacing.sm) {
                Text("この動画の続きをつくります。最後の\(continuation.frames)コマ（約0.9秒）を引き継ぐので、できる動画の最初の部分は元の動画の終わりと同じになります。つなぐときは「プロジェクトの動画」の「ここまでをつないで書き出す」を使います。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                if let enlargement = continuation.enlargement {
                    Text("拡大した動画の続きです。続きは元の大きさでつくってから、同じように\(enlargement.factor.label)に拡大します（\(enlargement.upscaler.label)）。")
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                HStack {
                    Image(systemName: "arrow.turn.down.right").foregroundStyle(palette.textSecondary)
                    Text("続き").font(.caption)
                    FileLink(path: continuation.videoURL.path) { quickLookURL = $0 }
                    Spacer()
                    Button("取り除く") { viewModel.continuation = nil }
                }
            }
            .padding(H3Spacing.md)
            .background(palette.surfaceMuted)
            .clipShape(RoundedRectangle(cornerRadius: H3Radius.control))
        }
    }

    // MARK: 音声に合わせる (lip sync)

    /// Forced audio: the chosen file's audio becomes the soundtrack and the
    /// picture is generated to match it. A one-line button until a file is
    /// chosen, so the panel stays as small as before for everyone else.
    @ViewBuilder
    private var lipSyncSection: some View {
        if let path = viewModel.forcedAudioPath {
            VStack(alignment: .leading, spacing: H3Spacing.sm) {
                Text("この音声に合わせて、口の動きなどの映像をつくります。動画の音は、この音声そのものになります。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                HStack {
                    Image(systemName: "waveform").foregroundStyle(palette.textSecondary)
                    Text("音声").font(.caption)
                    FileLink(path: path) { quickLookURL = $0 }
                    Spacer()
                    TextField("開始位置", value: $viewModel.forcedAudioOffset,
                              format: .number.precision(.fractionLength(0 ... 2)))
                        .textFieldStyle(.roundedBorder)
                        .multilineTextAlignment(.trailing)
                        .frame(width: 64)
                        .help("音声の何秒目から使うか")
                    Text("秒から").font(.caption)
                    Button("変更…") { viewModel.pickForcedAudio() }
                    Button("取り除く") { viewModel.forcedAudioPath = nil }
                }
            }
            .padding(H3Spacing.md)
            .background(palette.surfaceMuted)
            .clipShape(RoundedRectangle(cornerRadius: H3Radius.control))
        } else {
            Button {
                viewModel.pickForcedAudio()
            } label: {
                Label("音声に合わせる（リップシンク）…", systemImage: "waveform")
            }
            .font(.caption)
            .help("音声ファイルを選ぶと、その音声に合わせた映像をつくります")
        }
    }

    private func referenceIcon(for kind: H3ReferenceKind) -> String {
        switch kind {
        case .video, .videoAudio: return "video"
        case .audio: return "waveform"
        case .image: return "photo"
        }
    }

    // MARK: 動画の内容 (プロンプト)

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: H3Spacing.sm) {
            // Composer layout: the basic shape/size/length choices and the
            // generate button live in a toolbar along the bottom edge of the
            // prompt box. Return generates, Shift+Return starts a new line.
            VStack(spacing: 0) {
                ZStack(alignment: .topLeading) {
                    PromptTextView(text: $viewModel.prompt, suggestion: promptSuggestion) {
                        viewModel.requestGenerate()
                    }
                    .frame(height: 84)
                    if viewModel.prompt.isEmpty {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(promptSuggestion)
                                .font(.body)
                            Text("Tabで入力 ・ Enterで生成 ・ Shift+Enterで改行")
                                .font(.caption)
                        }
                        .foregroundStyle(palette.textSecondary.opacity(0.75))
                        .padding(.horizontal, 11)
                        .padding(.vertical, 8)
                        .allowsHitTesting(false)
                    }
                }
                HStack(spacing: H3Spacing.sm) {
                    aspectMenu
                    sizeMenu
                    durationMenu
                    // Several videos are kept in a project: without one,
                    // generating asks for it (BatchProjectSheet).
                    batchMenu
                    Spacer(minLength: 0)
                    continueButtons
                    generateButton
                }
                .padding(.horizontal, 8)
                .padding(.bottom, 8)
            }
            .background(palette.surface)
            .clipShape(RoundedRectangle(cornerRadius: H3Radius.editor))
            .overlay(
                RoundedRectangle(cornerRadius: H3Radius.editor)
                    .stroke(palette.border, lineWidth: 1)
            )

        }
    }

    // MARK: 画面の形・長さ (プロンプト欄の下端)

    private var aspectMenu: some View {
        Menu {
            Picker("画面の形", selection: Binding(
                get: { viewModel.sizeProfile.shape },
                set: { shape in
                    // Keep 小/中/大 and the upscale when the shape changes; a
                    // custom size goes back to the presets at 中.
                    let current = viewModel.sizeProfile
                    viewModel.sizeProfile = .preset(shape, current.tier ?? .medium,
                                                    upscaled: current.upscaled)
                }
            )) {
                ForEach(AspectShape.allCases) { shape in
                    Label(shape.label, systemImage: shape.systemImage).tag(shape)
                }
            }
            .pickerStyle(.inline)
        } label: {
            composerChip(systemImage: viewModel.sizeProfile.shape.systemImage,
                         text: viewModel.sizeProfile.shape.label)
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("画面の形")
    }

    private var sizeMenu: some View {
        Menu {
            let current = viewModel.sizeProfile
            Picker("大きさ", selection: Binding(
                get: { current.tier },
                set: { tier in
                    guard let tier else { return }
                    viewModel.sizeProfile = .preset(current.shape, tier, upscaled: current.upscaled)
                }
            )) {
                ForEach(SizeTier.allCases) { tier in
                    let profile = SizeProfile.preset(current.shape, tier, upscaled: current.upscaled)
                    Text("\(tier.label)（\(profile.resolutionLabel)）").tag(Optional(tier))
                }
                if current.isCustom {
                    Text("カスタム（\(current.resolutionLabel)）").tag(SizeTier?.none)
                }
            }
            .pickerStyle(.inline)
            Divider()
            Toggle(isOn: Binding(
                get: { current.upscaled },
                set: { viewModel.sizeProfile = viewModel.sizeProfile.withUpscale($0) }
            )) {
                Text("2倍に拡大（\(current.canvasLabel)で生成）")
            }
        } label: {
            composerChip(systemImage: "arrow.up.left.and.arrow.down.right", text: sizeChipText)
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("大きさ（\(viewModel.sizeProfile.label)）")
    }

    private var sizeChipText: String {
        let profile = viewModel.sizeProfile
        let name = profile.tier?.label ?? String(localized: "カスタム")
        return profile.upscaled ? String(localized: "\(name) ×2") : name
    }

    /// 1-15 s in one generation, and つなげて生成: 15 s of equal clips, or
    /// カスタムクリップ laid out clip by clip.
    private var durationMenu: some View {
        Menu {
            Picker("1回で生成", selection: durationSelection) {
                ForEach(secondsRange, id: \.self) { value in
                    Text("\(value) 秒").tag("single-\(value)")
                }
            }
            .pickerStyle(.inline)
            Picker("つなげて生成", selection: durationSelection) {
                Text("つないで15秒").tag("chain-\(chainedSecondsRange.lowerBound)")
            }
            .pickerStyle(.inline)
            // A toggle, for the menu's check mark while it is in use.
            Toggle("カスタムクリップ…", isOn: Binding(
                get: { viewModel.usesCustomClips },
                set: { _ in
                    viewModel.beginEditingClips()
                    showingClipEditor = true
                }))
        } label: {
            composerChip(systemImage: viewModel.isChained ? "film.stack" : "clock",
                         text: viewModel.durationLabel)
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("長さ")
    }

    /// After a generation: carry on from the video in the preview, or join
    /// a continuation with what it continues.
    @ViewBuilder
    private var continueButtons: some View {
        if let video = viewModel.joinableDisplayedVideo {
            Button {
                viewModel.exportJoined(endingAt: video)
            } label: {
                composerChip(systemImage: "film.stack", text: String(localized: "つないで書き出す"), menu: false)
            }
            .buttonStyle(.plain)
            .help("この動画と、その前の動画（続きの元）を、重なりでつないで1本にします")
        }
        if viewModel.canContinueDisplayedVideo {
            Button {
                viewModel.continueDisplayedVideo()
            } label: {
                composerChip(systemImage: "arrow.turn.down.right", text: String(localized: "続きを生成"), menu: false)
            }
            .buttonStyle(.plain)
            .help(viewModel.project == nil
                  ? String(localized: "表示中の動画の続きをつくります。先に、この動画を入れるプロジェクトを作ります")
                  : String(localized: "表示中の動画の続きをつくります（最後の約0.9秒を引き継ぎます）"))
        }
    }

    private var durationSelection: Binding<String> {
        Binding(
            get: {
                viewModel.usesCustomClips ? "custom"
                    : (viewModel.isChained ? "chain-" : "single-") + String(viewModel.seconds)
            },
            set: { tag in
                let parts = tag.split(separator: "-")
                guard parts.count == 2, let value = Int(parts[1]) else { return }
                viewModel.setDuration(seconds: value, chained: parts[0] == "chain")
            })
    }

    private var batchMenu: some View {
        Menu {
            Picker("本数", selection: $viewModel.batchCount) {
                ForEach([1, 2, 3, 4, 5, 6, 8, 10, 15, 20], id: \.self) { value in
                    Text("\(value) 本").tag(value)
                }
            }
            .pickerStyle(.inline)
        } label: {
            composerChip(systemImage: "square.stack.3d.down.right",
                         text: String(localized: "\(viewModel.batchCount)本"))
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .disabled(viewModel.isChained)
        .help(viewModel.isChained
              ? String(localized: "つなげて生成では1本ずつ作ります")
              : String(localized: "同じプロンプトと設定で、シードを変えて続けて生成し、プロジェクトに保存します（シード固定のときは1本目だけ固定し、2本目以降はランダム）。プロジェクトがないときは、生成するときに作成または選択します"))
    }

    private func composerChip(systemImage: String, text: String, menu: Bool = true) -> some View {
        HStack(spacing: 5) {
            Image(systemName: systemImage)
            Text(text)
            if menu {
                Image(systemName: "chevron.down").font(.system(size: 9, weight: .semibold))
            }
        }
        .font(.callout)
        .foregroundStyle(palette.textPrimary)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(palette.surfaceMuted)
        .clipShape(Capsule())
    }

    // MARK: 下部固定領域

    private var bottomBar: some View {
        HStack(spacing: H3Spacing.sm) {
            if isCollapsed {
                // Collapsed, the whole row (all but the generate button)
                // opens the panel - the chevron alone was too small a target.
                Button(action: expand) {
                    HStack(spacing: H3Spacing.sm) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(collapsedPromptLine)
                                .font(.callout)
                                .foregroundStyle(viewModel.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? palette.textSecondary : palette.textPrimary)
                                .lineLimit(1)
                            statusText.lineLimit(1)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        Image(systemName: "chevron.up")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(palette.textSecondary)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("クリックでパネルを広げる")

                // The prompt box (and its buttons) is hidden while collapsed;
                // 続きを生成 shows here too, as a finished video collapses it.
                continueButtons
                generateButton
            } else {
                statusText
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var statusText: some View {
        Group {
            // An empty prompt already shows its grey suggestion; no need
            // for a red error line as well.
            if let message = viewModel.validationMessage, !viewModel.promptIsEmpty {
                Text(message).foregroundStyle(palette.errorColor)
            } else {
                Text(([viewModel.draftSummaryText] + [viewModel.draftEstimateText].compactMap { $0 })
                    .joined(separator: summarySeparator))
                    .foregroundStyle(palette.textSecondary)
                    .help("このMacでの目安は、いまの設定とこれまでの生成実績から計算しています")
            }
        }
        .font(.caption)
    }

    private var collapsedPromptLine: String {
        let firstLine = viewModel.prompt
            .split(whereSeparator: \.isNewline)
            .first { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        return firstLine.map(String.init) ?? String(localized: "プロンプトを入力")
    }

    private func expand() {
        withAnimation(.easeInOut(duration: 0.15)) { isCollapsed = false }
    }

    /// Round icon at the prompt's bottom-right: generate, or stop while a
    /// generation runs. ⌘Return also generates.
    private var generateButton: some View {
        Button {
            if viewModel.isGenerating { viewModel.cancel() } else { viewModel.requestGenerate() }
        } label: {
            Image(systemName: viewModel.isGenerating ? "stop.fill" : "arrow.up")
                .font(.system(size: 14, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 30, height: 30)
                .background(Circle().fill(generateEnabled ? palette.accent : palette.textSecondary.opacity(0.4)))
        }
        .buttonStyle(.plain)
        .disabled(!generateEnabled)
        .keyboardShortcut(.return, modifiers: .command)
        .help(viewModel.isGenerating ? String(localized: "中止") : String(localized: "動画をつくる（Enter / ⌘Return）"))
        .accessibilityLabel(viewModel.isGenerating ? String(localized: "中止") : String(localized: "動画をつくる"))
    }

    private var generateEnabled: Bool {
        viewModel.isGenerating ? !viewModel.isCancelling : viewModel.canGenerate
    }
}

private struct ImagePickerRow: View {
    let label: String
    @Binding var path: String?
    let choose: () -> Void
    let preview: (URL) -> Void

    var body: some View {
        HStack {
            Text(label).font(.caption).frame(width: 84, alignment: .leading)
            if let path {
                FileLink(path: path, preview: preview)
            } else {
                Text("未選択").font(.caption)
            }
            Spacer()
            Button("変更…", action: choose)
            if path != nil {
                Button("取り除く") { path = nil }
            }
        }
    }
}

/// An input file's name as a link: clicking it shows the file in Quick
/// Look (image, video or audio) without leaving the app.
private struct FileLink: View {
    let path: String
    let preview: (URL) -> Void
    @State private var hovering = false

    var body: some View {
        Button {
            preview(URL(fileURLWithPath: path))
        } label: {
            Text(URL(fileURLWithPath: path).lastPathComponent)
                .font(.caption)
                .underline(hovering)
                .foregroundStyle(Color.accentColor)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .buttonStyle(.plain)
        .onHover { inside in
            hovering = inside
            if inside { NSCursor.pointingHand.push() } else { NSCursor.pop() }
        }
        .help("クリックで内容を確認")
    }
}

