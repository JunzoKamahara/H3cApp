import H3Engine
import SwiftUI

// RefMods (saved reference latents): making one from a reference the form
// holds (RefModExtractSheet) and choosing the members of a bundle
// (RefModMemberPickerSheet). The files themselves are listed in モデル管理
// (ModelManagerView) and kept in defaultRefModDirectory.

/// What RefModExtractSheet makes a RefMod from: one reference row, or (「画像を
/// まとめてRefModに…」) several image rows stacked into one RefMod.
struct RefModSources: Identifiable {
    let id = UUID()
    let references: [H3ReferenceInput]
}

/// 「RefModとして保存…」: encodes one of the form's references with the
/// Ref2VA VAE and writes a RefMod file into the RefMods folder
/// (GenerationViewModel.extractRefMod). With several image rows it stacks
/// them (upstream's "stack": one latent frame per image, the way the
/// published identity RefMods are made). Runs on the engine's queue, so it
/// waits for a generation in flight.
struct RefModExtractSheet: View {
    let sources: [H3ReferenceInput]
    @ObservedObject var viewModel: GenerationViewModel
    @ObservedObject var library: ModelLibrary
    @Binding var isPresented: Bool
    @Environment(\.colorScheme) private var colorScheme

    @State private var name: String
    @State private var resolutionCap = 1024
    @State private var latentFrames = 16
    @State private var maxSeconds = 15
    /// The new RefMod takes the source's row (same place in the list); off,
    /// the source stays and the RefMod is added at the end. Both would
    /// otherwise reach the generation as two copies of one reference.
    @State private var replaceSource = true
    /// A stack: the rows that go in (all at first), in the list's order.
    @State private var chosen: Set<UUID>
    @State private var running = false
    @State private var errorMessage: String?

    private var palette: H3Palette { H3Palette(colorScheme) }

    init(sources: [H3ReferenceInput], viewModel: GenerationViewModel, library: ModelLibrary, isPresented: Binding<Bool>) {
        self.sources = sources
        self.viewModel = viewModel
        self.library = library
        _isPresented = isPresented
        _name = State(initialValue: sources.first.map {
            URL(fileURLWithPath: $0.path).deletingPathExtension().lastPathComponent } ?? "")
        _chosen = State(initialValue: Set(sources.map(\.id)))
        // A stack holds one latent frame per image: 512 px keeps each near
        // 256 tokens, as in the published identity RefMods.
        _resolutionCap = State(initialValue: sources.count > 1 ? 512 : 1024)
    }

    private var source: H3ReferenceInput { sources[0] }
    private var stacking: Bool { sources.count > 1 }
    private var chosenSources: [H3ReferenceInput] { sources.filter { chosen.contains($0.id) } }

    /// A video row that also brings its sound: the RefMod holds the picture
    /// only, so replacing the row would drop the sound.
    private var sourceCarriesAudio: Bool {
        !stacking && (source.kind == .videoAudio || source.audioPath != nil || source.includeEmbeddedAudio)
    }

    private var kind: H3RefModKind {
        if stacking { return .image }
        switch source.kind {
        case .video, .videoAudio: return .video
        case .audio: return .audio
        default: return .image
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text(stacking ? "画像をまとめてRefModにする" : "RefModとして保存").font(.system(size: 16, weight: .semibold))
            Text("参照を一度だけVAEで符号化して、小さなファイル（.safetensors）に保存します。保存したRefModは、参照の「RefModを追加…」からどの生成でも使えます。Qwenには見せないので、プロンプトの<Picture n>は要りません。")
                .font(.caption)
                .foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            if stacking {
                Text("選んだ画像を1枚ずつ符号化して、1つのRefModにまとめます（ComfyUI-MiniMaxH3Modの「stack」と同じ形）。同じキャラクターや物を、向きや表情を変えて1体ずつ写した画像を選ぶと効きます。ほかの画像は1枚目の画像の大きさに合わせ、はみ出す部分は中央で切り取ります。文字や枠、背景も一緒に写るので、キャラクターだけが写った画像を選んでください。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(sources) { reference in
                        Toggle(isOn: Binding(
                            get: { chosen.contains(reference.id) },
                            set: { if $0 { chosen.insert(reference.id) } else { chosen.remove(reference.id) } }
                        )) {
                            Text(verbatim: URL(fileURLWithPath: reference.path).lastPathComponent)
                                .font(.caption)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                        .toggleStyle(.checkbox)
                    }
                }
                Text("参照のトークンは画像の枚数分増えます（解像度の上限512 pxなら1枚あたり約256）。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text(verbatim: URL(fileURLWithPath: source.path).lastPathComponent)
                    .font(.caption)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Grid(alignment: .leading, horizontalSpacing: H3Spacing.md, verticalSpacing: H3Spacing.sm) {
                GridRow {
                    Text("名前")
                    TextField("名前", text: $name).textFieldStyle(.roundedBorder).frame(width: 260)
                }
                if kind != .audio {
                    GridRow {
                        Text("解像度の上限")
                        Picker("", selection: $resolutionCap) {
                            Text("512 px").tag(512)
                            Text("768 px").tag(768)
                            Text("1024 px").tag(1024)
                            Text("1536 px").tag(1536)
                        }
                        .labelsHidden()
                        .frame(width: 120)
                        .help("短い辺がこれより大きい画像・動画は縮めてから符号化します。拡大はしません。")
                    }
                    if kind == .video {
                        GridRow {
                            Text("使うコマ数")
                            Stepper(value: $latentFrames, in: 5 ... 97) {
                                Text("\(latentFrames)")
                            }
                            .help("動画全体から均等に選んだコマ数。5、22、39…に切り下げられます。")
                        }
                    }
                } else {
                    GridRow {
                        Text("最大秒数")
                        Stepper(value: $maxSeconds, in: 1 ... 15) {
                            Text("\(maxSeconds)")
                        }
                        .help("先頭からこの秒数までを使います（参照の音声は合計15秒まで）。")
                    }
                }
            }
            Toggle(stacking ? "まとめた参照をRefModに置き換える" : "元の参照をRefModに置き換える", isOn: $replaceSource)
                .disabled(sourceCarriesAudio)
                .help(stacking
                      ? String(localized: "オンなら、選んだ画像の行を外し、最初の行の位置に作成したRefModを入れます。オフなら、元の行を残してRefModを最後に加えます。")
                      : String(localized: "オンなら、作成したRefModが元の参照の行と入れ替わります。オフなら、元の行を残してRefModを最後に加えます。"))
            if sourceCarriesAudio {
                Text("この参照は動画の音声も使うので、元の行を残してRefModを最後に加えます（RefModは映像だけです）。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if running, let state = viewModel.refModExtraction {
                HStack(spacing: H3Spacing.sm) {
                    ProgressView().controlSize(.small)
                    Text(refModPhaseText(state))
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                }
            } else if viewModel.isGenerating {
                Text("生成が終わってから始まります。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            }
            if let errorMessage {
                Text(errorMessage)
                    .font(.caption)
                    .foregroundStyle(palette.errorColor)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Spacer()
                if running {
                    Button("中止") { viewModel.cancelRefModExtraction() }
                } else {
                    Button("キャンセル") { isPresented = false }
                    Button("作成") { start() }
                        .keyboardShortcut(.defaultAction)
                        .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty
                                  || (stacking && chosenSources.count < 2))
                }
            }
        }
        .padding(H3Spacing.lg)
        .frame(width: 440)
    }

    private func start() {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        try? FileManager.default.createDirectory(atPath: defaultRefModDirectory, withIntermediateDirectories: true)
        let picked = stacking ? chosenSources : [source]
        guard !stacking || picked.count >= 2 else { return }
        var request = H3RefModExtractRequest(
            sourcePath: picked[0].path,
            outputPath: ModelLibrary.freePath(in: defaultRefModDirectory, stem: trimmed, extension: "safetensors"),
            kind: kind)
        request.name = trimmed
        // Always Full: a compressed RefMod (pooled to a small grid) kept only
        // rough colours of a character in the tests of 2026-11-05, so the
        // app no longer offers it. Compressed files made elsewhere still load.
        request.mode = .full
        request.resolutionCap = resolutionCap
        request.latentFrames = latentFrames
        request.maxSeconds = maxSeconds
        if stacking { request.imagePaths = picked.map(\.path) }
        running = true
        errorMessage = nil
        Task {
            do {
                let path = try await viewModel.extractRefMod(request)
                let refmod = H3ReferenceInput(kind: .refmod, path: path)
                let ids = Set(picked.map(\.id))
                if replaceSource, !sourceCarriesAudio,
                   let index = viewModel.referenceImages.firstIndex(where: { ids.contains($0.id) }) {
                    // The first picked row's place; the other picked rows go.
                    viewModel.referenceImages[index] = refmod
                    viewModel.referenceImages.removeAll { ids.contains($0.id) }
                } else {
                    viewModel.referenceImages.append(refmod)
                }
                isPresented = false
            } catch H3EngineError.cancelled {
                errorMessage = String(localized: "中止しました")
            } catch {
                errorMessage = error.localizedDescription
            }
            running = false
        }
    }
}

/// The engine's extraction phases in the user's words.
func refModPhaseText(_ state: GenerationViewModel.RefModExtractionState) -> String {
    switch state.phase {
    case "refmod read": return String(localized: "ファイルを読んでいます…")
    case "refmod stack":
        return String(localized: "画像を符号化しています（\(state.image)/\(state.images)枚目）…")
    case "video VAE encoder" where state.images > 1:
        return String(localized: "画像を符号化しています（\(state.image)/\(state.images)枚目）…")
    case "video VAE encoder": return String(localized: "映像を符号化しています（\(state.completed)/\(state.total)）…")
    case "audio VAE encoder": return String(localized: "音声を符号化しています…")
    case "refmod write": return String(localized: "書き出しています…")
    default: return String(localized: "準備しています…")
    }
}

/// A bundle holds several references: which of them join the generation
/// (one row each, in the bundle's order). onAdd gets the chosen members'
/// indices; the form appends them to its references, カスタムクリップ to the
/// clip's.
struct RefModMemberPickerSheet: View {
    let choice: RefModBundleChoice
    var onCancel: () -> Void
    var onAdd: ([Int]) -> Void
    @Environment(\.colorScheme) private var colorScheme
    @State private var selected: Set<Int>

    private var palette: H3Palette { H3Palette(colorScheme) }

    init(choice: RefModBundleChoice, onCancel: @escaping () -> Void, onAdd: @escaping ([Int]) -> Void) {
        self.choice = choice
        self.onCancel = onCancel
        self.onAdd = onAdd
        _selected = State(initialValue: Set(choice.info.members.filter(\.supported).map(\.id)))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("バンドルの中身を選ぶ").font(.system(size: 16, weight: .semibold))
            Text(verbatim: choice.info.name.isEmpty ? URL(fileURLWithPath: choice.path).lastPathComponent : choice.info.name)
                .font(.caption)
            List {
                ForEach(choice.info.members) { member in
                    Toggle(isOn: Binding(
                        get: { selected.contains(member.id) },
                        set: { if $0 { selected.insert(member.id) } else { selected.remove(member.id) } }
                    )) {
                        Text(refmodMemberSummary(member, inBundle: true)).font(.caption)
                    }
                    .disabled(!member.supported)
                }
            }
            .listStyle(.inset)
            .frame(minHeight: 160, maxHeight: 280)
            HStack {
                Spacer()
                Button("キャンセル") { onCancel() }
                Button("追加") {
                    onAdd(choice.info.members.map(\.id).filter { selected.contains($0) })
                }
                .keyboardShortcut(.defaultAction)
                .disabled(selected.isEmpty)
            }
        }
        .padding(H3Spacing.lg)
        .frame(width: 440)
    }
}

/// A RefMod among the references (the form's or a カスタムクリップ clip's):
/// its name and contents, and its strength (1 = the stored latent as it
/// is; less mixes it toward a blurred copy of itself, which weakens the
/// reference).
struct RefModReferenceRow: View {
    @ObservedObject var library: ModelLibrary
    let path: String
    let member: Int32
    @Binding var strength: Float
    /// The row's number in the form's list; nil for a clip's list.
    var number: Int? = nil
    var onRemove: () -> Void
    @Environment(\.colorScheme) private var colorScheme
    private var palette: H3Palette { H3Palette(colorScheme) }

    var body: some View {
        let entry = library.refmods.first { $0.path == path }
        let info = try? library.refmodInfo(path: path)?.get()
        let shown = info.flatMap { info -> H3RefModMember? in
            member >= 0 ? info.members.first { $0.id == Int(member) } : info.members.first
        }
        let title = entry?.name ?? URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
        HStack(alignment: .center) {
            Image(systemName: "cube")
                .foregroundStyle(palette.textSecondary)
            if let number {
                Text(verbatim: "\(number).").font(.caption)
            }
            VStack(alignment: .leading, spacing: 1) {
                Text(verbatim: title).font(.caption).lineLimit(1).truncationMode(.middle)
                if let shown {
                    Text(refmodMemberSummary(shown, inBundle: info?.isBundle ?? false))
                        .font(.caption2).foregroundStyle(palette.textSecondary)
                } else if case .failure(let error)? = library.refmodInfo(path: path) {
                    Text(error.localizedDescription).font(.caption2).foregroundStyle(palette.errorColor)
                }
            }
            Spacer()
            Text("強さ").font(.caption2).foregroundStyle(palette.textSecondary)
            Slider(value: Binding(
                get: { Double(strength) },
                set: { strength = min(max(Float($0), 0.05), 1) }
            ), in: 0.05 ... 1, step: 0.05)
            .frame(width: 90)
            .help("1はそのまま。小さくするほどぼかした参照に近づき、効きが弱まります（上流のApplyと同じ計算）。")
            Text(verbatim: String(format: "%.2f", strength)).font(.caption2).monospacedDigit()
                .frame(width: 30, alignment: .trailing)
            Button("取り除く") { onRemove() }
                .font(.caption)
        }
    }
}
