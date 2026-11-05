import H3Engine
import SwiftUI

// RefMods (saved reference latents): making one from a reference the form
// holds (RefModExtractSheet) and choosing the members of a bundle
// (RefModMemberPickerSheet). The files themselves are listed in モデル管理
// (ModelManagerView) and kept in defaultRefModDirectory.

/// 「RefModとして保存…」: encodes one of the form's references with the
/// Ref2VA VAE and writes a RefMod file into the RefMods folder
/// (GenerationViewModel.extractRefMod). Runs on the engine's queue, so it
/// waits for a generation in flight.
struct RefModExtractSheet: View {
    let source: H3ReferenceInput
    @ObservedObject var viewModel: GenerationViewModel
    @ObservedObject var library: ModelLibrary
    @Binding var isPresented: Bool
    @Environment(\.colorScheme) private var colorScheme

    @State private var name: String
    @State private var mode: H3RefModMode = .full
    @State private var resolutionCap = 1024
    @State private var latentFrames = 16
    @State private var poolLongEdge = 16
    @State private var optimizeSteps = 500
    @State private var maxSeconds = 15
    @State private var addToReferences = true
    @State private var running = false
    @State private var errorMessage: String?

    private var palette: H3Palette { H3Palette(colorScheme) }

    init(source: H3ReferenceInput, viewModel: GenerationViewModel, library: ModelLibrary, isPresented: Binding<Bool>) {
        self.source = source
        self.viewModel = viewModel
        self.library = library
        _isPresented = isPresented
        _name = State(initialValue: URL(fileURLWithPath: source.path).deletingPathExtension().lastPathComponent)
    }

    private var kind: H3RefModKind {
        switch source.kind {
        case .video, .videoAudio: return .video
        case .audio: return .audio
        default: return .image
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("RefModとして保存").font(.system(size: 16, weight: .semibold))
            Text("参照を一度だけVAEで符号化して、小さなファイル（.safetensors）に保存します。保存したRefModは、参照の「RefModを追加…」からどの生成でも使えます。Qwenには見せないので、プロンプトの<Picture n>は要りません。")
                .font(.caption)
                .foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            Text(verbatim: URL(fileURLWithPath: source.path).lastPathComponent)
                .font(.caption)
                .lineLimit(1)
                .truncationMode(.middle)
            Grid(alignment: .leading, horizontalSpacing: H3Spacing.md, verticalSpacing: H3Spacing.sm) {
                GridRow {
                    Text("名前")
                    TextField("名前", text: $name).textFieldStyle(.roundedBorder).frame(width: 260)
                }
                if kind != .audio {
                    GridRow {
                        Text("モード")
                        Picker("", selection: $mode) {
                            Text("Full（そのまま）").tag(H3RefModMode.full)
                            Text("圧縮（小さい格子に）").tag(H3RefModMode.compressed)
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .frame(width: 260)
                    }
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
                    if mode == .compressed {
                        GridRow {
                            Text("格子の長辺")
                            Stepper(value: $poolLongEdge, in: 2 ... 64, step: 2) {
                                Text("\(poolLongEdge)")
                            }
                            .help("長い辺をこの数のセルにまとめます。小さいほど参照のトークンが減り、細部も減ります。")
                        }
                        GridRow {
                            Text("調整ステップ")
                            Stepper(value: $optimizeSteps, in: 0 ... 5000, step: 50) {
                                Text("\(optimizeSteps)")
                            }
                            .help("まとめた格子を、元の符号化に近づくよう調整する回数。0は平均のまま。")
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
            Toggle("作成したRefModを参照に追加する", isOn: $addToReferences)
            if mode == .compressed && kind != .audio {
                Text("圧縮は上流（ComfyUI-MiniMaxH3Mod）と同じ計算です。人物の同一性を学習するものではなく、細部や動きは失われます。")
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
                        .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
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
        var request = H3RefModExtractRequest(
            sourcePath: source.path,
            outputPath: ModelLibrary.freePath(in: defaultRefModDirectory, stem: trimmed, extension: "safetensors"),
            kind: kind)
        request.name = trimmed
        request.mode = kind == .audio ? .full : mode
        request.resolutionCap = resolutionCap
        request.latentFrames = latentFrames
        request.poolLongEdge = poolLongEdge
        request.optimizeSteps = optimizeSteps
        request.maxSeconds = maxSeconds
        running = true
        errorMessage = nil
        Task {
            do {
                let path = try await viewModel.extractRefMod(request)
                if addToReferences {
                    viewModel.referenceImages.append(H3ReferenceInput(kind: .refmod, path: path))
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
    case "video VAE encoder": return String(localized: "映像を符号化しています（\(state.completed)/\(state.total)）…")
    case "audio VAE encoder": return String(localized: "音声を符号化しています…")
    case "refmod refine": return String(localized: "圧縮した格子を調整しています（\(state.completed)/\(state.total)）…")
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
