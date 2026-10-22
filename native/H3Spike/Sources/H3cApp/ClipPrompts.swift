import AppKit
import SwiftUI
import UniformTypeIdentifiers

// クリップごとの指示: in a つなげて生成 every clip used to get the same
// prompt, so a story told in it (dark -> two lights -> singing -> a city)
// started over in each clip. The prompt box now holds what stays the same,
// and each clip can add what happens in it (chainClipPrompt). They can be
// written here, or in a storyboard file read in - by hand or by any tool.

/// The storyboard file the sheet writes (and the simplest form it reads).
struct StoryboardFile: Codable {
    struct Clip: Codable {
        var clip: Int
        var start: Double
        var seconds: Double
        var prompt: String
        /// Whether the clip uses the reference images (omitted: yes).
        var references: Bool? = nil
    }

    var version = 1
    var prompt: String
    var clipSeconds: Int
    var clips: [Clip]

    /// Reads the file written above, or looser forms: "clips" as plain
    /// strings, or the whole file one array of strings. Returns the common
    /// prompt (nil if the file has none), each clip's text, and whether
    /// each uses the reference images.
    static func read(_ data: Data) throws -> (prompt: String?, clips: [String], references: [Bool]) {
        let json = try JSONSerialization.jsonObject(with: data)
        func clipTexts(_ value: Any?) -> (texts: [String], references: [Bool])? {
            guard let list = value as? [Any] else { return nil }
            var texts: [String] = []
            var references: [Bool] = []
            for item in list {
                if let text = item as? String {
                    texts.append(text)
                    references.append(true)
                } else if let object = item as? [String: Any], let text = object["prompt"] as? String {
                    texts.append(text)
                    references.append((object["references"] as? Bool) ?? true)
                } else {
                    return nil
                }
            }
            return (texts, references)
        }
        if let clips = clipTexts(json) { return (nil, clips.texts, clips.references) }
        guard let object = json as? [String: Any], let clips = clipTexts(object["clips"]) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let prompt = (object["prompt"] as? String) ?? (object["commonPrompt"] as? String)
        return (prompt, clips.texts, clips.references)
    }
}

/// The form's row for クリップごとの指示, shown while the length is
/// chained: one line until any are written.
struct ClipPromptsSection: View {
    @ObservedObject var viewModel: GenerationViewModel
    @State private var showingSheet = false
    @Environment(\.colorScheme) private var colorScheme
    private var palette: H3Palette { H3Palette(colorScheme) }

    var body: some View {
        Group {
            if viewModel.storedClipPrompts != nil || viewModel.storedClipReferences != nil {
                let plan = viewModel.currentChainPlan
                let written = (viewModel.storedClipPrompts ?? []).prefix(plan.count).filter {
                    !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                }.count
                let withoutReferences = (viewModel.storedClipReferences ?? []).prefix(plan.count)
                    .filter { !$0 }.count
                HStack {
                    Image(systemName: "list.number").foregroundStyle(palette.textSecondary)
                    Text("クリップごとの指示：\(plan.count)本のうち\(written)本").font(.caption)
                    if withoutReferences > 0 {
                        Text("（参照画像なし\(withoutReferences)本）").font(.caption)
                            .foregroundStyle(palette.textSecondary)
                    }
                    Spacer()
                    Button("編集…") { showingSheet = true }
                    Button("取り除く") {
                        viewModel.clipPrompts = []
                        viewModel.clipReferences = []
                    }
                }
                .padding(H3Spacing.md)
                .background(palette.surfaceMuted)
                .clipShape(RoundedRectangle(cornerRadius: H3Radius.control))
            } else {
                Button {
                    showingSheet = true
                } label: {
                    Label("クリップごとに指示を変える…", systemImage: "list.number")
                }
                .font(.caption)
                .help("つなぐ動画の1本ずつに、その動画で起きることを書けます。下の入力欄の文章は全部に共通で使います")
            }
        }
        .sheet(isPresented: $showingSheet) {
            ClipPromptsSheet(viewModel: viewModel, isPresented: $showingSheet)
        }
    }
}

struct ClipPromptsSheet: View {
    @ObservedObject var viewModel: GenerationViewModel
    @Binding var isPresented: Bool
    @State private var audioSeconds: Double?
    @State private var message: String?
    @Environment(\.colorScheme) private var colorScheme
    private var palette: H3Palette { H3Palette(colorScheme) }

    private var plan: [Int] { viewModel.currentChainPlan }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("クリップごとの指示").font(.headline)
            Text("つなぐ動画の1本ずつに、その動画で起きることを書きます。前の動画の終わりの様子から始めて、そこで起きる動き、終わりの様子、カメラの動きを書くと、話が先へ進みます。空のままの動画は、共通の指示だけでつくります。")
                .font(.caption)
                .foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            ScrollView {
                VStack(alignment: .leading, spacing: H3Spacing.md) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("共通の指示（全部の動画に使う：人物、服装、画風など）").font(.caption.weight(.semibold))
                        editor($viewModel.prompt, placeholder: "")
                    }
                    let starts = chainClipStarts(plan)
                    ForEach(plan.indices, id: \.self) { index in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: H3Spacing.sm) {
                                Text("\(index + 1)本目").font(.caption.weight(.semibold))
                                Text(timeRange(start: starts[index], frames: plan[index]))
                                    .font(.caption)
                                    .foregroundStyle(palette.textSecondary)
                                if let audio = audioRange(start: starts[index], frames: plan[index]) {
                                    Text(audio).font(.caption).foregroundStyle(palette.textSecondary)
                                }
                                Spacer()
                                if hasReferences {
                                    Toggle("参照画像を使う", isOn: referenceBinding(index))
                                        .toggleStyle(.checkbox)
                                        .font(.caption)
                                        .help("人物が出ない場面（背景だけの場面など）では外せます。外した動画は、前の動画の続きと文章だけでつくります")
                                }
                            }
                            editor(clipBinding(index), placeholder: String(localized: "（共通の指示だけでつくります）"))
                        }
                    }
                    if viewModel.clipPrompts.count > plan.count {
                        let unused = viewModel.clipPrompts[plan.count...].filter {
                            !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        }.count
                        if unused > 0 {
                            Text("このあとの\(unused)本分の指示は、今の長さでは使いません（長さを延ばすと使います）")
                                .font(.caption)
                                .foregroundStyle(palette.textSecondary)
                        }
                    }
                }
                .padding(.trailing, H3Spacing.sm)
            }
            if let message {
                Text(message).font(.caption).foregroundStyle(palette.errorColor)
            }
            HStack {
                Button("読み込む…") { importFile() }
                    .help("書き出したファイルか、同じ形のJSONファイルから読み込みます")
                Button("書き出す…") { exportFile() }
                Spacer()
                Button("すべて消す") {
                    viewModel.clipPrompts = []
                    viewModel.clipReferences = []
                }
                .disabled(viewModel.storedClipPrompts == nil && viewModel.storedClipReferences == nil)
                Button("完了") { isPresented = false }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(H3Spacing.lg)
        .frame(width: 600, height: 640)
        .onAppear {
            audioSeconds = viewModel.forcedAudioPath.flatMap(audioFileSeconds)
        }
    }

    private func editor(_ text: Binding<String>, placeholder: String) -> some View {
        ZStack(alignment: .topLeading) {
            TextEditor(text: text)
                .font(.body)
                .scrollContentBackground(.hidden)
                .padding(4)
            if text.wrappedValue.isEmpty && !placeholder.isEmpty {
                Text(placeholder)
                    .font(.body)
                    .foregroundStyle(palette.textSecondary.opacity(0.75))
                    .padding(.horizontal, 9)
                    .padding(.vertical, 4)
                    .allowsHitTesting(false)
            }
        }
        .frame(height: 58)
        .background(palette.surface)
        .clipShape(RoundedRectangle(cornerRadius: H3Radius.control))
        .overlay(RoundedRectangle(cornerRadius: H3Radius.control).stroke(palette.border, lineWidth: 1))
    }

    /// The form's reference images are in use (画像から, 参照画像).
    private var hasReferences: Bool {
        viewModel.creationMethod == .image && viewModel.imageInputMode == .referenceImage
            && !viewModel.referenceImages.isEmpty
    }

    private func referenceBinding(_ index: Int) -> Binding<Bool> {
        Binding(
            get: { viewModel.chainClipUsesReferences(viewModel.clipReferences, index: index) },
            set: { on in
                var flags = viewModel.clipReferences
                if flags.count <= index { flags += Array(repeating: true, count: index + 1 - flags.count) }
                flags[index] = on
                viewModel.clipReferences = flags
            })
    }

    private func clipBinding(_ index: Int) -> Binding<String> {
        Binding(
            get: { index < viewModel.clipPrompts.count ? viewModel.clipPrompts[index] : "" },
            set: { text in
                var prompts = viewModel.clipPrompts
                if prompts.count <= index { prompts += Array(repeating: "", count: index + 1 - prompts.count) }
                prompts[index] = text
                viewModel.clipPrompts = prompts
            })
    }

    private func timeRange(start: Int, frames: Int) -> String {
        let from = String(format: "%.1f", Double(start) / 24)
        let to = String(format: "%.1f", Double(start + frames) / 24)
        return String(localized: "\(from)〜\(to)秒")
    }

    /// With lip sync, where in the audio the clip is: it starts at the
    /// form's offset.
    private func audioRange(start: Int, frames: Int) -> String? {
        guard viewModel.forcedAudioPath != nil else { return nil }
        let offset = max(0, viewModel.forcedAudioOffset)
        let begin = offset + Double(start) / 24
        if let audioSeconds, begin >= audioSeconds - 0.1 {
            return String(localized: "（音声の終わりの後）")
        }
        let from = String(format: "%.1f", begin)
        let to = String(format: "%.1f", begin + Double(frames) / 24)
        return String(localized: "（音声 \(from)〜\(to)秒）")
    }

    private func importFile() {
        message = nil
        guard let path = chooseFile(allowedContentTypes: [.json], startingIn: viewModel.project?.url.path) else {
            return
        }
        do {
            let file = try StoryboardFile.read(Data(contentsOf: URL(fileURLWithPath: path)))
            if let prompt = file.prompt, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                viewModel.prompt = prompt
            }
            viewModel.clipPrompts = file.clips
            viewModel.clipReferences = file.references
        } catch {
            message = String(localized: "読み込めませんでした。クリップごとの指示を並べたJSONファイルを選んでください")
        }
    }

    private func exportFile() {
        message = nil
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "storyboard.json"
        if let project = viewModel.project { panel.directoryURL = project.url }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let starts = chainClipStarts(plan)
        let file = StoryboardFile(
            prompt: viewModel.prompt,
            clipSeconds: viewModel.chainClipLength == .five ? 5 : 7,
            clips: plan.indices.map { index in
                StoryboardFile.Clip(clip: index + 1,
                                    start: (Double(starts[index]) / 24 * 100).rounded() / 100,
                                    seconds: (Double(plan[index]) / 24 * 100).rounded() / 100,
                                    prompt: index < viewModel.clipPrompts.count ? viewModel.clipPrompts[index] : "",
                                    references: hasReferences
                                        ? viewModel.chainClipUsesReferences(viewModel.clipReferences, index: index) : nil)
            })
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            try encoder.encode(file).write(to: url, options: .atomic)
        } catch {
            message = String(localized: "書き出せませんでした（詳細: \(error.localizedDescription)）")
        }
    }
}
