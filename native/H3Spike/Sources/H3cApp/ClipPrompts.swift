import AppKit
import H3Engine
import SwiftUI
import UniformTypeIdentifiers

// カスタムクリップ's editor: in a つなげて生成 every clip used to get the
// same prompt, so a story told in it (dark -> two lights -> singing -> a
// city) started over in each clip. The prompt box now holds what stays the
// same, and each clip says what happens in it, how long it is, which images
// it uses and which audio it lip-syncs to (CustomClips.swift). Clips can be
// written here, or in a storyboard file read in - by hand or by any tool.

/// The storyboard file the editor writes (and the simpler forms it reads).
struct StoryboardFile: Codable {
    struct Clip: Codable {
        var clip: Int
        var start: Double
        var seconds: Double
        var prompt: String
        /// Uses the form's reference images (written before カスタムクリップ;
        /// false reads as no images).
        var references: Bool? = nil
        var images: ChainClipSpec.Images? = nil
        var referencePaths: [String]? = nil
        var firstFrame: String? = nil
        var lastFrame: String? = nil
        var audio: ChainClipSpec.Audio? = nil
        var audioPath: String? = nil
        var audioOffset: Double? = nil
    }

    var version = 2
    var prompt: String
    var clips: [Clip]

    /// Reads the file written below, or looser forms: "clips" as plain
    /// strings, or the whole file one array of strings. Returns the common
    /// prompt (nil if the file has none) and the clips.
    static func read(_ data: Data) throws -> (prompt: String?, clips: [ChainClipSpec]) {
        let json = try JSONSerialization.jsonObject(with: data)
        func clips(_ value: Any?) -> [ChainClipSpec]? {
            guard let list = value as? [Any] else { return nil }
            var clips: [ChainClipSpec] = []
            for item in list {
                var clip = ChainClipSpec()
                if let text = item as? String {
                    clip.prompt = text
                } else if let object = item as? [String: Any], let text = object["prompt"] as? String {
                    clip.prompt = text
                    if let seconds = object["seconds"] as? Double {
                        clip.seconds = Int(seconds.rounded()).clamped(to: ChainClipSpec.secondsRange)
                    }
                    if object["references"] as? Bool == false { clip.images = .references }
                    if let images = (object["images"] as? String).flatMap(ChainClipSpec.Images.init(rawValue:)) {
                        clip.images = images
                    }
                    clip.references = ((object["referencePaths"] as? [String]) ?? []).map { path in
                        .init(kind: (isVideoFile(path: path) ? H3ReferenceKind.video
                                     : isAudioFile(path: path) ? .audio : .image).storedName, path: path)
                    }
                    clip.firstFrame = object["firstFrame"] as? String
                    clip.lastFrame = object["lastFrame"] as? String
                    if let audio = (object["audio"] as? String).flatMap(ChainClipSpec.Audio.init(rawValue:)) {
                        clip.audio = audio
                    }
                    clip.audioPath = object["audioPath"] as? String
                    clip.audioOffset = (object["audioOffset"] as? Double) ?? 0
                } else {
                    return nil
                }
                clips.append(clip)
            }
            return clips.isEmpty ? nil : clips
        }
        if let list = clips(json) { return (nil, list) }
        guard let object = json as? [String: Any], let list = clips(object["clips"]) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let prompt = (object["prompt"] as? String) ?? (object["commonPrompt"] as? String)
        return (prompt, list)
    }
}

/// The form's row while the length is chained: what カスタムクリップ holds,
/// or a way into it from the equal clips of a length.
struct ClipPromptsSection: View {
    @ObservedObject var viewModel: GenerationViewModel
    @Binding var showingEditor: Bool
    @Environment(\.colorScheme) private var colorScheme
    private var palette: H3Palette { H3Palette(colorScheme) }

    var body: some View {
        if viewModel.usesCustomClips {
            HStack {
                Image(systemName: "list.number").foregroundStyle(palette.textSecondary)
                let total = String(format: "%.1f", viewModel.customClipsTotalSeconds)
                Text("カスタムクリップ：\(viewModel.customClips.count)本・計\(total)秒").font(.caption)
                Spacer()
                Button("編集…") { showingEditor = true }
            }
            .padding(H3Spacing.md)
            .background(palette.surfaceMuted)
            .clipShape(RoundedRectangle(cornerRadius: H3Radius.control))
        } else {
            Button {
                viewModel.useCustomClips()
                showingEditor = true
            } label: {
                Label("クリップを1本ずつ編集する…", systemImage: "list.number")
            }
            .font(.caption)
            .help("つなぐ動画の1本ずつに、長さ、起きること、画像、リップシンクの音声を決められます（カスタムクリップ）")
        }
    }
}

struct CustomClipsSheet: View {
    @ObservedObject var viewModel: GenerationViewModel
    @Binding var isPresented: Bool
    @State private var message: String?
    @Environment(\.colorScheme) private var colorScheme
    private var palette: H3Palette { H3Palette(colorScheme) }

    var body: some View {
        let plans = viewModel.chainClipPlans()
        let starts = chainClipStarts(frames: plans.map(\.frames), continues: plans.map(\.continues))
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("カスタムクリップ").font(.headline)
            Text("つなぐ動画を1本ずつ決めます。2本目からは、前の動画の最後の約0.9秒を引き継いで続きをつくります。前の動画の終わりの様子から始めて、そこで起きる動き、終わりの様子、カメラの動きを書くと、話が先へ進みます。")
                .font(.caption)
                .foregroundStyle(palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            ScrollView {
                VStack(alignment: .leading, spacing: H3Spacing.lg) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("共通の指示（全部の動画に使う：人物、服装、画風など）").font(.caption.weight(.semibold))
                        ClipTextEditor(text: $viewModel.prompt, placeholder: "")
                    }
                    ForEach($viewModel.customClips) { $clip in
                        let index = viewModel.customClips.firstIndex { $0.id == clip.id } ?? 0
                        CustomClipRow(viewModel: viewModel, clip: $clip, index: index,
                                      start: index < starts.count ? starts[index] : 0,
                                      plan: index < plans.count ? plans[index] : nil)
                        Divider()
                    }
                    Button {
                        var clip = ChainClipSpec()
                        if let last = viewModel.customClips.last { clip.seconds = max(2, last.seconds) }
                        viewModel.customClips.append(clip)
                    } label: {
                        Label("クリップを足す", systemImage: "plus")
                    }
                    .help("この下に、続きの動画を1本足します")
                }
                .padding(.trailing, H3Spacing.sm)
            }
            if let message {
                Text(message).font(.caption).foregroundStyle(palette.errorColor)
            }
            HStack {
                Button("読み込む…") { importFile() }
                    .help("書き出したファイルか、同じ形のJSONファイルから読み込みます")
                Button("書き出す…") { exportFile(starts: starts) }
                Spacer()
                Button("完了") { isPresented = false }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(H3Spacing.lg)
        .frame(width: 640, height: 700)
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
            viewModel.customClips = file.clips
            viewModel.chained = true
            viewModel.chainCustom = true
        } catch {
            message = String(localized: "読み込めませんでした。クリップごとの指示を並べたJSONファイルを選んでください")
        }
    }

    private func exportFile(starts: [Int]) {
        message = nil
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "storyboard.json"
        if let project = viewModel.project { panel.directoryURL = project.url }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let file = StoryboardFile(
            prompt: viewModel.prompt,
            clips: viewModel.customClips.enumerated().map { index, clip in
                StoryboardFile.Clip(
                    clip: index + 1,
                    start: index < starts.count ? (Double(starts[index]) / 24 * 100).rounded() / 100 : 0,
                    seconds: (Double(clip.frames) / 24 * 100).rounded() / 100,
                    prompt: clip.prompt,
                    images: index > 0 ? clip.images : nil,
                    referencePaths: index > 0 && clip.images == .references ? clip.references.map(\.path) : nil,
                    firstFrame: index > 0 && clip.images == .frames ? clip.firstFrame : nil,
                    lastFrame: index > 0 && clip.images == .frames ? clip.lastFrame : nil,
                    audio: index > 0 ? clip.audio : nil,
                    audioPath: index > 0 && clip.audio == .file ? clip.audioPath : nil,
                    audioOffset: index > 0 && clip.audio == .file ? clip.audioOffset : nil)
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

/// One clip in カスタムクリップ's editor.
private struct CustomClipRow: View {
    @ObservedObject var viewModel: GenerationViewModel
    @Binding var clip: ChainClipSpec
    let index: Int
    let start: Int
    let plan: ChainClipPlan?
    @Environment(\.colorScheme) private var colorScheme
    private var palette: H3Palette { H3Palette(colorScheme) }

    private var secondsRange: ClosedRange<Int> {
        index == 0 || clip.startsFresh ? ChainClipSpec.secondsRange : ChainClipSpec.continuingSecondsRange
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.sm) {
            HStack(spacing: H3Spacing.sm) {
                Text("\(index + 1)本目").font(.callout.weight(.semibold))
                Text(timeRange).font(.caption).foregroundStyle(palette.textSecondary)
                if let audio = audioRange {
                    Text(audio).font(.caption).foregroundStyle(palette.textSecondary)
                }
                Spacer()
                Stepper(value: Binding(get: { clip.seconds },
                                       set: { clip.seconds = $0.clamped(to: secondsRange) }),
                        in: secondsRange) {
                    Text("\(clip.seconds)秒").font(.caption).monospacedDigit()
                }
                .help("この動画の長さ")
                if viewModel.customClips.count > 1 {
                    Button {
                        let id = clip.id
                        viewModel.customClips.removeAll { $0.id == id }
                    } label: {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(.borderless)
                    .help("このクリップを取り除く")
                }
            }
            ClipTextEditor(text: $clip.prompt, placeholder: String(localized: "（共通の指示だけでつくります）"))
            if index == 0 {
                Text("1本目は、パネルの画像（画像から）とリップシンクの設定でつくります。")
                    .font(.caption)
                    .foregroundStyle(palette.textSecondary)
            } else {
                imagesSection
                audioSection
            }
        }
    }

    // MARK: Images

    @ViewBuilder
    private var imagesSection: some View {
        Toggle("同じ参照画像を使う", isOn: Binding(
            get: { clip.images == .same },
            set: { clip.images = $0 ? .same : .references }))
            .toggleStyle(.checkbox)
            .font(.caption)
            .help("パネルで選んだ参照画像を、この動画にも使います（パネルに参照画像がなければ、画像なしでつくります）")
        if clip.images != .same {
            VStack(alignment: .leading, spacing: H3Spacing.sm) {
                Picker("", selection: $clip.images) {
                    Text("参照画像").tag(ChainClipSpec.Images.references)
                    Text("最初・最後の画像").tag(ChainClipSpec.Images.frames)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 280)
                if clip.images == .references {
                    ForEach(Array(clip.references.enumerated()), id: \.offset) { offset, reference in
                        HStack {
                            Image(systemName: reference.kind == "audio" ? "waveform"
                                  : reference.kind == "video" ? "video" : "photo")
                                .foregroundStyle(palette.textSecondary)
                            Text(URL(fileURLWithPath: reference.path).lastPathComponent).font(.caption)
                            Spacer()
                            Button("取り除く") { clip.references.remove(at: offset) }.font(.caption)
                        }
                    }
                    HStack {
                        Button("ファイルを選ぶ…") {
                            for path in chooseFiles(allowedContentTypes: referenceMediaTypes)
                            where !clip.references.contains(where: { $0.path == path }) {
                                let kind: H3ReferenceKind = isVideoFile(path: path) ? .video
                                    : isAudioFile(path: path) ? .audio : .image
                                clip.references.append(.init(kind: kind.storedName, path: path))
                            }
                        }
                        .font(.caption)
                        if clip.references.isEmpty {
                            Text("選ばなければ、画像なしで前の動画の続きをつくります")
                                .font(.caption).foregroundStyle(palette.textSecondary)
                        }
                    }
                } else {
                    frameRow(label: String(localized: "最初の画像"), path: $clip.firstFrame)
                    frameRow(label: String(localized: "最後の画像"), path: $clip.lastFrame)
                    Text(clip.firstFrame != nil
                         ? String(localized: "最初の画像を選んだので、この動画は前の動画の続きにせず、その画像から始めます（前の動画とはカットでつながります）。")
                         : String(localized: "最後の画像だけなら、前の動画の続きのまま、その画像で終わるようにつくります。"))
                        .font(.caption)
                        .foregroundStyle(palette.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(.leading, H3Spacing.lg)
        }
    }

    private func frameRow(label: String, path: Binding<String?>) -> some View {
        HStack {
            Text(label).font(.caption)
            if let value = path.wrappedValue {
                Text(URL(fileURLWithPath: value).lastPathComponent).font(.caption)
                    .foregroundStyle(palette.textSecondary)
            }
            Spacer()
            Button(path.wrappedValue == nil ? String(localized: "選ぶ…") : String(localized: "変更…")) {
                if let chosen = chooseFile(allowedContentTypes: imageTypes) { path.wrappedValue = chosen }
            }
            .font(.caption)
            if path.wrappedValue != nil {
                Button("取り除く") { path.wrappedValue = nil }.font(.caption)
            }
        }
    }

    // MARK: Audio

    @ViewBuilder
    private var audioSection: some View {
        HStack {
            Picker("リップシンクの音声", selection: $clip.audio) {
                Text("前の動画から続ける").tag(ChainClipSpec.Audio.carryOn)
                Text("別の音声に変える").tag(ChainClipSpec.Audio.file)
                Text("リップシンクなし").tag(ChainClipSpec.Audio.none)
            }
            .font(.caption)
            .fixedSize()
            if clip.audio == .file {
                if let path = clip.audioPath {
                    Text(URL(fileURLWithPath: path).lastPathComponent).font(.caption)
                        .foregroundStyle(palette.textSecondary)
                }
                Button(clip.audioPath == nil ? String(localized: "選ぶ…") : String(localized: "変更…")) {
                    if let chosen = chooseFile(allowedContentTypes: forcedAudioTypes) { clip.audioPath = chosen }
                }
                .font(.caption)
                TextField("開始位置", value: $clip.audioOffset, format: .number.precision(.fractionLength(0 ... 2)))
                    .textFieldStyle(.roundedBorder)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 60)
                    .help("音声の何秒目から使うか")
                Text("秒から").font(.caption)
            }
        }
    }

    // MARK: Times

    private var timeRange: String {
        let from = String(format: "%.1f", Double(start) / 24)
        let to = String(format: "%.1f", Double(start + clip.frames) / 24)
        return String(localized: "\(from)〜\(to)秒")
    }

    /// With lip sync, where in the audio the clip is.
    private var audioRange: String? {
        guard let plan else { return nil }
        guard let path = plan.audioPath else {
            let lipSync = viewModel.forcedAudioPath != nil || viewModel.customClips.contains { $0.audio == .file }
            return lipSync ? String(localized: "（リップシンクなし）") : nil
        }
        let from = String(format: "%.1f", plan.audioOffset)
        let to = String(format: "%.1f", plan.audioOffset + Double(plan.frames) / 24)
        let name = URL(fileURLWithPath: path).lastPathComponent
        return String(localized: "（\(name) \(from)〜\(to)秒）")
    }
}

/// A short multi-line text box with a grey placeholder.
struct ClipTextEditor: View {
    @Binding var text: String
    var placeholder: String
    @Environment(\.colorScheme) private var colorScheme
    private var palette: H3Palette { H3Palette(colorScheme) }

    var body: some View {
        ZStack(alignment: .topLeading) {
            TextEditor(text: $text)
                .font(.body)
                .scrollContentBackground(.hidden)
                .padding(4)
            if text.isEmpty && !placeholder.isEmpty {
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
}
