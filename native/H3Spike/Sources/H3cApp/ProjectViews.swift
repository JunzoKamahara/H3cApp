import AppKit
import AVFoundation
import SwiftUI

/// Toolbar menu: the open project's name, and everything to do with
/// projects (the same items as the menu bar's プロジェクト menu).
struct ProjectMenu: View {
    @ObservedObject var viewModel: GenerationViewModel

    var body: some View {
        Menu {
            ProjectMenuItems(viewModel: viewModel, store: ProjectStore.shared, inMenuBar: false)
        } label: {
            Label(viewModel.project?.name ?? String(localized: "プロジェクトなし"), systemImage: "folder")
        }
        .disabled(viewModel.isGenerating)
        .help("プロンプトと設定、参照ファイル、生成した動画をフォルダにまとめて保存します")
    }
}

/// The project items, shared by the toolbar menu and the menu bar.
struct ProjectMenuItems: View {
    @ObservedObject var viewModel: GenerationViewModel
    @ObservedObject var store: ProjectStore
    /// Shortcuts belong to the menu bar only, so they aren't registered twice.
    var inMenuBar = true

    var body: some View {
        Button("新規プロジェクト…") { viewModel.showingNewProject = true }
            .keyboardShortcut(inMenuBar ? KeyboardShortcut("n", modifiers: [.command, .shift]) : nil)
        Button("プロジェクトとして保存…") { viewModel.showingSaveAsProject = true }
            .disabled(viewModel.project != nil)
            .help("今のプロンプトと設定、プレビューの動画を、名前を付けてプロジェクトとして保存します")
        Button("プロジェクトを開く…") { viewModel.chooseAndOpenProject() }
            .keyboardShortcut(inMenuBar ? KeyboardShortcut("o", modifiers: .command) : nil)
        let recents = store.existingRecents.filter { $0.standardizedFileURL != viewModel.project?.url.standardizedFileURL }
        Menu("最近のプロジェクト") {
            ForEach(recents, id: \.self) { url in
                Button(url.lastPathComponent) { viewModel.openProject(at: url) }
            }
        }
        .disabled(recents.isEmpty)
        Divider()
        Button("プロジェクトの動画…") { viewModel.showingProjectVideos = true }
            .disabled(viewModel.project == nil)
        Button("表示中の動画を参照に使う") {
            if let url = viewModel.resultURL { viewModel.useVideoAsReference(url) }
        }
        .disabled(viewModel.resultURL == nil)
        Button("プロジェクトをFinderで表示") {
            if let url = viewModel.project?.url { NSWorkspace.shared.activateFileViewerSelecting([url]) }
        }
        .disabled(viewModel.project == nil)
        Divider()
        Button("プロジェクトを閉じる") { viewModel.closeProject() }
            .disabled(viewModel.project == nil)
    }
}

/// 新規プロジェクト, and プロジェクトとして保存 (`savingAs`), which also moves
/// in the temp video shown in the preview.
struct NewProjectSheet: View {
    @ObservedObject var viewModel: GenerationViewModel
    @Binding var isPresented: Bool
    var savingAs = false
    @State private var name = ""
    @State private var parent = ProjectStore.defaultParentDirectory

    private var folder: URL {
        let folderName = ProjectFiles.folderName(for: name)
        return parent.appendingPathComponent(folderName.isEmpty ? "…" : folderName)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text(savingAs ? "プロジェクトとして保存" : "新規プロジェクト").font(.headline)
            TextField("プロジェクト名", text: $name)
                .textFieldStyle(.roundedBorder)
            HStack(alignment: .firstTextBaseline) {
                Text("保存先").foregroundStyle(.secondary)
                Text(folder.path)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                Spacer()
                Button("変更…") { chooseParent() }
            }
            .font(.callout)
            Text(description)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if let message = viewModel.projectMessage {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Spacer()
                Button("キャンセル") {
                    viewModel.projectMessage = nil
                    isPresented = false
                }
                    .keyboardShortcut(.cancelAction)
                Button(savingAs ? "保存" : "作成") {
                    viewModel.projectMessage = nil
                    if savingAs {
                        viewModel.saveAsProject(name: name, parent: parent)
                    } else {
                        viewModel.createProject(name: name, parent: parent)
                    }
                    if viewModel.project != nil { isPresented = false }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(H3Spacing.lg)
        .frame(width: 480)
    }

    private var description: String {
        guard savingAs else {
            return String(localized: "今のプロンプトと設定、参照ファイルがプロジェクトに入ります。生成した動画はこのフォルダに保存され、削除するまで残ります。")
        }
        if let pending = viewModel.savableTemporaryResult {
            return String(localized: "今のプロンプトと設定、参照ファイル、プレビューの動画（\(pending.url.lastPathComponent)）をプロジェクトとして保存します。これから生成する動画もこのフォルダに保存され、削除するまで残ります。")
        }
        return String(localized: "今のプロンプトと設定、参照ファイルをプロジェクトとして保存します（プレビューに保存できる動画はありません）。これから生成する動画はこのフォルダに保存され、削除するまで残ります。")
    }

    private func chooseParent() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.directoryURL = parent
        if panel.runModal() == .OK, let url = panel.url { parent = url }
    }
}

/// Generating several videos with no project open: the videos are kept in
/// a project, so this asks for a new or an existing one, then generates.
/// Cancelling goes back to the prompt without generating.
struct BatchProjectSheet: View {
    @ObservedObject var viewModel: GenerationViewModel
    @ObservedObject var store: ProjectStore
    @Binding var isPresented: Bool

    private enum Mode: Hashable { case new, existing }
    @State private var mode = Mode.new
    @State private var name = ""
    @State private var parent = ProjectStore.defaultParentDirectory
    @State private var selection: URL?
    @State private var chosenElsewhere: URL?
    @State private var overwriteTarget: URL?

    private var folder: URL {
        let folderName = ProjectFiles.folderName(for: name)
        return parent.appendingPathComponent(folderName.isEmpty ? "…" : folderName)
    }

    private var candidates: [URL] {
        var urls = store.existingRecents
        if let chosenElsewhere,
           !urls.contains(where: { $0.standardizedFileURL == chosenElsewhere.standardizedFileURL }) {
            urls.insert(chosenElsewhere, at: 0)
        }
        return urls
    }

    private var canConfirm: Bool {
        switch mode {
        case .new: return !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        case .existing: return selection != nil
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            Text("プロジェクトを選んで生成").font(.headline)
            Text("複数本（\(viewModel.batchCount)本）を生成するには、できた動画を保存するプロジェクトが必要です。新しいプロジェクトを作るか、既存のプロジェクトを選んでください。今のプロンプトと設定、参照ファイルはそのプロジェクトに保存されます。")
                .font(.callout)
                .fixedSize(horizontal: false, vertical: true)
            Picker("", selection: $mode) {
                Text("新しいプロジェクト").tag(Mode.new)
                Text("既存のプロジェクト").tag(Mode.existing)
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            switch mode {
            case .new:
                TextField("プロジェクト名", text: $name)
                    .textFieldStyle(.roundedBorder)
                HStack(alignment: .firstTextBaseline) {
                    Text("保存先").foregroundStyle(.secondary)
                    Text(folder.path)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Button("変更…") { chooseParent() }
                }
                .font(.callout)
            case .existing:
                if candidates.isEmpty {
                    Text("最近のプロジェクトはありません。「ほかのプロジェクトを選ぶ…」でフォルダを選んでください。")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    List(candidates, id: \.self, selection: $selection) { url in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(url.lastPathComponent)
                            Text(url.deletingLastPathComponent().path)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                    }
                    .frame(height: 150)
                }
                Button("ほかのプロジェクトを選ぶ…") { chooseExisting() }
            }

            if let message = viewModel.projectMessage {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Spacer()
                Button("キャンセル") {
                    viewModel.projectMessage = nil
                    isPresented = false
                }
                .keyboardShortcut(.cancelAction)
                Button(mode == .new ? "作成して生成" : "開いて生成") { confirm() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!canConfirm)
            }
        }
        .padding(H3Spacing.lg)
        .frame(width: 520)
        .alert(Text("「\(overwriteTarget?.lastPathComponent ?? "")」の設定を上書きしますか？"),
               isPresented: Binding(get: { overwriteTarget != nil },
                                    set: { if !$0 { overwriteTarget = nil } })) {
            Button("上書きして生成") {
                if let url = overwriteTarget { openAndGenerate(url) }
                overwriteTarget = nil
            }
            Button("キャンセル", role: .cancel) { overwriteTarget = nil }
        } message: {
            Text("このプロジェクトに保存されているプロンプトと設定は、今のものと異なります。上書きすると、今のプロンプトと設定で生成し、プロジェクトに保存します。")
        }
    }

    private func confirm() {
        viewModel.projectMessage = nil
        switch mode {
        case .new:
            if viewModel.createProjectAndGenerate(name: name, parent: parent) { isPresented = false }
        case .existing:
            guard let selection else { return }
            // Ask before replacing a project's own prompt and settings.
            if viewModel.projectDraftDiffers(at: selection) {
                overwriteTarget = selection
            } else {
                openAndGenerate(selection)
            }
        }
    }

    private func openAndGenerate(_ url: URL) {
        if viewModel.openProjectAndGenerate(at: url) { isPresented = false }
    }

    private func chooseParent() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.directoryURL = parent
        if panel.runModal() == .OK, let url = panel.url { parent = url }
    }

    private func chooseExisting() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = ProjectStore.defaultParentDirectory
        guard panel.runModal() == .OK, let url = panel.url else { return }
        chosenElsewhere = url
        selection = url
    }
}

/// The open project's videos, newest first.
struct ProjectVideosSheet: View {
    @ObservedObject var viewModel: GenerationViewModel
    @Binding var isPresented: Bool
    @State private var pendingDelete: ProjectVideo?

    var body: some View {
        VStack(alignment: .leading, spacing: H3Spacing.md) {
            HStack {
                Text(viewModel.project?.name ?? "").font(.headline)
                Text("\(viewModel.projectVideos.count)本").foregroundStyle(.secondary)
                Spacer()
                Button("閉じる") { isPresented = false }
                    .keyboardShortcut(.cancelAction)
            }
            if viewModel.projectVideos.isEmpty {
                Text("まだ動画はありません。生成した動画はここに保存されます。")
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(viewModel.projectVideos) { video in
                    row(video)
                }
            }
        }
        .padding(H3Spacing.lg)
        .frame(width: 720, height: 520)
        .onAppear { viewModel.reloadProjectVideos() }
        .confirmationDialog("この動画を削除しますか？", isPresented: Binding(
            get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }
        ), presenting: pendingDelete) { video in
            Button("ゴミ箱に移動", role: .destructive) { viewModel.deleteProjectVideo(video) }
        } message: { _ in
            Text("動画と設定の記録をゴミ箱に移動します。")
        }
    }

    private func row(_ video: ProjectVideo) -> some View {
        HStack(spacing: H3Spacing.md) {
            Button { show(video) } label: {
                VideoThumbnail(url: video.url)
            }
            .buttonStyle(.plain)
            .disabled(viewModel.isGenerating)
            .help(showHelp(video))
            VStack(alignment: .leading, spacing: 2) {
                Text(video.url.lastPathComponent)
                    .font(.callout.monospaced())
                    .lineLimit(1)
                    .truncationMode(.middle)
                if let record = video.record {
                    Text(summary(record)).font(.caption).foregroundStyle(.secondary)
                    Text(record.draft.prompt)
                        .font(.caption)
                        .lineLimit(1)
                        .truncationMode(.tail)
                } else {
                    Text("設定の記録がありません").font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer()
            Button("表示") { show(video) }
                .disabled(viewModel.isGenerating)
                .help(showHelp(video))
            Button("設定を戻す") {
                viewModel.restoreProjectVideo(video)
                isPresented = false
            }
            .disabled(video.record == nil)
            .help("この動画のプロンプトと設定をシード固定でパネルに戻します。そのまま生成すると同じ動画になります")
            Menu("その他") {
                Button("参照に使う") {
                    viewModel.useVideoAsReference(video.url)
                    isPresented = false
                }
                Button("Finderで表示") { NSWorkspace.shared.activateFileViewerSelecting([video.url]) }
                Divider()
                Button("削除…", role: .destructive) { pendingDelete = video }
            }
            .fixedSize()
        }
        .padding(.vertical, 2)
    }

    /// Shows the video and puts its settings (seed pinned) into the form,
    /// so the window and the panel describe the same video.
    private func show(_ video: ProjectVideo) {
        viewModel.showProjectVideo(video)
        viewModel.restoreProjectVideo(video)
        isPresented = false
    }

    private func showHelp(_ video: ProjectVideo) -> String {
        video.record == nil
            ? String(localized: "この動画をプレビューに表示します")
            : String(localized: "この動画をプレビューに表示し、プロンプトと設定もシード固定でパネルに戻します")
    }

    private func summary(_ record: ProjectVideoRecord) -> String {
        let draft = record.draft
        var parts: [String] = []
        if let profile = SizeProfile(rawValue: draft.sizeProfile) { parts.append(profile.label) }
        parts.append(String(localized: "\(draft.seconds)秒"))
        if let speed = SpeedMode(rawValue: draft.speedMode) { parts.append(speed.summaryLabel) }
        if draft.fastAttention { parts.append(String(localized: "高速モード（試験的）")) }
        parts.append(String(localized: "シード \(record.seed)"))
        parts.append(String(localized: "生成時間 \(formatElapsed(record.generationSeconds))"))
        return parts.joined(separator: summarySeparator)
    }
}

/// A still from one second into the video.
struct VideoThumbnail: View {
    let url: URL
    @State private var image: NSImage?

    var body: some View {
        ZStack {
            Color.black
            if let image {
                Image(nsImage: image).resizable().scaledToFit()
            }
        }
        .frame(width: 72, height: 72)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .task(id: url) { image = await Self.thumbnail(for: url) }
    }

    private static func thumbnail(for url: URL) async -> NSImage? {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 216, height: 216)
        guard let (image, _) = try? await generator.image(at: CMTime(seconds: 1, preferredTimescale: 600)) else {
            return nil
        }
        return NSImage(cgImage: image, size: NSSize(width: image.width, height: image.height))
    }
}
