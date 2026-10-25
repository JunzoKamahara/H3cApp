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
        Button("プロジェクトとして保存…") {
            viewModel.continueAfterSavingAsProject = false
            viewModel.showingSaveAsProject = true
        }
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
        Button("表示中の動画の続きを作る") {
            if let video = viewModel.continuableDisplayedVideo { viewModel.continueFromProjectVideo(video) }
        }
        .disabled(viewModel.continuableDisplayedVideo == nil || viewModel.isGenerating)
        Button("動画ファイルの続きを作る…") { viewModel.pickVideoToContinue() }
            .disabled(viewModel.project == nil || viewModel.isGenerating || viewModel.isImportingVideo)
            .help("ほかで作った動画や撮影した動画を、今の大きさでプロジェクトに取り込み、その続きをつくります")
        Button("動画ファイルを拡大…") { viewModel.pickVideoToUpscale() }
            .disabled(viewModel.isUpscalingVideo)
            .help("動画ファイルを選んで、拡大した動画（1.5〜4倍）を保存します")
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
            Text(viewModel.isChained
                 ? String(localized: "つなげて生成するには、つなぐ前の動画とつないだ動画を保存するプロジェクトが必要です。新しいプロジェクトを作るか、既存のプロジェクトを選んでください。今のプロンプトと設定、参照ファイルはそのプロジェクトに保存されます。")
                 : String(localized: "複数本（\(viewModel.batchCount)本）を生成するには、できた動画を保存するプロジェクトが必要です。新しいプロジェクトを作るか、既存のプロジェクトを選んでください。今のプロンプトと設定、参照ファイルはそのプロジェクトに保存されます。"))
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
    @State private var tree: [VideoNode] = []
    /// 2倍に拡大した動画をつくる…: shown from here, this being a sheet.
    @State private var upscaleRequest: GenerationViewModel.PendingUpscale?

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
                // Joined videos lead; the clips they were joined from fold
                // under them (a joined video can itself be a clip of another).
                List(tree, children: \.children) { node in
                    if let video = node.video {
                        row(video, clipIndex: node.clipIndex)
                    } else {
                        missingRow(node)
                    }
                }
            }
        }
        .padding(H3Spacing.lg)
        .frame(width: 720, height: 520)
        .onAppear {
            viewModel.reloadProjectVideos()
            tree = buildTree()
        }
        .onChange(of: viewModel.projectVideos) { _, _ in tree = buildTree() }
        .sheet(item: $upscaleRequest) { pending in
            UpscaleSheet(viewModel: viewModel, pending: pending) { upscaleRequest = nil }
        }
        .confirmationDialog("この動画を削除しますか？", isPresented: Binding(
            get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }
        ), presenting: pendingDelete) { video in
            Button("ゴミ箱に移動", role: .destructive) { viewModel.deleteProjectVideo(video) }
        } message: { video in
            if let clips = video.record?.joinedClips, viewModel.chainFolder(of: video) != nil {
                Text("つないだ動画と、そのもとの\(clips.count)本の動画をゴミ箱に移動します。")
            } else {
                Text("動画と設定の記録をゴミ箱に移動します。")
            }
        }
    }

    private func row(_ video: ProjectVideo, clipIndex: Int?) -> some View {
        HStack(spacing: H3Spacing.md) {
            Button { show(video) } label: {
                VideoThumbnail(url: video.url, size: clipIndex == nil ? 72 : 56)
                    .overlay(alignment: .bottomTrailing) {
                        // A joined video is marked so it isn't mistaken for
                        // the continuation it ends with.
                        if video.record?.joinedClips != nil {
                            Image(systemName: "film.stack")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundStyle(.white)
                                .padding(4)
                                .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 4))
                                .padding(3)
                        }
                    }
            }
            .buttonStyle(.plain)
            .disabled(viewModel.isGenerating)
            .help(showHelp(video))
            VStack(alignment: .leading, spacing: 2) {
                if let record = video.record, let clips = record.joinedClips {
                    Label(String(localized: "つないだ動画（\(clips.count)本・\(actualLength(record))）"),
                          systemImage: "film.stack")
                        .font(.callout.weight(.semibold))
                        .foregroundStyle(Color.accentColor)
                }
                if let clipIndex {
                    Text("\(clipIndex)本目").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                }
                Text(video.url.lastPathComponent)
                    .font(.callout.monospaced())
                    .lineLimit(1)
                    .truncationMode(.middle)
                if let record = video.record {
                    Text(summary(record)).font(.caption).foregroundStyle(.secondary)
                    if let previous = record.draft.continueFrom {
                        Label(String(localized: "\(URL(fileURLWithPath: previous).lastPathComponent) の続き"),
                              systemImage: "arrow.turn.down.right")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
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
                Button("この動画の続きを作る") {
                    viewModel.continueFromProjectVideo(video)
                    isPresented = false
                }
                .disabled(!viewModel.canContinue(video) || viewModel.isGenerating)
                Button("ここまでをつないで書き出す") {
                    viewModel.exportJoined(endingAt: video)
                }
                .disabled(viewModel.continuationChain(endingAt: video) == nil || viewModel.isExportingJoined)
                Button("拡大した動画をつくる…") {
                    Task { upscaleRequest = await viewModel.upscaleRequest(for: video) }
                }
                .disabled(!viewModel.canUpscale(video))
                .help("この動画を拡大した動画（1.5〜4倍）を、プロジェクトに新しくつくります（音声はそのまま）")
                Divider()
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

    /// The list as a tree: videos no joined video includes, newest first,
    /// with each joined video's clips under it in order.
    private func buildTree() -> [VideoNode] {
        guard let project = viewModel.project else { return [] }
        let included = Set(viewModel.projectVideos.flatMap { $0.record?.joinedClips ?? [] })
        return viewModel.projectVideos
            .filter { !included.contains(ProjectFiles.stored($0.url.path, in: project.url)) }
            .map { node(for: $0, path: $0.id, parentID: "", clipIndex: nil, depth: 0) }
    }

    private func node(for video: ProjectVideo?, path: String, parentID: String,
                      clipIndex: Int?, depth: Int) -> VideoNode {
        let id = parentID + "/" + path
        // depth guards against a record that (wrongly) includes itself.
        let children = depth < 8 ? video?.record?.joinedClips?.enumerated().map { index, clip in
            node(for: viewModel.projectVideo(atStoredPath: clip), path: clip, parentID: id,
                 clipIndex: index + 1, depth: depth + 1)
        } : nil
        return VideoNode(id: id, path: path, video: video, clipIndex: clipIndex, children: children)
    }

    /// A clip a joined video lists but the project no longer has.
    private func missingRow(_ node: VideoNode) -> some View {
        HStack(spacing: H3Spacing.md) {
            if let clipIndex = node.clipIndex {
                Text("\(clipIndex)本目").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            }
            Text(URL(fileURLWithPath: node.path).lastPathComponent)
                .font(.caption.monospaced())
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            Text("見つかりません").font(.caption).foregroundStyle(.secondary)
        }
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

    /// The length a video really has, from its frames ("19.3秒").
    private func actualLength(_ record: ProjectVideoRecord) -> String {
        guard record.actualFrameCount > 0 else { return String(localized: "\(record.draft.seconds)秒") }
        return formatPlaybackSeconds(Double(record.actualFrameCount) / Double(max(record.fps, 1)))
    }

    private func summary(_ record: ProjectVideoRecord) -> String {
        let draft = record.draft
        var parts: [String] = []
        if let profile = SizeProfile(rawValue: draft.sizeProfile) { parts.append(profile.label) }
        // How long the video plays (from its frames: a 5秒 request plays
        // 5.2秒, and a joined video's draft is only its last clip's).
        parts.append(String(localized: "再生時間 \(actualLength(record))"))
        if let speed = SpeedMode(rawValue: draft.speedMode) { parts.append(speed.summaryLabel) }
        if draft.fastAttention { parts.append(String(localized: "高速モード（試験的）")) }
        parts.append(String(localized: "シード \(record.seed)"))
        parts.append(record.joinedClips != nil
                     ? String(localized: "生成時間 合計\(formatElapsed(record.generationSeconds))")
                     : String(localized: "生成時間 \(formatElapsed(record.generationSeconds))"))
        return parts.joined(separator: summarySeparator)
    }
}

/// A still from one second into the video.
struct VideoThumbnail: View {
    let url: URL
    var size: CGFloat = 72
    @State private var image: NSImage?
    @State private var duration: Double?

    var body: some View {
        ZStack {
            Color.black
            if let image {
                Image(nsImage: image).resizable().scaledToFit()
            }
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(alignment: .bottomLeading) {
            // The playback length, read from the file itself.
            if let duration {
                Text(formatPlaybackSeconds(duration))
                    .font(.system(size: 10, weight: .semibold).monospacedDigit())
                    .foregroundStyle(.white)
                    .padding(.horizontal, 3)
                    .padding(.vertical, 1)
                    .background(.black.opacity(0.7), in: RoundedRectangle(cornerRadius: 3))
                    .padding(3)
            }
        }
        .task(id: url) {
            duration = try? await AVURLAsset(url: url).load(.duration).seconds
            image = await Self.thumbnail(for: url)
        }
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

/// One row of the project's video tree; a missing clip has no video.
struct VideoNode: Identifiable {
    let id: String
    let path: String
    let video: ProjectVideo?
    let clipIndex: Int?
    let children: [VideoNode]?
}

/// "5.2秒", "1分30.5秒".
func formatPlaybackSeconds(_ seconds: Double) -> String {
    let tenths = (seconds * 10).rounded() / 10
    guard tenths >= 60 else { return String(format: String(localized: "%.1f秒"), tenths) }
    let minutes = Int(tenths / 60)
    return String(format: String(localized: "%lld分%.1f秒"), minutes, tenths - Double(minutes * 60))
}
