import SwiftUI

// The preview fills the window; the composer (作り方, prompt, shape / size /
// length, generate) floats in front of it at the bottom centre and can be
// dragged elsewhere or collapsed. 詳細設定 is a separate dialog. Each window
// has its own state (see AppModels): a new window starts like a fresh launch,
// from the preset used last, unless it picks up the primary model.
struct ContentView: View {
    @ObservedObject var viewModel: GenerationViewModel
    /// False when this window picks up the primary model where its last
    /// window left it.
    let fresh: Bool
    @State private var composerOffset: CGSize = .zero
    @State private var composerCollapsed = false
    @State private var presetName = ""
    @State private var showingModelManager = false
    @State private var showingDownloadWizard = false
    // Auto-offer the download wizard once per launch when the engine can't
    // load a model - not on every relaunch attempt the user might trigger
    // via "もう一度試す", and not re-shown after they dismiss it once.
    @State private var hasOfferedDownloadWizard = false

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .bottom) {
                PreviewPane(viewModel: viewModel,
                            bottomInset: composerCollapsed ? 60 : 260)

                CreationFormView(
                    viewModel: viewModel, library: viewModel.library,
                    showingModelManager: $showingModelManager,
                    offset: $composerOffset, isCollapsed: $composerCollapsed,
                    bounds: geometry.size
                )
                .offset(composerOffset)
                .onHover { viewModel.isPointerOverComposer = $0 }
                .padding(.bottom, 20)
            }
            .background {
                // ⌘, opens this window's 詳細設定 - a key only, with no menu
                // item, and working while the prompt panel is folded away
                // with its 詳細設定 button.
                Button("詳細設定を開く") { viewModel.showingAdvancedSettings = true }
                    .keyboardShortcut(",", modifiers: .command)
                    .opacity(0)
                    .accessibilityHidden(true)
            }
            .sheet(item: $viewModel.pendingVideoImport) { pending in
                ImportVideoSheet(viewModel: viewModel, pending: pending)
            }
            .background {
                // 動画ファイルを2倍に拡大… (a second sheet, on its own view).
                Color.clear.sheet(item: $viewModel.pendingUpscale) { pending in
                    UpscaleSheet(viewModel: viewModel, pending: pending) { viewModel.pendingUpscale = nil }
                }
            }
            .overlay(alignment: .top) {
                // 動画ファイルの続きを作る: converting takes a little while.
                if viewModel.isImportingVideo || viewModel.isUpscalingVideo {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        if viewModel.isImportingVideo {
                            Text("動画を取り込んでいます（\(viewModel.importingVideoName)）…").font(.callout)
                        } else {
                            Text("動画を\(viewModel.upscalingFactor.label)に拡大しています（\(viewModel.upscalingVideoName)、\(Int((viewModel.upscaleProgress * 100).rounded()))%）…")
                                .font(.callout)
                                .monospacedDigit()
                            Button("中止") { viewModel.cancelUpscale() }
                                .controlSize(.small)
                        }
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(.regularMaterial, in: Capsule())
                    .padding(.top, 12)
                }
            }
        }
        .frame(minWidth: 760, minHeight: 600)
        // Once a generation starts (button, Return, ⌘Return or the HTTP
        // API), get the composer out of the way of the preview.
        .onChange(of: viewModel.isGenerating) { generating in
            if generating { withAnimation(.easeInOut(duration: 0.15)) { composerCollapsed = true } }
        }
        .alert("プリセットとして保存", isPresented: Binding(
            get: { viewModel.presetPendingName != nil },
            set: { if !$0 { viewModel.presetPendingName = nil } }
        )) {
            TextField("名前", text: $presetName)
            Button("保存") {
                if let preset = viewModel.presetPendingName {
                    viewModel.savePreset(preset, named: presetName)
                }
                presetName = ""
                viewModel.presetPendingName = nil
            }
            Button("キャンセル", role: .cancel) {
                presetName = ""
                viewModel.presetPendingName = nil
            }
        } message: {
            Text("形・大きさ・長さと詳細設定を保存します（プロンプトは含みません）。同じ名前のプリセットは上書きされます。")
        }
        .sheet(isPresented: $viewModel.showingAdvancedSettings) {
            AdvancedSettingsView(
                viewModel: viewModel, library: viewModel.library,
                showingModelManager: $showingModelManager
            )
        }
        // The window title (WindowGroup in H3cApp.swift) already names the
        // app; a toolbar title next to it showed the name twice.
        .toolbar {
            ToolbarItem(placement: .navigation) {
                ProjectMenu(viewModel: viewModel)
            }
            ToolbarItem(placement: .automatic) {
                Button {
                    showingModelManager = true
                } label: {
                    Label("モデル管理", systemImage: "square.stack.3d.up")
                }
            }
            ToolbarItem(placement: .automatic) {
                EngineStatusButton(
                    viewModel: viewModel, library: viewModel.library,
                    showingModelManager: $showingModelManager
                )
            }
        }
        .navigationTitle(viewModel.project.map { "\($0.name) — H3cApp" } ?? "H3cApp")
        .sheet(isPresented: $viewModel.showingNewProject) {
            NewProjectSheet(viewModel: viewModel, isPresented: $viewModel.showingNewProject)
        }
        .sheet(isPresented: $viewModel.showingSaveAsProject) {
            NewProjectSheet(viewModel: viewModel, isPresented: $viewModel.showingSaveAsProject, savingAs: true)
        }
        .sheet(isPresented: $viewModel.showingBatchProject) {
            BatchProjectSheet(viewModel: viewModel, store: ProjectStore.shared,
                              isPresented: $viewModel.showingBatchProject)
        }
        .sheet(isPresented: $viewModel.showingProjectVideos) {
            ProjectVideosSheet(viewModel: viewModel, isPresented: $viewModel.showingProjectVideos)
        }
        .alert("プロジェクト", isPresented: Binding(
            get: { viewModel.projectMessage != nil && !viewModel.showingNewProject
                && !viewModel.showingBatchProject && !viewModel.showingSaveAsProject },
            set: { if !$0 { viewModel.projectMessage = nil } }
        )) {
            if let file = viewModel.projectMessageFile {
                Button("Finderで表示") {
                    NSWorkspace.shared.activateFileViewerSelecting([file])
                    viewModel.projectMessage = nil
                }
            }
            Button("OK") { viewModel.projectMessage = nil }
        } message: {
            Text(viewModel.projectMessage ?? "")
        }
        .sheet(isPresented: $showingModelManager) {
            ModelManagerView(library: viewModel.library, isGenerating: viewModel.isGenerating) { id in
                viewModel.switchModel(to: id)
            }
        }
        .sheet(isPresented: $showingDownloadWizard) {
            ModelDownloadWizardView(
                suggestedDestination: viewModel.library.activeModel?.path ?? defaultH3ModelDownloadPath,
                onCompleted: { path in
                    if let id = viewModel.library.activeModelID {
                        viewModel.library.setModelPath(id: id, path: path)
                        viewModel.switchModel(to: id)
                    } else {
                        viewModel.library.addModel(path: path)
                        if let newID = viewModel.library.activeModelID { viewModel.switchModel(to: newID) }
                    }
                    showingDownloadWizard = false
                },
                onCancelled: { showingDownloadWizard = false },
                library: viewModel.library,
                onImported: { path, use in
                    viewModel.library.addModel(path: path)
                    if use || viewModel.library.models.count == 1,
                       let entry = viewModel.library.models.first(where: { $0.path == path }) {
                        viewModel.switchModel(to: entry.id)
                    }
                    showingDownloadWizard = false
                }
            )
        }
        .focusedSceneObject(viewModel)
        .onAppear {
            let store = ProjectStore.shared
            if fresh { viewModel.loadModel() }
            if let action = store.pendingAction {
                // Opened from the menu bar's プロジェクト menu with no window.
                store.pendingAction = nil
                store.didReopenAtLaunch = true
                if fresh { viewModel.applyLastUsedPreset() }
                switch action {
                case .new: viewModel.showingNewProject = true
                case .choose: DispatchQueue.main.async { viewModel.chooseAndOpenProject() }
                case .open(let url): viewModel.openProject(at: url)
                }
            } else if fresh && !viewModel.reopenLastProjectIfAny() {
                // The first window reopens the project open at the last
                // quit; otherwise the form starts from the preset used last.
                viewModel.applyLastUsedPreset()
            }
        }
        // initial: the engine can already have failed (no model registered)
        // before this window's first frame.
        .onChange(of: viewModel.engineState, initial: true) { _, newValue in
            if case .failed = newValue, !hasOfferedDownloadWizard {
                hasOfferedDownloadWizard = true
                showingDownloadWizard = true
            }
        }
    }
}

/// Calls `onClose` when the window holding this view is about to close.
struct WindowCloseObserver: NSViewRepresentable {
    let onClose: () -> Void

    func makeNSView(context: Context) -> ObserverView {
        let view = ObserverView()
        view.onClose = onClose
        return view
    }

    func updateNSView(_ nsView: ObserverView, context: Context) {
        nsView.onClose = onClose
    }

    final class ObserverView: NSView {
        var onClose: (() -> Void)?
        private var token: NSObjectProtocol?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            if let token { NotificationCenter.default.removeObserver(token) }
            token = nil
            guard let window else { return }
            token = NotificationCenter.default.addObserver(
                forName: NSWindow.willCloseNotification, object: window, queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated { self?.onClose?() }
            }
        }

        deinit {
            if let token { NotificationCenter.default.removeObserver(token) }
        }
    }
}

/// In full screen, hides the window's toolbar (project menu and title) so
/// the preview takes the whole screen; moving the pointer to the top edge
/// shows it. SwiftUI's own setting: the app's presentation options
/// (autoHideToolbar) didn't hide SwiftUI's toolbar.
struct FullScreenToolbarAutoHide: ViewModifier {
    func body(content: Content) -> some View {
        content.windowToolbarFullScreenVisibility(.onHover)
    }
}
