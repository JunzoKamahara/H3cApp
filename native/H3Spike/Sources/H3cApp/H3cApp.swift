import Foundation
import SwiftUI

@main
struct H3cApp: App {
    init() {
        // libh3.a resolves "h3_shaders.metal" relative to the process's
        // working directory (see h3.c). When running from inside an .app
        // bundle, LaunchServices doesn't guarantee any particular cwd, so
        // point it at Contents/Resources ourselves if the shader is there.
        if let resourcePath = Bundle.main.resourcePath,
           FileManager.default.fileExists(atPath: resourcePath + "/h3_shaders.metal") {
            FileManager.default.changeCurrentDirectoryPath(resourcePath)
        }
        // H3_ATTENTION_CACHE is no longer set here: GenerationViewModel now
        // picks the right cache (default/Ref2VA/Turbo) per job and passes it
        // through H3GenerationParams.attentionCachePath instead, since which
        // one is correct depends on that job's mode.

        // The primary model and the HTTP API exist from launch, independent
        // of windows (see AppModels).
        // Settles the app's folder (renamed from h3c-app) before anything
        // reads paths under it.
        _ = h3AppSupportDirectory
        _ = AppModels.shared
    }

    var body: some Scene {
        // The app is named H3cApp wherever the user sees it - this
        // window's title, the app/menu-bar name from Info.plist, the
        // bundle and executable filenames from package_app.sh, and the
        // Application Support folder. Only internal identifiers (the bundle
        // id, settings keys) still say h3c-app.
        WindowGroup("H3cApp", id: mainWindowID) {
            WindowRoot()
        }
        .commands {
            AboutCommand()
            GeneralSettingsCommand()
            ProjectCommands()
            VideoCommands()
            HelpCommands()
        }
        // Design spec section 3: 1180x780 initial content area, 920x640
        // minimum. .automatic lets the window be resized by hand within
        // that range instead of being pinned to one exact size.
        .windowResizability(.automatic)
        .defaultSize(width: 1180, height: 780)
        // Without this, AppKit cascades each new window progressively
        // down/right from the last - after many relaunches during
        // development the window had drifted low enough to push its bottom
        // (Save As...) past the dock. Centering keeps it reachable.
        .defaultPosition(.center)

        Window("全体設定", id: generalSettingsWindowID) {
            GeneralSettingsView()
        }
        .windowResizability(.contentSize)
        .defaultPosition(.center)
    }
}

let mainWindowID = "main"

/// The プロジェクト menu in the menu bar, acting on the frontmost window
/// (each window has its own project). Same items as the toolbar menu;
/// disabled while that window is generating. With no window open, creating
/// or opening a project opens a new window for it.
struct ProjectCommands: Commands {
    @FocusedObject private var viewModel: GenerationViewModel?
    @Environment(\.openWindow) private var openWindow
    @ObservedObject private var store = ProjectStore.shared

    var body: some Commands {
        CommandMenu("プロジェクト") {
            if let viewModel {
                ProjectMenuItems(viewModel: viewModel, store: store)
                    .disabled(viewModel.isGenerating)
            } else {
                Button("新規プロジェクト…") { openWindow(for: .new) }
                    .keyboardShortcut("n", modifiers: [.command, .shift])
                Button("プロジェクトを開く…") { openWindow(for: .choose) }
                    .keyboardShortcut("o", modifiers: .command)
                Menu("最近のプロジェクト") {
                    ForEach(store.existingRecents.filter { !store.isOpen($0) }, id: \.self) { url in
                        Button(url.lastPathComponent) { openWindow(for: .open(url)) }
                    }
                }
                .disabled(store.existingRecents.allSatisfy { store.isOpen($0) })
            }
        }
    }

    private func openWindow(for action: PendingProjectAction) {
        store.pendingAction = action
        openWindow(id: mainWindowID)
    }
}

/// The 動画 menu: enlarging the video in the frontmost window's preview,
/// or any video file - neither needs a project. Disabled while that window
/// is generating.
struct VideoCommands: Commands {
    @FocusedObject private var viewModel: GenerationViewModel?

    var body: some Commands {
        CommandMenu("動画") {
            let busy = viewModel.map { $0.isGenerating || $0.isUpscalingVideo } ?? true
            Button("表示中の動画を拡大…") { viewModel?.upscaleDisplayedVideo() }
                .disabled(busy || !(viewModel?.canUpscaleDisplayedVideo ?? false))
            Button("動画ファイルを拡大…") { viewModel?.pickVideoToUpscale() }
                .disabled(busy)
        }
    }
}

/// 全体設定… in the app menu, where Settings… normally sits (⌥⌘,): the
/// settings for the whole app. Each window's own settings (詳細設定) open
/// from its prompt panel, or with ⌘, (ContentView).
struct GeneralSettingsCommand: Commands {
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandGroup(replacing: .appSettings) {
            Button("全体設定…") { openWindow(id: generalSettingsWindowID) }
                .keyboardShortcut(",", modifiers: [.command, .option])
        }
    }
}
