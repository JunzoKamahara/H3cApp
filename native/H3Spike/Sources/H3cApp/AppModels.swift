import SwiftUI

/// The view models behind windows. The primary one lives as long as the app:
/// the HTTP API always talks to it, and a new window shows it whenever no
/// other window does. Closing that window keeps it - its engine, form and
/// result - so the API keeps working from the menu bar alone, and the next
/// window opened picks it up again. Other windows get their own model,
/// released with the window. Closing any window stops the generation it is
/// running and closes its project.
@MainActor
final class AppModels {
    static let shared = AppModels()

    let primary = GenerationViewModel()
    private var primaryShown = false
    private var primaryStarted = false
    private var shown: [ObjectIdentifier: WeakModel] = [:]

    private struct WeakModel {
        weak var model: GenerationViewModel?
    }

    private init() {
        APIHost.shared.start()
    }

    /// The model a new window shows, and whether it starts fresh (load the
    /// engine, reopen the last project or apply the last preset) rather
    /// than picking up the primary model where its last window left it.
    func acquire() -> (model: GenerationViewModel, fresh: Bool) {
        let model: GenerationViewModel
        let fresh: Bool
        if !primaryShown {
            primaryShown = true
            model = primary
            fresh = !primaryStarted
            primaryStarted = true
        } else {
            model = GenerationViewModel()
            fresh = true
        }
        shown[ObjectIdentifier(model)] = WeakModel(model: model)
        refreshAPIStatus()
        return (model, fresh)
    }

    /// The window showing `model` is closing. A generation it is running
    /// stops with it (and the rest of its batch): closing the window is how
    /// people expect to stop one, and nothing would show it any more.
    func release(_ model: GenerationViewModel) {
        model.cancel()
        shown[ObjectIdentifier(model)] = nil
        if model === primary { primaryShown = false }
        // The window's project closes with it (saved first), so another
        // window can open it; it still reopens at the next launch.
        model.closeProject(explicitly: false)
        refreshAPIStatus()
    }

    func isShown(_ model: GenerationViewModel) -> Bool {
        shown[ObjectIdentifier(model)]?.model != nil
    }

    private func refreshAPIStatus() {
        for model in shown.values.compactMap(\.model) {
            model.apiServerStatus = APIHost.shared.status(for: model)
        }
    }
}

/// Holds a window's model for the window's lifetime.
@MainActor
final class WindowModelSlot: ObservableObject {
    let model: GenerationViewModel
    let fresh: Bool

    init() {
        (model, fresh) = AppModels.shared.acquire()
    }
}

/// A window: its model from AppModels, handed back when the window closes.
struct WindowRoot: View {
    @StateObject private var slot = WindowModelSlot()

    var body: some View {
        ContentView(viewModel: slot.model, fresh: slot.fresh)
            .background(WindowCloseObserver { AppModels.shared.release(slot.model) })
            .modifier(FullScreenToolbarAutoHide())
    }
}
