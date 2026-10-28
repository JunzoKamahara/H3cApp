import SwiftUI

// Toolbar-right status chip (design spec section 3): shows readiness with
// an icon + word, not color alone, and opens a popover with the technical
// details (GPU/model) that used to sit directly in the form.
struct EngineStatusButton: View {
    @ObservedObject var viewModel: GenerationViewModel
    @ObservedObject var library: ModelLibrary
    @Binding var showingModelManager: Bool
    @State private var showingPopover = false
    @Environment(\.openWindow) private var openWindow

    private var statusText: String {
        switch viewModel.engineState {
        case .loading: return String(localized: "モデルを準備しています")
        case .ready: return String(localized: "準備完了")
        case .failed: return String(localized: "モデルが見つかりません")
        }
    }

    private var statusIcon: String {
        switch viewModel.engineState {
        case .loading: return "hourglass"
        case .ready: return "checkmark.circle.fill"
        case .failed: return "exclamationmark.triangle.fill"
        }
    }

    var body: some View {
        Button {
            showingPopover = true
        } label: {
            Label(statusText, systemImage: statusIcon)
                .font(.callout)
        }
        .buttonStyle(.plain)
        .popover(isPresented: $showingPopover, arrowEdge: .bottom) {
            statusDetail
        }
        .accessibilityLabel(statusText)
    }

    private var statusDetail: some View {
        VStack(alignment: .leading, spacing: H3Spacing.sm) {
            Text("動作環境").font(.headline)
            Text(viewModel.deviceLine).font(.body).textSelection(.enabled)
            Divider()
            Text("使用中のモデル").font(.caption).foregroundStyle(.secondary)
            Text(library.activeModel?.name ?? String(localized: "未登録")).font(.body).textSelection(.enabled)
            Text(viewModel.modelDirectory).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            Button("モデル管理…") {
                showingPopover = false
                showingModelManager = true
            }
            .font(.caption)
            Divider()
            Text("APIサーバー").font(.caption).foregroundStyle(.secondary)
            Text(viewModel.apiServerStatus).font(.caption).textSelection(.enabled)
            Button("全体設定…") {
                showingPopover = false
                openWindow(id: generalSettingsWindowID)
            }
            .font(.caption)
            if case .failed(let message) = viewModel.engineState {
                Divider()
                Text("読み込みエラー").font(.caption).foregroundStyle(.secondary)
                Text(message).font(.caption).foregroundStyle(.red)
                Button("もう一度試す") { viewModel.loadModel() }
            }
        }
        .padding(H3Spacing.lg)
        .frame(minWidth: 320)
    }
}
