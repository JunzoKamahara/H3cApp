import AppKit
import SwiftUI

let generalSettingsWindowID = "general-settings"

/// 全体設定: settings for the whole app rather than one window (詳細設定 is
/// per window), one tab each - 生成 for what every generation does, 保存画質
/// for how videos are saved, API for its addresses and token and whether
/// other devices may use it.
struct GeneralSettingsView: View {
    @AppStorage("H3cApp.generalSettingsTab") private var tab = Tab.generation

    enum Tab: String {
        case generation, video, api
    }

    var body: some View {
        TabView(selection: $tab) {
            GenerationSettingsTab()
                .tabItem { Label("生成", systemImage: "wand.and.stars") }
                .tag(Tab.generation)
            VideoSettingsTab()
                .tabItem { Label("保存画質", systemImage: "film") }
                .tag(Tab.video)
            APISettingsTab()
                .tabItem { Label("API", systemImage: "network") }
                .tag(Tab.api)
        }
        .frame(width: 560)
        .fixedSize(horizontal: false, vertical: true)
    }
}

/// 生成: what every window's generations do, whatever its own 詳細設定.
private struct GenerationSettingsTab: View {
    @ObservedObject private var preferences = AppPreferences.shared

    var body: some View {
        Form {
            Section("速度") {
                Toggle("条件の計算を近似で速くする", isOn: $preferences.useAdaLNCurve)
                Text("時刻ごとの条件（AdaLN）を小さな近似ファイルから計算し、1本あたり約10秒速くします。どの計算方式でも使えます。初回だけ、モデルごとに約7秒かけて約0.6GBのファイルを作ります。近似の誤差は元の計算の丸め誤差と同じ程度ですが、同じシードでも以前の版とは少し違う動画になります。")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Section {
                DenoisePreviewSettingRow(preferences: preferences, downloader: preferences.denoisePreviewDownloader)
            } header: {
                Text("途中の様子")
            } footer: {
                Text("どちらもすべてのウィンドウに効きます。生成中なら、次の1本から変わります。")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .formStyle(.grouped)
    }
}

/// 保存画質: the quality every video the app makes is saved at (HEVC).
private struct VideoSettingsTab: View {
    @AppStorage("h3c-app.videoQuality") private var savedQuality = VideoQuality.high

    var body: some View {
        Form {
            Section("保存する動画の画質") {
                Picker("保存する動画の画質", selection: $savedQuality) {
                    ForEach(VideoQuality.allCases) { quality in
                        Text(quality == .high ? String(localized: "\(quality.label)（標準）") : quality.label).tag(quality)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                Text("生成・つないで書き出す・拡大・取り込みの動画を、この画質のHEVCで保存します。あとから画質を上げることはできません。「標準」はファイルが小さく（高画質の約3分の1）、明るくなめらかな肌などに細かなブロックが出ることがあります。「最高画質」はブロックがほぼ消えますが、ファイルは高画質の約3倍になります。")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Text("書き出すときのコーデック（H.264で書き出す）と書き出しの画質は、書き出しの画面で選びます。")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .formStyle(.grouped)
    }
}

/// API: addresses and token to copy into scripts, and whether other
/// devices on the network may use it.
private struct APISettingsTab: View {
    @ObservedObject private var api = APIHost.shared
    @State private var showsToken = false
    @State private var confirmingNewToken = false
    /// What was copied last, to say so beside its button for a moment.
    @State private var copied: String?

    var body: some View {
        Form {
            Section("API") {
                LabeledContent("状態") {
                    if let error = api.startError {
                        Text("起動できませんでした（\(error)）").foregroundStyle(.red).textSelection(.enabled)
                    } else if api.isListening {
                        Text("受け付け中")
                    } else {
                        Text("起動しています…")
                    }
                }
                LabeledContent("アドレス") {
                    VStack(alignment: .trailing, spacing: 4) {
                        ForEach(api.addresses, id: \.self) { address in
                            HStack {
                                Text(address).monospaced().textSelection(.enabled)
                                copyButton(address, key: address)
                            }
                        }
                    }
                }
                LabeledContent("アクセストークン") {
                    VStack(alignment: .trailing, spacing: 6) {
                        if let value = api.token?.value {
                            if showsToken {
                                Text(value).monospaced().textSelection(.enabled)
                                    .lineLimit(2).frame(maxWidth: 300, alignment: .trailing)
                            } else {
                                Text(String(repeating: "•", count: 12) + value.suffix(4)).monospaced()
                            }
                        } else {
                            Text("なし").foregroundStyle(.secondary)
                        }
                        HStack {
                            Button { showsToken.toggle() } label: {
                                if showsToken { Text("隠す") } else { Text("表示") }
                            }
                            if let value = api.token?.value { copyButton(value, key: "token") }
                            Button("作り直す…") { confirmingNewToken = true }
                                .disabled(api.token == nil)
                        }
                    }
                }
                LabeledContent("トークンのファイル") {
                    VStack(alignment: .trailing, spacing: 4) {
                        Text(h3APITokenPath).font(.caption).textSelection(.enabled)
                        HStack {
                            Button("Finderで表示") {
                                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: h3APITokenPath)])
                            }
                            .disabled(api.token == nil)
                            copyButton(curlExample, key: "curl", title: "curlの例をコピー")
                        }
                    }
                }
                Group {
                    if api.allowsExternalAccess {
                        Text("リクエストには「Authorization: Bearer <トークン>」ヘッダーが必要です。ほかの機器からのアクセスを許可しているあいだ、トークンはアプリを終了しても変わりません。")
                    } else {
                        Text("リクエストには「Authorization: Bearer <トークン>」ヘッダーが必要です。トークンはアプリを起動するたびに変わるので、スクリプトでは毎回ファイルから読んでください。")
                    }
                }
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                Button("APIの説明") { NSWorkspace.shared.open(AppLinks.apiGuide) }
                    .buttonStyle(.link)
            }

            Section("ほかの機器からのアクセス") {
                Toggle("同じネットワークのほかの機器からAPIを使えるようにする", isOn: Binding(
                    get: { api.allowsExternalAccess },
                    set: { api.setAllowsExternalAccess($0) }
                ))
                Text("ほかのMacやスマートフォンから、上のアドレスとトークンでAPIを使えます。オフに戻すとトークンは作り直されます。")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if api.allowsExternalAccess {
                    Label("通信は暗号化されません。トークンやプロンプト、動画が同じネットワークの機器から読み取られるおそれがあるので、自宅など信頼できるネットワークでだけ使ってください。",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .formStyle(.grouped)
        .confirmationDialog("トークンを作り直しますか？", isPresented: $confirmingNewToken) {
            Button("作り直す", role: .destructive) { api.regenerateToken() }
        } message: {
            Text("今のトークンはすぐに使えなくなります。スクリプトやほかの機器には新しいトークンを設定し直してください。")
        }
    }

    /// A curl command for this Mac that reads the token from its file and
    /// passes it on standard input - never on the command line, where other
    /// accounts could see it with ps.
    private var curlExample: String {
        let home = NSHomeDirectory()
        let path = h3APITokenPath.hasPrefix(home + "/")
            ? "$HOME" + h3APITokenPath.dropFirst(home.count) : h3APITokenPath
        return "printf 'Authorization: Bearer %s\\n' \"$(cat \"\(path)\")\" | curl -H @- http://127.0.0.1:\(h3APIPort)/api/status"
    }

    private func copyButton(_ text: String, key: String, title: LocalizedStringKey = "コピー") -> some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
            copied = key
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                if copied == key { copied = nil }
            }
        } label: {
            if copied == key { Text("コピーしました") } else { Text(title) }
        }
    }
}
