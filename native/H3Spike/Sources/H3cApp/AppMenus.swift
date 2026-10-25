import AppKit
import SwiftUI

// H3cAppについて and the ヘルプ menu: what the app is built on and under
// which licenses (the model's notice "Powered by MiniMax H3" is the one its
// license encourages, §III.3(a)), and the way to the guides on the project
// site, in the app's language.

enum AppLinks {
    static let model = URL(string: "https://huggingface.co/MiniMaxAI/MiniMax-H3")!
    static let modelLicense = URL(string: "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE")!
    static let h3c = URL(string: "https://github.com/antirez/h3.c")!
    static let repository = URL(string: "https://github.com/JunzoKamahara/H3cApp")!
    static let releases = URL(string: "https://github.com/JunzoKamahara/H3cApp/releases")!
    static let issues = URL(string: "https://github.com/JunzoKamahara/H3cApp/issues")!
    static let taehv = URL(string: "https://github.com/madebyollin/taehv")!
    static let realESRGAN = URL(string: "https://github.com/xinntao/Real-ESRGAN")!

    /// The UI is in Japanese unless the app runs in English.
    static var isJapanese: Bool { Bundle.main.preferredLocalizations.first?.hasPrefix("ja") ?? true }

    /// A page of the project site in the app's language.
    static func site(_ page: String) -> URL {
        URL(string: "https://junzokamahara.github.io/H3cApp/" + (isJapanese ? "" : "en/") + page)!
    }

    static var apiGuide: URL {
        URL(string: "https://github.com/JunzoKamahara/H3cApp/blob/main/" + (isJapanese ? "API.ja.md" : "API.md") + "#api")!
    }

    /// A text file in the app's Resources (the licenses), in TextEdit;
    /// the repository's copy if it isn't there.
    static func openResource(_ name: String, fallback: URL) {
        guard let path = Bundle.main.path(forResource: name, ofType: nil) else {
            NSWorkspace.shared.open(fallback)
            return
        }
        let file = URL(fileURLWithPath: path)
        if let textEdit = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.TextEdit") {
            NSWorkspace.shared.open([file], withApplicationAt: textEdit, configuration: NSWorkspace.OpenConfiguration())
        } else {
            NSWorkspace.shared.open(file)
        }
    }
}

/// H3cAppについて: the standard panel with the model's notice and the
/// licenses as its credits.
struct AboutCommand: Commands {
    var body: some Commands {
        CommandGroup(replacing: .appInfo) {
            Button("H3cAppについて") {
                NSApp.orderFrontStandardAboutPanel(options: [.credits: Self.credits()])
                NSApp.activate(ignoringOtherApps: true)
            }
        }
    }

    static func credits() -> NSAttributedString {
        let text = NSMutableAttributedString()
        let paragraph = NSMutableParagraphStyle()
        paragraph.alignment = .center
        paragraph.paragraphSpacing = 6
        let body: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 11),
            .foregroundColor: NSColor.labelColor,
            .paragraphStyle: paragraph,
        ]
        func add(_ string: String, bold: Bool = false, link: URL? = nil) {
            var attributes = body
            if bold { attributes[.font] = NSFont.boldSystemFont(ofSize: 12) }
            if let link { attributes[.link] = link }
            text.append(NSAttributedString(string: string, attributes: attributes))
        }
        add("Powered by ", bold: true)
        add("MiniMax H3", bold: true, link: AppLinks.model)
        add("\n" + String(localized: "MiniMaxの公式製品ではない、独立したアプリです。") + "\n\n")

        add(String(localized: "ライセンス") + "\n", bold: true)
        add(String(localized: "H3cApp：MIT License。推論エンジンは、Salvatore Sanfilippo氏（antirez）の"))
        add("h3.c", link: AppLinks.h3c)
        add(String(localized: "を基にしています。") + "\n")
        add(String(localized: "動画生成モデル MiniMax H3："))
        add("MiniMax H3 Community License", link: AppLinks.modelLicense)
        add(String(localized: "。モデルはアプリに含まれず、別にダウンロードします。利用できる地域が限られ（EU・英国・韓国・米国は対象外）、利用規定があります。生成した動画を公開するときは、AIで生成したことをはっきり示してください。") + "\n")
        add(String(localized: "文章・画像の部分（Qwen3-VL）：Apache License 2.0。") + "\n")
        add(String(localized: "途中の様子のプレビュー用デコーダ（"))
        add("taeh3", link: AppLinks.taehv)
        add(String(localized: "）：MIT License。") + "\n")
        add(String(localized: "シェーダーの一部の設計（ccv）：BSD-3-Clause。") + "\n")
        add(String(localized: "AI超解像による2倍の拡大（"))
        add("Real-ESRGAN", link: AppLinks.realESRGAN)
        add(String(localized: "）：BSD-3-Clause。学習済みの重みをアプリに含めています。") + "\n\n")
        add(String(localized: "ライセンスの本文と第三者のライセンス表示は、ヘルプメニューから開けます。"))
        return text
    }
}

/// The ヘルプ menu: the project site's guides, the API, releases and
/// reports, and the licenses.
struct HelpCommands: Commands {
    var body: some Commands {
        CommandGroup(replacing: .help) {
            Button("H3cAppの使い方") { NSWorkspace.shared.open(AppLinks.site("guide.html")) }
                .keyboardShortcut("?", modifiers: .command)
            Button("はじめ方") { NSWorkspace.shared.open(AppLinks.site("getting-started.html")) }
            Button("詳細設定の説明") { NSWorkspace.shared.open(AppLinks.site("settings.html")) }
            Button("量子化モデル（GGUF）の使い方") { NSWorkspace.shared.open(AppLinks.site("gguf.html")) }
            Button("よくある質問") { NSWorkspace.shared.open(AppLinks.site("faq.html")) }
            Button("APIの説明") { NSWorkspace.shared.open(AppLinks.apiGuide) }
            Divider()
            Button("リリースノート") { NSWorkspace.shared.open(AppLinks.releases) }
            Button("不具合の報告・質問…") { NSWorkspace.shared.open(AppLinks.issues) }
            Divider()
            Button("ライセンスと利用条件の要点") { NSWorkspace.shared.open(AppLinks.site("faq.html#license")) }
            Button("MiniMax H3のライセンス") { NSWorkspace.shared.open(AppLinks.modelLicense) }
            Button("H3cAppのライセンス") {
                AppLinks.openResource("LICENSE", fallback: AppLinks.repository.appendingPathComponent("blob/main/LICENSE"))
            }
            Button("第三者のライセンス表示") {
                AppLinks.openResource("THIRD_PARTY_NOTICES.md",
                                      fallback: AppLinks.repository.appendingPathComponent("blob/main/THIRD_PARTY_NOTICES.md"))
            }
        }
    }
}
