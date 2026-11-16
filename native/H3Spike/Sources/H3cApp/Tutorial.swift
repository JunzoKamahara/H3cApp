import SwiftUI

// チュートリアル: short tours in speech bubbles (popovers) pointing at the
// controls they explain, one step at a time, with 戻る / 次へ and スキップ.
// A tour starts by itself the first time the app runs with a feature the
// user has not been shown yet (new users get the basics first), and any
// tour can be shown again from the ヘルプ menu. A step whose control is not
// on screen (an image's buttons while no image is shown, say) is shown as
// a card at the top of the window instead.
//
// Each window has its own controller (GenerationViewModel.tutorial); the
// controls register themselves with .tutorialAnchor(_:_:) while they are
// on screen. Which tours were shown is kept in UserDefaults.

enum TutorialAnchor: String {
    case kind, prompt, generate, preview, projectMenu
    case imageModel, imageSize, editReferences, imageActions
}

struct TutorialStep {
    /// The control the bubble points at; nil: a card at the top.
    let anchor: TutorialAnchor?
    let title: String
    let body: String
    /// Switch the panel to images first (when nothing is generating).
    var showsImageSide = false
    var arrowEdge: Edge = .top
}

struct Tutorial: Identifiable {
    let id: String
    let title: String
    let steps: [TutorialStep]
}

enum Tutorials {
    /// The ヘルプ menu's list, in order.
    static var all: [Tutorial] { [basics, images, previews] }

    static var basics: Tutorial {
        Tutorial(id: "basics", title: String(localized: "基本の使い方"), steps: [
            TutorialStep(anchor: .kind, title: String(localized: "つくるものを選ぶ"),
                         body: String(localized: "ここで動画か画像かを切り替えます。動画と画像は、それぞれ自分のプロンプトと設定を持ちます。")),
            TutorialStep(anchor: .prompt, title: String(localized: "プロンプトを書く"),
                         body: String(localized: "つくりたい場面を文章で書きます。灰色の例はTabで入力できます。Enterで生成、Shift+Enterで改行です。")),
            TutorialStep(anchor: .generate, title: String(localized: "生成する"),
                         body: String(localized: "このボタンかEnterで生成が始まります。生成中は中止ボタンに変わります。大きさや長さは、このボタンの左の項目で選びます。")),
            TutorialStep(anchor: .preview, title: String(localized: "できあがりを見る"),
                         body: String(localized: "できた動画や画像はここに出ます。ピンチやマウスホイールで拡大縮小、ドラッグで移動、ダブルクリックで元の大きさに戻ります。動画はクリックかスペースキーで再生・停止します。"),
                         arrowEdge: .bottom),
            TutorialStep(anchor: .projectMenu, title: String(localized: "プロジェクト"),
                         body: String(localized: "つくった動画や画像は、プロジェクトに保存されます。このメニューから一覧を開いたり、名前を付けて保存したりできます。"),
                         arrowEdge: .bottom),
        ])
    }

    static var images: Tutorial {
        Tutorial(id: "images-0.10", title: String(localized: "静止画をつくる・編集する"), steps: [
            TutorialStep(anchor: .kind, title: String(localized: "新機能：静止画"),
                         body: String(localized: "ここで「画像」を選ぶと、静止画をつくれます。手元の画像やつくった画像を、指示で直すこともできます。")),
            TutorialStep(anchor: .imageModel, title: String(localized: "画像モデルを選ぶ・入れる"),
                         body: String(localized: "静止画には画像モデルを使います。まだ入っていなければ、このメニューからダウンロードします（MiniMax H3 ImageかP-H3-Image）。"),
                         showsImageSide: true),
            TutorialStep(anchor: .imageSize, title: String(localized: "形・大きさ・枚数"),
                         body: String(localized: "形と大きさ、つくる枚数を選びます。P-H3-Imageは「特大」（約8メガピクセル）で最も良くなります。"),
                         showsImageSide: true),
            TutorialStep(anchor: .editReferences, title: String(localized: "画像を編集する"),
                         body: String(localized: "直したい画像を参照に入れ、変えたいことと残したいことをプロンプトに書いて生成します。例：Edit <Picture 1>. Replace the cup with a small bird. Keep everything else."),
                         showsImageSide: true),
            TutorialStep(anchor: .imageActions, title: String(localized: "できた画像を使う"),
                         body: String(localized: "できた画像の「この画像を編集」で続けて直したり、「高解像度化…」で大きくしたり、「この画像から動画を作る」でその画像から動き出す動画を準備したりできます。"),
                         arrowEdge: .bottom),
        ])
    }

    static var previews: Tutorial {
        Tutorial(id: "previews-0.10", title: String(localized: "プレビューの操作"), steps: [
            TutorialStep(anchor: .preview, title: String(localized: "新機能：プレビューの操作"),
                         body: String(localized: "画像も動画と同じように、ピンチやマウスホイールで拡大縮小、ドラッグで移動できます。ダブルクリックで元の大きさです。動画はスペースキーで再生・停止できます（プロンプトの入力中を除きます）。"),
                         arrowEdge: .bottom),
            TutorialStep(anchor: .preview, title: String(localized: "新機能：画像の途中の様子"),
                         body: String(localized: "画像の生成中は、ここにできあがりつつある様子がステップごとに出ます。思っていた画像と違えば、早めに止められます。全体設定の「生成」でオフにできます。"),
                         arrowEdge: .bottom),
        ])
    }

    /// For people who used an earlier version: the API now starts off.
    static var apiOff: Tutorial {
        Tutorial(id: "api-off-0.10", title: String(localized: "APIについてのお知らせ"), steps: [
            TutorialStep(anchor: nil, title: String(localized: "APIは初めはオフになりました"),
                         body: String(localized: "このバージョンから、HTTPのAPIは使うときだけオンにする形になりました。スクリプトなどから使うときは、全体設定（アプリのメニューの「全体設定…」、⌥⌘,）の「API」タブで「APIを使う」をオンにしてください。オンにした設定は次の起動にも引き継がれます。")),
        ])
    }
}

@MainActor
final class TutorialController: ObservableObject {
    @Published private(set) var tutorial: Tutorial?
    @Published private(set) var index = 0
    /// Anchors now on screen.
    @Published private(set) var visible: Set<TutorialAnchor> = []
    /// Anchors whose bubble was closed from outside during this step; the
    /// step goes on in the card instead.
    @Published private var closed: Set<TutorialAnchor> = []
    /// Bumped when a step wants the folded panel opened (ContentView).
    @Published private(set) var expandComposerRequest = 0
    private var queue: [Tutorial] = []
    /// Switches the panel to images for a step that shows its controls.
    var showImageSide: () -> Void = {}

    private static let seenKey = "h3c-app.tutorialsSeen"
    /// Only the first window starts tours by itself.
    private static var autoStarted = false

    var step: TutorialStep? {
        guard let tutorial, tutorial.steps.indices.contains(index) else { return nil }
        return tutorial.steps[index]
    }

    /// The anchor the bubble is attached to now; nil while no tour runs or
    /// the step's control is not on screen (the window's card shows it).
    var presentedAnchor: TutorialAnchor? {
        guard let anchor = step?.anchor, visible.contains(anchor), !closed.contains(anchor) else { return nil }
        return anchor
    }

    var showsCard: Bool { tutorial != nil && presentedAnchor == nil }

    func register(_ anchor: TutorialAnchor) { visible.insert(anchor) }
    func unregister(_ anchor: TutorialAnchor) { visible.remove(anchor) }
    func closedFromOutside(_ anchor: TutorialAnchor) { closed.insert(anchor) }

    /// From the ヘルプ menu: this tour now, whatever was shown before.
    func show(_ tutorial: Tutorial) {
        queue = []
        begin(tutorial)
    }

    /// At a window's first appearance: the tours this user has not seen.
    /// A new user (no model and no project yet) gets the basics and the
    /// image tour; someone updating from an earlier version only what is
    /// new to them.
    func startPendingIfAny(isNewUser: Bool) {
        guard !Self.autoStarted, tutorial == nil else { return }
        Self.autoStarted = true
        let defaults = UserDefaults.standard
        var seen = Set(defaults.stringArray(forKey: Self.seenKey) ?? [])
        let firstTime = defaults.object(forKey: Self.seenKey) == nil
        if firstTime && !isNewUser {
            // They know the basics already.
            seen.insert(Tutorials.basics.id)
        }
        // Someone updating hears first that their scripts need the API on.
        var pending = [Tutorials.basics, Tutorials.images, Tutorials.previews]
        // (Not if they have turned it on already.)
        if isNewUser || APIHost.shared.isEnabled {
            seen.insert(Tutorials.apiOff.id)
        } else {
            pending.insert(Tutorials.apiOff, at: 0)
        }
        defaults.set(Array(seen), forKey: Self.seenKey)
        queue = pending.filter { !seen.contains($0.id) }
        guard !queue.isEmpty else { return }
        begin(queue.removeFirst())
    }

    func next() {
        guard let tutorial else { return }
        if index + 1 < tutorial.steps.count {
            index += 1
            prepareStep()
        } else {
            finish()
        }
    }

    func back() {
        guard index > 0 else { return }
        index -= 1
        prepareStep()
    }

    /// スキップ: this tour and the ones queued after it.
    func skip() {
        for pending in queue { markSeen(pending.id) }
        queue = []
        tutorial = nil
    }

    private func finish() {
        tutorial = nil
        if !queue.isEmpty {
            let following = queue.removeFirst()
            // Let the last bubble close before the next tour's opens.
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 400_000_000)
                begin(following)
            }
        }
    }

    private func begin(_ tutorial: Tutorial) {
        markSeen(tutorial.id)
        self.tutorial = tutorial
        index = 0
        prepareStep()
    }

    private func prepareStep() {
        closed = []
        guard let step else { return }
        if step.showsImageSide { showImageSide() }
        if step.anchor != .projectMenu && step.anchor != .preview && step.anchor != .imageActions {
            expandComposerRequest += 1
        }
    }

    private func markSeen(_ id: String) {
        var seen = Set(UserDefaults.standard.stringArray(forKey: Self.seenKey) ?? [])
        seen.insert(id)
        UserDefaults.standard.set(Array(seen), forKey: Self.seenKey)
    }
}

/// The bubble's content: what the step is about, where it is in the tour,
/// and スキップ / 戻る / 次へ (完了 on the last step).
struct TutorialBubble: View {
    @ObservedObject var controller: TutorialController

    var body: some View {
        if let tutorial = controller.tutorial, let step = controller.step {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .firstTextBaseline) {
                    Image(systemName: "lightbulb.fill").foregroundStyle(.yellow)
                    Text(step.title).font(.headline)
                    Spacer(minLength: 8)
                    Text("\(controller.index + 1)/\(tutorial.steps.count)")
                        .font(.caption).monospacedDigit().foregroundStyle(.secondary)
                }
                Text(step.body)
                    .font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
                HStack {
                    Button("スキップ") { controller.skip() }
                        .help("このチュートリアルを閉じます。ヘルプメニューの「チュートリアル」からいつでも見られます")
                    Spacer()
                    if controller.index > 0 {
                        Button("戻る") { controller.back() }
                    }
                    Button(controller.index + 1 < tutorial.steps.count ? String(localized: "次へ") : String(localized: "完了")) {
                        controller.next()
                    }
                    .keyboardShortcut(.defaultAction)
                }
                .controlSize(.small)
            }
            .padding(14)
            .frame(width: 340)
        }
    }
}

private struct TutorialAnchorModifier: ViewModifier {
    @ObservedObject var controller: TutorialController
    let anchor: TutorialAnchor

    func body(content: Content) -> some View {
        content
            .onAppear { controller.register(anchor) }
            .onDisappear { controller.unregister(anchor) }
            .popover(isPresented: Binding(
                get: { controller.presentedAnchor == anchor },
                // Closed from outside (a click elsewhere): the tour waits
                // in the card at the top of the window instead of vanishing.
                set: { shown in if !shown && controller.presentedAnchor == anchor { controller.closedFromOutside(anchor) } }),
                     arrowEdge: controller.step?.arrowEdge ?? .top) {
                TutorialBubble(controller: controller)
            }
    }
}

extension View {
    /// Lets tutorial steps point at this view while it is on screen.
    func tutorialAnchor(_ anchor: TutorialAnchor, _ controller: TutorialController) -> some View {
        modifier(TutorialAnchorModifier(controller: controller, anchor: anchor))
    }
}

/// A step whose control is not on screen, at the top of the window.
struct TutorialCard: View {
    @ObservedObject var controller: TutorialController

    var body: some View {
        if controller.showsCard {
            TutorialBubble(controller: controller)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.secondary.opacity(0.25)))
                .shadow(radius: 8, y: 2)
                .padding(.top, 12)
        }
    }
}
