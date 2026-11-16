import SwiftUI

// チュートリアル: short tours in speech bubbles (popovers) pointing at the
// controls they explain, one step at a time, with 戻る / 次へ and スキップ.
// Three kinds start by themselves, once each:
// - the basics, at a new user's first launch;
// - first-use tours (Tutorials.firstUse), the first time a feature's
//   control comes on screen - 画像から, a continuation, the project list,
//   詳細設定, モデル管理 and so on (the anchor's registration offers it);
// - new-feature tours, at the first launch of a version that adds them.
// 全体設定 > チュートリアル turns the first-use and the new-feature tours
// off separately, and any tour can be shown again from the ヘルプ menu. A
// step whose control is not on screen (an image's buttons while no image is
// shown, say) is shown as a card at the top of the window instead.
//
// Each window has its own controller (GenerationViewModel.tutorial); the
// controls register themselves with .tutorialAnchor(_:_:) while they are
// on screen. Which tours were shown is kept in UserDefaults.

enum TutorialAnchor: String {
    case kind, prompt, generate, preview, projectMenu
    case imageModel, imageSize, editReferences, imageActions, upscaleButton
    case imageInput, continuation, lipSync, videoActions
    case projectList, advancedSettings, modelManager
}

/// What a step opens first so that its control is on screen (ContentView
/// does it; nothing while a generation runs).
enum TutorialReveal {
    /// The panel's image side.
    case imageSide
    /// The panel's video side, 画像から.
    case videoFromImage
    case projectList, advancedSettings, modelManager
}

struct TutorialStep {
    /// The control the bubble points at; nil: a card at the top.
    let anchor: TutorialAnchor?
    let title: String
    let body: String
    var reveal: TutorialReveal?
    var arrowEdge: Edge = .top
}

struct Tutorial: Identifiable {
    let id: String
    let title: String
    let steps: [TutorialStep]
}

enum Tutorials {
    /// The ヘルプ menu's list, in groups: the basics, the features as people
    /// come to them, then the newest.
    static var menuGroups: [[Tutorial]] {
        [[basics],
         [videoResult, videoFromImage, continuation, lipSync, projectList, advancedSettings, modelManager],
         [images, upscale, previews]]
    }

    /// Tours for a feature someone uses for the first time, by the control
    /// that shows it is in use.
    static var firstUse: [Tutorial] {
        [videoFromImage, continuation, lipSync, videoResult, projectList, advancedSettings, modelManager]
    }

    static func firstUse(for anchor: TutorialAnchor) -> Tutorial? {
        firstUse.first { $0.steps.first?.anchor == anchor }
    }

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

    // MARK: 初めて使う機能

    static var videoFromImage: Tutorial {
        Tutorial(id: "first-video-from-image", title: String(localized: "画像から動画をつくる"), steps: [
            TutorialStep(anchor: .imageInput, title: String(localized: "画像から動き始める"),
                         body: String(localized: "「最初・最後の画像」では、選んだ画像から動き始める動画をつくります。「最後の画像も指定…」で、終わりの画像も決められます。プロンプトには、画像の中で何がどう動くかを書きます。"),
                         reveal: .videoFromImage),
            TutorialStep(anchor: .imageInput, title: String(localized: "参照画像・動画・音声"),
                         body: String(localized: "「参照画像・動画・音声」では、人物や物の見た目、声などの特徴を参考にして新しい動画をつくります。最初のフレームが参照と同じになるとは限りません。"),
                         reveal: .videoFromImage),
        ])
    }

    static var continuation: Tutorial {
        Tutorial(id: "first-continuation", title: String(localized: "動画の続きをつくる"), steps: [
            TutorialStep(anchor: .continuation, title: String(localized: "続きをつくる"),
                         body: String(localized: "元の動画の最後の約0.9秒を引き継いで、続きの動画をつくります。プロンプトには、続きで起きることを書きます。")),
            TutorialStep(anchor: .continuation, title: String(localized: "つないで1本にする"),
                         body: String(localized: "続きができたら、パネルの「つないで書き出す」か、プロジェクトの一覧の「ここまでをつないで書き出す」で、元の動画と重なりをつないで1本にします。")),
        ])
    }

    static var lipSync: Tutorial {
        Tutorial(id: "first-lip-sync", title: String(localized: "音声に合わせる（リップシンク）"), steps: [
            TutorialStep(anchor: .lipSync, title: String(localized: "音声に合わせて映像をつくる"),
                         body: String(localized: "選んだ音声に合わせて、口の動きなどの映像をつくります。動画の音は、この音声そのものになります。音声の何秒目から使うかは、ここで決めます。プロンプトには、話している人の見た目や場面を書きます。")),
        ])
    }

    static var videoResult: Tutorial {
        Tutorial(id: "first-video-result", title: String(localized: "できた動画を使う"), steps: [
            TutorialStep(anchor: .videoActions, title: String(localized: "書き出す・設定を見る"),
                         body: String(localized: "「動画を書き出す…」（⌘S）で、好きな場所に保存します。「使用した設定…」では、この動画のプロンプトやシードなどを確かめられます。"),
                         arrowEdge: .bottom),
            TutorialStep(anchor: .videoActions, title: String(localized: "次の動画につなげる"),
                         body: String(localized: "「参照に使う」で、この動画を次の動画の参照にできます。パネルの「続きを生成」を押すと、この動画の続きをつくれます。"),
                         arrowEdge: .bottom),
        ])
    }

    static var projectList: Tutorial {
        Tutorial(id: "first-project-list", title: String(localized: "プロジェクトの動画と画像"), steps: [
            TutorialStep(anchor: .projectList, title: String(localized: "つくった動画と画像の一覧"),
                         body: String(localized: "このプロジェクトでつくった動画と画像の一覧です。「表示」でプレビューに出し、「設定を戻す」でそのときのプロンプトと設定をパネルに戻します。右上の切り替えで、すべてのプロジェクトから探すこともできます。"),
                         reveal: .projectList, arrowEdge: .bottom),
            TutorialStep(anchor: .projectList, title: String(localized: "「その他」でできること"),
                         body: String(localized: "動画の「その他」からは、続きをつくる、ここまでをつないで書き出す、拡大した動画をつくる、参照に使う、別のプロジェクトに移動するなどができます。画像の「その他」からは、編集や高解像度化ができます。"),
                         reveal: .projectList, arrowEdge: .bottom),
        ])
    }

    static var advancedSettings: Tutorial {
        Tutorial(id: "first-advanced-settings", title: String(localized: "詳細設定とプリセット"), steps: [
            TutorialStep(anchor: .advancedSettings, title: String(localized: "詳細設定"),
                         body: String(localized: "生成ステップ数、計算方式、速度、ランダムさ（シード）、LoRAなどを決めます。変えた設定は次の生成から使われます。"),
                         reveal: .advancedSettings, arrowEdge: .bottom),
            TutorialStep(anchor: .advancedSettings, title: String(localized: "プリセット"),
                         body: String(localized: "よく使う形・大きさ・長さと詳細設定は、パネルの「プリセット」に名前を付けて保存し、あとでまとめて戻せます（プロンプトは含みません）。"),
                         reveal: .advancedSettings, arrowEdge: .bottom),
        ])
    }

    static var modelManager: Tutorial {
        Tutorial(id: "first-model-manager", title: String(localized: "モデル管理"), steps: [
            TutorialStep(anchor: .modelManager, title: String(localized: "モデルの種類"),
                         body: String(localized: "動画をつくるH3モデル、画像モデル、LoRA、RefMod、拡大モデルを、上の切り替えごとに管理します。"),
                         reveal: .modelManager, arrowEdge: .bottom),
            TutorialStep(anchor: .modelManager, title: String(localized: "モデルを入れる"),
                         body: String(localized: "H3モデルは「ダウンロードして追加…」で入れるか、手元のGGUFやフォルダを登録します。拡大モデル（SwiftVR・SeedVR2）を入れると、動画や画像を大きくするときに選べるようになります。"),
                         reveal: .modelManager, arrowEdge: .bottom),
        ])
    }

    // MARK: 新機能

    static var images: Tutorial {
        Tutorial(id: "images-0.10", title: String(localized: "静止画をつくる・編集する"), steps: [
            TutorialStep(anchor: .kind, title: String(localized: "新機能：静止画"),
                         body: String(localized: "ここで「画像」を選ぶと、静止画をつくれます。手元の画像やつくった画像を、指示で直すこともできます。")),
            TutorialStep(anchor: .imageModel, title: String(localized: "画像モデルを選ぶ・入れる"),
                         body: String(localized: "静止画には画像モデルを使います。まだ入っていなければ、このメニューからダウンロードします（MiniMax H3 ImageかP-H3-Image）。"),
                         reveal: .imageSide),
            TutorialStep(anchor: .imageSize, title: String(localized: "形・大きさ・枚数"),
                         body: String(localized: "形と大きさ、つくる枚数を選びます。P-H3-Imageは「特大」（約8メガピクセル）で最も良くなります。"),
                         reveal: .imageSide),
            TutorialStep(anchor: .editReferences, title: String(localized: "画像を編集する"),
                         body: String(localized: "直したい画像を参照に入れ、変えたいことと残したいことをプロンプトに書いて生成します。例：Edit <Picture 1>. Replace the cup with a small bird. Keep everything else."),
                         reveal: .imageSide),
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

    static var upscale: Tutorial {
        Tutorial(id: "image-upscale", title: String(localized: "画像の高解像度化"), steps: [
            TutorialStep(anchor: .upscaleButton, title: String(localized: "新機能：画像の高解像度化"),
                         body: String(localized: "できた画像の「高解像度化…」で、画像を大きくして新しい画像としてプロジェクトに加えます。元の画像はそのまま残ります。プロジェクトの画像の一覧では「その他」メニューにあります。"),
                         arrowEdge: .bottom),
            TutorialStep(anchor: .upscaleButton, title: String(localized: "描き直すか、そのまま拡大するか"),
                         body: String(localized: "「画像モデルで描き直す」は細かいところを描き足しますが、顔や模様が少し変わることがあります（強さは控えめ・標準・強め）。元の絵のまま大きくしたいときは、Real-ESRGANやSwiftVRなど、動画の拡大と同じ方法を選びます。"),
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
    /// First-use tours offered before the launch's tours were sorted out
    /// (startPendingIfAny), to offer again then.
    private var deferredOffers: [TutorialAnchor] = []
    /// Opens what a step shows (ContentView).
    var reveal: (TutorialReveal) -> Void = { _ in }

    private static let seenKey = "h3c-app.tutorialsSeen"
    /// 全体設定 > チュートリアル: whether tours of new features start by
    /// themselves at launch (on by default). The basics for a new user and
    /// the API notice are not new-feature tours.
    nonisolated static let showsNewFeatureToursKey = "h3c-app.showNewFeatureTutorials"
    nonisolated static var showsNewFeatureTours: Bool {
        UserDefaults.standard.object(forKey: showsNewFeatureToursKey) as? Bool ?? true
    }
    /// 全体設定 > チュートリアル: whether the basics and the first-use tours
    /// start by themselves (on by default).
    nonisolated static let showsFirstUseToursKey = "h3c-app.showFirstUseTutorials"
    nonisolated static var showsFirstUseTours: Bool {
        UserDefaults.standard.object(forKey: showsFirstUseToursKey) as? Bool ?? true
    }
    /// Only the first window starts tours by itself.
    private static var autoStarted = false
    /// The panel's controls: a step at one of them unfolds the panel.
    private static let composerAnchors: Set<TutorialAnchor> = [
        .kind, .prompt, .generate, .imageModel, .imageSize, .editReferences,
        .imageInput, .continuation, .lipSync,
    ]

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

    func register(_ anchor: TutorialAnchor) {
        visible.insert(anchor)
        offerFirstUse(at: anchor)
    }
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
            // They know the basics and the features of the versions they
            // used already.
            seen.insert(Tutorials.basics.id)
            seen.formUnion(Tutorials.firstUse.map(\.id))
        }
        // Someone updating hears first that their scripts need the API on.
        var pending: [Tutorial] = []
        // 全体設定 > チュートリアル can keep either kind from starting by
        // itself; those stay unseen, so turning it back on shows them.
        if Self.showsFirstUseTours {
            pending.append(Tutorials.basics)
        }
        if Self.showsNewFeatureTours {
            pending += [Tutorials.images, Tutorials.upscale, Tutorials.previews]
        }
        // (Not if they have turned it on already.)
        if isNewUser || APIHost.shared.isEnabled {
            seen.insert(Tutorials.apiOff.id)
        } else {
            pending.insert(Tutorials.apiOff, at: 0)
        }
        defaults.set(Array(seen), forKey: Self.seenKey)
        queue = pending.filter { !seen.contains($0.id) }
        // Then what was on screen from the start (a project's video, say).
        let deferred = deferredOffers
        deferredOffers = []
        if !queue.isEmpty { begin(queue.removeFirst()) }
        for anchor in deferred where visible.contains(anchor) { offerFirstUse(at: anchor) }
    }

    /// A control of a feature came on screen: its first-use tour, once, now
    /// or after the tour that is running.
    private func offerFirstUse(at anchor: TutorialAnchor) {
        guard let tour = Tutorials.firstUse(for: anchor), Self.showsFirstUseTours,
              !hasSeen(tour.id), tutorial?.id != tour.id, !queue.contains(where: { $0.id == tour.id })
        else { return }
        guard Self.autoStarted else {
            if !deferredOffers.contains(anchor) { deferredOffers.append(anchor) }
            return
        }
        guard tutorial == nil else {
            queue.append(tour)
            return
        }
        // Let the control settle (a sheet sliding in) before the bubble.
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 600_000_000)
            guard !hasSeen(tour.id), visible.contains(anchor) else { return }
            if tutorial == nil {
                begin(tour)
            } else if tutorial?.id != tour.id, !queue.contains(where: { $0.id == tour.id }) {
                queue.append(tour)
            }
        }
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
        // A first-use tour waits for its control: one whose control has
        // gone again is offered the next time it comes back.
        while let following = queue.first {
            queue.removeFirst()
            if Tutorials.firstUse.contains(where: { $0.id == following.id }),
               let anchor = following.steps.first?.anchor, !visible.contains(anchor) {
                continue
            }
            // Let the last bubble close before the next tour's opens.
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 400_000_000)
                begin(following)
            }
            return
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
        if let what = step.reveal { reveal(what) }
        if let anchor = step.anchor, Self.composerAnchors.contains(anchor) {
            expandComposerRequest += 1
        }
    }

    private func hasSeen(_ id: String) -> Bool {
        (UserDefaults.standard.stringArray(forKey: Self.seenKey) ?? []).contains(id)
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
