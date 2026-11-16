import AVKit
import SwiftUI

// SwiftUI's `VideoPlayer` crashes on this OS build: its `_AVKit_SwiftUI`
// bridging type fails to resolve generic metadata for VideoPlayerView's
// superclass at runtime (Swift runtime bug, not app code - see
// getSuperclassMetadata in the crash report). Wrapping plain AppKit
// AVPlayerView ourselves avoids that code path entirely.
struct PlayerView: NSViewRepresentable {
    let player: AVPlayer

    // Zoomed in, this view is larger than the stage: SwiftUI's clipping
    // hides the overflow but AppKit still hit-tests it, so it swallowed
    // clicks on the header's buttons and the transport bar. All its input
    // is handled by the SwiftUI layer above it (ResultPlayerView), so it
    // takes none itself.
    final class PassThroughPlayerView: AVPlayerView {
        override func hitTest(_ point: NSPoint) -> NSView? { nil }
    }

    func makeNSView(context: Context) -> AVPlayerView {
        let view = PassThroughPlayerView()
        view.player = player
        // AVKit's own transport bar only fades in on mouse hover and turned
        // out to be easy to miss; a small always-visible SwiftUI bar below
        // (see VideoPlaybackModel/ResultPlayerView) replaces it entirely.
        view.controlsStyle = .none
        return view
    }

    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        if nsView.player !== player {
            nsView.player = player
        }
    }
}

@MainActor
final class VideoPlaybackModel: ObservableObject {
    let player: AVPlayer
    @Published var isPlaying = false
    @Published var currentTime: Double = 0
    @Published var duration: Double = 1
    private var timeObserverToken: Any?
    private var endObserver: NSObjectProtocol?

    init(url: URL) {
        player = AVPlayer(url: url)
        timeObserverToken = player.addPeriodicTimeObserver(
            forInterval: CMTime(seconds: 0.1, preferredTimescale: 600), queue: .main
        ) { [weak self] time in
            MainActor.assumeIsolated { self?.currentTime = time.seconds }
        }
        Task {
            if let asset = player.currentItem?.asset,
               let loadedDuration = try? await asset.load(.duration) {
                self.duration = max(loadedDuration.seconds, 0.1)
            }
        }
        endObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime, object: player.currentItem, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.isPlaying = false
                self.player.seek(to: .zero)
            }
        }
        // "新しい結果への切替時は先頭フレームで停止。ユーザーが再生を開始する。"
        // (design spec section 7) - don't autoplay.
    }

    func togglePlayback() {
        isPlaying.toggle()
        isPlaying ? player.play() : player.pause()
    }

    func seek(to seconds: Double) {
        // A zero-tolerance seek is exact but slow enough per call that
        // repeated calls while dragging a slider mostly just queue up and
        // lag behind; a small tolerance keeps scrubbing responsive.
        let time = CMTime(seconds: seconds, preferredTimescale: 600)
        player.seek(to: time, toleranceBefore: CMTime(seconds: 0.05, preferredTimescale: 600),
                    toleranceAfter: CMTime(seconds: 0.05, preferredTimescale: 600))
    }

    func cleanup() {
        if let timeObserverToken {
            player.removeTimeObserver(timeObserverToken)
        }
        timeObserverToken = nil
        if let endObserver {
            NotificationCenter.default.removeObserver(endObserver)
        }
        endObserver = nil
    }
}

// Playback only - export/settings/reuse live in ResultActionsView (see
// PreviewPane.swift), matching the design spec's split between "右ペイン
// 中央" (the player) and "右ペイン下" (result actions).
struct ResultPlayerView: View {
    @StateObject private var model: VideoPlaybackModel
    let aspectRatio: CGFloat
    // The video's own pixel size: shown at most 1:1 (never blown up to fill
    // a larger stage), scaled down to fit, then zoomed by pinch / mouse
    // wheel. Double-click resets to the fitted size.
    let nativeSize: CGSize
    /// False while the pointer is over something floating on the video
    /// (the composer): pinch and scroll-wheel don't zoom there.
    let zoomsHere: () -> Bool
    @State private var zoom: CGFloat = 1
    @State private var pan: CGSize = .zero
    // While the user is dragging, the slider shows this instead of
    // model.currentTime - otherwise the periodic time observer (which lags
    // one seek behind while scrubbing) snaps the thumb back every ~0.1s and
    // dragging looks like it does nothing.
    @State private var isScrubbing = false
    @State private var scrubTime: Double = 0

    init(url: URL, aspectRatio: CGFloat, nativeSize: CGSize, zoomsHere: @escaping () -> Bool = { true }) {
        self.aspectRatio = aspectRatio
        self.nativeSize = nativeSize
        self.zoomsHere = zoomsHere
        _model = StateObject(wrappedValue: VideoPlaybackModel(url: url))
    }

    var body: some View {
        VStack(spacing: H3Spacing.sm) {
            // Transport bar on top: the floating composer overlays the
            // bottom of the video.
            HStack {
                Button(action: model.togglePlayback) {
                    Image(systemName: model.isPlaying ? "pause.fill" : "play.fill")
                }
                .buttonStyle(.borderless)
                .accessibilityLabel(model.isPlaying ? String(localized: "一時停止") : String(localized: "再生"))
                Slider(
                    value: Binding(
                        get: { isScrubbing ? scrubTime : model.currentTime },
                        set: { newValue in
                            scrubTime = newValue
                            model.seek(to: newValue)
                        }
                    ),
                    in: 0 ... model.duration,
                    onEditingChanged: { editing in
                        isScrubbing = editing
                        if !editing { model.seek(to: scrubTime) }
                    }
                )
                .accessibilityLabel("再生位置")
                Text(String(format: "%.1fs / %.1fs", model.currentTime, model.duration))
                    .font(.caption)
                    .monospacedDigit()
                Button("\(Int((zoom * 100).rounded()))%") {
                    withAnimation(.easeInOut(duration: 0.15)) { zoom = 1; pan = .zero }
                }
                .buttonStyle(.borderless)
                .font(.caption)
                .monospacedDigit()
                .help("動画をクリックかスペースキーで再生・停止。ピンチ・マウスホイールで拡大縮小、ドラッグで移動、ダブルクリックで元の大きさ")
            }
            ZoomPanStage(nativeSize: nativeSize, zoom: $zoom, pan: $pan, zoomsHere: zoomsHere,
                         onTap: { model.togglePlayback() }) {
                PlayerView(player: model.player)
            }
            .background(SpaceKeyMonitor { model.togglePlayback() })
            .background(Color.black)
            .clipShape(RoundedRectangle(cornerRadius: H3Radius.stage))
        }
        .onDisappear { model.cleanup() }
    }
}

/// Content shown at most 1:1 (never blown up past its own pixel size),
/// scaled down to fit the stage, then zoomed by pinch / mouse wheel and
/// moved by dragging; double-click resets to the fitted size. Shared by the
/// video player and the image preview. Gestures sit on a SwiftUI layer above
/// the content, which may be an AppKit view that would otherwise take them.
struct ZoomPanStage<Content: View>: View {
    let nativeSize: CGSize
    @Binding var zoom: CGFloat
    @Binding var pan: CGSize
    /// False while the pointer is over something floating on the stage
    /// (the composer): pinch and scroll-wheel don't zoom there.
    var zoomsHere: () -> Bool = { true }
    /// A single click (the video plays or pauses); nil: none.
    var onTap: (() -> Void)?
    @ViewBuilder let content: () -> Content
    @State private var pinchBase: CGFloat?
    /// A pinch that began over the composer, left alone until it ends.
    @State private var pinchIgnored = false
    @State private var panBase: CGSize?
    static var zoomRange: ClosedRange<CGFloat> { 0.25 ... 8 }

    private func fittedSize(in area: CGSize) -> CGSize {
        guard nativeSize.width > 0, nativeSize.height > 0 else { return area }
        let scale = min(1, area.width / nativeSize.width, area.height / nativeSize.height)
        return CGSize(width: nativeSize.width * scale, height: nativeSize.height * scale)
    }

    private func clampedPan(_ proposed: CGSize, shown: CGSize, area: CGSize) -> CGSize {
        let maxX = max(0, (shown.width - area.width) / 2)
        let maxY = max(0, (shown.height - area.height) / 2)
        return CGSize(width: min(max(proposed.width, -maxX), maxX),
                      height: min(max(proposed.height, -maxY), maxY))
    }

    private func setZoom(_ value: CGFloat) {
        zoom = min(max(value, Self.zoomRange.lowerBound), Self.zoomRange.upperBound)
    }

    private func reset() {
        withAnimation(.easeInOut(duration: 0.15)) { zoom = 1; pan = .zero }
    }

    var body: some View {
        GeometryReader { geometry in
            let area = geometry.size
            let fitted = fittedSize(in: area)
            let shown = CGSize(width: fitted.width * zoom, height: fitted.height * zoom)
            let offset = clampedPan(pan, shown: shown, area: area)
            content()
                .frame(width: shown.width, height: shown.height)
                .offset(offset)
                .frame(width: area.width, height: area.height)
                .clipped()
                .overlay(
                    Color.clear
                        .contentShape(Rectangle())
                        .gesture(
                            MagnificationGesture()
                                .onChanged { value in
                                    if pinchBase == nil && !pinchIgnored && !zoomsHere() { pinchIgnored = true }
                                    guard !pinchIgnored else { return }
                                    let base = pinchBase ?? zoom
                                    pinchBase = base
                                    setZoom(base * value)
                                }
                                .onEnded { _ in
                                    pinchBase = nil
                                    pinchIgnored = false
                                }
                        )
                        .simultaneousGesture(
                            DragGesture(minimumDistance: 2)
                                .onChanged { value in
                                    let base = panBase ?? offset
                                    panBase = base
                                    pan = clampedPan(CGSize(width: base.width + value.translation.width,
                                                            height: base.height + value.translation.height),
                                                     shown: shown, area: area)
                                }
                                .onEnded { _ in panBase = nil }
                        )
                        .gesture(
                            TapGesture(count: 2)
                                .onEnded { reset() }
                                .exclusively(before: TapGesture(count: 1).onEnded { onTap?() })
                        )
                )
                .background(ScrollWheelMonitor(isActive: zoomsHere) { deltaY in
                    setZoom(zoom * exp(deltaY * 0.01))
                })
        }
    }
}

/// Space plays or pauses the video on the stage - unless typing is going
/// on (a text field or the prompt has focus), a modifier is held, or the
/// key belongs to another window or a sheet.
struct SpaceKeyMonitor: NSViewRepresentable {
    let onSpace: () -> Void

    func makeNSView(context: Context) -> MonitorView {
        let view = MonitorView()
        view.onSpace = onSpace
        return view
    }

    func updateNSView(_ nsView: MonitorView, context: Context) {
        nsView.onSpace = onSpace
    }

    final class MonitorView: NSView {
        var onSpace: (() -> Void)?
        private var monitor: Any?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            if let monitor { NSEvent.removeMonitor(monitor) }
            monitor = nil
            guard window != nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                guard let self, let window = self.window, event.window === window,
                      window.attachedSheet == nil, window.isKeyWindow,
                      event.keyCode == 49, !event.isARepeat,
                      event.modifierFlags.intersection([.command, .option, .control, .shift]).isEmpty,
                      !(window.firstResponder is NSText) else {
                    return event
                }
                self.onSpace?()
                return nil
            }
        }

        override func hitTest(_ point: NSPoint) -> NSView? { nil }

        deinit {
            if let monitor { NSEvent.removeMonitor(monitor) }
        }
    }
}

/// Reports mouse-wheel / two-finger-scroll deltas that happen over this
/// view (as a background, it doesn't take any other events), and consumes
/// them so they zoom instead of scrolling anything else.
struct ScrollWheelMonitor: NSViewRepresentable {
    /// False lets the events through (the pointer is over something on top).
    var isActive: () -> Bool = { true }
    let onScroll: (CGFloat) -> Void

    func makeNSView(context: Context) -> MonitorView {
        let view = MonitorView()
        view.onScroll = onScroll
        view.isActive = isActive
        return view
    }

    func updateNSView(_ nsView: MonitorView, context: Context) {
        nsView.onScroll = onScroll
        nsView.isActive = isActive
    }

    final class MonitorView: NSView {
        var onScroll: ((CGFloat) -> Void)?
        var isActive: () -> Bool = { true }
        private var monitor: Any?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            if let monitor { NSEvent.removeMonitor(monitor) }
            monitor = nil
            guard window != nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { [weak self] event in
                guard let self, event.window === self.window,
                      self.bounds.contains(self.convert(event.locationInWindow, from: nil)),
                      self.isActive() else {
                    return event
                }
                let delta = event.hasPreciseScrollingDeltas ? event.scrollingDeltaY : event.scrollingDeltaY * 8
                self.onScroll?(delta)
                return nil
            }
        }

        override func hitTest(_ point: NSPoint) -> NSView? { nil }

        deinit {
            if let monitor { NSEvent.removeMonitor(monitor) }
        }
    }
}
