import AppKit
import SwiftUI

/// The prompt editor: Return submits, Shift+Return inserts a newline, and
/// Tab on an empty prompt accepts the grey suggestion as real text.
/// AppKit-backed because SwiftUI's TextEditor can't tell the two apart on
/// macOS 13. Return while an input method is composing (e.g. Japanese
/// conversion) still just commits the conversion - the input method
/// consumes that key before it becomes an insertNewline: command.
///
/// Without onSubmit it is a plain multi-line box (Return inserts a newline),
/// used for カスタムクリップ's per-clip prompts.
///
/// onComposingChange reports text the input method has not committed yet.
/// NSTextView posts no textDidChange for it (setMarkedText), so the bound
/// text stays empty while the first word of a Japanese prompt is being
/// converted, and a grey placeholder drawn over an empty prompt would sit
/// on top of what is being typed.
struct PromptTextView: NSViewRepresentable {
    @Binding var text: String
    var suggestion: String = ""
    var onSubmit: (() -> Void)? = nil
    var onComposingChange: (Bool) -> Void = { _ in }

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.drawsBackground = false
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.borderType = .noBorder
        let size = scrollView.contentSize
        let textView = ComposingTextView(frame: NSRect(origin: .zero, size: size))
        textView.minSize = NSSize(width: 0, height: size.height)
        textView.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainer?.containerSize = NSSize(width: size.width, height: CGFloat.greatestFiniteMagnitude)
        textView.textContainer?.widthTracksTextView = true
        textView.delegate = context.coordinator
        textView.drawsBackground = false
        textView.isRichText = false
        textView.allowsUndo = true
        textView.font = NSFont.preferredFont(forTextStyle: .body)
        textView.textContainerInset = NSSize(width: 6, height: 8)
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.string = text
        textView.onMarkedTextChange = { [weak coordinator = context.coordinator] composing in
            coordinator?.reportComposing(composing)
        }
        scrollView.documentView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        context.coordinator.parent = self
        guard let textView = scrollView.documentView as? NSTextView else { return }
        // Don't disturb an in-progress input-method composition.
        if textView.string != text && !textView.hasMarkedText() {
            textView.string = text
        }
    }

    /// Reports when the input method's uncommitted text appears and goes.
    final class ComposingTextView: NSTextView {
        var onMarkedTextChange: ((Bool) -> Void)?

        private var composing: Bool { hasMarkedText() && markedRange().length > 0 }

        override func setMarkedText(_ string: Any, selectedRange: NSRange, replacementRange: NSRange) {
            super.setMarkedText(string, selectedRange: selectedRange, replacementRange: replacementRange)
            onMarkedTextChange?(composing)
        }

        override func unmarkText() {
            super.unmarkText()
            onMarkedTextChange?(composing)
        }

        override func insertText(_ string: Any, replacementRange: NSRange) {
            super.insertText(string, replacementRange: replacementRange)
            onMarkedTextChange?(composing)
        }
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: PromptTextView
        private var composing = false
        init(_ parent: PromptTextView) { self.parent = parent }

        func reportComposing(_ value: Bool) {
            guard value != composing else { return }
            composing = value
            parent.onComposingChange(value)
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            parent.text = textView.string
        }

        func textView(_ textView: NSTextView, doCommandBy selector: Selector) -> Bool {
            if selector == #selector(NSResponder.insertTab(_:)) {
                guard textView.string.isEmpty, !parent.suggestion.isEmpty else { return false }
                textView.string = parent.suggestion
                textView.setSelectedRange(NSRange(location: (parent.suggestion as NSString).length, length: 0))
                parent.text = parent.suggestion
                return true
            }
            guard selector == #selector(NSResponder.insertNewline(_:)), let onSubmit = parent.onSubmit else {
                return false
            }
            if NSApp.currentEvent?.modifierFlags.contains(.shift) == true {
                textView.insertNewlineIgnoringFieldEditor(nil)
            } else {
                onSubmit()
            }
            return true
        }
    }
}
