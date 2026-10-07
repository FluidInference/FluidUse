import AppKit
import SwiftUI

/// A highlighted range of a text (Unicode scalar offsets, as the engine reports them).
struct Highlight: Equatable {
    enum Style { case pii, unsupported }
    let start: Int
    let end: Int
    let style: Style
    let tooltip: String?
}

/// An NSTextView with live highlight attributes: the composer (orange PII marks) and editable reply bubbles
/// (red underlined unsupported claims with tooltips). Highlights are applied only when they belong to the current text.
struct RichTextView: NSViewRepresentable {
    @Binding var text: String
    /// The text the highlights were computed for; ignored unless equal to `text`.
    var highlightedText: String?
    var highlights: [Highlight]
    var fontSize: CGFloat = 15
    var scrolls = true
    var onSubmit: (() -> Void)?

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: RichTextView
        init(_ p: RichTextView) { parent = p }

        func textDidChange(_ n: Notification) {
            guard let tv = n.object as? NSTextView else { return }
            if parent.text != tv.string { parent.text = tv.string }
            RichTextView.paint(tv, parent: parent)
        }

        func textView(_ tv: NSTextView, doCommandBy sel: Selector) -> Bool {
            guard sel == #selector(NSResponder.insertNewline(_:)), let submit = parent.onSubmit else { return false }
            if NSApp.currentEvent?.modifierFlags.contains(.shift) == true {
                tv.insertNewlineIgnoringFieldEditor(nil)
            } else {
                submit()
            }
            return true
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    private func configure(_ tv: NSTextView) {
        tv.isRichText = false
        tv.allowsUndo = true
        tv.drawsBackground = false
        tv.isAutomaticQuoteSubstitutionEnabled = false
        tv.isAutomaticDashSubstitutionEnabled = false
        tv.isAutomaticTextReplacementEnabled = false
        tv.isContinuousSpellCheckingEnabled = false
        tv.font = .systemFont(ofSize: fontSize)
        tv.textColor = NSColor(Theme.ink)
        tv.insertionPointColor = NSColor(Theme.ink)
        tv.typingAttributes = [.font: NSFont.systemFont(ofSize: fontSize), .foregroundColor: NSColor(Theme.ink)]
        tv.textContainerInset = NSSize(width: 0, height: 2)
        tv.textContainer?.lineFragmentPadding = 0
        tv.string = text
    }

    func makeNSView(context: Context) -> NSView {
        if scrolls {
            let sv = NSTextView.scrollableTextView()
            sv.drawsBackground = false
            sv.hasVerticalScroller = true
            sv.autohidesScrollers = true
            let tv = sv.documentView as! NSTextView
            configure(tv)
            tv.delegate = context.coordinator
            return sv
        }
        let tv = SizingTextView()
        configure(tv)
        tv.isVerticallyResizable = false
        tv.textContainer?.widthTracksTextView = true
        tv.delegate = context.coordinator
        return tv
    }

    private func textView(_ v: NSView) -> NSTextView {
        (v as? NSScrollView)?.documentView as? NSTextView ?? v as! NSTextView
    }

    func updateNSView(_ v: NSView, context: Context) {
        context.coordinator.parent = self
        let tv = textView(v)
        if tv.string != text {
            let sel = tv.selectedRanges
            tv.string = text
            let n = (text as NSString).length
            tv.selectedRanges = sel.compactMap {
                let r = $0.rangeValue
                return r.location <= n ? NSValue(range: NSRange(location: r.location, length: min(r.length, n - r.location))) : nil
            }
            if tv.selectedRanges.isEmpty { tv.setSelectedRange(NSRange(location: n, length: 0)) }
        }
        Self.paint(tv, parent: self)
        if !scrolls { tv.invalidateIntrinsicContentSize() }
    }

    static func paint(_ tv: NSTextView, parent: RichTextView) {
        guard let storage = tv.textStorage else { return }
        let all = NSRange(location: 0, length: storage.length)
        storage.beginEditing()
        storage.setAttributes([.font: NSFont.systemFont(ofSize: parent.fontSize), .foregroundColor: NSColor(Theme.ink)], range: all)
        if parent.highlightedText == tv.string {
            for h in parent.highlights {
                guard let r = tv.string.nsRange(scalarStart: h.start, end: h.end), NSMaxRange(r) <= storage.length else { continue }
                switch h.style {
                case .pii:
                    storage.addAttribute(.backgroundColor, value: NSColor(Theme.mark), range: r)
                    storage.addAttribute(.foregroundColor, value: NSColor(Theme.piiInk), range: r)
                case .unsupported:
                    storage.addAttribute(.backgroundColor, value: NSColor(Theme.bad.opacity(0.18)), range: r)
                    storage.addAttribute(.underlineStyle, value: NSUnderlineStyle.thick.rawValue, range: r)
                    storage.addAttribute(.underlineColor, value: NSColor(Theme.bad), range: r)
                }
                if let tip = h.tooltip { storage.addAttribute(.toolTip, value: tip, range: r) }
            }
        }
        storage.endEditing()
    }

    func sizeThatFits(_ proposal: ProposedViewSize, nsView v: NSView, context: Context) -> CGSize? {
        guard !scrolls, let tv = v as? NSTextView, let lm = tv.layoutManager, let tc = tv.textContainer else { return nil }
        let width = proposal.width ?? 400
        tc.containerSize = NSSize(width: width, height: .greatestFiniteMagnitude)
        lm.ensureLayout(for: tc)
        let h = lm.usedRect(for: tc).height + tv.textContainerInset.height * 2
        return CGSize(width: width, height: ceil(max(h, fontSize * 1.3)))
    }
}

/// A non-scrolling text view for bubbles.
final class SizingTextView: NSTextView {
    override var intrinsicContentSize: NSSize { NSSize(width: NSView.noIntrinsicMetric, height: NSView.noIntrinsicMetric) }
}
