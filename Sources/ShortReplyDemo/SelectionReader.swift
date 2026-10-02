import AppKit
import ApplicationServices
import Carbon.HIToolbox

/// Reads the text selected in the frontmost app: Accessibility first, then a ⌘C round trip through the pasteboard.
enum SelectionReader {
    static var isTrusted: Bool { AXIsProcessTrusted() }

    /// Text to reply to. Order: the selection; then, when a text field has focus (a reply box), the post shown
    /// above it in the same dialog; then a ⌘C round trip. `typedTrigger` removes the trigger key's character if it
    /// landed in that text field.
    struct Context {
        let text: String?
        /// A text field (reply box) had focus, so a draft can be inserted without any UI.
        let fieldFocused: Bool
    }

    static func context(typedTrigger: Bool = false) async -> Context {
        if let text = accessibilitySelection(), !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return Context(text: text, fieldFocused: false)
        }
        if let focused = focusedElement(), isEditable(focused) {
            if typedTrigger {
                KeystrokeSender.send(
                    keyCode: UInt16(kVK_Delete), command: false,
                    to: NSWorkspace.shared.frontmostApplication?.processIdentifier)
            }
            return Context(text: postAbove(focused), fieldFocused: true)
        }
        return Context(text: await pasteboardSelection(), fieldFocused: false)
    }

    /// Removes the character a typed trigger key left in a focused text field, if any.
    static func deleteTypedTriggerIfEditing() {
        guard let focused = focusedElement(), isEditable(focused) else { return }
        KeystrokeSender.send(
            keyCode: UInt16(kVK_Delete), command: false, to: NSWorkspace.shared.frontmostApplication?.processIdentifier)
    }

    private static func focusedElement() -> AXUIElement? {
        guard let application = NSWorkspace.shared.frontmostApplication else { return nil }
        let element = AXUIElementCreateApplication(application.processIdentifier)
        AXUIElementSetAttributeValue(element, "AXEnhancedUserInterface" as CFString, kCFBooleanTrue)
        var focused: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXFocusedUIElementAttribute as CFString, &focused) == .success,
            let focused, CFGetTypeID(focused) == AXUIElementGetTypeID()
        else { return nil }
        return unsafeBitCast(focused, to: AXUIElement.self)
    }

    private static func attribute(_ element: AXUIElement, _ name: String) -> CFTypeRef? {
        var value: CFTypeRef?
        return AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success ? value : nil
    }

    private static func isEditable(_ element: AXUIElement) -> Bool {
        let role = attribute(element, kAXRoleAttribute) as? String ?? ""
        if role == kAXTextAreaRole || role == kAXTextFieldRole || role == kAXComboBoxRole { return true }
        var settable = DarwinBoolean(false)
        AXUIElementIsAttributeSettable(element, kAXValueAttribute as CFString, &settable)
        return settable.boolValue && attribute(element, kAXValueAttribute) is String
    }

    /// Static text above the focused field within the nearest enclosing container that holds a real paragraph:
    /// on a reply dialog that is the post being answered. Short fragments (names, times, "Replying to …") are dropped.
    private static func postAbove(_ focused: AXUIElement) -> String? {
        var node = focused
        for _ in 0..<48 {  // X nests the reply box dozens of AXGroups deep
            guard let parentValue = attribute(node, kAXParentAttribute),
                CFGetTypeID(parentValue) == AXUIElementGetTypeID()
            else { return nil }
            let parent = unsafeBitCast(parentValue, to: AXUIElement.self)
            var paragraphs: [String] = []
            var budget = 2500
            var reachedFocus = false
            collectText(parent, focused: focused, into: &paragraphs, budget: &budget, reachedFocus: &reachedFocus)
            let kept = Self.postParagraphs(paragraphs)
            if !kept.isEmpty { return kept.joined(separator: " ") }
            node = parent
        }
        return nil
    }

    /// The post body from a container's static texts in order: drops the header trio (display name, `@handle`,
    /// time), the "Replying to" line and UI labels, and keeps the rest so a post split around mention links
    /// ("… and", "@name", "…") comes back whole. Ignored when what remains is only short fragments.
    static func postParagraphs(_ paragraphs: [String]) -> [String] {
        var texts = paragraphs
        if let handle = texts.indices.first(where: { texts[$0].hasPrefix("@") && !texts[$0].contains(" ") }), handle > 0
        {
            var drop = [handle - 1, handle]
            if handle + 1 < texts.count, isTimestamp(texts[handle + 1]) { drop.append(handle + 1) }
            for index in drop.sorted(by: >) { texts.remove(at: index) }
        }
        let body = texts.filter {
            !$0.hasPrefix("Replying to") && $0 != "Show more" && $0 != "Post your reply" && !isTimestamp($0)
        }
        return body.contains(where: { $0.count >= 20 }) ? body : []
    }

    private static func isTimestamp(_ text: String) -> Bool {
        text.range(of: #"^(\d+[smhd]|[A-Z][a-z]{2} \d{1,2}(, \d{4})?|·)$"#, options: .regularExpression) != nil
    }

    private static func collectText(
        _ element: AXUIElement, focused: AXUIElement, into paragraphs: inout [String], budget: inout Int,
        reachedFocus: inout Bool
    ) {
        guard budget > 0, !reachedFocus else { return }
        budget -= 1
        if CFEqual(element, focused) {
            reachedFocus = true
            return
        }
        if attribute(element, kAXRoleAttribute) as? String == kAXStaticTextRole,
            let text = attribute(element, kAXValueAttribute) as? String
        {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { paragraphs.append(trimmed) }
            return
        }
        guard let children = attribute(element, kAXChildrenAttribute) as? [AXUIElement] else { return }
        for child in children {
            collectText(child, focused: focused, into: &paragraphs, budget: &budget, reachedFocus: &reachedFocus)
            if reachedFocus { return }
        }
    }

    /// `AXSelectedText` of the focused element, which Safari, Slack and native text views all expose.
    private static func accessibilitySelection() -> String? {
        // Chrome only exposes web-content accessibility (incl. AXSelectedText) once an assistive client asks for it
        // (focusedElement sets AXEnhancedUserInterface on the app).
        guard let focused = focusedElement() else { return nil }
        return attribute(focused, kAXSelectedTextAttribute) as? String
    }

    /// Sends ⌘C to the frontmost app and reads the pasteboard, restoring its previous contents afterwards.
    private static func pasteboardSelection() async -> String? {
        let pasteboard = NSPasteboard.general
        let previous = pasteboard.string(forType: .string)
        let changeCount = pasteboard.changeCount
        KeystrokeSender.send(keyCode: UInt16(kVK_ANSI_C), command: true)
        for _ in 0..<20 {
            try? await Task.sleep(for: .milliseconds(25))
            if pasteboard.changeCount != changeCount { break }
        }
        guard pasteboard.changeCount != changeCount else { return nil }
        let text = pasteboard.string(forType: .string)
        if let previous {
            pasteboard.clearContents()
            pasteboard.setString(previous, forType: .string)
        }
        return text
    }
}

enum KeystrokeSender {
    /// Posts a key press system-wide, or straight to one process when `pid` is given (no need for it to be frontmost).
    static func send(keyCode: UInt16, command: Bool, to pid: pid_t? = nil) {
        let source = CGEventSource(stateID: .combinedSessionState)
        guard let down = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: true),
            let up = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: false)
        else { return }
        if command {
            down.flags = .maskCommand
            up.flags = .maskCommand
        }
        if let pid {
            down.postToPid(pid)
            up.postToPid(pid)
        } else {
            down.post(tap: .cghidEventTap)
            up.post(tap: .cghidEventTap)
        }
    }
}

/// Global keys: 9 (or ⌃⌥R) drafts, 0 regenerates the last draft. Needs Accessibility trust for global key events. A global
/// monitor observes only, so a typed 9 / 0 still reaches the app you're typing in; the handlers delete it again
/// when a text field had focus.
final class HotkeyMonitor {
    enum Action: Sendable {
        case draft(typedTrigger: Bool)
        case regenerate(typedTrigger: Bool)
    }

    private var monitor: Any?

    init(handler: @escaping @Sendable (Action) -> Void) {
        monitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { event in
            let modifiers = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
            if event.keyCode == UInt16(kVK_ANSI_9) && modifiers.isEmpty { return handler(.draft(typedTrigger: true)) }
            if event.keyCode == UInt16(kVK_ANSI_0) && modifiers.isEmpty {
                return handler(.regenerate(typedTrigger: true))
            }
            if event.keyCode == UInt16(kVK_ANSI_R) && modifiers == [.control, .option] {
                handler(.draft(typedTrigger: false))
            }
        }
    }

    deinit {
        if let monitor { NSEvent.removeMonitor(monitor) }
    }
}
