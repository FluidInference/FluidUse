import AppKit
import ApplicationServices
import Foundation

/// Observes and drives another application's frontmost window through the
/// Accessibility API, the way Cua's native driver and Codex computer use do:
/// elements are addressed by snapshot tokens, values are set through `AXValue`,
/// and buttons and checkboxes are pressed with `AXPress`.
@MainActor
final class AccessibilityFormDriver: FormDriver {
    let application: NSRunningApplication
    /// Substring of the window title to target; nil uses the app's focused window.
    var windowTitleFilter: String?
    private let axApplication: AXUIElement
    private var targetWindow: AXUIElement?

    /// WebKit and Chromium apply an `AXValue` write to the focused field rather than the
    /// addressed one, so browsers are driven with real key events from the start.
    private static let browserBundlePrefixes = [
        "com.apple.Safari", "com.google.Chrome", "org.chromium", "com.microsoft.edgemac", "com.brave.Browser",
        "company.thebrowser", "org.mozilla.firefox", "com.vivaldi", "com.operasoftware",
    ]
    private var prefersKeystrokes: Bool {
        let bundle = application.bundleIdentifier ?? ""
        return Self.browserBundlePrefixes.contains { bundle.hasPrefix($0) }
    }
    private var elements: [String: AXUIElement] = [:]
    private var frames: [String: CGRect] = [:]
    private var observed: [String: FormElement] = [:]
    private let overlay = HighlightOverlay()

    static var isTrusted: Bool { AXIsProcessTrusted() }

    /// Running apps with a regular UI, excluding this process.
    static func candidates() -> [NSRunningApplication] {
        NSWorkspace.shared.runningApplications.filter {
            $0.activationPolicy == .regular && $0.processIdentifier != ProcessInfo.processInfo.processIdentifier
                && $0.localizedName != nil
        }
        .sorted { ($0.localizedName ?? "") < ($1.localizedName ?? "") }
    }

    init(application: NSRunningApplication) {
        self.application = application
        axApplication = AXUIElementCreateApplication(application.processIdentifier)
        // Chromium browsers build their web-content accessibility tree only when asked.
        AXUIElementSetAttributeValue(axApplication, "AXManualAccessibility" as CFString, kCFBooleanTrue)
        AXUIElementSetAttributeValue(axApplication, "AXEnhancedUserInterface" as CFString, kCFBooleanTrue)
    }

    /// Brings the app and the targeted window to the front so key events reach it.
    func activate() {
        application.activate()
        guard let targetWindow else { return }
        AXUIElementSetAttributeValue(targetWindow, kAXMainAttribute as CFString, kCFBooleanTrue)
        AXUIElementSetAttributeValue(targetWindow, kAXFocusedAttribute as CFString, kCFBooleanTrue)
        AXUIElementPerformAction(targetWindow, kAXRaiseAction as CFString)
    }

    /// The element holding keyboard focus, if it is `element` or an inner control drawn
    /// in the same place (composite web widgets focus a child of the addressed node),
    /// and the focused window is the targeted one. Key events go wherever focus is, so
    /// typing is refused unless this resolves.
    private func focusedCounterpart(of element: AXUIElement) -> AXUIElement? {
        guard let focused = attribute(axApplication, kAXFocusedUIElementAttribute),
            CFGetTypeID(focused) == AXUIElementGetTypeID()
        else { return nil }
        if let targetWindow, let window = attribute(axApplication, kAXFocusedWindowAttribute),
            CFGetTypeID(window) == AXUIElementGetTypeID(), !CFEqual(window, targetWindow)
        {
            return nil
        }
        let focusedElement = focused as! AXUIElement
        if CFEqual(focusedElement, element) { return focusedElement }
        guard let target = frame(of: element), let inner = frame(of: focusedElement), target.width > 0,
            target.height > 0
        else { return nil }
        let overlap = target.intersection(inner)
        guard !overlap.isNull, overlap.width * overlap.height >= 0.5 * inner.width * inner.height,
            target.contains(CGPoint(x: inner.midX, y: inner.midY))
        else { return nil }
        return focusedElement
    }

    private func isFocused(_ element: AXUIElement) -> Bool {
        focusedCounterpart(of: element) != nil
    }

    /// Titles of the app's windows, for choosing a target.
    func windowTitles() -> [String] {
        ((attribute(axApplication, kAXWindowsAttribute) as? [AXUIElement]) ?? []).compactMap {
            attribute($0, kAXTitleAttribute) as? String
        }
    }

    // MARK: Observation

    func snapshot() async throws -> PageSnapshot {
        guard Self.isTrusted else { throw DriverError.notTrusted }
        guard let window = resolveWindow() else { throw DriverError.noWindow(application.localizedName ?? "app") }
        targetWindow = window
        let title = attribute(window, kAXTitleAttribute) as? String ?? ""
        var statics: [(CGRect, String)] = []
        var controls: [(AXUIElement, String, [AXUIElement])] = []
        if prefersKeystrokes {
            // Chromium enables web-content accessibility a couple of seconds after the first
            // client query; until then the window holds only its own chrome.
            let deadline = ContinuousClock.now + .seconds(6)
            while ContinuousClock.now < deadline, !hasPopulatedWebArea(window) {
                try await Task.sleep(for: .milliseconds(300))
            }
            // The page subtree keeps filling in for a while after the web area appears.
            try await Task.sleep(for: .milliseconds(1500))
        }
        // Browsers hand back a truncated tree while a window switch or layout is in
        // flight, so walk until two consecutive passes agree on the control count.
        var previousCount = -1
        var stablePasses = 0
        for attempt in 0..<8 {
            statics = []
            controls = []
            walk(window, ancestors: []) { element, role, ancestors in
                if role == "AXStaticText", let frame = self.frame(of: element) {
                    let text = Self.clean(self.attribute(element, kAXValueAttribute) as? String ?? "")
                    if !text.isEmpty { statics.append((frame, text)) }
                } else if let mapped = Self.roleName(for: element, role: role, ancestors: ancestors) {
                    controls.append((element, mapped, ancestors))
                }
            }
            stablePasses = controls.count == previousCount ? stablePasses + 1 : 0
            if stablePasses >= 2 || (!prefersKeystrokes && attempt >= 1) { break }
            previousCount = controls.count
            try await Task.sleep(for: .milliseconds(500))
        }
        elements = [:]
        frames = [:]
        var result: [FormElement] = []
        for (index, (element, role, _)) in controls.enumerated() {
            guard let frame = frame(of: element) else { continue }
            let token = "ax-\(index + 1)"
            elements[token] = element
            frames[token] = frame
            // Buttons carry their own titles; page text near an untitled button is not its name.
            let label = Self.stripEnumerator(
                explicitLabel(element)
                    ?? (role == "Button" ? "" : Self.nearbyLabel(for: frame, role: role, statics: statics)))
            let value: String
            var checked: Bool?
            if role == "CheckBox" {
                checked = (attribute(element, kAXValueAttribute) as? NSNumber)?.intValue == 1
                value = ""
            } else {
                value = attribute(element, kAXValueAttribute) as? String ?? ""
            }
            let formElement = FormElement(
                token: token, role: role, label: label, value: value,
                placeholder: attribute(element, kAXPlaceholderValueAttribute) as? String ?? "",
                checked: checked, frame: frame)
            observed[token] = formElement
            result.append(formElement)
        }
        return PageSnapshot(
            title: Self.normalizeWindowTitle(title), url: application.localizedName ?? "", elements: result)
    }

    // MARK: Actions

    /// Scrolls the element into view. The outline overlay is off unless
    /// `CUA_DEMO_HIGHLIGHT` is set; on camera the typing itself is the cue.
    func highlight(_ token: String, on: Bool) async throws {
        guard let frame = frames[token] else { return }
        if on {
            if let element = elements[token] { AXUIElementPerformAction(element, "AXScrollToVisible" as CFString) }
            guard Self.showsOverlay else { return }
            // The element may have moved after scrolling; re-read its frame.
            if let element = elements[token], let updated = self.frame(of: element) {
                frames[token] = updated
                overlay.show(around: updated)
            } else {
                overlay.show(around: frame)
            }
        } else if Self.showsOverlay {
            overlay.hide()
        }
    }

    private static let showsOverlay = ProcessInfo.processInfo.environment["CUA_DEMO_HIGHLIGHT"] != nil

    /// Native apps take values through `AXValue`; browsers get key events. If a native
    /// app does not apply the first write, the driver falls back to key events too.
    func type(_ value: String, into token: String, characterDelay: Duration) async throws {
        guard let element = elements[token] else { throw DriverError.elementMissing(token) }
        let characters = Array(value)
        let probe = String(characters.prefix(1))
        if prefersKeystrokes {
            let typed = try await typeKeystrokes(value, into: element, token: token, characterDelay: characterDelay)
            guard Self.matches(currentValue(of: typed), value) || Self.matches(currentValue(of: element), value) else {
                throw DriverError.valueNotApplied(token)
            }
            return
        }
        try setValue(probe, on: element, token: token)
        if currentValue(of: element) == probe {
            for count in 2...max(characters.count, 2) where count <= characters.count {
                try setValue(String(characters.prefix(count)), on: element, token: token)
                try Task.checkCancellation()
                if characterDelay > .zero { try await Task.sleep(for: characterDelay) }
            }
            try setValue(value, on: element, token: token)
        } else {
            let typed = try await typeKeystrokes(value, into: element, token: token, characterDelay: characterDelay)
            guard Self.matches(currentValue(of: typed), value) || Self.matches(currentValue(of: element), value) else {
                throw DriverError.valueNotApplied(token)
            }
            return
        }
        guard currentValue(of: element) == value else { throw DriverError.valueNotApplied(token) }
    }

    /// Returns the element that actually received the keystrokes.
    @discardableResult
    private func typeKeystrokes(
        _ value: String, into element: AXUIElement, token: String, characterDelay: Duration
    ) async throws -> AXUIElement {
        activate()
        var element = element
        AXUIElementPerformAction(element, "AXScrollToVisible" as CFString)
        AXUIElementSetAttributeValue(element, kAXFocusedAttribute as CFString, kCFBooleanTrue)
        try await Task.sleep(for: .milliseconds(150))
        var counterpart = focusedCounterpart(of: element)
        if counterpart == nil, let relocated = relocate(token) {
            // Web frameworks re-render after a blur, replacing the accessibility node we
            // captured; find the control again by role and label, as upstream does.
            element = relocated
            AXUIElementPerformAction(element, "AXScrollToVisible" as CFString)
            AXUIElementSetAttributeValue(element, kAXFocusedAttribute as CFString, kCFBooleanTrue)
            try await Task.sleep(for: .milliseconds(150))
            counterpart = focusedCounterpart(of: element)
        }
        guard let focused = counterpart else { throw DriverError.focusLost(token) }
        let pid = application.processIdentifier
        let source = CGEventSource(stateID: .combinedSessionState)
        if !currentValue(of: focused).isEmpty || !currentValue(of: element).isEmpty {
            // Select all, then the typed text replaces it.
            let down = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: true)
            let up = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: false)
            down?.flags = .maskCommand
            up?.flags = .maskCommand
            down?.postToPid(pid)
            up?.postToPid(pid)
            try await Task.sleep(for: .milliseconds(50))
        }
        for scalar in value.unicodeScalars {
            try Task.checkCancellation()
            guard isFocused(element) else { throw DriverError.focusLost(token) }
            var unit = [UniChar](String(scalar).utf16)
            guard let down = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: true),
                let up = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: false)
            else { throw DriverError.actionFailed(token, -1) }
            down.keyboardSetUnicodeString(stringLength: unit.count, unicodeString: &unit)
            up.keyboardSetUnicodeString(stringLength: unit.count, unicodeString: &unit)
            down.postToPid(pid)
            up.postToPid(pid)
            try await Task.sleep(for: max(characterDelay, .milliseconds(8)))
        }
        try await Task.sleep(for: .milliseconds(150))
        return focused
    }

    /// Web widgets reformat as you type (phone masks, trimmed spaces), so the read-back
    /// is compared on letters and digits only.
    private static func matches(_ readback: String, _ value: String) -> Bool {
        func core(_ text: String) -> String { text.lowercased().filter { $0.isLetter || $0.isNumber } }
        let expected = core(value)
        return !expected.isEmpty && core(readback).contains(expected)
    }

    /// Re-walks the target window for a live element with the observed role and label,
    /// preferring the one nearest the original frame, and rebinds the token to it.
    private func relocate(_ token: String) -> AXUIElement? {
        guard let wanted = observed[token], let window = targetWindow ?? resolveWindow() else { return nil }
        var candidates: [(AXUIElement, CGRect)] = []
        walk(window, ancestors: []) { element, role, ancestors in
            guard Self.roleName(for: element, role: role, ancestors: ancestors) == wanted.role,
                let frame = self.frame(of: element),
                Self.stripEnumerator(self.explicitLabel(element) ?? "") == wanted.label
            else { return }
            candidates.append((element, frame))
        }
        let origin = wanted.frame
        guard
            let best = candidates.min(by: {
                hypot($0.1.midX - origin.midX, $0.1.midY - origin.midY)
                    < hypot($1.1.midX - origin.midX, $1.1.midY - origin.midY)
            })
        else { return nil }
        elements[token] = best.0
        frames[token] = best.1
        return best.0
    }

    private func currentValue(of element: AXUIElement) -> String {
        attribute(element, kAXValueAttribute) as? String ?? ""
    }

    func click(_ token: String) async throws {
        guard let element = elements[token] else { throw DriverError.elementMissing(token) }
        AXUIElementPerformAction(element, "AXScrollToVisible" as CFString)
        let status = AXUIElementPerformAction(element, kAXPressAction as CFString)
        guard status == .success else { throw DriverError.actionFailed(token, status.rawValue) }
    }

    func isChecked(_ token: String) async throws -> Bool? {
        guard let element = elements[token] else { return nil }
        return (attribute(element, kAXValueAttribute) as? NSNumber)?.intValue == 1
    }

    /// Browsers host their file panel in a separate sandbox helper; driving it with
    /// synthetic keys proved unreliable, so attaching stays a manual step on app targets.
    func attach(_ fileURL: URL, to token: String) async throws {
        throw DriverError.unsupported("File attachment through Accessibility")
    }

    /// Focuses the combo box, types to filter, and confirms the highlighted option.
    /// Works for react-select style controls and location typeaheads.
    func select(_ value: String, in token: String) async throws {
        guard let element = elements[token] else { throw DriverError.elementMissing(token) }
        let typed = try await typeKeystrokes(value, into: element, token: token, characterDelay: .milliseconds(12))
        try await Task.sleep(for: .milliseconds(700))
        try postKey(125, flags: [])  // Down: highlight the first match
        try await Task.sleep(for: .milliseconds(150))
        try postKey(36, flags: [])  // Return: choose it
        try await Task.sleep(for: .milliseconds(400))
        let chosen = [
            currentValue(of: typed), currentValue(of: element), description(of: element), description(of: typed),
        ]
        guard chosen.contains(where: { Self.matches($0, String(value.prefix(4))) }) else {
            throw DriverError.valueNotApplied(token)
        }
    }

    func selectAffirmative(in token: String) async throws {
        guard let element = elements[token] else { throw DriverError.elementMissing(token) }
        activate()
        AXUIElementPerformAction(element, "AXScrollToVisible" as CFString)
        AXUIElementSetAttributeValue(element, kAXFocusedAttribute as CFString, kCFBooleanTrue)
        try await Task.sleep(for: .milliseconds(150))
        guard isFocused(element) || relocate(token) != nil else { throw DriverError.focusLost(token) }
        try postKey(125, flags: [])  // Down: open the list on its first option
        try await Task.sleep(for: .milliseconds(400))
        try postKey(36, flags: [])  // Return: choose it
        try await Task.sleep(for: .milliseconds(400))
        let chosen = currentValue(of: element) + " " + description(of: element)
        guard !chosen.trimmingCharacters(in: .whitespaces).isEmpty else { throw DriverError.valueNotApplied(token) }
    }

    private func description(of element: AXUIElement) -> String {
        attribute(element, kAXDescriptionAttribute) as? String ?? ""
    }

    private func postKey(_ keyCode: CGKeyCode, flags: CGEventFlags) throws {
        let source = CGEventSource(stateID: .combinedSessionState)
        guard let down = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: true),
            let up = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: false)
        else { throw DriverError.unsupported("Key event") }
        down.flags = flags
        up.flags = flags
        down.postToPid(application.processIdentifier)
        up.postToPid(application.processIdentifier)
    }

    // MARK: Labels

    /// Apps that label controls (Safari, AppKit forms) expose it directly.
    private func explicitLabel(_ element: AXUIElement) -> String? {
        for name in [kAXDescriptionAttribute, kAXTitleAttribute] {
            if let text = attribute(element, name) as? String, !Self.clean(text).isEmpty { return Self.clean(text) }
        }
        if let title = attribute(element, kAXTitleUIElementAttribute), CFGetTypeID(title) == AXUIElementGetTypeID() {
            let text = attribute(title as! AXUIElement, kAXValueAttribute) as? String ?? ""
            if !Self.clean(text).isEmpty { return Self.clean(text) }
        }
        return nil
    }

    /// PDF viewers expose fields without names; take the nearest meaningful page text.
    /// Checkboxes read to the right, text fields read the caption directly above the
    /// box or the row text to the left, in that order.
    static func nearbyLabel(for frame: CGRect, role: String, statics: [(CGRect, String)]) -> String {
        func sameRow(_ rect: CGRect) -> Bool {
            let tolerance = max(frame.height, rect.height) * 0.75
            return abs(rect.midY - frame.midY) <= tolerance
        }
        let meaningful = statics.filter { isMeaningful($0.1) }
        if role == "CheckBox" {
            let right = meaningful.filter { sameRow($0.0) && $0.0.minX >= frame.minX && $0.0.minX - frame.maxX < 400 }
            if let best = right.min(by: { $0.0.minX - frame.maxX < $1.0.minX - frame.maxX }) { return best.1 }
        }
        let above = meaningful.filter {
            $0.0.maxY <= frame.minY + 2 && frame.minY - $0.0.maxY < frame.height * 1.5
                && $0.0.minX < frame.maxX && $0.0.maxX > frame.minX
        }
        if let best = above.min(by: { frame.minY - $0.0.maxY < frame.minY - $1.0.maxY }) { return best.1 }
        let left = meaningful.filter { sameRow($0.0) && $0.0.maxX <= frame.minX + 2 && frame.minX - $0.0.maxX < 700 }
        if let best = left.min(by: { frame.minX - $0.0.maxX < frame.minX - $1.0.maxX }) { return best.1 }
        let farAbove = meaningful.filter {
            $0.0.maxY <= frame.minY + 2 && frame.minY - $0.0.maxY < 80 && $0.0.minX < frame.maxX
                && $0.0.maxX > frame.minX
        }
        if let best = farAbove.min(by: { frame.minY - $0.0.maxY < frame.minY - $1.0.maxY }) { return best.1 }
        return ""
    }

    /// Form numbering such as "(a) ", "3a ", or "1. " is layout, not the field's name.
    static func stripEnumerator(_ label: String) -> String {
        label.replacingOccurrences(
            of: #"^(\(?[a-z0-9]{1,2}\)|[0-9]{1,2}[a-z]?[.:)])\s+"#, with: "", options: .regularExpression)
    }

    private static func isMeaningful(_ text: String) -> Bool {
        let letters = text.filter(\.isLetter).count
        return letters >= 2 && text.count <= 200
    }

    static func clean(_ text: String) -> String {
        var result = text.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        for pattern in [#"\s*\*+\s*$"#, #"\s*\(required\)\s*$"#, #"\s+required$"#, #"\s*:\s*$"#] {
            result = result.replacingOccurrences(
                of: pattern, with: "", options: [.regularExpression, .caseInsensitive])
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func normalizeWindowTitle(_ title: String) -> String {
        var result = title
        let patterns = [
            #"\s+[–—-]\s+Page \d+ of \d+$"#, #"\s+[–—-]\s+\d+ pages?$"#, #"\s+[–—-]\s+Edited$"#, #"\s+[–—-]\s+Locked$"#,
            #"\.(pdf|docx?|pages|txt)$"#, #"\s+-\s+Google Chrome\s+-\s+[^-]+$"#,
        ]
        for pattern in patterns {
            result = result.replacingOccurrences(of: pattern, with: "", options: .regularExpression)
        }
        return FormSchema.normalizeTitle(result)
    }

    // MARK: Roles

    private static func roleName(for element: AXUIElement, role: String, ancestors: [AXUIElement]) -> String? {
        var subroleValue: CFTypeRef?
        if AXUIElementCopyAttributeValue(element, kAXSubroleAttribute as CFString, &subroleValue) == .success,
            let subrole = subroleValue as? String,
            ["AXCloseButton", "AXMinimizeButton", "AXZoomButton", "AXFullScreenButton", "AXSearchField"].contains(
                subrole)
        {
            return nil
        }
        switch role {
        case "AXTextField", "AXTextArea": return "Edit"
        case "AXCheckBox": return "CheckBox"
        case "AXPopUpButton", "AXComboBox": return "ComboBox"
        case "AXButton": return "Button"
        default: return nil
        }
    }

    // MARK: AX plumbing

    private func resolveWindow() -> AXUIElement? {
        let windows = (attribute(axApplication, kAXWindowsAttribute) as? [AXUIElement]) ?? []
        if let filter = windowTitleFilter, !filter.isEmpty {
            return windows.first {
                (attribute($0, kAXTitleAttribute) as? String ?? "").localizedCaseInsensitiveContains(filter)
            }
        }
        if let focused = attribute(axApplication, kAXFocusedWindowAttribute),
            CFGetTypeID(focused) == AXUIElementGetTypeID()
        {
            return (focused as! AXUIElement)
        }
        return windows.first
    }

    private func walk(
        _ element: AXUIElement, ancestors: [AXUIElement],
        visit: (AXUIElement, String, [AXUIElement]) -> Void
    ) {
        let role = attribute(element, kAXRoleAttribute) as? String ?? ""
        // Window chrome is not part of the form.
        if ["AXToolbar", "AXMenuBar", "AXScrollBar", "AXSheet"].contains(role) { return }
        if role == "AXGroup", let subrole = attribute(element, kAXSubroleAttribute) as? String,
            subrole == "AXSidebar" || subrole == "AXTitlebar"
        {
            return
        }
        visit(element, role, ancestors)
        guard let children = attribute(element, kAXChildrenAttribute) as? [AXUIElement] else { return }
        for child in children { walk(child, ancestors: ancestors + [element], visit: visit) }
    }

    private func hasPopulatedWebArea(_ root: AXUIElement) -> Bool {
        var found = false
        walk(root, ancestors: []) { element, role, _ in
            if !found, role == "AXWebArea",
                let children = self.attribute(element, kAXChildrenAttribute) as? [AXUIElement], !children.isEmpty
            {
                found = true
            }
        }
        return found
    }

    private func attribute(_ element: AXUIElement, _ name: String) -> AnyObject? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success else { return nil }
        return value
    }

    private func frame(of element: AXUIElement) -> CGRect? {
        guard let position = attribute(element, kAXPositionAttribute), let size = attribute(element, kAXSizeAttribute),
            CFGetTypeID(position) == AXValueGetTypeID(), CFGetTypeID(size) == AXValueGetTypeID()
        else { return nil }
        var point = CGPoint.zero
        var dimensions = CGSize.zero
        AXValueGetValue(position as! AXValue, .cgPoint, &point)
        AXValueGetValue(size as! AXValue, .cgSize, &dimensions)
        // Browsers clip frames to the viewport, so a scrolled-out field reports a zero
        // height; it is still real and can be scrolled into view before acting.
        guard dimensions.width >= 0, dimensions.height >= 0 else { return nil }
        return CGRect(origin: point, size: dimensions)
    }

    private func setValue(_ value: String, on element: AXUIElement, token: String) throws {
        let status = AXUIElementSetAttributeValue(element, kAXValueAttribute as CFString, value as CFTypeRef)
        guard status == .success else { throw DriverError.actionFailed(token, status.rawValue) }
    }

    enum DriverError: Error, LocalizedError {
        case notTrusted
        case noWindow(String)
        case elementMissing(String)
        case valueNotApplied(String)
        case actionFailed(String, Int32)
        case focusLost(String)
        case unsupported(String)

        var errorDescription: String? {
            switch self {
            case .notTrusted:
                return
                    "Grant Accessibility access to the terminal running this demo (System Settings › Privacy & Security)"
            case .noWindow(let app): return "\(app) has no window to observe"
            case .elementMissing(let token): return "Element \(token) is no longer in the window"
            case .valueNotApplied(let token): return "The app did not accept the value for \(token)"
            case .actionFailed(let token, let code): return "Accessibility action on \(token) failed (AXError \(code))"
            case .focusLost(let token):
                return "Stopped: keyboard focus is not on \(token) in the target window, so nothing was typed"
            case .unsupported(let what): return "\(what) is not supported"
            }
        }
    }
}

/// A click-through floating window that outlines the element being acted on,
/// in screen coordinates, over whichever app owns it.
@MainActor
final class HighlightOverlay {
    private let panel: NSPanel

    init() {
        panel = NSPanel(
            contentRect: .zero, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        panel.level = .floating
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.ignoresMouseEvents = true
        panel.hasShadow = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        let view = NSView()
        view.wantsLayer = true
        view.layer?.borderColor = NSColor(calibratedRed: 1, green: 0.42, blue: 0, alpha: 1).cgColor
        view.layer?.borderWidth = 3
        view.layer?.cornerRadius = 4
        panel.contentView = view
    }

    /// `frame` uses Accessibility's top-left screen origin.
    func show(around frame: CGRect) {
        guard let screen = NSScreen.screens.first(where: { $0.frame.origin == .zero }) ?? NSScreen.main else { return }
        let padded = frame.insetBy(dx: -4, dy: -4)
        let flippedY = screen.frame.height - padded.maxY
        panel.setFrame(CGRect(x: padded.minX, y: flippedY, width: padded.width, height: padded.height), display: true)
        panel.orderFrontRegardless()
    }

    func hide() {
        panel.orderOut(nil)
    }
}
