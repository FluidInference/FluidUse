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
    private let axApplication: AXUIElement
    private var elements: [String: AXUIElement] = [:]
    private var frames: [String: CGRect] = [:]
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
    }

    func activate() {
        application.activate()
    }

    // MARK: Observation

    func snapshot() async throws -> PageSnapshot {
        guard Self.isTrusted else { throw DriverError.notTrusted }
        guard let window = focusedWindow() else { throw DriverError.noWindow(application.localizedName ?? "app") }
        let title = attribute(window, kAXTitleAttribute) as? String ?? ""
        var statics: [(CGRect, String)] = []
        var controls: [(AXUIElement, String, [AXUIElement])] = []
        walk(window, ancestors: []) { element, role, ancestors in
            if role == "AXStaticText", let frame = self.frame(of: element) {
                let text = Self.clean(self.attribute(element, kAXValueAttribute) as? String ?? "")
                if !text.isEmpty { statics.append((frame, text)) }
            } else if let mapped = Self.roleName(for: element, role: role, ancestors: ancestors) {
                controls.append((element, mapped, ancestors))
            }
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
            result.append(
                FormElement(
                    token: token, role: role, label: label, value: value,
                    placeholder: attribute(element, kAXPlaceholderValueAttribute) as? String ?? "",
                    checked: checked, frame: frame))
        }
        return PageSnapshot(
            title: Self.normalizeWindowTitle(title), url: application.localizedName ?? "", elements: result)
    }

    // MARK: Actions

    func highlight(_ token: String, on: Bool) async throws {
        guard let frame = frames[token] else { return }
        if on {
            if let element = elements[token] { AXUIElementPerformAction(element, "AXScrollToVisible" as CFString) }
            // The element may have moved after scrolling; re-read its frame.
            if let element = elements[token], let updated = self.frame(of: element) {
                frames[token] = updated
                overlay.show(around: updated)
            } else {
                overlay.show(around: frame)
            }
        } else {
            overlay.hide()
        }
    }

    func type(_ value: String, into token: String, characterDelay: Duration) async throws {
        guard let element = elements[token] else { throw DriverError.elementMissing(token) }
        let characters = Array(value)
        for count in 1...max(characters.count, 1) {
            try setValue(String(characters.prefix(count)), on: element, token: token)
            try Task.checkCancellation()
            if characterDelay > .zero { try await Task.sleep(for: characterDelay) }
        }
        try setValue(value, on: element, token: token)
        let readback = attribute(element, kAXValueAttribute) as? String ?? ""
        guard readback == value else { throw DriverError.valueNotApplied(token) }
    }

    func click(_ token: String) async throws {
        guard let element = elements[token] else { throw DriverError.elementMissing(token) }
        let status = AXUIElementPerformAction(element, kAXPressAction as CFString)
        guard status == .success else { throw DriverError.actionFailed(token, status.rawValue) }
    }

    func isChecked(_ token: String) async throws -> Bool? {
        guard let element = elements[token] else { return nil }
        return (attribute(element, kAXValueAttribute) as? NSNumber)?.intValue == 1
    }

    func attach(_ fileURL: URL, to token: String) async throws {
        throw DriverError.unsupported("File attachment through Accessibility")
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
        text.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func normalizeWindowTitle(_ title: String) -> String {
        var result = title
        let patterns = [
            #"\s+[–—-]\s+Page \d+ of \d+$"#, #"\s+[–—-]\s+\d+ pages?$"#, #"\s+[–—-]\s+Edited$"#, #"\s+[–—-]\s+Locked$"#,
            #"\.(pdf|docx?|pages|txt)$"#,
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

    private func focusedWindow() -> AXUIElement? {
        if let focused = attribute(axApplication, kAXFocusedWindowAttribute),
            CFGetTypeID(focused) == AXUIElementGetTypeID()
        {
            return (focused as! AXUIElement)
        }
        return (attribute(axApplication, kAXWindowsAttribute) as? [AXUIElement])?.first
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
        guard dimensions.width > 0, dimensions.height > 0 else { return nil }
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
