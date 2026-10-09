import ApplicationServices
import Foundation

/// Reads the posts on an X (x.com) page open in a browser window, through the Accessibility API.
/// Each post is an `article`; its permalink (`/<author>/status/<id>`) names it, and its text is the
/// static text that is not part of a link or button. Needs Accessibility permission.
public actor XTimelineReader {
    public struct Page: Sendable {
        public let windowTitle: String
        /// Posts in page order, top first. X only keeps the ones near the viewport in the tree.
        public let posts: [Bookmark]
        /// Screen frame of each post (Accessibility's top-left origin), by post id.
        public let frames: [String: CGRect]
        /// Screen frame of the browser window.
        public let windowFrame: CGRect

        /// Posts on screen, top first: entirely inside the window, or (for posts taller than it)
        /// with their top inside and most of the window filled.
        public var visiblePosts: [Bookmark] {
            posts.filter { post in
                guard let frame = frames[post.id], windowFrame.contains(frame.origin) else { return false }
                return windowFrame.contains(frame)
                    || windowFrame.intersection(frame).height >= windowFrame.height * 0.6
            }
            .sorted { frames[$0.id]!.minY < frames[$1.id]!.minY }
        }
    }

    private let application: AXUIElement
    private var webAreaReady = false

    private static let permalink = try! NSRegularExpression(
        pattern: #"^https://(?:x|twitter)\.com/([A-Za-z0-9_]+)/status/(\d+)$"#)
    /// Controls whose text is chrome, not post content.
    private static let skippedRoles: Set<String> = [
        "AXLink", "AXButton", "AXPopUpButton", "AXMenuButton", "AXToolbar", "AXScrollBar",
    ]

    /// `processIdentifier` of a running browser (Chrome, Safari, Arc, …).
    public init(processIdentifier: pid_t) {
        application = AXUIElementCreateApplication(processIdentifier)
        // Chromium browsers build their web-content accessibility tree only when asked.
        AXUIElementSetAttributeValue(application, "AXManualAccessibility" as CFString, kCFBooleanTrue)
        AXUIElementSetAttributeValue(application, "AXEnhancedUserInterface" as CFString, kCFBooleanTrue)
    }

    public static var isTrusted: Bool { AXIsProcessTrusted() }

    /// The frontmost window whose title marks an X page (`… / X`), or nil when none is open.
    public func read() async throws -> Page? {
        guard let window = xWindow() else { return nil }
        if !webAreaReady {
            // Chromium fills in web content a few seconds after the first accessibility client asks.
            let deadline = ContinuousClock.now + .seconds(8)
            while ContinuousClock.now < deadline, !hasWebArea(window, depth: 0) {
                try await Task.sleep(for: .milliseconds(300))
            }
            webAreaReady = true
        }
        var posts: [Bookmark] = []
        var frames: [String: CGRect] = [:]
        collectArticles(window, into: &posts, frames: &frames)
        return Page(
            windowTitle: attribute(window, kAXTitleAttribute) as? String ?? "", posts: posts, frames: frames,
            windowFrame: frame(of: window) ?? .zero)
    }

    /// True when the browser's focused window is the X page, so key events sent to the browser land there.
    public func xWindowIsFocused() -> Bool {
        guard let focused = attribute(application, kAXFocusedWindowAttribute),
            CFGetTypeID(focused) == AXUIElementGetTypeID()
        else { return false }
        let title = attribute(focused as! AXUIElement, kAXTitleAttribute) as? String ?? ""
        return title.contains(" / X") || title.hasSuffix(" on X")
    }

    /// Scrolls the X page so its last loaded post is in view, which makes X load the next ones. Uses the
    /// accessibility action, not key events, so it works while the browser is in the background.
    @discardableResult
    public func scrollToLastPost() -> Bool {
        guard let window = xWindow() else { return false }
        var articles: [AXUIElement] = []
        collectArticleElements(window, into: &articles)
        guard let last = articles.last else { return false }
        return AXUIElementPerformAction(last, "AXScrollToVisible" as CFString) == .success
    }

    private func collectArticleElements(_ element: AXUIElement, into articles: inout [AXUIElement]) {
        if attribute(element, kAXSubroleAttribute) as? String == "AXDocumentArticle" {
            articles.append(element)
            return
        }
        for child in children(element) { collectArticleElements(child, into: &articles) }
    }

    private func xWindow() -> AXUIElement? {
        let windows = (attribute(application, kAXWindowsAttribute) as? [AXUIElement]) ?? []
        return windows.first {
            let title = attribute($0, kAXTitleAttribute) as? String ?? ""
            return title.contains(" / X") || title.hasSuffix(" on X")
        }
    }

    private func collectArticles(
        _ element: AXUIElement, into posts: inout [Bookmark], frames: inout [String: CGRect]
    ) {
        if attribute(element, kAXSubroleAttribute) as? String == "AXDocumentArticle" {
            if let post = post(from: element), !posts.contains(where: { $0.id == post.id }) {
                posts.append(post)
                frames[post.id] = frame(of: element)
            }
            return
        }
        for child in children(element) { collectArticles(child, into: &posts, frames: &frames) }
    }

    private func frame(of element: AXUIElement) -> CGRect? {
        guard let position = attribute(element, kAXPositionAttribute), let size = attribute(element, kAXSizeAttribute),
            CFGetTypeID(position) == AXValueGetTypeID(), CFGetTypeID(size) == AXValueGetTypeID()
        else { return nil }
        var point = CGPoint.zero
        var dimensions = CGSize.zero
        AXValueGetValue(position as! AXValue, .cgPoint, &point)
        AXValueGetValue(size as! AXValue, .cgSize, &dimensions)
        return CGRect(origin: point, size: dimensions)
    }

    private func post(from article: AXUIElement) -> Bookmark? {
        var identity: (author: String, id: String)?
        var date = ""
        var texts: [String] = []
        var media: [String] = []
        func visit(_ element: AXUIElement) {
            let role = attribute(element, kAXRoleAttribute) as? String ?? ""
            let url = (attribute(element, "AXURL") as? URL)?.absoluteString
            if role == "AXLink", let url {
                if identity == nil,
                    let match = Self.permalink.firstMatch(in: url, range: NSRange(url.startIndex..., in: url)),
                    let author = Range(match.range(at: 1), in: url), let id = Range(match.range(at: 2), in: url)
                {
                    identity = (String(url[author]), String(url[id]))
                    date = attribute(element, kAXDescriptionAttribute) as? String ?? ""
                }
                if url.contains("/photo/") { media.append("photo") }
                if url.contains("/video/") { media.append("video") }
            }
            // A quoted post is a link without a URL; its text belongs to this post.
            if Self.skippedRoles.contains(role), role != "AXLink" || url != nil { return }
            if role == "AXStaticText", let value = attribute(element, kAXValueAttribute) as? String {
                let text = value.trimmingCharacters(in: .whitespacesAndNewlines)
                if text.count > 1 { texts.append(text) }
            }
            for child in children(element) { visit(child) }
        }
        visit(article)
        guard let identity, !texts.isEmpty else { return nil }
        return Bookmark(
            id: identity.id, author: identity.author, createdAt: date, text: texts.joined(separator: "\n"),
            media: media)
    }

    private func hasWebArea(_ element: AXUIElement, depth: Int) -> Bool {
        if attribute(element, kAXRoleAttribute) as? String == "AXWebArea" { return !children(element).isEmpty }
        guard depth < 12 else { return false }
        return children(element).contains { hasWebArea($0, depth: depth + 1) }
    }

    private func children(_ element: AXUIElement) -> [AXUIElement] {
        (attribute(element, kAXChildrenAttribute) as? [AXUIElement]) ?? []
    }

    private func attribute(_ element: AXUIElement, _ name: String) -> AnyObject? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success else { return nil }
        return value
    }
}
