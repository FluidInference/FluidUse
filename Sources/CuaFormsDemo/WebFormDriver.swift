import Foundation
import WebKit

/// Observes and mutates a web page the way Cua's driver does for native windows:
/// a snapshot of actionable elements with stable tokens, then token-addressed
/// value mutation and clicks. Everything runs in the page's main frame.
@MainActor
final class WebFormDriver: NSObject, WKNavigationDelegate {
    let webView: WKWebView
    private(set) var isLoading = false
    var onNavigation: (@MainActor () -> Void)?

    override init() {
        let configuration = WKWebViewConfiguration()
        let script = WKUserScript(
            source: Self.library, injectionTime: .atDocumentEnd, forMainFrameOnly: true)
        configuration.userContentController.addUserScript(script)
        webView = WKWebView(frame: .zero, configuration: configuration)
        super.init()
        webView.navigationDelegate = self
        webView.customUserAgent =
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) "
            + "Version/17.4 Safari/605.1.15"
    }

    func load(_ url: URL) {
        if url.isFileURL {
            webView.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
        } else {
            webView.load(URLRequest(url: url))
        }
    }

    // MARK: Observation

    func snapshot() async throws -> PageSnapshot {
        guard let string = try await evaluate("window.__cua.snapshot()").string, let data = string.data(using: .utf8),
            let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { throw DriverError.badSnapshot }
        let elements = (object["elements"] as? [[String: Any]] ?? []).map { raw -> FormElement in
            let frame = raw["frame"] as? [Double] ?? [0, 0, 0, 0]
            return FormElement(
                token: raw["token"] as? String ?? "",
                role: raw["role"] as? String ?? "",
                label: raw["label"] as? String ?? "",
                value: raw["value"] as? String ?? "",
                placeholder: raw["placeholder"] as? String ?? "",
                checked: raw["checked"] as? Bool,
                frame: CGRect(x: frame[0], y: frame[1], width: frame[2], height: frame[3]))
        }
        return PageSnapshot(
            title: object["title"] as? String ?? "", url: object["url"] as? String ?? "", elements: elements)
    }

    // MARK: Actions

    func highlight(_ token: String, on: Bool) async throws {
        _ = try await evaluate("window.__cua.highlight(\(js(token)), \(on))")
    }

    /// Types `value` into the element character by character so the recording shows
    /// the field filling in; each step dispatches an `input` event for reactive frameworks.
    func type(_ value: String, into token: String, characterDelay: Duration) async throws {
        _ = try await evaluate("window.__cua.focus(\(js(token)))")
        let characters = Array(value)
        for count in 1...max(characters.count, 1) {
            let prefix = String(characters.prefix(count))
            _ = try await evaluate("window.__cua.setValue(\(js(token)), \(js(prefix)), false)")
            try Task.checkCancellation()
            if characterDelay > .zero { try await Task.sleep(for: characterDelay) }
        }
        let applied = try await evaluate("window.__cua.setValue(\(js(token)), \(js(value)), true)")
        guard applied.bool == true else { throw DriverError.valueNotApplied(token) }
    }

    func click(_ token: String) async throws {
        let clicked = try await evaluate("window.__cua.click(\(js(token)))")
        guard clicked.bool == true else { throw DriverError.elementMissing(token) }
    }

    /// Sets the file input's list to the document, the way a user's picker would, and fires `change`.
    func attach(_ fileURL: URL, to token: String) async throws {
        let data = try Data(contentsOf: fileURL)
        let mime = fileURL.pathExtension.lowercased() == "pdf" ? "application/pdf" : "text/plain"
        let script =
            "window.__cua.attach(\(js(token)), \(js(data.base64EncodedString())), "
            + "\(js(fileURL.lastPathComponent)), \(js(mime)))"
        guard try await evaluate(script).bool == true else { throw DriverError.elementMissing(token) }
    }

    func isChecked(_ token: String) async throws -> Bool? {
        try await evaluate("window.__cua.checked(\(js(token)))").bool
    }

    // MARK: Plumbing

    /// JavaScript results the page library returns; reduced to Sendable values at the boundary.
    enum JSValue: Sendable {
        case string(String)
        case bool(Bool)
        case null

        var string: String? {
            if case .string(let value) = self { return value }
            return nil
        }
        var bool: Bool? {
            if case .bool(let value) = self { return value }
            return nil
        }
    }

    private func evaluate(_ script: String) async throws -> JSValue {
        // The async `evaluateJavaScript` overload traps on `undefined` results, so wrap the
        // completion-handler variant instead.
        try await withCheckedThrowingContinuation { continuation in
            webView.evaluateJavaScript(script) { result, error in
                if let error {
                    continuation.resume(throwing: error)
                } else if let string = result as? String {
                    continuation.resume(returning: .string(string))
                } else if let number = result as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() {
                    continuation.resume(returning: .bool(number.boolValue))
                } else {
                    continuation.resume(returning: .null)
                }
            }
        }
    }

    private func js(_ string: String) -> String {
        let data = try? JSONSerialization.data(withJSONObject: string, options: .fragmentsAllowed)
        return data.flatMap { String(data: $0, encoding: .utf8) } ?? "\"\""
    }

    nonisolated func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
        Task { @MainActor in
            isLoading = true
            onNavigation?()
        }
    }

    nonisolated func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        Task { @MainActor in
            isLoading = false
            onNavigation?()
        }
    }

    nonisolated func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        Task { @MainActor in
            isLoading = false
            onNavigation?()
        }
    }

    enum DriverError: Error, LocalizedError {
        case badSnapshot
        case elementMissing(String)
        case valueNotApplied(String)

        var errorDescription: String? {
            switch self {
            case .badSnapshot: return "The page did not return a usable element snapshot"
            case .elementMissing(let token): return "Element \(token) is no longer on the page"
            case .valueNotApplied(let token): return "The page rejected the value for \(token)"
            }
        }
    }

    /// Page-side library (`Resources/cua-observer.js`). Roles use the accessibility names
    /// the checkpoint was trained on: `Edit`, `CheckBox`, `ComboBox`, `Button`.
    static let library: String = {
        guard let url = Bundle.module.url(forResource: "cua-observer", withExtension: "js", subdirectory: "Resources"),
            let source = try? String(contentsOf: url, encoding: .utf8)
        else {
            fatalError("cua-observer.js is missing from the bundle")
        }
        return source
    }()
}
