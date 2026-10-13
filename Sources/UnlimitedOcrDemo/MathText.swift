import AppKit
import SwiftMath
import SwiftUI

/// Renders the model's LaTeX (SwiftMath, no web view): inline `\( … \)` as images flowing inside the text, display
/// `\[ … \]` blocks centered with their `\tag{n}` on the right. Anything SwiftMath cannot parse, or a formula that is
/// still streaming (no closing delimiter yet), stays raw text.
@MainActor
enum LatexRenderer {
    private static var cache: [String: NSImage] = [:]
    private static var failed: Set<String> = []

    /// Rewrites for commands the model writes but SwiftMath lacks (measured on ~2,200 formulas from real outputs).
    private static let rewrites: [(String, String)] = [
        (#"\\operatorname\s*\*?\s*\{([^}]*)\}"#, #"\\mathrm{$1}"#),
        (#"\\(boldsymbol|pmb|bm)\s*\{"#, #"\\mathbf{"#),
        (#"\\(big|Big|bigg|Bigg)[lrm]?(?![a-zA-Z])"#, ""),
        (#"\\dots(?![a-zA-Z])"#, #"\\ldots"#),
        (#"\\S(?![a-zA-Z])"#, "§"),
        (#"\\triangleq(?![a-zA-Z])"#, "≜"),
        (#"\\lesssim(?![a-zA-Z])"#, "≲"),
        (#"\\gtrsim(?![a-zA-Z])"#, "≳"),
        (#"\\rightleftharpoons(?![a-zA-Z])"#, "⇌"),
        (#"\\implies(?![a-zA-Z])"#, #"\\Rightarrow"#),
        (#"\\xrightarrow\s*\{[^}]*\}"#, #"\\longrightarrow"#),
        (#"\\(stackrel|overset)\s*\{[^}]*\}\s*\{([^}]*)\}"#, "$2"),
        (#"\\not\s*="#, #"\\neq"#),
        (#"\\begin\{array\}\s*\{[^}]*\}"#, #"\\begin{matrix}"#),
        (#"\\end\{array\}"#, #"\\end{matrix}"#),
    ]

    static func clean(_ latex: String) -> String {
        var t = latex
        for (pattern, template) in rewrites {
            t = t.replacingOccurrences(of: pattern, with: template, options: .regularExpression)
        }
        return t.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespaces)
    }

    static func image(_ latex: String, size: CGFloat, display: Bool) -> NSImage? {
        let dark = NSApp.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
        let key = "\(display ? "D" : "T")\(size)\(dark ? "d" : "l")|\(latex)"
        if let hit = cache[key] { return hit }
        if failed.contains(key) { return nil }
        let renderer = MTMathImage(
            latex: clean(latex), fontSize: size, textColor: dark ? NSColor(white: 0.92, alpha: 1) : .black,
            labelMode: display ? .display : .text, textAlignment: .left)
        let (error, image) = renderer.asImage()
        guard error == nil, let image else {
            failed.insert(key)
            return nil
        }
        cache[key] = image
        return image
    }
}

/// A paragraph whose `\( … \)` spans render as inline math.
struct MathText: View {
    let content: String
    var size: CGFloat = 14

    var body: some View { text }

    private var text: Text {
        var result = Text("")
        var rest = Substring(content)
        while let open = rest.range(of: "\\(") {
            result = result + Text(String(rest[..<open.lowerBound]))
            let afterOpen = rest[open.upperBound...]
            guard let close = afterOpen.range(of: "\\)") else {
                return result + Text(String(rest[open.lowerBound...]))  // still streaming
            }
            let latex = String(afterOpen[..<close.lowerBound])
            if let image = LatexRenderer.image(latex, size: size, display: false) {
                // Center the formula on the text's x-height.
                result = result + Text(Image(nsImage: image)).baselineOffset(-(image.size.height - size * 0.7) / 2)
            } else {
                result = result + Text("\\(" + latex + "\\)")
            }
            rest = afterOpen[close.upperBound...]
        }
        return result + Text(String(rest))
    }
}

/// A display equation block (`\[ … \]`, optional `\tag{n}`).
struct EquationView: View {
    let content: String
    var size: CGFloat = 16

    var body: some View {
        let parsed = Self.parse(content)
        if let latex = parsed.latex, let image = LatexRenderer.image(latex, size: size, display: true) {
            HStack {
                Spacer(minLength: 0)
                Image(nsImage: image)
                Spacer(minLength: 0)
                if let tag = parsed.tag { Text("(\(tag))").font(.system(size: 13)).foregroundStyle(.secondary) }
            }
            .padding(.vertical, 4)
        } else {
            Text(content).font(.system(size: 12, design: .monospaced))
        }
    }

    /// Body between `\[` and `\]` (nil while still streaming) and its `\tag{…}`.
    static func parse(_ content: String) -> (latex: String?, tag: String?) {
        guard let open = content.range(of: "\\["), let close = content.range(of: "\\]", options: .backwards),
            open.upperBound <= close.lowerBound
        else { return (nil, nil) }
        var body = String(content[open.upperBound..<close.lowerBound])
        var tag: String?
        if let match = body.range(of: #"\\tag\s*\{([^}]*)\}"#, options: .regularExpression) {
            tag = body[match].replacingOccurrences(of: #"\\tag\s*\{|\}"#, with: "", options: .regularExpression)
                .trimmingCharacters(in: .whitespaces)
            body.removeSubrange(match)
        }
        return (body, tag)
    }
}
