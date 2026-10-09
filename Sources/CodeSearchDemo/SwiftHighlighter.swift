import SwiftUI

/// Xcode-dark-style colouring for a Swift snippet whose lines start with a 7-character line-number gutter.
enum SwiftHighlighter {
    static let keyword = Color(red: 0.99, green: 0.37, blue: 0.64)
    static let type = Color(red: 0.36, green: 0.85, blue: 1.0)
    static let call = Color(red: 0.40, green: 0.72, blue: 0.64)
    static let declaration = Color(red: 0.25, green: 0.63, blue: 0.75)
    static let string = Color(red: 0.99, green: 0.42, blue: 0.36)
    static let number = Color(red: 0.82, green: 0.75, blue: 0.41)
    static let comment = Color(red: 0.47, green: 0.53, blue: 0.60)
    static let attribute = Color(red: 0.99, green: 0.56, blue: 0.25)
    static let plain = Color(white: 0.92)
    static let gutter = Color(white: 0.42)

    private static let keywords: Set<String> = [
        "func", "let", "var", "if", "else", "guard", "return", "for", "in", "while", "repeat", "switch", "case",
        "default", "break", "continue", "struct", "class", "enum", "actor", "protocol", "extension", "import", "init",
        "deinit", "self", "Self", "super", "static", "private", "fileprivate", "internal", "public", "open", "final",
        "override", "mutating", "nonisolated", "async", "await", "try", "throws", "rethrows", "throw", "do", "catch",
        "where", "nil", "true", "false", "inout", "some", "any", "typealias", "associatedtype", "lazy", "weak",
        "unowned", "defer", "is", "as", "subscript", "convenience", "required", "get", "set", "willSet", "didSet",
    ]
    private static let token = try! NSRegularExpression(
        pattern: #"(//.*$)|("(?:\\.|[^"\\])*")|(@[A-Za-z_]+)|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)"#)

    static func highlight(_ snippet: String) -> AttributedString {
        var result = AttributedString()
        for (index, line) in snippet.components(separatedBy: "\n").enumerated() {
            if index > 0 { result.append(AttributedString("\n")) }
            let gutterLength = min(7, line.count)
            var gutter = AttributedString(String(line.prefix(gutterLength)))
            gutter.foregroundColor = gutterColor
            result.append(gutter)
            result.append(code(String(line.dropFirst(gutterLength))))
        }
        return result
    }

    private static let gutterColor = gutter

    /// Colours one line of Swift.
    static func code(_ line: String) -> AttributedString {
        var result = AttributedString()
        let ns = line as NSString
        var cursor = 0
        var previousWord = ""
        for match in token.matches(in: line, range: NSRange(location: 0, length: ns.length)) {
            if match.range.location > cursor {
                result.append(
                    piece(ns.substring(with: NSRange(location: cursor, length: match.range.location - cursor)), plain))
            }
            let text = ns.substring(with: match.range)
            let color: Color
            if match.range(at: 1).location != NSNotFound {
                color = comment
            } else if match.range(at: 2).location != NSNotFound {
                color = string
            } else if match.range(at: 3).location != NSNotFound {
                color = attribute
            } else if match.range(at: 4).location != NSNotFound {
                color = number
            } else if keywords.contains(text) {
                color = keyword
            } else if ["func", "struct", "class", "enum", "actor", "protocol", "extension"].contains(previousWord) {
                color = declaration
            } else if text.first?.isUppercase == true {
                color = type
            } else if match.range.location + match.range.length < ns.length,
                ns.character(at: match.range.location + match.range.length) == 0x28  // "("
            {
                color = call
            } else {
                color = plain
            }
            result.append(piece(text, color))
            if match.range(at: 5).location != NSNotFound { previousWord = text }
            cursor = match.range.location + match.range.length
        }
        if cursor < ns.length { result.append(piece(ns.substring(from: cursor), plain)) }
        return result
    }

    private static func piece(_ text: String, _ color: Color) -> AttributedString {
        var part = AttributedString(text)
        part.foregroundColor = color
        return part
    }

    /// Fixed colours for the usual parts of an audio SDK, hashed ones for anything else.
    static func areaColor(_ area: String) -> Color {
        switch area {
        case "ASR": return .blue
        case "TTS": return .orange
        case "Diarizer": return .green
        case "VAD": return .purple
        case "Shared": return .teal
        case "ITN": return .pink
        case "CLI": return .yellow
        case "Tests": return Color(white: 0.6)
        default:
            // Stable across launches (String.hashValue is seeded per process).
            let palette: [Color] = [.mint, .indigo, .cyan, .red, .brown]
            return palette[area.unicodeScalars.reduce(0) { $0 &+ Int($1.value) } % palette.count]
        }
    }
}
