import SwiftUI

/// Dark-editor colouring for Python.
enum PythonHighlighter {
    static let keyword = Color(red: 0.99, green: 0.37, blue: 0.64)
    static let builtin = Color(red: 0.36, green: 0.85, blue: 1.0)
    static let call = Color(red: 0.40, green: 0.72, blue: 0.64)
    static let declaration = Color(red: 0.25, green: 0.63, blue: 0.75)
    static let string = Color(red: 0.99, green: 0.42, blue: 0.36)
    static let number = Color(red: 0.82, green: 0.75, blue: 0.41)
    static let comment = Color(red: 0.47, green: 0.53, blue: 0.60)
    static let decorator = Color(red: 0.99, green: 0.56, blue: 0.25)
    static let plain = Color(white: 0.92)

    private static let keywords: Set<String> = [
        "def", "return", "if", "elif", "else", "for", "while", "in", "not", "and", "or", "is", "import", "from", "as",
        "class", "try", "except", "finally", "raise", "with", "yield", "lambda", "pass", "break", "continue", "global",
        "nonlocal", "assert", "del", "None", "True", "False", "async", "await", "match", "case",
    ]
    private static let builtins: Set<String> = [
        "len", "range", "print", "int", "str", "float", "list", "dict", "set", "tuple", "bool", "sorted", "sum", "min",
        "max", "abs", "enumerate", "zip", "map", "filter", "any", "all", "isinstance", "self", "reversed", "round",
        "ord", "chr", "type", "object", "super",
    ]
    private static let token = try! NSRegularExpression(
        pattern:
            #"(#.*$)|([rbfu]{0,2}(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'))|(@[A-Za-z_][A-Za-z0-9_.]*)|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)"#
    )

    /// One coloured line per source line (a triple-quoted string spanning lines is tracked across lines).
    static func lines(_ source: String) -> [AttributedString] {
        var result: [AttributedString] = []
        var inDocstring: String?
        for line in source.components(separatedBy: "\n") {
            if let quote = inDocstring {
                result.append(piece(line, string))
                if line.contains(quote) { inDocstring = nil }
                continue
            }
            if let quote = ["\"\"\"", "'''"].first(where: { line.contains($0) }) {
                result.append(piece(line, string))
                if (line.components(separatedBy: quote).count - 1) % 2 == 1 { inDocstring = quote }
                continue
            }
            result.append(code(line))
        }
        return result
    }

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
                color = decorator
            } else if match.range(at: 4).location != NSNotFound {
                color = number
            } else if keywords.contains(text) {
                color = keyword
            } else if previousWord == "def" || previousWord == "class" {
                color = declaration
            } else if builtins.contains(text) {
                color = builtin
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
}
