import Foundation

/// A JSON value that keeps object key order. Intern-Decision renders the state with Python's `json.dumps`, where key
/// order is the caller's, and its prompt must be reproduced byte for byte; `JSONSerialization` cannot do that.
public indirect enum JSONValue: Sendable, Equatable {
    case object([(key: String, value: JSONValue)])
    case array([JSONValue])
    case string(String)
    case integer(Int)
    case number(Double)
    case bool(Bool)
    case null

    public static func == (lhs: JSONValue, rhs: JSONValue) -> Bool {
        switch (lhs, rhs) {
        case (.object(let a), .object(let b)):
            a.count == b.count && zip(a, b).allSatisfy { $0.key == $1.key && $0.value == $1.value }
        case (.array(let a), .array(let b)): a == b
        case (.string(let a), .string(let b)): a == b
        case (.integer(let a), .integer(let b)): a == b
        case (.number(let a), .number(let b)): a == b
        case (.bool(let a), .bool(let b)): a == b
        case (.null, .null): true
        default: false
        }
    }

    /// Python `json.dumps(value, ensure_ascii=False, indent=indent)`: a nested value's lines are indented by
    /// `indent` spaces per level, items end with `,`, keys are followed by `: `, empty containers are `{}` / `[]`.
    public func pythonDump(indent: Int) -> String {
        var out = ""
        dump(into: &out, indent: indent, depth: 0)
        return out
    }

    private func dump(into out: inout String, indent: Int, depth: Int) {
        switch self {
        case .object(let members):
            guard !members.isEmpty else {
                out += "{}"
                return
            }
            out += "{\n"
            for (index, member) in members.enumerated() {
                out += String(repeating: " ", count: indent * (depth + 1))
                out += JSONValue.pythonString(member.key) + ": "
                member.value.dump(into: &out, indent: indent, depth: depth + 1)
                out += index + 1 < members.count ? ",\n" : "\n"
            }
            out += String(repeating: " ", count: indent * depth) + "}"
        case .array(let items):
            guard !items.isEmpty else {
                out += "[]"
                return
            }
            out += "[\n"
            for (index, item) in items.enumerated() {
                out += String(repeating: " ", count: indent * (depth + 1))
                item.dump(into: &out, indent: indent, depth: depth + 1)
                out += index + 1 < items.count ? ",\n" : "\n"
            }
            out += String(repeating: " ", count: indent * depth) + "]"
        case .string(let text): out += JSONValue.pythonString(text)
        case .integer(let value): out += String(value)
        case .number(let value): out += JSONValue.pythonFloat(value)
        case .bool(let value): out += value ? "true" : "false"
        case .null: out += "null"
        }
    }

    /// `json.dumps(str, ensure_ascii=False)`: escapes `"`, `\` and control characters only.
    static func pythonString(_ text: String) -> String {
        var out = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            case _ where scalar.value < 0x20: out += String(format: "\\u%04x", scalar.value)
            default: out.unicodeScalars.append(scalar)
            }
        }
        return out + "\""
    }

    /// Python `float.__repr__`: shortest round-trip digits, `.0` on integral values, exponent form below 1e-4 and
    /// from 1e16 written as `1e-05` / `1e+16`.
    static func pythonFloat(_ value: Double) -> String {
        if value.isNaN { return "NaN" }
        if value.isInfinite { return value > 0 ? "Infinity" : "-Infinity" }
        if value == 0 { return value.sign == .minus ? "-0.0" : "0.0" }
        let magnitude = abs(value)
        if magnitude >= 1e-4 && magnitude < 1e16 {
            let text = "\(value)"  // Swift's shortest round-trip form, positional in this range
            return text.contains(".") || text.contains("e") ? text : text + ".0"
        }
        // Swift prints e.g. "1e-05" as "1e-05" and "1e+16" as "1e+16"; normalise the exponent to Python's form.
        let text = "\(value)"
        guard let e = text.firstIndex(where: { $0 == "e" || $0 == "E" }) else { return text }
        var mantissa = String(text[..<e])
        if mantissa.hasSuffix(".0") { mantissa.removeLast(2) }
        var exponent = String(text[text.index(after: e)...])
        let negative = exponent.hasPrefix("-")
        exponent = exponent.trimmingCharacters(in: CharacterSet(charactersIn: "+-"))
        while exponent.count < 2 { exponent = "0" + exponent }
        return mantissa + "e" + (negative ? "-" : "+") + exponent
    }

    // MARK: Parsing

    public enum ParseError: Error, LocalizedError {
        case invalid(String)
        public var errorDescription: String? {
            if case .invalid(let reason) = self { return "Invalid JSON: \(reason)" }
            return nil
        }
    }

    /// Parses JSON text keeping object key order (duplicate keys are kept in order, as Python's `object_pairs_hook`
    /// would see them). Integers without a fraction or exponent become `.integer`.
    public static func parse(_ text: String) throws -> JSONValue {
        var parser = Parser(scalars: Array(text.unicodeScalars))
        let value = try parser.value()
        parser.skipWhitespace()
        guard parser.index == parser.scalars.count else { throw ParseError.invalid("trailing characters") }
        return value
    }

    private struct Parser {
        let scalars: [Unicode.Scalar]
        var index = 0

        init(scalars: [Unicode.Scalar]) { self.scalars = scalars }

        mutating func skipWhitespace() {
            while index < scalars.count, " \t\n\r".unicodeScalars.contains(scalars[index]) { index += 1 }
        }

        mutating func value() throws -> JSONValue {
            skipWhitespace()
            guard index < scalars.count else { throw ParseError.invalid("unexpected end") }
            switch scalars[index] {
            case "{":
                index += 1
                var members: [(key: String, value: JSONValue)] = []
                skipWhitespace()
                if peek() == "}" {
                    index += 1
                    return .object(members)
                }
                while true {
                    skipWhitespace()
                    guard peek() == "\"" else { throw ParseError.invalid("expected object key") }
                    let key = try string()
                    skipWhitespace()
                    guard peek() == ":" else { throw ParseError.invalid("expected ':'") }
                    index += 1
                    members.append((key, try value()))
                    skipWhitespace()
                    if peek() == "," {
                        index += 1
                        continue
                    }
                    guard peek() == "}" else { throw ParseError.invalid("expected '}'") }
                    index += 1
                    return .object(members)
                }
            case "[":
                index += 1
                var items: [JSONValue] = []
                skipWhitespace()
                if peek() == "]" {
                    index += 1
                    return .array(items)
                }
                while true {
                    items.append(try value())
                    skipWhitespace()
                    if peek() == "," {
                        index += 1
                        continue
                    }
                    guard peek() == "]" else { throw ParseError.invalid("expected ']'") }
                    index += 1
                    return .array(items)
                }
            case "\"": return .string(try string())
            case "t":
                try literal("true")
                return .bool(true)
            case "f":
                try literal("false")
                return .bool(false)
            case "n":
                try literal("null")
                return .null
            default: return try number()
            }
        }

        func peek() -> Unicode.Scalar? { index < scalars.count ? scalars[index] : nil }

        mutating func literal(_ word: String) throws {
            for scalar in word.unicodeScalars {
                guard peek() == scalar else { throw ParseError.invalid("bad literal") }
                index += 1
            }
        }

        mutating func number() throws -> JSONValue {
            let start = index
            var isFloat = false
            while let scalar = peek(), "+-0123456789.eE".unicodeScalars.contains(scalar) {
                if ".eE".unicodeScalars.contains(scalar) { isFloat = true }
                index += 1
            }
            var text = ""
            text.unicodeScalars.append(contentsOf: scalars[start..<index])
            if !isFloat, let integer = Int(text) { return .integer(integer) }
            guard let double = Double(text) else { throw ParseError.invalid("bad number \(text)") }
            return .number(double)
        }

        mutating func string() throws -> String {
            index += 1  // opening quote
            var out = String.UnicodeScalarView()
            while let scalar = peek() {
                index += 1
                switch scalar {
                case "\"": return String(out)
                case "\\":
                    guard let escaped = peek() else { throw ParseError.invalid("bad escape") }
                    index += 1
                    switch escaped {
                    case "\"": out.append("\"")
                    case "\\": out.append("\\")
                    case "/": out.append("/")
                    case "b": out.append("\u{08}")
                    case "f": out.append("\u{0C}")
                    case "n": out.append("\n")
                    case "r": out.append("\r")
                    case "t": out.append("\t")
                    case "u":
                        var code = try hex4()
                        if (0xD800...0xDBFF).contains(code), peek() == "\\", index + 1 < scalars.count,
                            scalars[index + 1] == "u"
                        {
                            index += 2
                            let low = try hex4()
                            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                        }
                        guard let unicode = Unicode.Scalar(code) else { throw ParseError.invalid("bad \\u escape") }
                        out.append(unicode)
                    default: throw ParseError.invalid("bad escape \\\(escaped)")
                    }
                default: out.append(scalar)
                }
            }
            throw ParseError.invalid("unterminated string")
        }

        mutating func hex4() throws -> UInt32 {
            guard index + 4 <= scalars.count else { throw ParseError.invalid("bad \\u escape") }
            var text = ""
            text.unicodeScalars.append(contentsOf: scalars[index..<index + 4])
            index += 4
            guard let code = UInt32(text, radix: 16) else { throw ParseError.invalid("bad \\u escape") }
            return code
        }
    }
}

extension JSONValue: ExpressibleByStringLiteral, ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral,
    ExpressibleByBooleanLiteral, ExpressibleByArrayLiteral, ExpressibleByDictionaryLiteral, ExpressibleByNilLiteral
{
    public init(stringLiteral value: String) { self = .string(value) }
    public init(integerLiteral value: Int) { self = .integer(value) }
    public init(floatLiteral value: Double) { self = .number(value) }
    public init(booleanLiteral value: Bool) { self = .bool(value) }
    public init(arrayLiteral elements: JSONValue...) { self = .array(elements) }
    public init(dictionaryLiteral elements: (String, JSONValue)...) {
        self = .object(elements.map { (key: $0.0, value: $0.1) })
    }
    public init(nilLiteral: ()) { self = .null }
}
