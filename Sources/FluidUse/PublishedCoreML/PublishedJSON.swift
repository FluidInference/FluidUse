import Foundation

/// JSON value that keeps object member order.
///
/// The bridged runtimes treat member order as part of the request: a Choice question's criteria order
/// is the option order the model sees. `JSONSerialization` dictionaries do not preserve it.
public indirect enum PublishedJSON: Sendable, Equatable {
    public struct Member: Sendable, Equatable {
        public let key: String
        public let value: PublishedJSON

        public init(_ key: String, _ value: PublishedJSON) {
            self.key = key
            self.value = value
        }
    }

    case null
    case bool(Bool)
    case integer(Int)
    case number(Double)
    case string(String)
    case array([PublishedJSON])
    case object([Member])

    /// Compact UTF-8 encoding in member order. Non-finite numbers and duplicate keys are rejected.
    public func encoded() throws -> Data {
        var text = ""
        try write(into: &text)
        return Data(text.utf8)
    }

    private func write(into text: inout String) throws {
        switch self {
        case .null: text += "null"
        case .bool(let value): text += value ? "true" : "false"
        case .integer(let value): text += String(value)
        case .number(let value):
            guard value.isFinite else { throw PublishedCoreMLError.invalidRequest("JSON numbers must be finite") }
            text += value.description
        case .string(let value): Self.writeString(value, into: &text)
        case .array(let values):
            text += "["
            for (index, value) in values.enumerated() {
                if index > 0 { text += "," }
                try value.write(into: &text)
            }
            text += "]"
        case .object(let members):
            guard Set(members.map(\.key)).count == members.count else {
                throw PublishedCoreMLError.invalidRequest("JSON object keys must be unique")
            }
            text += "{"
            for (index, member) in members.enumerated() {
                if index > 0 { text += "," }
                Self.writeString(member.key, into: &text)
                text += ":"
                try member.value.write(into: &text)
            }
            text += "}"
        }
    }

    private static func writeString(_ value: String, into text: inout String) {
        text += "\""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": text += "\\\""
            case "\\": text += "\\\\"
            case "\n": text += "\\n"
            case "\r": text += "\\r"
            case "\t": text += "\\t"
            case _ where scalar.value < 0x20:
                text += String(format: "\\u%04x", scalar.value)
            default: text.unicodeScalars.append(scalar)
            }
        }
        text += "\""
    }
}

extension PublishedJSON: ExpressibleByStringLiteral, ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral,
    ExpressibleByBooleanLiteral, ExpressibleByNilLiteral
{
    public init(stringLiteral value: String) { self = .string(value) }
    public init(integerLiteral value: Int) { self = .integer(value) }
    public init(floatLiteral value: Double) { self = .number(value) }
    public init(booleanLiteral value: Bool) { self = .bool(value) }
    public init(nilLiteral: ()) { self = .null }
}
