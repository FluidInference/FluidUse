import Foundation

/// One typed question for a Decision 2.0 model (vLLM Semantic Router), as in its System One `questions` object.
/// Instructions and descriptions are text or structured JSON, rendered exactly as the upstream runtime renders them.
public enum Decision2Question: Sendable {
    /// Pick one of `options` (key, description); keys are reported in this order. A `.null` description is allowed.
    case choice(instructions: OrderedJSON, options: [(key: String, description: OrderedJSON)])
    /// Yes or no; reported under `false`, `true` in that order. Upstream's defaults are "No" / "Yes".
    case noul(instructions: OrderedJSON, no: OrderedJSON = "No", yes: OrderedJSON = "Yes")
    /// One of 2…10 ordered levels; keys are `"0"`, `"1"`, ….
    case score(instructions: OrderedJSON, levels: [OrderedJSON])

    public static func choice(_ instructions: String, _ options: KeyValuePairs<String, String>) -> Decision2Question {
        .choice(instructions: .string(instructions), options: options.map { ($0.key, .string($0.value)) })
    }

    public static func yesNo(_ instructions: String) -> Decision2Question {
        .noul(instructions: .string(instructions))
    }

    public static func score(_ instructions: String, _ levels: [String]) -> Decision2Question {
        .score(instructions: .string(instructions), levels: levels.map { .string($0) })
    }

    /// A question in the System One JSON form: `{"type": "choice" | "noul" | "score", "instructions": …,
    /// "criteria": …}`. Choice criteria keep their key order.
    public init(json: OrderedJSON) throws {
        guard case .object(let members) = json else { throw Decision2Error.invalidQuestion("question is not an object") }
        let field = { (name: String) in members.first(where: { $0.key == name })?.value }
        guard case .string(let type)? = field("type") else { throw Decision2Error.invalidQuestion("missing type") }
        guard let instructions = field("instructions"), instructions != .string(""), instructions != .null else {
            throw Decision2Error.invalidQuestion("missing instructions")
        }
        switch (type, field("criteria")) {
        case ("choice", .object(let options)?) where (2...255).contains(options.count):
            self = .choice(instructions: instructions, options: options.map { (key: $0.key, description: $0.value) })
        case ("noul", nil), ("noul", .null?):
            self = .noul(instructions: instructions)
        case ("noul", .object(let sides)?) where Set(sides.map(\.key)).isSubset(of: ["false", "true"]):
            let side = { (key: String) in sides.first(where: { $0.key == key })?.value }
            self = .noul(instructions: instructions, no: side("false") ?? "No", yes: side("true") ?? "Yes")
        case ("score", .array(let levels)?) where (2...10).contains(levels.count):
            self = .score(instructions: instructions, levels: levels)
        default:
            throw Decision2Error.invalidQuestion("unsupported \(type) criteria")
        }
    }

    var taskType: String {
        switch self {
        case .choice: "choice"
        case .noul: "noul"
        case .score: "score"
        }
    }

    var instructions: OrderedJSON {
        switch self {
        case .choice(let instructions, _), .noul(let instructions, _, _), .score(let instructions, _): instructions
        }
    }

    /// `(key, description)` pairs in the order the model reads them.
    var options: [(key: String, description: OrderedJSON)] {
        switch self {
        case .choice(_, let options): options
        case .noul(_, let no, let yes): [("false", no), ("true", yes)]
        case .score(_, let levels): levels.enumerated().map { (String($0.offset), $0.element) }
        }
    }
}

/// One answer, with the same fields as upstream `system_one()`.
public struct Decision2Answer: Sendable {
    public let id: String
    /// `choice`, `noul` or `score`.
    public let type: String
    public let keys: [String]
    /// Softmax over the options (after the package's Score offsets, when it has them).
    public let probabilities: [Double]

    /// The most probable key; ties go to the earlier option, as upstream.
    public var choice: String {
        var best = 0
        for i in probabilities.indices where probabilities[i] > probabilities[best] { best = i }
        return keys[best]
    }
    /// Probability of `true` for a yes/no question.
    public var yes: Double? { type == "noul" ? probabilities[keys.firstIndex(of: "true")!] : nil }
    /// Expected level for a Score question.
    public var score: Double? {
        type == "score" ? zip(keys, probabilities).reduce(0) { $0 + Double(Int($1.0)!) * $1.1 } : nil
    }
    /// 1 − normalized entropy (upstream's product confidence); nil for yes/no.
    public var confidence: Double? {
        guard type != "noul" else { return nil }
        let entropy = -probabilities.filter { $0 > 0 }.reduce(0) { $0 + $1 * log($1) }
        return max(0, min(1, 1 - entropy / log(Double(keys.count))))
    }

    public func probability(_ key: String) -> Double? { keys.firstIndex(of: key).map { probabilities[$0] } }
}

public struct Decision2Result: Sendable {
    /// In the order the questions were given.
    public let answers: [Decision2Answer]
    /// Upstream's `usage.input_tokens`: the sum of every question's own row.
    public let inputTokens: Int
    /// Core ML calls this request took (1 unless it had to be split).
    public let calls: Int
    public subscript(id: String) -> Decision2Answer? { answers.first(where: { $0.id == id }) }
}

public enum Decision2Error: Error, LocalizedError, Sendable {
    case invalidQuestion(String)
    case invalidAsset(String)
    case invalidOutput(String)
    case tooLong(String)

    public var errorDescription: String? {
        switch self {
        case .invalidQuestion(let reason): "Invalid Decision 2.0 question: \(reason)"
        case .invalidAsset(let reason): "Invalid Decision 2.0 asset: \(reason)"
        case .invalidOutput(let reason): "Invalid Decision 2.0 output: \(reason)"
        case .tooLong(let reason): "Decision 2.0 request too long: \(reason)"
        }
    }
}

extension OrderedJSON {
    /// Python `json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`, the form Decision 2.0
    /// renders structured payloads in. Keys sort by code point, as Python's `str` ordering.
    public var canonical: String {
        switch self {
        case .object(let members):
            let sorted = members.sorted { a, b in
                a.key.unicodeScalars.lexicographicallyPrecedes(b.key.unicodeScalars) { $0.value < $1.value }
            }
            return "{" + sorted.map { OrderedJSON.pythonString($0.key) + ":" + $0.value.canonical }.joined(separator: ",")
                + "}"
        case .array(let items): return "[" + items.map(\.canonical).joined(separator: ",") + "]"
        case .string(let text): return OrderedJSON.pythonString(text)
        case .integer(let value): return String(value)
        case .number(let value): return OrderedJSON.pythonFloat(value)
        case .bool(let value): return value ? "true" : "false"
        case .null: return "null"
        }
    }

    /// Upstream `_payload`: text verbatim, anything else canonical JSON.
    var decisionPayload: String {
        if case .string(let text) = self { return text }
        return canonical
    }
}
