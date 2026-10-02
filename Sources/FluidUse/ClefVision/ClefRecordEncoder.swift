import Foundation

/// Swift port of Clef's `encode_record` (Cloudflare/clef `joint_schema_model.py`, Apache-2.0): the token stream
/// the student was trained on, with the question / option spans the head averages over.
public struct ClefEncodedRecord: Sendable {
    public struct Question: Sendable {
        public let id: String
        public let typeIndex: Int
        public let span: Range<Int>
        public let optionSpans: [Range<Int>]
        public let optionIDs: [String]
    }

    public let inputIDs: [Int]
    public let questions: [Question]
    /// Token range of each image's `<|image_pad|>` run, in order.
    public let imageTokenRanges: [Range<Int>]
}

enum ClefRecordEncoder {
    static let systemPrompt =
        "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options."

    /// `imageTokenCounts`: merged vision tokens per image (grid_h * grid_w / merge²), in the order the images are given.
    static func encode(
        state: Any, questions: [(id: String, question: ClefQuestion)], imageTokenCounts: [Int],
        tokenizer: QwenBPETokenizer, config: ClefVisionConfig, maxLength: Int
    ) throws -> ClefEncodedRecord {
        func tokens(_ text: String) throws -> [Int] { try tokenizer.encode(text) }

        var schema = try tokens("\n\nSCHEMA FIELDS:\n")
        var encodedQuestions: [ClefEncodedRecord.Question] = []
        for (index, (id, question)) in questions.enumerated() {
            schema += try tokens("\nFIELD \(index + 1)\nID: \(id)\nTYPE: \(question.typeName)\nINSTRUCTION: ")
            let start = schema.count
            let instructions = question.instructions.flatMap { $0.isEmpty ? nil : $0 } ?? id
            schema += try tokens(ClefJSON.render(instructions))
            let end = schema.count
            schema += try tokens("\nALLOWED OPTIONS:\n")
            var optionSpans: [Range<Int>] = []
            var optionIDs: [String] = []
            for (optionIndex, option) in question.options().enumerated() {
                schema += try tokens("OPTION \(optionIndex + 1): ")
                let optionStart = schema.count
                var semantics: [String: Any] = ["option_id": option.id]
                if let description = option.description { semantics["description"] = description }
                schema += try tokens(ClefJSON.render(semantics))
                optionSpans.append(optionStart..<schema.count)
                optionIDs.append(option.id)
                schema += try tokens("\n")
            }
            schema += try tokens("END FIELD\n")
            encodedQuestions.append(
                .init(id: id, typeIndex: question.typeIndex, span: start..<end, optionSpans: optionSpans, optionIDs: optionIDs))
        }

        var prefix = try tokens("<|im_start|>system\n\(systemPrompt)<|im_end|>\n<|im_start|>user\nSTATE:\n")
        var imageRanges: [Range<Int>] = []
        if !imageTokenCounts.isEmpty {
            for count in imageTokenCounts {
                prefix.append(config.visionStartID)
                let start = prefix.count
                prefix += Array(repeating: config.imageTokenID, count: count)
                imageRanges.append(start..<prefix.count)
                prefix.append(config.visionEndID)
            }
            prefix += try tokens("\n")
        }
        let suffix = try tokens("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
        var stateIDs = try tokens(ClefJSON.render(state))
        let fixed = prefix.count + schema.count + suffix.count
        guard fixed <= maxLength else { throw ClefVisionError.tooLong(tokens: fixed, bucket: maxLength) }
        if stateIDs.count > maxLength - fixed { stateIDs = Array(stateIDs.prefix(maxLength - fixed)) }
        let offset = prefix.count + stateIDs.count
        let shifted = encodedQuestions.map { q in
            ClefEncodedRecord.Question(
                id: q.id, typeIndex: q.typeIndex, span: (q.span.lowerBound + offset)..<(q.span.upperBound + offset),
                optionSpans: q.optionSpans.map { ($0.lowerBound + offset)..<($0.upperBound + offset) }, optionIDs: q.optionIDs)
        }
        return ClefEncodedRecord(inputIDs: prefix + stateIDs + schema + suffix, questions: shifted, imageTokenRanges: imageRanges)
    }
}

/// `json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)`; strings render as themselves.
enum ClefJSON {
    static func render(_ value: Any) -> String {
        if let string = value as? String { return string }
        return dump(value)
    }

    static func dump(_ value: Any) -> String {
        switch value {
        case let string as String:
            return quote(string)
        case let bool as Bool:
            return bool ? "true" : "false"
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() { return number.boolValue ? "true" : "false" }
            if number.doubleValue.rounded() == number.doubleValue, !"\(number)".contains(".") { return "\(number.int64Value)" }
            return pythonFloat(number.doubleValue)
        case let int as Int:
            return String(int)
        case let double as Double:
            return pythonFloat(double)
        case let array as [Any]:
            return "[" + array.map(dump).joined(separator: ",") + "]"
        case let dict as [String: Any]:
            return "{" + dict.keys.sorted().map { quote($0) + ":" + dump(dict[$0]!) }.joined(separator: ",") + "}"
        case is NSNull:
            return "null"
        default:
            return quote(String(describing: value))
        }
    }

    /// Python's `repr(float)`: shortest round-trip, always with a decimal point or exponent.
    static func pythonFloat(_ value: Double) -> String {
        if value.isNaN { return "NaN" }
        if value.isInfinite { return value > 0 ? "Infinity" : "-Infinity" }
        var text = "\(value)"  // Swift prints shortest round-trip too
        if text.hasSuffix(".0") == false, !text.contains("."), !text.contains("e") { text += ".0" }
        if text.contains("e") {  // Swift: 1e-05 -> Python: 1e-05 (same), but Swift writes 1e-05 as "1e-05"
            text = text.replacingOccurrences(of: "e+", with: "e+")
        }
        return text
    }

    static func quote(_ string: String) -> String {
        var out = "\""
        for scalar in string.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)  // ensure_ascii=False keeps non-ASCII as is
                }
            }
        }
        return out + "\""
    }
}
