import Foundation

/// Hugging Face Unigram tokenizer used by both GLiNER 2.5 checkpoints.
/// Each schema item and each text word is tokenized separately, as in GLiNER's SchemaTransformer.
public struct GLiNER2Tokenizer: Sendable {
    private struct Piece: Sendable {
        let id: Int
        let score: Double
    }

    private let vocabulary: [String: Piece]
    private let specialTokens: [String: Int]
    private let specialTokenNames: [String]
    private let maximumPieceLength: Int
    private let unknownScore: Double
    private let unknownId: Int

    public let padTokenId: Int
    public let labelTokenId: Int
    public let separatorTokenId: Int

    public init(tokenizerJsonURL: URL) throws {
        let data = try Data(contentsOf: tokenizerJsonURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let model = root["model"] as? [String: Any],
            model["type"] as? String == "Unigram",
            let entries = model["vocab"] as? [[Any]],
            let unknownId = model["unk_id"] as? Int,
            let added = root["added_tokens"] as? [[String: Any]]
        else { throw GLiNER2Error.invalidAsset("Expected a Hugging Face Unigram tokenizer.json") }

        var lookup: [String: Piece] = [:]
        lookup.reserveCapacity(entries.count)
        var longest = 0
        var lowest = 0.0
        for (id, entry) in entries.enumerated() {
            guard entry.count == 2, let value = entry[0] as? String,
                let score = entry[1] as? Double
            else { throw GLiNER2Error.invalidAsset("Malformed Unigram vocabulary entry \(id)") }
            lookup[value] = Piece(id: id, score: score)
            longest = max(longest, value.unicodeScalars.count)
            lowest = min(lowest, score)
        }
        var specials: [String: Int] = [:]
        for entry in added where entry["special"] as? Bool == true {
            guard let content = entry["content"] as? String, let id = entry["id"] as? Int else {
                throw GLiNER2Error.invalidAsset("Malformed added special token")
            }
            specials[content] = id
        }
        guard let pad = specials["[PAD]"], let label = specials["[L]"],
            let separator = specials["[SEP_TEXT]"], unknownId >= 0, unknownId < entries.count
        else { throw GLiNER2Error.invalidAsset("Missing GLiNER special tokens") }

        self.vocabulary = lookup
        self.specialTokens = specials
        self.specialTokenNames = specials.keys.sorted { left, right in
            left.count == right.count ? left < right : left.count > right.count
        }
        self.maximumPieceLength = longest
        self.unknownScore = lowest - 10
        self.unknownId = unknownId
        self.padTokenId = pad
        self.labelTokenId = label
        self.separatorTokenId = separator
    }

    /// Tokenize a single schema item or text word without adding CLS/SEP.
    public func encode(_ item: String) -> [Int] {
        if let id = specialTokens[item] { return [id] }
        var remaining = item[...]
        var ids: [Int] = []
        while !remaining.isEmpty {
            let first = specialTokenNames.compactMap { name -> (Range<String.Index>, Int)? in
                guard let range = remaining.range(of: name, options: .literal), let id = specialTokens[name] else {
                    return nil
                }
                return (range, id)
            }.min { left, right in
                left.0.lowerBound < right.0.lowerBound
            }
            guard let (range, id) = first else {
                ids += encodeOrdinary(String(remaining))
                break
            }
            ids += encodeOrdinary(String(remaining[..<range.lowerBound]))
            ids.append(id)
            remaining = remaining[range.upperBound...]
        }
        return ids
    }

    private func encodeOrdinary(_ item: String) -> [Int] {
        let normalized = item.replacingOccurrences(
            of: #"\s{2,}|[\n\r\t]"#, with: " ", options: .regularExpression
        ).precomposedStringWithCanonicalMapping
        let trimmed = normalized.replacingOccurrences(of: #"\s+$"#, with: "", options: .regularExpression)
        guard !trimmed.isEmpty else { return [] }
        let metaspace = trimmed.replacingOccurrences(of: " ", with: "▁")
        let scalars = Array(((trimmed.hasPrefix(" ") ? "" : "▁") + metaspace).unicodeScalars)
        let count = scalars.count
        var scores = [Double](repeating: -.infinity, count: count + 1)
        var previous = [Int](repeating: -1, count: count + 1)
        var pieceIds = [Int](repeating: unknownId, count: count + 1)
        scores[0] = 0
        for start in 0..<count where scores[start].isFinite {
            let limit = min(count, start + maximumPieceLength)
            if start < limit {
                for end in (start + 1)...limit {
                    let text = String(String.UnicodeScalarView(scalars[start..<end]))
                    guard let piece = vocabulary[text] else { continue }
                    let candidate = scores[start] + piece.score
                    if candidate > scores[end] {
                        scores[end] = candidate
                        previous[end] = start
                        pieceIds[end] = piece.id
                    }
                }
            }
            let fallback = scores[start] + unknownScore
            if fallback > scores[start + 1] {
                scores[start + 1] = fallback
                previous[start + 1] = start
                pieceIds[start + 1] = unknownId
            }
        }
        var reversed: [Int] = []
        var position = count
        while position > 0 {
            reversed.append(pieceIds[position])
            position = previous[position]
        }
        var ids: [Int] = []
        for id in reversed.reversed() where id != unknownId || ids.last != unknownId {
            ids.append(id)
        }
        return ids
    }

    /// Python's word class includes Unicode letters, numbers, and underscore, but excludes combining marks.
    static func splitText(_ text: String) throws -> [String] {
        let pattern = try NSRegularExpression(
            pattern:
                #"(?:https?://[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*|\S"#,
            options: .caseInsensitive)
        let string = text as NSString
        return pattern.matches(in: text, range: NSRange(location: 0, length: string.length))
            .map { string.substring(with: $0.range).lowercased() }
    }

    /// Native GLiNER classification schema and word splitting. The caller must check bucket capacity.
    public func classificationSequence(
        text: String, task: String, labels: [String]
    ) throws -> (ids: [Int], markers: [Int]) {
        let sequence = try classificationSequence(text: text, heads: [(task, labels)])
        return (sequence.ids, sequence.markers[0])
    }

    /// Several classification heads in one schema, joined by `[SEP_STRUCT]` as GLiNER's SchemaTransformer does.
    /// `markers[h]` holds the token position of each `[L]` marker of head `h`.
    public func classificationSequence(
        text: String, heads: [(task: String, labels: [String])]
    ) throws -> (ids: [Int], markers: [[Int]]) {
        var source = text
        if source.isEmpty || !source.hasSuffix(".") && !source.hasSuffix("!") && !source.hasSuffix("?") {
            source += "."
        }
        var ids: [Int] = []
        var markers: [[Int]] = []
        for (index, head) in heads.enumerated() {
            if index > 0 { ids += encode("[SEP_STRUCT]") }
            ids += encode("(") + encode("[P]") + encode(head.task) + encode("(")
            var positions: [Int] = []
            for label in head.labels {
                positions.append(ids.count)
                ids.append(labelTokenId)
                ids += encode(label)
            }
            ids += encode(")") + encode(")")
            markers.append(positions)
        }
        ids.append(separatorTokenId)
        for word in try Self.splitText(source) {
            ids += encode(word)
        }
        return (ids, markers)
    }
}
