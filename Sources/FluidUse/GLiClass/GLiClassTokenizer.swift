import Foundation
import OSLog

/// Byte-level BPE encoder used by the Ettin/ModernBERT checkpoint behind GLiClass Edge.
public final class GLiClassTokenizer: Sendable {
    public let classTokenId: Int
    public let clsTokenId: Int
    public let sepTokenId: Int
    public let padTokenId: Int

    private struct AddedToken: Sendable {
        let content: String
        let id: Int
        let lstrip: Bool
        let rstrip: Bool
    }

    private enum Segment {
        case added(Int)
        case text(String)
    }

    private let vocab: [String: Int]
    private let mergeRank: [String: Int]
    private let addedTokens: [AddedToken]
    private let splitRegex: NSRegularExpression
    private let byteCharacters: [String]
    private let cache = TokenCache()

    private static let splitPattern =
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"

    public init(tokenizerJsonURL: URL) throws {
        let data = try Data(contentsOf: tokenizerJsonURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let model = root["model"] as? [String: Any], model["type"] as? String == "BPE",
            let rawVocab = model["vocab"] as? [String: Any], let rawMerges = model["merges"] as? [[String]]
        else {
            throw GLiClassError.invalidAsset("tokenizer.json must contain a byte-level BPE model")
        }
        var vocab: [String: Int] = [:]
        vocab.reserveCapacity(rawVocab.count)
        for (token, value) in rawVocab {
            guard let id = value as? Int else {
                throw GLiClassError.invalidAsset("tokenizer id for \(token.debugDescription) is not an integer")
            }
            vocab[token] = id
        }
        self.vocab = vocab

        var ranks: [String: Int] = [:]
        ranks.reserveCapacity(rawMerges.count)
        for (rank, pair) in rawMerges.enumerated() {
            guard pair.count == 2 else { throw GLiClassError.invalidAsset("tokenizer merge \(rank) is malformed") }
            ranks["\(pair[0]) \(pair[1])"] = rank
        }
        self.mergeRank = ranks

        var added: [AddedToken] = []
        if let entries = root["added_tokens"] as? [[String: Any]] {
            for entry in entries {
                guard let content = entry["content"] as? String, let id = entry["id"] as? Int else { continue }
                added.append(
                    AddedToken(
                        content: content, id: id, lstrip: entry["lstrip"] as? Bool ?? false,
                        rstrip: entry["rstrip"] as? Bool ?? false))
            }
        }
        self.addedTokens = added.sorted { $0.content.count > $1.content.count }
        self.splitRegex = try NSRegularExpression(pattern: Self.splitPattern)
        self.byteCharacters = Self.bytesToUnicode()

        func required(_ text: String) throws -> Int {
            guard let id = added.first(where: { $0.content == text })?.id ?? vocab[text] else {
                throw GLiClassError.invalidAsset("tokenizer.json has no \(text) token")
            }
            return id
        }
        self.classTokenId = try required("<<LABEL>>")
        self.clsTokenId = try required("[CLS]")
        self.sepTokenId = try required("[SEP]")
        self.padTokenId = try required("[PAD]")
    }

    /// Encode with the checkpoint's `[CLS] ... [SEP]` post-processing template.
    public func encode(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        let normalized = text.precomposedStringWithCanonicalMapping
        var ids: [Int] = []
        for segment in splitAddedTokens(normalized) {
            switch segment {
            case .added(let id): ids.append(id)
            case .text(let text): encodeText(text, into: &ids)
            }
        }
        return addSpecialTokens ? [clsTokenId] + ids + [sepTokenId] : ids
    }

    private func encodeText(_ text: String, into ids: inout [Int]) {
        guard !text.isEmpty else { return }
        // ByteLevel(add_prefix_space: true) applies to each non-special split.
        let input = text.first?.isWhitespace == true ? text : " " + text
        let ns = input as NSString
        for match in splitRegex.matches(in: input, range: NSRange(location: 0, length: ns.length)) {
            let piece = ns.substring(with: match.range)
            let symbols = piece.utf8.map { byteCharacters[Int($0)] }
            for token in bpe(symbols) {
                if let id = vocab[token] { ids.append(id) }
            }
        }
    }

    private func splitAddedTokens(_ text: String) -> [Segment] {
        var segments: [Segment] = []
        var buffer = ""
        var index = text.startIndex
        outer: while index < text.endIndex {
            for token in addedTokens {
                guard let end = text.index(index, offsetBy: token.content.count, limitedBy: text.endIndex),
                    text[index..<end] == token.content
                else { continue }
                if token.lstrip {
                    while buffer.last?.isWhitespace == true { buffer.removeLast() }
                }
                if !buffer.isEmpty {
                    segments.append(.text(buffer))
                    buffer = ""
                }
                segments.append(.added(token.id))
                index = end
                if token.rstrip {
                    while index < text.endIndex, text[index].isWhitespace { index = text.index(after: index) }
                }
                continue outer
            }
            buffer.append(text[index])
            index = text.index(after: index)
        }
        if !buffer.isEmpty { segments.append(.text(buffer)) }
        return segments
    }

    private func bpe(_ symbols: [String]) -> [String] {
        let key = symbols.joined(separator: "\u{0}")
        if let value = cache.lookup(key) { return value }
        var parts = symbols
        while parts.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for index in 0..<(parts.count - 1) {
                if let rank = mergeRank["\(parts[index]) \(parts[index + 1])"], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }
            if bestIndex < 0 { break }
            parts[bestIndex] += parts[bestIndex + 1]
            parts.remove(at: bestIndex + 1)
        }
        cache.store(key, parts)
        return parts
    }

    private struct TokenCache: Sendable {
        private let values = OSAllocatedUnfairLock<[String: [String]]>(initialState: [:])

        func lookup(_ key: String) -> [String]? { values.withLock { $0[key] } }
        func store(_ key: String, _ value: [String]) {
            values.withLock {
                if $0.count >= 8192 { $0.removeAll(keepingCapacity: true) }
                $0[key] = value
            }
        }
    }

    private static func bytesToUnicode() -> [String] {
        var scalars = [UInt32](repeating: 0, count: 256)
        var assigned = [Bool](repeating: false, count: 256)
        for range in [UInt32(33)...126, UInt32(161)...172, UInt32(174)...255] {
            for byte in range {
                scalars[Int(byte)] = byte
                assigned[Int(byte)] = true
            }
        }
        var offset: UInt32 = 0
        for byte in 0..<256 where !assigned[byte] {
            scalars[byte] = 256 + offset
            offset += 1
        }
        return scalars.map { String(UnicodeScalar($0)!) }
    }
}
