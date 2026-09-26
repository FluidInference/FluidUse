import Foundation
import os

/// Byte-level BPE tokenizer for Qwen `tokenizer.json` files (Qwen2 through Qwen3.5): NFC normalization, added tokens
/// matched literally, Qwen's split regex, GPT-2 byte-to-unicode mapping, then merges by rank. Encoding only; no special
/// tokens are added.
public final class QwenBPETokenizer: Sendable {
    private let vocabulary: [String: Int]
    private let mergeRanks: [String: Int]
    /// Added tokens, longest first, so a longer token wins over one it contains.
    private let addedTokens: [(content: String, id: Int)]
    private let byteToCharacter: [Character]
    /// Encoded pieces; ordinary text repeats the same words, so this saves most of the merge loops.
    private let cache = OSAllocatedUnfairLock<[String: [Int]]>(initialState: [:])

    public init(tokenizerJsonURL: URL) throws {
        let data = try Data(contentsOf: tokenizerJsonURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let model = root["model"] as? [String: Any], model["type"] as? String == "BPE",
            let vocab = model["vocab"] as? [String: Int], let merges = model["merges"] as? [Any],
            let added = root["added_tokens"] as? [[String: Any]]
        else { throw QwenTokenizerError.invalidAsset("Expected a Hugging Face byte-level BPE tokenizer.json") }
        guard (model["byte_fallback"] as? Bool ?? false) == false else {
            throw QwenTokenizerError.invalidAsset("byte_fallback BPE is not supported")
        }
        vocabulary = vocab
        var ranks: [String: Int] = [:]
        ranks.reserveCapacity(merges.count)
        for (rank, merge) in merges.enumerated() {
            let pair: [String]
            if let list = merge as? [String] {
                pair = list
            } else if let text = merge as? String {
                pair = text.split(separator: " ", maxSplits: 1).map(String.init)
            } else {
                pair = []
            }
            guard pair.count == 2 else { throw QwenTokenizerError.invalidAsset("Malformed merge \(rank)") }
            ranks[pair[0] + "\u{0}" + pair[1]] = rank
        }
        mergeRanks = ranks
        addedTokens = added.compactMap { entry in
            guard let content = entry["content"] as? String, let id = entry["id"] as? Int else { return nil }
            return (content, id)
        }.sorted { $0.content.count > $1.content.count }
        byteToCharacter = Self.bytesToUnicode()
        _ = try Self.splitter()
    }

    /// Qwen's pre-tokenizer split. Built per call: NSRegularExpression is not Sendable.
    private static func splitter() throws -> NSRegularExpression {
        try NSRegularExpression(
            pattern:
                #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        )
    }

    /// Token ids for `text`, without any special tokens around it.
    public func encode(_ text: String) throws -> [Int] {
        let splitter = try Self.splitter()
        var ids: [Int] = []
        for (segment, addedID) in splitAddedTokens(text.precomposedStringWithCanonicalMapping) {
            if let addedID {
                ids.append(addedID)
                continue
            }
            let string = segment as NSString
            for match in splitter.matches(in: segment, range: NSRange(location: 0, length: string.length)) {
                ids += encodePiece(string.substring(with: match.range))
            }
        }
        return ids
    }

    public func id(for token: String) -> Int? {
        addedTokens.first { $0.content == token }?.id ?? vocabulary[token]
    }

    private func splitAddedTokens(_ text: String) -> [(String, Int?)] {
        var parts: [(String, Int?)] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            var earliest: (range: Range<Substring.Index>, id: Int)?
            for (content, id) in addedTokens {
                guard let range = rest.range(of: content, options: .literal) else { continue }
                if earliest == nil || range.lowerBound < earliest!.range.lowerBound {
                    earliest = (range, id)
                }
            }
            guard let found = earliest else {
                parts.append((String(rest), nil))
                break
            }
            if found.range.lowerBound > rest.startIndex {
                parts.append((String(rest[..<found.range.lowerBound]), nil))
            }
            parts.append(("", found.id))
            rest = rest[found.range.upperBound...]
        }
        return parts
    }

    private func encodePiece(_ piece: String) -> [Int] {
        if let cached = cache.withLock({ $0[piece] }) { return cached }
        var symbols = piece.utf8.map { String(byteToCharacter[Int($0)]) }
        while symbols.count > 1 {
            var best: (rank: Int, index: Int)?
            for index in 0..<(symbols.count - 1) {
                if let rank = mergeRanks[symbols[index] + "\u{0}" + symbols[index + 1]],
                    best == nil || rank < best!.rank
                {
                    best = (rank, index)
                }
            }
            guard let (_, index) = best else { break }
            let merged = symbols[index] + symbols[index + 1]
            // Merge every occurrence of this pair left to right, as the reference BPE does.
            var next: [String] = []
            next.reserveCapacity(symbols.count)
            var i = 0
            while i < symbols.count {
                if i < symbols.count - 1, symbols[i] == symbols[index], symbols[i + 1] == symbols[index + 1] {
                    next.append(merged)
                    i += 2
                } else {
                    next.append(symbols[i])
                    i += 1
                }
            }
            symbols = next
        }
        let ids = symbols.compactMap { vocabulary[$0] }
        cache.withLock { storage in
            if storage.count > 100_000 { storage.removeAll(keepingCapacity: true) }
            storage[piece] = ids
        }
        return ids
    }

    /// GPT-2's reversible byte -> printable character table.
    private static func bytesToUnicode() -> [Character] {
        var bytes = Array(33...126) + Array(161...172) + Array(174...255)
        var codes = bytes
        var extra = 0
        for byte in 0..<256 where !bytes.contains(byte) {
            bytes.append(byte)
            codes.append(256 + extra)
            extra += 1
        }
        var table = [Character](repeating: " ", count: 256)
        for (byte, code) in zip(bytes, codes) {
            table[byte] = Character(UnicodeScalar(code)!)
        }
        return table
    }
}

public enum QwenTokenizerError: Error, LocalizedError, Sendable {
    case invalidAsset(String)

    public var errorDescription: String? {
        switch self {
        case .invalidAsset(let reason): "Invalid Qwen tokenizer asset: \(reason)"
        }
    }
}
