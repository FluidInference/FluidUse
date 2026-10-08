import Foundation
import os

/// Byte-level BPE tokenizer for the LFM2.5 `tokenizer.json` behind d1-omni-600M: no normalizer, added tokens matched
/// literally, LFM2's split regex (digits in runs of up to three), GPT-2 byte-to-unicode mapping, then merges by rank.
/// Encoding only; no special tokens are added.
public final class D1OmniTokenizer: Sendable {
    private let vocabulary: [String: Int]
    private let mergeRanks: [String: Int]
    /// Added tokens, longest first, so a longer token wins over one it contains.
    private let addedTokens: [(content: String, id: Int)]
    /// Added tokens grouped by first UTF-8 byte (longest first), so splitting costs one lookup per byte.
    private let addedTokensByFirstByte: [UInt8: [(bytes: [UInt8], id: Int)]]
    private let byteToCharacter: [Character]
    /// Encoded pieces; prompts repeat the same words, so this saves most of the merge loops.
    private let cache = OSAllocatedUnfairLock<[String: [Int]]>(initialState: [:])
    /// Whole encoded texts; game prompts repeat the same instruction and option labels every turn.
    private let textCache = OSAllocatedUnfairLock<[String: [Int]]>(initialState: [:])

    public let bosTokenId: Int

    public init(tokenizerJsonURL: URL) throws {
        let data = try Data(contentsOf: tokenizerJsonURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let model = root["model"] as? [String: Any], model["type"] as? String == "BPE",
            let vocab = model["vocab"] as? [String: Int], let merges = model["merges"] as? [Any],
            let added = root["added_tokens"] as? [[String: Any]]
        else { throw D1OmniError.invalidAsset("Expected a Hugging Face byte-level BPE tokenizer.json") }
        guard (model["byte_fallback"] as? Bool ?? false) == false else {
            throw D1OmniError.invalidAsset("byte_fallback BPE is not supported")
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
            guard pair.count == 2 else { throw D1OmniError.invalidAsset("Malformed merge \(rank)") }
            ranks[pair[0] + "\u{0}" + pair[1]] = rank
        }
        mergeRanks = ranks
        let addedList: [(content: String, id: Int)] = added.compactMap { entry in
            guard let content = entry["content"] as? String, let id = entry["id"] as? Int else { return nil }
            return (content, id)
        }
        addedTokens = addedList.sorted { $0.content.count > $1.content.count }
        addedTokensByFirstByte = Dictionary(
            grouping: addedTokens.map { (bytes: Array($0.content.utf8), id: $0.id) }.filter { !$0.bytes.isEmpty },
            by: { $0.bytes[0] }
        ).mapValues { $0.sorted { $0.bytes.count > $1.bytes.count } }
        guard let bos = addedList.first(where: { $0.content == "<|startoftext|>" })?.id else {
            throw D1OmniError.invalidAsset("Missing <|startoftext|>")
        }
        bosTokenId = bos
        byteToCharacter = Self.bytesToUnicode()
        _ = try Self.splitter()
    }

    /// LFM2's pre-tokenizer split. Built per call: NSRegularExpression is not Sendable.
    private static func splitter() throws -> NSRegularExpression {
        try NSRegularExpression(
            pattern:
                #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        )
    }

    /// Token ids for `text`, without any special tokens around it.
    public func encode(_ text: String) throws -> [Int] {
        if let cached = textCache.withLock({ $0[text] }) { return cached }
        let splitter = try Self.splitter()
        var ids: [Int] = []
        for (segment, addedID) in splitAddedTokens(text) {
            if let addedID {
                ids.append(addedID)
                continue
            }
            let string = segment as NSString
            for match in splitter.matches(in: segment, range: NSRange(location: 0, length: string.length)) {
                ids += encodePiece(string.substring(with: match.range))
            }
        }
        let encoded = ids
        textCache.withLock { storage in
            if storage.count > 10_000 { storage.removeAll(keepingCapacity: true) }
            storage[text] = encoded
        }
        return encoded
    }

    public func id(for token: String) -> Int? {
        addedTokens.first { $0.content == token }?.id ?? vocabulary[token]
    }

    /// Leftmost-longest literal added-token matches, scanned once over the UTF-8 bytes.
    private func splitAddedTokens(_ text: String) -> [(String, Int?)] {
        let bytes = Array(text.utf8)
        var parts: [(String, Int?)] = []
        var segmentStart = 0
        var index = 0
        while index < bytes.count {
            guard let candidates = addedTokensByFirstByte[bytes[index]],
                let match = candidates.first(where: { candidate in
                    index + candidate.bytes.count <= bytes.count
                        && bytes[index..<(index + candidate.bytes.count)].elementsEqual(candidate.bytes)
                })
            else {
                index += 1
                continue
            }
            if index > segmentStart {
                parts.append((String(decoding: bytes[segmentStart..<index], as: UTF8.self), nil))
            }
            parts.append(("", match.id))
            index += match.bytes.count
            segmentStart = index
        }
        if segmentStart < bytes.count {
            parts.append((String(decoding: bytes[segmentStart...], as: UTF8.self), nil))
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
