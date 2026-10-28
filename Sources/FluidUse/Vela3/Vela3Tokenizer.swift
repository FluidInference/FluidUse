import Foundation
import os

/// The byte-level BPE tokenizer of vela3-fast (UltraBERT's GPT-2 style `tokenizer.json`), with Hugging Face `tokenizers`
/// offsets: special tokens matched literally, the GPT-2 split regex, merges by rank (leftmost first on ties), and each
/// token's offsets running from the character of its first byte to the character of its last byte (so pieces of one
/// multi-byte character share its offsets). Offsets are Unicode scalar indices, as Python string indices. No offset
/// trimming: this tokenizer's post-processor is `TemplateProcessing`, so the leading space stays in a token's span.
public final class Vela3Tokenizer: Sendable {
    public struct Token: Sendable, Equatable {
        public let id: Int
        public let start: Int
        public let end: Int
    }

    private static let pattern = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#

    private let vocabulary: [String: Int]
    private let mergeRanks: [String: Int]
    /// Added tokens, longest first, so a longer token wins over one it contains.
    private let addedTokens: [(scalars: [Unicode.Scalar], id: Int)]
    private let byteToCharacter: [String]
    private let regex: NSRegularExpression
    /// Encoded pieces (token ids with their byte ranges); ordinary text repeats the same words.
    private let cache = OSAllocatedUnfairLock<[String: [(Int, Int, Int)]]>(initialState: [:])

    public init(tokenizerJsonURL: URL) throws {
        let data = try Data(contentsOf: tokenizerJsonURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let model = root["model"] as? [String: Any], model["type"] as? String == "BPE",
            let vocab = model["vocab"] as? [String: Int], let merges = model["merges"] as? [Any]
        else { throw Vela3Error.invalidAsset("tokenizer.json is not a byte-level BPE tokenizer") }
        vocabulary = vocab
        var ranks: [String: Int] = [:]
        ranks.reserveCapacity(merges.count)
        for (rank, merge) in merges.enumerated() {
            if let pair = merge as? [String], pair.count == 2 {
                ranks[pair[0] + " " + pair[1]] = rank
            } else if let text = merge as? String {
                ranks[text] = rank
            } else {
                throw Vela3Error.invalidAsset("tokenizer.json merge \(rank) is malformed")
            }
        }
        mergeRanks = ranks
        let added = (root["added_tokens"] as? [[String: Any]] ?? []).compactMap { t -> ([Unicode.Scalar], Int)? in
            guard let content = t["content"] as? String, let id = t["id"] as? Int, !content.isEmpty else { return nil }
            return (Array(content.unicodeScalars), id)
        }
        addedTokens = added.sorted { $0.0.count > $1.0.count }
        byteToCharacter = Self.bytesToUnicode()
        regex = try NSRegularExpression(pattern: Self.pattern)
    }

    public func encode(_ text: String) -> [Int] { tokens(Array(text.unicodeScalars)).map(\.id) }

    /// Tokens of a text with Unicode-scalar offsets into it.
    public func tokens(_ scalars: [Unicode.Scalar]) -> [Token] {
        var out: [Token] = []
        var segmentStart = 0
        var i = 0
        while i < scalars.count {
            if let match = addedTokens.first(where: { matches($0.scalars, scalars, at: i) }) {
                if segmentStart < i { out += textTokens(scalars, segmentStart..<i) }
                out.append(Token(id: match.id, start: i, end: i + match.scalars.count))
                i += match.scalars.count
                segmentStart = i
            } else {
                i += 1
            }
        }
        if segmentStart < scalars.count { out += textTokens(scalars, segmentStart..<scalars.count) }
        return out
    }

    private func matches(_ token: [Unicode.Scalar], _ s: [Unicode.Scalar], at i: Int) -> Bool {
        guard i + token.count <= s.count, s[i] == token[0] else { return false }
        for k in 1..<token.count where s[i + k] != token[k] { return false }
        return true
    }

    /// Split one stretch of ordinary text with the GPT-2 regex and BPE-encode each piece.
    private func textTokens(_ scalars: [Unicode.Scalar], _ range: Range<Int>) -> [Token] {
        let segment = Array(scalars[range])
        var view = String.UnicodeScalarView()
        view.append(contentsOf: segment)
        let string = String(view)
        // UTF-16 offset -> scalar index (within the segment)
        var scalarAt: [Int] = []
        scalarAt.reserveCapacity(string.utf16.count + 1)
        for (k, s) in segment.enumerated() { for _ in 0..<(s.value > 0xFFFF ? 2 : 1) { scalarAt.append(k) } }
        scalarAt.append(segment.count)
        var pieces: [Range<Int>] = []
        var last = 0
        for m in regex.matches(in: string, range: NSRange(location: 0, length: string.utf16.count)) {
            let a = scalarAt[m.range.location], b = scalarAt[m.range.location + m.range.length]
            if a > last { pieces.append(last..<a) }  // nothing the regex skipped is dropped
            if b > a { pieces.append(a..<b) }
            last = b
        }
        if last < segment.count { pieces.append(last..<segment.count) }
        var out: [Token] = []
        for piece in pieces {
            // the bytes of the piece and the scalar each byte belongs to
            var bytes: [UInt8] = []
            var owner: [Int] = []
            for k in piece {
                for b in String(segment[k]).utf8 {
                    bytes.append(b)
                    owner.append(range.lowerBound + k)
                }
            }
            for (id, b0, b1) in encodePiece(bytes) {
                out.append(Token(id: id, start: owner[b0], end: owner[b1 - 1] + 1))
            }
        }
        return out
    }

    /// BPE over the byte characters of one piece -> (token id, first byte, end byte).
    private func encodePiece(_ bytes: [UInt8]) -> [(Int, Int, Int)] {
        let key = bytes.map { byteToCharacter[Int($0)] }.joined()
        if let hit = cache.withLock({ $0[key] }) { return hit }
        var symbols: [(text: String, start: Int, end: Int)] = bytes.enumerated().map { (byteToCharacter[Int($0.element)], $0.offset, $0.offset + 1) }
        while symbols.count > 1 {
            var best = Int.max
            var at = -1
            for k in 0..<(symbols.count - 1) {
                if let r = mergeRanks[symbols[k].text + " " + symbols[k + 1].text], r < best {
                    best = r
                    at = k
                }
            }
            if at < 0 { break }
            symbols[at] = (symbols[at].text + symbols[at + 1].text, symbols[at].start, symbols[at + 1].end)
            symbols.remove(at: at + 1)
        }
        let out = symbols.map { (vocabulary[$0.text] ?? 0, $0.start, $0.end) }
        cache.withLock { c in
            if c.count > 50_000 { c.removeAll() }
            c[key] = out
        }
        return out
    }

    /// GPT-2's reversible byte -> printable character table.
    private static func bytesToUnicode() -> [String] {
        var bs: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs
        var n = 0
        for b in 0..<256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var table = [String](repeating: "", count: 256)
        for (b, c) in zip(bs, cs) { table[b] = String(Character(Unicode.Scalar(UInt32(c))!)) }
        return table
    }
}
