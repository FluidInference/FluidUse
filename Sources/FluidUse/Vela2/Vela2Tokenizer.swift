import Foundation
import os

/// The Vela 2.0 0.3B text tokenizer (the mmBERT / Gemma byte-fallback BPE of `LayaTokenizer`) with character offsets.
///
/// Reproduces HuggingFace `tokenizers` without a template: added tokens (newline runs, …) are matched first; every other
/// segment has spaces replaced by `▁`, a zero-width `▁` prepended and is split into `▁`-led pieces that are BPE-merged
/// with byte fallback. Offsets are Unicode scalar (code point) indices into the input, as Python reports them; a token
/// spans its first to last source character (the prepended `▁` has no width).
public final class Vela2Tokenizer: Sendable {
    struct Token {
        let id: Int
        let start: Int
        let end: Int
    }

    private typealias Key = LayaTokenizer.ScalarKey

    private let vocab: [Key: Int]
    private let mergeRank: [Key: Int]
    private let byteTokens: [Int?]
    private let addedByFirstScalar: [UInt32: [Added]]
    private let unknownId: Int
    private let cache = Cache()

    private struct Added: Sendable {
        let scalars: [UInt32]
        let id: Int
        let lstrip: Bool
        let rstrip: Bool
    }

    /// One normalized character: its text and the source span it came from (`start == end` for the prepended `▁`).
    private struct Char {
        let scalar: Unicode.Scalar
        let start: Int
        let end: Int
    }

    private static let space: Unicode.Scalar = "\u{2581}"

    public init(tokenizerJsonURL: URL) throws {
        let data = try Data(contentsOf: tokenizerJsonURL)
        let table = try LayaTokenizerFile.scan(data)
        guard let root = try JSONSerialization.jsonObject(with: data) as? NSDictionary,
            let model = root["model"] as? NSDictionary, model["type"] as? String == "BPE",
            model["byte_fallback"] as? Bool == true
        else { throw Vela2Error.invalidAsset("tokenizer.json must describe a byte-fallback BPE model") }
        var vocab = [Key: Int](minimumCapacity: table.vocab.count)
        for (token, id) in table.vocab { vocab[Key(token)] = id }
        self.vocab = vocab
        var ranks = [Key: Int](minimumCapacity: table.merges.count)
        for (rank, pair) in table.merges.enumerated() { ranks[Key("\(pair.0) \(pair.1)")] = rank }
        self.mergeRank = ranks
        self.byteTokens = (0..<256).map { vocab[Key(String(format: "<0x%02X>", $0))] }
        var byFirst: [UInt32: [Added]] = [:]
        for entry in root["added_tokens"] as? [[String: Any]] ?? [] {
            guard let content = entry["content"] as? String, let id = entry["id"] as? Int, !content.isEmpty else { continue }
            let scalars = content.unicodeScalars.map(\.value)
            byFirst[scalars[0], default: []].append(
                Added(scalars: scalars, id: id, lstrip: entry["lstrip"] as? Bool ?? false, rstrip: entry["rstrip"] as? Bool ?? false))
        }
        for key in byFirst.keys { byFirst[key]?.sort { $0.scalars.count > $1.scalars.count } }
        self.addedByFirstScalar = byFirst
        guard let unk = vocab[Key("<unk>")] else { throw Vela2Error.invalidAsset("tokenizer.json has no <unk>") }
        self.unknownId = unk
    }

    public func encode(_ text: String) -> [Int] { tokens(Array(text.unicodeScalars)).map(\.id) }

    func tokens(_ scalars: [Unicode.Scalar]) -> [Token] {
        var out: [Token] = []
        var segmentStart = 0
        var index = 0
        while index < scalars.count {
            if let candidates = addedByFirstScalar[scalars[index].value],
                let token = candidates.first(where: { matches($0, scalars, index) })
            {
                var end = index
                if token.lstrip { while end > segmentStart, scalars[end - 1].properties.isWhitespace { end -= 1 } }
                encodeSegment(scalars, segmentStart, end, into: &out)
                out.append(Token(id: token.id, start: index, end: index + token.scalars.count))
                index += token.scalars.count
                if token.rstrip { while index < scalars.count, scalars[index].properties.isWhitespace { index += 1 } }
                segmentStart = index
                continue
            }
            index += 1
        }
        encodeSegment(scalars, segmentStart, scalars.count, into: &out)
        return out
    }

    private func matches(_ token: Added, _ scalars: [Unicode.Scalar], _ index: Int) -> Bool {
        guard index + token.scalars.count <= scalars.count else { return false }
        for (k, value) in token.scalars.enumerated() where scalars[index + k].value != value { return false }
        return true
    }

    private func encodeSegment(_ scalars: [Unicode.Scalar], _ start: Int, _ end: Int, into out: inout [Token]) {
        guard start < end else { return }
        var chars: [Char] = []
        chars.reserveCapacity(end - start + 1)
        for i in start..<end { chars.append(Char(scalar: scalars[i] == " " ? Self.space : scalars[i], start: i, end: i + 1)) }
        if chars[0].scalar != Self.space { chars.insert(Char(scalar: Self.space, start: start, end: start), at: 0) }
        var piece: [Char] = []
        for char in chars {
            if char.scalar == Self.space, !piece.isEmpty {
                encodePiece(piece, into: &out)
                piece.removeAll(keepingCapacity: true)
            }
            piece.append(char)
        }
        if !piece.isEmpty { encodePiece(piece, into: &out) }
    }

    /// BPE over one piece; each output token covers the source span of the characters it merged.
    private func encodePiece(_ piece: [Char], into out: inout [Token]) {
        let key = piece.map(\.scalar.value)
        let merged: [(id: Int, first: Int, last: Int)]  // character index range within the piece
        if let cached = cache.lookup(key) {
            merged = cached
        } else {
            merged = bpe(piece.map(\.scalar))
            cache.store(key, merged)
        }
        for m in merged {
            var s = Int.max
            var e = Int.min
            for c in piece[m.first...m.last] where c.end > c.start {
                s = min(s, c.start)
                e = max(e, c.end)
            }
            if s == Int.max { (s, e) = (piece[m.first].start, piece[m.first].start) }
            out.append(Token(id: m.id, start: s, end: e))
        }
    }

    private func bpe(_ scalars: [Unicode.Scalar]) -> [(id: Int, first: Int, last: Int)] {
        // symbols with the character index range they cover
        var symbols: [(text: String, first: Int, last: Int)] = []
        var unknownRun = false
        for (i, scalar) in scalars.enumerated() {
            let symbol = String(Character(scalar))
            if vocab[Key(symbol)] != nil {
                symbols.append((symbol, i, i))
                unknownRun = false
                continue
            }
            let bytes = Array(symbol.utf8)
            if bytes.allSatisfy({ byteTokens[Int($0)] != nil }) {
                for byte in bytes { symbols.append((String(format: "<0x%02X>", byte), i, i)) }
                unknownRun = false
            } else if !unknownRun {
                symbols.append(("<unk>", i, i))
                unknownRun = true
            } else {
                symbols[symbols.count - 1].last = i
            }
        }
        while symbols.count > 1 {
            var bestRank = Int.max
            var best = -1
            for k in 0..<(symbols.count - 1) {
                if let rank = mergeRank[Key("\(symbols[k].text) \(symbols[k + 1].text)")], rank < bestRank {
                    bestRank = rank
                    best = k
                }
            }
            if best < 0 { break }
            symbols[best] = (symbols[best].text + symbols[best + 1].text, symbols[best].first, symbols[best + 1].last)
            symbols.remove(at: best + 1)
        }
        return symbols.map { (vocab[Key($0.text)] ?? unknownId, $0.first, $0.last) }
    }

    private struct Cache: Sendable {
        private let entries = OSAllocatedUnfairLock<[[UInt32]: [(id: Int, first: Int, last: Int)]]>(initialState: [:])

        func lookup(_ key: [UInt32]) -> [(id: Int, first: Int, last: Int)]? { entries.withLock { $0[key] } }

        func store(_ key: [UInt32], _ value: [(id: Int, first: Int, last: Int)]) {
            entries.withLock { table in
                if table.count >= 8192 { table.removeAll(keepingCapacity: true) }
                table[key] = value
            }
        }
    }
}
