import Foundation
import os

/// The byte-level BPE tokenizer of vela3-fast (UltraBERT's GPT-2 style `tokenizer.json`), with Hugging Face `tokenizers`
/// offsets: special tokens matched literally (`lstrip` / `rstrip` ones absorb the whitespace next to them, offsets
/// included), the GPT-2 split regex, merges by rank (leftmost first on ties, O(n log n)), and each
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
    private let addedTokens: [(scalars: [Unicode.Scalar], id: Int, lstrip: Bool, rstrip: Bool)]
    private let byteToCharacter: [String]
    private let regex: NSRegularExpression
    /// Encoded pieces (token ids with their byte ranges); ordinary text repeats the same words. Long pieces are not
    /// cached.
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
        let added = (root["added_tokens"] as? [[String: Any]] ?? []).compactMap { t -> ([Unicode.Scalar], Int, Bool, Bool)? in
            guard let content = t["content"] as? String, let id = t["id"] as? Int, !content.isEmpty else { return nil }
            return (Array(content.unicodeScalars), id, t["lstrip"] as? Bool ?? false, t["rstrip"] as? Bool ?? false)
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
                var start = i
                var end = i + match.scalars.count
                if match.lstrip { while start > segmentStart, scalars[start - 1].properties.isWhitespace { start -= 1 } }
                if match.rstrip { while end < scalars.count, scalars[end].properties.isWhitespace { end += 1 } }
                if segmentStart < start { out += textTokens(scalars, segmentStart..<start) }
                out.append(Token(id: match.id, start: start, end: end))
                i = end
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

    /// BPE over the byte characters of one piece -> (token id, first byte, end byte). Merges come off a min-heap of
    /// (rank, position) over a linked list of symbols, as Hugging Face's `Word::merge_all`.
    private func encodePiece(_ bytes: [UInt8]) -> [(Int, Int, Int)] {
        let cacheable = bytes.count <= 256
        let key = cacheable ? bytes.map { byteToCharacter[Int($0)] }.joined() : ""
        if cacheable, let hit = cache.withLock({ $0[key] }) { return hit }
        let n = bytes.count
        var text = bytes.map { byteToCharacter[Int($0)] }
        var end = (0..<n).map { $0 + 1 }
        var next = (0..<n).map { $0 + 1 < n ? $0 + 1 : -1 }
        var prev = (0..<n).map { $0 - 1 }
        var alive = [Bool](repeating: true, count: n)
        var version = [Int](repeating: 0, count: n)
        var heap = MergeHeap()
        func push(_ left: Int) {
            let right = next[left]
            guard right >= 0, let rank = mergeRanks[text[left] + " " + text[right]] else { return }
            heap.push(.init(rank: rank, left: left, leftVersion: version[left], right: right, rightVersion: version[right]))
        }
        for k in 0..<max(n - 1, 0) { push(k) }
        while let m = heap.pop() {
            guard alive[m.left], alive[m.right], next[m.left] == m.right, version[m.left] == m.leftVersion,
                version[m.right] == m.rightVersion
            else { continue }
            text[m.left] += text[m.right]
            end[m.left] = end[m.right]
            alive[m.right] = false
            next[m.left] = next[m.right]
            if next[m.left] >= 0 { prev[next[m.left]] = m.left }
            version[m.left] += 1
            if prev[m.left] >= 0 { push(prev[m.left]) }
            push(m.left)
        }
        var out: [(Int, Int, Int)] = []
        var k = n > 0 ? 0 : -1
        var start = 0
        while k >= 0 {
            out.append((vocabulary[text[k]] ?? 0, start, end[k]))
            start = end[k]
            k = next[k]
        }
        if cacheable {
            let result = out
            cache.withLock { c in
                if c.count > 50_000 { c.removeAll() }
                c[key] = result
            }
        }
        return out
    }

    private struct MergeHeap {
        struct Item {
            let rank: Int
            let left: Int
            let leftVersion: Int
            let right: Int
            let rightVersion: Int
            func before(_ o: Item) -> Bool { rank != o.rank ? rank < o.rank : left < o.left }
        }
        private var items: [Item] = []

        mutating func push(_ x: Item) {
            items.append(x)
            var i = items.count - 1
            while i > 0 {
                let p = (i - 1) / 2
                guard items[i].before(items[p]) else { break }
                items.swapAt(i, p)
                i = p
            }
        }

        mutating func pop() -> Item? {
            guard let top = items.first else { return nil }
            let last = items.removeLast()
            if !items.isEmpty {
                items[0] = last
                var i = 0
                while true {
                    let l = 2 * i + 1, r = l + 1
                    var m = i
                    if l < items.count, items[l].before(items[m]) { m = l }
                    if r < items.count, items[r].before(items[m]) { m = r }
                    if m == i { break }
                    items.swapAt(i, m)
                    i = m
                }
            }
            return top
        }
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
