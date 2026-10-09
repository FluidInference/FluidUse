import Foundation

/// Gemma BPE tokenizer for EmbeddingGemma 2 (`tokenizer.json`): spaces become `▁`, ranked merges over Unicode
/// scalars, UTF-8 byte fallback for scalars outside the vocabulary, `<bos>` … `<eos>` around the text.
/// Merges run from a priority queue, so long posts tokenize in O(n log n). Tokens are keyed by UTF-8 bytes, not
/// `String`: Swift string equality is canonical, which would merge distinct tokens such as CJK compatibility
/// ideographs with their unified forms.
public final class EmbeddingGemma2Tokenizer: Sendable {
    public let bosId: Int32
    public let eosId: Int32
    public let padId: Int32

    private let vocab: [[UInt8]: Int32]
    /// `left << 32 | right` → merge rank and the merged token's id.
    private let merges: [UInt64: (rank: Int32, id: Int32)]
    private let byteIds: [Int32]
    private let unkId: Int32

    public convenience init(tokenizerJsonURL: URL) throws {
        try self.init(data: Data(contentsOf: tokenizerJsonURL))
    }

    public init(data: Data) throws {
        // Parsed from raw bytes: JSONSerialization drops a leading U+FEFF from keys ("\u{FEFF}#" → "#"), and
        // `String`-keyed dictionaries collapse canonical equivalents; both would remap real tokens.
        var reader = TokenizerJSONReader(bytes: [UInt8](data))
        guard let (rawVocab, rawMerges) = reader.vocabAndMerges() else {
            throw EmbeddingGemma2Error.invalidAsset("tokenizer.json is not a Gemma BPE tokenizer")
        }
        var vocab: [[UInt8]: Int32] = [:]
        vocab.reserveCapacity(rawVocab.count)
        for (token, id) in rawVocab { vocab[token] = id }
        func id(_ token: String) -> Int32? { vocab[Array(token.utf8)] }
        guard let bos = id("<bos>"), let eos = id("<eos>"), let pad = id("<pad>") else {
            throw EmbeddingGemma2Error.invalidAsset("tokenizer.json has no <bos>, <eos> or <pad>")
        }
        var merges: [UInt64: (rank: Int32, id: Int32)] = [:]
        merges.reserveCapacity(rawMerges.count)
        for (rank, pair) in rawMerges.enumerated() {
            guard let left = vocab[pair.0], let right = vocab[pair.1], let merged = vocab[pair.0 + pair.1]
            else { continue }
            merges[Self.key(left, right)] = (Int32(rank), merged)
        }
        self.vocab = vocab
        self.merges = merges
        unkId = id("<unk>") ?? 3
        byteIds = (0..<256).map { id(String(format: "<0x%02X>", $0)) ?? id("<unk>") ?? 3 }
        bosId = bos
        eosId = eos
        padId = pad
    }

    /// `<bos>` + tokens + `<eos>`, keeping the first `maxLength - 2` tokens of longer text.
    public func encode(_ text: String, maxLength: Int) -> [Int32] {
        [bosId] + tokenize(text).prefix(max(maxLength - 2, 0)) + [eosId]
    }

    /// Token ids without special tokens.
    public func tokenize(_ text: String) -> [Int32] {
        let space = vocab[Array("\u{2581}".utf8)] ?? unkId
        var ids: [Int32] = []
        for scalar in text.unicodeScalars {
            if scalar == " " {
                ids.append(space)
                continue
            }
            let bytes = Array(String(scalar).utf8)
            if let id = vocab[bytes] {
                ids.append(id)
            } else {
                for byte in bytes { ids.append(byteIds[Int(byte)]) }
            }
        }
        return merge(ids)
    }

    private func merge(_ initial: [Int32]) -> [Int32] {
        let count = initial.count
        guard count > 1 else { return initial }
        var ids = initial
        var next = Array(1...count)  // `count` = end
        var previous = Array(-1..<(count - 1))
        var alive = [Bool](repeating: true, count: count)
        var heap = MergeHeap()
        for index in 0..<(count - 1) {
            if let merge = merges[Self.key(ids[index], ids[index + 1])] {
                heap.push(.init(rank: merge.rank, position: Int32(index), left: ids[index], right: ids[index + 1]))
            }
        }
        while let candidate = heap.pop() {
            let position = Int(candidate.position)
            let right = next[position]
            guard alive[position], right < count, ids[position] == candidate.left, ids[right] == candidate.right,
                let merge = merges[Self.key(candidate.left, candidate.right)]
            else { continue }
            ids[position] = merge.id
            alive[right] = false
            next[position] = next[right]
            if next[right] < count { previous[next[right]] = position }
            let before = previous[position]
            if before >= 0, let pair = merges[Self.key(ids[before], ids[position])] {
                heap.push(.init(rank: pair.rank, position: Int32(before), left: ids[before], right: ids[position]))
            }
            let after = next[position]
            if after < count, let pair = merges[Self.key(ids[position], ids[after])] {
                heap.push(.init(rank: pair.rank, position: Int32(position), left: ids[position], right: ids[after]))
            }
        }
        var result: [Int32] = []
        result.reserveCapacity(count)
        var index = 0
        while index < count {
            result.append(ids[index])
            index = next[index]
        }
        return result
    }

    private static func key(_ left: Int32, _ right: Int32) -> UInt64 {
        UInt64(UInt32(bitPattern: left)) << 32 | UInt64(UInt32(bitPattern: right))
    }
}

/// Just enough JSON to read `model.vocab` (token → id) and `model.merges` ([left, right] pairs) with every
/// string kept as its exact UTF-8 bytes.
private struct TokenizerJSONReader {
    let bytes: [UInt8]
    var index = 0

    init(bytes: [UInt8]) { self.bytes = bytes }

    mutating func vocabAndMerges() -> (vocab: [([UInt8], Int32)], merges: [([UInt8], [UInt8])])? {
        guard seek(key: "model") else { return nil }
        let modelStart = index
        guard seek(key: "vocab"), skipSpace(), consume(UInt8(ascii: "{")) else { return nil }
        var vocab: [([UInt8], Int32)] = []
        vocab.reserveCapacity(262_144)
        while skipSpace(), !consume(UInt8(ascii: "}")) {
            _ = consume(UInt8(ascii: ","))
            _ = skipSpace()
            guard let key = string(), skipSpace(), consume(UInt8(ascii: ":")), skipSpace(), let value = integer()
            else { return nil }
            vocab.append((key, Int32(value)))
        }
        index = modelStart
        guard seek(key: "merges"), skipSpace(), consume(UInt8(ascii: "[")) else { return nil }
        var merges: [([UInt8], [UInt8])] = []
        merges.reserveCapacity(600_000)
        while skipSpace(), !consume(UInt8(ascii: "]")) {
            _ = consume(UInt8(ascii: ","))
            guard skipSpace(), consume(UInt8(ascii: "[")), skipSpace(), let left = string(), skipSpace(),
                consume(UInt8(ascii: ",")), skipSpace(), let right = string(), skipSpace(), consume(UInt8(ascii: "]"))
            else { return nil }
            merges.append((left, right))
        }
        return (vocab, merges)
    }

    /// Moves past the next `"key":` at or after the cursor.
    private mutating func seek(key: String) -> Bool {
        let pattern = Array("\"\(key)\"".utf8)
        guard pattern.count <= bytes.count else { return false }
        while index + pattern.count <= bytes.count {
            if bytes[index] == pattern[0], Array(bytes[index..<(index + pattern.count)]) == pattern {
                index += pattern.count
                if skipSpace(), consume(UInt8(ascii: ":")) { return true }
            } else {
                index += 1
            }
        }
        return false
    }

    @discardableResult
    private mutating func skipSpace() -> Bool {
        while index < bytes.count, [0x20, 0x0A, 0x0D, 0x09].contains(bytes[index]) { index += 1 }
        return index < bytes.count
    }

    private mutating func consume(_ byte: UInt8) -> Bool {
        guard index < bytes.count, bytes[index] == byte else { return false }
        index += 1
        return true
    }

    private mutating func integer() -> Int? {
        var value = 0
        var digits = 0
        while index < bytes.count, (0x30...0x39).contains(bytes[index]) {
            value = value * 10 + Int(bytes[index] - 0x30)
            index += 1
            digits += 1
        }
        return digits > 0 ? value : nil
    }

    private mutating func string() -> [UInt8]? {
        guard consume(UInt8(ascii: "\"")) else { return nil }
        var out: [UInt8] = []
        var pendingHigh: UInt32?
        while index < bytes.count {
            let byte = bytes[index]
            index += 1
            if byte == UInt8(ascii: "\"") { return out }
            guard byte == UInt8(ascii: "\\") else {
                out.append(byte)
                continue
            }
            guard index < bytes.count else { return nil }
            let escape = bytes[index]
            index += 1
            switch escape {
            case UInt8(ascii: "n"): out.append(0x0A)
            case UInt8(ascii: "t"): out.append(0x09)
            case UInt8(ascii: "r"): out.append(0x0D)
            case UInt8(ascii: "b"): out.append(0x08)
            case UInt8(ascii: "f"): out.append(0x0C)
            case UInt8(ascii: "u"):
                guard index + 4 <= bytes.count,
                    let unit = UInt32(String(decoding: bytes[index..<(index + 4)], as: UTF8.self), radix: 16)
                else { return nil }
                index += 4
                if (0xD800..<0xDC00).contains(unit) {
                    pendingHigh = unit
                    continue
                }
                var scalarValue = unit
                if let high = pendingHigh, (0xDC00..<0xE000).contains(unit) {
                    scalarValue = 0x10000 + ((high - 0xD800) << 10) + (unit - 0xDC00)
                }
                pendingHigh = nil
                guard let scalar = Unicode.Scalar(scalarValue) else { return nil }
                out += Array(String(Character(scalar)).utf8)
            default: out.append(escape)  // \" \\ \/
            }
        }
        return nil
    }
}

/// Min-heap of merge candidates ordered by rank, then by position (leftmost first, as the reference does).
private struct MergeHeap {
    struct Entry {
        let rank: Int32
        let position: Int32
        let left: Int32
        let right: Int32

        func precedes(_ other: Entry) -> Bool {
            rank != other.rank ? rank < other.rank : position < other.position
        }
    }

    private var entries: [Entry] = []

    mutating func push(_ entry: Entry) {
        entries.append(entry)
        var child = entries.count - 1
        while child > 0 {
            let parent = (child - 1) / 2
            guard entries[child].precedes(entries[parent]) else { break }
            entries.swapAt(child, parent)
            child = parent
        }
    }

    mutating func pop() -> Entry? {
        guard let first = entries.first else { return nil }
        let last = entries.removeLast()
        if !entries.isEmpty {
            entries[0] = last
            var parent = 0
            while true {
                let left = 2 * parent + 1
                let right = left + 1
                var best = parent
                if left < entries.count, entries[left].precedes(entries[best]) { best = left }
                if right < entries.count, entries[right].precedes(entries[best]) { best = right }
                guard best != parent else { break }
                entries.swapAt(parent, best)
                parent = best
            }
        }
        return first
    }
}
