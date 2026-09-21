import Foundation
import os

/// SentencePiece-style BPE encoder for the mmBERT (Gemma vocabulary) tokenizer behind laya-multilingual.
///
/// Loads a HuggingFace `tokenizer.json` (256k vocabulary, 580k merges, byte fallback, added tokens)
/// and reproduces `tokenizers` encoding without special-token templates, which is how laya calls it.
/// Encoding only: laya never decodes.
public final class LayaTokenizer: Sendable {
    /// `<mask>` — one marker precedes every option in a laya sequence.
    public let maskTokenId: Int
    /// `<bos>` — laya's `[CLS]`.
    public let clsTokenId: Int
    /// `<eos>` — laya's `[SEP]`.
    public let sepTokenId: Int
    /// `<pad>` — fills the fixed-length model input.
    public let padTokenId: Int
    /// The literal mask token text, which laya strips from prompts so it can never appear inside them.
    public let maskToken: String

    /// Dictionary key that compares code points literally. Swift `String` equality applies Unicode
    /// canonical equivalence, which would merge distinct vocabulary entries such as U+4E86 and U+F9BA.
    struct ScalarKey: Hashable {
        let string: String

        init(_ string: String) { self.string = string }

        static func == (lhs: ScalarKey, rhs: ScalarKey) -> Bool {
            lhs.string.utf8.elementsEqual(rhs.string.utf8)
        }

        func hash(into hasher: inout Hasher) {
            var copy = string
            copy.withUTF8 { hasher.combine(bytes: UnsafeRawBufferPointer($0)) }
        }
    }

    private let vocab: [ScalarKey: Int]
    private let mergeRank: [ScalarKey: Int]
    private let byteTokens: [Int?]
    private let addedTokens: [ScalarKey: Int]
    /// Added tokens grouped by first scalar, longest first, so matching costs one lookup per position.
    private let addedTokensByFirstScalar: [UInt32: [AddedToken]]
    private let unknownTokenId: Int

    private struct AddedToken: Sendable {
        let scalars: [UInt32]
        let id: Int
        let lstrip: Bool
        let rstrip: Bool
    }
    /// Piece → ids memo. Prompts reuse the same words constantly; BPE merging is the dominant host cost.
    private let pieceCache = PieceCache()

    private static let spaceMarker: Unicode.Scalar = "\u{2581}"

    public init(tokenizerJsonURL: URL) throws {
        let data = try Data(contentsOf: tokenizerJsonURL)
        // Foundation's JSON parsers cannot load this vocabulary faithfully: JSONSerialization strips a
        // leading U+FEFF from string values (8 vocabulary keys start with a BOM) and Swift dictionaries
        // merge canonically equivalent keys (U+4E86 vs U+F9BA). The vocabulary and merges are therefore
        // read with a byte-level scanner; the small remaining fields go through JSONSerialization.
        let table = try LayaTokenizerFile.scan(data)
        guard let root = try JSONSerialization.jsonObject(with: data) as? NSDictionary,
            let model = root["model"] as? NSDictionary
        else {
            throw LayaError.invalidAsset("tokenizer.json is missing model")
        }
        guard model["type"] as? String == "BPE", model["byte_fallback"] as? Bool == true else {
            throw LayaError.invalidAsset("tokenizer.json must describe a byte-fallback BPE model")
        }

        var vocab = [ScalarKey: Int](minimumCapacity: table.vocab.count)
        for (token, id) in table.vocab {
            vocab[ScalarKey(token)] = id
        }
        guard vocab.count == table.vocab.count else {
            throw LayaError.invalidAsset(
                "tokenizer.json vocabulary has \(table.vocab.count - vocab.count) colliding keys")
        }
        self.vocab = vocab

        var mergeRank = [ScalarKey: Int](minimumCapacity: table.merges.count)
        for (rank, pair) in table.merges.enumerated() {
            mergeRank[ScalarKey("\(pair.0) \(pair.1)")] = rank
        }
        self.mergeRank = mergeRank

        var byteTokens = [Int?](repeating: nil, count: 256)
        for value in 0..<256 {
            byteTokens[value] = vocab[ScalarKey(String(format: "<0x%02X>", value))]
        }
        self.byteTokens = byteTokens

        var added = [ScalarKey: Int]()
        var byFirst: [UInt32: [AddedToken]] = [:]
        if let addedList = root["added_tokens"] as? [[String: Any]] {
            for entry in addedList {
                guard let content = entry["content"] as? String, let id = entry["id"] as? Int, !content.isEmpty
                else { continue }
                added[ScalarKey(content)] = id
                let scalars = content.unicodeScalars.map(\.value)
                let token = AddedToken(
                    scalars: scalars, id: id, lstrip: entry["lstrip"] as? Bool ?? false,
                    rstrip: entry["rstrip"] as? Bool ?? false)
                byFirst[scalars[0], default: []].append(token)
            }
        }
        // Longest content first so overlapping added tokens resolve like the Rust matcher.
        for key in byFirst.keys {
            byFirst[key]?.sort { $0.scalars.count > $1.scalars.count }
        }
        self.addedTokens = added
        self.addedTokensByFirstScalar = byFirst

        func requiredToken(_ content: String) throws -> Int {
            guard let id = added[ScalarKey(content)] ?? vocab[ScalarKey(content)] else {
                throw LayaError.invalidAsset("tokenizer.json has no \(content) token")
            }
            return id
        }
        self.maskToken = "<mask>"
        self.maskTokenId = try requiredToken(maskToken)
        self.clsTokenId = try requiredToken("<bos>")
        self.sepTokenId = try requiredToken("<eos>")
        self.padTokenId = try requiredToken("<pad>")
        self.unknownTokenId = try requiredToken("<unk>")
    }

    /// Encode text to token ids without any special-token template.
    ///
    /// Added tokens (`<mask>`, `<start_of_turn>`, newline runs, …) are matched first, longest match
    /// wins and `lstrip`/`rstrip` swallow adjacent whitespace like the Rust matcher; each remaining
    /// segment gets spaces replaced by `▁`, a leading `▁` prepended, and is split into `▁`-prefixed
    /// pieces that are BPE-merged with byte fallback for scalars outside the vocabulary.
    public func encode(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }
        let scalars = Array(text.unicodeScalars)
        var ids: [Int] = []
        var segment: [Unicode.Scalar] = []
        var index = 0
        while index < scalars.count {
            if let candidates = addedTokensByFirstScalar[scalars[index].value],
                let token = candidates.first(where: { matches($0, in: scalars, at: index) })
            {
                if token.lstrip {
                    while let last = segment.last, last.properties.isWhitespace { segment.removeLast() }
                }
                encodeSegment(segment, into: &ids)
                segment.removeAll(keepingCapacity: true)
                ids.append(token.id)
                index += token.scalars.count
                if token.rstrip {
                    while index < scalars.count, scalars[index].properties.isWhitespace { index += 1 }
                }
                continue
            }
            segment.append(scalars[index])
            index += 1
        }
        encodeSegment(segment, into: &ids)
        return ids
    }

    private func matches(_ token: AddedToken, in scalars: [Unicode.Scalar], at index: Int) -> Bool {
        guard index + token.scalars.count <= scalars.count else { return false }
        for (offset, value) in token.scalars.enumerated() where scalars[index + offset].value != value {
            return false
        }
        return true
    }

    // MARK: - Metaspace + BPE

    private func encodeSegment(_ segment: [Unicode.Scalar], into ids: inout [Int]) {
        guard !segment.isEmpty else { return }
        var scalars: [Unicode.Scalar] = []
        scalars.reserveCapacity(segment.count + 1)
        for scalar in segment {
            scalars.append(scalar == " " ? Self.spaceMarker : scalar)
        }
        if scalars.first != Self.spaceMarker {
            scalars.insert(Self.spaceMarker, at: 0)
        }
        var piece: [Unicode.Scalar] = []
        for scalar in scalars {
            if scalar == Self.spaceMarker, !piece.isEmpty {
                ids.append(contentsOf: encodePiece(piece))
                piece.removeAll(keepingCapacity: true)
            }
            piece.append(scalar)
        }
        if !piece.isEmpty {
            ids.append(contentsOf: encodePiece(piece))
        }
    }

    private func encodePiece(_ scalars: [Unicode.Scalar]) -> [Int] {
        let key = scalars.map(\.value)
        if let cached = pieceCache.lookup(key) { return cached }
        let ids = bpe(scalars)
        pieceCache.store(key, ids)
        return ids
    }

    /// Bounded memo behind an unfair lock; cleared wholesale when full.
    private struct PieceCache: Sendable {
        private static let capacity = 8192
        private let entries = OSAllocatedUnfairLock<[[UInt32]: [Int]]>(initialState: [:])

        func lookup(_ key: [UInt32]) -> [Int]? {
            entries.withLock { $0[key] }
        }

        func store(_ key: [UInt32], _ ids: [Int]) {
            entries.withLock { table in
                if table.count >= Self.capacity { table.removeAll(keepingCapacity: true) }
                table[key] = ids
            }
        }
    }

    /// Byte-level fallback happens per scalar before merging, exactly like the Rust BPE model.
    private func bpe(_ scalars: [Unicode.Scalar]) -> [Int] {
        var symbols: [String] = []
        symbols.reserveCapacity(scalars.count)
        var unknownRun = false
        for scalar in scalars {
            let symbol = String(Character(scalar))
            if vocab[ScalarKey(symbol)] != nil {
                symbols.append(symbol)
                unknownRun = false
                continue
            }
            let bytes = Array(symbol.utf8)
            if bytes.allSatisfy({ byteTokens[Int($0)] != nil }) {
                for byte in bytes {
                    symbols.append(String(format: "<0x%02X>", byte))
                }
                unknownRun = false
            } else if !unknownRun {
                // fuse_unk: consecutive unknown scalars collapse into one <unk>.
                symbols.append("<unk>")
                unknownRun = true
            }
        }
        while symbols.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for index in 0..<(symbols.count - 1) {
                if let rank = mergeRank[ScalarKey("\(symbols[index]) \(symbols[index + 1])")], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }
            if bestIndex < 0 { break }
            symbols[bestIndex] += symbols[bestIndex + 1]
            symbols.remove(at: bestIndex + 1)
        }
        return symbols.map { vocab[ScalarKey($0)] ?? unknownTokenId }
    }
}

/// Byte-level reader for the two large fields of a HuggingFace `tokenizer.json`: `model.vocab`
/// (string → int) and `model.merges` (`[a, b]` pairs or `"a b"` strings). Strings are decoded
/// scalar by scalar so a leading U+FEFF or a compatibility ideograph survives exactly as written.
enum LayaTokenizerFile {
    struct Table {
        var vocab: [(String, Int)] = []
        var merges: [(String, String)] = []
    }

    static func scan(_ data: Data) throws -> Table {
        var table = Table()
        try data.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
            let bytes = buffer.bindMemory(to: UInt8.self)
            var scanner = Scanner(bytes: bytes)
            guard let vocabStart = scanner.find(key: "\"vocab\"") else {
                throw LayaError.invalidAsset("tokenizer.json is missing model.vocab")
            }
            scanner.index = vocabStart
            scanner.skipWhitespace()
            try scanner.expect(UInt8(ascii: "{"))
            while true {
                scanner.skipWhitespace()
                if scanner.peek == UInt8(ascii: "}") {
                    scanner.index += 1
                    break
                }
                let key = try scanner.string()
                scanner.skipWhitespace()
                try scanner.expect(UInt8(ascii: ":"))
                scanner.skipWhitespace()
                table.vocab.append((key, try scanner.integer()))
                scanner.skipWhitespace()
                if scanner.peek == UInt8(ascii: ",") { scanner.index += 1 }
            }
            guard let mergesStart = scanner.find(key: "\"merges\"") else {
                throw LayaError.invalidAsset("tokenizer.json is missing model.merges")
            }
            scanner.index = mergesStart
            scanner.skipWhitespace()
            try scanner.expect(UInt8(ascii: "["))
            while true {
                scanner.skipWhitespace()
                if scanner.peek == UInt8(ascii: "]") { break }
                if scanner.peek == UInt8(ascii: "[") {
                    scanner.index += 1
                    scanner.skipWhitespace()
                    let left = try scanner.string()
                    scanner.skipWhitespace()
                    try scanner.expect(UInt8(ascii: ","))
                    scanner.skipWhitespace()
                    let right = try scanner.string()
                    scanner.skipWhitespace()
                    try scanner.expect(UInt8(ascii: "]"))
                    table.merges.append((left, right))
                } else {
                    let text = try scanner.string()
                    guard let space = text.firstIndex(of: " ") else {
                        throw LayaError.invalidAsset("tokenizer.json merge entry without a separator")
                    }
                    table.merges.append((String(text[..<space]), String(text[text.index(after: space)...])))
                }
                scanner.skipWhitespace()
                if scanner.peek == UInt8(ascii: ",") { scanner.index += 1 }
            }
        }
        return table
    }

    /// Minimal JSON lexer over UTF-8 bytes; only what the two fields need.
    private struct Scanner {
        let bytes: UnsafeBufferPointer<UInt8>
        var index = 0

        var peek: UInt8? { index < bytes.count ? bytes[index] : nil }

        mutating func skipWhitespace() {
            while let byte = peek, byte == 0x20 || byte == 0x0A || byte == 0x0D || byte == 0x09 { index += 1 }
        }

        mutating func expect(_ byte: UInt8) throws {
            guard peek == byte else { throw LayaError.invalidAsset("tokenizer.json: unexpected byte at \(index)") }
            index += 1
        }

        /// Position just after the first `"key":` at the current nesting or deeper; keys are ASCII.
        mutating func find(key: String) -> Int? {
            let pattern = Array(key.utf8)
            var cursor = index
            while cursor + pattern.count < bytes.count {
                if bytes[cursor] == pattern[0], (0..<pattern.count).allSatisfy({ bytes[cursor + $0] == pattern[$0] }) {
                    var after = cursor + pattern.count
                    while after < bytes.count, bytes[after] == 0x20 || bytes[after] == 0x0A { after += 1 }
                    if after < bytes.count, bytes[after] == UInt8(ascii: ":") { return after + 1 }
                }
                cursor += 1
            }
            return nil
        }

        mutating func integer() throws -> Int {
            var value = 0
            var digits = 0
            while let byte = peek, byte >= 0x30, byte <= 0x39 {
                value = value * 10 + Int(byte - 0x30)
                digits += 1
                index += 1
            }
            guard digits > 0 else { throw LayaError.invalidAsset("tokenizer.json: expected an integer at \(index)") }
            return value
        }

        mutating func string() throws -> String {
            try expect(UInt8(ascii: "\""))
            var scalars = String.UnicodeScalarView()
            var utf8 = [UInt8]()
            func flushUTF8() throws {
                guard !utf8.isEmpty else { return }
                // Not `String(bytes:encoding:)`: Foundation drops a leading U+FEFF while decoding.
                var iterator = utf8.makeIterator()
                var decoder = UTF8()
                loop: while true {
                    switch decoder.decode(&iterator) {
                    case .scalarValue(let scalar): scalars.append(scalar)
                    case .emptyInput: break loop
                    case .error: throw LayaError.invalidAsset("tokenizer.json: invalid UTF-8 at \(index)")
                    }
                }
                utf8.removeAll(keepingCapacity: true)
            }
            while true {
                guard let byte = peek else { throw LayaError.invalidAsset("tokenizer.json: unterminated string") }
                index += 1
                if byte == UInt8(ascii: "\"") { break }
                if byte != UInt8(ascii: "\\") {
                    utf8.append(byte)
                    continue
                }
                try flushUTF8()
                guard let escape = peek else { throw LayaError.invalidAsset("tokenizer.json: bad escape") }
                index += 1
                switch escape {
                case UInt8(ascii: "n"): scalars.append("\n")
                case UInt8(ascii: "r"): scalars.append("\r")
                case UInt8(ascii: "t"): scalars.append("\t")
                case UInt8(ascii: "b"): scalars.append("\u{08}")
                case UInt8(ascii: "f"): scalars.append("\u{0C}")
                case UInt8(ascii: "u"):
                    var value = try hex4()
                    if value >= 0xD800, value <= 0xDBFF, peek == UInt8(ascii: "\\") {
                        index += 1
                        try expect(UInt8(ascii: "u"))
                        let low = try hex4()
                        value = 0x10000 + ((value - 0xD800) << 10) + (low - 0xDC00)
                    }
                    guard let scalar = Unicode.Scalar(value) else {
                        throw LayaError.invalidAsset("tokenizer.json: invalid \\u escape at \(index)")
                    }
                    scalars.append(scalar)
                default: scalars.append(Unicode.Scalar(escape))
                }
            }
            try flushUTF8()
            return String(scalars)
        }

        private mutating func hex4() throws -> UInt32 {
            var value: UInt32 = 0
            for _ in 0..<4 {
                guard let byte = peek else { throw LayaError.invalidAsset("tokenizer.json: short \\u escape") }
                index += 1
                let digit: UInt32
                switch byte {
                case 0x30...0x39: digit = UInt32(byte - 0x30)
                case 0x41...0x46: digit = UInt32(byte - 0x41 + 10)
                case 0x61...0x66: digit = UInt32(byte - 0x61 + 10)
                default: throw LayaError.invalidAsset("tokenizer.json: bad hex digit")
                }
                value = value * 16 + digit
            }
            return value
        }
    }
}
