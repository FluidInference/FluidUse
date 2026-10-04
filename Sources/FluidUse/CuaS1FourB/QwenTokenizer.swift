import Foundation

/// Byte-level BPE encoder for the Qwen3.5 `tokenizer.json` (encode only).
///
/// Mirrors the HuggingFace `tokenizers` pipeline the reference uses: split out added (special)
/// tokens verbatim, NFC-normalize the remaining text, pre-tokenize with the file's split regex,
/// map UTF-8 bytes through the GPT-2 byte alphabet, then apply BPE merges by rank.
public final class QwenTokenizer: Sendable {
    private let vocab: [String: Int]
    private let mergeRank: [String: Int]
    private let splitRegex: NSRegularExpression
    private let byteChars: [String]
    /// Added tokens, longest first so overlapping prefixes resolve like the reference trie.
    private let addedTokens: [(text: String, id: Int)]

    public init(tokenizerJsonURL: URL) throws {
        let data = try Data(contentsOf: tokenizerJsonURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let model = root["model"] as? [String: Any],
            let vocabAny = model["vocab"] as? [String: Any],
            let mergesAny = model["merges"] as? [Any]
        else {
            throw CuaS1FourBError.invalidAsset("tokenizer.json is missing model.vocab / model.merges")
        }
        var vocab = [String: Int](minimumCapacity: vocabAny.count)
        for (token, id) in vocabAny {
            guard let id = id as? Int else { continue }
            vocab[token] = id
        }
        guard vocab.count == vocabAny.count else {
            throw CuaS1FourBError.invalidAsset(
                "tokenizer.json vocab keys collided (\(vocabAny.count) -> \(vocab.count))")
        }
        self.vocab = vocab

        var ranks = [String: Int](minimumCapacity: mergesAny.count)
        for (rank, merge) in mergesAny.enumerated() {
            if let pair = merge as? [String], pair.count == 2 {
                ranks["\(pair[0]) \(pair[1])"] = rank
            } else if let text = merge as? String {
                ranks[text] = rank
            }
        }
        self.mergeRank = ranks

        var added: [(String, Int)] = []
        for entry in root["added_tokens"] as? [[String: Any]] ?? [] {
            if let content = entry["content"] as? String, let id = entry["id"] as? Int {
                added.append((content, id))
            }
        }
        self.addedTokens = added.sorted { $0.0.count > $1.0.count }.map { (text: $0.0, id: $0.1) }

        self.splitRegex = try NSRegularExpression(pattern: Self.splitPattern(root))
        self.byteChars = Self.bytesToUnicode()
    }

    /// Token ids for `text`, exactly as `tokenizer(text)["input_ids"]` (no BOS/EOS is added by Qwen).
    public func encode(_ text: String) -> [Int] {
        var ids: [Int] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            if let (range, id) = firstAddedToken(in: rest) {
                encodeOrdinary(String(rest[rest.startIndex..<range.lowerBound]), into: &ids)
                ids.append(id)
                rest = rest[range.upperBound...]
            } else {
                encodeOrdinary(String(rest), into: &ids)
                break
            }
        }
        return ids
    }

    /// Id of a single-token string (special token or vocabulary entry), if any.
    public func tokenId(_ token: String) -> Int? {
        addedTokens.first { $0.text == token }?.id ?? vocab[token]
    }

    private func firstAddedToken(in text: Substring) -> (Range<Substring.Index>, Int)? {
        var best: (Range<Substring.Index>, Int)?
        for token in addedTokens {
            guard let range = text.range(of: token.text, options: .literal) else { continue }
            if let current = best, current.0.lowerBound <= range.lowerBound { continue }
            best = (range, token.id)
        }
        return best
    }

    private func encodeOrdinary(_ text: String, into ids: inout [Int]) {
        guard !text.isEmpty else { return }
        let normalized = text.precomposedStringWithCanonicalMapping as NSString
        for match in splitRegex.matches(
            in: normalized as String, range: NSRange(location: 0, length: normalized.length))
        {
            let piece = normalized.substring(with: match.range)
            let symbols = piece.utf8.map { byteChars[Int($0)] }
            for token in bpe(symbols) {
                if let id = vocab[token] { ids.append(id) }
            }
        }
    }

    private func bpe(_ initial: [String]) -> [String] {
        var symbols = initial
        while symbols.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for i in 0..<(symbols.count - 1) {
                if let rank = mergeRank["\(symbols[i]) \(symbols[i + 1])"], rank < bestRank {
                    bestRank = rank
                    bestIndex = i
                }
            }
            guard bestIndex >= 0 else { break }
            let left = symbols[bestIndex]
            let right = symbols[bestIndex + 1]
            var merged: [String] = []
            merged.reserveCapacity(symbols.count - 1)
            var i = 0
            while i < symbols.count {
                if i < symbols.count - 1, symbols[i] == left, symbols[i + 1] == right {
                    merged.append(left + right)
                    i += 2
                } else {
                    merged.append(symbols[i])
                    i += 1
                }
            }
            symbols = merged
        }
        return symbols
    }

    private static func splitPattern(_ root: [String: Any]) throws -> String {
        let pre = root["pre_tokenizer"] as? [String: Any]
        let steps = (pre?["pretokenizers"] as? [[String: Any]]) ?? (pre.map { [$0] } ?? [])
        for step in steps where step["type"] as? String == "Split" {
            if let pattern = step["pattern"] as? [String: Any], let regex = pattern["Regex"] as? String {
                return regex
            }
        }
        throw CuaS1FourBError.invalidAsset("tokenizer.json has no Split pre-tokenizer regex")
    }

    /// GPT-2 `bytes_to_unicode`: printable bytes map to themselves, the rest to U+0100 onwards.
    private static func bytesToUnicode() -> [String] {
        var printable = Array(33...126) + Array(161...172) + Array(174...255)
        var codepoints = printable
        var next = 0
        for byte in 0..<256 where !printable.contains(byte) {
            printable.append(byte)
            codepoints.append(256 + next)
            next += 1
        }
        var table = [String](repeating: "", count: 256)
        for (byte, codepoint) in zip(printable, codepoints) {
            // all code points are below U+0144, so the scalar always exists
            if let scalar = Unicode.Scalar(UInt32(codepoint)) { table[byte] = String(scalar) }
        }
        return table
    }
}
