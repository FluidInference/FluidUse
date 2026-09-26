import Foundation

/// Gemma BPE tokenizer as used by SigLIP 2's text encoder: spaces become `▁`, ranked merges over Unicode scalars,
/// UTF-8 byte fallback for scalars outside the vocabulary, `<eos>` appended, `<pad>` to a fixed length.
/// Text is lowercased first, matching SigLIP 2's training.
public final class SigLIP2Tokenizer: Sendable {
    public let length: Int
    public let padId: Int
    public let eosId: Int

    private let vocab: [String: Int]
    private let ranks: [String: Int]

    public convenience init(tokenizerJsonURL: URL, length: Int = 64) throws {
        try self.init(data: Data(contentsOf: tokenizerJsonURL), length: length)
    }

    public init(data: Data, length: Int = 64) throws {
        guard
            let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let model = root["model"] as? [String: Any],
            let vocab = model["vocab"] as? [String: Int],
            let merges = model["merges"] as? [[String]]
        else {
            throw SigLIP2Error.invalidAsset("tokenizer.json is not a Gemma BPE tokenizer")
        }
        guard let padId = vocab["<pad>"], let eosId = vocab["<eos>"] else {
            throw SigLIP2Error.invalidAsset("tokenizer.json has no <pad> or <eos> token")
        }
        var ranks: [String: Int] = [:]
        ranks.reserveCapacity(merges.count)
        for (rank, pair) in merges.enumerated() where pair.count == 2 {
            ranks[Self.key(pair[0], pair[1])] = rank
        }
        self.vocab = vocab
        self.ranks = ranks
        self.length = length
        self.padId = padId
        self.eosId = eosId
    }

    /// Token ids for `text`, `<eos>`-terminated and padded to `length`.
    public func encode(_ text: String) throws -> [Int32] {
        var ids = tokenize(text.lowercased())
        ids.append(eosId)
        guard ids.count <= length else {
            throw SigLIP2Error.invalidInput("Label needs \(ids.count) tokens; the text encoder takes \(length)")
        }
        ids.append(contentsOf: repeatElement(padId, count: length - ids.count))
        return ids.map(Int32.init)
    }

    /// Token ids without `<eos>` or padding.
    public func tokenize(_ text: String) -> [Int] {
        let normalized = text.replacingOccurrences(of: " ", with: "\u{2581}")
        var symbols = normalized.unicodeScalars.map { String($0) }
        while symbols.count > 1 {
            var best: (rank: Int, index: Int)?
            for index in 0..<(symbols.count - 1) {
                if let rank = ranks[Self.key(symbols[index], symbols[index + 1])], rank < (best?.rank ?? .max) {
                    best = (rank, index)
                }
            }
            guard let best else { break }
            symbols[best.index] += symbols[best.index + 1]
            symbols.remove(at: best.index + 1)
        }
        var ids: [Int] = []
        for symbol in symbols {
            if let id = vocab[symbol] {
                ids.append(id)
            } else {
                for byte in symbol.utf8 {
                    ids.append(vocab[String(format: "<0x%02X>", byte)] ?? vocab["<unk>"] ?? 3)
                }
            }
        }
        return ids
    }

    private static func key(_ left: String, _ right: String) -> String { left + "\u{0}" + right }
}
