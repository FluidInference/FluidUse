import Foundation

/// Streaming byte-level BPE decoder for the Unlimited-OCR `tokenizer.json`. Unlike `QwenBPETokenizer.decode`, it
/// keeps added tokens such as `<|det|>` (the layout markup) and drops only BOS / EOS. Bytes are buffered until they
/// form complete UTF-8, so multi-byte characters split across tokens come out whole.
public struct UnlimitedOcrTokenDecoder: Sendable {
    private let pieces: [Int: [UInt8]]
    private let skipped: Set<Int>
    private var pending: [UInt8] = []

    public init(tokenizerJsonURL: URL, skipping skipped: Set<Int>) throws {
        let data = try Data(contentsOf: tokenizerJsonURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let model = root["model"] as? [String: Any], let vocab = model["vocab"] as? [String: Int],
            let added = root["added_tokens"] as? [[String: Any]]
        else { throw UnlimitedOcrError.invalidAsset("tokenizer.json is not a byte-level BPE tokenizer") }
        var byteForScalar: [UInt32: UInt8] = [:]
        for (byte, scalar) in Self.bytesToUnicode().enumerated() { byteForScalar[scalar] = UInt8(byte) }
        var pieces = [Int: [UInt8]](minimumCapacity: vocab.count + added.count)
        for (token, id) in vocab {
            pieces[id] = token.unicodeScalars.compactMap { byteForScalar[$0.value] }
        }
        for entry in added {
            guard let content = entry["content"] as? String, let id = entry["id"] as? Int else { continue }
            pieces[id] = Array(content.utf8)
        }
        self.pieces = pieces
        self.skipped = skipped
    }

    /// Text completed by `id`; may be empty while a multi-byte character is still incomplete.
    public mutating func append(_ id: Int) -> String {
        guard !skipped.contains(id), let bytes = pieces[id] else { return "" }
        pending += bytes
        var end = pending.count
        // Hold back a trailing incomplete UTF-8 sequence (at most 3 bytes).
        for back in 1...min(3, pending.count) {
            let byte = pending[pending.count - back]
            if byte & 0xC0 == 0x80 { continue }  // continuation byte
            let length = byte >= 0xF0 ? 4 : byte >= 0xE0 ? 3 : byte >= 0xC0 ? 2 : 1
            if length > back { end = pending.count - back }
            break
        }
        let text = String(decoding: pending[..<end], as: UTF8.self)
        pending.removeFirst(end)
        return text
    }

    /// GPT-2's reversible byte -> printable code point table.
    private static func bytesToUnicode() -> [UInt32] {
        var bytes = Array(33...126) + Array(161...172) + Array(174...255)
        var codes = bytes
        var extra = 0
        for byte in 0..<256 where !bytes.contains(byte) {
            bytes.append(byte)
            codes.append(256 + extra)
            extra += 1
        }
        var table = [UInt32](repeating: 0, count: 256)
        for (byte, code) in zip(bytes, codes) { table[byte] = UInt32(code) }
        return table
    }
}
