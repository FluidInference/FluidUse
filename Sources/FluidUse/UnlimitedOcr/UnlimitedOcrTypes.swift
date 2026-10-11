import Foundation

public enum UnlimitedOcrError: Error, LocalizedError, Sendable {
    case invalidAsset(String)
    case invalidInput(String)
    case prediction(String)

    public var errorDescription: String? {
        switch self {
        case .invalidAsset(let reason): "Invalid Unlimited-OCR asset: \(reason)"
        case .invalidInput(let reason): "Invalid Unlimited-OCR input: \(reason)"
        case .prediction(let reason): "Unlimited-OCR prediction failed: \(reason)"
        }
    }
}

/// `config.json` of the Core ML bundle (written by model-lab `unlimited-ocr-coreml/conversion/export_assets.py`).
struct UnlimitedOcrConfig: Decodable, Sendable {
    let hiddenSize: Int
    let headDim: Int
    let layers: Int
    let vocabSize: Int
    let vocabHalf: Int
    let ropeTheta: Double
    let ring: Int
    let chunk: Int
    let capacities: [Int]
    let pageTokens: Int
    let imageSize: Int
    let bosId: Int
    let eosId: Int
    let imageTokenId: Int
    let promptSuffixIds: [String: [Int]]
    let ngramSize: Int
    let ngramWindow: Int
    let maxLength: Int
    let decoder: String
    let encoder: String

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case headDim = "head_dim"
        case layers
        case vocabSize = "vocab_size"
        case vocabHalf = "vocab_half"
        case ropeTheta = "rope_theta"
        case ring, chunk, capacities
        case pageTokens = "page_tokens"
        case imageSize = "image_size"
        case bosId = "bos_id"
        case eosId = "eos_id"
        case imageTokenId = "image_token_id"
        case promptSuffixIds = "prompt_suffix_ids"
        case ngramSize = "ngram_size"
        case ngramWindow = "ngram_window"
        case maxLength = "max_length"
        case decoder, encoder
    }
}

/// Progress of one `UnlimitedOcrManager.read` call.
public enum UnlimitedOcrEvent: Sendable {
    /// Page `index` (0-based) went through the vision encoder.
    case pageEncoded(index: Int, milliseconds: Double)
    /// The whole document (all page tokens + prompt) is in the decoder's cache.
    case prefilled(tokens: Int, capacity: Int, milliseconds: Double)
    /// Newly decoded text (raw model output, including `<PAGE>` and `<|det|>` layout markup).
    case text(String)
    case finished(UnlimitedOcrStats)
}

public struct UnlimitedOcrStats: Sendable {
    public let pages: Int
    public let promptTokens: Int
    public let outputTokens: Int
    public let encodeMilliseconds: Double
    public let prefillMilliseconds: Double
    public let decodeMilliseconds: Double
    public let stoppedAtEOS: Bool

    public var millisecondsPerToken: Double { decodeMilliseconds / Double(max(1, outputTokens)) }
}
