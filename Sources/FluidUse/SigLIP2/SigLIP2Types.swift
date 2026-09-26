import Foundation

public enum SigLIP2Error: Error, LocalizedError {
    case invalidAsset(String)
    case invalidInput(String)
    case predictionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidAsset(let reason): "Invalid SigLIP 2 asset: \(reason)"
        case .invalidInput(let reason): "Invalid SigLIP 2 input: \(reason)"
        case .predictionFailed(let reason): "SigLIP 2 prediction failed: \(reason)"
        }
    }
}

/// Preprocessing and scoring constants written by the converter (`config.json`).
public struct SigLIP2Config: Codable, Sendable {
    public let modelId: String
    public let imageSize: Int
    public let imageMean: [Float]
    public let imageStd: [Float]
    public let textLength: Int
    public let logitScale: Float
    public let logitBias: Float
    public let embeddingDim: Int
    public let precision: String

    enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
        case imageSize = "image_size"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case textLength = "text_length"
        case logitScale = "logit_scale"
        case logitBias = "logit_bias"
        case embeddingDim = "embedding_dim"
        case precision
    }

    /// Package base name, e.g. `siglip2-base-patch16-256`.
    public var name: String { modelId.split(separator: "/").last.map(String.init) ?? modelId }
}

/// One image scored against a label set.
public struct SigLIP2Answer: Sendable {
    public let labels: [String]
    /// Cosine similarity per label.
    public let similarities: [Float]
    /// Independent `sigmoid(scale · cos + bias)` per label, as SigLIP was trained.
    public let probabilities: [Float]

    public var selectedIndex: Int { similarities.indices.max { similarities[$0] < similarities[$1] } ?? 0 }
    public var selectedLabel: String { labels[selectedIndex] }
}
