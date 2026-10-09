import Foundation

public enum EvokeError: Error, LocalizedError {
    case invalidAsset(String)
    case predictionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidAsset(let reason): "Invalid Evoke asset: \(reason)"
        case .predictionFailed(let reason): "Evoke prediction failed: \(reason)"
        }
    }
}

/// Sparse terms: RoBERTa vocabulary id -> weight. Score two texts with `EvokeTerms.score`.
public typealias EvokeTerms = [Int: Float]

/// Which Evoke compiler transform to apply: queries keep fewer, sharper terms than documents.
public enum EvokeTextKind: String, Sendable {
    case query
    case document
}

/// Per-compiler pooling constants (`config.json` → `evoke.query` / `evoke.document`, from Evoke P2.2).
public struct EvokeTransform: Codable, Sendable, Equatable {
    public let activeDims: Int
    public let gamma: Float
    public let scale: Float

    public init(activeDims: Int, gamma: Float, scale: Float) {
        self.activeDims = activeDims
        self.gamma = gamma
        self.scale = scale
    }

    enum CodingKeys: String, CodingKey {
        case activeDims = "active_dims"
        case gamma
        case scale = "score_scale"
    }

    /// `log1p(max(v, 0)) ^ gamma * scale` over the first `activeDims` of the model's sorted top-k; keeps weights > 0.
    public func terms(maxLogits: [Float], vocabIds: [Int]) -> EvokeTerms {
        var terms: EvokeTerms = [:]
        for index in 0..<min(activeDims, maxLogits.count, vocabIds.count) {
            let weight = pow(log1p(max(maxLogits[index], 0)), gamma) * scale
            if weight > 0 { terms[vocabIds[index]] = weight }
        }
        return terms
    }
}

/// Converter-written `config.json`.
public struct EvokeConfig: Codable, Sendable {
    public struct Pooling: Codable, Sendable {
        public let query: EvokeTransform
        public let document: EvokeTransform
    }

    public let sequenceLengths: [Int]
    public let topK: Int
    public let evoke: Pooling

    enum CodingKeys: String, CodingKey {
        case sequenceLengths = "sequence_lengths"
        case topK = "top_k"
        case evoke
    }

    public func transform(for kind: EvokeTextKind) -> EvokeTransform {
        kind == .query ? evoke.query : evoke.document
    }
}

extension Dictionary where Key == Int, Value == Float {
    /// Sparse dot product: sum of query weight × document weight over shared terms.
    public static func score(_ query: EvokeTerms, _ document: EvokeTerms) -> Float {
        let (small, large) = query.count <= document.count ? (query, document) : (document, query)
        return small.reduce(0) { sum, entry in sum + entry.value * (large[entry.key] ?? 0) }
    }
}
