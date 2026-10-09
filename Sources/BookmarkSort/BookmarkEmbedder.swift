import FluidUse
import Foundation
import NaturalLanguage

/// Turns a post into a fixed-size vector for `LearnedBookmarkSorter`.
public protocol BookmarkEmbedding: Sendable {
    /// Short name for logs and per-embedder vector caches.
    var name: String { get }
    var dimension: Int { get }
    func embed(_ text: String) async throws -> [Float]
}

/// EmbeddingGemma 2 on Core ML (Neural Engine), with the document prefix that classified best in the A/B.
public struct GemmaBookmarkEmbedder: BookmarkEmbedding {
    public let manager: EmbeddingGemma2Manager
    public var name: String { "embeddinggemma-2" }
    public var dimension: Int { EmbeddingGemma2Manager.dimension }

    public init(manager: EmbeddingGemma2Manager) { self.manager = manager }

    public static func load() async throws -> GemmaBookmarkEmbedder {
        GemmaBookmarkEmbedder(manager: try await EmbeddingGemma2Manager.loadDefault())
    }

    public func embed(_ text: String) async throws -> [Float] {
        try await manager.embed(text, prompt: .document(title: nil))
    }
}

/// Sentence vectors from Apple's on-device contextual embeddings: the Latin- and Chinese-script
/// models are both run on every post and their mean-pooled token vectors concatenated, so English,
/// Chinese, and mixed posts share one space. Uses the OS models; nothing is downloaded by FluidUse.
public actor BookmarkEmbedder: BookmarkEmbedding {
    private let models: [NLContextualEmbedding]

    public nonisolated let dimension: Int
    public nonisolated var name: String { "apple-nl" }

    public init() async throws {
        var models: [NLContextualEmbedding] = []
        for script in [NLScript.latin, .simplifiedChinese] {
            guard let model = NLContextualEmbedding(script: script) else {
                throw BookmarkSortError.unavailable("No contextual embedding for \(script.rawValue)")
            }
            if !model.hasAvailableAssets {
                let result = try await model.requestAssets()
                guard result == .available else {
                    throw BookmarkSortError.unavailable("Embedding assets for \(script.rawValue) not available")
                }
            }
            try model.load()
            models.append(model)
        }
        self.models = models
        dimension = models.reduce(0) { $0 + $1.dimension }
    }

    public func embed(_ text: String) throws -> [Float] {
        var vector: [Float] = []
        vector.reserveCapacity(dimension)
        for model in models {
            var sum = [Double](repeating: 0, count: model.dimension)
            var tokens = 0
            if !text.isEmpty {
                let result = try model.embeddingResult(for: text, language: nil)
                result.enumerateTokenVectors(in: text.startIndex..<text.endIndex) { token, _ in
                    for i in 0..<min(token.count, sum.count) { sum[i] += token[i] }
                    tokens += 1
                    return true
                }
            }
            vector += sum.map { Float($0 / Double(max(tokens, 1))) }
        }
        return vector
    }
}

/// Per-dimension standardization followed by unit length, fitted on the training vectors.
public struct FeatureScaler: Sendable {
    let mean: [Float]
    let scale: [Float]

    public init(fitting vectors: [[Float]]) {
        let dimension = vectors.first?.count ?? 0
        var mean = [Float](repeating: 0, count: dimension)
        var variance = [Float](repeating: 0, count: dimension)
        for vector in vectors { for i in 0..<dimension { mean[i] += vector[i] } }
        mean = mean.map { $0 / Float(max(vectors.count, 1)) }
        for vector in vectors {
            for i in 0..<dimension { variance[i] += (vector[i] - mean[i]) * (vector[i] - mean[i]) }
        }
        self.mean = mean
        scale = variance.map { 1 / ((($0 / Float(max(vectors.count, 1))).squareRoot()) + 1e-6) }
    }

    public func transform(_ vector: [Float]) -> [Float] {
        var result = zip(vector, zip(mean, scale)).map { ($0 - $1.0) * $1.1 }
        let norm = result.reduce(0) { $0 + $1 * $1 }.squareRoot()
        if norm > 0 { result = result.map { $0 / norm } }
        return result
    }
}
