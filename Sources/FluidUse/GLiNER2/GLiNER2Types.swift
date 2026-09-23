import Foundation

/// Published GLiNER 2.5 Core ML classification variants.
public enum GLiNER2Variant: String, Sendable, CaseIterable {
    case base
    case multilingual

    public var repository: String {
        switch self {
        case .base: "FluidInference/gliner2-5-base-coreml"
        case .multilingual: "FluidInference/gliner2-5-multi-coreml"
        }
    }

    public var packageName: String {
        switch self {
        case .base: "gliner2_base_classification_embedding_w8_L128_K8.mlpackage"
        case .multilingual: "gliner2_multi_classification_embedding_w8_linear_L128_K8.mlpackage"
        }
    }
}

public enum GLiNER2Error: Error, LocalizedError, Sendable, Equatable {
    case invalidAsset(String)
    case invalidModel(String)
    case invalidInput(String)
    case invalidOutput(String)

    public var errorDescription: String? {
        switch self {
        case .invalidAsset(let reason): "Invalid GLiNER 2.5 asset: \(reason)"
        case .invalidModel(let reason): "Invalid GLiNER 2.5 model: \(reason)"
        case .invalidInput(let reason): "Invalid GLiNER 2.5 input: \(reason)"
        case .invalidOutput(let reason): "Invalid GLiNER 2.5 output: \(reason)"
        }
    }
}

/// Single-label classifier output in the caller's label order.
public struct GLiNER2Answer: Sendable {
    public let labels: [String]
    public let probabilities: [Float]
    public let logits: [Float]
    public let tokenCount: Int

    public var selectedIndex: Int {
        probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
    }

    public var selectedLabel: String { labels[selectedIndex] }
}
