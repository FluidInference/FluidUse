import Foundation

/// Published GLiNER 2.5 Core ML classification variants.
public enum GLiNER2Variant: String, Sendable, CaseIterable {
    case small
    case base
    case multilingual
    /// GLiNER2.5-Decide (DeBERTa-v3-large) with 128 tokens and up to 32 labels; its package scores four heads.
    /// The fp16 packages are used: on the GPU the W8 ones trigger a load-time preparation that peaks at several GB.
    case decide
    /// GLiNER2.5-Decide with 256 tokens, for longer documents or several heads per call.
    case decideLong

    public var repository: String {
        switch self {
        case .small: "FluidInference/gliner2-5-small-coreml"
        case .base: "FluidInference/gliner2-5-base-coreml"
        case .multilingual: "FluidInference/gliner2-5-multi-coreml"
        case .decide, .decideLong: "FluidInference/gliner2-5-decide-coreml"
        }
    }

    public var packageName: String {
        switch self {
        case .small: "gliner2_small_classification_embedding_w8_L128_K8.mlpackage"
        case .base: "gliner2_base_classification_embedding_w8_L128_K8.mlpackage"
        case .multilingual: "gliner2_multi_classification_embedding_w8_linear_L128_K8.mlpackage"
        case .decide: "gliner2_decide_classification_fp16_L128_H4_K32.mlpackage"
        case .decideLong: "gliner2_decide_classification_fp16_L256_H4_K32.mlpackage"
        }
    }

    var isDecide: Bool { self == .decide || self == .decideLong }

    /// Token budget for schema plus text.
    public var maximumLength: Int { self == .decideLong ? 256 : 128 }

    /// Labels per head.
    public var maximumOptions: Int { isDecide ? 32 : 8 }

    /// Classification heads per call.
    public var maximumHeads: Int { isDecide ? 4 : 1 }

    var tokenizerPath: String {
        isDecide ? "tokenizer.json" : "tokenizer/tokenizer.json"
    }

    /// Shape of `marker_indices`, `marker_mask`, and both outputs.
    var markerShape: [Int] {
        isDecide ? [1, maximumHeads, maximumOptions] : [1, maximumOptions]
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
