import Foundation

/// Errors from GLiClass asset loading, prompt construction, and inference.
public enum GLiClassError: Error, LocalizedError, Sendable, Equatable {
    case invalidAsset(String)
    case invalidModel(String)
    case invalidOutput(String)
    case invalidOptionCount(Int)
    case emptyOption(Int)
    case promptTooLong(optionCount: Int, maximumLength: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidAsset(let reason): return "Invalid GLiClass asset: \(reason)"
        case .invalidModel(let reason): return "Invalid GLiClass model: \(reason)"
        case .invalidOutput(let reason): return "Invalid GLiClass output: \(reason)"
        case .invalidOptionCount(let count):
            return "GLiClass requires 2–\(GLiClassManager.maximumOptions) options; received \(count)."
        case .emptyOption(let index): return "GLiClass option \(index) is empty."
        case .promptTooLong(let optionCount, let maximumLength):
            return "GLiClass's \(optionCount) option labels do not fit the \(maximumLength)-token bucket."
        }
    }
}

/// A single-label GLiClass result. Probabilities follow the supplied label order.
public struct GLiClassAnswer: Sendable {
    public let labels: [String]
    public let probabilities: [Float]
    public let logits: [Float]
    public let tokenCount: Int
    public let bucketLength: Int
    public let textWasTruncated: Bool

    public var selectedIndex: Int {
        probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
    }

    public var selectedLabel: String { labels[selectedIndex] }
}
