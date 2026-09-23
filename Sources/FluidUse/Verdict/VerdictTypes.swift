import Foundation

/// A Verdict request in the checkpoint's choice, score, or noul format.
public enum VerdictQuestion: Sendable, Equatable {
    public struct Option: Sendable, Equatable {
        public let id: String
        public let description: String

        public init(id: String, description: String) {
            self.id = id
            self.description = description
        }
    }

    public struct Level: Sendable, Equatable {
        public let id: String
        public let description: String
        public let value: Double

        public init(id: String, description: String, value: Double) {
            self.id = id
            self.description = description
            self.value = value
        }
    }

    case choice(question: String, options: [Option])
    case score(question: String, levels: [Level])
    case noul(proposition: String)
}

/// Verdict's calibrated decision. Abstention remains a distinct selected ID.
public struct VerdictAnswer: Sendable {
    public let candidateIDs: [String]
    public let probabilities: [Double]
    public let logits: [Float]
    public let tokenCount: Int
    public let selectedID: String
    public let isAbstention: Bool
    public let score: Double?
    public let probabilityTrue: Double?

    public var probabilityAbstain: Double { probabilities.last ?? 0 }
}

public enum VerdictError: Error, LocalizedError, Sendable, Equatable {
    case invalidInput(String)
    case invalidAsset(String)
    case invalidModel(String)
    case invalidOutput(String)

    public var errorDescription: String? {
        switch self {
        case .invalidInput(let reason): "Invalid Verdict input: \(reason)"
        case .invalidAsset(let reason): "Invalid Verdict asset: \(reason)"
        case .invalidModel(let reason): "Invalid Verdict model: \(reason)"
        case .invalidOutput(let reason): "Invalid Verdict output: \(reason)"
        }
    }
}
