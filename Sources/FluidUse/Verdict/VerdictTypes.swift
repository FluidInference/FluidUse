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
        public let value: VerdictNumber

        public init(id: String, description: String, value: VerdictNumber) {
            self.id = id
            self.description = description
            self.value = value
        }
    }

    case choice(question: String, options: [Option])
    case score(question: String, levels: [Level])
    case noul(proposition: String)
}

/// A score level value. The checkpoint was trained on Python's rendering, which prints integers
/// without a fraction (`2`) and floats with one (`2.0`), so the two spellings are kept distinct.
public enum VerdictNumber: Sendable, Equatable, ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral {
    case integer(Int)
    case real(Double)

    public init(integerLiteral value: Int) { self = .integer(value) }
    public init(floatLiteral value: Double) { self = .real(value) }

    public var doubleValue: Double {
        switch self {
        case .integer(let value): Double(value)
        case .real(let value): value
        }
    }

    /// Python `f"{value}"`; Swift's shortest round-trip `Double.description` matches `repr(float)`.
    var rendered: String {
        switch self {
        case .integer(let value): String(value)
        case .real(let value): value.description
        }
    }
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
