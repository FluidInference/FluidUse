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
        /// Rendered as a float (`2.0`), as the author's engine validates levels as floats.
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

/// How `VerdictManager.answer` turns logits into probabilities.
public enum VerdictCalibration: Sendable, Equatable {
    /// The released `calibrator.json`, as the author's serving engine applies it: the per-K temperature for the
    /// candidate count (abstention included), else the global temperature. Fitted by the author for open-domain use.
    case shipped
    /// Softmax of the raw logits (temperature 1).
    case uncalibrated
    /// One caller-chosen temperature for every candidate count, e.g. fitted on the caller's own calibration data.
    case temperature(Double)
}

/// The released calibrator: a global temperature and per-candidate-count overrides.
struct VerdictCalibrator: Sendable {
    let defaultTemperature: Double
    let temperatureByCount: [String: Double]

    init(data: Data) throws {
        guard let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let temperature = raw["temperature"] as? Double, temperature.isFinite, temperature > 0,
            let byCount = raw["per_k"] as? [String: Double],
            byCount.values.allSatisfy({ $0.isFinite && $0 > 0 })
        else { throw VerdictError.invalidAsset("Invalid calibrator.json") }
        defaultTemperature = temperature
        temperatureByCount = byCount
    }

    func temperature(candidates: Int, calibration: VerdictCalibration) -> Double {
        switch calibration {
        case .shipped: temperatureByCount[String(candidates)] ?? defaultTemperature
        case .uncalibrated: 1
        case .temperature(let value): value
        }
    }

    static func probabilities(logits: [Float], temperature: Double) throws -> [Double] {
        let scaled = logits.map { Double($0) / temperature }
        let peak = scaled.max() ?? 0
        let exponentials = scaled.map { exp($0 - peak) }
        let total = exponentials.reduce(0, +)
        guard total.isFinite, total > 0 else { throw VerdictError.invalidOutput("Invalid calibrated probabilities") }
        return exponentials.map { $0 / total }
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
