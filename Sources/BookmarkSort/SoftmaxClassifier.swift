import Accelerate
import Foundation

/// Multinomial logistic regression with L2, trained full-batch with Adam. Small enough to retrain
/// from scratch whenever the person files another example.
public struct SoftmaxClassifier: Sendable {
    public let classes: [String]
    let weights: [Float]  // classes × dimension, row-major
    let bias: [Float]
    let dimension: Int

    public struct Prediction: Sendable {
        /// Classes ordered from most to least likely, with probabilities.
        public let ranked: [(label: String, probability: Float)]

        public var label: String { ranked[0].label }
        public var confidence: Float { ranked[0].probability }
    }

    /// `l2` is the penalty on the weights per example, like scikit-learn's `1 / C` divided by the sample count.
    public static func train(
        features: [[Float]], labels: [String], l2: Float = 1.0 / 3, iterations: Int = 400, learningRate: Float = 0.05
    ) throws -> SoftmaxClassifier {
        guard let dimension = features.first?.count, features.count == labels.count else {
            throw BookmarkSortError.invalidInput("Training needs one label per feature vector")
        }
        let classes = Array(Set(labels)).sorted()
        guard classes.count >= 2 else { throw BookmarkSortError.invalidInput("Training needs at least two classes") }
        let classCount = classes.count
        let count = features.count
        let index = Dictionary(uniqueKeysWithValues: classes.enumerated().map { ($1, $0) })
        let targets = labels.map { index[$0]! }
        let matrix = features.flatMap { $0 }

        var weights = [Float](repeating: 0, count: classCount * dimension)
        var bias = [Float](repeating: 0, count: classCount)
        var moments = (
            weight: [Float](repeating: 0, count: weights.count),
            weightSquare: [Float](repeating: 0, count: weights.count),
            bias: [Float](repeating: 0, count: classCount), biasSquare: [Float](repeating: 0, count: classCount)
        )
        let (beta1, beta2, epsilon): (Float, Float, Float) = (0.9, 0.999, 1e-8)
        var logits = [Float](repeating: 0, count: count * classCount)
        var gradient = [Float](repeating: 0, count: weights.count)
        for step in 1...iterations {
            // logits = X · Wᵀ
            vDSP_mmul(
                matrix, 1, transpose(weights, rows: classCount, columns: dimension), 1, &logits, 1,
                vDSP_Length(count), vDSP_Length(classCount), vDSP_Length(dimension))
            var biasGradient = [Float](repeating: 0, count: classCount)
            for row in 0..<count {
                let base = row * classCount
                softmax(&logits, offset: base, count: classCount, bias: bias)
                logits[base + targets[row]] -= 1
                for column in 0..<classCount {
                    logits[base + column] /= Float(count)
                    biasGradient[column] += logits[base + column]
                }
            }
            // gradient = errorsᵀ · X + l2 · W
            vDSP_mmul(
                transpose(logits, rows: count, columns: classCount), 1, matrix, 1, &gradient, 1,
                vDSP_Length(classCount), vDSP_Length(dimension), vDSP_Length(count))
            let penalty = l2 / Float(count)
            let correction1 = 1 - pow(beta1, Float(step))
            let correction2 = 1 - pow(beta2, Float(step))
            for i in weights.indices {
                let g = gradient[i] + penalty * weights[i]
                moments.weight[i] = beta1 * moments.weight[i] + (1 - beta1) * g
                moments.weightSquare[i] = beta2 * moments.weightSquare[i] + (1 - beta2) * g * g
                weights[i] -=
                    learningRate * (moments.weight[i] / correction1)
                    / ((moments.weightSquare[i] / correction2).squareRoot() + epsilon)
            }
            for i in bias.indices {
                let g = biasGradient[i]
                moments.bias[i] = beta1 * moments.bias[i] + (1 - beta1) * g
                moments.biasSquare[i] = beta2 * moments.biasSquare[i] + (1 - beta2) * g * g
                bias[i] -=
                    learningRate * (moments.bias[i] / correction1)
                    / ((moments.biasSquare[i] / correction2).squareRoot() + epsilon)
            }
        }
        return SoftmaxClassifier(classes: classes, weights: weights, bias: bias, dimension: dimension)
    }

    public func predict(_ feature: [Float]) -> Prediction {
        var scores = [Float](repeating: 0, count: classes.count)
        for column in classes.indices {
            var dot: Float = 0
            weights.withUnsafeBufferPointer { pointer in
                vDSP_dotpr(pointer.baseAddress! + column * dimension, 1, feature, 1, &dot, vDSP_Length(dimension))
            }
            scores[column] = dot
        }
        Self.softmax(&scores, offset: 0, count: classes.count, bias: bias)
        let ranked = classes.indices.sorted { scores[$0] > scores[$1] }.map { (classes[$0], scores[$0]) }
        return Prediction(ranked: ranked)
    }

    private static func softmax(_ values: inout [Float], offset: Int, count: Int, bias: [Float]) {
        var maximum = -Float.infinity
        for i in 0..<count {
            values[offset + i] += bias[i]
            maximum = max(maximum, values[offset + i])
        }
        var total: Float = 0
        for i in 0..<count {
            values[offset + i] = exp(values[offset + i] - maximum)
            total += values[offset + i]
        }
        for i in 0..<count { values[offset + i] /= total }
    }

    private static func transpose(_ values: [Float], rows: Int, columns: Int) -> [Float] {
        var result = [Float](repeating: 0, count: values.count)
        vDSP_mtrans(values, 1, &result, 1, vDSP_Length(columns), vDSP_Length(rows))
        return result
    }
}

public enum BookmarkSortError: Error, LocalizedError, Sendable {
    case invalidInput(String)
    case unavailable(String)

    public var errorDescription: String? {
        switch self {
        case .invalidInput(let reason): "Invalid input: \(reason)"
        case .unavailable(let reason): "Unavailable: \(reason)"
        }
    }
}
