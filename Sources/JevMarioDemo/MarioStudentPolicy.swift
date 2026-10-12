import CoreML
import Foundation

/// The pixel student (model-lab `mario-tiny`, e.g. the ~9M-parameter MobileNetV3 trained with RAM tile / enemy
/// targets): the screen around Mario now and 4 frames earlier -> logits over run / jump / hop / wait (the first 4
/// outputs; multi-head models append survival logits). Under 1 ms per decision on the Neural Engine.
@available(macOS 15.0, *)
@MainActor
final class MarioStudentPolicy {
    static let moves: [MarioPlayer.Move] = [.run, .jump, .hop, .wait]
    static let cropLeft = 32, cropWidth = 192, scale = 2  // 2 blocks behind Mario to 10 ahead, half resolution
    private let model: MLModel
    /// Parameter count from the package metadata (export_coreml.py), e.g. "9.0M".
    let parameterLabel: String

    init(package: URL) async throws {
        let compiled = try await MLModel.compileModel(at: package)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        model = try await MLModel.load(contentsOf: compiled, configuration: configuration)
        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String]
        let count = Double(metadata?["parameters"] ?? "") ?? 0
        parameterLabel = count >= 1e6 ? String(format: "%.1fM", count / 1e6) : String(format: "%.0fK", count / 1e3)
    }

    /// Probabilities per move in `moves` order. `now` / `before` are raw 256x240 RGB frames.
    func decide(now: Data, before: Data, marioX: Int) throws -> [Float] {
        let height = 240 / Self.scale
        let width = Self.cropWidth / Self.scale
        let input = try MLMultiArray(shape: [1, 6, height as NSNumber, width as NSNumber], dataType: .float16)
        let out = input.dataPointer.assumingMemoryBound(to: Float16.self)
        for (slot, frame) in [now, before].enumerated() {
            frame.withUnsafeBytes { raw in
                let px = raw.bindMemory(to: UInt8.self)
                for y in 0..<height {
                    for x in 0..<width {
                        let sx = marioX - Self.cropLeft + x * Self.scale
                        let sy = y * Self.scale
                        let inside = sx >= 0 && sx < 256
                        for c in 0..<3 {
                            let value: Float16 = inside ? Float16(Float(px[(sy * 256 + sx) * 3 + c]) / 255) : 0
                            out[((slot * 3 + c) * height + y) * width + x] = value
                        }
                    }
                }
            }
        }
        let result = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["crop": input]))
        guard let logits = result.featureValue(for: "logits")?.multiArrayValue else { return [] }
        let values = (0..<Self.moves.count).map { Float(logits[$0].floatValue) }
        let peak = values.max() ?? 0
        let exps = values.map { expf($0 - peak) }
        let sum = exps.reduce(0, +)
        return exps.map { $0 / sum }
    }
}
