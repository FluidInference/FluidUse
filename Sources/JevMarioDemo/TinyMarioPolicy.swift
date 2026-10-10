import CoreML
import Foundation

/// The tiny pixel student (model-lab `mario-tiny`, ~0.46M parameters): the screen around Mario now and 4 frames
/// earlier -> logits over run / jump / hop / wait. About 0.3 ms per decision.
@available(macOS 15.0, *)
@MainActor
final class TinyMarioPolicy {
    static let moves: [MarioPlayer.Move] = [.run, .jump, .hop, .wait]
    static let cropLeft = 32, cropWidth = 192, scale = 2  // 2 blocks behind Mario to 10 ahead, half resolution
    private let model: MLModel

    init(package: URL) async throws {
        let compiled = try await MLModel.compileModel(at: package)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        model = try await MLModel.load(contentsOf: compiled, configuration: configuration)
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
