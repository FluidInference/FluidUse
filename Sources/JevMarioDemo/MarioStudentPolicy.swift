import CoreML
import Foundation

/// The pixel student (model-lab `mario-tiny`, a ~9M-parameter MobileNetV3 trained with RAM tile / enemy targets):
/// the screen around Mario now and 4 frames earlier -> logits over its moves (named in the package metadata; the first
/// `moves.count` outputs). A 4-move model (run / jump / hop / wait) decides when Mario lands; an 11-move model
/// (trained on the level solver) decides every 8 frames. Under 1 ms per decision on the Neural Engine.
@available(macOS 15.0, *)
@MainActor
final class MarioStudentPolicy {
    /// Move names in output order, from the package metadata.
    let moves: [String]
    /// Solver-style model: 8-frame macros, one decision per macro.
    var everyMacro: Bool { moves.count > 4 }

    static let right: UInt8 = 0x80, left: UInt8 = 0x40, down: UInt8 = 0x20, a: UInt8 = 0x01, b: UInt8 = 0x02
    /// The level solver's macros (model-lab mario-tiny/solver.py), 8 frames each.
    static let solverMacros: [String: [UInt8]] = [
        "run": Array(repeating: right | b, count: 8),
        "walk": Array(repeating: right, count: 8),
        "hold": Array(repeating: right | b | a, count: 8),
        "jump": [right | b] + Array(repeating: right | b | a, count: 7),
        "hop": [right | b] + Array(repeating: right | b | a, count: 2) + Array(repeating: right | b, count: 5),
        "up": [0] + Array(repeating: a, count: 7),
        "left": Array(repeating: left | b, count: 8),
        "leftjump": [left] + Array(repeating: left | a, count: 7),
        "wait": Array(repeating: 0, count: 8),
        "swim": Array(repeating: right | a, count: 2) + Array(repeating: right, count: 6),
        "down": Array(repeating: down, count: 8),
    ]
    static let cropLeft = 32, cropWidth = 192, scale = 2  // 2 blocks behind Mario to 10 ahead, half resolution
    private let model: MLModel
    /// Parameter count from the package metadata (export_coreml.py), e.g. "9.0M".
    let parameterLabel: String
    /// Where Core ML runs the model's ops (MLComputePlan), e.g. "Neural Engine · 185/185 ops".
    private(set) var placement = ""
    private let compiledURL: URL

    init(package: URL) async throws {
        let compiled = try await MLModel.compileModel(at: package)
        compiledURL = compiled
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        model = try await MLModel.load(contentsOf: compiled, configuration: configuration)
        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String]
        let count = Double(metadata?["parameters"] ?? "") ?? 0
        parameterLabel = count >= 1e6 ? String(format: "%.1fM", count / 1e6) : String(format: "%.0fK", count / 1e3)
        moves = (metadata?["moves"] ?? "run,jump,hop,wait").split(separator: ",").map(String.init)
        placement = await Self.placement(of: compiled)
    }

    /// Ops per preferred compute device from Core ML's compute plan (activity meters do not see the Neural Engine).
    static func placement(of compiled: URL) async -> String {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        guard let plan = try? await MLComputePlan.load(contentsOf: compiled, configuration: configuration),
            case .program(let program) = plan.modelStructure, let main = program.functions["main"]
        else { return "" }
        var ane = 0
        var gpu = 0
        var cpu = 0
        for op in main.block.operations where op.operatorName != "const" {
            switch plan.deviceUsage(for: op)?.preferred {
            case .neuralEngine: ane += 1
            case .gpu: gpu += 1
            case .cpu: cpu += 1
            default: break
            }
        }
        let total = ane + gpu + cpu
        guard total > 0 else { return "" }
        if ane == total { return "Neural Engine · \(ane)/\(total) ops" }
        return "Neural Engine \(ane) · GPU \(gpu) · CPU \(cpu) ops"
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
        let values = (0..<moves.count).map { Float(logits[$0].floatValue) }
        let peak = values.max() ?? 0
        let exps = values.map { expf($0 - peak) }
        let sum = exps.reduce(0, +)
        return exps.map { $0 / sum }
    }
}
