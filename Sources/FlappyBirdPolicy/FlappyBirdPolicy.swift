import FlappyBird
import FluidUse
import Foundation

/// Real Core ML choice between both projected actions, with no heuristic override.
public actor FlappyBirdPolicy {
    public enum Model: String, CaseIterable, Sendable {
        case gliclass
        case laya
        case gliner2Base
        case gliner2Multilingual

        public var title: String {
            switch self {
            case .gliclass: "GLiClass LUT8"
            case .laya: "Laya E8"
            case .gliner2Base: "GLiNER 2.5 base W8"
            case .gliner2Multilingual: "GLiNER 2.5 multilingual W8"
            }
        }
    }

    public struct Decision: Sendable {
        public let action: FlappyBird.Action
        public let milliseconds: Double
        public let flapProbability: Float
        public let tokens: Int
    }

    private enum Backend: Sendable {
        case gliclass(GLiClassManager)
        case laya(LayaManager)
        case gliner2(GLiNER2Manager)
    }

    private let backend: Backend
    public nonisolated let model: Model
    private static let instruction =
        "Choose the Flappy Bird action that avoids collision and stays near the gap center. "

    private init(model: Model, backend: Backend) {
        self.model = model
        self.backend = backend
    }

    /// Uses cached/published Core ML assets, or a model-specific local directory environment variable.
    public static func load(_ model: Model = .gliclass) async throws -> FlappyBirdPolicy {
        let policy: FlappyBirdPolicy
        switch model {
        case .gliclass:
            policy = try await loadGLiClass()
        case .laya:
            policy = try await loadLaya()
        case .gliner2Base, .gliner2Multilingual:
            policy = try await loadGLiNER2(model)
        }
        _ = try await policy.decide(game: FlappyBird(seed: 1), reversed: false)
        return policy
    }

    private static func loadGLiClass() async throws -> FlappyBirdPolicy {
        let configuration = GLiClassManager.Configuration(lengths: [128], precision: "lut8")
        let manager: GLiClassManager
        if let path = ProcessInfo.processInfo.environment["GLICLASS_MODEL_DIR"], !path.isEmpty {
            manager = try await GLiClassManager.load(
                from: URL(fileURLWithPath: path), configuration: configuration)
        } else {
            manager = try await GLiClassManager.load(configuration: configuration)
        }
        return FlappyBirdPolicy(model: .gliclass, backend: .gliclass(manager))
    }

    private static func loadLaya() async throws -> FlappyBirdPolicy {
        let configuration = LayaManager.Configuration(lengths: [128], precision: "e8")
        let manager: LayaManager
        if let path = ProcessInfo.processInfo.environment["LAYA_MODEL_DIR"], !path.isEmpty {
            manager = try await LayaManager.load(from: URL(fileURLWithPath: path), configuration: configuration)
        } else {
            manager = try await LayaManager.load(configuration: configuration)
        }
        return FlappyBirdPolicy(model: .laya, backend: .laya(manager))
    }

    private static func loadGLiNER2(_ model: Model) async throws -> FlappyBirdPolicy {
        let variant: GLiNER2Variant = model == .gliner2Base ? .base : .multilingual
        let directoryKey = model == .gliner2Base ? "GLINER2_BASE_MODEL_DIR" : "GLINER2_MULTI_MODEL_DIR"
        let manager: GLiNER2Manager
        if let path = ProcessInfo.processInfo.environment[directoryKey], !path.isEmpty {
            manager = try await GLiNER2Manager.load(from: URL(fileURLWithPath: path), variant: variant)
        } else {
            manager = try await GLiNER2Manager.load(variant: variant)
        }
        return FlappyBirdPolicy(model: model, backend: .gliner2(manager))
    }

    /// `reversed` changes presentation order, retaining the semantic action mapping.
    public func decide(game: FlappyBird, reversed: Bool) async throws -> Decision {
        let actions: [FlappyBird.Action] = reversed ? [.coast, .flap] : [.flap, .coast]
        let labels = actions.map { game.label(for: $0) }
        let started = DispatchTime.now().uptimeNanoseconds
        let probabilities: [Float]
        let selectedIndex: Int
        let tokenCount: Int
        let truncated: Bool
        switch backend {
        case .gliclass(let manager):
            let answer = try await manager.classify(text: game.observation, labels: labels, prompt: Self.instruction)
            probabilities = answer.probabilities
            selectedIndex = answer.selectedIndex
            tokenCount = answer.tokenCount
            truncated = answer.textWasTruncated
        case .laya(let manager):
            let answer = try await manager.answer(
                state: game.observation, question: .choice(Self.instruction, options: labels))
            probabilities = answer.probabilities
            selectedIndex = answer.selectedIndex
            tokenCount = answer.tokenCount
            truncated = answer.stateWasTruncated
        case .gliner2(let manager):
            let answer = try await manager.classify(text: game.observation, task: Self.instruction, labels: labels)
            probabilities = answer.probabilities
            selectedIndex = answer.selectedIndex
            tokenCount = answer.tokenCount
            truncated = false
        }
        let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
        guard !truncated else {
            throw LayaError.invalidOutput("Flappy Bird state exceeded the model token budget")
        }
        return Decision(
            action: actions[selectedIndex], milliseconds: milliseconds,
            flapProbability: probabilities[reversed ? 1 : 0], tokens: tokenCount)
    }
}
