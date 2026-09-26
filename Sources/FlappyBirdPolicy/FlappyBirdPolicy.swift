import FlappyBird
import FluidUse
import Foundation

/// Real Core ML choice between both projected actions, with no heuristic override.
public actor FlappyBirdPolicy {
    public enum Model: String, CaseIterable, Sendable {
        case gliclass
        case laya
        case gliner2Small
        case gliner2Base
        case gliner2Multilingual
        case verdict
        case kev05
        case kev06
        case kai
        case lex
        case lfm350
        case jeff
        case nanojev

        public var title: String {
            switch self {
            case .gliclass: "GLiClass LUT8"
            case .laya: "Laya E8"
            case .gliner2Small: "GLiNER 2.5 small W8"
            case .gliner2Base: "GLiNER 2.5 base W8"
            case .gliner2Multilingual: "GLiNER 2.5 multilingual W8"
            case .verdict: "Verdict FP16"
            case .kev05: "Kev 0.5B FP16"
            case .kev06: "Kev 0.6B FP16"
            case .kai: "Decision 1.0 Kai FP16"
            case .lex: "Decision 1.0 Lex FP16"
            case .lfm350: "LFM2.5-350M-RLCD FP16"
            case .jeff: "Jeff FP16"
            case .nanojev: "NanoJev FP16 (local)"
            }
        }

        var published: PublishedCoreMLModel? {
            switch self {
            case .kev05: .kev05
            case .kev06: .kev06
            case .kai: .kai
            case .lex: .lex
            case .lfm350: .lfm350
            case .jeff: .jeff
            case .nanojev: .nanojev
            case .gliclass, .laya, .gliner2Small, .gliner2Base, .gliner2Multilingual, .verdict: nil
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
        case verdict(VerdictManager)
        case published(PublishedCoreMLManager)
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
        case .gliner2Small, .gliner2Base, .gliner2Multilingual:
            policy = try await loadGLiNER2(model)
        case .verdict:
            policy = try await loadVerdict()
        case .kev05, .kev06, .kai, .lex, .lfm350, .jeff, .nanojev:
            policy = try await loadPublished(model)
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
        let variant: GLiNER2Variant
        let directoryKey: String
        switch model {
        case .gliner2Small: (variant, directoryKey) = (.small, "GLINER2_SMALL_MODEL_DIR")
        case .gliner2Base: (variant, directoryKey) = (.base, "GLINER2_BASE_MODEL_DIR")
        default: (variant, directoryKey) = (.multilingual, "GLINER2_MULTI_MODEL_DIR")
        }
        let manager: GLiNER2Manager
        if let path = ProcessInfo.processInfo.environment[directoryKey], !path.isEmpty {
            manager = try await GLiNER2Manager.load(from: URL(fileURLWithPath: path), variant: variant)
        } else {
            manager = try await GLiNER2Manager.load(variant: variant)
        }
        return FlappyBirdPolicy(model: model, backend: .gliner2(manager))
    }

    private static func loadVerdict() async throws -> FlappyBirdPolicy {
        let configuration = VerdictManager.Configuration(lengths: [128])
        let manager: VerdictManager
        if let path = ProcessInfo.processInfo.environment["VERDICT_MODEL_DIR"], !path.isEmpty {
            manager = try await VerdictManager.load(from: URL(fileURLWithPath: path), configuration: configuration)
        } else {
            manager = try await VerdictManager.load(configuration: configuration)
        }
        return FlappyBirdPolicy(model: .verdict, backend: .verdict(manager))
    }

    /// NanoJev weights are not redistributed: it needs `NANOJEV_MODEL_DIR` and `NANOJEV_PYTHON`.
    private static func loadPublished(_ model: Model) async throws -> FlappyBirdPolicy {
        guard let published = model.published else { throw LayaError.invalidOutput("\(model) is not bridged") }
        let manager: PublishedCoreMLManager
        if published == .nanojev {
            let environment = ProcessInfo.processInfo.environment
            guard let directory = environment["NANOJEV_MODEL_DIR"], let python = environment["NANOJEV_PYTHON"],
                !directory.isEmpty, !python.isEmpty
            else { throw PublishedCoreMLError.missingAsset("NANOJEV_MODEL_DIR and NANOJEV_PYTHON") }
            manager = try await PublishedCoreMLManager.start(
                model: .nanojev, from: URL(fileURLWithPath: directory), python: URL(fileURLWithPath: python))
        } else {
            manager = try await PublishedCoreMLManager.load(model: published)
        }
        return FlappyBirdPolicy(model: model, backend: .published(manager))
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
        case .verdict(let manager):
            let options = zip(actions, labels).map { VerdictQuestion.Option(id: $0.rawValue, description: $1) }
            let answer = try await manager.answer(
                context: game.observation, question: .choice(question: Self.instruction, options: options))
            // Drop abstention; the game needs one of the two actions.
            let substantive = actions.map { action in
                answer.candidateIDs.firstIndex(of: action.rawValue).map { answer.probabilities[$0] } ?? 0
            }
            probabilities = substantive.map(Float.init)
            selectedIndex = substantive.indices.max { substantive[$0] < substantive[$1] } ?? 0
            tokenCount = answer.tokenCount
            truncated = false
        case .published(let manager):
            (probabilities, selectedIndex, tokenCount) = try await Self.published(
                manager, game: game, actions: actions, labels: labels)
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

    private static func published(
        _ manager: PublishedCoreMLManager, game: FlappyBird, actions: [FlappyBird.Action], labels: [String]
    ) async throws -> (probabilities: [Float], selectedIndex: Int, tokens: Int) {
        let names = actions.map(\.rawValue)
        switch manager.model.family {
        case .systemOne:
            let options = zip(names, labels).map { DecisionOption($0, $1) }
            let response = try await manager.evaluate(
                SystemOneRequest(
                    state: game.observation, questions: [.choice("action", Self.instruction, options: options)]))
            guard case .choice(let selected, _, let distribution) = response["action"],
                let index = names.firstIndex(of: selected)
            else { throw PublishedCoreMLError.invalidResponse("No action choice") }
            return (distribution.map { Float($0.probability) }, index, response.inputTokens)
        case .constrainedSchema:
            let context = Self.instruction + game.observation + " Options: " + labels.joined(separator: "; ") + "."
            let decision = try await manager.constrained(context: context, fields: [.oneOf("action", names)])
            guard case .string(let selected) = decision["action"], let index = names.firstIndex(of: selected),
                let scores = decision.candidates.first?.scores
            else { throw PublishedCoreMLError.invalidResponse("No action value") }
            let logLikelihoods = names.map { name in
                scores.first { $0.value == .string(name) }?.logLikelihood ?? -.infinity
            }
            let peak = logLikelihoods.max() ?? 0
            let weights = logLikelihoods.map { exp($0 - peak) }
            let total = weights.reduce(0, +)
            return (weights.map { Float($0 / total) }, index, 0)
        case .labelClassification:
            let answer = try await manager.classify(
                text: game.observation, labels: labels, name: "Flappy Bird", description: Self.instruction)
            return (answer.probabilities.map(Float.init), answer.selectedIndex, 0)
        case .nanoJev:
            let options = zip(names, labels).map { DecisionOption($0, $1) }
            let decision = try await manager.decide(
                state: .string(game.observation),
                question: NanoJevQuestion(id: "action", instructions: Self.instruction, kind: .choice(options)))
            guard let index = names.firstIndex(of: decision.selectedID) else {
                throw PublishedCoreMLError.invalidResponse("No action choice")
            }
            let probabilities = names.map { name in
                decision.candidateIDs.firstIndex(of: name).map { Float(decision.probabilities[$0]) } ?? 0
            }
            return (probabilities, index, 0)
        }
    }
}
