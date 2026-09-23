import FluidUse
import Foundation

public enum DecisionModel: String, CaseIterable, Sendable {
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

    /// Largest option count the published package accepts; Kai/Lex Choice packages take three candidates.
    public var maximumOptions: Int {
        switch self {
        case .kai, .lex: 3
        case .nanojev: 4
        case .jeff, .gliner2Small, .gliner2Base, .gliner2Multilingual: 8
        case .gliclass, .verdict: 24
        case .laya, .kev05, .kev06, .lfm350: 32
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

/// One option: a short id (`name`) and the text the model reads (`label`).
public struct DecisionOptionText: Sendable, Equatable {
    public let name: String
    public let label: String

    public init(name: String, label: String) {
        self.name = name
        self.label = label
    }
}

public struct DecisionChoice: Sendable {
    /// Index into the options as presented.
    public let index: Int
    /// Per-option scores in presented order.
    public let probabilities: [Float]
    public let milliseconds: Double
    public let tokens: Int
}

/// One real Core ML choice among caller-described options, with no heuristic override.
public actor DecisionPolicy {
    private enum Backend: Sendable {
        case gliclass(GLiClassManager)
        case laya(LayaManager)
        case gliner2(GLiNER2Manager)
        case verdict(VerdictManager)
        case published(PublishedCoreMLManager)
    }

    private let backend: Backend
    public nonisolated let model: DecisionModel

    private init(model: DecisionModel, backend: Backend) {
        self.model = model
        self.backend = backend
    }

    /// Uses cached/published Core ML assets, or a model-specific local directory environment variable.
    public static func load(_ model: DecisionModel) async throws -> DecisionPolicy {
        let policy: DecisionPolicy
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
        return policy
    }

    private static func loadGLiClass() async throws -> DecisionPolicy {
        let configuration = GLiClassManager.Configuration(lengths: [128], precision: "lut8")
        let manager: GLiClassManager
        if let path = ProcessInfo.processInfo.environment["GLICLASS_MODEL_DIR"], !path.isEmpty {
            manager = try await GLiClassManager.load(
                from: URL(fileURLWithPath: path), configuration: configuration)
        } else {
            manager = try await GLiClassManager.load(configuration: configuration)
        }
        return DecisionPolicy(model: .gliclass, backend: .gliclass(manager))
    }

    private static func loadLaya() async throws -> DecisionPolicy {
        let configuration = LayaManager.Configuration(lengths: [128], precision: "e8")
        let manager: LayaManager
        if let path = ProcessInfo.processInfo.environment["LAYA_MODEL_DIR"], !path.isEmpty {
            manager = try await LayaManager.load(from: URL(fileURLWithPath: path), configuration: configuration)
        } else {
            manager = try await LayaManager.load(configuration: configuration)
        }
        return DecisionPolicy(model: .laya, backend: .laya(manager))
    }

    private static func loadGLiNER2(_ model: DecisionModel) async throws -> DecisionPolicy {
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
        return DecisionPolicy(model: model, backend: .gliner2(manager))
    }

    private static func loadVerdict() async throws -> DecisionPolicy {
        let configuration = VerdictManager.Configuration(lengths: [128])
        let manager: VerdictManager
        if let path = ProcessInfo.processInfo.environment["VERDICT_MODEL_DIR"], !path.isEmpty {
            manager = try await VerdictManager.load(from: URL(fileURLWithPath: path), configuration: configuration)
        } else {
            manager = try await VerdictManager.load(configuration: configuration)
        }
        return DecisionPolicy(model: .verdict, backend: .verdict(manager))
    }

    /// NanoJev weights are not redistributed: it needs `NANOJEV_MODEL_DIR` and `NANOJEV_PYTHON`.
    private static func loadPublished(_ model: DecisionModel) async throws -> DecisionPolicy {
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
        return DecisionPolicy(model: model, backend: .published(manager))
    }

    public func choose(
        state: String, instruction: String, options: [DecisionOptionText]
    ) async throws
        -> DecisionChoice
    {
        guard (1...model.maximumOptions).contains(options.count) else {
            throw LayaError.invalidOutput("\(model.title) accepts at most \(model.maximumOptions) options")
        }
        let names = options.map(\.name)
        let labels = options.map(\.label)
        let started = DispatchTime.now().uptimeNanoseconds
        let probabilities: [Float]
        let selectedIndex: Int
        let tokenCount: Int
        let truncated: Bool
        switch backend {
        case .gliclass(let manager):
            let answer = try await manager.classify(text: state, labels: labels, prompt: instruction)
            probabilities = answer.probabilities
            selectedIndex = answer.selectedIndex
            tokenCount = answer.tokenCount
            truncated = answer.textWasTruncated
        case .laya(let manager):
            let answer = try await manager.answer(
                state: state, question: .choice(instruction, options: labels))
            probabilities = answer.probabilities
            selectedIndex = answer.selectedIndex
            tokenCount = answer.tokenCount
            truncated = answer.stateWasTruncated
        case .gliner2(let manager):
            let answer = try await manager.classify(text: state, task: instruction, labels: labels)
            probabilities = answer.probabilities
            selectedIndex = answer.selectedIndex
            tokenCount = answer.tokenCount
            truncated = false
        case .verdict(let manager):
            let options = zip(names, labels).map { VerdictQuestion.Option(id: $0, description: $1) }
            let answer = try await manager.answer(
                context: state, question: .choice(question: instruction, options: options))
            // Drop abstention; the game needs one of the two actions.
            let substantive = names.map { name in
                answer.candidateIDs.firstIndex(of: name).map { answer.probabilities[$0] } ?? 0
            }
            probabilities = substantive.map(Float.init)
            selectedIndex = substantive.indices.max { substantive[$0] < substantive[$1] } ?? 0
            tokenCount = answer.tokenCount
            truncated = false
        case .published(let manager):
            (probabilities, selectedIndex, tokenCount) = try await Self.published(
                manager, state: state, instruction: instruction, names: names, labels: labels)
            truncated = false
        }
        let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
        guard !truncated else {
            throw LayaError.invalidOutput("State exceeded the model token budget")
        }
        return DecisionChoice(
            index: selectedIndex, probabilities: probabilities, milliseconds: milliseconds, tokens: tokenCount)
    }

    private static func published(
        _ manager: PublishedCoreMLManager, state: String, instruction: String, names: [String], labels: [String]
    ) async throws -> (probabilities: [Float], selectedIndex: Int, tokens: Int) {
        switch manager.model.family {
        case .systemOne:
            let options = zip(names, labels).map { DecisionOption($0, $1) }
            let response = try await manager.evaluate(
                SystemOneRequest(
                    state: state, questions: [.choice("action", instruction, options: options)]))
            guard case .choice(let selected, _, let distribution) = response["action"],
                let index = names.firstIndex(of: selected)
            else { throw PublishedCoreMLError.invalidResponse("No action choice") }
            return (distribution.map { Float($0.probability) }, index, response.inputTokens)
        case .constrainedSchema:
            let context = instruction + state + " Options: " + labels.joined(separator: "; ") + "."
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
                text: state, labels: labels, name: "Flappy Bird", description: instruction)
            return (answer.probabilities.map(Float.init), answer.selectedIndex, 0)
        case .nanoJev:
            let options = zip(names, labels).map { DecisionOption($0, $1) }
            let decision = try await manager.decide(
                state: .string(state),
                question: NanoJevQuestion(id: "action", instructions: instruction, kind: .choice(options)))
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
