import DecisionPolicy
import FlappyBird
import Foundation

/// Real Core ML choice between both projected actions, with no heuristic override.
public actor FlappyBirdPolicy {
    public typealias Model = DecisionModel

    public struct Decision: Sendable {
        public let action: FlappyBird.Action
        public let milliseconds: Double
        public let flapProbability: Float
        public let tokens: Int
    }

    private let policy: DecisionPolicy
    public nonisolated let model: Model
    private static let instruction =
        "Choose the Flappy Bird action that avoids collision and stays near the gap center. "

    private init(policy: DecisionPolicy) {
        self.policy = policy
        model = policy.model
    }

    /// Uses cached/published Core ML assets, or a model-specific local directory environment variable.
    public static func load(_ model: Model = .gliclass) async throws -> FlappyBirdPolicy {
        let policy = FlappyBirdPolicy(policy: try await DecisionPolicy.load(model))
        _ = try await policy.decide(game: FlappyBird(seed: 1), reversed: false)
        return policy
    }

    /// `reversed` changes presentation order, retaining the semantic action mapping.
    public func decide(game: FlappyBird, reversed: Bool) async throws -> Decision {
        let actions: [FlappyBird.Action] = reversed ? [.coast, .flap] : [.flap, .coast]
        let options = actions.map { DecisionOptionText(name: $0.rawValue, label: game.label(for: $0)) }
        let choice = try await policy.choose(state: game.observation, instruction: Self.instruction, options: options)
        return Decision(
            action: actions[choice.index], milliseconds: choice.milliseconds,
            flapProbability: choice.probabilities[reversed ? 1 : 0], tokens: choice.tokens)
    }
}
