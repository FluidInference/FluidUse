import DecisionPolicy
import Foundation
import LaneRunner

/// Real Core ML choice among the legal runner moves, each labeled with its consequence.
public actor LaneRunnerPolicy {
    public typealias Model = DecisionModel

    public struct Decision: Sendable {
        public let action: LaneRunner.Action
        /// Position of the chosen option in the presented order.
        public let position: Int
        public let milliseconds: Double
        public let confidence: Float
        public let tokens: Int
    }

    private let policy: DecisionPolicy
    public nonisolated let model: Model
    private static let instruction = "Choose the runner move that avoids crashing and keeps a way through. "

    private init(policy: DecisionPolicy) {
        self.policy = policy
        model = policy.model
    }

    public static func load(_ model: Model) async throws -> LaneRunnerPolicy {
        let policy = LaneRunnerPolicy(policy: try await DecisionPolicy.load(model))
        _ = try await policy.decide(game: LaneRunner(seed: 1), rotation: 0)
        return policy
    }

    /// `rotation` cycles the option order so a fixed position preference cannot look like skill.
    public func decide(game: LaneRunner, rotation: Int) async throws -> Decision {
        let legal = game.legalActions
        let shift = rotation % legal.count
        let actions = Array(legal[shift...] + legal[..<shift])
        let options = actions.map { DecisionOptionText(name: $0.rawValue, label: game.label(for: $0)) }
        let choice = try await policy.choose(state: game.observation, instruction: Self.instruction, options: options)
        return Decision(
            action: actions[choice.index], position: choice.index, milliseconds: choice.milliseconds,
            confidence: choice.probabilities[choice.index], tokens: choice.tokens)
    }
}
