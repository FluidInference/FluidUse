import DecisionPolicy
import Foundation
import LaneRunner
import LaneRunnerPolicy

private struct Row: Encodable {
    let seed: UInt64
    let policy: String
    let distance: Int
    let coins: Int
    let crash: String?
    let modelCalls: Int
    let medianModelMs: Double?
    let maxTokens: Int
    /// Decisions where the chosen move was unsafe although a safe one was offered.
    let unsafeChoices: Int
    /// How often each presented position was chosen.
    let positionCounts: [Int]
    let actionCounts: [String: Int]
}

@main
struct LaneRunnerCheck {
    static func main() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        let name = arguments.first ?? "heuristic"
        let seeds = parseList(arguments, "--seeds=") ?? [1, 2, 3, 4]
        let maxRows = parseList(arguments, "--max-rows=")?.first.map(Int.init) ?? 200
        let policy: LaneRunnerPolicy?
        switch name {
        case "heuristic", "random":
            policy = nil
        default:
            guard let model = DecisionModel(rawValue: name) else {
                let names = (["heuristic", "random"] + DecisionModel.allCases.map(\.rawValue))
                fputs("Use one of: \(names.joined(separator: ", "))\n", stderr)
                exit(2)
            }
            policy = try await LaneRunnerPolicy.load(model)
        }
        var rows: [Row] = []
        for seed in seeds {
            rows.append(try await run(seed: seed, name: name, policy: policy, maxRows: maxRows))
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        print(String(decoding: try encoder.encode(rows), as: UTF8.self))
    }

    private static func parseList(_ arguments: [String], _ prefix: String) -> [UInt64]? {
        guard let argument = arguments.first(where: { $0.hasPrefix(prefix) }) else { return nil }
        let components = argument.dropFirst(prefix.count).split(separator: ",")
        let parsed = components.compactMap { UInt64($0) }
        guard !parsed.isEmpty, parsed.count == components.count else {
            fputs("\(prefix) requires comma-separated unsigned integers\n", stderr)
            exit(2)
        }
        return parsed
    }

    private static func run(seed: UInt64, name: String, policy: LaneRunnerPolicy?, maxRows: Int) async throws -> Row {
        var game = LaneRunner(seed: seed)
        var random = seed &* 0x2545_F491_4F6C_DD1D | 1
        var latencies: [Double] = []
        var maxTokens = 0
        var unsafe = 0
        var positions = Array(repeating: 0, count: LaneRunner.Action.allCases.count)
        var actions: [String: Int] = [:]
        var decisions = 0
        while !game.isOver && game.distance < maxRows {
            let action: LaneRunner.Action
            if let policy {
                let decision = try await policy.decide(game: game, rotation: decisions)
                action = decision.action
                positions[decision.position] += 1
                latencies.append(decision.milliseconds)
                maxTokens = max(maxTokens, decision.tokens)
            } else if name == "random" {
                random ^= random << 13
                random ^= random >> 7
                random ^= random << 17
                action = game.legalActions[Int(random % UInt64(game.legalActions.count))]
            } else {
                action = game.heuristicAction
            }
            if !game.isSafe(action) && game.legalActions.contains(where: game.isSafe) { unsafe += 1 }
            actions[action.rawValue, default: 0] += 1
            decisions += 1
            game.step(action)
        }
        latencies.sort()
        return Row(
            seed: seed, policy: name, distance: game.distance, coins: game.coins, crash: game.crash,
            modelCalls: latencies.count, medianModelMs: latencies.isEmpty ? nil : latencies[latencies.count / 2],
            maxTokens: maxTokens, unsafeChoices: unsafe, positionCounts: positions, actionCounts: actions)
    }
}
