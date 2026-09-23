import FlappyBird
import FlappyBirdPolicy
import Foundation

private struct Row: Encodable {
    let seed: UInt64
    let policy: String
    let score: Int
    let survivalSeconds: Double
    let modelCalls: Int
    let medianModelMs: Double?
    let maxTokens: Int
    let flapChoices: Int
    let rawFlapChoices: Int
    let overrides: Int
    let flapsWithFlapFirst: Int
    let flapsWithCoastFirst: Int
    let flapFirstCalls: Int
    let coastFirstCalls: Int
    let decisionFramesOver100Ms: Int
    let ended: Bool
}

@main
struct FlappyBirdCheck {
    static func main() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        let name = arguments.first ?? "gliclass"
        guard let model = FlappyBirdPolicy.Model(rawValue: name) else {
            let names = FlappyBirdPolicy.Model.allCases.map(\.rawValue).joined(separator: ", ")
            fputs("Use one of: \(names)\n", stderr)
            exit(2)
        }
        let rawOnly = arguments.contains("--raw-only")
        let seedArgument = arguments.first { $0.hasPrefix("--seeds=") }
        let seeds: [UInt64]
        if let seedArgument {
            let components = seedArgument.dropFirst("--seeds=".count).split(separator: ",")
            let parsed = components.compactMap { UInt64($0) }
            guard !parsed.isEmpty, parsed.count == components.count else {
                fputs("--seeds requires comma-separated unsigned integers\n", stderr)
                exit(2)
            }
            seeds = parsed
        } else {
            seeds = [1, 2]
        }
        let policy = try await FlappyBirdPolicy.load(model)
        var rows: [Row] = []
        for seed in seeds {
            rows.append(try await run(seed: seed, policy: policy, guarded: false))
            if !rawOnly { rows.append(try await run(seed: seed, policy: policy, guarded: true)) }
            rows.append(try await run(seed: seed, policy: nil, guarded: false))
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        print(String(decoding: try encoder.encode(rows), as: UTF8.self))
    }

    private static func run(seed: UInt64, policy: FlappyBirdPolicy?, guarded: Bool) async throws -> Row {
        var game = FlappyBird(seed: seed)
        var latencies: [Double] = []
        var maxTokens = 0
        var flapChoices = 0
        var rawFlapChoices = 0
        var overrides = 0
        var flapsWithFlapFirst = 0
        var flapsWithCoastFirst = 0
        var flapFirstCalls = 0
        var coastFirstCalls = 0
        var over100 = 0
        var decisions = 0
        while !game.isOver && game.frames < 1_200 {
            let action: FlappyBird.Action
            if let policy {
                let reversed = decisions % 2 == 1
                let response = try await policy.decide(game: game, reversed: reversed)
                action = guarded ? game.guardedAction(preferred: response.action) : response.action
                if response.action == .flap { rawFlapChoices += 1 }
                if action != response.action { overrides += 1 }
                latencies.append(response.milliseconds)
                maxTokens = max(maxTokens, response.tokens)
                if response.milliseconds > 100 { over100 += 1 }
                if reversed {
                    coastFirstCalls += 1
                    if response.action == .flap { flapsWithCoastFirst += 1 }
                } else {
                    flapFirstCalls += 1
                    if response.action == .flap { flapsWithFlapFirst += 1 }
                }
            } else {
                action = game.heuristicAction
            }
            decisions += 1
            if action == .flap { flapChoices += 1 }
            game.step(action)
            for _ in 1..<FlappyBird.decisionFrames where !game.isOver && game.frames < 1_200 {
                game.step()
            }
        }
        latencies.sort()
        let policyName = policy.map { "\($0.model.rawValue)_\(guarded ? "guarded" : "raw")" } ?? "heuristic"
        return Row(
            seed: seed, policy: policyName,
            score: game.score, survivalSeconds: game.seconds, modelCalls: latencies.count,
            medianModelMs: latencies.isEmpty ? nil : latencies[latencies.count / 2],
            maxTokens: maxTokens, flapChoices: flapChoices, rawFlapChoices: rawFlapChoices, overrides: overrides,
            flapsWithFlapFirst: flapsWithFlapFirst,
            flapsWithCoastFirst: flapsWithCoastFirst,
            flapFirstCalls: flapFirstCalls,
            coastFirstCalls: coastFirstCalls,
            decisionFramesOver100Ms: over100, ended: game.isOver)
    }
}
