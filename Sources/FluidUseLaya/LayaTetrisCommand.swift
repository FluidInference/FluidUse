import FluidAudio
import FluidUse
import Foundation
import LayaTetris

/// `laya-tetris`: headless Tetris played by laya decisions.
///
/// Every legal landing of the current piece is described in one sentence and laya answers a
/// `noul` question, "Is this a clean placement?"; the highest P(true) wins. The game reports lines
/// cleared, pieces placed, and per-decision latency, which is the point: each decision is one
/// fixed-shape encoder pass on device. A feature-weighted heuristic policy is available as a baseline.
struct LayaTetrisCommand {
    private static let logger = AppLogger(category: "LayaTetris")

    static func run(arguments: [String]) async {
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }
        do {
            try await play(arguments: arguments)
        } catch {
            logger.error("laya-tetris failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    // MARK: - Options

    private struct Options {
        var modelDirectory: String?
        var precision = "fp16"
        var lengths: [Int] = [128]
        var maxPieces = 200
        var seed: UInt64 = 7
        var policy = "laya"
        var showEvery = 0
        var trace = 0
        var json = false
    }

    private static func parse(_ arguments: [String]) throws -> Options {
        var options = Options()
        var index = 0
        func value(_ flag: String) throws -> String {
            index += 1
            guard index < arguments.count else { throw LayaError.invalidAsset("\(flag) needs a value") }
            return arguments[index]
        }
        while index < arguments.count {
            switch arguments[index] {
            case "--model-dir": options.modelDirectory = try value("--model-dir")
            case "--precision": options.precision = try value("--precision")
            case "--lengths": options.lengths = try value("--lengths").split(separator: ",").compactMap { Int($0) }
            case "--pieces": options.maxPieces = Int(try value("--pieces")) ?? options.maxPieces
            case "--seed": options.seed = UInt64(try value("--seed")) ?? options.seed
            case "--policy": options.policy = try value("--policy")
            case "--show-every": options.showEvery = Int(try value("--show-every")) ?? 0
            case "--trace": options.trace = Int(try value("--trace")) ?? 0
            case "--json": options.json = true
            default: throw LayaError.invalidAsset("Unknown argument \(arguments[index])")
            }
            index += 1
        }
        guard ["laya", "heuristic", "random"].contains(options.policy) else {
            throw LayaError.invalidAsset("--policy must be laya, heuristic, or random")
        }
        return options
    }

    // MARK: - Game

    static let question = LayaTetris.question

    private static func play(arguments: [String]) async throws {
        let options = try parse(arguments)
        var manager: LayaManager?
        if options.policy == "laya" {
            let configuration = LayaManager.Configuration(lengths: options.lengths, precision: options.precision)
            if let directory = options.modelDirectory {
                manager = try await LayaManager.load(
                    from: URL(fileURLWithPath: directory), configuration: configuration)
            } else {
                manager = try await LayaManager.load(configuration: configuration)
            }
        }
        var game = TetrisGame(seed: options.seed)
        var rng = SplitMix64(seed: options.seed &+ 1)
        var decisionTimes: [Double] = []
        var tokenCounts: [Int] = []
        let started = Date()
        var pieces = 0
        while pieces < options.maxPieces, let piece = game.spawn() {
            let candidates = game.candidates(for: piece)
            guard !candidates.isEmpty else { break }
            let chosen: TetrisGame.Candidate
            switch options.policy {
            case "laya":
                guard let manager else { throw LayaError.invalidModel("The laya policy needs a loaded model") }
                var best: (score: Float, candidate: TetrisGame.Candidate)?
                for candidate in candidates {
                    let state = game.describe(candidate, piece: piece)
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    let answer = try await manager.answer(state: state, question: question)
                    decisionTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
                    tokenCounts.append(answer.tokenCount)
                    let score = answer.noul ?? 0
                    if pieces < options.trace {
                        print(String(format: "  %.3f  %@", score, state))
                    }
                    if best == nil || score > best!.score { best = (score, candidate) }
                }
                chosen = best!.candidate
                if pieces < options.trace {
                    print("  -> column \(chosen.column) rotation \(chosen.rotation)\n\(game.render(after: chosen))")
                }
            case "heuristic":
                chosen = candidates.max { $0.features.heuristic < $1.features.heuristic }!
            default:
                chosen = candidates[Int(rng.next() % UInt64(candidates.count))]
            }
            game.apply(chosen)
            pieces += 1
            if options.showEvery > 0, pieces % options.showEvery == 0 {
                print("piece \(pieces) · lines \(game.linesCleared)\n\(game.render())")
            }
        }
        let elapsed = Date().timeIntervalSince(started)
        decisionTimes.sort()
        let median = decisionTimes.isEmpty ? 0 : decisionTimes[decisionTimes.count / 2]
        let p95 =
            decisionTimes.isEmpty
            ? 0 : decisionTimes[min(decisionTimes.count - 1, Int(Double(decisionTimes.count) * 0.95))]
        let perMinute = elapsed > 0 ? Double(decisionTimes.count) / elapsed * 60 : 0
        if options.json {
            let payload: [String: Any] = [
                "policy": options.policy, "pieces": pieces, "lines": game.linesCleared, "game_over": game.isOver,
                "decisions": decisionTimes.count, "median_ms": median, "p95_ms": p95, "decisions_per_minute": perMinute,
                "elapsed_s": elapsed, "max_tokens": tokenCounts.max() ?? 0,
            ]
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            print(String(decoding: data, as: UTF8.self))
            return
        }
        print(game.render())
        print(
            "policy \(options.policy) · pieces \(pieces) · lines cleared \(game.linesCleared) · \(game.isOver ? "topped out" : "stopped at piece cap")"
        )
        if !decisionTimes.isEmpty {
            print(
                String(
                    format:
                        "%d laya decisions · median %.2f ms · p95 %.2f ms · %.0f decisions/min · prompts ≤ %d tokens · %.1f s total",
                    decisionTimes.count, median, p95, perMinute, tokenCounts.max() ?? 0, elapsed))
        } else {
            print(String(format: "%.2f s total", elapsed))
        }
    }

    private static func printUsage() {
        print(
            """
            Usage: swift run FluidUseLaya tetris [--model-dir DIR] [--precision fp16|e8] [--lengths 128] [--pieces 200] [--seed 7]
                                             [--policy laya|heuristic|random] [--show-every N] [--trace N] [--json]

            Plays headless 10x20 Tetris. With --policy laya (default) every legal landing is described in
            one sentence and scored by laya's P(clean); the best-scoring landing is played.
            """)
    }
}
