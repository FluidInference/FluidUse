import FluidUse
import Foundation
import Game2048

struct Game2048Command {
    private struct Options {
        var modelDirectory: String?
        var precision = "lut8"
        var layaModelDirectory: String?
        var layaPrecision = "e8"
        var policy = "gliclass"
        var games = 1
        var seed: UInt64 = 1
        var maximumMoves = 100_000
        var candidates = 2
        var modelMargin: Float = 0
        var lookahead = false
        var json = false
    }

    static func run(arguments: [String]) async {
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }
        do {
            try await benchmark(arguments: arguments)
        } catch {
            fputs("2048 failed: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }

    private static func parse(_ arguments: [String]) throws -> Options {
        var options = Options()
        var index = 0
        func value(_ flag: String) throws -> String {
            index += 1
            guard index < arguments.count else { throw GLiClassError.invalidAsset("\(flag) needs a value") }
            return arguments[index]
        }
        while index < arguments.count {
            switch arguments[index] {
            case "--model-dir": options.modelDirectory = try value("--model-dir")
            case "--precision": options.precision = try value("--precision")
            case "--laya-model-dir": options.layaModelDirectory = try value("--laya-model-dir")
            case "--laya-precision": options.layaPrecision = try value("--laya-precision")
            case "--policy": options.policy = try value("--policy")
            case "--games": options.games = Int(try value("--games")) ?? 0
            case "--seed": options.seed = UInt64(try value("--seed")) ?? options.seed
            case "--moves": options.maximumMoves = Int(try value("--moves")) ?? 0
            case "--candidates", "--gliclass-candidates": options.candidates = Int(try value("--candidates")) ?? 0
            case "--model-margin", "--gliclass-margin": options.modelMargin = Float(try value("--model-margin")) ?? -1
            case "--lookahead": options.lookahead = true
            case "--json": options.json = true
            default: throw GLiClassError.invalidAsset("Unknown argument \(arguments[index])")
            }
            index += 1
        }
        guard ["laya", "laya-choice", "gliclass", "heuristic", "random"].contains(options.policy) else {
            throw GLiClassError.invalidAsset("--policy must be laya, laya-choice, gliclass, heuristic, or random")
        }
        guard options.games > 0, options.maximumMoves > 0 else {
            throw GLiClassError.invalidAsset("--games and --moves must be positive")
        }
        guard (2...4).contains(options.candidates) else {
            throw GLiClassError.invalidAsset("--candidates must be between 2 and 4")
        }
        guard (0...1).contains(options.modelMargin) else {
            throw GLiClassError.invalidAsset("--model-margin must be between 0 and 1")
        }
        return options
    }

    private static func benchmark(arguments: [String]) async throws {
        let options = try parse(arguments)
        let manager: GLiClassManager?
        if options.policy == "gliclass" {
            let configuration = GLiClassManager.Configuration(lengths: [128], precision: options.precision)
            if let modelDirectory = options.modelDirectory {
                manager = try await GLiClassManager.load(
                    from: URL(fileURLWithPath: modelDirectory), configuration: configuration)
            } else {
                manager = try await GLiClassManager.load(configuration: configuration)
            }
        } else {
            manager = nil
        }
        let layaManager: LayaManager?
        if options.policy == "laya" || options.policy == "laya-choice" {
            let configuration = LayaManager.Configuration(lengths: [128], precision: options.layaPrecision)
            if let directory = options.layaModelDirectory {
                layaManager = try await LayaManager.load(
                    from: URL(fileURLWithPath: directory), configuration: configuration)
            } else {
                layaManager = try await LayaManager.load(configuration: configuration)
            }
        } else {
            layaManager = nil
        }

        var results: [[String: Any]] = []
        for gameIndex in 0..<options.games {
            let seed = options.seed &+ UInt64(gameIndex)
            var game = Game2048(seed: seed)
            var random = CommandRandom(seed: seed ^ 0x2048)
            var latencies: [Double] = []
            var heuristicAgreements = 0
            var comparisons = 0
            let started = Date()
            while !game.isOver, game.moves < options.maximumMoves {
                let legal = game.candidates()
                guard !legal.isEmpty else { break }
                let chosen: Game2048.Candidate
                switch options.policy {
                case "gliclass", "laya", "laya-choice":
                    let ranked = legal.sorted {
                        game.strategicScore($0, lookahead: options.lookahead)
                            > game.strategicScore($1, lookahead: options.lookahead)
                    }
                    let offered = Array(ranked.prefix(options.candidates))
                    if offered.count == 1 {
                        chosen = offered[0]
                    } else {
                        let probabilities: [Float]
                        if options.policy == "gliclass" {
                            guard let manager else { throw GLiClassError.invalidModel("GLiClass is not loaded") }
                            let before = DispatchTime.now().uptimeNanoseconds
                            let answer = try await manager.classify(
                                text: "Build the largest tile without filling the board.",
                                labels: offered.map(game.describe),
                                prompt:
                                    "Choose the safest 2048 swipe. Preserve empty cells, ordered high tiles, and merges."
                            )
                            latencies.append(Double(DispatchTime.now().uptimeNanoseconds - before) / 1e6)
                            probabilities = answer.probabilities
                        } else if options.policy == "laya-choice" {
                            guard let layaManager else { throw LayaError.invalidModel("Laya is not loaded") }
                            let choices = offered.enumerated().map {
                                LayaQuestion.Choice("move \($0.offset + 1)", description: game.describe($0.element))
                            }
                            let before = DispatchTime.now().uptimeNanoseconds
                            let answer = try await layaManager.answer(
                                state: "Build the largest 2048 tile without filling the board.",
                                question: .choice(
                                    "Choose the safest move. Preserve empty cells, ordered high tiles, and merges.",
                                    options: choices))
                            latencies.append(Double(DispatchTime.now().uptimeNanoseconds - before) / 1e6)
                            probabilities = answer.probabilities
                        } else {
                            guard let layaManager else { throw LayaError.invalidModel("Laya is not loaded") }
                            let question = LayaQuestion.noul(
                                "Is this a safe 2048 move that preserves empty space, keeps high tiles ordered in a corner, and enables future merges?"
                            )
                            var scores: [Float] = []
                            for candidate in offered {
                                let before = DispatchTime.now().uptimeNanoseconds
                                let answer = try await layaManager.answer(
                                    state: game.describe(candidate), question: question)
                                latencies.append(Double(DispatchTime.now().uptimeNanoseconds - before) / 1e6)
                                scores.append(answer.noul ?? 0)
                            }
                            probabilities = scores
                        }
                        comparisons += 1
                        let selected = probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
                        let runnerUp =
                            probabilities.indices.filter { $0 != selected }.map { probabilities[$0] }.max() ?? 0
                        let margin = probabilities[selected] - runnerUp
                        let index = selected == 0 || margin >= options.modelMargin ? selected : 0
                        if index == 0 { heuristicAgreements += 1 }
                        chosen = offered[index]
                    }
                case "heuristic":
                    chosen =
                        legal.max(by: {
                            game.strategicScore($0, lookahead: options.lookahead)
                                < game.strategicScore($1, lookahead: options.lookahead)
                        }) ?? legal[0]
                default:
                    chosen = legal[Int(random.next() % UInt64(legal.count))]
                }
                game.apply(chosen)
            }
            latencies.sort()
            let elapsed = Date().timeIntervalSince(started)
            results.append([
                "seed": seed,
                "score": game.score,
                "moves": game.moves,
                "maximum_tile": game.maximumTile,
                "game_over": game.isOver,
                "model_calls": latencies.count,
                "heuristic_agreement": comparisons > 0 ? Double(heuristicAgreements) / Double(comparisons) : 1,
                "median_ms": latencies.isEmpty ? 0 : latencies[latencies.count / 2],
                "elapsed_s": elapsed,
            ])
        }

        let scores = results.compactMap { $0["score"] as? Int }
        let moves = results.compactMap { $0["moves"] as? Int }
        let maximumTiles = results.compactMap { $0["maximum_tile"] as? Int }
        let payload: [String: Any] = [
            "policy": options.policy,
            "precision": options.policy.hasPrefix("laya") ? options.layaPrecision : options.precision,
            "candidates": options.candidates,
            "model_margin": options.modelMargin,
            "lookahead": options.lookahead,
            "games": results,
            "mean_score": Double(scores.reduce(0, +)) / Double(scores.count),
            "mean_moves": Double(moves.reduce(0, +)) / Double(moves.count),
            "best_tile": maximumTiles.max() ?? 0,
        ]
        if options.json {
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            print(String(decoding: data, as: UTF8.self))
        } else {
            for result in results {
                print(
                    "seed \(result["seed"] ?? 0): score \(result["score"] ?? 0), "
                        + "\(result["moves"] ?? 0) moves, max \(result["maximum_tile"] ?? 0)")
            }
            print(
                String(
                    format: "mean score %.0f · mean moves %.1f · best tile %d", payload["mean_score"] as? Double ?? 0,
                    payload["mean_moves"] as? Double ?? 0, payload["best_tile"] as? Int ?? 0))
        }
    }

    private static func printUsage() {
        print(
            """
            Usage: swift run -c release FluidUseLaya 2048 [--model-dir GLICLASS_DIR]
                       [--precision fp16|fp16-mask|lut8|lut6] [--laya-model-dir DIR]
                       [--laya-precision fp16|e8] [--policy laya|laya-choice|gliclass|heuristic|random]
                       [--games N] [--seed N] [--moves N] [--candidates 2|3|4]
                       [--model-margin 0...1] [--lookahead] [--json]

            Plays deterministic 4x4 2048. Every model receives the same heuristic shortlist;
            --lookahead enables one-move expectimax. GLiClass and laya-choice compare the shortlist
            in one pass; laya scores each candidate independently.
            """)
    }
}

private struct CommandRandom {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }
}
