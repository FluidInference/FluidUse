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
        var question: String?
        var shortlist = false
        var agreement = false
        var lookahead = 0
        var combine = "product"
        var describeStyle = TetrisGame.DescriptionStyle.plain
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
            case "--question": options.question = try value("--question")
            case "--shortlist": options.shortlist = true
            case "--agreement": options.agreement = true
            case "--lookahead": options.lookahead = Int(try value("--lookahead")) ?? 0
            case "--combine": options.combine = try value("--combine")
            case "--describe":
                guard let style = TetrisGame.DescriptionStyle(rawValue: try value("--describe")) else {
                    throw LayaError.invalidAsset("--describe must be plain or graded")
                }
                options.describeStyle = style
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

    /// Keep only landings that bury nothing, when any such landing exists.
    private static func shortlist(_ candidates: [TetrisGame.Candidate]) -> [TetrisGame.Candidate] {
        let clean = candidates.filter { $0.features.newHoles == 0 }
        return clean.isEmpty ? candidates : clean
    }

    private static func play(arguments: [String]) async throws {
        let options = try parse(arguments)
        let layaQuestion = options.question.map { LayaQuestion.noul($0) } ?? question
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
        var percentiles: [Double] = []
        var offered: [Int] = []
        var forced = 0
        var beforeFilter: [Int] = []
        var agreements = 0
        var agreementPieces = 0
        var tokenCounts: [Int] = []
        let started = Date()
        var pieces = 0
        while pieces < options.maxPieces, let piece = game.spawn() {
            let candidates = game.candidates(for: piece)
            guard !candidates.isEmpty else { break }
            var chosen: TetrisGame.Candidate
            switch options.policy {
            case "laya":
                guard let manager else { throw LayaError.invalidModel("The laya policy needs a loaded model") }
                var best: (score: Float, candidate: TetrisGame.Candidate)?
                var candidatesScored: [(TetrisGame.Candidate, Float)] = []
                // --shortlist: the harness enforces the hard constraint (never bury a cell when a
                // hole-free landing exists) and laya chooses among what survives, which is how a
                // System One model is meant to sit in a harness.
                let scored = options.shortlist ? shortlist(candidates) : candidates
                beforeFilter.append(candidates.count)
                offered.append(scored.count)
                if scored.count == 1 { forced += 1 }
                for candidate in scored {
                    let state = game.describe(candidate, piece: piece, style: options.describeStyle)
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    let answer = try await manager.answer(state: state, question: layaQuestion)
                    decisionTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
                    tokenCounts.append(answer.tokenCount)
                    let score = answer.noul ?? 0
                    if pieces < options.trace {
                        print(String(format: "  %.3f  %@", score, state))
                    }
                    candidatesScored.append((candidate, score))
                    if best == nil || score > best!.score { best = (score, candidate) }
                }
                chosen = best!.candidate
                // One-piece lookahead: re-rank the strongest landings by how good the board they
                // leave behind is for the piece that follows, which is the information a human has
                // and a one-shot judge does not.
                if options.lookahead > 0, let next = game.nextPiece, candidatesScored.count > 1 {
                    let top = candidatesScored.sorted { $0.1 > $1.1 }.prefix(options.lookahead)
                    var bestPair: (Float, TetrisGame.Candidate)?
                    for (candidate, own) in top {
                        var follow = game.candidates(for: next, on: candidate.board)
                        if options.shortlist { follow = shortlist(follow) }
                        guard !follow.isEmpty else { continue }
                        var bestNext: Float = 0
                        for option in follow.prefix(12) {
                            let sentence = game.describe(option, piece: next, style: options.describeStyle)
                            let t = DispatchTime.now().uptimeNanoseconds
                            let reply = try await manager.answer(state: sentence, question: layaQuestion)
                            decisionTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6)
                            tokenCounts.append(reply.tokenCount)
                            bestNext = max(bestNext, reply.noul ?? 0)
                        }
                        let combined: Float
                        switch options.combine {
                        case "min": combined = min(own, bestNext)  // no good move now AND no good reply
                        case "next": combined = bestNext  // rank purely by what it leaves behind
                        default: combined = own * bestNext
                        }
                        if bestPair == nil || combined > bestPair!.0 { bestPair = (combined, candidate) }
                    }
                    if let bestPair { chosen = bestPair.1 }
                }
                if options.agreement, scored.count > 1 {
                    // Where does laya's pick sit in the heuristic's ordering? 1.0 = the heuristic's own
                    // top choice, 0.5 = what picking at random would average.
                    let ranked = scored.sorted { $0.features.heuristic > $1.features.heuristic }
                    if let position = ranked.firstIndex(where: { $0.id == chosen.id }) {
                        percentiles.append(1.0 - Double(position) / Double(ranked.count - 1))
                    }
                    if ranked[0].id == chosen.id { agreements += 1 }
                    agreementPieces += 1
                }
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
                "landings_before_filter_mean": beforeFilter.isEmpty
                    ? 0 : Double(beforeFilter.reduce(0, +)) / Double(beforeFilter.count),
                "landings_offered_to_model_mean": offered.isEmpty
                    ? 0 : Double(offered.reduce(0, +)) / Double(offered.count),
                "forced_single_option_fraction": offered.isEmpty
                    ? 0 : Double(forced) / Double(offered.count),
                "heuristic_percentile_mean": percentiles.isEmpty
                    ? 0 : percentiles.reduce(0, +) / Double(percentiles.count),
                "heuristic_top_agreement": agreementPieces == 0
                    ? 0 : Double(agreements) / Double(agreementPieces),
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
            Usage: swift run FluidUseLaya tetris [--model-dir DIR] [--precision fp16|e8] [--lengths 128]
                                             [--pieces 200] [--seed 7] [--policy laya|heuristic|random]
                                             [--shortlist] [--describe plain|graded] [--lookahead N]
                                             [--combine product|min|next] [--agreement]
                                             [--show-every N] [--trace N] [--json]

            Plays headless 10x20 Tetris. With --policy laya (default) every legal landing is described in
            one sentence and scored by laya's P(clean); the best-scoring landing is played.

            The two flags that matter, and only together (76 -> 568 pieces over ten seeds):
              --shortlist        withhold landings that bury a cell when a clean landing exists
              --describe graded  wording that stays discriminative on a tall board; `plain` is the
                                 original, where every option read alike past 15 rows

            Measured alone, each is a regression: graded wording 48 pieces, shortlist 87, baseline 76.

              --lookahead N      also score the board each of the top N landings leaves for the next
                                 piece. Worse on average (445 vs 581) but ~4x the calls per piece
              --combine          how lookahead ranks: own score x best follow-up (default), min, or
                                 the follow-up alone
              --agreement        report where laya's pick sits in a Dellacherie ranking
            """)
    }
}
