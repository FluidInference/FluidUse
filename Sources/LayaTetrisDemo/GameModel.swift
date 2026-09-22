import FluidUse
import Foundation
import LayaTetris
import SwiftUI

/// One scored landing, shown in the decision console.
struct ScoredCandidate: Identifiable {
    let id: Int
    let candidate: TetrisGame.Candidate
    let sentence: String
    let probability: Float
    let milliseconds: Double
}

/// Drives the game loop for laya, GLiClass, and the non-model controls.
@MainActor
final class GameModel: ObservableObject {
    enum Policy: String, CaseIterable, Identifiable {
        case gliclass, laya, heuristic, random
        var id: String { rawValue }
    }

    // Model
    @Published private(set) var manager: LayaManager?
    @Published private(set) var gliClassManager: GLiClassManager?
    @Published private(set) var loadStatus = "Not loaded"
    @Published private(set) var isLoading = false
    @Published var errorMessage: String?

    // Game
    @Published private(set) var board: [[Bool]] = TetrisGame(seed: 1).board
    @Published private(set) var currentPiece = "–"
    @Published private(set) var candidates: [ScoredCandidate] = []
    @Published private(set) var chosen: TetrisGame.Candidate?
    @Published private(set) var evaluating: TetrisGame.Candidate?
    @Published private(set) var pieces = 0
    @Published private(set) var lines = 0
    @Published private(set) var isOver = false
    @Published private(set) var isRunning = false
    @Published private(set) var log: [String] = []

    // Stats
    @Published private(set) var decisions = 0
    @Published private(set) var medianMs = 0.0
    @Published private(set) var lastMs = 0.0
    @Published private(set) var decisionsPerMinute = 0.0
    @Published private(set) var promptTokens = 0
    @Published private(set) var bucket = 0
    /// Wall-clock time this game has been running, paused time excluded.
    @Published private(set) var elapsedSeconds: TimeInterval = 0
    @Published private(set) var gamesPlayed = 0
    /// Pieces and lines across every game in this run, including the one in progress.
    @Published private(set) var totalPieces = 0
    @Published private(set) var totalLines = 0

    // Controls
    @Published var policy: Policy = .gliclass
    /// Harness on: landings that bury a cell are withheld when a clean one exists, and the
    /// landing description stays discriminative on a tall board. 7.5x the pieces, measured.
    @Published var harness = true
    /// Re-rank the strongest landings by how good the board they leave is for the next piece.
    /// Measured worse than greedy on every combine tried (444 pieces against 581), because the
    /// model judges a hypothetical future board far less well than the move in front of it.
    /// Kept as a control. On average it plays worse (444 pieces against 581), but it spends far
    /// more model calls per piece, and on a good seed that buys a much longer single game:
    /// seed 24 runs 2,263 pieces and 896 lines over 50,487 calls, about four minutes.
    /// Off by default: greedy is the better player, and 1,199 pieces in 31 s is the honest demo.
    @Published var lookahead = 0
    /// Keep playing after a top-out: a new board on the next seed, totals carried forward.
    /// Enabled for the visual demo so a run continues even when one seed tops out early.
    @Published var marathon = true
    @Published var seed: UInt64 = 1
    /// Extra delay per scored candidate so the scoring can be watched; 0 runs flat out.
    @Published var stepDelayMs = 0.0
    /// Pause after each placed piece.
    @Published var pieceDelayMs = 0.0

    private var game = TetrisGame(seed: 7)
    private var rng = SplitMix64(seed: 8)
    private var pendingPiece: TetrisGame.Piece?
    private var task: Task<Void, Never>?
    /// Bumped by every reset/start so a cancelled run's late results are ignored.
    private var generation = 0
    private var latencies: [Double] = []
    private var runStart = Date()
    private var accumulatedSeconds: TimeInterval = 0
    private var clock: Task<Void, Never>?
    private var runDecisionSeconds = 0.0

    static let question = LayaTetris.question

    /// End-to-end model time per placed piece. This is the comparable speed number: GLiClass
    /// usually needs one encoder call while laya scores several surviving landings separately.
    var modelMillisecondsPerPiece: Double {
        totalPieces > 0 ? runDecisionSeconds * 1000 / Double(totalPieces) : 0
    }

    var hasLoadedModel: Bool {
        switch policy {
        case .gliclass: return gliClassManager != nil
        case .laya: return manager != nil
        case .heuristic, .random: return true
        }
    }

    /// `m:ss.t` so a demo clip reads a running clock rather than a raw double.
    var elapsedText: String {
        let minutes = Int(elapsedSeconds) / 60
        let seconds = elapsedSeconds - Double(minutes * 60)
        return String(format: "%d:%04.1f", minutes, seconds)  // 0:07.4
    }

    /// `LAYA_DEMO_AUTOLOAD=1` loads the model on launch; `LAYA_DEMO_AUTORUN=1` loads and plays without clicks; `LAYA_DEMO_QUIT_AFTER=<s>`
    /// prints the stats and exits, which is how the demo is smoke-tested headlessly.
    func applyEnvironment() {
        let environment = ProcessInfo.processInfo.environment
        if environment["TETRIS_POLICY"] == "laya" { policy = .laya }
        if let seedText = environment["LAYA_DEMO_SEED"], let value = UInt64(seedText) { seed = value }
        if let text = environment["LAYA_DEMO_LOOKAHEAD"], let value = Int(text) { lookahead = value }
        if environment["LAYA_DEMO_AUTOLOAD"] == "1" { loadModel() }
        guard environment["LAYA_DEMO_AUTORUN"] == "1" else { return }
        loadModel()
        Task {
            while !hasLoadedModel && isLoading { try? await Task.sleep(nanoseconds: 100_000_000) }
            guard hasLoadedModel else {
                if environment["LAYA_DEMO_QUIT_AFTER"] != nil {
                    print("autorun: model load failed: \(errorMessage ?? "unknown error")")
                    exit(1)
                }
                return
            }
            reset()
            toggle()
            if environment["LAYA_DEMO_STRESS"] == "1" {
                // Pause / reset / play every second to shake out state bugs.
                for _ in 0..<20 {
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                    if isRunning { toggle() }
                    reset()
                    toggle()
                    print("stress: reset ok, pieces \(pieces)")
                }
            }
            if let quitText = environment["LAYA_DEMO_QUIT_AFTER"], let seconds = Double(quitText) {
                try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                print(
                    String(
                        format:
                            "autorun: %d games · %d pieces · %d lines · %d decisions · median %.2f ms · %.0f decisions/min · %@",
                        gamesPlayed + 1, totalPieces, totalLines, decisions, medianMs, decisionsPerMinute,
                        (isOver ? "topped out" : "still playing") as NSString))
                exit(0)
            }
        }
    }

    func loadModel() {
        guard !isLoading, !hasLoadedModel else { return }
        isLoading = true
        loadStatus = "Loading \(policy.rawValue) (128-token bucket, CPU + Neural Engine)…"
        Task {
            do {
                let started = Date()
                switch policy {
                case .gliclass:
                    guard let directory = ProcessInfo.processInfo.environment["GLICLASS_MODEL_DIR"],
                        !directory.isEmpty
                    else { throw GLiClassError.invalidAsset("Set GLICLASS_MODEL_DIR for the GLiClass demo") }
                    let precision = ProcessInfo.processInfo.environment["GLICLASS_PRECISION"] ?? "fp16"
                    let loaded = try await GLiClassManager.load(
                        from: URL(fileURLWithPath: directory),
                        configuration: .init(lengths: [128], precision: precision))
                    _ = try await loaded.classify(
                        text: "The piece buries nothing and keeps the stack low.",
                        labels: ["a poor Tetris placement", "a clean Tetris placement"],
                        prompt: "Choose the better placement.")
                    gliClassManager = loaded
                    loadStatus = String(
                        format: "GLiClass Edge Apps v2 · %@ · L128 · CPU + ANE · loaded in %.1f s",
                        precision as NSString, Date().timeIntervalSince(started))
                case .laya:
                    let configuration = LayaManager.Configuration(lengths: [128])
                    let loaded: LayaManager
                    if let directory = ProcessInfo.processInfo.environment["LAYA_MODEL_DIR"], !directory.isEmpty {
                        loaded = try await LayaManager.load(
                            from: URL(fileURLWithPath: directory), configuration: configuration)
                    } else {
                        loaded = try await LayaManager.load(configuration: configuration)
                    }
                    _ = try await loaded.answer(state: "The piece leaves no holes.", question: Self.question)
                    manager = loaded
                    loadStatus = String(
                        format: "laya-multilingual · L128 · CPU + ANE · loaded in %.1f s",
                        Date().timeIntervalSince(started))
                case .heuristic, .random:
                    break
                }
            } catch {
                errorMessage = error.localizedDescription
                loadStatus = "Load failed"
            }
            isLoading = false
        }
    }

    func reset() {
        task?.cancel()
        task = nil
        generation += 1
        isRunning = false
        game = TetrisGame(seed: seed)
        pendingPiece = nil
        rng = SplitMix64(seed: seed &+ 1)
        board = game.board
        candidates = []
        chosen = nil
        evaluating = nil
        currentPiece = "–"
        pieces = 0
        lines = 0
        isOver = false
        decisions = 0
        latencies = []
        medianMs = 0
        lastMs = 0
        decisionsPerMinute = 0
        promptTokens = 0
        bucket = 0
        gamesPlayed = 0
        totalPieces = 0
        totalLines = 0
        runDecisionSeconds = 0
        pieceCallMs = []
        clock?.cancel()
        clock = nil
        accumulatedSeconds = 0
        elapsedSeconds = 0
        carriedLines = 0
        log = []
    }

    func toggle() {
        if isRunning {
            generation += 1
            task?.cancel()
            task = nil
            finishRun()
            return
        }
        guard hasLoadedModel else {
            errorMessage = "Load the model first, or pick the heuristic policy."
            return
        }
        // A fresh game must pick up the current seed. `game` is built in the property initialiser,
        // so without this, pressing Play before Reset silently replays the initialiser's seed.
        if isOver || pieces == 0 { reset() }
        generation += 1
        let run = generation
        isRunning = true
        runStart = Date()
        clock?.cancel()
        clock = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 100_000_000)
                guard !Task.isCancelled, let self, self.isRunning, self.generation == run else { return }
                self.elapsedSeconds = self.accumulatedSeconds + Date().timeIntervalSince(self.runStart)
            }
        }
        task = Task { [weak self] in
            await self?.play(run: run)
        }
    }

    private func finishRun() {
        guard isRunning else { return }
        accumulatedSeconds += Date().timeIntervalSince(runStart)
        elapsedSeconds = accumulatedSeconds
        clock?.cancel()
        clock = nil
        isRunning = false
        evaluating = nil
    }

    private func play(run: Int) async {
        defer {
            if run == generation { finishRun() }
        }
        while !Task.isCancelled, run == generation {
            // Resume scoring the same piece if Pause interrupted an inference call.
            if pendingPiece == nil { pendingPiece = game.spawn() }
            guard let piece = pendingPiece else { break }
            currentPiece = piece.name
            var all = game.candidates(for: piece)
            if harness {
                all = TetrisGame.shortlist(all)
            }
            guard !all.isEmpty else { break }
            candidates = []
            chosen = nil
            var best: (Float, TetrisGame.Candidate)?
            var scoredPairs: [(TetrisGame.Candidate, Float)] = []
            switch policy {
            case .gliclass:
                guard let gliClassManager else { break }
                let scored = Array(all.sorted { $0.features.heuristic > $1.features.heuristic }.prefix(2))
                if scored.count == 1 {
                    best = (Float(scored[0].features.heuristic), scored[0])
                    break
                }
                let sentences = scored.map { game.describe($0, piece: piece, style: harness ? .graded : .plain) }
                evaluating = scored[0]
                do {
                    let (answer, ms) = try await timedClassification(
                        manager: gliClassManager,
                        text: "Avoid holes, keep the stack low and smooth, and clear lines.", labels: sentences,
                        prompt: "Choose the best Tetris placement.")
                    guard run == generation, !Task.isCancelled else { return }
                    record(ms: ms, tokens: answer.tokenCount, bucket: answer.bucketLength)
                    for (index, candidate) in scored.enumerated() {
                        candidates.append(
                            ScoredCandidate(
                                id: candidate.id, candidate: candidate, sentence: sentences[index],
                                probability: answer.probabilities[index], milliseconds: ms))
                    }
                    best = (answer.probabilities[answer.selectedIndex], scored[answer.selectedIndex])
                } catch is CancellationError {
                    return
                } catch {
                    guard run == generation, !Task.isCancelled else { return }
                    errorMessage = error.localizedDescription
                    return
                }
            case .laya:
                guard let manager else { break }
                for candidate in all {
                    guard !Task.isCancelled, run == generation else { return }
                    evaluating = candidate
                    let sentence = game.describe(candidate, piece: piece, style: harness ? .graded : .plain)
                    do {
                        let (answer, ms) = try await timedAnswer(
                            manager: manager, state: sentence, question: Self.question)
                        // Paused or reset while the model was busy: drop the result quietly.
                        guard run == generation, !Task.isCancelled else { return }
                        record(ms: ms, tokens: answer.tokenCount, bucket: answer.bucketLength)
                        let p = answer.noul ?? 0
                        candidates.append(
                            ScoredCandidate(
                                id: candidate.id, candidate: candidate, sentence: sentence, probability: p,
                                milliseconds: ms))
                        scoredPairs.append((candidate, p))
                        if best == nil || p > best!.0 { best = (p, candidate) }
                    } catch is CancellationError {
                        return
                    } catch {
                        guard run == generation, !Task.isCancelled else { return }
                        errorMessage = error.localizedDescription
                        return
                    }
                    if stepDelayMs > 0 {
                        try? await Task.sleep(nanoseconds: UInt64(stepDelayMs * 1_000_000))
                    }
                }
            case .heuristic:
                let pick = all.max { $0.features.heuristic < $1.features.heuristic }!
                best = (Float(pick.features.heuristic), pick)
                candidates = all.map {
                    ScoredCandidate(
                        id: $0.id, candidate: $0,
                        sentence: game.describe($0, piece: piece, style: harness ? .graded : .plain),
                        probability: Float($0.features.heuristic), milliseconds: 0)
                }
            case .random:
                let pick = all[Int(rng.next() % UInt64(all.count))]
                best = (0, pick)
            }
            evaluating = nil
            guard run == generation, !Task.isCancelled else { return }
            guard var (score, pick) = best else { break }
            if policy == .laya, lookahead > 0, let manager, let next = game.nextPiece, scoredPairs.count > 1 {
                let top = scoredPairs.sorted { $0.1 > $1.1 }.prefix(lookahead)
                var bestPair: (Float, TetrisGame.Candidate)?
                for (candidate, own) in top {
                    var follow = game.candidates(for: next, on: candidate.board)
                    if harness {
                        follow = TetrisGame.shortlist(follow)
                    }
                    guard !follow.isEmpty else { continue }
                    var bestNext: Float = 0
                    for option in follow.prefix(12) {
                        if Task.isCancelled || run != generation { return }
                        let sentence = game.describe(option, piece: next, style: harness ? .graded : .plain)
                        do {
                            let (reply, ms) = try await timedAnswer(
                                manager: manager, state: sentence, question: Self.question)
                            guard run == generation, !Task.isCancelled else { return }
                            record(ms: ms, tokens: reply.tokenCount, bucket: reply.bucketLength)
                            bestNext = max(bestNext, reply.noul ?? 0)
                        } catch is CancellationError {
                            return
                        } catch {
                            guard run == generation, !Task.isCancelled else { return }
                            errorMessage = error.localizedDescription
                            return
                        }
                    }
                    let combined = own * bestNext
                    if bestPair == nil || combined > bestPair!.0 { bestPair = (combined, candidate) }
                }
                if let bestPair {
                    pick = bestPair.1
                    score = bestPair.0
                }
            }
            guard run == generation, !Task.isCancelled else { return }
            chosen = pick
            game.apply(pick)
            pendingPiece = nil
            pieces += 1
            lines = game.linesCleared
            totalPieces += 1
            totalLines = carriedLines + lines
            isOver = game.isOver
            let placed = String(
                format: "%@ → column %d rot %d · %@ %.3f · %d lines", piece.name as NSString, pick.column,
                pick.rotation,
                ([.laya, .gliclass].contains(policy) ? "model" : "score") as NSString, score, lines)
            log.insert(placed, at: 0)
            if log.count > 12 { log.removeLast() }
            printConsole(piece: piece, chosen: pick, score: score)
            board = game.board
            if isOver {
                guard marathon else { break }
                gamesPlayed += 1
                carriedLines = totalLines
                log.insert("game \(gamesPlayed) ended at \(pieces) pieces, \(lines) lines", at: 0)
                let nextSeed = seed &+ UInt64(gamesPlayed)
                game = TetrisGame(seed: nextSeed)
                rng = SplitMix64(seed: nextSeed &+ 1)
                board = game.board
                pieces = 0
                lines = 0
                isOver = false
                candidates = []
                chosen = nil
            }
            if pieceDelayMs > 0 {
                try? await Task.sleep(nanoseconds: UInt64(pieceDelayMs * 1_000_000))
            }
            // Control policies have no inference await. Yield even at zero delay so Pause,
            // Reset and the clock remain responsive during a marathon.
            await Task.yield()
            guard run == generation, !Task.isCancelled else { return }
        }
        guard run == generation, !Task.isCancelled else { return }
        if isOver { log.insert("Topped out after \(pieces) pieces, \(lines) lines.", at: 0) }
    }

    /// Measure away from MainActor so SwiftUI rendering time is not reported as model latency.
    private func timedClassification(
        manager: GLiClassManager, text: String, labels: [String], prompt: String
    ) async throws -> (GLiClassAnswer, Double) {
        let inference = Task.detached(priority: .userInitiated) {
            let started = DispatchTime.now().uptimeNanoseconds
            let answer = try await manager.classify(text: text, labels: labels, prompt: prompt)
            return (answer, Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6)
        }
        return try await withTaskCancellationHandler {
            try await inference.value
        } onCancel: {
            inference.cancel()
        }
    }

    private func timedAnswer(
        manager: LayaManager, state: String, question: LayaQuestion
    ) async throws -> (LayaAnswer, Double) {
        let inference = Task.detached(priority: .userInitiated) {
            let started = DispatchTime.now().uptimeNanoseconds
            let answer = try await manager.answer(state: state, question: question)
            return (answer, Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6)
        }
        return try await withTaskCancellationHandler {
            try await inference.value
        } onCancel: {
            inference.cancel()
        }
    }

    /// Terminal console for presentations (tmux next to asitop): one block per placed piece,
    /// the model-call line in red like the CUA-S1-FORMS demo.
    private var pieceCallMs: [Double] = []
    private var carriedLines = 0

    private func printConsole(piece: TetrisGame.Piece, chosen: TetrisGame.Candidate, score: Float) {
        if isRunning { elapsedSeconds = accumulatedSeconds + Date().timeIntervalSince(runStart) }
        let cyan = "\u{1B}[36m"
        let red = "\u{1B}[1;31m"
        let yellow = "\u{1B}[33m"
        let dim = "\u{1B}[2m"
        let reset = "\u{1B}[0m"
        let policyName: String
        switch policy {
        case .gliclass: policyName = "GLiClass Edge Apps v2"
        case .laya: policyName = "laya-multilingual"
        case .heuristic: policyName = "heuristic control"
        case .random: policyName = "random control"
        }
        print("\(cyan)▶ \(policyName)\(reset) · \(piece.name) piece · \(candidates.count) landings scored")
        if policy == .laya || policy == .gliclass {
            let sorted = candidates.sorted { $0.probability > $1.probability }
            for scored in sorted.prefix(3) {
                let marker = scored.candidate.id == chosen.id ? "→" : " "
                print(
                    "  \(marker) \(yellow)clean \(String(format: "%.1f%%", scored.probability * 100))\(reset)  \(scored.sentence)"
                )
            }
            if !pieceCallMs.isEmpty {
                let median = pieceCallMs.sorted()[pieceCallMs.count / 2]
                print(
                    "  \(red)model call \(String(format: "%.2f", median)) ms on Neural Engine\(reset) \(dim)(\(pieceCallMs.count) calls, \(String(format: "%.0f", pieceCallMs.reduce(0, +))) ms for this piece)\(reset)"
                )
            }
        } else {
            print("  → column \(chosen.column) rot \(chosen.rotation) · score \(String(format: "%.2f", score))")
        }
        print(
            // The wall clock goes in the console as well as the window: a viewer needs to see that
            // the recording is real time and not sped up.
            "  \(dim)\(elapsedText) elapsed · lines \(lines) · pieces \(pieces) · "
                + "\(String(format: "%.0f", decisionsPerMinute)) decisions/min\(reset)"
        )
        pieceCallMs.removeAll(keepingCapacity: true)
    }

    private func record(ms: Double, tokens: Int, bucket: Int) {
        pieceCallMs.append(ms)
        decisions += 1
        lastMs = ms
        latencies.append(ms)
        runDecisionSeconds += ms / 1000
        if latencies.count % 16 == 0 || latencies.count < 16 {
            let sorted = latencies.sorted()
            medianMs = sorted[sorted.count / 2]
        }
        let running = accumulatedSeconds + Date().timeIntervalSince(runStart)
        decisionsPerMinute = running > 0 ? Double(decisions) / running * 60 : 0
        promptTokens = max(promptTokens, tokens)
        self.bucket = bucket
    }
}
