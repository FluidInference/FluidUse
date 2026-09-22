import FluidUse
import Foundation
import Game2048
import SwiftUI

@MainActor
final class Game2048Model: ObservableObject {
    enum Policy: String, CaseIterable, Identifiable {
        case gliclass, heuristic, random
        var id: String { rawValue }
    }

    @Published private(set) var manager: GLiClassManager?
    @Published private(set) var loadStatus = "Not loaded"
    @Published private(set) var isLoading = false
    @Published var errorMessage: String?

    @Published private(set) var board = Game2048(seed: 1).board
    @Published private(set) var score = 0
    @Published private(set) var moves = 0
    @Published private(set) var maximumTile = 2
    @Published private(set) var modelCalls = 0
    @Published private(set) var modelMilliseconds = 0.0
    @Published private(set) var lastDirection: Game2048.Direction?
    @Published private(set) var elapsedSeconds: TimeInterval = 0
    @Published private(set) var gamesPlayed = 0
    @Published private(set) var isOver = false
    @Published private(set) var isRunning = false

    @Published var policy: Policy = .gliclass
    @Published var candidateCount = 2
    @Published var minimumMargin: Float = 0.50
    @Published var lookahead = true
    @Published var marathon = false
    @Published var seed: UInt64 = 46
    @Published var moveDelayMs = 0.0

    private var game = Game2048(seed: 1)
    private var random = DemoRandom(seed: 0x2048)
    private var task: Task<Void, Never>?
    private var clock: Task<Void, Never>?
    private var generation = 0
    private var runStart = Date()
    private var accumulatedSeconds: TimeInterval = 0

    var hasLoadedModel: Bool { policy != .gliclass || manager != nil }

    var modelMillisecondsPerMove: Double {
        moves > 0 ? modelMilliseconds / Double(moves) : 0
    }

    var elapsedText: String {
        let minutes = Int(elapsedSeconds) / 60
        let seconds = elapsedSeconds - Double(minutes * 60)
        return String(format: "%d:%04.1f", minutes, seconds)
    }

    func applyEnvironment() {
        let environment = ProcessInfo.processInfo.environment
        if let value = environment["GAME2048_SEED"].flatMap(UInt64.init) { seed = value }
        if let value = environment["GAME2048_CANDIDATES"].flatMap(Int.init), (2...4).contains(value) {
            candidateCount = value
        }
        if let value = environment["GAME2048_MARGIN"].flatMap(Float.init), (0...1).contains(value) {
            minimumMargin = value
        }
        if environment["GAME2048_AUTOLOAD"] == "1" { loadModel() }
        guard environment["GAME2048_AUTORUN"] == "1" else { return }
        loadModel()
        Task {
            while !hasLoadedModel && isLoading { try? await Task.sleep(for: .milliseconds(100)) }
            guard hasLoadedModel else { return }
            reset()
            toggle()
            if let text = environment["GAME2048_QUIT_AFTER"], let seconds = Double(text) {
                try? await Task.sleep(for: .seconds(seconds))
                print(
                    String(
                        format:
                            "autorun: score %d · %d moves · max %d · %d calls · %.2f model ms/move · %.2f s · %@",
                        score, moves, maximumTile, modelCalls, modelMillisecondsPerMove, elapsedSeconds,
                        (isOver ? "game over" : "still playing") as NSString))
                exit(0)
            }
        }
    }

    func loadModel() {
        guard !isLoading, manager == nil else { return }
        guard policy == .gliclass else { return }
        isLoading = true
        loadStatus = "Loading GLiClass Edge Apps v2…"
        Task {
            do {
                guard let directory = ProcessInfo.processInfo.environment["GLICLASS_MODEL_DIR"], !directory.isEmpty
                else { throw GLiClassError.invalidAsset("Set GLICLASS_MODEL_DIR for the 2048 demo") }
                let precision = ProcessInfo.processInfo.environment["GLICLASS_PRECISION"] ?? "lut8"
                let started = Date()
                let loaded = try await GLiClassManager.load(
                    from: URL(fileURLWithPath: directory),
                    configuration: .init(lengths: [128], precision: precision))
                _ = try await loaded.classify(
                    text: "Build the largest tile without filling the board.",
                    labels: ["swipe left: 8 empty cells", "swipe right: 5 empty cells"],
                    prompt: "Choose the safest 2048 swipe.")
                manager = loaded
                loadStatus = String(
                    format: "GLiClass Edge Apps v2 · %@ · L128 · CPU + ANE · loaded in %.1f s",
                    precision as NSString, Date().timeIntervalSince(started))
            } catch {
                errorMessage = error.localizedDescription
                loadStatus = "Load failed"
            }
            isLoading = false
        }
    }

    func reset() {
        task?.cancel()
        clock?.cancel()
        task = nil
        clock = nil
        generation += 1
        game = Game2048(seed: seed)
        random = DemoRandom(seed: seed ^ 0x2048)
        board = game.board
        score = game.score
        moves = game.moves
        maximumTile = game.maximumTile
        modelCalls = 0
        modelMilliseconds = 0
        lastDirection = nil
        elapsedSeconds = 0
        accumulatedSeconds = 0
        gamesPlayed = 0
        isOver = false
        isRunning = false
    }

    func toggle() {
        if isRunning {
            generation += 1
            task?.cancel()
            finishRun()
            return
        }
        guard hasLoadedModel else {
            errorMessage = "Load the model first, or choose a control policy."
            return
        }
        if isOver || moves == 0 { reset() }
        generation += 1
        let run = generation
        isRunning = true
        runStart = Date()
        clock = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(100))
                guard let self, self.isRunning, self.generation == run else { return }
                self.elapsedSeconds = self.accumulatedSeconds + Date().timeIntervalSince(self.runStart)
            }
        }
        task = Task { [weak self] in await self?.play(run: run) }
    }

    private func finishRun() {
        guard isRunning else { return }
        accumulatedSeconds += Date().timeIntervalSince(runStart)
        elapsedSeconds = accumulatedSeconds
        isRunning = false
        clock?.cancel()
        clock = nil
    }

    private func play(run: Int) async {
        defer {
            if generation == run { finishRun() }
        }
        while !Task.isCancelled, generation == run {
            let legal = game.candidates()
            guard !legal.isEmpty else { break }
            let chosen: Game2048.Candidate
            switch policy {
            case .gliclass:
                guard let manager else { return }
                let offered = Array(
                    legal.sorted {
                        game.strategicScore($0, lookahead: lookahead) > game.strategicScore($1, lookahead: lookahead)
                    }.prefix(candidateCount))
                if offered.count == 1 {
                    chosen = offered[0]
                } else {
                    do {
                        let labels = offered.map(game.describe)
                        let (answer, milliseconds) = try await timedClassification(
                            manager: manager, labels: labels)
                        guard generation == run, !Task.isCancelled else { return }
                        modelCalls += 1
                        modelMilliseconds += milliseconds
                        let selected = answer.selectedIndex
                        let runnerUp =
                            answer.probabilities.indices.filter { $0 != selected }
                            .map { answer.probabilities[$0] }.max() ?? 0
                        let margin = answer.probabilities[selected] - runnerUp
                        chosen = offered[selected == 0 || margin >= minimumMargin ? selected : 0]
                    } catch is CancellationError {
                        return
                    } catch {
                        errorMessage = error.localizedDescription
                        return
                    }
                }
            case .heuristic:
                guard
                    let best = legal.max(by: {
                        game.strategicScore($0, lookahead: lookahead) < game.strategicScore($1, lookahead: lookahead)
                    })
                else { return }
                chosen = best
            case .random:
                chosen = legal[Int(random.next() % UInt64(legal.count))]
            }
            game.apply(chosen)
            board = game.board
            score = game.score
            moves = game.moves
            maximumTile = game.maximumTile
            lastDirection = chosen.direction
            isOver = game.isOver
            if isOver {
                guard marathon else { break }
                gamesPlayed += 1
                let nextSeed = seed &+ UInt64(gamesPlayed)
                game = Game2048(seed: nextSeed)
                random = DemoRandom(seed: nextSeed ^ 0x2048)
                board = game.board
                score = game.score
                moves = game.moves
                maximumTile = game.maximumTile
                modelCalls = 0
                modelMilliseconds = 0
                isOver = false
            }
            if moveDelayMs > 0 { try? await Task.sleep(for: .milliseconds(moveDelayMs)) }
            await Task.yield()
        }
    }

    private func timedClassification(
        manager: GLiClassManager, labels: [String]
    ) async throws -> (GLiClassAnswer, Double) {
        let inference = Task.detached(priority: .userInitiated) {
            let started = DispatchTime.now().uptimeNanoseconds
            let answer = try await manager.classify(
                text: "Build the largest tile without filling the board.", labels: labels,
                prompt: "Choose the safest 2048 swipe. Preserve empty cells, ordered high tiles, and merges.")
            return (answer, Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6)
        }
        return try await withTaskCancellationHandler {
            try await inference.value
        } onCancel: {
            inference.cancel()
        }
    }
}

private struct DemoRandom {
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
