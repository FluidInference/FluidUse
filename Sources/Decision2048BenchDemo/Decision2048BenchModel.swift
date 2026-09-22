import FluidUse
import Foundation
import Game2048

struct Decision2048Side {
    let name: String
    let detail: String
    var board: [[Int]]
    var score = 0
    var moves = 0
    var maximumTile = 2
    var modelCalls = 0
    var modelMilliseconds = 0.0
    var isOver = false

    var modelMillisecondsPerMove: Double {
        moves > 0 ? modelMilliseconds / Double(moves) : 0
    }
}

@MainActor
final class Decision2048BenchModel: ObservableObject {
    @Published private(set) var gliClass: Decision2048Side
    @Published private(set) var laya: Decision2048Side
    @Published private(set) var loadStatus = "Models not loaded"
    @Published private(set) var isLoading = false
    @Published private(set) var isRunning = false
    @Published private(set) var elapsedSeconds: TimeInterval = 0
    @Published var errorMessage: String?
    @Published var seed: UInt64 = 3
    @Published var moveDelayMs = 10.0

    private var gliClassGame: Game2048
    private var layaGame: Game2048
    private var gliClassManager: GLiClassManager?
    private var layaManager: LayaManager?
    private var playTask: Task<Void, Never>?
    private var clockTask: Task<Void, Never>?
    private var generation = 0
    private var runStart = Date()
    private var accumulatedSeconds: TimeInterval = 0

    init() {
        let initial = Game2048(seed: 3)
        gliClassGame = initial
        layaGame = initial
        gliClass = Self.side(name: "GLiClass Edge Apps v2", detail: "32.7M · LUT8 · 1 call/move", game: initial)
        laya = Self.side(name: "Laya Multilingual", detail: "322M · E8 · 2 calls/move", game: initial)
    }

    var hasLoadedModels: Bool { gliClassManager != nil && layaManager != nil }

    var elapsedText: String {
        let minutes = Int(elapsedSeconds) / 60
        let seconds = elapsedSeconds - Double(minutes * 60)
        return String(format: "%d:%04.1f", minutes, seconds)
    }

    var speedupText: String {
        guard gliClass.modelMillisecondsPerMove > 0, laya.modelMillisecondsPerMove > 0 else { return "–" }
        return String(format: "%.1f×", laya.modelMillisecondsPerMove / gliClass.modelMillisecondsPerMove)
    }

    func applyEnvironment() {
        let environment = ProcessInfo.processInfo.environment
        if let value = environment["GAME2048_SEED"].flatMap(UInt64.init) { seed = value }
        if let value = environment["GAME2048_DELAY_MS"].flatMap(Double.init), value >= 0 { moveDelayMs = value }
        if environment["GAME2048_AUTOLOAD"] == "1" { loadModels() }
        guard environment["GAME2048_AUTORUN"] == "1" else { return }
        loadModels()
        Task {
            while isLoading { try? await Task.sleep(for: .milliseconds(100)) }
            guard hasLoadedModels else { return }
            reset()
            toggle()
            if let text = environment["GAME2048_QUIT_AFTER"], let seconds = Double(text) {
                try? await Task.sleep(for: .seconds(seconds))
                print(
                    String(
                        format:
                            "autorun: GLiClass %d score, %d moves, max %d, %.2f ms/move; Laya %d score, %d moves, max %d, %.2f ms/move; %.2f s",
                        gliClass.score, gliClass.moves, gliClass.maximumTile, gliClass.modelMillisecondsPerMove,
                        laya.score, laya.moves, laya.maximumTile, laya.modelMillisecondsPerMove, elapsedSeconds))
                if let errorMessage { print("autorun error: \(errorMessage)") }
                exit(0)
            }
        }
    }

    func loadModels() {
        guard !isLoading, !hasLoadedModels else { return }
        isLoading = true
        loadStatus = "Loading GLiClass LUT8…"
        Task {
            do {
                let environment = ProcessInfo.processInfo.environment
                guard let gliClassDirectory = environment["GLICLASS_MODEL_DIR"], !gliClassDirectory.isEmpty else {
                    throw GLiClassError.invalidAsset("Set GLICLASS_MODEL_DIR for the 2048 benchmark")
                }
                let loadedGLiClass = try await GLiClassManager.load(
                    from: URL(fileURLWithPath: gliClassDirectory),
                    configuration: .init(
                        lengths: [128], precision: environment["GLICLASS_PRECISION"] ?? "lut8"))
                loadStatus = "Loading Laya E8…"
                let layaConfiguration = LayaManager.Configuration(
                    lengths: [128], precision: environment["LAYA_PRECISION"] ?? "e8")
                let loadedLaya: LayaManager
                if let directory = environment["LAYA_MODEL_DIR"], !directory.isEmpty {
                    loadedLaya = try await LayaManager.load(
                        from: URL(fileURLWithPath: directory), configuration: layaConfiguration)
                } else {
                    loadedLaya = try await LayaManager.load(configuration: layaConfiguration)
                }
                _ = try await loadedGLiClass.classify(
                    text: "Build the largest tile without filling the board.",
                    labels: ["swipe left: 8 empty cells", "swipe right: 5 empty cells"],
                    prompt: "Choose the safest 2048 swipe.")
                _ = try await loadedLaya.answer(
                    state: "swipe left: 8 empty cells, gain 4, largest tile in a corner",
                    question: Self.layaQuestion)
                gliClassManager = loadedGLiClass
                layaManager = loadedLaya
                loadStatus = "GLiClass LUT8 + Laya E8 · L128 · CPU + ANE"
            } catch {
                errorMessage = error.localizedDescription
                loadStatus = "Load failed"
            }
            isLoading = false
        }
    }

    func reset() {
        stop()
        gliClassGame = Game2048(seed: seed)
        layaGame = Game2048(seed: seed)
        gliClass = Self.side(
            name: "GLiClass Edge Apps v2", detail: "32.7M · LUT8 · 1 call/move", game: gliClassGame)
        laya = Self.side(name: "Laya Multilingual", detail: "322M · E8 · 2 calls/move", game: layaGame)
        elapsedSeconds = 0
        accumulatedSeconds = 0
    }

    func toggle() {
        if isRunning {
            stop()
            return
        }
        guard hasLoadedModels else {
            errorMessage = "Load both models before starting the benchmark."
            return
        }
        if gliClass.isOver && laya.isOver { reset() }
        generation += 1
        let run = generation
        isRunning = true
        runStart = Date()
        clockTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(100))
                guard let self, self.isRunning, self.generation == run else { return }
                self.elapsedSeconds = self.accumulatedSeconds + Date().timeIntervalSince(self.runStart)
            }
        }
        playTask = Task { [weak self] in await self?.play(run: run) }
    }

    private func stop() {
        generation += 1
        playTask?.cancel()
        clockTask?.cancel()
        playTask = nil
        clockTask = nil
        if isRunning {
            accumulatedSeconds += Date().timeIntervalSince(runStart)
            elapsedSeconds = accumulatedSeconds
        }
        isRunning = false
    }

    private func play(run: Int) async {
        defer {
            if generation == run { stop() }
        }
        while !Task.isCancelled, generation == run, !gliClass.isOver || !laya.isOver {
            do {
                if !gliClass.isOver { try await playGLiClassMove(run: run) }
                if !laya.isOver { try await playLayaMove(run: run) }
            } catch is CancellationError {
                return
            } catch {
                errorMessage = error.localizedDescription
                print("2048 comparison failed: \(error.localizedDescription)")
                return
            }
            if moveDelayMs > 0 { try? await Task.sleep(for: .milliseconds(moveDelayMs)) }
            await Task.yield()
        }
    }

    private func playGLiClassMove(run: Int) async throws {
        guard let manager = gliClassManager else { throw GLiClassError.invalidModel("GLiClass is not loaded") }
        let offered = shortlist(game: gliClassGame)
        guard let first = offered.first else {
            gliClass.isOver = true
            return
        }
        var chosen = first
        if offered.count > 1 {
            let descriptions = offered.map(gliClassGame.describe)
            let inference = Task.detached(priority: .userInitiated) {
                let started = DispatchTime.now().uptimeNanoseconds
                let answer = try await manager.classify(
                    text: "Build the largest tile without filling the board.", labels: descriptions,
                    prompt: "Choose the safest 2048 swipe. Preserve empty cells, ordered high tiles, and merges.")
                return (answer.selectedIndex, Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6)
            }
            let result = try await inference.value
            guard generation == run, !Task.isCancelled else { throw CancellationError() }
            chosen = offered[result.0]
            gliClass.modelCalls += 1
            gliClass.modelMilliseconds += result.1
        }
        gliClassGame.apply(chosen)
        update(&gliClass, from: gliClassGame)
    }

    private func playLayaMove(run: Int) async throws {
        guard let manager = layaManager else { throw LayaError.invalidModel("Laya is not loaded") }
        let offered = shortlist(game: layaGame)
        guard let first = offered.first else {
            laya.isOver = true
            return
        }
        var chosen = first
        if offered.count > 1 {
            let descriptions = offered.map(layaGame.describe)
            let inference = Task.detached(priority: .userInitiated) {
                var scores: [Float] = []
                var milliseconds = 0.0
                for description in descriptions {
                    let started = DispatchTime.now().uptimeNanoseconds
                    let answer = try await manager.answer(state: description, question: Self.layaQuestion)
                    milliseconds += Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
                    scores.append(answer.noul ?? 0)
                }
                let selected = scores.indices.max { scores[$0] < scores[$1] } ?? 0
                return (selected, scores.count, milliseconds)
            }
            let result = try await inference.value
            guard generation == run, !Task.isCancelled else { throw CancellationError() }
            chosen = offered[result.0]
            laya.modelCalls += result.1
            laya.modelMilliseconds += result.2
        }
        layaGame.apply(chosen)
        update(&laya, from: layaGame)
    }

    private func shortlist(game: Game2048) -> [Game2048.Candidate] {
        Array(
            game.candidates().sorted {
                game.strategicScore($0, lookahead: true) > game.strategicScore($1, lookahead: true)
            }.prefix(2))
    }

    private func update(_ side: inout Decision2048Side, from game: Game2048) {
        side.board = game.board
        side.score = game.score
        side.moves = game.moves
        side.maximumTile = game.maximumTile
        side.isOver = game.isOver
    }

    private static func side(name: String, detail: String, game: Game2048) -> Decision2048Side {
        Decision2048Side(
            name: name, detail: detail, board: game.board, score: game.score, moves: game.moves,
            maximumTile: game.maximumTile, isOver: game.isOver)
    }

    private static let layaQuestion = LayaQuestion.noul(
        "Is this a safe 2048 move that preserves empty space, keeps high tiles ordered in a corner, and enables future merges?"
    )
}
