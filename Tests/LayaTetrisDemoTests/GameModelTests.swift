import Testing

import LayaTetris
@testable import LayaTetrisDemo

@Suite(.serialized)
struct GameModelTests {
    @MainActor
    @Test func testDemoDefaultsRunContinuouslyAtFullSpeed() {
        let model = GameModel()
        #expect(model.marathon)
        #expect(model.seed == 1)
        #expect(model.pieceDelayMs == 0)
    }

    @MainActor
    private func waitUntil(_ condition: () -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(5)
        while !condition(), ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        try #require(condition(), "Timed out waiting for the game loop")
    }

    @MainActor
    @Test func testPlayUsesSelectedSeedAndPauseFreezesBoardAndClock() async throws {
        let model = GameModel()
        defer { model.reset() }
        model.policy = .heuristic
        model.seed = 123
        model.pieceDelayMs = 1000
        var expected = TetrisGame(seed: 123)
        let spawned = expected.spawn()
        let piece = try #require(spawned)
        let candidates = TetrisGame.shortlist(expected.candidates(for: piece))
        expected.apply(try #require(candidates.max { $0.features.heuristic < $1.features.heuristic }))
        model.toggle()
        try await waitUntil { model.pieces == 1 }
        model.toggle()
        let pausedTime = model.elapsedSeconds
        #expect(model.board == expected.board)
        try await Task.sleep(for: .milliseconds(150))
        #expect(!model.isRunning)
        #expect(model.pieces == 1)
        #expect(model.elapsedSeconds == pausedTime)

        model.toggle()
        try await waitUntil { model.pieces == 2 }
        model.toggle()
        #expect(model.elapsedSeconds >= pausedTime)
        #expect(model.elapsedSeconds - pausedTime < 0.1, "Paused time must be excluded")
    }

    @MainActor
    @Test func testZeroDelayMarathonYieldsAndCarriesTotals() async throws {
        let model = GameModel()
        defer { model.reset() }
        model.policy = .random
        model.marathon = true
        model.toggle()
        try await waitUntil { model.gamesPlayed >= 2 }
        model.toggle()
        #expect(model.totalPieces > model.pieces)
        #expect(model.totalLines >= model.lines)
        let total = model.totalPieces
        let elapsed = model.elapsedSeconds
        try await Task.sleep(for: .milliseconds(120))
        #expect(model.totalPieces == total)
        #expect(model.elapsedSeconds == elapsed)
        #expect(!model.isRunning)
    }

    @MainActor
    @Test func testResetDiscardsSleepingRunAndClock() async throws {
        let model = GameModel()
        defer { model.reset() }
        model.policy = .random
        model.pieceDelayMs = 1000
        model.toggle()
        try await waitUntil { model.pieces == 1 }
        model.reset()
        try await Task.sleep(for: .milliseconds(150))
        #expect(model.board == TetrisGame(seed: model.seed).board)
        #expect(model.pieces == 0)
        #expect(model.totalPieces == 0)
        #expect(model.totalLines == 0)
        #expect(model.elapsedSeconds == 0)
        #expect(model.chosen == nil)
        #expect(!model.isRunning)
    }
}
