import Testing

@testable import GLiClass2048Demo

struct Game2048ModelTests {
    @MainActor
    @Test func defaultsToMeasuredTopTwoGLiClassConfiguration() {
        let model = Game2048Model()
        #expect(model.policy == .gliclass)
        #expect(model.candidateCount == 2)
        #expect(model.minimumMargin == 0.40)
        #expect(!model.marathon)
        #expect(model.seed == 9)
        #expect(model.moveDelayMs == 0)
    }

    @MainActor
    @Test func heuristicPolicyCanPlayAndReset() async throws {
        let model = Game2048Model()
        model.policy = .heuristic
        model.toggle()
        let deadline = ContinuousClock.now + .seconds(2)
        while model.moves < 5, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        #expect(model.moves >= 5)
        model.reset()
        #expect(model.moves == 0)
        #expect(model.score == 0)
        #expect(!model.isRunning)
    }
}
