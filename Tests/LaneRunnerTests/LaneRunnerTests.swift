import Testing

@testable import LaneRunner

struct LaneRunnerTests {
    @Test func seededTracksMatch() {
        var first = LaneRunner(seed: 7)
        var second = LaneRunner(seed: 7)
        for _ in 0..<100 {
            first.step(first.heuristicAction)
            second.step(second.heuristicAction)
            #expect(first == second)
        }
    }

    @Test func everyTrackIsSurvivable() {
        for seed in UInt64(1)...20 {
            var game = LaneRunner(seed: seed)
            while !game.isOver && game.distance < 300 { game.step(game.heuristicAction) }
            #expect(!game.isOver, "seed \(seed) crashed: \(game.crash ?? "")")
        }
    }

    @Test func obstaclesNeedTheirMove() {
        var game = LaneRunner(seed: 1)
        game.rows[0] = [.open, .low, .open]
        #expect(game.isSafe(.jump) == (game.label(for: .jump).contains("safe")))
        #expect(game.label(for: .stay).contains("crash"))
        #expect(game.label(for: .slide).contains("crash"))
        game.rows[0] = [.open, .high, .open]
        var sliding = game
        sliding.step(.slide)
        #expect(!sliding.isOver)
        game.rows[0] = [.open, .train, .open]
        var jumping = game
        jumping.step(.jump)
        #expect(jumping.isOver)
        #expect(jumping.crash == "hit a train")
    }

    @Test func edgeLanesOfferFourMoves() {
        var game = LaneRunner(seed: 1)
        game.lane = 0
        #expect(game.legalActions == [.right, .jump, .slide, .stay])
        game.lane = 1
        #expect(game.legalActions.count == 5)
    }
}
