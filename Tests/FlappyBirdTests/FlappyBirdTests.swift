import Testing

@testable import FlappyBird

struct FlappyBirdTests {
    @Test func seededCoursesMatch() {
        var first = FlappyBird(seed: 7)
        var second = FlappyBird(seed: 7)
        for _ in 0..<300 {
            first.step(first.heuristicAction)
            second.step(second.heuristicAction)
            #expect(first == second)
        }
    }

    @Test func flapChangesTrajectory() {
        let start = FlappyBird(seed: 1)
        let flap = start.projection(.flap)
        let coast = start.projection(.coast)
        #expect(flap.y < coast.y)
        #expect(start.frames == 0)
        #expect(start.y == 300)
    }

    @Test func collisionStopsTimeAndScore() {
        var game = FlappyBird(seed: 1)
        game.y = 13
        game.step(.flap)
        #expect(game.isOver)
        let stopped = game
        game.step()
        #expect(game == stopped)
    }

    @Test func pipeScoresExactlyOnce() {
        var game = FlappyBird(seed: 1)
        let center = game.pipes[0].center
        game.y = center
        game.pipes[0].x = FlappyBird.birdX - FlappyBird.radius - FlappyBird.pipeWidth + 1
        game.step()
        #expect(game.score == 1)
        for _ in 0..<5 { game.step() }
        #expect(game.score == 1)
    }

    @Test func forecastExposesCeilingDanger() {
        var game = FlappyBird(seed: 1)
        game.y = 15
        #expect(game.label(for: .flap).contains("collision"))
        #expect(!game.label(for: .coast).contains("collision"))
    }

    @Test func guardBlocksPredictedCeilingCollision() {
        var game = FlappyBird(seed: 1)
        game.y = 15
        #expect(game.projection(.flap).isOver)
        #expect(!game.projection(.coast).isOver)
        #expect(game.guardedAction(preferred: .flap) == .coast)
    }

    @Test func guardPreservesSafeModelChoice() {
        var game = FlappyBird(seed: 1)
        game.y = game.pipes[0].center
        #expect(game.guardedAction(preferred: .flap) == .flap)
    }
}
