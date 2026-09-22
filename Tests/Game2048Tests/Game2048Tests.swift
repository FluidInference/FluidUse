import Testing

@testable import Game2048

struct Game2048Tests {
    @Test func mergesEachTileOnlyOnce() throws {
        let game = Game2048(board: [[2, 2, 2, 2], [4, 4, 8, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        let left = try #require(game.candidates().first { $0.direction == .left })
        #expect(left.board[0] == [4, 4, 0, 0])
        #expect(left.board[1] == [8, 8, 0, 0])
        #expect(left.features.scoreGained == 12)
    }

    @Test func rejectsMovesThatDoNotChangeBoard() {
        let game = Game2048(board: [[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        let directions = Set(game.candidates().map(\.direction))
        #expect(!directions.contains(.left))
        #expect(directions.contains(.right))
        #expect(directions.contains(.down))
    }

    @Test func detectsFinishedBoard() {
        let game = Game2048(
            board: [
                [2, 4, 2, 4],
                [4, 2, 4, 2],
                [2, 4, 2, 4],
                [4, 2, 4, 2],
            ])
        #expect(game.isOver)
        #expect(game.candidates().isEmpty)
    }

    @Test func seededGamesAreDeterministic() {
        var first = Game2048(seed: 42)
        var second = Game2048(seed: 42)
        for _ in 0..<30 {
            let firstMoves = first.candidates()
            let secondMoves = second.candidates()
            #expect(firstMoves.map(\.direction) == secondMoves.map(\.direction))
            guard let firstPick = firstMoves.max(by: { $0.features.heuristic < $1.features.heuristic }),
                let secondPick = secondMoves.max(by: { $0.features.heuristic < $1.features.heuristic })
            else { break }
            first.apply(firstPick)
            second.apply(secondPick)
            #expect(first.board == second.board)
        }
    }
}
