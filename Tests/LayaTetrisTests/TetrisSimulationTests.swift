import Testing

@testable import LayaTetris

struct TetrisSimulationTests {
    @Test func testPreviewMatchesNextSpawnAcrossBagBoundaries() throws {
        var game = TetrisGame(seed: 24)
        var names: [String] = []
        for _ in 0..<28 {
            let expected = game.nextPiece?.name
            let spawned = game.spawn()
            let piece = try #require(spawned)
            if let expected { #expect(piece.name == expected) }
            names.append(piece.name)
        }
        for start in stride(from: 0, to: names.count, by: 7) {
            #expect(Set(names[start..<(start + 7)]) == Set(TetrisGame.pieces.map(\.name)))
        }
    }

    @Test func testHypotheticalBoardMatchesAppliedBoardWithoutMutatingGame() throws {
        var game = TetrisGame(seed: 7)
        let spawned = game.spawn()
        let piece = try #require(spawned)
        let landing = try #require(game.candidates(for: piece).first)
        let next = try #require(game.nextPiece)
        let originalBoard = game.board
        let follow = game.candidates(for: next, on: landing.board)
        #expect(game.board == originalBoard)
        var applied = game
        applied.apply(landing)
        let actual = applied.candidates(for: next)
        #expect(follow.map(\.board) == actual.map(\.board))
        #expect(follow.map(\.features.heuristic) == actual.map(\.features.heuristic))
    }

    @Test func testLookaheadCannotContinueAfterTopOut() {
        let game = TetrisGame(seed: 7)
        var board = game.board
        board[1][0] = true
        #expect(game.candidates(for: TetrisGame.pieces[1], on: board).isEmpty)
    }

    @Test func testShortlistFiltersWhenPossibleAndFallsBackWhenNecessary() {
        let game = TetrisGame(seed: 7)
        let candidates = game.candidates(for: TetrisGame.pieces[2])
        let burying = candidates.filter { $0.features.newHoles > 0 }
        #expect(!burying.isEmpty)
        let clean = TetrisGame.shortlist(candidates)
        #expect(!clean.isEmpty)
        #expect(clean.count < candidates.count)
        #expect(clean.allSatisfy { $0.features.newHoles == 0 })
        #expect(TetrisGame.shortlist(burying).map(\.id) == burying.map(\.id))
        #expect(TetrisGame.shortlist([]).isEmpty)
    }

    @Test func testGradedDescriptionDistinguishesTallLandings() throws {
        let game = TetrisGame(seed: 7)
        let piece = TetrisGame.pieces[1]
        var board = game.board
        for y in 5..<TetrisGame.height {
            board[y][0] = true
            board[y][1] = true
        }
        for y in 6..<TetrisGame.height {
            board[y][8] = true
            board[y][9] = true
        }
        let candidates = game.candidates(for: piece, on: board)
        let left = try #require(candidates.first { $0.column == 0 })
        let right = try #require(candidates.first { $0.column == 8 })
        #expect(game.describe(left, piece: piece, style: .graded).contains("only 3 rows"))
        #expect(game.describe(right, piece: piece, style: .graded).contains("4 rows"))
        #expect(game.describe(left, piece: piece, style: .plain) == game.describe(left, piece: piece))
    }
}
