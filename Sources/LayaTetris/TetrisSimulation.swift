import FluidUse
import Foundation

/// Headless 10×20 Tetris used by `FluidUseLaya tetris` and `LayaTetrisDemo`: legal landings, their
/// features, the one-sentence description laya scores, and a heuristic baseline.
public enum LayaTetris {
    /// The question both the CLI and the demo ask about every landing.
    public static let question = LayaQuestion.noul("Is this a clean placement?")
}

public struct SplitMix64 {
    private var state: UInt64
    public init(seed: UInt64) { state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

public struct TetrisGame {
    public static let width = 10
    public static let height = 20

    public struct Piece: Sendable {
        public let name: String
        /// Rotation states as cell offsets (column, row) with row 0 at the top of the piece box.
        public let rotations: [[(Int, Int)]]
    }

    public struct Features: Sendable {
        public let linesCleared: Int
        public let newHoles: Int
        public let landingHeight: Int
        public let maxHeight: Int
        public let bumpiness: Int
        public let bumpinessDelta: Int
        public let wellDepth: Int
        public let flushSides: Int

        /// Dellacherie-style linear evaluation used by the baseline policy.
        public var heuristic: Double {
            Double(linesCleared) * 3.4 - Double(newHoles) * 7.9 - Double(landingHeight) * 4.5 - Double(bumpiness) * 1.2
                - Double(wellDepth) * 3.4
        }
    }

    public struct Candidate: Identifiable, Sendable {
        public let id: Int
        public let rotation: Int
        public let column: Int
        public let board: [[Bool]]
        /// Board cells (x, y) the piece occupies before any line clears.
        public let cells: [(Int, Int)]
        public let features: Features
    }

    public static let pieces: [Piece] = [
        Piece(name: "I", rotations: [[(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 0), (0, 1), (0, 2), (0, 3)]]),
        Piece(name: "O", rotations: [[(0, 0), (1, 0), (0, 1), (1, 1)]]),
        Piece(
            name: "T",
            rotations: [
                [(0, 0), (1, 0), (2, 0), (1, 1)], [(0, 0), (0, 1), (0, 2), (1, 1)], [(1, 0), (0, 1), (1, 1), (2, 1)],
                [(1, 0), (0, 1), (1, 1), (1, 2)],
            ]),
        Piece(name: "S", rotations: [[(1, 0), (2, 0), (0, 1), (1, 1)], [(0, 0), (0, 1), (1, 1), (1, 2)]]),
        Piece(name: "Z", rotations: [[(0, 0), (1, 0), (1, 1), (2, 1)], [(1, 0), (0, 1), (1, 1), (0, 2)]]),
        Piece(
            name: "J",
            rotations: [
                [(0, 0), (0, 1), (1, 1), (2, 1)], [(0, 0), (1, 0), (0, 1), (0, 2)], [(0, 0), (1, 0), (2, 0), (2, 1)],
                [(1, 0), (1, 1), (0, 2), (1, 2)],
            ]),
        Piece(
            name: "L",
            rotations: [
                [(2, 0), (0, 1), (1, 1), (2, 1)], [(0, 0), (0, 1), (0, 2), (1, 2)], [(0, 0), (1, 0), (2, 0), (0, 1)],
                [(0, 0), (1, 0), (1, 1), (1, 2)],
            ]),
    ]

    public private(set) var board: [[Bool]]
    public private(set) var linesCleared = 0
    public private(set) var isOver = false
    private var bag: [Int] = []
    private var upcoming: [Piece] = []
    private var rng: SplitMix64

    public init(seed: UInt64) {
        board = Array(repeating: Array(repeating: false, count: Self.width), count: Self.height)
        rng = SplitMix64(seed: seed)
    }

    /// Next piece from a seven-bag randomizer, or nil once the stack has topped out.
    public mutating func spawn() -> Piece? {
        guard !isOver else { return nil }
        while upcoming.count < 2 { upcoming.append(draw()) }
        return upcoming.removeFirst()
    }

    /// The piece that will follow the one currently in play, which is what a Tetris player sees.
    public var nextPiece: Piece? { upcoming.first }

    private mutating func draw() -> Piece {
        if bag.isEmpty {
            bag = Array(0..<Self.pieces.count)
            for index in stride(from: bag.count - 1, to: 0, by: -1) {
                bag.swapAt(index, Int(rng.next() % UInt64(index + 1)))
            }
        }
        return Self.pieces[bag.removeLast()]
    }

    public func candidates(for piece: Piece) -> [Candidate] {
        candidates(for: piece, on: board)
    }

    /// Legal landings of `piece` on any board, used to evaluate the follow-up move.
    public func candidates(for piece: Piece, on board: [[Bool]]) -> [Candidate] {
        var result: [Candidate] = []
        let heightsBefore = columnHeights(board)
        let bumpinessBefore = bumpiness(heightsBefore)
        for (rotation, cells) in piece.rotations.enumerated() {
            let pieceWidth = cells.map(\.0).max()! + 1
            let pieceHeight = cells.map(\.1).max()! + 1
            for column in 0...(Self.width - pieceWidth) {
                // Hard drop: lowest row offset where every cell is free.
                var row = -pieceHeight
                while fits(cells, column: column, row: row + 1, on: board) { row += 1 }
                guard row >= 0 else { continue }
                var next = board
                let occupied = cells.map { (column + $0.0, row + $0.1) }
                for (dx, dy) in cells { next[row + dy][column + dx] = true }
                let landingHeight = Self.height - row - pieceHeight / 2
                let cleared = clearLines(&next)
                let heightsAfter = columnHeights(next)
                let holesBefore = holes(board)
                let holesAfter = holes(next)
                let bump = bumpiness(heightsAfter)
                let flush = flushSides(cells, column: column, row: row, on: board)
                result.append(
                    Candidate(
                        id: result.count, rotation: rotation, column: column, board: next, cells: occupied,
                        features: Features(
                            linesCleared: cleared, newHoles: max(0, holesAfter - holesBefore),
                            landingHeight: landingHeight, maxHeight: heightsAfter.max() ?? 0, bumpiness: bump,
                            bumpinessDelta: bump - bumpinessBefore, wellDepth: deepestWell(heightsAfter),
                            flushSides: flush)))
            }
        }
        return result
    }

    public mutating func apply(_ candidate: Candidate) {
        board = candidate.board
        linesCleared += candidate.features.linesCleared
        if board[0].contains(true) || board[1].contains(true) { isOver = true }
    }

    /// One natural sentence per landing, the only thing laya sees. Under the short question
    /// "Is this a clean placement?" the multilingual checkpoint ranks these sensibly: holes and a
    /// taller stack pull P(clean) down, a cleared line pushes it up; numeric feature dumps do not.
    /// Description style. `.plain` is the original wording; `.graded` keeps every clause
    /// discriminative when the board is tall, where `.plain` collapses to one identical sentence.
    public enum DescriptionStyle: String, Sendable {
        case plain, graded
    }

    public func describe(_ candidate: Candidate, piece: Piece, style: DescriptionStyle) -> String {
        guard style == .graded else { return describe(candidate, piece: piece) }
        let f = candidate.features
        var clauses: [String] = []
        // Line clears lead: they are the only move that reduces the stack, and in the original
        // wording they were mentioned last, after the clauses that saturate.
        if f.linesCleared > 0 {
            clauses.append("clears \(Self.words(f.linesCleared)) line\(f.linesCleared == 1 ? "" : "s")")
        }
        clauses.append(
            f.newHoles > 0
                ? "buries \(Self.words(f.newHoles)) cell\(f.newHoles == 1 ? "" : "s") that cannot be reached again"
                : "buries nothing")
        // Fine-grained height so two candidates on a tall board still read differently.
        let remaining = Self.height - f.maxHeight
        switch remaining {
        case ...1: clauses.append("fills the board to the very top")
        case 2...3: clauses.append("leaves only \(remaining) rows of space")
        case 4...6: clauses.append("leaves \(remaining) rows of space")
        case 7...10: clauses.append("leaves comfortable room above")
        default: clauses.append("keeps the stack low")
        }
        if f.bumpinessDelta > 2 {
            clauses.append("makes the surface much more uneven")
        } else if f.bumpinessDelta > 0 {
            clauses.append("makes the surface more uneven")
        } else if f.bumpinessDelta < 0 {
            clauses.append("evens out the surface")
        } else {
            clauses.append("leaves the surface as even as before")
        }
        if f.wellDepth >= 3 { clauses.append("leaves a deep gap \(f.wellDepth) rows deep") }
        let body =
            clauses.count > 1 ? clauses.dropLast().joined(separator: ", ") + ", and " + clauses.last! : clauses[0]
        return "The \(piece.name) piece dropped at column \(candidate.column) " + body + "."
    }

    public func describe(_ candidate: Candidate, piece: Piece) -> String {
        let f = candidate.features
        var clauses: [String] = []
        clauses.append(
            f.newHoles > 0
                ? "leaves \(Self.words(f.newHoles)) hole\(f.newHoles == 1 ? "" : "s") under it" : "leaves no holes")
        if f.bumpinessDelta > 2 {
            clauses.append("makes the surface much bumpier")
        } else if f.bumpinessDelta > 0 {
            clauses.append("makes the surface bumpier")
        } else if f.bumpinessDelta < 0 {
            clauses.append("makes the surface flatter")
        } else {
            clauses.append("keeps the surface flat")
        }
        if f.maxHeight >= 15 {
            clauses.append("the stack is getting dangerously tall")
        } else if f.landingHeight > 8 {
            clauses.append("makes the stack taller")
        } else {
            clauses.append("keeps the stack low")
        }
        if f.wellDepth >= 3 { clauses.append("leaves a deep well") }
        if f.linesCleared > 0 {
            clauses.append("clears \(Self.words(f.linesCleared)) line\(f.linesCleared == 1 ? "" : "s")")
        }
        let body =
            clauses.count > 1 ? clauses.dropLast().joined(separator: ", ") + ", and " + clauses.last! : clauses[0]
        return "The \(piece.name) piece dropped at column \(candidate.column) " + body + "."
    }

    private static func words(_ value: Int) -> String {
        let names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        return value < names.count ? names[value] : String(value)
    }

    public func render() -> String {
        Self.render(board)
    }

    public func render(after candidate: Candidate) -> String {
        Self.render(candidate.board)
    }

    private static func render(_ grid: [[Bool]]) -> String {
        grid.map { row in row.map { $0 ? "█" : "·" }.joined() }.joined(separator: "\n")
    }

    // MARK: Geometry

    private func fits(_ cells: [(Int, Int)], column: Int, row: Int, on board: [[Bool]]) -> Bool {
        for (dx, dy) in cells {
            let x = column + dx
            let y = row + dy
            guard x >= 0, x < Self.width, y < Self.height else { return false }
            if y >= 0, board[y][x] { return false }
        }
        return true
    }

    private func clearLines(_ grid: inout [[Bool]]) -> Int {
        let kept = grid.filter { !$0.allSatisfy { $0 } }
        let cleared = grid.count - kept.count
        grid = Array(repeating: Array(repeating: false, count: Self.width), count: cleared) + kept
        return cleared
    }

    private func columnHeights(_ grid: [[Bool]]) -> [Int] {
        (0..<Self.width).map { x in
            for y in 0..<Self.height where grid[y][x] { return Self.height - y }
            return 0
        }
    }

    private func holes(_ grid: [[Bool]]) -> Int {
        var count = 0
        for x in 0..<Self.width {
            var covered = false
            for y in 0..<Self.height {
                if grid[y][x] { covered = true } else if covered { count += 1 }
            }
        }
        return count
    }

    private func bumpiness(_ heights: [Int]) -> Int {
        zip(heights, heights.dropFirst()).reduce(0) { $0 + abs($1.0 - $1.1) }
    }

    private func deepestWell(_ heights: [Int]) -> Int {
        var deepest = 0
        for x in 0..<Self.width {
            let left = x == 0 ? Self.height : heights[x - 1]
            let right = x == Self.width - 1 ? Self.height : heights[x + 1]
            deepest = max(deepest, min(left, right) - heights[x])
        }
        return deepest
    }

    private func flushSides(_ cells: [(Int, Int)], column: Int, row: Int, on board: [[Bool]]) -> Int {
        var count = 0
        for (dx, dy) in cells {
            for neighbour in [(column + dx - 1, row + dy), (column + dx + 1, row + dy)] {
                if neighbour.0 < 0 || neighbour.0 >= Self.width {
                    count += 1
                } else if neighbour.1 >= 0, board[neighbour.1][neighbour.0] {
                    count += 1
                }
            }
        }
        return count
    }
}
