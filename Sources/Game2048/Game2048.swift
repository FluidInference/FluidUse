import Foundation

/// Deterministic 4×4 implementation of 2048 shared by the benchmark and SwiftUI demo.
public struct Game2048: Sendable {
    public static let size = 4

    public enum Direction: String, CaseIterable, Hashable, Sendable {
        case up, down, left, right
    }

    public struct Features: Sendable {
        public let emptyCells: Int
        public let maximumTile: Int
        public let maximumInCorner: Bool
        public let monotonicityPenalty: Int
        public let roughness: Int
        public let scoreGained: Int

        /// A conventional 2048 safety heuristic. It supplies a strong shortlist; GLiClass makes
        /// the final semantic comparison between the two best legal moves.
        public var heuristic: Double {
            let exponent = maximumTile > 0 ? Int(log2(Double(maximumTile))) : 0
            return Double(emptyCells * 270 + scoreGained * 2 + (maximumInCorner ? exponent * 120 : 0))
                - Double(monotonicityPenalty * 30 + roughness * 12)
        }
    }

    public struct Candidate: Sendable {
        public let direction: Direction
        public let board: [[Int]]
        public let features: Features
    }

    public private(set) var board: [[Int]]
    public private(set) var score = 0
    public private(set) var moves = 0
    public private(set) var isOver = false
    private var rng: SplitMix64

    public init(seed: UInt64) {
        board = Array(repeating: Array(repeating: 0, count: Self.size), count: Self.size)
        rng = SplitMix64(seed: seed)
        spawnTile()
        spawnTile()
    }

    /// Deterministic board initializer used by engine tests.
    init(board: [[Int]], seed: UInt64 = 1, score: Int = 0) {
        self.board = board
        self.score = score
        rng = SplitMix64(seed: seed)
        isOver = Self.legalCandidates(on: board).isEmpty
    }

    public var maximumTile: Int { board.flatMap { $0 }.max() ?? 0 }

    public func candidates() -> [Candidate] { Self.legalCandidates(on: board) }

    /// Expected strength of the best following move after every possible random tile spawn.
    /// Standard 2048 uses a 90% chance of spawning 2 and 10% chance of spawning 4.
    public func expectedReplyHeuristic(after candidate: Candidate) -> Double {
        var empty: [(Int, Int)] = []
        for row in 0..<Self.size {
            for column in 0..<Self.size where candidate.board[row][column] == 0 { empty.append((row, column)) }
        }
        guard !empty.isEmpty else { return candidate.features.heuristic }
        var total = 0.0
        for (row, column) in empty {
            for (tile, probability) in [(2, 0.9), (4, 0.1)] {
                var spawned = candidate.board
                spawned[row][column] = tile
                let replies = Self.legalCandidates(on: spawned)
                let best =
                    replies.map(\.features.heuristic).max()
                    ?? Self.features(board: spawned, scoreGained: 0).heuristic
                total += probability * best / Double(empty.count)
            }
        }
        return total
    }

    public func strategicScore(_ candidate: Candidate, lookahead: Bool) -> Double {
        guard lookahead else { return candidate.features.heuristic }
        return candidate.features.heuristic * 0.35 + expectedReplyHeuristic(after: candidate) * 0.65
    }

    public mutating func apply(_ candidate: Candidate) {
        guard !isOver else { return }
        board = candidate.board
        score += candidate.features.scoreGained
        moves += 1
        spawnTile()
        isOver = Self.legalCandidates(on: board).isEmpty
    }

    public func describe(_ candidate: Candidate) -> String {
        let features = candidate.features
        let corner = features.maximumInCorner ? "largest tile in a corner" : "largest tile away from a corner"
        return "swipe \(candidate.direction.rawValue): \(features.emptyCells) empty cells, "
            + "gain \(features.scoreGained), \(corner), monotonicity penalty \(features.monotonicityPenalty), "
            + "roughness \(features.roughness)"
    }

    private static func legalCandidates(on board: [[Int]]) -> [Candidate] {
        Direction.allCases.compactMap { direction in
            let moved = move(board, direction: direction)
            guard moved.board != board else { return nil }
            return Candidate(
                direction: direction, board: moved.board,
                features: features(board: moved.board, scoreGained: moved.scoreGained))
        }
    }

    private static func move(_ board: [[Int]], direction: Direction) -> (board: [[Int]], scoreGained: Int) {
        var result = board
        var gained = 0
        for index in 0..<size {
            let source: [Int]
            switch direction {
            case .left: source = board[index]
            case .right: source = Array(board[index].reversed())
            case .up: source = (0..<size).map { board[$0][index] }
            case .down: source = (0..<size).reversed().map { board[$0][index] }
            }
            let collapsed = collapse(source)
            gained += collapsed.scoreGained
            let line =
                (direction == .right || direction == .down) ? Array(collapsed.values.reversed()) : collapsed.values
            switch direction {
            case .left, .right: result[index] = line
            case .up, .down:
                for row in 0..<size { result[row][index] = line[row] }
            }
        }
        return (result, gained)
    }

    private static func collapse(_ line: [Int]) -> (values: [Int], scoreGained: Int) {
        let tiles = line.filter { $0 != 0 }
        var values: [Int] = []
        var gained = 0
        var index = 0
        while index < tiles.count {
            if index + 1 < tiles.count, tiles[index] == tiles[index + 1] {
                let merged = tiles[index] * 2
                values.append(merged)
                gained += merged
                index += 2
            } else {
                values.append(tiles[index])
                index += 1
            }
        }
        values += Array(repeating: 0, count: size - values.count)
        return (values, gained)
    }

    private static func features(board: [[Int]], scoreGained: Int) -> Features {
        let maximum = board.flatMap { $0 }.max() ?? 0
        let corners = [board[0][0], board[0][size - 1], board[size - 1][0], board[size - 1][size - 1]]
        var roughness = 0
        for row in 0..<size {
            for column in 0..<size {
                let current = exponent(board[row][column])
                if column + 1 < size, board[row][column] > 0, board[row][column + 1] > 0 {
                    roughness += abs(current - exponent(board[row][column + 1]))
                }
                if row + 1 < size, board[row][column] > 0, board[row + 1][column] > 0 {
                    roughness += abs(current - exponent(board[row + 1][column]))
                }
            }
        }
        var monotonicity = 0
        for row in board { monotonicity += lineMonotonicityPenalty(row.map(exponent)) }
        for column in 0..<size {
            monotonicity += lineMonotonicityPenalty((0..<size).map { exponent(board[$0][column]) })
        }
        return Features(
            emptyCells: board.flatMap { $0 }.filter { $0 == 0 }.count,
            maximumTile: maximum,
            maximumInCorner: maximum > 0 && corners.contains(maximum),
            monotonicityPenalty: monotonicity,
            roughness: roughness,
            scoreGained: scoreGained)
    }

    private static func exponent(_ tile: Int) -> Int {
        tile > 0 ? Int(log2(Double(tile))) : 0
    }

    private static func lineMonotonicityPenalty(_ line: [Int]) -> Int {
        var increasing = 0
        var decreasing = 0
        for (left, right) in zip(line, line.dropFirst()) {
            increasing += max(0, left - right)
            decreasing += max(0, right - left)
        }
        return min(increasing, decreasing)
    }

    private mutating func spawnTile() {
        var empty: [(Int, Int)] = []
        for row in 0..<Self.size {
            for column in 0..<Self.size where board[row][column] == 0 { empty.append((row, column)) }
        }
        guard !empty.isEmpty else { return }
        let location = empty[Int(rng.next() % UInt64(empty.count))]
        board[location.0][location.1] = rng.next() % 10 == 0 ? 4 : 2
    }
}

private struct SplitMix64: Sendable {
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
