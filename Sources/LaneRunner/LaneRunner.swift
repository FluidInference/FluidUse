import Foundation

/// Deterministic three-lane endless runner in the style of Subway Surfers. One step enters the next row.
public struct LaneRunner: Sendable, Equatable {
    public enum Action: String, CaseIterable, Sendable, Codable {
        case left, right, jump, slide, stay
    }

    public enum Obstacle: String, Sendable, Codable {
        case open, coin
        /// Low barrier: jump over it.
        case low
        /// Overhead barrier: slide under it.
        case high
        /// Blocks the whole lane: change lanes.
        case train

        var word: String {
            switch self {
            case .open: "open"
            case .coin: "coin"
            case .low: "low bar"
            case .high: "high bar"
            case .train: "train"
            }
        }
    }

    public static let lanes = 3
    /// Rows described to models and drawn by the app.
    public static let visibleRows = 3
    static let bufferedRows = 8
    /// Rows the consequence labels and heuristic search look past the next one.
    static let searchDepth = 3

    public internal(set) var lane = 1
    /// Upcoming rows; `rows[0]` is entered by the next step.
    public internal(set) var rows: [[Obstacle]] = []
    public private(set) var distance = 0
    public private(set) var coins = 0
    public private(set) var isOver = false
    public private(set) var crash: String?
    private var randomState: UInt64
    private var generated = 0

    public init(seed: UInt64) {
        randomState = seed
        while rows.count < Self.bufferedRows { appendRow() }
    }

    public var legalActions: [Action] {
        Action.allCases.filter { action in
            switch action {
            case .left: lane > 0
            case .right: lane < Self.lanes - 1
            case .jump, .slide, .stay: true
            }
        }
    }

    public mutating func step(_ action: Action) {
        guard !isOver else { return }
        let target = Self.target(lane: lane, action: action)
        let obstacle = rows[0][target]
        lane = target
        if let cause = Self.crashCause(obstacle, action) {
            isOver = true
            crash = cause
            return
        }
        if obstacle == .coin { coins += 1 }
        distance += 1
        rows.removeFirst()
        appendRow()
    }

    /// Heuristic reference: survive the search horizon, then prefer coins, then staying put.
    public var heuristicAction: Action {
        legalActions.max { lhs, rhs in heuristicValue(lhs) < heuristicValue(rhs) } ?? .stay
    }

    /// Human-readable model input: the current lane and the next three rows, left to right.
    public var observation: String {
        let ahead = rows.prefix(Self.visibleRows).map { row in row.map(\.word).joined(separator: ", ") }
        return "Lane \(lane + 1) of 3. Next row: \(ahead[0]). Then: \(ahead[1]). Then: \(ahead[2])."
    }

    /// Action labels state each move's consequence, as the Tetris harness does for placements.
    public func label(for action: Action) -> String {
        let target = Self.target(lane: lane, action: action)
        let obstacle = rows[0][target]
        if let cause = Self.crashCause(obstacle, action) { return "\(action.rawValue): crash, \(cause)" }
        var next = self
        next.step(action)
        let escape = next.survivableDepth(Self.searchDepth - 1) == Self.searchDepth - 1
        return "\(action.rawValue): safe\(obstacle == .coin ? ", coin" : "")\(escape ? "" : ", then trapped")"
    }

    /// Whether `action` crashes on the next row or leaves no way through the search horizon.
    public func isSafe(_ action: Action) -> Bool {
        guard Self.crashCause(rows[0][Self.target(lane: lane, action: action)], action) == nil else { return false }
        var next = self
        next.step(action)
        return next.survivableDepth(Self.searchDepth - 1) == Self.searchDepth - 1
    }

    static func target(lane: Int, action: Action) -> Int {
        switch action {
        case .left: max(lane - 1, 0)
        case .right: min(lane + 1, lanes - 1)
        case .jump, .slide, .stay: lane
        }
    }

    static func crashCause(_ obstacle: Obstacle, _ action: Action) -> String? {
        switch obstacle {
        case .open, .coin: nil
        case .train: "hit a train"
        case .low: action == .jump ? nil : "tripped on a low bar"
        case .high: action == .slide ? nil : "hit a high bar"
        }
    }

    private func heuristicValue(_ action: Action) -> Int {
        var next = self
        next.step(action)
        guard !next.isOver else { return -1_000 }
        // Same visible horizon as the labels: no peeking past the rows a model is shown.
        let depth = next.survivableDepth(Self.searchDepth - 1)
        return depth * 100 + (next.coins - coins) * 10 + (action == .stay ? 1 : 0)
    }

    /// Rows survivable within `limit` further steps under the best sequence.
    private func survivableDepth(_ limit: Int) -> Int {
        guard limit > 0, !isOver else { return 0 }
        var best = 0
        for action in legalActions {
            var next = self
            next.step(action)
            guard !next.isOver else { continue }
            best = max(best, 1 + next.survivableDepth(limit - 1))
            if best == limit { break }
        }
        return best
    }

    /// Obstacle rows alternate with open rows, and every obstacle row leaves at least one open lane,
    /// so two lane changes always reach it.
    private mutating func appendRow() {
        defer { generated += 1 }
        guard generated >= 2 else {
            rows.append(Array(repeating: .open, count: Self.lanes))
            return
        }
        if generated % 2 == 1 {
            rows.append((0..<Self.lanes).map { _ in nextRandom() % 4 == 0 ? .coin : .open })
            return
        }
        let free = Int(nextRandom() % UInt64(Self.lanes))
        rows.append(
            (0..<Self.lanes).map { index in
                guard index != free else { return nextRandom() % 3 == 0 ? .coin : .open }
                switch nextRandom() % 10 {
                case 0..<4: return .train
                case 4..<6: return .low
                case 6..<8: return .high
                default: return .open
                }
            })
    }

    private mutating func nextRandom() -> UInt64 {
        randomState &+= 0x9E37_79B9_7F4A_7C15
        var value = randomState
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }
}
