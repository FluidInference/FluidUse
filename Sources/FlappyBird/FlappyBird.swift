import Foundation

/// Deterministic Flappy Bird physics. Coordinates increase downward; one step is 1/60 second.
public struct FlappyBird: Sendable, Equatable {
    public enum Action: String, Sendable, Codable { case flap, coast }

    public struct Pipe: Sendable, Equatable, Identifiable {
        public let id: Int
        public internal(set) var x: Double
        public let center: Double
        public internal(set) var passed = false
    }

    public static let width = 420.0
    public static let height = 600.0
    public static let birdX = 100.0
    public static let radius = 12.0
    public static let pipeWidth = 64.0
    public static let gap = 180.0
    public static let stepSeconds = 1.0 / 60
    public static let decisionFrames = 6

    public internal(set) var y = 300.0
    public internal(set) var velocity = 0.0
    public internal(set) var pipes: [Pipe] = []
    public private(set) var frames = 0
    public private(set) var score = 0
    public private(set) var isOver = false
    private var randomState: UInt64
    private var nextID = 0

    public init(seed: UInt64) {
        randomState = seed
        for x in [420.0, 670.0, 920.0] { addPipe(x: x) }
    }

    public var seconds: Double { Double(frames) * Self.stepSeconds }
    public var nextPipe: Pipe? { pipes.first { $0.x + Self.pipeWidth >= Self.birdX - Self.radius } }

    /// A flap is an impulse on this frame only; coast lets gravity act.
    public mutating func step(_ action: Action = .coast) {
        guard !isOver else { return }
        if action == .flap { velocity = -260 }
        velocity += 800 * Self.stepSeconds
        y += velocity * Self.stepSeconds
        frames += 1
        for index in pipes.indices { pipes[index].x -= 135 * Self.stepSeconds }
        let hitBoundary = y - Self.radius <= 0 || y + Self.radius >= Self.height
        let hitPipe = pipes.contains { pipe in
            let overlaps =
                Self.birdX + Self.radius >= pipe.x
                && Self.birdX - Self.radius <= pipe.x + Self.pipeWidth
            return overlaps
                && (y - Self.radius <= pipe.center - Self.gap / 2
                    || y + Self.radius >= pipe.center + Self.gap / 2)
        }
        if hitBoundary || hitPipe {
            isOver = true
            return
        }
        for index in pipes.indices
        where !pipes[index].passed
            && pipes[index].x + Self.pipeWidth < Self.birdX - Self.radius
        {
            pipes[index].passed = true
            score += 1
        }
        pipes.removeAll { $0.x + Self.pipeWidth < 0 }
        if let last = pipes.last, last.x < Self.width + 250 { addPipe(x: last.x + 250) }
    }

    /// Transparent control policy; never used to override a model decision.
    public var heuristicAction: Action {
        let target = nextPipe?.center ?? Self.height / 2
        return velocity >= 0 && y + velocity * 0.12 > target + 15 ? .flap : .coast
    }

    /// Both actions receive the same 300 ms, no-further-flaps forecast.
    public func projection(_ action: Action) -> FlappyBird {
        var copy = self
        copy.step(action)
        for _ in 1..<18 { copy.step() }
        return copy
    }

    /// Keep a model choice unless its short forecast is clearly unsafe relative to the other action.
    public func guardedAction(preferred: Action) -> Action {
        let alternative: Action = preferred == .flap ? .coast : .flap
        let preferredProjection = projection(preferred)
        let alternativeProjection = projection(alternative)
        if preferredProjection.isOver && !alternativeProjection.isOver { return alternative }
        if alternativeProjection.isOver { return preferred }
        let center = nextPipe?.center ?? Self.height / 2
        let halfWidth = Self.gap / 2 - Self.radius - 12
        let preferredDistance = abs(preferredProjection.y - center)
        let alternativeDistance = abs(alternativeProjection.y - center)
        if preferredDistance > halfWidth && alternativeDistance <= halfWidth { return alternative }
        if preferredDistance > halfWidth && alternativeDistance < preferredDistance { return alternative }
        return preferred
    }

    /// Human-readable model input, including motion and the next pipe's geometry.
    public var observation: String {
        let target = nextPipe?.center ?? Self.height / 2
        let offset = Int(y - target)
        return "Bird \(abs(offset)) px \(offset >= 0 ? "below" : "above") gap center; "
            + "\(velocity >= 0 ? "falling" : "rising") \(Int(abs(velocity))) px/s. "
            + "Pipe \(Int((nextPipe?.x ?? Self.width) - Self.birdX)) px ahead."
    }

    /// Action labels expose simulated consequences; this is not a raw-image policy.
    public func label(for action: Action) -> String {
        let predicted = projection(action)
        if predicted.isOver { return "\(action.rawValue): collision within 300 ms" }
        let target = nextPipe?.center ?? Self.height / 2
        let offset = Int(predicted.y - target)
        let relation = offset >= 0 ? "below" : "above"
        let inside = abs(predicted.y - target) + Self.radius < Self.gap / 2
        return "\(action.rawValue): alive, \(abs(offset)) px \(relation) gap center, "
            + "\(inside ? "inside" : "outside") opening"
    }

    private mutating func addPipe(x: Double) {
        randomState &+= 0x9E37_79B9_7F4A_7C15
        var value = randomState
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        value ^= value >> 31
        let center = 190 + Double(value % 221)
        pipes.append(Pipe(id: nextID, x: x, center: center))
        nextID += 1
    }
}
