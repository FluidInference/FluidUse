import SwiftUI

// One overlay layer draws every message flying from the inbox head into its destination box.
// Frames come from anchor preferences (resolved once per layout); flights are spawned by the model's batched UI updates.

enum FlightSpot: Hashable {
    case inboxHead
    case team(String)
    case jailbreak
    case harmful
}

struct FlightSpotKey: PreferenceKey {
    static let defaultValue: [FlightSpot: Anchor<CGRect>] = [:]
    static func reduce(value: inout [FlightSpot: Anchor<CGRect>], nextValue: () -> [FlightSpot: Anchor<CGRect>]) {
        value.merge(nextValue()) { $1 }
    }
}

extension View {
    /// Marks this view's bounds as a flight endpoint.
    func flightSpot(_ spot: FlightSpot) -> some View {
        anchorPreference(key: FlightSpotKey.self, value: .bounds) { [spot: $0] }
    }
}

struct Flight {
    let start: Date
    let to: FlightSpot
    let color: Color
    let initial: String
    let label: String
}

/// Live flights (main actor only; not published — the overlay's TimelineView redraws from it every frame while `flying`).
@MainActor
final class FlightStore {
    static let duration = 0.55
    static let maxConcurrent = 28
    private(set) var flights: [Flight] = []

    func add(_ new: [Flight]) {
        flights.append(contentsOf: new)
        if flights.count > Self.maxConcurrent { flights.removeFirst(flights.count - Self.maxConcurrent) }  // drop the oldest
    }

    /// Drops landed flights; returns whether any are still in the air.
    func prune(_ now: Date) -> Bool {
        flights.removeAll { now.timeIntervalSince($0.start) > Self.duration }
        return !flights.isEmpty
    }

    func removeAll() { flights.removeAll() }
}

/// The single overlay: TimelineView + Canvas over the whole window, no hit testing.
@available(macOS 15.0, *)
struct FlightLayer: View {
    @EnvironmentObject var model: FrontDoorModel
    let rects: [FlightSpot: CGRect]

    var body: some View {
        TimelineView(.animation(minimumInterval: nil, paused: !model.flying)) { ctx in
            Canvas { gc, _ in
                guard let from = rects[.inboxHead] else { return }
                let now = ctx.date
                let a = CGPoint(x: from.midX, y: from.midY)
                for f in model.flightStore.flights {
                    guard let to = rects[f.to] else { continue }
                    let raw = now.timeIntervalSince(f.start) / FlightStore.duration
                    guard raw >= 0, raw <= 1 else { continue }
                    let p = 1 - pow(1 - raw, 3)  // ease-out
                    let b = CGPoint(x: to.midX, y: to.midY)
                    // gentle arc: control point above the straight line
                    let c = CGPoint(x: (a.x + b.x) / 2, y: min(a.y, b.y) - 24)
                    let x = (1 - p) * (1 - p) * a.x + 2 * (1 - p) * p * c.x + p * p * b.x
                    let y = (1 - p) * (1 - p) * a.y + 2 * (1 - p) * p * c.y + p * p * b.y
                    let tail = max(0, (raw - 0.7) / 0.3)  // last 30%: shrink and fade
                    let scale = 1 - 0.45 * tail
                    gc.opacity = 1 - 0.85 * tail
                    var g = gc
                    g.translateBy(x: x, y: y)
                    g.scaleBy(x: scale, y: scale)
                    let text = g.resolve(Text(f.label).font(.system(size: 9.5, weight: .semibold)).foregroundColor(.white))
                    let ts = text.measure(in: CGSize(width: 150, height: 20))
                    let w = ts.width + 22, h: CGFloat = 15
                    let pill = CGRect(x: -w / 2, y: -h / 2, width: w, height: h)
                    g.fill(Path(roundedRect: pill, cornerRadius: h / 2), with: .color(f.color.opacity(0.92)))
                    g.stroke(Path(roundedRect: pill, cornerRadius: h / 2), with: .color(.white.opacity(0.25)), lineWidth: 0.5)
                    let dot = CGRect(x: pill.minX + 2.5, y: -5, width: 10, height: 10)
                    g.fill(Path(ellipseIn: dot), with: .color(.black.opacity(0.35)))
                    g.draw(g.resolve(Text(f.initial).font(.system(size: 7, weight: .bold)).foregroundColor(.white)),
                           at: CGPoint(x: dot.midX, y: dot.midY))
                    g.draw(text, at: CGPoint(x: pill.minX + 15.5 + ts.width / 2, y: 0))
                }
            }
        }
        .allowsHitTesting(false)
    }
}
