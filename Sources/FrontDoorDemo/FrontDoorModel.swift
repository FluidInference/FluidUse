import FluidUse
import Foundation
import SwiftUI

/// Hand-off between the model loop (background, never waits on the UI) and the UI ticker (main actor, every ~50 ms).
final class RunBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var pending: [Processed] = []
    private var finishedAt: Date?
    private var cancelled = false

    func push(_ p: Processed) { lock.withLock { pending.append(p) } }
    func finish() { lock.withLock { finishedAt = Date() } }
    func cancel() { lock.withLock { cancelled = true } }
    var isCancelled: Bool { lock.withLock { cancelled } }
    /// Everything processed since the last drain, and when the model loop finished (nil while running).
    func drain() -> ([Processed], Date?) {
        lock.withLock {
            defer { pending.removeAll(keepingCapacity: true) }
            return (pending, finishedAt)
        }
    }
}

/// When each message "arrives": a burst that fills the queue, then a stream faster than the model can drain it,
/// so the model never waits for traffic.
struct ArrivalSchedule: Sendable {
    static let burst = 150
    static let interval = 0.004  // s between arrivals after the burst (250 msg/s)
    let start: Date
    let count: Int

    func arrived(at now: Date) -> Int {
        let t = now.timeIntervalSince(start)
        return min(count, Self.burst + max(0, Int(t / Self.interval)))
    }

    func time(of index: Int) -> Date {
        start.addingTimeInterval(Double(max(0, index - Self.burst + 1)) * Self.interval)
    }
}

@available(macOS 15.0, *)
@MainActor
final class FrontDoorModel: ObservableObject {
    enum Phase { case loading, idle, running, done, failed }

    static let laneRows = 20
    static let inboxRows = 30
    static let tick = Duration.milliseconds(50)

    @Published var phase = Phase.loading
    @Published var status = "Loading model…"
    @Published var selected: Processed?
    /// Published once per tick (coalesced).
    @Published private(set) var tally = Tally()
    @Published private(set) var arrivedCount = 0
    @Published private(set) var answeredRecent: [Processed] = []  // newest first, at most `laneRows`
    @Published private(set) var jailbreakRecent: [Processed] = []
    @Published private(set) var harmfulRecent: [Processed] = []
    @Published private(set) var elapsed = 0.0  // s since the run started (wall clock)

    var processedCount: Int { tally.processed }
    /// Waiting messages, oldest first (the first one is under the scanner).
    var waiting: ArraySlice<InboundMessage> { Traffic.all[min(processedCount, arrivedCount)..<arrivedCount] }
    /// End-to-end throughput: messages fully screened per wall-clock second.
    var wallPerSecond: Double { elapsed > 0 ? Double(processedCount) / elapsed : 0 }

    private var manager: Vela2Manager?
    private var buffer: RunBuffer?
    private var ticker: Task<Void, Never>?
    private var schedule: ArrivalSchedule?

    // MARK: startup

    func start() async {
        guard manager == nil else { return }
        let dir = Launch.modelDirectory
        print("loading \(dir.path)")
        do {
            let m = try await Task.detached { () async throws -> Vela2Manager in
                let m = try await Vela2Manager.load(from: dir)
                try await m.warm()
                // one pass over both call shapes so the first real message is steady-state
                _ = try await Engine.guardCheck(m, "hello")
                _ = try await Engine.routeCheck(m, "hello")
                return m
            }.value
            manager = m
            phase = .idle
            status = "\(m.modelName) ready · ANE ≤ \(m.aneMaxLength) tokens, GPU above"
            print("ready: \(m.modelName) (Core ML: ANE <= \(m.aneMaxLength) tokens, GPU above)")
            if Launch.demo { run() }
        } catch {
            phase = .failed
            status = "Failed to load model: \(error.localizedDescription)"
            print("load failed: \(error)")
        }
    }

    // MARK: run

    /// Start (or replay): the traffic arrives, every message is screened in order as fast as the model allows, then the run stops.
    func run() {
        guard let m = manager, phase != .loading, phase != .failed else { return }
        buffer?.cancel()
        ticker?.cancel()
        let buf = RunBuffer()
        let sched = ArrivalSchedule(start: Date(), count: Traffic.all.count)
        buffer = buf
        schedule = sched
        tally = Tally()
        arrivedCount = sched.arrived(at: sched.start)
        answeredRecent = []
        jailbreakRecent = []
        harmfulRecent = []
        elapsed = 0
        selected = nil
        phase = .running
        print("run: \(Traffic.all.count) messages")

        // The model loop: strictly sequential, off the main actor, never waits on the UI.
        Task.detached(priority: .userInitiated) {
            for msg in Traffic.all {
                if buf.isCancelled { return }
                while sched.arrived(at: Date()) <= msg.id {  // only waits if the model ever outruns the traffic
                    try? await Task.sleep(for: .milliseconds(1))
                }
                do {
                    let p = try await Engine.process(m, msg, arrived: sched.time(of: msg.id))
                    buf.push(p)
                    print(Engine.log(p))
                } catch {
                    print("error processing message \(msg.id + 1): \(error)")
                }
            }
            buf.finish()
        }
        // The UI ticker: applies whatever is done, at most every `tick`.
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: Self.tick)
                guard let self, self.buffer === buf else { return }
                if self.apply(buf.drain(), sched) { return }
            }
        }
    }

    /// Returns true when the run is over.
    private func apply(_ drained: ([Processed], Date?), _ sched: ArrivalSchedule) -> Bool {
        let (batch, finished) = drained
        let now = Date()
        var t = tally
        var answered = answeredRecent, jail = jailbreakRecent, harm = harmfulRecent
        for p in batch {
            t.add(p)
            switch p.verdict {
            case .answered: answered.insert(p, at: 0)
            case .jailbreak: jail.insert(p, at: 0)
            case .harmful: harm.insert(p, at: 0)
            }
        }
        if !batch.isEmpty {
            tally = t
            answeredRecent = Array(answered.prefix(Self.laneRows))
            jailbreakRecent = Array(jail.prefix(Self.laneRows))
            harmfulRecent = Array(harm.prefix(Self.laneRows))
        }
        let arrived = sched.arrived(at: now)
        if arrived != arrivedCount { arrivedCount = arrived }
        if let finished {
            elapsed = finished.timeIntervalSince(sched.start)
            phase = .done
            print(tally.summary)
            print(String(format: "end-to-end: %d messages in %.2f s = %.1f msg/s (wall clock, UI running)",
                         tally.processed, elapsed, wallPerSecond))
            print("run: done")
            return true
        }
        elapsed = now.timeIntervalSince(sched.start)
        return false
    }
}
