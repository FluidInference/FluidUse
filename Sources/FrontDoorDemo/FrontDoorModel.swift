import FluidUse
import Foundation
import SwiftUI

/// Hand-off between the model loop (background, never waits on the UI) and the UI ticker (main actor, every ~50 ms).
final class RunBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var pending: [Processed] = []
    private var finishedAt: Double?
    private var cancelled = false
    private let start = Date()
    private var pausedSince: Date?
    private var pausedTotal = 0.0

    /// Seconds the run has been running, excluding paused time.
    func active(at now: Date = Date()) -> Double {
        lock.withLock { now.timeIntervalSince(start) - pausedTotal - (pausedSince.map { now.timeIntervalSince($0) } ?? 0) }
    }
    var isPaused: Bool { lock.withLock { pausedSince != nil } }
    func setPaused(_ on: Bool) {
        lock.withLock {
            if on, pausedSince == nil { pausedSince = Date() }
            if !on, let p = pausedSince {
                pausedTotal += Date().timeIntervalSince(p)
                pausedSince = nil
            }
        }
    }

    func push(_ p: Processed) { lock.withLock { pending.append(p) } }
    func finish() {
        let t = active()
        lock.withLock { finishedAt = t }
    }
    func cancel() { lock.withLock { cancelled = true } }
    var isCancelled: Bool { lock.withLock { cancelled } }
    /// Everything processed since the last drain, and the active time at which the model loop finished (nil while running).
    func drain() -> ([Processed], Double?) {
        lock.withLock {
            defer { pending.removeAll(keepingCapacity: true) }
            return (pending, finishedAt)
        }
    }
}

/// When each message "arrives": a burst that fills the queue, then a stream faster than the model can drain it,
/// so the model never waits for traffic.
/// Times are active seconds since the run started (paused time excluded), so pausing also stops the arrivals.
struct ArrivalSchedule: Sendable {
    static let burst = 150
    static let interval = 0.004  // s between arrivals after the burst (250 msg/s)
    let count: Int

    func arrived(at t: Double) -> Int {
        min(count, Self.burst + max(0, Int(t / Self.interval)))
    }
}

@available(macOS 15.0, *)
@MainActor
final class FrontDoorModel: ObservableObject {
    enum Phase { case loading, idle, running, done, failed }

    static let inboxRows = 30
    static let tick = Duration.milliseconds(50)

    @Published var phase = Phase.loading
    @Published var status = "Loading model…"
    @Published var selected: Processed?
    /// Published once per tick (coalesced).
    @Published private(set) var tally = Tally()
    @Published private(set) var arrivedCount = 0
    /// Every processed row of each lane, oldest first (the lanes show them newest on top).
    @Published private(set) var answeredRows: [Processed] = []
    @Published private(set) var jailbreakRows: [Processed] = []
    @Published private(set) var harmfulRows: [Processed] = []
    @Published private(set) var elapsed = 0.0  // s the run has been running (wall clock, paused time excluded)
    @Published private(set) var paused = false
    /// Whether any flight is in the air (un-pauses the overlay's TimelineView).
    @Published private(set) var flying = false
    let flightStore = FlightStore()

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
        do {
            let dir = try await Launch.resolveModel()
            print("loading \(dir.path)")
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
        buffer?.setPaused(false)
        ticker?.cancel()
        let buf = RunBuffer()
        let sched = ArrivalSchedule(count: Traffic.all.count)
        buffer = buf
        schedule = sched
        tally = Tally()
        arrivedCount = sched.arrived(at: 0)
        paused = false
        answeredRows = []
        jailbreakRows = []
        harmfulRows = []
        elapsed = 0
        flightStore.removeAll()
        flying = false
        selected = nil
        phase = .running
        print("run: \(Traffic.all.count) messages")

        // The model loop: strictly sequential, off the main actor, never waits on the UI.
        Task.detached(priority: .userInitiated) {
            for msg in Traffic.all {
                if buf.isCancelled { return }
                // paused: wait between messages (the one in flight has finished)
                while buf.isPaused, !buf.isCancelled { try? await Task.sleep(for: .milliseconds(20)) }
                while sched.arrived(at: buf.active()) <= msg.id, !buf.isCancelled {  // only if the model outruns the traffic
                    try? await Task.sleep(for: .milliseconds(1))
                }
                do {
                    let p = try await Engine.process(m, msg, arrived: Date())
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
                if self.apply(buf.drain(), sched, buf) { break }
            }
            // let the last flights land, then pause the overlay
            while !Task.isCancelled {
                guard let self, self.buffer === buf else { return }
                if !self.flightStore.prune(Date()) {
                    self.flying = false
                    return
                }
                try? await Task.sleep(for: Self.tick)
            }
        }
    }

    /// Pause / resume between messages: stops arrivals, the model loop and the msg/s clock; flights in the air still land.
    func togglePause() {
        guard phase == .running, let buf = buffer else { return }
        paused.toggle()
        buf.setPaused(paused)
        print(paused ? "paused (\(tally.processed) processed)" : "resumed")
    }

    /// One flight per processed message, staggered across the tick so they stream rather than clump;
    /// at most `FlightStore.maxConcurrent` in the air (the newest win).
    private func spawnFlights(_ batch: [Processed], _ now: Date) {
        let alive = flightStore.prune(now)
        guard !batch.isEmpty else {
            if !alive, flying { flying = false }
            return
        }
        let recent = batch.suffix(FlightStore.maxConcurrent)
        let step = 0.05 / Double(recent.count)
        flightStore.add(recent.enumerated().map { i, p in
            let to: FlightSpot, color: Color
            switch p.verdict {
            case .answered(let team): (to, color) = (.team(team), Theme.team(team))
            case .jailbreak: (to, color) = (.jailbreak, Theme.bad)
            case .harmful: (to, color) = (.harmful, Theme.warn)
            }
            let words = (p.blocked ? p.message.text : p.masked).split(separator: " ").prefix(2).joined(separator: " ")
            let label = words.count > 14 ? String(words.prefix(13)) + "…" : words
            return Flight(start: now.addingTimeInterval(Double(i) * step), to: to, color: color,
                          initial: String(p.message.user.first(where: \.isLetter) ?? "?").uppercased(), label: label)
        })
        if !flying { flying = true }
    }

    /// Returns true when the run is over.
    private func apply(_ drained: ([Processed], Double?), _ sched: ArrivalSchedule, _ buf: RunBuffer) -> Bool {
        let (batch, finished) = drained
        let now = Date()
        let active = buf.active(at: now)
        var t = tally
        var answered: [Processed] = [], jail: [Processed] = [], harm: [Processed] = []
        for p in batch {
            t.add(p)
            switch p.verdict {
            case .answered: answered.append(p)
            case .jailbreak: jail.append(p)
            case .harmful: harm.append(p)
            }
        }
        spawnFlights(batch, now)
        if !batch.isEmpty {
            tally = t
            // append only to the lanes that changed (one publish each per tick)
            if !answered.isEmpty { answeredRows.append(contentsOf: answered) }
            if !jail.isEmpty { jailbreakRows.append(contentsOf: jail) }
            if !harm.isEmpty { harmfulRows.append(contentsOf: harm) }
        }
        let arrived = sched.arrived(at: active)
        if arrived != arrivedCount { arrivedCount = arrived }
        if let finished {
            elapsed = finished
            phase = .done
            print(tally.summary)
            print(String(format: "end-to-end: %d messages in %.2f s = %.1f msg/s (wall clock, UI running)",
                         tally.processed, elapsed, wallPerSecond))
            print("run: done")
            return true
        }
        if !buf.isPaused { elapsed = active }  // frozen while paused
        return false
    }
}
