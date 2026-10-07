import FluidUse
import Foundation
import SwiftUI

struct InboxItem: Identifiable, Hashable {
    let message: InboundMessage
    let arrived: Date
    var id: Int { message.id }
}

@available(macOS 15.0, *)
@MainActor
final class FrontDoorModel: ObservableObject {
    enum Phase { case loading, idle, running, done, failed }

    @Published var phase = Phase.loading
    @Published var status = "Loading model…"
    /// Waiting messages, oldest first (newest at the bottom, like a chat).
    @Published var inbox: [InboxItem] = []
    @Published var scanning: Int?
    /// Processed messages in processing order.
    @Published var processed: [Processed] = []
    @Published var selected: Processed?

    var tally: Tally { Tally(processed) }

    private var manager: Vela2Manager?
    private var arrivals: Task<Void, Never>?
    private var runner: Task<Void, Never>?
    private var generation = 0

    static let arrivalInterval = Duration.milliseconds(40)
    /// How long a message stays under the scanner (the model itself needs ~10 ms; this is for the eye).
    static let scanDwell = Duration.zero  // speed first: no artificial dwell

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

    /// Start (or replay): messages arrive one by one, each is screened in turn, then the run stops.
    func run() {
        guard let m = manager, phase != .loading, phase != .failed else { return }
        arrivals?.cancel()
        runner?.cancel()
        generation += 1
        let gen = generation
        inbox = []
        processed = []
        scanning = nil
        selected = nil
        phase = .running
        print("run: \(Traffic.all.count) messages")

        arrivals = Task { [weak self] in
            for msg in Traffic.all {
                guard let self, gen == self.generation, !Task.isCancelled else { return }
                self.inbox.append(InboxItem(message: msg, arrived: Date()))
                try? await Task.sleep(for: Self.arrivalInterval)
            }
        }
        runner = Task { [weak self] in
            await self?.consume(m, gen)
        }
    }

    /// Strictly sequential: the oldest waiting message, one model call chain at a time.
    private func consume(_ m: Vela2Manager, _ gen: Int) async {
        var next = 0
        while next < Traffic.all.count {
            guard gen == generation, !Task.isCancelled else { return }
            let want = Traffic.all[next].id
            guard let item = inbox.first(where: { $0.id == want }) else {
                try? await Task.sleep(for: .milliseconds(40))
                continue
            }
            scanning = item.id
            let began = ContinuousClock.now
            let msg = item.message, arrived = item.arrived
            let result = await Task.detached { try? await Engine.process(m, msg, arrived: arrived) }.value
            let rest = Self.scanDwell - (ContinuousClock.now - began)
            if rest > .zero { try? await Task.sleep(for: rest) }
            guard gen == generation, !Task.isCancelled else { return }
            if let result { print(Engine.log(result)) } else { print("error processing message \(msg.id + 1)") }
            inbox.removeAll { $0.id == want }
            if let result { processed.append(result) }
            scanning = nil
            next += 1
            await Task.yield()  // let the UI draw; no artificial pause
        }
        guard gen == generation else { return }
        withAnimation(.spring(duration: 0.5)) { phase = .done }
        print(tally.summary)
        print("run: done")
    }
}
