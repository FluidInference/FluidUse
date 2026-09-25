import Foundation
import SortAnything
import SwiftUI

/// Streams Fast Decisions documents through GLiNER2.5-Decide, all heads of a document in one call.
@MainActor
final class DecisionsModel: ObservableObject {
    enum Phase: Equatable {
        case loading(String)
        case ready
        case running
        case paused
        case finished
        case failed(String)
    }

    enum Mode: String, CaseIterable, Identifiable {
        /// One document at a time, answers revealed on screen.
        case show = "Show"
        /// Several calls in flight, as fast as the model goes.
        case turbo = "Turbo"
        var id: String { rawValue }
    }

    struct Decided: Identifiable {
        let document: DecisionDocument
        let result: DecisionSorter.Result
        var id: String { document.id }
    }

    struct DomainScore {
        var documents = 0
        var decisions = 0
        var correct = 0
        var accuracy: Double? { decisions > 0 ? Double(correct) / Double(decisions) : nil }
    }

    @Published private(set) var phase: Phase = .loading("Starting…")
    @Published var mode: Mode = .show
    /// Seconds each document stays on screen in Show mode.
    @Published var dwell: Double = 1.6
    @Published private(set) var current: Decided?
    @Published private(set) var scores: [String: DomainScore] = [:]
    @Published private(set) var documentsDone = 0
    @Published private(set) var decisions = 0
    @Published private(set) var correct = 0
    @Published private(set) var elapsed: Double = 0

    static let turboInFlight = 4
    static let turboFlush = 0.05
    /// DECISIONS_AUTOPLAY=N shows N documents in Show mode, then switches to Turbo, with no clicks.
    private let autoplayShowCount = ProcessInfo.processInfo.environment["DECISIONS_AUTOPLAY"].flatMap(Int.init)

    private(set) var documents: [DecisionDocument] = []
    private var queue: [DecisionDocument] = []
    private var sorter: DecisionSorter?
    private var runner: Task<Void, Never>?
    private var runStart: Date?
    private var elapsedBeforePause: Double = 0
    private var shownInShow = 0

    var total: Int { documents.count }
    var remaining: Int { queue.count }
    var decisionsPerSecond: Double { elapsed > 0 ? Double(decisions) / elapsed : 0 }
    var accuracy: Double? { decisions > 0 ? Double(correct) / Double(decisions) : nil }

    /// Set on the first call: SwiftUI can start the window's `.task` again, and a second run would toggle
    /// autoplay back off.
    private var preparing = false

    func prepare() async {
        guard !preparing else { return }
        preparing = true
        do {
            phase = .loading("Fetching Fast Decisions (1,700 documents)…")
            documents = try await FastDecisions.load()
            // Interleave domains so every tile moves from the start.
            documents = interleaved(documents)
            phase = .loading("Loading GLiNER2.5-Decide (first run downloads 923 MB)…")
            sorter = try await DecisionSorter.load { [weak self] file, bytes in
                guard bytes > 0 else { return }
                Task { @MainActor in self?.phase = .loading("Downloaded \(file)") }
            }
            phase = .loading("Compiling for this Mac…")
            _ = try await sorter?.decide(documents[0])
            reset()
            if autoplayShowCount != nil {
                mode = .show
                toggleRun()
            }
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    private func interleaved(_ documents: [DecisionDocument]) -> [DecisionDocument] {
        let byDomain = Dictionary(grouping: documents, by: \.domain)
        let longest = byDomain.values.map(\.count).max() ?? 0
        return (0..<longest).flatMap { index in
            FastDecisions.domains.compactMap { byDomain[$0].flatMap { index < $0.count ? $0[index] : nil } }
        }
    }

    func reset() {
        runner?.cancel()
        runner = nil
        queue = documents
        current = nil
        scores = Dictionary(uniqueKeysWithValues: FastDecisions.domains.map { ($0, DomainScore()) })
        documentsDone = 0
        decisions = 0
        correct = 0
        elapsed = 0
        elapsedBeforePause = 0
        shownInShow = 0
        phase = .ready
    }

    func toggleRun() {
        switch phase {
        case .running:
            runner?.cancel()
            runner = nil
            elapsedBeforePause = elapsed
            phase = .paused
        case .ready, .paused:
            phase = .running
            runStart = Date()
            runner = Task { [weak self] in await self?.run() }
        default:
            break
        }
    }

    private func run() async {
        guard let sorter else { return }
        while !Task.isCancelled, !queue.isEmpty {
            if mode == .turbo {
                await runTurbo(sorter)
            } else {
                await runShowStep(sorter)
            }
        }
        if !Task.isCancelled, queue.isEmpty {
            tick()
            phase = .finished
            print(
                "finished \(documentsDone) documents, \(decisions) decisions in \(String(format: "%.2f", elapsed)) s "
                    + "(\(String(format: "%.0f", decisionsPerSecond)) decisions/s), "
                    + "correct \(String(format: "%.1f", (accuracy ?? 0) * 100))%")
        }
    }

    private func runShowStep(_ sorter: DecisionSorter) async {
        guard let document = queue.first, let result = try? await sorter.decide(document) else { return }
        withAnimation(.easeOut(duration: 0.25)) {
            current = Decided(document: document, result: result)
            land([Decided(document: document, result: result)])
        }
        shownInShow += 1
        if let count = autoplayShowCount, shownInShow >= count { mode = .turbo }
        try? await Task.sleep(for: .seconds(dwell))
    }

    /// Runs the model off the main thread with several calls in flight and streams each result.
    private nonisolated static func turboStream(
        _ sorter: DecisionSorter, documents: [DecisionDocument], inFlight: Int
    ) -> AsyncStream<Decided> {
        AsyncStream { continuation in
            let producer = Task.detached {
                await withTaskGroup(of: Decided?.self) { group in
                    var next = 0
                    func launch() {
                        guard next < documents.count, !Task.isCancelled else { return }
                        let document = documents[next]
                        next += 1
                        group.addTask {
                            guard let result = try? await sorter.decide(document) else { return nil }
                            return Decided(document: document, result: result)
                        }
                    }
                    for _ in 0..<inFlight { launch() }
                    while let decided = await group.next() {
                        if let decided { continuation.yield(decided) }
                        launch()
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in producer.cancel() }
        }
    }

    private func runTurbo(_ sorter: DecisionSorter) async {
        let stream = Self.turboStream(sorter, documents: queue, inFlight: Self.turboInFlight)
        var pending: [Decided] = []
        var lastFlush = Date()
        for await decided in stream {
            pending.append(decided)
            if Date().timeIntervalSince(lastFlush) >= Self.turboFlush {
                current = pending.last
                land(pending)
                pending.removeAll(keepingCapacity: true)
                lastFlush = Date()
            }
            if Task.isCancelled || mode != .turbo { break }
        }
        if let last = pending.last { current = last }
        land(pending)
    }

    private func land(_ batch: [Decided]) {
        guard !batch.isEmpty else { return }
        var updated = scores
        for decided in batch {
            var score = updated[decided.document.domain, default: DomainScore()]
            score.documents += 1
            for answer in decided.result.answers {
                score.decisions += 1
                score.correct += answer.correct ? 1 : 0
                decisions += 1
                correct += answer.correct ? 1 : 0
            }
            updated[decided.document.domain] = score
        }
        scores = updated
        let done = Set(batch.map(\.id))
        queue.removeAll { done.contains($0.id) }
        documentsDone += batch.count
        tick()
    }

    private func tick() {
        if let runStart { elapsed = elapsedBeforePause + Date().timeIntervalSince(runStart) }
    }
}
