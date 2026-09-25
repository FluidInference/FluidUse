import Foundation
import SortAnything
import SwiftUI

/// Drives the stream: pulls DBpedia items, sorts them with Decide, and keeps the live statistics.
@MainActor
final class SortModel: ObservableObject {
    enum Phase: Equatable {
        case loading(String)
        case ready
        case running
        case paused
        case finished
        case failed(String)
    }

    enum Mode: String, CaseIterable, Identifiable {
        /// One card at a time, animated into its bucket.
        case show = "Show"
        /// Several calls in flight, as fast as the model goes.
        case turbo = "Turbo"
        var id: String { rawValue }
    }

    struct Placed: Identifiable {
        let item: SortItem
        let result: Sorter.Result
        var id: Int { item.id }
        var matchesGold: Bool { result.category == item.gold }
    }

    struct Flight: Identifiable {
        let placed: Placed
        var arrived = false
        var id: Int { placed.id }
    }

    @Published private(set) var phase: Phase = .loading("Starting…")
    @Published var mode: Mode = .show
    /// Cards per second in Show mode.
    @Published var pace: Double = 12
    @Published private(set) var categories: [String] = DBpediaSample.categories
    @Published private(set) var buckets: [String: [Placed]] = [:]
    @Published private(set) var queue: [SortItem] = []
    /// Items already in a bucket, so a card that lands after a pause and resume is not counted twice.
    private var landed: Set<Int> = []
    /// Cards currently travelling from the incoming slot to their bucket.
    @Published private(set) var flights: [Flight] = []
    @Published private(set) var sorted = 0
    @Published private(set) var scored = 0
    @Published private(set) var correct = 0
    @Published private(set) var elapsed: Double = 0
    @Published private(set) var lastMilliseconds: Double = 0

    let total = 1000
    static let turboInFlight = 4
    static let turboFlush = 0.05
    /// Cards that visibly fly per Turbo flush (about 60 per second at a 50 ms flush).
    static let turboFlightsPerFlush = 3
    private let autoplayShowCount = ProcessInfo.processInfo.environment["SORT_AUTOPLAY"].flatMap(Int.init)
    /// SORT_LOG=1 prints every decision to stdout (for a terminal next to the window).
    private let logDecisions = ProcessInfo.processInfo.environment["SORT_LOG"] == "1"
    private var shownInShow = 0
    private var items: [SortItem] = []
    private var sorter: Sorter?
    private var runner: Task<Void, Never>?
    private var modelMilliseconds: [Double] = []
    private var runStart: Date?
    private var elapsedBeforePause: Double = 0

    var itemsPerSecond: Double { elapsed > 0 ? Double(sorted) / elapsed : 0 }
    var accuracy: Double? { scored > 0 ? Double(correct) / Double(scored) : nil }
    var medianMilliseconds: Double? {
        guard !modelMilliseconds.isEmpty else { return nil }
        return modelMilliseconds.sorted()[modelMilliseconds.count / 2]
    }
    var maximumCategories: Int { sorter?.maximumCategories ?? 32 }

    /// Set on the first call: SwiftUI can start the window's `.task` again, and a second run would toggle
    /// autoplay back off.
    private var preparing = false

    func prepare() async {
        guard !preparing else { return }
        preparing = true
        do {
            phase = .loading("Fetching 1,000 Wikipedia abstracts…")
            items = try await DBpediaSample.load(count: total)
            phase = .loading("Loading GLiNER2.5-Decide (first run downloads 923 MB)…")
            sorter = try await Sorter.load { [weak self] file, bytes in
                guard bytes > 0 else { return }
                Task { @MainActor in self?.phase = .loading("Downloaded \(file)") }
            }
            phase = .loading("Compiling for this Mac…")
            _ = try await sorter?.sort(items[0], into: categories)
            reset()
            // SORT_AUTOSTART=show|turbo starts a run without a click; SORT_AUTOPLAY=N flies N cards in Show mode and
            // then switches to Turbo (for recordings).
            let environment = ProcessInfo.processInfo.environment
            print("ready at \(Date().timeIntervalSince1970)")
            if autoplayShowCount != nil {
                // Give a recorder a moment to show the empty board before cards start flying.
                try? await Task.sleep(for: .seconds(1.5))
                mode = .show
                toggleRun()
            } else if let start = environment["SORT_AUTOSTART"], let mode = Mode(rawValue: start.capitalized) {
                self.mode = mode
                toggleRun()
            }
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func reset() {
        runner?.cancel()
        runner = nil
        categories = DBpediaSample.categories
        buckets = Dictionary(uniqueKeysWithValues: categories.map { ($0, []) })
        queue = items
        flights = []
        landed = []
        sorted = 0
        scored = 0
        correct = 0
        elapsed = 0
        elapsedBeforePause = 0
        modelMilliseconds = []
        lastMilliseconds = 0
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

    /// Adds a category; it takes effect from the next item.
    func addCategory(_ raw: String) {
        let name = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !name.isEmpty, !categories.contains(name), categories.count < maximumCategories else { return }
        categories.append(name)
        buckets[name] = buckets[name] ?? []
    }

    /// Removes a category from future decisions; items already sorted stay in its bucket.
    func removeCategory(_ name: String) {
        guard categories.count > 2 else { return }
        categories.removeAll { $0 == name }
        if buckets[name]?.isEmpty == true { buckets[name] = nil }
    }

    var bucketNames: [String] {
        categories + buckets.keys.filter { !categories.contains($0) }.sorted()
    }

    private func run() async {
        guard let sorter else { return }
        while !Task.isCancelled, !queue.isEmpty {
            if mode == .turbo {
                await runTurbo(sorter)
            } else {
                await runShow(sorter)
            }
        }
        if !Task.isCancelled, queue.isEmpty {
            tick()
            phase = .finished
            print("finished at \(Date().timeIntervalSince1970)")
            print(
                "finished \(sorted) items in \(String(format: "%.2f", elapsed)) s "
                    + "(\(String(format: "%.0f", itemsPerSecond)) items/s), "
                    + "matches \(String(format: "%.1f", (accuracy ?? 0) * 100))% (\(mode.rawValue))")
        }
    }

    /// Seconds a card takes to fly from the incoming slot to its bucket.
    static let travel = 0.45

    /// Classifies ahead in the background and launches one card every `1 / pace` seconds; several cards can be in
    /// the air at once. Returns when the queue is empty, the run is paused, or the mode changes.
    private func runShow(_ sorter: Sorter) async {
        let stream = Self.turboStream(sorter, items: queue, categories: categories, inFlight: 2)
        for await placed in stream {
            if Task.isCancelled || mode != .show { break }
            launch(placed)
            let started = Date()
            try? await Task.sleep(for: .seconds(1 / pace))
            if Date().timeIntervalSince(started) < 1 / pace { break }  // cancelled mid-sleep
        }
        // On a mode switch, let cards already in the air land before Turbo snapshots the queue. On pause, return at
        // once: they land by themselves, and `land` ignores anything sorted twice.
        while !Task.isCancelled, !flights.isEmpty { try? await Task.sleep(for: .milliseconds(20)) }
    }

    /// A decorative flight only animates: the item has already been counted by Turbo.
    private func launch(_ placed: Placed, decorative: Bool = false) {
        guard !flights.contains(where: { $0.id == placed.id }) else { return }
        flights.append(Flight(placed: placed))
        Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(20))
            withAnimation(.easeInOut(duration: Self.travel)) {
                if let index = self?.flights.firstIndex(where: { $0.id == placed.id }) {
                    self?.flights[index].arrived = true
                }
            }
            try? await Task.sleep(for: .seconds(Self.travel))
            guard let self else { return }
            flights.removeAll { $0.id == placed.id }
            guard !decorative else { return }
            land([placed])
            shownInShow += 1
            if let count = autoplayShowCount, shownInShow >= count { mode = .turbo }
        }
    }

    /// Sorts `items` with `inFlight` calls always running, independent of the main thread, and streams each result.
    private nonisolated static func turboStream(
        _ sorter: Sorter, items: [SortItem], categories: [String], inFlight: Int
    ) -> AsyncStream<Placed> {
        AsyncStream { continuation in
            let producer = Task.detached {
                await withTaskGroup(of: Placed?.self) { group in
                    var next = 0
                    func launch() {
                        guard next < items.count, !Task.isCancelled else { return }
                        let item = items[next]
                        next += 1
                        group.addTask {
                            guard let result = try? await sorter.sort(item, into: categories) else { return nil }
                            return Placed(item: item, result: result)
                        }
                    }
                    for _ in 0..<inFlight { launch() }
                    while let placed = await group.next() {
                        if let placed { continuation.yield(placed) }
                        launch()
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in producer.cancel() }
        }
    }

    /// Runs the model pipeline off the main thread and applies results to the screen every `turboFlush` seconds,
    /// without per-card animation, until the queue is empty, the run is paused, or the mode changes.
    private func runTurbo(_ sorter: Sorter) async {
        let stream = Self.turboStream(
            sorter, items: queue, categories: categories, inFlight: Self.turboInFlight)
        var pending: [Placed] = []
        var lastFlush = Date()
        for await placed in stream {
            pending.append(placed)
            if Date().timeIntervalSince(lastFlush) >= Self.turboFlush {
                // Counts land at full speed; a few cards per flush also fly, so Turbo still shows motion.
                for placed in pending.suffix(Self.turboFlightsPerFlush) where flights.count < 40 {
                    launch(placed, decorative: true)
                }
                land(pending)
                pending.removeAll(keepingCapacity: true)
                lastFlush = Date()
            }
            if Task.isCancelled || mode != .turbo { break }
        }
        land(pending)
    }

    private func land(_ incoming: [Placed]) {
        let batch = incoming.filter { self.landed.insert($0.id).inserted }
        guard !batch.isEmpty else { return }
        var updated = buckets
        for placed in batch {
            updated[placed.result.category, default: []].insert(placed, at: 0)
            if categories.contains(placed.item.gold) {
                scored += 1
                correct += placed.matchesGold ? 1 : 0
            }
            modelMilliseconds.append(placed.result.milliseconds)
        }
        buckets = updated
        let batchIDs = Set(batch.map(\.id))
        queue.removeAll { batchIDs.contains($0.id) }
        sorted += batch.count
        lastMilliseconds = batch.last?.result.milliseconds ?? lastMilliseconds
        tick()
        if logDecisions { log(batch) }
    }

    private func log(_ batch: [Placed]) {
        let (cyan, yellow, red, green, dim, reset) =
            ("\u{1B}[1;36m", "\u{1B}[33m", "\u{1B}[31m", "\u{1B}[32m", "\u{1B}[2m", "\u{1B}[0m")
        var lines = ""
        for (offset, placed) in batch.enumerated() {
            let number = sorted - batch.count + offset + 1
            let mark = placed.matchesGold ? "\(green)✓\(reset)" : "\(red)✗ label: \(placed.item.gold)\(reset)"
            lines += "\(cyan)▶ #\(number) \(placed.item.title)\(reset)\n"
            lines +=
                "  \(yellow)→ \(placed.result.category)\(reset) · "
                + "\(Int(placed.result.confidence * 100))% · "
                + "\(red)model call \(String(format: "%.1f", placed.result.milliseconds)) ms\(reset)  \(mark)\n"
        }
        lines += "\(dim)  sorted \(sorted)/\(total) · \(String(format: "%.1f", elapsed)) s\(reset)\n"
        print(lines, terminator: "")
    }

    private func tick() {
        if let runStart { elapsed = elapsedBeforePause + Date().timeIntervalSince(runStart) }
    }
}
