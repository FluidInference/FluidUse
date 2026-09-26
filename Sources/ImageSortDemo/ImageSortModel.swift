import CoreGraphics
import Foundation
import ImageIO
import ImageSort
import SwiftUI

/// Drives the stream: loads Pets photos, sorts them with SigLIP 2, and grows the photo chart.
@MainActor
final class ImageSortModel: ObservableObject {
    enum Phase: Equatable {
        case loading(String)
        case ready
        case running
        case paused
        case finished
        case failed(String)
    }

    enum Mode: String, CaseIterable, Identifiable {
        /// One photo at a time, with its top-5 breeds.
        case show = "Show"
        /// Several calls in flight, as fast as the model goes.
        case turbo = "Turbo"
        var id: String { rawValue }
    }

    struct Placed: Identifiable, Sendable {
        let item: PetItem
        let result: ImageSorter.Result
        let tile: CGImage?
        var id: Int { item.id }
        var matchesGold: Bool { result.breed == item.breed }
    }

    @Published private(set) var phase: Phase = .loading("Starting…")
    @Published var mode: Mode = .show
    /// Photos per second in Show mode.
    @Published var pace: Double = 8
    @Published private(set) var remaining = 0
    @Published private(set) var sorted = 0
    @Published private(set) var correct = 0
    @Published private(set) var elapsed: Double = 0
    @Published private(set) var counts: [String: Int] = [:]
    @Published private(set) var chartImage: CGImage?
    /// The photo just sorted, shown large with its top-5 breeds.
    @Published private(set) var current: Placed?
    @Published private(set) var currentImage: CGImage?
    /// Where the latest tile landed, for the highlight in Show mode.
    @Published private(set) var lastSlot: CGRect?

    let breeds = PetsSample.breeds
    let total: Int
    static let turboInFlight = 4
    static let turboFlush = 0.05
    static let cats: Set<String> = [
        "abyssinian", "bengal", "birman", "bombay", "british shorthair", "egyptian mau", "maine coon", "persian",
        "ragdoll", "russian blue", "siamese", "sphynx",
    ]

    private let environment = ProcessInfo.processInfo.environment
    /// IMAGE_SORT_LOG=1 prints every decision to stdout (for a terminal next to the window).
    private let logDecisions = ProcessInfo.processInfo.environment["IMAGE_SORT_LOG"] == "1"
    private var chart: PhotoChart?
    private var queue: [PetItem] = []
    private var landed: Set<Int> = []
    private var items: [PetItem] = []
    private var sorter: ImageSorter?
    private var runner: Task<Void, Never>?
    private var modelMilliseconds: [Double] = []
    private var runStart: Date?
    private var elapsedBeforePause: Double = 0
    private var shownInShow = 0
    private var preparing = false

    init() {
        total = ProcessInfo.processInfo.environment["IMAGE_SORT_COUNT"].flatMap(Int.init) ?? 1000
    }

    var photosPerSecond: Double { elapsed > 0 ? Double(sorted) / elapsed : 0 }
    var accuracy: Double? { sorted > 0 ? Double(correct) / Double(sorted) : nil }
    var medianMilliseconds: Double? {
        guard !modelMilliseconds.isEmpty else { return nil }
        return modelMilliseconds.sorted()[modelMilliseconds.count / 2]
    }

    func prepare() async {
        guard !preparing else { return }
        preparing = true
        do {
            phase = .loading("Fetching \(total) Oxford-IIIT Pets photos…")
            items = try await PetsSample.load(count: total) { [weak self] done, all in
                Task { @MainActor in self?.phase = .loading("Cached \(done) / \(all) photos") }
            }
            phase = .loading("Loading SigLIP 2 and embedding 37 breed names…")
            sorter = try await ImageSorter.load()
            _ = try await sorter?.sort(items[0])
            reset()
            print("ready at \(Date().timeIntervalSince1970)")
            // IMAGE_SORT_WAIT=1 keeps the chart empty until Start (Space). IMAGE_SORT_AUTOPLAY=N sorts N photos in
            // Show mode and then switches to Turbo; IMAGE_SORT_AUTOSTART=show|turbo starts in that mode.
            if environment["IMAGE_SORT_WAIT"] == "1" {
                return
            } else if environment["IMAGE_SORT_AUTOPLAY"].flatMap(Int.init) != nil {
                try? await Task.sleep(for: .seconds(1.5))
                mode = .show
                toggleRun()
            } else if let start = environment["IMAGE_SORT_AUTOSTART"], let mode = Mode(rawValue: start.capitalized) {
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
        chart = PhotoChart(rows: breeds.count) { [breeds] row in
            Self.cats.contains(breeds[row])
                ? CGColor(red: 1, green: 0.6, blue: 0.2, alpha: 0.10)
                : CGColor(red: 0.3, green: 0.55, blue: 1, alpha: 0.10)
        }
        chartImage = chart?.snapshot()
        queue = items
        remaining = items.count
        landed = []
        counts = [:]
        current = nil
        currentImage = nil
        lastSlot = nil
        sorted = 0
        correct = 0
        elapsed = 0
        elapsedBeforePause = 0
        modelMilliseconds = []
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
                await runShow(sorter)
            }
        }
        if !Task.isCancelled, queue.isEmpty {
            tick()
            phase = .finished
            print(
                "finished \(sorted) photos in \(String(format: "%.2f", elapsed)) s "
                    + "(\(String(format: "%.0f", photosPerSecond)) photos/s), "
                    + "correct \(String(format: "%.1f", (accuracy ?? 0) * 100))% (\(mode.rawValue))")
        }
    }

    private func runShow(_ sorter: ImageSorter) async {
        for await placed in Self.stream(sorter, items: queue, inFlight: 2) {
            if Task.isCancelled || mode != .show { break }
            land([placed], highlight: true)
            shownInShow += 1
            if let count = environment["IMAGE_SORT_AUTOPLAY"].flatMap(Int.init), shownInShow >= count {
                mode = .turbo
            }
            let started = Date()
            try? await Task.sleep(for: .seconds(1 / pace))
            if Date().timeIntervalSince(started) < 1 / pace { break }
        }
    }

    private func runTurbo(_ sorter: ImageSorter) async {
        var pending: [Placed] = []
        var lastFlush = Date()
        for await placed in Self.stream(sorter, items: queue, inFlight: Self.turboInFlight) {
            pending.append(placed)
            if Date().timeIntervalSince(lastFlush) >= Self.turboFlush {
                land(pending, highlight: false)
                pending.removeAll(keepingCapacity: true)
                lastFlush = Date()
            }
            if Task.isCancelled || mode != .turbo { break }
        }
        land(pending, highlight: false)
    }

    /// Sorts `items` with `inFlight` calls always running off the main thread and streams each result.
    private nonisolated static func stream(
        _ sorter: ImageSorter, items: [PetItem], inFlight: Int
    )
        -> AsyncStream<Placed>
    {
        AsyncStream { continuation in
            let producer = Task.detached {
                await withTaskGroup(of: Placed?.self) { group in
                    var next = 0
                    func launch() {
                        guard next < items.count, !Task.isCancelled else { return }
                        let item = items[next]
                        next += 1
                        group.addTask {
                            guard let result = try? await sorter.sort(item) else { return nil }
                            return Placed(item: item, result: result, tile: thumbnail(item.file, size: 48))
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

    nonisolated static func thumbnail(_ file: URL, size: Int) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(file as CFURL, nil) else { return nil }
        return CGImageSourceCreateThumbnailAtIndex(
            source, 0,
            [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: size,
                kCGImageSourceCreateThumbnailWithTransform: true,
            ] as CFDictionary)
    }

    private func land(_ incoming: [Placed], highlight: Bool) {
        let batch = incoming.filter { self.landed.insert($0.id).inserted }
        guard let last = batch.last, let chart else { return }
        var updated = counts
        var slot: CGRect?
        for placed in batch {
            let index = updated[placed.result.breed, default: 0]
            let row = breeds.firstIndex(of: placed.result.breed) ?? 0
            chart.draw(placed.tile, row: row, index: index, wrong: !placed.matchesGold)
            slot = PhotoChart.slot(row: row, index: index)
            updated[placed.result.breed] = index + 1
            correct += placed.matchesGold ? 1 : 0
            modelMilliseconds.append(placed.result.milliseconds)
        }
        counts = updated
        chartImage = chart.snapshot()
        lastSlot = highlight ? slot : nil
        current = last
        currentImage = Self.thumbnail(last.item.file, size: 480)
        let ids = Set(batch.map(\.id))
        queue.removeAll { ids.contains($0.id) }
        remaining = queue.count
        sorted += batch.count
        tick()
        if logDecisions { log(batch) }
    }

    private func log(_ batch: [Placed]) {
        let (cyan, yellow, red, green, dim, reset) =
            ("\u{1B}[1;36m", "\u{1B}[33m", "\u{1B}[31m", "\u{1B}[32m", "\u{1B}[2m", "\u{1B}[0m")
        var lines = ""
        for (offset, placed) in batch.enumerated() {
            let number = sorted - batch.count + offset + 1
            let mark = placed.matchesGold ? "\(green)✓\(reset)" : "\(red)✗ label: \(placed.item.breed)\(reset)"
            lines += "\(cyan)▶ #\(number) photo \(placed.item.id).jpg\(reset)\n"
            lines +=
                "  \(yellow)→ \(placed.result.breed)\(reset) · \(Int(placed.result.share * 100))% of 37 · "
                + "\(red)model call \(String(format: "%.1f", placed.result.milliseconds)) ms\(reset)  \(mark)\n"
        }
        lines += "\(dim)  sorted \(sorted)/\(total) · \(String(format: "%.1f", elapsed)) s\(reset)\n"
        print(lines, terminator: "")
    }

    private func tick() {
        if let runStart { elapsed = elapsedBeforePause + Date().timeIntervalSince(runStart) }
    }
}
