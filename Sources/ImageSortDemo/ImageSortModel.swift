import CoreGraphics
import Foundation
import ImageIO
import ImageSort
import SwiftUI

/// Drives the stream: loads Pets photos, sorts them with SigLIP 2, and keeps the live statistics.
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
        /// One photo at a time, animated into its bucket.
        case show = "Show"
        /// Several calls in flight, as fast as the model goes.
        case turbo = "Turbo"
        var id: String { rawValue }
    }

    struct Placed: Identifiable {
        let item: PetItem
        let result: ImageSorter.Result
        let thumbnail: CGImage?
        var id: Int { item.id }
        var matchesGold: Bool { result.breed == item.breed }
    }

    struct Flight: Identifiable {
        let placed: Placed
        var arrived = false
        var id: Int { placed.id }
    }

    @Published private(set) var phase: Phase = .loading("Starting…")
    @Published var mode: Mode = .show
    /// Photos per second in Show mode.
    @Published var pace: Double = 10
    @Published private(set) var buckets: [String: [Placed]] = [:]
    @Published private(set) var queue: [PetItem] = [] {
        didSet { refreshIncoming() }
    }
    /// Thumbnail of the next photo, decoded once rather than on every redraw.
    @Published private(set) var incomingImage: CGImage?
    private var incomingID: Int?
    @Published private(set) var flights: [Flight] = []
    @Published private(set) var sorted = 0
    @Published private(set) var correct = 0
    @Published private(set) var elapsed: Double = 0

    let breeds = PetsSample.breeds
    let total: Int
    static let turboInFlight = 4
    static let turboFlush = 0.05
    static let turboFlightsPerFlush = 2
    static let travel = 0.45
    static let cats: Set<String> = [
        "abyssinian", "bengal", "birman", "bombay", "british shorthair", "egyptian mau", "maine coon", "persian",
        "ragdoll", "russian blue", "siamese", "sphynx",
    ]

    private let environment = ProcessInfo.processInfo.environment
    /// IMAGE_SORT_LOG=1 prints every decision to stdout (for a terminal next to the window).
    private let logDecisions = ProcessInfo.processInfo.environment["IMAGE_SORT_LOG"] == "1"
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
    var modelName: String { sorter?.modelName ?? "SigLIP 2" }

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
            // IMAGE_SORT_AUTOSTART=show|turbo starts without a click; IMAGE_SORT_AUTOPLAY=N flies N photos in Show
            // mode and then switches to Turbo (for recordings).
            if environment["IMAGE_SORT_AUTOPLAY"].flatMap(Int.init) != nil {
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
        buckets = Dictionary(uniqueKeysWithValues: breeds.map { ($0, []) })
        queue = items
        flights = []
        landed = []
        counts = [:]
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
        let stream = Self.stream(sorter, items: queue, inFlight: 2)
        for await placed in stream {
            if Task.isCancelled || mode != .show { break }
            launch(placed)
            let started = Date()
            try? await Task.sleep(for: .seconds(1 / pace))
            if Date().timeIntervalSince(started) < 1 / pace { break }
        }
        while !Task.isCancelled, !flights.isEmpty { try? await Task.sleep(for: .milliseconds(20)) }
    }

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
            if let count = environment["IMAGE_SORT_AUTOPLAY"].flatMap(Int.init), shownInShow >= count { mode = .turbo }
        }
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
                            return Placed(item: item, result: result, thumbnail: thumbnail(item.file))
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

    nonisolated static func thumbnail(_ file: URL, size: Int = 220) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(file as CFURL, nil) else { return nil }
        return CGImageSourceCreateThumbnailAtIndex(
            source, 0,
            [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: size,
                kCGImageSourceCreateThumbnailWithTransform: true,
            ] as CFDictionary)
    }

    private func runTurbo(_ sorter: ImageSorter) async {
        let stream = Self.stream(sorter, items: queue, inFlight: Self.turboInFlight)
        var pending: [Placed] = []
        var lastFlush = Date()
        for await placed in stream {
            pending.append(placed)
            if Date().timeIntervalSince(lastFlush) >= Self.turboFlush {
                for placed in pending.suffix(Self.turboFlightsPerFlush) where flights.count < 30 {
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
            updated[placed.result.breed, default: []].insert(placed, at: 0)
            if updated[placed.result.breed]!.count > 6 { updated[placed.result.breed]!.removeLast() }
            correct += placed.matchesGold ? 1 : 0
            modelMilliseconds.append(placed.result.milliseconds)
            counts[placed.result.breed, default: 0] += 1
        }
        buckets = updated
        let ids = Set(batch.map(\.id))
        queue.removeAll { ids.contains($0.id) }
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

    /// Photos per bucket (the bucket itself only keeps the latest few for display).
    @Published private(set) var counts: [String: Int] = [:]

    private func refreshIncoming() {
        let next = queue.first { item in !flights.contains { $0.id == item.id } }
        guard next?.id != incomingID else { return }
        incomingID = next?.id
        incomingImage = next.flatMap { Self.thumbnail($0.file, size: 400) }
    }

    private func tick() {
        if let runStart { elapsed = elapsedBeforePause + Date().timeIntervalSince(runStart) }
    }
}
