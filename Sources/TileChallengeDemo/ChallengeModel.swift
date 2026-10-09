import CoreGraphics
import FluidUse
import Foundation
import SwiftUI

/// Solves "select all images with …" photo grids with EmbeddingGemma 2: each tile is embedded (vision on the GPU,
/// text model on the Neural Engine) and ticked when the asked-for category beats all the others. Photos and labels are
/// 21 everyday categories from Caltech-256; the grids are built here, not taken from any real CAPTCHA.
@MainActor
final class ChallengeModel: ObservableObject {
    enum Phase: Equatable {
        case loading(String)
        case ready
        case running(String)
        case done
        case failed(String)
    }

    struct Tile: Identifiable {
        let id: Int
        let image: CGImage
        let category: Int
        var picked: Bool? = nil
        var score: Float = 0
        var isTarget: Bool
    }

    struct Grid {
        let prompt: String
        var tiles: [Tile]
        var verified = false
        var solved: Bool { tiles.allSatisfy { $0.picked == $0.isTarget } }
    }

    @Published private(set) var phase: Phase = .loading("Starting…")
    @Published private(set) var grid: Grid?
    @Published private(set) var history: [Bool] = []
    @Published private(set) var tilesDone = 0
    @Published private(set) var tilesCorrect = 0
    @Published private(set) var gridsSolved = 0
    @Published private(set) var busySeconds: Double = 0
    @Published private(set) var lastGridMilliseconds: Double = 0
    @Published private(set) var summary = ""

    /// Categories, and the lookalike groups the hard round draws its distractors from.
    static let categories = [
        "traffic light", "fire hydrant", "bus", "bicycle", "motorcycle", "fire truck", "airplane", "helicopter", "boat",
        "bridge", "lighthouse", "palm tree", "ladder", "umbrella", "horse", "zebra", "giraffe", "elephant", "camel",
        "sunflower", "cactus",
    ]
    static let lookalikes: [[String]] = [
        ["bus", "bicycle", "motorcycle", "fire truck"], ["airplane", "helicopter", "boat"],
        ["horse", "zebra", "giraffe", "elephant", "camel"], ["palm tree", "cactus", "sunflower"],
        ["bridge", "lighthouse", "ladder", "traffic light", "fire hydrant", "umbrella"],
    ]
    static let turboGrids = max(
        1, CommandLine.arguments.first { $0.hasPrefix("--grids=") }.flatMap { Int($0.dropFirst(8)) } ?? 1000)

    private var text: EmbeddingGemma2Manager?
    private var vision: EmbeddingGemma2Vision?
    private var photos: [(file: URL, category: Int)] = []
    private var labelVectors: [[Float]] = []  // one per category
    private var runTask: Task<Void, Never>?
    private var generator = SeededGenerator(seed: 7)

    var tilesPerSecond: Double { busySeconds > 0 ? Double(tilesDone) / busySeconds : 0 }
    var accuracy: Double { tilesDone > 0 ? Double(tilesCorrect) / Double(tilesDone) : 0 }
    var gridsDone: Int { history.count }

    func prepare() async {
        do {
            struct Item: Decodable {
                let file: String
                let label: String
            }
            let directory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("FluidUse/Datasets/caltech256-select")
            let items = try JSONDecoder().decode(
                [Item].self, from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
            photos = items.compactMap { item in
                Self.categories.firstIndex(of: item.label).map { (directory.appendingPathComponent(item.file), $0) }
            }
            guard photos.count > 100 else {
                throw EmbeddingGemma2Error.invalidAsset("No photos in \(directory.path)")
            }
            phase = .loading("Loading EmbeddingGemma 2…")
            let loadStart = DispatchTime.now().uptimeNanoseconds
            let text = try await EmbeddingGemma2Manager.loadDefault()
            let vision = try await EmbeddingGemma2Vision.load(text: text)
            for category in Self.categories { labelVectors.append(try await text.embed("a photo of a \(category).")) }
            _ = try await vision.embed(image: try EmbeddingGemma2Vision.image(contentsOf: photos[0].file))
            self.text = text
            self.vision = vision
            DemoLog.model(
                String(
                    format: "models loaded in %.1f s · vision on the GPU, text on the Neural Engine · %d photos",
                    Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1e9, photos.count))
            phase = .ready
            if !CommandLine.arguments.contains("--manual") { start() }
        } catch {
            phase = .failed(error.localizedDescription)
            DemoLog.line("failed: \(error.localizedDescription)", color: 196, bold: true)
        }
    }

    func start() {
        guard phase == .ready || phase == .done else { return }
        reset()
        runTask = Task { await run() }
    }

    func stop() {
        runTask?.cancel()
        runTask = nil
        if case .running = phase { phase = .done }
    }

    func reset() {
        runTask?.cancel()
        runTask = nil
        grid = nil
        history = []
        tilesDone = 0
        tilesCorrect = 0
        gridsSolved = 0
        busySeconds = 0
        summary = ""
        generator = SeededGenerator(seed: 7)
        if vision != nil { phase = .ready }
    }

    private func run() async {
        do {
            DemoLog.event("▶ round 1: everyday things")
            phase = .running("Round 1 · everyday things")
            for _ in 0..<3 { try await showGrid(makeGrid(hard: false)) }
            DemoLog.event("▶ round 2: lookalikes")
            phase = .running("Round 2 · lookalikes (bus vs fire truck, horse vs zebra…)")
            for _ in 0..<3 { try await showGrid(makeGrid(hard: true)) }
            DemoLog.event("▶ turbo: \(Self.turboGrids) grids, no pauses")
            phase = .running("Turbo · \(Self.turboGrids) grids, no pauses")
            try await turbo()
            summary = String(
                format: "%d grids, %d tiles in %.1f s · %.0f tiles/s · %.1f%% of tiles right · %d/%d grids perfect",
                gridsDone, tilesDone, busySeconds, tilesPerSecond, 100 * accuracy, gridsSolved, gridsDone)
            DemoLog.event("■ " + summary)
            phase = .done
        } catch is CancellationError {
        } catch {
            phase = .failed(error.localizedDescription)
            DemoLog.line("failed: \(error.localizedDescription)", color: 196, bold: true)
        }
    }

    typealias Spec = (prompt: String, target: Int, picks: [(URL, Int)])

    /// A 3x3 grid with 2–5 photos of the target; distractors from any category, or (hard) from its lookalikes.
    private func makeGrid(hard: Bool) -> Spec {
        let positives = Int.random(in: 2...5, using: &generator)
        let group = Self.lookalikes.randomElement(using: &generator)!
        let pool = hard ? group.compactMap { Self.categories.firstIndex(of: $0) } : Array(Self.categories.indices)
        let target = pool.randomElement(using: &generator)!
        let yes = photos.filter { $0.category == target }.shuffled(using: &generator).prefix(positives)
        let no = photos.filter { $0.category != target && pool.contains($0.category) }.shuffled(using: &generator)
            .prefix(9 - yes.count)
        let picks = (Array(yes) + Array(no)).shuffled(using: &generator).map { ($0.file, $0.category) }
        let name = Self.categories[target]
        return (("aeiou".contains(name.first!) ? "an " : "a ") + name, target, picks)
    }

    /// Zero-shot decision: the asked-for category must beat every other category.
    private func decide(_ vector: [Float], target: Int) -> (picked: Bool, score: Float) {
        let scores = labelVectors.map { zip($0, vector).reduce(Float(0)) { $0 + $1.0 * $1.1 } }
        let best = scores.indices.max { scores[$0] < scores[$1] } ?? 0
        return (best == target, scores[target])
    }

    private nonisolated static func decode(_ files: [URL]) throws -> [CGImage] {
        try files.map { try EmbeddingGemma2Vision.image(contentsOf: $0) }
    }

    /// A grid at a watchable pace: tiles appear, get ticked as each embedding lands, then Verify.
    private func showGrid(_ spec: Spec) async throws {
        guard let vision else { return }
        let images = try await Task.detached { try Self.decode(spec.picks.map(\.0)) }.value
        grid = Grid(
            prompt: spec.prompt,
            tiles: images.indices.map {
                Tile(id: $0, image: images[$0], category: spec.picks[$0].1, isTarget: spec.picks[$0].1 == spec.target)
            })
        try await Task.sleep(for: .milliseconds(450))
        let begin = DispatchTime.now().uptimeNanoseconds
        let vectors = try await vision.embed(images: images, maxInFlight: 9)
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - begin) / 1e9
        for index in vectors.indices {
            let decision = decide(vectors[index], target: spec.target)
            withAnimation(.easeOut(duration: 0.12)) {
                grid?.tiles[index].picked = decision.picked
                grid?.tiles[index].score = decision.score
            }
            try await Task.sleep(for: .milliseconds(70))
        }
        record(elapsed: elapsed)
        try await Task.sleep(for: .milliseconds(350))
        withAnimation(.easeOut(duration: 0.15)) { grid?.verified = true }
        log()
        try await Task.sleep(for: .milliseconds(700))
    }

    /// As fast as the chips allow: the next batch of grids is decoded and embedded while this one is shown.
    private func turbo() async throws {
        guard let vision else { return }
        let batch = 4
        var remaining = Self.turboGrids
        func prepareBatch() -> [Spec] {
            let count = min(batch, remaining)
            remaining -= count
            return (0..<count).map { _ in makeGrid(hard: Bool.random(using: &generator)) }
        }
        func compute(
            _ specs: [Spec]
        ) async throws
            -> (images: [CGImage], vectors: [[Float]], seconds: Double)
        {
            let files = specs.flatMap { $0.picks.map(\.0) }
            let begin = DispatchTime.now().uptimeNanoseconds
            let images = try await Task.detached { try Self.decode(files) }.value
            let vectors = try await vision.embed(images: images, maxInFlight: 12)
            return (images, vectors, Double(DispatchTime.now().uptimeNanoseconds - begin) / 1e9)
        }
        var specs = prepareBatch()
        var pending = Task { try await compute(specs) }
        while !specs.isEmpty {
            try Task.checkCancellation()
            let (images, vectors, seconds) = try await pending.value
            let current = specs
            specs = prepareBatch()
            let nextSpecs = specs
            if !nextSpecs.isEmpty { pending = Task { try await compute(nextSpecs) } }
            for (gridIndex, spec) in current.enumerated() {
                var tiles: [Tile] = []
                for tileIndex in 0..<9 {
                    let flat = gridIndex * 9 + tileIndex
                    let decision = decide(vectors[flat], target: spec.target)
                    tiles.append(
                        Tile(
                            id: tileIndex, image: images[flat], category: spec.picks[tileIndex].1,
                            picked: decision.picked,
                            score: decision.score, isTarget: spec.picks[tileIndex].1 == spec.target))
                }
                grid = Grid(prompt: spec.prompt, tiles: tiles, verified: true)
                record(elapsed: seconds / Double(current.count))
                if gridsDone % 25 == 0 { log() }
                try await Task.sleep(for: .milliseconds(16))  // one frame per grid, so every grid is drawn
            }
        }
    }

    private func record(elapsed: Double) {
        guard let grid else { return }
        busySeconds += elapsed
        lastGridMilliseconds = elapsed * 1000
        tilesDone += grid.tiles.count
        tilesCorrect += grid.tiles.filter { $0.picked == $0.isTarget }.count
        if grid.solved { gridsSolved += 1 }
        history.append(grid.solved)
    }

    private func log() {
        guard let grid else { return }
        DemoLog.line(
            String(
                format: "%@ grid %d “%@” %@ · %.0f ms · %.0f tiles/s · %.1f%% tiles right", grid.solved ? "✓" : "✗",
                gridsDone, grid.prompt, grid.solved ? "solved" : "missed", lastGridMilliseconds, tilesPerSecond,
                100 * accuracy), color: grid.solved ? 120 : 203)
    }
}

/// Deterministic shuffles, so every run sees the same grids.
struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }
}
