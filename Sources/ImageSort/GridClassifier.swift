import CoreGraphics
import FluidUse
import Foundation

/// Labels every cell of a `columns × rows` grid over a frame with SigLIP 2 on Core ML.
/// Cells are scored concurrently; one call handles one cell.
public final class GridClassifier: Sendable {
    public struct Cell: Sendable {
        public let label: Int
        /// Softmax share of the chosen label among all labels.
        public let share: Float
        /// The five most likely labels with their shares, best first.
        public let top: [Ranked]
    }

    public struct Ranked: Sendable {
        public let label: Int
        public let share: Float
    }

    public struct Grid: Sendable {
        public let cells: [Cell]
        public let milliseconds: Double
    }

    public let labels: [String]
    public let columns: Int
    public let rows: Int
    private let manager: SigLIP2Manager
    private let embeddings: [[Float]]

    private init(manager: SigLIP2Manager, labels: [String], embeddings: [[Float]], columns: Int, rows: Int) {
        self.manager = manager
        self.labels = labels
        self.embeddings = embeddings
        self.columns = columns
        self.rows = rows
    }

    /// Loads the encoders from `SIGLIP2_MODEL_DIR` and embeds `template` (with `{}` replaced) once per label.
    public static func load(
        labels: [String], template: String = "a photo of {}.", columns: Int, rows: Int
    ) async throws -> GridClassifier {
        guard let path = ProcessInfo.processInfo.environment["SIGLIP2_MODEL_DIR"], !path.isEmpty else {
            throw SigLIP2Error.invalidAsset("Set SIGLIP2_MODEL_DIR to the converted siglip2-base-patch16-256 folder")
        }
        let manager = try await SigLIP2Manager.load(from: URL(fileURLWithPath: path))
        let embeddings = try await manager.embed(
            labels: labels.map { template.replacingOccurrences(of: "{}", with: $0) })
        return GridClassifier(manager: manager, labels: labels, embeddings: embeddings, columns: columns, rows: rows)
    }

    /// Classifies every cell of `frame`, keeping `inFlight` calls running.
    public func classify(_ frame: CGImage, inFlight: Int = 6) async throws -> Grid {
        let start = DispatchTime.now().uptimeNanoseconds
        let width = frame.width / columns
        let height = frame.height / rows
        let crops = (0..<(columns * rows)).map { index in
            frame.cropping(
                to: CGRect(x: (index % columns) * width, y: (index / columns) * height, width: width, height: height))
        }
        var cells = [Cell?](repeating: nil, count: crops.count)
        try await withThrowingTaskGroup(of: (Int, Cell).self) { group in
            var next = 0
            func launch() {
                guard next < crops.count else { return }
                let index = next
                next += 1
                group.addTask { [self] in (index, try await cell(crops[index])) }
            }
            for _ in 0..<inFlight { launch() }
            while let (index, cell) = try await group.next() {
                cells[index] = cell
                launch()
            }
        }
        let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
        return Grid(cells: cells.map { $0 ?? Cell(label: 0, share: 0, top: []) }, milliseconds: milliseconds)
    }

    private func cell(_ crop: CGImage?) async throws -> Cell {
        guard let crop else { throw SigLIP2Error.invalidInput("Empty cell") }
        let answer = manager.score(
            imageEmbedding: try await manager.embed(image: crop), labels: labels, labelEmbeddings: embeddings)
        let scale = manager.config.logitScale
        let best = answer.similarities[answer.selectedIndex]
        let weights = answer.similarities.map { exp(scale * ($0 - best)) }
        let total = weights.reduce(0, +)
        let top = weights.indices.sorted { weights[$0] > weights[$1] }.prefix(5).map {
            Ranked(label: $0, share: weights[$0] / total)
        }
        return Cell(label: answer.selectedIndex, share: 1 / total, top: top)
    }
}
