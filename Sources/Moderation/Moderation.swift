@preconcurrency import CoreML
import FluidUse
import Foundation

/// Real Civil Comments test comments (CC0) with clear human labels, plus native PyTorch P(toxic) for parity.
public struct CommentSample: Sendable, Decodable {
    public struct Row: Sendable, Decodable {
        public let text: String
        /// Share of human raters who called the comment toxic.
        public let toxicity: Double
        /// d1-omni-600M PyTorch (MPS fp16) P(toxic).
        public let native: Double
        public var isToxic: Bool { toxicity >= 0.5 }
    }

    public let source: String
    public let threshold: Double
    public let rows: [Row]

    public static func bundled() throws -> CommentSample {
        guard let url = Bundle.module.url(forResource: "civil-comments-5000", withExtension: "json") else {
            throw D1OmniError.invalidAsset("civil-comments-5000.json missing from the bundle")
        }
        return try JSONDecoder().decode(CommentSample.self, from: Data(contentsOf: url))
    }
}

public struct ModeratedComment: Sendable, Identifiable {
    public enum Engine: String, Sendable {
        case neuralEngine = "ANE"
        case gpu = "GPU"
    }
    public let id: Int
    public let text: String
    public let humanToxicity: Double
    public let probability: Float
    public let flagged: Bool
    public let engine: Engine
    public let tokens: Int
    /// Core ML call time divided by the rows in that call.
    public let milliseconds: Double
}

/// Moderation over a comment firehose: one shared queue feeds a Neural Engine worker (one comment per call) and a
/// GPU worker (eight per call), so both chips run at once.
@available(macOS 15.0, iOS 18.0, *)
public actor ModerationEngine {
    public enum Mode: String, Sendable, CaseIterable { case ane, gpu, both }

    public static let question = D1OmniQuestion.noul(
        "Is this comment toxic?",
        criteria: (
            yes:
                "toxic: a rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion",
            no: "not toxic"
        ))

    private let neuralEngine: D1OmniManager?
    private let gpu: D1OmniManager?
    public nonisolated let mode: Mode

    private init(neuralEngine: D1OmniManager?, gpu: D1OmniManager?, mode: Mode) {
        self.neuralEngine = neuralEngine
        self.gpu = gpu
        self.mode = mode
    }

    /// `directory` is a `D1OmniModelStore` snapshot: yes/no functions with one row per call run on the Neural Engine,
    /// eight per call on the GPU.
    public static func load(from directory: URL, mode: Mode) async throws -> ModerationEngine {
        async let ane =
            mode == .gpu
            ? nil : try D1OmniManager.load(from: directory, computeUnits: .cpuAndNeuralEngine, batch: 1, options: 2)
        async let gpu =
            mode == .ane ? nil : try D1OmniManager.load(from: directory, computeUnits: .cpuAndGPU, batch: 8, options: 2)
        let engine = ModerationEngine(neuralEngine: try await ane, gpu: try await gpu, mode: mode)
        try await engine.warmUp()
        return engine
    }

    private func warmUp() async throws {
        let text = String(repeating: "warm up ", count: 40)
        for manager in [neuralEngine, gpu].compactMap({ $0 }) {
            for length in manager.lengths {
                let state = String(text.prefix(max(1, length * 3 - 150)))
                _ = try? await manager.answer(
                    states: Array(repeating: state, count: manager.batch), question: Self.question)
            }
        }
    }

    /// Streams results in completion order. Short comments go to the Neural Engine first and long ones to the GPU
    /// first, where each is fastest; either worker takes any comment once its preferred queue is empty.
    public func run(
        _ rows: [CommentSample.Row], threshold: Double
    ) async throws -> AsyncThrowingStream<
        [ModeratedComment], Error
    > {
        guard let lengthSource = neuralEngine ?? gpu else { throw D1OmniError.invalidAsset("no engine") }
        var tokenCounts: [Int] = []
        tokenCounts.reserveCapacity(rows.count)
        for row in rows {
            tokenCounts.append(try await lengthSource.encode(state: row.text, question: Self.question).ids.count)
        }
        let queue = CommentQueue(tokenCounts: tokenCounts, lengths: lengthSource.lengths)
        let question = Self.question
        let neuralEngine = neuralEngine
        let gpu = gpu
        return AsyncThrowingStream { continuation in
            let task = Task {
                try await withThrowingTaskGroup(of: Void.self) { group in
                    if let neuralEngine {
                        group.addTask {
                            while let indices = await queue.take(count: 1, preferLong: false) {
                                try Task.checkCancellation()
                                let answers = try await neuralEngine.answer(
                                    states: indices.map { rows[$0].text }, question: question)
                                continuation.yield(
                                    Self.results(indices, answers, rows, threshold, .neuralEngine))
                            }
                        }
                    }
                    if let gpu {
                        group.addTask {
                            while let indices = await queue.take(count: gpu.batch, preferLong: true) {
                                try Task.checkCancellation()
                                let answers = try await gpu.answer(
                                    states: indices.map { rows[$0].text }, question: question)
                                continuation.yield(Self.results(indices, answers, rows, threshold, .gpu))
                            }
                        }
                    }
                    try await group.waitForAll()
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private static func results(
        _ indices: [Int], _ answers: [D1OmniAnswer], _ rows: [CommentSample.Row], _ threshold: Double,
        _ engine: ModeratedComment.Engine
    ) -> [ModeratedComment] {
        zip(indices, answers).map { index, answer in
            let probability = answer.probabilities[0]
            return ModeratedComment(
                id: index, text: rows[index].text, humanToxicity: rows[index].toxicity, probability: probability,
                flagged: Double(probability) >= threshold, engine: engine, tokens: answer.tokenCount,
                milliseconds: answer.predictionMilliseconds / Double(indices.count))
        }
    }
}

/// Pending comment indices grouped by the smallest bucket that fits them.
actor CommentQueue {
    private var pending: [[Int]]

    init(tokenCounts: [Int], lengths: [Int]) {
        var pending = Array(repeating: [Int](), count: lengths.count)
        for (index, count) in tokenCounts.enumerated() {
            guard let bucket = lengths.firstIndex(where: { $0 >= count }) else { continue }
            pending[bucket].append(index)
        }
        // Reverse so popLast keeps stream order within a bucket.
        self.pending = pending.map { Array($0.reversed()) }
    }

    /// Up to `count` indices from one bucket: the longest non-empty bucket if `preferLong`, else the shortest.
    func take(count: Int, preferLong: Bool) -> [Int]? {
        let order = preferLong ? Array(pending.indices.reversed()) : Array(pending.indices)
        guard let bucket = order.first(where: { !pending[$0].isEmpty }) else { return nil }
        var taken: [Int] = []
        while taken.count < count, let index = pending[bucket].popLast() { taken.append(index) }
        return taken
    }
}
