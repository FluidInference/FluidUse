import Foundation
import Moderation
import SwiftUI

@available(macOS 15.0, *)
@MainActor
final class ModerationModel: ObservableObject {
    @Published var mode: ModerationEngine.Mode = .both
    @Published private(set) var isLoading = false
    @Published private(set) var isRunning = false
    @Published private(set) var loadedMode: ModerationEngine.Mode?
    @Published private(set) var feed: [ModeratedComment] = []
    @Published private(set) var flaggedFeed: [ModeratedComment] = []
    @Published private(set) var checked = 0
    @Published private(set) var flagged = 0
    @Published private(set) var agreeing = 0
    @Published private(set) var aneCount = 0
    @Published private(set) var gpuCount = 0
    @Published private(set) var startedAt: Date?
    @Published private(set) var finishedSeconds: Double?
    @Published private(set) var lastMs = 0.0
    @Published var errorMessage: String?

    let sample: CommentSample?
    var total: Int { sample?.rows.count ?? 0 }
    private var engine: ModerationEngine?
    private var runTask: Task<Void, Never>?
    private var autostart = false
    private static let feedLength = 5000
    private static let flaggedLength = 5000

    init() {
        do {
            sample = try CommentSample.bundled()
        } catch {
            sample = nil
            errorMessage = error.localizedDescription
        }
    }

    var seconds: Double {
        finishedSeconds ?? startedAt.map { -$0.timeIntervalSinceNow } ?? 0
    }

    func rate(at date: Date) -> Double {
        let elapsed = finishedSeconds ?? startedAt.map { date.timeIntervalSince($0) } ?? 0
        return elapsed > 0.05 ? Double(checked) / elapsed : 0
    }

    /// `MODERATION_MODE` (ane, gpu, both) preselects; the model loads and runs unless `MODERATION_AUTOSTART=0`.
    func applyLaunchEnvironment() {
        let environment = ProcessInfo.processInfo.environment
        if let mode = environment["MODERATION_MODE"].flatMap(ModerationEngine.Mode.init(rawValue:)) {
            self.mode = mode
        }
        autostart = environment["MODERATION_AUTOSTART"] != "0"
        load()
    }

    func load() {
        guard !isLoading, loadedMode != mode else { return }
        reset()
        engine = nil
        loadedMode = nil
        isLoading = true
        let mode = mode
        Task {
            defer { isLoading = false }
            do {
                engine = try await Task.detached {
                    try await ModerationEngine.load(from: Launch.resolveModel(), mode: mode)
                }.value
                loadedMode = mode
                if autostart {
                    autostart = false
                    start()
                }
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    func reset() {
        runTask?.cancel()
        runTask = nil
        isRunning = false
        feed = []
        flaggedFeed = []
        checked = 0
        flagged = 0
        agreeing = 0
        aneCount = 0
        gpuCount = 0
        startedAt = nil
        finishedSeconds = nil
        lastMs = 0
    }

    func start() {
        guard let engine, let sample, !isRunning else { return }
        reset()
        isRunning = true
        Self.clearLog()
        Self.log("\u{1B}[1;36m■ d1-omni-600M · \(mode.rawValue) · \(sample.rows.count) comments\u{1B}[0m\n")
        let rows = sample.rows
        let threshold = sample.threshold
        runTask = Task {
            do {
                // Model calls and stream draining stay off the main actor; the UI takes a batch every 50 ms.
                let buffer = ResultBuffer()
                // The clock includes tokenization, as a live feed would pay it per comment.
                startedAt = Date()
                let producer = Task.detached {
                    let stream = try await engine.run(rows, threshold: threshold)
                    for try await batch in stream { await buffer.append(batch) }
                    await buffer.finish()
                }
                while true {
                    try await Task.sleep(for: .milliseconds(50))
                    let (batch, done) = await buffer.drain()
                    apply(batch)
                    if done { break }
                }
                try await producer.value
                finishedSeconds = startedAt.map { -$0.timeIntervalSinceNow }
                Self.log(
                    String(
                        format:
                            "\u{1B}[1;32m■ done · %d comments in %.1f s · %.0f/s · flagged %d · %.1f%% agree with humans\u{1B}[0m\n",
                        checked, finishedSeconds ?? 0, rate(at: Date()), flagged,
                        100 * Double(agreeing) / Double(max(checked, 1))))
            } catch is CancellationError {
            } catch {
                errorMessage = error.localizedDescription
            }
            isRunning = false
        }
    }

    private func apply(_ batch: [ModeratedComment]) {
        guard !batch.isEmpty, let sample else { return }
        for result in batch {
            checked += 1
            if result.flagged { flagged += 1 }
            if result.flagged == sample.rows[result.id].isToxic { agreeing += 1 }
            if result.engine == .neuralEngine { aneCount += 1 } else { gpuCount += 1 }
        }
        lastMs = batch.last?.milliseconds ?? lastMs
        Self.log(batch.map(Self.logLine).joined())
        // Toxic comments never enter the live feed; they fly into the toxic box instead.
        feed = Array((batch.filter { !$0.flagged }.reversed() + feed).prefix(Self.feedLength))
        let newFlags = batch.filter(\.flagged)
        if !newFlags.isEmpty {
            withAnimation(.spring(duration: 0.55, bounce: 0.15)) {
                flaggedFeed = Array((newFlags.reversed() + flaggedFeed).prefix(Self.flaggedLength))
            }
        }
    }
}

@available(macOS 15.0, *)
extension ModerationModel {
    /// Per-comment lines, ANSI-colored, appended to `MODERATION_DEMO_LOG` (default /tmp/moderation-demo.log) for a
    /// `tail -f` pane next to macmon.
    static let logURL = URL(
        fileURLWithPath: ProcessInfo.processInfo.environment["MODERATION_DEMO_LOG"] ?? "/tmp/moderation-demo.log")

    static func clearLog() {
        try? Data("\u{1B}[2J\u{1B}[3J\u{1B}[H".utf8).write(to: logURL)
    }

    static func logLine(_ row: ModeratedComment) -> String {
        let text = row.text.replacingOccurrences(of: "\n", with: " ").prefix(80)
        let head = String(
            format: "%@ %3.0f%% %@ %4.1fms ", row.flagged ? "✗ TOXIC" : "✓ ok   ",
            (row.flagged ? row.probability : 1 - row.probability) * 100, row.engine.rawValue, row.milliseconds)
        return row.flagged
            ? "\u{1B}[1;31m\(head)\(text)\u{1B}[0m\n" : "\u{1B}[2m\(head)\u{1B}[0m\(text)\n"
    }

    static func log(_ text: String) {
        guard !text.isEmpty else { return }
        if let handle = try? FileHandle(forWritingTo: logURL) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(text.utf8))
        } else {
            try? text.write(to: logURL, atomically: true, encoding: .utf8)
        }
    }
}

/// Collects streamed results between UI refreshes.
actor ResultBuffer {
    private var pending: [ModeratedComment] = []
    private var finished = false

    func append(_ batch: [ModeratedComment]) { pending += batch }
    func finish() { finished = true }

    func drain() -> ([ModeratedComment], Bool) {
        defer { pending.removeAll(keepingCapacity: true) }
        return (pending, finished)
    }
}
