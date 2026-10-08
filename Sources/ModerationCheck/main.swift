import FluidUse
import Foundation
import Moderation

/// Headless moderation run over the bundled 5,000 Civil Comments: throughput, parity with native PyTorch, and
/// agreement with the human labels.
///
///     swift run -c release ModerationCheck [ane|gpu|both] [--limit=N]    (D1_MODERATION_DIR=<dir> to skip the download)
@main
struct ModerationCheck {
    static func main() async throws {
        guard #available(macOS 15.0, *) else { fatalError("macOS 15 required") }
        let arguments = Array(CommandLine.arguments.dropFirst())
        let mode = arguments.first.flatMap(ModerationEngine.Mode.init(rawValue:)) ?? .both
        let limit = arguments.first { $0.hasPrefix("--limit=") }.flatMap { Int($0.dropFirst(8)) }
        let directory: URL
        if let path = ProcessInfo.processInfo.environment["D1_MODERATION_DIR"], !path.isEmpty {
            directory = URL(fileURLWithPath: path)
        } else {
            directory = try await D1OmniModelStore.ensure { file, bytes in
                if bytes > 0 { print("downloaded \(file) (\(bytes / 1_000_000) MB)") }
            }
        }
        let sample = try CommentSample.bundled()
        let rows = limit.map { Array(sample.rows.prefix($0)) } ?? sample.rows
        let loadStarted = Date()
        let engine = try await ModerationEngine.load(from: directory, mode: mode)
        let loadSeconds = -loadStarted.timeIntervalSinceNow

        let prepared = Date()
        let stream = try await engine.run(rows, threshold: sample.threshold)
        let started = Date()
        var results: [ModeratedComment] = []
        for try await batch in stream { results += batch }
        let seconds = -started.timeIntervalSinceNow
        let prepSeconds = started.timeIntervalSince(prepared)

        var flips = 0
        var maxDelta = 0.0
        var correct = 0
        var truePositives = 0
        var flagged = 0
        var toxic = 0
        var byEngine: [String: Int] = [:]
        for result in results {
            let row = rows[result.id]
            maxDelta = max(maxDelta, abs(Double(result.probability) - row.native))
            if result.flagged != (row.native >= sample.threshold) { flips += 1 }
            if result.flagged == row.isToxic { correct += 1 }
            if result.flagged { flagged += 1 }
            if row.isToxic { toxic += 1 }
            if result.flagged && row.isToxic { truePositives += 1 }
            byEngine[result.engine.rawValue, default: 0] += 1
        }
        let n = Double(max(results.count, 1))
        print(
            String(
                format:
                    "mode=%@ comments=%d load=%.1fs tokenize=%.2fs run=%.2fs → %.0f comments/s  %@\n"
                    + "vs native: flag flips %d, max|dp| %.3f\n"
                    + "vs humans: accuracy %.1f%%, toxic recall %.1f%%, precision %.1f%% (flagged %d, toxic %d)",
                mode.rawValue, results.count, loadSeconds, prepSeconds, seconds, n / seconds,
                byEngine.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: ", "), flips,
                maxDelta, 100 * Double(correct) / n, 100 * Double(truePositives) / Double(max(toxic, 1)),
                100 * Double(truePositives) / Double(max(flagged, 1)), flagged, toxic))
    }
}
