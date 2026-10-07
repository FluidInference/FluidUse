import FluidUse
import Foundation
import SwiftUI

/// The five questions (triage_core.questions), answered together in one model call.
enum TriageQuestions {
    static let all: [(id: String, question: Decision2Question)] = [
        (
            "type",
            .choice(
                "What kind of issue is this?",
                [
                    "bug": "Something isn't working: errors, crashes, wrong behavior, failing tests",
                    "enhancement": "New feature or improvement request",
                    "documentation": "Docs, README, guides, examples or website content",
                    "question": "A usage question or request for help, not a defect",
                    "proposal": "Architecture, design, roadmap or research proposal",
                ])
        ),
        (
            "wg",
            .choice(
                "Which workgroup should own this issue?",
                [
                    "wg/mom-routing":
                        "Routing decisions, signals, model selection rules, Mixture-of-Models routing logic",
                    "wg/router-models-inference-runtime":
                        "Classifier/embedding models, model training, inference runtime (Candle, ONNX, Rust bindings)",
                    "wg/data-plane-networking":
                        "Envoy, ext_proc, gateway, proxy, request/response handling, networking",
                    "wg/enterprise-environment":
                        "Kubernetes, Helm, operators, auth, deployment environments, ROCm/GPU platforms",
                    "wg/evaluation-quality":
                        "Benchmarks, evaluations, accuracy, testing quality, CI performance checks",
                    "wg/developer-experience-ecosystem":
                        "Docs, CLI, dashboard UI, examples, install, local dev experience",
                    "wg/agentic-context": "Agents, tools, MCP, memory, context management, agent workflows",
                    "wg/platform-operations":
                        "Observability, metrics, logging, tracing, operations and reliability",
                    "owner/maintainers": "Repository governance, community, releases, process",
                ])
        ),
        (
            "priority",
            .score(
                "How urgent is this issue?",
                [
                    "P2: nice-to-have or exploratory", "P1: important, should be done",
                    "P0: critical, must be fixed now",
                ])
        ),
        (
            "needs_info",
            .yesNo(
                "Is important information missing (e.g. reproduction steps, versions, logs) so the maintainers must ask the author for more details?"
            )
        ),
        ("good_first", .yesNo("Is this a small, well-scoped task suitable for a first-time contributor?")),
    ]
}

/// One issue's labels and the probabilities behind them (server.triage).
struct TriageResult: Sendable {
    // The priority score (expected level, 0-2) ranks issues well but its argmax is P0 for most issues; cut at this
    // repo's p90 / p50 so P0 is the top ~10 % and P1 the next ~40 %.
    static let p0At = 1.452
    static let p1At = 1.251

    let ms: Double
    let labels: [String]
    let type: [(String, Double)]
    let wg: [(String, Double)]
    let priority: [(String, Double)]
    let priorityScore: Double
    let needsInfo: Double
    let goodFirst: Double

    init(_ result: Decision2Result, ms: Double) {
        func probs(_ id: String) -> [(String, Double)] {
            guard let a = result[id] else { return [] }
            return Array(zip(a.keys, a.probabilities))
        }
        let level = result["priority"]?.score ?? 0
        self.ms = ms
        type = probs("type")
        wg = probs("wg")
        priority = probs("priority")
        priorityScore = level
        needsInfo = result["needs_info"]?.yes ?? 0
        goodFirst = result["good_first"]?.yes ?? 0
        var labels = [
            result["type"]?.choice ?? "?", result["wg"]?.choice ?? "?",
            level >= Self.p0At ? "priority/P0" : level >= Self.p1At ? "priority/P1" : "priority/P2",
        ]
        if needsInfo >= 0.6 { labels.append("needs-info") }
        if goodFirst >= 0.6 { labels.append("good first issue") }
        self.labels = labels
    }
}

/// Per-row state, so a result redraws only its own row.
@MainActor final class RowState: ObservableObject, Identifiable {
    let issue: Issue
    @Published var result: TriageResult?
    @Published var live = false
    var labeledAt: Date?

    nonisolated var id: Int { issue.number }

    init(issue: Issue) { self.issue = issue }
}

struct WorkgroupCount: Identifiable, Equatable {
    let name: String
    var count: Int
    var id: String { name }
}

/// Counters for the stats bar and the sidebar.
@MainActor final class TriageStats: ObservableObject {
    @Published var done = 0
    @Published var decisions = 0
    @Published var avgMs: Double?
    @Published var rate: Double?
    @Published var workgroups = LabelPalette.workgroups.map { WorkgroupCount(name: $0, count: 0) }
}

/// The row the list should keep centered.
@MainActor final class ScrollState: ObservableObject {
    @Published var target: Int?
    var anchor = UnitPoint.center
}

@MainActor final class TriageModel: ObservableObject {
    enum Phase: Equatable {
        case loadingIssues, loadingModel, ready, running, finished
        case failed(String)
    }

    typealias Answer = @Sendable (OrderedJSON) async throws -> Decision2Result

    @Published var phase = Phase.loadingIssues {
        didSet { if case .failed(let message) = phase { print("error: \(message)") } }
    }
    @Published var rows: [RowState] = []
    @Published var selected: RowState?
    let stats = TriageStats()
    let scroll = ScrollState()
    private var answer: Answer?
    private var started = false

    var openCount: Int { rows.filter(\.issue.isOpen).count }

    static func issuesURL() -> URL {
        if let path = CommandLine.arguments.dropFirst().first(where: { !$0.hasPrefix("-") }) {
            return URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Caches/FluidUse/issue-triage/issues.json")
    }

    func start() async {
        guard !started else { return }
        started = true
        let url = Self.issuesURL()
        guard FileManager.default.fileExists(atPath: url.path) else {
            phase = .failed("No issues at \(url.path).\nRun Sources/IssueTriageDemo/fetch-issues.sh first.")
            return
        }
        do {
            let issues = try await Task.detached(priority: .userInitiated) { try Issue.load(from: url) }.value
            rows = issues.map(RowState.init)
            print("loaded \(issues.count) issues from \(url.path)")
            phase = .loadingModel
            guard #available(macOS 15.0, *) else {
                phase = .failed("Decision 2.0 on Core ML needs macOS 15 or later.")
                return
            }
            let loadStart = Date()
            answer = try await Task.detached(priority: .userInitiated) { () -> Answer in
                let directory = try await Decision2ModelStore.ensure(.kai)
                let manager = try await Decision2Manager.load(from: directory)
                try await manager.warm()
                return { state in try await manager.answer(state: state, questions: TriageQuestions.all) }
            }.value
            print(String(format: "Decision-2.0-Kai-0.6B loaded and warmed in %.1f s", Date().timeIntervalSince(loadStart)))
            phase = .ready
            if CommandLine.arguments.contains("--auto") {
                try await Task.sleep(for: .milliseconds(1500))
                await triageAll()
            }
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func triageAll() async {
        guard phase == .ready, let answer else { return }
        phase = .running
        let states = rows.map(\.issue.state)
        let numbers = rows.map(\.issue.number)
        // Inference runs back to back off the main actor; the UI only consumes results, so rendering never delays
        // the next call.
        let (results, continuation) = AsyncThrowingStream.makeStream(of: (Int, TriageResult).self)
        let producer = Task.detached(priority: .userInitiated) {
            do {
                for (index, state) in states.enumerated() {
                    try Task.checkCancellation()
                    let start = DispatchTime.now().uptimeNanoseconds
                    let result = try await answer(state)
                    let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                    let triage = TriageResult(result, ms: ms)
                    print(
                        String(format: "%5.1f ms  #%-5d ", ms, numbers[index])
                            + triage.labels.joined(separator: "  "))
                    continuation.yield((index, triage))
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        defer { producer.cancel() }

        rows.first?.live = true
        scroll.target = rows.first?.id
        let t0 = Date()
        var sumMs = 0.0
        var counts = Dictionary(uniqueKeysWithValues: LabelPalette.workgroups.map { ($0, 0) })
        do {
            for try await (index, result) in results {
                let row = rows[index]
                row.labeledAt = Date()
                row.result = result
                withAnimation(.easeOut(duration: 0.6)) { row.live = false }
                if index + 1 < rows.count {
                    rows[index + 1].live = true
                    if index % 2 == 1 { scroll.target = rows[index + 1].id }
                }
                counts[result.labels[1], default: 0] += 1
                sumMs += result.ms
                let done = index + 1
                stats.done = done
                stats.decisions = done * TriageQuestions.all.count
                stats.avgMs = sumMs / Double(done)
                stats.rate = Double(done) / Date().timeIntervalSince(t0)
                if done % 3 == 0 || done == rows.count { updateWorkgroups(counts) }
            }
        } catch {
            phase = .failed(error.localizedDescription)
            return
        }
        let seconds = Date().timeIntervalSince(t0)
        print(
            String(
                format: "triaged %d issues · %d decisions · %.1f ms per issue · %.1f issues/s · %.1f s total",
                stats.done, stats.decisions, sumMs / Double(max(1, stats.done)), Double(stats.done) / seconds, seconds))
        phase = .finished
        scroll.anchor = .top
        scroll.target = rows.first?.id
    }

    private func updateWorkgroups(_ counts: [String: Int]) {
        let order = Dictionary(uniqueKeysWithValues: LabelPalette.workgroups.enumerated().map { ($1, $0) })
        let sorted = counts.map { WorkgroupCount(name: $0.key, count: $0.value) }
            .sorted { ($0.count, -(order[$0.name] ?? 99)) > ($1.count, -(order[$1.name] ?? 99)) }
        if sorted != stats.workgroups {
            withAnimation(.easeInOut(duration: 0.25)) { stats.workgroups = sorted }
        }
    }
}
