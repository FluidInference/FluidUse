import FluidUse
import Foundation
import SwiftUI

/// One entry in the task list: a plain-English request, the Python feature it exercises, and its asserts.
struct CodingTask: Decodable, Identifiable {
    let id: String
    let topic: String
    let prompt: String
    let tests: [String]
    let imports: [String]

    /// What the model sees: the request plus the first assert, so it knows the function name and signature.
    var userMessage: String { "\(prompt)\nYour code should pass this test:\n\(tests[0])" }
}

/// Plays the task list through Qwen2.5-Coder-0.5B on Core ML: each task is written live, then its asserts run with
/// `python3`. Pause stops after the current task; Replay starts over.
@available(macOS 15.0, *)
@MainActor
final class WriterModel: ObservableObject {
    enum Step: Equatable {
        case loading(String)
        case idle
        case writing
        case finished
        case failed(String)
    }

    enum Status: Equatable {
        case pending
        case writing
        case testing
        case passed(Int, Int)
        case failed(Int, Int)
        case broken(String)
    }

    struct Check: Identifiable {
        let id: Int
        let test: String
        let passed: Bool
    }

    @Published private(set) var step: Step = .loading("Starting…")
    @Published private(set) var paused = false
    @Published private(set) var statuses: [String: Status] = [:]
    @Published private(set) var current: String?
    @Published private(set) var code = ""
    @Published private(set) var checks: [Check] = []
    @Published private(set) var runError: String?
    @Published private(set) var tokensPerSecond: Double = 0
    @Published private(set) var lastSeconds: Double = 0
    @Published var customTask = ""
    @Published private(set) var customTitle: String?

    let tasks: [CodingTask]
    private var writer: CodeWriterManager?
    private var showTask: Task<Void, Never>?

    var passedCount: Int { statuses.values.filter { if case .passed = $0 { return true } else { return false } }.count }
    var doneCount: Int {
        statuses.values.filter {
            switch $0 {
            case .passed, .failed, .broken: return true
            default: return false
            }
        }.count
    }
    var isRunning: Bool { step == .writing }
    var writerMissing: Bool { writer == nil }
    /// True while tokens are streaming into the editor (cursor shown).
    @Published private(set) var isWriting = false
    var canPlay: Bool { (step == .idle || step == .finished || paused) && writer != nil && !isBusyCustom }
    @Published private(set) var isBusyCustom = false

    init() {
        guard let url = Bundle.module.url(forResource: "tasks", withExtension: "json", subdirectory: "Resources"),
            let data = try? Data(contentsOf: url),
            let tasks = try? JSONDecoder().decode([CodingTask].self, from: data)
        else {
            self.tasks = []
            step = .failed("Resources/tasks.json is missing or invalid")
            return
        }
        self.tasks = tasks
        for task in tasks { statuses[task.id] = .pending }
    }

    /// `CODE_WRITER_MODEL_DIR` or `--model=<dir>` for a local folder; otherwise the pinned Hugging Face snapshot.
    static func localModelDirectory() -> URL? {
        if let path = ProcessInfo.processInfo.environment["CODE_WRITER_MODEL_DIR"], !path.isEmpty {
            return URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        }
        if let argument = CommandLine.arguments.first(where: { $0.hasPrefix("--model=") }) {
            return URL(fileURLWithPath: (String(argument.dropFirst(8)) as NSString).expandingTildeInPath)
        }
        return nil
    }

    func start() {
        guard writer == nil, case .loading = step else { return }
        Task {
            do {
                var directory = Self.localModelDirectory()
                if directory == nil {
                    step = .loading("Downloading \(CodeWriterModelStore.repository) (about 1 GB, first launch only)…")
                    DemoLog.model("fetching \(CodeWriterModelStore.repository)")
                    directory = try await CodeWriterModelStore.ensure { file, bytes in
                        if bytes > 0 { DemoLog.model("downloaded \(file)") }
                    }
                }
                guard let directory else { return }
                DemoLog.model("loading \(directory.path)")
                step = .loading("Loading Qwen2.5-Coder 0.5B (first launch compiles for the Neural Engine)…")
                let started = Date()
                let writer = try await CodeWriterManager.load(from: directory)
                step = .loading("Warming up…")
                try await writer.warmUp()
                self.writer = writer
                DemoLog.model(String(format: "ready in %.1f s", Date().timeIntervalSince(started)))
                step = .idle
                if CommandLine.arguments.contains("--autostart") { play() }
            } catch {
                DemoLog.line("load failed: \(error.localizedDescription)", color: 196)
                step = .failed(error.localizedDescription)
            }
        }
    }

    func play() {
        guard canPlay else { return }
        paused = false
        if step == .finished { reset() }
        step = .writing
        showTask = Task { await runShow() }
    }

    func pause() {
        guard isRunning else { return }
        paused = true
        DemoLog.event("⏸ pausing after this task")
    }

    func replay() {
        showTask?.cancel()
        showTask = nil
        reset()
        paused = false
        step = .idle
        play()
    }

    private func reset() {
        for task in tasks { statuses[task.id] = .pending }
        current = nil
        code = ""
        checks = []
        runError = nil
        customTitle = nil
    }

    private func runShow() async {
        DemoLog.event("▶ writing \(tasks.count) tasks")
        for task in tasks where statuses[task.id] == .pending {
            if Task.isCancelled { return }
            if paused {
                step = .idle
                return
            }
            await write(task)
        }
        guard !Task.isCancelled else { return }
        step = .finished
        DemoLog.event("■ \(passedCount)/\(tasks.count) tasks pass all their tests")
    }

    private func write(_ task: CodingTask) async {
        guard let writer else { return }
        current = task.id
        customTitle = nil
        code = ""
        checks = []
        runError = nil
        statuses[task.id] = .writing
        DemoLog.line("✎ \(task.topic): \(task.prompt)", color: 81)
        do {
            isWriting = true
            defer { isWriting = false }
            let completion = try await writer.write(task: task.userMessage) { text in
                let shown = CodeWriterManager.extractCode(text)
                Task { @MainActor in self.code = shown }
            }
            code = completion.code
            tokensPerSecond = completion.timing.tokensPerSecond
            lastSeconds = completion.timing.totalSeconds
            DemoLog.model(
                String(
                    format: "%d tokens in %.2f s (prefill %.0f ms on the Neural Engine, %.0f tok/s on the GPU)",
                    completion.timing.generatedTokens, completion.timing.totalSeconds,
                    completion.timing.prefillSeconds * 1000, completion.timing.tokensPerSecond))
            statuses[task.id] = .testing
            let outcome = await PythonRunner.run(code: completion.code, imports: task.imports, tests: task.tests)
            checks = outcome.checks.enumerated().map {
                Check(id: $0.offset, test: $0.element.test, passed: $0.element.passed)
            }
            runError = outcome.error
            let passed = outcome.checks.filter(\.passed).count
            if let error = outcome.error, outcome.checks.isEmpty {
                statuses[task.id] = .broken(error)
                DemoLog.line("  ✗ \(error)", color: 196)
            } else if outcome.passed {
                statuses[task.id] = .passed(passed, task.tests.count)
                DemoLog.line("  ✓ \(passed)/\(task.tests.count) tests pass", color: 46)
            } else {
                statuses[task.id] = .failed(passed, task.tests.count)
                DemoLog.line("  ✗ \(passed)/\(task.tests.count) tests pass", color: 208)
            }
        } catch {
            statuses[task.id] = .broken(error.localizedDescription)
            runError = error.localizedDescription
            DemoLog.line("  ✗ \(error.localizedDescription)", color: 196)
        }
    }

    /// A task typed by hand: written the same way; no tests, so the result is only checked to parse as Python.
    func writeCustom() {
        let prompt = customTask.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let writer, !prompt.isEmpty, !isRunning, !isBusyCustom else { return }
        isBusyCustom = true
        current = nil
        customTitle = prompt
        code = ""
        checks = []
        runError = nil
        DemoLog.line("✎ (typed) \(prompt)", color: 81)
        Task {
            defer { isBusyCustom = false }
            do {
                isWriting = true
                let completion = try await writer.write(task: prompt) { text in
                    let shown = CodeWriterManager.extractCode(text)
                    Task { @MainActor in self.code = shown }
                }
                isWriting = false
                code = completion.code
                tokensPerSecond = completion.timing.tokensPerSecond
                lastSeconds = completion.timing.totalSeconds
                runError = await PythonRunner.compiles(completion.code)
                DemoLog.line(
                    runError == nil ? "  ✓ valid Python" : "  ✗ \(runError!)", color: runError == nil ? 46 : 196)
            } catch {
                isWriting = false
                runError = error.localizedDescription
            }
        }
    }

    func select(_ task: CodingTask) {
        guard !isRunning, !isBusyCustom, statuses[task.id] != .pending else { return }
        // Re-show a finished task's result by writing it again (greedy, so the code is the same).
        Task {
            await write(task)
        }
    }
}
