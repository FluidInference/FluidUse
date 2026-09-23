import Darwin
import Foundation

/// Mac-only bridge to the published model-specific Core ML serving code.
///
/// The model packages remain local and run through Core ML. Python performs the released tokenizer,
/// prompt rendering, and decoding contract. Start one session per model and reuse it: the worker loads
/// its packages once and requests are served one at a time in call order. A request that times out or is
/// cancelled terminates the worker, because its reply can no longer be matched to a caller; start a new
/// session afterwards. Model errors (for example an over-long input) are reported per request and leave
/// the session usable.
public actor PublishedCoreMLManager {
    public struct Configuration: Sendable {
        /// Package precision; `nil` selects the model's first entry in `PublishedCoreMLModel.precisions`.
        public var precision: String?
        /// Limit for loading packages before the worker reports ready.
        public var startupTimeout: Duration
        /// Limit for one request. Kai and Lex load a different package when the question type changes.
        public var requestTimeout: Duration
        /// Copy the worker's standard error to this process's standard error in addition to keeping its tail.
        public var forwardsStandardError: Bool
        /// Extra environment variables for the worker.
        public var environment: [String: String]

        public init(
            precision: String? = nil, startupTimeout: Duration = .seconds(600),
            requestTimeout: Duration = .seconds(300), forwardsStandardError: Bool = false,
            environment: [String: String] = [:]
        ) {
            self.precision = precision
            self.startupTimeout = startupTimeout
            self.requestTimeout = requestTimeout
            self.forwardsStandardError = forwardsStandardError
            self.environment = environment
        }
    }

    public nonisolated let model: PublishedCoreMLModel
    public nonisolated let precision: String
    private let configuration: Configuration
    private let process: Process
    private let input: FileHandle
    private let replies: LineChannel
    private let standardError: OutputTail
    private var closedReason: String?
    private var writeInFlight = false
    /// Internal for tests: whether the stdin descriptor has been released.
    private(set) var inputClosed = false
    private var busy = false
    private var queue: [CheckedContinuation<Void, Never>] = []

    /// Download the pinned snapshot, build its locked Python environment with `uv`, and start a session.
    /// - Parameter cacheDirectory: The parent Models directory, not the repository subdirectory.
    public static func load(
        model: PublishedCoreMLModel, cacheDirectory: URL? = nil, uv: URL? = nil,
        configuration: Configuration = Configuration(), progress: PublishedCoreMLModelStore.Progress? = nil
    ) async throws -> PublishedCoreMLManager {
        let directory = try await PublishedCoreMLModelStore.ensure(
            model: model, precision: configuration.precision, cacheDirectory: cacheDirectory, progress: progress)
        let python = try await PublishedCoreMLModelStore.prepareEnvironment(for: model, in: directory, uv: uv)
        return try await start(model: model, from: directory, python: python, configuration: configuration)
    }

    /// Start a serving session from a materialized repository and a Python environment holding its
    /// dependencies. NanoJev may fetch its pinned upstream tokenizer/source files; it never fetches or
    /// redistributes trained weights.
    /// - Parameters:
    ///   - directory: Local repository root containing the selected `.mlpackage` and tokenizer files.
    ///   - python: Python 3.12 interpreter with the published toolkit's dependencies installed.
    public static func start(
        model: PublishedCoreMLModel, from directory: URL, python: URL,
        configuration: Configuration = Configuration()
    ) async throws -> PublishedCoreMLManager {
        let manager = try PublishedCoreMLManager(
            model: model, directory: directory, python: python, configuration: configuration)
        try await manager.waitUntilReady()
        return manager
    }

    private init(
        model: PublishedCoreMLModel, directory: URL, python: URL, configuration: Configuration
    ) throws {
        guard directory.isFileURL, python.isFileURL else {
            throw PublishedCoreMLError.invalidRequest("Local file URLs are required")
        }
        let precision = configuration.precision ?? model.precisions[0]
        let manager = FileManager.default
        guard manager.fileExists(atPath: directory.appendingPathComponent(model.rootMarker).path) else {
            throw PublishedCoreMLError.missingAsset(model.rootMarker)
        }
        for path in try model.requiredPackages(precision: precision) {
            guard manager.fileExists(atPath: directory.appendingPathComponent(path).path) else {
                throw PublishedCoreMLError.missingAsset(path)
            }
        }
        guard manager.isExecutableFile(atPath: python.path) else {
            throw PublishedCoreMLError.missingAsset(python.path)
        }
        guard
            let worker = Bundle.module.url(
                forResource: "published-coreml-worker", withExtension: "py", subdirectory: "Resources")
        else { throw PublishedCoreMLError.missingAsset("published-coreml-worker.py") }

        let inputPipe = Pipe()
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        let replies = LineChannel()
        let standardError = OutputTail()
        let forwards = configuration.forwardsStandardError
        let task = Process()
        task.executableURL = python
        task.arguments = [worker.path, "--model", model.rawValue, "--root", directory.path, "--precision", precision]
        // The locked environment must not pick up the host's Python search paths.
        var inherited = ProcessInfo.processInfo.environment
        for key in ["PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV", "CONDA_PREFIX"] {
            inherited.removeValue(forKey: key)
        }
        task.environment =
            inherited
            .merging(["PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1", "TOKENIZERS_PARALLELISM": "false"]) {
                _, new in new
            }
            .merging(configuration.environment) { _, new in new }
        task.standardInput = inputPipe
        task.standardOutput = outputPipe
        task.standardError = errorPipe
        outputPipe.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if chunk.isEmpty { handle.readabilityHandler = nil }
            replies.receive(chunk)
        }
        errorPipe.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            guard !chunk.isEmpty else {
                handle.readabilityHandler = nil
                return
            }
            standardError.append(chunk)
            if forwards { FileHandle.standardError.write(chunk) }
        }
        // A write after the worker exits must fail with EPIPE instead of raising SIGPIPE in the host.
        _ = fcntl(inputPipe.fileHandleForWriting.fileDescriptor, F_SETNOSIGPIPE, 1)

        self.model = model
        self.precision = precision
        self.configuration = configuration
        self.process = task
        self.input = inputPipe.fileHandleForWriting
        self.replies = replies
        self.standardError = standardError
        do { try task.run() } catch {
            outputPipe.fileHandleForReading.readabilityHandler = nil
            errorPipe.fileHandleForReading.readabilityHandler = nil
            throw PublishedCoreMLError.runtime("Could not start Python: \(error.localizedDescription)")
        }
    }

    deinit {
        Self.stop(process)
    }

    /// Whether the session has ended (timeout, cancellation, worker exit, or `shutdown`).
    public var isClosed: Bool { closedReason != nil }

    /// Worker process identifier, for tests that stall the worker.
    nonisolated var workerProcessIdentifier: Int32 { process.processIdentifier }

    /// SIGTERM, then SIGKILL after a grace period: a stopped or wedged worker ignores SIGTERM, and only its
    /// death unblocks a write into its full stdin pipe.
    static func stop(_ process: Process) {
        guard process.isRunning else { return }
        process.terminate()
        let identifier = process.processIdentifier
        DispatchQueue.global().asyncAfter(deadline: .now() + 1) {
            if process.isRunning { kill(identifier, SIGKILL) }
        }
    }

    private func waitUntilReady() async throws {
        do {
            let line = try await replies.next(timeout: configuration.startupTimeout, waitingFor: "model loading")
            guard let line else { throw exitError("Worker exited while loading") }
            guard line == Data("ready".utf8) else {
                throw PublishedCoreMLError.invalidResponse(
                    "Expected ready, got \(String(decoding: line, as: UTF8.self))")
            }
        } catch {
            close(reason: "startup failed")
            throw error
        }
    }

    /// Evaluate one request in the model's published JSON schema and return its JSON answer.
    ///
    /// The request bytes reach the runtime unchanged apart from line breaks, so object member order
    /// (for example Choice criteria order) is preserved. The answer is returned as the runtime wrote it.
    public func evaluate(_ request: Data) async throws -> Data {
        guard (try? JSONSerialization.jsonObject(with: request)) is [String: Any] else {
            throw PublishedCoreMLError.invalidRequest("Top-level JSON must be an object")
        }
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        if let closedReason { throw PublishedCoreMLError.closed(closedReason) }

        // Raw line breaks can only be insignificant whitespace in valid JSON.
        var line = Data(request.map { $0 == 0x0A || $0 == 0x0D ? 0x20 : $0 })
        line.append(0x0A)
        // One deadline covers writing the request and waiting for the reply. A worker that stops reading stdin
        // blocks a large write indefinitely; stopping it at the deadline, or when the caller cancels, makes the
        // write fail with EPIPE.
        let started = ContinuousClock.now
        let deadline = started + configuration.requestTimeout
        let process = process
        let watchdog = Task { [timeout = configuration.requestTimeout] in
            try? await Task.sleep(for: timeout)
            if !Task.isCancelled { Self.stop(process) }
        }
        writeInFlight = true
        do {
            try await withTaskCancellationHandler {
                try await Self.write(line, to: input)
            } onCancel: {
                Self.stop(process)
            }
            watchdog.cancel()
            finishWrite()
        } catch {
            watchdog.cancel()
            finishWrite()
            if let closedReason { throw PublishedCoreMLError.closed(closedReason) }
            if Task.isCancelled {
                close(reason: "a request was cancelled")
                throw CancellationError()
            }
            if ContinuousClock.now >= deadline {
                close(reason: "a request timed out")
                throw PublishedCoreMLError.timedOut("writing a request")
            }
            close(reason: "worker stopped accepting requests")
            throw exitError("Worker stopped accepting requests")
        }
        if let closedReason { throw PublishedCoreMLError.closed(closedReason) }
        let reply: Data?
        do {
            let remaining = deadline - ContinuousClock.now
            guard remaining > .zero else { throw PublishedCoreMLError.timedOut("a model answer") }
            reply = try await replies.next(timeout: remaining, waitingFor: "a model answer")
        } catch {
            close(reason: error is CancellationError ? "a request was cancelled" : "a request timed out")
            throw error
        }
        guard let reply else {
            close(reason: "worker exited")
            throw exitError("Worker exited during a request")
        }
        if reply.starts(with: Data("ok ".utf8)) { return Data(reply.dropFirst(3)) }
        if reply.starts(with: Data("error ".utf8)) {
            let message = (try? JSONSerialization.jsonObject(with: reply.dropFirst(6), options: .fragmentsAllowed))
            throw PublishedCoreMLError.runtime(message as? String ?? String(decoding: reply, as: UTF8.self))
        }
        close(reason: "protocol error")
        throw PublishedCoreMLError.invalidResponse(String(decoding: reply.prefix(200), as: UTF8.self))
    }

    /// Stop the worker. Later requests throw `PublishedCoreMLError.closed`.
    public func shutdown() {
        close(reason: "shut down")
    }

    /// Recent worker standard error, useful when a model reports a failure.
    public nonisolated var standardErrorTail: String { standardError.text }

    private func close(reason: String) {
        guard closedReason == nil else { return }
        closedReason = reason
        Self.stop(process)
        // Closing the descriptor under a blocked write could let it be reused; finishWrite closes it once the
        // write unwinds, including when another call closed the session meanwhile.
        if !writeInFlight { closeInput() }
    }

    private func finishWrite() {
        writeInFlight = false
        if closedReason != nil { closeInput() }
    }

    private func closeInput() {
        guard !inputClosed else { return }
        inputClosed = true
        try? input.close()
    }

    private func exitError(_ message: String) -> PublishedCoreMLError {
        let tail = standardError.text
        return .runtime(tail.isEmpty ? message : "\(message): \(tail)")
    }

    private static func write(_ data: Data, to handle: FileHandle) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try handle.write(contentsOf: data)
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func acquire() async {
        guard busy else {
            busy = true
            return
        }
        await withCheckedContinuation { queue.append($0) }
    }

    private func release() {
        if queue.isEmpty { busy = false } else { queue.removeFirst().resume() }
    }
}
