import Foundation
import os

/// Bounded tail of a child process's standard error, kept for error messages.
final class OutputTail: Sendable {
    private let data = OSAllocatedUnfairLock(initialState: Data())
    private let limit: Int

    init(limit: Int = 16_384) { self.limit = limit }

    func append(_ chunk: Data) {
        data.withLock { data in
            data.append(chunk)
            if data.count > limit { data = Data(data.suffix(limit)) }
        }
    }

    var text: String {
        String(decoding: data.withLock { $0 }, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Newline-delimited messages from a child process, awaited one at a time with a deadline.
///
/// `receive` is fed from a `readabilityHandler`; an empty chunk marks end of file. At most one reader
/// waits at a time. A timed-out or cancelled wait leaves the stream position undefined, so callers
/// must stop using the stream afterwards.
final class LineChannel: Sendable {
    private struct Waiter: Sendable {
        let id: UInt64
        let continuation: CheckedContinuation<Data?, Error>
        var timer: Task<Void, Never>?
    }

    private struct State: Sendable {
        var buffer = Data()
        var lines: [Data] = []
        var closed = false
        var failure: PublishedCoreMLError?
        var waiter: Waiter?
        var nextID: UInt64 = 0

        mutating func takeReadyWaiter() -> (Waiter, Result<Data?, Error>)? {
            guard let waiter else { return nil }
            let result: Result<Data?, Error>
            if !lines.isEmpty {
                result = .success(lines.removeFirst())
            } else if let failure {
                result = .failure(failure)
            } else if closed {
                result = .success(nil)
            } else {
                return nil
            }
            self.waiter = nil
            return (waiter, result)
        }
    }

    private let state = OSAllocatedUnfairLock(initialState: State())
    private let limit: Int

    init(limit: Int = 4_194_304) { self.limit = limit }

    func receive(_ chunk: Data) {
        let ready = state.withLock { state -> (Waiter, Result<Data?, Error>)? in
            if chunk.isEmpty {
                state.closed = true
            } else if !state.closed {
                state.buffer.append(chunk)
                while let index = state.buffer.firstIndex(of: 0x0A) {
                    state.lines.append(Data(state.buffer[state.buffer.startIndex..<index]))
                    state.buffer = Data(state.buffer[(index + 1)...])
                }
                if state.buffer.count > limit {
                    state.failure = .invalidResponse("Worker message exceeded \(limit) bytes")
                    state.closed = true
                }
            }
            return state.takeReadyWaiter()
        }
        if let (waiter, result) = ready {
            waiter.timer?.cancel()
            waiter.continuation.resume(with: result)
        }
    }

    /// The next line, or `nil` once the stream has ended.
    func next(timeout: Duration, waitingFor description: String) async throws -> Data? {
        let id = state.withLock { state in
            state.nextID += 1
            return state.nextID
        }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Data?, Error>) in
                let immediate = state.withLock { state -> Result<Data?, Error>? in
                    if Task.isCancelled { return .failure(CancellationError()) }
                    if !state.lines.isEmpty { return .success(state.lines.removeFirst()) }
                    if let failure = state.failure { return .failure(failure) }
                    if state.closed { return .success(nil) }
                    precondition(state.waiter == nil, "LineChannel supports one reader at a time")
                    state.waiter = Waiter(id: id, continuation: continuation)
                    return nil
                }
                if let immediate {
                    continuation.resume(with: immediate)
                    return
                }
                let timer = Task { [weak self] in
                    try? await Task.sleep(for: timeout)
                    guard !Task.isCancelled else { return }
                    self?.fail(id: id, with: PublishedCoreMLError.timedOut(description))
                }
                let attached = state.withLock { state -> Bool in
                    guard state.waiter?.id == id else { return false }
                    state.waiter?.timer = timer
                    return true
                }
                if !attached { timer.cancel() }
            }
        } onCancel: {
            fail(id: id, with: CancellationError())
        }
    }

    private func fail(id: UInt64, with error: Error) {
        let waiter = state.withLock { state -> Waiter? in
            guard let waiter = state.waiter, waiter.id == id else { return nil }
            state.waiter = nil
            return waiter
        }
        waiter?.timer?.cancel()
        waiter?.continuation.resume(throwing: error)
    }
}

enum PublishedProcess {
    /// Run a command to completion without blocking a thread; cancelling the task terminates it.
    static func run(_ executable: URL, arguments: [String]) async throws -> (code: Int32, standardError: String) {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        let errorPipe = Pipe()
        process.standardError = errorPipe
        let tail = OutputTail()
        let reader = errorPipe.fileHandleForReading
        reader.readabilityHandler = { handle in
            let chunk = handle.availableData
            if chunk.isEmpty { handle.readabilityHandler = nil } else { tail.append(chunk) }
        }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                process.terminationHandler = { finished in
                    reader.readabilityHandler = nil
                    tail.append(reader.readDataToEndOfFile())
                    continuation.resume(returning: (finished.terminationStatus, tail.text))
                }
                do {
                    try Task.checkCancellation()
                    try process.run()
                } catch {
                    process.terminationHandler = nil
                    reader.readabilityHandler = nil
                    continuation.resume(throwing: error)
                }
            }
        } onCancel: {
            if process.isRunning { process.terminate() }
        }
    }
}
