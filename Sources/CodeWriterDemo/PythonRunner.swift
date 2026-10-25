import Foundation

/// Runs model-written Python with the system `python3`: each assert separately, so the console can show which passed.
enum PythonRunner {
    struct Outcome: Sendable {
        /// One entry per assert, in order; nil when the code itself failed before the tests ran.
        let checks: [(test: String, passed: Bool)]
        /// Error text when the code did not load (syntax error, exception at import time, timeout).
        let error: String?
        let seconds: Double

        var passed: Bool { error == nil && !checks.isEmpty && checks.allSatisfy(\.passed) }
    }

    /// The harness: load the code, then run every assert in its own try block and print `PASS`/`FAIL` per test.
    static func harness(code: String, imports: [String], tests: [String]) -> String {
        let quoted = tests.map { "    " + pythonLiteral($0) + "," }.joined(separator: "\n")
        return """
            import sys
            _source = \(pythonLiteral((imports + [code]).joined(separator: "\n")))
            _tests = [
            \(quoted)
            ]
            try:
                exec(compile(_source, "solution.py", "exec"), globals())
            except BaseException as error:
                print("LOAD " + type(error).__name__ + ": " + str(error).splitlines()[0] if str(error) else "LOAD " + type(error).__name__)
                sys.exit(0)
            for _test in _tests:
                try:
                    exec(_test, globals())
                    print("PASS")
                except BaseException as error:
                    print("FAIL " + type(error).__name__)
            """
    }

    static func run(code: String, imports: [String], tests: [String], timeout: Double = 10) async -> Outcome {
        let started = Date()
        let script = harness(code: code, imports: imports, tests: tests)
        let output: String
        do {
            output = try await execute(script, timeout: timeout)
        } catch {
            return Outcome(checks: [], error: error.localizedDescription, seconds: Date().timeIntervalSince(started))
        }
        let lines = output.split(whereSeparator: \.isNewline).map(String.init)
        if let load = lines.first(where: { $0.hasPrefix("LOAD ") }) {
            return Outcome(checks: [], error: String(load.dropFirst(5)), seconds: Date().timeIntervalSince(started))
        }
        let results = lines.filter { $0.hasPrefix("PASS") || $0.hasPrefix("FAIL") }
        let checks = zip(tests, results).map { (test: $0, passed: $1 == "PASS") }
        let error = checks.count == tests.count ? nil : "the tests did not finish"
        return Outcome(checks: checks, error: error, seconds: Date().timeIntervalSince(started))
    }

    /// Whether `code` parses as Python (for tasks typed in by hand, which have no tests).
    static func compiles(_ code: String) async -> String? {
        let script = """
            import ast
            try:
                ast.parse(\(pythonLiteral(code)))
                print("OK")
            except SyntaxError as error:
                print("SyntaxError line " + str(error.lineno) + ": " + str(error.msg))
            """
        let output = (try? await execute(script, timeout: 10)) ?? "python3 did not run"
        let line = output.trimmingCharacters(in: .whitespacesAndNewlines)
        return line == "OK" ? nil : line
    }

    private static func execute(_ script: String, timeout: Double) async throws -> String {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(
            "python-writer-\(UUID().uuidString).py")
        try script.write(to: file, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: file) }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["python3", file.path]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        try process.run()
        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning {
            if Date() > deadline {
                process.terminate()
                throw RunnerError.timeout(timeout)
            }
            try await Task.sleep(for: .milliseconds(20))
        }
        return String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
    }

    /// A Python string literal for arbitrary text (JSON string syntax is valid Python for these escapes).
    static func pythonLiteral(_ text: String) -> String {
        let data = (try? JSONSerialization.data(withJSONObject: [text], options: [.withoutEscapingSlashes])) ?? Data()
        let array = String(decoding: data, as: UTF8.self)
        return String(array.dropFirst().dropLast())
    }

    enum RunnerError: Error, LocalizedError {
        case timeout(Double)

        var errorDescription: String? {
            switch self {
            case .timeout(let seconds): return "timed out after \(Int(seconds)) s"
            }
        }
    }
}
