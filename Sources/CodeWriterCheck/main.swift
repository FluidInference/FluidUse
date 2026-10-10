import FluidUse
import Foundation

// Parity + latency check for the Swift code-writer host: replays HumanEval prompts through CodeWriterManager, compares
// each token stream with a reference run (Python, same prompt, greedy) and runs the HumanEval tests with python3.
//   swift run -c release CodeWriterCheck <model dir | hf> <HumanEval.jsonl> <reference.jsonl> [limit]
// `hf` downloads the pinned snapshot through CodeWriterModelStore.

let arguments = CommandLine.arguments
guard arguments.count >= 4 else {
    FileHandle.standardError.write(
        Data("usage: CodeWriterCheck <model dir | hf> <HumanEval.jsonl> <reference.jsonl> [limit]\n".utf8))
    exit(2)
}
let limit = arguments.count > 4 ? Int(arguments[4]) ?? .max : .max
let system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

struct Problem: Decodable {
    let taskID: String
    let prompt: String
    let entryPoint: String
    let test: String

    enum CodingKeys: String, CodingKey {
        case taskID = "task_id"
        case prompt
        case entryPoint = "entry_point"
        case test
    }
}

struct Reference: Decodable {
    let taskID: String
    let tokens: [Int]
    let passed: Bool

    enum CodingKeys: String, CodingKey {
        case taskID = "task_id"
        case tokens
        case passed
    }
}

func lines(_ path: String) throws -> [Data] {
    try String(contentsOf: URL(fileURLWithPath: path), encoding: .utf8)
        .split(whereSeparator: \.isNewline).map { Data($0.utf8) }
}

let problems = try lines(arguments[2]).prefix(limit).map { try JSONDecoder().decode(Problem.self, from: $0) }
let references = Dictionary(
    uniqueKeysWithValues: try lines(arguments[3]).map { try JSONDecoder().decode(Reference.self, from: $0) }
        .map { ($0.taskID, $0) })

/// Mirrors `humaneval.py`: keep the prompt's imports/helpers whether the model wrote the body or the whole function.
func program(_ problem: Problem, _ code: String) -> String {
    let body =
        code.contains("def \(problem.entryPoint)")
        ? problem.prompt + "\n    pass\n\n" + code : problem.prompt + code
    return body + "\n\n" + problem.test + "\n\ncheck(\(problem.entryPoint))\n"
}

func passes(_ source: String) throws -> Bool {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent("codewriter-\(UUID().uuidString).py")
    try source.write(to: file, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(at: file) }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = ["python3", file.path]
    process.standardOutput = FileHandle.nullDevice
    process.standardError = FileHandle.nullDevice
    try process.run()
    let deadline = Date().addingTimeInterval(10)
    while process.isRunning, Date() < deadline { usleep(20_000) }
    if process.isRunning {
        process.terminate()
        return false
    }
    return process.terminationStatus == 0
}

@available(macOS 15.0, *)
func run() async throws {
    let directory =
        arguments[1] == "hf"
        ? try await CodeWriterModelStore.ensure { file, bytes in if bytes > 0 { print("downloaded \(file)") } }
        : URL(fileURLWithPath: arguments[1])
    let manager = try await CodeWriterManager.load(from: directory)
    try await manager.warmUp()
    var passed = 0
    var referencePassed = 0
    var identical = 0
    var prefills: [Double] = []
    var perToken: [Double] = []
    for (index, problem) in problems.enumerated() {
        let user = "Complete the following python code:\n```python\n\(problem.prompt)\n```"
        let completion = try await manager.write(task: user, system: system, maxNewTokens: 512)
        let ok = try passes(program(problem, completion.code))
        passed += ok ? 1 : 0
        prefills.append(completion.timing.prefillSeconds)
        if completion.timing.generatedTokens > 0 {
            perToken.append(completion.timing.decodeSeconds / Double(completion.timing.generatedTokens))
        }
        var note = ""
        if let reference = references[problem.taskID] {
            referencePassed += reference.passed ? 1 : 0
            if completion.tokens == reference.tokens {
                identical += 1
            } else {
                let diverge =
                    zip(completion.tokens, reference.tokens).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                    ?? min(completion.tokens.count, reference.tokens.count)
                note = " differs@\(diverge) ref=\(reference.passed ? "PASS" : "fail")"
            }
        }
        print(
            "\(index + 1)/\(problems.count) \(problem.taskID) \(ok ? "PASS" : "fail") "
                + "\(completion.timing.generatedTokens) tok\(note)")
    }
    func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        return sorted.isEmpty ? 0 : sorted[sorted.count / 2]
    }
    print("pass@1 \(passed)/\(problems.count); reference \(referencePassed); identical token streams \(identical)")
    print(
        String(
            format: "prefill p50: %.1f ms   decode p50: %.1f ms/token (%.0f tok/s)", median(prefills) * 1000,
            median(perToken) * 1000, 1 / max(median(perToken), 1e-9)))
}

if #available(macOS 15.0, *) {
    try await run()
} else {
    FileHandle.standardError.write(Data("CodeWriterCheck needs macOS 15\n".utf8))
    exit(2)
}
