import Foundation
import FluidUse

/// `FluidUseOfficialBench run …` — feeds authors' official benchmark requests to FluidUse models.
///
/// `run` reads one request per line and writes one result row per line, in order, keeping failures.
/// Bridge models receive each line's bytes unchanged. Verdict lines are `{"context", "question", "labels"}` for
/// raw logits or `{"context", "question": {typed}}` for the calibrated serving path.
let arguments = Array(CommandLine.arguments.dropFirst())
do {
    switch arguments.first {
    case "run": try await RunCommand.run(Options(Array(arguments.dropFirst())))
    default:
        print(
            """
            usage: FluidUseOfficialBench run --model MODEL --in requests.jsonl --out results.jsonl [options]
            MODEL: verdict or a PublishedCoreMLModel raw value (kev-0-5b, kev-0.6b, lfm2-5-350m-rlcd, jeff, …)
            options: --precision P  --cache DIR  --lengths 128,512 (verdict)  --root DIR --python PATH (local bridge)
                     --strict-context (reject instead of shortening Kev state)  --meta meta.json
                     --request-timeout SECONDS (bridge; a closed session is restarted and timed separately)
            """)
        exit(2)
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8))
    exit(1)
}

struct Options {
    private var values: [String: String] = [:]
    private var flags: Set<String> = []

    init(_ arguments: [String]) throws {
        var index = 0
        while index < arguments.count {
            let key = arguments[index]
            guard key.hasPrefix("--") else { throw BenchError.usage("Unexpected argument \(key)") }
            if index + 1 < arguments.count, !arguments[index + 1].hasPrefix("--") {
                values[String(key.dropFirst(2))] = arguments[index + 1]
                index += 2
            } else {
                flags.insert(String(key.dropFirst(2)))
                index += 1
            }
        }
    }

    subscript(_ key: String) -> String? { values[key] }
    func flag(_ key: String) -> Bool { flags.contains(key) }

    func required(_ key: String) throws -> String {
        guard let value = values[key] else { throw BenchError.usage("Missing --\(key)") }
        return value
    }

    var cache: URL? { values["cache"].map { URL(fileURLWithPath: $0, isDirectory: true) } }
}

enum BenchError: Error, LocalizedError {
    case usage(String)
    case request(String)

    var errorDescription: String? {
        switch self {
        case .usage(let message), .request(let message): message
        }
    }
}

/// One loaded model behind a uniform request → JSON answer call.
enum LoadedModel {
    case bridge(PublishedCoreMLManager)
    case verdict(VerdictManager)

    static func load(_ options: Options) async throws -> (LoadedModel, [String: Any]) {
        let name = try options.required("model")
        let started = ContinuousClock.now
        var meta: [String: Any] = ["model": name]
        if name == "verdict" {
            let lengths = (options["lengths"] ?? "128,512").split(separator: ",").compactMap { Int($0) }
            let manager = try await VerdictManager.load(
                cacheDirectory: options.cache, configuration: .init(lengths: lengths))
            meta["lengths"] = lengths
            meta["execution"] = "Swift/Core ML in process"
            meta["startup_ms"] = milliseconds(ContinuousClock.now - started)
            return (.verdict(manager), meta)
        }
        guard let model = PublishedCoreMLModel(rawValue: name) else {
            throw BenchError.usage("Unknown model \(name)")
        }
        var environment: [String: String] = [:]
        if options.flag("strict-context") { environment["FLUIDUSE_STRICT_CONTEXT"] = "1" }
        var configuration = PublishedCoreMLManager.Configuration(
            precision: options["precision"], environment: environment)
        if let seconds = options["request-timeout"].flatMap(Double.init) {
            configuration.requestTimeout = .milliseconds(Int(seconds * 1000))
        }
        let manager: PublishedCoreMLManager
        if let root = options["root"] {
            manager = try await PublishedCoreMLManager.start(
                model: model, from: URL(fileURLWithPath: root, isDirectory: true),
                python: URL(fileURLWithPath: try options.required("python")), configuration: configuration)
        } else {
            manager = try await PublishedCoreMLManager.load(
                model: model, cacheDirectory: options.cache, configuration: configuration)
        }
        meta["precision"] = manager.precision
        meta["strict_context"] = options.flag("strict-context")
        meta["execution"] = "Bridge: Python worker performs tokenization, Core ML predict, and decoding"
        meta["startup_ms"] = milliseconds(ContinuousClock.now - started)
        return (.bridge(manager), meta)
    }

    /// Answer one request line; returns the answer as JSON bytes.
    func answer(_ line: Data) async throws -> Data {
        switch self {
        case .bridge(let manager):
            return try await manager.evaluate(line)
        case .verdict(let manager):
            guard let object = try JSONSerialization.jsonObject(with: line) as? [String: Any],
                let context = object["context"] as? String
            else { throw BenchError.request("Verdict lines need a context") }
            if let labels = object["labels"] as? [String] {
                let logits = try await manager.logits(
                    question: object["question"] as? String, context: context, labels: labels)
                return try JSONSerialization.data(withJSONObject: ["logits": logits.map(Double.init)])
            }
            guard let question = object["question"] as? [String: Any] else {
                throw BenchError.request("Verdict lines need labels or a typed question")
            }
            let answer = try await manager.answer(context: context, question: try Self.verdictQuestion(question))
            return try JSONSerialization.data(withJSONObject: [
                "candidate_ids": answer.candidateIDs, "logits": answer.logits.map(Double.init),
                "probabilities": answer.probabilities, "selected_id": answer.selectedID,
                "is_abstention": answer.isAbstention, "token_count": answer.tokenCount,
            ])
        }
    }

    private static func verdictQuestion(_ object: [String: Any]) throws -> VerdictQuestion {
        let options = (object["options"] as? [[String: Any]] ?? []).compactMap { option -> VerdictQuestion.Option? in
            guard let id = option["id"] as? String, let description = option["description"] as? String else {
                return nil
            }
            return VerdictQuestion.Option(id: id, description: description)
        }
        guard object["type"] as? String == "choice", let prompt = object["question"] as? String,
            options.count == (object["options"] as? [Any])?.count
        else { throw BenchError.request("Only choice questions with id/description options are supported") }
        return .choice(question: prompt, options: options)
    }
}

func milliseconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds) * 1000 + Double(duration.components.attoseconds) / 1e15
}

@discardableResult
func writeSummary(
    _ meta: [String: Any], _ requests: Int, _ counts: [String: Int], _ restarts: [[String: Any]], _ options: Options,
    failure: String? = nil
) throws -> [String: Any] {
    var summary = meta
    summary["requests"] = requests
    summary["counts"] = counts
    summary["worker_restarts"] = restarts
    if let failure { summary["aborted"] = "worker restart failed: \(failure)" }
    if let path = options["meta"] {
        try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: path))
    }
    return summary
}

enum RunCommand {
    static func run(_ options: Options) async throws {
        let input = URL(fileURLWithPath: try options.required("in"))
        let output = URL(fileURLWithPath: try options.required("out"))
        let lines = try Data(contentsOf: input).split(separator: 0x0A, omittingEmptySubsequences: true)
        var (model, meta) = try await LoadedModel.load(options)
        var restarts: [[String: Any]] = []
        FileManager.default.createFile(atPath: output.path, contents: nil)
        let handle = try FileHandle(forWritingTo: output)
        defer { try? handle.close() }
        var counts: [String: Int] = [:]
        for (index, line) in lines.enumerated() {
            let started = ContinuousClock.now
            var row: [String: Any] = ["index": index]
            do {
                let answer = try await model.answer(Data(line))
                row["status"] = "ok"
                row["answer"] = try JSONSerialization.jsonObject(with: answer, options: .fragmentsAllowed)
            } catch {
                row["status"] = "error"
                row["error"] = error.localizedDescription
            }
            row["latency_ms"] = milliseconds(ContinuousClock.now - started)
            counts[row["status"] as! String, default: 0] += 1
            var encoded = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
            encoded.append(0x0A)
            try handle.write(contentsOf: encoded)
            // A timeout, cancellation, or worker exit closes the bridge session. Restart after the row is
            // recorded, and time the restart separately from any request.
            if case .bridge(let manager) = model, await manager.isClosed, index + 1 < lines.count {
                let restartStarted = ContinuousClock.now
                do {
                    (model, _) = try await LoadedModel.load(options)
                } catch {
                    try writeSummary(meta, lines.count, counts, restarts, options, failure: error.localizedDescription)
                    throw error
                }
                restarts.append([
                    "after_index": index, "startup_ms": milliseconds(ContinuousClock.now - restartStarted),
                ])
            }
            if (index + 1) % 100 == 0 {
                FileHandle.standardError.write(Data("\(index + 1)/\(lines.count) \(counts)\n".utf8))
            }
        }
        let summary = try writeSummary(meta, lines.count, counts, restarts, options)
        print(
            String(decoding: try JSONSerialization.data(withJSONObject: summary, options: [.sortedKeys]), as: UTF8.self)
        )
    }
}
