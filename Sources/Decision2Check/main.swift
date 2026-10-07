import FluidUse
import Foundation

/// Decision 2.0 checks.
///
///     swift run -c release Decision2Check parity <model directory> <fixtures.json>
///     swift run -c release Decision2Check example kai|eos     (downloads the pinned snapshot, answers the card example)
///
/// Fixtures: typed-decisions TEST requests with the expected token ids of every question row, the Python Core ML
/// runtime's probabilities and the upstream runtime's answers (decision2-kai/swift_fixtures.py).
@main
struct Decision2Check {
    struct Fixture: Decodable {
        let id: String
        let state: String
        let questions: String
        let ids: [String: [Int]]
        let python: [String: [String: Double]]
        let upstream: [String: [String: Double]]
    }

    static func main() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard #available(macOS 15.0, *) else { fatalError("macOS 15 required") }
        if arguments.first == "parity", arguments.count == 3 {
            try await parity(directory: URL(fileURLWithPath: arguments[1]), fixtures: URL(fileURLWithPath: arguments[2]))
        } else if arguments.first == "example", arguments.count == 2, let model = Decision2Model(rawValue: "decision-2.0-\(arguments[1])-coreml") {
            try await example(model)
        } else {
            fputs("usage: Decision2Check parity <model dir> <fixtures.json> | example kai|eos\n", stderr)
            exit(2)
        }
    }

    @available(macOS 15.0, *)
    static func example(_ model: Decision2Model) async throws {
        let directory = try await Decision2ModelStore.ensure(model) { file, bytes in
            if bytes > 0 { print("downloaded \(file) (\(bytes / 1_000_000) MB)") }
        }
        let manager = try await Decision2Manager.load(from: directory)
        try await manager.warm()
        var times: [Double] = []
        var result: Decision2Result!
        for _ in 0..<5 {
        let start = Date()
        result = try await manager.answer(
            state: "The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
            questions: [
                ("route", .choice("Which team should handle this request?", [
                    "returns": "Refunds, replacements and damaged deliveries",
                    "billing": "Payments, invoices and charges",
                    "technical": "Product setup and faults",
                ])),
                ("receipt", .yesNo("Does the customer have a receipt?")),
                ("urgency", .score("How urgent is this request?", ["Routine", "Soon", "Today"])),
            ])
        times.append(Date().timeIntervalSince(start) * 1000)
        }
        for a in result.answers {
            let detail = a.yes.map { String(format: "yes %.3f", $0) } ?? a.score.map { String(format: "score %.3f", $0) } ?? a.choice
            print("\(a.id): \(detail)  \(zip(a.keys, a.probabilities).map { "\($0)=\(String(format: "%.3f", $1))" }.joined(separator: " "))")
        }
        print("\(manager.modelName): \(result.answers.count) questions, \(result.calls) call, ms per request: "
            + times.map { String(format: "%.1f", $0) }.joined(separator: ", "))
    }

    @available(macOS 15.0, *)
    static func parity(directory: URL, fixtures url: URL) async throws {
        let fixtures = try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: url))
        let manager = try await Decision2Manager.load(from: directory)
        try await manager.warm()
        var tokenMismatch = 0, decisions = 0, flipsPython = 0, flipsUpstream = 0
        var maxPython = 0.0, maxUpstream = 0.0
        var times: [Double] = []
        func label(_ probs: [String: Double], _ keys: [String]) -> String {
            var best = keys[0]
            for k in keys where probs[k]! > probs[best]! { best = k }
            return best
        }
        for fixture in fixtures {
            let state = try OrderedJSON.parse(fixture.state)
            guard case .object(let members) = try OrderedJSON.parse(fixture.questions) else { fatalError("questions") }
            let questions = try members.map { (id: $0.key, question: try Decision2Question(json: $0.value)) }
            for (id, question) in questions where try manager.tokens(state: state, question: question) != fixture.ids[id]! {
                tokenMismatch += 1
                if tokenMismatch <= 3 { print("token mismatch \(fixture.id)/\(id)") }
            }
            let start = Date()
            let result = try await manager.answer(state: state, questions: questions)
            times.append(Date().timeIntervalSince(start) * 1000)
            for a in result.answers {
                decisions += 1
                let mine = Dictionary(uniqueKeysWithValues: zip(a.keys, a.probabilities))
                let py = fixture.python[a.id]!, up = fixture.upstream[a.id]!
                maxPython = max(maxPython, a.keys.map { abs(mine[$0]! - py[$0]!) }.max()!)
                maxUpstream = max(maxUpstream, a.keys.map { abs(mine[$0]! - up[$0]!) }.max()!)
                flipsPython += label(mine, a.keys) != label(py, a.keys) ? 1 : 0
                flipsUpstream += label(mine, a.keys) != label(up, a.keys) ? 1 : 0
            }
        }
        times.sort()
        print("\(manager.modelName): \(fixtures.count) requests, \(decisions) decisions")
        print("token mismatches vs upstream encoder: \(tokenMismatch)")
        print(String(format: "vs Python Core ML runtime: %d flips, max |dp| %.5f", flipsPython, maxPython))
        print(String(format: "vs upstream PyTorch: %d flips, max |dp| %.4f", flipsUpstream, maxUpstream))
        print(String(format: "request p50 %.1f ms, p95 %.1f ms", times[times.count / 2], times[times.count * 95 / 100]))
    }
}
