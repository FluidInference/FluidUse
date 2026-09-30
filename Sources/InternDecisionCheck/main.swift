import FluidUse
import Foundation

/// Intern-Decision-0.8B checks against the checkpoint's own fp32 engine.
///
///     swift run -c release InternDecisionCheck parity <model directory> <parity.json>
///     swift run -c release InternDecisionCheck bench <model directory> [iterations]
///
/// `parity.json` holds records `{request: {state, questions}, ids, positions, answers: {field: {labels,
/// probabilities, decision}}}` written by the mobius fixtures script (reference probabilities after temperature).
@main
struct InternDecisionCheck {
    static func main() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        switch arguments.first {
        case "parity" where arguments.count == 3:
            try await parity(directory: URL(fileURLWithPath: arguments[1]), records: URL(fileURLWithPath: arguments[2]))
        case "bench" where arguments.count >= 2:
            try await bench(
                directory: URL(fileURLWithPath: arguments[1]), iterations: Int(arguments.dropFirst(2).first ?? "") ?? 30
            )
        default:
            fputs("usage: InternDecisionCheck parity <dir> <parity.json> | bench <dir> [iterations]\n", stderr)
            exit(2)
        }
    }

    static func request(
        _ value: OrderedJSON
    ) throws -> (OrderedJSON, [(name: String, question: InternDecisionQuestion)]) {
        guard case .object(let members) = value, let state = members.first(where: { $0.key == "state" })?.value,
            case .object(let questions)? = members.first(where: { $0.key == "questions" })?.value
        else { throw InternDecisionError.invalidRequest("record needs state and questions") }
        return (state, try questions.map { (name: $0.key, question: try InternDecisionQuestion(json: $0.value)) })
    }

    static func parity(directory: URL, records: URL) async throws {
        let manager = try await InternDecisionManager.load(from: directory)
        guard case .array(let items) = try OrderedJSON.parse(String(contentsOf: records, encoding: .utf8)) else {
            throw InternDecisionError.invalidRequest("parity file must be a list")
        }
        var fields = 0
        var flips = 0
        var tokenMismatches = 0
        var worst: Float = 0
        var skipped = 0
        let start = Date()
        for item in items {
            guard case .object(let members) = item else { continue }
            guard let record = members.first(where: { $0.key == "request" })?.value else {
                print("record without request, skipped")
                continue
            }
            let (state, questions) = try request(record)
            let (ids, _) = try manager.encode(state: state, questions: questions)
            if case .array(let expectedIDs)? = members.first(where: { $0.key == "ids" })?.value {
                let expected = expectedIDs.compactMap { if case .integer(let n) = $0 { n } else { nil } }
                if expected != ids {
                    tokenMismatches += 1
                    let prefix = zip(ids, expected).prefix { $0 == $1 }.count
                    print(
                        "TOKEN MISMATCH at \(prefix): swift \(Array(ids.dropFirst(prefix).prefix(6))) ref \(Array(expected.dropFirst(prefix).prefix(6)))"
                    )
                }
            }
            if ids.count > manager.maxTokens {
                skipped += 1
                continue
            }
            let result: InternDecisionResult
            do { result = try await manager.decide(state: state, questions: questions) } catch InternDecisionError
                .tooLong
            {
                skipped += 1
                continue
            }
            guard case .object(let answers)? = members.first(where: { $0.key == "answers" })?.value else { continue }
            for answer in result.answers {
                guard case .object(let expected)? = answers.first(where: { $0.key == answer.field })?.value,
                    case .array(let probabilities)? = expected.first(where: { $0.key == "probabilities" })?.value,
                    case .string(let decision)? = expected.first(where: { $0.key == "decision" })?.value
                else { continue }
                let reference = probabilities.map { value -> Float in
                    switch value {
                    case .number(let x): Float(x)
                    case .integer(let n): Float(n)
                    default: .nan
                    }
                }
                fields += 1
                worst = max(worst, zip(reference, answer.probabilities).map { abs($0 - $1) }.max() ?? 0)
                if decision != answer.decision {
                    flips += 1
                    print("FLIP \(answer.field): swift \(answer.decision) ref \(decision)")
                }
            }
        }
        print(
            String(
                format: "records %d (skipped %d), fields %d, token mismatches %d, flips %d, max |dp| %.4f, %.1f s",
                items.count, skipped, fields, tokenMismatches, flips, worst, Date().timeIntervalSince(start)))
        exit(flips == 0 && tokenMismatches == 0 ? 0 : 1)
    }

    /// The model card's request shape: about 320 tokens, one choice, one yes/no, one score question.
    static func bench(directory: URL, iterations: Int) async throws {
        let manager = try await InternDecisionManager.load(from: directory)
        let state: OrderedJSON = [
            "channel": "email",
            "message":
                "Charged twice for my annual renewal ($240 each). Emailed last week, no reply. Refund the duplicate before Friday.",
        ]
        let questions: [(name: String, question: InternDecisionQuestion)] = [
            (
                "team",
                .choice(
                    "Which team should handle this ticket?",
                    options: [("billing", "Payments and refunds."), ("technical", "Bugs."), ("sales", "Renewals.")])
            ),
            ("frustrated", .noul("Is the customer frustrated?")),
            ("urgency", .score("How urgent is this ticket?", levels: ["Low", "Medium", "High"])),
        ]
        var times: [Double] = []
        var last: InternDecisionResult?
        for i in 0..<(iterations + 5) {
            let start = Date()
            last = try await manager.decide(state: state, questions: questions)
            if i >= 5 { times.append(Date().timeIntervalSince(start) * 1000) }
        }
        times.sort()
        guard let result = last, !times.isEmpty else {
            fputs("iterations must be at least 1\n", stderr)
            exit(2)
        }
        print("tokens \(result.inputTokens), bucket \(result.bucketLength)")
        for answer in result.answers {
            print("  \(answer.field): \(answer.decision) (\(String(format: "%.3f", answer.confidence)))")
        }
        print(
            String(
                format: "p50 %.1f ms, p95 %.1f ms over %d calls", times[times.count / 2],
                times[min(times.count - 1, Int(Double(times.count) * 0.95))], times.count))
    }
}
