import FluidUse
import Foundation

/// Vela 2.0 0.3B parity against the Python engine.
///
///     swift run -c release Vela2Check parity <model directory> <fixtures03.json> [ane]
///
/// Fixtures (vela2/fixtures03.py): guardrail-style requests with the engine's token ids / offsets, encoder sequences,
/// and calibrated answers from the Python engine on the same Core ML encoder (GPU).
@main
struct Vela2Check {
    struct Fixture: Decodable {
        struct Part: Decodable { let role: String; let text: String }
        struct Question: Decodable {
            let id: String; let type: String; let text: String; let over: String
            let options: [String: String]?; let labels: [String: String]?; let abstain: Bool?
        }
        struct Request: Decodable { let parts: [Part]; let questions: [Question] }
        struct Tokens: Decodable { let ids: [Int]; let offsets: [[Int]] }
        struct Row: Decodable { let ids: [Int] }
        struct SpanOut: Decodable { let start: Int; let end: Int; let label: String; let probability: Double; let text: String }
        struct Answer: Decodable {
            let id: String; let type: String; let answer: String?; let probabilities: [String: Double]?
            let spans: [SpanOut]?; let threshold: Double?
        }
        let request: Request
        let tokens: [String: Tokens]
        let rows: [Row]
        let answers: [Answer]
    }

    static func main() async throws {
        let a = Array(CommandLine.arguments.dropFirst())
        guard #available(macOS 15.0, *), a.count >= 3, a[0] == "parity" else {
            fputs("usage: Vela2Check parity <model dir> <fixtures03.json> [ane]\n", stderr)
            exit(2)
        }
        // Option order matters: decode the fixture's dicts preserving key order from the raw JSON.
        let data = try Data(contentsOf: URL(fileURLWithPath: a[2]))
        let fixtures = try JSONDecoder().decode([Fixture].self, from: data)
        let raw = try OrderedJSON.parse(String(decoding: data, as: UTF8.self))
        let manager = try await Vela2Manager.load(from: URL(fileURLWithPath: a[1]), aneMaxLength: a.count > 3 ? nil : 0)
        try await manager.warm()
        var tokenBad = 0, seqBad = 0, choiceBad = 0, spanBad = 0, choices = 0, spanSets = 0
        var maxDp = 0.0, maxSpanDp = 0.0
        var enc: [Double] = [], total: [Double] = []
        guard case .array(let rawCases) = raw else { fatalError() }
        for (fixture, rawCase) in zip(fixtures, rawCases) {
            let parts = fixture.request.parts.map { Vela2Part($0.role, $0.text) }
            for p in parts {
                let (ids, offs) = manager.tokenize(p.text)
                let want = fixture.tokens[p.role]!
                if ids != want.ids || offs.map({ [$0.0, $0.1] }) != want.offsets {
                    tokenBad += 1
                    print("token mismatch [\(p.role)]: \(p.text.prefix(60))")
                }
            }
            // ordered options / labels from the raw JSON
            func ordered(_ q: OrderedJSON, _ key: String) -> [(name: String, description: String)] {
                guard case .object(let m) = q, case .object(let o)? = m.first(where: { $0.key == key })?.value else { return [] }
                return o.map { ($0.key, { if case .string(let s) = $0 { s } else { "" } }($0.value)) }
            }
            guard case .object(let caseMembers) = rawCase, case .object(let req)? = caseMembers.first(where: { $0.key == "request" })?.value,
                case .array(let rawQs)? = req.first(where: { $0.key == "questions" })?.value
            else { fatalError() }
            let questions: [Vela2Question] = zip(fixture.request.questions, rawQs).map { q, rq in
                q.type == "span"
                    ? .span(id: q.id, text: q.text, labels: ordered(rq, "labels"), over: q.over)
                    : .choice(id: q.id, text: q.text, options: ordered(rq, "options"), over: [q.over], abstain: q.abstain ?? true)
            }
            let seqs = try manager.sequences(parts: parts, questions: questions)
            if seqs != fixture.rows.map(\.ids) {
                seqBad += 1
                print("sequence mismatch: \(parts[0].text.prefix(60))")
            }
            let r = try await manager.predict(parts: parts, questions: questions)
            enc.append(r.encoderMs)
            total.append(r.totalMs)
            for want in fixture.answers {
                if want.type == "span" {
                    spanSets += 1
                    let got = r[span: want.id]!.spans
                    let same = got.count == want.spans!.count
                        && zip(got, want.spans!).allSatisfy { $0.start == $1.start && $0.end == $1.end && $0.label == $1.label }
                    if !same {
                        spanBad += 1
                        print("span diff \(want.id): \(got.map { "\($0.label):\($0.text)" }) vs \(want.spans!.map { "\($0.label):\($0.text)" })")
                    } else {
                        for (g, w) in zip(got, want.spans!) { maxSpanDp = max(maxSpanDp, abs(g.probability - w.probability)) }
                    }
                } else {
                    choices += 1
                    let got = r[choice: want.id]!
                    if got.answer != want.answer { choiceBad += 1; print("choice diff \(want.id): \(got.answer) vs \(want.answer!)") }
                    for (k, v) in want.probabilities! { maxDp = max(maxDp, abs(got.probability(k)! - v)) }
                }
            }
        }
        enc.sort(); total.sort()
        print("\(manager.modelName): \(fixtures.count) requests (\(a.count > 3 ? "ANE <= 128 tokens, GPU above" : "GPU only"))")
        print("token mismatches \(tokenBad), sequence mismatches \(seqBad)")
        print(String(format: "choices: %d / %d differ, max |dp| %.5f", choiceBad, choices, maxDp))
        print(String(format: "span sets: %d / %d differ, max |d span p| %.5f", spanBad, spanSets, maxSpanDp))
        print(String(format: "encoder p50 %.2f ms, full request p50 %.2f ms", enc[enc.count / 2], total[total.count / 2]))
    }
}
