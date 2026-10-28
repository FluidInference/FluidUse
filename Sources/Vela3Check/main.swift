import FluidUse
import Foundation

/// Vela 3.0 parity against the Python runtime of FluidInference/vela-3.0-coreml (`vela3_coreml.py`, same Core ML
/// encoders on the GPU, the release's own rendering / thresholds / span decoder), plus a timing run.
///
///     swift run -c release Vela3Check parity <model directory> <fixtures.json>
///     swift run -c release Vela3Check download            # fetch the pinned snapshot into the FluidUse cache
///
/// Fixtures (vela-3.0-coreml/conversion/fixtures.py): token ids and offsets of varied texts, encoder sequences and
/// answers of typed decisions, PII and unsupported-claim calls.
@main
struct Vela3Check {
    struct Fixtures: Decodable {
        struct Tok: Decodable { let text: String; let ids: [Int]; let offsets: [[Int]] }
        struct Decision: Decodable {
            let state: String; let id: String; let type: String; let instructions: String; let options: [[String]]
            let ids: [Int]; let names: [String]; let probabilities: [Double]
        }
        struct Span: Decodable { let label: String; let start: Int; let end: Int }
        struct SpanCase: Decodable {
            let kind: String; let text: String?; let context: String?; let answer: String?
            let ids: [Int]; let spans: [Span]; let threshold: Double
        }
        let tokens: [Tok]
        let decisions: [Decision]
        let spans: [SpanCase]
    }

    static func main() async throws {
        let a = Array(CommandLine.arguments.dropFirst())
        guard #available(macOS 15.0, *) else { exit(2) }
        if a.first == "download" {
            let dir = try await Vela3ModelStore.ensure { file, bytes in print(bytes == 0 ? "downloading \(file)" : "  \(bytes / 1_048_576) MB") }
            print(dir.path)
            return
        }
        guard a.count >= 3, a[0] == "parity" else {
            fputs("usage: Vela3Check parity <model dir> <fixtures.json> | Vela3Check download\n", stderr)
            exit(2)
        }
        let fx = try JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: URL(fileURLWithPath: a[2])))
        let t0 = Date()
        let m = try await Vela3Manager.load(from: URL(fileURLWithPath: a[1]))
        try await m.warm()
        print(String(format: "loaded + warmed in %.1f s", Date().timeIntervalSince(t0)))

        var bad = 0
        for t in fx.tokens {
            let (ids, offs) = m.tokenize(t.text)
            if ids != t.ids || offs.map({ [$0.0, $0.1] }) != t.offsets {
                bad += 1
                print("token mismatch: \(t.text.debugDescription.prefix(80))")
            }
        }
        print("tokens: \(fx.tokens.count) texts, \(bad) mismatches")

        var seqBad = 0, flips = 0, maxDp = 0.0
        var decisionMs: [Double] = []
        for d in fx.decisions {
            let q: Vela3Question =
                d.type == "noul"
                ? .yesNo(id: d.id, instructions: d.instructions)
                : .choice(id: d.id, instructions: d.instructions, options: d.options.map { ($0[0], $0[1]) })
            if m.decisionSequence(state: d.state, question: q) != d.ids { seqBad += 1 }
            let r = try await m.decide(state: d.state, questions: [q])
            decisionMs.append(r.timing.totalMs)
            let got = r.decisions[0]
            let want = zip(d.names, d.probabilities).max { $0.1 < $1.1 }!.0
            if got.answer != want { flips += 1 }
            for (n, p) in zip(d.names, d.probabilities) { maxDp = max(maxDp, abs((got.probability(n) ?? -1) - p)) }
        }
        print(String(format: "decisions: %d questions, %d sequence mismatches, %d answers differ, max |dp| %.5f", fx.decisions.count, seqBad, flips, maxDp))

        var spanSeqBad = 0, spanBad = 0, nSpans = 0
        var spanMs: [Double] = []
        for c in fx.spans {
            let (ids, r): ([Int], Vela3SpanResult)
            if c.kind == "pii" {
                ids = m.spanSequence(pii: c.text!)
                r = try await m.pii(c.text!)
            } else {
                ids = m.spanSequence(context: c.context!, answer: c.answer!)
                r = try await m.unsupported(context: c.context!, answer: c.answer!)
            }
            spanMs.append(r.timing.totalMs)
            if ids != c.ids { spanSeqBad += 1 }
            let got = Set(r.spans.map { "\($0.label)|\($0.start)|\($0.end)" })
            let want = Set(c.spans.map { "\($0.label)|\($0.start)|\($0.end)" })
            nSpans += want.count
            if got != want || abs(r.threshold - c.threshold) > 1e-9 {
                spanBad += 1
                print("span mismatch [\(c.kind)]: want \(want.subtracting(got).sorted()) got extra \(got.subtracting(want).sorted())")
            }
        }
        print("spans: \(fx.spans.count) calls (\(nSpans) spans), \(spanSeqBad) sequence mismatches, \(spanBad) span sets differ")
        func median(_ x: [Double]) -> Double { x.isEmpty ? 0 : x.sorted()[x.count / 2] }
        print(String(format: "median call: decision %.1f ms, span %.1f ms (GPU, end to end)", median(decisionMs), median(spanMs)))
        if bad + seqBad + flips + spanSeqBad + spanBad > 0 { exit(1) }
    }
}
