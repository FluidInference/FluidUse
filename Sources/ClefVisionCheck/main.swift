import CoreGraphics
import FluidUse
import Foundation
import ImageIO

/// clef-vision-0.8b checks against the Python fixtures (`Tests/FluidUseTests/Fixtures/clef-vision`).
///
///     swift run -c release ClefVisionCheck encode <fixtures dir>                    # tokens / spans / positions, no model
///     swift run -c release ClefVisionCheck parity <fixtures dir> [model dir]        # full Core ML pipeline vs reference logits
///     swift run -c release ClefVisionCheck latency [model dir] [rounds]             # warm timings per stage
@main
struct ClefVisionCheck {
    struct Fixtures: Decodable {
        struct Question: Decodable {
            let id: String
            let type: Int
            let span: [Int]
            let option_spans: [[Int]]
            let option_ids: [String]
            let logits: [Float]
            let probabilities: [Float]
        }
        struct Record: Decodable {
            let id: String
            let task: String
            let state: JSONValue
            let questions_request: [String: JSONValue]
            let images: [String]
            let input_ids: [Int]
            let image_grid_thw: [[Int]]
            let position_ids: [[Int]]
            let questions: [Question]
        }
        let image_token_id: Int
        let records: [Record]
    }

    /// Minimal JSON value for fixtures: round-trips into `Any` for the encoder.
    enum JSONValue: Decodable {
        case string(String), number(Double), bool(Bool), null, array([JSONValue]), object([String: JSONValue])
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if c.decodeNil() { self = .null } else if let b = try? c.decode(Bool.self) { self = .bool(b) }
            else if let n = try? c.decode(Double.self) { self = .number(n) } else if let s = try? c.decode(String.self) { self = .string(s) }
            else if let a = try? c.decode([JSONValue].self) { self = .array(a) } else { self = .object(try c.decode([String: JSONValue].self)) }
        }
        var any: Any {
            switch self {
            case .string(let s): return s
            case .number(let n): return n.rounded() == n && abs(n) < 1e15 ? Int(n) as Any : n
            case .bool(let b): return b
            case .null: return NSNull()
            case .array(let a): return a.map(\.any)
            case .object(let o): return o.mapValues(\.any)
            }
        }
    }

    static func questions(_ request: [String: JSONValue], order: [String]) -> [(id: String, question: ClefQuestion)] {
        order.map { id in
            guard case .object(let q) = request[id]!, case .string(let type) = q["type"]! else { fatalError("bad fixture") }
            var instructions: String?
            if case .string(let s)? = q["instructions"] { instructions = s }
            switch type {
            case "noul":
                var criteria: [String: String] = [:]
                if case .object(let c)? = q["criteria"] { for (k, v) in c { if case .string(let s) = v { criteria[k] = s } } }
                return (id, .noul(instructions: instructions, criteria: criteria))
            case "choice":
                var criteria: [String: String] = [:]
                if case .object(let c)? = q["criteria"] { for (k, v) in c { if case .string(let s) = v { criteria[k] = s } } }
                return (id, .choice(instructions: instructions, criteria: criteria))
            default:
                var criteria: [String] = []
                if case .array(let c)? = q["criteria"] { for v in c { if case .string(let s) = v { criteria.append(s) } } }
                return (id, .score(instructions: instructions, criteria: criteria))
            }
        }
    }

    static func load(_ directory: URL) throws -> Fixtures {
        try JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: directory.appendingPathComponent("fixtures.json")))
    }

    static func image(_ url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw ClefVisionError.invalidInput("cannot read \(url.lastPathComponent)") }
        return image
    }

    static func main() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard #available(macOS 15.0, *) else { print("needs macOS 15"); return }
        switch arguments.first {
        case "encode" where arguments.count == 2:
            try await encode(fixtures: URL(fileURLWithPath: arguments[1]))
        case "parity" where arguments.count >= 2:
            let model = arguments.count > 2 ? URL(fileURLWithPath: arguments[2]) : try await ClefVisionModelStore.ensure()
            try await parity(fixtures: URL(fileURLWithPath: arguments[1]), model: model)
        case "latency":
            let model = arguments.count > 1 ? URL(fileURLWithPath: arguments[1]) : try await ClefVisionModelStore.ensure()
            try await latency(model: model, rounds: arguments.count > 2 ? Int(arguments[2]) ?? 5 : 5)
        case "dump" where arguments.count == 4:
            // dump <fixtures dir> <record id> <out dir>: Swift patches + vision tokens per image, as raw float32
            let fixtures = try load(URL(fileURLWithPath: arguments[1]))
            guard let record = fixtures.records.first(where: { $0.id == arguments[2] }) else { print("no such record"); return }
            let out = URL(fileURLWithPath: arguments[3])
            try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
            let manager = try await ClefVisionManager.load(from: try await ClefVisionModelStore.ensure(buckets: []), buckets: [])
            var grids: [[Int]] = []
            for (i, name) in record.images.enumerated() {
                let r = try await manager.visionTokens(for: try image(URL(fileURLWithPath: arguments[1]).appendingPathComponent(name)))
                grids.append([r.gridH, r.gridW])
                try r.patches.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: out.appendingPathComponent("patches-\(i).f32"))
                try r.tokens.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: out.appendingPathComponent("tokens-\(i).f32"))
                print("image \(i): grid \(r.gridH)x\(r.gridW) patches \(r.patches.count / 1536) tokens \(r.tokens.count / 1024)")
            }
            try JSONSerialization.data(withJSONObject: ["grids": grids]).write(to: out.appendingPathComponent("grids.json"))
        default:
            print("usage: ClefVisionCheck encode <fixtures> | parity <fixtures> [model] | latency [model] [rounds]")
        }
    }

    @available(macOS 15.0, *)
    static func encode(fixtures directory: URL) async throws {
        let fixtures = try load(directory)
        let encoder = try ClefVisionManager.encoder(from: try await ClefVisionModelStore.ensureEncoderAssets())
        var failures = 0
        for record in fixtures.records {
            let counts = record.image_grid_thw.map { $0[1] * $0[2] / 4 }
            let encoded = try encoder.encode(
                state: record.state.any, questions: questions(record.questions_request, order: record.questions.map(\.id)),
                imageTokenCounts: counts)
            let idsMatch = encoded.inputIDs == record.input_ids
            var spansMatch = encoded.questions.count == record.questions.count
            for (q, ref) in zip(encoded.questions, record.questions) {
                spansMatch = spansMatch && q.span == ref.span[0]..<ref.span[1] && q.optionIDs == ref.option_ids
                    && q.optionSpans.map { [$0.lowerBound, $0.upperBound] } == ref.option_spans
            }
            let positions = ClefVisionHost.mropePositions(
                inputIDs: encoded.inputIDs, imageGrids: record.image_grid_thw.map { (h: $0[1], w: $0[2]) },
                imageTokenID: fixtures.image_token_id, mergeSize: 2)
            let positionsMatch = positions == record.position_ids
            if !(idsMatch && spansMatch && positionsMatch) {
                failures += 1
                if !idsMatch, let first = zip(encoded.inputIDs, record.input_ids).enumerated().first(where: { $0.element.0 != $0.element.1 }) {
                    print("  first token mismatch at \(first.offset): ours \(first.element.0) ref \(first.element.1) (lengths \(encoded.inputIDs.count) vs \(record.input_ids.count))")
                }
            }
            print("\(record.task.padding(toLength: 18, withPad: " ", startingAt: 0)) tokens \(record.input_ids.count) ids \(idsMatch ? "ok" : "DIFF") spans \(spansMatch ? "ok" : "DIFF") positions \(positionsMatch ? "ok" : "DIFF")")
        }
        print(failures == 0 ? "ENCODE PASS" : "ENCODE FAIL (\(failures))")
    }

    @available(macOS 15.0, *)
    static func parity(fixtures directory: URL, model: URL) async throws {
        let fixtures = try load(directory)
        let manager = try await ClefVisionManager.load(from: model)
        var worst: Float = 0, worstProb: Float = 0, agree = 0, total = 0
        for record in fixtures.records {
            let images = try record.images.map { try image(directory.appendingPathComponent($0)) }
            let result = try await manager.answer(
                state: record.state.any, images: images,
                questions: questions(record.questions_request, order: record.questions.map(\.id)))
            var line = "\(record.task.padding(toLength: 18, withPad: " ", startingAt: 0)) tokens \(result.inputTokens)/\(record.input_ids.count)"
            for (answer, ref) in zip(result.answers, record.questions) {
                let diff = zip(answer.logits, ref.logits).map { abs($0 - $1) }.max() ?? 0
                let pdiff = zip(answer.probabilities, ref.probabilities).map { abs($0 - $1) }.max() ?? 0
                worst = max(worst, diff); worstProb = max(worstProb, pdiff)
                let ok = answer.choice == ref.option_ids[ref.probabilities.indices.max { ref.probabilities[$0] < ref.probabilities[$1] }!]
                agree += ok ? 1 : 0; total += 1
                line += "  \(answer.questionID): |Δlogit| \(String(format: "%.3e", diff)) |Δp| \(String(format: "%.3e", pdiff)) \(ok ? "✓" : "✗")"
            }
            line += "  [vision \(Int(result.visionMilliseconds)) ms, lm \(Int(result.languageMilliseconds)) ms, head \(Int(result.headMilliseconds)) ms]"
            print(line)
        }
        print("PARITY max |Δlogit| \(String(format: "%.3e", worst)) max |Δp| \(String(format: "%.3e", worstProb)) argmax \(agree)/\(total) \(worstProb < 0.02 && agree == total ? "PASS" : "CHECK")")
    }

    @available(macOS 15.0, *)
    static func latency(model: URL, rounds: Int) async throws {
        let manager = try await ClefVisionManager.load(from: model)
        try await manager.warm()
        let fixtures = try load(URL(fileURLWithPath: "Tests/FluidUseTests/Fixtures/clef-vision"))
        for record in fixtures.records {
            let images = try record.images.map { try image(URL(fileURLWithPath: "Tests/FluidUseTests/Fixtures/clef-vision/\($0)")) }
            var vision: [Double] = [], lm: [Double] = [], head: [Double] = []
            for _ in 0..<rounds {
                let r = try await manager.answer(
                    state: record.state.any, images: images,
                    questions: questions(record.questions_request, order: record.questions.map(\.id)))
                vision.append(r.visionMilliseconds); lm.append(r.languageMilliseconds); head.append(r.headMilliseconds)
            }
            func median(_ v: [Double]) -> Int { Int(v.sorted()[v.count / 2]) }
            print("\(record.task.padding(toLength: 18, withPad: " ", startingAt: 0)) \(images.count) image(s) \(record.input_ids.count) tokens: vision \(median(vision)) ms  lm \(median(lm)) ms  head \(median(head)) ms  total \(median(vision) + median(lm) + median(head)) ms")
        }
    }
}
