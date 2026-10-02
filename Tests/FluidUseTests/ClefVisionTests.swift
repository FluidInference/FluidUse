import XCTest

@testable import FluidUse

/// clef-vision-0.8b host pieces that need no model: Clef's JSON rendering, smart-resize, M-RoPE positions, and
/// the record encoder against the Python fixtures (tokenizer from the pinned bundle; skipped when offline).
final class ClefVisionTests: XCTestCase {
    static let fixturesURL = Bundle.module.resourceURL!.appendingPathComponent("Fixtures/clef-vision")

    struct Fixtures: Decodable {
        struct Question: Decodable {
            let id: String
            let span: [Int]
            let option_spans: [[Int]]
            let option_ids: [String]
        }
        struct Record: Decodable {
            let task: String
            let state: AnyJSON
            let questions_request: [String: AnyJSON]
            let input_ids: [Int]
            let image_grid_thw: [[Int]]
            let position_ids: [[Int]]
            let questions: [Question]
        }
        let image_token_id: Int
        let records: [Record]
    }

    enum AnyJSON: Decodable {
        case string(String), number(Double), bool(Bool), null, array([AnyJSON]), object([String: AnyJSON])
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if c.decodeNil() { self = .null } else if let b = try? c.decode(Bool.self) { self = .bool(b) }
            else if let n = try? c.decode(Double.self) { self = .number(n) } else if let s = try? c.decode(String.self) { self = .string(s) }
            else if let a = try? c.decode([AnyJSON].self) { self = .array(a) } else { self = .object(try c.decode([String: AnyJSON].self)) }
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
        var string: String? { if case .string(let s) = self { return s }; return nil }
    }

    func testClefJSONMatchesPythonDumps() throws {
        // json.dumps(..., ensure_ascii=False, separators=(",", ":"), sort_keys=True); strings render bare
        XCTAssertEqual(ClefJSON.render("plain text"), "plain text")
        XCTAssertEqual(ClefJSON.render(["task": "x", "a": 1]), #"{"a":1,"task":"x"}"#)
        XCTAssertEqual(ClefJSON.render(["total": 1250.0, "ok": true, "n": NSNull()]), #"{"n":null,"ok":true,"total":1250.0}"#)
        XCTAssertEqual(ClefJSON.render(["list": ["é", "a\"b", "x\ny"]]), "{\"list\":[\"é\",\"a\\\"b\",\"x\\ny\"]}")
        // values parsed by JSONSerialization arrive as NSNumber: 0/1 must stay integers, 1250.0 a float, bools bools
        let parsed = try JSONSerialization.jsonObject(
            with: Data(#"{"zero":0,"one":1,"total":1250.0,"flag":true,"neg":-0.0,"mixed":[1,2.0],"big":12345678901234567890,"tiny":1e-05}"#.utf8))
        XCTAssertEqual(
            ClefJSON.render(parsed),
            #"{"big":12345678901234567890,"flag":true,"mixed":[1,2.0],"neg":-0.0,"one":1,"tiny":1e-05,"total":1250.0,"zero":0}"#)
    }

    func testEmptyOptionsAreRejected() throws {
        XCTAssertTrue(ClefQuestion.choice(instructions: nil, criteria: [:]).options().isEmpty)
        XCTAssertEqual(ClefQuestion.noul(instructions: nil).options().map(\.id), ["true", "false"])
        XCTAssertEqual(ClefQuestion.score(instructions: nil, criteria: ["a", "b", "c"]).options().map(\.id), ["0", "1", "2"])
    }

    func testSmartResizeKeepsBudgetAndFactor() {
        // 480x640 over the 448² cap: beta = sqrt(307200/200704) -> floor(480/beta/32)*32 x floor(640/beta/32)*32
        let (h, w) = ClefImagePreprocessor.smartResize(height: 480, width: 640, factor: 32, minPixels: 16384, maxPixels: 200_704)
        XCTAssertEqual([h, w], [384, 512])
        // 500x375 (Oxford Pets) rounds to 512x384 = 196608 <= cap, giving the fixtures' 32x24 patch grid
        let pets = ClefImagePreprocessor.smartResize(height: 500, width: 375, factor: 32, minPixels: 16384, maxPixels: 200_704)
        XCTAssertEqual([pets.0, pets.1], [512, 384])
        // tiny image is scaled up to the minimum
        let (sh, sw) = ClefImagePreprocessor.smartResize(height: 40, width: 60, factor: 32, minPixels: 16384, maxPixels: 200_704)
        XCTAssertGreaterThanOrEqual(sh * sw, 16384)
    }

    func testMRoPEPositionsForTextThenImage() {
        // 2 text tokens, a 2x4 merged image (grid 4x8 patches), 1 text token
        let ids = [10, 11] + Array(repeating: 99, count: 8) + [12]
        let positions = ClefVisionHost.mropePositions(inputIDs: ids, imageGrids: [(h: 4, w: 8)], imageTokenID: 99, mergeSize: 2)
        XCTAssertEqual(positions[0].prefix(2).map { $0 }, [0, 1])
        XCTAssertEqual(Array(positions[0][2..<10]), Array(repeating: 2, count: 8))  // t = cursor
        XCTAssertEqual(Array(positions[1][2..<10]), [2, 2, 2, 2, 3, 3, 3, 3])  // h = cursor + row
        XCTAssertEqual(Array(positions[2][2..<10]), [2, 3, 4, 5, 2, 3, 4, 5])  // w = cursor + col
        XCTAssertEqual([positions[0][10], positions[1][10], positions[2][10]], [6, 6, 6])  // cursor += max(2, 4)
    }

    func testFixturePositionsMatchPython() throws {
        let fixtures = try JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: Self.fixturesURL.appendingPathComponent("fixtures.json")))
        for record in fixtures.records {
            let positions = ClefVisionHost.mropePositions(
                inputIDs: record.input_ids, imageGrids: record.image_grid_thw.map { (h: $0[1], w: $0[2]) },
                imageTokenID: fixtures.image_token_id, mergeSize: 2)
            XCTAssertEqual(positions, record.position_ids, record.task)
        }
    }

    /// Record encoding against Clef's own `encode_record` output. Fetches only the tokenizer and manifest (a few MB).
    func testRecordEncoderMatchesClef() async throws {
        guard #available(macOS 15.0, *) else { throw XCTSkip("needs macOS 15") }
        let directory: URL
        do { directory = try await ClefVisionModelStore.ensureEncoderAssets() } catch { throw XCTSkip("tokenizer unavailable: \(error)") }
        let manager = try ClefVisionManager.encoder(from: directory)
        let fixtures = try JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: Self.fixturesURL.appendingPathComponent("fixtures.json")))
        for record in fixtures.records {
            let questions: [(id: String, question: ClefQuestion)] = record.questions.map { q in
                guard case .object(let spec) = record.questions_request[q.id]! else { fatalError() }
                let instructions = spec["instructions"]?.string
                switch spec["type"]!.string! {
                case "noul":
                    var criteria: [String: String] = [:]
                    if case .object(let c)? = spec["criteria"] { criteria = c.compactMapValues(\.string) }
                    return (q.id, .noul(instructions: instructions, criteria: criteria))
                case "choice":
                    guard case .object(let c)? = spec["criteria"] else { fatalError() }
                    return (q.id, .choice(instructions: instructions, criteria: c.compactMapValues(\.string)))
                default:
                    guard case .array(let c)? = spec["criteria"] else { fatalError() }
                    return (q.id, .score(instructions: instructions, criteria: c.compactMap(\.string)))
                }
            }
            let encoded = try manager.encode(
                state: record.state.any, questions: questions, imageTokenCounts: record.image_grid_thw.map { $0[1] * $0[2] / 4 })
            XCTAssertEqual(encoded.inputIDs, record.input_ids, "\(record.task) token ids")
            for (q, ref) in zip(encoded.questions, record.questions) {
                XCTAssertEqual([q.span.lowerBound, q.span.upperBound], ref.span, "\(record.task) \(q.id) span")
                XCTAssertEqual(q.optionIDs, ref.option_ids, "\(record.task) \(q.id) options")
                XCTAssertEqual(q.optionSpans.map { [$0.lowerBound, $0.upperBound] }, ref.option_spans, "\(record.task) \(q.id) option spans")
            }
        }
    }
}
