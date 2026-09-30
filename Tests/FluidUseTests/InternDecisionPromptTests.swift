import XCTest

@testable import FluidUse

/// Prompt rendering against the checkpoint's own compiler and chat template (`Fixtures/intern-decision-prompts.json`
/// was written by the reference `inference.py`); the Core ML path is covered by `InternDecisionCheck parity`.
final class InternDecisionPromptTests: XCTestCase {
    struct Fixture {
        let request: JSONValue
        let prompt: String
        let fields: [(name: String, labels: [String])]
    }

    static func fixtures() throws -> [Fixture] {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "intern-decision-prompts", withExtension: "json", subdirectory: "Fixtures"))
        guard case .array(let items) = try JSONValue.parse(String(contentsOf: url, encoding: .utf8)) else {
            throw XCTSkip("fixture is not a list")
        }
        return try items.map { item in
            guard case .object(let members) = item, let request = members.first(where: { $0.key == "request" })?.value,
                case .string(let prompt)? = members.first(where: { $0.key == "prompt" })?.value,
                case .array(let fields)? = members.first(where: { $0.key == "fields" })?.value
            else { throw XCTSkip("malformed fixture") }
            let parsed = fields.map { field -> (name: String, labels: [String]) in
                guard case .object(let m) = field,
                    case .string(let name)? = m.first(where: { $0.key == "name" })?.value,
                    case .array(let labels)? = m.first(where: { $0.key == "labels" })?.value
                else { return ("", []) }
                return (name, labels.compactMap { if case .string(let s) = $0 { s } else { nil } })
            }
            return Fixture(request: request, prompt: prompt, fields: parsed)
        }
    }

    /// A manager with no Core ML model: enough for `prompt(state:questions:)`.
    static func promptOnly() throws -> InternDecisionManager {
        let system =
            "You are a careful decision assistant. Use the state and decision schema in the user message to make the requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its listed options and return one valid JSON object mapping each field name to its chosen symbol. Use the field names and symbols exactly as given. Do not include explanations, Markdown, or extra text."
        let tokenizerURL = FileManager.default.temporaryDirectory.appendingPathComponent("empty-tokenizer.json")
        try #"{"model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#.write(
            to: tokenizerURL, atomically: true, encoding: .utf8)
        return InternDecisionManager(
            tokenizer: try QwenBPETokenizer(tokenizerJsonURL: tokenizerURL), systemPrompt: system, temperature: 2.7478,
            symbols: Array("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"), buckets: [],
            models: InternDecisionManager.Models(computeUnits: .cpuOnly), embeddings: Data(), hiddenSize: 1024,
            rotaryDim: 64, ropeTheta: 1e7, padID: 248044, markerID: 248077)
    }

    func testPromptsMatchReferenceCompiler() throws {
        let manager = try Self.promptOnly()
        for (index, fixture) in try Self.fixtures().enumerated() {
            let (state, questions) = try InternDecisionPromptTests.request(fixture.request)
            let prompt = try manager.prompt(state: state, questions: questions)
            if prompt != fixture.prompt {
                let common = zip(prompt, fixture.prompt).prefix { $0 == $1 }.count
                XCTFail(
                    "fixture \(index) differs at \(common): swift \(prompt.dropFirst(common).prefix(60).debugDescription) ref \(fixture.prompt.dropFirst(common).prefix(60).debugDescription)"
                )
            }
            XCTAssertEqual(questions.map(\.name), fixture.fields.map(\.name))
            XCTAssertEqual(questions.map { $0.question.options.map(\.label) }, fixture.fields.map(\.labels))
        }
    }

    static func request(_ value: JSONValue) throws -> (JSONValue, [(name: String, question: InternDecisionQuestion)]) {
        guard case .object(let members) = value, let state = members.first(where: { $0.key == "state" })?.value,
            case .object(let questions)? = members.first(where: { $0.key == "questions" })?.value
        else { throw XCTSkip("record needs state and questions") }
        return (state, try questions.map { (name: $0.key, question: try InternDecisionQuestion(json: $0.value)) })
    }

    func testPythonDumpMatchesJsonDumps() {
        let value: JSONValue = [
            "s": "a \"q\" \\ \t\n\u{01} é 🎮", "i": 3, "f": 32.5, "one": 1.0, "big": 1e16, "tiny": 1.5e-07,
            "t": true, "n": nil, "eo": [:], "ea": [], "nested": ["z": [1, ["y": "x"]]],
        ]
        let expected = """
            {
              "s": "a \\"q\\" \\\\ \\t\\n\\u0001 é 🎮",
              "i": 3,
              "f": 32.5,
              "one": 1.0,
              "big": 1e+16,
              "tiny": 1.5e-07,
              "t": true,
              "n": null,
              "eo": {},
              "ea": [],
              "nested": {
                "z": [
                  1,
                  {
                    "y": "x"
                  }
                ]
              }
            }
            """
        XCTAssertEqual(value.pythonDump(indent: 2), expected)
        XCTAssertEqual(JSONValue.string("plain").pythonDump(indent: 2), "\"plain\"")
    }

    func testParseKeepsKeyOrderAndRoundTrips() throws {
        let text = "{\"b\": 1, \"a\": [true, null, 2.5, \"x\\u00e9\\ud83c\\udfae\"], \"c\": {}}"
        let value = try JSONValue.parse(text)
        guard case .object(let members) = value else { return XCTFail("not an object") }
        XCTAssertEqual(members.map(\.key), ["b", "a", "c"])
        XCTAssertEqual(members[1].value, [true, nil, 2.5, "xé🎮"])
        XCTAssertEqual(try JSONValue.parse(value.pythonDump(indent: 2)), value)
    }

    func testNoulDefaultsAndScoreLabels() {
        XCTAssertEqual(InternDecisionQuestion.noul("Fine?").options.map(\.label), ["no", "yes"])
        XCTAssertEqual(
            InternDecisionQuestion.score("How many?", levels: ["Zero", "One"]).options.map(\.label), ["0", "1"])
        XCTAssertThrowsError(
            try InternDecisionManager.validated(
                [("bad", .scoreKeyed("x", levels: [(label: "high", description: "h")]))], symbolCount: 62))
        XCTAssertThrowsError(try InternDecisionManager.validated([], symbolCount: 62))
    }

    func testBestIndexBreaksTiesBySmallerLabel() {
        let answer = InternDecisionAnswer(
            field: "f", labels: ["b", "a", "c"], probabilities: [0.4, 0.4, 0.2], rawProbabilities: [0.4, 0.4, 0.2])
        XCTAssertEqual(answer.decision, "a")
        XCTAssertEqual(answer.expectedScore, nil)
        let score = InternDecisionAnswer(
            field: "s", labels: ["0", "1", "2"], probabilities: [0.25, 0.5, 0.25], rawProbabilities: [0.25, 0.5, 0.25])
        XCTAssertEqual(score.expectedScore ?? -1, 1.0, accuracy: 1e-6)
    }
}
