import Foundation
import XCTest

@testable import FluidUse

/// Real-model checks through the full path: pinned download/verification, `uv sync`, worker start, typed request.
/// Set `FLUIDUSE_PUBLISHED_COREML_CACHE` to a Models directory (the first run downloads several GB and needs `uv`).
/// NanoJev additionally needs `FLUIDUSE_NANOJEV_DIR` (local Mobius conversion) and `FLUIDUSE_NANOJEV_PYTHON`.
final class PublishedCoreMLIntegrationTests: XCTestCase {
    private let billing = SystemOneQuestion.choice(
        "team", "Which team should handle this request?",
        options: [DecisionOption("Billing", "Charges and refunds"), DecisionOption("Support", "Technical problems")])
    private let state = "Please refund the duplicate charge. I need this fixed today."

    private func session(_ model: PublishedCoreMLModel, precision: String? = nil) async throws -> PublishedCoreMLManager
    {
        guard let path = ProcessInfo.processInfo.environment["FLUIDUSE_PUBLISHED_COREML_CACHE"], !path.isEmpty else {
            throw XCTSkip("Set FLUIDUSE_PUBLISHED_COREML_CACHE to run published Core ML bridge tests")
        }
        return try await PublishedCoreMLManager.load(
            model: model, cacheDirectory: URL(fileURLWithPath: path, isDirectory: true),
            configuration: .init(precision: precision))
    }

    private func choice(_ answer: SystemOneAnswer?) -> (selected: String, keys: [String])? {
        guard case .choice(let selected, _, let probabilities) = answer else { return nil }
        return (selected, probabilities.map(\.key))
    }

    func testKev05KeepsOptionOrder() async throws {
        let manager = try await session(.kev05)
        let forward = try await manager.evaluate(SystemOneRequest(state: state, questions: [billing]))
        XCTAssertEqual(choice(forward["team"])?.selected, "Billing")
        let reversed = SystemOneQuestion.choice(
            "team", "Which team should handle this request?",
            options: [
                DecisionOption("Support", "Technical problems"), DecisionOption("Billing", "Charges and refunds"),
            ])
        let answer = try await manager.evaluate(SystemOneRequest(state: state, questions: [reversed]))
        XCTAssertEqual(choice(answer["team"])?.keys, ["Support", "Billing"])
        XCTAssertEqual(choice(answer["team"])?.selected, "Billing")
    }

    func testKev06W8ServesRepeatedRequests() async throws {
        let manager = try await session(.kev06, precision: "w8")
        for _ in 0..<2 {
            let response = try await manager.evaluate(SystemOneRequest(state: state, questions: [billing]))
            XCTAssertEqual(choice(response["team"])?.selected, "Billing")
            XCTAssertEqual(response.model, "kev-0.6b")
        }
    }

    private func systemOneExample() -> SystemOneRequest {
        SystemOneRequest(
            state: .object([.init("message", .string(state))]),
            questions: [
                .noul("refund_requested", "Does the customer explicitly request a refund?"),
                billing,
                .score(
                    "urgency", "How urgent is the request?", levels: ["No deadline", "Needed soon", "Needed today"]),
            ])
    }

    func testKaiAnswersAllQuestionTypes() async throws {
        let response = try await session(.kai).evaluate(systemOneExample())
        XCTAssertEqual(response.answers.map(\.id), ["refund_requested", "team", "urgency"])
        guard case .noul(let refund) = response["refund_requested"] else { return XCTFail("No noul answer") }
        XCTAssertGreaterThan(refund, 0.5)
        XCTAssertEqual(choice(response["team"])?.selected, "Billing")
        guard case .score(_, _, let levels) = response["urgency"] else { return XCTFail("No score answer") }
        XCTAssertEqual(levels.count, 3)
    }

    func testLexW8AnswersAllQuestionTypes() async throws {
        let response = try await session(.lex, precision: "w8").evaluate(systemOneExample())
        XCTAssertEqual(response.model, "Decision-1.0-Lex")
        XCTAssertEqual(choice(response["team"])?.selected, "Billing")
        guard case .score(let expected, _, _) = response["urgency"] else { return XCTFail("No score answer") }
        XCTAssertGreaterThan(expected, 1.5)
    }

    func testLFMChoosesSchemaValues() async throws {
        let decision = try await session(.lfm350).constrained(
            context: "Our company invoice contains a duplicate charge, and payment is due tomorrow.",
            fields: [.oneOf("route", ["billing", "technical", "sales"]), .boolean("urgent")])
        XCTAssertEqual(decision["route"], .string("billing"))
        XCTAssertEqual(decision["urgent"], .bool(true))
        XCTAssertEqual(decision.candidates.map { $0.scores.count }, [3, 2])
    }

    func testJeffW8ScoresLabels() async throws {
        let result = try await session(.jeff, precision: "w8").classify(
            text: "The invoice was charged twice and the customer asks for a refund.",
            labels: ["billing: invoice or payment issue", "support: technical product issue"],
            name: "Choose the correct support queue")
        XCTAssertEqual(result.selectedIndex, 0)
        XCTAssertGreaterThan(result.probabilities[0], 0.9)
    }

    func testNanoJevLocalConversion() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let directory = environment["FLUIDUSE_NANOJEV_DIR"], let python = environment["FLUIDUSE_NANOJEV_PYTHON"]
        else { throw XCTSkip("Set FLUIDUSE_NANOJEV_DIR and FLUIDUSE_NANOJEV_PYTHON to run NanoJev") }
        let manager = try await PublishedCoreMLManager.start(
            model: .nanojev, from: URL(fileURLWithPath: directory, isDirectory: true),
            python: URL(fileURLWithPath: python))
        let decision = try await manager.decide(
            state: "The invoice says paid and shows a zero balance.",
            question: NanoJevQuestion(id: "paid", instructions: "Has the invoice been paid?", kind: .boolean))
        XCTAssertEqual(decision.selectedID, "true")
    }
}
