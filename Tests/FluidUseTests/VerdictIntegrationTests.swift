import Foundation
import XCTest

@testable import FluidUse

final class VerdictIntegrationTests: XCTestCase {
    func testNativeRenderingAndAbstention() throws {
        let request = try VerdictManager.render(
            context: "No information is available.",
            question: .choice(
                question: "Which card was stolen?",
                options: [
                    .init(id: "visa", description: "a Visa debit card"),
                    .init(id: "mastercard", description: "a Mastercard debit card"),
                ]))
        XCTAssertEqual(request.ids, ["visa", "mastercard", VerdictManager.abstentionID])
        XCTAssertEqual(
            request.text,
            "<<LABEL>>It is a Visa debit card<<LABEL>>It is a Mastercard debit card"
                + "<<LABEL>>insufficient evidence<<SEP>>Question: Which card was stolen?"
                + "\n\nContext:\nNo information is available.")
    }

    func testScoreValuesKeepNativeIntegerRendering() throws {
        let request = try VerdictManager.render(
            context: "The delivery was late.",
            question: .score(
                question: "How severe?",
                levels: [
                    .init(id: "low", description: "Low urgency", value: 0),
                    .init(id: "high", description: "High urgency", value: 2),
                ]))
        XCTAssertTrue(request.text.contains("<<LABEL>>Low urgency (Value: 0)"))
        XCTAssertTrue(request.text.contains("<<LABEL>>High urgency (Value: 2)"))
    }

    func testScoreValuesKeepPythonFloatRendering() throws {
        let request = try VerdictManager.render(
            context: "c",
            question: .score(
                question: "q",
                levels: [
                    .init(id: "a", description: "A", value: .real(2)),
                    .init(id: "b", description: "B", value: 2.5),
                    .init(id: "c", description: "C", value: 1e-05),
                    .init(id: "d", description: "D", value: -3),
                ]))
        XCTAssertTrue(request.text.contains("<<LABEL>>A (Value: 2.0)"))
        XCTAssertTrue(request.text.contains("<<LABEL>>B (Value: 2.5)"))
        XCTAssertTrue(request.text.contains("<<LABEL>>C (Value: 1e-05)"))
        XCTAssertTrue(request.text.contains("<<LABEL>>D (Value: -3)"))
        XCTAssertEqual(request.values, [2, 2.5, 1e-05, -3])
    }

    func testPinnedChecksumsAreSHA256() {
        let assets = VerdictModelStore.shared + VerdictModelStore.buckets.values.flatMap { $0 }
        XCTAssertEqual(assets.count, 10)
        for asset in assets {
            XCTAssertEqual(asset.sha256.count, 64, asset.path)
            XCTAssertTrue(asset.sha256.allSatisfy(\.isHexDigit), asset.path)
        }
    }

    func testPublishedL128ChoiceAndAbstention() async throws {
        guard let path = ProcessInfo.processInfo.environment["FLUIDUSE_VERDICT_MODEL_DIR"], !path.isEmpty else {
            throw XCTSkip("Set FLUIDUSE_VERDICT_MODEL_DIR to a published Verdict package directory")
        }
        let manager = try await VerdictManager.load(
            from: URL(fileURLWithPath: path, isDirectory: true), configuration: .init(lengths: [128]))
        let choice = try await manager.answer(
            context: "I lost my wallet yesterday and need to stop my debit card immediately.",
            question: .choice(
                question: "What is the primary customer inquiry?",
                options: [
                    .init(id: "card_lost", description: "Reporting a lost or stolen card"),
                    .init(id: "pin_reset", description: "Requesting a PIN reset"),
                ]))
        XCTAssertEqual(choice.tokenCount, 53)
        XCTAssertEqual(choice.selectedID, "card_lost")
        XCTAssertEqual(choice.probabilities[0], 0.6283802556009165, accuracy: 0.01)

        let abstention = try await manager.answer(
            context: "No information is available.",
            question: .choice(
                question: "Which card was stolen?",
                options: [
                    .init(id: "visa", description: "a Visa debit card"),
                    .init(id: "mastercard", description: "a Mastercard debit card"),
                ]))
        XCTAssertTrue(abstention.isAbstention)
        XCTAssertEqual(abstention.selectedID, VerdictManager.abstentionID)
    }
}
