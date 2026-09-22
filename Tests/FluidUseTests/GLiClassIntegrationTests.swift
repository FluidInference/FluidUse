import Foundation
import XCTest

@testable import FluidUse

/// Reference parity and inference checks against the real GLiClass Edge Apps v2 assets.
final class GLiClassIntegrationTests: XCTestCase {
    private struct TokenizerCase: Decodable {
        let text: String
        let ids: [Int]
    }

    private struct SequenceCase: Decodable {
        let text: String
        let ids: [Int]
        let markers: [Int]
    }

    private var fixtureDirectory: URL {
        Bundle.module.resourceURL!.appendingPathComponent("Fixtures")
    }

    private func modelDirectory() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["FLUIDUSE_GLICLASS_MODEL_DIR"], !path.isEmpty else {
            throw XCTSkip("Set FLUIDUSE_GLICLASS_MODEL_DIR to enable real GLiClass integration tests")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    func testTokenizerMatchesHuggingFaceFixtures() throws {
        let tokenizer = try GLiClassTokenizer(
            tokenizerJsonURL: try modelDirectory().appendingPathComponent("tokenizer.json"))
        let cases = try JSONDecoder().decode(
            [TokenizerCase].self,
            from: Data(contentsOf: fixtureDirectory.appendingPathComponent("gliclass-tokenizer-cases.json")))
        for item in cases {
            XCTAssertEqual(tokenizer.encode(item.text), item.ids, item.text.debugDescription)
        }
    }

    func testRenderedSequenceMatchesReferencePipeline() async throws {
        let directory = try modelDirectory()
        let manager = try await GLiClassManager.load(from: directory)
        let item = try JSONDecoder().decode(
            SequenceCase.self,
            from: Data(contentsOf: fixtureDirectory.appendingPathComponent("gliclass-sequence-case.json")))
        let labels = [
            "a poor Tetris placement that creates holes or a dangerous tall stack",
            "a clean Tetris placement that avoids holes and keeps the stack low",
        ]
        let sequence = manager.tokenSequence(
            text:
                "The I piece dropped at column 0 leaves no holes, keeps the surface flat, keeps the stack low, and clears one line.",
            labels: labels, prompt: "Which label best describes this placement?")
        XCTAssertEqual(sequence.ids, item.ids)
        XCTAssertEqual(sequence.markers, item.markers)

        let tokenizer = try GLiClassTokenizer(tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json"))
        let cases: [(text: String, labels: [String], prompt: String?)] = [
            (
                "Avoid holes, keep the stack low and smooth, and clear lines.",
                [
                    "The I piece clears two lines and buries nothing.",
                    "The T piece buries one cell and leaves two rows of space.",
                ],
                "Choose the best Tetris placement."
            ),
            ("café 中文 🧱", ["first option", "second option", "third option"], nil),
            ("contains [MASK] token", ["label  one", "|||IP_ADDRESS|||"], "specials: "),
        ]
        for item in cases {
            let optimized = manager.tokenSequence(text: item.text, labels: item.labels, prompt: item.prompt).ids
            let rendered = item.labels.map { "<<LABEL>>\($0)" }.joined() + "<<SEP>>" + (item.prompt ?? "") + item.text
            XCTAssertEqual(optimized, tokenizer.encode(rendered))
        }
    }

    func testRealModelPrefersCleanPlacement() async throws {
        let manager = try await GLiClassManager.load(from: try modelDirectory())
        let labels = [
            "a poor Tetris placement that creates holes or a dangerous tall stack",
            "a clean Tetris placement that avoids holes and keeps the stack low",
        ]
        let prompt = "Which label best describes this placement?"
        let bad = try await manager.classify(
            text: "The T piece creates 3 holes, leaves a very rough surface, and raises the stack to 17 rows.",
            labels: labels, prompt: prompt)
        let good = try await manager.classify(
            text: "The I piece creates no holes, keeps a flat low surface, and clears 2 lines.",
            labels: labels, prompt: prompt)
        XCTAssertGreaterThan(good.probabilities[1], bad.probabilities[1])
        XCTAssertEqual(good.probabilities.reduce(0, +), 1, accuracy: 1e-5)
        XCTAssertEqual(good.bucketLength, 128)
    }

    func testRepeatedPredictionsDoNotRetainPreviousInputState() async throws {
        let manager = try await GLiClassManager.load(from: try modelDirectory())
        let labels = ["a poor placement", "a clean placement"]
        let prompt = "Choose the better Tetris placement."
        let first = try await manager.classify(
            text: "The I piece clears two lines, buries nothing, and keeps the stack low.",
            labels: labels, prompt: prompt)
        _ = try await manager.classify(
            text: "The T piece buries three cells and leaves only one row of space above a very uneven surface.",
            labels: labels, prompt: prompt)
        let repeated = try await manager.classify(
            text: "The I piece clears two lines, buries nothing, and keeps the stack low.",
            labels: labels, prompt: prompt)

        XCTAssertEqual(repeated.logits, first.logits)
        XCTAssertEqual(repeated.probabilities, first.probabilities)
        XCTAssertEqual(repeated.selectedIndex, first.selectedIndex)
    }

    func testRejectsAnOptionWhoseTextDoesNotFitTheBucket() async throws {
        let manager = try await GLiClassManager.load(from: try modelDirectory())
        let labels = ["short option", String(repeating: "long option ", count: 200)]

        do {
            _ = try await manager.classify(text: "state", labels: labels)
            XCTFail("Expected the incomplete final option to be rejected")
        } catch let error as GLiClassError {
            XCTAssertEqual(error, .promptTooLong(optionCount: 2, maximumLength: 128))
        }
    }

    func testModelNamesCoverPublishedPrecisions() throws {
        XCTAssertEqual(
            try GLiClassManager.modelName(length: 128, precision: "fp16"),
            "gliclass_edge_apps_fp16_L128_options25")
        XCTAssertEqual(
            try GLiClassManager.modelName(length: 128, precision: "lut8"),
            "gliclass_edge_apps_lut8_kmeans_per_tensor_L128_options25")
        XCTAssertEqual(
            try GLiClassManager.modelName(length: 128, precision: "fp16-mask"),
            "gliclass_edge_apps_float_mask_fp16_L128_options25")
        XCTAssertThrowsError(try GLiClassManager.modelName(length: 128, precision: "int8"))
    }
}
