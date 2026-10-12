import Foundation
import XCTest

@testable import FluidUse

final class EvokeTests: XCTestCase {
    private struct Reference: Decodable {
        struct TokenCase: Decodable {
            let text: String
            let ids: [Int]
        }
        let tokens: [TokenCase]
    }

    func testTransformAppliesLog1pPowerScaleAndDropsNonPositive() {
        let transform = EvokeTransform(activeDims: 3, gamma: 2, scale: 0.5)
        let terms = transform.terms(maxLogits: [3, 1, -2, 9], vocabIds: [10, 11, 12, 13])
        XCTAssertEqual(Set(terms.keys), [10, 11], "negative logit dropped, entries past activeDims ignored")
        XCTAssertEqual(terms[10]!, pow(log1p(3), 2) * 0.5, accuracy: 1e-6)
        XCTAssertEqual(terms[11]!, pow(log1p(1), 2) * 0.5, accuracy: 1e-6)
    }

    func testScoreIsSparseDotOverSharedTerms() {
        let query: EvokeTerms = [1: 2, 2: 3]
        let document: EvokeTerms = [2: 4, 3: 5]
        XCTAssertEqual(EvokeTerms.score(query, document), 12)
        XCTAssertEqual(EvokeTerms.score(document, query), 12)
        XCTAssertEqual(EvokeTerms.score(query, [:]), 0)
    }

    func testConfigDecodesPublishedPoolingConstants() throws {
        let json = """
            {"sequence_lengths": [64, 128], "top_k": 192,
             "evoke": {"query": {"active_dims": 50, "gamma": 1.8518644571304321, "score_scale": 0.6963680386543274},
                       "document": {"active_dims": 192, "gamma": 0.5627960562705994, "score_scale": 1.0}}}
            """
        let config = try JSONDecoder().decode(EvokeConfig.self, from: Data(json.utf8))
        XCTAssertEqual(config.transform(for: .query).activeDims, 50)
        XCTAssertEqual(config.transform(for: .document).activeDims, 192)
        XCTAssertEqual(config.sequenceLengths, [64, 128])
    }

    func testPinnedTokenizerMatchesHuggingFaceRoberta() throws {
        guard let directory = ProcessInfo.processInfo.environment["FLUIDUSE_EVOKE_MODEL_DIR"] else {
            throw XCTSkip("Set FLUIDUSE_EVOKE_MODEL_DIR to check the pinned Hub tokenizer")
        }
        let tokenizer = try QwenBPETokenizer(
            tokenizerJsonURL: URL(fileURLWithPath: directory).appendingPathComponent("tokenizer.json"),
            preTokenizer: .gpt2)
        let file = try XCTUnwrap(
            Bundle.module.url(forResource: "evoke-reference", withExtension: "json", subdirectory: "Fixtures"))
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: file))
        for row in reference.tokens {
            XCTAssertEqual([0] + (try tokenizer.encode(row.text)) + [2], row.ids, row.text)
        }
    }
}
