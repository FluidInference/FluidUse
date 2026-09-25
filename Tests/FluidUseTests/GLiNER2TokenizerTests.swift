import Foundation
import XCTest

@testable import FluidUse

final class GLiNER2TokenizerTests: XCTestCase {
    private struct Reference: Decodable {
        let text: String
        let task: String
        let labels: [String]
        let ids: [Int]
        let markers: [Int]
    }

    private func tokenizer(for variant: GLiNER2Variant) throws -> GLiNER2Tokenizer {
        let variable = variant == .base ? "FLUIDUSE_GLINER2_BASE_MODEL_DIR" : "FLUIDUSE_GLINER2_MULTI_MODEL_DIR"
        guard let directory = ProcessInfo.processInfo.environment[variable] else {
            throw XCTSkip("Set \(variable) to check the pinned Hub tokenizer")
        }
        return try GLiNER2Tokenizer(
            tokenizerJsonURL: URL(fileURLWithPath: directory).appendingPathComponent("tokenizer/tokenizer.json"))
    }

    func testWordSplitterMatchesUpstreamUnicodeBehavior() throws {
        XCTAssertEqual(try GLiNER2Tokenizer.splitText("नमस्ते"), ["नमस", "्", "त", "े"])
        XCTAssertEqual(try GLiNER2Tokenizer.splitText("cafe\u{301}"), ["cafe", "\u{301}"])
        XCTAssertEqual(try GLiNER2Tokenizer.splitText("x-y 中文"), ["x-y", "中文"])
    }

    func testPinnedTokenizersMatchUpstreamEdgeSequences() throws {
        for variant in [GLiNER2Variant.base, .multilingual] {
            let tokenizer = try tokenizer(for: variant)
            let name = variant == .base ? "gliner2-base-edge-sequences" : "gliner2-multilingual-edge-sequences"
            let file = try XCTUnwrap(
                Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures"))
            let rows = try JSONDecoder().decode([Reference].self, from: Data(contentsOf: file))
            for row in rows {
                let actual = try tokenizer.classificationSequence(text: row.text, task: row.task, labels: row.labels)
                XCTAssertEqual(actual.ids, row.ids, "\(variant) \(row.text) \(row.labels)")
                XCTAssertEqual(actual.markers, row.markers, "\(variant) \(row.text) \(row.labels)")
            }
        }
    }

    func testEmbeddedSpecialTokensAndUnknownRuns() throws {
        let expected: [(GLiNER2Variant, [Int], [Int])] = [
            (.base, [1204, 128007, 2982], [507, 3]),
            (.multilingual, [260, 330, 250108, 260, 277], [260, 3]),
        ]
        for (variant, special, unknown) in expected {
            let tokenizer = try tokenizer(for: variant)
            XCTAssertEqual(tokenizer.encode("x[L]y"), special)
            XCTAssertEqual(tokenizer.encode("🫩🫩"), unknown)
        }
    }
}
