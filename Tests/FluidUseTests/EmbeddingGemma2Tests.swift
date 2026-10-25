import Foundation
import XCTest

@testable import FluidUse

/// Tokenizer behaviour on a hand-built `tokenizer.json`, plus parity and packing checks against the real assets.
final class EmbeddingGemma2Tests: XCTestCase {
    /// Tiny Gemma-style tokenizer: a BOM-prefixed token next to its plain twin, a CJK compatibility ideograph next
    /// to its canonical equivalent, byte fallback, and three ranked merges.
    private let miniTokenizer = #"""
        {"model": {"type": "BPE", "vocab": {
            "<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "﻿#": 4, "#": 5, "a": 6, "b": 7, "ab": 8,
            "▁": 9, "▁a": 10, "豈": 11, "豈": 12, "<0xC3>": 13, "<0xA9>": 14, "aa": 15},
          "merges": [["a", "b"], ["▁", "a"], ["a", "a"]]}}
        """#

    private func mini() throws -> EmbeddingGemma2Tokenizer {
        try EmbeddingGemma2Tokenizer(data: Data(miniTokenizer.utf8))
    }

    func testKeepsBOMPrefixedTokenDistinct() throws {
        XCTAssertEqual(try mini().encode("#", maxLength: 8), [2, 5, 1])
    }

    func testKeepsCanonicallyEquivalentTokensDistinct() throws {
        let tokenizer = try mini()
        XCTAssertEqual(tokenizer.tokenize("\u{F900}"), [11])
        XCTAssertEqual(tokenizer.tokenize("\u{8C48}"), [12])
    }

    func testFallsBackToUTF8Bytes() throws {
        XCTAssertEqual(try mini().tokenize("é"), [13, 14])
    }

    func testMergesByRankThenLeftmost() throws {
        let tokenizer = try mini()
        XCTAssertEqual(tokenizer.tokenize("ab"), [8])
        XCTAssertEqual(tokenizer.tokenize(" a"), [10])
        XCTAssertEqual(tokenizer.tokenize("aaa"), [15, 6])
    }

    func testTruncatesBetweenSpecialTokens() throws {
        XCTAssertEqual(try mini().encode("ababab", maxLength: 4), [2, 8, 8, 1])
    }

    func testPromptPrefixes() {
        XCTAssertEqual(EmbeddingGemma2Prompt.document(title: nil).apply(to: "x"), "title: none | text: x")
        XCTAssertEqual(EmbeddingGemma2Prompt.clustering.apply(to: "x"), "task: clustering | query: x")
    }

    // MARK: Real assets (set FLUIDUSE_EMBEDDINGGEMMA2_MODEL_DIR to the downloaded embeddinggemma-2-coreml folder)

    private struct TokenizerCase: Decodable {
        let text: String
        let ids: [Int32]
    }

    private func modelDirectory() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["FLUIDUSE_EMBEDDINGGEMMA2_MODEL_DIR"], !path.isEmpty
        else { throw XCTSkip("Set FLUIDUSE_EMBEDDINGGEMMA2_MODEL_DIR to enable EmbeddingGemma 2 integration tests") }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    private func cases() throws -> [TokenizerCase] {
        let url = Bundle.module.resourceURL!.appendingPathComponent("Fixtures/embeddinggemma2-tokenizer-cases.json")
        return try JSONDecoder().decode([TokenizerCase].self, from: Data(contentsOf: url))
    }

    func testTokenizerMatchesHuggingFace() throws {
        let tokenizer = try EmbeddingGemma2Tokenizer(
            tokenizerJsonURL: try modelDirectory().appendingPathComponent("tokenizer.json"))
        for item in try cases() {
            XCTAssertEqual(tokenizer.encode(item.text, maxLength: .max), item.ids, item.text)
        }
    }

    func testPackedMatchesOneAtATime() async throws {
        guard #available(macOS 15, iOS 18, *) else { throw XCTSkip("needs macOS 15") }
        let manager = try await EmbeddingGemma2Manager.load(from: try modelDirectory())
        let texts = try cases().map(\.text).filter { !$0.isEmpty }
        let packed = try await manager.embed(texts, prompt: .clustering)
        for (text, vector) in zip(texts, packed) {
            let single = try await manager.embed(text, prompt: .clustering)
            XCTAssertEqual(vector.count, EmbeddingGemma2Manager.dimension)
            let cosine = zip(single, vector).reduce(Float(0)) { $0 + $1.0 * $1.1 }
            XCTAssertGreaterThan(cosine, 0.995, text)
            XCTAssertEqual(vector.reduce(0) { $0 + $1 * $1 }, 1, accuracy: 0.01)
        }
    }
}

/// Window arithmetic for the audio path, against the Hugging Face feature extractor's counts.
final class EmbeddingGemma2AudioWindowTests: XCTestCase {
    func testFullWindowMatchesFeatureExtractor() {
        // 10 s at 16 kHz: the extractor yields 999 frames and the encoder 250 tokens.
        let frames = EmbeddingGemma2Audio.validFrames(samples: 160_000)
        XCTAssertEqual(frames, 999)
        XCTAssertEqual(EmbeddingGemma2Audio.tokenCount(validFrames: frames), 250)
    }

    func testShortClipMatchesFeatureExtractor() {
        // 55,680 samples (3.48 s): 347 frames, 87 audio tokens.
        let frames = EmbeddingGemma2Audio.validFrames(samples: 55_680)
        XCTAssertEqual(frames, 347)
        XCTAssertEqual(EmbeddingGemma2Audio.tokenCount(validFrames: frames), 87)
    }

    func testTooShortHasNoFrames() {
        XCTAssertEqual(EmbeddingGemma2Audio.validFrames(samples: 100), 0)
    }
}

/// Image resize targets, against Hugging Face `get_aspect_ratio_preserving_size`.
final class EmbeddingGemma2VisionSizeTests: XCTestCase {
    func testTargetSizesMatchHuggingFace() {
        let cases: [(width: Int, height: Int, budget: EmbeddingGemma2Vision.Budget, expected: (Int, Int))] = [
            (500, 334, .detailed, (960, 624)), (334, 500, .fast, (288, 480)), (1920, 1080, .balanced, (720, 384)),
            (4000, 100, .fast, (2496, 48)), (64, 64, .detailed, (768, 768)),
        ]
        for item in cases {
            let size = EmbeddingGemma2Vision.targetSize(width: item.width, height: item.height, budget: item.budget)
            XCTAssertEqual(size.width, item.expected.0, "\(item.width)x\(item.height)")
            XCTAssertEqual(size.height, item.expected.1, "\(item.width)x\(item.height)")
            XCTAssertLessThanOrEqual((size.width / 16) * (size.height / 16), item.budget.patches)
        }
    }
}
