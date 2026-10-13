import XCTest

@testable import FluidUse

/// Unlimited-OCR host pieces that need no model: output parsing, page geometry, bucket math and the pinned assets.
final class UnlimitedOcrTests: XCTestCase {
    /// Start of the model's own output for page 4 of Baidu's paper (fp32 reference run).
    static let output = """
        <PAGE><|det|>title [115, 100, 293, 116]<|/det|>2. Related Works
        <|det|>title [114, 133, 383, 149]<|/det|>2.1. Pipeline-based Framework
        <|det|>text [113, 159, 886, 370]<|/det|>Traditional OCR models, particularly those designed for document parsing
        <PAGE><|det|>sub_title [100, 50, 400, 60]<|/det|>3.1. Next
        <|det|>text [100, 70, 900
        """

    func testParsesPagesBlocksAndBoxes() {
        let document = UnlimitedOcrDocument(parsing: Self.output)
        XCTAssertEqual(document.pages.count, 2)
        let first = document.pages[0].blocks
        XCTAssertEqual(first.map(\.label), ["title", "title", "text"])
        XCTAssertEqual(first[0].content, "2. Related Works")
        XCTAssertEqual(first[0].boxes.count, 1)
        let box = first[0].boxes[0]
        XCTAssertEqual(box.minX, 115.0 / 999, accuracy: 1e-9)
        XCTAssertEqual(box.minY, 100.0 / 999, accuracy: 1e-9)
        XCTAssertEqual(box.maxX, 293.0 / 999, accuracy: 1e-9)
        XCTAssertEqual(box.maxY, 116.0 / 999, accuracy: 1e-9)
    }

    func testStreamingTailHasNoBoxUntilComplete() {
        let blocks = UnlimitedOcrDocument(parsing: Self.output).pages[1].blocks
        XCTAssertEqual(blocks.map(\.label), ["sub_title", "text"])
        XCTAssertTrue(blocks[1].boxes.isEmpty)
        XCTAssertEqual(blocks[1].content, "")
    }

    func testMarkdownDropsLayoutMarkup() {
        let markdown = UnlimitedOcrDocument(parsing: Self.output).markdown
        XCTAssertFalse(markdown.contains("<|det|>"))
        XCTAssertFalse(markdown.contains("<PAGE>"))
        XCTAssertTrue(markdown.hasPrefix("2. Related Works\n\n2.1. Pipeline-based Framework"))
        XCTAssertTrue(markdown.contains("\n\n---\n\n3.1. Next"))
    }

    func testFitRectCentersAnA4Page() {
        let rect = UnlimitedOcrPagePreprocessor.fitRect(width: 595, height: 842)
        XCTAssertEqual(rect.height, 1024)
        XCTAssertEqual(rect.width, (595.0 * 1024 / 842).rounded())
        XCTAssertEqual(rect.minX, ((1024 - rect.width) / 2).rounded(.down))
        XCTAssertEqual(rect.minY, 0)
    }

    func testBucketsHoldOneTwoFourAndEightPages() throws {
        let json = """
            {"hidden_size": 1280, "head_dim": 128, "layers": 12, "vocab_size": 129280, "vocab_half": 64640,
             "rope_theta": 10000.0, "ring": 128, "chunk": 128, "capacities": [384, 640, 1152, 2304],
             "page_tokens": 273, "image_size": 1024, "bos_id": 0, "eos_id": 1, "image_token_id": 128815,
             "prompt_suffix_ids": {"multi": [37460, 4366, 76466, 16], "single": [34030, 76466, 16]},
             "ngram_size": 35, "ngram_window": 1024, "max_length": 32768,
             "decoder": "UnlimitedOcrDecoder.mlpackage", "encoder": "UnlimitedOcrPageEncoder.mlpackage"}
            """
        let config = try JSONDecoder().decode(UnlimitedOcrConfig.self, from: Data(json.utf8))
        XCTAssertEqual(config.prefixLength(pages: 1, prompt: "multi"), 278)
        XCTAssertEqual(config.prefixLength(pages: 3, prompt: "multi"), 824)
        let buckets = (1...8).map { config.capacity(forPrefix: config.prefixLength(pages: $0, prompt: "multi")) }
        XCTAssertEqual(buckets, [384, 640, 1152, 1152, 2304, 2304, 2304, 2304])
        XCTAssertNil(config.capacity(forPrefix: config.prefixLength(pages: 9, prompt: "multi")))
        XCTAssertEqual(config.maxPages, 8)
    }

    func testModelStorePinsEveryBundleFile() {
        let paths = UnlimitedOcrModelStore.assets.map(\.path)
        XCTAssertEqual(Set(paths).count, paths.count)
        for required in [
            "config.json", "embeddings.f16", "tokenizer.json",
            "UnlimitedOcrDecoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
            "UnlimitedOcrPageEncoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
        ] {
            XCTAssertTrue(paths.contains(required), required)
        }
        XCTAssertTrue(UnlimitedOcrModelStore.assets.allSatisfy { $0.sha256.count == 64 })
        XCTAssertEqual(UnlimitedOcrModelStore.revision.count, 40)
        let mlx = Set(UnlimitedOcrModelStore.assets(for: .mlx).map(\.path))
        XCTAssertTrue(mlx.contains("mlx/decoder.safetensors"))
        XCTAssertTrue(mlx.contains("UnlimitedOcrPageEncoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin"))
        XCTAssertFalse(mlx.contains("UnlimitedOcrDecoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin"))
        XCTAssertFalse(mlx.contains("embeddings.f16"))
    }
}
