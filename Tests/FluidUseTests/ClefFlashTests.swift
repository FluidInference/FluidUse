import XCTest

@testable import FluidUse

/// clef-flash host pieces that need no model: manifest parsing, text-only RoPE tables, and the pinned asset list.
@available(macOS 15.0, *)
final class ClefFlashTests: XCTestCase {
    static let manifest: [String: Any] = [
        "hidden_size": 4096, "rotary_dim": 64, "rope_theta": 10_000_000, "pad_id": 248044, "vocab_size": 248320,
        "buckets": [512, 256, 2048, 1024], "max_questions": 16, "max_options": 96, "image_token_id": 248056,
        "vision_start_token_id": 248053, "vision_end_token_id": 248054,
        "parts": (0..<8).map { String(format: "part%02d.mlpackage", $0) }, "head": "Head.mlpackage",
    ]

    func testManifestParses() throws {
        let config = try ClefFlashManager.Config(json: Self.manifest)
        XCTAssertEqual(config.buckets, [256, 512, 1024, 2048])
        XCTAssertEqual(config.parts.count, 8)
        XCTAssertEqual(config.rotaryDim, 64)
        var missing = Self.manifest
        missing.removeValue(forKey: "pad_id")
        XCTAssertThrowsError(try ClefFlashManager.Config(json: missing))
    }

    func testRopeTablesMatchPlainRope() throws {
        let config = try ClefFlashManager.Config(json: Self.manifest)
        let (cos, sin) = try ClefFlashManager.ropeTables(length: 256, config: config)
        XCTAssertEqual(cos.shape, [256, 64])
        let c = cos.dataPointer.assumingMemoryBound(to: Float16.self)
        let s = sin.dataPointer.assumingMemoryBound(to: Float16.self)
        for position in [0, 1, 37, 255] {
            for i in [0, 5, 31] {
                let angle = Double(position) / pow(10_000_000, Double(2 * i) / 64)
                XCTAssertEqual(Double(c[position * 64 + i]), Foundation.cos(angle), accuracy: 1e-3)
                XCTAssertEqual(Double(s[position * 64 + i]), Foundation.sin(angle), accuracy: 1e-3)
                // halves repeat (rotate_half layout)
                XCTAssertEqual(c[position * 64 + i], c[position * 64 + 32 + i])
                XCTAssertEqual(s[position * 64 + i], s[position * 64 + 32 + i])
            }
        }
    }

    func testStoreListsEveryBundleFile() {
        let paths = Set(ClefFlashModelStore.assets.map(\.path))
        for file in ["config.json", "tokenizer.json", "embeddings.f16", "output_embeddings.f16"] {
            XCTAssertTrue(paths.contains(file), file)
        }
        for part in 0..<8 {
            XCTAssertTrue(paths.contains(String(format: "part%02d.mlpackage/Manifest.json", part)))
        }
        XCTAssertTrue(paths.contains("Head.mlpackage/Manifest.json"))
        XCTAssertTrue(ClefFlashModelStore.assets.allSatisfy { $0.sha256.count == 64 })
    }
}
