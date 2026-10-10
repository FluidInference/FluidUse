import XCTest

@testable import FluidUse

/// clef-text-0.6b host pieces that need no model: RoPE tables, the causal + padding mask, the pinned asset list, and
/// the demo's mock backlog.
@available(macOS 15.0, *)
final class ClefTextTests: XCTestCase {
    func testRopeTablesMatchPlainRope() throws {
        let (cos, sin) = try ClefTextManager.ropeTables(length: 64, headDim: 128, theta: 1_000_000)
        XCTAssertEqual(cos.shape, [64, 128])
        let c = cos.dataPointer.assumingMemoryBound(to: Float16.self)
        let s = sin.dataPointer.assumingMemoryBound(to: Float16.self)
        for position in [0, 1, 17, 63] {
            for i in [0, 9, 63] {
                let angle = Double(position) / pow(1_000_000, Double(2 * i) / 128)
                XCTAssertEqual(Double(c[position * 128 + i]), Foundation.cos(angle), accuracy: 1e-3)
                XCTAssertEqual(Double(s[position * 128 + i]), Foundation.sin(angle), accuracy: 1e-3)
                XCTAssertEqual(c[position * 128 + i], c[position * 128 + 64 + i])
            }
        }
    }

    func testMaskIsCausalAndHidesPadding() throws {
        let length = 8
        let real = 5
        let mask = try ClefTextManager.mask(length: length, realTokens: real)
        let m = mask.dataPointer.assumingMemoryBound(to: Float16.self)
        func visible(_ q: Int, _ k: Int) -> Bool { m[q * length + k] == 0 }
        for q in 0..<real {
            for k in 0..<length { XCTAssertEqual(visible(q, k), k <= q, "q \(q) k \(k)") }
        }
        // padded query rows see every real token and stay causal among themselves (outputs ignored, values finite)
        XCTAssertTrue(visible(6, 0) && visible(6, 4) && visible(6, 6))
        XCTAssertFalse(visible(6, 7))
    }

    func testStoreListsEveryBundleFile() {
        let paths = Set(ClefTextModelStore.assets.map(\.path))
        for file in [
            "config.json", "tokenizer.json", "embeddings.f16", "Decoder.mlpackage/Manifest.json",
            "Head.mlpackage/Manifest.json",
        ] {
            XCTAssertTrue(paths.contains(file), file)
        }
        XCTAssertEqual(ClefTextModelStore.revision.count, 40)
        XCTAssertTrue(ClefTextModelStore.assets.allSatisfy { $0.sha256.count == 64 })
    }
}
