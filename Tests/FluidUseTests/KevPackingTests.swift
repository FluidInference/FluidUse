import XCTest

@testable import FluidUse

/// Packing of question branches into one fused call: lane placement and grouping, and the question rendering the
/// packed branches are built from. The fused Core ML path itself is covered by `KevCheck fast-parity`.
final class KevPackingTests: XCTestCase {
    func testQuestionsNeverCrossALane() throws {
        guard #available(macOS 15.0, iOS 18.0, *) else { throw XCTSkip("KevFastManager needs macOS 15") }
        // 100 + 40 would cross the first 128-token lane, so the second question starts the next lane
        XCTAssertEqual(KevFastManager.laneStarts([100, 40, 20]), [0, 128, 168])
        XCTAssertEqual(KevFastManager.laneStarts([64, 64, 1]), [0, 64, 128])
        XCTAssertEqual(KevFastManager.laneStarts([128, 128]), [0, 128])
    }

    func testGroupsRespectPackedLengthAndReadouts() throws {
        guard #available(macOS 15.0, iOS 18.0, *) else { throw XCTSkip("KevFastManager needs macOS 15") }
        // 12 questions of 16 tokens fill exactly 192 packed tokens
        XCTAssertEqual(KevFastManager.packGroups(Array(repeating: 16, count: 12), packedLen: 192), [Array(0..<12)])
        // a 13th spills into a second call
        XCTAssertEqual(
            KevFastManager.packGroups(Array(repeating: 16, count: 13), packedLen: 192), [Array(0..<12), [12]])
        // lane padding counts toward the packed length: 100 + (skip to 128) + 100 = 228 > 192
        XCTAssertEqual(KevFastManager.packGroups([100, 100], packedLen: 192), [[0], [1]])
        // at most 16 readouts per call even when tokens remain
        XCTAssertEqual(
            KevFastManager.packGroups(Array(repeating: 1, count: 20), packedLen: 256), [Array(0..<16), Array(16..<20)])
    }

    func testYesNoQuestionRendersKevOptions() {
        let question = KevQuestion.noul("Is this person an athlete?")
        XCTAssertEqual(question.optionTexts, ["no", "yes"])
        XCTAssertEqual(question.keys, ["false", "true"])
    }

    func testPinnedSnapshotListsEveryFileOnce() {
        let paths = KevModelStore.assets.map(\.path)
        XCTAssertEqual(Set(paths).count, paths.count)
        XCTAssertTrue(paths.contains("fused/KevFused.mlpackage/Data/com.apple.CoreML/weights/weight.bin"))
        XCTAssertTrue(paths.contains("tokenizer.json"))
        XCTAssertTrue(KevModelStore.assets.allSatisfy { $0.sha256.count == 64 })
        XCTAssertEqual(KevModelStore.revision.count, 40)
    }
}
