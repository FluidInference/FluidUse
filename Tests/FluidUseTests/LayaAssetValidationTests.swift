import Foundation
import XCTest

@testable import FluidUse

final class LayaAssetValidationTests: XCTestCase {
    func testOversizedTokenizerIntegerThrowsInsteadOfTrapping() {
        let json = #"{"vocab":{"a":9999999999999999999999999999999999999999},"merges":[]}"#
        XCTAssertThrowsError(try LayaTokenizerFile.scan(Data(json.utf8)))
    }

    func testInvalidSurrogatePairThrowsInsteadOfTrapping() {
        let json = #"{"vocab":{"\uD800\u0000":1},"merges":[]}"#
        XCTAssertThrowsError(try LayaTokenizerFile.scan(Data(json.utf8)))
    }

    func testValidSurrogatePairPreservesScalar() throws {
        let json = #"{"vocab":{"\uD83D\uDE00":1},"merges":[]}"#
        let table = try LayaTokenizerFile.scan(Data(json.utf8))
        XCTAssertEqual(table.vocab.first?.0, "😀")
    }

    func testConcurrentFileInstallationLeavesOneCompleteFileAndNoStagingFiles() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let destination = directory.appendingPathComponent("cached-file")
        let first = directory.appendingPathComponent("download-one")
        let second = directory.appendingPathComponent("download-two")
        // File-installation test data, not model artifacts or inference doubles.
        let firstData = Data("first completed download".utf8)
        let secondData = Data("second completed download".utf8)
        try firstData.write(to: first)
        try secondData.write(to: second)
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { try LayaModelStore.installDownloadedFile(first, at: destination) }
            group.addTask { try LayaModelStore.installDownloadedFile(second, at: destination) }
            try await group.waitForAll()
        }
        let installed = try Data(contentsOf: destination)
        XCTAssertTrue(installed == firstData || installed == secondData)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), ["cached-file"])
    }
}
