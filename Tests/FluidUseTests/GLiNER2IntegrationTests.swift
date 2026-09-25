import Foundation
import XCTest

@testable import FluidUse

/// Reference sequences and scores are generated with the published Python preprocessing and W8 packages.
final class GLiNER2IntegrationTests: XCTestCase {
    private struct Reference: Decodable {
        let task: String
        let text: String
        let labels: [String]
        let ids: [Int]
        let markers: [Int]
        let probabilities: [Float]
    }

    private func directory(for variant: GLiNER2Variant) throws -> URL {
        let variable: String
        switch variant {
        case .small: variable = "FLUIDUSE_GLINER2_SMALL_MODEL_DIR"
        case .base: variable = "FLUIDUSE_GLINER2_BASE_MODEL_DIR"
        case .multilingual: variable = "FLUIDUSE_GLINER2_MULTI_MODEL_DIR"
        case .decide, .decideLong: variable = "FLUIDUSE_GLINER2_DECIDE_MODEL_DIR"
        }
        guard let path = ProcessInfo.processInfo.environment[variable], !path.isEmpty else {
            throw XCTSkip("Set \(variable) to run real GLiNER 2.5 integration tests")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    private func references(for variant: GLiNER2Variant) throws -> [Reference] {
        let name = variant == .multilingual ? "gliner2-multilingual-sequences" : "gliner2-base-sequences"
        guard let file = Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures") else {
            XCTFail("Missing \(name) fixture")
            return []
        }
        return try JSONDecoder().decode([Reference].self, from: Data(contentsOf: file))
    }

    func testBaseMatchesReference() async throws {
        try await check(variant: .base)
    }

    func testSmallUsesPublishedTokenizerAndPackage() async throws {
        let manager = try await GLiNER2Manager.load(
            from: directory(for: .small), variant: .small, computeUnits: .cpuOnly)
        for item in try references(for: .small) {
            let sequence = try manager.tokenSequence(text: item.text, task: item.task, labels: item.labels)
            XCTAssertEqual(sequence.ids, item.ids)
            XCTAssertEqual(sequence.markers, item.markers)
            let answer = try await manager.classify(text: item.text, task: item.task, labels: item.labels)
            XCTAssertEqual(answer.probabilities.count, item.labels.count)
            XCTAssertTrue(answer.probabilities.allSatisfy(\.isFinite))
        }
    }

    func testMultilingualMatchesReference() async throws {
        try await check(variant: .multilingual)
    }

    private func check(variant: GLiNER2Variant) async throws {
        let manager = try await GLiNER2Manager.load(
            from: directory(for: variant), variant: variant, computeUnits: .cpuOnly)
        for item in try references(for: variant) {
            let sequence = try manager.tokenSequence(text: item.text, task: item.task, labels: item.labels)
            XCTAssertEqual(sequence.ids, item.ids, "\(variant) \(item.text)")
            XCTAssertEqual(sequence.markers, item.markers, "\(variant) \(item.text)")
            let answer = try await manager.classify(text: item.text, task: item.task, labels: item.labels)
            XCTAssertEqual(answer.tokenCount, item.ids.count)
            XCTAssertEqual(answer.probabilities.count, item.labels.count)
            for (actual, expected) in zip(answer.probabilities, item.probabilities) {
                XCTAssertEqual(actual, expected, accuracy: 0.01, "\(variant) \(item.text)")
            }
        }
    }

    func testRejectsOverlongSchema() async throws {
        let manager = try await GLiNER2Manager.load(from: directory(for: .base), variant: .base, computeUnits: .cpuOnly)
        do {
            _ = try await manager.classify(
                text: "short", task: "decision", labels: [String(repeating: "very long option ", count: 100)])
            XCTFail("Expected an overlong schema error")
        } catch let error as GLiNER2Error {
            guard case .invalidInput = error else { return XCTFail("Wrong error: \(error)") }
        }
    }

    func testModelNamesPointToPublishedW8Packages() {
        XCTAssertEqual(
            GLiNER2Variant.small.packageName, "gliner2_small_classification_embedding_w8_L128_K8.mlpackage")
        XCTAssertEqual(
            GLiNER2Variant.base.packageName, "gliner2_base_classification_embedding_w8_L128_K8.mlpackage")
        XCTAssertEqual(
            GLiNER2Variant.multilingual.packageName,
            "gliner2_multi_classification_embedding_w8_linear_L128_K8.mlpackage")
        XCTAssertEqual(GLiNER2Variant.decide.packageName, "gliner2_decide_classification_fp16_L128_H4_K32.mlpackage")
        XCTAssertEqual(
            GLiNER2Variant.decideLong.packageName, "gliner2_decide_classification_fp16_L256_H4_K32.mlpackage")
        XCTAssertEqual(GLiNER2Variant.decide.maximumOptions, 32)
        XCTAssertEqual(GLiNER2Variant.decideLong.maximumLength, 256)
        XCTAssertEqual(GLiNER2Variant.decide.maximumHeads, 4)
        XCTAssertEqual(GLiNER2Variant.base.maximumOptions, 8)
    }
}
