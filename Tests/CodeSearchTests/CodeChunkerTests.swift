import XCTest

@testable import CodeSearch

final class CodeChunkerTests: XCTestCase {
    private let source = """
        import Foundation

        /// Converts audio.
        public struct AudioConverter {
            /// Resamples to 16 kHz mono.
            public func resample(_ samples: [Float]) -> [Float] {
                samples
            }

            class func shared() -> AudioConverter { AudioConverter() }

            init() {}
        }

        extension AudioConverter {
            @discardableResult
            static func write(_ samples: [Float]) -> Bool { true }
        }
        """

    func testDeclarationsAndOwners() {
        let chunks = CodeChunker.chunks(source: source, path: "Sources/AudioConverter.swift")
        XCTAssertEqual(
            chunks.map(\.name),
            [
                "AudioConverter", "AudioConverter.resample", "AudioConverter.shared", "AudioConverter.init",
                "AudioConverter",
                "AudioConverter.write",
            ])
        XCTAssertEqual(chunks.map(\.kind), ["struct", "func", "func", "init", "extension", "func"])
    }

    func testLinesAndDocComments() {
        let chunks = CodeChunker.chunks(source: source, path: "Sources/AudioConverter.swift")
        let resample = chunks[1]
        XCTAssertEqual(resample.line, 6)
        XCTAssertTrue(resample.code.hasPrefix("/// Resamples to 16 kHz mono."))
        // The next declaration's doc comment stays with it.
        XCTAssertFalse(chunks[0].code.contains("Resamples"))
        XCTAssertTrue(chunks[5].code.hasPrefix("@discardableResult"))
    }

    func testDocumentFormat() {
        let chunk = CodeChunker.chunks(source: source, path: "Sources/AudioConverter.swift")[1]
        XCTAssertTrue(chunk.document.hasPrefix("title: AudioConverter.swift AudioConverter.resample | text: "))
    }
}
