import XCTest

@testable import BookmarkSort

final class TopicDiscoveryTests: XCTestCase {
    func testPhrasesStayInsideClausesAndSkipStopWords() {
        let phrases = TopicDiscovery.phrases(in: "Sourdough rose today. Cold proof works, honestly")
        XCTAssertTrue(phrases.contains("sourdough rose"))
        XCTAssertTrue(phrases.contains("cold proof"))
        XCTAssertFalse(phrases.contains("rose cold"))  // crosses the full stop
        XCTAssertFalse(phrases.contains("today"))
        XCTAssertFalse(phrases.contains("honestly"))
    }

    func testChinesePhrasesAreTwoToFourCharacterRuns() {
        let phrases = TopicDiscovery.phrases(in: "文字狱")
        XCTAssertEqual(Set(phrases), ["文字", "字狱", "文字狱"])
    }

    /// Three tight groups around orthogonal directions; k-means must recover them exactly, and the same seed must
    /// give the same answer.
    func testSphericalKMeansRecoversSeparatedGroups() {
        var vectors: [[Float]] = []
        var truth: [Int] = []
        for group in 0..<3 {
            for member in 0..<20 {
                var vector = [Float](repeating: 0, count: 8)
                vector[group] = 1
                vector[3 + member % 5] = 0.05 * Float(member % 4)
                vectors.append(TopicDiscovery.normalized(vector))
                truth.append(group)
            }
        }
        let assignment = TopicDiscovery.sphericalKMeans(vectors, k: 3, seed: 1)
        XCTAssertEqual(assignment, TopicDiscovery.sphericalKMeans(vectors, k: 3, seed: 1))
        for group in 0..<3 {
            let labels = Set(truth.indices.filter { truth[$0] == group }.map { assignment[$0] })
            XCTAssertEqual(labels.count, 1)
        }
        XCTAssertEqual(Set(assignment).count, 3)
    }

    func testAssignJoinsNearestTopicAndSubtopic() {
        let child = TopicNode(id: "0.1", name: "child", members: [], children: [], centroid: [0, 1, 0])
        let other = TopicNode(id: "0.0", name: "other", members: [], children: [], centroid: [0, 0, 1])
        var topics = [
            TopicNode(id: "0", name: "near", members: [1], children: [other, child], centroid: [0.6, 0.8, 0]),
            TopicNode(id: "1", name: "far", members: [2], children: [], centroid: [-1, 0, 0]),
        ]
        let path = TopicDiscovery.assign(7, vector: [0, 1, 0], into: &topics)
        XCTAssertEqual(path.map(\.id), ["0", "0.1"])
        XCTAssertEqual(topics[0].members, [1, 7])
        XCTAssertEqual(topics[0].children[1].members, [7])
        XCTAssertTrue(topics[1].members == [2])
    }

    func testCarryColorsKeepsMatchingTopicsColour() {
        var old = TopicNode(id: "0", name: "a", members: [], children: [], centroid: [1, 0])
        old.colorIndex = 4
        var new = [
            TopicNode(id: "0", name: "b", members: [], children: [], centroid: [0, 1]),
            TopicNode(id: "1", name: "a2", members: [], children: [], centroid: [0.99, 0.14]),
        ]
        TopicDiscovery.carryColors(from: [old], to: &new, paletteSize: 10)
        XCTAssertEqual(new[1].colorIndex, 4)
        XCTAssertNotEqual(new[0].colorIndex, 4)
    }
}
