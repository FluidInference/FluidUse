import Foundation
import XCTest

@testable import FluidUse

/// Replays outputs pinned from the author's serving engine (`core/engine_encoder.py` @ 30f15564, bundled FP32 ONNX,
/// shipped `calibrator.json`) on real held-out contexts: prompt text, raw logits, applied temperature, and calibrated
/// probabilities. Regenerate with `Tools/official-benchmarks/verdict/pin_upstream_calibration.py`.
final class VerdictCalibrationTests: XCTestCase {
    private struct Fixture: Decodable {
        struct Case: Decodable {
            let name: String
            let context: String
            let query: [String: AnyDecodable]
            let prompt: String
            let candidateIDs: [String]
            let logits: [Float]
            let temperature: Double
            let probabilities: [Double]

            enum CodingKeys: String, CodingKey {
                case name, context, query, prompt, logits, temperature, probabilities
                case candidateIDs = "candidate_ids"
            }
        }

        let calibratorSHA256: String
        let cases: [Case]

        enum CodingKeys: String, CodingKey {
            case cases
            case calibratorSHA256 = "calibrator_sha256"
        }
    }

    private struct AnyDecodable: Decodable {
        let value: Any

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let string = try? container.decode(String.self) {
                value = string
            } else if let number = try? container.decode(Double.self) {
                value = number
            } else if let array = try? container.decode([[String: AnyDecodable]].self) {
                value = array.map { $0.mapValues(\.value) }
            } else {
                value = try container.decode([String: AnyDecodable].self).mapValues(\.value)
            }
        }
    }

    private func load() throws -> (Fixture, Data) {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "verdict-upstream-calibration", withExtension: "json", subdirectory: "Fixtures"))
        let data = try Data(contentsOf: url)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let calibrator = try JSONSerialization.data(withJSONObject: try XCTUnwrap(object["calibrator"]))
        return (try JSONDecoder().decode(Fixture.self, from: data), calibrator)
    }

    private func question(_ query: [String: AnyDecodable]) throws -> VerdictQuestion {
        let values = query.mapValues(\.value)
        switch values["type"] as? String {
        case "choice":
            let options = try XCTUnwrap(values["options"] as? [[String: Any]])
            return .choice(
                question: try XCTUnwrap(values["question"] as? String),
                options: try options.map {
                    .init(
                        id: try XCTUnwrap($0["id"] as? String), description: try XCTUnwrap($0["description"] as? String)
                    )
                })
        case "score":
            let levels = try XCTUnwrap(values["levels"] as? [[String: Any]])
            return .score(
                question: try XCTUnwrap(values["question"] as? String),
                levels: try levels.map {
                    .init(
                        id: try XCTUnwrap($0["id"] as? String),
                        description: try XCTUnwrap($0["description"] as? String),
                        value: try XCTUnwrap($0["value"] as? Double))
                })
        default:
            return .noul(proposition: try XCTUnwrap(values["proposition"] as? String))
        }
    }

    func testRenderingMatchesUpstreamEngine() throws {
        let (fixture, _) = try load()
        XCTAssertEqual(fixture.cases.count, 6)
        for item in fixture.cases {
            let rendered = try VerdictManager.render(context: item.context, question: try question(item.query))
            XCTAssertEqual(rendered.text, item.prompt, item.name)
            XCTAssertEqual(rendered.ids, item.candidateIDs, item.name)
        }
    }

    func testShippedCalibrationMatchesUpstreamEngine() throws {
        let (fixture, calibratorData) = try load()
        XCTAssertEqual(fixture.calibratorSHA256, "af2a876993148efa0726b6ccf710fe2303897d20c0ce8c7c9036eb50f64d23de")
        let calibrator = try VerdictCalibrator(data: calibratorData)
        for item in fixture.cases {
            let temperature = calibrator.temperature(candidates: item.candidateIDs.count, calibration: .shipped)
            XCTAssertEqual(temperature, item.temperature, accuracy: 1e-6, item.name)
            let probabilities = try VerdictCalibrator.probabilities(logits: item.logits, temperature: temperature)
            for (ours, upstream) in zip(probabilities, item.probabilities) {
                XCTAssertEqual(ours, upstream, accuracy: 1e-5, item.name)
            }
        }
        // K=8 has no per-K entry: the engine falls back to the global temperature.
        XCTAssertEqual(calibrator.temperature(candidates: 8, calibration: .shipped), 2.8039, accuracy: 1e-6)
    }

    func testExplicitCalibrationChoices() throws {
        let (fixture, calibratorData) = try load()
        let calibrator = try VerdictCalibrator(data: calibratorData)
        let item = try XCTUnwrap(fixture.cases.first { $0.name == "choice-k5" })
        XCTAssertEqual(calibrator.temperature(candidates: 5, calibration: .uncalibrated), 1)
        XCTAssertEqual(calibrator.temperature(candidates: 5, calibration: .temperature(1.4)), 1.4)
        let raw = try VerdictCalibrator.probabilities(logits: item.logits, temperature: 1)
        let shipped = try VerdictCalibrator.probabilities(logits: item.logits, temperature: item.temperature)
        XCTAssertEqual(raw.firstIndex(of: raw.max()!), shipped.firstIndex(of: shipped.max()!))
        XCTAssertGreaterThan(raw.max()!, shipped.max()!)
    }
}
