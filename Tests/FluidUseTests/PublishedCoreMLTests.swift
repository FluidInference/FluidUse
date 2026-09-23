import Foundation
import XCTest

@testable import FluidUse

final class PublishedCoreMLTests: XCTestCase {
    private let kaiExample = SystemOneRequest(
        state: .object([.init("message", "Please refund the duplicate charge. I need this fixed today.")]),
        questions: [
            .noul("refund_requested", "Does the customer explicitly request a refund?"),
            .choice(
                "team", "Which team should handle this request?",
                options: [
                    DecisionOption("Billing", "Charges and refunds"), DecisionOption("Support", "Technical problems"),
                ]
            ),
            .score("urgency", "How urgent is the request?", levels: ["No deadline", "Needed soon", "Needed today"]),
        ])

    func testJSONKeepsMemberOrderAndEscapes() throws {
        let value: PublishedJSON = .object([
            .init("z", 1), .init("a", .array([true, nil, 2.5])), .init("text", "line\nquote\"tab\t\u{1}é"),
        ])
        let text = String(decoding: try value.encoded(), as: UTF8.self)
        XCTAssertEqual(text, #"{"z":1,"a":[true,null,2.5],"text":"line\nquote\"tab\t\u0001é"}"#)
        let parsed = try JSONSerialization.jsonObject(with: try value.encoded()) as? [String: Any]
        XCTAssertEqual(parsed?["text"] as? String, "line\nquote\"tab\t\u{1}é")
        XCTAssertThrowsError(try PublishedJSON.number(.nan).encoded())
        XCTAssertThrowsError(try PublishedJSON.object([.init("a", 1), .init("a", 2)]).encoded())
    }

    func testSystemOneRequestMatchesPublishedKaiExample() throws {
        let text = String(decoding: try kaiExample.json(defaultModel: "Decision-1.0-Kai").encoded(), as: UTF8.self)
        XCTAssertEqual(
            text,
            #"{"model":"Decision-1.0-Kai","state":{"message":"Please refund the duplicate charge. I need this fixed today."},"#
                + #""questions":{"refund_requested":{"type":"noul","instructions":"Does the customer explicitly request a refund?"},"#
                + #""team":{"type":"choice","instructions":"Which team should handle this request?","#
                + #""criteria":{"Billing":"Charges and refunds","Support":"Technical problems"}},"#
                + #""urgency":{"type":"score","instructions":"How urgent is the request?","#
                + #""criteria":["No deadline","Needed soon","Needed today"]}}}"#)
    }

    func testSystemOneNoulCriteriaAndNullDescriptions() throws {
        let request = SystemOneRequest(
            state: "s",
            questions: [
                .noul("q", "Is it paid?", trueCriterion: "The balance is zero"),
                .choice("c", "Pick", options: [DecisionOption("a"), DecisionOption("b", "B")]),
            ], model: "kev-latest")
        XCTAssertEqual(
            String(decoding: try request.json(defaultModel: "unused").encoded(), as: UTF8.self),
            #"{"model":"kev-latest","state":"s","questions":{"q":{"type":"noul","instructions":"Is it paid?","#
                + #""criteria":{"true":"The balance is zero"}},"c":{"type":"choice","instructions":"Pick","#
                + #""criteria":{"a":null,"b":"B"}}}}"#)
        let duplicate = SystemOneRequest(state: "s", questions: [.noul("q", "a"), .noul("q", "b")])
        XCTAssertThrowsError(try duplicate.json(defaultModel: "m"))
    }

    func testSystemOneResponseFollowsRequestOrder() throws {
        let reply = Data(
            #"""
            {"model":"Decision-1.0-Kai","answers":{
              "urgency":{"type":"score","probabilities":{"0":0.1,"1":0.3,"2":0.6},"confidence":0.6,"score":1.5,
                "legend":{"0":"No deadline","1":"Needed soon","2":"Needed today"}},
              "team":{"type":"choice","probabilities":{"Support":0.2,"Billing":0.8},"confidence":0.8,"choice":"Billing"},
              "refund_requested":{"type":"noul","noul":0.97}},
             "usage":{"input_tokens":88,"output_tokens":0}}
            """#.utf8)
        let response = try SystemOneResponse(json: reply, request: kaiExample)
        XCTAssertEqual(response.answers.map(\.id), ["refund_requested", "team", "urgency"])
        XCTAssertEqual(response["refund_requested"], .noul(probabilityTrue: 0.97))
        XCTAssertEqual(
            response["team"],
            .choice(
                selected: "Billing", confidence: 0.8,
                probabilities: [.init(key: "Billing", probability: 0.8), .init(key: "Support", probability: 0.2)]))
        XCTAssertEqual(
            response["urgency"], .score(expectedLevel: 1.5, confidence: 0.6, probabilities: [0.1, 0.3, 0.6]))
        XCTAssertEqual(response.inputTokens, 88)
    }

    func testConstrainedSchemaAndDecision() throws {
        let fields: [ConstrainedField] = [.oneOf("route", ["billing", "technical", "sales"]), .boolean("urgent")]
        XCTAssertEqual(
            String(decoding: try ConstrainedField.schema(fields).encoded(), as: UTF8.self),
            #"{"type":"object","properties":{"route":{"type":"string","enum":["billing","technical","sales"]},"#
                + #""urgent":{"type":"boolean"}},"required":["route","urgent"],"additionalProperties":false}"#)
        let reply = Data(
            #"""
            {"text":"{\"route\": \"billing\", \"urgent\": true}",
             "scores":{"route":[{"value":"billing","log_likelihood":-1.5},{"value":"technical","log_likelihood":-4.0},
                                {"value":"sales","log_likelihood":-6.0}],
                       "urgent":[{"value":true,"log_likelihood":-0.5},{"value":false,"log_likelihood":-2.0}]},
             "branches":5,"model_calls":1}
            """#.utf8)
        let decision = try ConstrainedDecision(json: reply, fields: fields)
        XCTAssertEqual(decision["route"], .string("billing"))
        XCTAssertEqual(decision["urgent"], .bool(true))
        XCTAssertEqual(
            decision.candidates[1].scores,
            [.init(value: .bool(true), logLikelihood: -0.5), .init(value: .bool(false), logLikelihood: -2.0)])
        XCTAssertThrowsError(try ConstrainedField.schema([.boolean("a"), .boolean("a")]))
        let outside = Data(
            #"{"text":"{\"route\": \"other\", \"urgent\": 1}","scores":{},"branches":1,"model_calls":1}"#.utf8)
        XCTAssertThrowsError(try ConstrainedDecision(json: outside, fields: fields))
    }

    func testNanoJevRequestShape() throws {
        let question = NanoJevQuestion(
            id: "action", instructions: "Choose the next action.",
            kind: .choice([DecisionOption("left", "Move the aim left."), DecisionOption("noop")]))
        XCTAssertEqual(
            String(decoding: try question.request(state: "The target is left.").encoded(), as: UTF8.self),
            #"{"states":[{"id":"request","state":"The target is left.","questions":{"action":{"type":"choice","#
                + #""instructions":"Choose the next action.","criteria":{"left":"Move the aim left.","noop":"noop"}}}}]}"#
        )
    }

    func testManifestPinsEveryDownloadableModel() throws {
        for model in PublishedCoreMLModel.allCases {
            guard model.isDownloadable else {
                XCTAssertThrowsError(try PublishedCoreMLModelStore.manifest(for: model))
                continue
            }
            let manifest = try PublishedCoreMLModelStore.manifest(for: model)
            XCTAssertEqual(manifest.revision.count, 40, model.rawValue)
            XCTAssertTrue(
                manifest.files.allSatisfy {
                    $0.sha256.count == 64 && $0.sha256.allSatisfy(\.isHexDigit) && $0.size > 0
                },
                model.rawValue)
            XCTAssertTrue(manifest.files.contains { $0.path == model.rootMarker }, model.rawValue)
            let project = model.projectDirectory.map { $0.isEmpty ? "" : $0 + "/" } ?? ""
            XCTAssertTrue(manifest.files.contains { $0.path == project + "uv.lock" }, model.rawValue)
            for precision in model.precisions {
                let selected = try PublishedCoreMLModelStore.selectedFiles(for: model, precision: precision)
                let packages = Set(
                    selected.compactMap { file -> String? in
                        guard let range = file.path.range(of: ".mlpackage/") else { return nil }
                        return String(file.path[..<range.lowerBound]) + ".mlpackage"
                    })
                XCTAssertEqual(packages, Set(try model.requiredPackages(precision: precision)), "\(model) \(precision)")
            }
        }
    }

    func testSelectedFilesSkipOtherPrecisions() throws {
        let fp16 = try PublishedCoreMLModelStore.selectedFiles(for: .kai, precision: "fp16").map(\.path)
        XCTAssertTrue(fp16.contains("coreml/choice.mlpackage/Data/com.apple.CoreML/weights/weight.bin"))
        XCTAssertFalse(fp16.contains { $0.contains("embedding-w8") })
        let lexW8 = try PublishedCoreMLModelStore.selectedFiles(for: .lex, precision: "w8").map(\.path)
        XCTAssertTrue(lexW8.contains("coreml/choice.mlpackage/Manifest.json"))
        XCTAssertTrue(lexW8.contains("coreml/score-embedding-w8.mlpackage/Manifest.json"))
        XCTAssertFalse(lexW8.contains("coreml/score.mlpackage/Manifest.json"))
        XCTAssertThrowsError(try PublishedCoreMLModelStore.selectedFiles(for: .lfm350, precision: "w8"))
    }

    func testManifestMatchCheckUsesSizeAndDigest() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        try Data("abc".utf8).write(to: file)
        let digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        XCTAssertTrue(try PublishedCoreMLModelStore.matches(file, .init(path: "x", size: 3, sha256: digest)))
        XCTAssertFalse(try PublishedCoreMLModelStore.matches(file, .init(path: "x", size: 4, sha256: digest)))
        XCTAssertFalse(
            try PublishedCoreMLModelStore.matches(
                file, .init(path: "x", size: 3, sha256: String(repeating: "0", count: 64))))
    }

    func testLineChannelSplitsChunksAndEnds() async throws {
        let channel = LineChannel()
        channel.receive(Data("rea".utf8))
        channel.receive(Data("dy\nok {}\npartial".utf8))
        let first = try await channel.next(timeout: .seconds(5), waitingFor: "test")
        let second = try await channel.next(timeout: .seconds(5), waitingFor: "test")
        XCTAssertEqual(first, Data("ready".utf8))
        XCTAssertEqual(second, Data("ok {}".utf8))
        channel.receive(Data())
        let end = try await channel.next(timeout: .seconds(5), waitingFor: "test")
        XCTAssertNil(end)
    }

    func testLineChannelDeliversToWaitingReader() async throws {
        let channel = LineChannel()
        async let line = channel.next(timeout: .seconds(5), waitingFor: "test")
        try await Task.sleep(for: .milliseconds(50))
        channel.receive(Data("late\n".utf8))
        let value = try await line
        XCTAssertEqual(value, Data("late".utf8))
    }

    func testLineChannelTimesOut() async throws {
        let channel = LineChannel()
        do {
            _ = try await channel.next(timeout: .milliseconds(20), waitingFor: "nothing")
            XCTFail("Expected a timeout")
        } catch {
            XCTAssertEqual(error as? PublishedCoreMLError, .timedOut("nothing"))
        }
    }

    func testLineChannelHonorsCancellation() async throws {
        let channel = LineChannel()
        let task = Task { try await channel.next(timeout: .seconds(30), waitingFor: "nothing") }
        try await Task.sleep(for: .milliseconds(20))
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch {
            XCTAssertTrue(error is CancellationError)
        }
    }

    func testLineChannelRejectsOversizedMessages() async throws {
        let channel = LineChannel(limit: 8)
        channel.receive(Data("0123456789".utf8))
        do {
            _ = try await channel.next(timeout: .seconds(5), waitingFor: "test")
            XCTFail("Expected an oversized-message error")
        } catch {
            XCTAssertEqual(error as? PublishedCoreMLError, .invalidResponse("Worker message exceeded 8 bytes"))
        }
    }

    func testFamiliesAndModelNames() {
        XCTAssertEqual(PublishedCoreMLModel.kai.systemOneModelName, "Decision-1.0-Kai")
        XCTAssertEqual(PublishedCoreMLModel.kev06.family, .systemOne)
        XCTAssertNil(PublishedCoreMLModel.jeff.systemOneModelName)
        XCTAssertEqual(PublishedCoreMLModel.kev06.repository, "FluidInference/kev-0.6b-coreml")
        XCTAssertThrowsError(try PublishedCoreMLModel.kev05.requiredPackages(precision: "w8"))
        XCTAssertEqual(
            try PublishedCoreMLModel.lex.requiredPackages(precision: "w8"),
            ["coreml/choice.mlpackage", "coreml/noul-embedding-w8.mlpackage", "coreml/score-embedding-w8.mlpackage"])
    }
}
