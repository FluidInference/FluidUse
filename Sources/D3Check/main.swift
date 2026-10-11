import CoreGraphics
import FluidUse
import Foundation
import ImageIO

/// d3 checks.
///
///     swift run -c release D3Check example lite|nano|mini     (downloads the pinned snapshot, answers the card request)
///     swift run -c release D3Check parity <model dir> <fixtures dir>   (marked screenshots vs the Python Core ML host)
///
/// Fixtures: model-lab models/d3-demos/browser/make_swift_fixtures.py (token ids + probabilities per screenshot).
@main
struct D3Check {
    struct VisionFixtures: Decodable { let fixtures: [VisionFixture] }

    struct VisionFixture: Decodable {
        struct Option: Decodable {
            let key: String
            let description: String
        }
        let image: String
        let state: String
        let instructions: String
        let options: [Option]
        let ids: [Int]
        let probabilities: [Double]
    }

    @available(macOS 15.0, *)
    static func parity(directory: URL, fixtures folder: URL) async throws {
        let fixtures = try JSONDecoder().decode(
            VisionFixtures.self, from: Data(contentsOf: folder.appendingPathComponent("fixtures.json"))
        ).fixtures
        let started = Date()
        let manager = try await D3Manager.load(from: directory)
        print(String(format: "loaded in %.1f s", Date().timeIntervalSince(started)))
        var worst = 0.0
        var mismatches = 0
        var flips = 0
        for (index, fixture) in fixtures.enumerated() {
            guard let source = CGImageSourceCreateWithURL(folder.appendingPathComponent(fixture.image) as CFURL, nil),
                let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
            else { throw D3Error.invalidAsset(fixture.image) }
            let question = D3Question.choice(
                fixture.instructions, options: fixture.options.map { ($0.key, $0.description) })
            for pass in 0..<(index == 0 ? 2 : 1) {  // the first call per bucket also pays the GPU setup
                let encoded = try await manager.encode(image)
                let ids = try await manager.tokens(state: .string(fixture.state), question: question, images: [encoded])
                let answer = try await manager.answer(
                    state: .string(fixture.state), images: [encoded], questions: [("click", question)])[0]
                guard pass == (index == 0 ? 1 : 0) else { continue }
                if ids != fixture.ids { mismatches += 1 }
                let d = zip(answer.probabilities, fixture.probabilities).map { abs($0 - $1) }.max() ?? 0
                worst = max(worst, d)
                let theirs = fixture.probabilities.indices.max { fixture.probabilities[$0] < fixture.probabilities[$1] }
                let ours = answer.probabilities.indices.max { answer.probabilities[$0] < answer.probabilities[$1] }
                if theirs != ours { flips += 1 }
                print(
                    String(
                        format: "%@: %d tokens (%@), choice %@, max|dp| %.4f, vision %.0f ms + text %.0f ms",
                        fixture.image, ids.count,
                        ids == fixture.ids ? "ids match" : "IDS DIFFER (\(fixture.ids.count))",
                        answer.best, d, encoded.milliseconds, answer.milliseconds))
            }
        }
        print(
            "\(fixtures.count) screenshots: token mismatches \(mismatches), flips \(flips), max|dp| \(String(format: "%.4f", worst))"
        )
        if mismatches > 0 || flips > 0 { exit(1) }
    }

    @available(macOS 15.0, *)
    static func example(_ model: D3Model) async throws {
        let directory = try await D3ModelStore.ensure(model) { file, bytes in
            if bytes > 0 { print("downloaded \(file) (\(bytes / 1_000_000) MB)") }
        }
        let manager = try await D3Manager.load(from: directory)
        let state = OrderedJSON.string(
            "The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.")
        let questions: [(id: String, question: D3Question)] = [
            (
                "route",
                .choice(
                    "Which team should handle this request?",
                    options: [
                        ("returns", "Refunds, replacements and damaged deliveries"),
                        ("billing", "Payments, invoices and charges"), ("technical", "Product setup and faults"),
                    ])
            ),
            ("receipt", .noul("Does the customer have a receipt?")),
            ("urgency", .score("How urgent is this request?", levels: ["Routine", "Soon", "Today"])),
        ]
        _ = try await manager.answer(state: state, images: [], questions: questions)  // first call pays the setup
        let started = Date()
        let answers = try await manager.answer(state: state, images: [], questions: questions)
        let ms = Date().timeIntervalSince(started) * 1000
        for a in answers {
            let detail =
                a.kind == "noul"
                ? String(format: "P(true) %.3f", a.yes)
                : a.kind == "score" ? String(format: "score %.2f", a.score) : a.best
            print("\(a.id): \(detail)  \(zip(a.keys, a.probabilities).map { "\($0) \(String(format: "%.3f", $1))" })")
        }
        print(String(format: "%@: 3 questions in %.0f ms", manager.modelName, ms))
    }

    static func main() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard #available(macOS 15.0, *) else { fatalError("macOS 15 required") }
        if arguments.first == "parity", arguments.count == 3 {
            try await parity(
                directory: URL(fileURLWithPath: arguments[1]), fixtures: URL(fileURLWithPath: arguments[2]))
        } else if arguments.first == "example", arguments.count == 2,
            let model = D3Model(rawValue: "d3-\(arguments[1])-coreml")
        {
            try await example(model)
        } else {
            fputs("usage: D3Check example lite|nano|mini | parity <model dir> <fixtures dir>\n", stderr)
            exit(2)
        }
    }
}
