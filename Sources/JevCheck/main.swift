import CoreGraphics
import FluidUse
import Foundation
import ImageIO

/// JEV-9B Core ML checks against the Python fixtures (model-lab `jev-9b-coreml/conversion/export_fixtures.py`).
///
///     swift run -c release JevCheck parity <fixtures dir> <model dir>     # token ids, positions, probabilities
///     swift run -c release JevCheck latency <fixtures dir> <model dir>    # warm timings, one image + 3 questions
@main
@available(macOS 15.0, *)
struct JevCheck {
    struct Fixtures: Decodable {
        struct Record: Decodable {
            let state: [Part]
            let kind: String
            let question: String
            let options: [String]?
            let inputIds: [Int]
            let positionIds: [[Int]]
            let probabilities: [Float]
        }
        enum Part: Decodable {
            case text(String)
            case image(String)
            init(from decoder: Decoder) throws {
                let c = try decoder.singleValueContainer()
                if let s = try? c.decode(String.self) {
                    self = .text(s)
                } else {
                    self = .image(try c.decode([String: String].self)["image"] ?? "")
                }
            }
        }
        let bucket: Int
        let records: [Record]
    }

    static func image(_ url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw JevError.invalidInput("cannot read \(url.path)") }
        return image
    }

    static func question(_ r: Fixtures.Record) -> JevQuestion {
        switch r.kind {
        case "noul": return JevQuestion(.noul, r.question)
        case "score": return JevQuestion(.score, r.question)
        default: return JevQuestion(.choice(r.options ?? []), r.question)
        }
    }

    static func state(_ r: Fixtures.Record, in dir: URL) throws -> [JevStatePart] {
        try r.state.map {
            switch $0 {
            case .text(let s): return .text(s)
            case .image(let name): return .image(try image(dir.appendingPathComponent(name)))
            }
        }
    }

    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count >= 4 else {
            print("usage: JevCheck parity|latency <fixtures dir> <model dir>")
            exit(2)
        }
        let fixturesDir = URL(fileURLWithPath: args[2])
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let fixtures = try decoder.decode(
            Fixtures.self, from: Data(contentsOf: fixturesDir.appendingPathComponent("fixtures.json")))
        let loadStarted = Date()
        let manager = try await JevManager.load(from: URL(fileURLWithPath: args[3]), bucket: fixtures.bucket)
        print(String(format: "loaded in %.1f s", Date().timeIntervalSince(loadStarted)))
        switch args[1] {
        case "parity":
            var idFailures = 0
            var flips = 0
            var maxDrift: Float = 0
            for (i, r) in fixtures.records.enumerated() {
                let st = try state(r, in: fixturesDir)
                let q = question(r)
                let (ids, positions) = try await manager.tokens(state: st, question: q)
                let idsOK = ids == r.inputIds && positions == r.positionIds
                if !idsOK { idFailures += 1 }
                let result = try await manager.decide(state: st, questions: [q])
                let p = result.decisions[0].probabilities
                let drift = zip(p, r.probabilities).map { abs($0 - $1) }.max() ?? 0
                maxDrift = max(maxDrift, drift)
                let argmax = { (v: [Float]) in v.indices.max { v[$0] < v[$1] }! }
                let sorted = r.probabilities.sorted(by: >)
                let nearTie = sorted.count > 1 && sorted[0] - sorted[1] < 0.02  // either answer is within noise
                if argmax(p) != argmax(r.probabilities) && !nearTie { flips += 1 }
                print(
                    String(
                        format: "%2d %-6@ tokens %3d %@  drift %.4f  swift %@  python %@", i, r.kind as NSString,
                        ids.count, idsOK ? "ids OK " : "ids MISMATCH", drift,
                        p.map { String(format: "%.3f", $0) }.joined(separator: " "),
                        r.probabilities.map { String(format: "%.3f", $0) }.joined(separator: " ")))
            }
            print(
                "records \(fixtures.records.count): token/position mismatches \(idFailures), argmax flips (python margin"
                    + " >= 0.02) \(flips), max prob drift \(maxDrift)")
            if idFailures > 0 || flips > 0 { exit(1) }
        case "latency":
            guard
                let r = fixtures.records.first(where: {
                    $0.state.contains { if case .image = $0 { true } else { false } }
                })
            else { exit(1) }
            let st = try state(r, in: fixturesDir)
            let qs = [
                JevQuestion(.noul, "Is there an enemy in this picture?"),
                JevQuestion(.noul, "Is there a pipe in this picture?"),
                JevQuestion(.noul, "Is there a gap in this picture?"),
            ]
            _ = try await manager.decide(state: st, questions: qs)
            var totals: [Double] = []
            for _ in 0..<10 {
                let result = try await manager.decide(state: st, questions: qs)
                totals.append(result.totalMilliseconds)
                print(
                    String(
                        format: "vision %.0f ms  decoder %.0f ms  total %.0f ms", result.visionMilliseconds,
                        result.decoderMilliseconds, result.totalMilliseconds))
            }
            print(String(format: "median %.0f ms for 1 image + 3 questions", totals.sorted()[totals.count / 2]))
        default:
            print("unknown mode \(args[1])")
            exit(2)
        }
    }
}
