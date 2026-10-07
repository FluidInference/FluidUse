import AppKit
import FluidUse
import Foundation

// Chatbot Front Door: a support bot's incoming traffic screened and routed on this Mac by Vela-2.0-0.3B (Core ML).
//
//     swift run -c release FrontDoorDemo [--model <dir>]     (or VELA_DIR=<dir>)
//     swift run -c release FrontDoorDemo --demo               (starts the run by itself)
//     swift run -c release FrontDoorDemo --selftest           (no window: all 1,000 messages + model-vs-expected)

setvbuf(stdout, nil, _IOLBF, 0)

// Keep AppKit from treating command-line arguments (e.g. the `--model` path) as documents to open.
UserDefaults.standard.register(defaults: ["NSTreatUnknownArgumentsAsOpen": "NO"])

enum Launch {
    /// `--model <dir>` or `VELA_DIR`; otherwise the pinned snapshot from the Hub (`Vela2ModelStore`).
    static let modelDirectory: URL? = {
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--model"), i + 1 < args.count { return URL(fileURLWithPath: args[i + 1]) }
        if let env = ProcessInfo.processInfo.environment["VELA_DIR"], !env.isEmpty { return URL(fileURLWithPath: env) }
        return nil
    }()

    static func resolveModel() async throws -> URL {
        if let dir = modelDirectory { return dir }
        return try await Vela2ModelStore.ensure { file, bytes in
            if bytes > 0 { print("downloaded \(file) (\(bytes / 1_000_000) MB)") }
        }
    }
    static let selftest = CommandLine.arguments.contains("--selftest")
    static let demo = CommandLine.arguments.contains("--demo")
}

if #available(macOS 15.0, *) {
    if Launch.selftest {
        Task.detached {
            do {
                try await SelfTest.run(directory: Launch.resolveModel())
                exit(0)
            } catch {
                fputs("selftest failed: \(error)\n", stderr)
                exit(1)
            }
        }
        dispatchMain()
    } else {
        FrontDoorApp.main()
    }
} else {
    fputs("FrontDoorDemo needs macOS 15 or later\n", stderr)
    exit(1)
}

@available(macOS 15.0, *)
enum SelfTest {
    static func run(directory: URL) async throws {
        print("loading \(directory.path)")
        let m = try await Vela2Manager.load(from: directory)
        try await m.warm()
        _ = try await Engine.guardCheck(m, "hello")
        _ = try await Engine.routeCheck(m, "hello")
        print("loaded \(m.modelName); ANE for sequences ≤ \(m.aneMaxLength) tokens, GPU above")
        var done: [Processed] = []
        var confusion: [String: Int] = [:]
        var topicHit = 0, topicN = 0
        var debugMs = 0.0  // time spent on debug-only route calls for false-blocked benign messages (excluded from throughput)
        let started = ContinuousClock.now
        for msg in Traffic.all {
            let p = try await Engine.process(m, msg, arrived: Date())
            done.append(p)
            let got = p.verdict.label
            confusion["\(msg.expect)→\(got)", default: 0] += 1
            var ok = got == msg.expect
            var topicNote = ""
            if msg.expect == "ok" {
                // probe.py's metric: the topic answer for every benign message (a false-blocked one gets a debug-only route call)
                let team: String
                if let r = p.route {
                    team = r.team
                } else {
                    let d0 = ContinuousClock.now
                    team = try await Engine.routeCheck(m, msg.text).team
                    let d = ContinuousClock.now - d0
                    debugMs += Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15
                }
                topicN += 1
                if team == msg.expectTopic { topicHit += 1 } else { ok = false }
                if p.route == nil { topicNote = ", topic would be \(team)" }
            }
            let debug = String(format: "   [expected %@%@ · attack %.2f harm %.2f%@]", msg.expect, msg.expectTopic.map { "/\($0)" } ?? "",
                               p.guardCheck.attack, p.guardCheck.harm, topicNote)
            print((ok ? "   " : "XX ") + Engine.log(p) + (ok ? "" : debug))
        }
        let wall = ContinuousClock.now - started
        let wallS = Double(wall.components.seconds) + Double(wall.components.attoseconds) / 1e18 - debugMs / 1000
        let t = Tally(done)
        print(t.summary)
        print(String(format: "end-to-end: %d messages in %.2f s = %.1f msg/s (wall clock, no UI, incl. logging)", done.count, wallS, Double(done.count) / wallS))
        let order = ["ok", "jailbreak", "harmful"]
        print("model vs expected (rows = expected, cols = model: ok / jailbreak / harmful):")
        for e in order {
            print("  " + e.padding(toLength: 10, withPad: " ", startingAt: 0)
                  + order.map { String(format: "%4d", confusion["\(e)→\($0)"] ?? 0) }.joined())
        }
        print("topic (benign messages routed to the expected team, probe.py metric): \(topicHit)/\(topicN)")
        let g = done.map(\.guardCheck.timing.encoderMs).sorted()
        let r = done.compactMap(\.route?.timing.encoderMs).sorted()
        print(String(format: "guard encoder p50 %.1f ms (ANE %d/%d) · route encoder p50 %.1f ms (ANE %d/%d)",
                     g[g.count / 2], done.filter(\.guardCheck.timing.onNeuralEngine).count, done.count,
                     r.isEmpty ? 0 : r[r.count / 2], done.compactMap(\.route).filter(\.timing.onNeuralEngine).count, r.count))
    }
}
