import AppKit
import FluidUse
import Foundation

// On-device Guardrail: Vela-2.0-0.3B on Core ML (Neural Engine for ≤128-token checks, GPU above).
//
//     swift run -c release GuardrailDemo [--model <dir>]     (or VELA_DIR=<dir>)
//     swift run -c release GuardrailDemo --selftest           (no window: runs every scenario through all lanes)

setvbuf(stdout, nil, _IOLBF, 0)

enum Launch {
    static let modelDirectory: URL = {
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--model"), i + 1 < args.count { return URL(fileURLWithPath: args[i + 1]) }
        if let env = ProcessInfo.processInfo.environment["VELA_DIR"], !env.isEmpty { return URL(fileURLWithPath: env) }
        return URL(fileURLWithPath: "/Users/hanweng/Documents/vela2/release03")
    }()
    static let selftest = CommandLine.arguments.contains("--selftest")
}

if #available(macOS 15.0, *) {
    if Launch.selftest {
        Task.detached {
            do {
                try await SelfTest.run(directory: Launch.modelDirectory)
                exit(0)
            } catch {
                fputs("selftest failed: \(error)\n", stderr)
                exit(1)
            }
        }
        dispatchMain()
    } else {
        GuardrailApp.main()
    }
} else {
    fputs("GuardrailDemo needs macOS 15 or later\n", stderr)
    exit(1)
}

@available(macOS 15.0, *)
enum SelfTest {
    static func run(directory: URL) async throws {
        print("loading \(directory.path)")
        let m = try await Vela2Manager.load(from: directory)
        try await m.warm()
        print("loaded \(m.modelName); ANE for sequences ≤ \(m.aneMaxLength) tokens, GPU above")
        // The first screen pass per message warms the per-text token cache (as the app's earlier keystrokes would).
        var screens: [Timing] = [], piis: [Timing] = [], sends: [Timing] = [], replies: [Timing] = []
        var checks = 0, onANE = 0
        func count(_ t: Timing) {
            checks += 1
            if t.onNeuralEngine { onANE += 1 }
        }
        for sc in Scenario.all {
            print("\n== \(sc.name)")
            var lastUser = ""
            for msg in sc.suggest {
                print("» \(msg)")
                _ = try await Guard.screen(m, msg)
                let s = try await Guard.screen(m, msg)
                print(Guard.log(s)); screens.append(s.timing); count(s.timing)
                let p = try await Guard.pii(m, msg)
                print(Guard.log(p)); piis.append(p.timing); count(p.timing)
                let f = try await Guard.send(m, msg)
                print(Guard.log(f)); sends.append(f.timing); count(f.timing)
                if !f.blocked {
                    lastUser = f.masked
                    print("           masked: \(f.masked)")
                }
            }
            if !lastUser.isEmpty {
                let r = try await Guard.reply(m, request: lastUser, source: sc.source, answer: sc.reply)
                print(Guard.log(r)); replies.append(r.timing); count(r.timing)
            }
        }
        func summary(_ name: String, _ ts: [Timing]) {
            let enc = ts.map(\.encoderMs).sorted(), tot = ts.map(\.totalMs).sorted()
            guard !enc.isEmpty else { return }
            print(String(format: "%@ %d checks, on ANE %d/%d, encoder p50 %.1f ms (max %.1f), total p50 %.1f ms",
                         name.padding(toLength: 8, withPad: " ", startingAt: 0), ts.count,
                         ts.filter(\.onNeuralEngine).count, ts.count, enc[enc.count / 2], enc.last!, tot[tot.count / 2]))
        }
        print("\n== summary")
        summary("live", screens)
        summary("pii", piis)
        summary("send", sends)
        summary("reply", replies)
        print("checks: \(checks) · on ANE: \(onANE)")
    }
}
