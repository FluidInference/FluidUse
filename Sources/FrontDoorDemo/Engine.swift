import FluidUse
import Foundation

// MARK: - Check results

/// Where a check ran and how long it took.
struct Timing: Sendable, Equatable, Hashable {
    let onNeuralEngine: Bool
    let encoderMs: Double
    let totalMs: Double
    let buckets: [Int]
    /// Token count of each encoder sequence the check ran as.
    let tokens: [Int]

    init(_ r: Vela2Result, tokens: [Int]) {
        onNeuralEngine = r.onNeuralEngine
        encoderMs = r.encoderMs
        totalMs = r.totalMs
        buckets = r.buckets
        self.tokens = tokens
    }

    /// "92→128" (tokens → bucket), joined with "+" for multi-sequence checks.
    var shape: String { zip(tokens, buckets).map { "\($0)→\($1)" }.joined(separator: " + ") }
    var shortUnit: String { onNeuralEngine ? "ANE" : "GPU" }
    var longUnit: String { onNeuralEngine ? "Neural Engine" : "GPU" }
}

struct PIISpan: Sendable, Equatable, Hashable {
    let label: String
    let start: Int  // Unicode scalar offsets
    let end: Int
    let text: String
    let probability: Double
}

struct TopicP: Sendable, Hashable {
    let name: String
    let p: Double
}

struct GuardCheck: Sendable, Hashable {
    let attack: Double  // P(jailbreak)
    let harm: Double  // P(unsafe)
    let timing: Timing
}

struct RouteCheck: Sendable, Hashable {
    let team: String
    let topics: [TopicP]  // option order
    let pii: [PIISpan]
    let timing: Timing
}

enum Verdict: Sendable, Hashable {
    case answered(team: String)
    case jailbreak
    case harmful

    /// `ok` | `jailbreak` | `harmful` (messages.py labels).
    var label: String {
        switch self {
        case .answered: "ok"
        case .jailbreak: "jailbreak"
        case .harmful: "harmful"
        }
    }
}

struct Processed: Sendable, Identifiable, Hashable {
    let message: InboundMessage
    let arrived: Date
    let guardCheck: GuardCheck
    let route: RouteCheck?
    let verdict: Verdict

    var id: Int { message.id }
    var pii: [PIISpan] { route?.pii ?? [] }
    var masked: String { maskPII(message.text, pii) }
    var blocked: Bool { route == nil }
    var timings: [Timing] { [guardCheck.timing] + (route.map { [$0.timing] } ?? []) }
    /// Both calls end to end (tokenize + encoder + heads).
    var totalMs: Double { timings.reduce(0) { $0 + $1.totalMs } }
}

/// Replaces each span with `[LABEL]`, on Unicode scalars.
func maskPII(_ text: String, _ spans: [PIISpan]) -> String {
    let s = Array(text.unicodeScalars)
    var out = String.UnicodeScalarView()
    var last = 0
    for sp in spans.sorted(by: { $0.start < $1.start }) where sp.start >= last && sp.end <= s.count {
        out.append(contentsOf: s[last..<sp.start])
        out.append(contentsOf: "[\(sp.label)]".unicodeScalars)
        last = sp.end
    }
    if last < s.count { out.append(contentsOf: s[last...]) }
    return String(out)
}

// MARK: - The front door

enum Policy {
    /// Attack / harm probability that blocks a message.
    static let flag = 0.8
}

@available(macOS 15.0, *)
enum Engine {
    static func spans(_ r: Vela2Result, _ id: String) -> [PIISpan] {
        (r[span: id]?.spans ?? []).map { PIISpan(label: $0.label, start: $0.start, end: $0.end, text: $0.text, probability: $0.probability) }
            .sorted { $0.start < $1.start }
    }

    static func run(_ m: Vela2Manager, _ text: String, _ qs: [Vela2Question]) async throws -> (Vela2Result, Timing) {
        let parts = [Vela2Part("user", text)]
        let r = try await m.predict(parts: parts, questions: qs)
        let tokens = (try? m.sequences(parts: parts, questions: qs).map(\.count)) ?? []
        return (r, Timing(r, tokens: tokens))
    }

    static func guardCheck(_ m: Vela2Manager, _ text: String) async throws -> GuardCheck {
        let (r, t) = try await run(m, text, Questions.guardCall)
        return GuardCheck(
            attack: r[choice: "attack"]?.probability("jailbreak") ?? 0,
            harm: r[choice: "p_harm"]?.probability("unsafe") ?? 0, timing: t)
    }

    static func routeCheck(_ m: Vela2Manager, _ text: String) async throws -> RouteCheck {
        let (r, t) = try await run(m, text, Questions.routeCall)
        let c = r[choice: "topic"]
        let topics = c.map { zip($0.names, $0.probabilities).map { TopicP(name: $0.0, p: $0.1) } } ?? []
        return RouteCheck(team: c?.answer ?? "general", topics: topics, pii: spans(r, "pii"), timing: t)
    }

    /// Guard first (ANE-sized), then — only if it passes — route + PII.
    static func process(_ m: Vela2Manager, _ msg: InboundMessage, arrived: Date) async throws -> Processed {
        let g = try await guardCheck(m, msg.text)
        if g.attack >= Policy.flag {
            return Processed(message: msg, arrived: arrived, guardCheck: g, route: nil, verdict: .jailbreak)
        }
        if g.harm >= Policy.flag {
            return Processed(message: msg, arrived: arrived, guardCheck: g, route: nil, verdict: .harmful)
        }
        let r = try await routeCheck(m, msg.text)
        return Processed(message: msg, arrived: arrived, guardCheck: g, route: r, verdict: .answered(team: r.team))
    }

    // MARK: log lines

    static func unit(_ t: Timing, _ lane: String) -> String {
        String(format: "%5.1f ms %@ %@", t.encoderMs, t.shortUnit, lane)
    }

    static func log(_ p: Processed) -> String {
        var s = unit(p.guardCheck.timing, "guard")
        switch p.verdict {
        case .jailbreak:
            s += String(format: "  → BLOCKED jailbreak (%.2f)", p.guardCheck.attack)
        case .harmful:
            s += String(format: "  → BLOCKED harmful (%.2f)", p.guardCheck.harm)
        case .answered(let team):
            s += "  " + unit(p.route!.timing, "route") + "  → \(team)"
            if !p.pii.isEmpty { s += "  🔒[\(p.pii.map(\.label).joined(separator: ", "))]" }
        }
        let flat = p.message.text.replacingOccurrences(of: "\n", with: " ")
        let preview = flat.count > 48 ? String(flat.prefix(47)) + "…" : flat
        return s + "  | " + preview
    }
}

/// Totals over processed messages (shared by the UI header, final banner and the selftest).
struct Tally: Equatable {
    var processed = 0, answered = 0, jailbreak = 0, harmful = 0, withPII = 0
    var checks = 0, onANE = 0
    var totalMs = 0.0
    var byTeam: [String: Int] = [:]

    init() {}

    init(_ ps: [Processed]) {
        for p in ps { add(p) }
    }

    mutating func add(_ p: Processed) {
        processed += 1
        switch p.verdict {
        case .answered(let t):
            answered += 1
            byTeam[t, default: 0] += 1
        case .jailbreak: jailbreak += 1
        case .harmful: harmful += 1
        }
        if !p.pii.isEmpty { withPII += 1 }
        for t in p.timings {
            checks += 1
            if t.onNeuralEngine { onANE += 1 }
        }
        totalMs += p.totalMs
    }

    var blocked: Int { jailbreak + harmful }
    var avgMs: Double { processed > 0 ? totalMs / Double(processed) : 0 }
    /// Sequential on-device throughput at that average (model time only).
    var perSecond: Double { avgMs > 0 ? 1000 / avgMs : 0 }
    var aneShare: Double { checks > 0 ? Double(onANE) / Double(checks) : 0 }

    var summary: String {
        let teams = Questions.teams.map { "\($0.name) \(byTeam[$0.name] ?? 0)" }.joined(separator: ", ")
        return String(
            format: "summary: %d messages · answered %d (%@) · blocked %d (jailbreak %d, harmful %d) · with PII %d · avg %.1f ms/msg model time (≈%.0f msg/s back to back) · checks on Neural Engine %d of %d",
            processed, answered, teams, blocked, jailbreak, harmful, withPII, avgMs, perSecond, onANE, checks)
    }
}
