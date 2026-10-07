import AppKit
import SwiftUI

enum Theme {
    static func hex(_ v: UInt32, _ a: Double = 1) -> Color {
        Color(.sRGB, red: Double((v >> 16) & 0xff) / 255, green: Double((v >> 8) & 0xff) / 255, blue: Double(v & 0xff) / 255, opacity: a)
    }
    static let bg = hex(0x0b0d12)
    static let panel = hex(0x13161e)
    static let panel2 = hex(0x1a1e29)
    static let line = hex(0x262b38)
    static let ink = hex(0xeef1f7)
    static let dim = hex(0x8b93a7)
    static let accent = hex(0x6e8bff)
    static let ok = hex(0x3ccf8e)
    static let bad = hex(0xff5d6c)
    static let warn = hex(0xffb547)
    static let pii = hex(0xf5a524)
    static let mark = hex(0xf5a524, 0.28)
    static let piiInk = hex(0xffd28f)
    static let ane = hex(0x3ccf8e)
    static let gpu = hex(0x8fa6ff)

    static func team(_ name: String) -> Color {
        switch name {
        case "billing": hex(0x6e8bff)
        case "technical": hex(0xb18cff)
        case "account": hex(0x2fd3c4)
        case "shipping": hex(0xff8fc7)
        default: hex(0xa5afc4)
        }
    }

    static func teamIcon(_ name: String) -> String {
        switch name {
        case "billing": "💳"
        case "technical": "🛠"
        case "account": "👤"
        case "shipping": "📦"
        default: "💬"
        }
    }

    static let avatarColors: [UInt32] = [0x6e8bff, 0xb18cff, 0x2fd3c4, 0xff8fc7, 0xf5a524, 0x3ccf8e, 0xff7a59, 0x5ec8ff]
    static func avatar(_ name: String) -> Color {
        hex(avatarColors[name.unicodeScalars.reduce(0) { ($0 &* 31 &+ Int($1.value)) & 0xffff } % avatarColors.count])
    }
}

// MARK: - small pieces

enum ChipStyle { case plain, ok, bad, warn, dim }

struct Chip: View {
    let text: String
    var style: ChipStyle = .plain
    var size: CGFloat = 11.5
    var body: some View {
        let (bg, fg): (Color, Color) = switch style {
        case .plain: (Color.white.opacity(0.08), Theme.ink)
        case .ok: (Theme.ok.opacity(0.16), Theme.hex(0x8ff0c4))
        case .bad: (Theme.bad.opacity(0.18), Theme.hex(0xffadb5))
        case .warn: (Theme.warn.opacity(0.16), Theme.piiInk)
        case .dim: (Color.white.opacity(0.08), Theme.dim)
        }
        Text(text).font(.system(size: size)).lineLimit(1).fixedSize()
            .padding(.horizontal, 7).padding(.vertical, 2)
            .background(bg, in: Capsule()).foregroundStyle(fg)
    }
}

/// "⚡ ANE · 3.6 ms" / "GPU · 6.1 ms" (encoder time; tooltip has the rest).
struct UnitBadge: View {
    let timing: Timing
    var what: String? = nil
    var detailed = false
    var body: some View {
        let color = timing.onNeuralEngine ? Theme.ane : Theme.gpu
        HStack(spacing: 4) {
            if let what { Text(what).opacity(0.75) }
            Text((timing.onNeuralEngine ? "⚡ " : "") + (detailed ? timing.longUnit : timing.shortUnit)
                 + String(format: " · %.1f ms", timing.encoderMs))
                .fontWeight(.semibold)
            if detailed {
                Text(String(format: "total %.1f ms · tokens→bucket %@", timing.totalMs, timing.shape)).opacity(0.7)
            }
        }
        .font(.system(size: detailed ? 11.5 : 10.5).monospacedDigit())
        .lineLimit(1).fixedSize()
        .padding(.horizontal, 7).padding(.vertical, 2)
        .foregroundStyle(color)
        .background(color.opacity(0.13), in: Capsule())
        .overlay(Capsule().stroke(color.opacity(0.35)))
        .help("Encoder \(String(format: "%.2f", timing.encoderMs)) ms on the \(timing.longUnit) (tokens → bucket: \(timing.shape)); whole check incl. tokenize + heads \(String(format: "%.2f", timing.totalMs)) ms")
    }
}

struct Avatar: View {
    let name: String
    var size: CGFloat = 30
    var body: some View {
        let initial = name.first(where: { $0.isLetter }).map { String($0).uppercased() } ?? "?"
        Text(initial)
            .font(.system(size: size * 0.45, weight: .bold))
            .foregroundStyle(Theme.bg)
            .frame(width: size, height: size)
            .background(Theme.avatar(name), in: Circle())
    }
}

struct Bar: View {
    let label: String
    let p: Double?
    let color: Color
    var body: some View {
        HStack(spacing: 8) {
            Text(label).font(.system(size: 13)).frame(width: 150, alignment: .leading).lineLimit(1)
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(Theme.panel2)
                    Capsule().fill(color).frame(width: g.size.width * (p ?? 0))
                }
            }
            .frame(height: 9)
            Text(p.map { "\(Int(($0 * 100).rounded()))%" } ?? "–").font(.system(size: 13).monospacedDigit())
                .foregroundStyle(Theme.dim).frame(width: 42, alignment: .trailing)
        }
    }
}

/// Masked text with `[LABEL]` tokens rendered as orange pills.
func maskedAttributed(_ s: String, size: CGFloat = 14) -> AttributedString {
    var out = AttributedString()
    var rest = Substring(s)
    while let open = rest.firstIndex(of: "["), let close = rest[open...].firstIndex(of: "]") {
        let label = rest[rest.index(after: open)..<close]
        if !label.isEmpty, label.allSatisfy({ $0.isUppercase || $0 == "_" }) {
            out += AttributedString(String(rest[..<open]))
            var tok = AttributedString("\u{2009}\(label)\u{2009}")
            tok.font = .system(size: size - 2, weight: .semibold)
            tok.foregroundColor = Theme.piiInk
            tok.backgroundColor = Theme.pii.opacity(0.22)
            out += tok
            rest = rest[rest.index(after: close)...]
        } else {
            out += AttributedString(String(rest[...open]))
            rest = rest[rest.index(after: open)...]
        }
    }
    out += AttributedString(String(rest))
    return out
}

/// The original text with each PII span highlighted and labelled (Unicode-scalar offsets).
func highlightedAttributed(_ text: String, _ spans: [PIISpan], size: CGFloat = 15) -> AttributedString {
    let s = Array(text.unicodeScalars)
    func str(_ r: Range<Int>) -> String { String(String.UnicodeScalarView(s[r])) }
    var out = AttributedString()
    var last = 0
    for sp in spans.sorted(by: { $0.start < $1.start }) where sp.start >= last && sp.end <= s.count {
        out += AttributedString(str(last..<sp.start))
        var hit = AttributedString(str(sp.start..<sp.end))
        hit.backgroundColor = Theme.mark
        hit.foregroundColor = Theme.piiInk
        out += hit
        var tag = AttributedString("\u{2009}\(sp.label)")
        tag.font = .system(size: size - 5, weight: .bold)
        tag.foregroundColor = Theme.pii
        tag.baselineOffset = 4
        out += tag
        last = sp.end
    }
    if last < s.count { out += AttributedString(str(last..<s.count)) }
    return out
}

func clock(_ d: Date) -> String { d.formatted(date: .omitted, time: .standard) }

// MARK: - main view

@available(macOS 15.0, *)
struct ContentView: View {
    @EnvironmentObject var model: FrontDoorModel

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(Theme.line)
            HStack(alignment: .top, spacing: 10) {
                InboxColumn().frame(width: 280)
                VStack(spacing: 10) {
                    AnsweredLane().frame(maxHeight: .infinity)
                    HStack(alignment: .top, spacing: 10) {
                        BlockedLane(kind: .jailbreak)
                        BlockedLane(kind: .harmful)
                    }
                    .frame(height: 230)
                }
            }
            .padding(12)
        }
        .background(Theme.bg)
        .foregroundStyle(Theme.ink)
        .preferredColorScheme(.dark)
        .overlayPreferenceValue(FlightSpotKey.self) { anchors in
            GeometryReader { g in
                FlightLayer(rects: anchors.mapValues { g[$0] })
            }
            .allowsHitTesting(false)
        }
        .sheet(item: $model.selected) { p in DetailSheet(p: p) }
    }

    // MARK: header

    var header: some View {
        let t = model.tally
        let has = t.processed > 0
        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Text("🚪 Chatbot Front Door").font(.system(size: 16, weight: .bold))
                Text("screened on this Mac by Vela-2.0-0.3B (Core ML) · nothing sent anywhere")
                    .font(.system(size: 11.5)).foregroundStyle(Theme.dim).lineLimit(1).truncationMode(.tail)
                Spacer(minLength: 6)
                controls
            }
            HStack(spacing: 6) {
                MiniStat(value: "\(t.processed)", unit: "/\(Traffic.all.count)", label: "processed", color: Theme.ink)
                MiniStat(value: "\(t.answered)", label: "answered", color: Theme.ok)
                MiniStat(value: "\(t.blocked)", label: "blocked", color: Theme.bad)
                MiniStat(value: "\(t.withPII)", label: "PII 🔒", color: Theme.pii)
                MiniStat(value: has ? String(format: "%.1f", t.avgMs) : "–", unit: "ms", label: "avg / msg", color: Theme.ink)
                    .help("Average model time per message (guard + route calls, incl. tokenize and heads)")
                MiniStat(value: has ? String(format: "%.0f", model.wallPerSecond) : "–", unit: "msg/s", label: "end to end", color: Theme.ink)
                    .help("Messages fully screened per wall-clock second since the run started")
                MiniStat(value: has ? String(format: "%.0f%%", t.aneShare * 100) : "–", label: "checks on ANE ⚡", color: Theme.ane)
                    .help("\(t.onANE) of \(t.checks) model calls ran on the Neural Engine")
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
    }

    @ViewBuilder var controls: some View {
        switch model.phase {
        case .loading:
            ProgressView().controlSize(.small)
            Text(model.status).font(.system(size: 11.5)).foregroundStyle(Theme.dim).lineLimit(1)
        case .failed:
            Text(model.status).font(.system(size: 11.5)).foregroundStyle(Theme.bad).lineLimit(2)
        case .idle:
            Button { model.run() } label: { Text("▶  Start").font(.system(size: 12.5, weight: .semibold)).padding(.horizontal, 6) }
                .buttonStyle(.borderedProminent).tint(Theme.accent).focusable(false)
        case .running:
            if model.paused {
                Text("paused").font(.system(size: 11.5, weight: .semibold)).foregroundStyle(Theme.warn)
            } else {
                ProgressView().controlSize(.small)
                Text("screening…").font(.system(size: 11.5)).foregroundStyle(Theme.dim)
            }
            Button { model.togglePause() } label: {
                Text(model.paused ? "▶  Resume" : "❚❚  Pause").font(.system(size: 12.5, weight: .semibold)).padding(.horizontal, 6)
            }
            .buttonStyle(.bordered).focusable(false)
            .keyboardShortcut(.space, modifiers: [])
            .help("Pause / resume (Space)")
        case .done:
            Button { model.run() } label: { Text("↻  Replay").font(.system(size: 12.5, weight: .semibold)).padding(.horizontal, 6) }
                .buttonStyle(.borderedProminent).tint(Theme.accent).focusable(false)
        }
    }
}

struct MiniStat: View {
    let value: String
    var unit: String? = nil
    let label: String
    let color: Color
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(value).font(.system(size: 16, weight: .bold).monospacedDigit()).foregroundStyle(color)
                if let unit { Text(unit).font(.system(size: 10.5).monospacedDigit()).foregroundStyle(Theme.dim) }
            }
            Text(label).font(.system(size: 10)).foregroundStyle(Theme.dim).lineLimit(1)
        }
        .padding(.horizontal, 9).padding(.vertical, 5)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.panel, in: RoundedRectangle(cornerRadius: 9))
        .overlay(RoundedRectangle(cornerRadius: 9).stroke(Theme.line))
    }
}

/// Rounded panel with a header row.
struct Lane<Header: View, Content: View>: View {
    var tint: Color = Theme.line
    @ViewBuilder var header: Header
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) { header }
            content.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        }
        .padding(.horizontal, 10).padding(.vertical, 9)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Theme.panel, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(tint))
    }
}

struct LaneTitle: View {
    let text: String
    var count: Int? = nil
    var color: Color = Theme.dim
    var body: some View {
        Text(text.uppercased()).font(.system(size: 10.5, weight: .semibold)).kerning(0.8).foregroundStyle(color).lineLimit(1)
        if let count {
            Text("\(count)").font(.system(size: 10.5, weight: .bold).monospacedDigit())
                .padding(.horizontal, 6).padding(.vertical, 1)
                .background(color.opacity(0.16), in: Capsule()).foregroundStyle(color)
        }
    }
}

// MARK: - inbox

@available(macOS 15.0, *)
struct InboxColumn: View {
    @EnvironmentObject var model: FrontDoorModel

    var body: some View {
        let waiting = model.waiting
        Lane {
            LaneTitle(text: "📥 Inbox", count: waiting.count, color: Theme.ink)
            Spacer()
            Text("waiting · next on top").font(.system(size: 10)).foregroundStyle(Theme.dim)
        } content: {
            Group {
            if waiting.isEmpty, model.phase == .done {
                FinalSummary()
            } else if waiting.isEmpty {
                Text(emptyText).font(.system(size: 12)).foregroundStyle(Theme.dim).multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 4) {
                        ForEach(waiting.prefix(FrontDoorModel.inboxRows)) { msg in
                            InboxRow(message: msg, scanning: msg.id == waiting.first?.id)
                        }
                        if waiting.count > FrontDoorModel.inboxRows {
                            Text("+ \(waiting.count - FrontDoorModel.inboxRows) more waiting")
                                .font(.system(size: 11)).foregroundStyle(Theme.dim).padding(.top, 4)
                        }
                    }
                }
                .defaultScrollAnchor(.top)
                .scrollIndicators(.never)
            }
            }
            .overlay(alignment: .top) { Color.clear.frame(height: 24).flightSpot(.inboxHead) }
        }
    }

    var emptyText: String {
        switch model.phase {
        case .loading: "Loading the model…"
        case .idle: "Press Start to open the front door.\n\(Traffic.all.count) messages are waiting."
        case .running: "Waiting for traffic…"
        case .done: "All caught up ✓"
        case .failed: "Model failed to load."
        }
    }
}

struct InboxRow: View {
    let message: InboundMessage
    let scanning: Bool

    var body: some View {
        HStack(spacing: 6) {
            Avatar(name: message.user, size: 16)
            Text(message.user).font(.system(size: 11.5, weight: .semibold)).lineLimit(1).frame(width: 78, alignment: .leading)
            Text(message.text).font(.system(size: 11.5)).foregroundStyle(Theme.ink.opacity(0.75)).lineLimit(1)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 7).padding(.vertical, 4)
        .background(scanning ? Theme.accent.opacity(0.14) : Theme.panel2, in: RoundedRectangle(cornerRadius: 7))
        .overlay { if scanning { Shimmer().clipShape(RoundedRectangle(cornerRadius: 7)).allowsHitTesting(false) } }
        .overlay(RoundedRectangle(cornerRadius: 7).stroke(scanning ? Theme.accent.opacity(0.8) : .clear))
    }
}

/// A soft light band sweeping left to right.
struct Shimmer: View {
    var body: some View {
        TimelineView(.animation) { ctx in
            GeometryReader { g in
                let period = 0.9
                let phase = ctx.date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: period) / period
                let w = g.size.width * 0.4
                LinearGradient(colors: [.clear, Theme.accent.opacity(0.30), .white.opacity(0.08), .clear],
                               startPoint: .leading, endPoint: .trailing)
                    .frame(width: w)
                    .offset(x: -w + phase * (g.size.width + w))
            }
        }
    }
}

// MARK: - lanes

@available(macOS 15.0, *)
struct AnsweredLane: View {
    @EnvironmentObject var model: FrontDoorModel

    var body: some View {
        let t = model.tally
        Lane(tint: Theme.ok.opacity(0.35)) {
            LaneTitle(text: "✅ Answered", count: t.answered, color: Theme.ok)
            Spacer(minLength: 4)
            ForEach(Questions.teams, id: \.name) { team in
                TeamTag(team: team.name, count: t.byTeam[team.name] ?? 0).flightSpot(.team(team.name))
            }
        } content: {
            LaneList(rows: model.answeredRows, empty: "Allowed messages land here, routed to a team · 🔒 personal info redacted")
        }
    }
}

enum BlockKind { case jailbreak, harmful }

@available(macOS 15.0, *)
struct BlockedLane: View {
    @EnvironmentObject var model: FrontDoorModel
    let kind: BlockKind

    var body: some View {
        let total = kind == .jailbreak ? model.tally.jailbreak : model.tally.harmful
        Lane(tint: Theme.bad.opacity(0.35)) {
            LaneTitle(text: kind == .jailbreak ? "⛔ Blocked · jailbreak / injection" : "⛔ Blocked · harmful", count: total, color: Theme.bad)
            Spacer(minLength: 0)
        } content: {
            LaneList(rows: kind == .jailbreak ? model.jailbreakRows : model.harmfulRows, empty: "Never reaches the bot")
                .overlay(alignment: .top) { Color.clear.frame(height: 22).flightSpot(kind == .jailbreak ? .jailbreak : .harmful) }
        }
    }
}

/// Every row of a lane, newest on top, scrollable all the way down. Stays pinned to the top (showing new rows) only
/// while the user is at the top; once they scroll down, their position holds while rows keep arriving above.
@available(macOS 15.0, *)
struct LaneList: View {
    @EnvironmentObject var model: FrontDoorModel
    let rows: [Processed]  // oldest first
    let empty: String
    @State private var position = ScrollPosition(edge: .top)
    @State private var atTop = true
    @State private var follow = true
    @State private var offsetY: CGFloat = 0
    private static let scrollTest = ProcessInfo.processInfo.environment["FRONTDOOR_SCROLLTEST"] == "1"

    var body: some View {
        if rows.isEmpty {
            Text(empty).font(.system(size: 11.5)).foregroundStyle(Theme.dim).frame(maxWidth: .infinity, maxHeight: .infinity)
                .onAppear { follow = true; position = ScrollPosition(edge: .top) }
        } else {
            ScrollView {
                LazyVStack(spacing: 3) {
                    ForEach(rows.reversed()) { p in
                        LaneRow(p: p).onTapGesture { model.selected = p }
                    }
                }
                .scrollTargetLayout()
            }
            .scrollPosition($position, anchor: .top)
            .scrollIndicators(.automatic)
            .onScrollGeometryChange(for: CGFloat.self, of: { $0.contentOffset.y + $0.contentInsets.top }) { _, y in
                offsetY = y
                atTop = y <= 4
            }
            .onScrollPhaseChange { _, phase in
                // decided by the user's own scrolling only, not by rows arriving
                if phase == .interacting || phase == .tracking { follow = false }
                if phase == .idle { follow = atTop }
            }
            .onChange(of: rows.count) {
                if follow { position.scrollTo(edge: .top) }
            }
            .onAppear { scrollTestHook() }
        }
    }

    /// Hidden `FRONTDOOR_SCROLLTEST=1`: after 2 s, park the lane mid-list as a user would, to check it holds still.
    private func scrollTestHook() {
        guard Self.scrollTest, empty.hasPrefix("Allowed") else { return }
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(3))
            follow = false
            if rows.count > 12 { position.scrollTo(id: rows[rows.count - 12].id, anchor: .top) }
        }
    }
}

struct TeamTag: View {
    let team: String
    var count: Int? = nil
    var body: some View {
        let c = Theme.team(team)
        Text(count.map { "\(team) \($0)" } ?? team)
            .font(.system(size: 10, weight: .semibold).monospacedDigit()).lineLimit(1).fixedSize()
            .padding(.horizontal, 6).padding(.vertical, 1)
            .foregroundStyle(c).background(c.opacity(0.14), in: Capsule())
    }
}

/// One compact line: avatar, name, (team tag), 🔒, text (redacted when it carries personal info).
struct LaneRow: View {
    let p: Processed

    var body: some View {
        HStack(spacing: 6) {
            Avatar(name: p.message.user, size: 16)
            Text(p.message.user).font(.system(size: 11.5, weight: .semibold)).lineLimit(1).frame(width: 84, alignment: .leading)
            if let r = p.route { TeamTag(team: r.team).frame(width: 62, alignment: .leading) }
            if !p.pii.isEmpty { Text("🔒").font(.system(size: 9.5)) }
            Group {
                if p.blocked {
                    Text(p.message.text).foregroundStyle(Theme.ink.opacity(0.55))
                } else if p.pii.isEmpty {
                    Text(p.message.text).foregroundStyle(Theme.ink.opacity(0.85))
                } else {
                    Text(maskedAttributed(p.masked, size: 11.5)).foregroundStyle(Theme.ink.opacity(0.85))
                }
            }
            .font(.system(size: 11.5)).lineLimit(1)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 7).padding(.vertical, 3)
        .background(Theme.panel2, in: RoundedRectangle(cornerRadius: 6))
        .contentShape(Rectangle())
    }
}

// MARK: - final summary (fills the emptied inbox)

@available(macOS 15.0, *)
struct FinalSummary: View {
    @EnvironmentObject var model: FrontDoorModel

    var body: some View {
        let t = model.tally
        VStack(alignment: .leading, spacing: 8) {
            Text("✓ Run complete").font(.system(size: 15, weight: .bold)).foregroundStyle(Theme.ok)
            Text(String(format: "%d messages in %.1f s · nothing sent anywhere", t.processed, model.elapsed))
                .font(.system(size: 11.5)).foregroundStyle(Theme.dim)
            VStack(spacing: 4) {
                row("✅ Answered", "\(t.answered)", Theme.ok)
                ForEach(Questions.teams, id: \.name) { team in
                    row("    \(team.name)", "\(t.byTeam[team.name] ?? 0)", Theme.team(team.name), small: true)
                }
                row("⛔ Jailbreak / injection", "\(t.jailbreak)", Theme.bad)
                row("⛔ Harmful", "\(t.harmful)", Theme.bad)
                row("🔒 PII redacted", "\(t.withPII)", Theme.pii)
                Divider().overlay(Theme.line).padding(.vertical, 2)
                row("avg per message", String(format: "%.1f ms", t.avgMs), Theme.ink)
                row("end to end", String(format: "%.0f msg/s", model.wallPerSecond), Theme.ink)
                row("checks on ANE ⚡", String(format: "%.0f%% (%d/%d)", t.aneShare * 100, t.onANE, t.checks), Theme.ane)
            }
            .padding(9)
            .background(Theme.panel2, in: RoundedRectangle(cornerRadius: 9))
            Spacer(minLength: 0)
            Button { model.run() } label: {
                Text("↻  Replay").font(.system(size: 12.5, weight: .semibold)).frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).tint(Theme.accent).focusable(false)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    func row(_ label: String, _ value: String, _ color: Color, small: Bool = false) -> some View {
        HStack {
            Text(label).font(.system(size: small ? 11 : 12)).foregroundStyle(small ? Theme.dim : Theme.ink)
            Spacer()
            Text(value).font(.system(size: small ? 11 : 12, weight: .bold).monospacedDigit()).foregroundStyle(color)
        }
    }
}

// MARK: - detail sheet

@available(macOS 15.0, *)
struct DetailSheet: View {
    @Environment(\.dismiss) private var dismiss
    let p: Processed

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 10) {
                Avatar(name: p.message.user, size: 34)
                VStack(alignment: .leading, spacing: 2) {
                    Text(p.message.user).font(.system(size: 15, weight: .semibold))
                    Text("arrived \(clock(p.arrived)) · message \(p.message.id + 1) of \(Traffic.all.count)")
                        .font(.system(size: 11.5)).foregroundStyle(Theme.dim)
                }
                Spacer()
                verdictChip
                Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
            }

            section("Original message" + (p.pii.isEmpty ? "" : " · personal info highlighted"))
            Text(highlightedAttributed(p.message.text, p.pii)).font(.system(size: 15)).lineSpacing(3).textSelection(.enabled)
                .padding(12).frame(maxWidth: .infinity, alignment: .leading)
                .background(Theme.panel2, in: RoundedRectangle(cornerRadius: 10))
            if !p.pii.isEmpty {
                FlowRow(spacing: 6) {
                    ForEach(p.pii, id: \.self) { s in
                        Chip(text: "\(s.label) · \(s.text) · \(Int((s.probability * 100).rounded()))%", style: .warn)
                    }
                }
                section("What the bot receives")
                Text(maskedAttributed(p.masked, size: 14)).font(.system(size: 14))
                    .padding(12).frame(maxWidth: .infinity, alignment: .leading)
                    .background(Theme.panel2, in: RoundedRectangle(cornerRadius: 10))
            }

            HStack { section("Check 1 · guard (attack + harm)"); Spacer(); UnitBadge(timing: p.guardCheck.timing, detailed: true) }
            Bar(label: "jailbreak / injection", p: p.guardCheck.attack, color: p.guardCheck.attack >= Policy.flag ? Theme.bad : Theme.dim)
            Bar(label: "harmful", p: p.guardCheck.harm, color: p.guardCheck.harm >= Policy.flag ? Theme.bad : Theme.dim)

            if let r = p.route {
                HStack { section("Check 2 · route + personal info"); Spacer(); UnitBadge(timing: r.timing, detailed: true) }
                ForEach(r.topics, id: \.name) { t in
                    Bar(label: "\(Theme.teamIcon(t.name)) \(t.name)", p: t.p, color: t.name == r.team ? Theme.team(t.name) : Theme.dim.opacity(0.6))
                }
            } else {
                HStack { section("Check 2 · route + personal info"); Spacer(); Chip(text: "skipped — blocked at the door", style: .dim) }
            }
            Text(String(format: "Total on device: %.1f ms (%@)", p.totalMs,
                        p.timings.map { String(format: "%.1f ms %@", $0.totalMs, $0.shortUnit) }.joined(separator: " + ")))
                .font(.system(size: 11.5).monospacedDigit()).foregroundStyle(Theme.dim)
        }
        .padding(22)
        .frame(width: 680)
        .background(Theme.panel)
        .foregroundStyle(Theme.ink)
        .preferredColorScheme(.dark)
    }

    @ViewBuilder var verdictChip: some View {
        switch p.verdict {
        case .answered(let team): Chip(text: "✅ Answered → \(Theme.teamIcon(team)) \(team)", style: .ok, size: 12.5)
        case .jailbreak: Chip(text: "⛔ Blocked · jailbreak / prompt injection", style: .bad, size: 12.5)
        case .harmful: Chip(text: "⛔ Blocked · harmful", style: .bad, size: 12.5)
        }
    }

    func section(_ s: String) -> some View {
        Text(s.uppercased()).font(.system(size: 11, weight: .semibold)).kerning(0.8).foregroundStyle(Theme.dim)
    }
}

// MARK: - a wrapping row

struct FlowRow: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = arrange(proposal.width ?? .infinity, subviews)
        return CGSize(width: proposal.width ?? rows.map(\.width).max() ?? 0, height: rows.last.map { $0.y + $0.height } ?? 0)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        for row in arrange(bounds.width, subviews) {
            for (i, x, size) in row.items {
                subviews[i].place(at: CGPoint(x: bounds.minX + x, y: bounds.minY + row.y + (row.height - size.height) / 2),
                                  proposal: ProposedViewSize(size))
            }
        }
    }

    private struct Row { var items: [(Int, CGFloat, CGSize)] = []; var y: CGFloat = 0; var width: CGFloat = 0; var height: CGFloat = 0 }

    private func arrange(_ maxWidth: CGFloat, _ subviews: Subviews) -> [Row] {
        var rows: [Row] = [Row()]
        for (i, v) in subviews.enumerated() {
            var size = v.sizeThatFits(.unspecified)
            if size.width > maxWidth { size = v.sizeThatFits(ProposedViewSize(width: maxWidth, height: nil)) }
            if !rows[rows.count - 1].items.isEmpty, rows[rows.count - 1].width + spacing + size.width > maxWidth {
                let y = rows[rows.count - 1].y + rows[rows.count - 1].height + spacing
                rows.append(Row(y: y))
            }
            var r = rows[rows.count - 1]
            r.items.append((i, r.items.isEmpty ? 0 : r.width + spacing, size))
            r.width = r.items.last!.1 + size.width
            r.height = max(r.height, size.height)
            rows[rows.count - 1] = r
        }
        return rows
    }
}
