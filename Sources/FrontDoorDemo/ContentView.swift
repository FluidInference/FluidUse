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
    @Namespace private var flight

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(Theme.line)
            HStack(alignment: .top, spacing: 14) {
                InboxColumn(flight: flight).frame(width: 340)
                VStack(spacing: 14) {
                    AnsweredLane(flight: flight).frame(maxHeight: .infinity)
                    HStack(alignment: .top, spacing: 14) {
                        BlockedLane(kind: .jailbreak, flight: flight)
                        BlockedLane(kind: .harmful, flight: flight)
                    }
                    .frame(height: 268)
                }
            }
            .padding(.horizontal, 22).padding(.top, 14).padding(.bottom, 18)
        }
        .background(Theme.bg)
        .foregroundStyle(Theme.ink)
        .preferredColorScheme(.dark)
        .sheet(item: $model.selected) { p in DetailSheet(p: p) }
    }

    // MARK: header

    var header: some View {
        let t = model.tally
        return VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 14) {
                Text("🚪 Chatbot Front Door").font(.system(size: 19, weight: .bold)).kerning(0.3)
                (Text("Every message screened on this Mac by ") + Text("Vela-2.0-0.3B").bold().foregroundColor(Theme.ink)
                    + Text(" (Core ML) · nothing sent anywhere"))
                    .font(.system(size: 13)).foregroundStyle(Theme.dim).lineLimit(1)
                Spacer(minLength: 8)
                controls
            }
            HStack(spacing: 10) {
                Stat(value: "\(t.processed)", unit: "/ \(Traffic.all.count)", label: "processed", color: Theme.ink)
                Stat(value: "\(t.answered)", label: "answered", color: Theme.ok)
                Stat(value: "\(t.blocked)", label: "blocked", color: Theme.bad)
                Stat(value: "\(t.withPII)", label: "with personal info 🔒", color: Theme.pii)
                Stat(value: t.processed > 0 ? String(format: "%.1f", t.avgMs) : "–", unit: "ms", label: "avg per message (both checks)", color: Theme.ink)
                Stat(value: t.processed > 0 ? String(format: "%.0f", t.perSecond) : "–", unit: "msg/s", label: "on-device throughput", color: Theme.ink)
                    .help("1000 ÷ average milliseconds per message: what this Mac sustains screening messages one after another")
                Stat(value: "\(t.onANE)", unit: "of \(t.checks)", label: "checks on Neural Engine ⚡", color: Theme.ane)
            }
        }
        .padding(.horizontal, 22).padding(.vertical, 14)
    }

    @ViewBuilder var controls: some View {
        switch model.phase {
        case .loading:
            ProgressView().controlSize(.small)
            Text(model.status).font(.system(size: 12)).foregroundStyle(Theme.dim)
        case .failed:
            Text(model.status).font(.system(size: 12)).foregroundStyle(Theme.bad).lineLimit(2)
        case .idle:
            Text(model.status).font(.system(size: 12)).foregroundStyle(Theme.dim).lineLimit(1)
            Button { model.run() } label: { Text("▶  Start").font(.system(size: 13, weight: .semibold)).padding(.horizontal, 10) }
                .buttonStyle(.borderedProminent).tint(Theme.accent).controlSize(.large)
        case .running:
            ProgressView().controlSize(.small)
            Text("Screening live traffic…").font(.system(size: 12)).foregroundStyle(Theme.dim)
        case .done:
            Button { model.run() } label: { Text("↻  Replay").font(.system(size: 13, weight: .semibold)).padding(.horizontal, 10) }
                .buttonStyle(.borderedProminent).tint(Theme.accent).controlSize(.large)
        }
    }
}

struct Stat: View {
    let value: String
    var unit: String? = nil
    let label: String
    let color: Color
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(value).font(.system(size: 22, weight: .bold).monospacedDigit()).foregroundStyle(color)
                    .contentTransition(.numericText())
                if let unit { Text(unit).font(.system(size: 12).monospacedDigit()).foregroundStyle(Theme.dim) }
            }
            Text(label).font(.system(size: 11)).foregroundStyle(Theme.dim).lineLimit(1)
        }
        .padding(.horizontal, 14).padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.panel, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Theme.line))
    }
}

/// Rounded panel with a header row.
struct Lane<Header: View, Content: View>: View {
    var tint: Color = Theme.line
    @ViewBuilder var header: Header
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) { header }
            content.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        }
        .padding(.horizontal, 14).padding(.vertical, 12)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Theme.panel, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(tint))
    }
}

struct LaneTitle: View {
    let text: String
    var count: Int? = nil
    var color: Color = Theme.dim
    var body: some View {
        Text(text.uppercased()).font(.system(size: 11.5, weight: .semibold)).kerning(1).foregroundStyle(color)
        if let count {
            Text("\(count)").font(.system(size: 11.5, weight: .bold).monospacedDigit())
                .padding(.horizontal, 7).padding(.vertical, 1)
                .background(color.opacity(0.16), in: Capsule()).foregroundStyle(color)
                .contentTransition(.numericText())
        }
    }
}

// MARK: - inbox

@available(macOS 15.0, *)
struct InboxColumn: View {
    @EnvironmentObject var model: FrontDoorModel
    let flight: Namespace.ID

    var body: some View {
        Lane {
            LaneTitle(text: "📥 Inbox", count: model.inbox.count, color: Theme.ink)
            Spacer()
            Text("incoming · newest at the bottom").font(.system(size: 11)).foregroundStyle(Theme.dim)
        } content: {
            if model.inbox.isEmpty, model.phase == .done {
                FinalSummary().transition(.scale(scale: 0.95).combined(with: .opacity))
            } else if model.inbox.isEmpty {
                VStack(spacing: 8) {
                    Spacer()
                    Text(emptyText).font(.system(size: 13)).foregroundStyle(Theme.dim).multilineTextAlignment(.center)
                    Spacer()
                }
                .frame(maxWidth: .infinity)
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(spacing: 8) {
                            ForEach(model.inbox) { item in
                                InboxRow(item: item, scanning: model.scanning == item.id)
                                    .id(item.id)
                            }
                        }
                    }
                    .defaultScrollAnchor(.top)
                    .scrollIndicators(.never)
                }
            }
        }
    }

    var emptyText: String {
        switch model.phase {
        case .loading: "Loading the model…"
        case .idle: "Press Start to open the front door.\n40 customers are waiting."
        case .running: "Waiting for the next message…"
        case .done: "All caught up ✓"
        case .failed: "Model failed to load."
        }
    }
}

struct InboxRow: View {
    let item: InboxItem
    let scanning: Bool

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Avatar(name: item.message.user, size: 30)
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text(item.message.user).font(.system(size: 13, weight: .semibold)).lineLimit(1)
                    Spacer(minLength: 6)
                    Text(clock(item.arrived)).font(.system(size: 11).monospacedDigit()).foregroundStyle(Theme.dim)
                }
                Text(item.message.text).font(.system(size: 12.5)).foregroundStyle(Theme.ink.opacity(0.78)).lineLimit(2)
                if scanning {
                    Text("🔍 screening on device…").font(.system(size: 11, weight: .medium)).foregroundStyle(Theme.accent)
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(scanning ? Theme.accent.opacity(0.10) : Theme.panel2, in: RoundedRectangle(cornerRadius: 11))
        .overlay { if scanning { Shimmer().clipShape(RoundedRectangle(cornerRadius: 11)).allowsHitTesting(false) } }
        .overlay(RoundedRectangle(cornerRadius: 11).stroke(scanning ? Theme.accent.opacity(0.8) : Theme.line, lineWidth: scanning ? 1.5 : 1))
    }
}

/// A soft light band sweeping left to right.
struct Shimmer: View {
    var body: some View {
        TimelineView(.animation) { ctx in
            GeometryReader { g in
                let period = 1.1
                let phase = ctx.date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: period) / period
                let w = g.size.width * 0.45
                LinearGradient(colors: [.clear, Theme.accent.opacity(0.30), .white.opacity(0.10), .clear],
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
    let flight: Namespace.ID

    var body: some View {
        let answered = model.processed.filter { !$0.blocked }
        Lane(tint: Theme.ok.opacity(0.35)) {
            LaneTitle(text: "✅ Answered · routed to a team", count: answered.count, color: Theme.ok)
            Spacer()
            Text("🔒 personal info is redacted before the bot sees it").font(.system(size: 11)).foregroundStyle(Theme.dim)
        } content: {
            HStack(alignment: .top, spacing: 10) {
                ForEach(Questions.teams, id: \.name) { team in
                    let cards = answered.filter { $0.route?.team == team.name }.reversed()
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(spacing: 6) {
                            Text(Theme.teamIcon(team.name)).font(.system(size: 12))
                            Text(team.name.capitalized).font(.system(size: 12.5, weight: .semibold)).foregroundStyle(Theme.team(team.name))
                            Spacer(minLength: 2)
                            Text("\(cards.count)").font(.system(size: 11.5, weight: .bold).monospacedDigit())
                                .foregroundStyle(Theme.team(team.name)).contentTransition(.numericText())
                        }
                        .padding(.horizontal, 9).padding(.vertical, 5)
                        .background(Theme.team(team.name).opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
                        ScrollView {
                            VStack(spacing: 8) {
                                ForEach(Array(cards)) { p in
                                    LaneCard(p: p)
                                        .onTapGesture { model.selected = p }
                                }
                            }
                        }
                        .scrollIndicators(.never)
                    }
                    .frame(maxWidth: .infinity)
                }
            }
        }
    }
}

enum BlockKind { case jailbreak, harmful }

@available(macOS 15.0, *)
struct BlockedLane: View {
    @EnvironmentObject var model: FrontDoorModel
    let kind: BlockKind
    let flight: Namespace.ID

    var body: some View {
        let cards = model.processed.filter { kind == .jailbreak ? $0.verdict == .jailbreak : $0.verdict == .harmful }.reversed()
        Lane(tint: Theme.bad.opacity(0.35)) {
            LaneTitle(text: kind == .jailbreak ? "⛔ Blocked · jailbreak / prompt injection" : "⛔ Blocked · harmful",
                      count: cards.count, color: Theme.bad)
            Spacer()
            Text("never reaches the bot").font(.system(size: 11)).foregroundStyle(Theme.dim)
        } content: {
            ScrollView {
                LazyVGrid(columns: [GridItem(.flexible(), spacing: 8), GridItem(.flexible(), spacing: 8)], spacing: 8) {
                    ForEach(Array(cards)) { p in
                        LaneCard(p: p)
                            .onTapGesture { model.selected = p }
                    }
                }
            }
            .scrollIndicators(.never)
        }
    }
}

/// A processed message in its lane.
struct LaneCard: View {
    let p: Processed

    var body: some View {
        let tint: Color = p.route.map { Theme.team($0.team) } ?? Theme.bad
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Avatar(name: p.message.user, size: 18)
                Text(p.message.user).font(.system(size: 11.5, weight: .semibold)).lineLimit(1)
                Spacer(minLength: 2)
                if !p.pii.isEmpty { Text("🔒").font(.system(size: 11)).help("\(p.pii.count) personal-info span(s) redacted") }
                Text(p.arrived.formatted(date: .omitted, time: .shortened)).font(.system(size: 10).monospacedDigit()).foregroundStyle(Theme.dim)
            }
            Group {
                if p.blocked {
                    Text(p.message.text).foregroundStyle(Theme.ink.opacity(0.6)).strikethrough(color: Theme.bad.opacity(0.5))
                } else {
                    Text(maskedAttributed(p.masked, size: 12)).foregroundStyle(Theme.ink.opacity(0.88))
                }
            }
            .font(.system(size: 12)).lineLimit(p.blocked ? 2 : 3).frame(maxWidth: .infinity, alignment: .leading)
            FlowRow(spacing: 4) {
                switch p.verdict {
                case .jailbreak: Chip(text: String(format: "attack %.0f%%", p.guardCheck.attack * 100), style: .bad, size: 10.5)
                case .harmful: Chip(text: String(format: "harm %.0f%%", p.guardCheck.harm * 100), style: .bad, size: 10.5)
                case .answered: EmptyView()
                }
                UnitBadge(timing: p.guardCheck.timing)
                if let r = p.route { UnitBadge(timing: r.timing) }
            }
        }
        .padding(.leading, 12).padding(.trailing, 9).padding(.vertical, 8)
        .background(Theme.panel2, in: RoundedRectangle(cornerRadius: 10))
        .overlay(alignment: .leading) {
            UnevenRoundedRectangle(topLeadingRadius: 10, bottomLeadingRadius: 10).fill(tint).frame(width: 3)
        }
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(tint.opacity(0.28)))
        .contentShape(RoundedRectangle(cornerRadius: 10))
        .help("Click for details")
    }
}

// MARK: - final summary (fills the emptied inbox)

@available(macOS 15.0, *)
struct FinalSummary: View {
    @EnvironmentObject var model: FrontDoorModel

    var body: some View {
        let t = model.tally
        let share = t.checks > 0 ? Double(t.onANE) / Double(t.checks) * 100 : 0
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Text("✓").font(.system(size: 20, weight: .bold)).foregroundStyle(Theme.ok)
                Text("Run complete").font(.system(size: 17, weight: .bold))
            }
            Text("\(t.processed) messages screened on this Mac — nothing sent anywhere.")
                .font(.system(size: 13)).foregroundStyle(Theme.dim).fixedSize(horizontal: false, vertical: true)
            VStack(spacing: 7) {
                row("✅ Answered", "\(t.answered)", Theme.ok)
                ForEach(Questions.teams, id: \.name) { team in
                    row("     \(Theme.teamIcon(team.name)) \(team.name)", "\(t.byTeam[team.name] ?? 0)", Theme.team(team.name), small: true)
                }
                row("⛔ Blocked · jailbreak", "\(t.jailbreak)", Theme.bad)
                row("⛔ Blocked · harmful", "\(t.harmful)", Theme.bad)
                row("🔒 Personal info redacted", "\(t.withPII)", Theme.pii)
            }
            .padding(12)
            .background(Theme.panel2, in: RoundedRectangle(cornerRadius: 11))
            HStack(spacing: 8) {
                Stat(value: String(format: "%.1f", t.avgMs), unit: "ms", label: "avg per message", color: Theme.ink)
                Stat(value: String(format: "%.0f%%", share), unit: "\(t.onANE)/\(t.checks)", label: "checks on ANE ⚡", color: Theme.ane)
            }
            Text("Guard checks run on the Neural Engine; route + personal-info checks (larger schema) on the GPU.")
                .font(.system(size: 11.5)).foregroundStyle(Theme.dim).fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            Button { model.run() } label: {
                Text("↻  Replay").font(.system(size: 13, weight: .semibold)).frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).tint(Theme.accent).controlSize(.large)
        }
        .padding(14)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Theme.ok.opacity(0.07), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Theme.ok.opacity(0.4)))
    }

    func row(_ label: String, _ value: String, _ color: Color, small: Bool = false) -> some View {
        HStack {
            Text(label).font(.system(size: small ? 12 : 13, weight: small ? .regular : .medium))
                .foregroundStyle(small ? Theme.dim : Theme.ink)
            Spacer()
            Text(value).font(.system(size: small ? 12.5 : 14, weight: .bold).monospacedDigit()).foregroundStyle(color)
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
