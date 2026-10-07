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
    static let userBubble = hex(0x2a3a7a)
    static let barDim = hex(0x56607a)
    static let ane = hex(0x3ccf8e)
    static let gpu = hex(0x8fa6ff)
}

// MARK: - small pieces

struct Card<Content: View>: View {
    var title: String?
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let title {
                Text(title.uppercased()).font(.system(size: 11, weight: .semibold)).kerning(1).foregroundStyle(Theme.dim)
            }
            content
        }
        .padding(.horizontal, 16).padding(.vertical, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.panel, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Theme.line))
    }
}

enum ChipStyle { case plain, ok, bad, warn, dim }

struct Chip: View {
    let text: String
    var style: ChipStyle = .plain
    var body: some View {
        let (bg, fg): (Color, Color) = switch style {
        case .plain: (Color.white.opacity(0.08), Theme.ink)
        case .ok: (Theme.ok.opacity(0.16), Theme.hex(0x8ff0c4))
        case .bad: (Theme.bad.opacity(0.18), Theme.hex(0xffadb5))
        case .warn: (Theme.warn.opacity(0.16), Theme.piiInk)
        case .dim: (Color.white.opacity(0.08), Theme.dim)
        }
        Text(text).font(.system(size: 11.5)).lineLimit(1).fixedSize()
            .padding(.horizontal, 8).padding(.vertical, 2)
            .background(bg, in: Capsule()).foregroundStyle(fg)
    }
}

/// "⚡ Neural Engine · 3.6 ms (total 4.1)" / "GPU · 16.8 ms (total 17.9)".
struct UnitBadge: View {
    let timing: Timing
    var compact = false
    var body: some View {
        let color = timing.onNeuralEngine ? Theme.ane : Theme.gpu
        HStack(spacing: 4) {
            Text((timing.onNeuralEngine ? "⚡ " : "") + (compact ? timing.shortUnit : timing.longUnit)
                 + String(format: " · %.1f ms", timing.encoderMs))
                .font(.system(size: 11.5, weight: .semibold).monospacedDigit())
            Text(String(format: "total %.1f", timing.totalMs)).font(.system(size: 10).monospacedDigit()).opacity(0.7)
        }
        .lineLimit(1).fixedSize()
        .padding(.horizontal, 8).padding(.vertical, 2)
        .foregroundStyle(color)
        .background(color.opacity(0.13), in: Capsule())
        .overlay(Capsule().stroke(color.opacity(0.35)))
        .help("Encoder \(String(format: "%.2f", timing.encoderMs)) ms on the \(timing.longUnit) (tokens → bucket: \(timing.shape)); full check incl. tokenize + heads \(String(format: "%.2f", timing.totalMs)) ms")
    }
}

struct Bar: View {
    let label: String
    let p: Double?
    let color: Color
    var body: some View {
        HStack(spacing: 8) {
            Text(label).font(.system(size: 13)).frame(width: 118, alignment: .leading).lineLimit(1)
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
        .animation(.spring(duration: 0.35), value: p)
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
            tok.font = .system(size: size - 1.5, weight: .semibold)
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

// MARK: - main view

@available(macOS 15.0, *)
struct ContentView: View {
    @EnvironmentObject var model: GuardrailModel

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(Theme.line)
            HStack(alignment: .top, spacing: 14) {
                sourceColumn.frame(width: 300)
                chat.frame(maxWidth: .infinity)
                decisionColumn.frame(width: 350)
            }
            .padding(.horizontal, 22).padding(.top, 14).padding(.bottom, 18)
        }
        .background(Theme.bg)
        .foregroundStyle(Theme.ink)
        .preferredColorScheme(.dark)
    }

    // header

    var header: some View {
        HStack(spacing: 14) {
            Text("🛡️ On-device Guardrail").font(.system(size: 18, weight: .bold)).kerning(0.3)
            (Text("Every message and reply checked by ") + Text("Vela-2.0-0.3B").bold().foregroundColor(Theme.ink)
                + Text(" on this Mac (Core ML) · nothing sent anywhere"))
                .font(.system(size: 13)).foregroundStyle(Theme.dim).lineLimit(1)
            Spacer(minLength: 8)
            if !model.ready {
                ProgressView().controlSize(.small)
                Text(model.status).font(.system(size: 12)).foregroundStyle(Theme.dim)
            }
            Text("checks: \(model.checks) · on ANE: \(model.onANE)")
                .font(.system(size: 12, weight: .medium).monospacedDigit())
                .padding(.horizontal, 10).padding(.vertical, 4)
                .background(Theme.panel2, in: Capsule()).overlay(Capsule().stroke(Theme.line))
                .help(model.status)
            Picker("", selection: Binding(get: { model.scenario }, set: { model.setScenario($0) })) {
                ForEach(Scenario.all) { Text($0.name).tag($0) }
            }
            .labelsHidden().frame(width: 280)
        }
        .padding(.horizontal, 22).padding(.vertical, 14)
    }

    // left

    var sourceColumn: some View {
        Card(title: "Source document") {
            TextEditor(text: $model.source)
                .font(.system(size: 14)).scrollContentBackground(.hidden)
                .lineSpacing(3)
                .padding(8)
                .background(Theme.panel2, in: RoundedRectangle(cornerRadius: 10))
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Theme.line))
            Text("Replies are checked against this text. Edit it to change what counts as supported.")
                .font(.system(size: 12)).foregroundStyle(Theme.dim)
        }
        .frame(maxHeight: .infinity)
    }

    // center

    var chat: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 14) {
                        if model.messages.isEmpty {
                            VStack(spacing: 6) {
                                Text("Type a message or pick a suggestion below.")
                                Text("Personal info is highlighted as you type and masked before sending; replies are checked against the source on the left.")
                            }
                            .font(.system(size: 12)).foregroundStyle(Theme.dim).multilineTextAlignment(.center)
                            .frame(maxWidth: 400).padding(.top, 160)
                        }
                        ForEach(model.messages) { m in
                            MessageView(message: m).id(m.id)
                        }
                    }
                    .padding(18)
                }
                .onChange(of: model.messages.count) {
                    if let last = model.messages.last { withAnimation { proxy.scrollTo(last.id, anchor: .bottom) } }
                }
                .onChange(of: model.messages.last?.reply?.answer) {
                    if let last = model.messages.last { withAnimation { proxy.scrollTo(last.id, anchor: .bottom) } }
                }
            }
            Divider().overlay(Theme.line)
            composer
        }
        .background(Theme.panel, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Theme.line))
        .frame(maxHeight: .infinity)
    }

    var composer: some View {
        let pii = model.livePII
        return VStack(alignment: .leading, spacing: 8) {
            ZStack(alignment: .topLeading) {
                RichTextView(
                    text: $model.input, highlightedText: pii?.text,
                    highlights: (pii?.spans ?? []).map {
                        Highlight(start: $0.start, end: $0.end, style: .pii,
                                  tooltip: "\($0.label) · \(Int(($0.probability * 100).rounded()))%")
                    },
                    fontSize: 15, scrolls: true, onSubmit: { model.send() })
                if model.input.isEmpty {
                    Text("Type a message… personal info is highlighted as you type")
                        .font(.system(size: 15)).foregroundStyle(Theme.dim).padding(.top, 2).allowsHitTesting(false)
                }
            }
            .frame(height: 64)
            .padding(.horizontal, 14).padding(.vertical, 10)
            .background(Theme.panel2, in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(Theme.line))

            FlowRow(spacing: 8) {
                ForEach(model.scenario.suggest, id: \.self) { s in
                    Button { model.input = s } label: {
                        Text(s).font(.system(size: 12)).lineLimit(1).truncationMode(.tail)
                            .foregroundStyle(Theme.dim)
                            .padding(.horizontal, 10).padding(.vertical, 3)
                            .background(Theme.panel2, in: Capsule()).overlay(Capsule().stroke(Theme.line))
                    }
                    .buttonStyle(.plain)
                }
            }

            HStack(spacing: 8) {
                liveChips
                Spacer(minLength: 4)
                Button(action: model.send) {
                    Text("Send").font(.system(size: 14, weight: .semibold)).foregroundStyle(.white)
                        .padding(.horizontal, 18).padding(.vertical, 7)
                        .background(Theme.accent, in: RoundedRectangle(cornerRadius: 10))
                }
                .buttonStyle(.plain).disabled(!model.ready)
                .keyboardShortcut(.return, modifiers: .command)
            }
            .frame(minHeight: 32)
        }
        .padding(.horizontal, 14).padding(.vertical, 12)
    }

    @ViewBuilder var liveChips: some View {
        let hasText = !model.input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        if hasText, let s = model.screen, s.text == model.input {
            HStack(spacing: 6) {
                Text("live").font(.system(size: 11)).foregroundStyle(Theme.dim)
                if let r = s.blockReason {
                    Chip(text: "⛔ \(r)", style: .bad)
                } else {
                    Chip(text: String(format: "attack %d%% · harm %d%%", Int((s.attack * 100).rounded()), Int((s.harm * 100).rounded())), style: .ok)
                }
                UnitBadge(timing: s.timing, compact: true)
                if let p = model.livePII, p.text == model.input {
                    Chip(text: p.spans.isEmpty ? "no personal info" : "🔒 \(p.spans.count) personal info", style: p.spans.isEmpty ? .dim : .warn)
                    UnitBadge(timing: p.timing, compact: true)
                }
            }
        } else if hasText {
            Text("checking…").font(.system(size: 12)).foregroundStyle(Theme.dim)
        }
    }

    // right

    var decisionColumn: some View {
        let p = model.panel
        return VStack(spacing: 14) {
            Card(title: "The model's last check") {
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text(p.timing.map { String(format: "%.1f", $0.encoderMs) } ?? "–")
                        .font(.system(size: 38, weight: .heavy).monospacedDigit())
                    Text("ms").foregroundStyle(Theme.dim)
                    if let t = p.timing {
                        Text(t.onNeuralEngine ? "⚡ Neural Engine" : "GPU")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(t.onNeuralEngine ? Theme.ane : Theme.gpu)
                            .padding(.leading, 6)
                    }
                }
                if let t = p.timing {
                    Text(String(format: "encoder · %@ tokens · full check %.1f ms %@", t.shape, t.totalMs, p.what))
                        .font(.system(size: 12)).foregroundStyle(Theme.dim)
                } else {
                    Text(model.ready ? "waiting for a message" : model.status).font(.system(size: 12)).foregroundStyle(Theme.dim)
                }
                if let v = p.verdict {
                    Text(v.text).font(.system(size: 14, weight: .bold)).foregroundStyle(v.bad ? Theme.bad : Theme.ok)
                }
            }
            Card {
                ScrollView {
                    VStack(alignment: .leading, spacing: 6) {
                        Bar(label: "Prompt attack", p: p.attack, color: (p.attack ?? 0) >= Policy.flag ? Theme.bad : Theme.barDim)
                        Bar(label: "Harmful", p: p.harm, color: (p.harm ?? 0) >= Policy.flag ? Theme.bad : Theme.barDim)
                        Bar(label: "Needs fact-check", p: p.factcheck, color: Theme.warn)
                        section("Route (subject area)")
                        if p.route.isEmpty {
                            Text(p.timing == nil ? "—" : p.routeNote).font(.system(size: 12)).foregroundStyle(Theme.dim)
                        }
                        ForEach(Array(p.route.enumerated()), id: \.offset) { i, r in
                            Bar(label: r.name, p: r.p, color: i == 0 ? Theme.accent : Theme.hex(0x3b4766))
                        }
                        section("Personal information found")
                        if p.pii.isEmpty {
                            Text("none").font(.system(size: 12)).foregroundStyle(Theme.dim)
                        }
                        ForEach(Array(p.pii.enumerated()), id: \.offset) { _, s in
                            VStack(spacing: 3) {
                                HStack {
                                    Text(s.text).font(.system(size: 13)).lineLimit(1)
                                    Spacer()
                                    Text(s.label).font(.system(size: 11.5, weight: .semibold)).foregroundStyle(Theme.piiInk)
                                }
                                Line().stroke(style: StrokeStyle(lineWidth: 1, dash: [3, 3])).foregroundStyle(Theme.line).frame(height: 1)
                            }
                        }
                        section("What leaves the device")
                        Group {
                            if p.leavesBlocked {
                                Text("Nothing — blocked on device").foregroundStyle(Theme.hex(0xffadb5))
                            } else if let l = p.leaves {
                                Text(maskedAttributed(l, size: 13))
                            } else {
                                Text("—").foregroundStyle(Theme.dim)
                            }
                        }
                        .font(.system(size: 13)).lineSpacing(2)
                        .padding(.horizontal, 11).padding(.vertical, 9)
                        .frame(maxWidth: .infinity, minHeight: 40, alignment: .leading)
                        .background(Theme.panel2, in: RoundedRectangle(cornerRadius: 10))
                        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Theme.line))
                        section("Where it ran")
                        Text("≤ 128 tokens → Neural Engine · longer → GPU.\nLive screen (attack + harm) fits the ANE; the PII schema alone (~192 tokens) and the full send check (~535) run on the GPU; replies vs a short source fit the ANE.")
                            .font(.system(size: 11.5)).foregroundStyle(Theme.dim).fixedSize(horizontal: false, vertical: true)
                    }
                }
                .scrollIndicators(.never)
            }
            .frame(maxHeight: .infinity, alignment: .top)
        }
    }

    func section(_ s: String) -> some View {
        Text(s).font(.system(size: 12)).foregroundStyle(Theme.dim).padding(.top, 10).padding(.bottom, 2)
    }
}

struct Line: Shape {
    func path(in r: CGRect) -> Path {
        Path { $0.move(to: CGPoint(x: 0, y: r.midY)); $0.addLine(to: CGPoint(x: r.width, y: r.midY)) }
    }
}

// MARK: - messages

@available(macOS 15.0, *)
struct MessageView: View {
    @EnvironmentObject var model: GuardrailModel
    let message: ChatMessage

    var body: some View {
        switch message.kind {
        case .blocked(let c):
            HStack {
                Spacer(minLength: 120)
                bubble(fill: Theme.hex(0x3a1820), stroke: Theme.hex(0x6b2230), corner: .bottomTrailing) {
                    who("You")
                    Text(c.text).font(.system(size: 14)).textSelection(.enabled)
                    FlowRow(spacing: 6) {
                        Chip(text: "⛔ Blocked on device · \(c.blockReason ?? "")", style: .bad)
                        Chip(text: String(format: "attack %d%% · harm %d%%", Int((c.attack * 100).rounded()), Int((c.harm * 100).rounded())), style: .dim)
                        UnitBadge(timing: c.timing)
                    }
                }
            }
        case .user(let c):
            HStack {
                Spacer(minLength: 120)
                bubble(fill: Theme.userBubble, stroke: .clear, corner: .bottomTrailing) {
                    who("You · as sent")
                    Text(maskedAttributed(c.masked)).font(.system(size: 14)).lineSpacing(2).textSelection(.enabled)
                    FlowRow(spacing: 6) {
                        if !c.pii.isEmpty { Chip(text: "🔒 \(c.pii.count) personal info masked", style: .warn) }
                        Chip(text: "→ \(c.domain.first?.name ?? "?")")
                        if c.factcheck { Chip(text: "needs fact-check", style: .warn) }
                        UnitBadge(timing: c.timing)
                    }
                }
            }
        case .bot:
            HStack {
                bubble(fill: Theme.panel2, stroke: Theme.line, corner: .bottomLeading) {
                    if message.typing {
                        who("Assistant")
                        Text("typing…").italic().foregroundStyle(Theme.dim)
                    } else {
                        HStack(spacing: 4) {
                            who("Assistant")
                            Text("· scripted reply, click to edit").font(.system(size: 11)).foregroundStyle(Theme.dim)
                        }
                        RichTextView(
                            text: Binding(get: { message.text }, set: { model.editReply(message.id, $0) }),
                            highlightedText: message.reply?.answer,
                            highlights: (message.reply?.unsupported ?? []).map {
                                Highlight(start: $0.start, end: $0.end, style: .unsupported,
                                          tooltip: "Not supported by the source (\(Int(($0.probability * 100).rounded()))%)")
                            },
                            fontSize: 14, scrolls: false)
                        HStack(spacing: 6) {
                            replyStatus
                            Spacer(minLength: 8)
                            Button("↻ re-check") { model.checkReply(message.id) }
                                .buttonStyle(.plain).font(.system(size: 11.5)).foregroundStyle(Theme.accent)
                        }
                    }
                }
                .frame(maxWidth: 560, alignment: .leading)
                Spacer(minLength: 120)
            }
        }
    }

    @ViewBuilder var replyStatus: some View {
        if message.checking {
            Chip(text: "checking against source…", style: .dim)
        } else if let r = message.reply {
            let n = r.unsupported.count
            if r.answer != message.text {
                Chip(text: "edited — re-check", style: .dim)
            } else if n > 0 {
                Chip(text: "⚠ \(n) claim\(n > 1 ? "s" : "") not in the source", style: .bad)
            } else {
                Chip(text: "✓ supported by the source", style: .ok)
            }
            UnitBadge(timing: r.timing)
        }
    }

    func who(_ s: String) -> some View {
        Text(s).font(.system(size: 11)).foregroundStyle(Theme.dim)
    }

    enum Corner { case bottomLeading, bottomTrailing }

    func bubble<C: View>(fill: Color, stroke: Color, corner: Corner, @ViewBuilder _ content: () -> C) -> some View {
        let shape = UnevenRoundedRectangle(
            topLeadingRadius: 16, bottomLeadingRadius: corner == .bottomLeading ? 4 : 16,
            bottomTrailingRadius: corner == .bottomTrailing ? 4 : 16, topTrailingRadius: 16)
        return VStack(alignment: .leading, spacing: 4) { content() }
            .padding(.horizontal, 14).padding(.vertical, 10)
            .background(fill, in: shape)
            .overlay(shape.stroke(stroke))
            .transition(.opacity.combined(with: .move(edge: .bottom)))
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
