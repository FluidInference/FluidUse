import SwiftUI

extension Color {
    init(hex: UInt32, alpha: Double = 1) {
        self.init(
            .sRGB, red: Double((hex >> 16) & 0xff) / 255, green: Double((hex >> 8) & 0xff) / 255,
            blue: Double(hex & 0xff) / 255, opacity: alpha)
    }
}

/// GitHub dark palette, as in the web demo.
enum Theme {
    static let bg = Color(hex: 0x0d1117)
    static let bg2 = Color(hex: 0x010409)
    static let panel = Color(hex: 0x151b23)
    static let line = Color(hex: 0x3d444d)
    static let line2 = Color(hex: 0x2a313c)
    static let ink = Color(hex: 0xf0f6fc)
    static let dim = Color(hex: 0x9198a1)
    static let link = Color(hex: 0x4493f8)
    static let green = Color(hex: 0x3fb950)
    static let purple = Color(hex: 0xab7df8)
    static let btn = Color(hex: 0x212830)
    static let live = Color(.sRGB, red: 137 / 255, green: 87 / 255, blue: 229 / 255, opacity: 0.16)
    static let gradient = LinearGradient(
        colors: [Color(hex: 0x8957e5), Color(hex: 0x4493f8)], startPoint: .leading, endPoint: .trailing)
}

struct ContentView: View {
    @EnvironmentObject var model: TriageModel

    var body: some View {
        VStack(spacing: 0) {
            Header(open: model.openCount)
            VStack(spacing: 16) {
                Toolbar()
                HStack(alignment: .top, spacing: 16) {
                    Sidebar(stats: model.stats)
                    VStack(spacing: 16) {
                        if model.phase == .running || model.phase == .finished {
                            StatsBar(stats: model.stats, total: model.rows.count)
                        }
                        IssueBox(rows: model.rows, open: model.openCount, scroll: model.scroll)
                    }
                }
                .frame(maxHeight: .infinity, alignment: .top)
            }
            .padding(24)
            .frame(maxWidth: 1560)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(Theme.bg)
        .foregroundStyle(Theme.ink)
        .font(.system(size: 14))
        .preferredColorScheme(.dark)
        .sheet(item: $model.selected) { row in DetailSheet(row: row) }
    }
}

// MARK: - Header and toolbar

struct Header: View {
    let open: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Image(systemName: "book.closed")
                    .font(.system(size: 13))
                    .foregroundStyle(Theme.dim)
                    .frame(width: 32, height: 32)
                    .overlay(Circle().stroke(Theme.line))
                Text("routerlabs").font(.system(size: 16))
                Text("/").font(.system(size: 16)).foregroundStyle(Theme.dim)
                Text("semantic-switch").font(.system(size: 16, weight: .semibold))
                Text("Public")
                    .font(.system(size: 12)).foregroundStyle(Theme.dim)
                    .padding(.horizontal, 7)
                    .overlay(Capsule().stroke(Theme.line))
                    .padding(.leading, 6)
                Spacer()
                Text("Local demo · labels by an on-device model")
                    .font(.system(size: 11)).foregroundStyle(Theme.dim)
                    .padding(.horizontal, 8).padding(.vertical, 1)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line, style: StrokeStyle(dash: [3, 2])))
            }
            .frame(height: 32)
            HStack(spacing: 6) {
                tab("Code")
                HStack(spacing: 6) {
                    Image(systemName: "smallcircle.filled.circle")
                    Text("Issues").fontWeight(.semibold)
                    Text("\(open)")
                        .font(.system(size: 12, weight: .medium))
                        .padding(.horizontal, 6)
                        .background(Capsule().fill(Color(hex: 0x2f3742)))
                }
                .padding(.horizontal, 10).padding(.vertical, 8)
                .overlay(alignment: .bottom) { Rectangle().fill(Color(hex: 0xf78166)).frame(height: 2) }
                ForEach(["Pull requests", "Discussions", "Actions", "Projects", "Insights"], id: \.self) { tab($0) }
            }
        }
        .padding(.horizontal, 24).padding(.top, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.bg2)
        .overlay(alignment: .bottom) { Rectangle().fill(Theme.line2).frame(height: 1) }
    }

    func tab(_ name: String) -> some View {
        Text(name).padding(.horizontal, 10).padding(.vertical, 8)
    }
}

struct Toolbar: View {
    @EnvironmentObject var model: TriageModel

    var body: some View {
        HStack(spacing: 8) {
            Text("is:issue sort:created-desc")
                .foregroundStyle(Theme.dim)
                .padding(.horizontal, 12).padding(.vertical, 5)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 6).fill(Theme.bg))
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line))
            FakeButton(title: "Labels")
            FakeButton(title: "Milestones")
            Button {
                Task { await model.triageAll() }
            } label: {
                Text(triageTitle)
                    .fontWeight(.semibold)
                    .padding(.horizontal, 16).padding(.vertical, 5)
                    .background(RoundedRectangle(cornerRadius: 6).fill(Theme.gradient))
                    .opacity(model.phase == .ready ? 1 : 0.6)
            }
            .buttonStyle(.plain)
            .disabled(model.phase != .ready)
            FakeButton(title: "New issue", fill: Color(hex: 0x238636))
        }
    }

    var triageTitle: String {
        switch model.phase {
        case .loadingIssues: "Loading issues…"
        case .loadingModel: "Loading model…"
        case .ready: "✦ Triage with Decision 2.0"
        case .running: "✦ Triaging…"
        case .finished: "✓ Triaged"
        case .failed: "✦ Triage with Decision 2.0"
        }
    }
}

struct FakeButton: View {
    let title: String
    var fill = Theme.btn

    var body: some View {
        Text(title)
            .fontWeight(.medium)
            .padding(.horizontal, 16).padding(.vertical, 5)
            .background(RoundedRectangle(cornerRadius: 6).fill(fill))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line))
    }
}

// MARK: - Sidebar and stats

struct Sidebar: View {
    @ObservedObject var stats: TriageStats
    @EnvironmentObject var model: TriageModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("ROUTED TO WORKGROUPS")
                .font(.system(size: 12, weight: .semibold)).kerning(0.5)
                .foregroundStyle(Theme.dim)
                .padding(.bottom, 12)
            let maxCount = max(1, stats.workgroups.map(\.count).max() ?? 1)
            ForEach(stats.workgroups) { wg in
                VStack(spacing: 4) {
                    HStack {
                        Text(wg.name.replacingOccurrences(of: "wg/", with: "")
                            .replacingOccurrences(of: "owner/", with: ""))
                        Spacer()
                        Text("\(wg.count)").fontWeight(.bold).monospacedDigit()
                    }
                    .font(.system(size: 13))
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Theme.line2)
                            Capsule().fill(LabelChip.base(wg.name))
                                .frame(width: geo.size.width * Double(wg.count) / Double(maxCount))
                        }
                    }
                    .frame(height: 8)
                }
                .padding(.bottom, 10)
            }
            if let note {
                Text(note).font(.system(size: 12)).foregroundStyle(Theme.dim).padding(.top, 4)
            }
        }
        .padding(.horizontal, 16).padding(.vertical, 14)
        .frame(width: 290, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 6).fill(Theme.panel))
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line))
    }

    var note: String? {
        switch model.phase {
        case .loadingIssues: "Loading issues…"
        case .loadingModel: "Loading Decision-2.0-Kai-0.6B…"
        case .ready: "Press ✦ Triage to start"
        case .running, .finished: nil
        case .failed(let message): message
        }
    }
}

struct StatsBar: View {
    @ObservedObject var stats: TriageStats
    let total: Int

    var body: some View {
        VStack(spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 18) {
                stat("\(stats.done)", "/ \(total) issues triaged")
                stat("\(stats.decisions)", "labels decided")
                stat(stats.avgMs.map { String(format: "%.0f", $0) } ?? "–", "ms per issue")
                stat(stats.rate.map { String(format: "%.1f", $0) } ?? "–", "issues / sec")
                Spacer(minLength: 0)
                (Text("5 decisions per issue in one call · ")
                    + Text("Decision-2.0-Kai-0.6B").foregroundColor(Theme.ink).bold()
                    + Text(" · Core ML on this Mac"))
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                    .layoutPriority(1)
            }
            .font(.system(size: 13))
            .foregroundStyle(Theme.dim)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Theme.line2)
                    Capsule().fill(Theme.gradient)
                        .frame(width: geo.size.width * Double(stats.done) / Double(max(1, total)))
                }
            }
            .frame(height: 4)
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
        .background(RoundedRectangle(cornerRadius: 6).fill(Theme.panel))
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line))
    }

    func stat(_ value: String, _ caption: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 4) {
            Text(value).font(.system(size: 18, weight: .bold)).monospacedDigit().foregroundStyle(Theme.ink)
            Text(caption)
        }
        .fixedSize()
    }
}

// MARK: - Issue list

struct IssueBox: View {
    let rows: [RowState]
    let open: Int
    let scroll: ScrollState

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 16) {
                HStack(spacing: 6) {
                    Image(systemName: "smallcircle.filled.circle")
                    Text("\(open) Open")
                }
                .foregroundStyle(Theme.ink).fontWeight(.semibold)
                HStack(spacing: 6) {
                    Image(systemName: "checkmark")
                    Text("\(rows.count - open) Closed")
                }
                Spacer(minLength: 16)
                HStack(spacing: 22) {
                    ForEach(["Author", "Labels", "Projects", "Milestones", "Assignees", "Sort"], id: \.self) {
                        Text("\($0) ▾")
                    }
                }
                .lineLimit(1)
            }
            .foregroundStyle(Theme.dim)
            .padding(.horizontal, 16).padding(.vertical, 14)
            .background(Theme.panel)
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(rows) { row in IssueRow(row: row) }
                    }
                }
                .background(ScrollDriver(scroll: scroll, proxy: proxy))
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line))
        .frame(maxHeight: .infinity)
    }
}

/// Keeps the current row centered; the list itself observes nothing that changes during a run.
struct ScrollDriver: View {
    @ObservedObject var scroll: ScrollState
    let proxy: ScrollViewProxy

    var body: some View {
        Color.clear.onChange(of: scroll.target) { _, target in
            guard let target else { return }
            let anchor = scroll.anchor
            withAnimation(.easeInOut(duration: 0.25)) { proxy.scrollTo(target, anchor: anchor) }
            guard anchor == .top else { return }
            // lazy rows above are estimated until laid out; settle once the jump has landed
            Task { @MainActor in
                try? await Task.sleep(for: .milliseconds(400))
                proxy.scrollTo(target, anchor: anchor)
            }
        }
    }
}

struct IssueRow: View {
    @ObservedObject var row: RowState
    @EnvironmentObject var model: TriageModel
    @State private var hover = false

    var body: some View {
        let issue = row.issue
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: issue.isOpen ? "smallcircle.filled.circle" : "checkmark.circle")
                .foregroundStyle(issue.isOpen ? Theme.green : Theme.purple)
                .padding(.top, 3)
            VStack(alignment: .leading, spacing: 2) {
                TitleFlow {
                    Text(issue.title).font(.system(size: 16, weight: .semibold))
                    if let result = row.result {
                        let pop = row.labeledAt.map { Date().timeIntervalSince($0) < 0.5 } ?? false
                        ForEach(result.labels, id: \.self) { LabelChip(name: $0, pop: pop) }
                    }
                }
                Text("#\(issue.number) \(issue.isOpen ? "opened" : "was closed") \(issue.created) by \(issue.author)")
                    .font(.system(size: 12)).foregroundStyle(Theme.dim)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            HStack(spacing: 3) {
                if issue.comments > 0 {
                    Image(systemName: "bubble.left")
                    Text("\(issue.comments)")
                }
            }
            .font(.system(size: 12)).foregroundStyle(Theme.dim)
            .frame(width: 50, alignment: .trailing)
            .padding(.top, 3)
        }
        .padding(.horizontal, 16).padding(.vertical, 8)
        .background(row.live ? Theme.live : hover ? Theme.panel : Color.clear)
        .overlay(alignment: .top) { Rectangle().fill(Theme.line2).frame(height: 1) }
        .contentShape(Rectangle())
        .onHover { hover = $0 }
        .onTapGesture { model.selected = row }
        .id(row.id)
    }
}

/// A label chip: the label color at 18 % fill and 45 % border, text lightened 45 % toward white.
struct LabelChip: View {
    let name: String
    var pop = false
    @State private var shown = false

    static func base(_ name: String) -> Color {
        let (r, g, b) = LabelPalette.rgb(name)
        return Color(.sRGB, red: r, green: g, blue: b)
    }

    var body: some View {
        let (r, g, b) = LabelPalette.rgb(name)
        let light = { (v: Double) in v + (1 - v) * 0.45 }
        Text(name)
            .font(.system(size: 12, weight: .medium))
            .lineLimit(1)
            .foregroundStyle(Color(.sRGB, red: light(r), green: light(g), blue: light(b)))
            .padding(.horizontal, 7)
            .frame(height: 20)
            .background(Capsule().fill(Color(.sRGB, red: r, green: g, blue: b, opacity: 0.18)))
            .overlay(Capsule().stroke(Color(.sRGB, red: r, green: g, blue: b, opacity: 0.45)))
            .scaleEffect(!pop || shown ? 1 : 0.2)
            .opacity(!pop || shown ? 1 : 0)
            .onAppear {
                guard pop else { return }
                withAnimation(.spring(response: 0.35, dampingFraction: 0.75)) { shown = true }
            }
    }
}

/// The title, then chips flowing after it on the same line when they fit, else wrapping below.
struct TitleFlow: Layout {
    var spacing: CGFloat = 5
    var lineSpacing: CGFloat = 4

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let frames = arrange(width: proposal.width ?? 10_000, subviews)
        let width = frames.map(\.maxX).max() ?? 0
        let height = frames.map(\.maxY).max() ?? 0
        return CGSize(width: proposal.width ?? width, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        for (frame, subview) in zip(arrange(width: bounds.width, subviews), subviews) {
            subview.place(
                at: CGPoint(x: bounds.minX + frame.minX, y: bounds.minY + frame.minY),
                proposal: ProposedViewSize(frame.size))
        }
    }

    private func arrange(width: CGFloat, _ subviews: Subviews) -> [CGRect] {
        guard let first = subviews.first else { return [] }
        let title = first.sizeThatFits(ProposedViewSize(width: width, height: nil))
        var frames = [CGRect(origin: .zero, size: title)]
        // chips sit beside the title's last line; a wrapped title fills the width, so chips go below it
        let firstLine: CGFloat = 21
        var x = title.width + spacing + 2
        var lineTop = max(0, title.height - firstLine)
        var lineHeight = min(title.height, firstLine)
        for subview in subviews.dropFirst() {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > width, x > 0 {
                x = 0
                lineTop += lineHeight + lineSpacing
                lineHeight = size.height
            }
            frames.append(CGRect(x: x, y: lineTop + (lineHeight - size.height) / 2, width: size.width, height: size.height))
            x += size.width + spacing
        }
        return frames
    }
}

// MARK: - Detail sheet

struct DetailSheet: View {
    @ObservedObject var row: RowState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .firstTextBaseline) {
                (Text(row.issue.title).font(.system(size: 18, weight: .semibold))
                    + Text("  #\(row.issue.number)").font(.system(size: 18)).foregroundColor(Theme.dim))
                Spacer()
                Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            if let r = row.result {
                HStack(spacing: 4) { ForEach(r.labels, id: \.self) { LabelChip(name: $0) } }
                    .padding(.vertical, 6)
                Text("5 decisions in one call · \(String(format: "%.0f", r.ms)) ms")
                    .font(.system(size: 12)).foregroundStyle(Theme.dim)
                section("Type")
                bars(r.type)
                section("Owning workgroup")
                bars(r.wg)
                section("Priority · urgency score \(String(format: "%.2f", r.priorityScore)) / 2")
                bars(r.priority) { ["0": "P2 · nice-to-have", "1": "P1 · important", "2": "P0 · critical"][$0] ?? $0 }
                section("Flags")
                bars([("needs-info", r.needsInfo), ("good first issue", r.goodFirst)])
            } else {
                Text("Not triaged yet — press ✦ Triage with Decision 2.0.")
                    .foregroundStyle(Theme.dim).padding(.top, 12)
            }
        }
        .padding(.horizontal, 20).padding(.vertical, 18)
        .frame(width: 640, alignment: .leading)
        .background(Theme.panel)
        .foregroundStyle(Theme.ink)
        .font(.system(size: 13))
        .preferredColorScheme(.dark)
    }

    func section(_ title: String) -> some View {
        Text(title.uppercased())
            .font(.system(size: 12)).kerning(0.5).foregroundStyle(Theme.dim)
            .padding(.top, 12).padding(.bottom, 4)
    }

    func bars(_ probs: [(String, Double)], _ format: @escaping (String) -> String = { $0 }) -> some View {
        VStack(spacing: 4) {
            ForEach(probs.sorted { $0.1 > $1.1 }, id: \.0) { key, p in
                HStack(spacing: 8) {
                    Text(format(key)).frame(width: 230, alignment: .leading).lineLimit(1)
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Theme.line2)
                            Capsule().fill(Theme.link).frame(width: geo.size.width * p)
                        }
                    }
                    .frame(height: 8)
                    Text("\(Int((p * 100).rounded()))%")
                        .monospacedDigit().foregroundStyle(Theme.dim)
                        .frame(width: 44, alignment: .trailing)
                }
            }
        }
    }
}
