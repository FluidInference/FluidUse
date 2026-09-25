import SortAnything
import SwiftUI

private struct FramesKey: PreferenceKey {
    static let defaultValue: [String: CGRect] = [:]
    static func reduce(value: inout [String: CGRect], nextValue: () -> [String: CGRect]) {
        value.merge(nextValue()) { $1 }
    }
}

extension View {
    fileprivate func reportFrame(_ key: String) -> some View {
        background(
            GeometryReader { proxy in
                Color.clear.preference(key: FramesKey.self, value: [key: proxy.frame(in: .named("board"))])
            })
    }
}

private let incomingKey = "__incoming"

struct ContentView: View {
    @EnvironmentObject private var model: SortModel
    @State private var frames: [String: CGRect] = [:]
    @State private var newCategory = ""

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            categoryBar
            Divider()
            switch model.phase {
            case .loading(let message):
                status(message, spinning: true)
            case .failed(let message):
                status("Failed: \(message)", spinning: false)
            default:
                board
            }
            Divider()
            footer
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    // MARK: Header and controls

    private var header: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .center, spacing: 18) {
                titleBlock
                Spacer(minLength: 12)
                stats
                controls
            }
            VStack(alignment: .leading, spacing: 10) {
                titleBlock
                HStack(alignment: .center, spacing: 14) {
                    stats
                    Spacer(minLength: 8)
                    controls
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    private var titleBlock: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Sort anything").font(.system(size: 26, weight: .bold))
            Text("GLiNER2.5-Decide · Core ML · on this Mac · categories chosen at run time")
                .font(.callout).foregroundStyle(.secondary).lineLimit(1)
        }
    }

    @ViewBuilder
    private var stats: some View {
        HStack(spacing: 16) {
            stat("Sorted", "\(model.sorted) / \(model.total)")
            stat("Items / s", model.sorted > 0 ? String(format: "%.0f", model.itemsPerSecond) : "–")
            stat("Elapsed", String(format: "%.1f s", model.elapsed))
            stat("ms / item", model.medianMilliseconds.map { String(format: "%.1f", $0) } ?? "–")
            stat("Matches label", model.accuracy.map { String(format: "%.1f%%", $0 * 100) } ?? "–")
        }
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(alignment: .trailing, spacing: 2) {
            Text(title.uppercased()).font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
            Text(value).font(.system(size: 20, weight: .semibold, design: .rounded)).monospacedDigit()
                .contentTransition(.numericText()).fixedSize()
        }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Button(model.phase == .running ? "Pause" : "Start") { model.toggleRun() }
                    .keyboardShortcut(.space, modifiers: [])
                    .disabled(!(model.phase == .ready || model.phase == .running || model.phase == .paused))
                    .buttonStyle(.borderedProminent)
                Button("Reset") { model.reset() }.disabled(model.phase == .running)
            }
            Picker("Mode", selection: $model.mode) {
                ForEach(SortModel.Mode.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented).labelsHidden().frame(width: 150)
            if model.mode == .show {
                HStack(spacing: 6) {
                    Text("Pace").font(.caption).fixedSize()
                    Slider(value: $model.pace, in: 2...40).frame(width: 90)
                    Text(String(format: "%.0f/s", model.pace)).font(.caption).monospacedDigit().fixedSize()
                }
            } else {
                Text("\(SortModel.turboInFlight) calls in flight").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var categoryBar: some View {
        HStack(spacing: 8) {
            Text("Categories").font(.callout.weight(.semibold))
            FlowLayout(spacing: 6) {
                ForEach(model.categories, id: \.self) { name in
                    HStack(spacing: 4) {
                        Text(name)
                        Button {
                            model.removeCategory(name)
                        } label: {
                            Image(systemName: "xmark").font(.caption2.weight(.bold))
                        }
                        .buttonStyle(.plain).foregroundStyle(.secondary)
                    }
                    .padding(.horizontal, 10).padding(.vertical, 5)
                    .background(Capsule().fill(color(for: name).opacity(0.18)))
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            TextField("Add a category…", text: $newCategory)
                .textFieldStyle(.roundedBorder).frame(minWidth: 110, idealWidth: 180, maxWidth: 180)
                .onSubmit {
                    model.addCategory(newCategory)
                    newCategory = ""
                }
        }
        .padding(.horizontal, 20).padding(.vertical, 10)
    }

    // MARK: Board

    private var board: some View {
        GeometryReader { outer in
            let incomingWidth = min(240, max(170, outer.size.width * 0.2))
            HStack(alignment: .top, spacing: 14) {
                incoming(showQueue: outer.size.height > 520).frame(width: incomingWidth)
                GeometryReader { proxy in
                    let columns = max(2, min(6, Int(proxy.size.width / 170)))
                    let rows = (model.bucketNames.count + columns - 1) / columns
                    let height = max(64, min(120, (proxy.size.height - CGFloat(rows - 1) * 10) / CGFloat(rows)))
                    ScrollView {
                        LazyVGrid(
                            columns: Array(repeating: GridItem(.flexible(), spacing: 10), count: columns), spacing: 10
                        ) {
                            ForEach(model.bucketNames, id: \.self) { name in
                                BucketView(
                                    name: name, placed: model.buckets[name] ?? [], color: color(for: name),
                                    active: model.categories.contains(name), height: height
                                )
                                .reportFrame(name)
                            }
                        }
                    }
                }
            }
        }
        .padding(14)
        .coordinateSpace(name: "board")
        .onPreferenceChange(FramesKey.self) { frames = $0 }
        .overlay(alignment: .topLeading) { flyingCard }
    }

    private func incoming(showQueue: Bool) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("INCOMING · \(model.queue.count) left").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            ZStack {
                RoundedRectangle(cornerRadius: 12).fill(Color.secondary.opacity(0.06))
                if let next = model.queue.first(where: { item in !model.flights.contains { $0.id == item.id } }) {
                    ItemCard(item: next, result: nil, tint: .secondary)
                }
            }
            .frame(height: 140)
            .reportFrame(incomingKey)
            ForEach(
                Array(model.queue.dropFirst().prefix(showQueue ? 5 : 0).enumerated()), id: \.element.id
            ) { offset, item in
                ItemCard(item: item, result: nil, tint: .secondary, compact: true)
                    .opacity(1 - Double(offset) * 0.16)
            }
            Spacer(minLength: 0)
        }
    }

    private var flyingCard: some View {
        ZStack(alignment: .topLeading) {
            ForEach(model.flights) { flight in
                if let from = frames[incomingKey], let to = frames[flight.placed.result.category] {
                    let target = flight.arrived ? to : from
                    ItemCard(
                        item: flight.placed.item, result: flight.placed.result,
                        tint: color(for: flight.placed.result.category)
                    )
                    .frame(width: from.width - 8, height: from.height - 8)
                    .scaleEffect(flight.arrived ? 0.4 : 1)
                    .opacity(flight.arrived ? 0.2 : 1)
                    .position(x: target.midX, y: target.midY)
                }
            }
        }
        .allowsHitTesting(false)
    }

    private func status(_ message: String, spinning: Bool) -> some View {
        VStack(spacing: 12) {
            if spinning { ProgressView() }
            Text(message).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var footer: some View {
        HStack {
            Text(DBpediaSample.attribution)
            Spacer()
            Text("Model: fastino/GLiNER2.5-Decide (Apache-2.0) → FluidInference/gliner2-5-decide-coreml")
        }
        .font(.caption).foregroundStyle(.secondary)
        .padding(.horizontal, 20).padding(.vertical, 8)
    }

    private func color(for name: String) -> Color {
        let palette: [Color] = [
            .blue, .orange, .green, .pink, .purple, .teal, .red, .indigo, .mint, .brown, .cyan, .yellow,
        ]
        let index = model.bucketNames.firstIndex(of: name) ?? abs(name.hashValue)
        return palette[index % palette.count]
    }
}

private struct ItemCard: View {
    let item: SortItem
    let result: Sorter.Result?
    let tint: Color
    var compact = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(item.title).font(compact ? .callout.weight(.semibold) : .headline).lineLimit(1)
            if !compact {
                Text(item.content).font(.caption).foregroundStyle(.secondary).lineLimit(4)
            }
            if let result {
                Text(
                    "→ \(result.category) · \(Int(result.confidence * 100))% · \(String(format: "%.0f ms", result.milliseconds))"
                )
                .font(.caption.weight(.semibold)).foregroundStyle(tint)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(tint.opacity(0.5), lineWidth: 1))
        .shadow(color: .black.opacity(0.08), radius: 3, y: 1)
    }
}

private struct BucketView: View {
    let name: String
    let placed: [SortModel.Placed]
    let color: Color
    let active: Bool
    let height: CGFloat

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(name).font(.headline)
                Spacer()
                Text("\(placed.count)").font(.title3.weight(.bold)).monospacedDigit()
                    .contentTransition(.numericText())
            }
            ForEach(placed.prefix(height >= 104 ? 3 : height >= 80 ? 2 : 1)) { entry in
                HStack(spacing: 4) {
                    Image(systemName: entry.matchesGold ? "checkmark.circle.fill" : "xmark.circle.fill")
                        .foregroundStyle(entry.matchesGold ? Color.green : Color.red)
                        .help("DBpedia label: \(entry.item.gold)")
                    Text(entry.item.title).lineLimit(1)
                }
                .font(.caption)
            }
            Spacer(minLength: 0)
        }
        .padding(10)
        .frame(height: height)
        .background(RoundedRectangle(cornerRadius: 12).fill(color.opacity(active ? 0.12 : 0.05)))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(color.opacity(active ? 0.45 : 0.15), lineWidth: 1))
    }
}

/// Lays children out left to right, wrapping to a new line when the width runs out.
private struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = arrange(width: proposal.width ?? .infinity, subviews: subviews)
        let height = rows.map(\.height).reduce(0, +) + CGFloat(max(rows.count - 1, 0)) * spacing
        return CGSize(width: proposal.width ?? rows.map(\.width).max() ?? 0, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var y = bounds.minY
        for row in arrange(width: bounds.width, subviews: subviews) {
            var x = bounds.minX
            for index in row.indices {
                let size = subviews[index].sizeThatFits(.unspecified)
                subviews[index].place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += row.height + spacing
        }
    }

    private func arrange(width: CGFloat, subviews: Subviews) -> [(indices: [Int], width: CGFloat, height: CGFloat)] {
        var rows: [(indices: [Int], width: CGFloat, height: CGFloat)] = []
        var current: (indices: [Int], width: CGFloat, height: CGFloat) = ([], 0, 0)
        for index in subviews.indices {
            let size = subviews[index].sizeThatFits(.unspecified)
            let needed = current.indices.isEmpty ? size.width : current.width + spacing + size.width
            if needed > width, !current.indices.isEmpty {
                rows.append(current)
                current = ([index], size.width, size.height)
            } else {
                current = (current.indices + [index], needed, max(current.height, size.height))
            }
        }
        if !current.indices.isEmpty { rows.append(current) }
        return rows
    }
}
