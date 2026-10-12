import FluidUse
import SwiftUI

@available(macOS 15.0, *)
struct ContentView: View {
    @State private var player = MarioPlayer()
    let modelDirectory: URL?  // JEV-9B bundle; nil = download on first use
    let bridgeCommand: [String]

    var body: some View {
        HStack(alignment: .top, spacing: 24) {
            VStack(alignment: .leading, spacing: 10) {
                ZStack(alignment: .topTrailing) {
                    if let frame = player.frame {
                        Image(decorative: frame, scale: 1).resizable().interpolation(.none)
                            .aspectRatio(256 / 240, contentMode: .fit)
                    } else {
                        Rectangle().fill(.black).aspectRatio(256 / 240, contentMode: .fit)
                    }
                    if player.thinking {
                        Text("thinking…").font(.system(.body, design: .monospaced)).padding(6)
                            .background(.black.opacity(0.6)).foregroundStyle(.white).padding(8)
                    }
                }
                .frame(minWidth: 384, maxWidth: .infinity)
                HStack {
                    Text(player.level.isEmpty ? "" : "World \(player.level)").font(.title3.bold())
                    Spacer()
                    ForEach(Array(player.results.enumerated()), id: \.offset) { _, r in
                        Text(r.level).font(.system(.caption, design: .monospaced))
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(r.flag ? Color.green.opacity(0.35) : Color.red.opacity(0.35), in: Capsule())
                    }
                }
                ProgressView(value: Double(min(player.x, MarioPlayer.flagX)), total: Double(MarioPlayer.flagX))
                Text(player.status).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 14) {
                Picker("Model", selection: $player.brain) {
                    ForEach(MarioPlayer.Brain.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented).labelsHidden()
                if player.brain == .tiny {
                    Text(
                        "Every move: a \(player.studentParameters.isEmpty ? "small" : player.studentParameters + "-parameter") model looks at the pixels around Mario"
                    )
                    .font(.callout).foregroundStyle(.secondary)
                    if !player.studentPlacement.isEmpty {
                        Label(player.studentPlacement, systemImage: "cpu").font(.callout.bold()).foregroundStyle(.green)
                    }
                    ForEach(Array(MarioStudentPolicy.moves.enumerated()), id: \.offset) { i, move in
                        let p = i < player.moveProbabilities.count ? player.moveProbabilities[i] : 0
                        HStack {
                            Text(move.rawValue).font(.headline).frame(width: 50, alignment: .leading)
                            Bar(value: p, color: move == player.move ? .orange : .gray)
                            Text(String(format: "%.2f", p)).font(.system(.caption, design: .monospaced))
                        }
                    }
                } else {
                    Text("Every move: JEV-9B answers three yes/no questions about the screen just ahead of Mario")
                        .font(.callout).foregroundStyle(.secondary)
                }
                ForEach(player.brain == .jev ? MarioPlayer.questions : []) { q in
                    HStack(spacing: 12) {
                        Group {
                            if let image = player.viewImages[q.view] {
                                Image(decorative: image, scale: 1).resizable().interpolation(.none).scaledToFit()
                            } else {
                                Rectangle().fill(.quaternary)
                            }
                        }
                        .frame(width: 96, height: 96)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(q.id).font(.headline)
                            let p = player.probabilities[q.id] ?? 0
                            Bar(value: p, color: p > q.threshold ? .orange : .gray)
                            Text(String(format: "P(yes) %.2f", p)).font(.system(.caption, design: .monospaced))
                        }
                    }
                }
                HStack(alignment: .firstTextBaseline) {
                    Text(player.move.rawValue.uppercased()).font(.system(size: 34, weight: .bold, design: .rounded))
                        .foregroundStyle(player.move == .run ? .green : .orange)
                    Spacer()
                    VStack(alignment: .trailing) {
                        Text(
                            String(
                                format: player.lastMilliseconds < 10 ? "%.2f ms" : "%.0f ms", player.lastMilliseconds)
                        ).font(
                            .system(.title3, design: .monospaced))
                        Text("decision \(player.decisions)").foregroundStyle(.secondary)
                    }
                }
                HStack {
                    Toggle("Max speed", isOn: $player.turbo).toggleStyle(.switch)
                    Spacer()
                    Text(String(format: "%.0f fps · %.1f× real time", player.gameFPS, player.gameFPS / 60))
                        .font(.system(.body, design: .monospaced))
                        .foregroundStyle(player.turbo ? .orange : .secondary)
                }
                LevelGrid(selected: $player.selected, reliable: MarioPlayer.reliable, results: player.results)
                Button(player.finished ? "Play selected" : "Restart with selected") {
                    player.restart(modelDirectory: modelDirectory, bridgeCommand: bridgeCommand)
                }
                .buttonStyle(.borderedProminent)
                .disabled(player.selected.isEmpty)
            }
            .frame(width: 340)
        }
        .padding(24)
        .task { player.start(modelDirectory: modelDirectory, bridgeCommand: bridgeCommand) }
    }
}

/// All 32 levels as toggles (4 per world); a dot marks levels the student clears from every start, and finished
/// levels show green (flag) or red.
@available(macOS 15.0, *)
struct LevelGrid: View {
    @Binding var selected: Set<String>
    let reliable: Set<String>
    let results: [(level: String, flag: Bool)]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Levels").font(.headline)
                Spacer()
                Button("Reliable") { selected = reliable }.buttonStyle(.link)
                Button("All") { selected = Set(MarioPlayer.allLevels) }.buttonStyle(.link)
                Button("None") { selected = [] }.buttonStyle(.link)
            }
            Grid(horizontalSpacing: 4, verticalSpacing: 4) {
                ForEach(0..<4) { row in
                    GridRow {
                        ForEach(0..<8) { col in
                            cell(MarioPlayer.allLevels[col * 4 + row])
                        }
                    }
                }
            }
        }
    }

    func cell(_ level: String) -> some View {
        let on = selected.contains(level)
        let outcome = results.last { $0.level == level }?.flag
        return Button {
            if on { selected.remove(level) } else { selected.insert(level) }
        } label: {
            HStack(spacing: 2) {
                Text(level).font(.system(.caption, design: .monospaced))
                if reliable.contains(level) { Circle().fill(.green).frame(width: 4, height: 4) }
            }
            .frame(maxWidth: .infinity, minHeight: 22)
            .background(
                outcome.map { $0 ? Color.green.opacity(0.35) : Color.red.opacity(0.3) }
                    ?? (on ? Color.accentColor.opacity(0.35) : Color.secondary.opacity(0.12)),
                in: RoundedRectangle(cornerRadius: 5)
            )
            .overlay(RoundedRectangle(cornerRadius: 5).stroke(on ? Color.accentColor : .clear, lineWidth: 1))
        }
        .buttonStyle(.plain)
    }
}

/// A probability bar that redraws instantly (ProgressView animates and lags behind sub-second decisions).
struct Bar: View {
    let value: Float
    let color: Color

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(.quaternary)
                Capsule().fill(color).frame(width: geo.size.width * CGFloat(min(max(value, 0), 1)))
            }
        }
        .frame(height: 8)
        .transaction { $0.animation = nil }
    }
}
