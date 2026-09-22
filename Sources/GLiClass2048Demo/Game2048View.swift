import Game2048
import SwiftUI

struct Game2048View: View {
    @EnvironmentObject private var model: Game2048Model

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            board
            scoreboard
            controls
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .alert(
            "2048 model",
            isPresented: Binding(
                get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })
        ) {
            Button("OK") { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("on-device GLiClass plays 2048").font(.title2.bold())
            Text("A heuristic shortlists safe swipes; GLiClass compares their resulting boards in one encoder pass.")
                .font(.caption).foregroundStyle(.secondary)
            HStack(spacing: 6) {
                Text(model.loadStatus).font(.caption.monospaced())
                    .foregroundStyle(model.hasLoadedModel ? Color.green : Color.secondary)
                if model.gamesPlayed > 0 {
                    Text("· game \(model.gamesPlayed + 1)").font(.caption.monospaced()).foregroundStyle(.orange)
                }
            }
        }
    }

    private var board: some View {
        VStack(spacing: 8) {
            ForEach(0..<Game2048.size, id: \.self) { row in
                HStack(spacing: 8) {
                    ForEach(0..<Game2048.size, id: \.self) { column in
                        tile(model.board[row][column])
                    }
                }
            }
        }
        .padding(10)
        .background(Color(red: 0.42, green: 0.38, blue: 0.34))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .aspectRatio(1, contentMode: .fit)
    }

    private func tile(_ value: Int) -> some View {
        RoundedRectangle(cornerRadius: 8)
            .fill(tileColor(value))
            .overlay {
                if value > 0 {
                    Text("\(value)")
                        .font(.system(size: value >= 1024 ? 19 : 25, weight: .bold, design: .rounded))
                        .foregroundStyle(value <= 4 ? Color.black.opacity(0.72) : Color.white)
                        .minimumScaleFactor(0.55)
                }
            }
            .aspectRatio(1, contentMode: .fit)
    }

    private func tileColor(_ value: Int) -> Color {
        switch value {
        case 0: return Color.white.opacity(0.15)
        case 2: return Color(red: 0.93, green: 0.89, blue: 0.82)
        case 4: return Color(red: 0.93, green: 0.86, blue: 0.70)
        case 8: return Color(red: 0.95, green: 0.58, blue: 0.32)
        case 16: return Color(red: 0.96, green: 0.45, blue: 0.25)
        case 32: return Color(red: 0.96, green: 0.36, blue: 0.23)
        case 64: return Color(red: 0.95, green: 0.25, blue: 0.15)
        case 128: return Color(red: 0.93, green: 0.78, blue: 0.28)
        case 256: return Color(red: 0.93, green: 0.75, blue: 0.20)
        case 512: return Color(red: 0.93, green: 0.70, blue: 0.14)
        case 1024: return Color(red: 0.91, green: 0.65, blue: 0.08)
        default: return Color(red: 0.35, green: 0.25, blue: 0.65)
        }
    }

    private var scoreboard: some View {
        HStack(spacing: 0) {
            readout(model.elapsedText, "time", .primary)
            Divider().frame(height: 28)
            readout("\(model.score)", "score", .secondary)
            Divider().frame(height: 28)
            readout("\(model.moves)", "moves", .green)
            Divider().frame(height: 28)
            readout("\(model.maximumTile)", "max tile", .orange)
            Divider().frame(height: 28)
            readout(
                model.modelMillisecondsPerMove > 0
                    ? String(format: "%.1f", model.modelMillisecondsPerMove) : "–",
                "model ms/move", .blue)
        }
        .padding(.vertical, 8)
        .background(Color.secondary.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func readout(_ value: String, _ label: String, _ color: Color) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.system(size: 16, weight: .semibold, design: .monospaced))
                .foregroundStyle(color).minimumScaleFactor(0.6).lineLimit(1)
            Text(label).font(.system(size: 9)).foregroundStyle(.secondary).lineLimit(1)
        }
        .frame(maxWidth: .infinity)
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button(model.hasLoadedModel ? "Loaded" : "Load model") { model.loadModel() }
                    .disabled(model.isLoading || model.hasLoadedModel || model.policy != .gliclass)
                Button(model.isRunning ? "Pause" : (model.isOver ? "Restart" : "Play")) { model.toggle() }
                    .keyboardShortcut(.space, modifiers: [])
                Button("Reset") { model.reset() }.disabled(model.isRunning)
                Spacer()
                if let direction = model.lastDirection {
                    Text("last: \(direction.rawValue)").font(.caption.monospaced()).foregroundStyle(.secondary)
                }
            }
            Toggle("Marathon: start the next seeded game after game over", isOn: $model.marathon)
                .font(.caption)
            Picker("Policy", selection: $model.policy) {
                ForEach(Game2048Model.Policy.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .disabled(model.isRunning)
            HStack {
                Text("Candidates").font(.caption)
                Picker("", selection: $model.candidateCount) {
                    Text("top 2").tag(2)
                    Text("top 3").tag(3)
                    Text("all 4").tag(4)
                }
                .pickerStyle(.segmented)
                .disabled(model.isRunning)
            }
            HStack {
                Text("Override margin").font(.caption)
                Slider(
                    value: Binding(
                        get: { Double(model.minimumMargin) }, set: { model.minimumMargin = Float($0) }),
                    in: 0...1, step: 0.05
                )
                .disabled(model.isRunning)
                Text(String(format: "%.2f", model.minimumMargin)).font(.caption.monospaced()).frame(width: 34)
            }
            HStack {
                Text("Seed").font(.caption)
                TextField("seed", value: $model.seed, format: .number).frame(width: 90).disabled(model.isRunning)
                Spacer()
            }
            Text(String(format: "Pause per move: %.0f ms (may raise model latency)", model.moveDelayMs))
                .font(.caption)
            Slider(value: $model.moveDelayMs, in: 0...500, step: 10)
        }
    }
}
