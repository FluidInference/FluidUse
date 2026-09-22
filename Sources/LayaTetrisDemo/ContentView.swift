import LayaTetris
import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var model: GameModel

    var body: some View {
        // Always one column: header, board, scoreboard, controls. A side-by-side layout left a
        // large dead gap beside the board at any window size worth recording.
        VStack(alignment: .leading, spacing: 10) {
            header
            board
            scoreboard
            controls
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .padding(16)
        .alert(
            "Tetris model",
            isPresented: Binding(get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })
        ) {
            Button("OK") { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
    }

    private var board: some View {
        // Capped so a tall window does not stretch the playfield into a sliver. The 10x20 ratio
        // ties width to height, so this is also what keeps the cells a sensible size.
        BoardView(board: model.board, chosen: model.chosen, evaluating: model.evaluating)
            .aspectRatio(CGFloat(TetrisGame.width) / CGFloat(TetrisGame.height), contentMode: .fit)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("on-device models play Tetris").font(.title2.bold())
            Text(
                "GLiClass compares the two strongest legal landings in one encoder pass; laya can score every landing for comparison."
            )
            .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 6) {
                Text(model.loadStatus).font(.caption.monospaced()).foregroundStyle(
                    model.hasLoadedModel ? Color.green : Color.secondary)
                if model.marathon, model.gamesPlayed > 0 {
                    Text("· game \(model.gamesPlayed + 1)").font(.caption.monospaced())
                        .foregroundStyle(.orange)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Live scoreboard under the board: the clock plus what a viewer needs to read the run.
    private var scoreboard: some View {
        HStack(spacing: 0) {
            readout(model.elapsedText, model.isOver ? "topped out" : "time", model.isOver ? .red : .primary)
            Divider().frame(height: 26)
            readout("\(model.totalPieces)", "pieces", .secondary)
            Divider().frame(height: 26)
            readout("\(model.totalLines)", "lines", .green)
            Divider().frame(height: 26)
            readout(
                model.modelMillisecondsPerPiece > 0 ? String(format: "%.1f", model.modelMillisecondsPerPiece) : "–",
                "model ms/move", .orange)
            Divider().frame(height: 26)
            readout("\(model.decisions)", "calls", .blue)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 6)
        .background(Color.secondary.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func readout(_ value: String, _ label: String, _ accent: Color) -> some View {
        VStack(spacing: 1) {
            Text(value)
                .font(.system(size: 14, weight: .semibold, design: .monospaced))
                .foregroundStyle(accent)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
            Text(label).font(.system(size: 9)).foregroundStyle(.secondary).lineLimit(1)
        }
        .frame(maxWidth: .infinity)
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Button(model.hasLoadedModel ? "Loaded" : "Load model") { model.loadModel() }
                    .disabled(model.isLoading || model.hasLoadedModel || [.heuristic, .random].contains(model.policy))
                Button(model.isRunning ? "Pause" : (model.isOver ? "Restart" : "Play")) { model.toggle() }
                    .keyboardShortcut(.space, modifiers: [])
                    .disabled(!model.hasLoadedModel)
                Button("Reset") { model.reset() }.disabled(model.isRunning)
            }
            Toggle("Harness: withhold burying moves, graded wording", isOn: $model.harness)
                .font(.caption)
                .disabled(model.isRunning)
            Toggle("Marathon: new board after each top-out", isOn: $model.marathon)
                .font(.caption)
            HStack {
                Text("Lookahead").font(.caption)
                Picker("", selection: $model.lookahead) {
                    Text("off").tag(0)
                    Text("2").tag(2)
                    Text("4").tag(4)
                    Text("6").tag(6)
                }
                .pickerStyle(.segmented)
                .disabled(model.isRunning)
            }
            Picker("Policy", selection: $model.policy) {
                ForEach(GameModel.Policy.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .disabled(model.isRunning)
            HStack {
                Text("Seed").font(.caption)
                TextField("seed", value: $model.seed, format: .number).frame(width: 70).disabled(model.isRunning)
                Spacer()
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(String(format: "Delay per scored landing: %.0f ms", model.stepDelayMs)).font(.caption)
                Slider(value: $model.stepDelayMs, in: 0...300, step: 10)
                Text(String(format: "Pause per piece: %.0f ms (may raise model latency)", model.pieceDelayMs))
                    .font(.caption)
                Slider(value: $model.pieceDelayMs, in: 0...1000, step: 20)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

}

struct BoardView: View {
    let board: [[Bool]]
    let chosen: TetrisGame.Candidate?
    let evaluating: TetrisGame.Candidate?

    var body: some View {
        Canvas { context, size in
            let columns = TetrisGame.width
            let rows = TetrisGame.height
            let cell = min(size.width / CGFloat(columns), size.height / CGFloat(rows))
            let originX = (size.width - cell * CGFloat(columns)) / 2
            let originY = (size.height - cell * CGFloat(rows)) / 2
            func rect(_ x: Int, _ y: Int) -> CGRect {
                CGRect(x: originX + CGFloat(x) * cell, y: originY + CGFloat(y) * cell, width: cell, height: cell)
                    .insetBy(dx: 1, dy: 1)
            }
            context.fill(
                Path(CGRect(x: originX, y: originY, width: cell * CGFloat(columns), height: cell * CGFloat(rows))),
                with: .color(Color.black.opacity(0.85)))
            for y in 0..<rows {
                for x in 0..<columns where board[y][x] {
                    context.fill(Path(roundedRect: rect(x, y), cornerRadius: 2), with: .color(Color.cyan.opacity(0.85)))
                }
            }
            if let evaluating {
                for (x, y) in evaluating.cells where y >= 0 {
                    context.stroke(
                        Path(roundedRect: rect(x, y), cornerRadius: 2), with: .color(Color.orange.opacity(0.9)),
                        lineWidth: 1.5)
                }
            }
            if let chosen {
                for (x, y) in chosen.cells where y >= 0 {
                    context.fill(Path(roundedRect: rect(x, y), cornerRadius: 2), with: .color(Color.green.opacity(0.9)))
                }
            }
        }
        .background(Color.black.opacity(0.9))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
