import LayaTetris
import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var model: GameModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            BoardView(board: model.board, chosen: model.chosen, evaluating: model.evaluating)
                .frame(width: 260, height: 520)
            controls
        }
        // Anchor at the top so an undersized window clips the controls, never the header.
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .padding(16)
        .alert(
            "laya",
            isPresented: Binding(get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })
        ) {
            Button("OK") { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("laya plays Tetris").font(.title2.bold())
            Text(
                "Every legal landing is described in one sentence; laya answers “Is this a clean placement?” and the highest P(true) is played."
            )
            .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Text(model.loadStatus).font(.caption.monospaced()).foregroundStyle(
                model.manager == nil ? Color.secondary : Color.green)
        }
        .frame(width: 260, alignment: .leading)
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Button(model.manager == nil ? "Load model" : "Loaded") { model.loadModel() }
                    .disabled(model.isLoading || model.manager != nil)
                Button(model.isRunning ? "Pause" : (model.isOver ? "Restart" : "Play")) { model.toggle() }
                    .keyboardShortcut(.space, modifiers: [])
                    .disabled(model.policy == .laya && model.manager == nil)
                Button("Reset") { model.reset() }.disabled(model.isRunning)
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
                Text(String(format: "Pause per piece: %.0f ms", model.pieceDelayMs)).font(.caption)
                Slider(value: $model.pieceDelayMs, in: 0...1000, step: 20)
            }
        }
        .frame(width: 260)
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
