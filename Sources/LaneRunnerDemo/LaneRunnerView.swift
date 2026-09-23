import LaneRunner
import SwiftUI

struct LaneRunnerView: View {
    @EnvironmentObject private var model: LaneRunnerModel

    var body: some View {
        HStack(alignment: .top, spacing: 28) {
            VStack(alignment: .leading, spacing: 12) {
                Text("RUNNER / LOCAL").font(.caption.monospaced()).foregroundStyle(.orange)
                Text("Three lanes. Five moves.").font(.largeTitle.bold())
                board
                Text("Jump low bars, slide under high bars, change lanes around trains.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 16) {
                Text("Who’s running?").font(.title2.bold())
                Picker("Controller", selection: $model.control) {
                    ForEach(LaneRunnerModel.Control.allCases) { Text($0.title).tag($0) }
                }
                .labelsHidden()
                .disabled(model.isRunning || model.isLoading)
                .onChange(of: model.control) { _, _ in model.reset() }
                if model.usesModel {
                    Button(model.isLoading ? "Loading…" : (model.canPlay ? "Model ready" : "Load model")) {
                        model.loadModel()
                    }
                    .disabled(model.isLoading || model.canPlay)
                }
                HStack {
                    Button(model.isRunning ? "Pause" : (model.game.isOver ? "Try again" : "Run")) { model.toggle() }
                        .buttonStyle(.borderedProminent).tint(.orange).disabled(!model.canPlay)
                    Button("Reset") { model.reset() }
                }
                if model.control == .manual {
                    HStack {
                        Button("←") { model.queue(.left) }.keyboardShortcut(.leftArrow, modifiers: [])
                        Button("↑ jump") { model.queue(.jump) }.keyboardShortcut(.upArrow, modifiers: [])
                        Button("↓ slide") { model.queue(.slide) }.keyboardShortcut(.downArrow, modifiers: [])
                        Button("→") { model.queue(.right) }.keyboardShortcut(.rightArrow, modifiers: [])
                    }
                    .disabled(!model.isRunning)
                }
                HStack {
                    Text("Track seed")
                    TextField("Seed", value: $model.seed, format: .number)
                        .frame(width: 90).disabled(model.isRunning)
                        .onChange(of: model.seed) { _, _ in model.reset() }
                }
                Text(String(format: "Row every %.0f ms", model.rowMs)).font(.subheadline)
                Slider(value: $model.rowMs, in: 150...800, step: 50)
                Divider()
                HStack {
                    metric("DISTANCE", "\(model.game.distance)")
                    metric("COINS", "\(model.game.coins)")
                }
                HStack {
                    metric("MODEL CALL", model.calls == 0 ? "—" : String(format: "%.1f ms", model.modelMs))
                    metric("CONFIDENCE", model.calls == 0 ? "—" : String(format: "%.0f%%", model.confidence * 100))
                }
                Text("\(model.calls) replies · \(model.lateRows) late rows · \(model.unsafeChoices) unsafe picks")
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
                Text(model.lastAction.uppercased()).font(.title3.bold().monospaced())
                Divider()
                Text("WHAT THE MODEL READS").font(.caption.monospaced()).foregroundStyle(.secondary)
                Text(model.game.observation).font(.caption.monospaced()).textSelection(.enabled)
                ForEach(model.game.legalActions, id: \.self) { action in
                    Text(model.game.label(for: action)).font(.caption.monospaced())
                        .foregroundStyle(model.game.isSafe(action) ? Color.primary : Color.red)
                }
                Text("Option order rotates each row. A reply that misses its row counts as late, and the runner stays.")
                    .font(.caption).foregroundStyle(.secondary)
                if let crash = model.game.crash {
                    Text("Crashed: \(crash)").font(.caption.bold()).foregroundStyle(.red)
                }
                if let error = model.errorMessage {
                    Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
                }
            }
            .frame(width: 330)
            .padding(.top, 6)
        }
        .padding(24)
        .onDisappear { model.pause() }
    }

    private func metric(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.system(size: 10, weight: .semibold, design: .monospaced)).foregroundStyle(.secondary)
            Text(value).font(.system(size: 24, weight: .semibold, design: .rounded)).monospacedDigit()
        }.frame(maxWidth: .infinity, alignment: .leading)
    }

    private var board: some View {
        Canvas { context, size in
            let laneWidth = size.width / CGFloat(LaneRunner.lanes)
            let rowHeight = size.height / 7
            context.fill(
                Path(CGRect(origin: .zero, size: size)), with: .color(Color(red: 0.10, green: 0.09, blue: 0.14)))
            for lane in 1..<LaneRunner.lanes {
                let x = CGFloat(lane) * laneWidth
                context.stroke(
                    Path {
                        $0.move(to: CGPoint(x: x, y: 0))
                        $0.addLine(to: CGPoint(x: x, y: size.height))
                    },
                    with: .color(.white.opacity(0.15)), style: StrokeStyle(lineWidth: 2, dash: [10, 10]))
            }
            for (index, row) in model.game.rows.prefix(6).enumerated() {
                let y = size.height - rowHeight * CGFloat(index + 2)
                let visible = index < LaneRunner.visibleRows
                for (lane, obstacle) in row.enumerated() {
                    let cell = CGRect(x: CGFloat(lane) * laneWidth, y: y, width: laneWidth, height: rowHeight)
                    draw(obstacle, in: cell.insetBy(dx: 14, dy: 10), context: context, dim: !visible)
                }
            }
            let player = CGRect(
                x: CGFloat(model.game.lane) * laneWidth + laneWidth / 2 - 18, y: size.height - rowHeight + 14,
                width: 36, height: rowHeight - 28)
            context.fill(
                Path(roundedRect: player, cornerRadius: 10), with: .color(model.game.isOver ? .red : .mint))
        }
        .frame(width: 390, height: 600)
        .clipShape(RoundedRectangle(cornerRadius: 18))
        .overlay(alignment: .center) {
            if !model.isRunning {
                VStack(spacing: 8) {
                    Text(model.game.isOver ? "Run over" : (model.game.distance == 0 ? "Ready" : "Paused"))
                        .font(.title2.bold())
                    Text(
                        model.game.isOver
                            ? "\(model.game.distance) rows · \(model.game.coins) coins"
                            : "Choose a runner, then press Run"
                    )
                    .font(.caption)
                }
                .padding(20).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
            }
        }
        .accessibilityLabel("Lane runner, \(model.game.distance) rows")
    }

    private func draw(_ obstacle: LaneRunner.Obstacle, in rect: CGRect, context: GraphicsContext, dim: Bool) {
        let opacity = dim ? 0.35 : 1
        switch obstacle {
        case .open:
            break
        case .coin:
            let side = min(rect.width, rect.height) * 0.5
            context.fill(
                Path(ellipseIn: CGRect(x: rect.midX - side / 2, y: rect.midY - side / 2, width: side, height: side)),
                with: .color(.yellow.opacity(opacity)))
        case .train:
            context.fill(Path(roundedRect: rect, cornerRadius: 8), with: .color(.red.opacity(opacity)))
        case .low:
            context.fill(
                Path(CGRect(x: rect.minX, y: rect.maxY - 14, width: rect.width, height: 14)),
                with: .color(.orange.opacity(opacity)))
        case .high:
            context.fill(
                Path(CGRect(x: rect.minX, y: rect.minY, width: rect.width, height: 14)),
                with: .color(.purple.opacity(opacity)))
        }
    }
}
