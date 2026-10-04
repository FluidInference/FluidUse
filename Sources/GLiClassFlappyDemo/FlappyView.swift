import FlappyBird
import SwiftUI

struct FlappyView: View {
    @EnvironmentObject private var model: FlappyModel

    var body: some View {
        HStack(alignment: .top, spacing: 28) {
            VStack(alignment: .leading, spacing: 12) {
                Text("FLAPPY / LOCAL").font(.caption.monospaced()).foregroundStyle(.teal)
                Text("One bird. Two choices.").font(.largeTitle.bold())
                board
                Text("FLAP gives one upward impulse. COAST lets gravity act.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 18) {
                Text("Who’s flying?").font(.title2.bold())
                Picker("Controller", selection: $model.control) {
                    ForEach(FlappyModel.Control.allCases) { Text($0.rawValue).tag($0) }
                }
                .labelsHidden()
                .disabled(model.isRunning || model.isLoading)
                .onChange(of: model.control) { _, _ in model.reset() }
                if model.usesModel {
                    Toggle("Safety guard", isOn: $model.safetyGuard)
                        .disabled(model.isRunning)
                        .onChange(of: model.safetyGuard) { _, _ in model.reset() }
                        .font(.caption)
                }
                if model.usesModel {
                    Button(model.isLoading ? "Loading…" : (model.canPlay ? "Model ready" : "Load model")) {
                        model.loadModel()
                    }
                    .disabled(model.isLoading || model.canPlay)
                }
                HStack {
                    Button(model.isRunning ? "Pause" : (model.game.isOver ? "Try again" : "Play")) { model.toggle() }
                        .buttonStyle(.borderedProminent).tint(.teal).disabled(!model.canPlay)
                    Button("Reset") { model.reset() }
                }
                if model.control == .manual {
                    Button("Flap · Space") { model.flap() }
                        .keyboardShortcut(.space, modifiers: []).disabled(!model.isRunning)
                }
                HStack {
                    Text("Course seed")
                    TextField("Seed", value: $model.seed, format: .number)
                        .frame(width: 90).disabled(model.isRunning)
                        .onChange(of: model.seed) { _, _ in model.reset() }
                }
                Divider()
                HStack {
                    metric("PIPES", "\(model.game.score)")
                    metric("SURVIVAL", String(format: "%.1f s", model.game.seconds))
                }
                HStack {
                    metric("MODEL CALL", model.calls == 0 ? "—" : String(format: "%.1f ms", model.modelMs))
                    metric("RESPONSE AGE", model.calls == 0 ? "—" : String(format: "%.1f ms", model.responseMs))
                }
                Text(
                    "\(model.calls) replies · \(model.overrides) guard overrides · \(model.lateReplies) over 100 ms · \(model.tokens) tokens"
                )
                .font(.caption.monospaced()).foregroundStyle(.secondary)
                HStack {
                    Text(model.lastAction.uppercased()).font(.title3.bold().monospaced())
                    Spacer()
                    Text(model.calls == 0 ? "" : String(format: "P(flap) %.0f%%", model.probability * 100))
                        .font(.caption.monospaced())
                }
                Divider()
                Text(String(format: "Added response delay: %.0f ms", model.addedDelayMs)).font(.subheadline)
                Slider(value: $model.addedDelayMs, in: 0...500, step: 25)
                    .disabled(!model.usesModel)
                Text("Physics runs at 60 Hz while the model thinks. Requests are at most 10 Hz, one at a time.")
                    .font(.caption).foregroundStyle(.secondary)
                Text(
                    "All models see the state and 300 ms action forecasts. Option order alternates. The safety guard changes forecasted unsafe choices."
                )
                .font(.caption).foregroundStyle(.secondary)
                Text("Use the same seed to compare courses. Confidence is a model score, not a safety guarantee.")
                    .font(.caption).foregroundStyle(.secondary)
                if let error = model.errorMessage {
                    Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
                }
            }
            .frame(width: 310)
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
            let sx = size.width / FlappyBird.width
            let sy = size.height / FlappyBird.height
            context.scaleBy(x: sx, y: sy)
            context.fill(
                Path(CGRect(x: 0, y: 0, width: 420, height: 600)),
                with: .color(Color(red: 0.06, green: 0.13, blue: 0.19)))
            for x in stride(from: 0.0, to: 420, by: 30) {
                for y in stride(from: 0.0, to: 600, by: 30) {
                    context.fill(
                        Path(ellipseIn: CGRect(x: x, y: y, width: 2, height: 2)), with: .color(.white.opacity(0.09)))
                }
            }
            for pipe in model.game.pipes {
                let top = pipe.center - FlappyBird.gap / 2
                let bottom = pipe.center + FlappyBird.gap / 2
                context.fill(
                    Path(CGRect(x: pipe.x, y: 0, width: FlappyBird.pipeWidth, height: top)), with: .color(.teal))
                context.fill(
                    Path(CGRect(x: pipe.x, y: bottom, width: FlappyBird.pipeWidth, height: 600 - bottom)),
                    with: .color(.teal))
            }
            let bird = CGRect(x: FlappyBird.birdX - 12, y: model.game.y - 12, width: 24, height: 24)
            context.fill(Path(ellipseIn: bird), with: .color(.yellow))
            context.fill(
                Path(ellipseIn: CGRect(x: 103, y: model.game.y - 6, width: 5, height: 5)), with: .color(.black))
            context.fill(Path(CGRect(x: 108, y: model.game.y + 1, width: 10, height: 4)), with: .color(.orange))
        }
        .frame(width: 420, height: 600)
        .clipShape(RoundedRectangle(cornerRadius: 18))
        .overlay(alignment: .center) {
            if !model.isRunning {
                VStack(spacing: 8) {
                    Text(model.game.isOver ? "Flight over" : (model.game.frames == 0 ? "Ready for takeoff" : "Paused"))
                        .font(.title2.bold())
                    Text(
                        model.game.isOver
                            ? "\(model.game.score) pipes · \(String(format: "%.1f", model.game.seconds)) seconds"
                            : "Choose a controller, then press Play"
                    )
                    .font(.caption)
                }
                .padding(20).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
            }
        }
        .onTapGesture { model.flap() }
        .accessibilityLabel("Flappy Bird game, \(model.game.score) pipes passed")
    }
}
