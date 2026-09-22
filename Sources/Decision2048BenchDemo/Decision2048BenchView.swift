import Game2048
import SwiftUI

struct Decision2048BenchView: View {
    @EnvironmentObject private var model: Decision2048BenchModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            HStack(alignment: .top, spacing: 18) {
                competitor(model.gliClass, accent: .green)
                competitor(model.laya, accent: .blue)
            }
            comparison
            controls
        }
        .padding(18)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .alert(
            "2048 Bench",
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
            Text("2048 Bench · decision models").font(.title.bold())
            Text(
                "Laya vs GLiClass on the same seed and top-two expectimax shortlist; each model makes the final choice."
            )
            .font(.callout).foregroundStyle(.secondary)
            Text(model.loadStatus).font(.caption.monospaced())
                .foregroundStyle(model.hasLoadedModels ? Color.green : Color.secondary)
        }
    }

    private func competitor(_ side: Decision2048Side, accent: Color) -> some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(side.name).font(.headline).foregroundStyle(accent)
                    Text(side.detail).font(.caption.monospaced()).foregroundStyle(.secondary)
                }
                Spacer()
                if side.isOver { Text("finished").font(.caption.bold()).foregroundStyle(.red) }
            }
            board(side.board)
            HStack(spacing: 0) {
                readout("\(side.score)", "score", .primary)
                Divider().frame(height: 30)
                readout("\(side.moves)", "moves", accent)
                Divider().frame(height: 30)
                readout("\(side.maximumTile)", "max", .orange)
                Divider().frame(height: 30)
                readout(String(format: "%.1f", side.modelMillisecondsPerMove), "ms/move", .secondary)
            }
            .padding(.vertical, 8)
            .background(Color.secondary.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            Text("\(side.modelCalls.formatted()) model calls")
                .font(.caption.monospaced()).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    private func board(_ values: [[Int]]) -> some View {
        VStack(spacing: 7) {
            ForEach(0..<Game2048.size, id: \.self) { row in
                HStack(spacing: 7) {
                    ForEach(0..<Game2048.size, id: \.self) { column in tile(values[row][column]) }
                }
            }
        }
        .padding(9)
        .background(Color(red: 0.42, green: 0.38, blue: 0.34))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .aspectRatio(1, contentMode: .fit)
    }

    private func tile(_ value: Int) -> some View {
        RoundedRectangle(cornerRadius: 7)
            .fill(tileColor(value))
            .overlay {
                if value > 0 {
                    Text("\(value)").font(.system(size: 22, weight: .bold, design: .rounded))
                        .foregroundStyle(value <= 4 ? Color.black.opacity(0.72) : Color.white)
                        .minimumScaleFactor(0.45).lineLimit(1)
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

    private func readout(_ value: String, _ label: String, _ color: Color) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.system(size: 16, weight: .semibold, design: .monospaced))
                .foregroundStyle(color).minimumScaleFactor(0.55).lineLimit(1)
            Text(label).font(.system(size: 9)).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    private var comparison: some View {
        HStack {
            Label(model.elapsedText, systemImage: "clock").font(.headline.monospaced())
            Spacer()
            Text("GLiClass model-time advantage: \(model.speedupText)")
                .font(.headline.monospaced()).foregroundStyle(.green)
        }
        .padding(12)
        .background(Color.secondary.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                Button(model.hasLoadedModels ? "Loaded" : "Load both models") { model.loadModels() }
                    .disabled(model.isLoading || model.hasLoadedModels)
                Button(model.isRunning ? "Pause" : "Run comparison") { model.toggle() }
                    .disabled(!model.hasLoadedModels)
                    .keyboardShortcut(.space, modifiers: [])
                Button("Reset") { model.reset() }.disabled(model.isRunning)
                Spacer()
                Text("Seed")
                TextField("seed", value: $model.seed, format: .number).frame(width: 80).disabled(model.isRunning)
            }
            HStack {
                Text("Visual delay: \(Int(model.moveDelayMs)) ms/round").font(.caption)
                Slider(value: $model.moveDelayMs, in: 0...100, step: 5).disabled(model.isRunning)
            }
            Text("Inference alternates between models to avoid ANE contention. Delay is excluded from model latency.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}
