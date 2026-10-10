import FluidUse
import SwiftUI

@available(macOS 15.0, *)
struct ContentView: View {
    @State private var player = MarioPlayer()
    let modelDirectory: URL
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
                .frame(width: 768)
                ProgressView(value: Double(min(player.x, MarioPlayer.flagX)), total: Double(MarioPlayer.flagX))
                Text(player.status).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 14) {
                Text("JEV-9B · Core ML").font(.title2.bold())
                Text("Every move: three yes/no questions about the screen just ahead of Mario")
                    .font(.callout).foregroundStyle(.secondary)
                ForEach(MarioPlayer.questions) { q in
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
                            ProgressView(value: Double(p)).tint(p > q.threshold ? .orange : .gray)
                            Text(String(format: "P(yes) %.2f", p)).font(.system(.caption, design: .monospaced))
                        }
                    }
                }
                HStack(alignment: .firstTextBaseline) {
                    Text(player.move.rawValue.uppercased()).font(.system(size: 34, weight: .bold, design: .rounded))
                        .foregroundStyle(player.move == .run ? .green : .orange)
                    Spacer()
                    VStack(alignment: .trailing) {
                        Text(String(format: "%.0f ms", player.lastMilliseconds)).font(
                            .system(.title3, design: .monospaced))
                        Text("decision \(player.decisions)").foregroundStyle(.secondary)
                    }
                }
                Button("Restart") { player.restart(modelDirectory: modelDirectory, bridgeCommand: bridgeCommand) }
                    .disabled(!player.finished)
            }
            .frame(width: 340)
        }
        .padding(24)
        .task { player.start(modelDirectory: modelDirectory, bridgeCommand: bridgeCommand) }
    }
}
