import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var model: ChallengeModel

    var body: some View {
        VStack(spacing: 0) {
            Header()
            Divider()
            HStack(alignment: .top, spacing: 24) {
                ChallengeCard()
                VStack(alignment: .leading, spacing: 14) {
                    Stats()
                    HistoryStrip()
                    Spacer()
                }
            }
            .padding(20)
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

struct Header: View {
    @EnvironmentObject private var model: ChallengeModel

    var body: some View {
        HStack(spacing: 10) {
            Text("Picture challenge").font(.title2.bold())
            Text(status).font(.callout).foregroundStyle(.secondary)
            Spacer()
            Button {
                if case .running = model.phase { model.stop() } else { model.start() }
            } label: {
                if case .running = model.phase {
                    Label("Stop", systemImage: "pause.fill")
                } else {
                    Label("Run", systemImage: "play.fill")
                }
            }
            .keyboardShortcut(.space, modifiers: [])
            .disabled(!canRun)
        }
        .padding(.horizontal, 20).padding(.vertical, 14)
    }

    private var canRun: Bool {
        switch model.phase {
        case .ready, .done, .running: true
        default: false
        }
    }

    private var status: String {
        switch model.phase {
        case .loading(let message): message
        case .ready: "Ready · Space runs it"
        case .running(let stage): stage
        case .done: model.summary
        case .failed(let message): message
        }
    }
}

struct ChallengeCard: View {
    @EnvironmentObject private var model: ChallengeModel

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Select all images with").font(.system(size: 15))
                Text(model.grid?.prompt ?? "…").font(.system(size: 26, weight: .bold))
                    .contentTransition(.opacity)
            }
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color.indigo)
            LazyVGrid(columns: Array(repeating: GridItem(.fixed(150), spacing: 4), count: 3), spacing: 4) {
                ForEach(model.grid?.tiles ?? []) { tile in
                    TileView(tile: tile, verified: model.grid?.verified ?? false)
                }
            }
            .frame(width: 458, height: 458)
            .padding(4)
            HStack {
                if let grid = model.grid, grid.verified {
                    Label(
                        grid.solved ? "Solved" : "Missed",
                        systemImage: grid.solved ? "checkmark.seal.fill" : "xmark.seal.fill"
                    )
                    .foregroundStyle(grid.solved ? .green : .red).font(.headline)
                }
                Spacer()
                Text(String(format: "%.0f ms", model.lastGridMilliseconds)).font(.callout.monospacedDigit())
                    .foregroundStyle(.secondary)
                Text("VERIFY").font(.callout.bold()).foregroundStyle(.white)
                    .padding(.horizontal, 18).padding(.vertical, 8)
                    .background(RoundedRectangle(cornerRadius: 4).fill(Color.indigo))
            }
            .padding(12)
        }
        .frame(width: 466)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.primary.opacity(0.15)))
    }
}

struct TileView: View {
    let tile: ChallengeModel.Tile
    let verified: Bool

    var body: some View {
        Image(decorative: tile.image, scale: 1).resizable().scaledToFill()
            .frame(width: 150, height: 150).clipped()
            .scaleEffect(tile.picked == true ? 0.86 : 1)
            .overlay(alignment: .topLeading) {
                if tile.picked == true {
                    Image(systemName: "checkmark.circle.fill").font(.system(size: 26))
                        .foregroundStyle(.white, Color.indigo).padding(6)
                }
            }
            .overlay(alignment: .bottomTrailing) {
                if tile.picked != nil {
                    Text(String(format: "%.2f", tile.score)).font(.caption2.monospacedDigit().bold())
                        .padding(.horizontal, 5).padding(.vertical, 2)
                        .background(Capsule().fill(.black.opacity(0.55))).foregroundStyle(.white).padding(5)
                }
            }
            .overlay {
                if verified {
                    Rectangle().stroke(tile.picked == tile.isTarget ? Color.clear : Color.red, lineWidth: 5)
                }
            }
            .background(Color.indigo.opacity(tile.picked == true ? 0.25 : 0))
    }
}

struct Stats: View {
    @EnvironmentObject private var model: ChallengeModel

    var body: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
            Tile(value: "\(model.gridsDone)", caption: "grids solved")
            Tile(value: String(format: "%.0f", model.tilesPerSecond), caption: "photos / second")
            Tile(
                value: model.gridsDone == 0 ? "–" : String(format: "%.0f ms", model.lastGridMilliseconds),
                caption: "per 9-photo grid")
            Tile(
                value: model.tilesDone == 0 ? "–" : String(format: "%.1f%%", 100 * model.accuracy),
                caption: "photos judged right")
        }
        .frame(width: 440)
    }
}

struct HistoryStrip: View {
    @EnvironmentObject private var model: ChallengeModel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("\(model.gridsSolved) of \(model.gridsDone) grids perfect").font(.caption).foregroundStyle(.secondary)
            LazyVGrid(columns: Array(repeating: GridItem(.fixed(9), spacing: 3), count: 36), spacing: 3) {
                ForEach(Array(model.history.suffix(216).enumerated()), id: \.offset) { _, solved in
                    RoundedRectangle(cornerRadius: 2).fill(solved ? Color.green : Color.red).frame(width: 9, height: 9)
                }
            }
            .frame(width: 440, alignment: .leading)
            Text(
                "EmbeddingGemma 2 · photos on the GPU, labels on the Neural Engine · zero-shot · nothing leaves this Mac"
            )
            .font(.caption2).foregroundStyle(.secondary)
        }
    }
}

struct Tile: View {
    let value: String
    let caption: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value).font(.system(size: 30, weight: .semibold, design: .rounded)).monospacedDigit()
                .contentTransition(.numericText())
            Text(caption).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.primary.opacity(0.05)))
    }
}
