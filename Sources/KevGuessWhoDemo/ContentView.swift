import SortAnything
import SwiftUI

struct ContentView: View {
    @EnvironmentObject var model: GuessWhoModel

    private let columns = [GridItem(.adaptive(minimum: 116), spacing: 8)]

    var body: some View {
        VStack(spacing: 0) {
            header
            HStack(alignment: .top, spacing: 16) {
                ScrollView {
                    LazyVGrid(columns: columns, spacing: 8) {
                        ForEach(model.cards) { card in
                            CardView(card: card, isSecret: model.secret == card.id, solved: isSolved)
                        }
                    }
                    .padding(.bottom, 8)
                }
                .frame(maxWidth: .infinity)
                sidebar.frame(width: 300)
            }
            .padding(16)
            Text(DBpediaSample.attribution + " · questions answered by Kev-0.8B (jaredpalmer/kev, Apache-2.0)")
                .font(.caption2).foregroundStyle(.secondary).padding(.bottom, 8)
        }
        .background(Color(red: 0.07, green: 0.08, blue: 0.10))
        .foregroundStyle(.white)
    }

    private var isSolved: Bool {
        if case .solved = model.phase { return true }
        return false
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 28) {
            VStack(alignment: .leading, spacing: 2) {
                Text("GUESS WHO").font(.system(size: 13, weight: .heavy)).tracking(3).foregroundStyle(.orange)
                Text(status).font(.system(size: 28, weight: .bold)).lineLimit(1).minimumScaleFactor(0.5)
                    .contentTransition(.opacity)
            }
            .layoutPriority(1)
            Spacer(minLength: 12)
            controls
            stat(String(format: "%.1f ms", model.lastCallMs), "last call")
            stat(String(format: "%.1f ms", model.medianCallMs), "median call")
            stat(String(format: "%.0f/s", model.scanRate), "decisions / s")
            stat("\(model.totalDecisions)", "decisions")
        }
        .padding(.horizontal, 20).padding(.vertical, 14)
        .background(Color.white.opacity(0.04))
    }

    private var status: String {
        switch model.phase {
        case .loading(let text): text
        case .dealing: "Dealing \(model.cards.count) new cards…"
        case .scanning:
            "Reading \(model.scanned)/\(model.cards.count) bios · \(GuessWhoModel.questions.count) questions each"
        case .asking(let question): question
        case .solved(let name): "It's \(name)!"
        case .failed(let error): "Error: \(error)"
        }
    }

    private func stat(_ value: String, _ label: String) -> some View {
        VStack(alignment: .trailing, spacing: 2) {
            Text(value).font(.system(size: 22, weight: .semibold, design: .monospaced)).lineLimit(1)
                .minimumScaleFactor(0.6).contentTransition(.numericText())
            Text(label).font(.caption).foregroundStyle(.secondary).lineLimit(1)
        }
    }

    private var controls: some View {
        HStack(spacing: 8) {
            Button {
                model.togglePause()
            } label: {
                Label(model.paused ? "Play" : "Pause", systemImage: model.paused ? "play.fill" : "pause.fill")
                    .frame(width: 78)
            }
            .keyboardShortcut(.space, modifiers: [])
            .help("Play / pause (Space)")
            Button {
                model.reset()
            } label: {
                Label("Reset", systemImage: "arrow.counterclockwise").frame(width: 78)
            }
            .keyboardShortcut("r", modifiers: [.command])
            .help("New wall (⌘R)")
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .tint(Color.orange.opacity(0.85))
        .disabled(!model.ready)
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Game \(model.game) · \(model.remaining) of \(model.cards.count) left")
                .font(.headline)
            if model.scanSeconds > 0 {
                Text(
                    String(
                        format: "Scan: %d bios × %d questions = %d decisions in %.2f s", model.scanned,
                        GuessWhoModel.questions.count, model.scanned * GuessWhoModel.questions.count, model.scanSeconds)
                )
                .font(.callout).foregroundStyle(.secondary)
            }
            Divider().overlay(Color.white.opacity(0.2))
            ForEach(Array(model.asked.enumerated()), id: \.offset) { index, turn in
                HStack(alignment: .top, spacing: 8) {
                    Text("\(index + 1).").monospacedDigit().foregroundStyle(.secondary)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(turn.question).font(.callout)
                        Text("\(turn.answer ? "YES" : "NO") · \(turn.removed) flipped down")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(turn.answer ? .green : .red)
                    }
                }
                .transition(.move(edge: .leading).combined(with: .opacity))
            }
            Spacer()
            Text(
                "One fused Core ML call per card answers every question on that bio; each turn asks the question that splits the cards still up closest to half."
            )
            .font(.caption).foregroundStyle(.secondary)
        }
        .animation(.easeOut(duration: 0.3), value: model.asked.count)
    }
}

struct CardView: View {
    let card: GuessWhoModel.Card
    let isSecret: Bool
    let solved: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 4) {
                Text(icon).font(.system(size: 13))
                Text(card.item.title).font(.system(size: 12, weight: .semibold)).lineLimit(2)
            }
            Spacer(minLength: 0)
            HStack(spacing: 2) {
                ForEach(0..<GuessWhoModel.questions.count, id: \.self) { q in
                    Circle()
                        .fill(dotColor(q))
                        .frame(width: 5, height: 5)
                }
            }
        }
        .padding(7)
        .frame(maxWidth: .infinity, minHeight: 62, alignment: .topLeading)
        .background(RoundedRectangle(cornerRadius: 8).fill(background))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(
                    solved && isSecret ? Color.orange : Color.white.opacity(card.answers == nil ? 0.08 : 0.2),
                    lineWidth: solved && isSecret ? 3 : 1)
        )
        .opacity(card.down ? 0.18 : 1)
        .rotation3DEffect(.degrees(card.flipping ? 90 : 0), axis: (x: 1, y: 0, z: 0))
        .scaleEffect(solved && isSecret ? 1.12 : 1)
        .animation(.spring(duration: 0.4), value: solved)
        .animation(.easeOut(duration: 0.15), value: card.answers == nil)
    }

    private var icon: String {
        switch card.item.gold {
        case "athlete": "🏅"
        case "politician": "🏛️"
        default: "🎨"
        }
    }

    private var background: Color {
        if solved && isSecret { return Color.orange.opacity(0.35) }
        return card.answers == nil ? Color.white.opacity(0.05) : Color(red: 0.16, green: 0.20, blue: 0.28)
    }

    private func dotColor(_ q: Int) -> Color {
        guard let answers = card.answers else { return Color.white.opacity(0.1) }
        return answers[q] ? Color.green : Color.white.opacity(0.25)
    }
}
