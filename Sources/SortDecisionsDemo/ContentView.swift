import SortAnything
import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var model: DecisionsModel

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            switch model.phase {
            case .loading(let message):
                status(message, spinning: true)
            case .failed(let message):
                status("Failed: \(message)", spinning: false)
            default:
                HStack(alignment: .top, spacing: 16) {
                    DocumentPanel(decided: model.current).frame(width: 560)
                    domains
                }
                .padding(16)
            }
            Divider()
            footer
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 22) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Sort decisions").font(.system(size: 26, weight: .bold))
                Text("GLiNER2.5-Decide · Core ML · on this Mac · every question for a document in one call")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Spacer()
            stat("Documents", "\(model.documentsDone) / \(model.total)")
            stat("Decisions", "\(model.decisions)")
            stat("Decisions / s", model.decisions > 0 ? String(format: "%.0f", model.decisionsPerSecond) : "–")
            stat("Elapsed", String(format: "%.1f s", model.elapsed))
            stat("Match Fastino's labels", model.accuracy.map { String(format: "%.1f%%", $0 * 100) } ?? "–")
            controls
        }
        .padding(.horizontal, 20).padding(.vertical, 14)
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(alignment: .trailing, spacing: 2) {
            Text(title.uppercased()).font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
            Text(value).font(.system(size: 22, weight: .semibold, design: .rounded)).monospacedDigit()
        }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Button(model.phase == .running ? "Pause" : "Start") { model.toggleRun() }
                    .keyboardShortcut(.space, modifiers: [])
                    .disabled(!(model.phase == .ready || model.phase == .running || model.phase == .paused))
                    .buttonStyle(.borderedProminent)
                Button("Reset") { model.reset() }.disabled(model.phase == .running)
            }
            Picker("Mode", selection: $model.mode) {
                ForEach(DecisionsModel.Mode.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented).labelsHidden().frame(width: 150)
            Text(
                model.mode == .turbo
                    ? "\(DecisionsModel.turboInFlight) calls in flight"
                    : String(format: "%.1f s per document", model.dwell)
            )
            .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var domains: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 196), spacing: 10)], spacing: 10) {
                ForEach(FastDecisions.domains, id: \.self) { domain in
                    DomainTile(
                        domain: domain, score: model.scores[domain] ?? .init(),
                        active: model.current?.document.domain == domain)
                }
            }
        }
    }

    private func status(_ message: String, spinning: Bool) -> some View {
        VStack(spacing: 12) {
            if spinning { ProgressView() }
            Text(message).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var footer: some View {
        HStack {
            Text(FastDecisions.attribution + " · scored as on the dataset card: exact match per decision")
            Spacer()
            Text("Model: fastino/GLiNER2.5-Decide (Apache-2.0) → FluidInference/gliner2-5-decide-coreml")
        }
        .font(.caption).foregroundStyle(.secondary)
        .padding(.horizontal, 20).padding(.vertical, 8)
    }
}

func humanize(_ name: String) -> String {
    name.replacingOccurrences(of: "_", with: " ")
}

private struct DocumentPanel: View {
    let decided: DecisionsModel.Decided?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let decided {
                HStack {
                    Text(humanize(decided.document.domain).uppercased())
                        .font(.caption.weight(.bold)).foregroundStyle(.secondary)
                    Spacer()
                    Text(
                        "\(decided.result.answers.count) questions · 1 call · "
                            + String(format: "%.0f ms", decided.result.milliseconds)
                    )
                    .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                }
                ScrollView {
                    Text(decided.document.input)
                        .font(.system(size: 13)).frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(height: 300)
                .padding(12)
                .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .textBackgroundColor)))
                VStack(spacing: 8) {
                    ForEach(decided.result.answers, id: \.task) { answer in
                        AnswerRow(answer: answer)
                    }
                }
                .id(decided.id)
                .transition(.opacity)
            } else {
                Text("Press Start").foregroundStyle(.secondary).frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            Spacer(minLength: 0)
        }
        .padding(16)
        .background(RoundedRectangle(cornerRadius: 14).fill(Color.secondary.opacity(0.07)))
    }
}

private struct AnswerRow: View {
    let answer: DecisionSorter.Answer

    var body: some View {
        HStack(spacing: 10) {
            Text(humanize(answer.task)).font(.callout.weight(.semibold)).frame(width: 140, alignment: .leading)
            Text(humanize(answer.label))
                .font(.callout.weight(.bold))
                .padding(.horizontal, 10).padding(.vertical, 4)
                .background(Capsule().fill((answer.correct ? Color.green : Color.red).opacity(0.22)))
            Text(String(format: "%.0f%%", answer.confidence * 100)).font(.caption).foregroundStyle(.secondary)
                .monospacedDigit()
            Spacer()
            Image(systemName: answer.correct ? "checkmark.circle.fill" : "xmark.circle.fill")
                .foregroundStyle(answer.correct ? Color.green : Color.red)
            if !answer.correct {
                Text("label: " + answer.gold.map(humanize).joined(separator: ", "))
                    .font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor)))
    }
}

private struct DomainTile: View {
    let domain: String
    let score: DecisionsModel.DomainScore
    let active: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(humanize(domain)).font(.headline).lineLimit(1)
                Spacer()
                Text(score.accuracy.map { String(format: "%.0f%%", $0 * 100) } ?? "–")
                    .font(.title3.weight(.bold)).monospacedDigit()
            }
            ProgressView(value: Double(score.documents), total: 100)
            Text("\(score.documents)/100 documents · \(score.decisions) decisions")
                .font(.caption).foregroundStyle(.secondary).monospacedDigit()
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 12).fill(Color.accentColor.opacity(active ? 0.22 : 0.07)))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.accentColor.opacity(active ? 0.8 : 0.15)))
    }
}
