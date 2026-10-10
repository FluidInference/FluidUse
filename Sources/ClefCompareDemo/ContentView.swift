import SwiftUI

struct ContentView: View {
    @EnvironmentObject var model: CompareModel

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if model.ready {
                HStack(alignment: .top, spacing: 16) {
                    IncomingLane().frame(width: 300)
                    ComparedFeed().frame(maxWidth: .infinity)
                }
                .padding(18)
                Divider()
                composer
            } else {
                Spacer()
                ProgressView(model.status).controlSize(.large)
                Spacer()
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 28) {
            VStack(alignment: .leading, spacing: 4) {
                Text("clef-flash 9B  vs  clef-text 0.6B").font(.system(size: 26, weight: .bold))
                Text("Same tickets, same questions · 9B on the GPU · 0.6B distilled student on the Neural Engine")
                    .font(.system(size: 14)).foregroundStyle(.secondary)
            }
            Spacer()
            if let agreement = model.agreement {
                Stat(value: String(format: "%.0f%%", agreement * 100), label: "decisions match")
            }
            if let big = model.bigMedian, let small = model.smallMedian, small > 0 {
                Stat(value: String(format: "%.0f vs %.0f ms", big, small), label: "per ticket · 9B vs 0.6B")
            }
        }
        .padding(.horizontal, 28).padding(.vertical, 16)
    }

    private var composer: some View {
        HStack(spacing: 12) {
            TextField("Type your own ticket and press Return…", text: $model.draft)
                .textFieldStyle(.roundedBorder).font(.system(size: 15))
                .onSubmit { model.submitDraft() }
            Button(model.running ? "Pause stream" : "Resume stream") { model.toggleStream() }
            Button("Restart") { model.restart() }
        }
        .padding(.horizontal, 28).padding(.vertical, 14)
    }
}

struct Stat: View {
    let value: String
    let label: String

    var body: some View {
        VStack(alignment: .trailing, spacing: 0) {
            Text(value).font(.system(size: 34, weight: .heavy, design: .rounded)).monospacedDigit()
            Text(label).font(.system(size: 12)).foregroundStyle(.secondary)
        }
    }
}

struct IncomingLane: View {
    @EnvironmentObject var model: CompareModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "tray.and.arrow.down.fill").foregroundStyle(.blue)
                Text("Incoming").font(.system(size: 17, weight: .semibold))
                Spacer()
                Text("\(model.tickets.count) triaged").font(.system(size: 13)).foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            if let current = model.current {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("both models reading…").font(.system(size: 12, weight: .semibold)).foregroundStyle(.blue)
                    }
                    Text(current.text).font(.system(size: 14)).fixedSize(horizontal: false, vertical: true)
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 10).fill(Color.blue.opacity(0.12)))
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.blue, lineWidth: 2))
                .id(current.id)
            }
            ForEach(model.incoming) { ticket in
                Text(ticket.text).font(.system(size: 13)).foregroundStyle(.secondary).lineLimit(2)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor)))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10).stroke(
                            ticket.typed ? Color.accentColor : .clear, lineWidth: 2)
                    )
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
            Spacer(minLength: 0)
        }
        .padding(14)
        .frame(maxHeight: .infinity, alignment: .top)
        .background(RoundedRectangle(cornerRadius: 14).fill(Color.blue.opacity(0.05)))
        .clipped()
    }
}

struct ComparedFeed: View {
    @EnvironmentObject var model: CompareModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                Text("Decided").font(.system(size: 17, weight: .semibold))
                Spacer()
                Label("9B · GPU", systemImage: "cpu").font(.system(size: 12)).foregroundStyle(.secondary)
                Label("0.6B · Neural Engine", systemImage: "brain").font(.system(size: 12)).foregroundStyle(.secondary)
            }
            ScrollView {
                LazyVStack(spacing: 8) {
                    ForEach(model.tickets.reversed()) { ticket in
                        CompareRow(ticket: ticket).transition(.move(edge: .top).combined(with: .opacity))
                    }
                }
            }
        }
        .padding(14)
        .frame(maxHeight: .infinity, alignment: .top)
        .background(RoundedRectangle(cornerRadius: 14).fill(Color.green.opacity(0.04)))
    }
}

struct CompareRow: View {
    let ticket: CompareModel.Ticket

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(ticket.text).font(.system(size: 14)).lineLimit(2)
                Spacer()
                if ticket.agreements == 3 {
                    Image(systemName: "equal.circle.fill").foregroundStyle(.green)
                } else {
                    Text("\(ticket.agreements)/3 match").font(.system(size: 12, weight: .semibold)).foregroundStyle(
                        .orange)
                }
            }
            DecisionLine(label: "9B", detail: "GPU", decision: ticket.big, ticket: ticket)
            DecisionLine(label: "0.6B", detail: "ANE", decision: ticket.small, ticket: ticket)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(
            RoundedRectangle(cornerRadius: 10).stroke(
                ticket.typed ? Color.accentColor : (ticket.agreements < 3 ? Color.orange.opacity(0.6) : .clear),
                lineWidth: 2))
    }
}

struct DecisionLine: View {
    let label: String
    let detail: String
    let decision: CompareModel.Decision
    let ticket: CompareModel.Ticket

    private static let teamColors: [String: Color] = [
        "billing": .orange, "engineering": .red, "account": .purple, "product": .teal,
    ]
    private var urgencyColor: Color { [Color.green, .blue, .orange, .red][decision.urgency] }

    var body: some View {
        HStack(spacing: 10) {
            Text(label).font(.system(size: 12, weight: .bold)).frame(width: 36, alignment: .leading)
            Text(detail).font(.system(size: 11)).foregroundStyle(.secondary).frame(width: 30, alignment: .leading)
            Text(decision.team.capitalized)
                .font(.system(size: 12, weight: .semibold)).lineLimit(1)
                .frame(width: 100).padding(.vertical, 4)
                .background(
                    RoundedRectangle(cornerRadius: 6).fill((Self.teamColors[decision.team] ?? .gray).opacity(0.2))
                )
                .foregroundStyle(Self.teamColors[decision.team] ?? .gray)
                .overlay(mismatch(ticket.teamAgrees))
            Text(CompareModel.urgencyLabels[decision.urgency])
                .font(.system(size: 11, weight: .semibold)).frame(width: 64).padding(.vertical, 3)
                .background(Capsule().fill(urgencyColor.opacity(0.18))).foregroundStyle(urgencyColor)
                .overlay(mismatch(ticket.urgencyAgrees, capsule: true))
            Text(decision.refund ? "Refund" : "No refund")
                .font(.system(size: 11, weight: .semibold)).frame(width: 72).padding(.vertical, 3)
                .background(Capsule().fill((decision.refund ? Color.pink : .gray).opacity(0.15)))
                .foregroundStyle(decision.refund ? Color.pink : .secondary)
                .overlay(mismatch(ticket.refundAgrees, capsule: true))
            Spacer()
            Text(String(format: "%.0f%%", decision.teamConfidence * 100))
                .font(.system(size: 12)).foregroundStyle(.secondary).monospacedDigit()
            Text(String(format: "%.0f ms", decision.milliseconds))
                .font(.system(size: 13, weight: .semibold)).monospacedDigit().frame(width: 64, alignment: .trailing)
        }
    }

    @ViewBuilder private func mismatch(_ agrees: Bool, capsule: Bool = false) -> some View {
        if !agrees {
            if capsule {
                Capsule().stroke(Color.orange, lineWidth: 2)
            } else {
                RoundedRectangle(cornerRadius: 6).stroke(Color.orange, lineWidth: 2)
            }
        }
    }
}
