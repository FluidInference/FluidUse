import SwiftUI

struct ContentView: View {
    @EnvironmentObject var model: TriageModel

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if model.ready {
                board
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
        HStack(alignment: .firstTextBaseline, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Cloudflare clef-flash · 9B decision model")
                    .font(.system(size: 26, weight: .bold))
                Text("Running on this Mac · Core ML on the GPU · no network")
                    .font(.system(size: 15)).foregroundStyle(.secondary)
            }
            Spacer()
            if let median = model.medianMilliseconds {
                VStack(alignment: .trailing, spacing: 0) {
                    Text(String(format: "%.2f s", median / 1000))
                        .font(.system(size: 44, weight: .heavy, design: .rounded)).monospacedDigit()
                    Text("per ticket · 3 decisions in one pass")
                        .font(.system(size: 13)).foregroundStyle(.secondary)
                }
            }
        }
        .padding(.horizontal, 28).padding(.vertical, 18)
    }

    private var board: some View {
        HStack(alignment: .top, spacing: 16) {
            IncomingLane()
                .frame(width: 340)
            DecidedFeed()
                .frame(maxWidth: .infinity)
        }
        .padding(18)
    }

    private var composer: some View {
        HStack(spacing: 12) {
            TextField("Type your own ticket and press Return…", text: $model.draft)
                .textFieldStyle(.roundedBorder).font(.system(size: 15))
                .onSubmit { model.submitDraft() }
            Button(model.running ? "Pause stream" : "Resume stream") { model.toggleStream() }
            Button("Restart") { model.restart() }
            if model.busy { ProgressView().controlSize(.small) }
        }
        .padding(.horizontal, 28).padding(.vertical, 14)
    }
}

struct IncomingLane: View {
    @EnvironmentObject var model: TriageModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "tray.and.arrow.down.fill").foregroundStyle(.blue)
                Text("Incoming").font(.system(size: 17, weight: .semibold))
                Spacer()
                Text("\(model.triagedCount) triaged").font(.system(size: 13)).foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            if let current = model.current {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("clef-flash is reading…").font(.system(size: 12, weight: .semibold)).foregroundStyle(.blue)
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

struct DecidedFeed: View {
    @EnvironmentObject var model: TriageModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                Text("Decided").font(.system(size: 17, weight: .semibold))
                Spacer()
                ForEach(TriageModel.Team.allCases) { team in
                    HStack(spacing: 5) {
                        Circle().fill(team.color).frame(width: 8, height: 8)
                        Text(team.title).font(.system(size: 13))
                        Text("\(model.count(for: team))").font(.system(size: 13, weight: .semibold)).monospacedDigit()
                    }
                    .padding(.horizontal, 9).padding(.vertical, 4)
                    .background(Capsule().fill(team.color.opacity(0.12)))
                }
            }
            ScrollView {
                LazyVStack(spacing: 8) {
                    ForEach(model.tickets.reversed()) { ticket in
                        TicketRow(ticket: ticket)
                            .transition(.move(edge: .top).combined(with: .opacity))
                    }
                }
            }
        }
        .padding(14)
        .frame(maxHeight: .infinity, alignment: .top)
        .background(RoundedRectangle(cornerRadius: 14).fill(Color.green.opacity(0.04)))
    }
}

struct TicketRow: View {
    let ticket: TriageModel.Ticket

    private var urgencyColor: Color { [Color.green, .blue, .orange, .red][ticket.urgency] }

    var body: some View {
        HStack(spacing: 12) {
            Text(ticket.team.title)
                .font(.system(size: 13, weight: .semibold)).lineLimit(1)
                .frame(width: 104)
                .padding(.vertical, 6)
                .background(RoundedRectangle(cornerRadius: 7).fill(ticket.team.color.opacity(0.2)))
                .foregroundStyle(ticket.team.color)
            Text(ticket.text).font(.system(size: 14)).lineLimit(2)
                .frame(maxWidth: .infinity, alignment: .leading)
            Pill(text: TriageModel.urgencyLabels[ticket.urgency], color: urgencyColor)
                .frame(width: 70, alignment: .leading)
            Pill(text: "Refund", color: .pink).opacity(ticket.refund > 0.5 ? 1 : 0)
            Text(String(format: "%.0f%%", ticket.teamConfidence * 100))
                .font(.system(size: 12)).foregroundStyle(.secondary).monospacedDigit().frame(
                    width: 36, alignment: .trailing)
            Text(String(format: "%.0f ms", ticket.milliseconds))
                .font(.system(size: 13, weight: .semibold)).monospacedDigit().frame(width: 60, alignment: .trailing)
        }
        .padding(.horizontal, 12).padding(.vertical, 9)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(ticket.typed ? Color.accentColor : .clear, lineWidth: 2))
    }
}

struct Pill: View {
    let text: String
    let color: Color

    var body: some View {
        Text(text).font(.system(size: 11, weight: .semibold)).lineLimit(1).fixedSize()
            .padding(.horizontal, 7).padding(.vertical, 3)
            .background(Capsule().fill(color.opacity(0.18)))
            .foregroundStyle(color)
    }
}
