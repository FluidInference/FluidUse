import Foundation
import FluidUse
import SwiftUI

/// Support-ticket triage with Cloudflare clef-flash (9B) running locally: each ticket gets three typed decisions
/// (team, urgency, refund) from one forward pass. Tickets are fictional mock data.
@MainActor
final class TriageModel: ObservableObject {
    enum Team: String, CaseIterable, Identifiable {
        case billing, engineering, account, product
        var id: String { rawValue }
        var title: String { rawValue.capitalized }
        var color: Color {
            switch self {
            case .billing: return .orange
            case .engineering: return .red
            case .account: return .purple
            case .product: return .teal
            }
        }
    }

    struct Ticket: Identifiable {
        let id = UUID()
        let text: String
        let team: Team
        let teamConfidence: Float
        let urgency: Int
        let refund: Float
        let milliseconds: Double
        let typed: Bool
    }

    struct Incoming: Identifiable, Equatable {
        let id = UUID()
        let text: String
        let typed: Bool
    }

    static let urgencyLabels = ["Low", "Normal", "High", "Critical"]

    static let questions: [(id: String, question: ClefQuestion)] = [
        (
            "team",
            .choice(
                instructions: "Which team should handle this ticket?",
                criteria: [
                    "billing": "Charges, refunds, invoices", "engineering": "Bugs, outages, API errors",
                    "account": "Login, security, account changes", "product": "Feature requests and questions",
                ])
        ),
        ("urgency", .score(instructions: "How urgent is this ticket?", criteria: urgencyLabels)),
        ("refund", .noul(instructions: "Is the customer asking for money back?")),
    ]

    // Fictional tickets (mock data, not real customers).
    static let mockTickets = [
        "Hi, I was charged twice for my Pro plan this month. Can you refund the duplicate?",
        "The dashboard has been throwing 500 errors since this morning and none of our team can log in.",
        "Would love a dark mode for the mobile app, the white screen is brutal at night.",
        "I forgot my password and the reset email never arrives. Checked spam already.",
        "Our checkout is down and customers can't pay. This is costing us thousands per hour!!",
        "Can I get an invoice with our VAT number on it for last quarter?",
        "Your latest update deleted all my saved drafts. Two weeks of work gone. I want my money back.",
        "Quick question: do you support exporting reports to CSV?",
        "Please cancel my subscription and close my account, I'm moving to another tool.",
        "API requests started timing out after we upgraded to v3 of the SDK.",
        "Someone logged into my account from another country. I didn't do that. Please lock it.",
        "Is there a discount for nonprofits? We're a small animal shelter.",
        "Webhooks stopped firing for new orders about an hour ago, nothing in the logs.",
        "I was promised a refund three weeks ago and still haven't seen it on my card.",
        "Can you add Spanish to the language options? Half our team would use it.",
        "How do I change the email address on my account? I left my old job.",
        "The iOS app crashes every time I open the calendar tab.",
        "We were billed for 50 seats but we only have 12 users. Please fix and credit us.",
        "Two-factor codes aren't arriving by SMS, I'm locked out before a client meeting.",
        "Would be great if the search bar supported filters by date.",
    ]

    @Published var status = "Loading…"
    @Published var ready = false
    @Published var tickets: [Ticket] = []
    @Published var running = false
    @Published var draft = ""
    @Published var busy = false
    /// Tickets waiting to be read (front = next), and the one the model is reading right now.
    @Published var incoming: [Incoming] = []
    @Published var current: Incoming?

    /// The loaded model as a closure (ClefFlashManager needs macOS 15; the package targets 14): answers + total ms.
    private var answer: ((String) async throws -> ([ClefAnswer], Double))?
    private var next = 0
    /// The fixed 1,000-ticket backlog the stream works through (fictional).
    static let backlog = MockTickets.make(count: 1000)
    /// Wall time spent triaging (pauses excluded), and whether the backlog is done.
    @Published var activeSeconds: Double = 0
    @Published var finished = false
    /// Bumped by `restart()`; a decision that finishes for an older generation is dropped.
    private var generation = 0
    private var streamTask: Task<Void, Never>?

    var medianMilliseconds: Double? {
        let times = tickets.map(\.milliseconds).sorted()
        return times.isEmpty ? nil : times[times.count / 2]
    }

    func count(for team: Team) -> Int { tickets.reduce(0) { $0 + ($1.team == team ? 1 : 0) } }

    /// Which model the board runs: `CLEF_MODEL=text` for clef-text-0.6b on the Neural Engine, else clef-flash 9B.
    enum Model {
        case flash, text
        var title: String {
            self == .text ? "clef-text · 0.6B decision model" : "Cloudflare clef-flash · 9B decision model"
        }
        var subtitle: String {
            self == .text
                ? "Distilled from Cloudflare clef-flash · Core ML on the GPU · no network"
                : "Running on this Mac · Core ML on the GPU · no network"
        }
        var short: String { self == .text ? "clef-text 0.6B" : "clef-flash" }
    }

    static let model: Model = ProcessInfo.processInfo.environment["CLEF_MODEL"] == "text" ? .text : .flash

    func start() async {
        guard answer == nil else { return }
        do {
            guard #available(macOS 15.0, *) else {
                status = "These Core ML models need macOS 15"
                return
            }
            let started = Date()
            switch Self.model {
            case .flash:
                let directory: URL
                if let path = ProcessInfo.processInfo.environment["CLEF_FLASH_BUNDLE"], !path.isEmpty {
                    directory = URL(fileURLWithPath: path)
                } else {
                    status = "Downloading clef-flash Core ML (FluidInference/clef-flash-coreml, ~11 GB)…"
                    directory = try await ClefFlashModelStore.ensure()
                }
                status = "Loading clef-flash 9B (first launch compiles the packages)…"
                let manager = try await ClefFlashManager.load(from: directory, bucket: 512)
                try await manager.warm()
                // a couple of full-length tickets page every part's weights in before the board starts timing
                for text in Self.mockTickets.prefix(2) {
                    _ = try await manager.answer(state: text, questions: Self.questions)
                }
                answer = { text in
                    let result = try await manager.answer(state: text, questions: Self.questions)
                    return (result.answers, result.totalMilliseconds)
                }
            case .text:
                let directory: URL
                if let path = ProcessInfo.processInfo.environment["CLEF_TEXT_BUNDLE"], !path.isEmpty {
                    directory = URL(fileURLWithPath: path)
                } else {
                    status = "Downloading clef-text-0.6b Core ML (FluidInference/clef-text-0.6b-coreml, ~1.3 GB)…"
                    directory = try await ClefTextModelStore.ensure()
                }
                status = "Loading clef-text 0.6B…"
                let manager = try await ClefTextManager.load(from: directory, buckets: [512], computeUnits: .cpuAndGPU)
                try await manager.warm()
                for text in Self.mockTickets.prefix(2) {
                    _ = try await manager.answer(state: text, questions: Self.questions)
                }
                answer = { text in
                    let result = try await manager.answer(state: text, questions: Self.questions)
                    return (result.answers, result.totalMilliseconds)
                }
            }
            print("loaded in \(String(format: "%.1f", Date().timeIntervalSince(started))) s")
            ready = true
            status = ""
            toggleStream()
        } catch {
            status = "Failed to load: \(error.localizedDescription)"
        }
    }

    func toggleStream() {
        if running {
            streamTask?.cancel()
            running = false
            return
        }
        running = true
        streamTask = Task { [weak self] in
            // back to back: the next ticket starts the moment the previous one is decided
            while let self, !Task.isCancelled {
                self.refill()
                guard !self.incoming.isEmpty else {
                    self.finished = true
                    self.running = false
                    self.current = nil
                    break
                }
                let ticket = self.incoming.removeFirst()  // no animation: a ticket takes ~50 ms on the 0.6B
                self.current = ticket
                let started = Date()
                await self.triage(ticket.text, typed: ticket.typed)
                self.activeSeconds += Date().timeIntervalSince(started)
                await Task.yield()
            }
        }
    }

    /// Keep the visible queue topped up from the backlog.
    private func refill() {
        while incoming.count < 8, next < Self.backlog.count {
            incoming.append(Incoming(text: Self.backlog[next], typed: false))
            next += 1
        }
    }

    /// Backlog tickets decided so far (typed ones not counted).
    var backlogDone: Int { tickets.filter { !$0.typed }.count }

    func submitDraft() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        draft = ""
        // typed tickets jump the queue
        incoming.insert(Incoming(text: text, typed: true), at: 0)
        if !running {
            Task {
                let ticket = incoming.removeFirst()
                current = ticket
                await triage(ticket.text, typed: true)
            }
        }
    }

    /// Clear the board, put the mock queue back at its first ticket and stream again.
    func restart() {
        streamTask?.cancel()
        running = false
        generation += 1
        withAnimation(.easeOut(duration: 0.2)) {
            tickets.removeAll()
            incoming.removeAll()
            current = nil
        }
        next = 0
        activeSeconds = 0
        finished = false
        toggleStream()
    }

    var triagedCount: Int { tickets.count }

    private func triage(_ text: String, typed: Bool) async {
        guard let answer else { return }
        let generation = generation
        busy = true
        defer { busy = false }
        do {
            let (answers, milliseconds) = try await answer(text)
            guard generation == self.generation else { return }
            let team = answers[0]
            let urgency = answers[1]
            let refund = answers[2]
            let teamIndex = team.probabilities.indices.max { team.probabilities[$0] < team.probabilities[$1] } ?? 0
            let urgencyIndex =
                urgency.probabilities.indices.max { urgency.probabilities[$0] < urgency.probabilities[$1] } ?? 0
            let ticket = Ticket(
                text: text, team: Team(rawValue: team.optionIDs[teamIndex]) ?? .product,
                teamConfidence: team.probabilities[teamIndex], urgency: urgencyIndex, refund: refund.noul ?? 0,
                milliseconds: milliseconds, typed: typed)
            print(
                String(
                    format: "%5.0f ms  %-11@ %-8@ refund %.2f  %@", milliseconds, ticket.team.rawValue,
                    Self.urgencyLabels[urgencyIndex], ticket.refund, text))
            withAnimation(.easeOut(duration: 0.12)) {
                tickets.append(ticket)
                if tickets.count > 1200 { tickets.removeFirst(tickets.count - 1200) }
            }
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }
}
