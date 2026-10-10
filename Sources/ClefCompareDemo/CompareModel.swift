import Foundation
import FluidUse
import SwiftUI

/// Side by side on the same tickets: Cloudflare clef-flash (9B, Core ML on the GPU) and clef-text-0.6b (distilled
/// from it, Core ML on the Neural Engine). Both answer the same three typed questions per ticket, concurrently.
/// Tickets are fictional mock data.
@MainActor
final class CompareModel: ObservableObject {
    struct Decision: Equatable {
        let team: String
        let urgency: Int
        let refund: Bool
        let teamConfidence: Float
        let milliseconds: Double
    }

    struct Ticket: Identifiable {
        let id = UUID()
        let text: String
        let big: Decision
        let small: Decision
        let typed: Bool
        var teamAgrees: Bool { big.team == small.team }
        var urgencyAgrees: Bool { big.urgency == small.urgency }
        var refundAgrees: Bool { big.refund == small.refund }
        var agreements: Int { [teamAgrees, urgencyAgrees, refundAgrees].filter { $0 }.count }
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

    @Published var status = "Loading both models…"
    @Published var ready = false
    @Published var tickets: [Ticket] = []
    @Published var running = false
    @Published var draft = ""
    @Published var incoming: [Incoming] = []
    @Published var current: Incoming?

    /// Loaded models as closures (the managers need macOS 15; the package targets 14): answers + total ms.
    private var bigAnswer: (@Sendable (String) async throws -> ([ClefAnswer], Double))?
    private var smallAnswer: (@Sendable (String) async throws -> ([ClefAnswer], Double))?
    private var next = 0
    private var generation = 0
    private var streamTask: Task<Void, Never>?

    var agreement: Double? {
        let decisions = tickets.count * 3
        return decisions == 0 ? nil : Double(tickets.reduce(0) { $0 + $1.agreements }) / Double(decisions)
    }
    func median(_ values: [Double]) -> Double? {
        let sorted = values.sorted()
        return sorted.isEmpty ? nil : sorted[sorted.count / 2]
    }
    var bigMedian: Double? { median(tickets.map(\.big.milliseconds)) }
    var smallMedian: Double? { median(tickets.map(\.small.milliseconds)) }

    static func directory(_ variable: String) -> URL? {
        guard let path = ProcessInfo.processInfo.environment[variable], !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path)
    }

    func start() async {
        guard bigAnswer == nil else { return }
        do {
            guard #available(macOS 15.0, *) else {
                status = "These Core ML models need macOS 15"
                return
            }
            status = "Loading clef-text-0.6b (Neural Engine)…"
            let smallDirectory: URL
            if let local = Self.directory("CLEF_TEXT_BUNDLE") {
                smallDirectory = local
            } else {
                status = "Downloading clef-text-0.6b (FluidInference/clef-text-0.6b-coreml, ~1.3 GB)…"
                smallDirectory = try await ClefTextModelStore.ensure()
            }
            let small = try await ClefTextManager.load(from: smallDirectory, buckets: [512])
            try await small.warm()
            let bigDirectory: URL
            if let local = Self.directory("CLEF_FLASH_BUNDLE") {
                bigDirectory = local
            } else {
                status = "Downloading clef-flash Core ML (FluidInference/clef-flash-coreml, ~11 GB)…"
                bigDirectory = try await ClefFlashModelStore.ensure()
            }
            status = "Loading clef-flash 9B (GPU)…"
            let big = try await ClefFlashManager.load(from: bigDirectory, bucket: 512)
            try await big.warm()
            for text in Self.mockTickets.prefix(2) {
                _ = try await big.answer(state: text, questions: Self.questions)
                _ = try await small.answer(state: text, questions: Self.questions)
            }
            bigAnswer = { text in
                let r = try await big.answer(state: text, questions: Self.questions)
                return (r.answers, r.totalMilliseconds)
            }
            smallAnswer = { text in
                let r = try await small.answer(state: text, questions: Self.questions)
                return (r.answers, r.totalMilliseconds)
            }
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
            while let self, !Task.isCancelled {
                self.refill()
                let ticket = withAnimation(.easeOut(duration: 0.2)) { self.incoming.removeFirst() }
                withAnimation(.easeOut(duration: 0.2)) { self.current = ticket }
                await self.triage(ticket)
                await Task.yield()
            }
        }
    }

    private func refill() {
        while incoming.count < 8 {
            incoming.append(Incoming(text: Self.mockTickets[next % Self.mockTickets.count], typed: false))
            next += 1
        }
    }

    func submitDraft() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        draft = ""
        withAnimation(.easeOut(duration: 0.2)) { incoming.insert(Incoming(text: text, typed: true), at: 0) }
        if !running {
            Task {
                let ticket = incoming.removeFirst()
                current = ticket
                await triage(ticket)
            }
        }
    }

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
        toggleStream()
    }

    private static func decision(_ answers: [ClefAnswer], _ ms: Double) -> Decision {
        let team = answers[0]
        let urgency = answers[1]
        let refund = answers[2]
        let teamIndex = team.probabilities.indices.max { team.probabilities[$0] < team.probabilities[$1] } ?? 0
        let urgencyIndex =
            urgency.probabilities.indices.max { urgency.probabilities[$0] < urgency.probabilities[$1] } ?? 0
        return Decision(
            team: team.optionIDs[teamIndex], urgency: urgencyIndex, refund: (refund.noul ?? 0) > 0.5,
            teamConfidence: team.probabilities[teamIndex], milliseconds: ms)
    }

    private func triage(_ ticket: Incoming) async {
        guard let bigAnswer, let smallAnswer else { return }
        let generation = generation
        do {
            // different hardware (GPU vs Neural Engine), so the two run at the same time
            async let big = bigAnswer(ticket.text)
            async let small = smallAnswer(ticket.text)
            let (b, s) = try await (big, small)
            guard generation == self.generation else { return }
            let row = Ticket(
                text: ticket.text, big: Self.decision(b.0, b.1), small: Self.decision(s.0, s.1), typed: ticket.typed)
            print(
                String(
                    format: "9B %4.0f ms  0.6B %4.0f ms  agree %d/3  %@", row.big.milliseconds,
                    row.small.milliseconds, row.agreements, ticket.text))
            withAnimation(.spring(duration: 0.35)) {
                tickets.append(row)
                if tickets.count > 200 { tickets.removeFirst(tickets.count - 200) }
                current = nil
            }
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }
}
