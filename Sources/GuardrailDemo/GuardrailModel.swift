import FluidUse
import Foundation
import SwiftUI

/// The right panel: whatever the model checked last.
struct Panel {
    var timing: Timing?
    var what = ""
    var verdict: (text: String, bad: Bool)?
    var attack: Double?
    var harm: Double?
    var factcheck: Double?
    var route: [(name: String, p: Double)] = []
    var routeNote = "computed on send"
    var pii: [PIISpan] = []
    var leaves: String?  // masked text, or nil
    var leavesBlocked = false
}

struct ChatMessage: Identifiable {
    enum Kind {
        case user(SendCheck)
        case blocked(SendCheck)
        case bot
    }

    let id = UUID()
    var kind: Kind
    // bot only
    var text = ""
    var typing = false
    var checking = false
    var reply: ReplyCheck?
}

@available(macOS 15.0, *)
@MainActor
final class GuardrailModel: ObservableObject {
    @Published var status = "Loading model…"
    @Published var ready = false
    @Published var scenario = Scenario.all[0]
    @Published var source = Scenario.all[0].source
    @Published var input = "" { didSet { if input != oldValue { scheduleLive() } } }
    @Published var messages: [ChatMessage] = []
    @Published var panel = Panel()
    // live lane results (dropped when stale)
    @Published var screen: ScreenCheck?
    @Published var livePII: PIICheck?
    @Published var checks = 0
    @Published var onANE = 0

    private var manager: Vela2Manager?
    private var liveTask: Task<Void, Never>?
    private var seq = 0
    private var lastUser = ""

    // MARK: startup

    func start() async {
        guard manager == nil else { return }
        let dir = Launch.modelDirectory
        print("loading \(dir.path)")
        do {
            let m = try await Task.detached { () async throws -> Vela2Manager in
                let m = try await Vela2Manager.load(from: dir)
                try await m.warm()
                // first pass over each lane's schema so the first real message is steady-state
                _ = try await Guard.screen(m, "hello")
                _ = try await Guard.pii(m, "hello")
                _ = try await Guard.send(m, "hello")
                _ = try await Guard.reply(m, request: "hi", source: "The sky is blue.", answer: "The sky is green.")
                return m
            }.value
            manager = m
            ready = true
            status = "\(m.modelName) ready · ANE ≤ \(m.aneMaxLength) tokens, GPU above"
            print("ready: \(m.modelName) (Core ML: ANE <= \(m.aneMaxLength) tokens, GPU above)")
            scheduleLive()
            if CommandLine.arguments.contains("--autoplay") { await autoplay() }
        } catch {
            status = "Failed to load model: \(error.localizedDescription)"
            print("load failed: \(error)")
        }
    }

    /// Hidden `--autoplay [scenario id]` (for screenshots): send each suggestion of the scenario, then leave the
    /// first one back in the composer so the live lane and its highlight show.
    private func autoplay() async {
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--autoplay"), i + 1 < args.count, let s = Scenario.all.first(where: { $0.id == args[i + 1] }) {
            setScenario(s)
        }
        for msg in scenario.suggest {
            input = msg
            try? await Task.sleep(for: .milliseconds(600))
            send()
            try? await Task.sleep(for: .milliseconds(1500))
        }
        input = scenario.suggest[scenario.suggest.count - 1]
    }

    private func count(_ t: Timing) {
        checks += 1
        if t.onNeuralEngine { onANE += 1 }
    }

    // MARK: scenarios

    func setScenario(_ s: Scenario) {
        scenario = s
        source = s.source
        messages = []
        lastUser = ""
        input = ""
        panel = Panel()
    }

    // MARK: lane 1 — while typing

    private func scheduleLive() {
        liveTask?.cancel()
        seq += 1
        let my = seq
        let text = input
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            screen = nil
            livePII = nil
            return
        }
        guard let m = manager else { return }
        liveTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(120))
            guard !Task.isCancelled else { return }
            // (a) attack + harm: short schema → Neural Engine for short messages
            guard let s = try? await Task.detached(operation: { try await Guard.screen(m, text) }).value else { return }
            guard let self, my == self.seq, text == self.input else { return }
            self.count(s.timing)
            self.screen = s
            self.showScreen(s)
            print(Guard.log(s))
            // (b) PII spans: the 17-label schema alone is ~192 tokens → GPU
            guard !Task.isCancelled,
                let p = try? await Task.detached(operation: { try await Guard.pii(m, text) }).value
            else { return }
            guard my == self.seq, text == self.input else { return }
            self.count(p.timing)
            self.livePII = p
            self.showPII(p)
            print(Guard.log(p))
        }
    }

    private func showScreen(_ s: ScreenCheck) {
        var p = panel
        p.timing = s.timing
        p.what = "· live screen: 2 decisions"
        p.attack = s.attack
        p.harm = s.harm
        p.factcheck = nil
        p.route = []
        p.routeNote = "computed on send"
        if let r = s.blockReason {
            p.verdict = ("⛔ Would be blocked: \(r)", true)
        } else {
            p.verdict = ("✓ Looks fine so far", false)
        }
        panel = p
    }

    private func showPII(_ c: PIICheck) {
        var p = panel
        p.timing = c.timing
        p.what = "· live personal-info spans"
        p.pii = c.spans
        p.leaves = (screen?.blockReason == nil) ? Guard.mask(c.text, c.spans) : nil
        p.leavesBlocked = screen?.blockReason != nil
        panel = p
    }

    // MARK: lane 2 — on send

    func send() {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, let m = manager else { return }
        input = ""
        let sc = scenario
        Task { [weak self] in
            guard let c = try? await Task.detached(operation: { try await Guard.send(m, text) }).value, let self else { return }
            self.count(c.timing)
            print(Guard.log(c))
            self.showSend(c)
            if c.blocked {
                self.messages.append(ChatMessage(kind: .blocked(c)))
                return
            }
            self.lastUser = c.masked
            self.messages.append(ChatMessage(kind: .user(c)))
            var bot = ChatMessage(kind: .bot)
            bot.typing = true
            let id = bot.id
            self.messages.append(bot)
            try? await Task.sleep(for: .milliseconds(700))
            guard self.scenario == sc, let i = self.messages.firstIndex(where: { $0.id == id }) else { return }
            self.messages[i].typing = false
            self.messages[i].text = sc.reply
            self.checkReply(id)
        }
    }

    private func showSend(_ c: SendCheck) {
        panel = Panel(
            timing: c.timing, what: "· \(c.decisions) decisions in 1 call",
            verdict: c.blockReason.map { ("⛔ Blocked: \($0)", true) } ?? ("✓ Allowed · route → \(c.domain.first?.name ?? "?")", false),
            attack: c.attack, harm: c.harm, factcheck: c.factcheckP, route: c.domain, pii: c.pii,
            leaves: c.blocked ? nil : c.masked, leavesBlocked: c.blocked)
    }

    // MARK: lane 3 — reply vs source

    func checkReply(_ id: UUID) {
        guard let m = manager, let i = messages.firstIndex(where: { $0.id == id }) else { return }
        let answer = messages[i].text
        let request = lastUser
        let source = source
        messages[i].checking = true
        Task { [weak self] in
            let r = try? await Task.detached(operation: { try await Guard.reply(m, request: request, source: source, answer: answer) }).value
            guard let self, let i = self.messages.firstIndex(where: { $0.id == id }) else { return }
            self.messages[i].checking = false
            guard let r else { return }
            self.count(r.timing)
            print(Guard.log(r))
            self.messages[i].reply = r
            var p = self.panel
            p.timing = r.timing
            p.what = "· reply vs source"
            let n = r.unsupported.count
            p.verdict = n > 0 ? ("⚠ \(n) unsupported claim\(n > 1 ? "s" : "")", true) : ("✓ Reply grounded in the source", false)
            self.panel = p
        }
    }

    func editReply(_ id: UUID, _ text: String) {
        guard let i = messages.firstIndex(where: { $0.id == id }), messages[i].text != text else { return }
        messages[i].text = text
    }
}
