import DecisionPolicy
import Foundation
import LaneRunner
import LaneRunnerPolicy
import SwiftUI

@MainActor
final class LaneRunnerModel: ObservableObject {
    enum Control: Hashable, Identifiable {
        case model(DecisionModel)
        case heuristic
        case manual

        static var allCases: [Control] { DecisionModel.allCases.map(Control.model) + [.heuristic, .manual] }
        var id: String { title }
        var title: String {
            switch self {
            case .model(let model): model.title
            case .heuristic: "Heuristic"
            case .manual: "Manual (arrow keys)"
            }
        }
        var model: DecisionModel? {
            if case .model(let model) = self { return model }
            return nil
        }
    }

    @Published private(set) var game = LaneRunner(seed: 1)
    @Published private(set) var isRunning = false
    @Published private(set) var isLoading = false
    @Published private(set) var loadedModels: Set<DecisionModel> = []
    @Published private(set) var calls = 0
    @Published private(set) var lateRows = 0
    @Published private(set) var unsafeChoices = 0
    @Published private(set) var modelMs = 0.0
    @Published private(set) var confidence: Float = 0
    @Published private(set) var lastAction = "—"
    @Published var errorMessage: String?
    @Published var control: Control = .model(.gliner2Multilingual)
    @Published var seed: UInt64 = 1
    @Published var rowMs = 400.0

    private var policies: [DecisionModel: LaneRunnerPolicy] = [:]
    private var loopTask: Task<Void, Never>?
    private var generation = 0
    private var inFlight = false
    private var reply: (row: Int, decision: LaneRunnerPolicy.Decision)?
    private var queuedAction: LaneRunner.Action = .stay
    private var decisionNumber = 0
    private var autostart = false

    var usesModel: Bool { control.model != nil }
    var canPlay: Bool { control.model.map { loadedModels.contains($0) } ?? true }

    func loadModel() {
        guard let selected = control.model, !loadedModels.contains(selected), !isLoading else { return }
        isLoading = true
        errorMessage = nil
        Task {
            defer { isLoading = false }
            do {
                // Checksums and Core ML compilation must not run on the main actor.
                let started = Date()
                policies[selected] = try await Task.detached { try await LaneRunnerPolicy.load(selected) }.value
                loadedModels.insert(selected)
                FileHandle.standardError.write(
                    Data(String(format: "Loaded %@ in %.1f s\n", selected.title, -started.timeIntervalSinceNow).utf8))
                if autostart { toggle() }
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    /// `LANE_RUNNER_MODEL`, `LANE_RUNNER_SEED`, and `LANE_RUNNER_ROW_MS` preselect a run; the model then
    /// loads and runs without any clicks.
    func applyLaunchEnvironment() {
        let environment = ProcessInfo.processInfo.environment
        if let seed = environment["LANE_RUNNER_SEED"].flatMap(UInt64.init) { self.seed = seed }
        if let rowMs = environment["LANE_RUNNER_ROW_MS"].flatMap(Double.init) { self.rowMs = rowMs }
        game = LaneRunner(seed: seed)
        guard let name = environment["LANE_RUNNER_MODEL"] else { return }
        if name == "heuristic" {
            control = .heuristic
            toggle()
        } else if let model = DecisionModel(rawValue: name) {
            control = .model(model)
            autostart = true
            loadModel()
        }
    }

    func reset() {
        pause()
        game = LaneRunner(seed: seed)
        calls = 0
        lateRows = 0
        unsafeChoices = 0
        modelMs = 0
        confidence = 0
        lastAction = "—"
        decisionNumber = 0
        reply = nil
        errorMessage = nil
    }

    func pause() {
        generation += 1
        loopTask?.cancel()
        loopTask = nil
        inFlight = false
        queuedAction = .stay
        isRunning = false
    }

    func toggle() {
        if isRunning {
            pause()
            return
        }
        guard canPlay else { return }
        if game.isOver || game.distance == 0 { reset() }
        isRunning = true
        let run = generation
        loopTask = Task { [weak self] in
            while let self, self.isRunning, self.generation == run, !Task.isCancelled {
                self.requestDecision(run: run)
                try? await Task.sleep(for: .milliseconds(self.rowMs))
                guard self.isRunning, self.generation == run else { return }
                self.advanceRow()
            }
        }
    }

    func queue(_ action: LaneRunner.Action) {
        guard control == .manual, isRunning else { return }
        queuedAction = action
    }

    /// Applies one row: a reply that arrived for this row, else the runner stays in its lane.
    func advanceRow() {
        guard isRunning else { return }
        let action: LaneRunner.Action
        switch control {
        case .heuristic:
            action = game.heuristicAction
            lastAction = action.rawValue
        case .manual:
            action = queuedAction
            queuedAction = .stay
            lastAction = action.rawValue
        case .model:
            if let reply, reply.row == game.distance {
                action = reply.decision.action
                lastAction = action.rawValue
            } else {
                action = .stay
                lateRows += 1
                lastAction = "late → stay"
            }
            reply = nil
        }
        if !game.isSafe(action) && game.legalActions.contains(where: game.isSafe) { unsafeChoices += 1 }
        game.step(action)
        if game.isOver { pause() }
    }

    private func requestDecision(run: Int) {
        guard !inFlight, let selected = control.model, let policy = policies[selected] else { return }
        inFlight = true
        let snapshot = game
        let rotation = decisionNumber
        decisionNumber += 1
        Task { [weak self] in
            do {
                let decision = try await policy.decide(game: snapshot, rotation: rotation)
                guard let self, self.generation == run else { return }
                self.inFlight = false
                self.reply = (snapshot.distance, decision)
                self.calls += 1
                self.modelMs = decision.milliseconds
                self.confidence = decision.confidence
            } catch {
                guard let self, self.generation == run else { return }
                self.pause()
                self.errorMessage = error.localizedDescription
            }
        }
    }
}
