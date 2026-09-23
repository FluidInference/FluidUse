import FlappyBird
import FlappyBirdPolicy
import Foundation
import SwiftUI

@MainActor
final class FlappyModel: ObservableObject {
    enum Control: String, CaseIterable, Identifiable {
        case gliclass = "GLiClass LUT8"
        case gliner2Multilingual = "GLiNER 2.5 multilingual"
        case gliner2Base = "GLiNER 2.5 base"
        case gliner2Small = "GLiNER 2.5 small"
        case laya = "Laya E8"
        case verdict = "Verdict"
        case kev05 = "Kev 0.5B"
        case kev06 = "Kev 0.6B"
        case kai = "Decision 1.0 Kai"
        case lex = "Decision 1.0 Lex"
        case lfm350 = "LFM2.5-350M-RLCD"
        case jeff = "Jeff"
        case nanojev = "NanoJev (local)"
        case heuristic = "Heuristic"
        case manual = "Manual"
        var id: String { rawValue }
        var model: FlappyBirdPolicy.Model? {
            switch self {
            case .gliclass: .gliclass
            case .gliner2Multilingual: .gliner2Multilingual
            case .gliner2Base: .gliner2Base
            case .gliner2Small: .gliner2Small
            case .laya: .laya
            case .verdict: .verdict
            case .kev05: .kev05
            case .kev06: .kev06
            case .kai: .kai
            case .lex: .lex
            case .lfm350: .lfm350
            case .jeff: .jeff
            case .nanojev: .nanojev
            case .heuristic, .manual: nil
            }
        }
    }

    @Published private(set) var game = FlappyBird(seed: 1)
    @Published private(set) var isRunning = false
    @Published private(set) var isLoading = false
    @Published private(set) var loadedModels: Set<FlappyBirdPolicy.Model> = []
    @Published private(set) var calls = 0
    @Published private(set) var overrides = 0
    @Published private(set) var lateReplies = 0
    @Published private(set) var modelMs = 0.0
    @Published private(set) var responseMs = 0.0
    @Published private(set) var probability: Float = 0
    @Published private(set) var tokens = 0
    @Published private(set) var lastAction = "—"
    @Published var errorMessage: String?
    @Published var control: Control = .gliclass
    @Published var safetyGuard = true
    @Published var seed: UInt64 = 1
    @Published var addedDelayMs = 0.0

    private var policies: [FlappyBirdPolicy.Model: FlappyBirdPolicy] = [:]
    private var clockTask: Task<Void, Never>?
    private var decisionTask: Task<Void, Never>?
    private var generation = 0
    private var pending = false
    private var queuedAction: FlappyBird.Action = .coast
    private var decisionNumber = 0
    private var lastRequestedFrame = -FlappyBird.decisionFrames

    var usesModel: Bool { control.model != nil }
    var canPlay: Bool { control.model.map { loadedModels.contains($0) } ?? true }

    func loadModel() {
        guard let selected = control.model, !loadedModels.contains(selected), !isLoading else { return }
        isLoading = true
        errorMessage = nil
        Task {
            defer { isLoading = false }
            do {
                policies[selected] = try await FlappyBirdPolicy.load(selected)
                loadedModels.insert(selected)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    func reset() {
        pause()
        game = FlappyBird(seed: seed)
        calls = 0
        overrides = 0
        lateReplies = 0
        modelMs = 0
        responseMs = 0
        probability = 0
        tokens = 0
        lastAction = "—"
        decisionNumber = 0
        lastRequestedFrame = -FlappyBird.decisionFrames
        errorMessage = nil
    }

    func pause() {
        generation += 1
        clockTask?.cancel()
        decisionTask?.cancel()
        clockTask = nil
        decisionTask = nil
        pending = false
        queuedAction = .coast
        isRunning = false
    }

    func toggle() {
        if isRunning {
            pause()
            return
        }
        guard canPlay else { return }
        if game.isOver || game.frames == 0 { reset() }
        isRunning = true
        requestDecision()
        let run = generation
        clockTask = Task { [weak self] in
            var previous = DispatchTime.now().uptimeNanoseconds
            var accumulator = 0.0
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(8))
                guard let self, self.isRunning, self.generation == run, !Task.isCancelled else { return }
                let now = DispatchTime.now().uptimeNanoseconds
                let elapsed = Double(now - previous) / 1e9
                previous = now
                guard elapsed < 0.5 else {
                    self.pause()
                    self.errorMessage = "Paused after a clock interruption. Press Play to resume."
                    return
                }
                accumulator += elapsed
                while accumulator >= FlappyBird.stepSeconds && self.isRunning {
                    self.advanceFrame()
                    accumulator -= FlappyBird.stepSeconds
                }
            }
        }
    }

    func flap() {
        guard control == .manual, isRunning else { return }
        queuedAction = .flap
    }

    // Fixed-step logic is separate from wall-clock scheduling so lifecycle checks need no sleeps.
    func advanceFrame() {
        guard isRunning else { return }
        if control == .heuristic && game.frames % FlappyBird.decisionFrames == 0 {
            queuedAction = game.heuristicAction
            lastAction = queuedAction.rawValue
        }
        game.step(queuedAction)
        queuedAction = .coast
        if game.isOver {
            pause()
            return
        }
        if !pending && game.frames - lastRequestedFrame >= FlappyBird.decisionFrames {
            requestDecision()
        }
    }

    private func requestDecision() {
        guard let selected = control.model, let policy = policies[selected] else { return }
        pending = true
        lastRequestedFrame = game.frames
        let snapshot = game
        let run = generation
        let reversed = decisionNumber % 2 != 0
        decisionNumber += 1
        let delay = addedDelayMs
        let started = DispatchTime.now().uptimeNanoseconds
        decisionTask = Task { [weak self] in
            do {
                let decision = try await policy.decide(game: snapshot, reversed: reversed)
                if delay > 0 { try await Task.sleep(for: .milliseconds(delay)) }
                guard let self, !Task.isCancelled, self.generation == run, self.isRunning else { return }
                self.pending = false
                self.calls += 1
                self.modelMs = decision.milliseconds
                self.responseMs = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
                if self.responseMs > 100 { self.lateReplies += 1 }
                self.probability = decision.flapProbability
                self.tokens = decision.tokens
                let applied =
                    self.safetyGuard
                    ? self.game.guardedAction(preferred: decision.action) : decision.action
                if applied != decision.action { self.overrides += 1 }
                self.lastAction =
                    applied == decision.action
                    ? applied.rawValue : "\(decision.action.rawValue) → \(applied.rawValue)"
                self.queuedAction = applied
            } catch {
                guard let self, self.generation == run, !Task.isCancelled else { return }
                self.pause()
                self.errorMessage = error.localizedDescription
            }
        }
    }
}
