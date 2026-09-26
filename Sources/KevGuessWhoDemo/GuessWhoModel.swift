import Foundation
import FluidUse
import SortAnything
import SwiftUI

/// Blocks the inference producer and the turn pacing while the demo is paused.
actor PauseGate {
    private var paused = false

    func set(_ value: Bool) { paused = value }

    func wait() async throws {
        while paused {
            try await Task.sleep(for: .milliseconds(50))
        }
    }
}

/// A self-playing Guess Who: Kev reads every card's Wikipedia abstract once, answering every question in the pool in
/// one fused Core ML call per card, then plays turns until one card (the hidden person) is left.
@MainActor
final class GuessWhoModel: ObservableObject {
    struct Card: Identifiable {
        let id: Int
        let item: SortItem
        /// Kev's yes/no per pool question, filled by the scan.
        var answers: [Bool]?
        var down = false
        var flipping = false
    }

    enum Phase: Equatable {
        case loading(String)
        case dealing
        case scanning
        case asking(String)
        case solved(String)
        case failed(String)
    }

    static let questions = [
        "Is this person an athlete?",
        "Is this person a musician or singer?",
        "Is this person a politician?",
        "Is this person a woman?",
        "Was this person born before 1950?",
        "Is this person from the United States?",
        "Is this person from Europe?",
        "Does this person play football (soccer)?",
        "Is this person a painter or visual artist?",
        "Is this person an actor?",
        "Has this person competed at the Olympic Games?",
        "Is this person a writer or poet?",
    ]
    static let cardsPerGame = 80

    @Published var cards: [Card] = []
    @Published var phase: Phase = .loading("Loading Kev-0.8B…")
    @Published var secret: Int?
    @Published var asked: [(question: String, answer: Bool, removed: Int)] = []
    @Published var game = 0
    @Published var scanned = 0
    @Published var scanSeconds: Double = 0
    @Published var lastCallMs: Double = 0
    @Published var medianCallMs: Double = 0
    @Published var totalDecisions = 0
    @Published var paused = false
    @Published var ready = false

    /// KevFastManager.answer (macOS 15+; the package targets macOS 14).
    private var answer: (@Sendable (String, [KevQuestion]) async throws -> [KevAnswer])?
    /// KevFastManager.warm: a function left idle pays a ~0.3–0.8 s re-setup on its next call, so each scan re-warms
    /// first instead of stalling mid-wall.
    private var warm: (@Sendable () async throws -> Void)?
    private var people: [SortItem] = []
    private var callTimes: [Double] = []
    private var started = false
    private var runner: Task<Void, Never>?
    private let gate = PauseGate()

    /// Decisions per wall-clock second in this game's scan.
    var scanRate: Double { scanSeconds > 0 ? Double(scanned * Self.questions.count) / scanSeconds : 0 }
    var remaining: Int { cards.filter { !$0.down }.count }

    func start() async {
        guard !started else { return }
        started = true
        do {
            guard #available(macOS 15.0, *) else {
                phase = .failed("Kev's fused Core ML path needs macOS 15")
                return
            }
            let directory = Self.modelDirectory()
            phase = .loading("Loading Kev-0.8B Core ML from \(directory.lastPathComponent)…")
            let manager = try await KevFastManager.load(from: directory)
            phase = .loading("Compiling the GPU functions…")
            try await manager.warm()
            answer = { try await manager.answer(state: $0, questions: $1) }
            warm = { try await manager.warm() }
            phase = .loading("Fetching Wikipedia people (DBpedia)…")
            people = try await DBpediaSample.load(count: 960, seed: 7, classes: ["artist", "athlete", "politician"])
            log(
                "ready · Kev-0.8B fused Core ML (GPU) · \(people.count) Wikipedia people · \(Self.questions.count) questions"
            )
            ready = true
            run()
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func togglePause() {
        paused.toggle()
        log(paused ? "paused" : "playing")
        let value = paused
        Task { await gate.set(value) }
    }

    /// Abandons the current game and deals a fresh wall.
    func reset() {
        guard ready else { return }
        runner?.cancel()
        callTimes = []
        lastCallMs = 0
        medianCallMs = 0
        totalDecisions = 0
        scanSeconds = 0
        log("reset")
        run()
    }

    private func run() {
        runner = Task {
            do {
                while !Task.isCancelled {
                    try await playGame()
                    try await pause(for: .seconds(3))
                }
            } catch is CancellationError {
            } catch {
                phase = .failed(error.localizedDescription)
            }
        }
    }

    /// Sleeps `duration` of unpaused time.
    private func pause(for duration: Duration) async throws {
        try await gate.wait()
        try await Task.sleep(for: duration)
        try await gate.wait()
    }

    private func playGame() async throws {
        guard let answer else { return }
        let offset = (game * Self.cardsPerGame) % max(people.count - Self.cardsPerGame, 1)
        game += 1
        cards = people[offset..<offset + Self.cardsPerGame].enumerated().map { Card(id: $0.offset, item: $0.element) }
        asked = []
        secret = nil
        scanned = 0

        phase = .dealing
        let warmStart = DispatchTime.now().uptimeNanoseconds
        try await warm?()
        log(
            String(
                format: "GPU functions re-warmed in %.2f s",
                Double(DispatchTime.now().uptimeNanoseconds - warmStart) / 1e9))
        phase = .scanning
        log("game \(game) · reading \(cards.count) bios × \(Self.questions.count) questions")
        let questions = Self.questions.map { KevQuestion.noul($0) }
        let scanStart = DispatchTime.now().uptimeNanoseconds
        let texts = cards.map(\.item.text)
        let gate = gate
        // Inference runs back to back off the main actor; the UI only consumes results, so rendering never
        // delays the next call.
        let (results, continuation) = AsyncThrowingStream.makeStream(of: (Int, [Bool], Double).self)
        let producer = Task.detached(priority: .userInitiated) {
            do {
                for (index, text) in texts.enumerated() {
                    try Task.checkCancellation()
                    try await gate.wait()
                    let start = DispatchTime.now().uptimeNanoseconds
                    let answers = try await answer(text, questions)
                    let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                    continuation.yield((index, answers.map { $0.best == "true" }, ms))
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        defer { producer.cancel() }
        for try await (index, answers, ms) in results {
            cards[index].answers = answers
            scanned += 1
            record(ms: ms, decisions: answers.count)
            scanSeconds = Double(DispatchTime.now().uptimeNanoseconds - scanStart) / 1e9
            let yes = answers.indices.filter { answers[$0] }.map { Self.labels[$0] }
            log(
                String(format: "card %2d/%d  %5.1f ms  12 decisions  ", index + 1, cards.count, ms)
                    + "\(cards[index].item.title) → yes: \(yes.isEmpty ? "none" : yes.joined(separator: ", "))",
                model: true)
        }
        log(
            String(
                format: "scan done · %d decisions in %.2f s · %.0f decisions/s", scanned * Self.questions.count,
                scanSeconds, scanRate))

        // Hide someone whose answers differ from every other card's, so the questions can single them out.
        let signatures = cards.map { $0.answers ?? [] }
        let unique = cards.indices.filter { i in !cards.indices.contains { $0 != i && signatures[$0] == signatures[i] }
        }
        let hidden = unique.randomElement() ?? Int.random(in: 0..<cards.count)
        secret = hidden
        log("hidden person picked · \(cards.count) cards up")
        var unused = Array(Self.questions.indices)
        while remaining > 1, !unused.isEmpty {
            let alive = cards.indices.filter { !cards[$0].down }
            // the unused question that splits the cards still up closest to half
            let pick = unused.min { a, b in
                split(a, alive) < split(b, alive)
            }!
            unused.removeAll { $0 == pick }
            let target = cards[hidden].answers![pick]
            let losers = alive.filter { cards[$0].answers![pick] != target }
            if losers.isEmpty { continue }
            phase = .asking(Self.questions[pick])
            try await pause(for: .milliseconds(900))
            for index in losers {
                withAnimation(.easeIn(duration: 0.35)) { cards[index].flipping = true }
            }
            try await pause(for: .milliseconds(350))
            for index in losers {
                withAnimation(.easeOut(duration: 0.25)) {
                    cards[index].down = true
                    cards[index].flipping = false
                }
            }
            asked.append((Self.questions[pick], target, losers.count))
            log(
                "Q\(asked.count) \(Self.questions[pick]) → \(target ? "YES" : "NO") · \(losers.count) flipped · \(remaining) left"
            )
            try await pause(for: .milliseconds(700))
        }
        let left = cards.filter { !$0.down }
        phase = .solved(
            left.count == 1 ? left[0].item.title : "\(left.count) left — it was \(cards[hidden].item.title)")
        log("solved in \(asked.count) questions · it's \(cards[hidden].item.title)")
    }

    static let labels = [
        "athlete", "musician", "politician", "woman", "born<1950", "US", "Europe", "football", "painter", "actor",
        "Olympian", "writer",
    ]

    /// One console line; model calls are marked red for the log pane.
    private func log(_ text: String, model: Bool = false) {
        let time = Date().formatted(.dateTime.hour(.twoDigits(amPM: .omitted)).minute().second())
        print(model ? "\(time) \u{1B}[31m[Core ML]\u{1B}[0m \(text)" : "\(time) \u{1B}[36m[game]\u{1B}[0m \(text)")
    }

    /// How far a question is from a 50/50 split of the cards still up.
    private func split(_ question: Int, _ alive: [Int]) -> Int {
        let yes = alive.filter { cards[$0].answers![question] }.count
        return abs(2 * yes - alive.count)
    }

    private func record(ms: Double, decisions: Int) {
        lastCallMs = ms
        callTimes.append(ms)
        if callTimes.count > 200 { callTimes.removeFirst(callTimes.count - 200) }
        medianCallMs = callTimes.sorted()[callTimes.count / 2]
        totalDecisions += decisions
    }

    static func modelDirectory() -> URL {
        if let path = ProcessInfo.processInfo.environment["KEV_MODEL_DIR"] { return URL(fileURLWithPath: path) }
        if CommandLine.arguments.count > 1 { return URL(fileURLWithPath: CommandLine.arguments[1]) }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
            "Documents/mobius-kev-0.8b/models/computer-use/kev-0.8b/coreml/build/kev-model")
    }
}
