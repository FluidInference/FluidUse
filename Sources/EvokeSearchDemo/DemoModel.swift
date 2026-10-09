import FluidUse
import Foundation
import Observation

@MainActor
@Observable
final class DemoModel {
    enum Phase: Equatable {
        case loading(String)
        case ready
        case failed(String)
    }

    var phase: Phase = .loading("Loading Core ML models…")
    var query = ""
    var mode: SearchMode = .evoke
    var hits: [SearchHit] = []
    var queryTerms: [(word: String, weight: Float, evoked: Bool)] = []
    var queryLatencyMs: Double?
    var indexSummary = ""

    private var encoder: EvokeManager?
    private var index: SearchIndex?
    private var searchTask: Task<Void, Never>?
    /// Last encoded query: the Evoke flip reuses what the final keystroke already computed.
    private var cachedQuery: (text: String, atoms: EvokeTerms, ms: Double)?
    private var autoplayTask: Task<Void, Never>?
    var isAutoplaying = false
    var caption = ""

    static let suggestions = [
        "football transfer rumors", "new album release", "healthy breakfast ideas", "video game launch",
        "climate change protest", "startup fundraising", "red carpet fashion", "space exploration",
        "wedding anniversary", "back to school",
    ]

    func start() async {
        let tweets = Launch.postsFile.map(MockTimeline.load(from:)) ?? MockTimeline.tweets
        do {
            let encoder = try await EvokeManager.loadDefault(lengths: [64]) { file, bytes in
                Task { @MainActor in self.phase = .loading(bytes > 0 ? "Downloaded \(file)" : "Downloading \(file)…") }
            }
            phase = .loading("Indexing \(tweets.count) posts…")
            _ = try await encoder.terms(for: "warm up", kind: .document)
            let clock = ContinuousClock()
            var atoms: [EvokeTerms] = []
            let elapsed = try await clock.measure {
                for tweet in tweets {
                    atoms.append(try await encoder.terms(for: tweet.text, kind: .document))
                    if atoms.count % 50 == 0 { phase = .loading("Indexing \(atoms.count)/\(tweets.count) posts…") }
                }
            }
            var words: [Int: String] = [:]
            for id in Set(atoms.flatMap(\.keys)) {
                if let word = encoder.word(for: id) { words[id] = word }
            }
            // Warm the query path so the first keystroke isn't a cold Neural Engine call.
            _ = try await encoder.terms(for: "warm up", kind: .query)
            self.encoder = encoder
            self.index = SearchIndex(tweets: tweets, atoms: atoms, words: words)
            let seconds = Double(elapsed.components.attoseconds) / 1e18 + Double(elapsed.components.seconds)
            indexSummary = String(
                format: "%d posts indexed in %.2f s on Neural Engine · %.1f terms/post", atoms.count, seconds,
                Double(atoms.map(\.count).reduce(0, +)) / Double(atoms.count))
            phase = .ready
            search()
            startAutoplay()
        } catch {
            phase = .failed("\(error)")
        }
    }

    /// Scripted loop: type each query, show keyword results, then flip to Evoke.
    func startAutoplay() {
        resumeTask?.cancel()
        autoplayTask?.cancel()
        searchTask?.cancel()
        isAutoplaying = true
        tickStart = .now
        encodesSinceTick = 0
        autoplayTask = Task {
            while !Task.isCancelled {
                for q in Self.suggestions {
                    guard await play(q) else { return }
                }
            }
        }
    }

    private var resumeTask: Task<Void, Never>?

    /// Pauses on user interaction; resumes by itself after 20 s idle unless `resume` is false.
    func stopAutoplay(resume: Bool = true) {
        autoplayTask?.cancel()
        autoplayTask = nil
        isAutoplaying = false
        caption = ""
        resumeTask?.cancel()
        guard resume else { return }
        resumeTask = Task {
            try? await Task.sleep(for: .seconds(20))
            guard !Task.isCancelled else { return }
            startAutoplay()
        }
    }

    /// No pacing: every keystroke is encoded and rendered before the next one is typed.
    private func play(_ text: String) async -> Bool {
        mode = .evoke
        for i in text.indices {
            guard !Task.isCancelled else { return false }
            query = String(text[...i])
            await runSearch()
            encodesSinceTick += 1
            let now = ContinuousClock.now
            if now - tickStart >= .milliseconds(500) {
                let seconds =
                    Double((now - tickStart).components.attoseconds) / 1e18
                    + Double((now - tickStart).components.seconds)
                caption = String(
                    format: "Live: %.0f searches/s · every keystroke encoded on the Neural Engine",
                    Double(encodesSinceTick) / seconds)
                encodesSinceTick = 0
                tickStart = now
            }
            // One hop through the main queue so SwiftUI can draw this keystroke.
            await Task.yield()
        }
        return !Task.isCancelled
    }

    private var encodesSinceTick = 0
    private var tickStart = ContinuousClock.now

    func search() {
        // Autoplay drives runSearch itself; ignore the TextField's onChange echo.
        guard !isAutoplaying else { return }
        searchTask?.cancel()
        searchTask = Task { await runSearch() }
    }

    /// id -> readable word (nil cached too), so the expansion panel skips BPE decoding per keystroke.
    private var wordCache: [Int: String?] = [:]

    private func cachedWord(_ id: Int, _ encoder: EvokeManager) -> String? {
        if let hit = wordCache[id] { return hit }
        let word = encoder.word(for: id)
        wordCache[id] = word
        return word
    }

    private func encodeQuery(_ text: String, _ encoder: EvokeManager) async -> EvokeTerms? {
        if let cachedQuery, cachedQuery.text == text { return cachedQuery.atoms }
        guard let ids = try? encoder.tokenize(text),
            let (atoms, ms) = try? await encoder.terms(tokenIds: ids, kind: .query)
        else { return nil }
        cachedQuery = (text, atoms, ms)
        return atoms
    }

    private func runSearch() async {
        guard let index, let encoder else { return }
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            hits = index.tweets.map { SearchHit(tweet: $0, score: 0, terms: []) }
            queryTerms = []
            queryLatencyMs = nil
            return
        }
        switch mode {
        case .keyword:
            // Encode anyway, as search-as-you-type would; the Evoke flip then reuses the result.
            _ = await encodeQuery(text, encoder)
            hits = index.keyword(text)
            queryTerms = SearchIndex.terms(text).map { ($0, 1, false) }
            queryLatencyMs = nil
        case .evoke:
            guard let atoms = await encodeQuery(text, encoder), !Task.isCancelled else { return }
            queryLatencyMs = cachedQuery?.ms
            hits = index.evoke(atoms)
            let literal = Set(SearchIndex.terms(text))
            var terms: [(word: String, weight: Float, evoked: Bool)] = []
            for (id, weight) in atoms.sorted(by: { $0.value > $1.value }) {
                guard let word = cachedWord(id, encoder), !terms.contains(where: { $0.word == word }) else {
                    continue
                }
                terms.append((word, weight, !literal.contains(word)))
                if terms.count == 14 { break }
            }
            queryTerms = terms
        }
    }
}
