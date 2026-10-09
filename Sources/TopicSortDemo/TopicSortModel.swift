import BookmarkSort
import FluidUse
import Foundation
import SwiftUI

/// Streams posts in, embeds each with EmbeddingGemma 2 on the Neural Engine and files it into the nearest broad
/// topic as it arrives. Broad topics are found from the posts themselves (first after 40 posts, re-sorted as the
/// stream grows); any broad topic can then be split into specific subtopics.
@MainActor
final class TopicSortModel: ObservableObject {
    enum Phase: Equatable {
        case loading(String)
        /// Model and posts loaded; waiting for Start.
        case ready
        case streaming
        case paused
        case resorting
        case splitting(String)
        case done
        case failed(String)
    }

    @Published private(set) var phase: Phase = .loading("Starting…")
    /// Posts that have arrived, in arrival order.
    @Published private(set) var posts: [Bookmark] = []
    @Published private(set) var topics: [TopicNode] = []
    /// Post index → ids of its topic and subtopic (ids, not nodes: nodes would copy members and centroids).
    @Published private(set) var paths: [Int: [String]] = [:]
    @Published private(set) var postsPerSecond: Double = 0
    @Published private(set) var lastEvent = ""
    @Published var selection: String? = TopicSortModel.allID

    static let allID = "all"
    static let palette: [Color] = [.blue, .orange, .green, .pink, .purple, .teal, .red, .indigo, .brown, .mint]
    /// Posts embedded concurrently per batch; the UI updates once per batch.
    static let batchSize = max(1, value("batch").flatMap(Int.init) ?? 64)
    static let resortPoints = [40, 160, 400, 1000, 4000]

    private var source: [Bookmark] = []
    private var vectors: [[Float]] = []
    private var manager: EmbeddingGemma2Manager?
    private let cache = PhraseVectorCache()
    private var topicCount = 6
    private var runTask: Task<Void, Never>?
    private var pauseRequested = false
    /// Streaming time so far, excluding pauses.
    private var activeSeconds: Double = 0
    /// Bumped by Reset, so work that awaited across a Reset (a split) drops its result.
    private var generation = 0

    var totalCount: Int { source.count }
    var subtopicCount: Int { topics.reduce(0) { $0 + $1.children.count } }
    /// Splits only once the stream is finished: a re-sort replaces the broad topics and would drop subtopics.
    var canSplit: Bool { phase == .done }
    var isRunning: Bool { phase == .streaming || phase == .resorting }

    func node(_ id: String) -> TopicNode? {
        for topic in topics {
            if topic.id == id { return topic }
            if let child = topic.children.first(where: { $0.id == id }) { return child }
        }
        return nil
    }

    func color(_ node: TopicNode) -> Color { Self.palette[node.colorIndex % Self.palette.count] }

    /// Topic, then subtopic, of post `index`.
    func path(_ index: Int) -> [TopicNode] { (paths[index] ?? []).compactMap(node) }

    /// Feed for the selection: newest first (the latest 200 for all posts); a split topic is grouped by subtopic.
    var feedSections: [(id: String, title: String?, color: Color?, items: [Int])] {
        guard let selection, selection != Self.allID, let node = node(selection) else {
            return [(Self.allID, nil, nil, Array(posts.indices.reversed().prefix(200)))]
        }
        if node.children.isEmpty { return [(node.id, nil, nil, node.members.sorted(by: >))] }
        return node.children.map { ($0.id, $0.name, color($0), $0.members.sorted(by: >)) }
    }

    private static func value(_ name: String) -> String? {
        CommandLine.arguments.first { $0.hasPrefix("--\(name)=") }.map { String($0.dropFirst(name.count + 3)) }
    }

    /// Loads the posts and the model, then waits for Start (or starts at once with `--autostart`).
    func prepare() async {
        do {
            // Default: the bundled fictional posts (Tools/make_mock_posts.py); --posts= streams your own file.
            let bundled = Bundle.module.url(
                forResource: "mock-posts", withExtension: "jsonl", subdirectory: "Resources")
            guard
                let path = Self.value("posts") ?? ProcessInfo.processInfo.environment["TOPIC_SORT_POSTS"]
                    ?? bundled?.path
            else { throw BookmarkSortError.invalidInput("Missing Resources/mock-posts.jsonl") }
            source = try String(contentsOfFile: path, encoding: .utf8).split(separator: "\n").map {
                try JSONDecoder().decode(Bookmark.self, from: Data($0.utf8))
            }
            if let limit = Self.value("limit").flatMap(Int.init) { source = Array(source.prefix(limit)) }
            topicCount = max(2, Self.value("topics").flatMap(Int.init) ?? 6)
            phase = .loading("Loading EmbeddingGemma 2 on the Neural Engine…")
            DemoLog.event("Sort by topic · \(source.count) posts · EmbeddingGemma 2 on the Neural Engine")
            let loadStart = DispatchTime.now().uptimeNanoseconds
            // The first load of a new model copy compiles all functions for the Neural Engine (minutes, once);
            // a ticking status keeps that from looking like a hang.
            let ticker = Task { @MainActor [weak self] in
                var seconds = 0
                while !Task.isCancelled {
                    try? await Task.sleep(for: .seconds(1))
                    seconds += 1
                    guard let self, case .loading = self.phase else { continue }
                    self.phase = .loading(
                        "Loading on the Neural Engine… \(seconds) s"
                            + (seconds >= 8 ? " (first launch compiles the model once, ~2 min)" : ""))
                }
            }
            defer { ticker.cancel() }
            let manager = try await EmbeddingGemma2Manager.loadDefault { file, bytes in
                if bytes == 0 { DemoLog.model("fetching \(file) from Hugging Face…") }
            }
            self.manager = manager
            _ = try await manager.embed("warm up", prompt: .clustering)
            DemoLog.model(
                String(
                    format: "model loaded in %.1f s · 7 functions (embed_32…512, pack_256 = 8 posts per call)",
                    Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1e9))
            phase = .ready
            lastEvent = "\(source.count) posts queued. Press Start."
            if CommandLine.arguments.contains("--autostart") { start() }
        } catch {
            phase = .failed(error.localizedDescription)
            DemoLog.line("failed: \(error.localizedDescription)", color: 196, bold: true)
        }
    }

    /// Start from the beginning, or resume after Pause.
    func start() {
        if phase == .paused {
            pauseRequested = false
            phase = .streaming
            DemoLog.event("▶ resume at \(posts.count) posts")
            return
        }
        guard phase == .ready else { return }
        DemoLog.event("▶ start: streaming \(source.count) posts")
        pauseRequested = false
        runTask = Task { await run() }
    }

    func pause() {
        guard isRunning else { return }
        pauseRequested = true
        DemoLog.event("❚❚ pause at \(posts.count) posts")
    }

    /// Clears every sorted post and topic; Start replays the stream from the first post.
    func reset() {
        DemoLog.event("↺ reset")
        generation += 1
        runTask?.cancel()
        runTask = nil
        pauseRequested = false
        posts = []
        vectors = []
        topics = []
        paths = [:]
        postsPerSecond = 0
        activeSeconds = 0
        selection = Self.allID
        guard manager != nil else { return }
        phase = .ready
        lastEvent = "\(source.count) posts queued. Press Start."
    }

    private func run() async {
        guard let manager else { return }
        phase = .streaming
        let rate = Self.value("rate").flatMap(Double.init) ?? 0
        var nextResort = Self.resortPoints.makeIterator()
        var resortAt = nextResort.next()
        do {
            while posts.count < source.count {
                if pauseRequested {
                    phase = .paused
                    lastEvent = String(format: "Paused at %d posts", posts.count)
                    while pauseRequested { try await Task.sleep(for: .milliseconds(50)) }
                    phase = .streaming
                }
                try Task.checkCancellation()
                let tick = DispatchTime.now().uptimeNanoseconds
                let batch = Array(source[posts.count..<min(posts.count + Self.batchSize, source.count)])
                let embedded = try await Self.embed(batch, manager: manager)
                let embedMilliseconds = Double(DispatchTime.now().uptimeNanoseconds - tick) / 1e6
                try Task.checkCancellation()
                // Work on copies and publish once per batch; mutating @Published storage per post copies it each time.
                var newPosts = posts
                var newTopics = topics
                var newPaths = paths
                for (post, vector) in zip(batch, embedded) {
                    let index = newPosts.count
                    vectors.append(vector)
                    newPosts.append(post)
                    if !newTopics.isEmpty {
                        newPaths[index] = TopicDiscovery.assign(index, vector: vector, into: &newTopics).map(\.id)
                    }
                }
                posts = newPosts
                topics = newTopics
                paths = newPaths
                activeSeconds += Double(DispatchTime.now().uptimeNanoseconds - tick) / 1e9
                postsPerSecond = Double(posts.count) / max(activeSeconds, 1e-6)
                logBatch(batch.count, milliseconds: embedMilliseconds)
                if let point = resortAt, posts.count >= point {
                    try await resort()
                    resortAt = nextResort.next()
                }
                if rate > 0 { try await Task.sleep(for: .seconds(Double(batch.count) / rate)) }
            }
            try await resort()
            phase = .done
            lastEvent = String(
                format: "%d posts sorted at %.0f posts/s. Pick a topic and press Split.", posts.count, postsPerSecond)
            DemoLog.event(
                String(format: "streamed %d posts at %.0f posts/s (embed + file)", posts.count, postsPerSecond))
            printTree()
            if CommandLine.arguments.contains("--auto-split"),
                let largest = topics.max(by: { $0.members.count < $1.members.count })
            {
                try await Task.sleep(for: .seconds(1))
                selection = largest.id
                await split(largest.id)
            }
            if CommandLine.arguments.contains("--split-all") {
                for topic in topics where topic.children.isEmpty { await split(topic.id) }
            }
            if let dump = Self.value("dump") {
                // One line per post: id, broad topic, subtopic (if split), for scoring against labelled data.
                let lines = posts.indices.map { index -> String in
                    let fields = [posts[index].id] + path(index).map(\.name)
                    return String(decoding: try! JSONSerialization.data(withJSONObject: fields), as: UTF8.self)
                }
                try lines.joined(separator: "\n").write(toFile: dump, atomically: true, encoding: .utf8)
            }
            if let snapshot = Self.value("snapshot") { try SnapshotRenderer.write(model: self, to: snapshot) }
        } catch is CancellationError {
        } catch {
            phase = .failed(error.localizedDescription)
            DemoLog.line("failed: \(error.localizedDescription)", color: 196, bold: true)
        }
    }

    /// Embeds a batch (packed eight to a Neural Engine call), results in input order.
    private nonisolated static func embed(
        _ batch: [Bookmark], manager: EmbeddingGemma2Manager
    ) async throws
        -> [[Float]]
    {
        try await manager.embed(batch.map(\.classificationText), prompt: .clustering)
    }

    /// Re-finds the broad topics from every post so far, keeping colours where topics carry over.
    private func resort() async throws {
        guard let manager else { return }
        phase = .resorting
        let start = DispatchTime.now().uptimeNanoseconds
        var fresh = try await TopicDiscovery().discoverBroad(
            vectors: vectors, texts: posts.map(\.classificationText), topicCount: topicCount, cache: cache,
            manager: manager)
        try Task.checkCancellation()
        TopicDiscovery.carryColors(from: topics, to: &fresh, paletteSize: Self.palette.count)
        withAnimation(.easeInOut(duration: 0.3)) {
            topics = fresh
            rebuildPaths()
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        lastEvent = String(
            format: "Re-sorted %d posts into %d broad topics in %.2f s", posts.count, topics.count, seconds)
        DemoLog.line("↻ " + lastEvent, color: 220)
        for topic in topics { DemoLog.line("    \(DemoLog.topic(topic))  \(topic.members.count)") }
        phase = .streaming
    }

    /// Sorts one broad topic into specific subtopics.
    func split(_ id: String) async {
        guard let manager, let index = topics.firstIndex(where: { $0.id == id }), topics[index].children.isEmpty
        else { return }
        let topic = topics[index]
        let previous = phase
        let started = generation
        phase = .splitting(topic.name)
        let start = DispatchTime.now().uptimeNanoseconds
        do {
            let children = try await TopicDiscovery().split(
                topic, vectors: vectors, texts: posts.map(\.classificationText), phraseCache: cache
            ) { phrase in try await manager.embed(phrase) }
            let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            // Reset (or anything else) may have replaced the topics while this awaited.
            guard generation == started, let current = topics.firstIndex(where: { $0.id == topic.id }) else {
                if case .splitting = phase { phase = previous }
                return
            }
            withAnimation(.spring(response: 0.6, dampingFraction: 0.85)) {
                topics[current].children = children
                rebuildPaths()
            }
            lastEvent = String(
                format: "Split “%@” into %d subtopics in %.2f s", topic.name, children.count, seconds)
            DemoLog.line("✂ " + lastEvent, color: 220, bold: true)
            for child in children { DemoLog.line("    \(DemoLog.topic(child))  \(child.members.count)") }
        } catch {
            lastEvent = "Split failed: \(error.localizedDescription)"
        }
        if case .splitting = phase { phase = previous }
    }

    func merge(_ id: String) {
        guard let index = topics.firstIndex(where: { $0.id == id }) else { return }
        DemoLog.event("⤺ merged “\(topics[index].name)” back into one topic")
        withAnimation(.spring(response: 0.5, dampingFraction: 0.85)) {
            topics[index].children = []
            rebuildPaths()
        }
        if selection?.hasPrefix(id + ".") == true { selection = id }
    }

    func rename(_ id: String, to name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        for index in topics.indices {
            if topics[index].id == id { topics[index].name = trimmed }
            for child in topics[index].children.indices where topics[index].children[child].id == id {
                topics[index].children[child].name = trimmed
            }
        }
        rebuildPaths()
    }

    private func rebuildPaths() {
        var result: [Int: [String]] = [:]
        for topic in topics {
            for member in topic.members { result[member] = [topic.id] }
            for child in topic.children { for member in child.members { result[member] = [topic.id, child.id] } }
        }
        paths = result
    }

    private func printTree() {
        for topic in topics {
            DemoLog.line("[\(topic.members.count)] \(DemoLog.topic(topic))")
            for child in topic.children { DemoLog.line("    [\(child.members.count)] \(DemoLog.topic(child))") }
        }
    }

    /// One line per batch: Neural Engine time and rate, plus one of the batch's posts and where it went.
    private func logBatch(_ count: Int, milliseconds: Double) {
        let calls = (count + EmbeddingGemma2Manager.packSlots - 1) / EmbeddingGemma2Manager.packSlots
        DemoLog.model(
            String(
                format: "%d posts · ~%d calls · %.0f ms · %.0f posts/s · %d/%d", count, calls, milliseconds,
                Double(count) / max(milliseconds / 1000, 1e-6), posts.count, source.count))
        guard let last = posts.indices.last else { return }
        let text = posts[last].text.replacingOccurrences(of: "\n", with: " ")
        let snippet = text.count > 70 ? String(text.prefix(70)) + "…" : text
        let destination =
            topics.isEmpty ? "(topics after the first batch)" : path(last).map(DemoLog.topic).joined(separator: " › ")
        DemoLog.line("      “\(snippet)” → \(destination)")
    }
}

extension TopicDiscovery {
    /// Broad topics only (no subtopics), names from EmbeddingGemma 2.
    func discoverBroad(
        vectors: [[Float]], texts: [String], topicCount: Int, cache: PhraseVectorCache,
        manager: EmbeddingGemma2Manager
    ) async throws -> [TopicNode] {
        var broad = self
        broad.maxDepth = 1
        return try await broad.discover(
            vectors: vectors, texts: texts, topicCount: topicCount, phraseCache: cache
        ) { phrase in try await manager.embed(phrase) }
    }
}
