import Accelerate
import AVFoundation
import FluidUse
import Foundation
import SwiftUI
import os

/// Indexes audio collections with EmbeddingGemma 2 (10 s windows: audio model on the GPU, text model on the Neural
/// Engine) and searches every window with a text query.
@MainActor
final class AudioSearchModel: ObservableObject {
    enum Phase: Equatable {
        case loading(String)
        case ready
        case indexing
        case done
        case failed(String)
    }

    struct Collection: Identifiable {
        let id: Int
        let name: String
        let color: Color
        let files: [URL]
        let suggestions: [String]
    }

    struct Entry: Sendable {
        let collection: Int
        let file: URL
        let start: TimeInterval
        let duration: TimeInterval
        let embedding: [Float]
    }

    struct Result: Identifiable {
        let entry: Entry
        let score: Float
        var id: String { "\(entry.file.path)#\(entry.start)" }
    }

    @Published private(set) var phase: Phase = .loading("Starting…")
    @Published private(set) var collections: [Collection] = []
    @Published private(set) var windowsDone = 0
    @Published private(set) var windowsTotal = 0
    @Published private(set) var audioSeconds: Double = 0
    @Published private(set) var indexSeconds: Double = 0
    @Published private(set) var results: [Result] = []
    @Published private(set) var queryMilliseconds: Double = 0
    @Published private(set) var playing: String?
    @Published var query = ""
    /// Hands-free mode (default; `--manual` turns it off): index on launch, then type each suggested query and play
    /// its top three windows, round and round. Typing or pressing play yourself switches it off.
    @Published private(set) var autoPlay = !CommandLine.arguments.contains("--manual")
    /// Indexing passes so far. The first grows the index live; later ones rebuild it in the background while the
    /// current index keeps answering searches, to keep showing the indexing speed.
    @Published private(set) var pass = 0
    @Published private(set) var isIndexing = false
    @Published private(set) var indexedWindows = 0
    /// Hands-free show: read everything once (searchable as it grows), then alternate `--segment=` seconds (default
    /// 30) of listening (type a query, play its top three) and of search speed (queries back to back, as fast as the
    /// Neural Engine embeds them).
    enum Segment { case listen, burst }
    @Published private(set) var segment: Segment?
    @Published private(set) var segmentRemaining = 0
    /// Search-speed counters for the current burst.
    @Published private(set) var burstQueries = 0
    @Published private(set) var burstPerSecond: Double = 0
    @Published private(set) var burstMilliseconds: Double = 0

    private var entries: [Entry] = []
    /// Every window's embedding, row after row, so a query is one matrix-vector product.
    private var matrix: [Float] = []
    private var text: EmbeddingGemma2Manager?
    private var audio: EmbeddingGemma2Audio?
    private var player: AVAudioPlayer?
    private var searchTask: Task<Void, Never>?
    private var runTask: Task<Void, Never>?
    private var autoTask: Task<Void, Never>?
    private var autoTyping = false
    /// The last query text hands-free mode put in the field; the field's change callback echoing it is not typing.
    private var autoQuery = ""
    /// Seconds of each result to play in hands-free mode (`--clip=`).
    private let clipSeconds = max(
        1, CommandLine.arguments.first { $0.hasPrefix("--clip=") }.flatMap { Double($0.dropFirst(7)) } ?? 4)
    private let segmentSeconds = max(
        5, CommandLine.arguments.first { $0.hasPrefix("--segment=") }.flatMap { Double($0.dropFirst(10)) } ?? 30)

    var realTimeFactor: Double { indexSeconds > 0 ? audioSeconds / indexSeconds : 0 }
    var allSuggestions: [String] { collections.flatMap(\.suggestions) }

    func prepare() async {
        do {
            collections = Self.defaultCollections()
            guard !collections.isEmpty else {
                throw EmbeddingGemma2Error.invalidAsset("No audio found: pass --audio=<file or folder>")
            }
            DemoLog.event(
                "Search audio · \(collections.map { "\($0.name) (\($0.files.count))" }.joined(separator: " · "))")
            let loadStart = DispatchTime.now().uptimeNanoseconds
            let ticker = Task { @MainActor [weak self] in
                var seconds = 0
                while !Task.isCancelled {
                    try? await Task.sleep(for: .seconds(1))
                    seconds += 1
                    guard let self, case .loading = self.phase else { continue }
                    self.phase = .loading(
                        "Loading EmbeddingGemma 2… \(seconds) s"
                            + (seconds >= 8 ? " (first launch downloads and compiles the models once)" : ""))
                }
            }
            defer { ticker.cancel() }
            let text = try await EmbeddingGemma2Manager.loadDefault()
            let audio = try await EmbeddingGemma2Audio.load(text: text)
            _ = try await audio.embed(window: [Float](repeating: 0, count: 16_000))
            _ = try await text.embed("warm up", prompt: .searchQuery)
            // The search-speed segment embeds queries eight per call (pack_256): load that path now too.
            _ = try await text.embed(Array(repeating: "warm up", count: 64), prompt: .searchQuery)
            self.text = text
            self.audio = audio
            DemoLog.model(
                String(
                    format: "models loaded in %.1f s · audio encoder on the GPU, text model on the Neural Engine",
                    Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1e9))
            phase = .ready
            if autoPlay {
                startAutoPlay()
            } else if CommandLine.arguments.contains("--autostart") {
                start()
            }
        } catch {
            phase = .failed(error.localizedDescription)
            DemoLog.line("failed: \(error.localizedDescription)", color: 196, bold: true)
        }
    }

    func start() {
        guard phase == .ready || phase == .done, !isIndexing else { return }
        runTask = Task { await index() }
    }

    func reset() {
        autoTask?.cancel()
        autoTask = nil
        runTask?.cancel()
        runTask = nil
        stop()
        entries = []
        matrix = []
        indexedWindows = 0
        pass = 0
        isIndexing = false
        results = []
        windowsDone = 0
        windowsTotal = 0
        audioSeconds = 0
        indexSeconds = 0
        if audio != nil { phase = .ready }
        DemoLog.event("↺ reset")
    }

    /// One indexing pass over every collection. Pass 1 adds windows to the live index as they finish (searchable at
    /// once); later passes build a new index beside the current one and swap it in at the end.
    private func index() async {
        guard let audio else { return }
        pass += 1
        isIndexing = true
        phase = .indexing
        let live = pass == 1
        // Round-robin across collections, so a growing index covers every collection early.
        var queues = collections.map { collection in collection.files.map { (collection.id, $0) } }
        var ordered: [(Int, URL)] = []
        while queues.contains(where: { !$0.isEmpty }) {
            for index in queues.indices where !queues[index].isEmpty { ordered.append(queues[index].removeFirst()) }
        }
        let files = ordered
        DemoLog.event(
            "▶ pass \(pass): reading \(files.count) files" + (live ? "" : " (searches keep using pass \(pass - 1))"))
        do {
            let decodeStart = DispatchTime.now().uptimeNanoseconds
            let recordings = try await Task.detached(priority: .userInitiated) {
                try files.map { try EmbeddingGemma2Audio.samples(contentsOf: $0.1) }
            }.value
            DemoLog.line(
                String(
                    format: "decoded %d files to 16 kHz in %.1f s", files.count,
                    Double(DispatchTime.now().uptimeNanoseconds - decodeStart) / 1e9), color: 245)
            let total = recordings.reduce(0.0) { $0 + Double($1.count) } / Double(EmbeddingGemma2Audio.sampleRate)
            let start = DispatchTime.now().uptimeNanoseconds
            let progress = IndexProgress()
            let inbox = WindowInbox()
            var staged: [Entry] = []
            // The embedding tasks report into `progress` and `inbox`; the UI drains them ten times a second.
            let watcher = Task { @MainActor [weak self] in
                while !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(100))
                    guard let self else { return }
                    self.refresh(progress: progress, total: total, start: start)
                    let fresh = inbox.drain()
                    if live { self.add(fresh) } else { staged += fresh }
                }
            }
            _ = try await audio.embed(
                recordings: recordings, progress: { done, count in progress.update(done: done, total: count) },
                onWindow: { recording, window in
                    inbox.append(
                        Entry(
                            collection: files[recording].0, file: files[recording].1, start: window.start,
                            duration: window.duration, embedding: window.embedding))
                })
            watcher.cancel()
            try Task.checkCancellation()
            refresh(progress: progress, total: total, start: start)
            if live {
                add(inbox.drain())
            } else {
                staged += inbox.drain()
                entries = staged
                matrix = staged.flatMap(\.embedding)
                indexedWindows = staged.count
            }
            isIndexing = false
            phase = .done
            DemoLog.event(
                String(
                    format: "pass %d: indexed %@ of audio (%d windows) in %.1f s = %.0fx real time", pass,
                    Self.clock(total), indexedWindows, indexSeconds, realTimeFactor))
            if let preset = CommandLine.arguments.first(where: { $0.hasPrefix("--query=") }), pass == 1 {
                query = String(preset.dropFirst(8))
            }
            if !autoPlay, !query.isEmpty, pass == 1 { search() }
        } catch is CancellationError {
            // Cut short (a timed speed run, or Reset): the previous index stays in place.
            isIndexing = false
            if phase == .indexing { phase = indexedWindows > 0 ? .done : .ready }
            DemoLog.event(
                String(
                    format: "pass %d: read %@ of audio in %.1f s = %.0fx real time (stopped there)", pass,
                    Self.clock(audioSeconds), indexSeconds, realTimeFactor))
        } catch {
            isIndexing = false
            phase = .failed(error.localizedDescription)
            DemoLog.line("failed: \(error.localizedDescription)", color: 196, bold: true)
        }
    }

    private func add(_ fresh: [Entry]) {
        guard !fresh.isEmpty else { return }
        entries += fresh
        matrix += fresh.flatMap(\.embedding)
        indexedWindows = entries.count
    }

    private func refresh(progress: IndexProgress, total: Double, start: UInt64) {
        let (done, count) = progress.snapshot
        guard count > 0 else { return }
        windowsDone = done
        windowsTotal = count
        audioSeconds = total * Double(done) / Double(count)
        indexSeconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
    }

    /// Ranks every indexed window against the query (debounced while typing). Typing by hand ends hands-free mode.
    func search() {
        if autoTyping || query == autoQuery { return }
        stopAutoPlay()
        searchTask?.cancel()
        let query = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty, !entries.isEmpty else {
            results = []
            return
        }
        searchTask = Task {
            try? await Task.sleep(for: .milliseconds(120))
            guard !Task.isCancelled, let ranked = await rank(query), !Task.isCancelled else { return }
            results = ranked
        }
    }

    /// Top 25 windows for `query`: one text embedding, then one matrix-vector product over every window.
    private func rank(_ query: String) async -> [Result]? {
        guard let text else { return nil }
        let begin = DispatchTime.now().uptimeNanoseconds
        guard let vector = try? await text.embed(query, prompt: .searchQuery) else { return nil }
        var scores = [Float](repeating: 0, count: entries.count)
        cblas_sgemv(
            CblasRowMajor, CblasNoTrans, Int32(entries.count), Int32(vector.count), 1, matrix, Int32(vector.count),
            vector, 1, 0, &scores, 1)
        let ranked = scores.indices.sorted { scores[$0] > scores[$1] }.prefix(25).map {
            Result(entry: entries[$0], score: scores[$0])
        }
        queryMilliseconds = Double(DispatchTime.now().uptimeNanoseconds - begin) / 1e6
        DemoLog.line(
            String(format: "🔎 “%@” → %d windows ranked in %.0f ms", query, entries.count, queryMilliseconds), color: 81)
        for result in ranked.prefix(3) {
            DemoLog.line(
                String(
                    format: "      %.3f  %@  %@ @ %@", result.score, collections[result.entry.collection].name,
                    result.entry.file.lastPathComponent, Self.clock(result.entry.start)))
        }
        return ranked
    }

    func toggleAutoPlay() {
        if autoPlay { stopAutoPlay() } else if !entries.isEmpty { startAutoPlay() }
    }

    /// The hands-free show: the first read (searchable as it grows), then listen / search-speed segments, each
    /// `segmentSeconds` long, round and round.
    private func startAutoPlay() {
        autoTask?.cancel()
        autoPlay = true
        DemoLog.event(
            "▶ hands-free: \(Int(segmentSeconds)) s listening to the top 3, then \(Int(segmentSeconds)) s of searches as fast as it can"
        )
        autoTask = Task { [weak self] in
            var nextQuery = 0
            // Indexing comes first, on its own; the show starts once every file is searchable.
            if let self, self.indexedWindows == 0 || self.isIndexing {
                if !self.isIndexing, self.indexedWindows == 0 {
                    self.isIndexing = true
                    self.runTask = Task { await self.index() }
                }
                while !Task.isCancelled, self.isIndexing { try? await Task.sleep(for: .milliseconds(100)) }
            }
            while let self, !Task.isCancelled, self.indexedWindows > 0 {
                // 🔎 Listen
                self.segment = .listen
                let deadline = ContinuousClock.now + .seconds(self.segmentSeconds)
                let ticker = self.countDown(to: deadline)
                let queries = self.allSuggestions
                while !Task.isCancelled, !queries.isEmpty, ContinuousClock.now < deadline {
                    let query = queries[nextQuery % queries.count]
                    nextQuery += 1
                    // Type it like a person would, then search.
                    self.autoTyping = true
                    self.autoQuery = ""
                    self.query = ""
                    for character in query {
                        self.autoQuery.append(character)
                        self.query.append(character)
                        try? await Task.sleep(for: .milliseconds(30))
                    }
                    self.autoTyping = false
                    guard !Task.isCancelled, let ranked = await self.rank(query) else { break }
                    self.results = ranked
                    for (rank, result) in ranked.prefix(3).enumerated() {
                        let left = (deadline - ContinuousClock.now).components.seconds
                        guard !Task.isCancelled, left > 0 else { break }
                        DemoLog.line(
                            String(
                                format: "   ▶ #%d  %@  %@ @ %@", rank + 1,
                                self.collections[result.entry.collection].name,
                                result.entry.file.lastPathComponent, Self.clock(result.entry.start)), color: 120)
                        await self.playClip(
                            result, seconds: min(self.clipSeconds, result.entry.duration, Double(left)))
                    }
                    try? await Task.sleep(for: .milliseconds(800))
                }
                ticker.cancel()
                self.stop()
                guard !Task.isCancelled else { return }
                // ⚡ Search speed
                self.segment = .burst
                let burstEnd = ContinuousClock.now + .seconds(self.segmentSeconds)
                let burstTicker = self.countDown(to: burstEnd)
                await self.burst(until: burstEnd)
                burstTicker.cancel()
            }
        }
    }

    /// Queries back to back until `deadline`: 64 at a time, embedded eight per Neural Engine call, then ranked
    /// against every window in one matrix multiply. The field and the results show the latest query of each batch.
    private func burst(until deadline: ContinuousClock.Instant) async {
        guard let text, !entries.isEmpty else { return }
        let pool = queryPool
        let dimension = EmbeddingGemma2Manager.dimension
        let batchSize = 64
        burstQueries = 0
        burstPerSecond = 0
        burstMilliseconds = 0
        DemoLog.event(
            "⚡ search speed: \(pool.count) different queries, back to back, against \(entries.count)+ windows")
        let start = DispatchTime.now().uptimeNanoseconds
        var embedNanoseconds: UInt64 = 0
        var rankNanoseconds: UInt64 = 0
        var next = 0
        var lastLog = start
        var recent: [(time: UInt64, count: Int)] = [(start, 0)]
        while !Task.isCancelled, ContinuousClock.now < deadline {
            let batch = (0..<batchSize).map { pool[(next + $0) % pool.count] }
            next += batchSize
            let embedStart = DispatchTime.now().uptimeNanoseconds
            guard let vectors = try? await text.embed(batch, prompt: .searchQuery) else { return }
            let rankStart = DispatchTime.now().uptimeNanoseconds
            // The index may still be growing: rank against this moment's windows.
            let snapshot = entries
            let index = matrix
            let windows = snapshot.count
            let queryMatrix = vectors.flatMap { $0 }
            var scores = [Float](repeating: 0, count: batchSize * windows)
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans, Int32(batchSize), Int32(windows), Int32(dimension), 1,
                queryMatrix, Int32(dimension), index, Int32(dimension), 0, &scores, Int32(windows))
            // Every query gets its own top 10, as a real search would; only the last one is shown.
            var shown: [Result] = []
            for row in 0..<batchSize {
                let top = Self.topIndices(scores, row: row, width: windows, count: 10)
                if row == batchSize - 1 {
                    shown = top.map { Result(entry: entries[$0], score: scores[row * windows + $0]) }
                }
            }
            let end = DispatchTime.now().uptimeNanoseconds
            embedNanoseconds += rankStart - embedStart
            rankNanoseconds += end - rankStart
            burstQueries += batchSize
            // Rate over the last second, so the tiles show the current speed.
            recent.append((end, batchSize))
            recent.removeAll { end - $0.time > 1_000_000_000 }
            let window = Double(end - (recent.first.map { $0.time } ?? start)) / 1e9
            let recentQueries = recent.dropFirst().reduce(0) { $0 + $1.count }
            if window > 0.2, recentQueries > 0 {
                burstPerSecond = Double(recentQueries) / window
                burstMilliseconds = 1000 / burstPerSecond
            }
            autoQuery = batch[batchSize - 1]
            query = autoQuery
            results = shown
            queryMilliseconds = burstMilliseconds
            if end - lastLog > 1_000_000_000 {
                lastLog = end
                DemoLog.line(
                    String(
                        format:
                            "⚡ %d queries · %.0f/s · %.2f ms each (embed %.2f + rank %.2f) · %.1f M window scores/s",
                        burstQueries, burstPerSecond, burstMilliseconds,
                        Double(embedNanoseconds) / 1e6 / Double(burstQueries),
                        Double(rankNanoseconds) / 1e6 / Double(burstQueries),
                        burstPerSecond * Double(entries.count) / 1e6), color: 81)
            }
        }
        let total = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        DemoLog.event(
            String(
                format: "⚡ %d searches in %.0f s = %.0f/s on average, %.0f/s at the end", burstQueries, total,
                Double(burstQueries) / total, burstPerSecond))
    }

    /// Indices of the `count` highest scores in one row of a row-major score matrix, best first.
    static func topIndices(_ scores: [Float], row: Int, width: Int, count: Int) -> [Int] {
        var best: [(index: Int, score: Float)] = []
        best.reserveCapacity(count + 1)
        for column in 0..<width {
            let score = scores[row * width + column]
            if best.count < count || score > best[best.count - 1].score {
                let position = best.firstIndex { score > $0.score } ?? best.count
                best.insert((column, score), at: position)
                if best.count > count { best.removeLast() }
            }
        }
        return best.map(\.index)
    }

    private func countDown(to deadline: ContinuousClock.Instant) -> Task<Void, Never> {
        Task { @MainActor [weak self] in
            while !Task.isCancelled, let self {
                self.segmentRemaining = max(0, Int((deadline - ContinuousClock.now).components.seconds))
                try? await Task.sleep(for: .milliseconds(250))
            }
        }
    }

    /// Everything the search-speed segment cycles through: the suggestions plus a spread of topics and sounds.
    var queryPool: [String] {
        var pool = allSuggestions + Self.extraQueries
        let sounds = Set(
            collections.flatMap(\.files).compactMap { file -> String? in
                let name = file.deletingPathExtension().lastPathComponent
                guard let range = name.range(of: "__") else { return nil }
                return "the sound of " + name[..<range.lowerBound].replacingOccurrences(of: "_", with: " ")
            })
        pool += sounds.sorted()
        return pool
    }

    static let extraQueries = [
        "revenue growth this quarter", "guidance for next year", "operating margins improved",
        "cash flow and dividends",
        "share buyback program", "supply chain problems", "demand from customers", "pricing pressure",
        "new product launch", "hiring and headcount", "interest rates and inflation", "currency headwinds",
        "an acquisition of a company", "questions from analysts", "thank you for joining the call",
        "the chief executive officer speaks", "regulation and government policy", "research and development spending",
        "the weather and climate", "a scientific discovery", "a sports team won the game", "a famous city in Europe",
        "the history of a war", "a recipe for dinner", "travelling by train", "music and concerts",
        "a disease and its treatment", "planets and space", "an election and voters", "children at school",
        "a river and a mountain", "computers and the internet", "money and banks", "an old church",
        "farming and crops", "the ocean and fish", "a museum exhibition", "an earthquake", "a new law was passed",
        "a festival with fireworks",
    ]

    private func stopAutoPlay() {
        guard autoPlay || autoTask != nil else { return }
        autoTask?.cancel()
        autoTask = nil
        autoTyping = false
        autoPlay = false
        segment = nil
        stop()
        DemoLog.event("❚❚ hands-free off")
    }

    /// Plays `seconds` of a result's window and returns when it ends (or is stopped).
    private func playClip(_ result: Result, seconds: TimeInterval) async {
        stop()
        guard let player = try? AVAudioPlayer(contentsOf: result.entry.file) else { return }
        player.currentTime = result.entry.start
        player.play()
        self.player = player
        playing = result.id
        try? await Task.sleep(for: .seconds(seconds))
        if playing == result.id { stop() }
    }

    /// Play button: plays (or stops) one window; ends hands-free mode.
    func play(_ result: Result) {
        stopAutoPlay()
        if playing == result.id {
            stop()
            return
        }
        Task { await playClip(result, seconds: result.entry.duration) }
    }

    func stop() {
        player?.stop()
        player = nil
        playing = nil
    }

    static func clock(_ seconds: TimeInterval) -> String {
        let total = Int(seconds.rounded())
        return total >= 3600
            ? String(format: "%d:%02d:%02d", total / 3600, total / 60 % 60, total % 60)
            : String(format: "%d:%02d", total / 60, total % 60)
    }

    /// `--audio=<file or folder>` (repeatable), else the public datasets already on this Mac.
    static func defaultCollections() -> [Collection] {
        let palette: [Color] = [.blue, .orange, .green, .pink, .purple, .teal]
        let given = CommandLine.arguments.filter { $0.hasPrefix("--audio=") }.map { String($0.dropFirst(8)) }
        if !given.isEmpty {
            return given.enumerated().compactMap { index, path in
                let files = audioFiles(at: URL(fileURLWithPath: path))
                guard !files.isEmpty else { return nil }
                return Collection(
                    id: index, name: URL(fileURLWithPath: path).lastPathComponent,
                    color: palette[index % palette.count], files: files, suggestions: [])
            }
        }
        let fluidAudio = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FluidAudio")
        let fluidUse = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FluidUse/Datasets")
        let candidates: [(String, URL, [String])] = [
            (
                "Earnings calls", fluidAudio.appendingPathComponent("earnings22-1h"),
                ["forward-looking statements disclaimer", "the operator opens the line for questions"]
            ),
            (
                "FLEURS English", fluidAudio.appendingPathComponent("FLEURS/en_us"),
                ["the history of an ancient empire", "animals in the wild"]
            ),
            ("FLEURS French", fluidAudio.appendingPathComponent("FLEURS/fr_fr"), ["a story about a famous painter"]),
            (
                "Sounds (ESC-50)", fluidUse.appendingPathComponent("esc50/wav"),
                ["a baby crying", "someone laughing", "a dog barking", "church bells"]
            ),
        ]
        var collections: [Collection] = []
        for candidate in candidates {
            let files = audioFiles(at: candidate.1)
            guard !files.isEmpty else { continue }
            let index = collections.count
            collections.append(
                Collection(
                    id: index, name: candidate.0, color: palette[index % palette.count], files: files,
                    suggestions: candidate.2))
        }
        return collections
    }

    static func audioFiles(at url: URL) -> [URL] {
        let extensions: Set<String> = ["wav", "flac", "mp3", "m4a", "aiff", "caf"]
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else { return [] }
        guard isDirectory.boolValue else { return [url] }
        let names = (try? FileManager.default.contentsOfDirectory(atPath: url.path)) ?? []
        return names.filter { extensions.contains(($0 as NSString).pathExtension.lowercased()) }.sorted()
            .map { url.appendingPathComponent($0) }
    }
}

/// Windows finished by the embedding tasks, waiting for the UI to add them to the index.
final class WindowInbox: Sendable {
    private let pending = OSAllocatedUnfairLock(initialState: [AudioSearchModel.Entry]())

    func append(_ entry: AudioSearchModel.Entry) { pending.withLock { $0.append(entry) } }

    func drain() -> [AudioSearchModel.Entry] {
        pending.withLock { entries in
            defer { entries = [] }
            return entries
        }
    }
}

/// Window counts from the embedding tasks, read by the UI on a timer.
final class IndexProgress: Sendable {
    private let counts = OSAllocatedUnfairLock(initialState: (done: 0, total: 0))

    func update(done: Int, total: Int) { counts.withLock { $0 = (done, total) } }

    var snapshot: (Int, Int) { counts.withLock { ($0.done, $0.total) } }
}
