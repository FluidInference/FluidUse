import Accelerate
import AVFoundation
import FluidUse
import Foundation
import SwiftUI
import os

/// Indexes audio collections with EmbeddingGemma 2 (10 s windows: audio model on the GPU, text model on the Neural
/// Engine) and searches every window with a text query. Play runs the show once: index everything, listen to the
/// top three of a few queries, then run searches back to back as fast as they go. Pause holds it anywhere (timers
/// included); Replay starts over from an empty index.
@MainActor
final class AudioSearchModel: ObservableObject {
    enum Step: Equatable {
        case loading(String)
        case idle
        case indexing
        case listening
        case speed
        case finished
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

    @Published private(set) var step: Step = .loading("Starting…")
    @Published private(set) var paused = false
    @Published private(set) var collections: [Collection] = []
    @Published private(set) var windowsDone = 0
    @Published private(set) var windowsTotal = 0
    @Published private(set) var audioSeconds: Double = 0
    @Published private(set) var indexSeconds: Double = 0
    @Published private(set) var indexedWindows = 0
    @Published private(set) var results: [Result] = []
    @Published private(set) var queryMilliseconds: Double = 0
    @Published private(set) var playing: String?
    @Published private(set) var stepRemaining = 0
    @Published private(set) var burstQueries = 0
    @Published private(set) var burstPerSecond: Double = 0
    @Published private(set) var burstMilliseconds: Double = 0
    @Published var query = ""

    private var entries: [Entry] = []
    /// Every window's embedding, row after row, so a query is one matrix-vector product.
    private var matrix: [Float] = []
    private var text: EmbeddingGemma2Manager?
    private var audio: EmbeddingGemma2Audio?
    private var player: AVAudioPlayer?
    private var searchTask: Task<Void, Never>?
    private var showTask: Task<Void, Never>?
    /// The last query the show put in the field; the field's change callback echoing it is not typing.
    private var showQuery = ""
    /// Paused time so far, so timers and speeds count only running time.
    private var pausedNanoseconds: UInt64 = 0
    private var pauseBegan: UInt64?
    private let gate = PauseGate()
    /// Seconds of each result to play (`--clip=`) and length of the listen and speed steps (`--segment=`).
    private let clipSeconds = max(
        1, CommandLine.arguments.first { $0.hasPrefix("--clip=") }.flatMap { Double($0.dropFirst(7)) } ?? 3)
    private let segmentSeconds = max(
        5, CommandLine.arguments.first { $0.hasPrefix("--segment=") }.flatMap { Double($0.dropFirst(10)) } ?? 30)

    var realTimeFactor: Double { indexSeconds > 0 ? audioSeconds / indexSeconds : 0 }
    var allSuggestions: [String] { collections.flatMap(\.suggestions) }
    var isRunning: Bool { [.indexing, .listening, .speed].contains(step) }
    var canPlay: Bool { step == .idle || (isRunning && paused) }
    var canReplay: Bool { isRunning || step == .finished }

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
                    guard let self, case .loading = self.step else { continue }
                    self.step = .loading(
                        "Loading EmbeddingGemma 2… \(seconds) s"
                            + (seconds >= 8 ? " (first launch downloads and compiles the models once)" : ""))
                }
            }
            defer { ticker.cancel() }
            let text = try await EmbeddingGemma2Manager.loadDefault()
            let audio = try await EmbeddingGemma2Audio.load(text: text)
            _ = try await audio.embed(window: [Float](repeating: 0, count: 16_000))
            _ = try await text.embed("warm up", prompt: .searchQuery)
            // The speed step embeds queries eight per call (pack_256): load that path now too.
            _ = try await text.embed(Array(repeating: "warm up", count: 64), prompt: .searchQuery)
            self.text = text
            self.audio = audio
            DemoLog.model(
                String(
                    format: "models loaded in %.1f s · audio encoder on the GPU, text model on the Neural Engine",
                    Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1e9))
            step = .idle
            if CommandLine.arguments.contains("--autostart") { play() }
        } catch {
            step = .failed(error.localizedDescription)
            DemoLog.line("failed: \(error.localizedDescription)", color: 196, bold: true)
        }
    }

    // MARK: Play / Pause / Replay

    /// Starts the show, or resumes it after Pause.
    func play() {
        if isRunning, paused {
            if let began = pauseBegan { pausedNanoseconds += DispatchTime.now().uptimeNanoseconds - began }
            pauseBegan = nil
            paused = false
            gate.set(paused: false)
            player?.play()
            DemoLog.event("▶ resume")
            return
        }
        guard step == .idle else { return }
        showTask = Task { await show() }
    }

    func pause() {
        guard isRunning, !paused else { return }
        paused = true
        pauseBegan = DispatchTime.now().uptimeNanoseconds
        gate.set(paused: true)
        player?.pause()
        DemoLog.event("❚❚ pause")
    }

    /// Clears the index and runs the show again from the start.
    func replay() {
        guard canReplay else { return }
        showTask?.cancel()
        showTask = nil
        stop()
        paused = false
        pauseBegan = nil
        gate.set(paused: false)
        entries = []
        matrix = []
        indexedWindows = 0
        results = []
        query = ""
        showQuery = ""
        windowsDone = 0
        windowsTotal = 0
        audioSeconds = 0
        indexSeconds = 0
        burstQueries = 0
        burstPerSecond = 0
        burstMilliseconds = 0
        step = .idle
        DemoLog.event("↺ replay")
        showTask = Task { await show() }
    }

    /// Nanoseconds of running (unpaused) time on a monotonic clock.
    private var activeNow: UInt64 {
        let now = DispatchTime.now().uptimeNanoseconds
        return now - pausedNanoseconds - (pauseBegan.map { now - $0 } ?? 0)
    }

    private func waitWhilePaused() async {
        while paused, !Task.isCancelled { try? await Task.sleep(for: .milliseconds(50)) }
    }

    /// Index, listen, speed, done.
    private func show() async {
        DemoLog.event(
            "▶ play: index everything, top 3 of \(Self.showcase.count) example searches, \(Int(segmentSeconds)) s of searches as fast as they go"
        )
        await index()
        guard !Task.isCancelled, indexedWindows > 0 else { return }
        await listen()
        guard !Task.isCancelled else { return }
        stop()
        await speed()
        guard !Task.isCancelled else { return }
        step = .finished
        DemoLog.event(
            String(
                format: "■ done: %@ of audio indexed in %.1f s (%.0fx real time); %d searches at %.0f/s",
                Self.clock(audioSeconds), indexSeconds, realTimeFactor, burstQueries, burstPerSecond))
    }

    // MARK: Index

    private func index() async {
        guard let audio else { return }
        step = .indexing
        // Round-robin across collections.
        var queues = collections.map { collection in collection.files.map { (collection.id, $0) } }
        var ordered: [(Int, URL)] = []
        while queues.contains(where: { !$0.isEmpty }) {
            for index in queues.indices where !queues[index].isEmpty { ordered.append(queues[index].removeFirst()) }
        }
        let files = ordered
        DemoLog.event("📥 indexing \(files.count) files")
        do {
            let decodeStart = DispatchTime.now().uptimeNanoseconds
            let recordings = try await Task.detached(priority: .userInitiated) {
                try files.map { try EmbeddingGemma2Audio.samples(contentsOf: $0.1) }
            }.value
            DemoLog.line(
                String(
                    format: "decoded %d files to 16 kHz in %.1f s", files.count,
                    Double(DispatchTime.now().uptimeNanoseconds - decodeStart) / 1e9), color: 245)
            await waitWhilePaused()
            let total = recordings.reduce(0.0) { $0 + Double($1.count) } / Double(EmbeddingGemma2Audio.sampleRate)
            let start = activeNow
            let progress = IndexProgress()
            let inbox = WindowInbox()
            let gate = gate
            // The embedding tasks report into `progress` and `inbox`; the UI drains them ten times a second.
            let watcher = Task { @MainActor [weak self] in
                while !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(100))
                    guard let self else { return }
                    self.refresh(progress: progress, total: total, start: start)
                    self.add(inbox.drain())
                }
            }
            defer { watcher.cancel() }
            _ = try await audio.embed(
                recordings: recordings, progress: { done, count in progress.update(done: done, total: count) },
                onWindow: { recording, window in
                    inbox.append(
                        Entry(
                            collection: files[recording].0, file: files[recording].1, start: window.start,
                            duration: window.duration, embedding: window.embedding))
                },
                beforeWindow: { await gate.wait() })
            try Task.checkCancellation()
            refresh(progress: progress, total: total, start: start)
            add(inbox.drain())
            DemoLog.event(
                String(
                    format: "📥 indexed %@ of audio (%d windows) in %.1f s = %.0fx real time", Self.clock(total),
                    indexedWindows, indexSeconds, realTimeFactor))
        } catch is CancellationError {
        } catch {
            step = .failed(error.localizedDescription)
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
        indexSeconds = Double(activeNow - start) / 1e9
    }

    // MARK: Listen

    /// The listening step's examples: each is typed, then its top three play. Two phrases from the earnings call,
    /// six everyday sounds; every one returns three right hits.
    static let showcase = [
        "forward-looking statements disclaimer", "a baby crying", "church bells ringing",
        "the operator opens the line for questions", "a siren", "someone laughing", "a cow mooing", "birds chirping",
    ]

    static let earningsExamples: Set = [
        "forward-looking statements disclaimer", "the operator opens the line for questions",
    ]

    /// Types each showcase query (those whose collection is present), then plays its top three.
    private func listen() async {
        step = .listening
        // Earnings-call examples need that collection; sound examples need the sound clips.
        let names = Set(collections.map(\.name))
        var queries = Self.showcase.filter { query in
            Self.earningsExamples.contains(query) ? names.contains("Earnings calls") : names.contains("Sounds (ESC-50)")
        }
        if queries.isEmpty { queries = Array(allSuggestions.prefix(8)) }
        guard !queries.isEmpty else { return }
        for (number, query) in queries.enumerated() {
            guard !Task.isCancelled else { return }
            stepRemaining = queries.count - number
            showQuery = ""
            self.query = ""
            for character in query {
                await waitWhilePaused()
                showQuery.append(character)
                self.query.append(character)
                try? await Task.sleep(for: .milliseconds(30))
            }
            await waitWhilePaused()
            guard !Task.isCancelled, let ranked = await rank(query) else { return }
            results = ranked
            for (rank, result) in ranked.prefix(3).enumerated() {
                guard !Task.isCancelled else { return }
                DemoLog.line(
                    String(
                        format: "   ▶ #%d  %@  %@ @ %@", rank + 1, collections[result.entry.collection].name,
                        result.entry.file.lastPathComponent, Self.clock(result.entry.start)), color: 120)
                await playClip(result, seconds: min(clipSeconds, result.entry.duration))
            }
            await waitWhilePaused()
            try? await Task.sleep(for: .milliseconds(700))
        }
    }

    /// Plays `seconds` of a result's window, holding while paused; returns when it ends or is stopped.
    private func playClip(_ result: Result, seconds: TimeInterval) async {
        stop()
        guard let player = try? AVAudioPlayer(contentsOf: result.entry.file) else { return }
        player.currentTime = result.entry.start
        player.play()
        self.player = player
        playing = result.id
        var played: UInt64 = 0
        var last = DispatchTime.now().uptimeNanoseconds
        while !Task.isCancelled, playing == result.id, Double(played) / 1e9 < seconds {
            try? await Task.sleep(for: .milliseconds(50))
            let now = DispatchTime.now().uptimeNanoseconds
            if !paused { played += now - last }
            last = now
        }
        if playing == result.id { stop() }
    }

    // MARK: Speed

    /// Queries back to back for `segmentSeconds` of running time: 64 at a time, embedded eight per Neural Engine
    /// call, ranked against every window in one matrix multiply. The field shows the latest query of each batch.
    private func speed() async {
        guard let text, !entries.isEmpty else { return }
        step = .speed
        let pool = queryPool
        let dimension = EmbeddingGemma2Manager.dimension
        let windows = entries.count
        let batchSize = 64
        burstQueries = 0
        burstPerSecond = 0
        burstMilliseconds = 0
        DemoLog.event("⚡ search speed: \(pool.count) different queries, back to back, against \(windows) windows")
        let start = activeNow
        let length = UInt64(segmentSeconds * 1e9)
        let ticker = tick { [weak self] in
            guard let self else { return }
            self.stepRemaining = max(0, Int((Double(length) - Double(self.activeNow - start)) / 1e9))
        }
        defer { ticker.cancel() }
        var embedNanoseconds: UInt64 = 0
        var rankNanoseconds: UInt64 = 0
        var next = 0
        var lastLog = start
        var recent: [(time: UInt64, count: Int)] = [(start, 0)]
        while !Task.isCancelled, activeNow - start < length {
            if paused {
                await waitWhilePaused()
                recent = [(activeNow, 0)]
                continue
            }
            let batch = (0..<batchSize).map { pool[(next + $0) % pool.count] }
            next += batchSize
            let embedStart = DispatchTime.now().uptimeNanoseconds
            guard let vectors = try? await text.embed(batch, prompt: .searchQuery) else { return }
            let rankStart = DispatchTime.now().uptimeNanoseconds
            let queryMatrix = vectors.flatMap { $0 }
            var scores = [Float](repeating: 0, count: batchSize * windows)
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans, Int32(batchSize), Int32(windows), Int32(dimension), 1,
                queryMatrix, Int32(dimension), matrix, Int32(dimension), 0, &scores, Int32(windows))
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
            // Rate over the last second of running time, so the tiles show the current speed.
            let now = activeNow
            recent.append((now, batchSize))
            recent.removeAll { now - $0.time > 1_000_000_000 }
            let span = Double(now - (recent.first?.time ?? start)) / 1e9
            let recentQueries = recent.dropFirst().reduce(0) { $0 + $1.count }
            if span > 0.2, recentQueries > 0 {
                burstPerSecond = Double(recentQueries) / span
                burstMilliseconds = 1000 / burstPerSecond
            }
            showQuery = batch[batchSize - 1]
            query = showQuery
            results = shown
            queryMilliseconds = burstMilliseconds
            if now - lastLog > 1_000_000_000 {
                lastLog = now
                DemoLog.line(
                    String(
                        format:
                            "⚡ %d searches · %.0f/s · %.2f ms each (embed %.2f + rank %.2f) · %.1f M window scores/s",
                        burstQueries, burstPerSecond, burstMilliseconds,
                        Double(embedNanoseconds) / 1e6 / Double(burstQueries),
                        Double(rankNanoseconds) / 1e6 / Double(burstQueries),
                        burstPerSecond * Double(windows) / 1e6), color: 81)
            }
        }
        // The tiles keep the whole step's average.
        let elapsed = Double(activeNow - start) / 1e9
        if elapsed > 0, burstQueries > 0 {
            burstPerSecond = Double(burstQueries) / elapsed
            burstMilliseconds = 1000 / burstPerSecond
        }
    }

    private func tick(_ update: @escaping @MainActor () -> Void) -> Task<Void, Never> {
        Task { @MainActor in
            while !Task.isCancelled {
                update()
                try? await Task.sleep(for: .milliseconds(250))
            }
        }
    }

    // MARK: Search by hand (when the show is not running)

    /// Ranks every indexed window against the typed query (debounced).
    func search() {
        if query == showQuery || isRunning { return }
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
        return ranked
    }

    /// Row play button (when the show is not running): plays or stops one window.
    func play(_ result: Result) {
        guard !isRunning else { return }
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

    /// Everything the search-speed segment cycles through: the suggestions plus a spread of topics and sounds.
    var queryPool: [String] {
        var pool = Self.showcase + allSuggestions + Self.extraQueries
        // Sound clips have random file names; their classes come from labels.json next to the folder.
        var sounds = Set<String>()
        for collection in collections {
            guard let folder = collection.files.first?.deletingLastPathComponent(),
                let data = try? Data(
                    contentsOf: folder.deletingLastPathComponent().appendingPathComponent("labels.json")),
                let labels = try? JSONSerialization.jsonObject(with: data) as? [String: [String: String]]
            else { continue }
            for label in labels.values {
                if let category = label["category"] {
                    sounds.insert("the sound of " + category.replacingOccurrences(of: "_", with: " "))
                }
            }
        }
        pool += sounds.sorted()
        var seen = Set<String>()
        return pool.filter { seen.insert($0).inserted }
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

/// Holds embedding tasks while the show is paused.
final class PauseGate: Sendable {
    private let state = OSAllocatedUnfairLock(initialState: false)

    func set(paused: Bool) { state.withLock { $0 = paused } }

    func wait() async {
        while state.withLock({ $0 }), !Task.isCancelled { try? await Task.sleep(for: .milliseconds(50)) }
    }
}

/// Window counts from the embedding tasks, read by the UI on a timer.
final class IndexProgress: Sendable {
    private let counts = OSAllocatedUnfairLock(initialState: (done: 0, total: 0))

    func update(done: Int, total: Int) { counts.withLock { $0 = (done, total) } }

    var snapshot: (Int, Int) { counts.withLock { ($0.done, $0.total) } }
}
