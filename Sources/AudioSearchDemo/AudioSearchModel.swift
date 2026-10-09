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
    /// Hands-free show: alternating `--segment=` seconds (default 30) of speed (reading everything again) and search.
    enum Segment { case speed, search }
    @Published private(set) var segment: Segment?
    @Published private(set) var segmentRemaining = 0

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

    /// The hands-free show: a speed segment (read every file again as fast as possible; the very first one runs
    /// until the index is complete), then a search segment (type a query, play its top three, next query), each
    /// `segmentSeconds` long, round and round.
    private func startAutoPlay() {
        autoTask?.cancel()
        autoPlay = true
        DemoLog.event(
            "▶ hands-free: \(Int(segmentSeconds)) s reading as fast as it can, then \(Int(segmentSeconds)) s of searches"
        )
        autoTask = Task { [weak self] in
            var nextQuery = 0
            while let self, !Task.isCancelled {
                // ⚡ Speed
                self.segment = .speed
                self.stop()
                self.results = []
                self.autoQuery = ""
                self.query = ""
                let firstPass = self.indexedWindows == 0
                // Mark it now: the countdown below polls `isIndexing` before the task gets to run.
                self.isIndexing = true
                let run = Task { await self.index() }
                self.runTask = run
                if firstPass {
                    let ticker = self.countUp()
                    await run.value
                    ticker.cancel()
                } else {
                    await self.countdown(self.segmentSeconds) { !self.isIndexing && self.phase != .indexing }
                    if self.isIndexing { run.cancel() }
                    await run.value
                }
                guard !Task.isCancelled, self.indexedWindows > 0 else { return }
                // 🔎 Search
                self.segment = .search
                let deadline = ContinuousClock.now + .seconds(self.segmentSeconds)
                let ticker = Task { @MainActor [weak self] in
                    while !Task.isCancelled, let self {
                        self.segmentRemaining = max(
                            0, Int((deadline - ContinuousClock.now).components.seconds))
                        try? await Task.sleep(for: .milliseconds(250))
                    }
                }
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
            }
        }
    }

    /// Counts the speed segment's seconds up (the first pass runs to completion, however long that is).
    private func countUp() -> Task<Void, Never> {
        let start = ContinuousClock.now
        return Task { @MainActor [weak self] in
            while !Task.isCancelled, let self {
                self.segmentRemaining = Int((ContinuousClock.now - start).components.seconds)
                try? await Task.sleep(for: .milliseconds(250))
            }
        }
    }

    /// Waits `seconds` (or until `done()`), counting down `segmentRemaining`.
    private func countdown(_ seconds: Double, until done: () -> Bool) async {
        let deadline = ContinuousClock.now + .seconds(seconds)
        while !Task.isCancelled, ContinuousClock.now < deadline, !done() {
            segmentRemaining = max(0, Int((deadline - ContinuousClock.now).components.seconds))
            try? await Task.sleep(for: .milliseconds(250))
        }
    }

    private func stopAutoPlay() {
        guard autoPlay || autoTask != nil else { return }
        autoTask?.cancel()
        autoTask = nil
        autoTyping = false
        autoPlay = false
        segment = nil
        // A timed re-read stops with the show; the first pass finishes so the index is complete.
        if pass > 1, isIndexing { runTask?.cancel() }
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
