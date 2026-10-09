import Accelerate
import CodeSearch
import FluidUse
import Foundation
import SwiftUI

/// Indexes a repository's Swift declarations with EmbeddingGemma 2 (Neural Engine) and searches them in plain English.
/// Play runs the show once: index, eight example questions with their top three, then searches back to back as fast
/// as they go. Pause holds it anywhere (timers included); Replay starts over from an empty index.
@MainActor
final class CodeSearchModel: ObservableObject {
    enum Step: Equatable {
        case loading(String)
        case idle
        case indexing
        case examples
        case speed
        case finished
        case failed(String)
    }

    /// A line in the live list while indexing (a function just indexed) or during search speed (question → answer).
    struct FeedItem: Identifiable {
        let id: Int
        let question: String?
        let chunk: CodeChunk
    }

    struct Result: Identifiable {
        let chunk: CodeChunk
        let score: Float
        var id: String { "\(chunk.path):\(chunk.line)" }
    }

    @Published private(set) var step: Step = .loading("Starting…")
    @Published private(set) var paused = false
    @Published private(set) var repositoryName = ""
    @Published private(set) var fileCount = 0
    @Published private(set) var lineCount = 0
    @Published private(set) var chunksTotal = 0
    @Published private(set) var chunksDone = 0
    @Published private(set) var indexSeconds: Double = 0
    @Published private(set) var results: [Result] = []
    @Published private(set) var selected: Result?
    @Published private(set) var grepMatches: Int?
    @Published private(set) var queryMilliseconds: Double = 0
    @Published private(set) var exampleNumber = 0
    @Published private(set) var stepRemaining = 0
    @Published private(set) var burstQueries = 0
    @Published private(set) var burstPerSecond: Double = 0
    @Published private(set) var burstMilliseconds: Double = 0
    @Published private(set) var feed: [FeedItem] = []
    @Published var query = ""

    private var repository = URL(fileURLWithPath: "/")
    private var chunks: [CodeChunk] = []
    private var indexed: [CodeChunk] = []
    /// Every indexed chunk's embedding, row after row, so a query is one matrix-vector product.
    private var matrix: [Float] = []
    private var text: EmbeddingGemma2Manager?
    private var showTask: Task<Void, Never>?
    private var searchTask: Task<Void, Never>?
    private var showQuery = ""
    private var snippets: [String: String] = [:]
    /// Every indexed file's text, lowercased, for the exact-phrase grep comparison.
    private var sources: [String: String] = [:]
    private var feedSerial = 0
    private var pausedNanoseconds: UInt64 = 0
    private var pauseBegan: UInt64?
    /// Tokens kept per chunk (title + doc comment + the start of the code): 128 indexed FluidAudio in ~28 s with the
    /// best top-1 on our question set.
    private let maxTokens =
        CommandLine.arguments.first { $0.hasPrefix("--tokens=") }.flatMap { Int($0.dropFirst(9)) } ?? 64
    private let speedSeconds = max(
        5, CommandLine.arguments.first { $0.hasPrefix("--segment=") }.flatMap { Double($0.dropFirst(10)) } ?? 30)
    private let dwellSeconds = max(
        0.5, CommandLine.arguments.first { $0.hasPrefix("--dwell=") }.flatMap { Double($0.dropFirst(8)) } ?? 1.6)

    var chunksPerSecond: Double { indexSeconds > 0 ? Double(chunksDone) / indexSeconds : 0 }
    var isRunning: Bool { [.indexing, .examples, .speed].contains(step) }
    var canPlay: Bool { step == .idle || (isRunning && paused) }
    var canReplay: Bool { isRunning || step == .finished }

    /// Plain-English questions about FluidAudio whose answers share few or none of their words.
    static let examples = [
        "turn text into speech audio", "variational Bayes clustering of speaker embeddings",
        "boost recognition of custom vocabulary words", "convert a word into phonemes",
        "stream microphone audio into the recognizer", "agglomerative hierarchical clustering",
        "print a progress bar while downloading", "resample an audio file to 16 kHz mono",
    ]

    private var prepareTask: Task<Void, Never>?

    /// Loads once, in a task the model owns, so the window going away or redrawing cannot cancel it.
    func start() {
        guard prepareTask == nil else { return }
        prepareTask = Task { await prepare() }
    }

    func prepare() async {
        do {
            let path =
                CommandLine.arguments.first { $0.hasPrefix("--repo=") }.map { String($0.dropFirst(7)) }
                ?? "~/Documents/FluidAudio"
            repository = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
            repositoryName = repository.lastPathComponent
            chunks = CodeChunker.chunks(repository: repository)
            guard !chunks.isEmpty else { throw EmbeddingGemma2Error.invalidAsset("No Swift files in \(path)") }
            let paths = Set(chunks.map(\.path))
            fileCount = paths.count
            for path in paths {
                sources[path] =
                    (try? String(contentsOf: repository.appendingPathComponent(path), encoding: .utf8))?.lowercased()
            }
            lineCount = sources.values.reduce(0) {
                $0 + $1.split(separator: "\n", omittingEmptySubsequences: false).count
            }
            chunksTotal = chunks.count
            DemoLog.event(
                "Code search · \(repositoryName): \(fileCount) Swift files, \(lineCount) lines, \(chunks.count) functions and types"
            )
            step = .loading("Loading EmbeddingGemma 2 on the Neural Engine…")
            let start = DispatchTime.now().uptimeNanoseconds
            // The first load of a new build compiles the model for the Neural Engine (a few minutes, once).
            let ticker = Task { @MainActor [weak self] in
                var seconds = 0
                while !Task.isCancelled {
                    try? await Task.sleep(for: .seconds(1))
                    seconds += 1
                    guard let self, case .loading = self.step else { continue }
                    self.step = .loading(
                        "Loading EmbeddingGemma 2… \(seconds) s"
                            + (seconds >= 8 ? " (first launch compiles the model once, ~2 min)" : ""))
                }
            }
            defer { ticker.cancel() }
            let text = try await EmbeddingGemma2Manager.loadDefault()
            _ = try await text.embed("warm up", prompt: .codeRetrieval)
            _ = try await text.embed(Array(repeating: "warm up", count: 64), prompt: .codeRetrieval)
            self.text = text
            DemoLog.model(
                String(
                    format: "model loaded in %.1f s · 100%% of its ops on the Neural Engine",
                    Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9))
            step = .idle
            if CommandLine.arguments.contains("--autostart") { play() }
        } catch {
            step = .failed(error.localizedDescription)
            DemoLog.line("failed: \(error.localizedDescription)", color: 196, bold: true)
        }
    }

    // MARK: Play / Pause / Replay

    func play() {
        if isRunning, paused {
            if let began = pauseBegan { pausedNanoseconds += DispatchTime.now().uptimeNanoseconds - began }
            pauseBegan = nil
            paused = false
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
        DemoLog.event("❚❚ pause")
    }

    func replay() {
        guard canReplay else { return }
        showTask?.cancel()
        paused = false
        pauseBegan = nil
        indexed = []
        matrix = []
        chunksDone = 0
        indexSeconds = 0
        results = []
        feed = []
        selected = nil
        grepMatches = nil
        query = ""
        showQuery = ""
        burstQueries = 0
        burstPerSecond = 0
        burstMilliseconds = 0
        exampleNumber = 0
        step = .idle
        DemoLog.event("↺ replay")
        showTask = Task { await show() }
    }

    private var activeNow: UInt64 {
        let now = DispatchTime.now().uptimeNanoseconds
        return now - pausedNanoseconds - (pauseBegan.map { now - $0 } ?? 0)
    }

    private func waitWhilePaused() async {
        while paused, !Task.isCancelled { try? await Task.sleep(for: .milliseconds(50)) }
    }

    /// Holds `seconds` of running time.
    private func dwell(_ seconds: Double) async {
        let start = activeNow
        while !Task.isCancelled, Double(activeNow - start) / 1e9 < seconds {
            try? await Task.sleep(for: .milliseconds(50))
        }
    }

    private func show() async {
        DemoLog.event(
            "▶ play: index \(repositoryName), \(Self.examples.count) example questions, \(Int(speedSeconds)) s of searches as fast as they go"
        )
        await index()
        guard !Task.isCancelled, !indexed.isEmpty else { return }
        await examplesStep()
        guard !Task.isCancelled else { return }
        await speed()
        guard !Task.isCancelled else { return }
        step = .finished
        DemoLog.event(
            String(
                format: "■ done: %d functions and types indexed in %.1f s (%.0f/s); %d searches at %.0f/s",
                indexed.count, indexSeconds, chunksPerSecond, burstQueries, burstPerSecond))
    }

    // MARK: Index

    private func index() async {
        guard let text else { return }
        step = .indexing
        DemoLog.event("📥 indexing \(chunks.count) functions and types (\(maxTokens) tokens each, at most)")
        let start = activeNow
        // Small batches keep the live list moving; packing still fills each Neural Engine call.
        let batchSize = 64
        feed = []
        do {
            for batchStart in stride(from: 0, to: chunks.count, by: batchSize) {
                await waitWhilePaused()
                try Task.checkCancellation()
                let batch = Array(chunks[batchStart..<min(batchStart + batchSize, chunks.count)])
                let vectors = try await text.embed(batch.map(\.document), prompt: .none, maxTokens: maxTokens)
                try Task.checkCancellation()
                indexed += batch
                matrix += vectors.flatMap { $0 }
                chunksDone = indexed.count
                push(batch.suffix(4).map { (nil, $0) })
                if let last = batch.last { selected = Result(chunk: last, score: 0) }
                indexSeconds = Double(activeNow - start) / 1e9
            }
            DemoLog.event(
                String(
                    format: "📥 indexed %d functions and types from %d files (%d lines) in %.1f s = %.0f per second",
                    indexed.count, fileCount, lineCount, indexSeconds, chunksPerSecond))
        } catch is CancellationError {
        } catch {
            step = .failed(error.localizedDescription)
            DemoLog.line("failed: \(error.localizedDescription)", color: 196, bold: true)
        }
    }

    // MARK: Examples

    private func examplesStep() async {
        step = .examples
        feed = []
        selected = nil
        for (number, question) in Self.examples.enumerated() {
            guard !Task.isCancelled else { return }
            exampleNumber = number + 1
            selected = nil
            grepMatches = nil
            showQuery = ""
            query = ""
            for character in question {
                await waitWhilePaused()
                showQuery.append(character)
                query.append(character)
                try? await Task.sleep(for: .milliseconds(12))
            }
            await waitWhilePaused()
            guard let ranked = await rank(question) else { return }
            results = ranked
            selected = ranked.first
            grepMatches = grepCount(question)
            DemoLog.line(
                String(
                    format: "      exact-phrase grep: %d files · #1 %@ (%@:%d)", grepMatches ?? 0,
                    ranked.first?.chunk.name ?? "-", ranked.first?.chunk.path ?? "-", ranked.first?.chunk.line ?? 0),
                color: 120)
            // Walk the top three, the first for longest.
            for (rank, result) in ranked.prefix(3).enumerated() {
                guard !Task.isCancelled else { return }
                selected = result
                await dwell(rank == 0 ? dwellSeconds * 0.6 : dwellSeconds * 0.2)
            }
            selected = ranked.first
        }
    }

    /// Files containing the question verbatim (case-insensitive), over their full text: what `grep -ril` finds.
    private func grepCount(_ phrase: String) -> Int {
        let needle = phrase.lowercased()
        return sources.values.filter { $0.contains(needle) }.count
    }

    // MARK: Speed

    private func speed() async {
        guard let text, !indexed.isEmpty else { return }
        step = .speed
        let pool = Self.speedQuestions
        let dimension = EmbeddingGemma2Manager.dimension
        let count = indexed.count
        let batchSize = 64
        burstQueries = 0
        feed = []
        DemoLog.event("⚡ search speed: \(pool.count) different questions, back to back, against \(count) functions")
        let start = activeNow
        let length = UInt64(speedSeconds * 1e9)
        var next = 0
        var lastLog = start
        var recent: [(time: UInt64, count: Int)] = [(start, 0)]
        while !Task.isCancelled, activeNow - start < length {
            stepRemaining = max(0, Int((Double(length) - Double(activeNow - start)) / 1e9))
            if paused {
                await waitWhilePaused()
                recent = [(activeNow, 0)]
                continue
            }
            let batch = (0..<batchSize).map { pool[(next + $0) % pool.count] }
            next += batchSize
            guard let vectors = try? await text.embed(batch, prompt: .codeRetrieval) else { return }
            let queryMatrix = vectors.flatMap { $0 }
            var scores = [Float](repeating: 0, count: batchSize * count)
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans, Int32(batchSize), Int32(count), Int32(dimension), 1,
                queryMatrix, Int32(dimension), matrix, Int32(dimension), 0, &scores, Int32(count))
            // Every question gets its own top 10, as a real search would; a few answers per batch go to the live list.
            var shown: [Result] = []
            var answered: [(String?, CodeChunk)] = []
            for row in 0..<batchSize {
                let top = Self.topIndices(scores, row: row, width: count, count: 10)
                if row % 16 == 15, let best = top.first { answered.append((batch[row], indexed[best])) }
                if row == batchSize - 1 {
                    shown = top.map { Result(chunk: indexed[$0], score: scores[row * count + $0]) }
                }
            }
            push(answered)
            burstQueries += batchSize
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
            selected = shown.first
            grepMatches = nil
            queryMilliseconds = burstMilliseconds
            if now - lastLog > 1_000_000_000 {
                lastLog = now
                DemoLog.line(
                    String(
                        format: "⚡ %d searches · %.0f/s · %.2f ms each · %.1f M functions ranked/s", burstQueries,
                        burstPerSecond, burstMilliseconds, burstPerSecond * Double(count) / 1e6), color: 81)
            }
        }
        let elapsed = Double(activeNow - start) / 1e9
        if elapsed > 0, burstQueries > 0 {
            burstPerSecond = Double(burstQueries) / elapsed
            burstMilliseconds = 1000 / burstPerSecond
        }
    }

    private func push(_ items: [(String?, CodeChunk)]) {
        for (question, chunk) in items {
            feedSerial += 1
            feed.insert(FeedItem(id: feedSerial, question: question, chunk: chunk), at: 0)
        }
        if feed.count > 18 { feed.removeLast(feed.count - 18) }
    }

    /// The part of the codebase a path belongs to (`ASR`, `TTS`, `Diarizer`, …), for the coloured tag.
    static func area(_ path: String) -> String {
        let parts = path.split(separator: "/").map(String.init)
        if parts.first == "Tests" { return "Tests" }
        if parts.count > 2, parts[0] == "Sources" {
            if parts[1].hasSuffix("CLI") { return "CLI" }
            return parts.count > 3 ? parts[2] : parts[1]
        }
        return parts.first ?? path
    }

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

    // MARK: Search by hand (when the show is not running)

    func search() {
        if query == showQuery || isRunning { return }
        searchTask?.cancel()
        let question = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty, !indexed.isEmpty else {
            results = []
            selected = nil
            return
        }
        searchTask = Task {
            try? await Task.sleep(for: .milliseconds(120))
            guard !Task.isCancelled, let ranked = await rank(question), !Task.isCancelled else { return }
            results = ranked
            selected = ranked.first
            grepMatches = grepCount(question)
        }
    }

    func select(_ result: Result) {
        guard !isRunning else { return }
        selected = result
    }

    /// Top 25 chunks for `question`: one embedding, then one matrix-vector product over every chunk.
    private func rank(_ question: String) async -> [Result]? {
        guard let text else { return nil }
        let begin = DispatchTime.now().uptimeNanoseconds
        guard let vector = try? await text.embed(question, prompt: .codeRetrieval) else { return nil }
        var scores = [Float](repeating: 0, count: indexed.count)
        cblas_sgemv(
            CblasRowMajor, CblasNoTrans, Int32(indexed.count), Int32(vector.count), 1, matrix, Int32(vector.count),
            vector, 1, 0, &scores, 1)
        let ranked = scores.indices.sorted { scores[$0] > scores[$1] }.prefix(25).map {
            Result(chunk: indexed[$0], score: scores[$0])
        }
        queryMilliseconds = Double(DispatchTime.now().uptimeNanoseconds - begin) / 1e6
        DemoLog.line(
            String(format: "🔎 “%@” → %d functions ranked in %.0f ms", question, indexed.count, queryMilliseconds),
            color: 81)
        return ranked
    }

    /// The declaration as it is in the file (indentation kept), up to 60 lines.
    func snippet(_ chunk: CodeChunk) -> String {
        if let cached = snippets[chunk.path + ":\(chunk.line)"] { return cached }
        guard let text = try? String(contentsOf: repository.appendingPathComponent(chunk.path), encoding: .utf8)
        else { return chunk.code }
        let lines = text.components(separatedBy: "\n")
        var first = chunk.line - 1
        while first > 0, lines[first - 1].trimmingCharacters(in: .whitespaces).hasPrefix("///") { first -= 1 }
        let slice = Array(lines[first..<min(lines.count, chunk.line - 1 + 60)])
        let indent =
            slice.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            .map { $0.prefix { $0 == " " }.count }.min() ?? 0
        let result = slice.enumerated().map { offset, line in
            String(format: "%5d  ", first + offset + 1)
                + String(line.dropFirst(min(indent, line.prefix { $0 == " " }.count)))
        }
        .joined(separator: "\n")
        snippets[chunk.path + ":\(chunk.line)"] = result
        return result
    }

    /// Questions for the speed step: the examples plus a spread of developer questions about an audio SDK.
    static let speedQuestions =
        examples + [
            "download every file of a Hugging Face model repository", "softmax over an array of floats",
            "cosine distance between two speaker embeddings", "edit distance between two sequences",
            "write float samples to a WAV file", "k-means clustering", "split a phoneme string into tokens for SSML",
            "verify a downloaded file's checksum", "detect when someone starts and stops speaking",
            "split a long recording into overlapping chunks", "word error rate between a transcript and the reference",
            "convert stereo audio to mono", "normalize numbers and dates in a transcript",
            "decode audio tokens with beam search", "load a Core ML model from disk", "compile an mlpackage",
            "choose which compute units to run on", "retry a failed network request", "parse a JSON config file",
            "measure real-time factor", "log a message", "handle a missing model file", "cache downloaded models",
            "read a WAV file header", "compute a mel spectrogram", "apply a Hann window",
            "run a fast Fourier transform",
            "merge speaker segments that are close together", "assign each segment to a speaker",
            "extract speaker embeddings", "find the best matching speaker", "remove silence from audio",
            "voice activity detection threshold", "transcribe a file in chunks", "keep decoder state between chunks",
            "pick the token with the highest score", "map token ids back to text", "byte pair encoding tokenizer",
            "pad audio to a fixed length", "convert floats to 16-bit integers", "play audio through the speakers",
            "record from the microphone", "generate speech with a cloned voice", "choose a voice for text to speech",
            "add pauses between sentences", "spell out an abbreviation", "convert numbers to words",
            "benchmark a dataset and report accuracy", "download a test dataset", "compare two transcripts",
            "streaming end of utterance detection", "keyword spotting", "translate speech to another language",
            "detect the spoken language", "denoise a recording", "cancel echo from a speaker",
            "separate overlapping speakers", "count the number of speakers", "export results as JSON",
            "command line argument parsing", "print usage help", "measure memory usage", "thread-safe actor state",
        ]
}
