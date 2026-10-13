import AppKit
import FluidUse
import Foundation
import PDFKit
import UnlimitedOcrMLX

/// Drives one read: PDF -> page tensors -> UnlimitedOcrManager stream -> parsed document for the UI. Opening a PDF
/// only loads its pages; `start()` reads them, `togglePause()` holds / continues the generation.
@available(macOS 15.0, *)
@MainActor
final class ReaderModel: ObservableObject {
    enum Phase: Equatable {
        case loading, ready, loaded
        case encoding(Int)
        case prefilling, reading, paused, done
        case failed(String)
    }

    @Published var phase: Phase = .loading
    @Published var pageImages: [NSImage] = []
    @Published var document = UnlimitedOcrDocument(parsing: "")
    @Published var title = ""
    @Published var outputTokens = 0
    @Published var tokensPerSecond = 0.0
    @Published var elapsed = 0.0
    @Published var maxPages = 0
    @Published var status = "Loading the model…"

    private var manager: UnlimitedOcrFastManager?
    private var readTask: Task<Void, Never>?
    private var pdfPages: [PDFPage] = []
    private var text = ""
    private var pausedAt: Date?
    private var loggedBlocks = 0
    /// Bumped per read; events from a cancelled read are dropped.
    private var generation = 0
    private var pausedTotal = 0.0

    /// Start (or restart) is available whenever a document is loaded, including mid-read and while paused.
    var canStart: Bool {
        switch phase {
        case .loading, .ready: return false
        case .failed: return manager != nil && !pdfPages.isEmpty
        default: return !pdfPages.isEmpty
        }
    }

    /// The button reads "Restart" once a read has begun.
    var hasStarted: Bool {
        switch phase {
        case .loading, .ready, .loaded: return false
        default: return true
        }
    }

    var canPause: Bool { phase == .reading || phase == .paused }

    func boot() async {
        do {
            let started = Date()
            let directory: URL
            if let path = ProcessInfo.processInfo.environment["UNLIMITED_OCR_BUNDLE"], !path.isEmpty {
                directory = URL(fileURLWithPath: path)
            } else {
                status = "Downloading Unlimited-OCR (\(UnlimitedOcrModelStore.repository), ~3.9 GB)…"
                directory = try await UnlimitedOcrModelStore.ensure(runtime: .mlx)
            }
            status = "Loading the model…"
            // Core ML page encoder + MLX text decoder (no per-size buckets, so nothing to prepare per document).
            let manager = try await UnlimitedOcrFastManager.load(from: directory)
            try await manager.warm()
            self.manager = manager
            maxPages = manager.maxPages
            print(
                String(
                    format: "[unlimited-ocr] loaded in %.1f s, up to %d pages per read",
                    Date().timeIntervalSince(started), maxPages))
            phase = .ready
            if let path = ProcessInfo.processInfo.environment["UNLIMITED_OCR_PDF"], !path.isEmpty {
                open(URL(fileURLWithPath: path))
            }
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    /// Loads the PDF's pages (up to `maxPages`) without reading them.
    func open(_ url: URL) {
        guard manager != nil, let pdf = PDFDocument(url: url) else {
            phase = .failed("Could not open \(url.lastPathComponent)")
            return
        }
        stop()
        let count = min(pdf.pageCount, maxPages)
        title = url.deletingPathExtension().lastPathComponent
        pdfPages = (0..<count).compactMap { pdf.page(at: $0) }
        pageImages = pdfPages.map { $0.thumbnail(of: NSSize(width: 900, height: 1200), for: .mediaBox) }
        reset()
        phase = .loaded
    }

    /// Reads the loaded pages from the beginning.
    func start() {
        guard let manager, canStart else { return }
        if hasStarted { print("[unlimited-ocr] restart") }
        stop()
        reset()
        generation += 1
        let pages = pdfPages
        let current = generation
        readTask = Task { [weak self] in
            await self?.read(pages, with: manager, generation: current)
        }
    }

    func togglePause() {
        guard let manager else { return }
        if phase == .reading {
            phase = .paused
            pausedAt = Date()
            manager.pause()
            print("[unlimited-ocr] paused at \(outputTokens) tokens")
        } else if phase == .paused {
            if let pausedAt { pausedTotal += Date().timeIntervalSince(pausedAt) }
            pausedAt = nil
            phase = .reading
            manager.resume()
            print("[unlimited-ocr] resumed")
        }
    }

    private func stop() {
        readTask?.cancel()
        readTask = nil
        if let manager { manager.resume() }
    }

    private func reset() {
        text = ""
        document = UnlimitedOcrDocument(parsing: "")
        outputTokens = 0
        tokensPerSecond = 0
        elapsed = 0
        pausedAt = nil
        pausedTotal = 0
        loggedBlocks = 0
    }

    /// One colored log line per finished layout block (the side terminal's `tail -F`).
    private func logFinishedBlocks(final: Bool = false) {
        let blocks = document.pages.flatMap { page in page.blocks.map { (page.id, $0) } }
        let done = final ? blocks.count : max(0, blocks.count - 1)  // the last block may still be streaming
        guard done > loggedBlocks else { return }
        for (page, block) in blocks[loggedBlocks..<done] {
            let snippet = block.content.replacingOccurrences(of: "\n", with: " ").prefix(44)
            print(
                String(
                    format: "\u{1B}[2m%6.1fs\u{1B}[0m  \u{1B}[38;5;214mp%-2d\u{1B}[0m \u{1B}[38;5;75m%-13@\u{1B}[0m %@"
                        + "  \u{1B}[2m%d tok · %.0f tok/s\u{1B}[0m",
                    elapsed, page + 1, block.label as NSString, String(snippet) as NSString, outputTokens,
                    tokensPerSecond))
        }
        loggedBlocks = done
    }

    private func read(_ pdfPages: [PDFPage], with manager: UnlimitedOcrFastManager, generation: Int) async {
        let started = Date()
        do {
            phase = .encoding(pdfPages.count)
            let pages = try pdfPages.map { try UnlimitedOcrPagePreprocessor.page(from: $0) }
            var decodeStart: Date?
            var lastParse = Date.distantPast
            for try await event in manager.read(pages: pages) {
                guard generation == self.generation, !Task.isCancelled else { return }
                switch event {
                case .pageEncoded(let index, _):
                    phase = index + 1 == pdfPages.count ? .prefilling : .encoding(pdfPages.count)
                case .prefilled(let tokens, _, _):
                    print(
                        String(
                            format: "[unlimited-ocr] %d pages encoded + prefilled (%d tokens) in %.2f s",
                            pdfPages.count, tokens, Date().timeIntervalSince(started)))
                    if phase != .paused { phase = .reading }
                    decodeStart = Date()
                case .text(let delta):
                    text += delta
                    outputTokens += 1
                    if Date().timeIntervalSince(lastParse) > 0.05 {
                        document = UnlimitedOcrDocument(parsing: text)
                        lastParse = Date()
                        logFinishedBlocks()
                    }
                    if let decodeStart {
                        let active = Date().timeIntervalSince(decodeStart) - pausedTotal
                        tokensPerSecond = Double(outputTokens) / max(0.001, active)
                    }
                    elapsed = Date().timeIntervalSince(started) - pausedTotal
                case .finished(let stats):
                    document = UnlimitedOcrDocument(parsing: text)
                    logFinishedBlocks(final: true)
                    outputTokens = stats.outputTokens
                    tokensPerSecond = 1000 / stats.millisecondsPerToken
                    elapsed = Date().timeIntervalSince(started) - pausedTotal
                    phase = .done
                    print(
                        String(
                            format: "[unlimited-ocr] %d pages -> %d tokens in %.1f s (%.1f tok/s)",
                            stats.pages, stats.outputTokens, elapsed, tokensPerSecond))
                }
            }
        } catch is CancellationError {
        } catch {
            if generation == self.generation { phase = .failed(error.localizedDescription) }
        }
    }
}
