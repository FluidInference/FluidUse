import AppKit
import FluidUse
import Foundation
import PDFKit

/// Drives one read: PDF -> page tensors -> UnlimitedOcrManager stream -> parsed document for the UI.
@available(macOS 15.0, *)
@MainActor
final class ReaderModel: ObservableObject {
    enum Phase: Equatable {
        case loading, ready
        case encoding(Int)
        case prefilling, reading, done
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

    private var manager: UnlimitedOcrManager?
    private var readTask: Task<Void, Never>?
    private var text = ""

    static var bundleURL: URL {
        if let path = ProcessInfo.processInfo.environment["UNLIMITED_OCR_BUNDLE"], !path.isEmpty {
            return URL(fileURLWithPath: path)
        }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
            "Documents/unlimited-ocr-work/bundle")
    }

    func start() async {
        do {
            let started = Date()
            let manager = try await UnlimitedOcrManager.load(from: Self.bundleURL)
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

    func open(_ url: URL) {
        guard let manager, let pdf = PDFDocument(url: url) else {
            phase = .failed("Could not open \(url.lastPathComponent)")
            return
        }
        readTask?.cancel()
        let count = min(pdf.pageCount, maxPages)
        title = url.deletingPathExtension().lastPathComponent
        pageImages = (0..<count).compactMap {
            pdf.page(at: $0)?.thumbnail(of: NSSize(width: 900, height: 1200), for: .mediaBox)
        }
        text = ""
        document = UnlimitedOcrDocument(parsing: "")
        outputTokens = 0
        tokensPerSecond = 0
        elapsed = 0
        let pdfPages = (0..<count).compactMap { pdf.page(at: $0) }
        readTask = Task { [weak self] in
            await self?.read(pdfPages, with: manager)
        }
    }

    private func read(_ pdfPages: [PDFPage], with manager: UnlimitedOcrManager) async {
        let started = Date()
        do {
            phase = .encoding(pdfPages.count)
            let pages = try pdfPages.map { try UnlimitedOcrPagePreprocessor.page(from: $0) }
            var decodeStart: Date?
            var lastParse = Date.distantPast
            for try await event in manager.read(pages: pages) {
                switch event {
                case .pageEncoded(let index, _):
                    phase = index + 1 == pdfPages.count ? .prefilling : .encoding(pdfPages.count)
                case .prefilled(let tokens, _, _):
                    print(
                        String(
                            format: "[unlimited-ocr] %d pages encoded + prefilled (%d tokens) in %.2f s",
                            pdfPages.count, tokens, Date().timeIntervalSince(started)))
                    phase = .reading
                    decodeStart = Date()
                case .text(let delta):
                    text += delta
                    outputTokens += 1
                    if Date().timeIntervalSince(lastParse) > 0.05 {
                        document = UnlimitedOcrDocument(parsing: text)
                        lastParse = Date()
                    }
                    if let decodeStart {
                        tokensPerSecond = Double(outputTokens) / max(0.001, Date().timeIntervalSince(decodeStart))
                    }
                    elapsed = Date().timeIntervalSince(started)
                case .finished(let stats):
                    document = UnlimitedOcrDocument(parsing: text)
                    outputTokens = stats.outputTokens
                    tokensPerSecond = 1000 / stats.millisecondsPerToken
                    elapsed = Date().timeIntervalSince(started)
                    phase = .done
                    print(
                        String(
                            format: "[unlimited-ocr] %d pages -> %d tokens in %.1f s (%.1f tok/s)",
                            stats.pages, stats.outputTokens, elapsed, tokensPerSecond))
                }
            }
        } catch is CancellationError {
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }
}
