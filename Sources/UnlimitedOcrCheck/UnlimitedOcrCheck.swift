import CoreGraphics
import FluidUse
import Foundation
import ImageIO
import PDFKit

// Runs UnlimitedOcrManager on page images or a PDF and optionally diffs against a reference output.md.
//   UnlimitedOcrCheck <bundle dir | hub> <page.png ... | doc.pdf> [--ref output.md] [--out text.md] [--max-new N]
//       [--first-page K --pages N] [--npy pixels.npy] [--cache dir]
@main
struct UnlimitedOcrCheck {
    static func option(_ name: String, _ args: inout [String]) -> String? {
        guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
        let value = args[i + 1]
        args.removeSubrange(i...(i + 1))
        return value
    }

    /// Reference pixels from tools/ref_generate.py (`pixels.npy`, float32 [N, 3, 1024, 1024]) for exact parity.
    static func loadNpyPages(_ path: String) throws -> [UnlimitedOcrPage] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let headerLength = Int(data[8]) | Int(data[9]) << 8
        let body = data.dropFirst(10 + headerLength)
        let plane = 3 * UnlimitedOcrPagePreprocessor.size * UnlimitedOcrPagePreprocessor.size
        let floats = body.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        return stride(from: 0, to: floats.count, by: plane).map {
            UnlimitedOcrPage(pixels: Array(floats[$0..<($0 + plane)]))
        }
    }

    static func main() async throws {
        setvbuf(stdout, nil, _IOLBF, 0)
        var args = Array(CommandLine.arguments.dropFirst())
        let refPath = option("--ref", &args)
        let outPath = option("--out", &args)
        let npyPath = option("--npy", &args)
        let cachePath = option("--cache", &args)
        let warmAll = args.contains("--warm-all")
        args.removeAll { $0 == "--warm-all" }
        let maxNew = option("--max-new", &args).flatMap(Int.init)
        let firstPage = option("--first-page", &args).flatMap(Int.init) ?? 1
        let pageCount = option("--pages", &args).flatMap(Int.init)
        guard #available(macOS 15.0, *) else {
            print("UnlimitedOcrCheck needs macOS 15 (Core ML state)")
            exit(1)
        }
        if npyPath != nil { args.append("-") }
        guard args.count >= 2 else {
            print(
                "usage: UnlimitedOcrCheck <bundle dir | hub> <page.png ... | doc.pdf> [--ref output.md] [--out text.md]"
            )
            exit(2)
        }
        // "hub" downloads the pinned bundle (into --cache <dir> or the FluidUse cache).
        let bundle =
            args[0] == "hub"
            ? try await UnlimitedOcrModelStore.ensure(cacheDirectory: cachePath.map { URL(fileURLWithPath: $0) })
            : URL(fileURLWithPath: args[0])
        let inputs = args.dropFirst().map { URL(fileURLWithPath: $0) }

        var pixels: [UnlimitedOcrPage] = []
        if let npyPath {
            pixels = try loadNpyPages(npyPath)
        }
        for url in inputs where url.lastPathComponent != "-" {
            if url.pathExtension.lowercased() == "pdf" {
                guard let document = PDFDocument(url: url) else { fatalError("cannot open \(url.path)") }
                let last = min(document.pageCount, firstPage - 1 + (pageCount ?? document.pageCount))
                for index in (firstPage - 1)..<last {
                    pixels.append(try UnlimitedOcrPagePreprocessor.page(from: document.page(at: index)!))
                }
            } else {
                guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                    let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
                else { fatalError("cannot read \(url.path)") }
                pixels.append(try UnlimitedOcrPagePreprocessor.page(from: image))
            }
        }

        let loadStart = Date()
        let manager = try await UnlimitedOcrManager.load(from: bundle)
        try await manager.warm(pages: warmAll ? nil : [pixels.count])
        print(String(format: "loaded in %.1f s, %d page(s)", Date().timeIntervalSince(loadStart), pixels.count))

        var text = ""
        for try await event in manager.read(pages: pixels, maxNewTokens: maxNew) {
            switch event {
            case .pageEncoded(let index, let ms):
                print(String(format: "page %d encoded in %.0f ms", index + 1, ms))
            case .prefilled(let tokens, let capacity, let ms):
                print(String(format: "prefill %d tokens (bucket %d) in %.0f ms", tokens, capacity, ms))
            case .text(let delta):
                text += delta
            case .finished(let stats):
                print(
                    String(
                        format: "%d tokens in %.1f s (%.1f ms/token), EOS %@", stats.outputTokens,
                        stats.decodeMilliseconds / 1000, stats.millisecondsPerToken, stats.stoppedAtEOS ? "yes" : "no"))
            }
        }
        if let outPath { try text.write(toFile: outPath, atomically: true, encoding: .utf8) }
        print("----- output (first 600 chars) -----")
        print(String(text.prefix(600)))
        if let refPath {
            let ref = try String(contentsOfFile: refPath, encoding: .utf8)
            let got = text.trimmingCharacters(in: .whitespacesAndNewlines)
            // tools/ref_generate.py decodes the EOS token into output.md; the manager skips it.
            let want = ref.replacingOccurrences(of: "<｜end▁of▁sentence｜>", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let common = zip(got, want).prefix { $0 == $1 }.count
            print("----- vs reference -----")
            print(
                "identical: \(got == want) · common prefix \(common) of \(want.count) reference chars (got \(got.count))"
            )
            if got != want {
                print("ref: …\(want.dropFirst(max(0, common - 40)).prefix(120))")
                print("got: …\(got.dropFirst(max(0, common - 40)).prefix(120))")
            }
        }
    }
}
