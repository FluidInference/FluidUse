import AppKit
import BookmarkSort
import FluidUse
import Foundation

/// Scores the sorter against bookmarks someone already filed by hand.
///
///     swift run -c release BookmarkSortCheck --gold=gold.jsonl --taxonomy=taxonomy.json \
///         [--variant=decideLong] [--dump=predictions.jsonl]      zero-shot GLiNER 2.5 on the taxonomy's labels
///     swift run -c release BookmarkSortCheck --gold=gold.jsonl --learned [--folds=5] [--embedder=gemma|apple]
///                                                                 embeddings + classifiers, k-fold cross-validated
///     swift run -c release BookmarkSortCheck --tokenizer-check=tok_ref.jsonl
///                                                                 EmbeddingGemma 2 tokenizer vs `{"text", "ids"}` lines
///
/// `gold.jsonl` holds one `{"id", "text", "quoted"?, "folder", "category"?}` object per line.
@main
struct BookmarkSortCheck {
    struct Gold: Decodable {
        let id: String
        let text: String
        let quoted: String?
        let folder: String
        let category: String?
    }

    static func main() async throws {
        let arguments = CommandLine.arguments.dropFirst()
        func value(_ name: String) -> String? {
            arguments.first { $0.hasPrefix("--\(name)=") }.map { String($0.dropFirst(name.count + 3)) }
        }
        if let target = value("harvest").flatMap(Int.init) {
            try await harvest(target: target, outputPath: value("out"))
            return
        }
        if arguments.contains("--chrome") {
            let chrome = NSRunningApplication.runningApplications(withBundleIdentifier: "com.google.Chrome")
            guard let chrome = chrome.first else { throw BookmarkSortError.unavailable("Chrome is not running") }
            let page = try await XTimelineReader(processIdentifier: chrome.processIdentifier).read()
            print(page?.windowTitle ?? "no X window")
            for post in page?.posts ?? [] {
                print(
                    "@\(post.author) \(post.id) \(post.media)\n  \(post.text.prefix(200).replacingOccurrences(of: "\n", with: " ⏎ "))"
                )
            }
            return
        }
        if arguments.contains("--learned"), let goldPath = value("gold") {
            try await crossValidate(
                goldPath: goldPath, folds: value("folds").flatMap(Int.init) ?? 5, embedderName: value("embedder"),
                dumpPath: value("dump"))
            return
        }
        if let path = value("pack-check") {
            try await checkPacking(jsonLinesPath: path)
            return
        }
        if let path = value("tokenizer-check") {
            try await checkTokenizer(referencePath: path)
            return
        }
        guard let taxonomyPath = value("taxonomy"), let goldPath = value("gold") else {
            print("usage: BookmarkSortCheck --taxonomy=taxonomy.json --gold=gold.jsonl [--variant=decideLong]")
            exit(2)
        }
        guard let variant = GLiNER2Variant(rawValue: value("variant") ?? "decideLong") else {
            print("unknown variant; one of \(GLiNER2Variant.allCases.map(\.rawValue))")
            exit(2)
        }
        let taxonomy = try BookmarkTaxonomy.load(URL(fileURLWithPath: taxonomyPath))
        let gold = try String(contentsOfFile: goldPath, encoding: .utf8).split(separator: "\n").map {
            try JSONDecoder().decode(Gold.self, from: Data($0.utf8))
        }
        let sorter = try await BookmarkSorter.load(variant: variant, taxonomy: taxonomy)
        let bookmarks = gold.map { Bookmark(id: $0.id, author: "", text: $0.text, quotedText: $0.quoted) }
        _ = try await sorter.sort(bookmarks[0])

        var folderCorrect = 0
        var categoryTotal = 0
        var categoryCorrect = 0
        var truncated = 0
        var latencies: [Double] = []
        var confusions: [String: Int] = [:]
        var dump = ""
        let wall = DispatchTime.now().uptimeNanoseconds
        for (item, bookmark) in zip(gold, bookmarks) {
            let result = try await sorter.sort(bookmark)
            latencies.append(result.milliseconds)
            truncated += result.truncated ? 1 : 0
            folderCorrect += result.folder == item.folder ? 1 : 0
            if !result.folder.isEmpty, result.folder != item.folder {
                confusions["folder  \(item.folder) → \(result.folder)", default: 0] += 1
            }
            if let expected = item.category {
                categoryTotal += 1
                let predicted = result.folder == item.folder ? result.category : nil
                if predicted == expected {
                    categoryCorrect += 1
                } else if let predicted {
                    confusions["theme   \(short(expected)) → \(short(predicted))", default: 0] += 1
                }
            }
            let record: [String: Any] = [
                "id": item.id, "folder": result.folder, "category": result.category ?? NSNull(),
                "gold_folder": item.folder, "gold_category": item.category ?? NSNull(),
                "folder_confidence": result.folderConfidence, "truncated": result.truncated,
            ]
            if value("dump") != nil,
                let line = String(data: try JSONSerialization.data(withJSONObject: record), encoding: .utf8)
            {
                dump += line + "\n"
            }
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - wall) / 1e9
        if let path = value("dump") { try dump.write(toFile: path, atomically: true, encoding: .utf8) }
        latencies.sort()
        func percent(_ part: Int, _ whole: Int) -> String {
            String(format: "%.1f%% (%d/%d)", 100 * Double(part) / Double(max(whole, 1)), part, whole)
        }
        print(
            """
            \(variant.rawValue)  items \(gold.count)
            folder accuracy     \(percent(folderCorrect, gold.count))
            category accuracy   \(percent(categoryCorrect, categoryTotal))  (end to end: a wrong folder counts wrong)
            truncated           \(percent(truncated, gold.count))
            wall \(String(format: "%.2f", seconds)) s  p50 \(String(format: "%.2f", latencies[latencies.count / 2])) ms  \
            p95 \(String(format: "%.2f", latencies[latencies.count * 95 / 100])) ms
            most common mistakes:
            """)
        for (key, count) in confusions.sorted(by: { $0.value > $1.value }).prefix(12) {
            print("  \(count)\t\(key)")
        }
    }

    static func crossValidate(goldPath: String, folds: Int, embedderName: String?, dumpPath: String?) async throws {
        let examples = try FiledBookmark.load(jsonLines: URL(fileURLWithPath: goldPath))
        let embedder: any BookmarkEmbedding =
            embedderName == "apple" ? try await BookmarkEmbedder() : try await GemmaBookmarkEmbedder.load()
        _ = try await embedder.embed("warm up")
        let start = DispatchTime.now().uptimeNanoseconds
        var vectors: [[Float]] = []
        for example in examples { vectors.append(try await embedder.embed(example.bookmark.classificationText)) }
        let embedMilliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 / Double(examples.count)
        if let dumpPath {
            let lines = zip(examples, vectors).map { example, vector in
                String(
                    decoding: try! JSONSerialization.data(withJSONObject: ["id": example.id, "v": vector]),
                    as: UTF8.self)
            }
            try lines.joined(separator: "\n").write(toFile: dumpPath, atomically: true, encoding: .utf8)
        }
        // Fixed interleaved folds keep the run reproducible.
        let order = examples.indices.sorted { ($0 * 7919) % examples.count < ($1 * 7919) % examples.count }
        var folderCorrect = 0
        var categoryTotal = 0
        var hits = [0, 0, 0]
        var confident = (count: 0, correct: 0)
        for fold in 0..<folds {
            let test = Set(order.enumerated().filter { $0.offset % folds == fold }.map(\.element))
            let train = examples.indices.filter { !test.contains($0) }
            let sorter = try LearnedBookmarkSorter.train(
                on: train.map { examples[$0] }, rawVectors: train.map { vectors[$0] }, embedder: embedder)
            for index in test {
                let result = sorter.sort(rawVector: vectors[index])
                let item = examples[index]
                folderCorrect += result.folder == item.folder ? 1 : 0
                guard let expected = item.category else { continue }
                categoryTotal += 1
                let ranked = result.folder == item.folder ? result.categories.map(\.name) : []
                for k in 0..<3 where ranked.prefix(k + 1).contains(expected) { hits[k] += 1 }
                if result.isConfident(threshold: 0.7) {
                    confident.count += 1
                    confident.correct += ranked.first == expected ? 1 : 0
                }
            }
        }
        func percent(_ part: Int, _ whole: Int) -> String {
            String(format: "%.1f%% (%d/%d)", 100 * Double(part) / Double(max(whole, 1)), part, whole)
        }
        print(
            """
            learned (\(embedder.name)), \(folds)-fold  items \(examples.count)  embed \(String(format: "%.1f", embedMilliseconds)) ms/post
            folder accuracy     \(percent(folderCorrect, examples.count))
            category top-1      \(percent(hits[0], categoryTotal))  (end to end)
            category top-3      \(percent(hits[2], categoryTotal))
            confident (≥0.7)    \(percent(confident.correct, confident.count)) correct, \
            covering \(percent(confident.count, categoryTotal))
            """)
    }

    /// Collects posts from the X page in Chrome (background is fine: it scrolls with the accessibility action on
    /// the last post, no key events). Writes one `Bookmark` JSON per line to `outputPath` as it goes, else stdout.
    static func harvest(target: Int, outputPath: String?) async throws {
        let chrome = NSRunningApplication.runningApplications(withBundleIdentifier: "com.google.Chrome")
        guard let chrome = chrome.first else { throw BookmarkSortError.unavailable("Chrome is not running") }
        let reader = XTimelineReader(processIdentifier: chrome.processIdentifier)
        var seen = Set<String>()
        var stale = 0
        var output: FileHandle?
        if let outputPath {
            if let existing = try? String(contentsOfFile: outputPath, encoding: .utf8) {
                for line in existing.split(separator: "\n") {
                    if let post = try? JSONDecoder().decode(Bookmark.self, from: Data(line.utf8)) {
                        seen.insert(post.id)
                    }
                }
            } else {
                FileManager.default.createFile(atPath: outputPath, contents: nil)
            }
            output = FileHandle(forWritingAtPath: outputPath)
            output?.seekToEndOfFile()
        }
        while seen.count < target, stale < 12 {
            let before = seen.count
            for post in try await reader.read()?.posts ?? [] where !seen.contains(post.id) {
                seen.insert(post.id)
                let line = String(decoding: try JSONEncoder().encode(post), as: UTF8.self) + "\n"
                if let output { output.write(Data(line.utf8)) } else { print(line, terminator: "") }
            }
            stale = seen.count == before ? stale + 1 : 0
            FileHandle.standardError.write(Data("\r\(seen.count)/\(target) posts".utf8))
            await reader.scrollToLastPost()
            try await Task.sleep(for: .milliseconds(stale > 0 ? 2500 : 1500))
        }
        FileHandle.standardError.write(Data("\nharvested \(seen.count) posts\n".utf8))
        try output?.close()
    }

    /// Packed (`pack_256`) vs one-at-a-time embeddings for the `text` field of each JSON line: agreement and speed.
    static func checkPacking(jsonLinesPath: String) async throws {
        struct Line: Decodable { let text: String }
        let texts = try String(contentsOfFile: jsonLinesPath, encoding: .utf8).split(separator: "\n").map {
            try JSONDecoder().decode(Line.self, from: Data($0.utf8)).text
        }
        let manager = try await EmbeddingGemma2Manager.loadDefault()
        _ = try await manager.embed(["warm up"], prompt: .clustering)
        var start = DispatchTime.now().uptimeNanoseconds
        var single: [[Float]] = []
        for text in texts { single.append(try await manager.embed(text, prompt: .clustering)) }
        let singleSeconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        start = DispatchTime.now().uptimeNanoseconds
        var packed: [[Float]] = []
        for batch in stride(from: 0, to: texts.count, by: 64) {
            packed += try await manager.embed(Array(texts[batch..<min(batch + 64, texts.count)]), prompt: .clustering)
        }
        let packedSeconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let cosines = zip(single, packed).map { zip($0, $1).reduce(Float(0)) { $0 + $1.0 * $1.1 } }.sorted()
        print(
            String(
                format: "%d texts: one at a time %.0f/s, packed %.0f/s; cos min %.5f, mean %.6f", texts.count,
                Double(texts.count) / singleSeconds, Double(texts.count) / packedSeconds, cosines.first ?? 0,
                cosines.reduce(0, +) / Float(max(cosines.count, 1))))
    }

    static func checkTokenizer(referencePath: String) async throws {
        struct Reference: Decodable {
            let text: String
            let ids: [Int32]
        }
        let directory = try await EmbeddingGemma2ModelStore.ensure()
        var start = DispatchTime.now().uptimeNanoseconds
        let tokenizer = try EmbeddingGemma2Tokenizer(
            tokenizerJsonURL: directory.appendingPathComponent("tokenizer.json"))
        print(String(format: "loaded tokenizer in %.0f ms", Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6))
        let references = try String(contentsOfFile: referencePath, encoding: .utf8).split(separator: "\n").map {
            try JSONDecoder().decode(Reference.self, from: Data($0.utf8))
        }
        var mismatches = 0
        start = DispatchTime.now().uptimeNanoseconds
        for reference in references {
            let ids = tokenizer.encode(reference.text, maxLength: .max)
            if ids != reference.ids {
                mismatches += 1
                if mismatches <= 3 {
                    let first =
                        zip(ids, reference.ids).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                        ?? min(ids.count, reference.ids.count)
                    print(
                        "mismatch at \(first): \(ids.dropFirst(first).prefix(6)) vs \(reference.ids.dropFirst(first).prefix(6))"
                    )
                }
            }
        }
        let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 / Double(references.count)
        print(
            "tokenizer: \(references.count - mismatches)/\(references.count) identical, \(String(format: "%.2f", milliseconds)) ms/text"
        )
    }

    static func short(_ name: String) -> String {
        String(name.prefix { $0 != "·" }).trimmingCharacters(in: .whitespaces)
    }
}
