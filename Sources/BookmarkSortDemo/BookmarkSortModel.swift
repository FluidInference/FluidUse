import AppKit
import BookmarkSort
import FluidUse
import Foundation

/// Watches the X page open in Chrome, sorts every post it has not seen yet, and learns from confirmations.
@MainActor
final class BookmarkSortModel: ObservableObject {
    enum Phase: Equatable {
        case loading(String)
        case waitingForPage(String)
        case watching
        case failed(String)
    }

    enum Status: Equatable {
        /// Confident on both levels: filed without asking.
        case filed
        /// Waiting for the person to pick one of the suggestions.
        case suggested
        /// The person picked this category.
        case confirmed(String)
        /// Already one of the filed examples, so the prediction proves nothing.
        case known(String)
    }

    struct Sorted: Identifiable {
        let post: Bookmark
        let result: SortDecision
        var status: Status
        var id: String { post.id }
    }

    @Published private(set) var phase: Phase = .loading("Starting…")
    @Published private(set) var items: [Sorted] = []
    @Published private(set) var onPage: Set<String> = []
    @Published private(set) var exampleCount = 0
    @Published private(set) var pageTitle = ""
    @Published var threshold: Float = 0.7
    /// Page through the browser on its own, reading each post as it comes on screen.
    @Published var autoScroll = CommandLine.arguments.contains("--auto-scroll")
    private let highlight = TweetHighlight()
    /// How the current sorter decides, for the header.
    @Published private(set) var modeDescription = ""

    private var sorter: (any BookmarkSorting)?
    private var embedder: (any BookmarkEmbedding)?
    private var examples: [FiledBookmark] = []
    private var vectors: [String: [Float]] = [:]
    private var folderOfCategory: [String: String] = [:]

    static let cacheDirectory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("FluidUse/bookmark-sort", isDirectory: true)
    static var confirmedURL: URL { cacheDirectory.appendingPathComponent("confirmed.jsonl") }
    /// One cache per embedder: vectors from different models don't mix.
    static func vectorsURL(_ embedder: any BookmarkEmbedding) -> URL {
        cacheDirectory.appendingPathComponent(
            embedder.name == "apple-nl" ? "vectors.json" : "vectors-\(embedder.name).json")
    }

    var newCount: Int { items.filter { if case .known = $0.status { false } else { true } }.count }
    var filedCount: Int { items.filter { $0.status == .filed }.count }
    var pendingCount: Int { items.filter { $0.status == .suggested }.count }
    var medianMilliseconds: Double {
        let values = items.suffix(max(items.count - 1, 0)).map(\.result.milliseconds).sorted()
        return values.isEmpty ? 0 : values[values.count / 2]
    }

    func start() async {
        do {
            try await prepare()
            try await watch()
        } catch is CancellationError {
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    private func prepare() async throws {
        guard XTimelineReader.isTrusted else {
            throw BookmarkSortError.unavailable(
                "Grant Accessibility access to the terminal that launched this demo (System Settings › Privacy)")
        }
        let arguments = CommandLine.arguments
        func value(_ name: String) -> String? {
            arguments.first { $0.hasPrefix("--\(name)=") }.map { String($0.dropFirst(name.count + 3)) }
        }
        guard let path = value("examples") ?? ProcessInfo.processInfo.environment["BOOKMARK_SORT_EXAMPLES"] else {
            try await prepareZeroShot(taxonomyPath: value("taxonomy"), variant: value("variant"))
            return
        }
        let embedder: any BookmarkEmbedding
        if value("embedder") == "apple" {
            phase = .loading("Loading Apple on-device embeddings…")
            embedder = try await BookmarkEmbedder()
        } else {
            phase = .loading("Loading EmbeddingGemma 2 on the Neural Engine…")
            embedder = try await GemmaBookmarkEmbedder.load()
        }
        self.embedder = embedder
        examples = try FiledBookmark.load(jsonLines: URL(fileURLWithPath: path))
        if let data = try? Data(contentsOf: Self.confirmedURL), let text = String(data: data, encoding: .utf8) {
            examples += text.split(separator: "\n").compactMap {
                try? JSONDecoder().decode(FiledBookmark.self, from: Data($0.utf8))
            }
        }
        if let data = try? Data(contentsOf: Self.vectorsURL(embedder)) {
            vectors = (try? JSONDecoder().decode([String: [Float]].self, from: data)) ?? [:]
        }
        for (index, example) in examples.enumerated() where vectors[example.id] == nil {
            if index % 20 == 0 { phase = .loading("Learning from your filed posts… \(index)/\(examples.count)") }
            vectors[example.id] = try await embedder.embed(example.bookmark.classificationText)
        }
        try FileManager.default.createDirectory(at: Self.cacheDirectory, withIntermediateDirectories: true)
        try JSONEncoder().encode(vectors).write(to: Self.vectorsURL(embedder))
        try await retrain()
        print("trained on \(examples.count) filed posts")
    }

    /// No examples: GLiNER2.5-Decide scores the taxonomy's labels directly.
    private func prepareZeroShot(taxonomyPath: String?, variant name: String?) async throws {
        let taxonomy = try taxonomyPath.map { try BookmarkTaxonomy.load(URL(fileURLWithPath: $0)) } ?? .aiBookmarks
        guard let variant = GLiNER2Variant(rawValue: name ?? "decide") else {
            throw BookmarkSortError.invalidInput("Unknown variant \(name ?? "")")
        }
        phase = .loading("Loading GLiNER2.5 (\(variant.rawValue)) on Core ML…")
        sorter = try await BookmarkSorter.load(variant: variant, taxonomy: taxonomy)
        threshold = 0.5
        modeDescription = "Zero-shot · GLiNER2.5-Decide on Core ML · no examples, no training · nothing leaves this Mac"
        print("zero-shot \(variant.rawValue): \(taxonomy.folders.map(\.name))")
    }

    private func retrain() async throws {
        guard let embedder else { return }
        let examples = examples
        let raw = examples.map { vectors[$0.id]! }
        sorter = try await Task.detached(priority: .userInitiated) {
            try LearnedBookmarkSorter.train(on: examples, rawVectors: raw, embedder: embedder)
        }.value
        exampleCount = examples.count
        modeDescription =
            "Learned from \(examples.count) posts you filed · "
            + (embedder is GemmaBookmarkEmbedder
                ? "EmbeddingGemma 2 on the Neural Engine" : "Apple on-device embeddings")
            + " · nothing leaves this Mac"
        folderOfCategory = [:]
        for example in examples { if let category = example.category { folderOfCategory[category] = example.folder } }
    }

    private func watch() async throws {
        guard let chrome = NSRunningApplication.runningApplications(withBundleIdentifier: "com.google.Chrome").first
        else { throw BookmarkSortError.unavailable("Open Google Chrome on an x.com page") }
        let reader = XTimelineReader(processIdentifier: chrome.processIdentifier)
        let dwell = CommandLine.arguments.first { $0.hasPrefix("--dwell=") }.flatMap { Double($0.dropFirst(8)) } ?? 0.9
        phase = .waitingForPage("Open an x.com page in Chrome")
        var quietScrolls = 0
        while !Task.isCancelled {
            guard let page = try await reader.read(), let sorter else {
                phase = .waitingForPage("Open an x.com page in Chrome")
                highlight.hide()
                try await Task.sleep(for: .milliseconds(700))
                continue
            }
            phase = .watching
            pageTitle = page.windowTitle.components(separatedBy: " - Google Chrome").first ?? page.windowTitle
            onPage = Set(page.visiblePosts.map(\.id))
            let fresh = page.visiblePosts.filter { post in !items.contains { $0.id == post.id } }
            for post in fresh {
                guard let frame = page.frames[post.id] else { continue }
                highlight.show(around: frame, .init(caption: "reading…", color: .orange))
                try await Task.sleep(for: .milliseconds(250))
                let result = try await sorter.sort(post)
                let status: Status
                if let filed = examples.first(where: { $0.id == post.id }) {
                    status = .known(filed.category ?? filed.folder)
                } else {
                    status = result.isConfident(threshold: threshold) ? .filed : .suggested
                    log(post, result, confident: status == .filed)
                }
                items.insert(Sorted(post: post, result: result, status: status), at: 0)
                highlight.update(Self.look(result, status))
                try await Task.sleep(for: .seconds(dwell))
            }
            if autoScroll {
                quietScrolls = fresh.isEmpty ? quietScrolls + 1 : 0
                if quietScrolls > 8 {
                    autoScroll = false
                    print("reached the end of the page")
                } else {
                    if await reader.xWindowIsFocused() {
                        try await scroll(chrome, by: page.windowFrame.height * 0.6)
                    } else {
                        quietScrolls = 0  // paused: keys would land in another Chrome window
                    }
                }
                try await Task.sleep(for: .milliseconds(900))
            } else {
                try await Task.sleep(for: .milliseconds(700))
            }
        }
    }

    /// Scrolls the browser with Down-arrow presses sent to it alone, so this app keeps focus.
    /// Chromium ignores wheel events posted to a background process; key events reach the page.
    private func scroll(_ application: NSRunningApplication, by pixels: CGFloat) async throws {
        for _ in 0..<max(1, Int(pixels / 40)) {
            for down in [true, false] {
                CGEvent(keyboardEventSource: nil, virtualKey: 0x7D, keyDown: down)?
                    .postToPid(application.processIdentifier)
            }
            try await Task.sleep(for: .milliseconds(30))
        }
    }

    private static func look(_ result: SortDecision, _ status: Status) -> TweetHighlight.Look {
        let name = result.category.map(short) ?? result.folder
        let confidence = result.categoryConfidence ?? result.folderConfidence
        let caption = String(format: "%@ · %.0f%% · %.0f ms", name, confidence * 100, result.milliseconds)
        switch status {
        case .filed: return .init(caption: caption, color: name == "Not AI" ? .gray : .green)
        case .known(let category): return .init(caption: "already filed · \(short(category))", color: .gray)
        default: return .init(caption: caption + " ?", color: .orange)
        }
    }

    /// Files `id` under `category` and learns from it.
    func confirm(_ id: String, category: String) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        let item = items[index]
        items[index].status = .confirmed(category)
        print("filed @\(item.post.author) → \(Self.short(category))")
        guard sorter is LearnedBookmarkSorter else { return }
        let folder = folderOfCategory[category] ?? category
        let filed = FiledBookmark(
            id: item.post.id, text: item.post.text, quoted: item.post.quotedText, folder: folder,
            category: folder == category ? nil : category)
        Task {
            do {
                guard let embedder else { return }
                if vectors[filed.id] == nil {
                    vectors[filed.id] = try await embedder.embed(filed.bookmark.classificationText)
                }
                examples.removeAll { $0.id == filed.id }
                examples.append(filed)
                let line = String(decoding: try JSONEncoder().encode(filed), as: UTF8.self) + "\n"
                if let handle = try? FileHandle(forWritingTo: Self.confirmedURL) {
                    handle.seekToEndOfFile()
                    handle.write(Data(line.utf8))
                    try handle.close()
                } else {
                    try line.write(to: Self.confirmedURL, atomically: true, encoding: .utf8)
                }
                try JSONEncoder().encode(vectors).write(to: Self.vectorsURL(embedder))
                try await retrain()
                print("learned @\(item.post.author) → \(Self.short(category)); \(examples.count) filed posts")
            } catch {
                print("could not learn from @\(item.post.author): \(error.localizedDescription)")
            }
        }
    }

    /// Categories the person can file into, grouped by folder.
    var folders: [(folder: String, categories: [String])] {
        guard let sorter else { return [] }
        return sorter.folders.map { folder in
            let categories = sorter.categories(in: folder)
            return (folder, categories.isEmpty ? [folder] : categories)
        }
    }

    private func log(_ post: Bookmark, _ result: SortDecision, confident: Bool) {
        let where_ = result.category.map { "\(result.folder) › \(Self.short($0))" } ?? result.folder
        let confidence = result.categoryConfidence ?? result.folderConfidence
        print(
            String(
                format: "%6.1f ms  %@  @%@ → %@  (%.2f)", result.milliseconds, confident ? "FILED  " : "SUGGEST",
                post.author, where_, confidence))
    }

    /// `十三、华夷之辨与身份论战 · Identity Wars…` → `十三、华夷之辨与身份论战`.
    static func short(_ name: String) -> String {
        String(name.prefix { $0 != "·" }).trimmingCharacters(in: .whitespaces)
    }

    /// English half of a bilingual category name, if any.
    static func english(_ name: String) -> String? {
        guard let dot = name.firstIndex(of: "·") else { return nil }
        return name[name.index(after: dot)...].trimmingCharacters(in: .whitespaces)
    }
}
