import FluidUse
import Foundation

/// Files bookmarks into a `BookmarkTaxonomy` zero-shot, with a GLiNER 2.5 classifier on Core ML:
/// first the folder, then a category inside it. No examples needed; nothing leaves the machine.
public final class BookmarkSorter: BookmarkSorting {
    public let taxonomy: BookmarkTaxonomy
    public let variant: GLiNER2Variant
    private let manager: GLiNER2Manager

    public init(manager: GLiNER2Manager, variant: GLiNER2Variant, taxonomy: BookmarkTaxonomy) throws {
        guard !taxonomy.folders.isEmpty else { throw GLiNER2Error.invalidInput("The taxonomy has no folders") }
        self.manager = manager
        self.variant = variant
        self.taxonomy = taxonomy
    }

    /// Downloads (once) and loads `variant`.
    public static func load(
        variant: GLiNER2Variant = .decideLong, taxonomy: BookmarkTaxonomy,
        progress: GLiNER2ModelStore.Progress? = nil
    ) async throws -> BookmarkSorter {
        let manager = try await GLiNER2Manager.load(variant: variant, progress: progress)
        return try BookmarkSorter(manager: manager, variant: variant, taxonomy: taxonomy)
    }

    public var folders: [String] { taxonomy.folders.map(\.name) }

    public func categories(in folder: String) -> [String] {
        taxonomy.folders.first { $0.name == folder }?.categories.map(\.name) ?? []
    }

    public func sort(_ bookmark: Bookmark) async throws -> SortDecision {
        let start = DispatchTime.now().uptimeNanoseconds
        let text = bookmark.classificationText
        let folders = taxonomy.folders
        let folderRanking = try await rank(text: text, task: taxonomy.folderTask, labels: folders.map(\.modelLabel))
        let folder = folders[folderRanking.ranked[0].index]
        var categories: [(name: String, probability: Float)] = []
        var truncated = folderRanking.truncated
        if !folder.categories.isEmpty {
            let ranking = try await rank(
                text: text, task: taxonomy.categoryTask, labels: folder.categories.map(\.modelLabel))
            categories = ranking.ranked.map { (folder.categories[$0.index].name, $0.probability) }
            truncated = truncated || ranking.truncated
        }
        return SortDecision(
            folder: folder.name, folderConfidence: folderRanking.ranked[0].probability, categories: categories,
            milliseconds: Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6, truncated: truncated)
    }

    /// `labels` ranked best first. More labels than one head holds run as a knockout: each group of
    /// `maximumOptions` picks a winner, then the winners are ranked against each other.
    private func rank(
        text: String, task: String, labels: [String]
    ) async throws
        -> (ranked: [(index: Int, probability: Float)], truncated: Bool)
    {
        if labels.count == 1 { return ([(0, 1)], false) }
        let capacity = manager.maximumOptions
        guard labels.count > capacity else { return try await score(text: text, task: task, labels: labels) }
        var winners: [Int] = []
        var truncated = false
        for group in stride(from: 0, to: labels.count, by: capacity) {
            let indices = Array(group..<min(group + capacity, labels.count))
            let result = try await score(text: text, task: task, labels: indices.map { labels[$0] })
            winners.append(indices[result.ranked[0].index])
            truncated = truncated || result.truncated
        }
        let final = try await rank(text: text, task: task, labels: winners.map { labels[$0] })
        return (final.ranked.map { (winners[$0.index], $0.probability) }, truncated || final.truncated)
    }

    private func score(
        text: String, task: String, labels: [String]
    ) async throws
        -> (ranked: [(index: Int, probability: Float)], truncated: Bool)
    {
        let fitted = try fit(text, heads: [(task, labels)])
        let answer = try await manager.classifyConcurrently(text: fitted, task: task, labels: labels)
        let ranked = labels.indices.sorted { answer.probabilities[$0] > answer.probabilities[$1] }
            .map { ($0, answer.probabilities[$0]) }
        return (ranked, fitted.count < text.count)
    }

    /// Longest prefix of `text` that fits the token budget alongside the schema.
    func fit(_ text: String, heads: [(task: String, labels: [String])]) throws -> String {
        let limit = manager.maximumLength
        if try manager.tokenCount(text: text, heads: heads) <= limit { return text }
        let characters = Array(text)
        var low = 0
        var high = characters.count
        while low < high {
            let middle = (low + high + 1) / 2
            if try manager.tokenCount(text: String(characters[..<middle]), heads: heads) <= limit {
                low = middle
            } else {
                high = middle - 1
            }
        }
        guard low > 0 else {
            throw GLiNER2Error.invalidInput("The labels alone exceed \(limit) tokens")
        }
        return String(characters[..<low])
    }
}
