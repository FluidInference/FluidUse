import Foundation

/// Where a sorter put one post.
public struct SortDecision: Sendable {
    public let folder: String
    public let folderConfidence: Float
    /// Categories inside `folder`, most likely first; empty when the folder has none.
    public let categories: [(name: String, probability: Float)]
    public let milliseconds: Double
    /// The post was shortened to fit the model's token budget.
    public let truncated: Bool

    public init(
        folder: String, folderConfidence: Float, categories: [(name: String, probability: Float)],
        milliseconds: Double, truncated: Bool = false
    ) {
        self.folder = folder
        self.folderConfidence = folderConfidence
        self.categories = categories
        self.milliseconds = milliseconds
        self.truncated = truncated
    }

    public var category: String? { categories.first?.name }
    public var categoryConfidence: Float? { categories.first?.probability }

    /// Confident enough on both levels to file without asking.
    public func isConfident(threshold: Float) -> Bool {
        folderConfidence >= threshold && (categoryConfidence ?? 1) >= threshold
    }
}

/// Anything that can file a post: zero-shot (`BookmarkSorter`) or learned (`LearnedBookmarkSorter`).
public protocol BookmarkSorting: Sendable {
    func sort(_ bookmark: Bookmark) async throws -> SortDecision
    var folders: [String] { get }
    func categories(in folder: String) -> [String]
}
