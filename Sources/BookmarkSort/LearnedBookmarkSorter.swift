import Foundation

/// A post someone already filed: the examples `LearnedBookmarkSorter` learns from.
/// Stored one per line as `{"id", "text", "quoted"?, "folder", "category"?}`.
public struct FiledBookmark: Codable, Sendable {
    public let id: String
    public let text: String
    public let quoted: String?
    public let folder: String
    public let category: String?

    public init(id: String, text: String, quoted: String? = nil, folder: String, category: String? = nil) {
        self.id = id
        self.text = text
        self.quoted = quoted
        self.folder = folder
        self.category = category
    }

    public var bookmark: Bookmark { Bookmark(id: id, author: "", text: text, quotedText: quoted) }

    public static func load(jsonLines url: URL) throws -> [FiledBookmark] {
        try String(contentsOf: url, encoding: .utf8).split(separator: "\n").map {
            try JSONDecoder().decode(FiledBookmark.self, from: Data($0.utf8))
        }
    }
}

/// Sorts posts the way the person already does, learning from the ones they filed:
/// on-device embeddings, then a folder classifier and one category classifier per folder.
public final class LearnedBookmarkSorter: BookmarkSorting {
    public let embedder: any BookmarkEmbedding
    let scaler: FeatureScaler
    let folderModel: SoftmaxClassifier
    let categoryModels: [String: SoftmaxClassifier]
    let singleCategory: [String: String]

    /// Trains on `examples`. Embedding them is the slow part (5–30 ms each); `vectors` reuses earlier ones by id.
    public static func train(
        on examples: [FiledBookmark], embedder: any BookmarkEmbedding, vectors: [String: [Float]] = [:]
    ) async throws -> LearnedBookmarkSorter {
        var raw: [[Float]] = []
        for example in examples {
            if let cached = vectors[example.id] {
                raw.append(cached)
            } else {
                raw.append(try await embedder.embed(example.bookmark.classificationText))
            }
        }
        return try train(on: examples, rawVectors: raw, embedder: embedder)
    }

    /// Trains on precomputed `embedder` vectors, one per example, in order.
    public static func train(
        on examples: [FiledBookmark], rawVectors: [[Float]], embedder: any BookmarkEmbedding
    ) throws -> LearnedBookmarkSorter {
        let scaler = FeatureScaler(fitting: rawVectors)
        let features = rawVectors.map(scaler.transform)
        let folderModel = try SoftmaxClassifier.train(features: features, labels: examples.map(\.folder))
        var categoryModels: [String: SoftmaxClassifier] = [:]
        var singleCategory: [String: String] = [:]
        for folder in Set(examples.map(\.folder)) {
            let members = examples.indices.filter { examples[$0].folder == folder && examples[$0].category != nil }
            let names = Set(members.compactMap { examples[$0].category })
            if names.count == 1 {
                singleCategory[folder] = names.first
            } else if names.count > 1 {
                categoryModels[folder] = try SoftmaxClassifier.train(
                    features: members.map { features[$0] }, labels: members.map { examples[$0].category! })
            }
        }
        return LearnedBookmarkSorter(
            embedder: embedder, scaler: scaler, folderModel: folderModel, categoryModels: categoryModels,
            singleCategory: singleCategory)
    }

    init(
        embedder: any BookmarkEmbedding, scaler: FeatureScaler, folderModel: SoftmaxClassifier,
        categoryModels: [String: SoftmaxClassifier], singleCategory: [String: String]
    ) {
        self.embedder = embedder
        self.scaler = scaler
        self.folderModel = folderModel
        self.categoryModels = categoryModels
        self.singleCategory = singleCategory
    }

    public var folders: [String] { folderModel.classes }

    public func categories(in folder: String) -> [String] {
        categoryModels[folder]?.classes ?? singleCategory[folder].map { [$0] } ?? []
    }

    public func sort(_ bookmark: Bookmark) async throws -> SortDecision {
        let start = DispatchTime.now().uptimeNanoseconds
        let raw = try await embedder.embed(bookmark.classificationText)
        return sort(rawVector: raw, start: start)
    }

    /// Sorts an already embedded post.
    public func sort(rawVector: [Float], start: UInt64 = DispatchTime.now().uptimeNanoseconds) -> SortDecision {
        let feature = scaler.transform(rawVector)
        let folder = folderModel.predict(feature)
        var categories: [(name: String, probability: Float)] = []
        if let model = categoryModels[folder.label] {
            categories = model.predict(feature).ranked.map { ($0.label, $0.probability) }
        } else if let only = singleCategory[folder.label] {
            categories = [(only, 1)]
        }
        return SortDecision(
            folder: folder.label, folderConfidence: folder.confidence, categories: categories,
            milliseconds: Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
    }
}
