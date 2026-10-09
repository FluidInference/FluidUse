import Foundation

/// One saved post, reduced to what the sorter reads.
public struct Bookmark: Codable, Sendable, Identifiable, Hashable {
    public let id: String
    public let author: String
    public let createdAt: String
    public let text: String
    /// Text of the quoted post, if any.
    public let quotedText: String?
    /// Media kinds in the post (`photo`, `video`, `animated_gif`).
    public let media: [String]
    /// X's bookmark order; larger is more recent.
    public let sortIndex: String?

    public init(
        id: String, author: String, createdAt: String = "", text: String, quotedText: String? = nil,
        media: [String] = [], sortIndex: String? = nil
    ) {
        self.id = id
        self.author = author
        self.createdAt = createdAt
        self.text = text
        self.quotedText = quotedText
        self.media = media
        self.sortIndex = sortIndex
    }

    /// Post text followed by the quoted post, with `t.co` links removed.
    public var classificationText: String {
        var parts = [Self.stripLinks(text)]
        if let quotedText, !quotedText.isEmpty { parts.append(Self.stripLinks(quotedText)) }
        return parts.filter { !$0.isEmpty }.joined(separator: "\n")
    }

    static func stripLinks(_ text: String) -> String {
        text.replacingOccurrences(of: #"https://t\.co/\S+"#, with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Bookmarks exported by the xarchive browser extension (`xarchive_<user>_<date>.json`).
public enum XArchiveExport {
    struct File: Decodable {
        struct Entry: Decodable {
            struct Author: Decodable {
                let screenName: String

                enum CodingKeys: String, CodingKey { case screenName = "screen_name" }
            }
            struct Media: Decodable { let type: String }
            struct Quoted: Decodable {
                let fullText: String?

                enum CodingKeys: String, CodingKey { case fullText = "full_text" }
            }

            let tweetId: String
            let sortIndex: String?
            let status: String?
            let createdAt: String?
            let fullText: String?
            let author: Author?
            let media: [Media]?
            let quotedTweet: Quoted?

            enum CodingKeys: String, CodingKey {
                case status, author, media
                case tweetId = "tweet_id"
                case sortIndex = "sort_index"
                case createdAt = "created_at"
                case fullText = "full_text"
                case quotedTweet = "quoted_tweet"
            }
        }

        let bookmarks: [Entry]
    }

    /// Available bookmarks in `data`, in export order. Deleted or protected posts are skipped.
    public static func decode(_ data: Data) throws -> [Bookmark] {
        try JSONDecoder().decode(File.self, from: data).bookmarks.compactMap { entry in
            guard entry.status == nil || entry.status == "available", let text = entry.fullText else { return nil }
            return Bookmark(
                id: entry.tweetId, author: entry.author?.screenName ?? "", createdAt: entry.createdAt ?? "",
                text: text, quotedText: entry.quotedTweet?.fullText, media: entry.media?.map(\.type) ?? [],
                sortIndex: entry.sortIndex)
        }
    }

    public static func load(_ url: URL) throws -> [Bookmark] {
        try decode(Data(contentsOf: url))
    }
}
