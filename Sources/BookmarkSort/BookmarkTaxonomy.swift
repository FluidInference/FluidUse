import Foundation

/// Folders and the categories inside them, as the person already uses them.
///
/// `name` is what gets written out; `label` is the short phrase the model scores (defaults to `name`).
/// A folder without categories is sorted into, but not subdivided.
public struct BookmarkTaxonomy: Codable, Sendable {
    public struct Category: Codable, Sendable, Hashable {
        public let name: String
        public let label: String?

        public init(name: String, label: String? = nil) {
            self.name = name
            self.label = label
        }

        public var modelLabel: String { label ?? name }
    }

    public struct Folder: Codable, Sendable, Hashable {
        public let name: String
        public let label: String?
        public let categories: [Category]

        public init(name: String, label: String? = nil, categories: [Category] = []) {
            self.name = name
            self.label = label
            self.categories = categories
        }

        public var modelLabel: String { label ?? name }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            name = try container.decode(String.self, forKey: .name)
            label = try container.decodeIfPresent(String.self, forKey: .label)
            categories = try container.decodeIfPresent([Category].self, forKey: .categories) ?? []
        }
    }

    /// Head question for picking a folder.
    public let folderTask: String
    /// Head question for picking a category inside a folder.
    public let categoryTask: String
    public let folders: [Folder]

    public init(folderTask: String = "topic", categoryTask: String = "theme", folders: [Folder]) {
        self.folderTask = folderTask
        self.categoryTask = categoryTask
        self.folders = folders
    }

    public static func load(_ url: URL) throws -> BookmarkTaxonomy {
        try JSONDecoder().decode(BookmarkTaxonomy.self, from: Data(contentsOf: url))
    }
}

extension BookmarkTaxonomy {
    /// A starter set for AI and ML bookmarks: one folder, scored as a flat list.
    public static let aiBookmarks = BookmarkTaxonomy(
        folderTask: "topic", categoryTask: "tweet topic",
        folders: [
            Folder(
                name: "AI bookmarks",
                categories: [
                    Category(name: "Speech & audio", label: "speech recognition, text to speech, audio models"),
                    Category(name: "Vision & multimodal", label: "image, video and vision-language models"),
                    Category(name: "Language models", label: "large language models and chatbots"),
                    Category(name: "Coding agents", label: "AI coding assistants and agents"),
                    Category(name: "On-device", label: "on-device AI on Apple silicon and phones"),
                    Category(name: "Research papers", label: "research paper and benchmark results"),
                    Category(name: "Industry news", label: "AI startups, funding and company news"),
                    Category(name: "Not AI", label: "something unrelated to AI"),
                ])
        ])
}
