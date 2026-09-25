import Foundation

/// One Wikipedia abstract from the DBpedia-14 test split with its gold category.
public struct SortItem: Codable, Sendable, Identifiable, Hashable {
    public let id: Int
    public let title: String
    public let content: String
    /// Demo category name of the gold DBpedia class.
    public let gold: String

    public var text: String { title + "\n" + content }
}

/// Balanced, seeded sample of the DBpedia-14 test split (CC BY-SA 3.0, Wikipedia via DBpedia),
/// fetched from the Hugging Face dataset viewer API and cached locally. Nothing is bundled.
public enum DBpediaSample {
    public static let attribution =
        "DBpedia-14 test split (Zhang et al., 2015) · Wikipedia text via DBpedia · CC BY-SA 3.0"

    /// Short names for the 14 DBpedia classes, in dataset label order.
    public static let categories = [
        "company", "school", "artist", "athlete", "politician", "transportation", "building", "nature", "village",
        "animal", "plant", "album", "film", "book",
    ]

    static let rowsPerClass = 5000
    static let endpoint = "https://datasets-server.huggingface.co/rows"

    private struct Page: Decodable {
        struct Entry: Decodable {
            struct Row: Decodable {
                let label: Int
                let title: String
                let content: String
            }
            let rowIndex: Int
            let row: Row

            enum CodingKeys: String, CodingKey {
                case row
                case rowIndex = "row_idx"
            }
        }
        let rows: [Entry]
    }

    public static func cacheURL(count: Int, seed: UInt64) -> URL {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("FluidUse/sort-anything/dbpedia-test-\(count)-seed\(seed).json")
    }

    /// `count` items, as even across the 14 classes as possible, in a seeded shuffled order.
    public static func load(count: Int = 1000, seed: UInt64 = 0) async throws -> [SortItem] {
        let cache = cacheURL(count: count, seed: seed)
        if let data = try? Data(contentsOf: cache), let items = try? JSONDecoder().decode([SortItem].self, from: data),
            items.count == count
        {
            return items
        }
        var generator = SeededGenerator(seed: seed)
        let perClass = (count + categories.count - 1) / categories.count
        var items: [SortItem] = []
        for label in categories.indices {
            let offset = label * rowsPerClass + Int(generator.next() % UInt64(rowsPerClass - perClass))
            var components = URLComponents(string: endpoint)!
            components.queryItems = [
                URLQueryItem(name: "dataset", value: "fancyzhx/dbpedia_14"),
                URLQueryItem(name: "config", value: "dbpedia_14"),
                URLQueryItem(name: "split", value: "test"),
                URLQueryItem(name: "offset", value: String(offset)),
                URLQueryItem(name: "length", value: String(perClass)),
            ]
            let (data, response) = try await URLSession.shared.data(from: components.url!)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                throw URLError(.badServerResponse)
            }
            for entry in try JSONDecoder().decode(Page.self, from: data).rows where entry.row.label == label {
                items.append(
                    SortItem(
                        id: entry.rowIndex, title: entry.row.title.trimmingCharacters(in: .whitespaces),
                        content: entry.row.content.trimmingCharacters(in: .whitespaces), gold: categories[label]))
            }
        }
        items.shuffle(using: &generator)
        items = Array(items.prefix(count))
        try FileManager.default.createDirectory(
            at: cache.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(items).write(to: cache)
        return items
    }
}

/// xorshift64*, so a seed gives the same sample on every machine.
struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }

    mutating func next() -> UInt64 {
        state ^= state >> 12
        state ^= state << 25
        state ^= state >> 27
        return state &* 0x2545_F491_4F6C_DD1D
    }
}
