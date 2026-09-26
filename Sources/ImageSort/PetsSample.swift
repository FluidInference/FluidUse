import Foundation

/// One photo from the Oxford-IIIT Pets test split with its gold breed.
public struct PetItem: Codable, Sendable, Identifiable, Hashable {
    public let id: Int
    public let breed: String
    /// Cached JPEG on disk.
    public let file: URL
}

/// Seeded sample of the Oxford-IIIT Pets test split (CC BY-SA 4.0), fetched from the Hugging Face dataset viewer
/// API on first use and cached locally. Nothing is bundled.
public enum PetsSample {
    public static let attribution = "Oxford-IIIT Pets (Parkhi et al., 2012) · CC BY-SA 4.0"
    public static let testCount = 3669

    /// The 37 breeds in dataset label order.
    public static let breeds = [
        "abyssinian", "american bulldog", "american pit bull terrier", "basset hound", "beagle", "bengal", "birman",
        "bombay", "boxer", "british shorthair", "chihuahua", "egyptian mau", "english cocker spaniel",
        "english setter", "german shorthaired", "great pyrenees", "havanese", "japanese chin", "keeshond",
        "leonberger", "maine coon", "miniature pinscher", "newfoundland", "persian", "pomeranian", "pug", "ragdoll",
        "russian blue", "saint bernard", "samoyed", "scottish terrier", "shiba inu", "siamese", "sphynx",
        "staffordshire bull terrier", "wheaten terrier", "yorkshire terrier",
    ]

    /// Prompt per breed, as scored in the mobius Pets check.
    public static func prompt(for breed: String) -> String { "a photo of a \(breed), a type of pet." }

    static let endpoint = "https://datasets-server.huggingface.co/rows"
    static let pageSize = 100

    private struct Page: Decodable {
        struct Entry: Decodable {
            struct Row: Decodable {
                struct Image: Decodable { let src: String }
                let image: Image
                let label: Int
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

    public static func cacheDirectory() -> URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FluidUse/image-sort/oxford-pets-test")
    }

    /// `count` photos in a seeded shuffled order (every eligible photo when `count` is nil); `testOnly` restricts
    /// the pool to the 3,669 test photos. The viewer API fetches the
    /// 3,669 test photos; a cache that also holds the train split (ids from 3,669) samples from both.
    public static func load(
        count: Int? = 1000, seed: UInt64 = 0, testOnly: Bool = false,
        progress: (@Sendable (Int, Int) -> Void)? = nil
    ) async throws -> [PetItem] {
        let directory = cacheDirectory()
        let manifestURL = directory.appendingPathComponent("manifest.json")
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)

        var labels: [Int: Int] = [:]
        if let data = try? Data(contentsOf: manifestURL),
            let saved = try? JSONDecoder().decode([Int: Int].self, from: data),
            saved.count >= testCount
        {
            labels = saved
        }
        var sources: [Int: String] = [:]
        if labels.count < testCount {
            for offset in stride(from: 0, to: testCount, by: pageSize) {
                for entry in try await page(offset: offset) {
                    labels[entry.rowIndex] = entry.row.label
                    sources[entry.rowIndex] = entry.row.image.src
                }
            }
            try JSONEncoder().encode(labels).write(to: manifestURL)
        }

        var generator = SeededGenerator(seed: seed)
        let pool = labels.keys.filter { !testOnly || $0 < testCount }.sorted()
        let chosen = Array(pool.shuffled(using: &generator).prefix(count ?? pool.count))
        let missing = chosen.filter { !manager.fileExists(atPath: file(for: $0).path) }
        if !missing.isEmpty {
            if sources.isEmpty {
                for offset in stride(from: 0, to: testCount, by: pageSize) {
                    for entry in try await page(offset: offset) { sources[entry.rowIndex] = entry.row.image.src }
                }
            }
            try await download(
                missing, sources: sources, done: chosen.count - missing.count, total: chosen.count, progress)
        }
        return chosen.map { PetItem(id: $0, breed: breeds[labels[$0]!], file: file(for: $0)) }
    }

    static func file(for row: Int) -> URL { cacheDirectory().appendingPathComponent("\(row).jpg") }

    private static func page(offset: Int) async throws -> [Page.Entry] {
        var components = URLComponents(string: endpoint)!
        components.queryItems = [
            URLQueryItem(name: "dataset", value: "timm/oxford-iiit-pet"),
            URLQueryItem(name: "config", value: "default"),
            URLQueryItem(name: "split", value: "test"),
            URLQueryItem(name: "offset", value: String(offset)),
            URLQueryItem(name: "length", value: String(pageSize)),
        ]
        let data = try await fetch(components.url!)
        return try JSONDecoder().decode(Page.self, from: data).rows
    }

    /// GET with backoff: the dataset viewer answers bursts with 429 and a `Retry-After`.
    static func fetch(_ url: URL, attempts: Int = 8) async throws -> Data {
        var delay = 2.0
        for attempt in 1...attempts {
            let (data, response) = try await URLSession.shared.data(from: url)
            let http = response as? HTTPURLResponse
            if http?.statusCode == 200 { return data }
            guard attempt < attempts, http?.statusCode == 429 || (http?.statusCode ?? 0) >= 500 else { break }
            let wait = (http?.value(forHTTPHeaderField: "Retry-After")).flatMap(Double.init) ?? delay
            try await Task.sleep(for: .seconds(min(wait, 60)))
            delay *= 2
        }
        throw URLError(.badServerResponse)
    }

    private static func download(
        _ rows: [Int], sources: [Int: String], done: Int, total: Int, _ progress: (@Sendable (Int, Int) -> Void)?
    ) async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            var next = 0
            var finished = done
            func launch() {
                guard next < rows.count else { return }
                let row = rows[next]
                next += 1
                group.addTask {
                    guard let source = sources[row], let url = URL(string: source) else { throw URLError(.badURL) }
                    try await fetch(url).write(to: file(for: row), options: .atomic)
                }
            }
            for _ in 0..<8 { launch() }
            while try await group.next() != nil {
                finished += 1
                progress?(finished, total)
                launch()
            }
        }
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
