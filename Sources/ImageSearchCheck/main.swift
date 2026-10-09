import FluidUse
import Foundation

/// Embeds images with EmbeddingGemma 2 and scores zero-shot Oxford Pets (photos cached by ImageSortDemo).
///
///     swift run -c release ImageSearchCheck [--budget=70|140|280] [--per-breed=10] [--dump=vectors.jsonl]
@main
struct ImageSearchCheck {
    static let breeds = [
        "abyssinian", "american bulldog", "american pit bull terrier", "basset hound", "beagle", "bengal", "birman",
        "bombay", "boxer", "british shorthair", "chihuahua", "egyptian mau", "english cocker spaniel",
        "english setter", "german shorthaired", "great pyrenees", "havanese", "japanese chin", "keeshond",
        "leonberger", "maine coon", "miniature pinscher", "newfoundland", "persian", "pomeranian", "pug", "ragdoll",
        "russian blue", "saint bernard", "samoyed", "scottish terrier", "shiba inu", "siamese", "sphynx",
        "staffordshire bull terrier", "wheaten terrier", "yorkshire terrier",
    ]

    static func main() async throws {
        let arguments = CommandLine.arguments.dropFirst()
        func value(_ name: String) -> String? {
            arguments.first { $0.hasPrefix("--\(name)=") }.map { String($0.dropFirst(name.count + 3)) }
        }
        let budget = value("budget").flatMap(Int.init).flatMap(EmbeddingGemma2Vision.Budget.init) ?? .fast
        let perBreed = value("per-breed").flatMap(Int.init) ?? 10
        let directory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FluidUse/image-sort/oxford-pets-test")
        let manifest = try JSONDecoder().decode(
            [String: Int].self, from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
        // Same sample as the PyTorch check: test split (ids < 3,669), the first `perBreed` ids of each breed.
        var byBreed: [Int: [Int]] = [:]
        for (key, label) in manifest { if let id = Int(key), id < 3669 { byBreed[label, default: []].append(id) } }
        let sample = (0..<breeds.count).flatMap { (byBreed[$0] ?? []).sorted().prefix(perBreed).map { ($0, $0) } }
            .map { id, _ in (id: id, label: manifest[String(id)]!) }

        var start = DispatchTime.now().uptimeNanoseconds
        let text = try await EmbeddingGemma2Manager.loadDefault()
        let vision = try await EmbeddingGemma2Vision.load(text: text)
        let warmup = try EmbeddingGemma2Vision.image(contentsOf: directory.appendingPathComponent("0.jpg"))
        _ = try await vision.embed(image: warmup, budget: budget)
        print(String(format: "models loaded in %.1f s", seconds(since: start)))

        start = DispatchTime.now().uptimeNanoseconds
        let images = try sample.map {
            try EmbeddingGemma2Vision.image(contentsOf: directory.appendingPathComponent("\($0.id).jpg"))
        }
        let decode = seconds(since: start)
        start = DispatchTime.now().uptimeNanoseconds
        let vectors = try await vision.embed(images: images, budget: budget)
        let embed = seconds(since: start)
        var labels: [[Float]] = []
        for breed in breeds { labels.append(try await text.embed("a photo of a \(breed), a type of pet.")) }
        var correct = 0
        for (item, vector) in zip(sample, vectors) {
            let scores = labels.map { zip($0, vector).reduce(Float(0)) { $0 + $1.0 * $1.1 } }
            if scores.indices.max(by: { scores[$0] < scores[$1] }) == item.label { correct += 1 }
        }
        print(
            String(
                format: "%d tokens: %d photos, decode %.1f s, embed %.1f s = %.0f images/s; zero-shot Pets %.1f%%",
                budget.rawValue, sample.count, decode, embed, Double(sample.count) / embed,
                100 * Double(correct) / Double(sample.count)))
        if let dump = value("dump") {
            let lines = zip(sample, vectors).map { item, vector in
                String(
                    decoding: try! JSONSerialization.data(withJSONObject: ["id": item.id, "v": vector]), as: UTF8.self)
            }
            try lines.joined(separator: "\n").write(toFile: dump, atomically: true, encoding: .utf8)
        }
    }

    static func seconds(since start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 }
}
