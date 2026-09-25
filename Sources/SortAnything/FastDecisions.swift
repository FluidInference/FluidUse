import FluidUse
import Foundation

/// One Fast Decisions row: a document and the decisions a product has to make about it.
public struct DecisionDocument: Codable, Sendable, Identifiable, Hashable {
    public struct Head: Codable, Sendable, Hashable {
        public let task: String
        public let labels: [String]
        /// Gold labels; one for single-label heads.
        public let gold: [String]
        public let multiLabel: Bool
    }

    public let id: String
    public let domain: String
    public let input: String
    public let heads: [Head]
}

/// Fastino's Fast Decisions development split (Apache-2.0): 17 domains × 100 rows, fetched from Hugging Face at a
/// pinned revision and cached. The published benchmark numbers use a held-out test split that is not public.
public enum FastDecisions {
    public static let attribution = "fastino/fast-decisions, development split · Apache-2.0"
    public static let revision = "1a33070cabf94ce2e29105482dd2ef6c157ad7f2"
    public static let domains = [
        "support_intent", "support_topic", "document_type", "review_sentiment", "agent_handoff", "email_triage",
        "ticket_route", "product_feedback", "banking_intent", "clinic_request", "travel_request", "news_topic",
        "paper_field", "sports_recap", "restaurant_review", "benefits_request", "screen_tags",
    ]

    private struct Row: Decodable {
        struct Output: Decodable {
            struct Classification: Decodable {
                let task: String
                let trueLabel: [String]
                let labels: [String]
                let multiLabel: Bool?

                enum CodingKeys: String, CodingKey {
                    case task, labels
                    case trueLabel = "true_label"
                    case multiLabel = "multi_label"
                }
            }
            let classifications: [Classification]
        }
        let input: String
        let output: Output
    }

    public static func load() async throws -> [DecisionDocument] {
        let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FluidUse/sort-decisions/fast-decisions-\(revision.prefix(8)).json")
        if let data = try? Data(contentsOf: cache),
            let documents = try? JSONDecoder().decode([DecisionDocument].self, from: data)
        {
            return documents
        }
        var documents: [DecisionDocument] = []
        for domain in domains {
            let url = URL(
                string: "https://huggingface.co/datasets/fastino/fast-decisions/resolve/\(revision)/\(domain).jsonl")!
            let (data, response) = try await URLSession.shared.data(from: url)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
            for (index, line) in data.split(separator: UInt8(ascii: "\n")).enumerated() where !line.isEmpty {
                let row = try JSONDecoder().decode(Row.self, from: Data(line))
                documents.append(
                    DecisionDocument(
                        id: "\(domain)-\(index)", domain: domain, input: row.input,
                        heads: row.output.classifications.map {
                            .init(
                                task: $0.task, labels: $0.labels, gold: $0.trueLabel,
                                multiLabel: $0.multiLabel ?? false)
                        }))
            }
        }
        try FileManager.default.createDirectory(
            at: cache.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(documents).write(to: cache)
        return documents
    }
}

/// Answers every head of a document in one GLiNER2.5-Decide call (fp16 256-token package).
public final class DecisionSorter: Sendable {
    public struct Answer: Sendable, Hashable {
        public let task: String
        public let label: String
        public let confidence: Float
        public let gold: [String]
        /// Fast Decisions card scoring: the prediction and gold compared as sets.
        public var correct: Bool { gold == [label] }
    }

    public struct Result: Sendable {
        public let answers: [Answer]
        public let milliseconds: Double
        public let truncated: Bool
    }

    private let manager: GLiNER2Manager

    private init(manager: GLiNER2Manager) { self.manager = manager }

    public static func load(progress: GLiNER2ModelStore.Progress? = nil) async throws -> DecisionSorter {
        let manager: GLiNER2Manager
        if let path = ProcessInfo.processInfo.environment["GLINER2_DECIDE_MODEL_DIR"], !path.isEmpty {
            manager = try await GLiNER2Manager.load(from: URL(fileURLWithPath: path), variant: .decideLong)
        } else {
            manager = try await GLiNER2Manager.load(variant: .decideLong, progress: progress)
        }
        return DecisionSorter(manager: manager)
    }

    /// All heads in one call; trailing words are dropped until the request fits 256 tokens.
    public func decide(_ document: DecisionDocument) async throws -> Result {
        let start = DispatchTime.now().uptimeNanoseconds
        let heads = document.heads.map { (task: $0.task, labels: $0.labels) }
        var words = document.input.split(separator: " ", omittingEmptySubsequences: false)
        var truncated = false
        while true {
            do {
                let answers = try await manager.classifyConcurrently(
                    text: words.joined(separator: " "), heads: heads)
                return Result(
                    answers: zip(document.heads, answers).map { head, answer in
                        Answer(
                            task: head.task, label: answer.selectedLabel,
                            confidence: answer.probabilities[answer.selectedIndex], gold: head.gold.sorted())
                    },
                    milliseconds: Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6, truncated: truncated)
            } catch GLiNER2Error.invalidInput(let reason) where reason.contains("tokens") && words.count > 8 {
                words.removeLast(max(1, words.count / 8))
                truncated = true
            }
        }
    }
}
