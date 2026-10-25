import FluidUse
import Foundation

/// A term that contributed to a match. `evoked` = the word never appears in the post itself.
struct MatchTerm: Hashable, Sendable {
    let word: String
    let evoked: Bool
}

struct SearchHit: Identifiable, Sendable {
    let tweet: Tweet
    let score: Float
    let terms: [MatchTerm]
    var id: Int { tweet.id }
}

enum SearchMode: String, CaseIterable, Sendable {
    case keyword = "Keyword"
    case evoke = "Evoke"
}

/// In-memory inverted indexes: BM25 over literal words, and a sparse dot product over Evoke atoms.
struct SearchIndex: Sendable {
    let tweets: [Tweet]
    private let semantic: [Int: [(doc: Int, weight: Float)]]
    private let words: [Int: String]
    private let lexical: [String: [(doc: Int, tf: Int)]]
    private let docLengths: [Int]
    private let lowercased: [String]

    init(tweets: [Tweet], atoms: [EvokeTerms], words: [Int: String]) {
        self.tweets = tweets
        self.words = words
        lowercased = tweets.map { $0.text.lowercased() }
        var semantic: [Int: [(doc: Int, weight: Float)]] = [:]
        for (doc, docAtoms) in atoms.enumerated() {
            for (id, weight) in docAtoms { semantic[id, default: []].append((doc, weight)) }
        }
        self.semantic = semantic
        var lexical: [String: [(doc: Int, tf: Int)]] = [:]
        var lengths: [Int] = []
        for (doc, text) in lowercased.enumerated() {
            let terms = Self.terms(text)
            lengths.append(terms.count)
            for (term, tf) in Dictionary(terms.map { ($0, 1) }, uniquingKeysWith: +) {
                lexical[term, default: []].append((doc, tf))
            }
        }
        self.lexical = lexical
        docLengths = lengths
    }

    private static let stopWords: Set<String> = [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "my", "of",
        "on", "or", "that", "the", "this", "to", "was", "what", "with", "your",
    ]

    static func terms(_ text: String) -> [String] {
        text.lowercased().split { !($0.isLetter || $0.isNumber) }.map(String.init)
            .filter { $0.count > 1 && !stopWords.contains($0) }
    }

    func keyword(_ query: String, limit: Int = 20) -> [SearchHit] {
        let n = Float(tweets.count)
        let avg = Float(docLengths.reduce(0, +)) / n
        var scores: [Int: Float] = [:]
        var matched: [Int: [String]] = [:]
        for term in Set(Self.terms(query)) {
            guard let postings = lexical[term] else { continue }
            let df = Float(postings.count)
            let idf = log(1 + (n - df + 0.5) / (df + 0.5))
            for (doc, tf) in postings {
                let f = Float(tf)
                let norm = f * 2.2 / (f + 1.2 * (0.25 + 0.75 * Float(docLengths[doc]) / avg))
                scores[doc, default: 0] += idf * norm
                matched[doc, default: []].append(term)
            }
        }
        return scores.sorted { $0.value > $1.value }.prefix(limit).map { doc, score in
            SearchHit(
                tweet: tweets[doc], score: score,
                terms: matched[doc, default: []].map { MatchTerm(word: $0, evoked: false) })
        }
    }

    func evoke(_ query: EvokeTerms, limit: Int = 8) -> [SearchHit] {
        var scores: [Int: Float] = [:]
        var contributions: [Int: [(id: Int, value: Float)]] = [:]
        for (id, qw) in query {
            guard let postings = semantic[id] else { continue }
            for (doc, dw) in postings {
                scores[doc, default: 0] += qw * dw
                contributions[doc, default: []].append((id, qw * dw))
            }
        }
        // Keep hits within reach of the best one, and chips that carry real weight.
        let best = scores.values.max() ?? 0
        return scores.filter { $0.value >= best * 0.25 }.sorted { $0.value > $1.value }.prefix(limit).map {
            doc, score in
            let top = contributions[doc, default: []].sorted { $0.value > $1.value }
            let floor = (top.first?.value ?? 0) * 0.15
            var seen = Set<String>()
            let terms = top.filter { $0.value >= floor }.compactMap { c -> MatchTerm? in
                guard let word = words[c.id], seen.insert(word).inserted else { return nil }
                return MatchTerm(word: word, evoked: !lowercased[doc].contains(word))
            }
            return SearchHit(tweet: tweets[doc], score: score, terms: Array(terms.prefix(5)))
        }
    }
}
