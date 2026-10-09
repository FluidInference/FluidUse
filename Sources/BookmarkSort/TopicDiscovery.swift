import Accelerate
import Foundation

/// A discovered topic: the posts in it (indices into the input), a name, and subtopics for big topics.
public struct TopicNode: Identifiable, Sendable {
    /// Stable across re-sorts when `carryColors` matches the topic to an earlier one.
    public var id: String
    public var name: String
    public var members: [Int]
    public var children: [TopicNode]
    /// Unit-length mean of the members' vectors; new posts join the topic whose centre is nearest.
    public var centroid: [Float]
    /// Stable colour slot, carried across re-sorts by matching topics to the previous ones.
    public var colorIndex = 0

    /// `children`, or nil for a leaf (the shape `OutlineGroup` wants).
    public var subtopics: [TopicNode]? { children.isEmpty ? nil : children }
}

/// Sorts posts into topics, then splits big topics into subtopics, from their embeddings alone: spherical k-means
/// (cosine) per level, deterministic seeds. Names are phrases taken from the posts themselves, ranked by how close
/// the embedding model puts each phrase to the topic's centre and how far from its sibling topics.
public struct TopicDiscovery: Sendable {
    /// Posts a topic needs before it is split into subtopics.
    public var splitThreshold = 80
    public var maxDepth = 2
    public var seed: UInt64 = 7

    public init() {}

    /// Topics for unit-length `vectors`, largest first. `embedPhrase` embeds a candidate name (unit length);
    /// `phraseCache` keeps those vectors between calls so re-sorting a growing stream stays fast. `topicCount`
    /// fixes the number of top-level topics; otherwise it follows the post count.
    public func discover(
        vectors: [[Float]], texts: [String], members: [Int]? = nil, topicCount: Int? = nil, path: String = "",
        excluding parentWords: Set<String> = [], phraseCache: PhraseVectorCache = PhraseVectorCache(),
        embedPhrase: @escaping @Sendable (String) async throws -> [Float]
    ) async throws -> [TopicNode] {
        // Missing phrases are embedded 16 at a time; Core ML's async prediction keeps the Neural Engine busy.
        func phraseVectors(_ phrases: [String]) async throws -> [String: [Float]] {
            var result: [String: [Float]] = [:]
            var missing: [String] = []
            for phrase in phrases {
                if let cached = await phraseCache.vector(for: phrase) {
                    result[phrase] = cached
                } else {
                    missing.append(phrase)
                }
            }
            for start in stride(from: 0, to: missing.count, by: 16) {
                let chunk = missing[start..<min(start + 16, missing.count)]
                try await withThrowingTaskGroup(of: (String, [Float]).self) { group in
                    for phrase in chunk { group.addTask { (phrase, try await embedPhrase(phrase)) } }
                    for try await (phrase, vector) in group {
                        result[phrase] = vector
                        await phraseCache.store(vector, for: phrase)
                    }
                }
            }
            return result
        }

        func build(_ members: [Int], depth: Int, path: String, parentWords: Set<String>) async throws -> [TopicNode] {
            let count = members.count
            let k =
                depth == 0
                ? topicCount ?? min(8, max(3, Int((Double(count) / 60).rounded())))
                : min(6, max(2, Int((Double(count) / 45).rounded())))
            guard count > k else { return [] }
            let assignment = Self.sphericalKMeans(members.map { vectors[$0] }, k: k, seed: seed &+ UInt64(depth))
            var groups = [[Int]](repeating: [], count: k)
            for (offset, cluster) in assignment.enumerated() { groups[cluster].append(members[offset]) }
            groups = groups.filter { !$0.isEmpty }.sorted { $0.count > $1.count }
            let centroids = groups.map { Self.centroid($0.map { vectors[$0] }) }
            var nodes: [TopicNode] = []
            for (index, group) in groups.enumerated() {
                let siblings = centroids.enumerated().filter { $0.offset != index }.map(\.element)
                let phrases = try await name(
                    group, centroid: centroids[index], siblings: siblings, texts: texts, exclude: parentWords,
                    embed: phraseVectors)
                let id = "\(path)\(index)"
                var node = TopicNode(
                    id: id, name: phrases.isEmpty ? "Topic \(index + 1)" : phrases.joined(separator: " · "),
                    members: group, children: [], centroid: centroids[index])
                if depth + 1 < maxDepth, group.count >= splitThreshold {
                    node.children = try await build(
                        group, depth: depth + 1, path: id + ".", parentWords: parentWords.union(phrases))
                }
                nodes.append(node)
            }
            return nodes
        }
        return try await build(members ?? Array(vectors.indices), depth: 0, path: path, parentWords: parentWords)
    }

    /// Subtopics for one topic (2–6 by size), named without repeating the topic's own name.
    public func split(
        _ topic: TopicNode, vectors: [[Float]], texts: [String], phraseCache: PhraseVectorCache,
        embedPhrase: @escaping @Sendable (String) async throws -> [Float]
    ) async throws -> [TopicNode] {
        var single = self
        single.maxDepth = 1
        let words = Set(topic.name.components(separatedBy: " · "))
        let count = topic.members.count
        var children = try await single.discover(
            vectors: vectors, texts: texts, members: topic.members,
            topicCount: min(6, max(2, Int((Double(count) / 45).rounded()))), path: topic.id + ".", excluding: words,
            phraseCache: phraseCache, embedPhrase: embedPhrase)
        for index in children.indices { children[index].colorIndex = topic.colorIndex }
        return children
    }

    /// Adds post `index` to the nearest topic (and its nearest subtopic), nudging their centres toward it.
    /// Returns the path, top-level first.
    @discardableResult
    public static func assign(_ index: Int, vector: [Float], into topics: inout [TopicNode]) -> [TopicNode] {
        func nearest(_ nodes: [TopicNode]) -> Int? {
            nodes.indices.max { dot(nodes[$0].centroid, vector) < dot(nodes[$1].centroid, vector) }
        }
        func join(_ node: inout TopicNode) {
            let weight = Float(node.members.count)
            node.centroid = normalized(zip(node.centroid, vector).map { $0 * weight + $1 })
            node.members.append(index)
        }
        guard let top = nearest(topics) else { return [] }
        join(&topics[top])
        var path = [topics[top]]
        if let child = nearest(topics[top].children) {
            join(&topics[top].children[child])
            path.append(topics[top].children[child])
        }
        return path
    }

    /// Gives each new topic the colour and id of the most similar old one (greedy, centroid similarity above 0.8),
    /// so colours, the selection and renames follow a topic across re-sorts; the rest get unused colours and fresh
    /// ids. Children are re-pathed under their parent's id.
    public static func carryColors(from old: [TopicNode], to new: inout [TopicNode], paletteSize: Int) {
        var pairs: [(similarity: Float, new: Int, old: Int)] = []
        for (n, node) in new.enumerated() {
            for (o, previous) in old.enumerated() { pairs.append((dot(node.centroid, previous.centroid), n, o)) }
        }
        var matchedNew = Set<Int>()
        var matchedOld = Set<Int>()
        var usedColors = Set<Int>()
        for pair in pairs.sorted(by: { $0.similarity > $1.similarity })
        where pair.similarity > 0.8 && !matchedNew.contains(pair.new) && !matchedOld.contains(pair.old) {
            new[pair.new].colorIndex = old[pair.old].colorIndex
            new[pair.new].id = old[pair.old].id
            matchedNew.insert(pair.new)
            matchedOld.insert(pair.old)
            usedColors.insert(old[pair.old].colorIndex)
        }
        var usedIDs = Set(new.indices.filter { matchedNew.contains($0) }.map { new[$0].id })
        var next = 0
        var serial = 0
        for index in new.indices where !matchedNew.contains(index) {
            while usedColors.contains(next % paletteSize), usedColors.count < paletteSize { next += 1 }
            new[index].colorIndex = next % paletteSize
            usedColors.insert(new[index].colorIndex)
            while usedIDs.contains("t\(serial)") || old.contains(where: { $0.id == "t\(serial)" }) { serial += 1 }
            new[index].id = "t\(serial)"
            usedIDs.insert(new[index].id)
        }
        for index in new.indices {
            for child in new[index].children.indices {
                new[index].children[child].id = "\(new[index].id).\(child)"
            }
        }
    }

    static func normalized(_ vector: [Float]) -> [Float] {
        let norm = vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
        return norm > 0 ? vector.map { $0 / norm } : vector
    }

    /// Up to two phrases from the group's posts: frequent ones, ranked by similarity to the group centre minus
    /// half the similarity to the nearest sibling, skipping near-duplicates of phrases already picked.
    private func name(
        _ group: [Int], centroid: [Float], siblings: [[Float]], texts: [String], exclude: Set<String>,
        embed: ([String]) async throws -> [String: [Float]]
    ) async throws -> [String] {
        let excluded = Set(exclude.map { $0.lowercased() })
        var counts: [String: Int] = [:]
        for member in group {
            for phrase in Set(Self.phrases(in: texts[member])) where !excluded.contains(phrase) {
                counts[phrase, default: 0] += 1
            }
        }
        let minimum = max(2, Int(0.05 * Double(group.count)))
        let candidates = counts.filter { $0.value >= minimum }
            .sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
            .prefix(80).map(\.key)
        let vectors = try await embed(Array(candidates))
        var scored: [(phrase: String, vector: [Float], score: Float)] = []
        for phrase in candidates {
            guard let vector = vectors[phrase] else { continue }
            let nearestSibling = siblings.map { Self.dot(vector, $0) }.max() ?? 0
            scored.append((phrase, vector, Self.dot(vector, centroid) - 0.5 * nearestSibling))
        }
        var picked: [(phrase: String, vector: [Float])] = []
        for candidate in scored.sorted(by: { $0.score > $1.score }) {
            let words = Set(candidate.phrase.split(separator: " "))
            guard
                picked.allSatisfy({
                    Self.dot($0.vector, candidate.vector) < 0.8
                        && words.isDisjoint(with: $0.phrase.split(separator: " "))
                })
            else { continue }
            picked.append((candidate.phrase, candidate.vector))
            if picked.count == 2 { break }
        }
        return picked.map { $0.phrase.prefix(1).uppercased() + $0.phrase.dropFirst() }
    }

    private static let stopWords: Set<String> = Set(
        """
        the a an and or of to in on for is are was were be been it this that with as at by from we our you your i my me \
        they their he she his her its not but if so do does did have has had just more most than then there what which \
        who how all any can will would should could about into out up over new now one also like get got make made via \
        use using very really much many some here when where why them only even still because don't it's i'm you're \
        people quote replying show today tonight tip day week finally again still always every never actually honestly \
        somehow something whole watched maybe way happy worth month apparently literally genuinely thing things \
        anyone everyone someone feel feels felt time times year years last first next
        """.split(whereSeparator: \.isWhitespace).map(String.init))

    /// Candidate names in `text`: lowercased Latin runs of 1–2 words inside one clause (no stop word at either
    /// end), and 2–4 character runs of Chinese.
    static func phrases(in text: String) -> [String] {
        let cleaned = text.replacingOccurrences(of: #"@\w+|https?://\S+|#"#, with: " ", options: .regularExpression)
        var result: [String] = []
        for clause in cleaned.split(whereSeparator: { ".,!?:;()[]\"“”—–|/\n".contains($0) }) {
            let words = clause.matches(of: /[A-Za-z][A-Za-z0-9\-]+/).map { String($0.output).lowercased() }
            for length in 1...2 where words.count >= length {
                for start in 0...(words.count - length) {
                    let gram = Array(words[start..<(start + length)])
                    guard !stopWords.contains(gram.first!), !stopWords.contains(gram.last!),
                        gram.allSatisfy({ $0.count >= 3 })
                    else { continue }
                    result.append(gram.joined(separator: " "))
                }
            }
        }
        for run in cleaned.matches(of: /[一-鿿]{2,}/) {
            let characters = Array(run.output)
            for length in 2...4 where characters.count >= length {
                for start in 0...(characters.count - length) {
                    result.append(String(characters[start..<(start + length)]))
                }
            }
        }
        return result
    }

    /// Lloyd's algorithm on the unit sphere with k-means++ seeding; best of several restarts by total similarity.
    /// Similarities are one matrix multiply per iteration (Accelerate).
    static func sphericalKMeans(_ vectors: [[Float]], k: Int, seed: UInt64, restarts: Int = 8) -> [Int] {
        let n = vectors.count
        let d = vectors.first?.count ?? 0
        let points = vectors.flatMap { $0 }
        var generator = SplitMix64(seed: seed)
        var best: (assignment: [Int], score: Float) = ([], -.infinity)
        var similarities = [Float](repeating: 0, count: n * k)
        func similarity(to centres: [Float], count: Int) {
            // [n × d] · [count × d]ᵀ → [n × count]
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans, Int32(n), Int32(count), Int32(d), 1, points, Int32(d),
                centres, Int32(d), 0, &similarities, Int32(count))
        }
        for _ in 0..<restarts {
            var centres = [Float](repeating: 0, count: k * d)
            let first = Int.random(in: 0..<n, using: &generator)
            centres.replaceSubrange(0..<d, with: points[(first * d)..<((first + 1) * d)])
            var closest = [Float](repeating: -1, count: n)
            for chosenCount in 1..<k {
                var latest = [Float](centres[((chosenCount - 1) * d)..<(chosenCount * d)])
                var column = [Float](repeating: 0, count: n)
                cblas_sgemv(
                    CblasRowMajor, CblasNoTrans, Int32(n), Int32(d), 1, points, Int32(d), &latest, 1, 0, &column, 1)
                for index in 0..<n { closest[index] = max(closest[index], column[index]) }
                let distances = closest.map { max(0, 1 - $0) }
                var target = Float.random(
                    in: 0..<max(distances.reduce(0, +), .leastNonzeroMagnitude), using: &generator)
                var chosen = n - 1
                for (index, distance) in distances.enumerated() {
                    target -= distance
                    if target <= 0 {
                        chosen = index
                        break
                    }
                }
                centres.replaceSubrange(
                    (chosenCount * d)..<((chosenCount + 1) * d), with: points[(chosen * d)..<((chosen + 1) * d)])
            }
            var assignment = [Int](repeating: -1, count: n)
            var score: Float = 0
            for _ in 0..<50 {
                similarity(to: centres, count: k)
                score = 0
                var changed = false
                for index in 0..<n {
                    var bestCluster = 0
                    var bestSimilarity = -Float.infinity
                    for cluster in 0..<k where similarities[index * k + cluster] > bestSimilarity {
                        bestSimilarity = similarities[index * k + cluster]
                        bestCluster = cluster
                    }
                    if assignment[index] != bestCluster {
                        assignment[index] = bestCluster
                        changed = true
                    }
                    score += bestSimilarity
                }
                if !changed { break }
                var sums = [Float](repeating: 0, count: k * d)
                for index in 0..<n {
                    let offset = assignment[index] * d
                    for dimension in 0..<d { sums[offset + dimension] += points[index * d + dimension] }
                }
                for cluster in 0..<k {
                    let row = Array(sums[(cluster * d)..<((cluster + 1) * d)])
                    let norm = sqrt(row.reduce(0) { $0 + $1 * $1 })
                    if norm > 0 { for dimension in 0..<d { centres[cluster * d + dimension] = row[dimension] / norm } }
                }
            }
            if score > best.score { best = (assignment, score) }
        }
        return best.assignment
    }

    static func centroid(_ vectors: [[Float]]) -> [Float] {
        guard let first = vectors.first else { return [] }
        var sum = [Float](repeating: 0, count: first.count)
        for vector in vectors { for index in sum.indices { sum[index] += vector[index] } }
        return normalized(sum)
    }

    static func dot(_ a: [Float], _ b: [Float]) -> Float {
        var total: Float = 0
        for index in 0..<min(a.count, b.count) { total += a[index] * b[index] }
        return total
    }
}

/// Phrase vectors already computed, shared across re-sorts and splits.
public actor PhraseVectorCache {
    private var vectors: [String: [Float]] = [:]

    public init() {}

    func vector(for phrase: String) -> [Float]? { vectors[phrase] }
    func store(_ vector: [Float], for phrase: String) { vectors[phrase] = vector }
}

/// Small deterministic generator so the same posts always give the same topics.
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }
}
