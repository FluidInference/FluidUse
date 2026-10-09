import Foundation

/// Colored stdout log for the side terminal (`demo.sh` tails it under macmon): one line per encoded keystroke.
enum DemoLog {
    private static let dim = "\u{1B}[2m"
    private static let bold = "\u{1B}[1m"
    private static let cyan = "\u{1B}[36m"
    private static let green = "\u{1B}[32m"
    private static let amber = "\u{1B}[33m"
    private static let reset = "\u{1B}[0m"

    static func loaded(posts: Int) {
        print("\(bold)evoke\(reset) Granite-Embedding-30M-Sparse · Core ML fp16 L64 · Neural Engine · \(posts) posts")
    }

    static func indexing(_ done: Int, of total: Int) {
        print("\(dim)index \(done)/\(total)\(reset)")
    }

    static func indexed(posts: Int, seconds: Double, termsPerPost: Double) {
        print(
            String(
                format: "\(green)indexed\(reset) %d posts in %.2f s (%.2f ms/post) · %.0f terms/post", posts, seconds,
                seconds * 1000 / Double(max(posts, 1)), termsPerPost))
    }

    static func search(
        _ query: String, ms: Double, terms: [(word: String, weight: Float, evoked: Bool)], hits: [SearchHit]
    ) {
        let expansion = terms.prefix(6).map { $0.evoked ? "\(amber)\($0.word)\(reset)" : $0.word }
            .joined(separator: " ")
        let top = hits.first.map { "@\($0.tweet.handle)" } ?? "-"
        let padded = query.count < 24 ? query + String(repeating: " ", count: 24 - query.count) : query
        print(
            String(format: "\(cyan)%6.2f ms\(reset) ANE  ", ms) + "\(bold)\(padded)\(reset) → \(expansion)"
                + "  \(dim)\(hits.count) hits · top \(top)\(reset)")
    }
}
