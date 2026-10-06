import CryptoKit
import FluidUse
import Foundation

/// One issue as shown on the page. The real author is replaced by a stable made-up handle; the model state is built
/// once at load.
struct Issue: Identifiable, Sendable {
    let number: Int
    let title: String
    let isOpen: Bool
    let author: String
    let created: String
    let comments: Int
    let state: OrderedJSON

    var id: Int { number }

    /// Loads `gh issue list … --json number,title,body,labels,state,author,createdAt,comments` output, newest first.
    static func load(from url: URL) throws -> [Issue] {
        guard let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [[String: Any]] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let iso = ISO8601DateFormatter()
        let day = DateFormatter()
        day.locale = Locale(identifier: "en_US")
        day.dateFormat = "MMM d, yyyy"
        return raw.compactMap { item -> Issue? in
            guard let number = item["number"] as? Int, let title = item["title"] as? String else { return nil }
            let login = (item["author"] as? [String: Any])?["login"] as? String ?? "ghost"
            let createdAt = item["createdAt"] as? String ?? ""
            return Issue(
                number: number, title: title, isOpen: (item["state"] as? String) == "OPEN",
                author: Issue.handle(login),
                created: iso.date(from: createdAt).map { day.string(from: $0) } ?? createdAt,
                comments: (item["comments"] as? [Any])?.count ?? 0,
                state: .object([
                    ("repository", .string("vllm-project/semantic-router")),
                    ("issue_title", .string(title)),
                    ("issue_body", .string(Issue.clean(item["body"] as? String ?? ""))),
                ]))
        }
        .sorted { $0.number > $1.number }
    }

    /// triage_core.clean: drop HTML comments, collapse 3+ newlines, strip, keep the first 700 code points.
    static func clean(_ body: String, limit: Int = 700) -> String {
        // `.` must cross newlines (Python re.S), set with the inline flag.
        var text = body.replacingOccurrences(of: "(?s)<!--.*?-->", with: "", options: .regularExpression)
        text = text.replacingOccurrences(of: "\n{3,}", with: "\n\n", options: .regularExpression)
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return String(String.UnicodeScalarView(text.unicodeScalars.prefix(limit)))
    }

    static let adjectives = [
        "swift", "quiet", "brave", "lucky", "cosmic", "rusty", "sunny", "clever", "mellow", "pixel", "fuzzy", "neon",
    ]
    static let nouns = [
        "otter", "falcon", "maple", "comet", "badger", "lynx", "cactus", "harbor", "pebble", "willow", "koala", "ember",
    ]

    /// server.handle: the SHA-256 of the login as a 256-bit integer h → "{ADJ[h % 12]}-{NOUN[(h // 12) % 12]}{h % 97}".
    static func handle(_ login: String) -> String {
        let digest = Array(SHA256.hash(data: Data(login.utf8)))
        func mod(_ m: Int) -> Int { digest.reduce(0) { ($0 * 256 + Int($1)) % m } }
        let h144 = mod(144)
        return "\(adjectives[h144 % 12])-\(nouns[h144 / 12])\(mod(97))"
    }
}

/// The repository's label colors (labels.json), for the labels the model can give.
enum LabelPalette {
    static let colors: [String: String] = [
        "bug": "d73a4a", "enhancement": "a2eeef", "documentation": "ededed", "question": "d876e3",
        "proposal": "5319E7",
        "wg/mom-routing": "7B3FE4", "wg/router-models-inference-runtime": "0E8A16",
        "wg/data-plane-networking": "0366D6", "wg/enterprise-environment": "D93F0B",
        "wg/evaluation-quality": "e247cd", "wg/developer-experience-ecosystem": "1D76DB",
        "wg/agentic-context": "8250DF", "owner/maintainers": "5319E7", "wg/platform-operations": "D93F0B",
        "priority/P0": "b60205", "priority/P1": "5f9df6", "priority/P2": "e4f815",
        "needs-info": "D876E3", "good first issue": "7057ff",
    ]

    /// Sidebar workgroups, in labels.json order (ties keep this order).
    static let workgroups = [
        "wg/mom-routing", "wg/router-models-inference-runtime", "wg/data-plane-networking",
        "wg/enterprise-environment", "wg/evaluation-quality", "wg/developer-experience-ecosystem",
        "wg/agentic-context", "owner/maintainers", "wg/platform-operations",
    ]

    static func rgb(_ name: String) -> (Double, Double, Double) {
        let hex = UInt32(colors[name] ?? "8b949e", radix: 16) ?? 0x8b949e
        return (Double((hex >> 16) & 0xff) / 255, Double((hex >> 8) & 0xff) / 255, Double(hex & 0xff) / 255)
    }
}
