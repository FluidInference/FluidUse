import FluidUse
import Foundation

// Parity + latency check for the Swift host: replays a responses.jsonl (post, expected reply) through
// ShortReplyManager and reports exact matches and per-reply timing.
//   swift run -c release ShortReplyCheck <model dir> <responses.jsonl> [column] [limit]

let arguments = CommandLine.arguments
guard arguments.count >= 3 else {
    FileHandle.standardError.write(Data("usage: ShortReplyCheck <model dir> <responses.jsonl> [column] [limit]\n".utf8))
    exit(2)
}
let directory = URL(fileURLWithPath: arguments[1])
let column = arguments.count > 3 ? arguments[3] : "tuned"
let limit = arguments.count > 4 ? Int(arguments[4]) ?? .max : .max

struct Row: Decodable {
    let id: String
    let post: String
    let replies: [String: String]

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: AnyKey.self)
        id = try container.decode(String.self, forKey: AnyKey("id"))
        post = try container.decode(String.self, forKey: AnyKey("post"))
        var replies: [String: String] = [:]
        for key in container.allKeys where key.stringValue != "id" && key.stringValue != "post" {
            replies[key.stringValue] = try container.decode(String.self, forKey: key)
        }
        self.replies = replies
    }
}

struct AnyKey: CodingKey {
    var stringValue: String
    var intValue: Int? { nil }
    init(_ string: String) { stringValue = string }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { nil }
}

let rows = try String(contentsOf: URL(fileURLWithPath: arguments[2]), encoding: .utf8)
    .split(whereSeparator: \.isNewline).prefix(limit)
    .map { try JSONDecoder().decode(Row.self, from: Data($0.utf8)) }

@available(macOS 15.0, *)
func run() async throws {
    let manager = try await ShortReplyManager.load(from: directory)
    try await manager.warmUp()
    var matches = 0
    var totals: [Double] = []
    var prefills: [Double] = []
    var perToken: [Double] = []
    for (index, row) in rows.enumerated() {
        let draft = try await manager.draft(for: row.post)
        let expected = row.replies[column] ?? ""
        let same = draft.reply == expected
        matches += same ? 1 : 0
        totals.append(draft.timing.totalSeconds)
        prefills.append(draft.timing.prefillSeconds)
        if draft.timing.generatedTokens > 0 {
            perToken.append(draft.timing.decodeSeconds / Double(draft.timing.generatedTokens))
        }
        if !same {
            print("  differs #\(index + 1): swift=\(draft.reply.debugDescription) python=\(expected.debugDescription)")
        }
    }
    func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        return sorted[sorted.count / 2]
    }
    print("exact reply match: \(matches)/\(rows.count)")
    print(
        String(
            format: "prefill p50: %.1f ms   decode p50: %.1f ms/token   reply p50: %.0f ms",
            median(prefills) * 1000, median(perToken) * 1000, median(totals) * 1000))
}

if #available(macOS 15.0, *) {
    try await run()
} else {
    FileHandle.standardError.write(Data("ShortReplyCheck needs macOS 15\n".utf8))
    exit(2)
}
