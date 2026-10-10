import Foundation
import FluidUse

// ClefTextCheck <bundle> <requests.jsonl> (string states): token count + per-question probabilities for each request (one JSON line
// out per request), so the Python Core ML / PyTorch pipeline can be diffed against the Swift host.
guard CommandLine.arguments.count == 3 else {
    print("usage: ClefTextCheck <bundle dir> <requests.jsonl>")
    exit(2)
}
@available(macOS 15.0, *)
func run() async throws {
    let bundle = URL(fileURLWithPath: CommandLine.arguments[1])
    let manager = try await ClefTextManager.load(from: bundle, buckets: [256, 512, 1024, 2048])
    let lines = try String(contentsOfFile: CommandLine.arguments[2], encoding: .utf8).split(separator: "\n")
    for line in lines {
        guard let request = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
            let state = request["state"] as? String, let raw = request["questions"] as? [[Any]]
        else { continue }
        var questions: [(id: String, question: ClefQuestion)] = []
        for item in raw {
            guard let id = item[0] as? String, let q = item[1] as? [String: Any], let type = q["type"] as? String else {
                continue
            }
            let instructions = q["instructions"] as? String
            switch type {
            case "choice":
                questions.append(
                    (id, .choice(instructions: instructions, criteria: q["criteria"] as? [String: String] ?? [:])))
            case "score":
                questions.append((id, .score(instructions: instructions, criteria: q["criteria"] as? [String] ?? [])))
            default: questions.append((id, .noul(instructions: instructions)))
            }
        }
        let record = try await manager.encode(state: state, questions: questions)
        let result = try await manager.answer(state: state, questions: questions)
        let answers = Dictionary(
            uniqueKeysWithValues: result.answers.map {
                ($0.questionID, Dictionary(uniqueKeysWithValues: zip($0.optionIDs, $0.probabilities.map(Double.init))))
            })
        let out: [String: Any] = [
            "tokens": record.inputIDs.count, "ids_head": Array(record.inputIDs.prefix(40)), "answers": answers,
            "ms": result.totalMilliseconds,
        ]
        print(String(data: try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys]), encoding: .utf8)!)
    }
}

if #available(macOS 15.0, *) {
    try await run()
} else {
    fatalError("needs macOS 15")
}
