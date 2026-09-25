import Foundation
import SortAnything

/// Headless Sort Decisions: every Fast Decisions document, all of its heads in one GLiNER2.5-Decide call.
///
///     swift run -c release SortDecisionsCheck [--inflight=4] [--dump=path.jsonl]
@main
struct SortDecisionsCheck {
    static func main() async throws {
        let arguments = CommandLine.arguments.dropFirst()
        func value(_ name: String) -> String? {
            arguments.first { $0.hasPrefix("--\(name)=") }.map { String($0.dropFirst(name.count + 3)) }
        }
        let inflight = max(1, value("inflight").flatMap(Int.init) ?? 4)
        let documents = try await FastDecisions.load()
        let sorter = try await DecisionSorter.load()
        _ = try await sorter.decide(documents[0])

        let wall = DispatchTime.now().uptimeNanoseconds
        let results = try await withThrowingTaskGroup(of: (Int, DecisionSorter.Result).self) { group in
            var results = [DecisionSorter.Result?](repeating: nil, count: documents.count)
            var next = 0
            func launch() {
                guard next < documents.count else { return }
                let index = next
                next += 1
                group.addTask { (index, try await sorter.decide(documents[index])) }
            }
            for _ in 0..<inflight { launch() }
            while let (index, result) = try await group.next() {
                results[index] = result
                launch()
            }
            return results.compactMap { $0 }
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - wall) / 1e9
        var perDomain: [String: (hits: Int, heads: Int)] = [:]
        var dump = ""
        for (document, result) in zip(documents, results) {
            for answer in result.answers {
                perDomain[document.domain, default: (0, 0)].hits += answer.correct ? 1 : 0
                perDomain[document.domain, default: (0, 0)].heads += 1
                dump += "{\"id\":\"\(document.id)\",\"task\":\"\(answer.task)\",\"label\":\"\(answer.label)\"}\n"
            }
        }
        if let path = value("dump") { try dump.write(toFile: path, atomically: true, encoding: .utf8) }
        let accuracies = FastDecisions.domains.map { domain in
            Double(perDomain[domain]!.hits) / Double(perDomain[domain]!.heads)
        }
        for (domain, accuracy) in zip(FastDecisions.domains, accuracies) {
            print("\(domain.padding(toLength: 18, withPad: " ", startingAt: 0)) \(String(format: "%.3f", accuracy))")
        }
        let heads = perDomain.values.map(\.heads).reduce(0, +)
        print(
            """
            documents \(documents.count)  decisions \(heads)  in flight \(inflight)  \
            truncated \(results.filter(\.truncated).count)
            average \(String(format: "%.4f", accuracies.reduce(0, +) / Double(accuracies.count)))  \
            pooled \(String(format: "%.4f", Double(perDomain.values.map(\.hits).reduce(0, +)) / Double(heads)))
            wall \(String(format: "%.2f", seconds)) s  \(String(format: "%.1f", Double(documents.count) / seconds)) docs/s  \
            \(String(format: "%.1f", Double(heads) / seconds)) decisions/s
            """)
    }
}
