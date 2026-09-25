import Foundation
import SortAnything

/// Headless Sort Anything: sorts a balanced DBpedia-14 test sample and reports accuracy and throughput.
///
///     swift run -c release SortAnythingCheck [--count=1000] [--seed=0] [--inflight=1] [--dump=path.jsonl]
@main
struct SortAnythingCheck {
    static func main() async throws {
        let arguments = CommandLine.arguments.dropFirst()
        func value(_ name: String) -> String? {
            arguments.first { $0.hasPrefix("--\(name)=") }.map { String($0.dropFirst(name.count + 3)) }
        }
        let count = value("count").flatMap(Int.init) ?? 1000
        let seed = value("seed").flatMap(UInt64.init) ?? 0
        let items = try await DBpediaSample.load(count: count, seed: seed)
        let sorter = try await Sorter.load()
        let categories = DBpediaSample.categories
        _ = try await sorter.sort(items[0], into: categories)

        let inflight = max(1, value("inflight").flatMap(Int.init) ?? 1)
        var correct = 0
        var truncated = 0
        var latencies: [Double] = []
        var dump = ""
        let wall = DispatchTime.now().uptimeNanoseconds
        let results = try await withThrowingTaskGroup(of: (Int, Sorter.Result).self) { group in
            var results = [Sorter.Result?](repeating: nil, count: items.count)
            var next = 0
            while next < min(inflight, items.count) {
                let index = next
                group.addTask { (index, try await sorter.sort(items[index], into: categories)) }
                next += 1
            }
            while let (index, result) = try await group.next() {
                results[index] = result
                if next < items.count {
                    let index = next
                    group.addTask { (index, try await sorter.sort(items[index], into: categories)) }
                    next += 1
                }
            }
            return results.compactMap { $0 }
        }
        for (item, result) in zip(items, results) {
            correct += result.category == item.gold ? 1 : 0
            truncated += result.truncated ? 1 : 0
            latencies.append(result.milliseconds)
            if value("dump") != nil {
                dump += "{\"id\":\(item.id),\"predicted\":\"\(result.category)\",\"gold\":\"\(item.gold)\"}\n"
            }
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - wall) / 1e9
        latencies.sort()
        if let path = value("dump") { try dump.write(toFile: path, atomically: true, encoding: .utf8) }
        print(
            """
            items \(items.count)  in flight \(inflight)  accuracy \(String(format: "%.3f", Double(correct) / Double(items.count)))  \
            truncated \(truncated)
            wall \(String(format: "%.2f", seconds)) s  \
            \(String(format: "%.1f", Double(items.count) / seconds)) items/s  \
            p50 \(String(format: "%.2f", latencies[latencies.count / 2])) ms  \
            p95 \(String(format: "%.2f", latencies[latencies.count * 95 / 100])) ms
            """)
    }
}
