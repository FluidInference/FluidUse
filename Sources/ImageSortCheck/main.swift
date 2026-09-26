import FluidUse
import Foundation
import ImageSort

// Headless checks for the SigLIP 2 image sorter.
//   ImageSortCheck tokenizer <cases.json>   token ids must equal the Python tokenizer's
//   ImageSortCheck [--count=N] [--inflight=N] [--predictions=out.json]   zero-shot Pets accuracy and speed
setvbuf(stdout, nil, _IOLBF, 0)
let arguments = CommandLine.arguments.dropFirst()

func option(_ name: String) -> String? {
    arguments.first { $0.hasPrefix("--\(name)=") }.map { String($0.dropFirst(name.count + 3)) }
}

if arguments.first == "tokenizer", let path = arguments.dropFirst().first {
    struct Case: Decodable {
        let text: String
        let ids: [Int32]
    }
    guard let directory = ProcessInfo.processInfo.environment["SIGLIP2_MODEL_DIR"] else {
        fatalError("Set SIGLIP2_MODEL_DIR")
    }
    let tokenizer = try SigLIP2Tokenizer(
        tokenizerJsonURL: URL(fileURLWithPath: directory).appendingPathComponent("tokenizer.json"))
    let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: URL(fileURLWithPath: path)))
    var mismatches = 0
    for item in cases {
        let ids = try tokenizer.encode(item.text)
        if ids != item.ids {
            mismatches += 1
            if mismatches <= 5 {
                print("mismatch: \(item.text.debugDescription)\n  swift \(ids)\n  python \(item.ids)")
            }
        }
    }
    print("tokenizer: \(cases.count - mismatches)/\(cases.count) identical")
    exit(mismatches == 0 ? 0 : 1)
}

let count = option("count").flatMap(Int.init)
let inFlight = option("inflight").flatMap(Int.init) ?? 4
let items = try await PetsSample.load(count: count ?? PetsSample.testCount) { done, total in
    if done % 250 == 0 || done == total { print("cached \(done)/\(total) photos") }
}
let sorter = try await ImageSorter.load()
_ = try await sorter.sort(items[0])

let start = Date()
let results = try await withThrowingTaskGroup(of: (PetItem, ImageSorter.Result).self) { group in
    var next = 0
    var results: [(PetItem, ImageSorter.Result)] = []
    func launch() {
        guard next < items.count else { return }
        let item = items[next]
        next += 1
        group.addTask { (item, try await sorter.sort(item)) }
    }
    for _ in 0..<inFlight { launch() }
    while let result = try await group.next() {
        results.append(result)
        launch()
    }
    return results
}
let seconds = Date().timeIntervalSince(start)
let correct = results.filter { $0.0.breed == $0.1.breed }.count
let milliseconds = results.map(\.1.milliseconds).sorted()
print(
    "\(sorter.modelName): \(correct)/\(results.count) = \(String(format: "%.2f", 100 * Double(correct) / Double(results.count)))% · "
        + "\(String(format: "%.2f", seconds)) s (\(String(format: "%.0f", Double(results.count) / seconds)) photos/s, "
        + "\(inFlight) in flight) · median call \(String(format: "%.1f", milliseconds[milliseconds.count / 2])) ms")
if let path = option("predictions") {
    let predictions = Dictionary(uniqueKeysWithValues: results.map { (String($0.0.id), $0.1.breed) })
    try JSONEncoder().encode(predictions).write(to: URL(fileURLWithPath: path))
}
