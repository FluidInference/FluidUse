import FluidUse
import Foundation

/// Sorts text into caller-chosen categories with GLiNER2.5-Decide on Core ML.
/// Calls are not serialized: callers may keep several `sort` calls in flight.
public final class Sorter: Sendable {
    public struct Result: Sendable {
        public let category: String
        public let confidence: Float
        public let milliseconds: Double
        /// The abstract was shortened to fit the model's token budget.
        public let truncated: Bool
    }

    public static let task = "category"

    private let manager: GLiNER2Manager

    private init(manager: GLiNER2Manager) { self.manager = manager }

    /// Downloads (once) and loads the fp16 128-token Decide package.
    public static func load(progress: GLiNER2ModelStore.Progress? = nil) async throws -> Sorter {
        let manager: GLiNER2Manager
        if let path = ProcessInfo.processInfo.environment["GLINER2_DECIDE_MODEL_DIR"], !path.isEmpty {
            manager = try await GLiNER2Manager.load(from: URL(fileURLWithPath: path), variant: .decide)
        } else {
            manager = try await GLiNER2Manager.load(variant: .decide, progress: progress)
        }
        return Sorter(manager: manager)
    }

    public var maximumCategories: Int { manager.maximumOptions }

    /// Picks one of `categories` for `item`, dropping trailing words until the request fits.
    public func sort(_ item: SortItem, into categories: [String]) async throws -> Result {
        let start = DispatchTime.now().uptimeNanoseconds
        var words = item.content.split(separator: " ", omittingEmptySubsequences: true)
        var truncated = false
        while true {
            let text = item.title + "\n" + words.joined(separator: " ")
            do {
                let answer = try await manager.classifyConcurrently(text: text, task: Self.task, labels: categories)
                let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                return Result(
                    category: answer.selectedLabel, confidence: answer.probabilities[answer.selectedIndex],
                    milliseconds: milliseconds, truncated: truncated)
            } catch GLiNER2Error.invalidInput(let reason) where reason.contains("tokens") && words.count > 8 {
                words.removeLast(max(1, words.count / 8))
                truncated = true
            }
        }
    }
}
