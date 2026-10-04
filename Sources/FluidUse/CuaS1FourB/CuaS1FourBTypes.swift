import CoreGraphics
import Foundation

/// Errors from the Cua-S1-4B runtime.
public enum CuaS1FourBError: Error, LocalizedError, Sendable {
    case invalidAsset(String)
    case invalidModel(String)
    case invalidInput(String)
    case promptTooLong(tokens: Int, maximum: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidAsset(let detail): return "Cua-S1-4B asset: \(detail)"
        case .invalidModel(let detail): return "Cua-S1-4B model: \(detail)"
        case .invalidInput(let detail): return "Cua-S1-4B input: \(detail)"
        case .promptTooLong(let tokens, let maximum):
            return "Cua-S1-4B prompt is \(tokens) tokens; the largest loaded bucket holds \(maximum)"
        }
    }
}

/// Which LoRA adapter (and so which converted decoder) a manager runs.
public enum CuaS1FourBModality: String, Sendable, CaseIterable {
    /// Accessibility-tree text state.
    case text
    /// Screenshot state (vision tower + decoder trained with the multimodal adapter).
    case multimodal
}

/// One candidate `(element, action)` decision for a screen state, as in `cua_s1.four_b.Option`.
public struct CuaS1FourBOption: Sendable, Hashable {
    public var elementId: String
    public var role: String
    public var label: String
    public var action: String
    /// Only meaningful for `fill`: which extracted value would be entered.
    public var entityId: String?

    public init(elementId: String, role: String, label: String, action: String, entityId: String? = nil) {
        self.elementId = elementId
        self.role = role
        self.label = label
        self.action = action
        self.entityId = entityId
    }
}

/// The screen state and closed option list for one decision.
public struct CuaS1FourBState: Sendable {
    public var app: String
    public var taskFamily: String
    /// The episode goal when the state itself does not show it.
    public var goal: String?
    /// Accessibility tree text (text modality).
    public var accessibilityTree: String?
    /// Screenshot (multimodal modality).
    public var screenshot: CGImage?
    public var options: [CuaS1FourBOption]

    public init(
        app: String, taskFamily: String, goal: String? = nil, accessibilityTree: String? = nil,
        screenshot: CGImage? = nil, options: [CuaS1FourBOption]
    ) {
        self.app = app
        self.taskFamily = taskFamily
        self.goal = goal
        self.accessibilityTree = accessibilityTree
        self.screenshot = screenshot
        self.options = options
    }
}

/// Scored options for one state, in the caller's option order.
public struct CuaS1FourBDecision: Sendable {
    public struct Scored: Sendable {
        public let option: CuaS1FourBOption
        public let letter: String
        /// Raw answer-letter logit at the last prompt position.
        public let logit: Float
        /// Softmax over all option letters (the `FourBModel.forward` readout).
        public let probability: Float
    }

    public let options: [Scored]
    /// Prompt length in tokens and the bucket it ran in.
    public let tokens: Int
    public let bucketLength: Int

    /// The single best option overall (nil only for an empty decision, which `decide` never returns).
    public var best: Scored? { options.max { $0.logit < $1.logit } }

    /// Per element, the best of that element's own options (Cua's benchmark readout).
    public func bestPerElement() -> [String: Scored] {
        var result: [String: Scored] = [:]
        for scored in options {
            let id = scored.option.elementId
            if let current = result[id], current.logit >= scored.logit { continue }
            result[id] = scored
        }
        return result
    }
}
