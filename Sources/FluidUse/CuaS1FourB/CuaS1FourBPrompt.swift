import Foundation

/// Prompt contract of `cua_s1.four_b` (letters, `build_prompt`) rendered with the Qwen3.5 chat template.
///
/// The template's generation prompt opens a thinking block (`<think>\n`); Cua trains and evaluates the
/// adapters with exactly that suffix, so the letter logits are read after it.
public enum CuaS1FourBPrompt {
    public static let letters = (UInt8(ascii: "A")...UInt8(ascii: "Z")).map { String(UnicodeScalar($0)) }

    public static let systemPrompt =
        "You are a one-pass computer-use decision model. You are shown the current state of a screen and a "
        + "fixed, closed list of candidate (element, action) options, each given a single letter. Choose exactly "
        + "one option: the single best next action to take. Answer with ONLY that option's letter -- no words, "
        + "no punctuation, no explanation."

    /// Placeholder the host expands to one `<|image_pad|>` per merged image token.
    public static let imagePlaceholder = "<|vision_start|><|image_pad|><|vision_end|>"

    public static func optionLine(letter: String, option: CuaS1FourBOption) -> String {
        var action = option.action
        if option.action == "fill", let entity = option.entityId, !entity.isEmpty {
            action += " (with entity '\(entity)')"
        }
        return "\(letter). \(option.role) \"\(option.label)\" -> \(action)"
    }

    /// `build_prompt`'s user text.
    public static func userText(state: CuaS1FourBState, modality: CuaS1FourBModality) throws -> String {
        guard !state.options.isEmpty else { throw CuaS1FourBError.invalidInput("no options") }
        guard state.options.count <= letters.count else {
            throw CuaS1FourBError.invalidInput("\(state.options.count) options exceeds the 26-letter budget")
        }
        let lines = zip(letters, state.options).map { optionLine(letter: $0, option: $1) }.joined(separator: "\n")
        var text = ""
        if let goal = state.goal, !goal.isEmpty { text += "Goal: \(goal)\n\n" }
        text += "App: \(state.app)\nTask family: \(state.taskFamily)\n\n"
        switch modality {
        case .text:
            guard let tree = state.accessibilityTree, !tree.isEmpty else {
                throw CuaS1FourBError.invalidInput("text modality requires an accessibility tree")
            }
            text += "Accessibility tree:\n\(tree)\n\n"
        case .multimodal:
            text += "The current screenshot is attached.\n\n"
        }
        return text + "Options:\n\(lines)\n\nAnswer with a single letter."
    }

    /// The full chat string passed to the tokenizer (`apply_chat_template(..., add_generation_prompt=True)`).
    public static func chat(state: CuaS1FourBState, modality: CuaS1FourBModality) throws -> String {
        let user = try userText(state: state, modality: modality)
        let content = modality == .multimodal ? imagePlaceholder + user : user
        return "<|im_start|>system\n\(systemPrompt)<|im_end|>\n<|im_start|>user\n\(content)<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n"
    }
}
