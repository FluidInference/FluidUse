import Foundation

/// Swift port of upstream `cua_s1.schema`: the byte-level context and option
/// rendering the checkpoint was trained on. Keep the strings identical.
public enum FormAction: String, Sendable {
    case fill, check, click, skip
    /// Host rule, not a model option: attach the source document to a file input.
    case attach
    /// Host rule, not a model option: a predetermined answer keyed by the question text.
    case answer
}

/// A predetermined answer from a separate step (a saved answer sheet or an LLM run once per
/// applicant), applied by the harness to any element whose label contains `question`.
/// The model never sees these; it was trained on short captions, not questions.
public struct PredeterminedAnswer: Identifiable, Sendable {
    public let question: String
    public let value: String
    public var id: String { question }

    public init(question: String, value: String) {
        self.question = question
        self.value = value
    }

    /// Lines of the form `question text contains => answer`; `#` starts a comment.
    public static func parse(_ text: String) -> [PredeterminedAnswer] {
        text.components(separatedBy: .newlines).compactMap { line in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty, !trimmed.hasPrefix("#"), let range = trimmed.range(of: "=>") else { return nil }
            let question = trimmed[..<range.lowerBound].trimmingCharacters(in: .whitespaces)
            let value = trimmed[range.upperBound...].trimmingCharacters(in: .whitespaces)
            guard !question.isEmpty, !value.isEmpty else { return nil }
            return PredeterminedAnswer(question: question, value: value)
        }
    }

    public static func match(_ label: String, in answers: [PredeterminedAnswer]) -> PredeterminedAnswer? {
        answers.first { label.localizedCaseInsensitiveContains($0.question) }
    }
}

/// A labeled value from the source document; rendered as one `fill` option.
public struct Entity: Identifiable, Hashable, Sendable {
    public let label: String
    public let value: String
    public var enabled = true
    public var id: String { label + "\u{0}" + value }

    public init(label: String, value: String, enabled: Bool = true) {
        self.label = label
        self.value = value
        self.enabled = enabled
    }

    public var option: String { "fill \(label): \(value)" }
}

/// One observed UI element. `token` is the stable action target inside the page.
public struct FormElement: Identifiable, Sendable {
    public let token: String
    public let role: String
    public let label: String
    public let value: String
    public let placeholder: String
    public let checked: Bool?
    public let frame: CGRect

    public init(
        token: String, role: String, label: String, value: String, placeholder: String, checked: Bool?, frame: CGRect
    ) {
        self.token = token
        self.role = role
        self.label = label
        self.value = value
        self.placeholder = placeholder
        self.checked = checked
        self.frame = frame
    }

    public var id: String { token }

    /// Roles the planner scores. Upstream: `filter_elements`.
    public var isActionable: Bool {
        ["button", "checkbox", "combobox", "edit", "textfield"].contains(normalizedRole)
    }

    public var isFileUpload: Bool { role == "FileUpload" }

    /// A combo box whose label is a statement to agree with is a consent control; the
    /// model knows those as checkboxes. Other combo boxes are scored as text fields so a
    /// matching entity can be chosen in them.
    public var scoringRole: String {
        guard role == "ComboBox" else { return role }
        let lowered = label.lowercased()
        let consent = [
            "i acknowledge", "i agree", "i certify", "i confirm", "i have read", "by submitting", "i understand",
            "i accept", "i consent",
        ]
        return consent.contains(where: lowered.contains) ? "CheckBox" : "Edit"
    }

    /// The element as the model sees it.
    public var forScoring: FormElement {
        guard role == "ComboBox" else { return self }
        return FormElement(
            token: token, role: scoringRole, label: label, value: value, placeholder: placeholder,
            checked: scoringRole == "CheckBox" ? (value.isEmpty ? false : true) : nil, frame: frame)
    }

    public var normalizedRole: String {
        role.replacingOccurrences(of: "_", with: "").replacingOccurrences(of: " ", with: "").lowercased()
    }
}

public struct PageSnapshot: Sendable {
    public let title: String
    public let url: String
    public let elements: [FormElement]

    public init(title: String, url: String, elements: [FormElement]) {
        self.title = title
        self.url = url
        self.elements = elements
    }
}

public enum FormSchema {
    public static let fixedActions: [FormAction] = [.check, .click, .skip]
    public static let appSuffixes = [
        " - Google Chrome", " - Microsoft Edge", " - Mozilla Firefox", " - Brave", " - Safari",
    ]

    /// `render_context` — one element, truncated exactly like upstream.
    public static func renderContext(formTitle: String, element: FormElement) -> String {
        let state: String
        if element.role == "CheckBox" {
            state = element.checked == true ? "checked" : "unchecked"
        } else {
            state = "value=\"\(String(element.value.prefix(48)))\""
        }
        let hint = element.placeholder.isEmpty ? "" : " hint=\"\(String(element.placeholder.prefix(72)))\""
        return "TASK fill the form from the document, then submit\n"
            + "FORM \(String(formTitle.prefix(64)))\n"
            + "ELEMENT \(element.role) \"\(String(element.label.prefix(72)))\" \(state)\(hint)"
    }

    /// `render_options` — entity pointers followed by the fixed actions.
    public static func renderOptions(entities: [Entity]) -> [String] {
        entities.map(\.option) + fixedActions.map(\.rawValue)
    }

    /// `decode` — option index to action and optional entity pointer.
    public static func decode(optionIndex: Int, entityCount: Int) -> (FormAction, Int?) {
        if optionIndex < entityCount { return (.fill, optionIndex) }
        return (fixedActions[optionIndex - entityCount], nil)
    }

    public static func normalizeTitle(_ title: String) -> String {
        for suffix in appSuffixes where title.hasSuffix(suffix) {
            return String(title.dropLast(suffix.count))
        }
        return title
    }

    /// Upstream only ever auto-clicks a control whose label normalizes to a submit label.
    public static func isSubmitControl(_ element: FormElement) -> Bool {
        guard ["button", "axbutton"].contains(element.normalizedRole) else { return false }
        let words = element.label.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber })
        let normalized = words.joined(separator: " ")
        return submitLabels.contains(normalized)
    }

    /// Upstream's runtime set is {"submit", "submit form"}; the training catalogue is
    /// broader. The demo accepts the training set so a "Submit application" button counts.
    public static let submitLabels: Set<String> = [
        "submit", "submit form", "submit registration", "submit application", "submit claim",
        "send", "continue", "register", "apply now", "save and continue", "complete registration",
        "next", "finish", "confirm and submit", "sign up", "book appointment", "create account", "done",
    ]

    /// `derive_entities` — conservative first/last/full-name variants.
    public static func deriveEntities(_ entities: [Entity]) -> [Entity] {
        var byLabel: [String: Entity] = [:]
        for entity in entities where byLabel[entity.label.lowercased()] == nil {
            byLabel[entity.label.lowercased()] = entity
        }
        var output = entities
        let nameKeys = ["name", "full name", "patient", "applicant", "patient name", "legal name"]
        let name = entities.first { nameKeys.contains($0.label.lowercased()) }
        if let name, name.value.trimmingCharacters(in: .whitespaces).contains(" "), byLabel["first name"] == nil {
            let trimmed = name.value.trimmingCharacters(in: .whitespaces)
            let first = String(trimmed.prefix { $0 != " " })
            let last = trimmed.dropFirst(first.count).trimmingCharacters(in: .whitespaces)
            output.append(Entity(label: "First name", value: first))
            output.append(Entity(label: "Last name", value: last))
        }
        if let first = byLabel["first name"], let last = byLabel["last name"], name == nil {
            output.append(Entity(label: "Full name", value: "\(first.value) \(last.value)"))
        }
        return output
    }
}
