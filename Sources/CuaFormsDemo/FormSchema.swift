import Foundation

/// Swift port of upstream `cua_s1.schema`: the byte-level context and option
/// rendering the checkpoint was trained on. Keep the strings identical.
enum FormAction: String, Sendable {
    case fill, check, click, skip
    /// Host rule, not a model option: attach the source document to a file input.
    case attach
}

/// A labeled value from the source document; rendered as one `fill` option.
struct Entity: Identifiable, Hashable, Sendable {
    let label: String
    let value: String
    var enabled = true
    var id: String { label + "\u{0}" + value }

    var option: String { "fill \(label): \(value)" }
}

/// One observed UI element. `token` is the stable action target inside the page.
struct FormElement: Identifiable, Sendable {
    let token: String
    let role: String
    let label: String
    let value: String
    let placeholder: String
    let checked: Bool?
    let frame: CGRect

    var id: String { token }

    /// Roles the planner scores. Upstream: `filter_elements`.
    var isActionable: Bool {
        ["button", "checkbox", "combobox", "edit", "textfield"].contains(normalizedRole)
    }

    var isFileUpload: Bool { role == "FileUpload" }

    /// A combo box whose label is a statement to agree with is a consent control; the
    /// model knows those as checkboxes. Other combo boxes are scored as text fields so a
    /// matching entity can be chosen in them.
    var scoringRole: String {
        guard role == "ComboBox" else { return role }
        let lowered = label.lowercased()
        let consent = [
            "i acknowledge", "i agree", "i certify", "i confirm", "i have read", "by submitting", "i understand",
            "i accept", "i consent",
        ]
        return consent.contains(where: lowered.contains) ? "CheckBox" : "Edit"
    }

    /// The element as the model sees it.
    var forScoring: FormElement {
        guard role == "ComboBox" else { return self }
        return FormElement(
            token: token, role: scoringRole, label: label, value: value, placeholder: placeholder,
            checked: scoringRole == "CheckBox" ? (value.isEmpty ? false : true) : nil, frame: frame)
    }

    var normalizedRole: String {
        role.replacingOccurrences(of: "_", with: "").replacingOccurrences(of: " ", with: "").lowercased()
    }
}

struct PageSnapshot: Sendable {
    let title: String
    let url: String
    let elements: [FormElement]
}

enum FormSchema {
    static let fixedActions: [FormAction] = [.check, .click, .skip]
    static let appSuffixes = [
        " - Google Chrome", " - Microsoft Edge", " - Mozilla Firefox", " - Brave", " - Safari",
    ]

    /// `render_context` — one element, truncated exactly like upstream.
    static func renderContext(formTitle: String, element: FormElement) -> String {
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
    static func renderOptions(entities: [Entity]) -> [String] {
        entities.map(\.option) + fixedActions.map(\.rawValue)
    }

    /// `decode` — option index to action and optional entity pointer.
    static func decode(optionIndex: Int, entityCount: Int) -> (FormAction, Int?) {
        if optionIndex < entityCount { return (.fill, optionIndex) }
        return (fixedActions[optionIndex - entityCount], nil)
    }

    static func normalizeTitle(_ title: String) -> String {
        for suffix in appSuffixes where title.hasSuffix(suffix) {
            return String(title.dropLast(suffix.count))
        }
        return title
    }

    /// Upstream only ever auto-clicks a control whose label normalizes to a submit label.
    static func isSubmitControl(_ element: FormElement) -> Bool {
        guard ["button", "axbutton"].contains(element.normalizedRole) else { return false }
        let words = element.label.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber })
        let normalized = words.joined(separator: " ")
        return submitLabels.contains(normalized)
    }

    /// Upstream's runtime set is {"submit", "submit form"}; the training catalogue is
    /// broader. The demo accepts the training set so a "Submit application" button counts.
    static let submitLabels: Set<String> = [
        "submit", "submit form", "submit registration", "submit application", "submit claim",
        "send", "continue", "register", "apply now", "save and continue", "complete registration",
        "next", "finish", "confirm and submit", "sign up", "book appointment", "create account", "done",
    ]

    /// `derive_entities` — conservative first/last/full-name variants.
    static func deriveEntities(_ entities: [Entity]) -> [Entity] {
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
