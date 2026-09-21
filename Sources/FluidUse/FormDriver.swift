import Foundation

/// A UI surface the planner can observe and act on: the embedded web page or a
/// running application's window through the Accessibility API.
@MainActor
public protocol FormDriver: AnyObject {
    func snapshot() async throws -> PageSnapshot
    func highlight(_ token: String, on: Bool) async throws
    func type(_ value: String, into token: String, characterDelay: Duration) async throws
    func click(_ token: String) async throws
    func isChecked(_ token: String) async throws -> Bool?
    func attach(_ fileURL: URL, to token: String) async throws
    /// Chooses `value` in a combo box or select, typing to filter where the control allows it.
    func select(_ value: String, in token: String) async throws
    /// Chooses the affirmative option (Yes / I agree / I acknowledge) in a combo box used as consent.
    func selectAffirmative(in token: String) async throws
}

extension WebFormDriver: FormDriver {
    public func select(_ value: String, in token: String) async throws {
        try await type(value, into: token, characterDelay: .zero)
    }

    public func selectAffirmative(in token: String) async throws {
        for candidate in ["Yes", "I agree", "I acknowledge", "I accept", "Agree"] {
            if (try? await type(candidate, into: token, characterDelay: .zero)) != nil { return }
        }
        throw DriverError.valueNotApplied(token)
    }
}
