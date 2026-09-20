import Foundation

/// A UI surface the planner can observe and act on: the embedded web page or a
/// running application's window through the Accessibility API.
@MainActor
protocol FormDriver: AnyObject {
    func snapshot() async throws -> PageSnapshot
    func highlight(_ token: String, on: Bool) async throws
    func type(_ value: String, into token: String, characterDelay: Duration) async throws
    func click(_ token: String) async throws
    func isChecked(_ token: String) async throws -> Bool?
    func attach(_ fileURL: URL, to token: String) async throws
}

extension WebFormDriver: FormDriver {}
