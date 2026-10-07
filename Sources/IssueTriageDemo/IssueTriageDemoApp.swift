import AppKit
import SwiftUI

@main
struct IssueTriageDemoApp: App {
    @StateObject private var model = TriageModel()

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        // Bare SwiftPM executables start as background processes; make this one a regular windowed app.
        for key in UserDefaults.standard.dictionaryRepresentation().keys where key.hasPrefix("NSWindow Frame") {
            UserDefaults.standard.removeObject(forKey: key)
        }
        // The issues path is an argument; AppKit would otherwise take it as a document to open and show no window.
        var arguments = UserDefaults.standard.volatileDomain(forName: UserDefaults.argumentDomain)
        arguments["NSTreatUnknownArgumentsAsOpen"] = "NO"
        UserDefaults.standard.setVolatileDomain(arguments, forName: UserDefaults.argumentDomain)
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Issue Triage — Decision-2.0-Kai-0.6B on Core ML") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 1000, minHeight: 640)
                .task { await model.start() }
        }
        .defaultSize(width: 1500, height: 950)
        .windowResizability(.contentMinSize)
    }
}
