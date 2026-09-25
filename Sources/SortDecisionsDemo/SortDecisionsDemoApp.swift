import AppKit
import SwiftUI

@main
struct SortDecisionsDemoApp: App {
    @StateObject private var model = DecisionsModel()

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        // Bare SwiftPM executables start as background processes; make this one a regular windowed app.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Sort decisions — GLiNER2.5-Decide on-device") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 1280, minHeight: 820)
                .task { await model.prepare() }
        }
        .defaultSize(width: 1500, height: 940)
        .windowResizability(.contentMinSize)
    }
}
