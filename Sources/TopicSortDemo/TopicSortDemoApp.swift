import AppKit
import SwiftUI

@main
struct TopicSortDemoApp: App {
    @StateObject private var model = TopicSortModel()

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        // Bare SwiftPM executables start as background processes; make this one a regular windowed app.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Sort by topic — on-device") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 900, minHeight: 600)
                .task { await model.prepare() }
        }
        .defaultSize(width: 1100, height: 950)
    }
}
