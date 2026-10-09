import AppKit
import SwiftUI

@main
struct AudioSearchDemoApp: App {
    @StateObject private var model = AudioSearchModel()

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        // Bare SwiftPM executables start as background processes; make this one a regular windowed app.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Search audio — on-device") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 900, minHeight: 640)
                .task { await model.prepare() }
        }
        .defaultSize(width: 1100, height: 900)
    }
}
