import AppKit
import SwiftUI

@main
struct CuaFormsDemoApp: App {
    @StateObject private var model = DemoModel()

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        // Bare SwiftPM executables start as background processes; make this one a
        // regular windowed app with a menu bar and Dock presence.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("CUA-S1-FORMS on-device") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 1180, idealWidth: 1560, minHeight: 720, idealHeight: 940)
        }
    }
}
