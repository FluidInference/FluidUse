import AppKit
import SwiftUI

@main
struct CodeSearchDemoApp: App {
    @StateObject private var model = CodeSearchModel()

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        // Bare SwiftPM executables start as background processes; make this one a regular windowed app.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Code search — on-device") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 1000, minHeight: 700)
                .onAppear { model.start() }
        }
        .defaultSize(width: 1200, height: 950)
    }
}
