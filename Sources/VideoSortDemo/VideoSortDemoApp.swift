import AppKit
import SwiftUI

@main
struct VideoSortDemoApp: App {
    @StateObject private var model = VideoSortModel()

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        // Bare SwiftPM executables start as background processes; make this one a regular windowed app.
        for key in UserDefaults.standard.dictionaryRepresentation().keys where key.hasPrefix("NSWindow Frame") {
            UserDefaults.standard.removeObject(forKey: key)
        }
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("SigLIP 2 on-device — video") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 720, minHeight: 520)
                .task { await model.prepare() }
        }
        .defaultSize(width: 1560, height: 980)
        .windowResizability(.contentMinSize)
    }
}
