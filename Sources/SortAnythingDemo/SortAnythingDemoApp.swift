import AppKit
import SwiftUI

@main
struct SortAnythingDemoApp: App {
    @StateObject private var model = SortModel()

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        // Bare SwiftPM executables start as background processes; make this one a regular windowed app.
        // Forget any saved window frame so the default size below (all 14 buckets visible) applies on launch.
        for key in UserDefaults.standard.dictionaryRepresentation().keys where key.hasPrefix("NSWindow Frame") {
            UserDefaults.standard.removeObject(forKey: key)
        }
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Sort anything — GLiNER2.5-Decide on-device") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 760, minHeight: 560)
                .task { await model.prepare() }
        }
        .defaultSize(width: 1500, height: 940)
        .windowResizability(.contentMinSize)
    }
}
