import AppKit
import SwiftUI

@main
struct BookmarkSortDemoApp: App {
    @StateObject private var model = BookmarkSortModel()

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
        WindowGroup("Sort bookmarks — on-device") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 460, minHeight: 520)
                .task { await model.start() }
        }
        .defaultSize(width: 620, height: 1000)
        .windowResizability(.contentMinSize)
    }
}
