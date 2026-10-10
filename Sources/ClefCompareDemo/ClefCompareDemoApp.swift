import AppKit
import SwiftUI

@main
struct ClefCompareDemoApp: App {
    @StateObject private var model = CompareModel()

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        for key in UserDefaults.standard.dictionaryRepresentation().keys where key.hasPrefix("NSWindow Frame") {
            UserDefaults.standard.removeObject(forKey: key)  // open at the default size, not a remembered one
        }
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("clef-flash 9B vs clef-text 0.6B") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 1000, minHeight: 640)
                .task { await model.start() }
        }
        .defaultSize(width: 1320, height: 840)
        .windowResizability(.contentMinSize)
    }
}
