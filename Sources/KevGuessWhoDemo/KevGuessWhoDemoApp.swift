import AppKit
import SwiftUI

@main
struct KevGuessWhoDemoApp: App {
    @StateObject private var model = GuessWhoModel()

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
        WindowGroup("Guess Who — Kev-0.8B on Core ML") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 900, minHeight: 600)
                .task { await model.start() }
        }
        .defaultSize(width: 1560, height: 960)
        .windowResizability(.contentMinSize)
    }
}
