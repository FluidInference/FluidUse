import AppKit
import SwiftUI

@available(macOS 15.0, *)
struct FrontDoorApp: App {
    @StateObject private var model = FrontDoorModel()

    init() {
        // Bare SwiftPM executables start as background processes; make this one a regular windowed app.
        for key in UserDefaults.standard.dictionaryRepresentation().keys where key.hasPrefix("NSWindow Frame") {
            UserDefaults.standard.removeObject(forKey: key)
        }
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Chatbot Front Door — Vela-2.0-0.3B on Core ML") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 980, minHeight: 680)
                .task { await model.start() }
        }
        .defaultSize(width: 1100, height: 800)
        .windowResizability(.contentMinSize)
    }
}
