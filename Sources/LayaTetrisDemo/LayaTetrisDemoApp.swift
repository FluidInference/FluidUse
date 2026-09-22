import AppKit
import SwiftUI

@main
struct LayaTetrisDemoApp: App {
    @StateObject private var model = GameModel()

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        // Bare SwiftPM executables start as background processes; make this one a
        // regular windowed app with a menu bar and Dock presence.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("laya plays Tetris") {
            ContentView()
                .environmentObject(model)
                // Tall enough for header + 520 pt board + controls; still fits a
                // 14" display with the menu and title bars.
                .frame(minWidth: 340, idealWidth: 460, minHeight: 620, idealHeight: 940)
                .onAppear { model.applyEnvironment() }
        }
        .windowResizability(.contentMinSize)
    }
}
