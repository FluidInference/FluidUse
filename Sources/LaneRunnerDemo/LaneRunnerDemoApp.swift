import AppKit
import SwiftUI

@main
struct LaneRunnerDemoApp: App {
    @StateObject private var model = LaneRunnerModel()

    init() {
        // Launched from a terminal without a bundle: become a regular, focusable foreground app.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Lane Runner · Local model lab") {
            LaneRunnerView().environmentObject(model)
                .frame(minWidth: 800, minHeight: 700)
                .onAppear { model.applyLaunchEnvironment() }
        }
        .defaultSize(width: 820, height: 720)
    }
}
