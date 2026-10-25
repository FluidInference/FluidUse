import AppKit
import SwiftUI

@available(macOS 15.0, *)
struct ModerationDemoApp: App {
    @StateObject private var model = ModerationModel()

    init() {
        // Launched from a terminal without a bundle: become a regular, focusable foreground app.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Moderation · d1-omni-600M on device") {
            ModerationView().environmentObject(model)
                .frame(minWidth: 940, minHeight: 600)
                .onAppear { model.applyLaunchEnvironment() }
        }
        .defaultSize(width: 1040, height: 680)
    }
}
