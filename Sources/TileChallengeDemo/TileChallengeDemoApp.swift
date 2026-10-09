import AppKit
import SwiftUI

@main
struct TileChallengeDemoApp: App {
    @StateObject private var model = ChallengeModel()

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        // Bare SwiftPM executables start as background processes; make this one a regular windowed app.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Picture challenge — on-device") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 980, minHeight: 680)
                .task { await model.prepare() }
        }
        .defaultSize(width: 1120, height: 760)
    }
}
