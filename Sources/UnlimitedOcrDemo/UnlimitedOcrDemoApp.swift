import AppKit
import SwiftUI

@main
struct UnlimitedOcrDemoApp: App {
    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        // Bare SwiftPM executables start as background processes; make this one a regular windowed app.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Unlimited-OCR — on this Mac") {
            if #available(macOS 15.0, *) {
                RootView()
            } else {
                Text("Unlimited-OCR needs macOS 15 (Core ML state).").padding(40)
            }
        }
        .defaultSize(width: 1320, height: 860)
        .windowResizability(.contentMinSize)
    }
}

@available(macOS 15.0, *)
struct RootView: View {
    @StateObject private var model = ReaderModel()

    var body: some View {
        ContentView()
            .environmentObject(model)
            .frame(minWidth: 1000, minHeight: 640)
            .task { await model.boot() }
    }
}
