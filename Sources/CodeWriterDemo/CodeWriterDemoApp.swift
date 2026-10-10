import AppKit
import SwiftUI

@available(macOS 15.0, *)
@main
struct CodeWriterDemoApp: App {
    @StateObject private var model = WriterModel()

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        // Bare SwiftPM executables start as background processes; make this one a regular windowed app.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Python writer — on-device") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 1000, minHeight: 680)
                .onAppear { model.start() }
        }
        .defaultSize(width: 1280, height: 860)
    }
}
