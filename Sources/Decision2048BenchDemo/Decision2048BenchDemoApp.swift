import SwiftUI

@main
struct Decision2048BenchDemoApp: App {
    @StateObject private var model = Decision2048BenchModel()

    var body: some Scene {
        WindowGroup {
            Decision2048BenchView()
                .environmentObject(model)
                .frame(minWidth: 900, minHeight: 720)
                .task { model.applyEnvironment() }
        }
        .defaultSize(width: 1060, height: 820)
    }
}
