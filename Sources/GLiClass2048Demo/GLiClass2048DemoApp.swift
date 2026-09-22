import SwiftUI

@main
struct GLiClass2048DemoApp: App {
    @StateObject private var model = Game2048Model()

    var body: some Scene {
        WindowGroup {
            Game2048View()
                .environmentObject(model)
                .frame(minWidth: 500, minHeight: 760)
                .task { model.applyEnvironment() }
        }
        .defaultSize(width: 560, height: 860)
    }
}
