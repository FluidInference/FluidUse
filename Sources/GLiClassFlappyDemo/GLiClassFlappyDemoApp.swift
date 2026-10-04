import SwiftUI

@main
struct GLiClassFlappyDemoApp: App {
    @StateObject private var model = FlappyModel()

    var body: some Scene {
        WindowGroup("Flappy Bird · Local model lab") {
            FlappyView().environmentObject(model)
                .frame(minWidth: 820, minHeight: 730)
        }
        .defaultSize(width: 860, height: 770)
    }
}
