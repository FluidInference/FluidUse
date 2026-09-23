import SwiftUI

@main
struct LaneRunnerDemoApp: App {
    @StateObject private var model = LaneRunnerModel()

    var body: some Scene {
        WindowGroup("Lane Runner · Local model lab") {
            LaneRunnerView().environmentObject(model)
                .frame(minWidth: 800, minHeight: 700)
        }
        .defaultSize(width: 820, height: 720)
    }
}
