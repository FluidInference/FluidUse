import FluidUse
import SwiftUI

/// JEV-9B (Core ML) plays Super Mario Bros. 1-1.
///
///     JEV_MODEL_DIR=<bundle> swift run -c release JevMarioDemo     # or Sources/JevMarioDemo/demo.sh
///
/// The model bundle defaults to the pinned Hugging Face download (`JevModelStore`). The emulator runs in
/// `nes_bridge.py` through `uv` (`JEV_MARIO_BRIDGE` overrides the command).
@main
@available(macOS 15.0, *)
struct JevMarioDemoApp: App {
    @State private var modelDirectory: URL?
    @State private var error: String?

    var body: some Scene {
        WindowGroup("JEV-9B plays Mario") {
            Group {
                if let modelDirectory {
                    ContentView(modelDirectory: modelDirectory, bridgeCommand: Self.bridgeCommand)
                } else {
                    Text(error ?? "Downloading JEV-9B Core ML (~10 GB, first run only)…").padding(40)
                }
            }
            .task {
                guard modelDirectory == nil else { return }
                if let path = ProcessInfo.processInfo.environment["JEV_MODEL_DIR"] {
                    modelDirectory = URL(fileURLWithPath: path)
                    return
                }
                do { modelDirectory = try await JevModelStore.ensure() } catch {
                    self.error = error.localizedDescription
                }
            }
        }
        .windowResizability(.contentSize)
    }

    static var bridgeCommand: [String] {
        if let custom = ProcessInfo.processInfo.environment["JEV_MARIO_BRIDGE"] {
            return custom.split(separator: " ").map(String.init)
        }
        let script = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("nes_bridge.py")
        return [
            "uv", "run", "--no-project", "--python", "3.12", "--with", "gym-super-mario-bros==7.4.0", "--with",
            "nes-py",
            "--with", "gym==0.26.2", "--with", "numpy<2", "python", script.path,
        ]
    }
}
