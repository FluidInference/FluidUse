import FluidUse
import SwiftUI

/// JEV-9B or the small pixel student (Core ML) plays Super Mario Bros.
///
///     swift run -c release JevMarioDemo                                   # JEV-9B (downloaded on first use)
///     JEV_MARIO_BRAIN=student MARIO_STUDENT_MODEL=<.mlpackage> swift run -c release JevMarioDemo
///
/// JEV-9B loads from `JEV_MODEL_DIR` or the pinned Hugging Face download (`JevModelStore`), only when picked. The
/// emulator runs in `nes_bridge.py` through `uv` (`JEV_MARIO_BRIDGE` overrides the command).
@main
@available(macOS 15.0, *)
struct JevMarioDemoApp: App {
    init() { setvbuf(stdout, nil, _IOLBF, 0) }  // the decision log is tailed live (demo.sh)

    var body: some Scene {
        WindowGroup("Mario on Core ML") {
            ContentView(
                modelDirectory: ProcessInfo.processInfo.environment["JEV_MODEL_DIR"].flatMap {
                    $0.isEmpty ? nil : URL(fileURLWithPath: $0)
                },
                bridgeCommand: Self.bridgeCommand)
        }
        .defaultSize(width: 900, height: 540)
        .windowResizability(.contentMinSize)
    }

    static var bridgeCommand: [String] {
        if let custom = ProcessInfo.processInfo.environment["JEV_MARIO_BRIDGE"] {
            return custom.split(separator: " ").map(String.init)
        }
        let script = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent(
            "retro_bridge.py")
        return [
            "uv", "run", "--no-project", "--python", "3.12", "--with", "stable-retro", "--with",
            "gym-super-mario-bros==7.4.0", "--with",
            "nes-py",
            "--with", "gym==0.26.2", "--with", "numpy<2", "python", script.path,
        ]
    }
}
