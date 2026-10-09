import AppKit
import SwiftUI

// Search-as-you-type over a mock timeline with Granite-Embedding-30M-Sparse (Evoke) on the Neural Engine.
//
//     swift run -c release EvokeSearchDemo [--posts <posts.json>]     (or EVOKE_POSTS=<file>; EVOKE_MODEL_DIR=<dir>)

enum Launch {
    /// `--posts <file>` or `EVOKE_POSTS`: a JSON array of `{"text": …}`; otherwise the built-in 40 posts.
    static let postsFile: URL? = {
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--posts"), i + 1 < args.count { return URL(fileURLWithPath: args[i + 1]) }
        if let env = ProcessInfo.processInfo.environment["EVOKE_POSTS"], !env.isEmpty {
            return URL(fileURLWithPath: env)
        }
        return nil
    }()
}

@main
struct EvokeSearchDemoApp: App {
    init() {
        // Launched from `swift run`: become a regular foreground app with a Dock icon and focus.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Evoke Search") {
            RootView()
                .frame(width: 940)
                .frame(minHeight: 760)
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
    }
}
