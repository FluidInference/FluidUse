import AppKit
import FluidUse
import Foundation

// Comment moderation firehose: d1-omni-600M on Core ML, Neural Engine and GPU together.
//
//     swift run -c release ModerationDemo [--model <dir>]     (or D1_MODERATION_DIR=<dir>)

enum Launch {
    /// `--model <dir>` or `D1_MODERATION_DIR`; otherwise the pinned snapshot from the Hub (`D1OmniModelStore`).
    static let modelDirectory: URL? = {
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--model"), i + 1 < args.count { return URL(fileURLWithPath: args[i + 1]) }
        if let env = ProcessInfo.processInfo.environment["D1_MODERATION_DIR"], !env.isEmpty {
            return URL(fileURLWithPath: env)
        }
        return nil
    }()

    static func resolveModel() async throws -> URL {
        if let dir = modelDirectory { return dir }
        return try await D1OmniModelStore.ensure { file, bytes in
            if bytes > 0 { print("downloaded \(file) (\(bytes / 1_000_000) MB)") }
        }
    }
}

if #available(macOS 15.0, *) {
    ModerationDemoApp.main()
} else {
    fputs("ModerationDemo needs macOS 15 or later\n", stderr)
    exit(1)
}
