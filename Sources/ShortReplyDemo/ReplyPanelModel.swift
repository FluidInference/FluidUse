import AppKit
import FluidUse
import Foundation

/// State behind the floating panel: model loading, the current post, the draft and its timing.
@available(macOS 15.0, *)
@MainActor
final class ReplyPanelModel: ObservableObject {
    enum Phase: Equatable {
        case loading
        case idle
        case drafting
        case done
        case failed(String)
    }

    @Published var phase: Phase = .loading
    @Published var post = ""
    @Published var reply = ""
    @Published var timing: ShortReplyManager.Timing?
    @Published var drafts = 0
    @Published var trimmed = false
    /// The post text as read from the app; drafts of the same raw text sample a new reply instead of repeating.
    private(set) var rawPost = ""
    private var variation = 0
    private var previousReplies: Set<String> = []
    var sourceApplication: NSRunningApplication?

    private var manager: ShortReplyManager?

    static var modelDirectory: URL {
        if let path = ProcessInfo.processInfo.environment["SHORT_REPLY_MODEL_DIR"], !path.isEmpty {
            return URL(fileURLWithPath: path)
        }
        return URL(fileURLWithPath: ".mobius/short-reply-lm/coreml-demo")
    }

    func load() async {
        let started = ContinuousClock.now
        do {
            let manager = try await ShortReplyManager.load(from: Self.modelDirectory)
            try await manager.warmUp()
            self.manager = manager
            let elapsed = started.duration(to: .now)
            print(
                "model ready in \(elapsed.components.seconds) s · prefill on ANE, decode on GPU · \(Self.modelDirectory.lastPathComponent)"
            )
            phase = .idle
        } catch {
            print("model load failed: \(error.localizedDescription)")
            phase = .failed(error.localizedDescription)
        }
    }

    func draft(post text: String) async {
        guard let manager else {
            show(error: "Model is still loading.")
            return
        }
        if text == rawPost {
            variation += 1
        } else {
            rawPost = text
            variation = 0
            previousReplies = []
        }
        post = text
        reply = ""
        timing = nil
        trimmed = false
        phase = .drafting
        do {
            let draft = try await manager.draft(for: text, variation: variation, avoiding: previousReplies)
            previousReplies.insert(draft.reply)
            reply = draft.reply
            timing = draft.timing
            trimmed = draft.trimmed
            post = draft.post
            drafts += 1
            phase = .done
            let ms = Int(draft.timing.totalSeconds * 1000)
            let prefill = Int(draft.timing.prefillSeconds * 1000)
            print(
                "#\(drafts) \(ms) ms (prefill \(prefill) ms ANE · \(draft.timing.generatedTokens) tokens GPU) · post \(text.count) chars\(variation > 0 ? " · regenerate #\(variation)" : "")"
            )
            print("   post:  \(text.prefix(110).replacingOccurrences(of: "\n", with: " "))")
            print("   reply: \(draft.reply)")
        } catch {
            show(error: error.localizedDescription)
        }
    }

    func show(error: String) {
        phase = .failed(error)
        print("error: \(error)")
    }

    func copyReply() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(reply, forType: .string)
    }

    /// Copies the reply, hands focus back to the source app, and sends it ⌘V directly (delivered to its process,
    /// so it lands even if macOS declines the activation request). `replace` selects the field first (⌘A), for a
    /// regenerated reply going into a box that already holds the previous one.
    func insertReply(replace: Bool = false) {
        copyReply()
        guard let application = sourceApplication else { return }
        if NSApp.isActive {
            NSApp.yieldActivation(to: application)
        }
        application.activate()
        let pid = application.processIdentifier
        let name = application.localizedName ?? "app"
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(200))
            if replace {
                // Purge the box first: select all, delete, and give the editor a moment between each step.
                KeystrokeSender.send(keyCode: 0, command: true, to: pid)  // ⌘A
                try? await Task.sleep(for: .milliseconds(80))
                KeystrokeSender.send(keyCode: 51, command: false, to: pid)  // Delete
                try? await Task.sleep(for: .milliseconds(80))
            }
            KeystrokeSender.send(keyCode: 9, command: true, to: pid)  // ⌘V
            print("insert: \(replace ? "replaced in" : "pasted into") \(name) (pid \(pid))")
        }
    }

    var isRegenerate: Bool { variation > 0 }
}
