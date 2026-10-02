import AppKit
import SwiftUI

// Menu-bar app: select a post anywhere, press ⌃⌥R, get a drafted reply in a floating panel.
// Top-level code (not @main) so the macOS 15 Core ML APIs can be gated at runtime while the package targets macOS 14.

@available(macOS 15.0, *)
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem?
    private let model = ReplyPanelModel()
    private var panel: ReplyPanel?
    private var hotkey: HotkeyMonitor?
    /// With a reply box focused, 9 drafts and pastes without showing the panel.
    private var autoInsert = true
    private var autoInsertItem: NSMenuItem?
    /// Bare 9 / 0 keys are observed system-wide; switch them off outside a demo (⌃⌥R always works).
    private var bareKeys = true
    private var bareKeysItem: NSMenuItem?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        item.button?.title = "↩︎"
        item.button?.toolTip = "Short Reply — 9 drafts a reply into the focused reply box, 0 regenerates it"
        let menu = NSMenu()
        for (title, action) in [
            ("Draft reply for selection  9 / ⌃⌥R", #selector(draftFromMenu)),
            ("Draft reply from clipboard", #selector(draftFromClipboard)),
        ] {
            let item = NSMenuItem(title: title, action: action, keyEquivalent: "")
            item.target = self
            menu.addItem(item)
        }
        menu.addItem(.separator())
        let auto = NSMenuItem(
            title: "Auto-insert into the focused reply box (no panel)", action: #selector(toggleAutoInsert),
            keyEquivalent: "")
        auto.target = self
        auto.state = .on
        menu.addItem(auto)
        autoInsertItem = auto
        let bare = NSMenuItem(
            title: "Bare 9 / 0 keys (demo mode)", action: #selector(toggleBareKeys), keyEquivalent: "")
        bare.target = self
        bare.state = .on
        menu.addItem(bare)
        bareKeysItem = bare
        menu.addItem(.separator())
        // Quit keeps a nil target so the responder chain reaches NSApplication.
        menu.addItem(withTitle: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        item.menu = menu
        statusItem = item

        panel = ReplyPanel(model: model)
        hotkey = HotkeyMonitor { [weak self] action in
            Task { @MainActor in
                guard let self else { return }
                switch action {
                case .draft(let typed):
                    if typed && !self.bareKeys { return }
                    self.draftFromSelection(typedTrigger: typed)
                case .regenerate(let typed):
                    if typed && !self.bareKeys { return }
                    if typed { SelectionReader.deleteTypedTriggerIfEditing() }
                    self.regenerate()
                }
            }
        }
        if !SelectionReader.isTrusted {
            print(
                "accessibility: not trusted yet — grant access in System Settings > Privacy & Security > Accessibility")
        }
        Task { await model.load() }
    }

    @objc func draftFromMenu() { draftFromSelection(typedTrigger: false) }

    /// 0: a new sampled reply for the last post, replacing the box contents (or a first draft if there is none yet).
    func regenerate() {
        if model.phase == .drafting { return }  // a draft is in flight; ignore the key
        guard !model.rawPost.isEmpty else { return draftFromSelection(typedTrigger: false) }
        let previous = model.sourceApplication
        Task { @MainActor in
            await model.draft(post: model.rawPost)
            guard case .done = model.phase else { return panel?.present() ?? () }
            if autoInsert, previous != nil {
                model.sourceApplication = previous
                model.insertReply(replace: true)
            } else {
                panel?.present()
            }
        }
    }

    @objc func toggleAutoInsert() {
        autoInsert.toggle()
        autoInsertItem?.state = autoInsert ? .on : .off
    }

    @objc func toggleBareKeys() {
        bareKeys.toggle()
        bareKeysItem?.state = bareKeys ? .on : .off
    }

    func draftFromSelection(typedTrigger: Bool) {
        if model.phase == .drafting { return }
        let previous = NSWorkspace.shared.frontmostApplication
        Task { @MainActor in
            let context = await SelectionReader.context(typedTrigger: typedTrigger)
            await draft(context.text, source: previous, insertDirectly: autoInsert && context.fieldFocused)
        }
    }

    @objc func draftFromClipboard() {
        Task { @MainActor in
            await draft(NSPasteboard.general.string(forType: .string), source: nil, insertDirectly: false)
        }
    }

    private func draft(_ text: String?, source: NSRunningApplication?, insertDirectly: Bool) async {
        guard let text, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            model.show(error: "Select a post, or open its reply box, then press 9.")
            panel?.present()
            return
        }
        model.sourceApplication = source
        if !insertDirectly { panel?.present() }
        await model.draft(post: text)
        guard insertDirectly else { return }
        if case .done = model.phase {
            model.insertReply(replace: model.isRegenerate)
        } else {
            panel?.present()  // an error: show it
        }
    }
}

setvbuf(stdout, nil, _IOLBF, 0)
if #available(macOS 15.0, *) {
    let application = NSApplication.shared
    let delegate = AppDelegate()
    application.delegate = delegate
    application.setActivationPolicy(.accessory)
    application.run()
} else {
    FileHandle.standardError.write(Data("ShortReplyDemo needs macOS 15\n".utf8))
    exit(2)
}
