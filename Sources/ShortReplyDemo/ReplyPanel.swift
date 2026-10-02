import AppKit
import SwiftUI

/// Floating, non-activating panel that shows the selected post, the draft, timing, and Copy / Insert.
@available(macOS 15.0, *)
@MainActor
final class ReplyPanel {
    private let window: NSPanel
    private let model: ReplyPanelModel

    init(model: ReplyPanelModel) {
        self.model = model
        window = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 440, height: 250),
            styleMask: [.titled, .closable, .nonactivatingPanel, .utilityWindow, .fullSizeContentView],
            backing: .buffered, defer: false)
        window.title = "Short Reply"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.isFloatingPanel = true
        window.level = .floating
        window.hidesOnDeactivate = false
        window.isReleasedWhenClosed = false
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.contentView = NSHostingView(
            rootView: ReplyPanelView(model: model, onClose: { [weak self] in self?.window.orderOut(nil) }))
    }

    func hide() { window.orderOut(nil) }

    /// Shows the panel near the mouse without taking focus from the app the post was selected in.
    func present() {
        let mouse = NSEvent.mouseLocation
        var origin = NSPoint(x: mouse.x + 16, y: mouse.y - window.frame.height - 16)
        if let screen = NSScreen.screens.first(where: { $0.frame.contains(mouse) }) ?? NSScreen.main {
            origin.x = min(
                max(origin.x, screen.visibleFrame.minX + 8), screen.visibleFrame.maxX - window.frame.width - 8)
            origin.y = min(
                max(origin.y, screen.visibleFrame.minY + 8), screen.visibleFrame.maxY - window.frame.height - 8)
        }
        window.setFrameOrigin(origin)
        window.orderFrontRegardless()
    }
}

@available(macOS 15.0, *)
struct ReplyPanelView: View {
    @ObservedObject var model: ReplyPanelModel
    var onClose: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Short Reply").font(.headline)
                Spacer()
                statusLabel
                Button(action: onClose) { Image(systemName: "xmark.circle.fill") }
                    .buttonStyle(.plain).foregroundStyle(.secondary)
            }
            if !model.post.isEmpty {
                Text(model.post)
                    .font(.callout).foregroundStyle(.secondary).lineLimit(3)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            Group {
                switch model.phase {
                case .loading:
                    Text("Loading model…").foregroundStyle(.secondary)
                case .idle:
                    Text("Select a post and press 9.").foregroundStyle(.secondary)
                case .drafting:
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("Drafting…")
                    }
                case .done:
                    Text(model.reply).font(.title3.weight(.medium)).textSelection(.enabled)
                case .failed(let message):
                    Text(message).foregroundStyle(.red)
                }
            }
            .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
            HStack {
                if let timing = model.timing {
                    Text(
                        String(
                            format: "%d ms · prefill %d ms on ANE · %d tokens on GPU%@",
                            Int(timing.totalSeconds * 1000), Int(timing.prefillSeconds * 1000), timing.generatedTokens,
                            model.trimmed ? " · long post, start used" : "")
                    )
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                Spacer()
                Button("Again") { Task { await model.draft(post: model.post) } }.disabled(model.phase != .done)
                Button("Copy") { model.copyReply() }.disabled(model.phase != .done)
                Button("Insert") { model.insertReply() }.disabled(
                    model.phase != .done || model.sourceApplication == nil
                )
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(16)
        .frame(width: 440)
    }

    private var statusLabel: some View {
        Text(model.phase == .loading ? "loading" : "Qwen3-0.6B · 724 MB · on-device")
            .font(.caption).foregroundStyle(.tertiary)
    }
}
