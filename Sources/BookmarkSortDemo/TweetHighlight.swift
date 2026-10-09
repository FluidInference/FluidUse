import AppKit
import SwiftUI

/// Click-through outline around the post being read in the browser, with the decision as a caption.
@MainActor
final class TweetHighlight {
    struct Look: Equatable {
        var caption: String
        var color: Color
    }

    private final class State: ObservableObject {
        @Published var look = Look(caption: "", color: .orange)
    }

    private struct Outline: View {
        @ObservedObject var state: State

        var body: some View {
            ZStack(alignment: .topLeading) {
                RoundedRectangle(cornerRadius: 10)
                    .stroke(state.look.color, lineWidth: 4)
                    .background(RoundedRectangle(cornerRadius: 10).fill(state.look.color.opacity(0.08)))
                    .padding(.top, Self.captionHeight)
                Text(state.look.caption)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 10)
                    .frame(height: Self.captionHeight - 4)
                    .background(Capsule().fill(state.look.color))
            }
            .animation(.easeOut(duration: 0.15), value: state.look)
        }

        static let captionHeight: CGFloat = 30
    }

    private let panel: NSPanel
    private let state = State()

    init() {
        panel = NSPanel(
            contentRect: .zero, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        panel.level = .floating
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.ignoresMouseEvents = true
        panel.hasShadow = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.contentView = NSHostingView(rootView: Outline(state: state))
    }

    /// `frame` uses Accessibility's top-left screen origin; the caption sits just above it.
    func show(around frame: CGRect, _ look: Look) {
        state.look = look
        guard let screen = NSScreen.screens.first(where: { $0.frame.origin == .zero }) ?? NSScreen.main else { return }
        let padded = frame.insetBy(dx: -6, dy: -6)
        let height = padded.height + Outline.captionHeight
        let flippedY = screen.frame.height - padded.maxY
        panel.setFrame(CGRect(x: padded.minX, y: flippedY, width: padded.width, height: height), display: true)
        panel.orderFrontRegardless()
    }

    func update(_ look: Look) { state.look = look }

    func hide() { panel.orderOut(nil) }
}
