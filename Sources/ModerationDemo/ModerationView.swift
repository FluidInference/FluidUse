import Moderation
import SwiftUI

@available(macOS 15.0, *)
struct ModerationView: View {
    @EnvironmentObject private var model: ModerationModel

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            controls.frame(width: 220)
            GeometryReader { feedSize in
                HStack(alignment: .top, spacing: 16) {
                    feedColumn
                    toxicColumn(flyFrom: feedSize.size.width - 300)
                        .frame(width: 300)
                }
            }
        }
        .padding(16)
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Comment moderation").font(.headline)
            Text("d1-omni-600M · on device · no network").font(.caption2).foregroundStyle(.secondary)
            Picker("Chips", selection: $model.mode) {
                Text("Neural Engine").tag(ModerationEngine.Mode.ane)
                Text("GPU").tag(ModerationEngine.Mode.gpu)
                Text("Neural Engine + GPU").tag(ModerationEngine.Mode.both)
            }
            .labelsHidden().controlSize(.small)
            .disabled(model.isRunning || model.isLoading)
            .onChange(of: model.mode) { _, _ in model.load() }
            HStack(spacing: 6) {
                Button(model.isLoading ? "Loading…" : (model.isRunning ? "Running…" : "Start")) { model.start() }
                    .buttonStyle(.borderedProminent).tint(.green)
                    .disabled(model.loadedMode == nil || model.isRunning)
                    .keyboardShortcut(.space, modifiers: [])
                Button("Reset") { model.reset() }.disabled(model.isRunning)
            }
            .controlSize(.small)
            Divider()
            TimelineView(.periodic(from: .now, by: 0.1)) { context in
                VStack(alignment: .leading, spacing: 8) {
                    metric("COMMENTS / S", String(format: "%.0f", model.rate(at: context.date)), big: true)
                    HStack {
                        metric("CHECKED", "\(model.checked)")
                        metric("TIME", String(format: "%.1fs", elapsed(at: context.date)))
                    }
                }
            }
            HStack {
                metric("FLAGGED", "\(model.flagged)")
                metric("MS/CALL", model.checked == 0 ? "—" : String(format: "%.1f", model.lastMs))
            }
            HStack {
                metric(
                    "VS HUMANS",
                    model.checked == 0
                        ? "—" : String(format: "%.1f%%", 100 * Double(model.agreeing) / Double(model.checked)))
                metric("OF", "\(model.total)")
            }
            HStack {
                metric("ANE", "\(model.aneCount)")
                metric("GPU", "\(model.gpuCount)")
            }
            Divider()
            Text(
                "Civil Comments test set (CC0). Flagged at P(toxic) ≥ 0.8; “raters” = share of ~10 people who called it toxic."
            )
            .font(.caption2).foregroundStyle(.secondary)
            if let error = model.errorMessage {
                Text(error).font(.caption2).foregroundStyle(.red).textSelection(.enabled)
            }
            Spacer()
        }
    }

    private func elapsed(at date: Date) -> Double {
        model.finishedSeconds ?? model.startedAt.map { date.timeIntervalSince($0) } ?? 0
    }

    private var feedColumn: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("LIVE FEED · passed").font(.caption.monospaced()).foregroundStyle(.green)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 4) {
                    ForEach(model.feed) { row in feedRow(row) }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    /// Flagged comments slide in from the live feed's side, so each toxic one visibly leaves the feed.
    private func toxicColumn(flyFrom distance: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("TOXIC · \(model.flagged) removed").font(.caption.monospaced()).foregroundStyle(.red)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 3) {
                    ForEach(model.flaggedFeed) { row in
                        toxicRow(row)
                            .transition(
                                .asymmetric(
                                    insertion: .offset(x: -distance).combined(
                                        with: .scale(scale: 1.4, anchor: .leading)
                                    )
                                    .combined(with: .opacity),
                                    removal: .opacity))
                    }
                }
            }
            // Flagged rows fly in from the feed, outside this scroll view's bounds.
            .scrollClipDisabled()
        }
        .padding(10)
        .frame(maxHeight: .infinity, alignment: .topLeading)
        .background(Color.red.opacity(0.08), in: .rect(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.red.opacity(0.35)))
    }

    private func feedRow(_ row: ModeratedComment) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Text("✓").font(.body.bold()).foregroundStyle(.green)
            Text(row.text.replacingOccurrences(of: "\n", with: " "))
                .font(.caption).lineLimit(2).foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
            VStack(alignment: .trailing, spacing: 2) {
                Text(String(format: "ok %.0f%%", (1 - row.probability) * 100))
                    .font(.caption.monospaced().bold()).foregroundStyle(.secondary)
                Text(String(format: "raters %.0f%% · %@", row.humanToxicity * 100, row.engine.rawValue))
                    .font(.caption2.monospaced()).foregroundStyle(.secondary)
            }
            .frame(width: 110, alignment: .trailing)
        }
        .padding(6)
        .background(Color.primary.opacity(0.04), in: .rect(cornerRadius: 8))
    }

    private func toxicRow(_ row: ModeratedComment) -> some View {
        HStack(spacing: 6) {
            Text(String(format: "%.0f%%", row.probability * 100))
                .font(.system(size: 10, weight: .bold, design: .monospaced)).foregroundStyle(.red)
                .frame(width: 30, alignment: .trailing)
            Text(row.text.replacingOccurrences(of: "\n", with: " "))
                .font(.system(size: 11)).lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
            Text(String(format: "raters %.0f%%", row.humanToxicity * 100))
                .font(.system(size: 9, design: .monospaced)).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 6).padding(.vertical, 3)
        .background(Color.red.opacity(0.16), in: .rect(cornerRadius: 5))
    }

    private func metric(_ title: String, _ value: String, big: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.system(size: 9, weight: .semibold, design: .monospaced)).foregroundStyle(.secondary)
            Text(value).font(.system(size: big ? 34 : 16, weight: .semibold, design: .rounded)).monospacedDigit()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
