import AppKit
import SwiftUI
import UniformTypeIdentifiers
import WebKit

struct ContentView: View {
    @EnvironmentObject private var model: DemoModel

    var body: some View {
        HSplitView {
            sidebar
                .frame(minWidth: 380, idealWidth: 420, maxWidth: 520)
            ZStack(alignment: .topTrailing) {
                if model.target == .web {
                    WebViewContainer(webView: model.driver.webView)
                } else {
                    SnapshotSchematic(snapshot: model.lastSnapshot, rows: model.rows)
                }
                hud.padding(12)
            }
            .frame(minWidth: 640)
        }
    }

    // MARK: Sidebar

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 10) {
            modelSection
            Divider()
            documentSection
            Divider()
            pageSection
            Divider()
            runSection
            Divider()
            decisionsSection
            if let message = model.errorMessage {
                Text(message).font(.callout).foregroundStyle(.red).textSelection(.enabled)
            }
        }
        .padding(14)
    }

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("CUA-S1-FORMS · Core ML").font(.headline)
                Spacer()
                Button("Load model") { model.loadModel() }
            }
            Text(model.modelStatus).font(.callout).foregroundStyle(.secondary)
            if let placement = model.placement {
                Text(
                    "Compute plan: \(placement.neuralEngine)/\(placement.total) ops on Neural Engine, \(placement.cpu) CPU"
                )
                .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var documentSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Source document").font(.headline)
                Spacer()
                Button("Sample profile") { model.loadSampleDocument() }
                Button("Open…") { openDocument() }
            }
            if !model.documentName.isEmpty {
                Text("\(model.documentName) · \(model.enabledEntities.count) of \(model.entities.count) entities")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Text("Answer sheet").font(.subheadline).fontWeight(.medium)
                Spacer()
                if !model.answersName.isEmpty {
                    Text("\(model.answersName) · \(model.answers.count) answers").font(.caption).foregroundStyle(
                        .secondary)
                    Button("Clear") { model.clearAnswers() }
                }
                Button("Open…") { openAnswers() }
            }
            if !model.entities.isEmpty {
                ScrollView {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(model.entities) { entity in
                            Toggle(isOn: Binding(get: { entity.enabled }, set: { _ in model.toggle(entity) })) {
                                HStack(spacing: 4) {
                                    Text(entity.label).fontWeight(.medium)
                                    Text(entity.value).foregroundStyle(.secondary).lineLimit(1)
                                }
                                .font(.caption)
                            }
                            .toggleStyle(.checkbox)
                        }
                    }
                }
                .frame(maxHeight: 150)
            }
        }
    }

    private var pageSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Target").font(.headline)
                Spacer()
                Picker("", selection: $model.target) {
                    Text("Embedded web page").tag(DemoModel.Target.web)
                    ForEach(model.applications, id: \.processIdentifier) { app in
                        Text(app.localizedName ?? "?").tag(DemoModel.Target.application(app.processIdentifier))
                    }
                }
                .labelsHidden()
                .frame(maxWidth: 220)
                Button {
                    model.refreshApplications()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
            }
            if model.target == .web {
                HStack {
                    TextField("https://…", text: $model.urlField)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit { model.navigate() }
                    Button("Go") { model.navigate() }
                    Button("Sample form") { model.loadSampleForm() }
                }
            } else {
                HStack {
                    TextField("Window title contains… (empty = front window)", text: $model.windowFilter)
                        .textFieldStyle(.roundedBorder)
                    Button("Observe window") { model.observe() }
                    Button("Sample PDF in Preview") { model.openSamplePDF() }
                    if !model.accessibilityTrusted {
                        Text("Accessibility access not granted").font(.caption).foregroundStyle(.red)
                    }
                }
            }
            if !model.pageTitle.isEmpty {
                Text(model.pageTitle).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
        }
    }

    private var runSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Button("Plan only") { model.run(execute: false) }
                Button("Fill form") { model.run(execute: true) }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: .command)
                if model.hasExecutablePlan {
                    Button("Execute plan") { model.executePlan() }
                }
                if let countdown = model.countdown {
                    Button("Disarm (\(countdown))") { model.disarm() }
                } else {
                    Button("Arm 5 s") { model.arm() }
                }
                if model.isRunning {
                    Button("Stop") { model.stop() }
                    ProgressView().controlSize(.small)
                }
                Spacer()
                Toggle("Allow submit click", isOn: $model.allowSubmit)
            }
            Text("Hotkeys from any app: 9 run · ⌃⌥⌘A arm 5 s · ⌃⌥⌘S stop")
                .font(.caption2).foregroundStyle(.secondary)
            HStack {
                Text("Min confidence \(model.minConfidence, format: .number.precision(.fractionLength(2)))")
                    .font(.caption).frame(width: 130, alignment: .leading)
                Slider(value: $model.minConfidence, in: 0...1)
            }
            HStack {
                Text("Typing \(Int(model.characterDelayMilliseconds)) ms/char")
                    .font(.caption).frame(width: 130, alignment: .leading)
                Slider(value: $model.characterDelayMilliseconds, in: 0...60)
            }
        }
        .disabled(model.isRunning && false)
    }

    private var decisionsSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Decisions").font(.headline)
                Spacer()
                if model.scoredCount > 0, let median = model.medianLatency {
                    Text("\(model.scoredCount) scored · \(model.actionCount) actions · median \(Self.format(median))")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 3) {
                        ForEach(model.rows) { row in
                            HStack(spacing: 4) {
                                if row.status == "planned" && (row.action == .fill || row.action == .check) {
                                    Toggle(
                                        "",
                                        isOn: Binding(get: { row.approved }, set: { _ in model.toggleApproval(row) })
                                    )
                                    .toggleStyle(.checkbox).labelsHidden()
                                }
                                DecisionRowView(row: row)
                            }
                            .id(row.id)
                        }
                    }
                }
                .onChange(of: model.rows.count) { _, _ in
                    if let last = model.rows.last { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
            .frame(maxHeight: .infinity)
        }
    }

    // MARK: HUD

    private var hud: some View {
        VStack(alignment: .trailing, spacing: 2) {
            if let countdown = model.countdown {
                Text("starting in \(countdown)")
                    .font(.system(.title2, design: .rounded)).fontWeight(.bold)
            }
            HStack(spacing: 6) {
                Circle().fill(model.placement == nil ? Color.gray : Color.green).frame(width: 8, height: 8)
                Text(model.placement == nil ? "On-device · Core ML" : "On-device · Neural Engine")
                    .font(.system(.caption, design: .rounded)).fontWeight(.semibold)
            }
            if let latency = model.lastLatency {
                Text(
                    "decision \(Self.format(latency))"
                        + (model.medianLatency.map { " · median \(Self.format($0))" } ?? "")
                )
                .font(.system(.caption2, design: .monospaced))
            }
            if let placement = model.placement {
                Text("\(placement.neuralEngine)/\(placement.total) ops on ANE")
                    .font(.system(.caption2, design: .monospaced))
            }
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
        .allowsHitTesting(false)
    }

    static func format(_ duration: Duration) -> String {
        let milliseconds =
            Double(duration.components.seconds) * 1000
            + Double(duration.components.attoseconds) / 1e15
        return String(format: "%.2f ms", milliseconds)
    }

    private func openAnswers() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.plainText]
        panel.allowsMultipleSelection = false
        panel.message = "Choose a text file with `question text contains => answer` lines"
        if panel.runModal() == .OK, let url = panel.url {
            model.loadAnswers(url)
        }
    }

    private func openDocument() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.pdf, .plainText]
        panel.allowsMultipleSelection = false
        panel.message = "Choose a PDF or text file with `Label: value` lines"
        if panel.runModal() == .OK, let url = panel.url {
            model.loadDocument(url)
        }
    }
}

private struct DecisionRowView: View {
    let row: DecisionRow

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(row.element.role).font(.caption2).foregroundStyle(.secondary).frame(width: 60, alignment: .leading)
            VStack(alignment: .leading, spacing: 1) {
                Text(row.element.label.isEmpty ? "(unlabeled)" : row.element.label).font(.caption).lineLimit(1)
                HStack(spacing: 4) {
                    Text(row.action.rawValue).fontWeight(.semibold).foregroundStyle(color)
                    if row.action == .fill, let entity = row.entity {
                        Text("← \(entity.label): \(entity.value)").lineLimit(1)
                    }
                    if row.action == .answer, let entity = row.entity {
                        Text("← \(entity.value)").lineLimit(1)
                    }
                }
                .font(.caption2)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 1) {
                if let confidence = row.confidence, let latency = row.latency {
                    Text("\(Int((confidence * 100).rounded()))% · \(ContentView.format(latency))")
                        .font(.system(.caption2, design: .monospaced))
                }
                Text(row.status).font(.caption2).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }

    private var color: Color {
        switch row.action {
        case .fill: return .blue
        case .check: return .purple
        case .click: return .orange
        case .skip: return .secondary
        case .attach: return .teal
        case .answer: return .green
        }
    }
}

/// What the model observed, drawn to scale: control frames with their derived labels,
/// tinted by the decision once one exists.
private struct SnapshotSchematic: View {
    let snapshot: PageSnapshot?
    let rows: [DecisionRow]

    var body: some View {
        GeometryReader { geometry in
            if let snapshot, !snapshot.elements.isEmpty {
                let bounds = snapshot.elements.map(\.frame).reduce(snapshot.elements[0].frame) { $0.union($1) }
                    .insetBy(dx: -40, dy: -40)
                let scale = min(geometry.size.width / bounds.width, geometry.size.height / bounds.height)
                ZStack(alignment: .topLeading) {
                    Color(nsColor: .textBackgroundColor)
                    ForEach(snapshot.elements) { element in
                        let frame = element.frame
                        let decision = rows.first { $0.element.token == element.token }
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 2)
                                .fill(color(for: decision).opacity(0.18))
                            RoundedRectangle(cornerRadius: 2)
                                .stroke(color(for: decision), lineWidth: 1)
                            Text(element.label.isEmpty ? element.role : element.label)
                                .font(.system(size: max(7, 9 * scale)))
                                .lineLimit(1)
                                .padding(.leading, 2)
                        }
                        .frame(width: max(frame.width * scale, 6), height: max(frame.height * scale, 6))
                        .offset(x: (frame.minX - bounds.minX) * scale, y: (frame.minY - bounds.minY) * scale)
                    }
                    Text("\(snapshot.url) · \(snapshot.elements.count) controls observed through Accessibility")
                        .font(.caption).foregroundStyle(.secondary).padding(8)
                }
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "macwindow.on.rectangle").font(.largeTitle).foregroundStyle(.secondary)
                    Text("Observe window to see what the model will score").foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    private func color(for row: DecisionRow?) -> Color {
        switch row?.action {
        case .fill: return .blue
        case .check: return .purple
        case .click: return .orange
        case .attach: return .teal
        case .answer: return .green
        case .skip: return .gray
        case nil: return .secondary
        }
    }
}

private struct WebViewContainer: NSViewRepresentable {
    let webView: WKWebView

    func makeNSView(context: Context) -> WKWebView { webView }
    func updateNSView(_ nsView: WKWebView, context: Context) {}
}
