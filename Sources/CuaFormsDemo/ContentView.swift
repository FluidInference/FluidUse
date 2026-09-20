import AppKit
import SwiftUI
import UniformTypeIdentifiers
import WebKit

struct ContentView: View {
    @EnvironmentObject private var model: DemoModel

    @State private var lowerPane = LowerPane.console

    enum LowerPane: String, CaseIterable {
        case console = "Console"
        case decisions = "Decisions"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            modelSection
            Divider()
            documentSection
            Divider()
            pageSection
            Divider()
            runSection
            Divider()
            HStack {
                Picker("", selection: $lowerPane) {
                    ForEach(LowerPane.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 200)
                Spacer()
                if model.scoredCount > 0, let median = model.medianLatency {
                    Text("\(model.scoredCount) scored · \(model.actionCount) actions · median \(Self.format(median))")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            hud
            Group {
                if model.target == .web {
                    VSplitView {
                        WebViewContainer(webView: model.driver.webView).frame(minHeight: 200)
                        lowerContent.frame(minHeight: 160)
                    }
                } else {
                    lowerContent
                }
            }
            .frame(maxHeight: .infinity)
            if let message = model.errorMessage {
                Text(message).font(.callout).foregroundStyle(.red).textSelection(.enabled)
            }
        }
        .padding(14)
    }

    @ViewBuilder private var lowerContent: some View {
        switch lowerPane {
        case .console: ConsoleView(lines: model.console)
        case .decisions: decisionsList
        }
    }

    // MARK: Sections

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

    @State private var showDocument = false

    /// One summary line by default; expand for the entity list and the answer sheet.
    private var documentSection: some View {
        DisclosureGroup(isExpanded: $showDocument) {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Button("Sample profile") { model.loadSampleDocument() }
                    Button("Open document…") { openDocument() }
                    Spacer()
                    if !model.answersName.isEmpty { Button("Clear answers") { model.clearAnswers() } }
                    Button("Open answer sheet…") { openAnswers() }
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
            .padding(.top, 4)
        } label: {
            HStack(spacing: 6) {
                Text("Source").font(.headline)
                Text(
                    model.documentName.isEmpty
                        ? "none" : "\(model.documentName) · \(model.enabledEntities.count) entities"
                )
                .font(.caption).foregroundStyle(.secondary)
                if !model.answersName.isEmpty {
                    Text("· \(model.answersName) · \(model.answers.count) answers").font(.caption).foregroundStyle(
                        .secondary)
                }
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

    private var decisionsList: some View {
        VStack(alignment: .leading, spacing: 4) {
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
        HStack(spacing: 10) {
            Circle().fill(model.placement == nil ? Color.gray : Color.green).frame(width: 8, height: 8)
            Text(model.placement == nil ? "On-device · Core ML" : "On-device · Neural Engine")
                .font(.system(.caption, design: .rounded)).fontWeight(.semibold)
            if let placement = model.placement {
                Text("\(placement.neuralEngine)/\(placement.total) ops on ANE").font(
                    .system(.caption2, design: .monospaced))
            }
            if let latency = model.lastLatency {
                Text(
                    "decision \(Self.format(latency))"
                        + (model.medianLatency.map { " · median \(Self.format($0))" } ?? "")
                )
                .font(.system(.caption2, design: .monospaced))
            }
            if let countdown = model.countdown {
                Text("starting in \(countdown)").font(.system(.caption, design: .rounded)).fontWeight(.bold)
            }
            Spacer()
            utilization
        }
    }

    private var utilization: some View {
        UtilizationView(monitor: model.monitor)
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

/// Observes the monitor directly so the overlay refreshes with each sample.
private struct UtilizationView: View {
    @ObservedObject var monitor: SystemMonitor

    var body: some View {
        HStack(spacing: 10) {
            Text(String(format: "CPU app %.0f%%", monitor.processCPUPercent))
            Text(String(format: "system %.0f%%", monitor.systemCPUPercent))
            if let ane = monitor.anePowerMilliwatts {
                Text("ANE \(ane) mW")
            } else {
                Text(String(format: "ANE busy %.1f%%", monitor.aneDutyPercent))
            }
            if let cpu = monitor.cpuPowerMilliwatts { Text("CPU \(cpu) mW") }
        }
        .font(.system(.caption2, design: .monospaced))
    }
}

/// Terminal-style log of every model call: the exact context, the options, and the
/// ranked choices, so the decisions are visibly the model's.
private struct ConsoleView: View {
    let lines: [String]

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(lines.enumerated()), id: \.offset) { index, line in
                        Text(line)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(color(for: line))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(index)
                    }
                }
                .padding(10)
            }
            .background(Color(red: 0.07, green: 0.08, blue: 0.1))
            .onChange(of: lines.count) { _, count in
                if count > 0 { proxy.scrollTo(count - 1, anchor: .bottom) }
            }
        }
    }

    private func color(for line: String) -> Color {
        if line.hasPrefix("▶") { return Color(red: 0.55, green: 0.8, blue: 1) }
        if line.hasPrefix("$") { return Color(red: 0.75, green: 0.75, blue: 0.75) }
        if line.hasPrefix("■") { return Color(red: 0.55, green: 0.85, blue: 0.55) }
        if line.contains("→") { return Color(red: 1, green: 0.85, blue: 0.4) }
        return Color(red: 0.8, green: 0.82, blue: 0.85)
    }
}

private struct WebViewContainer: NSViewRepresentable {
    let webView: WKWebView

    func makeNSView(context: Context) -> WKWebView { webView }
    func updateNSView(_ nsView: WKWebView, context: Context) {}
}
