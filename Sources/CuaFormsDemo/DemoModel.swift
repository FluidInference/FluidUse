@preconcurrency import CoreML
import AppKit
import FluidAudio
import Foundation
import SwiftUI

/// One scored element, in the order the model saw it.
struct DecisionRow: Identifiable, Sendable {
    let id = UUID()
    let element: FormElement
    let action: FormAction
    let entity: Entity?
    /// Nil for host rules that did not go through the model.
    let confidence: Float?
    let latency: Duration?
    var status: String
    /// Plan-only rows can be unchecked before "Execute plan" so a wrong fill never runs.
    var approved = true
}

/// Operation placement from the public compute plan, so the badge is measured, not asserted.
struct ComputePlacement: Sendable {
    let neuralEngine: Int
    let cpu: Int
    let gpu: Int
    var total: Int { neuralEngine + cpu + gpu }
}

@MainActor
final class DemoModel: ObservableObject {
    @Published var modelStatus = "Model not loaded"
    @Published var placement: ComputePlacement?
    @Published var entities: [Entity] = []
    @Published var answers: [PredeterminedAnswer] = []
    @Published var answersName = ""
    @Published var documentName = ""
    private(set) var documentURL: URL?
    @Published var pageTitle = ""
    @Published var pageURL = ""
    @Published var urlField = ""
    @Published var rows: [DecisionRow] = []
    @Published var isRunning = false
    @Published var minConfidence: Double = 0.5
    @Published var characterDelayMilliseconds: Double = 18
    @Published var allowSubmit = false
    @Published var lastLatency: Duration?
    @Published var errorMessage: String?
    /// Seconds left before an armed run starts; nil when not armed.
    @Published var countdown: Int?
    /// Model input and output per decision, verbatim, for the console pane.
    @Published var console: [String] = []
    let monitor = SystemMonitor()
    private var countdownTask: Task<Void, Never>?
    private var hotkeyMonitors: [Any] = []

    /// Where decisions are executed: the embedded page or another app's window.
    enum Target: Hashable {
        case web
        case application(pid_t)
    }

    let driver = WebFormDriver()
    @Published var target: Target = .web
    @Published var applications: [NSRunningApplication] = []
    @Published var lastSnapshot: PageSnapshot?
    /// Substring of the target window's title; empty means the app's focused window.
    @Published var windowFilter = ""
    private var axDriver: AccessibilityFormDriver?
    private var manager: CuaS1FormsManager?
    private var runTask: Task<Void, Never>?

    init() {
        driver.onNavigation = { [weak self] in self?.refreshPageInfo() }
        refreshApplications()
        installHotkeys()
        monitor.start()
        if ProcessInfo.processInfo.environment["CUA_DEMO_AUTORUN"] != nil { autorun() }
    }

    // MARK: Recording triggers

    /// Starts (or, with a reviewed plan pending, executes) after a visible countdown so a
    /// screen recording can begin first. The overlay shows the remaining seconds.
    func arm(seconds: Int = 5) {
        guard !isRunning, countdown == nil else { return }
        countdown = seconds
        countdownTask = Task {
            var remaining = seconds
            while remaining > 0 {
                try? await Task.sleep(for: .seconds(1))
                guard !Task.isCancelled else {
                    countdown = nil
                    return
                }
                remaining -= 1
                countdown = remaining
            }
            countdown = nil
            trigger()
        }
    }

    func disarm() {
        countdownTask?.cancel()
        countdown = nil
    }

    /// What the hotkey and the countdown do: execute a reviewed plan if there is one,
    /// otherwise fill the form.
    func trigger() {
        if hasExecutablePlan { executePlan() } else { run(execute: true) }
    }

    /// ⌃⌥⌘F starts the run from any app, ⌃⌥⌘A arms the countdown, ⌃⌥⌘S stops. Global
    /// monitoring needs the same Accessibility permission the driver already has.
    private func installHotkeys() {
        let required: NSEvent.ModifierFlags = [.control, .option, .command]
        let handle: (NSEvent) -> Void = { [weak self] event in
            guard let key = event.charactersIgnoringModifiers?.lowercased() else { return }
            let modifiers = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
            // A bare 9 is the recording trigger; it is ignored while a run is in progress.
            if key == "9", modifiers.isEmpty {
                Task { @MainActor in self?.trigger() }
                return
            }
            guard modifiers == required else { return }
            Task { @MainActor in
                switch key {
                case "f": self?.trigger()
                case "a": self?.arm()
                case "s":
                    self?.disarm()
                    self?.stop()
                default: break
                }
            }
        }
        if let global = NSEvent.addGlobalMonitorForEvents(matching: .keyDown, handler: handle) {
            hotkeyMonitors.append(global)
        }
        if let local = NSEvent.addLocalMonitorForEvents(
            matching: .keyDown,
            handler: { event in
                handle(event)
                return event
            })
        {
            hotkeyMonitors.append(local)
        }
    }

    var accessibilityTrusted: Bool { AccessibilityFormDriver.isTrusted }

    func refreshApplications() {
        applications = AccessibilityFormDriver.candidates()
        if case .application(let pid) = target, !applications.contains(where: { $0.processIdentifier == pid }) {
            target = .web
        }
    }

    /// The driver for the selected target; the Accessibility driver is rebuilt per app.
    private func currentDriver() throws -> any FormDriver {
        switch target {
        case .web:
            return driver
        case .application(let pid):
            if let axDriver, axDriver.application.processIdentifier == pid {
                axDriver.windowTitleFilter = windowFilter
                return axDriver
            }
            guard let app = applications.first(where: { $0.processIdentifier == pid }) else {
                throw AccessibilityFormDriver.DriverError.noWindow("pid \(pid)")
            }
            let created = AccessibilityFormDriver(application: app)
            created.windowTitleFilter = windowFilter
            axDriver = created
            return created
        }
    }

    /// Appends the exact model input and its top choices to the console.
    private func logDecision(context: String, options: [String], result: CuaS1FormsResult, latency: Duration) {
        var lines = context.components(separatedBy: "\n").map { "  " + $0 }
        let shown = options.prefix(4).joined(separator: " | ")
        lines.append("  options[\(options.count)]: \(shown)\(options.count > 4 ? " | …" : "")")
        let ranked = result.probabilities.enumerated().sorted { $0.element > $1.element }.prefix(3)
        let top = ranked.map { String(format: "%@ %.1f%%", options[$0.offset], $0.element * 100) }.joined(
            separator: "  ·  ")
        lines.append("  → \(top)")
        lines.append("  model call \(ContentView.format(latency)) on Neural Engine")
        appendConsole(["▶ CUA-S1-FORMS"] + lines)
    }

    private func appendConsole(_ lines: [String]) {
        console.append(contentsOf: lines)
        if console.count > 600 { console.removeFirst(console.count - 600) }
    }

    /// Applies an answer-sheet entry without consulting the model and logs it as such.
    private func applyAnswer(
        _ answer: PredeterminedAnswer, to element: FormElement, driver: any FormDriver, execute: Bool
    ) async throws {
        var row = DecisionRow(
            element: element, action: .answer, entity: Entity(label: answer.question, value: answer.value),
            confidence: nil, latency: nil, status: execute ? "answer sheet" : "planned")
        print("\(element.role) \"\(element.label.prefix(60))\" -> answer sheet [\(answer.value)]")
        appendConsole(["■ host · answer sheet (model not consulted): \"\(element.label.prefix(60))\" → \(answer.value)"]
        )
        rows.append(row)
        let rowIndex = rows.count - 1
        guard execute else { return }
        try await driver.highlight(element.token, on: true)
        defer { Task { try? await driver.highlight(element.token, on: false) } }
        do {
            switch element.role {
            case "ComboBox":
                if element.value.localizedCaseInsensitiveContains(answer.value) {
                    row.status = "already selected"
                } else {
                    if ["yes", "i agree", "i acknowledge", "i accept"].contains(answer.value.lowercased()) {
                        // Consent lists word their one option differently ("Acknowledge/Confirm"),
                        // and typing "Yes" leaves the control with no matches; take the first option.
                        try await driver.selectAffirmative(in: element.token)
                        row.status = "first option · answer sheet"
                    } else {
                        try await driver.select(answer.value, in: element.token)
                        row.status = "selected · answer sheet"
                    }
                }
            case "CheckBox":
                let wantsChecked = ["yes", "true", "checked", "on"].contains(answer.value.lowercased())
                if element.checked == wantsChecked {
                    row.status = "already \(wantsChecked ? "checked" : "unchecked")"
                } else {
                    try await driver.click(element.token)
                    row.status = "\(wantsChecked ? "checked" : "unchecked") · answer sheet"
                }
            default:
                if element.value == answer.value {
                    row.status = "already filled"
                } else {
                    let perCharacter = min(
                        Duration.milliseconds(characterDelayMilliseconds),
                        .milliseconds(1200 / max(answer.value.count, 1)))
                    try await driver.type(answer.value, into: element.token, characterDelay: perCharacter)
                    row.status = "filled · answer sheet"
                }
            }
        } catch {
            row.status = "answer failed: \(error.localizedDescription)"
        }
        rows[rowIndex].status = row.status
        appendConsole(["  ✓ \(row.status)"])
        try await Task.sleep(for: .milliseconds(120))
    }

    /// Re-observe the target without scoring, for the schematic pane.
    func observe() {
        Task {
            do {
                let snapshot = try await currentDriver().snapshot()
                lastSnapshot = snapshot
                pageTitle = snapshot.title
                errorMessage = nil
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    /// `CUA_DEMO_AUTORUN=1`: load the model, the sample profile and the sample form, fill it,
    /// and print every decision. `CUA_DEMO_QUIT=1` exits afterwards (smoke test / recording).
    private func autorun() {
        Task {
            if let path = ProcessInfo.processInfo.environment["CUA_DEMO_DOCUMENT"] {
                loadDocument(URL(fileURLWithPath: path))
            } else {
                loadSampleDocument()
            }
            if let path = ProcessInfo.processInfo.environment["CUA_DEMO_ANSWERS"] {
                loadAnswers(URL(fileURLWithPath: path))
            }
            if let filter = ProcessInfo.processInfo.environment["CUA_DEMO_WINDOW"] { windowFilter = filter }
            if ProcessInfo.processInfo.environment["CUA_DEMO_SAMPLE_PDF"] != nil {
                openSamplePDF()
                while target == .web { try? await Task.sleep(for: .milliseconds(200)) }
            } else if let appName = ProcessInfo.processInfo.environment["CUA_DEMO_TARGET"],
                let app = applications.first(where: { $0.localizedName == appName })
            {
                target = .application(app.processIdentifier)
            } else if let page = ProcessInfo.processInfo.environment["CUA_DEMO_URL"], let url = URL(string: page) {
                driver.load(url)
            } else {
                loadSampleForm()
            }
            do {
                let loaded = try await CuaS1FormsManager.load(computeUnits: .cpuAndNeuralEngine)
                manager = loaded
                modelStatus = "Loaded · 706K params · FP16 · CPU + Neural Engine"
                placement = await Self.measurePlacement()
                if let placement {
                    print("compute plan: \(placement.neuralEngine)/\(placement.total) ops on ANE, \(placement.cpu) CPU")
                }
            } catch {
                print("model load failed: \(error.localizedDescription)")
                if ProcessInfo.processInfo.environment["CUA_DEMO_QUIT"] != nil { exit(1) }
                return
            }
            if target == .web {
                while driver.isLoading || driver.webView.url == nil { try? await Task.sleep(for: .milliseconds(100)) }
            }
            try? await Task.sleep(for: .milliseconds(500))
            if ProcessInfo.processInfo.environment["CUA_DEMO_PREPARE"] != nil {
                observe()
                appendConsole(["ready · press 9 in the target app to run"])
                return
            }
            run(execute: ProcessInfo.processInfo.environment["CUA_DEMO_PLAN_ONLY"] == nil)
            _ = await runTask?.value
            if let median = medianLatency {
                print("scored \(scoredCount) elements, \(actionCount) actions, median \(ContentView.format(median))")
            }
            if let errorMessage { print("error: \(errorMessage)") }
            if let number = NSApp.windows.first?.windowNumber { print("window id: \(number)") }
            if ProcessInfo.processInfo.environment["CUA_DEMO_QUIT"] != nil { exit(errorMessage == nil ? 0 : 1) }
        }
    }

    var enabledEntities: [Entity] { entities.filter(\.enabled) }
    var maximumEntities: Int { CuaS1FormsManager.maximumOptions - FormSchema.fixedActions.count }
    var scoredCount: Int { rows.count }
    var actionCount: Int { rows.filter { $0.action != .skip }.count }

    var medianLatency: Duration? {
        let sorted = rows.compactMap(\.latency).sorted()
        guard !sorted.isEmpty else { return nil }
        return sorted[sorted.count / 2]
    }

    // MARK: Model

    func loadModel() {
        modelStatus = "Downloading / loading…"
        errorMessage = nil
        Task {
            do {
                let loaded = try await CuaS1FormsManager.load(computeUnits: .cpuAndNeuralEngine)
                manager = loaded
                modelStatus = "Loaded · 706K params · FP16 · CPU + Neural Engine"
                placement = await Self.measurePlacement()
            } catch {
                modelStatus = "Load failed"
                errorMessage = error.localizedDescription
            }
        }
    }

    /// Reads preferred-device assignments from the compiled artifact FluidAudio cached.
    private static func measurePlacement() async -> ComputePlacement? {
        guard #available(macOS 14.4, *) else { return nil }
        let manager = FileManager.default
        guard let support = manager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            return nil
        }
        let root = support.appendingPathComponent("FluidAudio/Models", isDirectory: true)
        let target = ModelNames.CuaS1Forms.modelFile
        guard let enumerator = manager.enumerator(at: root, includingPropertiesForKeys: nil) else { return nil }
        var compiled: URL?
        while let url = enumerator.nextObject() as? URL {
            if url.lastPathComponent == target {
                compiled = url
                break
            }
        }
        guard let compiled else { return nil }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        guard let plan = try? await MLComputePlan.load(contentsOf: compiled, configuration: configuration),
            case .program(let program) = plan.modelStructure
        else { return nil }
        var counts = (ane: 0, cpu: 0, gpu: 0)
        func visit(_ operations: [MLModelStructure.Program.Operation]) {
            for operation in operations {
                if let usage = plan.deviceUsage(for: operation) {
                    switch usage.preferred {
                    case .neuralEngine: counts.ane += 1
                    case .cpu: counts.cpu += 1
                    case .gpu: counts.gpu += 1
                    @unknown default: break
                    }
                }
                for block in operation.blocks { visit(block.operations) }
            }
        }
        for function in program.functions.values { visit(function.block.operations) }
        return ComputePlacement(neuralEngine: counts.ane, cpu: counts.cpu, gpu: counts.gpu)
    }

    // MARK: Document

    func loadDocument(_ url: URL) {
        errorMessage = nil
        do {
            entities = try DocumentEntities.extract(from: url)
            documentName = url.lastPathComponent
            documentURL = url
            if entities.isEmpty {
                errorMessage = "No `Label: value` lines found in \(url.lastPathComponent)"
            } else if entities.count > maximumEntities {
                for index in entities.indices where index >= maximumEntities { entities[index].enabled = false }
                errorMessage = "The model scores at most \(maximumEntities) entities; extra ones start disabled"
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Loads an answer sheet (`question contains => answer` lines) applied by the harness.
    func loadAnswers(_ url: URL) {
        do {
            answers = PredeterminedAnswer.parse(try String(contentsOf: url, encoding: .utf8))
            answersName = url.lastPathComponent
            if answers.isEmpty { errorMessage = "No `question => answer` lines found in \(url.lastPathComponent)" }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func clearAnswers() {
        answers = []
        answersName = ""
    }

    func loadSampleDocument() {
        guard
            let url = Bundle.module.url(
                forResource: "sample-applicant", withExtension: "pdf", subdirectory: "Resources")
        else {
            errorMessage = "Sample document is missing from the bundle"
            return
        }
        loadDocument(url)
    }

    func toggle(_ entity: Entity) {
        guard let index = entities.firstIndex(where: { $0.id == entity.id }) else { return }
        entities[index].enabled.toggle()
    }

    // MARK: Page

    func navigate() {
        var text = urlField.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty else { return }
        if !text.contains("://") { text = "https://" + text }
        guard let url = URL(string: text) else {
            errorMessage = "Not a valid URL"
            return
        }
        rows = []
        driver.load(url)
    }

    func loadSampleForm() {
        guard let url = Bundle.module.url(forResource: "sample-form", withExtension: "html", subdirectory: "Resources")
        else {
            errorMessage = "Sample form is missing from the bundle"
            return
        }
        rows = []
        urlField = ""
        driver.load(url)
    }

    /// Copies the bundled fillable PDF to a temporary file (Preview auto-saves into it),
    /// opens it, and targets Preview once it is running.
    func openSamplePDF() {
        guard
            let source = Bundle.module.url(
                forResource: "sample-application", withExtension: "pdf", subdirectory: "Resources")
        else {
            errorMessage = "Sample PDF is missing from the bundle"
            return
        }
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("CuaFormsDemo", isDirectory: true)
        let copy = folder.appendingPathComponent("Example Robotics - Job Application.pdf")
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try? FileManager.default.removeItem(at: copy)
            try FileManager.default.copyItem(at: source, to: copy)
        } catch {
            errorMessage = error.localizedDescription
            return
        }
        guard let preview = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.Preview") else {
            errorMessage = "Preview is not available"
            return
        }
        rows = []
        lastSnapshot = nil
        NSWorkspace.shared.open([copy], withApplicationAt: preview, configuration: NSWorkspace.OpenConfiguration()) {
            _, error in
            Task { @MainActor in
                if let error {
                    self.errorMessage = error.localizedDescription
                    return
                }
                try? await Task.sleep(for: .seconds(2))
                self.refreshApplications()
                if let app = self.applications.first(where: { $0.bundleIdentifier == "com.apple.Preview" }) {
                    self.target = .application(app.processIdentifier)
                    self.observe()
                }
            }
        }
    }

    private func refreshPageInfo() {
        pageTitle = driver.webView.title ?? ""
        pageURL = driver.webView.url?.absoluteString ?? ""
    }

    // MARK: Run

    /// Plan every actionable element; with `execute`, apply fills and checks as each
    /// decision lands, then at most one submit click when authorized.
    func run(execute: Bool) {
        guard !isRunning else { return }
        guard let manager else {
            errorMessage = "Load the model first"
            return
        }
        let entities = enabledEntities
        guard !entities.isEmpty else {
            errorMessage = "Open a document first"
            return
        }
        guard entities.count <= maximumEntities else {
            errorMessage = "Disable entities until at most \(maximumEntities) remain"
            return
        }
        isRunning = true
        errorMessage = nil
        rows = []
        runTask = Task {
            defer { isRunning = false }
            do {
                let driver = try currentDriver()
                if let axDriver = driver as? AccessibilityFormDriver, execute {
                    axDriver.activate()
                    try await Task.sleep(for: .milliseconds(400))
                }
                try await performRun(driver: driver, manager: manager, entities: entities, execute: execute)
            } catch is CancellationError {
                errorMessage = "Stopped"
            } catch {
                errorMessage = error.localizedDescription
                print("error: \(error.localizedDescription)")
            }
        }
    }

    func stop() {
        runTask?.cancel()
    }

    func toggleApproval(_ row: DecisionRow) {
        guard let index = rows.firstIndex(where: { $0.id == row.id }) else { return }
        rows[index].approved.toggle()
    }

    var hasExecutablePlan: Bool {
        rows.contains { ($0.action == .fill || $0.action == .check) && $0.status == "planned" }
    }

    /// Acts on the approved fill/check rows of a plan-only run, in order, using the same
    /// element tokens; the page must not have changed since the plan.
    func executePlan() {
        guard !isRunning, hasExecutablePlan else { return }
        isRunning = true
        errorMessage = nil
        runTask = Task {
            defer { isRunning = false }
            do {
                let driver = try currentDriver()
                if let axDriver = driver as? AccessibilityFormDriver {
                    axDriver.activate()
                    try await Task.sleep(for: .milliseconds(400))
                }
                let delay = Duration.milliseconds(characterDelayMilliseconds)
                for index in rows.indices where rows[index].status == "planned" {
                    let row = rows[index]
                    if row.action == .answer, let entity = row.entity {
                        rows[index].status = "answer sheet"
                        try await applyAnswer(
                            PredeterminedAnswer(question: entity.label, value: entity.value), to: row.element,
                            driver: driver, execute: true)
                        rows.removeLast()  // applyAnswer appends its own row; keep the planned one updated
                        continue
                    }
                    guard row.action == .fill || row.action == .check else { continue }
                    guard row.approved else {
                        rows[index].status = "vetoed"
                        continue
                    }
                    try Task.checkCancellation()
                    try await driver.highlight(row.element.token, on: true)
                    defer { Task { try? await driver.highlight(row.element.token, on: false) } }
                    if row.action == .fill, let value = row.entity?.value {
                        let perCharacter = min(delay, .milliseconds(1200 / max(value.count, 1)))
                        try await driver.type(value, into: row.element.token, characterDelay: perCharacter)
                        rows[index].status = "filled"
                    } else if row.action == .check {
                        try await driver.click(row.element.token)
                        let checked = try await driver.isChecked(row.element.token)
                        rows[index].status = checked == true ? "checked" : "check not confirmed"
                    }
                    try await Task.sleep(for: .milliseconds(120))
                }
            } catch is CancellationError {
                errorMessage = "Stopped"
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func performRun(
        driver: any FormDriver, manager: CuaS1FormsManager, entities: [Entity], execute: Bool
    ) async throws {
        let snapshot = try await driver.snapshot()
        lastSnapshot = snapshot
        pageTitle = snapshot.title
        print("observed \(snapshot.elements.count) elements in \"\(snapshot.title)\"")
        appendConsole(["", "$ observe \(snapshot.url) → \(snapshot.elements.count) controls in \"\(snapshot.title)\""])
        let title = FormSchema.normalizeTitle(snapshot.title)
        let options = FormSchema.renderOptions(entities: entities)
        let threshold = Float(minConfidence)
        let delay = Duration.milliseconds(characterDelayMilliseconds)
        var pendingClicks: [DecisionRow] = []

        for element in snapshot.elements where element.isActionable {
            try Task.checkCancellation()
            if element.role != "Button", let answer = PredeterminedAnswer.match(element.label, in: answers) {
                try await applyAnswer(answer, to: element, driver: driver, execute: execute)
                continue
            }
            let context = FormSchema.renderContext(formTitle: title, element: element.forScoring)
            // Time the call off the main actor so UI work does not inflate the number.
            let (result, latency) = try await Task.detached(priority: .userInitiated) {
                let clock = ContinuousClock()
                let start = clock.now
                let result = try await manager.score(context: context, options: options)
                return (result, clock.now - start)
            }.value
            lastLatency = latency
            let (action, entityIndex) = FormSchema.decode(
                optionIndex: result.selectedIndex, entityCount: entities.count)
            let confidence: Float = result.probabilities[result.selectedIndex]
            monitor.recordModelCall(latency)
            logDecision(context: context, options: options, result: result, latency: latency)
            var row = DecisionRow(
                element: element, action: action, entity: entityIndex.map { entities[$0] },
                confidence: confidence, latency: latency, status: action == .skip ? "skipped" : "planned")
            rows.append(row)
            let rowIndex = rows.count - 1
            print(
                "\(element.role) \"\(element.label)\" -> \(action.rawValue)"
                    + (row.entity.map { " [\($0.label): \($0.value)]" } ?? "")
                    + String(format: " %.1f%% %@", confidence * 100, ContentView.format(latency)))

            guard action != .skip, confidence >= threshold else {
                if action != .skip { rows[rowIndex].status = "below threshold" }
                continue
            }
            if action == .click {
                if FormSchema.isSubmitControl(element) {
                    pendingClicks.append(row)
                } else {
                    rows[rowIndex].status = "ignored · not a submit control"
                }
                continue
            }
            guard execute else { continue }
            try await driver.highlight(element.token, on: true)
            defer { Task { try? await driver.highlight(element.token, on: false) } }
            switch action {
            case .fill where element.role == "ComboBox":
                let value = row.entity?.value ?? ""
                if element.value.localizedCaseInsensitiveContains(value) {
                    row.status = "already selected"
                } else {
                    do {
                        try await driver.select(value, in: element.token)
                        row.status = "selected · host rule"
                    } catch {
                        row.status = "select failed: \(error.localizedDescription)"
                    }
                }
            case .check where element.role == "ComboBox":
                do {
                    try await driver.selectAffirmative(in: element.token)
                    row.status = "affirmed · host rule"
                } catch {
                    row.status = "select failed: \(error.localizedDescription)"
                }
            case .fill:
                let value = row.entity?.value ?? ""
                if element.value == value {
                    row.status = "already filled"
                } else {
                    // Same-role scoring; value mutation goes through the page's native setter.
                    let perCharacter = min(delay, .milliseconds(1200 / max(value.count, 1)))
                    try await driver.type(value, into: element.token, characterDelay: perCharacter)
                    row.status = "filled"
                }
            case .check:
                if element.checked == true {
                    row.status = "already checked"
                } else {
                    try await driver.click(element.token)
                    let checked = try await driver.isChecked(element.token)
                    row.status = checked == true ? "checked" : "check not confirmed"
                }
            default:
                break
            }
            rows[rowIndex].status = row.status
            appendConsole(["  ✓ \(row.status)"])
            try await Task.sleep(for: .milliseconds(120))
        }

        // File inputs are outside the model's option vocabulary (upstream trains "Upload file"
        // as a skip), so attaching the source document is a host rule, reported as such.
        if execute, let documentURL {
            for element in snapshot.elements where element.isFileUpload {
                try Task.checkCancellation()
                var row = DecisionRow(
                    element: element, action: .attach, entity: nil, confidence: nil, latency: nil,
                    status: "host rule")
                if !element.value.isEmpty {
                    row.status = "already attached"
                    rows.append(row)
                    continue
                }
                try await driver.highlight(element.token, on: true)
                do {
                    try await driver.attach(documentURL, to: element.token)
                    row.status = "attached \(documentURL.lastPathComponent) · host rule"
                } catch {
                    row.status = "attach failed: \(error.localizedDescription)"
                }
                rows.append(row)
                print("FileUpload \"\(element.label)\" -> attach (host rule) \(row.status)")
                try await Task.sleep(for: .milliseconds(300))
                try await driver.highlight(element.token, on: false)
            }
        }

        guard execute, allowSubmit else { return }
        let submit = pendingClicks.filter { FormSchema.isSubmitControl($0.element) }.max {
            ($0.confidence ?? 0) < ($1.confidence ?? 0)
        }
        guard let submit, let rowIndex = rows.firstIndex(where: { $0.id == submit.id }) else { return }
        try await Task.sleep(for: .milliseconds(400))
        try await driver.highlight(submit.element.token, on: true)
        try await driver.click(submit.element.token)
        rows[rowIndex].status = "clicked"
    }
}
