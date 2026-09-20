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

    /// Where decisions are executed: the embedded page or another app's window.
    enum Target: Hashable {
        case web
        case application(pid_t)
    }

    let driver = WebFormDriver()
    @Published var target: Target = .web
    @Published var applications: [NSRunningApplication] = []
    @Published var lastSnapshot: PageSnapshot?
    private var axDriver: AccessibilityFormDriver?
    private var manager: CuaS1FormsManager?
    private var runTask: Task<Void, Never>?

    init() {
        driver.onNavigation = { [weak self] in self?.refreshPageInfo() }
        refreshApplications()
        if ProcessInfo.processInfo.environment["CUA_DEMO_AUTORUN"] != nil { autorun() }
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
            if let axDriver, axDriver.application.processIdentifier == pid { return axDriver }
            guard let app = applications.first(where: { $0.processIdentifier == pid }) else {
                throw AccessibilityFormDriver.DriverError.noWindow("pid \(pid)")
            }
            let created = AccessibilityFormDriver(application: app)
            axDriver = created
            return created
        }
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
            if let appName = ProcessInfo.processInfo.environment["CUA_DEMO_TARGET"],
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
                if let axDriver = driver as? AccessibilityFormDriver, execute { axDriver.activate() }
                try await performRun(driver: driver, manager: manager, entities: entities, execute: execute)
            } catch is CancellationError {
                errorMessage = "Stopped"
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    func stop() {
        runTask?.cancel()
    }

    private func performRun(
        driver: any FormDriver, manager: CuaS1FormsManager, entities: [Entity], execute: Bool
    ) async throws {
        let snapshot = try await driver.snapshot()
        lastSnapshot = snapshot
        pageTitle = snapshot.title
        print("observed \(snapshot.elements.count) elements in \"\(snapshot.title)\"")
        let title = FormSchema.normalizeTitle(snapshot.title)
        let options = FormSchema.renderOptions(entities: entities)
        let threshold = Float(minConfidence)
        let delay = Duration.milliseconds(characterDelayMilliseconds)
        var pendingClicks: [DecisionRow] = []

        for element in snapshot.elements where element.isActionable {
            try Task.checkCancellation()
            let context = FormSchema.renderContext(formTitle: title, element: element)
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
                pendingClicks.append(row)
                continue
            }
            guard execute else { continue }
            try await driver.highlight(element.token, on: true)
            defer { Task { try? await driver.highlight(element.token, on: false) } }
            switch action {
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
