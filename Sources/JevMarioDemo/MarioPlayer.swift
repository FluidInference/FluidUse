import CoreGraphics
import FluidUse
import Foundation
import Observation

/// The control loop. While Mario is on the ground the emulator pauses for a decision, then the move plays out frame by
/// frame at 60 fps.
/// - JEV-9B: three yes/no questions about close-ups just ahead of him (enemy? pipe or step? gap?); any yes -> long
///   running jump, else run. Mirrors model-lab `jev-9b-coreml/conversion/mario.py`.
/// - Student: a small pixel model (~9M parameters) picks run / jump / hop / wait from the pixels around Mario
///   (model-lab `mario-tiny`).
@available(macOS 15.0, *)
@MainActor
@Observable
final class MarioPlayer {
    struct Question: Identifiable {
        let id: String  // panel label
        let view: String
        let text: String
        let threshold: Float
    }

    enum Move: String { case run, jump, hop, wait }

    enum Brain: String, CaseIterable, Identifiable {
        case jev = "JEV-9B · 3 questions"
        case tiny = "Student · pixels"
        var id: String { rawValue }
    }

    /// (x offset from Mario, width, top relative to his feet, height) in screen px, and the state text.
    static let views: [String: (dx: Int, width: Int, dy: Int, height: Int, state: String)] = [
        "ahead": (16, 64, -48, 80, "Close-up of the area just ahead of Mario in a Super Mario Bros. level:"),
        "above": (
            16, 64, -48, 48,
            "Close-up of the space just ahead of Mario in a Super Mario Bros. level, at the height of his body:"
        ),
        "ground": (16, 96, -32, 64, "Close-up of the ground in a Super Mario Bros. level:"),
    ]
    static let questions: [Question] = [
        .init(
            id: "enemy ahead", view: "ahead", text: "Is there an enemy (a brown Goomba or a Koopa) in this picture?",
            threshold: 0.5),
        .init(
            id: "pipe / step ahead", view: "above",
            text: "Is there a solid obstacle (a pipe, a block or a step) in this picture?", threshold: 0.5),
        .init(
            id: "gap ahead", view: "ground", text: "Is the brick floor in this picture interrupted by a gap?",
            threshold: 0.5),
    ]

    static let right: UInt8 = 0b1000_0000, a: UInt8 = 0b0000_0001, b: UInt8 = 0b0000_0010
    static let moves: [Move: [UInt8]] = [
        .run: Array(repeating: right | b, count: 6),
        .jump: [right | b] + Array(repeating: right | a | b, count: 22) + Array(repeating: right | b, count: 4),
        .hop: [right | b] + Array(repeating: right | a | b, count: 8) + Array(repeating: right | b, count: 4),
        .wait: Array(repeating: 0, count: 6),
    ]
    static let flagX = 3160

    // published state
    var frame: CGImage?
    var viewImages: [String: CGImage] = [:]
    var probabilities: [String: Float] = [:]
    var move: Move = .run
    var thinking = false
    var decisions = 0
    var lastMilliseconds: Double = 0
    var x = 40
    var status = "Loading…"
    var finished = false
    var brain: Brain = ProcessInfo.processInfo.environment["JEV_MARIO_BRAIN"] == "student" ? .tiny : .jev
    var moveProbabilities: [Float] = []  // student: per MarioStudentPolicy.moves
    var studentParameters = ""
    /// Max speed: no 60 fps pacing, the emulator runs as fast as the bridge and the model allow.
    var turbo = ProcessInfo.processInfo.environment["MARIO_TURBO"] == "1"
    var gameFPS: Double = 0  // emulator frames per second over the last second
    private var frameTimes: [Date] = []
    var studentPlacement = ""

    private var manager: JevManager?
    private var tiny: MarioStudentPolicy?
    private var history: [Data] = []  // last 5 frames' pixels (the student also sees 4 frames back)
    private var bridge: MarioBridge?
    private var loop: Task<Void, Never>?
    private var started = false

    /// Starts the first game; SwiftUI may re-run the view's `.task`, so later calls are ignored (use `restart`).
    func start(modelDirectory: URL?, bridgeCommand: [String]) {
        guard !started else { return }
        started = true
        run(modelDirectory: modelDirectory, bridgeCommand: bridgeCommand)
    }

    private func run(modelDirectory: URL?, bridgeCommand: [String]) {
        loop?.cancel()
        loop = Task {
            do {
                let loadStarted = Date()
                switch brain {
                case .jev where manager == nil:
                    status =
                        modelDirectory == nil
                        ? "Downloading JEV-9B Core ML (~13 GB, first use only)…" : "Loading JEV-9B…"
                    let directory: URL
                    if let modelDirectory {
                        directory = modelDirectory
                    } else {
                        directory = try await JevModelStore.ensure()
                    }
                    status = "Loading JEV-9B (Core ML)…"
                    manager = try await JevManager.load(from: directory, bucket: 256)
                case .tiny where tiny == nil:
                    guard let path = ProcessInfo.processInfo.environment["MARIO_STUDENT_MODEL"] else {
                        throw DemoError.noStudentModel
                    }
                    let policy = try await MarioStudentPolicy(package: URL(fileURLWithPath: path))
                    studentParameters = policy.parameterLabel
                    studentPlacement = policy.placement
                    print("student: \(policy.parameterLabel) params, \(policy.placement)")
                    tiny = policy
                default: break
                }
                status = String(format: "\(brain.rawValue) ready in %.1f s", Date().timeIntervalSince(loadStarted))
                if bridge == nil { bridge = try MarioBridge(command: bridgeCommand) }
                try await play()
            } catch is CancellationError {
            } catch {
                status = "Error: \(error.localizedDescription)"
            }
        }
    }

    enum DemoError: Error, LocalizedError {
        case noStudentModel
        var errorDescription: String? { "Set MARIO_STUDENT_MODEL to the student's .mlpackage path" }
    }

    func restart(modelDirectory: URL?, bridgeCommand: [String]) {
        finished = false
        decisions = 0
        run(modelDirectory: modelDirectory, bridgeCommand: bridgeCommand)
    }

    /// Levels to play in order as (level, idle frames before the start), from MARIO_PLAYLIST ("1-1:0,3-2:41", idle
    /// optional) or MARIO_LEVEL. The student's default playlist is every level it has reached the flag on.
    static var playlist: [(level: String, idle: Int)] {
        let env = ProcessInfo.processInfo.environment
        let spec = env["MARIO_PLAYLIST"].flatMap { $0.isEmpty ? nil : $0 } ?? env["MARIO_LEVEL"] ?? "1-1"
        return spec.split(separator: ",").map { entry in
            let parts = entry.split(separator: ":")
            return (String(parts[0]), parts.count > 1 ? Int(parts[1]) ?? 0 : 0)
        }
    }

    var level = ""
    var results: [(level: String, flag: Bool)] = []

    private func play() async throws {
        results = []
        for (index, entry) in Self.playlist.enumerated() {
            try Task.checkCancellation()
            level = entry.level
            decisions = 0
            let flag = try await playLevel(entry.level, idle: entry.idle)
            results.append((entry.level, flag))
            print("LEVEL \(entry.level)\(entry.idle > 0 ? " (start +\(entry.idle))" : ""): \(status)")
            if index < Self.playlist.count - 1 { try await Task.sleep(for: .seconds(turbo ? 0.4 : 1.5)) }
        }
        finished = true
        let flags = results.filter(\.flag).count
        status = "\(flags)/\(results.count) levels reached the flag"
        print("PLAYLIST: \(status)")
    }

    private func playLevel(_ level: String, idle: Int) async throws -> Bool {
        guard let bridge else { return false }
        var step = try await bridge.reset(level: level, idle: idle)
        show(step)
        history = Array(repeating: step.pixels, count: 5)
        while !step.done {
            try Task.checkCancellation()
            // 1. decide (emulator paused)
            if brain == .tiny, let tiny {
                let started = Date()
                moveProbabilities = try tiny.decide(now: step.pixels, before: history[0], marioX: step.screenX)
                lastMilliseconds = Date().timeIntervalSince(started) * 1000
                let best = moveProbabilities.indices.max { moveProbabilities[$0] < moveProbabilities[$1] } ?? 0
                move = MarioStudentPolicy.moves[best]
                decisions += 1
                print(String(format: "%3d  x %4d  %@  %.2f ms", decisions, step.x, move.rawValue, lastMilliseconds))
                step = try await perform(move, from: step, bridge: bridge)
                continue
            }
            guard let manager else { return false }
            thinking = true
            let started = Date()
            var probs: [String: Float] = [:]
            var images: [String: CGImage] = [:]
            for (name, spec) in Self.views {
                guard let image = Self.view(step.frame, x: step.screenX, y: step.screenY, spec: spec) else { continue }
                images[name] = image
                let asked = Self.questions.filter { $0.view == name }
                let result = try await manager.decide(
                    state: [.text(spec.state), .image(image)], questions: asked.map { JevQuestion(.noul, $0.text) })
                for (q, d) in zip(asked, result.decisions) { probs[q.id] = d.yes ?? 0 }
            }
            lastMilliseconds = Date().timeIntervalSince(started) * 1000
            viewImages = images
            probabilities = probs
            move = Self.questions.contains { (probs[$0.id] ?? 0) > $0.threshold } ? .jump : .run
            decisions += 1
            thinking = false
            let line = Self.questions.map { String(format: "%@ %.2f", $0.id, probs[$0.id] ?? 0) }.joined(
                separator: "  ")
            print(
                String(format: "%3d  x %4d  %@  %@  %.0f ms", decisions, step.x, move.rawValue, line, lastMilliseconds))
            step = try await perform(move, from: step, bridge: bridge)
        }
        status =
            step.flag
            ? "Reached the flag in \(decisions) decisions" : "Mario died at x \(step.x) after \(decisions) decisions"
        return step.flag
    }

    /// Play the move at 60 fps, then keep running until Mario lands (max 90 frames).
    private func perform(
        _ move: Move, from start: MarioBridge.Step, bridge: MarioBridge
    ) async throws
        -> MarioBridge.Step
    {
        var step = start
        let buttons = Self.moves[move] ?? []
        var index = 0
        var airborne = 0
        while !step.done {
            let press: UInt8
            if index < buttons.count {
                press = buttons[index]
                index += 1
            } else if !step.onGround && airborne < 90 {
                press = Self.right | Self.b
                airborne += 1
            } else {
                break
            }
            let tick = Date()
            step = try await bridge.step(press)
            show(step)
            history = Array(history.dropFirst()) + [step.pixels]
            let now = Date()
            frameTimes.append(now)
            frameTimes.removeAll { now.timeIntervalSince($0) > 1 }
            gameFPS = Double(frameTimes.count)
            let elapsed = now.timeIntervalSince(tick)
            if !turbo, elapsed < 1 / 60 { try await Task.sleep(for: .seconds(1 / 60 - elapsed)) }
        }
        return step
    }

    private func show(_ step: MarioBridge.Step) {
        frame = step.frame
        x = step.x
    }

    /// A close-up: Mario's feet are 32 px below the RAM y; upscaled 4x with nearest-neighbour (PIL NEAREST).
    static func view(
        _ frame: CGImage, x: Int, y: Int, spec: (dx: Int, width: Int, dy: Int, height: Int, state: String)
    )
        -> CGImage?
    {
        let x0 = min(max(x + spec.dx, 0), frame.width - spec.width)
        let y0 = min(max(y + 32 + spec.dy, 0), frame.height - spec.height)
        guard let crop = frame.cropping(to: CGRect(x: x0, y: y0, width: spec.width, height: spec.height)) else {
            return nil
        }
        let scale = 4
        guard
            let context = CGContext(
                data: nil, width: spec.width * scale, height: spec.height * scale, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        context.interpolationQuality = .none
        context.draw(crop, in: CGRect(x: 0, y: 0, width: spec.width * scale, height: spec.height * scale))
        return context.makeImage()
    }
}
