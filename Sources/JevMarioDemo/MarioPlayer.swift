import CoreGraphics
import FluidUse
import Foundation
import Observation

/// The control loop: while Mario is on the ground, JEV-9B answers three yes/no questions about close-ups of the
/// screen just ahead of him (enemy? pipe or step? gap?); if any says yes he takes a long running jump, otherwise he
/// runs right. The move plays out frame by frame. Mirrors model-lab `jev-9b-coreml/conversion/mario.py`.
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

    enum Move: String { case run, jump }

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
    var status = "Loading JEV-9B (Core ML)…"
    var finished = false

    private var manager: JevManager?
    private var bridge: MarioBridge?
    private var loop: Task<Void, Never>?
    private var started = false

    /// Starts the first game; SwiftUI may re-run the view's `.task`, so later calls are ignored (use `restart`).
    func start(modelDirectory: URL, bridgeCommand: [String]) {
        guard !started else { return }
        started = true
        run(modelDirectory: modelDirectory, bridgeCommand: bridgeCommand)
    }

    private func run(modelDirectory: URL, bridgeCommand: [String]) {
        loop?.cancel()
        loop = Task {
            do {
                if manager == nil {
                    let started = Date()
                    manager = try await JevManager.load(from: modelDirectory, bucket: 256)
                    status = String(format: "Model loaded in %.0f s", Date().timeIntervalSince(started))
                }
                if bridge == nil { bridge = try MarioBridge(command: bridgeCommand) }
                try await play()
            } catch is CancellationError {
            } catch {
                status = "Error: \(error.localizedDescription)"
            }
        }
    }

    func restart(modelDirectory: URL, bridgeCommand: [String]) {
        finished = false
        decisions = 0
        run(modelDirectory: modelDirectory, bridgeCommand: bridgeCommand)
    }

    private func play() async throws {
        guard let manager, let bridge else { return }
        var step = try await bridge.reset()
        show(step)
        while !step.done {
            try Task.checkCancellation()
            // 1. decide (emulator paused)
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
            // 2. play the move at 60 fps, then keep running until Mario lands
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
                let elapsed = Date().timeIntervalSince(tick)
                if elapsed < 1 / 60 { try await Task.sleep(for: .seconds(1 / 60 - elapsed)) }
            }
        }
        finished = true
        status =
            step.flag
            ? "Reached the flag in \(decisions) decisions" : "Mario died at x \(step.x) after \(decisions) decisions"
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
