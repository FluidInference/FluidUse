import LaneRunner
import SceneKit
import SwiftUI

/// Third-person 3D view of the runner, skinned with Kenney's CC0 Train Kit and Platformer Kit models.
@MainActor
final class RunnerScene {
    let scene = SCNScene()
    private let world = SCNNode()
    private let player = SCNNode()
    private let templates: [String: SCNNode]
    private static let laneWidth: CGFloat = 1.7
    private static let rowLength: CGFloat = 3.2
    private static let drawnRows = 8

    init() {
        var templates: [String: SCNNode] = [:]
        for (kit, names) in [
            ("train", ["train-electric-subway-a", "train-electric-subway-b", "track"]),
            (
                "platformer",
                ["character-oobi", "coin-gold", "fence-low-straight", "tree-pine", "tree"]
            ),
        ] {
            for name in names {
                guard
                    let url = Bundle.module.url(
                        forResource: name, withExtension: "obj", subdirectory: "Resources/Kenney/\(kit)"),
                    let loaded = try? SCNScene(url: url)
                else { continue }
                let node = SCNNode()
                for child in loaded.rootNode.childNodes { node.addChildNode(child) }
                let (low, high) = node.boundingBox
                // Center on x/z and stand on y = 0 so placement is by lane and row only.
                node.pivot = SCNMatrix4MakeTranslation((low.x + high.x) / 2, low.y, (low.z + high.z) / 2)
                templates[name] = node
            }
        }
        self.templates = templates
        buildStage()
    }

    /// Rebuild the track for `game`. After a step, the rows slide one row toward the camera over `duration`.
    func show(_ game: LaneRunner, passed: [LaneRunner.Obstacle]?, action: LaneRunner.Action?, duration: Double) {
        world.removeAllActions()
        world.childNodes.forEach { $0.removeFromParentNode() }
        var rows = Array(game.rows.prefix(Self.drawnRows).enumerated().map { (index: $0.offset + 1, row: $0.element) })
        if let passed { rows.insert((0, passed), at: 0) }
        for (index, row) in rows {
            let z = -CGFloat(index) * Self.rowLength
            addScenery(z: z, seed: game.distance + index)
            for (lane, obstacle) in row.enumerated() {
                addObstacle(obstacle, x: Self.x(lane), z: z, variant: (game.distance + index + lane) % 2)
            }
        }
        let moveX = SCNAction.move(
            to: SCNVector3(Self.x(game.lane), 0, 0), duration: min(duration * 0.5, 0.2))
        moveX.timingMode = .easeOut
        player.removeAllActions()
        player.scale = SCNVector3(1.15, 1.15, 1.15)
        player.eulerAngles = SCNVector3(0, CGFloat.pi, 0)
        if game.isOver {
            // The crash row is still rows[0]: stop the runner against it.
            world.position.z = 0
            let fall = SCNAction.rotateTo(x: -.pi / 2, y: .pi, z: 0, duration: 0.25)
            player.runAction(.group([moveX, fall]))
            return
        }
        player.runAction(.group([moveX, Self.pose(action, duration: duration)]))
        guard passed != nil else {
            world.position.z = 0
            return
        }
        world.position.z = -Self.rowLength
        world.runAction(.move(to: SCNVector3(0, 0, 0), duration: duration))
    }

    private static func x(_ lane: Int) -> CGFloat { CGFloat(lane - 1) * laneWidth }

    private static func pose(_ action: LaneRunner.Action?, duration: Double) -> SCNAction {
        switch action {
        case .jump:
            let up = SCNAction.moveBy(x: 0, y: 1.2, z: 0, duration: duration * 0.45)
            up.timingMode = .easeOut
            let down = SCNAction.moveBy(x: 0, y: -1.2, z: 0, duration: duration * 0.45)
            down.timingMode = .easeIn
            return .sequence([up, down])
        case .slide:
            return .sequence([
                .scale(to: 0.55, duration: duration * 0.15), .wait(duration: duration * 0.6),
                .scale(to: 1.15, duration: duration * 0.2),
            ])
        default:
            let bob = SCNAction.moveBy(x: 0, y: 0.08, z: 0, duration: duration * 0.25)
            return .sequence([bob, bob.reversed()])
        }
    }

    private func clone(_ name: String) -> SCNNode {
        templates[name]?.clone() ?? SCNNode(geometry: SCNBox(width: 1, height: 1, length: 1, chamferRadius: 0.1))
    }

    private func addObstacle(_ obstacle: LaneRunner.Obstacle, x: CGFloat, z: CGFloat, variant: Int) {
        for offset in [-1.0, 0.0, 1.0] as [CGFloat] {
            let track = clone("track")
            track.scale = SCNVector3(1.4, 1, 1.07)
            track.position = SCNVector3(x, 0, z + offset * Self.rowLength / 3)
            world.addChildNode(track)
        }
        switch obstacle {
        case .open:
            return
        case .coin:
            let coin = clone("coin-gold")
            coin.scale = SCNVector3(1.4, 1.4, 1.4)
            coin.position = SCNVector3(x, 0.45, z)
            coin.runAction(.repeatForever(.rotateBy(x: 0, y: .pi * 2, z: 0, duration: 1.2)))
            world.addChildNode(coin)
        case .train:
            let train = clone(variant == 0 ? "train-electric-subway-a" : "train-electric-subway-b")
            train.scale = SCNVector3(1.15, 1.15, 1.15)
            train.position = SCNVector3(x, 0.1, z)
            world.addChildNode(train)
        case .low:
            let fence = clone("fence-low-straight")
            fence.scale = SCNVector3(1.5, 1.8, 2)
            fence.position = SCNVector3(x, 0.1, z)
            world.addChildNode(fence)
        case .high:
            world.addChildNode(Self.overheadBar(x: x, z: z))
        }
    }

    /// Two posts and a striped beam with open space below: something to slide under, not a wall.
    private static func overheadBar(x: CGFloat, z: CGFloat) -> SCNNode {
        let bar = SCNNode()
        let wood = NSColor(red: 0.62, green: 0.38, blue: 0.24, alpha: 1)
        for side: CGFloat in [-1, 1] {
            let post = SCNNode(geometry: SCNCylinder(radius: 0.07, height: 1.15))
            post.geometry?.firstMaterial?.diffuse.contents = wood
            post.position = SCNVector3(side * laneWidth * 0.44, 0.575, 0)
            bar.addChildNode(post)
        }
        let stripes = 6
        let stripeWidth = laneWidth * 0.95 / CGFloat(stripes)
        for index in 0..<stripes {
            let stripe = SCNNode(geometry: SCNBox(width: stripeWidth, height: 0.22, length: 0.14, chamferRadius: 0.02))
            stripe.geometry?.firstMaterial?.diffuse.contents = index.isMultiple(of: 2) ? NSColor.systemRed : .white
            stripe.position = SCNVector3(
                -laneWidth * 0.475 + stripeWidth * (CGFloat(index) + 0.5), 0.95, 0)
            bar.addChildNode(stripe)
        }
        bar.position = SCNVector3(x, 0.1, z)
        return bar
    }

    private func addScenery(z: CGFloat, seed: Int) {
        for side: CGFloat in [-1, 1] {
            let tree = clone(seed % 3 == 0 ? "tree" : "tree-pine")
            tree.scale = SCNVector3(2.2, 2.2, 2.2)
            tree.position = SCNVector3(side * (Self.laneWidth * 1.5 + 1.3 + CGFloat(seed % 2) * 0.6), 0, z)
            world.addChildNode(tree)
        }
    }

    private func buildStage() {
        scene.background.contents = NSColor(red: 0.53, green: 0.78, blue: 0.95, alpha: 1)
        scene.fogStartDistance = 14
        scene.fogEndDistance = 30
        scene.fogColor = NSColor(red: 0.53, green: 0.78, blue: 0.95, alpha: 1)

        let ground = SCNNode(geometry: SCNPlane(width: 60, height: 120))
        ground.geometry?.firstMaterial?.diffuse.contents = NSColor(red: 0.47, green: 0.66, blue: 0.36, alpha: 1)
        ground.eulerAngles.x = -.pi / 2
        ground.position = SCNVector3(0, -0.01, -40)
        scene.rootNode.addChildNode(ground)
        let ballast = SCNNode(geometry: SCNPlane(width: Self.laneWidth * 3 + 0.6, height: 120))
        ballast.geometry?.firstMaterial?.diffuse.contents = NSColor(red: 0.55, green: 0.52, blue: 0.48, alpha: 1)
        ballast.eulerAngles.x = -.pi / 2
        ballast.position = SCNVector3(0, 0, -40)
        scene.rootNode.addChildNode(ballast)

        scene.rootNode.addChildNode(world)
        player.addChildNode(clone("character-oobi"))
        scene.rootNode.addChildNode(player)

        let camera = SCNNode()
        camera.camera = SCNCamera()
        camera.camera?.fieldOfView = 55
        camera.camera?.zFar = 60
        camera.position = SCNVector3(0, 3.4, 5.2)
        camera.look(at: SCNVector3(0, 0.6, -7))
        scene.rootNode.addChildNode(camera)

        let sun = SCNNode()
        sun.light = SCNLight()
        sun.light?.type = .directional
        sun.light?.castsShadow = true
        sun.light?.shadowColor = NSColor.black.withAlphaComponent(0.35)
        sun.eulerAngles = SCNVector3(-1.0, 0.5, 0)
        scene.rootNode.addChildNode(sun)
        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 550
        scene.rootNode.addChildNode(ambient)
    }
}

struct RunnerSceneView: NSViewRepresentable {
    let runner: RunnerScene

    func makeNSView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = runner.scene
        view.antialiasingMode = .multisampling4X
        view.rendersContinuously = true
        view.allowsCameraControl = false
        return view
    }

    func updateNSView(_ view: SCNView, context: Context) {}
}
