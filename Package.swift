// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "FluidUse",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "FluidUse", targets: ["FluidUse"])
    ],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.15.8")
    ],
    targets: [
        .target(
            name: "FluidUse",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio")
            ],
            resources: [.copy("Resources")]
        ),
        .executableTarget(
            name: "FluidUseDemo",
            dependencies: ["FluidUse", .product(name: "FluidAudio", package: "FluidAudio")],
            resources: [.copy("Resources")]
        ),
        .target(name: "LayaTetris", dependencies: ["FluidUse"]),
        .target(name: "Game2048"),
        .executableTarget(
            name: "FluidUseLaya",
            dependencies: ["FluidUse", "Game2048", "LayaTetris", .product(name: "FluidAudio", package: "FluidAudio")]
        ),
        .executableTarget(
            name: "LayaTetrisDemo",
            dependencies: ["FluidUse", "LayaTetris"],
            exclude: ["README.md"]
        ),
        .executableTarget(
            name: "GLiClass2048Demo",
            dependencies: ["FluidUse", "Game2048"],
            exclude: ["README.md"]
        ),
        .executableTarget(
            name: "Decision2048BenchDemo",
            dependencies: ["FluidUse", "Game2048"],
            exclude: ["README.md"]
        ),
        .testTarget(
            name: "FluidUseTests", dependencies: ["FluidUse", "LayaTetris"],
            resources: [.copy("Fixtures")]
        ),
        .testTarget(name: "LayaTetrisTests", dependencies: ["LayaTetris"]),
        .testTarget(name: "LayaTetrisDemoTests", dependencies: ["LayaTetrisDemo", "LayaTetris"]),
        .testTarget(name: "Game2048Tests", dependencies: ["Game2048"]),
        .testTarget(name: "GLiClass2048DemoTests", dependencies: ["GLiClass2048Demo", "Game2048"]),
        .testTarget(name: "Decision2048BenchDemoTests", dependencies: ["Decision2048BenchDemo", "Game2048"]),
    ]
)
