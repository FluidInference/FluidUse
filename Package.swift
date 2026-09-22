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
        .executableTarget(
            name: "FluidUseLaya",
            dependencies: ["FluidUse", "LayaTetris", .product(name: "FluidAudio", package: "FluidAudio")]
        ),
        .executableTarget(
            name: "LayaTetrisDemo",
            dependencies: ["FluidUse", "LayaTetris"],
            exclude: ["README.md"]
        ),
        .testTarget(name: "FluidUseTests", dependencies: ["FluidUse", "LayaTetris"]),
        .testTarget(name: "LayaTetrisTests", dependencies: ["LayaTetris"]),
        .testTarget(name: "LayaTetrisDemoTests", dependencies: ["LayaTetrisDemo", "LayaTetris"]),
    ]
)
