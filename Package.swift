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
    ]
)
