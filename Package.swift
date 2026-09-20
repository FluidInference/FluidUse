// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "CuaFormsDemo",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(name: "FluidAudio", path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "CuaFormsDemo",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio")
            ],
            resources: [.copy("Resources")]
        )
    ]
)
