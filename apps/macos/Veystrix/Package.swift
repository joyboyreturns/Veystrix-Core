// swift-tools-version: 6.0
import PackageDescription

// Veystrix for macOS: a native SwiftUI front end for the Veystrix engine.
// VeystrixKit holds all logic (no UI) and is unit-tested without any model or GPU.
let package = Package(
    name: "Veystrix",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "Veystrix", targets: ["Veystrix"]),
        .library(name: "VeystrixKit", targets: ["VeystrixKit"]),
    ],
    dependencies: [
        // Sparkle 2 (MIT-style license, see LICENSES/Sparkle-2.10.1.txt). Pinned exactly.
        .package(url: "https://github.com/sparkle-project/Sparkle", exact: "2.10.1"),
    ],
    targets: [
        .target(name: "VeystrixKit", path: "Sources/VeystrixKit"),
        .executableTarget(
            name: "Veystrix",
            dependencies: [
                "VeystrixKit",
                .product(name: "Sparkle", package: "Sparkle"),
            ],
            path: "Sources/Veystrix"
        ),
        .testTarget(name: "VeystrixKitTests", dependencies: ["VeystrixKit"], path: "Tests/VeystrixKitTests"),
    ],
    swiftLanguageModes: [.v5]
)
