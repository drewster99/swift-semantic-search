// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "SemanticSearch",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1)
    ],
    products: [
        .library(name: "SemanticSearch", targets: ["SemanticSearch"])
    ],
    dependencies: [
        // Pinned tightly: mlx-swift-lm has rewritten the MLXEmbedders public API on
        // its 3.x line (loadModelContainer now takes Downloader/TokenizerLoader
        // instead of HubApi). Stay on the patched 2.x fork until we explicitly migrate.
        .package(url: "https://github.com/drewster99/mlx-swift-lm", .upToNextMinor(from: "2.31.4")),
        .package(url: "https://github.com/drewster99/mlx-swift", .upToNextMinor(from: "0.31.7"))
    ],
    targets: [
        .target(
            name: "SemanticSearch",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXLinalg", package: "mlx-swift")
            ],
            path: "Sources/SemanticSearch"
        ),
        .testTarget(
            name: "SemanticSearchTests",
            dependencies: ["SemanticSearch"],
            path: "Tests/SemanticSearchTests",
            resources: [
                .process("Fixtures")
            ]
        )
    ]
)
