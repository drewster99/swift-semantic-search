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
        // its main branch (loadModelContainer now takes Downloader/TokenizerLoader
        // instead of HubApi). Stay on 2.29.x until we explicitly migrate.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "2.29.2"))
    ],
    targets: [
        .target(
            name: "SemanticSearch",
            dependencies: [
                .product(name: "MLXEmbedders", package: "mlx-swift-lm")
            ],
            path: "Sources/SemanticSearch"
        ),
        .testTarget(
            name: "SemanticSearchTests",
            dependencies: ["SemanticSearch"],
            path: "Tests/SemanticSearchTests"
        )
    ]
)
