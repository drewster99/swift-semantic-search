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
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "2.29.2")
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
