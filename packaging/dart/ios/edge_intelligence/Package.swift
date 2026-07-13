// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "edge_intelligence",
    platforms: [
        .iOS("13.0")
    ],
    products: [
        .library(name: "edge-intelligence", targets: ["edge_intelligence"])
    ],
    dependencies: [
        .package(name: "FlutterFramework", path: "../FlutterFramework")
    ],
    targets: [
        .target(
            name: "edge_intelligence",
            dependencies: [
                "el_ffi",
                .product(name: "FlutterFramework", package: "FlutterFramework")
            ]
        ),
        .binaryTarget(
            name: "el_ffi",
            path: "../Frameworks/el_ffi.xcframework"
        )
    ]
)
