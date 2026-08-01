// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "KernelGoblin",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "KernelGoblinTrellis2", targets: ["KernelGoblinTrellis2"]),
        .executable(name: "kg-trellis2", targets: ["KernelGoblinTrellis2CLI"]),
        .executable(
            name: "kg-trellis2-dense-bench",
            targets: ["KernelGoblinTrellis2DenseBenchmark"]
        ),
        .executable(
            name: "kg-trellis2-attention-bench",
            targets: ["KernelGoblinTrellis2AttentionBenchmark"]
        ),
        .executable(
            name: "kg-trellis2-pbr-bake-bench",
            targets: ["KernelGoblinTrellis2PBRBenchmark"]
        ),
    ],
    targets: [
        .target(
            name: "KernelGoblinTrellis2",
            path: "Sources/KernelGoblinTrellis2",
            resources: [.copy("Metal")]
        ),
        .executableTarget(
            name: "KernelGoblinTrellis2CLI",
            dependencies: ["KernelGoblinTrellis2"],
            path: "Sources/KernelGoblinTrellis2CLI"
        ),
        .executableTarget(
            name: "KernelGoblinTrellis2DenseBenchmark",
            dependencies: ["KernelGoblinTrellis2"],
            path: "Benchmarks/KernelGoblinTrellis2Dense"
        ),
        .executableTarget(
            name: "KernelGoblinTrellis2AttentionBenchmark",
            dependencies: ["KernelGoblinTrellis2"],
            path: "Benchmarks/KernelGoblinTrellis2Attention"
        ),
        .executableTarget(
            name: "KernelGoblinTrellis2PBRBenchmark",
            dependencies: ["KernelGoblinTrellis2"],
            path: "Benchmarks/KernelGoblinTrellis2PBR"
        ),
        .testTarget(
            name: "KernelGoblinTrellis2Tests",
            dependencies: ["KernelGoblinTrellis2"],
            path: "Tests/KernelGoblinTrellis2Tests",
            resources: [.copy("Fixtures")]
        ),
    ]
)
