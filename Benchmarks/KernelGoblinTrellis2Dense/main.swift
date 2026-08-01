import Foundation
import KernelGoblinTrellis2
import Metal

private struct Workload {
    let name: String
    let rows: Int
    let inputChannels: Int
    let outputChannels: Int
}

private let workloads = [
    Workload(name: "ss_input", rows: 4096, inputChannels: 8, outputChannels: 1536),
    Workload(name: "cross_kv", rows: 1029, inputChannels: 1024, outputChannels: 3072),
    Workload(name: "conditioning", rows: 1, inputChannels: 1536, outputChannels: 9216),
    Workload(name: "ss_mlp_up", rows: 4096, inputChannels: 1536, outputChannels: 8192),
]

@main
enum DenseBenchmark {
    static func main() throws {
        let warmup = try integerArgument("--warmup", default: 5)
        let iterations = try integerArgument("--iterations", default: 20)
        guard warmup >= 1, iterations >= 3 else {
            throw BenchmarkError.invalidArguments
        }
        let context = try MetalContext()
        let kernel = try DenseKernel(context: context)
        print("device=\(context.device.name)")
        print("os=\(ProcessInfo.processInfo.operatingSystemVersionString)")
        print("metal_language=3.0 warmup=\(warmup) iterations=\(iterations)")
        print("timing=dispatch + completion wait; graph construction, allocations, and validation excluded")
        for workload in workloads {
            try run(
                workload, context: context, kernel: kernel,
                warmup: warmup, iterations: iterations
            )
        }
    }

    private static func run(
        _ workload: Workload,
        context: MetalContext,
        kernel: DenseKernel,
        warmup: Int,
        iterations: Int
    ) throws {
        var state: UInt32 = 0x9e3779b9
        func randomFloat() -> Float {
            state = state &* 1_664_525 &+ 1_013_904_223
            return Float(Int32(bitPattern: state)) / Float(Int32.max) * 0.125
        }
        var inputs = (0..<(workload.rows * workload.inputChannels)).map { _ in
            randomFloat()
        }
        var checkpoint = (0..<(workload.outputChannels * workload.inputChannels
            + workload.outputChannels)).map { _ in bf16(randomFloat()) }
        guard let inputBuffer = context.device.makeBuffer(
            bytes: &inputs,
            length: inputs.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ), let checkpointBuffer = context.device.makeBuffer(
            bytes: &checkpoint,
            length: checkpoint.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared
        ), let tiledOutput = context.device.makeBuffer(
            length: workload.rows * workload.outputChannels * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ), let simdOutput = context.device.makeBuffer(
            length: workload.rows * workload.outputChannels * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ), let mpsOutput = context.device.makeBuffer(
            length: workload.rows * workload.outputChannels * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw BenchmarkError.allocationFailed
        }
        let biasOffset = workload.outputChannels * workload.inputChannels
            * MemoryLayout<UInt16>.stride
        func invoke(_ implementation: BF16LinearImplementation, _ output: MTLBuffer) throws {
            try kernel.linearBF16WeightsF32Output(
                input: inputBuffer, checkpoint: checkpointBuffer, weightOffset: 0,
                biasOffset: biasOffset, rows: workload.rows,
                inputChannels: workload.inputChannels,
                outputChannels: workload.outputChannels, output: output,
                implementation: implementation
            )
        }
        func invokeMPSGraph() throws {
            try kernel.linearBF16WeightsF32Output(
                input: inputBuffer,
                checkpoint: checkpointBuffer,
                weightOffset: 0,
                biasOffset: workload.outputChannels * workload.inputChannels
                    * MemoryLayout<UInt16>.stride,
                rows: workload.rows,
                inputChannels: workload.inputChannels,
                outputChannels: workload.outputChannels,
                output: mpsOutput,
                implementation: .mpsGraphModelPrecision
            )
        }
        try invoke(.tiled, tiledOutput)
        try invoke(.simdgroupMatrix, simdOutput)
        try invokeMPSGraph()
        let outputCount = workload.rows * workload.outputChannels
        let tiled = tiledOutput.contents().assumingMemoryBound(to: Float.self)
        let simd = simdOutput.contents().assumingMemoryBound(to: Float.self)
        let mps = mpsOutput.contents().assumingMemoryBound(to: Float.self)
        var simdSquaredError = 0.0
        var mpsSquaredError = 0.0
        var squaredReference = 0.0
        var simdMaximumError: Float = 0
        var mpsMaximumError: Float = 0
        for index in 0..<outputCount {
            let simdError = simd[index] - tiled[index]
            let mpsError = mps[index] - tiled[index]
            simdMaximumError = max(simdMaximumError, abs(simdError))
            mpsMaximumError = max(mpsMaximumError, abs(mpsError))
            simdSquaredError += Double(simdError) * Double(simdError)
            mpsSquaredError += Double(mpsError) * Double(mpsError)
            squaredReference += Double(tiled[index]) * Double(tiled[index])
        }
        let denominator = max(squaredReference, Double.leastNonzeroMagnitude)
        let simdNormalizedRMS = sqrt(simdSquaredError / denominator)
        let mpsNormalizedRMS = sqrt(mpsSquaredError / denominator)
        guard simdNormalizedRMS <= 5e-4 else {
            throw BenchmarkError.conformanceFailed(workload.name, "simdgroup", simdNormalizedRMS)
        }
        // The production MPSGraph path executes the upstream BF16 matrix
        // boundary, so validate it with dtype-derived aggregate and absolute
        // gates rather than the F32 accumulation bound used by Metal oracles.
        guard mpsNormalizedRMS <= 2e-3, mpsMaximumError <= 2e-3 else {
            throw BenchmarkError.conformanceFailed(workload.name, "mpsgraph", mpsNormalizedRMS)
        }
        for row in [0, workload.rows / 2, workload.rows - 1] {
            for channel in [0, workload.outputChannels / 2, workload.outputChannels - 1] {
                var expected = f32(checkpoint[
                    workload.outputChannels * workload.inputChannels + channel
                ])
                var absoluteProducts: Float = abs(expected)
                for inputChannel in 0..<workload.inputChannels {
                    let product = inputs[row * workload.inputChannels + inputChannel]
                        * f32(checkpoint[channel * workload.inputChannels + inputChannel])
                    absoluteProducts += abs(product)
                    expected += product
                }
                let gamma = Float(workload.inputChannels + 1) * Float.ulpOfOne
                let errorBound = gamma / (1 - gamma) * absoluteProducts + 2e-6
                let index = row * workload.outputChannels + channel
                guard abs(tiled[index] - expected) <= errorBound,
                      abs(simd[index] - expected) <= errorBound else {
                    throw BenchmarkError.cpuReferenceFailed(workload.name, row, channel)
                }
            }
        }
        for _ in 0..<warmup {
            try invoke(.tiled, tiledOutput)
            try invoke(.simdgroupMatrix, simdOutput)
            try invokeMPSGraph()
        }
        var tiledTimes: [Double] = []
        var simdTimes: [Double] = []
        var mpsTimes: [Double] = []
        tiledTimes.reserveCapacity(iterations)
        simdTimes.reserveCapacity(iterations)
        mpsTimes.reserveCapacity(iterations)
        for iteration in 0..<iterations {
            let implementations: [(String, () throws -> Void)] = [
                ("tiled", { try invoke(.tiled, tiledOutput) }),
                ("simdgroup", { try invoke(.simdgroupMatrix, simdOutput) }),
                ("mpsgraph_bf16", { try invokeMPSGraph() }),
            ]
            let offset = iteration % implementations.count
            let order = Array(implementations[offset...]) + Array(implementations[..<offset])
            for (name, invokeTimed) in order {
                let start = DispatchTime.now().uptimeNanoseconds
                try invokeTimed()
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
                switch name {
                case "tiled": tiledTimes.append(elapsed)
                case "simdgroup": simdTimes.append(elapsed)
                default: mpsTimes.append(elapsed)
                }
            }
        }
        print(
            "workload=\(workload.name) m=\(workload.rows) n=\(workload.outputChannels) " +
            "k=\(workload.inputChannels) simd_max_abs_diff=\(simdMaximumError) " +
            "simd_normalized_rms=\(simdNormalizedRMS) mps_max_abs_diff=\(mpsMaximumError) " +
            "mps_normalized_rms=\(mpsNormalizedRMS)"
        )
        print(summary(name: "tiled", values: tiledTimes))
        print(summary(name: "simdgroup", values: simdTimes))
        print(summary(name: "mpsgraph_bf16", values: mpsTimes))
        print(String(
            format: "simdgroup_speedup=%.3fx mpsgraph_bf16_speedup=%.3fx",
            median(tiledTimes) / median(simdTimes), median(tiledTimes) / median(mpsTimes)
        ))
    }

    private static func summary(name: String, values: [Double]) -> String {
        let sorted = values.sorted()
        let p95 = sorted[min(sorted.count - 1, Int(ceil(Double(sorted.count) * 0.95)) - 1)]
        return "implementation=\(name) median_ms=\(median(sorted)) p95_ms=\(p95)"
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2)
            ? (sorted[middle - 1] + sorted[middle]) / 2
            : sorted[middle]
    }

    private static func integerArgument(_ name: String, default defaultValue: Int) throws -> Int {
        guard let index = CommandLine.arguments.firstIndex(of: name) else { return defaultValue }
        guard index + 1 < CommandLine.arguments.count,
              let value = Int(CommandLine.arguments[index + 1]) else {
            throw BenchmarkError.invalidArguments
        }
        return value
    }

    private static func bf16(_ value: Float) -> UInt16 {
        UInt16(truncatingIfNeeded: value.bitPattern >> 16)
    }

    private static func f32(_ value: UInt16) -> Float {
        Float(bitPattern: UInt32(value) << 16)
    }
}

private enum BenchmarkError: Error {
    case invalidArguments
    case allocationFailed
    case conformanceFailed(String, String, Double)
    case cpuReferenceFailed(String, Int, Int)
}
