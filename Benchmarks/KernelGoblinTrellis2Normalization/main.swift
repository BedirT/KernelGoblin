import Foundation
import KernelGoblinTrellis2
import Metal

@main
enum NormalizationBenchmark {
    static func main() throws {
        let warmup = try integerArgument("--warmup", default: 5)
        let iterations = try integerArgument("--iterations", default: 20)
        guard warmup >= 1, iterations >= 3 else { throw BenchmarkError.invalidArguments }
        let context = try MetalContext()
        let kernel = try NormalizationKernel(context: context)
        print("device=\(context.device.name)")
        print("os=\(ProcessInfo.processInfo.operatingSystemVersionString)")
        print("warmup=\(warmup) iterations=\(iterations)")
        print("timing=dispatch + synchronized completion wait; allocations and validation excluded")
        try benchmarkLayerNorm(
            context: context, kernel: kernel, warmup: warmup, iterations: iterations
        )
        try benchmarkRMSNorm(
            context: context, kernel: kernel, warmup: warmup, iterations: iterations
        )
    }

    private static func benchmarkLayerNorm(
        context: MetalContext, kernel: NormalizationKernel,
        warmup: Int, iterations: Int
    ) throws {
        let rows = 4096, channels = 1536
        var inputValues = fixtureValues(count: rows * channels)
        var checkpointValues = (0..<channels).map {
            bf16(0.8 + Float($0 % 17) * 0.01)
        }
        checkpointValues += (0..<channels).map {
            bf16(-0.05 + Float($0 % 11) * 0.005)
        }
        let buffers = try makeBuffers(
            context: context, input: &inputValues, checkpoint: &checkpointValues
        )
        let biasOffset = channels * MemoryLayout<UInt16>.stride
        func invoke(_ implementation: NormalizationImplementation, _ output: MTLBuffer) throws {
            try kernel.layerNormF32(
                input: buffers.input, checkpoint: buffers.checkpoint,
                rows: rows, channels: channels, weightOffset: 0,
                biasOffset: biasOffset, output: output,
                implementation: implementation
            )
        }
        try benchmark(
            name: "layer_norm", elements: inputValues.count,
            scalar: { try invoke(.scalar, buffers.scalar) },
            simdgroup: { try invoke(.simdgroup, buffers.simdgroup) },
            scalarOutput: buffers.scalar, simdgroupOutput: buffers.simdgroup,
            warmup: warmup, iterations: iterations
        )
    }

    private static func benchmarkRMSNorm(
        context: MetalContext, kernel: NormalizationKernel,
        warmup: Int, iterations: Int
    ) throws {
        let rows = 4096, heads = 12, dimensions = 128
        var inputValues = fixtureValues(count: rows * heads * dimensions)
        var checkpointValues = (0..<(heads * dimensions)).map {
            bf16(0.9 + Float($0 % 13) * 0.008)
        }
        let buffers = try makeBuffers(
            context: context, input: &inputValues, checkpoint: &checkpointValues
        )
        func invoke(_ implementation: NormalizationImplementation, _ output: MTLBuffer) throws {
            try kernel.multiheadRMSNormF32(
                input: buffers.input, checkpoint: buffers.checkpoint, gammaOffset: 0,
                rows: rows, heads: heads, dimensions: dimensions, output: output,
                implementation: implementation
            )
        }
        try benchmark(
            name: "multihead_rms_norm", elements: inputValues.count,
            scalar: { try invoke(.scalar, buffers.scalar) },
            simdgroup: { try invoke(.simdgroup, buffers.simdgroup) },
            scalarOutput: buffers.scalar, simdgroupOutput: buffers.simdgroup,
            warmup: warmup, iterations: iterations
        )
    }

    private static func benchmark(
        name: String, elements: Int,
        scalar: @escaping () throws -> Void,
        simdgroup: @escaping () throws -> Void,
        scalarOutput: MTLBuffer, simdgroupOutput: MTLBuffer,
        warmup: Int, iterations: Int
    ) throws {
        try scalar()
        try simdgroup()
        let reference = scalarOutput.contents().assumingMemoryBound(to: Float.self)
        let candidate = simdgroupOutput.contents().assumingMemoryBound(to: Float.self)
        var maximumError: Float = 0
        var squaredError = 0.0
        var squaredReference = 0.0
        for index in 0..<elements {
            let error = candidate[index] - reference[index]
            maximumError = max(maximumError, abs(error))
            squaredError += Double(error) * Double(error)
            squaredReference += Double(reference[index]) * Double(reference[index])
        }
        let normalizedRMS = sqrt(
            squaredError / max(squaredReference, Double.leastNonzeroMagnitude)
        )
        guard maximumError <= 2e-5, normalizedRMS <= 1e-5 else {
            throw BenchmarkError.conformanceFailed(name, maximumError, normalizedRMS)
        }
        for _ in 0..<warmup { try scalar(); try simdgroup() }
        var scalarTimes: [Double] = []
        var simdTimes: [Double] = []
        for iteration in 0..<iterations {
            let order: [(String, () throws -> Void)] = iteration.isMultiple(of: 2)
                ? [("scalar", scalar), ("simdgroup", simdgroup)]
                : [("simdgroup", simdgroup), ("scalar", scalar)]
            for (label, invocation) in order {
                let start = DispatchTime.now().uptimeNanoseconds
                try invocation()
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
                if label == "scalar" { scalarTimes.append(elapsed) }
                else { simdTimes.append(elapsed) }
            }
        }
        print("workload=\(name) elements=\(elements) max_abs_diff=\(maximumError) normalized_rms=\(normalizedRMS)")
        print(summary(name: "scalar", values: scalarTimes))
        print(summary(name: "simdgroup", values: simdTimes))
        print(String(format: "simdgroup_speedup=%.3fx", median(scalarTimes) / median(simdTimes)))
    }

    private static func makeBuffers(
        context: MetalContext, input: inout [Float], checkpoint: inout [UInt16]
    ) throws -> (input: MTLBuffer, checkpoint: MTLBuffer, scalar: MTLBuffer, simdgroup: MTLBuffer) {
        guard let inputBuffer = context.device.makeBuffer(
            bytes: &input, length: input.count * 4, options: .storageModeShared
        ), let checkpointBuffer = context.device.makeBuffer(
            bytes: &checkpoint, length: checkpoint.count * 2, options: .storageModeShared
        ), let scalar = context.device.makeBuffer(
            length: input.count * 4, options: .storageModeShared
        ), let simdgroup = context.device.makeBuffer(
            length: input.count * 4, options: .storageModeShared
        ) else { throw BenchmarkError.allocationFailed }
        return (inputBuffer, checkpointBuffer, scalar, simdgroup)
    }

    private static func fixtureValues(count: Int) -> [Float] {
        (0..<count).map {
            Float(sin(Double($0) * 0.013) * 0.7 + cos(Double($0) * 0.007) * 0.2)
        }
    }

    private static func bf16(_ value: Float) -> UInt16 {
        UInt16(truncatingIfNeeded: value.bitPattern >> 16)
    }

    private static func integerArgument(_ name: String, default defaultValue: Int) throws -> Int {
        guard let index = CommandLine.arguments.firstIndex(of: name) else { return defaultValue }
        guard index + 1 < CommandLine.arguments.count,
              let value = Int(CommandLine.arguments[index + 1]) else {
            throw BenchmarkError.invalidArguments
        }
        return value
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2)
            ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
    }

    private static func summary(name: String, values: [Double]) -> String {
        let sorted = values.sorted()
        let p95 = sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * 0.95))]
        return String(format: "implementation=%@ median_ms=%.6f p95_ms=%.6f", name, median(values), p95)
    }
}

private enum BenchmarkError: Error {
    case invalidArguments
    case allocationFailed
    case conformanceFailed(String, Float, Double)
}
