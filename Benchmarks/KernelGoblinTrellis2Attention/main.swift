import Foundation
import KernelGoblinTrellis2
import Metal

private struct Workload {
    let name: String
    let queryCount: Int
    let keyCount: Int
    let heads: Int
    let dimensions: Int
}

private let workloads = [
    Workload(name: "self_q4096_k4096", queryCount: 4096, keyCount: 4096, heads: 12, dimensions: 128),
    Workload(name: "cross_q4096_k1029", queryCount: 4096, keyCount: 1029, heads: 12, dimensions: 128),
    Workload(name: "cross_q1024_k1029", queryCount: 1024, keyCount: 1029, heads: 12, dimensions: 128),
]

@main
enum AttentionBenchmark {
    static func main() throws {
        guard #available(macOS 15.0, *) else { throw BenchmarkError.unsupportedOS }
        let warmup = try integerArgument("--warmup", default: 3)
        let iterations = try integerArgument("--iterations", default: 10)
        guard warmup >= 1, iterations >= 3 else { throw BenchmarkError.invalidArguments }

        let context = try MetalContext()
        let kernel = try AttentionKernel(context: context)
        print("device=\(context.device.name)")
        print("os=\(ProcessInfo.processInfo.operatingSystemVersionString)")
        print("warmup=\(warmup) iterations=\(iterations)")
        print("layout=input_output=[batch,query_or_key,head,dimension] contiguous; mpsgraph_transpose=[batch,head,query_or_key,dimension]")
        print("timing=dispatch + synchronized completion wait; graph construction, first compile, allocations, and validation excluded")
        for workload in workloads {
            try run(workload, context: context, kernel: kernel, warmup: warmup, iterations: iterations)
        }
    }

    @available(macOS 15.0, *)
    private static func run(
        _ workload: Workload,
        context: MetalContext,
        kernel: AttentionKernel,
        warmup: Int,
        iterations: Int
    ) throws {
        var queries = fixtureValues(
            count: workload.queryCount * workload.heads * workload.dimensions,
            multiplier: 1_664_525, increment: 1_013_904_223, scale: 1
        )
        var keys = fixtureValues(
            count: workload.keyCount * workload.heads * workload.dimensions,
            multiplier: 22_695_477, increment: 1, scale: 1
        )
        var values = fixtureValues(
            count: workload.keyCount * workload.heads * workload.dimensions,
            multiplier: 1_103_515_245, increment: 12_345, scale: 0.5
        )
        let outputCount = workload.queryCount * workload.heads * workload.dimensions
        guard let queryBuffer = context.device.makeBuffer(
            bytes: &queries, length: queries.count * 4, options: .storageModeShared
        ), let keyBuffer = context.device.makeBuffer(
            bytes: &keys, length: keys.count * 4, options: .storageModeShared
        ), let valueBuffer = context.device.makeBuffer(
            bytes: &values, length: values.count * 4, options: .storageModeShared
        ), let metalOutput = context.device.makeBuffer(
            length: outputCount * 4, options: .storageModeShared
        ), let mpsOutput = context.device.makeBuffer(
            length: outputCount * 4, options: .storageModeShared
        ) else {
            throw BenchmarkError.allocationFailed
        }
        func runMetal() throws {
            try kernel.fusedF32(
                queries: queryBuffer, keys: keyBuffer, values: valueBuffer,
                queryCount: workload.queryCount, keyCount: workload.keyCount,
                heads: workload.heads, dimensions: workload.dimensions,
                output: metalOutput, implementation: .metal
            )
        }
        func runMPSGraph() throws {
            try kernel.fusedF32(
                queries: queryBuffer, keys: keyBuffer, values: valueBuffer,
                queryCount: workload.queryCount, keyCount: workload.keyCount,
                heads: workload.heads, dimensions: workload.dimensions,
                output: mpsOutput, implementation: .mpsGraph
            )
        }

        // These first calls force MPSGraph compilation before validation or timing.
        try runMetal()
        try runMPSGraph()
        let metal = metalOutput.contents().assumingMemoryBound(to: Float.self)
        let mps = mpsOutput.contents().assumingMemoryBound(to: Float.self)
        var squaredError = 0.0
        var squaredReference = 0.0
        var maximumError: Float = 0
        for index in 0..<outputCount {
            let error = mps[index] - metal[index]
            maximumError = max(maximumError, abs(error))
            squaredError += Double(error) * Double(error)
            squaredReference += Double(metal[index]) * Double(metal[index])
        }
        let normalizedRMS = sqrt(
            squaredError / max(squaredReference, Double.leastNonzeroMagnitude)
        )
        guard normalizedRMS <= 0.005 else {
            throw BenchmarkError.conformanceFailed(workload.name, normalizedRMS)
        }
        let sampledMaximumError = sampledCPUError(
            workload: workload, queries: queries, keys: keys, values: values,
            metal: metal, mps: mps
        )
        guard sampledMaximumError <= 2e-5 else {
            throw BenchmarkError.cpuReferenceFailed(workload.name, sampledMaximumError)
        }

        for _ in 0..<warmup {
            try runMetal()
            try runMPSGraph()
        }
        var metalTimes: [Double] = []
        var mpsTimes: [Double] = []
        for iteration in 0..<iterations {
            let order: [(String, () throws -> Void)] = iteration.isMultiple(of: 2)
                ? [("metal", runMetal), ("mpsgraph", runMPSGraph)]
                : [("mpsgraph", runMPSGraph), ("metal", runMetal)]
            for (name, invocation) in order {
                let start = DispatchTime.now().uptimeNanoseconds
                try invocation()
                let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
                if name == "metal" { metalTimes.append(milliseconds) }
                else { mpsTimes.append(milliseconds) }
            }
        }
        print(
            "workload=\(workload.name) b=1 h=\(workload.heads) q=\(workload.queryCount) " +
            "k=\(workload.keyCount) d=\(workload.dimensions) max_abs_diff=\(maximumError) " +
            "normalized_rms=\(normalizedRMS) sampled_cpu_max_abs_diff=\(sampledMaximumError)"
        )
        print(summary(name: "metal", values: metalTimes))
        print(summary(name: "mpsgraph", values: mpsTimes))
        print(String(format: "mpsgraph_speedup=%.3fx", median(metalTimes) / median(mpsTimes)))
    }

    private static func sampledCPUError(
        workload: Workload,
        queries: [Float],
        keys: [Float],
        values: [Float],
        metal: UnsafePointer<Float>,
        mps: UnsafePointer<Float>
    ) -> Float {
        var maximumError: Float = 0
        let sampledQueries = [0, workload.queryCount / 2, workload.queryCount - 1]
        let sampledHeads = [0, workload.heads / 2, workload.heads - 1]
        let sampledDimensions = [0, workload.dimensions / 2, workload.dimensions - 1]
        let scale = 1 / sqrt(Float(workload.dimensions))
        for query in sampledQueries {
            for head in sampledHeads {
                let queryBase = (query * workload.heads + head) * workload.dimensions
                var scores = [Float](repeating: 0, count: workload.keyCount)
                for key in 0..<workload.keyCount {
                    let keyBase = (key * workload.heads + head) * workload.dimensions
                    var score: Float = 0
                    for dimension in 0..<workload.dimensions {
                        score.addProduct(
                            queries[queryBase + dimension], keys[keyBase + dimension]
                        )
                    }
                    scores[key] = score * scale
                }
                let maximum = scores.max()!
                let denominator = scores.reduce(Float(0)) { $0 + exp($1 - maximum) }
                for dimension in sampledDimensions {
                    var expected: Float = 0
                    for key in 0..<workload.keyCount {
                        let keyBase = (key * workload.heads + head) * workload.dimensions
                        expected += exp(scores[key] - maximum) / denominator
                            * values[keyBase + dimension]
                    }
                    let index = queryBase + dimension
                    maximumError = max(
                        maximumError, abs(metal[index] - expected), abs(mps[index] - expected)
                    )
                }
            }
        }
        return maximumError
    }

    private static func fixtureValues(
        count: Int, multiplier: UInt32, increment: UInt32, scale: Float
    ) -> [Float] {
        (0..<count).map { index in
            let bits = UInt32(truncatingIfNeeded: index) &* multiplier &+ increment
            let value = Float(Int32(bitPattern: bits)) / Float(Int32.max) * scale
            return Float(bitPattern: value.bitPattern & 0xffff_0000)
        }
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
}

private enum BenchmarkError: Error {
    case unsupportedOS
    case invalidArguments
    case allocationFailed
    case conformanceFailed(String, Double)
    case cpuReferenceFailed(String, Float)
}
