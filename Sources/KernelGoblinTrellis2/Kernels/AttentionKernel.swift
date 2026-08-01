import Foundation
import Metal

public enum AttentionImplementation: Sendable {
    case automatic
    case metal
    case mpsGraph
}

public struct AttentionSegments: Equatable, Sendable {
    public let offsets: [Int]

    public init(offsets: [Int]) throws {
        guard offsets.count >= 2, offsets[0] == 0 else {
            throw NativeRuntimeError.invalidArgument(
                "attention segment offsets must start at zero and contain an end offset"
            )
        }
        for index in 1..<offsets.count where offsets[index] < offsets[index - 1] {
            throw NativeRuntimeError.invalidArgument(
                "attention segment offsets must be nondecreasing"
            )
        }
        guard offsets[offsets.count - 1] <= Int(Int32.max) else {
            throw NativeRuntimeError.invalidArgument(
                "attention segments exceed the upstream signed 32-bit layout"
            )
        }
        self.offsets = offsets
    }

    public var segmentCount: Int { offsets.count - 1 }
    public var totalCount: Int { offsets[offsets.count - 1] }
}

public final class AttentionKernel: @unchecked Sendable {
    private struct Parameters {
        var queryCount: UInt32
        var keyCount: UInt32
        var heads: UInt32
        var dimensions: UInt32
        var scale: Float
    }

    private let context: MetalContext
    private let pipeline: MTLComputePipelineState
    private let simdgroupPipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.library(named: "identity")
        guard let function = library.makeFunction(name: "kg_fused_attention_f32"),
              let simdgroupFunction = library.makeFunction(
                  name: "kg_simdgroup_attention_f32"
              ) else {
            throw NativeRuntimeError.invalidArgument("fused attention Metal function is missing")
        }
        self.pipeline = try context.device.makeComputePipelineState(function: function)
        self.simdgroupPipeline = try context.device.makeComputePipelineState(
            function: simdgroupFunction
        )
    }

    public func fusedF32(
        queries: MTLBuffer, keys: MTLBuffer, values: MTLBuffer,
        queryCount: Int, keyCount: Int, heads: Int, dimensions: Int,
        output: MTLBuffer,
        implementation: AttentionImplementation = .automatic
    ) throws {
        let querySegments = try AttentionSegments(offsets: [0, queryCount])
        let keySegments = try AttentionSegments(offsets: [0, keyCount])
        try segmentedF32(
            queries: queries, keys: keys, values: values,
            querySegments: querySegments, keySegments: keySegments,
            heads: heads, dimensions: dimensions, output: output,
            implementation: implementation
        )
    }

    public func segmentedF32(
        queries: MTLBuffer, keys: MTLBuffer, values: MTLBuffer,
        querySegments: AttentionSegments, keySegments: AttentionSegments,
        heads: Int, dimensions: Int, output: MTLBuffer,
        implementation: AttentionImplementation = .automatic
    ) throws {
        let queryElements = try checkedProduct(querySegments.totalCount, heads, dimensions)
        let keyElements = try checkedProduct(keySegments.totalCount, heads, dimensions)
        let queryBytes = try checkedBytes(queryElements)
        let keyBytes = try checkedBytes(keyElements)
        guard querySegments.totalCount > 0, keySegments.totalCount > 0, heads > 0,
              dimensions > 0, dimensions <= 256,
              dimensions.nonzeroBitCount == 1,
              dimensions <= pipeline.maxTotalThreadsPerThreadgroup,
              querySegments.segmentCount == keySegments.segmentCount,
              querySegments.totalCount <= Int(Int32.max),
              keySegments.totalCount <= Int(Int32.max),
              heads <= Int(UInt32.max), dimensions <= Int(UInt32.max),
              queries.length >= queryBytes, keys.length >= keyBytes,
              values.length >= keyBytes, output.length >= queryBytes,
              output !== keys, output !== values else {
            throw NativeRuntimeError.invalidArgument("invalid fused attention buffers or dimensions")
        }
        let rowBytes = try checkedBytes(try checkedProduct(heads, dimensions, 1))
        for segment in 0..<querySegments.segmentCount {
            let queryCount = querySegments.offsets[segment + 1]
                - querySegments.offsets[segment]
            let keyCount = keySegments.offsets[segment + 1]
                - keySegments.offsets[segment]
            if queryCount == 0 { continue }
            let groups = try checkedProduct(queryCount, heads, 1)
            let localQueryElements = try checkedProduct(queryCount, heads, dimensions)
            let localKeyElements = try checkedProduct(keyCount, heads, dimensions)
            guard keyCount > 0, queryCount <= Int(Int32.max),
                  keyCount <= Int(Int32.max), groups <= Int(UInt32.max),
                  localQueryElements <= Int(UInt32.max),
                  localKeyElements <= Int(UInt32.max) else {
                throw NativeRuntimeError.invalidArgument(
                    "attention segment exceeds Metal 32-bit indexing"
                )
            }
        }
        let operationCount = querySegments.totalCount.multipliedReportingOverflow(
            by: keySegments.totalCount
        )
        let canUseMPSGraph = context.mpsGraphAttention.isSupported
            && querySegments.segmentCount == 1
            && dimensions == 128
            && !operationCount.overflow
            && operationCount.partialValue >= 1_000_000
        if implementation == .mpsGraph && !canUseMPSGraph {
            throw NativeRuntimeError.invalidArgument(
                "MPSGraph attention requires one large 128-wide segment on macOS 15 or newer"
            )
        }
        if implementation == .mpsGraph || (implementation == .automatic && canUseMPSGraph) {
            if #available(macOS 15.0, *) {
                context.mpsGraphAttention.run(
                    queries: queries,
                    keys: keys,
                    values: values,
                    queryCount: querySegments.totalCount,
                    keyCount: keySegments.totalCount,
                    heads: heads,
                    dimensions: dimensions,
                    output: output
                )
            }
            return
        }
        let useSIMDGroup = (dimensions == 64 || dimensions == 128)
            && simdgroupPipeline.threadExecutionWidth == 32
            && simdgroupPipeline.maxTotalThreadsPerThreadgroup >= 256
        try context.runCompute(label: "fused attention") { encoder in
            encoder.setComputePipelineState(useSIMDGroup ? simdgroupPipeline : pipeline)
            for segment in 0..<querySegments.segmentCount {
                let queryStart = querySegments.offsets[segment]
                let queryCount = querySegments.offsets[segment + 1] - queryStart
                let keyStart = keySegments.offsets[segment]
                let keyCount = keySegments.offsets[segment + 1] - keyStart
                if queryCount == 0 { continue }
                var parameters = Parameters(
                    queryCount: UInt32(queryCount), keyCount: UInt32(keyCount),
                    heads: UInt32(heads), dimensions: UInt32(dimensions),
                    scale: 1 / sqrt(Float(dimensions))
                )
                encoder.setBuffer(queries, offset: queryStart * rowBytes, index: 0)
                encoder.setBuffer(keys, offset: keyStart * rowBytes, index: 1)
                encoder.setBuffer(values, offset: keyStart * rowBytes, index: 2)
                encoder.setBuffer(output, offset: queryStart * rowBytes, index: 3)
                encoder.setBytes(
                    &parameters, length: MemoryLayout<Parameters>.stride, index: 4
                )
                if useSIMDGroup {
                    encoder.dispatchThreadgroups(
                        MTLSize(width: (queryCount + 7) / 8, height: heads, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                    )
                } else {
                    let groups = try checkedProduct(queryCount, heads, 1)
                    encoder.dispatchThreadgroups(
                        MTLSize(width: groups, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(
                            width: dimensions, height: 1, depth: 1
                        )
                    )
                }
            }
        }
    }

}

private func checkedProduct(_ first: Int, _ second: Int, _ third: Int) throws -> Int {
    let a = first.multipliedReportingOverflow(by: second)
    let b = a.partialValue.multipliedReportingOverflow(by: third)
    guard !a.overflow, !b.overflow else {
        throw NativeRuntimeError.invalidArgument("attention element count overflows Int")
    }
    return b.partialValue
}

private func checkedBytes(_ elements: Int) throws -> Int {
    let result = elements.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
    guard !result.overflow else {
        throw NativeRuntimeError.invalidArgument("attention byte count overflows Int")
    }
    return result.partialValue
}
