import Foundation
import Metal

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

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.library(named: "identity")
        guard let function = library.makeFunction(name: "kg_fused_attention_f32") else {
            throw NativeRuntimeError.invalidArgument("fused attention Metal function is missing")
        }
        self.pipeline = try context.device.makeComputePipelineState(function: function)
    }

    public func fusedF32(
        queries: MTLBuffer, keys: MTLBuffer, values: MTLBuffer,
        queryCount: Int, keyCount: Int, heads: Int, dimensions: Int,
        output: MTLBuffer
    ) throws {
        let queryElements = try checkedProduct(queryCount, heads, dimensions)
        let keyElements = try checkedProduct(keyCount, heads, dimensions)
        let queryBytes = try checkedBytes(queryElements)
        let keyBytes = try checkedBytes(keyElements)
        let groups = try checkedProduct(queryCount, heads, 1)
        guard queryCount > 0, keyCount > 0, heads > 0,
              dimensions > 0, dimensions <= 256,
              dimensions.nonzeroBitCount == 1,
              dimensions <= pipeline.maxTotalThreadsPerThreadgroup,
              queryCount <= Int(UInt32.max), keyCount <= Int(UInt32.max),
              heads <= Int(UInt32.max), dimensions <= Int(UInt32.max),
              queries.length >= queryBytes, keys.length >= keyBytes,
              values.length >= keyBytes, output.length >= queryBytes else {
            throw NativeRuntimeError.invalidArgument("invalid fused attention buffers or dimensions")
        }
        var parameters = Parameters(
            queryCount: UInt32(queryCount), keyCount: UInt32(keyCount),
            heads: UInt32(heads), dimensions: UInt32(dimensions),
            scale: 1 / sqrt(Float(dimensions))
        )
        guard let command = context.queue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder() else {
            throw NativeRuntimeError.allocationFailed("could not create fused attention command")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queries, offset: 0, index: 0)
        encoder.setBuffer(keys, offset: 0, index: 1)
        encoder.setBuffer(values, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: dimensions, height: 1, depth: 1)
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if command.status == .error {
            throw NativeRuntimeError.allocationFailed(
                "Metal fused attention failed: \(command.error?.localizedDescription ?? "unknown error")"
            )
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
