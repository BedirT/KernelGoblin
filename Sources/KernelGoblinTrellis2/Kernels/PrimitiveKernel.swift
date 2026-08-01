import Foundation
import Metal

public final class PrimitiveKernel: @unchecked Sendable {
    private struct TimestepEmbeddingParameters {
        var rows: UInt32
        var dimensions: UInt32
        var logMaxPeriod: Float
    }

    private let context: MetalContext
    private let timestepPipeline: MTLComputePipelineState
    private let siluPipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.library(named: "identity")
        guard let timestep = library.makeFunction(name: "kg_timestep_embedding_f32"),
              let silu = library.makeFunction(name: "kg_silu_f32") else {
            throw NativeRuntimeError.invalidArgument("primitive Metal functions are missing")
        }
        self.timestepPipeline = try context.device.makeComputePipelineState(function: timestep)
        self.siluPipeline = try context.device.makeComputePipelineState(function: silu)
    }

    public func timestepEmbeddingF32(
        timesteps: MTLBuffer,
        rows: Int,
        dimensions: Int,
        maxPeriod: Float = 10_000,
        output: MTLBuffer
    ) throws {
        guard rows > 0, dimensions >= 2, maxPeriod > 0,
              rows <= Int(UInt32.max), dimensions <= Int(UInt32.max),
              fitsUInt32(rows, dimensions),
              timesteps.length >= rows * MemoryLayout<Float>.stride,
              output.length >= rows * dimensions * MemoryLayout<Float>.stride else {
            throw NativeRuntimeError.invalidArgument("invalid timestep embedding buffers or dimensions")
        }
        var parameters = TimestepEmbeddingParameters(
            rows: UInt32(rows), dimensions: UInt32(dimensions), logMaxPeriod: log(maxPeriod)
        )
        let parameterData = withUnsafeBytes(of: &parameters) { Data($0) }
        try dispatch(
            pipeline: timestepPipeline,
            count: rows * dimensions,
            buffers: [(timesteps, 0), (output, 1)],
            bytes: (parameterData, 2)
        )
    }

    public func siluF32(input: MTLBuffer, count: Int, output: MTLBuffer) throws {
        guard count > 0, count <= Int(UInt32.max),
              input.length >= count * MemoryLayout<Float>.stride,
              output.length >= count * MemoryLayout<Float>.stride else {
            throw NativeRuntimeError.invalidArgument("invalid SiLU buffers or count")
        }
        var metalCount = UInt32(count)
        let countData = withUnsafeBytes(of: &metalCount) { Data($0) }
        try dispatch(
            pipeline: siluPipeline,
            count: count,
            buffers: [(input, 0), (output, 1)],
            bytes: (countData, 2)
        )
    }

    private func dispatch(
        pipeline: MTLComputePipelineState,
        count: Int,
        buffers: [(MTLBuffer, Int)],
        bytes: (Data, Int)
    ) throws {
        guard let command = context.queue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder() else {
            throw NativeRuntimeError.allocationFailed("could not create primitive Metal command")
        }
        encoder.setComputePipelineState(pipeline)
        for (buffer, index) in buffers { encoder.setBuffer(buffer, offset: 0, index: index) }
        bytes.0.withUnsafeBytes { raw in
            encoder.setBytes(raw.baseAddress!, length: raw.count, index: bytes.1)
        }
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if command.status == .error {
            throw NativeRuntimeError.allocationFailed(
                "Metal primitive command failed: \(command.error?.localizedDescription ?? "unknown error")"
            )
        }
    }
}

private func fitsUInt32(_ lhs: Int, _ rhs: Int) -> Bool {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    return !result.overflow && result.partialValue <= Int(UInt32.max)
}
