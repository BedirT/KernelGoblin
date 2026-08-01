import Foundation
import Metal

public final class PrimitiveKernel: @unchecked Sendable {
    private struct TimestepEmbeddingParameters {
        var rows: UInt32
        var dimensions: UInt32
        var logMaxPeriod: Float
    }

    private struct SplitQKVParameters {
        var rows: UInt32
        var channels: UInt32
    }

    private struct ModulateParameters {
        var rows: UInt32
        var channels: UInt32
        var shiftOffset: UInt32
        var scaleOffset: UInt32
    }

    private struct ResidualParameters {
        var count: UInt32
        var channels: UInt32
        var gateOffset: UInt32
        var hasGate: UInt32
    }

    private struct AddCheckpointBF16Parameters {
        var offset: UInt64
        var count: UInt32
    }

    private let context: MetalContext
    private let timestepPipeline: MTLComputePipelineState
    private let siluPipeline: MTLComputePipelineState
    private let roundBF16Pipeline: MTLComputePipelineState
    private let splitQKVPipeline: MTLComputePipelineState
    private let splitKVPipeline: MTLComputePipelineState
    private let geluPipeline: MTLComputePipelineState
    private let modulatePipeline: MTLComputePipelineState
    private let modulateBF16Pipeline: MTLComputePipelineState
    private let residualPipeline: MTLComputePipelineState
    private let residualBF16Pipeline: MTLComputePipelineState
    private let addPipeline: MTLComputePipelineState
    private let addCheckpointBF16Pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.library(named: "identity")
        guard let timestep = library.makeFunction(name: "kg_timestep_embedding_f32"),
              let silu = library.makeFunction(name: "kg_silu_f32"),
              let roundBF16 = library.makeFunction(name: "kg_round_bf16_f32"),
              let splitQKV = library.makeFunction(name: "kg_split_qkv_f32"),
              let splitKV = library.makeFunction(name: "kg_split_kv_f32"),
              let gelu = library.makeFunction(name: "kg_gelu_tanh_f32"),
              let modulate = library.makeFunction(name: "kg_modulate_f32"),
              let modulateBF16 = library.makeFunction(name: "kg_modulate_bf16_f32"),
              let residual = library.makeFunction(name: "kg_residual_f32"),
              let residualBF16 = library.makeFunction(name: "kg_residual_bf16_f32"),
              let add = library.makeFunction(name: "kg_add_f32"),
              let addCheckpoint = library.makeFunction(name: "kg_add_checkpoint_bf16_f32") else {
            throw NativeRuntimeError.invalidArgument("primitive Metal functions are missing")
        }
        self.timestepPipeline = try context.device.makeComputePipelineState(function: timestep)
        self.siluPipeline = try context.device.makeComputePipelineState(function: silu)
        self.roundBF16Pipeline = try context.device.makeComputePipelineState(function: roundBF16)
        self.splitQKVPipeline = try context.device.makeComputePipelineState(function: splitQKV)
        self.splitKVPipeline = try context.device.makeComputePipelineState(function: splitKV)
        self.geluPipeline = try context.device.makeComputePipelineState(function: gelu)
        self.modulatePipeline = try context.device.makeComputePipelineState(function: modulate)
        self.modulateBF16Pipeline = try context.device.makeComputePipelineState(function: modulateBF16)
        self.residualPipeline = try context.device.makeComputePipelineState(function: residual)
        self.residualBF16Pipeline = try context.device.makeComputePipelineState(function: residualBF16)
        self.addPipeline = try context.device.makeComputePipelineState(function: add)
        self.addCheckpointBF16Pipeline = try context.device.makeComputePipelineState(
            function: addCheckpoint
        )
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

    public func roundBF16F32(input: MTLBuffer, count: Int, output: MTLBuffer) throws {
        guard count > 0, count <= Int(UInt32.max),
              input.length >= count * MemoryLayout<Float>.stride,
              output.length >= count * MemoryLayout<Float>.stride else {
            throw NativeRuntimeError.invalidArgument("invalid BF16 rounding buffers or count")
        }
        var metalCount = UInt32(count)
        let countData = withUnsafeBytes(of: &metalCount) { Data($0) }
        try dispatch(
            pipeline: roundBF16Pipeline, count: count,
            buffers: [(input, 0), (output, 1)], bytes: (countData, 2)
        )
    }

    public func splitQKVF32(
        input: MTLBuffer, rows: Int, channels: Int,
        query: MTLBuffer, key: MTLBuffer, value: MTLBuffer
    ) throws {
        let elements = rows.multipliedReportingOverflow(by: channels)
        let bytes = elements.partialValue.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
        let inputBytes = bytes.partialValue.multipliedReportingOverflow(by: 3)
        guard rows > 0, channels > 0, rows <= Int(UInt32.max),
              channels <= Int(UInt32.max), !elements.overflow, !bytes.overflow,
              !inputBytes.overflow, elements.partialValue <= Int(UInt32.max),
              input.length >= inputBytes.partialValue,
              query.length >= bytes.partialValue, key.length >= bytes.partialValue,
              value.length >= bytes.partialValue else {
            throw NativeRuntimeError.invalidArgument("invalid QKV split buffers or dimensions")
        }
        var parameters = SplitQKVParameters(rows: UInt32(rows), channels: UInt32(channels))
        let data = withUnsafeBytes(of: &parameters) { Data($0) }
        try dispatch(
            pipeline: splitQKVPipeline, count: elements.partialValue,
            buffers: [(input, 0), (query, 1), (key, 2), (value, 3)], bytes: (data, 4)
        )
    }

    public func splitKVF32(
        input: MTLBuffer, rows: Int, channels: Int,
        key: MTLBuffer, value: MTLBuffer
    ) throws {
        let elements = rows.multipliedReportingOverflow(by: channels)
        let bytes = elements.partialValue.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
        let inputBytes = bytes.partialValue.multipliedReportingOverflow(by: 2)
        guard rows > 0, channels > 0, rows <= Int(UInt32.max),
              channels <= Int(UInt32.max), !elements.overflow, !bytes.overflow,
              !inputBytes.overflow, elements.partialValue <= Int(UInt32.max),
              input.length >= inputBytes.partialValue,
              key.length >= bytes.partialValue, value.length >= bytes.partialValue else {
            throw NativeRuntimeError.invalidArgument("invalid KV split buffers or dimensions")
        }
        var parameters = SplitQKVParameters(rows: UInt32(rows), channels: UInt32(channels))
        let data = withUnsafeBytes(of: &parameters) { Data($0) }
        try dispatch(
            pipeline: splitKVPipeline, count: elements.partialValue,
            buffers: [(input, 0), (key, 1), (value, 2)], bytes: (data, 3)
        )
    }

    public func geluTanhF32(input: MTLBuffer, count: Int, output: MTLBuffer) throws {
        try dispatchCounted(pipeline: geluPipeline, input: input, count: count, output: output)
    }

    public func addF32(lhs: MTLBuffer, rhs: MTLBuffer, count: Int, output: MTLBuffer) throws {
        try validateFloatBuffers([lhs, rhs, output], count: count)
        var metalCount = UInt32(count)
        let data = withUnsafeBytes(of: &metalCount) { Data($0) }
        try dispatch(
            pipeline: addPipeline, count: count,
            buffers: [(lhs, 0), (rhs, 1), (output, 2)], bytes: (data, 3)
        )
    }

    public func addCheckpointBF16F32(
        input: MTLBuffer, checkpoint: MTLBuffer, checkpointOffset: Int,
        count: Int, output: MTLBuffer
    ) throws {
        let checkpointBytes = count.multipliedReportingOverflow(by: 2)
        let checkpointEnd = checkpointOffset.addingReportingOverflow(checkpointBytes.partialValue)
        guard checkpointOffset >= 0, checkpointOffset % 2 == 0,
              !checkpointBytes.overflow, !checkpointEnd.overflow,
              checkpoint.length >= checkpointEnd.partialValue else {
            throw NativeRuntimeError.invalidArgument("invalid BF16 checkpoint addition range")
        }
        try validateFloatBuffers([input, output], count: count)
        var parameters = AddCheckpointBF16Parameters(
            offset: UInt64(checkpointOffset), count: UInt32(count)
        )
        let data = withUnsafeBytes(of: &parameters) { Data($0) }
        try dispatch(
            pipeline: addCheckpointBF16Pipeline, count: count,
            buffers: [(input, 0), (checkpoint, 1), (output, 2)], bytes: (data, 3)
        )
    }

    public func modulateF32(
        input: MTLBuffer, modulation: MTLBuffer, rows: Int, channels: Int,
        shiftOffset: Int, scaleOffset: Int, output: MTLBuffer
    ) throws {
        try modulate(
            pipeline: modulatePipeline, input: input, modulation: modulation,
            rows: rows, channels: channels, shiftOffset: shiftOffset,
            scaleOffset: scaleOffset, output: output
        )
    }

    public func modulateBF16F32(
        input: MTLBuffer, modulation: MTLBuffer, rows: Int, channels: Int,
        shiftOffset: Int, scaleOffset: Int, output: MTLBuffer
    ) throws {
        try modulate(
            pipeline: modulateBF16Pipeline, input: input, modulation: modulation,
            rows: rows, channels: channels, shiftOffset: shiftOffset,
            scaleOffset: scaleOffset, output: output
        )
    }

    private func modulate(
        pipeline: MTLComputePipelineState, input: MTLBuffer, modulation: MTLBuffer,
        rows: Int, channels: Int, shiftOffset: Int, scaleOffset: Int,
        output: MTLBuffer
    ) throws {
        let elements = try elementCount(rows, channels)
        let lastOffset = max(shiftOffset, scaleOffset)
        let modulationElements = lastOffset.addingReportingOverflow(channels)
        let modulationBytes = modulationElements.partialValue.multipliedReportingOverflow(by: 4)
        guard shiftOffset >= 0, scaleOffset >= 0,
              shiftOffset <= Int(UInt32.max), scaleOffset <= Int(UInt32.max),
              !modulationElements.overflow, !modulationBytes.overflow,
              modulation.length >= modulationBytes.partialValue else {
            throw NativeRuntimeError.invalidArgument("invalid modulation offsets")
        }
        try validateFloatBuffers([input, output], count: elements)
        var parameters = ModulateParameters(
            rows: UInt32(rows), channels: UInt32(channels),
            shiftOffset: UInt32(shiftOffset), scaleOffset: UInt32(scaleOffset)
        )
        let data = withUnsafeBytes(of: &parameters) { Data($0) }
        try dispatch(
            pipeline: pipeline, count: elements,
            buffers: [(input, 0), (modulation, 1), (output, 2)], bytes: (data, 3)
        )
    }

    public func residualF32(
        residual: MTLBuffer, branch: MTLBuffer, modulation: MTLBuffer,
        rows: Int, channels: Int, gateOffset: Int? = nil, output: MTLBuffer
    ) throws {
        let elements = try elementCount(rows, channels)
        guard let offset = gateOffset else {
            var parameters = ResidualParameters(
                count: UInt32(elements), channels: UInt32(channels),
                gateOffset: 0, hasGate: 0
            )
            return try dispatchResidual(
                pipeline: residualPipeline,
                residual: residual, branch: branch, modulation: modulation,
                output: output, elements: elements, parameters: &parameters
            )
        }
        let modulationElements = offset.addingReportingOverflow(channels)
        let modulationBytes = modulationElements.partialValue.multipliedReportingOverflow(by: 4)
        guard offset >= 0, offset <= Int(UInt32.max),
              !modulationElements.overflow, !modulationBytes.overflow,
              modulation.length >= modulationBytes.partialValue else {
            throw NativeRuntimeError.invalidArgument("invalid residual gate offset")
        }
        var parameters = ResidualParameters(
            count: UInt32(elements), channels: UInt32(channels),
            gateOffset: UInt32(offset), hasGate: 1
        )
        try dispatchResidual(
            pipeline: residualPipeline,
            residual: residual, branch: branch, modulation: modulation,
            output: output, elements: elements, parameters: &parameters
        )
    }

    public func residualBF16F32(
        residual: MTLBuffer, branch: MTLBuffer, modulation: MTLBuffer,
        rows: Int, channels: Int, gateOffset: Int? = nil, output: MTLBuffer
    ) throws {
        let elements = try elementCount(rows, channels)
        let offset = gateOffset ?? 0
        let modulationElements = offset.addingReportingOverflow(channels)
        let modulationBytes = modulationElements.partialValue.multipliedReportingOverflow(by: 4)
        guard offset >= 0, offset <= Int(UInt32.max),
              !modulationElements.overflow, !modulationBytes.overflow,
              gateOffset == nil || modulation.length >= modulationBytes.partialValue else {
            throw NativeRuntimeError.invalidArgument("invalid BF16 residual gate offset")
        }
        var parameters = ResidualParameters(
            count: UInt32(elements), channels: UInt32(channels),
            gateOffset: UInt32(offset), hasGate: gateOffset == nil ? 0 : 1
        )
        try dispatchResidual(
            pipeline: residualBF16Pipeline,
            residual: residual, branch: branch, modulation: modulation,
            output: output, elements: elements, parameters: &parameters
        )
    }

    private func dispatchResidual(
        pipeline: MTLComputePipelineState,
        residual: MTLBuffer, branch: MTLBuffer, modulation: MTLBuffer,
        output: MTLBuffer, elements: Int, parameters: inout ResidualParameters
    ) throws {
        try validateFloatBuffers([residual, branch, output], count: elements)
        let data = withUnsafeBytes(of: &parameters) { Data($0) }
        try dispatch(
            pipeline: pipeline, count: elements,
            buffers: [(residual, 0), (branch, 1), (modulation, 2), (output, 3)],
            bytes: (data, 4)
        )
    }

    private func dispatchCounted(
        pipeline: MTLComputePipelineState, input: MTLBuffer,
        count: Int, output: MTLBuffer
    ) throws {
        try validateFloatBuffers([input, output], count: count)
        var metalCount = UInt32(count)
        let data = withUnsafeBytes(of: &metalCount) { Data($0) }
        try dispatch(
            pipeline: pipeline, count: count,
            buffers: [(input, 0), (output, 1)], bytes: (data, 2)
        )
    }

    private func validateFloatBuffers(_ buffers: [MTLBuffer], count: Int) throws {
        guard count > 0, count <= Int(UInt32.max),
              buffers.allSatisfy({ $0.length >= count * 4 }) else {
            throw NativeRuntimeError.invalidArgument("invalid elementwise buffers or count")
        }
    }

    private func elementCount(_ rows: Int, _ channels: Int) throws -> Int {
        let result = rows.multipliedReportingOverflow(by: channels)
        guard rows > 0, channels > 0, !result.overflow,
              result.partialValue <= Int(UInt32.max) else {
            throw NativeRuntimeError.invalidArgument("elementwise dimensions exceed UInt32")
        }
        return result.partialValue
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
