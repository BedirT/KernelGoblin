import Foundation
import Metal

public enum NormalizationImplementation: Sendable {
    case automatic
    case scalar
    case simdgroup
}

public final class NormalizationKernel: @unchecked Sendable {
    private struct LayerNormParameters {
        var weightOffset: UInt64
        var biasOffset: UInt64
        var rows: UInt32
        var channels: UInt32
        var hasAffine: UInt32
        var epsilon: Float
    }

    private struct RMSNormParameters {
        var gammaOffset: UInt64
        var rows: UInt32
        var heads: UInt32
        var dimensions: UInt32
        var epsilon: Float
    }

    private let context: MetalContext
    private let layerNormPipeline: MTLComputePipelineState
    private let layerNormF32AffinePipeline: MTLComputePipelineState
    private let layerNormF16AffinePipeline: MTLComputePipelineState
    private let rmsNormPipeline: MTLComputePipelineState
    private let layerNormSIMDGroupPipeline: MTLComputePipelineState
    private let rmsNormSIMDGroupPipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.library(named: "identity")
        guard let layerNorm = library.makeFunction(name: "kg_layer_norm_f32"),
              let layerNormF32Affine = library.makeFunction(
                  name: "kg_layer_norm_f32_affine_f32"
              ),
              let layerNormF16Affine = library.makeFunction(
                  name: "kg_layer_norm_f32_affine_f16"
              ),
              let rmsNorm = library.makeFunction(name: "kg_multihead_rms_norm_f32"),
              let layerNormSIMDGroup = library.makeFunction(
                  name: "kg_layer_norm_simdgroup_f32"
              ),
              let rmsNormSIMDGroup = library.makeFunction(
                  name: "kg_multihead_rms_norm_simdgroup_f32"
              ) else {
            throw NativeRuntimeError.invalidArgument("normalization Metal functions are missing")
        }
        self.layerNormPipeline = try context.device.makeComputePipelineState(function: layerNorm)
        self.layerNormF32AffinePipeline = try context.device.makeComputePipelineState(
            function: layerNormF32Affine
        )
        self.layerNormF16AffinePipeline = try context.device.makeComputePipelineState(
            function: layerNormF16Affine
        )
        self.rmsNormPipeline = try context.device.makeComputePipelineState(function: rmsNorm)
        self.layerNormSIMDGroupPipeline = try context.device.makeComputePipelineState(
            function: layerNormSIMDGroup
        )
        self.rmsNormSIMDGroupPipeline = try context.device.makeComputePipelineState(
            function: rmsNormSIMDGroup
        )
    }

    public func layerNormF32(
        input: MTLBuffer, checkpoint: MTLBuffer, rows: Int, channels: Int,
        weightOffset: Int? = nil, biasOffset: Int? = nil, epsilon: Float = 1e-6,
        output: MTLBuffer,
        implementation: NormalizationImplementation = .automatic
    ) throws {
        try layerNorm(
            pipeline: layerNormPipeline, affineWidth: 2,
            input: input, checkpoint: checkpoint, rows: rows, channels: channels,
            weightOffset: weightOffset, biasOffset: biasOffset,
            epsilon: epsilon, output: output,
            implementation: implementation
        )
    }

    public func layerNormF32WeightsF32(
        input: MTLBuffer, checkpoint: MTLBuffer, rows: Int, channels: Int,
        weightOffset: Int? = nil, biasOffset: Int? = nil, epsilon: Float = 1e-5,
        output: MTLBuffer
    ) throws {
        try layerNorm(
            pipeline: layerNormF32AffinePipeline, affineWidth: 4,
            input: input, checkpoint: checkpoint, rows: rows, channels: channels,
            weightOffset: weightOffset, biasOffset: biasOffset,
            epsilon: epsilon, output: output
        )
    }

    public func layerNormF32WeightsF16(
        input: MTLBuffer, checkpoint: MTLBuffer, rows: Int, channels: Int,
        weightOffset: Int? = nil, biasOffset: Int? = nil, epsilon: Float = 1e-6,
        output: MTLBuffer
    ) throws {
        try layerNorm(
            pipeline: layerNormF16AffinePipeline, affineWidth: 2,
            input: input, checkpoint: checkpoint, rows: rows, channels: channels,
            weightOffset: weightOffset, biasOffset: biasOffset,
            epsilon: epsilon, output: output
        )
    }

    private func layerNorm(
        pipeline: MTLComputePipelineState, affineWidth: Int,
        input: MTLBuffer, checkpoint: MTLBuffer, rows: Int, channels: Int,
        weightOffset: Int?, biasOffset: Int?, epsilon: Float,
        output: MTLBuffer,
        implementation: NormalizationImplementation = .scalar
    ) throws {
        let affine = weightOffset != nil || biasOffset != nil
        let affineBytes = try checkedByteCount(
            rows: 1, channels: channels, width: affineWidth
        )
        let weightEnd = try weightOffset.map { try checkedSum($0, affineBytes) }
        let biasEnd = try biasOffset.map { try checkedSum($0, affineBytes) }
        guard rows > 0, channels > 0, epsilon > 0,
              rows <= Int(UInt32.max), channels <= Int(UInt32.max),
              (!affine || (weightOffset != nil && biasOffset != nil)),
              weightOffset.map({ $0 >= 0 && $0 % affineWidth == 0 }) ?? true,
              biasOffset.map({ $0 >= 0 && $0 % affineWidth == 0 }) ?? true,
              fitsBuffer(rows, channels, width: 4, buffer: input),
              fitsBuffer(rows, channels, width: 4, buffer: output),
              weightEnd.map({ checkpoint.length >= $0 }) ?? true,
              biasEnd.map({ checkpoint.length >= $0 }) ?? true else {
            throw NativeRuntimeError.invalidArgument("invalid LayerNorm buffers or dimensions")
        }
        var parameters = LayerNormParameters(
            weightOffset: UInt64(weightOffset ?? 0), biasOffset: UInt64(biasOffset ?? 0),
            rows: UInt32(rows), channels: UInt32(channels),
            hasAffine: affine ? 1 : 0, epsilon: epsilon
        )
        let useSIMDGroup = pipeline === layerNormPipeline
            && channels >= 128
            && layerNormSIMDGroupPipeline.threadExecutionWidth == 32
            && implementation != .scalar
        if implementation == .simdgroup && !useSIMDGroup {
            throw NativeRuntimeError.invalidArgument(
                "SIMD-group LayerNorm requires BF16 affine data and at least 128 channels"
            )
        }
        if useSIMDGroup {
            try dispatchSIMDGroup(
                pipeline: layerNormSIMDGroupPipeline, groups: rows,
                buffers: [(input, 0), (checkpoint, 1), (output, 2)],
                parameters: &parameters
            )
        } else {
            try dispatch(
                pipeline: pipeline, count: rows,
                buffers: [(input, 0), (checkpoint, 1), (output, 2)], parameters: &parameters
            )
        }
    }

    public func multiheadRMSNormF32(
        input: MTLBuffer, checkpoint: MTLBuffer, gammaOffset: Int,
        rows: Int, heads: Int, dimensions: Int, epsilon: Float = 1e-12,
        output: MTLBuffer,
        implementation: NormalizationImplementation = .automatic
    ) throws {
        let groupCount = try checkedElementCount(rows, heads)
        let gammaBytes = try checkedByteCount(rows: heads, channels: dimensions, width: 2)
        let gammaEnd = try checkedSum(gammaOffset, gammaBytes)
        guard rows > 0, heads > 0, dimensions > 0, epsilon > 0,
              rows <= Int(UInt32.max), heads <= Int(UInt32.max),
              dimensions <= Int(UInt32.max), gammaOffset >= 0, gammaOffset % 2 == 0,
              groupCount <= Int(UInt32.max),
              fitsBuffer(groupCount, dimensions, width: 4, buffer: input),
              fitsBuffer(groupCount, dimensions, width: 4, buffer: output),
              checkpoint.length >= gammaEnd else {
            throw NativeRuntimeError.invalidArgument("invalid RMSNorm buffers or dimensions")
        }
        var parameters = RMSNormParameters(
            gammaOffset: UInt64(gammaOffset), rows: UInt32(rows), heads: UInt32(heads),
            dimensions: UInt32(dimensions), epsilon: epsilon
        )
        let useSIMDGroup = dimensions >= 64
            && rmsNormSIMDGroupPipeline.threadExecutionWidth == 32
            && implementation != .scalar
        if implementation == .simdgroup && !useSIMDGroup {
            throw NativeRuntimeError.invalidArgument(
                "SIMD-group RMSNorm requires at least 64 dimensions"
            )
        }
        if useSIMDGroup {
            try dispatchSIMDGroup(
                pipeline: rmsNormSIMDGroupPipeline, groups: groupCount,
                buffers: [(input, 0), (checkpoint, 1), (output, 2)],
                parameters: &parameters
            )
        } else {
            try dispatch(
                pipeline: rmsNormPipeline, count: groupCount,
                buffers: [(input, 0), (checkpoint, 1), (output, 2)], parameters: &parameters
            )
        }
    }

    private func dispatch<T>(
        pipeline: MTLComputePipelineState, count: Int,
        buffers: [(MTLBuffer, Int)], parameters: inout T
    ) throws {
        try context.runCompute(label: "normalization kernel") { encoder in
            encoder.setComputePipelineState(pipeline)
            for (buffer, index) in buffers {
                encoder.setBuffer(buffer, offset: 0, index: index)
            }
            let parameterData = withUnsafeBytes(of: &parameters) { Data($0) }
            parameterData.withUnsafeBytes { raw in
                encoder.setBytes(raw.baseAddress!, length: raw.count, index: 3)
            }
            let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
            )
        }
    }

    private func dispatchSIMDGroup<T>(
        pipeline: MTLComputePipelineState, groups: Int,
        buffers: [(MTLBuffer, Int)], parameters: inout T
    ) throws {
        try context.runCompute(label: "SIMD-group normalization kernel") { encoder in
            encoder.setComputePipelineState(pipeline)
            for (buffer, index) in buffers {
                encoder.setBuffer(buffer, offset: 0, index: index)
            }
            let parameterData = withUnsafeBytes(of: &parameters) { Data($0) }
            parameterData.withUnsafeBytes { raw in
                encoder.setBytes(raw.baseAddress!, length: raw.count, index: 3)
            }
            encoder.dispatchThreadgroups(
                MTLSize(width: groups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        }
    }
}

private func fitsBuffer(_ rows: Int, _ channels: Int, width: Int, buffer: MTLBuffer) -> Bool {
    let elements = rows.multipliedReportingOverflow(by: channels)
    let bytes = elements.partialValue.multipliedReportingOverflow(by: width)
    return !elements.overflow && !bytes.overflow && buffer.length >= bytes.partialValue
}

private func checkedElementCount(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard !result.overflow else {
        throw NativeRuntimeError.invalidArgument("normalization element count overflows Int")
    }
    return result.partialValue
}

private func checkedByteCount(rows: Int, channels: Int, width: Int) throws -> Int {
    let elements = try checkedElementCount(rows, channels)
    return try checkedElementCount(elements, width)
}

private func checkedSum(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.addingReportingOverflow(rhs)
    guard !result.overflow else {
        throw NativeRuntimeError.invalidArgument("normalization byte range overflows Int")
    }
    return result.partialValue
}
