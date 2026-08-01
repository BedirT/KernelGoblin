import Foundation
import Metal

public enum ConvolutionWeightType: Sendable {
    case f16
    case f32

    var byteWidth: Int { self == .f16 ? 2 : 4 }
}

public final class SparseStructureKernel: @unchecked Sendable {
    private struct Conv3DParameters {
        var weightOffset: UInt64
        var biasOffset: UInt64
        var inputResolution: UInt32
        var outputResolution: UInt32
        var inputChannels: UInt32
        var outputChannels: UInt32
        var kernelSize: UInt32
        var stride: UInt32
        var padding: UInt32
        var weightsAreF16: UInt32
    }

    private struct PixelShuffle3DParameters {
        var inputResolution: UInt32
        var outputChannels: UInt32
        var factor: UInt32
    }

    private let context: MetalContext
    private let conv3DPipeline: MTLComputePipelineState
    private let roundF16Pipeline: MTLComputePipelineState
    private let pixelShufflePipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.library(named: "sparse_structure")
        guard let conv3D = library.makeFunction(name: "kg_conv3d_voxel_major_f32"),
              let roundF16 = library.makeFunction(name: "kg_round_f16_f32"),
              let pixelShuffle = library.makeFunction(
                  name: "kg_pixel_shuffle_3d_voxel_major_f32"
              ) else {
            throw NativeRuntimeError.invalidArgument(
                "sparse-structure Metal functions are missing"
            )
        }
        self.conv3DPipeline = try context.device.makeComputePipelineState(function: conv3D)
        self.roundF16Pipeline = try context.device.makeComputePipelineState(function: roundF16)
        self.pixelShufflePipeline = try context.device.makeComputePipelineState(
            function: pixelShuffle
        )
    }

    public func conv3DF32(
        input: MTLBuffer, checkpoint: MTLBuffer,
        weightOffset: Int, biasOffset: Int,
        inputResolution: Int, inputChannels: Int, outputChannels: Int,
        kernelSize: Int = 3, stride: Int = 1, padding: Int = 1,
        weightType: ConvolutionWeightType, output: MTLBuffer
    ) throws {
        guard inputResolution > 0, inputChannels > 0, outputChannels > 0,
              kernelSize > 0, stride > 0, padding >= 0,
              weightOffset >= 0, biasOffset >= 0 else {
            throw NativeRuntimeError.invalidArgument("invalid Conv3D dimensions")
        }
        let padded = try checkedAdd(inputResolution, try checkedMultiply(padding, 2))
        guard padded >= kernelSize else {
            throw NativeRuntimeError.invalidArgument("Conv3D kernel exceeds padded input")
        }
        let outputResolution = (padded - kernelSize) / stride + 1
        let inputElements = try checkedProduct4(
            inputResolution, inputResolution, inputResolution, inputChannels
        )
        let outputElements = try checkedProduct4(
            outputResolution, outputResolution, outputResolution, outputChannels
        )
        let weightElements = try checkedProduct4(
            try checkedProduct4(outputChannels, inputChannels, 1, 1),
            kernelSize, kernelSize, kernelSize
        )
        let inputBytes = try checkedMultiply(inputElements, 4)
        let outputBytes = try checkedMultiply(outputElements, 4)
        let weightBytes = try checkedMultiply(weightElements, weightType.byteWidth)
        let biasBytes = try checkedMultiply(outputChannels, weightType.byteWidth)
        let weightEnd = try checkedAdd(weightOffset, weightBytes)
        let biasEnd = try checkedAdd(biasOffset, biasBytes)
        guard input.length >= inputBytes, output.length >= outputBytes,
              weightEnd <= checkpoint.length, biasEnd <= checkpoint.length,
              outputElements <= Int(UInt32.max),
              [inputResolution, outputResolution, inputChannels, outputChannels,
               kernelSize, stride, padding].allSatisfy({ $0 <= Int(UInt32.max) }) else {
            throw NativeRuntimeError.invalidArgument("invalid Conv3D buffers or byte ranges")
        }
        var parameters = Conv3DParameters(
            weightOffset: UInt64(weightOffset), biasOffset: UInt64(biasOffset),
            inputResolution: UInt32(inputResolution),
            outputResolution: UInt32(outputResolution),
            inputChannels: UInt32(inputChannels), outputChannels: UInt32(outputChannels),
            kernelSize: UInt32(kernelSize), stride: UInt32(stride),
            padding: UInt32(padding), weightsAreF16: weightType == .f16 ? 1 : 0
        )
        try dispatch(
            pipeline: conv3DPipeline, count: outputElements,
            buffers: [(input, 0), (checkpoint, 1), (output, 2)],
            parameters: &parameters, parameterIndex: 3
        )
    }

    public func roundF16F32(input: MTLBuffer, count: Int, output: MTLBuffer) throws {
        let bytes = try checkedMultiply(count, 4)
        guard count > 0, count <= Int(UInt32.max),
              input.length >= bytes, output.length >= bytes else {
            throw NativeRuntimeError.invalidArgument("invalid F16 rounding buffers")
        }
        var count = UInt32(count)
        try dispatch(
            pipeline: roundF16Pipeline, count: Int(count),
            buffers: [(input, 0), (output, 1)], parameters: &count, parameterIndex: 2
        )
    }

    public func pixelShuffle3DF32(
        input: MTLBuffer, inputResolution: Int, outputChannels: Int,
        factor: Int = 2, output: MTLBuffer
    ) throws {
        guard inputResolution > 0, outputChannels > 0, factor > 0 else {
            throw NativeRuntimeError.invalidArgument("invalid PixelShuffle3D dimensions")
        }
        let outputResolution = try checkedMultiply(inputResolution, factor)
        let factorCubed = try checkedProduct4(factor, factor, factor, 1)
        let inputChannels = try checkedMultiply(outputChannels, factorCubed)
        let inputElements = try checkedProduct4(
            inputResolution, inputResolution, inputResolution, inputChannels
        )
        let outputElements = try checkedProduct4(
            outputResolution, outputResolution, outputResolution, outputChannels
        )
        let inputBytes = try checkedMultiply(inputElements, 4)
        let outputBytes = try checkedMultiply(outputElements, 4)
        guard input.length >= inputBytes, output.length >= outputBytes,
              outputElements <= Int(UInt32.max),
              inputResolution <= Int(UInt32.max), outputChannels <= Int(UInt32.max),
              factor <= Int(UInt32.max) else {
            throw NativeRuntimeError.invalidArgument("invalid PixelShuffle3D buffers")
        }
        var parameters = PixelShuffle3DParameters(
            inputResolution: UInt32(inputResolution),
            outputChannels: UInt32(outputChannels), factor: UInt32(factor)
        )
        try dispatch(
            pipeline: pixelShufflePipeline, count: outputElements,
            buffers: [(input, 0), (output, 1)], parameters: &parameters,
            parameterIndex: 2
        )
    }

    private func dispatch<T>(
        pipeline: MTLComputePipelineState, count: Int,
        buffers: [(MTLBuffer, Int)], parameters: inout T, parameterIndex: Int
    ) throws {
        try context.runCompute(label: "sparse-structure kernel") { encoder in
            encoder.setComputePipelineState(pipeline)
            for (buffer, index) in buffers {
                encoder.setBuffer(buffer, offset: 0, index: index)
            }
            let parameterData = withUnsafeBytes(of: &parameters) { Data($0) }
            parameterData.withUnsafeBytes { bytes in
                encoder.setBytes(
                    bytes.baseAddress!, length: bytes.count, index: parameterIndex
                )
            }
            let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
            )
        }
    }
}

private func checkedMultiply(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard lhs >= 0, rhs >= 0, !result.overflow else {
        throw NativeRuntimeError.invalidArgument("sparse tensor size overflows Int")
    }
    return result.partialValue
}

private func checkedAdd(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.addingReportingOverflow(rhs)
    guard lhs >= 0, rhs >= 0, !result.overflow else {
        throw NativeRuntimeError.invalidArgument("sparse tensor extent overflows Int")
    }
    return result.partialValue
}

private func checkedProduct4(_ a: Int, _ b: Int, _ c: Int, _ d: Int) throws -> Int {
    try checkedMultiply(try checkedMultiply(try checkedMultiply(a, b), c), d)
}
