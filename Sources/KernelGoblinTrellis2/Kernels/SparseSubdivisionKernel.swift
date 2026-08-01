import Metal

public final class SparseSubdivisionKernel: @unchecked Sendable {
    private struct Parameters {
        var childCount: UInt32
        var inputChannels: UInt32
        var outputChannels: UInt32
    }

    private let context: MetalContext
    private let selectPipeline: MTLComputePipelineState
    private let skipPipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.library(named: "sparse_subdivision")
        guard let select = library.makeFunction(name: "kg_channel_to_spatial_f32"),
              let skip = library.makeFunction(name: "kg_channel_to_spatial_skip_f32") else {
            throw NativeRuntimeError.invalidArgument("sparse subdivision Metal functions are missing")
        }
        self.selectPipeline = try context.device.makeComputePipelineState(function: select)
        self.skipPipeline = try context.device.makeComputePipelineState(function: skip)
    }

    public func selectConvolutionFeatures(
        input: MTLBuffer, parentIndices: MTLBuffer, childIndices: MTLBuffer,
        childCount: Int, outputChannels: Int, output: MTLBuffer
    ) throws {
        let inputChannels = try sparseSubdivisionKernelProduct(outputChannels, 8)
        try run(
            pipeline: selectPipeline, input: input,
            parentIndices: parentIndices, childIndices: childIndices,
            childCount: childCount, inputChannels: inputChannels,
            outputChannels: outputChannels, output: output
        )
    }

    public func selectSkipFeatures(
        input: MTLBuffer, parentIndices: MTLBuffer, childIndices: MTLBuffer,
        childCount: Int, inputChannels: Int, outputChannels: Int,
        output: MTLBuffer
    ) throws {
        guard inputChannels.isMultiple(of: 8),
              outputChannels.isMultiple(of: inputChannels / 8) else {
            throw NativeRuntimeError.invalidArgument("invalid C2S skip channel ratio")
        }
        try run(
            pipeline: skipPipeline, input: input,
            parentIndices: parentIndices, childIndices: childIndices,
            childCount: childCount, inputChannels: inputChannels,
            outputChannels: outputChannels, output: output
        )
    }

    private func run(
        pipeline: MTLComputePipelineState, input: MTLBuffer,
        parentIndices: MTLBuffer, childIndices: MTLBuffer,
        childCount: Int, inputChannels: Int, outputChannels: Int,
        output: MTLBuffer
    ) throws {
        let outputElements = childCount.multipliedReportingOverflow(by: outputChannels)
        let inputBytes = childCount.multipliedReportingOverflow(by: MemoryLayout<UInt32>.stride)
        let outputBytes = outputElements.partialValue.multipliedReportingOverflow(
            by: MemoryLayout<Float>.stride
        )
        guard childCount > 0, inputChannels > 0, outputChannels > 0,
              childCount <= Int(UInt32.max), inputChannels <= Int(UInt32.max),
              outputChannels <= Int(UInt32.max), !outputElements.overflow,
              outputElements.partialValue <= Int(UInt32.max), !inputBytes.overflow,
              !outputBytes.overflow,
              parentIndices.storageMode != .private, childIndices.storageMode != .private,
              parentIndices.length >= inputBytes.partialValue,
              childIndices.length >= inputBytes.partialValue,
              output.length >= outputBytes.partialValue,
              output !== input else {
            throw NativeRuntimeError.invalidArgument("invalid sparse subdivision buffers")
        }
        let parents = parentIndices.contents().assumingMemoryBound(to: UInt32.self)
        let children = childIndices.contents().assumingMemoryBound(to: UInt32.self)
        var maximumParent: UInt32 = 0
        for index in 0..<childCount {
            guard children[index] < 8 else {
                throw NativeRuntimeError.invalidArgument("invalid sparse child index")
            }
            maximumParent = max(maximumParent, parents[index])
        }
        let requiredInputElements = (Int(maximumParent) + 1).multipliedReportingOverflow(
            by: inputChannels
        )
        let requiredInputBytes = requiredInputElements.partialValue.multipliedReportingOverflow(
            by: MemoryLayout<Float>.stride
        )
        guard !requiredInputElements.overflow,
              !requiredInputBytes.overflow,
              input.length >= requiredInputBytes.partialValue else {
            throw NativeRuntimeError.invalidArgument("sparse subdivision parent is out of range")
        }
        var parameters = Parameters(
            childCount: UInt32(childCount), inputChannels: UInt32(inputChannels),
            outputChannels: UInt32(outputChannels)
        )
        try context.runCompute(label: "sparse channel-to-spatial") { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(parentIndices, offset: 0, index: 1)
            encoder.setBuffer(childIndices, offset: 0, index: 2)
            encoder.setBuffer(output, offset: 0, index: 3)
            encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 4)
            let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(
                MTLSize(width: outputElements.partialValue, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
            )
        }
    }
}

private func sparseSubdivisionKernelProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard lhs > 0, rhs > 0, !result.overflow else {
        throw NativeRuntimeError.invalidArgument("sparse subdivision size overflows Int")
    }
    return result.partialValue
}
