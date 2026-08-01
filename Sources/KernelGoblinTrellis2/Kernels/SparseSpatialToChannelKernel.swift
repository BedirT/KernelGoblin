import Metal

public final class SparseSpatialToChannelKernel: @unchecked Sendable {
    private struct Parameters {
        var coarseCount: UInt32
        var inputChannels: UInt32
        var outputChannels: UInt32
    }

    private let context: MetalContext
    private let packPipeline: MTLComputePipelineState
    private let skipPipeline: MTLComputePipelineState
    private let posteriorMeanPipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.library(named: "sparse_spatial_to_channel")
        guard let pack = library.makeFunction(name: "kg_spatial_to_channel_pack_f32"),
              let skip = library.makeFunction(name: "kg_spatial_to_channel_skip_f32"),
              let posteriorMean = library.makeFunction(
                  name: "kg_sparse_posterior_mean_f32"
              )
        else {
            throw NativeRuntimeError.invalidArgument(
                "spatial-to-channel Metal functions are missing"
            )
        }
        self.packPipeline = try context.device.makeComputePipelineState(function: pack)
        self.skipPipeline = try context.device.makeComputePipelineState(function: skip)
        self.posteriorMeanPipeline = try context.device.makeComputePipelineState(
            function: posteriorMean
        )
    }

    public func pack(
        input: MTLBuffer, sourceIndices: MTLBuffer,
        coarseCount: Int, channels: Int, output: MTLBuffer
    ) throws {
        let outputChannels = try sparseS2CProduct(channels, 8)
        try run(
            pipeline: packPipeline, input: input, sourceIndices: sourceIndices,
            coarseCount: coarseCount, inputChannels: channels,
            outputChannels: outputChannels, output: output
        )
    }

    public func skipMean(
        input: MTLBuffer, sourceIndices: MTLBuffer,
        coarseCount: Int, inputChannels: Int, outputChannels: Int,
        output: MTLBuffer
    ) throws {
        let packedChannels = try sparseS2CProduct(inputChannels, 8)
        guard packedChannels.isMultiple(of: outputChannels) else {
            throw NativeRuntimeError.invalidArgument(
                "spatial-to-channel skip channels cannot be evenly reduced"
            )
        }
        try run(
            pipeline: skipPipeline, input: input, sourceIndices: sourceIndices,
            coarseCount: coarseCount, inputChannels: inputChannels,
            outputChannels: outputChannels, output: output
        )
    }

    public func selectPosteriorMean(
        input: MTLBuffer, rows: Int, output: MTLBuffer
    ) throws {
        let inputElements = try sparseS2CProduct(rows, 64)
        let outputElements = try sparseS2CProduct(rows, 32)
        let inputBytes = try sparseS2CProduct(
            inputElements, MemoryLayout<Float>.stride
        )
        let outputBytes = try sparseS2CProduct(
            outputElements, MemoryLayout<Float>.stride
        )
        guard rows > 0, rows <= Int(UInt32.max),
              outputElements <= Int(UInt32.max), input.length >= inputBytes,
              output.length >= outputBytes, input !== output else {
            throw NativeRuntimeError.invalidArgument("invalid posterior mean buffers")
        }
        var metalRows = UInt32(rows)
        try context.runCompute(label: "sparse posterior mean selection") { encoder in
            encoder.setComputePipelineState(posteriorMeanPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(&metalRows, length: MemoryLayout<UInt32>.stride, index: 2)
            let width = min(posteriorMeanPipeline.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(
                MTLSize(width: outputElements, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
            )
        }
    }

    private func run(
        pipeline: MTLComputePipelineState,
        input: MTLBuffer, sourceIndices: MTLBuffer,
        coarseCount: Int, inputChannels: Int, outputChannels: Int,
        output: MTLBuffer
    ) throws {
        let slots = try sparseS2CProduct(coarseCount, 8)
        let mapBytes = try sparseS2CProduct(slots, MemoryLayout<Int32>.stride)
        let outputElements = try sparseS2CProduct(coarseCount, outputChannels)
        let outputBytes = try sparseS2CProduct(
            outputElements, MemoryLayout<Float>.stride
        )
        guard coarseCount > 0, inputChannels > 0, outputChannels > 0,
              coarseCount <= Int(UInt32.max), inputChannels <= Int(UInt32.max),
              outputChannels <= Int(UInt32.max), outputElements <= Int(UInt32.max),
              sourceIndices.storageMode != .private,
              sourceIndices.length >= mapBytes, output.length >= outputBytes,
              input !== output, input !== sourceIndices, output !== sourceIndices
        else {
            throw NativeRuntimeError.invalidArgument(
                "invalid spatial-to-channel Metal buffers"
            )
        }
        let map = sourceIndices.contents().assumingMemoryBound(to: Int32.self)
        var maximumSource: Int32 = -1
        for index in 0..<slots { maximumSource = max(maximumSource, map[index]) }
        let inputElements = try sparseS2CProduct(Int(maximumSource) + 1, inputChannels)
        let inputBytes = try sparseS2CProduct(
            inputElements, MemoryLayout<Float>.stride
        )
        guard input.length >= inputBytes else {
            throw NativeRuntimeError.invalidArgument(
                "spatial-to-channel source index is out of range"
            )
        }
        var parameters = Parameters(
            coarseCount: UInt32(coarseCount), inputChannels: UInt32(inputChannels),
            outputChannels: UInt32(outputChannels)
        )
        try context.runCompute(label: "sparse spatial-to-channel") { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(sourceIndices, offset: 0, index: 1)
            encoder.setBuffer(output, offset: 0, index: 2)
            encoder.setBytes(
                &parameters, length: MemoryLayout<Parameters>.stride, index: 3
            )
            let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(
                MTLSize(width: outputElements, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
            )
        }
    }
}
