import Foundation
import Metal

public final class DINOv3Kernel: @unchecked Sendable {
    private struct PatchEmbedParameters {
        var weightOffset: UInt64
        var biasOffset: UInt64
        var imageHeight: UInt32
        var imageWidth: UInt32
        var patchSize: UInt32
        var inputChannels: UInt32
        var outputChannels: UInt32
        var outputTokenOffset: UInt32
    }

    private struct RopeParameters {
        var tokenCount: UInt32
        var prefixTokens: UInt32
        var patchesH: UInt32
        var patchesW: UInt32
        var heads: UInt32
        var dimensions: UInt32
        var theta: Float
    }

    private let context: MetalContext
    private let patchPipeline: MTLComputePipelineState
    private let ropePipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.library(named: "identity")
        guard let patch = library.makeFunction(name: "kg_dino_patch_embed_f32"),
              let rope = library.makeFunction(name: "kg_dino_rope_f32") else {
            throw NativeRuntimeError.invalidArgument("DINOv3 Metal functions are missing")
        }
        patchPipeline = try context.device.makeComputePipelineState(function: patch)
        ropePipeline = try context.device.makeComputePipelineState(function: rope)
    }

    public func patchEmbedF32(
        image: MTLBuffer,
        checkpoint: MTLBuffer,
        weightOffset: Int,
        biasOffset: Int,
        imageHeight: Int,
        imageWidth: Int,
        patchSize: Int,
        inputChannels: Int,
        outputChannels: Int,
        outputTokenOffset: Int,
        output: MTLBuffer
    ) throws {
        guard imageHeight > 0, imageWidth > 0, patchSize > 0,
              imageHeight % patchSize == 0, imageWidth % patchSize == 0,
              inputChannels > 0, outputChannels > 0, outputTokenOffset >= 0,
              weightOffset >= 0, biasOffset >= 0,
              [imageHeight, imageWidth, patchSize, inputChannels, outputChannels,
               outputTokenOffset].allSatisfy({ $0 <= Int(UInt32.max) }) else {
            throw NativeRuntimeError.invalidArgument("invalid DINO patch embedding dimensions")
        }
        let patchCount = try dinoProduct(imageHeight / patchSize, imageWidth / patchSize)
        let imageElements = try dinoProduct(try dinoProduct(imageHeight, imageWidth), inputChannels)
        let kernelElements = try dinoProduct(
            try dinoProduct(try dinoProduct(outputChannels, inputChannels), patchSize),
            patchSize
        )
        let outputTokens = try dinoSum(outputTokenOffset, patchCount)
        let outputElements = try dinoProduct(outputTokens, outputChannels)
        let imageBytes = try dinoBytes(imageElements)
        let outputBytes = try dinoBytes(outputElements)
        let weightEnd = try dinoSum(weightOffset, try dinoBytes(kernelElements))
        let biasEnd = try dinoSum(biasOffset, try dinoBytes(outputChannels))
        guard image.length >= imageBytes,
              checkpoint.length >= weightEnd, checkpoint.length >= biasEnd,
              output.length >= outputBytes,
              output !== image, output !== checkpoint else {
            throw NativeRuntimeError.invalidArgument("invalid DINO patch embedding buffers")
        }
        var parameters = PatchEmbedParameters(
            weightOffset: UInt64(weightOffset), biasOffset: UInt64(biasOffset),
            imageHeight: UInt32(imageHeight), imageWidth: UInt32(imageWidth),
            patchSize: UInt32(patchSize), inputChannels: UInt32(inputChannels),
            outputChannels: UInt32(outputChannels),
            outputTokenOffset: UInt32(outputTokenOffset)
        )
        try dispatch(
            pipeline: patchPipeline,
            count: try dinoProduct(patchCount, outputChannels),
            buffers: [(image, 0), (checkpoint, 1), (output, 2)],
            parameters: &parameters,
            parameterIndex: 3
        )
    }

    public func ropeF32(
        input: MTLBuffer,
        tokenCount: Int,
        prefixTokens: Int,
        patchesH: Int,
        patchesW: Int,
        heads: Int,
        dimensions: Int,
        theta: Float,
        output: MTLBuffer
    ) throws {
        let patchCount = try dinoProduct(patchesH, patchesW)
        let expectedTokens = try dinoSum(prefixTokens, patchCount)
        let elements = try dinoProduct(try dinoProduct(tokenCount, heads), dimensions)
        let bytes = try dinoBytes(elements)
        guard tokenCount == expectedTokens, prefixTokens > 0, heads > 0,
              dimensions >= 4, dimensions % 4 == 0, theta > 0,
              [tokenCount, prefixTokens, patchesH, patchesW, heads, dimensions]
                .allSatisfy({ $0 > 0 && $0 <= Int(UInt32.max) }),
              input.length >= bytes, output.length >= bytes,
              input !== output else {
            throw NativeRuntimeError.invalidArgument("invalid DINO RoPE buffers or dimensions")
        }
        var parameters = RopeParameters(
            tokenCount: UInt32(tokenCount), prefixTokens: UInt32(prefixTokens),
            patchesH: UInt32(patchesH), patchesW: UInt32(patchesW),
            heads: UInt32(heads), dimensions: UInt32(dimensions), theta: theta
        )
        try dispatch(
            pipeline: ropePipeline, count: elements,
            buffers: [(input, 0), (output, 1)],
            parameters: &parameters, parameterIndex: 2
        )
    }

    private func dispatch<Parameters>(
        pipeline: MTLComputePipelineState,
        count: Int,
        buffers: [(MTLBuffer, Int)],
        parameters: inout Parameters,
        parameterIndex: Int
    ) throws {
        guard count > 0, count <= Int(UInt32.max),
              let command = context.queue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder() else {
            throw NativeRuntimeError.allocationFailed("could not create DINO Metal command")
        }
        encoder.setComputePipelineState(pipeline)
        for (buffer, index) in buffers { encoder.setBuffer(buffer, offset: 0, index: index) }
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
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw NativeRuntimeError.allocationFailed(
                "DINO Metal command failed: \(command.error?.localizedDescription ?? "unknown error")"
            )
        }
    }
}

private func dinoProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard !result.overflow else {
        throw NativeRuntimeError.invalidArgument("DINO tensor size overflow")
    }
    return result.partialValue
}

private func dinoSum(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.addingReportingOverflow(rhs)
    guard !result.overflow else {
        throw NativeRuntimeError.invalidArgument("DINO tensor range overflow")
    }
    return result.partialValue
}

private func dinoBytes(_ elements: Int) throws -> Int {
    try dinoProduct(elements, MemoryLayout<Float>.stride)
}
