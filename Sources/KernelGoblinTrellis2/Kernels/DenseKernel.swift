import Metal

public final class DenseKernel: @unchecked Sendable {
    private struct LinearF32Parameters {
        var weightOffset: UInt64
        var biasOffset: UInt64
        var rows: UInt32
        var inputChannels: UInt32
        var outputChannels: UInt32
        var hasBias: UInt32
    }

    private let context: MetalContext
    private let linearF32Pipeline: MTLComputePipelineState
    private let linearBF16WeightsPipeline: MTLComputePipelineState
    private static let tileSize = 16

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.library(named: "identity")
        guard let function = library.makeFunction(name: "kg_linear_tiled_f32") else {
            throw NativeRuntimeError.invalidArgument(
                "kg_linear_tiled_f32 is missing from Metal library"
            )
        }
        self.linearF32Pipeline = try context.device.makeComputePipelineState(function: function)
        guard let bf16Function = library.makeFunction(
            name: "kg_linear_tiled_bf16_weights_f32_output"
        ) else {
            throw NativeRuntimeError.invalidArgument(
                "kg_linear_tiled_bf16_weights_f32_output is missing from Metal library"
            )
        }
        self.linearBF16WeightsPipeline = try context.device.makeComputePipelineState(
            function: bf16Function
        )
    }

    public func linearF32(
        input: MTLBuffer,
        checkpoint: MTLBuffer,
        weightOffset: Int,
        biasOffset: Int? = nil,
        rows: Int,
        inputChannels: Int,
        outputChannels: Int,
        output: MTLBuffer
    ) throws {
        try linear(
            pipeline: linearF32Pipeline,
            elementWidth: MemoryLayout<Float>.stride,
            input: input,
            checkpoint: checkpoint,
            weightOffset: weightOffset,
            biasOffset: biasOffset,
            rows: rows,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            output: output
        )
    }

    public func linearBF16WeightsF32Output(
        input: MTLBuffer,
        checkpoint: MTLBuffer,
        weightOffset: Int,
        biasOffset: Int? = nil,
        rows: Int,
        inputChannels: Int,
        outputChannels: Int,
        output: MTLBuffer
    ) throws {
        try linear(
            pipeline: linearBF16WeightsPipeline,
            elementWidth: MemoryLayout<UInt16>.stride,
            input: input,
            checkpoint: checkpoint,
            weightOffset: weightOffset,
            biasOffset: biasOffset,
            rows: rows,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            output: output
        )
    }

    private func linear(
        pipeline: MTLComputePipelineState,
        elementWidth: Int,
        input: MTLBuffer,
        checkpoint: MTLBuffer,
        weightOffset: Int,
        biasOffset: Int?,
        rows: Int,
        inputChannels: Int,
        outputChannels: Int,
        output: MTLBuffer
    ) throws {
        guard rows > 0, inputChannels > 0, outputChannels > 0,
              weightOffset >= 0, biasOffset.map({ $0 >= 0 }) ?? true else {
            throw NativeRuntimeError.invalidArgument("linear dimensions and offsets must be positive")
        }
        let inputBytes = try checkedBytes(rows, inputChannels, MemoryLayout<Float>.stride)
        let weightBytes = try checkedBytes(outputChannels, inputChannels, elementWidth)
        let outputBytes = try checkedBytes(rows, outputChannels, MemoryLayout<Float>.stride)
        let weightEnd = try checkedAdd(weightOffset, weightBytes)
        let biasBytes = try checkedBytes(outputChannels, 1, elementWidth)
        let biasEnd = try biasOffset.map { try checkedAdd($0, biasBytes) }
        guard input.length >= inputBytes,
              checkpoint.length >= weightEnd,
              output.length >= outputBytes,
              output !== input, output !== checkpoint,
              biasEnd.map({ checkpoint.length >= $0 }) ?? true,
              weightOffset % elementWidth == 0,
              biasOffset.map({ $0 % elementWidth == 0 }) ?? true
        else {
            throw NativeRuntimeError.invalidArgument("linear buffers or tensor alignment are invalid")
        }
        guard rows <= Int(UInt32.max), inputChannels <= Int(UInt32.max),
              outputChannels <= Int(UInt32.max),
              fitsUInt32Product(rows, outputChannels),
              fitsUInt32Product(rows, inputChannels),
              fitsUInt32Product(outputChannels, inputChannels) else {
            throw NativeRuntimeError.invalidArgument("linear dimensions exceed Metal UInt32 indexing")
        }
        var parameters = LinearF32Parameters(
            weightOffset: UInt64(weightOffset),
            biasOffset: UInt64(biasOffset ?? weightOffset),
            rows: UInt32(rows),
            inputChannels: UInt32(inputChannels),
            outputChannels: UInt32(outputChannels),
            hasBias: biasOffset == nil ? 0 : 1
        )
        try context.runCompute(label: "dense kernel") { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(checkpoint, offset: 0, index: 1)
            encoder.setBuffer(output, offset: 0, index: 2)
            encoder.setBytes(
                &parameters, length: MemoryLayout<LinearF32Parameters>.stride, index: 3
            )
            let tile = Self.tileSize
            guard pipeline.maxTotalThreadsPerThreadgroup >= tile * tile else {
                throw NativeRuntimeError.invalidArgument(
                    "Metal device cannot dispatch the required dense tile"
                )
            }
            encoder.dispatchThreadgroups(
                MTLSize(
                    width: (outputChannels + tile - 1) / tile,
                    height: (rows + tile - 1) / tile,
                    depth: 1
                ),
                threadsPerThreadgroup: MTLSize(width: tile, height: tile, depth: 1)
            )
        }
    }
}

private func checkedBytes(_ first: Int, _ second: Int, _ width: Int) throws -> Int {
    let a = first.multipliedReportingOverflow(by: second)
    let b = a.partialValue.multipliedReportingOverflow(by: width)
    guard !a.overflow, !b.overflow else {
        throw NativeRuntimeError.invalidArgument("tensor byte count overflow")
    }
    return b.partialValue
}

private func checkedAdd(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.addingReportingOverflow(rhs)
    guard !result.overflow else {
        throw NativeRuntimeError.invalidArgument("tensor byte range overflows Int")
    }
    return result.partialValue
}

private func fitsUInt32Product(_ lhs: Int, _ rhs: Int) -> Bool {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    return !result.overflow && result.partialValue <= Int(UInt32.max)
}
