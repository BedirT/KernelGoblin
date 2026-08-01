import Metal

public enum BF16LinearImplementation: Sendable {
    case automatic
    case tiled
    case simdgroupMatrix
    case mpsGraph
    case mpsGraphModelPrecision
}

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
    private let linearF16WeightsPipeline: MTLComputePipelineState
    private let linearBF16SIMDGroupPipeline: MTLComputePipelineState?
    private static let tileSize = 16

    public init(context: MetalContext, enableSIMDGroupMatrix: Bool = true) throws {
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
        guard let f16Function = library.makeFunction(
            name: "kg_linear_tiled_f16_weights_f32_output"
        ) else {
            throw NativeRuntimeError.invalidArgument(
                "kg_linear_tiled_f16_weights_f32_output is missing from Metal library"
            )
        }
        self.linearF16WeightsPipeline = try context.device.makeComputePipelineState(
            function: f16Function
        )
        do {
            guard enableSIMDGroupMatrix else {
                self.linearBF16SIMDGroupPipeline = nil
                return
            }
            let simdLibrary = try context.library(named: "dense_simdgroup")
            guard let function = simdLibrary.makeFunction(
                name: "kg_linear_simdgroup_bf16_weights_f32_output"
            ) else {
                throw NativeRuntimeError.invalidArgument(
                    "kg_linear_simdgroup_bf16_weights_f32_output is missing from Metal library"
                )
            }
            self.linearBF16SIMDGroupPipeline = try context.device.makeComputePipelineState(
                function: function
            )
        } catch {
            self.linearBF16SIMDGroupPipeline = nil
        }
    }

    public var supportsSIMDGroupMatrix: Bool { linearBF16SIMDGroupPipeline != nil }
    public var supportsMPSGraph: Bool { context.mpsGraphDense.isSupported }

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
        output: MTLBuffer,
        implementation: BF16LinearImplementation = .automatic
    ) throws {
        try validateLinear(
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
        let rowInputCount = rows.multipliedReportingOverflow(by: inputChannels)
        let operationCount = rowInputCount.partialValue.multipliedReportingOverflow(
            by: outputChannels
        )
        let useMPSGraph = supportsMPSGraph && ((implementation == .mpsGraph
            || implementation == .mpsGraphModelPrecision)
            || (implementation == .automatic
                && !operationCount.overflow
                && !rowInputCount.overflow
                && rows >= 8
                && operationCount.partialValue >= 1_000_000))
        if useMPSGraph {
            if #available(macOS 15.2, *) {
                try context.mpsGraphDense.run(
                    input: input,
                    checkpoint: checkpoint,
                    weightOffset: weightOffset,
                    biasOffset: biasOffset,
                    rows: rows,
                    inputChannels: inputChannels,
                    outputChannels: outputChannels,
                    output: output,
                    modelPrecision: implementation != .mpsGraph
                )
            }
            return
        }
        if implementation == .mpsGraph || implementation == .mpsGraphModelPrecision {
            throw NativeRuntimeError.invalidArgument(
                "MPSGraph dense requires macOS 15.2 or newer"
            )
        }
        let selectedPipeline: MTLComputePipelineState
        let usesSIMDGroupMatrix: Bool
        switch implementation {
        case .automatic:
            // A single row does not amortize the cooperative matrix setup on
            // Apple M3. Keep tiny conditioning projections on the tiled path.
            if rows >= 8, let linearBF16SIMDGroupPipeline {
                selectedPipeline = linearBF16SIMDGroupPipeline
                usesSIMDGroupMatrix = true
            } else {
                selectedPipeline = linearBF16WeightsPipeline
                usesSIMDGroupMatrix = false
            }
        case .tiled:
            selectedPipeline = linearBF16WeightsPipeline
            usesSIMDGroupMatrix = false
        case .simdgroupMatrix:
            guard let linearBF16SIMDGroupPipeline else {
                throw NativeRuntimeError.invalidArgument(
                    "Metal device cannot create the SIMD-group matrix dense pipeline"
                )
            }
            selectedPipeline = linearBF16SIMDGroupPipeline
            usesSIMDGroupMatrix = true
        case .mpsGraph, .mpsGraphModelPrecision:
            fatalError("MPSGraph dispatch returned before Metal pipeline selection")
        }
        try linear(
            pipeline: selectedPipeline,
            elementWidth: MemoryLayout<UInt16>.stride,
            simdgroupMatrix: usesSIMDGroupMatrix,
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

    public func linearF16WeightsF32Output(
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
            pipeline: linearF16WeightsPipeline,
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
        simdgroupMatrix: Bool = false,
        input: MTLBuffer,
        checkpoint: MTLBuffer,
        weightOffset: Int,
        biasOffset: Int?,
        rows: Int,
        inputChannels: Int,
        outputChannels: Int,
        output: MTLBuffer
    ) throws {
        try validateLinear(
            elementWidth: elementWidth,
            input: input,
            checkpoint: checkpoint,
            weightOffset: weightOffset,
            biasOffset: biasOffset,
            rows: rows,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            output: output,
            tile: simdgroupMatrix ? 8 : Self.tileSize
        )
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
            let tile = simdgroupMatrix ? 8 : Self.tileSize
            let threads = simdgroupMatrix ? 32 : tile * tile
            guard pipeline.maxTotalThreadsPerThreadgroup >= threads,
                  !simdgroupMatrix || pipeline.threadExecutionWidth == 32 else {
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
                threadsPerThreadgroup: simdgroupMatrix
                    ? MTLSize(width: threads, height: 1, depth: 1)
                    : MTLSize(width: tile, height: tile, depth: 1)
            )
        }
    }

    private func validateLinear(
        elementWidth: Int,
        input: MTLBuffer,
        checkpoint: MTLBuffer,
        weightOffset: Int,
        biasOffset: Int?,
        rows: Int,
        inputChannels: Int,
        outputChannels: Int,
        output: MTLBuffer,
        tile: Int = 8
    ) throws {
        guard rows > 0, inputChannels > 0, outputChannels > 0,
              weightOffset >= 0, biasOffset.map({ $0 >= 0 }) ?? true else {
            throw NativeRuntimeError.invalidArgument("linear dimensions and offsets must be positive")
        }
        guard rows <= Int(UInt32.max),
              inputChannels <= Int(UInt32.max) - tile,
              outputChannels <= Int(UInt32.max),
              fitsUInt32Product(rows, outputChannels),
              fitsUInt32Product(rows, inputChannels),
              fitsUInt32Product(outputChannels, inputChannels) else {
            throw NativeRuntimeError.invalidArgument("linear dimensions exceed Metal UInt32 indexing")
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
