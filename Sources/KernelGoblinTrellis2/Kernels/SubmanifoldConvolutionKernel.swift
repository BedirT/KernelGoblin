import Metal

public enum SubmanifoldConvolutionImplementation: Sendable {
    case automatic
    case scalar
    case simdgroupMatrix
}

public final class SubmanifoldConvolutionKernel: @unchecked Sendable {
    private struct Parameters {
        var weightOffset: UInt64
        var biasOffset: UInt64
        var tokenCount: UInt32
        var inputChannels: UInt32
        var outputChannels: UInt32
        var hasBias: UInt32
    }

    private let context: MetalContext
    private let scalarPipeline: MTLComputePipelineState
    private let simdPipeline: MTLComputePipelineState?

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.library(named: "sparse_convolution")
        guard let scalar = library.makeFunction(name: "kg_subm_conv3x3_f16_scalar_f32") else {
            throw NativeRuntimeError.invalidArgument(
                "kg_subm_conv3x3_f16_scalar_f32 is missing from Metal library"
            )
        }
        self.scalarPipeline = try context.device.makeComputePipelineState(function: scalar)
        do {
            let simdLibrary = try context.library(named: "sparse_convolution_simdgroup")
            guard let function = simdLibrary.makeFunction(
                name: "kg_subm_conv3x3_f16_simdgroup_f32"
            ) else {
                throw NativeRuntimeError.invalidArgument(
                    "kg_subm_conv3x3_f16_simdgroup_f32 is missing from Metal library"
                )
            }
            self.simdPipeline = try context.device.makeComputePipelineState(function: function)
        } catch {
            self.simdPipeline = nil
        }
    }

    public var supportsSIMDGroupMatrix: Bool { simdPipeline != nil }

    public func convolveF16WeightsF32Output(
        input: MTLBuffer,
        neighbors: MTLBuffer?,
        checkpoint: MTLBuffer,
        weightOffset: Int,
        biasOffset: Int? = nil,
        tokenCount: Int,
        inputChannels: Int,
        outputChannels: Int,
        output: MTLBuffer,
        implementation: SubmanifoldConvolutionImplementation = .automatic
    ) throws {
        guard tokenCount >= 0, inputChannels > 0, outputChannels > 0,
              weightOffset >= 0, biasOffset.map({ $0 >= 0 }) ?? true else {
            throw NativeRuntimeError.invalidArgument("invalid submanifold convolution dimensions")
        }
        if tokenCount == 0 { return }
        let inputElements = try checkedProduct(tokenCount, inputChannels)
        let outputElements = try checkedProduct(tokenCount, outputChannels)
        let neighborElements = try checkedProduct(
            tokenCount, SparseNeighborhood3x3.neighborCount
        )
        let weightElements = try checkedProduct(
            try checkedProduct(outputChannels, SparseNeighborhood3x3.neighborCount),
            inputChannels
        )
        let inputBytes = try checkedProduct(inputElements, MemoryLayout<Float>.stride)
        let outputBytes = try checkedProduct(outputElements, MemoryLayout<Float>.stride)
        let neighborBytes = try checkedProduct(neighborElements, MemoryLayout<Int32>.stride)
        let weightEnd = try checkedSum(
            weightOffset, try checkedProduct(weightElements, MemoryLayout<UInt16>.stride)
        )
        let biasEnd = try biasOffset.map {
            try checkedSum($0, try checkedProduct(outputChannels, MemoryLayout<UInt16>.stride))
        }
        guard tokenCount <= Int(UInt32.max), inputChannels <= Int(UInt32.max),
              outputChannels <= Int(UInt32.max), outputElements <= Int(UInt32.max),
              input.length >= inputBytes, output.length >= outputBytes,
              neighbors.map({ $0.length >= neighborBytes }) ?? false,
              neighbors?.storageMode != .private,
              checkpoint.length >= weightEnd,
              biasEnd.map({ checkpoint.length >= $0 }) ?? true,
              weightOffset.isMultiple(of: 2),
              biasOffset.map({ $0.isMultiple(of: 2) }) ?? true,
              output !== input, output !== checkpoint, output !== neighbors else {
            throw NativeRuntimeError.invalidArgument(
                "invalid submanifold convolution buffers or byte ranges"
            )
        }
        let neighborValues = neighbors!.contents().assumingMemoryBound(to: Int32.self)
        for index in 0..<neighborElements {
            guard neighborValues[index] >= -1,
                  neighborValues[index] < Int32(tokenCount) else {
                throw NativeRuntimeError.invalidArgument(
                    "sparse neighbor index is outside the input token range"
                )
            }
        }
        var parameters = Parameters(
            weightOffset: UInt64(weightOffset),
            biasOffset: UInt64(biasOffset ?? weightOffset),
            tokenCount: UInt32(tokenCount),
            inputChannels: UInt32(inputChannels),
            outputChannels: UInt32(outputChannels),
            hasBias: biasOffset == nil ? 0 : 1
        )
        let pipeline: MTLComputePipelineState
        let usesSIMD: Bool
        switch implementation {
        case .automatic:
            if tokenCount >= 8, inputChannels >= 8, outputChannels >= 8,
               let simdPipeline {
                pipeline = simdPipeline
                usesSIMD = true
            } else {
                pipeline = scalarPipeline
                usesSIMD = false
            }
        case .scalar:
            pipeline = scalarPipeline
            usesSIMD = false
        case .simdgroupMatrix:
            guard let simdPipeline else {
                throw NativeRuntimeError.invalidArgument(
                    "Metal device cannot create sparse SIMD-group matrix pipeline"
                )
            }
            pipeline = simdPipeline
            usesSIMD = true
        }
        try context.runCompute(label: "submanifold sparse convolution") { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(neighbors, offset: 0, index: 1)
            encoder.setBuffer(checkpoint, offset: 0, index: 2)
            encoder.setBuffer(output, offset: 0, index: 3)
            encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 4)
            if usesSIMD {
                guard pipeline.maxTotalThreadsPerThreadgroup >= 32,
                      pipeline.threadExecutionWidth == 32 else {
                    throw NativeRuntimeError.invalidArgument(
                        "Metal device cannot dispatch sparse SIMD-group matrix tile"
                    )
                }
                encoder.dispatchThreadgroups(
                    MTLSize(
                        width: (outputChannels + 7) / 8,
                        height: (tokenCount + 7) / 8,
                        depth: 1
                    ),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            } else {
                let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
                encoder.dispatchThreads(
                    MTLSize(width: outputElements, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
                )
            }
        }
    }
}

private func checkedProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard lhs >= 0, rhs >= 0, !result.overflow else {
        throw NativeRuntimeError.invalidArgument("sparse convolution size overflows Int")
    }
    return result.partialValue
}

private func checkedSum(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.addingReportingOverflow(rhs)
    guard lhs >= 0, rhs >= 0, !result.overflow else {
        throw NativeRuntimeError.invalidArgument("sparse convolution range overflows Int")
    }
    return result.partialValue
}
