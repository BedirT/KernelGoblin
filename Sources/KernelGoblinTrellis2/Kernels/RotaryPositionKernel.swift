import Metal

public final class RotaryPositionKernel: @unchecked Sendable {
    private struct Parameters {
        var tokens: UInt32
        var heads: UInt32
        var dimensions: UInt32
        var frequencyDimensions: UInt32
        var minimumFrequency: Float
        var maximumFrequency: Float
    }

    private let context: MetalContext
    private let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.library(named: "identity")
        guard let function = library.makeFunction(name: "kg_rope3d_f32") else {
            throw NativeRuntimeError.invalidArgument("3D RoPE Metal function is missing")
        }
        self.pipeline = try context.device.makeComputePipelineState(function: function)
    }

    public func apply3DF32(
        input: MTLBuffer, coordinates: MTLBuffer, tokens: Int, heads: Int,
        dimensions: Int, minimumFrequency: Float = 1,
        maximumFrequency: Float = 10_000, output: MTLBuffer
    ) throws {
        let elements = try ropeProduct(tokens, heads, dimensions)
        let tensorBytes = try ropeBytes(elements, MemoryLayout<Float>.stride)
        let coordinateElements = try ropeProduct(tokens, 4, 1)
        let coordinateBytes = try ropeBytes(coordinateElements, MemoryLayout<Int32>.stride)
        let frequencyDimensions = dimensions / 2 / 3
        guard tokens > 0, heads > 0, dimensions > 0, dimensions % 2 == 0,
              frequencyDimensions > 0, minimumFrequency > 0, maximumFrequency > 0,
              tokens <= Int(UInt32.max), heads <= Int(UInt32.max),
              dimensions <= Int(UInt32.max), frequencyDimensions <= Int(UInt32.max),
              elements <= Int(UInt32.max),
              input !== output,
              input.length >= tensorBytes, output.length >= tensorBytes,
              coordinates.length >= coordinateBytes else {
            throw NativeRuntimeError.invalidArgument("invalid 3D RoPE buffers or dimensions")
        }
        var parameters = Parameters(
            tokens: UInt32(tokens), heads: UInt32(heads),
            dimensions: UInt32(dimensions),
            frequencyDimensions: UInt32(frequencyDimensions),
            minimumFrequency: minimumFrequency, maximumFrequency: maximumFrequency
        )
        try context.runCompute(label: "3D RoPE") { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(coordinates, offset: 0, index: 1)
            encoder.setBuffer(output, offset: 0, index: 2)
            encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 3)
            let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(
                MTLSize(width: elements, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
            )
        }
    }
}

private func ropeProduct(_ first: Int, _ second: Int, _ third: Int) throws -> Int {
    let a = first.multipliedReportingOverflow(by: second)
    let b = a.partialValue.multipliedReportingOverflow(by: third)
    guard !a.overflow, !b.overflow else {
        throw NativeRuntimeError.invalidArgument("3D RoPE element count overflows Int")
    }
    return b.partialValue
}

private func ropeBytes(_ count: Int, _ width: Int) throws -> Int {
    let result = count.multipliedReportingOverflow(by: width)
    guard !result.overflow else {
        throw NativeRuntimeError.invalidArgument("3D RoPE byte count overflows Int")
    }
    return result.partialValue
}
