import Metal

public final class PBRFieldKernel: @unchecked Sendable {
    private let context: MetalContext
    private let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.library(named: "pbr_field")
        guard let function = library.makeFunction(name: "kg_trellis_texture_to_pbr_f32") else {
            throw NativeRuntimeError.invalidArgument("TRELLIS PBR field function is missing")
        }
        self.pipeline = try context.device.makeComputePipelineState(function: function)
    }

    public func callAsFunction(
        raw: MTLBuffer, count: Int, output: MTLBuffer
    ) throws {
        let bytes = count.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
        guard count > 0, count <= Int(UInt32.max), !bytes.overflow,
              raw.length >= bytes.partialValue,
              output.length >= bytes.partialValue else {
            throw NativeRuntimeError.invalidArgument("invalid TRELLIS PBR field buffers")
        }
        var metalCount = UInt32(count)
        try context.runCompute(label: "TRELLIS texture-to-PBR transform") { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(raw, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(&metalCount, length: MemoryLayout<UInt32>.stride, index: 2)
            let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
            )
        }
    }
}
