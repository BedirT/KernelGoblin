import Metal

public struct FlexibleDualGridHeadResult: @unchecked Sendable {
    public let dualOffsets: MTLBuffer
    public let intersections: MTLBuffer
    public let splitWeights: MTLBuffer
    public let count: Int
}

public final class FlexibleDualGridHeadKernel: @unchecked Sendable {
    private let context: MetalContext
    private let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.library(named: "flexible_dual_grid")
        guard let function = library.makeFunction(name: "kg_flexible_dual_grid_head_f32") else {
            throw NativeRuntimeError.invalidArgument("flexible dual-grid head function is missing")
        }
        self.pipeline = try context.device.makeComputePipelineState(function: function)
    }

    public func callAsFunction(rawHead: MTLBuffer, count: Int) throws -> FlexibleDualGridHeadResult {
        let rawBytes = try flexibleDualGridHeadBytes(count, 7, MemoryLayout<Float>.stride)
        let dualBytes = try flexibleDualGridHeadBytes(count, 3, MemoryLayout<Float>.stride)
        let intersectionBytes = try flexibleDualGridHeadBytes(count, 3, 1)
        let splitBytes = try flexibleDualGridHeadBytes(count, 1, MemoryLayout<Float>.stride)
        guard count > 0, count <= Int(UInt32.max),
              rawHead.length >= rawBytes else {
            throw NativeRuntimeError.invalidArgument("invalid flexible dual-grid raw head")
        }
        let dual = try context.makeBuffer(
            length: dualBytes, label: "TRELLIS flexible dual-grid offsets"
        )
        let intersections = try context.makeBuffer(
            length: intersectionBytes, label: "TRELLIS flexible dual-grid intersections"
        )
        let split = try context.makeBuffer(
            length: splitBytes, label: "TRELLIS flexible dual-grid split weights"
        )
        var metalCount = UInt32(count)
        try context.runCompute(label: "flexible dual-grid head") { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(rawHead, offset: 0, index: 0)
            encoder.setBuffer(dual, offset: 0, index: 1)
            encoder.setBuffer(intersections, offset: 0, index: 2)
            encoder.setBuffer(split, offset: 0, index: 3)
            encoder.setBytes(&metalCount, length: MemoryLayout<UInt32>.stride, index: 4)
            let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
            )
        }
        return FlexibleDualGridHeadResult(
            dualOffsets: dual, intersections: intersections,
            splitWeights: split, count: count
        )
    }
}

private func flexibleDualGridHeadBytes(
    _ count: Int, _ channels: Int, _ width: Int
) throws -> Int {
    let elements = count.multipliedReportingOverflow(by: channels)
    let bytes = elements.partialValue.multipliedReportingOverflow(by: width)
    guard count > 0, channels > 0, width > 0, !elements.overflow, !bytes.overflow else {
        throw NativeRuntimeError.invalidArgument("flexible dual-grid size overflows Int")
    }
    return bytes.partialValue
}
