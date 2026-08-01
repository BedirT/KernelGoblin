import Metal

public struct ShapeSparseDecoderResult: @unchecked Sendable {
    public let rawHead: MTLBuffer
    public let coordinates: [SparseStructureCoordinate]
    public let spatialShape: SparseSpatialShape
    public let subdivisionGuides: [SparseSubdivision2x]
    public let subdivisionLogits: [MTLBuffer]
}

public final class ShapeSparseDecoder: @unchecked Sendable {
    public static let stageChannels = [1024, 512, 256, 128, 64]
    public static let stageBlocks = [4, 16, 8, 4, 0]

    private let context: MetalContext
    private let dense: DenseKernel
    private let normalization: NormalizationKernel
    private let sparseStructure: SparseStructureKernel
    private let convNeXt: SparseConvNeXtBlock
    private let c2s: SparseC2SBlock

    public init(context: MetalContext) throws {
        self.context = context
        self.dense = try DenseKernel(context: context)
        self.normalization = try NormalizationKernel(context: context)
        self.sparseStructure = try SparseStructureKernel(context: context)
        self.convNeXt = try SparseConvNeXtBlock(context: context)
        self.c2s = try SparseC2SBlock(context: context)
    }

    public func callAsFunction(
        latent: MTLBuffer,
        coordinates: [SparseStructureCoordinate],
        spatialShape: SparseSpatialShape,
        checkpoint: MappedCheckpoint
    ) throws -> ShapeSparseDecoderResult {
        let tokenCount = coordinates.count
        let latentElements = try shapeDecoderProduct(tokenCount, 32)
        let latentBytes = try shapeDecoderProduct(
            latentElements, MemoryLayout<Float>.stride
        )
        guard tokenCount > 0, latent.length >= latentBytes else {
            throw NativeRuntimeError.invalidArgument("invalid shape decoder latent")
        }
        let fromWeight = try shapeDecoderTensor(
            checkpoint, "from_latent.weight", .f16, [1024, 32]
        )
        let fromBias = try shapeDecoderTensor(
            checkpoint, "from_latent.bias", .f16, [1024]
        )
        let outputWeight = try shapeDecoderTensor(
            checkpoint, "output_layer.weight", .f16, [7, 64]
        )
        let outputBias = try shapeDecoderTensor(
            checkpoint, "output_layer.bias", .f16, [7]
        )
        return try autoreleasepool {
            let checkpointBuffer = try checkpoint.acquireBuffer()
            let initialElements = try shapeDecoderProduct(tokenCount, 1024)
            let initialBytes = try shapeDecoderProduct(
                initialElements, MemoryLayout<Float>.stride
            )
            var current = try context.makeBuffer(
                length: initialBytes, label: "TRELLIS shape decoder from-latent"
            )
            try dense.linearF16WeightsF32Output(
                input: latent, checkpoint: checkpointBuffer,
                weightOffset: Int(fromWeight.fileOffset), biasOffset: Int(fromBias.fileOffset),
                rows: tokenCount, inputChannels: 32, outputChannels: 1024,
                output: current
            )
            try sparseStructure.roundF16F32(
                input: current, count: initialElements, output: current
            )
            var currentCoordinates = coordinates
            var currentShape = spatialShape
            var currentNeighborhood = try SparseNeighborhood3x3(
                coordinates: currentCoordinates, spatialShape: currentShape
            )
            var currentNeighborBuffer = try requireShapeDecoderBuffer(
                currentNeighborhood.makeMetalBuffer(context: context)
            )
            var guides: [SparseSubdivision2x] = []
            var guideLogits: [MTLBuffer] = []
            for stage in Self.stageChannels.indices {
                let channels = Self.stageChannels[stage]
                for blockIndex in 0..<Self.stageBlocks[stage] {
                    let layout = try SparseConvNeXtOffsets(
                        checkpoint: checkpoint, stage: stage,
                        block: blockIndex, channels: channels
                    )
                    let elements = try shapeDecoderProduct(
                        currentCoordinates.count, channels
                    )
                    let bytes = try shapeDecoderProduct(
                        elements, MemoryLayout<Float>.stride
                    )
                    let expandedBytes = try shapeDecoderProduct(bytes, 4)
                    let convolutionOutput = try context.makeBuffer(
                        length: bytes, label: "TRELLIS decoder ConvNeXt convolution"
                    )
                    let normalized = try context.makeBuffer(
                        length: bytes, label: "TRELLIS decoder ConvNeXt normalization"
                    )
                    let expanded = try context.makeBuffer(
                        length: expandedBytes, label: "TRELLIS decoder ConvNeXt expansion"
                    )
                    let branch = try context.makeBuffer(
                        length: bytes, label: "TRELLIS decoder ConvNeXt branch"
                    )
                    let output = try context.makeBuffer(
                        length: bytes, label: "TRELLIS decoder ConvNeXt output"
                    )
                    try convNeXt(
                        input: current, neighbors: currentNeighborBuffer,
                        checkpoint: checkpointBuffer, offsets: layout,
                        tokenCount: currentCoordinates.count, channels: channels,
                        convolutionOutput: convolutionOutput, normalized: normalized,
                        expanded: expanded, branch: branch, output: output
                    )
                    current = output
                }
                guard stage < Self.stageChannels.count - 1 else { continue }
                let outputChannels = Self.stageChannels[stage + 1]
                let c2sLayout = try SparseC2SCheckpointLayout(
                    checkpoint: checkpoint, stage: stage,
                    block: Self.stageBlocks[stage],
                    inputChannels: channels, outputChannels: outputChannels
                )
                let result = try c2s(
                    input: current, coordinates: currentCoordinates,
                    spatialShape: currentShape, neighborBuffer: currentNeighborBuffer,
                    checkpoint: checkpointBuffer, layout: c2sLayout,
                    inputChannels: channels, outputChannels: outputChannels
                )
                guard let features = result.features,
                      let neighbors = result.neighborBuffer else {
                    throw NativeRuntimeError.invalidArgument(
                        "shape decoder subdivision removed every active voxel"
                    )
                }
                guides.append(result.subdivision)
                guideLogits.append(result.subdivisionLogits)
                current = features
                currentCoordinates = result.coordinates
                currentShape = result.neighborhood.spatialShape
                currentNeighborhood = result.neighborhood
                currentNeighborBuffer = neighbors
            }
            let finalCount = currentCoordinates.count
            let normalizedElements = try shapeDecoderProduct(finalCount, 64)
            let normalizedBytes = try shapeDecoderProduct(
                normalizedElements, MemoryLayout<Float>.stride
            )
            let normalized = try context.makeBuffer(
                length: normalizedBytes, label: "TRELLIS shape decoder final normalization"
            )
            try normalization.layerNormF32WeightsF16(
                input: current, checkpoint: checkpointBuffer,
                rows: finalCount, channels: 64, epsilon: 1e-5, output: normalized
            )
            let rawElements = try shapeDecoderProduct(finalCount, 7)
            let rawBytes = try shapeDecoderProduct(
                rawElements, MemoryLayout<Float>.stride
            )
            let rawHead = try context.makeBuffer(
                length: rawBytes, label: "TRELLIS shape decoder raw head"
            )
            try dense.linearF16WeightsF32Output(
                input: normalized, checkpoint: checkpointBuffer,
                weightOffset: Int(outputWeight.fileOffset),
                biasOffset: Int(outputBias.fileOffset),
                rows: finalCount, inputChannels: 64, outputChannels: 7,
                output: rawHead
            )
            return ShapeSparseDecoderResult(
                rawHead: rawHead, coordinates: currentCoordinates,
                spatialShape: currentShape, subdivisionGuides: guides,
                subdivisionLogits: guideLogits
            )
        }
    }
}

private func shapeDecoderProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard lhs >= 0, rhs >= 0, !result.overflow else {
        throw NativeRuntimeError.invalidArgument("shape decoder tensor size overflows Int")
    }
    return result.partialValue
}

private func shapeDecoderTensor(
    _ checkpoint: MappedCheckpoint, _ name: String,
    _ dtype: TensorDataType, _ shape: [UInt64]
) throws -> TensorDescriptor {
    let descriptor = try checkpoint.descriptor(named: name)
    guard descriptor.dtype == dtype, descriptor.shape == shape else {
        throw NativeRuntimeError.invalidArgument("\(name) does not match shape decoder contract")
    }
    return descriptor
}

private func requireShapeDecoderBuffer(_ value: MTLBuffer?) throws -> MTLBuffer {
    guard let value else {
        throw NativeRuntimeError.invalidArgument("shape decoder has no active sparse coordinates")
    }
    return value
}
