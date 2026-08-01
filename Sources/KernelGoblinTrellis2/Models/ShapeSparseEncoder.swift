import Metal

public struct ShapeSparseEncoderResult: @unchecked Sendable {
    public let latent: MTLBuffer
    public let coordinates: [SparseStructureCoordinate]
    public let spatialShape: SparseSpatialShape
    /// Coarsest-to-finest guides, ready for the guided texture decoder.
    public let subdivisionGuides: [SparseSubdivision2x]
}

public final class ShapeSparseEncoder: @unchecked Sendable {
    public static let stageChannels = [64, 128, 256, 512, 1024]
    public static let stageBlocks = [0, 4, 8, 16, 4]

    private let context: MetalContext
    private let dense: DenseKernel
    private let normalization: NormalizationKernel
    private let sparseStructure: SparseStructureKernel
    private let convNeXt: SparseConvNeXtBlock
    private let s2c: SparseS2CBlock
    private let spatialToChannel: SparseSpatialToChannelKernel

    public init(context: MetalContext) throws {
        self.context = context
        self.dense = try DenseKernel(context: context)
        self.normalization = try NormalizationKernel(context: context)
        self.sparseStructure = try SparseStructureKernel(context: context)
        self.convNeXt = try SparseConvNeXtBlock(context: context)
        self.s2c = try SparseS2CBlock(context: context)
        self.spatialToChannel = try SparseSpatialToChannelKernel(context: context)
    }

    public func callAsFunction(
        input: MTLBuffer,
        coordinates: [SparseStructureCoordinate],
        spatialShape: SparseSpatialShape,
        checkpoint: MappedCheckpoint
    ) throws -> ShapeSparseEncoderResult {
        let tokenCount = coordinates.count
        let inputElements = try sparseS2CProduct(tokenCount, 6)
        let inputBytes = try sparseS2CProduct(
            inputElements, MemoryLayout<Float>.stride
        )
        guard tokenCount > 0, input.length >= inputBytes else {
            throw NativeRuntimeError.invalidArgument("invalid shape encoder input")
        }
        let inputWeight = try shapeEncoderTensor(
            checkpoint, "input_layer.weight", [64, 6]
        )
        let inputBias = try shapeEncoderTensor(
            checkpoint, "input_layer.bias", [64]
        )
        let latentWeight = try shapeEncoderTensor(
            checkpoint, "to_latent.weight", [64, 1024]
        )
        let latentBias = try shapeEncoderTensor(
            checkpoint, "to_latent.bias", [64]
        )

        return try autoreleasepool {
            let checkpointBuffer = try checkpoint.acquireBuffer()
            let initialElements = try sparseS2CProduct(tokenCount, 64)
            let initialBytes = try sparseS2CProduct(
                initialElements, MemoryLayout<Float>.stride
            )
            var current = try context.makeBuffer(
                length: initialBytes, label: "TRELLIS shape encoder input projection"
            )
            try dense.linearF16WeightsF32Output(
                input: input, checkpoint: checkpointBuffer,
                weightOffset: Int(inputWeight.fileOffset),
                biasOffset: Int(inputBias.fileOffset), rows: tokenCount,
                inputChannels: 6, outputChannels: 64, output: current
            )
            // The encoder torso begins at the explicit upstream FP16 boundary.
            try sparseStructure.roundF16F32(
                input: current, count: initialElements, output: current
            )
            var currentCoordinates = coordinates
            var currentShape = spatialShape
            var neighborhood = try SparseNeighborhood3x3(
                coordinates: currentCoordinates, spatialShape: currentShape
            )
            guard var neighborBuffer = try neighborhood.makeMetalBuffer(context: context)
            else {
                throw NativeRuntimeError.invalidArgument(
                    "shape encoder unexpectedly has no active coordinates"
                )
            }
            var downsampleGuides: [SparseSubdivision2x] = []
            downsampleGuides.reserveCapacity(4)

            for stage in Self.stageChannels.indices {
                let channels = Self.stageChannels[stage]
                for blockIndex in 0..<Self.stageBlocks[stage] {
                    let layout = try SparseConvNeXtOffsets(
                        checkpoint: checkpoint, stage: stage,
                        block: blockIndex, channels: channels
                    )
                    let elements = try sparseS2CProduct(
                        currentCoordinates.count, channels
                    )
                    let bytes = try sparseS2CProduct(
                        elements, MemoryLayout<Float>.stride
                    )
                    let expandedBytes = try sparseS2CProduct(bytes, 4)
                    let convolutionOutput = try context.makeBuffer(
                        length: bytes, label: "TRELLIS encoder ConvNeXt convolution"
                    )
                    let normalized = try context.makeBuffer(
                        length: bytes, label: "TRELLIS encoder ConvNeXt normalization"
                    )
                    let expanded = try context.makeBuffer(
                        length: expandedBytes, label: "TRELLIS encoder ConvNeXt expansion"
                    )
                    let branch = try context.makeBuffer(
                        length: bytes, label: "TRELLIS encoder ConvNeXt branch"
                    )
                    let output = try context.makeBuffer(
                        length: bytes, label: "TRELLIS encoder ConvNeXt output"
                    )
                    try convNeXt(
                        input: current, neighbors: neighborBuffer,
                        checkpoint: checkpointBuffer, offsets: layout,
                        tokenCount: currentCoordinates.count, channels: channels,
                        convolutionOutput: convolutionOutput, normalized: normalized,
                        expanded: expanded, branch: branch, output: output
                    )
                    current = output
                }
                guard stage < Self.stageChannels.count - 1 else { continue }
                let outputChannels = Self.stageChannels[stage + 1]
                let layout = try SparseS2CCheckpointLayout(
                    checkpoint: checkpoint, stage: stage,
                    block: Self.stageBlocks[stage], inputChannels: channels,
                    outputChannels: outputChannels
                )
                let result = try s2c(
                    input: current, coordinates: currentCoordinates,
                    spatialShape: currentShape, neighborBuffer: neighborBuffer,
                    checkpoint: checkpointBuffer, layout: layout,
                    inputChannels: channels, outputChannels: outputChannels
                )
                current = result.features
                currentCoordinates = result.coordinates
                currentShape = result.neighborhood.spatialShape
                neighborhood = result.neighborhood
                neighborBuffer = result.neighborBuffer
                downsampleGuides.append(result.subdivision)
            }

            let finalCount = currentCoordinates.count
            let normalizedElements = try sparseS2CProduct(finalCount, 1024)
            let normalizedBytes = try sparseS2CProduct(
                normalizedElements, MemoryLayout<Float>.stride
            )
            let normalized = try context.makeBuffer(
                length: normalizedBytes,
                label: "TRELLIS shape encoder final normalization"
            )
            // This normalization is outside the FP16 torso and is non-affine.
            try normalization.layerNormF32WeightsF16(
                input: current, checkpoint: checkpointBuffer,
                rows: finalCount, channels: 1024, epsilon: 1e-5,
                output: normalized
            )
            let posteriorElements = try sparseS2CProduct(finalCount, 64)
            let posteriorBytes = try sparseS2CProduct(
                posteriorElements, MemoryLayout<Float>.stride
            )
            let posterior = try context.makeBuffer(
                length: posteriorBytes, label: "TRELLIS shape encoder posterior"
            )
            try dense.linearF16WeightsF32Output(
                input: normalized, checkpoint: checkpointBuffer,
                weightOffset: Int(latentWeight.fileOffset),
                biasOffset: Int(latentBias.fileOffset), rows: finalCount,
                inputChannels: 1024, outputChannels: 64, output: posterior
            )
            let latentElements = try sparseS2CProduct(finalCount, 32)
            let latentBytes = try sparseS2CProduct(
                latentElements, MemoryLayout<Float>.stride
            )
            let latent = try context.makeBuffer(
                length: latentBytes, label: "TRELLIS shape encoder posterior mean"
            )
            try spatialToChannel.selectPosteriorMean(
                input: posterior, rows: finalCount, output: latent
            )
            return ShapeSparseEncoderResult(
                latent: latent, coordinates: currentCoordinates,
                spatialShape: currentShape,
                subdivisionGuides: Array(downsampleGuides.reversed())
            )
        }
    }
}

private func shapeEncoderTensor(
    _ checkpoint: MappedCheckpoint, _ name: String, _ shape: [UInt64]
) throws -> TensorDescriptor {
    let descriptor = try checkpoint.descriptor(named: name)
    guard descriptor.dtype == .f16, descriptor.shape == shape else {
        throw NativeRuntimeError.invalidArgument(
            "\(name) does not match shape encoder contract"
        )
    }
    return descriptor
}
