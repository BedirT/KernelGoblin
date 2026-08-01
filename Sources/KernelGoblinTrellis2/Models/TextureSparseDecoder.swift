import Metal

public struct TextureSparseDecoderResult: @unchecked Sendable {
    public let rawHead: MTLBuffer
    public let pbrFields: MTLBuffer
    public let coordinates: [SparseStructureCoordinate]
    public let spatialShape: SparseSpatialShape
}

public final class TextureSparseDecoder: @unchecked Sendable {
    private let context: MetalContext
    private let dense: DenseKernel
    private let normalization: NormalizationKernel
    private let sparseStructure: SparseStructureKernel
    private let convNeXt: SparseConvNeXtBlock
    private let c2s: SparseC2SBlock
    private let pbr: PBRFieldKernel

    public init(context: MetalContext) throws {
        self.context = context
        self.dense = try DenseKernel(context: context)
        self.normalization = try NormalizationKernel(context: context)
        self.sparseStructure = try SparseStructureKernel(context: context)
        self.convNeXt = try SparseConvNeXtBlock(context: context)
        self.c2s = try SparseC2SBlock(context: context)
        self.pbr = try PBRFieldKernel(context: context)
    }

    public func callAsFunction(
        latent: MTLBuffer,
        coordinates: [SparseStructureCoordinate],
        spatialShape: SparseSpatialShape,
        subdivisionGuides: [SparseSubdivision2x],
        checkpoint: MappedCheckpoint
    ) throws -> TextureSparseDecoderResult {
        let tokenCount = coordinates.count
        let latentElements = try textureDecoderProduct(tokenCount, 32)
        let latentBytes = try textureDecoderProduct(
            latentElements, MemoryLayout<Float>.stride
        )
        guard tokenCount > 0, latent.length >= latentBytes,
              subdivisionGuides.count == ShapeSparseDecoder.stageChannels.count - 1 else {
            throw NativeRuntimeError.invalidArgument("invalid texture decoder input")
        }
        let fromWeight = try textureDecoderTensor(
            checkpoint, "from_latent.weight", .f16, [1024, 32]
        )
        let fromBias = try textureDecoderTensor(
            checkpoint, "from_latent.bias", .f16, [1024]
        )
        let outputWeight = try textureDecoderTensor(
            checkpoint, "output_layer.weight", .f16, [6, 64]
        )
        let outputBias = try textureDecoderTensor(
            checkpoint, "output_layer.bias", .f16, [6]
        )
        return try autoreleasepool {
            let checkpointBuffer = try checkpoint.acquireBuffer()
            let initialElements = try textureDecoderProduct(tokenCount, 1024)
            let initialBytes = try textureDecoderProduct(
                initialElements, MemoryLayout<Float>.stride
            )
            var current = try context.makeBuffer(
                length: initialBytes, label: "TRELLIS texture decoder from-latent"
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
            var currentNeighborBuffer = try requireTextureDecoderBuffer(
                currentNeighborhood.makeMetalBuffer(context: context)
            )
            for stage in ShapeSparseDecoder.stageChannels.indices {
                let channels = ShapeSparseDecoder.stageChannels[stage]
                for blockIndex in 0..<ShapeSparseDecoder.stageBlocks[stage] {
                    let layout = try SparseConvNeXtOffsets(
                        checkpoint: checkpoint, stage: stage,
                        block: blockIndex, channels: channels
                    )
                    let elements = try textureDecoderProduct(
                        currentCoordinates.count, channels
                    )
                    let bytes = try textureDecoderProduct(
                        elements, MemoryLayout<Float>.stride
                    )
                    let expandedBytes = try textureDecoderProduct(bytes, 4)
                    let convolutionOutput = try context.makeBuffer(
                        length: bytes, label: "TRELLIS texture ConvNeXt convolution"
                    )
                    let normalized = try context.makeBuffer(
                        length: bytes, label: "TRELLIS texture ConvNeXt normalization"
                    )
                    let expanded = try context.makeBuffer(
                        length: expandedBytes, label: "TRELLIS texture ConvNeXt expansion"
                    )
                    let branch = try context.makeBuffer(
                        length: bytes, label: "TRELLIS texture ConvNeXt branch"
                    )
                    let output = try context.makeBuffer(
                        length: bytes, label: "TRELLIS texture ConvNeXt output"
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
                guard stage < ShapeSparseDecoder.stageChannels.count - 1 else { continue }
                let outputChannels = ShapeSparseDecoder.stageChannels[stage + 1]
                let c2sLayout = try SparseC2SCheckpointLayout(
                    checkpoint: checkpoint, stage: stage,
                    block: ShapeSparseDecoder.stageBlocks[stage],
                    inputChannels: channels, outputChannels: outputChannels,
                    predictsSubdivision: false
                )
                let result = try c2s(
                    input: current, coordinates: currentCoordinates,
                    spatialShape: currentShape, neighborBuffer: currentNeighborBuffer,
                    checkpoint: checkpointBuffer, layout: c2sLayout,
                    inputChannels: channels, outputChannels: outputChannels,
                    guide: subdivisionGuides[stage]
                )
                guard let features = result.features,
                      let neighbors = result.neighborBuffer else {
                    throw NativeRuntimeError.invalidArgument(
                        "texture decoder guide removed every active voxel"
                    )
                }
                current = features
                currentCoordinates = result.coordinates
                currentShape = result.neighborhood.spatialShape
                currentNeighborhood = result.neighborhood
                currentNeighborBuffer = neighbors
            }
            let finalCount = currentCoordinates.count
            let normalizedElements = try textureDecoderProduct(finalCount, 64)
            let normalizedBytes = try textureDecoderProduct(
                normalizedElements, MemoryLayout<Float>.stride
            )
            let normalized = try context.makeBuffer(
                length: normalizedBytes, label: "TRELLIS texture decoder final normalization"
            )
            try normalization.layerNormF32WeightsF16(
                input: current, checkpoint: checkpointBuffer,
                rows: finalCount, channels: 64, epsilon: 1e-5, output: normalized
            )
            let rawElements = try textureDecoderProduct(finalCount, 6)
            let rawBytes = try textureDecoderProduct(
                rawElements, MemoryLayout<Float>.stride
            )
            let rawHead = try context.makeBuffer(
                length: rawBytes, label: "TRELLIS texture decoder raw head"
            )
            try dense.linearF16WeightsF32Output(
                input: normalized, checkpoint: checkpointBuffer,
                weightOffset: Int(outputWeight.fileOffset),
                biasOffset: Int(outputBias.fileOffset),
                rows: finalCount, inputChannels: 64, outputChannels: 6,
                output: rawHead
            )
            let pbrFields = try context.makeBuffer(
                length: rawBytes, label: "TRELLIS decoded PBR fields"
            )
            try pbr(raw: rawHead, count: rawElements, output: pbrFields)
            return TextureSparseDecoderResult(
                rawHead: rawHead, pbrFields: pbrFields,
                coordinates: currentCoordinates,
                spatialShape: currentShape
            )
        }
    }
}

private func textureDecoderTensor(
    _ checkpoint: MappedCheckpoint, _ name: String,
    _ dtype: TensorDataType, _ shape: [UInt64]
) throws -> TensorDescriptor {
    let descriptor = try checkpoint.descriptor(named: name)
    guard descriptor.dtype == dtype, descriptor.shape == shape else {
        throw NativeRuntimeError.invalidArgument("\(name) does not match texture decoder contract")
    }
    return descriptor
}

private func requireTextureDecoderBuffer(_ value: MTLBuffer?) throws -> MTLBuffer {
    guard let value else {
        throw NativeRuntimeError.invalidArgument("texture decoder has no active sparse coordinates")
    }
    return value
}

private func textureDecoderProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard lhs >= 0, rhs >= 0, !result.overflow else {
        throw NativeRuntimeError.invalidArgument("texture decoder tensor size overflows Int")
    }
    return result.partialValue
}
