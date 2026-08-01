import Metal

public struct SparseS2CCheckpointLayout: Sendable {
    let normWeight: Int
    let normBias: Int
    let convolutionDownWeight: Int
    let convolutionDownBias: Int
    let convolutionOutputWeight: Int
    let convolutionOutputBias: Int

    init(
        normWeight: Int, normBias: Int,
        convolutionDownWeight: Int, convolutionDownBias: Int,
        convolutionOutputWeight: Int, convolutionOutputBias: Int
    ) {
        self.normWeight = normWeight
        self.normBias = normBias
        self.convolutionDownWeight = convolutionDownWeight
        self.convolutionDownBias = convolutionDownBias
        self.convolutionOutputWeight = convolutionOutputWeight
        self.convolutionOutputBias = convolutionOutputBias
    }

    public init(
        checkpoint: MappedCheckpoint, stage: Int, block: Int,
        inputChannels: Int, outputChannels: Int
    ) throws {
        guard stage >= 0, block >= 0, inputChannels > 0,
              outputChannels > 0, outputChannels.isMultiple(of: 8) else {
            throw NativeRuntimeError.invalidArgument("invalid S2C checkpoint location")
        }
        let prefix = "blocks.\(stage).\(block)."
        func tensor(_ name: String, _ shape: [UInt64]) throws -> TensorDescriptor {
            let value = try checkpoint.descriptor(named: prefix + name)
            guard value.dtype == .f16, value.shape == shape else {
                throw NativeRuntimeError.invalidArgument(
                    "\(prefix + name) does not match its F16 S2C tensor contract"
                )
            }
            return value
        }
        let ci = UInt64(inputChannels), co = UInt64(outputChannels)
        let normWeight = try tensor("norm1.weight", [ci])
        let normBias = try tensor("norm1.bias", [ci])
        let downWeight = try tensor("conv1.weight", [co / 8, 3, 3, 3, ci])
        let downBias = try tensor("conv1.bias", [co / 8])
        let outputWeight = try tensor("conv2.weight", [co, 3, 3, 3, co])
        let outputBias = try tensor("conv2.bias", [co])
        self.init(
            normWeight: Int(normWeight.fileOffset), normBias: Int(normBias.fileOffset),
            convolutionDownWeight: Int(downWeight.fileOffset),
            convolutionDownBias: Int(downBias.fileOffset),
            convolutionOutputWeight: Int(outputWeight.fileOffset),
            convolutionOutputBias: Int(outputBias.fileOffset)
        )
    }
}

public struct SparseS2CResult: @unchecked Sendable {
    public let features: MTLBuffer
    public let coordinates: [SparseStructureCoordinate]
    public let neighborhood: SparseNeighborhood3x3
    public let neighborBuffer: MTLBuffer
    public let subdivision: SparseSubdivision2x
}

public final class SparseS2CBlock: @unchecked Sendable {
    private let context: MetalContext
    private let normalization: NormalizationKernel
    private let primitive: PrimitiveKernel
    private let sparseStructure: SparseStructureKernel
    private let convolution: SubmanifoldConvolutionKernel
    private let spatialToChannel: SparseSpatialToChannelKernel

    public init(context: MetalContext) throws {
        self.context = context
        self.normalization = try NormalizationKernel(context: context)
        self.primitive = try PrimitiveKernel(context: context)
        self.sparseStructure = try SparseStructureKernel(context: context)
        self.convolution = try SubmanifoldConvolutionKernel(context: context)
        self.spatialToChannel = try SparseSpatialToChannelKernel(context: context)
    }

    public func callAsFunction(
        input: MTLBuffer,
        coordinates: [SparseStructureCoordinate],
        spatialShape: SparseSpatialShape,
        neighborBuffer: MTLBuffer,
        checkpoint: MTLBuffer,
        layout: SparseS2CCheckpointLayout,
        inputChannels: Int,
        outputChannels: Int
    ) throws -> SparseS2CResult {
        let fineCount = coordinates.count
        let inputElements = try sparseS2CProduct(fineCount, inputChannels)
        let inputBytes = try sparseS2CProduct(
            inputElements, MemoryLayout<Float>.stride
        )
        guard fineCount > 0, inputChannels > 0, outputChannels > 0,
              outputChannels.isMultiple(of: 8),
              try sparseS2CProduct(inputChannels, 8) >= outputChannels,
              try sparseS2CProduct(inputChannels, 8) % outputChannels == 0,
              input.length >= inputBytes else {
            throw NativeRuntimeError.invalidArgument("invalid S2C sparse input")
        }
        let topology = try SparseSpatial2Channel2x(
            coordinates: coordinates, spatialShape: spatialShape
        )
        let coarseCount = topology.coordinates.count
        let coarseElements = try sparseS2CProduct(coarseCount, outputChannels)
        let coarseBytes = try sparseS2CProduct(
            coarseElements, MemoryLayout<Float>.stride
        )
        let fineDownChannels = outputChannels / 8
        let fineDownElements = try sparseS2CProduct(fineCount, fineDownChannels)
        let fineDownBytes = try sparseS2CProduct(
            fineDownElements, MemoryLayout<Float>.stride
        )
        let sourceMap = try topology.makeSourceIndexBuffer(context: context)

        var normalized: MTLBuffer? = try context.makeBuffer(
            length: inputBytes, label: "TRELLIS S2C input normalization"
        )
        try normalization.layerNormF32WeightsF16(
            input: input, checkpoint: checkpoint, rows: fineCount,
            channels: inputChannels, weightOffset: layout.normWeight,
            biasOffset: layout.normBias, epsilon: 1e-6, output: normalized!
        )
        try sparseStructure.roundF16F32(
            input: normalized!, count: inputElements, output: normalized!
        )
        try primitive.siluF32(
            input: normalized!, count: inputElements, output: normalized!
        )
        try sparseStructure.roundF16F32(
            input: normalized!, count: inputElements, output: normalized!
        )
        var convolvedDown: MTLBuffer? = try context.makeBuffer(
            length: fineDownBytes, label: "TRELLIS S2C fine convolution"
        )
        try convolution.convolveF16WeightsF32Output(
            input: normalized!, neighbors: neighborBuffer, checkpoint: checkpoint,
            weightOffset: layout.convolutionDownWeight,
            biasOffset: layout.convolutionDownBias,
            tokenCount: fineCount, inputChannels: inputChannels,
            outputChannels: fineDownChannels, output: convolvedDown!
        )
        normalized = nil
        try sparseStructure.roundF16F32(
            input: convolvedDown!, count: fineDownElements, output: convolvedDown!
        )
        var packedBranch: MTLBuffer? = try context.makeBuffer(
            length: coarseBytes, label: "TRELLIS S2C packed convolution"
        )
        try spatialToChannel.pack(
            input: convolvedDown!, sourceIndices: sourceMap,
            coarseCount: coarseCount, channels: fineDownChannels,
            output: packedBranch!
        )
        convolvedDown = nil
        var activated: MTLBuffer? = try context.makeBuffer(
            length: coarseBytes, label: "TRELLIS S2C output normalization"
        )
        try normalization.layerNormF32WeightsF16(
            input: packedBranch!, checkpoint: checkpoint,
            rows: coarseCount, channels: outputChannels,
            epsilon: 1e-6, output: activated!
        )
        packedBranch = nil
        try sparseStructure.roundF16F32(
            input: activated!, count: coarseElements, output: activated!
        )
        try primitive.siluF32(
            input: activated!, count: coarseElements, output: activated!
        )
        try sparseStructure.roundF16F32(
            input: activated!, count: coarseElements, output: activated!
        )
        let neighborhood = try SparseNeighborhood3x3(
            coordinates: topology.coordinates, spatialShape: topology.spatialShape
        )
        guard let coarseNeighbors = try neighborhood.makeMetalBuffer(context: context)
        else {
            throw NativeRuntimeError.invalidArgument(
                "spatial-to-channel unexpectedly produced no coordinates"
            )
        }
        let branch = try context.makeBuffer(
            length: coarseBytes, label: "TRELLIS S2C output convolution"
        )
        try convolution.convolveF16WeightsF32Output(
            input: activated!, neighbors: coarseNeighbors, checkpoint: checkpoint,
            weightOffset: layout.convolutionOutputWeight,
            biasOffset: layout.convolutionOutputBias,
            tokenCount: coarseCount, inputChannels: outputChannels,
            outputChannels: outputChannels, output: branch
        )
        activated = nil
        let skip = try context.makeBuffer(
            length: coarseBytes, label: "TRELLIS S2C skip reduction"
        )
        try spatialToChannel.skipMean(
            input: input, sourceIndices: sourceMap, coarseCount: coarseCount,
            inputChannels: inputChannels, outputChannels: outputChannels,
            output: skip
        )
        let output = try context.makeBuffer(
            length: coarseBytes, label: "TRELLIS S2C output"
        )
        try primitive.addF32(
            lhs: branch, rhs: skip, count: coarseElements, output: output
        )
        try sparseStructure.roundF16F32(
            input: output, count: coarseElements, output: output
        )
        return SparseS2CResult(
            features: output, coordinates: topology.coordinates,
            neighborhood: neighborhood, neighborBuffer: coarseNeighbors,
            subdivision: topology.subdivision
        )
    }
}
