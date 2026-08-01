import Metal

public struct SparseC2SCheckpointLayout: Sendable {
    let subdivisionWeight: Int?
    let subdivisionBias: Int?
    let normWeight: Int
    let normBias: Int
    let convolutionUpWeight: Int
    let convolutionUpBias: Int
    let convolutionOutputWeight: Int
    let convolutionOutputBias: Int

    init(
        subdivisionWeight: Int?, subdivisionBias: Int?,
        normWeight: Int, normBias: Int,
        convolutionUpWeight: Int, convolutionUpBias: Int,
        convolutionOutputWeight: Int, convolutionOutputBias: Int
    ) {
        self.subdivisionWeight = subdivisionWeight
        self.subdivisionBias = subdivisionBias
        self.normWeight = normWeight
        self.normBias = normBias
        self.convolutionUpWeight = convolutionUpWeight
        self.convolutionUpBias = convolutionUpBias
        self.convolutionOutputWeight = convolutionOutputWeight
        self.convolutionOutputBias = convolutionOutputBias
    }

    public init(
        checkpoint: MappedCheckpoint, stage: Int, block: Int,
        inputChannels: Int, outputChannels: Int,
        predictsSubdivision: Bool = true
    ) throws {
        guard stage >= 0, block >= 0, inputChannels > 0, outputChannels > 0 else {
            throw NativeRuntimeError.invalidArgument("invalid C2S checkpoint location")
        }
        let prefix = "blocks.\(stage).\(block)."
        func tensor(_ name: String, _ shape: [UInt64]) throws -> TensorDescriptor {
            let value = try checkpoint.descriptor(named: prefix + name)
            guard value.dtype == .f16, value.shape == shape else {
                throw NativeRuntimeError.invalidArgument(
                    "\(prefix + name) does not match its F16 C2S tensor contract"
                )
            }
            return value
        }
        guard outputChannels <= Int(UInt64.max / 8) else {
            throw NativeRuntimeError.invalidArgument("C2S channel count overflows")
        }
        let ci = UInt64(inputChannels), co = UInt64(outputChannels)
        let subdivisionWeight = try predictsSubdivision
            ? tensor("to_subdiv.weight", [8, ci]) : nil
        let subdivisionBias = try predictsSubdivision
            ? tensor("to_subdiv.bias", [8]) : nil
        let normWeight = try tensor("norm1.weight", [ci])
        let normBias = try tensor("norm1.bias", [ci])
        let convolutionUpWeight = try tensor("conv1.weight", [co * 8, 3, 3, 3, ci])
        let convolutionUpBias = try tensor("conv1.bias", [co * 8])
        let convolutionOutputWeight = try tensor("conv2.weight", [co, 3, 3, 3, co])
        let convolutionOutputBias = try tensor("conv2.bias", [co])
        self.init(
            subdivisionWeight: subdivisionWeight.map { Int($0.fileOffset) },
            subdivisionBias: subdivisionBias.map { Int($0.fileOffset) },
            normWeight: Int(normWeight.fileOffset), normBias: Int(normBias.fileOffset),
            convolutionUpWeight: Int(convolutionUpWeight.fileOffset),
            convolutionUpBias: Int(convolutionUpBias.fileOffset),
            convolutionOutputWeight: Int(convolutionOutputWeight.fileOffset),
            convolutionOutputBias: Int(convolutionOutputBias.fileOffset)
        )
    }
}

public struct SparseC2SResult: @unchecked Sendable {
    public let features: MTLBuffer?
    public let coordinates: [SparseStructureCoordinate]
    public let neighborhood: SparseNeighborhood3x3
    public let neighborBuffer: MTLBuffer?
    public let subdivisionLogits: MTLBuffer?
    public let subdivision: SparseSubdivision2x
}

public final class SparseC2SBlock: @unchecked Sendable {
    private let context: MetalContext
    private let dense: DenseKernel
    private let normalization: NormalizationKernel
    private let primitive: PrimitiveKernel
    private let sparseStructure: SparseStructureKernel
    private let convolution: SubmanifoldConvolutionKernel
    private let subdivisionKernel: SparseSubdivisionKernel

    public init(context: MetalContext) throws {
        self.context = context
        self.dense = try DenseKernel(context: context)
        self.normalization = try NormalizationKernel(context: context)
        self.primitive = try PrimitiveKernel(context: context)
        self.sparseStructure = try SparseStructureKernel(context: context)
        self.convolution = try SubmanifoldConvolutionKernel(context: context)
        self.subdivisionKernel = try SparseSubdivisionKernel(context: context)
    }

    public func callAsFunction(
        input: MTLBuffer,
        coordinates: [SparseStructureCoordinate],
        spatialShape: SparseSpatialShape,
        neighborBuffer: MTLBuffer,
        checkpoint: MTLBuffer,
        layout: SparseC2SCheckpointLayout,
        inputChannels: Int,
        outputChannels: Int,
        guide: SparseSubdivision2x? = nil
    ) throws -> SparseC2SResult {
        let tokenCount = coordinates.count
        let parentElements = try sparseC2SProduct(tokenCount, inputChannels)
        let parentBytes = try sparseC2SProduct(parentElements, MemoryLayout<Float>.stride)
        guard tokenCount > 0, inputChannels > 0, inputChannels.isMultiple(of: 8),
              outputChannels > 0, outputChannels.isMultiple(of: inputChannels / 8),
              input.length >= parentBytes else {
            throw NativeRuntimeError.invalidArgument("invalid C2S sparse input")
        }
        let subdivisionLogits: MTLBuffer?
        let subdivision: SparseSubdivision2x
        if let subdivisionWeight = layout.subdivisionWeight,
           let subdivisionBias = layout.subdivisionBias {
            guard guide == nil else {
                throw NativeRuntimeError.invalidArgument(
                    "predicted C2S subdivision cannot also accept a guide"
                )
            }
            let logitElements = try sparseC2SProduct(tokenCount, 8)
            let logitBytes = try sparseC2SProduct(
                logitElements, MemoryLayout<Float>.stride
            )
            let logits = try context.makeBuffer(
                length: logitBytes, label: "TRELLIS subdivision logits"
            )
            try dense.linearF16WeightsF32Output(
                input: input, checkpoint: checkpoint,
                weightOffset: subdivisionWeight, biasOffset: subdivisionBias,
                rows: tokenCount, inputChannels: inputChannels, outputChannels: 8,
                output: logits
            )
            try sparseStructure.roundF16F32(
                input: logits, count: logitElements, output: logits
            )
            subdivision = try SparseSubdivision2x(
                parentCoordinates: coordinates, logits: logits
            )
            subdivisionLogits = logits
        } else {
            guard layout.subdivisionWeight == nil,
                  layout.subdivisionBias == nil,
                  let guide else {
                throw NativeRuntimeError.invalidArgument(
                    "guided C2S subdivision requires an explicit guide"
                )
            }
            try guide.validate(parentCoordinates: coordinates)
            subdivision = guide
            subdivisionLogits = nil
        }
        let childShape = try SparseSpatialShape(
            width: sparseC2SSpatialDouble(spatialShape.width),
            height: sparseC2SSpatialDouble(spatialShape.height),
            depth: sparseC2SSpatialDouble(spatialShape.depth)
        )
        let childNeighborhood = try SparseNeighborhood3x3(
            coordinates: subdivision.coordinates, spatialShape: childShape
        )
        guard !subdivision.coordinates.isEmpty else {
            return SparseC2SResult(
                features: nil, coordinates: [], neighborhood: childNeighborhood,
                neighborBuffer: nil, subdivisionLogits: subdivisionLogits,
                subdivision: subdivision
            )
        }
        let maps = try requireSubdivisionMaps(
            subdivision.makeMetalBuffers(context: context)
        )
        let childCount = subdivision.coordinates.count
        var normalized: MTLBuffer? = try context.makeBuffer(
            length: parentBytes, label: "TRELLIS C2S normalized"
        )
        try normalization.layerNormF32WeightsF16(
            input: input, checkpoint: checkpoint, rows: tokenCount,
            channels: inputChannels, weightOffset: layout.normWeight,
            biasOffset: layout.normBias, epsilon: 1e-6, output: normalized!
        )
        try sparseStructure.roundF16F32(
            input: normalized!, count: parentElements, output: normalized!
        )
        try primitive.siluF32(input: normalized!, count: parentElements, output: normalized!)
        try sparseStructure.roundF16F32(
            input: normalized!, count: parentElements, output: normalized!
        )
        let expandedChannels = try sparseC2SProduct(outputChannels, 8)
        let expandedElements = try sparseC2SProduct(tokenCount, expandedChannels)
        let expandedBytes = try sparseC2SProduct(
            expandedElements, MemoryLayout<Float>.stride
        )
        var convolvedUp: MTLBuffer? = try context.makeBuffer(
            length: expandedBytes,
            label: "TRELLIS C2S parent convolution"
        )
        try convolution.convolveF16WeightsF32Output(
            input: normalized!, neighbors: neighborBuffer, checkpoint: checkpoint,
            weightOffset: layout.convolutionUpWeight,
            biasOffset: layout.convolutionUpBias, tokenCount: tokenCount,
            inputChannels: inputChannels, outputChannels: expandedChannels,
            output: convolvedUp!
        )
        normalized = nil
        let childElements = try sparseC2SProduct(childCount, outputChannels)
        let childBytes = try sparseC2SProduct(childElements, MemoryLayout<Float>.stride)
        var upsampled: MTLBuffer? = try context.makeBuffer(
            length: childBytes, label: "TRELLIS C2S selected convolution"
        )
        try subdivisionKernel.selectConvolutionFeatures(
            input: convolvedUp!, parentIndices: maps.0, childIndices: maps.1,
            childCount: childCount, outputChannels: outputChannels, output: upsampled!
        )
        convolvedUp = nil
        var activated: MTLBuffer? = try context.makeBuffer(
            length: childBytes, label: "TRELLIS C2S output normalization"
        )
        try normalization.layerNormF32WeightsF16(
            input: upsampled!, checkpoint: checkpoint, rows: childCount,
            channels: outputChannels, epsilon: 1e-6, output: activated!
        )
        upsampled = nil
        try sparseStructure.roundF16F32(
            input: activated!, count: childElements, output: activated!
        )
        try primitive.siluF32(input: activated!, count: childElements, output: activated!)
        try sparseStructure.roundF16F32(
            input: activated!, count: childElements, output: activated!
        )
        let childNeighborBuffer = try requireSubdivisionBuffer(
            childNeighborhood.makeMetalBuffer(context: context)
        )
        let branch = try context.makeBuffer(
            length: childBytes, label: "TRELLIS C2S output convolution"
        )
        try convolution.convolveF16WeightsF32Output(
            input: activated!, neighbors: childNeighborBuffer, checkpoint: checkpoint,
            weightOffset: layout.convolutionOutputWeight,
            biasOffset: layout.convolutionOutputBias, tokenCount: childCount,
            inputChannels: outputChannels, outputChannels: outputChannels,
            output: branch
        )
        activated = nil
        let skip = try context.makeBuffer(
            length: childBytes, label: "TRELLIS C2S skip"
        )
        try subdivisionKernel.selectSkipFeatures(
            input: input, parentIndices: maps.0, childIndices: maps.1,
            childCount: childCount, inputChannels: inputChannels,
            outputChannels: outputChannels, output: skip
        )
        let output = try context.makeBuffer(
            length: childBytes, label: "TRELLIS C2S output"
        )
        try primitive.addF32(lhs: branch, rhs: skip, count: childElements, output: output)
        try sparseStructure.roundF16F32(input: output, count: childElements, output: output)
        return SparseC2SResult(
            features: output, coordinates: subdivision.coordinates,
            neighborhood: childNeighborhood, neighborBuffer: childNeighborBuffer,
            subdivisionLogits: subdivisionLogits, subdivision: subdivision
        )
    }
}

private func sparseC2SProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard lhs >= 0, rhs >= 0, !result.overflow else {
        throw NativeRuntimeError.invalidArgument("C2S tensor size overflows Int")
    }
    return result.partialValue
}

private func sparseC2SSpatialDouble(_ value: Int) throws -> Int {
    let result = value.multipliedReportingOverflow(by: 2)
    guard value > 0, !result.overflow, result.partialValue <= Int(Int32.max) else {
        throw NativeRuntimeError.invalidArgument("C2S spatial shape overflows")
    }
    return result.partialValue
}

private func requireSubdivisionMaps(
    _ value: (MTLBuffer, MTLBuffer)?
) throws -> (MTLBuffer, MTLBuffer) {
    guard let value else {
        throw NativeRuntimeError.invalidArgument("subdivision unexpectedly produced no maps")
    }
    return value
}

private func requireSubdivisionBuffer(_ value: MTLBuffer?) throws -> MTLBuffer {
    guard let value else {
        throw NativeRuntimeError.invalidArgument("subdivision unexpectedly produced no neighbors")
    }
    return value
}
