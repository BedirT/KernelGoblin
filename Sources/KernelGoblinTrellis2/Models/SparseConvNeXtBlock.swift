import Metal

public struct SparseConvNeXtOffsets: Sendable {
    public let convolutionWeight: Int
    public let convolutionBias: Int
    public let normalizationWeight: Int
    public let normalizationBias: Int
    public let mlpUpWeight: Int
    public let mlpUpBias: Int
    public let mlpDownWeight: Int
    public let mlpDownBias: Int

    init(
        convolutionWeight: Int, convolutionBias: Int,
        normalizationWeight: Int, normalizationBias: Int,
        mlpUpWeight: Int, mlpUpBias: Int,
        mlpDownWeight: Int, mlpDownBias: Int
    ) {
        self.convolutionWeight = convolutionWeight
        self.convolutionBias = convolutionBias
        self.normalizationWeight = normalizationWeight
        self.normalizationBias = normalizationBias
        self.mlpUpWeight = mlpUpWeight
        self.mlpUpBias = mlpUpBias
        self.mlpDownWeight = mlpDownWeight
        self.mlpDownBias = mlpDownBias
    }

    public init(checkpoint: MappedCheckpoint, stage: Int, block: Int, channels: Int) throws {
        guard stage >= 0, block >= 0, channels > 0 else {
            throw NativeRuntimeError.invalidArgument("invalid ConvNeXt checkpoint location")
        }
        let prefix = "blocks.\(stage).\(block)."
        func descriptor(_ name: String, shape: [UInt64]) throws -> TensorDescriptor {
            let value = try checkpoint.descriptor(named: prefix + name)
            guard value.dtype == .f16, value.shape == shape else {
                throw NativeRuntimeError.invalidArgument(
                    "\(prefix + name) does not match its F16 ConvNeXt tensor contract"
                )
            }
            return value
        }
        let c = UInt64(channels), expanded = UInt64(channels * 4)
        let convWeight = try descriptor("conv.weight", shape: [c, 3, 3, 3, c])
        let convBias = try descriptor("conv.bias", shape: [c])
        let normWeight = try descriptor("norm.weight", shape: [c])
        let normBias = try descriptor("norm.bias", shape: [c])
        let upWeight = try descriptor("mlp.0.weight", shape: [expanded, c])
        let upBias = try descriptor("mlp.0.bias", shape: [expanded])
        let downWeight = try descriptor("mlp.2.weight", shape: [c, expanded])
        let downBias = try descriptor("mlp.2.bias", shape: [c])
        self.init(
            convolutionWeight: Int(convWeight.fileOffset),
            convolutionBias: Int(convBias.fileOffset),
            normalizationWeight: Int(normWeight.fileOffset),
            normalizationBias: Int(normBias.fileOffset),
            mlpUpWeight: Int(upWeight.fileOffset), mlpUpBias: Int(upBias.fileOffset),
            mlpDownWeight: Int(downWeight.fileOffset), mlpDownBias: Int(downBias.fileOffset)
        )
    }
}

public final class SparseConvNeXtBlock: @unchecked Sendable {
    private let convolution: SubmanifoldConvolutionKernel
    private let normalization: NormalizationKernel
    private let dense: DenseKernel
    private let primitive: PrimitiveKernel
    private let sparseStructure: SparseStructureKernel

    public init(context: MetalContext) throws {
        self.convolution = try SubmanifoldConvolutionKernel(context: context)
        self.normalization = try NormalizationKernel(context: context)
        self.dense = try DenseKernel(context: context)
        self.primitive = try PrimitiveKernel(context: context)
        self.sparseStructure = try SparseStructureKernel(context: context)
    }

    public func callAsFunction(
        input: MTLBuffer,
        neighbors: MTLBuffer?,
        checkpoint: MTLBuffer,
        offsets: SparseConvNeXtOffsets,
        tokenCount: Int,
        channels: Int,
        convolutionOutput: MTLBuffer,
        normalized: MTLBuffer,
        expanded: MTLBuffer,
        branch: MTLBuffer,
        output: MTLBuffer,
        convolutionImplementation: SubmanifoldConvolutionImplementation = .automatic
    ) throws {
        guard tokenCount >= 0, channels > 0 else {
            throw NativeRuntimeError.invalidArgument("invalid ConvNeXt sparse feature shape")
        }
        if tokenCount == 0 { return }
        let elements = try sparseConvNeXtProduct(tokenCount, channels)
        let expandedChannels = try sparseConvNeXtProduct(channels, 4)
        let expandedElements = try sparseConvNeXtProduct(tokenCount, expandedChannels)
        guard elements <= Int(UInt32.max), expandedElements <= Int(UInt32.max),
              [input, convolutionOutput, normalized, branch, output].allSatisfy({
                  $0.length >= elements * MemoryLayout<Float>.stride
              }), expanded.length >= expandedElements * MemoryLayout<Float>.stride else {
            throw NativeRuntimeError.invalidArgument("invalid ConvNeXt workspace buffers")
        }
        let allBuffers = [input, checkpoint, convolutionOutput, normalized, expanded, branch, output]
            + (neighbors.map { [$0] } ?? [])
        guard Set(allBuffers.map(ObjectIdentifier.init)).count == allBuffers.count else {
            throw NativeRuntimeError.invalidArgument("ConvNeXt buffers must not alias")
        }
        try convolution.convolveF16WeightsF32Output(
            input: input, neighbors: neighbors, checkpoint: checkpoint,
            weightOffset: offsets.convolutionWeight,
            biasOffset: offsets.convolutionBias,
            tokenCount: tokenCount, inputChannels: channels,
            outputChannels: channels, output: convolutionOutput,
            implementation: convolutionImplementation
        )
        try sparseStructure.roundF16F32(
            input: convolutionOutput, count: elements, output: convolutionOutput
        )
        try normalization.layerNormF32WeightsF16(
            input: convolutionOutput, checkpoint: checkpoint,
            rows: tokenCount, channels: channels,
            weightOffset: offsets.normalizationWeight,
            biasOffset: offsets.normalizationBias, epsilon: 1e-6,
            output: normalized
        )
        try sparseStructure.roundF16F32(input: normalized, count: elements, output: normalized)
        try dense.linearF16WeightsF32Output(
            input: normalized, checkpoint: checkpoint,
            weightOffset: offsets.mlpUpWeight, biasOffset: offsets.mlpUpBias,
            rows: tokenCount, inputChannels: channels,
            outputChannels: expandedChannels, output: expanded
        )
        try sparseStructure.roundF16F32(
            input: expanded, count: expandedElements, output: expanded
        )
        try primitive.siluF32(input: expanded, count: expandedElements, output: expanded)
        try sparseStructure.roundF16F32(
            input: expanded, count: expandedElements, output: expanded
        )
        try dense.linearF16WeightsF32Output(
            input: expanded, checkpoint: checkpoint,
            weightOffset: offsets.mlpDownWeight, biasOffset: offsets.mlpDownBias,
            rows: tokenCount, inputChannels: expandedChannels,
            outputChannels: channels, output: branch
        )
        try sparseStructure.roundF16F32(input: branch, count: elements, output: branch)
        try primitive.addF32(lhs: input, rhs: branch, count: elements, output: output)
        try sparseStructure.roundF16F32(input: output, count: elements, output: output)
    }

    public static func workspaceBytes(tokenCount: Int, channels: Int) throws -> Int {
        guard tokenCount >= 0, channels > 0 else {
            throw NativeRuntimeError.invalidArgument("invalid ConvNeXt workspace shape")
        }
        if tokenCount == 0 { return 0 }
        let elements = try sparseConvNeXtProduct(tokenCount, channels)
        // Four N*C buffers plus the N*4C expansion and the N*27 neighbor map.
        let featureBytes = try sparseConvNeXtProduct(elements, 8 * MemoryLayout<Float>.stride)
        let neighborBytes = try sparseConvNeXtProduct(
            tokenCount, SparseNeighborhood3x3.neighborCount * MemoryLayout<Int32>.stride
        )
        return try sparseConvNeXtSum(featureBytes, neighborBytes)
    }
}

private func sparseConvNeXtProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard lhs > 0, rhs > 0, !result.overflow else {
        throw NativeRuntimeError.invalidArgument("ConvNeXt tensor size overflows Int")
    }
    return result.partialValue
}

private func sparseConvNeXtSum(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.addingReportingOverflow(rhs)
    guard !result.overflow else {
        throw NativeRuntimeError.invalidArgument("ConvNeXt workspace size overflows Int")
    }
    return result.partialValue
}
