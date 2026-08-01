import Metal

public enum SLatFlowConfiguration: Sendable {
    case shape
    case texture

    public var inputChannels: Int {
        switch self {
        case .shape: 32
        case .texture: 64
        }
    }
}

public final class SLatFlow: @unchecked Sendable {
    public static let modelChannels = 1536
    public static let outputChannels = 32
    public static let blockCount = 30

    public let configuration: SLatFlowConfiguration
    private let context: MetalContext
    private let dense: DenseKernel
    private let primitives: PrimitiveKernel
    private let normalization: NormalizationKernel
    private let block: SLatBlock

    public init(
        context: MetalContext, configuration: SLatFlowConfiguration = .shape
    ) throws {
        self.configuration = configuration
        self.context = context
        self.dense = try DenseKernel(context: context)
        self.primitives = try PrimitiveKernel(context: context)
        self.normalization = try NormalizationKernel(context: context)
        self.block = try SLatBlock(context: context)
    }

    public func forwardF32(
        input: MTLBuffer, timestep: MTLBuffer, conditioning: MTLBuffer,
        coordinates: MTLBuffer, checkpoint: MappedCheckpoint, tokens: Int,
        conditioningTokens: Int
    ) throws -> MTLBuffer {
        try forwardF32(
            input: input, timestep: timestep, conditioning: conditioning,
            coordinates: coordinates, checkpoint: checkpoint, tokens: tokens,
            conditioningTokens: conditioningTokens, trace: nil
        )
    }

    func forwardF32(
        input: MTLBuffer, timestep: MTLBuffer, conditioning: MTLBuffer,
        coordinates: MTLBuffer, checkpoint: MappedCheckpoint, tokens: Int,
        conditioningTokens: Int, trace: ((Int, MTLBuffer) -> Void)?
    ) throws -> MTLBuffer {
        let inputBytes = try shapeFlowBytes(tokens, configuration.inputChannels)
        let hiddenBytes = try shapeFlowBytes(tokens, Self.modelChannels)
        let conditioningBytes = try shapeFlowBytes(conditioningTokens, 1024)
        let coordinateBytes = try shapeFlowBytes(
            tokens, 4, elementWidth: MemoryLayout<Int32>.stride
        )
        guard input.length >= inputBytes, timestep.length >= 4,
              conditioning.length >= conditioningBytes,
              coordinates.length >= coordinateBytes else {
            throw NativeRuntimeError.invalidArgument("invalid shape-flow inputs")
        }

        let inputWeight = try requireTensor(
            checkpoint, "input_layer.weight", .bf16,
            [1536, UInt64(configuration.inputChannels)]
        )
        let inputBias = try requireTensor(checkpoint, "input_layer.bias", .bf16, [1536])
        var hidden = try makeBuffer(length: hiddenBytes, label: "shape-flow input projection")
        try dense.linearBF16WeightsF32Output(
            input: input, checkpoint: checkpoint.buffer,
            weightOffset: Int(inputWeight.fileOffset), biasOffset: Int(inputBias.fileOffset),
            rows: tokens, inputChannels: configuration.inputChannels,
            outputChannels: Self.modelChannels, output: hidden
        )
        try primitives.roundBF16F32(input: hidden, count: hiddenBytes / 4, output: hidden)

        let modulation = try makeConditioning(timestep: timestep, checkpoint: checkpoint)
        let roundedConditioning = try makeBuffer(
            length: conditioningBytes, label: "shape-flow rounded image conditioning"
        )
        try primitives.roundBF16F32(
            input: conditioning, count: conditioningBytes / 4, output: roundedConditioning
        )
        for index in 0..<Self.blockCount {
            hidden = try block.forwardF32(
                input: hidden, sharedModulation: modulation,
                conditioning: roundedConditioning, checkpoint: checkpoint,
                block: index, tokens: tokens, conditioningTokens: conditioningTokens,
                coordinates: coordinates
            )
            trace?(index, hidden)
        }

        let normalized = try makeBuffer(length: hiddenBytes, label: "shape-flow output norm")
        try normalization.layerNormF32(
            input: hidden, checkpoint: checkpoint.buffer, rows: tokens,
            channels: Self.modelChannels, epsilon: 1e-5, output: normalized
        )
        let outputWeight = try requireTensor(
            checkpoint, "out_layer.weight", .bf16, [32, 1536]
        )
        let outputBias = try requireTensor(checkpoint, "out_layer.bias", .bf16, [32])
        let outputBytes = try shapeFlowBytes(tokens, Self.outputChannels)
        let output = try makeBuffer(length: outputBytes, label: "shape-flow output")
        try dense.linearBF16WeightsF32Output(
            input: normalized, checkpoint: checkpoint.buffer,
            weightOffset: Int(outputWeight.fileOffset), biasOffset: Int(outputBias.fileOffset),
            rows: tokens, inputChannels: Self.modelChannels,
            outputChannels: Self.outputChannels, output: output
        )
        return output
    }

    private func makeConditioning(
        timestep: MTLBuffer, checkpoint: MappedCheckpoint
    ) throws -> MTLBuffer {
        let firstWeight = try requireTensor(
            checkpoint, "t_embedder.mlp.0.weight", .bf16, [1536, 256]
        )
        let firstBias = try requireTensor(
            checkpoint, "t_embedder.mlp.0.bias", .bf16, [1536]
        )
        let secondWeight = try requireTensor(
            checkpoint, "t_embedder.mlp.2.weight", .bf16, [1536, 1536]
        )
        let secondBias = try requireTensor(
            checkpoint, "t_embedder.mlp.2.bias", .bf16, [1536]
        )
        let modulationWeight = try requireTensor(
            checkpoint, "adaLN_modulation.1.weight", .bf16, [9216, 1536]
        )
        let modulationBias = try requireTensor(
            checkpoint, "adaLN_modulation.1.bias", .bf16, [9216]
        )
        let frequency = try makeBuffer(length: 256 * 4, label: "shape-flow timestep sinusoid")
        let first = try makeBuffer(length: 1536 * 4, label: "shape-flow timestep first")
        let activated = try makeBuffer(length: 1536 * 4, label: "shape-flow timestep SiLU")
        let embedding = try makeBuffer(length: 1536 * 4, label: "shape-flow timestep embedding")
        let modulationInput = try makeBuffer(length: 1536 * 4, label: "shape-flow adaLN SiLU")
        let modulation = try makeBuffer(length: 9216 * 4, label: "shape-flow adaLN")
        try primitives.timestepEmbeddingF32(
            timesteps: timestep, rows: 1, dimensions: 256, output: frequency
        )
        try dense.linearBF16WeightsF32Output(
            input: frequency, checkpoint: checkpoint.buffer,
            weightOffset: Int(firstWeight.fileOffset), biasOffset: Int(firstBias.fileOffset),
            rows: 1, inputChannels: 256, outputChannels: 1536, output: first
        )
        try primitives.siluF32(input: first, count: 1536, output: activated)
        try dense.linearBF16WeightsF32Output(
            input: activated, checkpoint: checkpoint.buffer,
            weightOffset: Int(secondWeight.fileOffset), biasOffset: Int(secondBias.fileOffset),
            rows: 1, inputChannels: 1536, outputChannels: 1536, output: embedding
        )
        try primitives.siluF32(input: embedding, count: 1536, output: modulationInput)
        try dense.linearBF16WeightsF32Output(
            input: modulationInput, checkpoint: checkpoint.buffer,
            weightOffset: Int(modulationWeight.fileOffset),
            biasOffset: Int(modulationBias.fileOffset), rows: 1,
            inputChannels: 1536, outputChannels: 9216, output: modulation
        )
        try primitives.roundBF16F32(input: modulation, count: 9216, output: modulation)
        return modulation
    }

    private func requireTensor(
        _ checkpoint: MappedCheckpoint, _ name: String,
        _ dtype: TensorDataType, _ shape: [UInt64]
    ) throws -> TensorDescriptor {
        let tensor = try checkpoint.descriptor(named: name)
        guard tensor.dtype == dtype, tensor.shape == shape else {
            throw NativeRuntimeError.invalidArgument("incompatible shape-flow tensor \(name)")
        }
        return tensor
    }

    private func makeBuffer(length: Int, label: String) throws -> MTLBuffer {
        try context.makeBuffer(length: length, label: label)
    }
}

public typealias SLatShapeFlow = SLatFlow

private func shapeFlowBytes(
    _ rows: Int, _ channels: Int,
    elementWidth: Int = MemoryLayout<Float>.stride
) throws -> Int {
    let elements = rows.multipliedReportingOverflow(by: channels)
    let bytes = elements.partialValue.multipliedReportingOverflow(by: elementWidth)
    guard rows > 0, channels > 0, elementWidth > 0,
          !elements.overflow, !bytes.overflow else {
        throw NativeRuntimeError.invalidArgument("shape-flow buffer size overflows Int")
    }
    return bytes.partialValue
}
