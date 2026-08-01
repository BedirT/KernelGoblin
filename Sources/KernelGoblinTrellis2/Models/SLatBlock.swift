import Metal

public final class SLatBlock: @unchecked Sendable {
    public static let channels = 1536

    private let context: MetalContext
    private let primitives: PrimitiveKernel
    private let normalization: NormalizationKernel
    private let selfAttention: SLatSelfAttention
    private let crossAttention: SLatCrossAttention
    private let feedForward: SLatFeedForward

    public init(context: MetalContext) throws {
        self.context = context
        self.primitives = try PrimitiveKernel(context: context)
        self.normalization = try NormalizationKernel(context: context)
        self.selfAttention = try SLatSelfAttention(context: context)
        self.crossAttention = try SLatCrossAttention(context: context)
        self.feedForward = try SLatFeedForward(context: context)
    }

    public func forwardF32(
        input: MTLBuffer, sharedModulation: MTLBuffer, conditioning: MTLBuffer,
        checkpoint: MappedCheckpoint, block: Int, tokens: Int,
        conditioningTokens: Int, coordinates: MTLBuffer
    ) throws -> MTLBuffer {
        // This vertical slice accepts exactly one sparse sequence. A future
        // stage API will carry batch segment offsets explicitly.
        try forwardF32(
            input: input, sharedModulation: sharedModulation, conditioning: conditioning,
            checkpoint: checkpoint, block: block, tokens: tokens,
            conditioningTokens: conditioningTokens, coordinates: coordinates,
            trace: nil
        )
    }

    func forwardF32(
        input: MTLBuffer, sharedModulation: MTLBuffer, conditioning: MTLBuffer,
        checkpoint: MappedCheckpoint, block: Int, tokens: Int,
        conditioningTokens: Int, coordinates: MTLBuffer,
        trace: ((String, MTLBuffer) -> Void)?
    ) throws -> MTLBuffer {
        let tensorBytes = try blockBytes(tokens, Self.channels)
        let tensorElements = tensorBytes / 4
        let modulationElements = Self.channels * 6
        guard block >= 0, input.length >= tensorBytes,
              sharedModulation.length >= modulationElements * 4 else {
            throw NativeRuntimeError.invalidArgument("invalid SLat block inputs")
        }
        let blockModulation = try checkpoint.descriptor(named: "blocks.\(block).modulation")
        let normWeight = try checkpoint.descriptor(named: "blocks.\(block).norm2.weight")
        let normBias = try checkpoint.descriptor(named: "blocks.\(block).norm2.bias")
        guard blockModulation.dtype == .bf16, blockModulation.shape == [9216],
              normWeight.dtype == .bf16, normWeight.shape == [1536],
              normBias.dtype == .bf16, normBias.shape == [1536] else {
            throw NativeRuntimeError.invalidArgument("incompatible SLat block checkpoint")
        }
        let modulation = try makeBuffer(length: modulationElements * 4, label: "SLat block modulation")
        try primitives.addCheckpointBF16F32(
            input: sharedModulation, checkpoint: try checkpoint.acquireBuffer(),
            checkpointOffset: Int(blockModulation.fileOffset),
            count: modulationElements, output: modulation
        )
        try primitives.roundBF16F32(
            input: modulation, count: modulationElements, output: modulation
        )

        let norm1 = try makeBuffer(length: tensorBytes, label: "SLat norm1")
        try normalization.layerNormF32(
            input: input, checkpoint: try checkpoint.acquireBuffer(), rows: tokens,
            channels: Self.channels, output: norm1
        )
        try primitives.roundBF16F32(input: norm1, count: tensorElements, output: norm1)
        trace?("norm1", norm1)
        let selfInput = try makeBuffer(length: tensorBytes, label: "SLat modulated self input")
        try primitives.modulateBF16F32(
            input: norm1, modulation: modulation, rows: tokens, channels: Self.channels,
            shiftOffset: 0, scaleOffset: Self.channels, output: selfInput
        )
        try primitives.roundBF16F32(input: selfInput, count: tensorElements, output: selfInput)
        trace?("self_input", selfInput)
        let selfOutput = try selfAttention.forwardF32(
            input: selfInput, checkpoint: checkpoint, block: block, tokens: tokens,
            coordinates: coordinates
        )
        trace?("self_output", selfOutput)
        let afterSelf = try makeBuffer(length: tensorBytes, label: "SLat self residual")
        try primitives.residualBF16F32(
            residual: input, branch: selfOutput, modulation: modulation,
            rows: tokens, channels: Self.channels, gateOffset: Self.channels * 2,
            output: afterSelf
        )
        try primitives.roundBF16F32(input: afterSelf, count: tensorElements, output: afterSelf)
        trace?("after_self", afterSelf)

        let norm2 = try makeBuffer(length: tensorBytes, label: "SLat norm2")
        try normalization.layerNormF32(
            input: afterSelf, checkpoint: try checkpoint.acquireBuffer(), rows: tokens,
            channels: Self.channels, weightOffset: Int(normWeight.fileOffset),
            biasOffset: Int(normBias.fileOffset), output: norm2
        )
        try primitives.roundBF16F32(input: norm2, count: tensorElements, output: norm2)
        trace?("norm2", norm2)
        let crossOutput = try crossAttention.forwardF32(
            input: norm2, conditioning: conditioning, checkpoint: checkpoint,
            block: block, tokens: tokens, conditioningTokens: conditioningTokens
        )
        trace?("cross_output", crossOutput)
        let afterCross = try makeBuffer(length: tensorBytes, label: "SLat cross residual")
        try primitives.residualBF16F32(
            residual: afterSelf, branch: crossOutput, modulation: modulation,
            rows: tokens, channels: Self.channels, output: afterCross
        )
        try primitives.roundBF16F32(input: afterCross, count: tensorElements, output: afterCross)
        trace?("after_cross", afterCross)

        let norm3 = try makeBuffer(length: tensorBytes, label: "SLat norm3")
        try normalization.layerNormF32(
            input: afterCross, checkpoint: try checkpoint.acquireBuffer(), rows: tokens,
            channels: Self.channels, output: norm3
        )
        try primitives.roundBF16F32(input: norm3, count: tensorElements, output: norm3)
        trace?("norm3", norm3)
        let mlpInput = try makeBuffer(length: tensorBytes, label: "SLat modulated MLP input")
        try primitives.modulateBF16F32(
            input: norm3, modulation: modulation, rows: tokens, channels: Self.channels,
            shiftOffset: Self.channels * 3, scaleOffset: Self.channels * 4,
            output: mlpInput
        )
        try primitives.roundBF16F32(input: mlpInput, count: tensorElements, output: mlpInput)
        trace?("mlp_input", mlpInput)
        let mlpOutput = try feedForward.forwardF32(
            input: mlpInput, checkpoint: checkpoint, block: block, tokens: tokens,
            trace: trace
        )
        trace?("mlp_output", mlpOutput)
        let output = try makeBuffer(length: tensorBytes, label: "SLat block output")
        try primitives.residualBF16F32(
            residual: afterCross, branch: mlpOutput, modulation: modulation,
            rows: tokens, channels: Self.channels, gateOffset: Self.channels * 5,
            output: output
        )
        try primitives.roundBF16F32(input: output, count: tensorElements, output: output)
        trace?("output", output)
        return output
    }

    private func makeBuffer(length: Int, label: String) throws -> MTLBuffer {
        try context.makeBuffer(length: length, label: label)
    }
}

private func blockBytes(_ rows: Int, _ channels: Int) throws -> Int {
    let elements = rows.multipliedReportingOverflow(by: channels)
    let bytes = elements.partialValue.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
    guard rows > 0, channels > 0, !elements.overflow, !bytes.overflow else {
        throw NativeRuntimeError.invalidArgument("SLat block buffer size overflows Int")
    }
    return bytes.partialValue
}
