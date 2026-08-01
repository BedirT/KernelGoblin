import Metal

public final class SLatFeedForward: @unchecked Sendable {
    public static let channels = 1536
    public static let hiddenChannels = 8192

    private let context: MetalContext
    private let dense: DenseKernel
    private let primitives: PrimitiveKernel

    public init(context: MetalContext) throws {
        self.context = context
        self.dense = try DenseKernel(context: context)
        self.primitives = try PrimitiveKernel(context: context)
    }

    public func forwardF32(
        input: MTLBuffer, checkpoint: MappedCheckpoint, block: Int, tokens: Int
    ) throws -> MTLBuffer {
        try forwardF32(
            input: input, checkpoint: checkpoint, block: block, tokens: tokens,
            trace: nil
        )
    }

    func forwardF32(
        input: MTLBuffer, checkpoint: MappedCheckpoint, block: Int, tokens: Int,
        trace: ((String, MTLBuffer) -> Void)?
    ) throws -> MTLBuffer {
        guard block >= 0, tokens > 0 else {
            throw NativeRuntimeError.invalidArgument("invalid feed-forward block or token count")
        }
        let prefix = "blocks.\(block).mlp.mlp"
        let upWeight = try checkpoint.descriptor(named: "\(prefix).0.weight")
        let upBias = try checkpoint.descriptor(named: "\(prefix).0.bias")
        let downWeight = try checkpoint.descriptor(named: "\(prefix).2.weight")
        let downBias = try checkpoint.descriptor(named: "\(prefix).2.bias")
        guard upWeight.dtype == .bf16, upWeight.shape == [8192, 1536],
              upBias.dtype == .bf16, upBias.shape == [8192],
              downWeight.dtype == .bf16, downWeight.shape == [1536, 8192],
              downBias.dtype == .bf16, downBias.shape == [1536] else {
            throw NativeRuntimeError.invalidArgument("incompatible SLat feed-forward checkpoint")
        }
        let hiddenBytes = try feedForwardBytes(tokens, Self.hiddenChannels)
        let outputBytes = try feedForwardBytes(tokens, Self.channels)
        guard input.length >= outputBytes else {
            throw NativeRuntimeError.invalidArgument("feed-forward input buffer is too small")
        }
        let hidden = try makeBuffer(length: hiddenBytes, label: "SLat MLP hidden")
        try dense.linearBF16WeightsF32Output(
            input: input, checkpoint: checkpoint.buffer,
            weightOffset: Int(upWeight.fileOffset), biasOffset: Int(upBias.fileOffset),
            rows: tokens, inputChannels: Self.channels,
            outputChannels: Self.hiddenChannels, output: hidden
        )
        try primitives.roundBF16F32(input: hidden, count: hiddenBytes / 4, output: hidden)
        trace?("mlp_hidden_linear", hidden)
        let activated = try makeBuffer(length: hiddenBytes, label: "SLat MLP GELU")
        try primitives.geluTanhF32(input: hidden, count: hiddenBytes / 4, output: activated)
        try primitives.roundBF16F32(input: activated, count: hiddenBytes / 4, output: activated)
        trace?("mlp_hidden_gelu", activated)
        let output = try makeBuffer(length: outputBytes, label: "SLat MLP output")
        try dense.linearBF16WeightsF32Output(
            input: activated, checkpoint: checkpoint.buffer,
            weightOffset: Int(downWeight.fileOffset), biasOffset: Int(downBias.fileOffset),
            rows: tokens, inputChannels: Self.hiddenChannels,
            outputChannels: Self.channels, output: output
        )
        try primitives.roundBF16F32(input: output, count: outputBytes / 4, output: output)
        return output
    }

    private func makeBuffer(length: Int, label: String) throws -> MTLBuffer {
        try context.makeBuffer(length: length, label: label)
    }
}

private func feedForwardBytes(_ rows: Int, _ channels: Int) throws -> Int {
    let elements = rows.multipliedReportingOverflow(by: channels)
    let bytes = elements.partialValue.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
    guard rows > 0, channels > 0, !elements.overflow, !bytes.overflow else {
        throw NativeRuntimeError.invalidArgument("SLat feed-forward buffer size overflows Int")
    }
    return bytes.partialValue
}
