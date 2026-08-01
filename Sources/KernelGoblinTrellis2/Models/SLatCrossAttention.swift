import Metal

public final class SLatCrossAttention: @unchecked Sendable {
    public static let channels = 1536
    public static let contextChannels = 1024
    public static let heads = 12
    public static let headDimensions = 128

    private let context: MetalContext
    private let dense: DenseKernel
    private let primitives: PrimitiveKernel
    private let normalization: NormalizationKernel
    private let attention: AttentionKernel

    public init(context: MetalContext) throws {
        self.context = context
        self.dense = try DenseKernel(context: context)
        self.primitives = try PrimitiveKernel(context: context)
        self.normalization = try NormalizationKernel(context: context)
        self.attention = try AttentionKernel(context: context)
    }

    public func forwardF32(
        input: MTLBuffer, conditioning: MTLBuffer, checkpoint: MappedCheckpoint,
        block: Int, tokens: Int, conditioningTokens: Int
    ) throws -> MTLBuffer {
        guard block >= 0, tokens > 0, conditioningTokens > 0 else {
            throw NativeRuntimeError.invalidArgument("invalid cross-attention block or token count")
        }
        let prefix = "blocks.\(block).cross_attn"
        let qWeight = try checkpoint.descriptor(named: "\(prefix).to_q.weight")
        let qBias = try checkpoint.descriptor(named: "\(prefix).to_q.bias")
        let kvWeight = try checkpoint.descriptor(named: "\(prefix).to_kv.weight")
        let kvBias = try checkpoint.descriptor(named: "\(prefix).to_kv.bias")
        let qGamma = try checkpoint.descriptor(named: "\(prefix).q_rms_norm.gamma")
        let kGamma = try checkpoint.descriptor(named: "\(prefix).k_rms_norm.gamma")
        let outWeight = try checkpoint.descriptor(named: "\(prefix).to_out.weight")
        let outBias = try checkpoint.descriptor(named: "\(prefix).to_out.bias")
        let queryBytes = try crossByteCount(tokens, Self.channels)
        let keyBytes = try crossByteCount(conditioningTokens, Self.channels)
        let contextBytes = try crossByteCount(conditioningTokens, Self.contextChannels)
        guard qWeight.dtype == .bf16, qWeight.shape == [1536, 1536],
              qBias.dtype == .bf16, qBias.shape == [1536],
              kvWeight.dtype == .bf16, kvWeight.shape == [3072, 1024],
              kvBias.dtype == .bf16, kvBias.shape == [3072],
              qGamma.dtype == .bf16, qGamma.shape == [12, 128],
              kGamma.dtype == .bf16, kGamma.shape == [12, 128],
              outWeight.dtype == .bf16, outWeight.shape == [1536, 1536],
              outBias.dtype == .bf16, outBias.shape == [1536],
              input.length >= queryBytes, conditioning.length >= contextBytes else {
            throw NativeRuntimeError.invalidArgument("incompatible SLat cross-attention checkpoint")
        }

        let query = try makeBuffer(length: queryBytes, label: "SLat cross query")
        try dense.linearBF16WeightsF32Output(
            input: input, checkpoint: checkpoint.buffer,
            weightOffset: Int(qWeight.fileOffset), biasOffset: Int(qBias.fileOffset),
            rows: tokens, inputChannels: Self.channels, outputChannels: Self.channels,
            output: query
        )
        try primitives.roundBF16F32(
            input: query, count: queryBytes / 4, output: query
        )
        let (key, value) = try projectAndSplitKV(
            conditioning: conditioning, checkpoint: checkpoint,
            conditioningTokens: conditioningTokens, weight: kvWeight, bias: kvBias,
            tensorBytes: keyBytes
        )
        let normalizedQuery = try makeBuffer(length: queryBytes, label: "SLat normalized cross query")
        let normalizedKey = try makeBuffer(length: keyBytes, label: "SLat normalized cross key")
        try normalization.multiheadRMSNormF32(
            input: query, checkpoint: checkpoint.buffer, gammaOffset: Int(qGamma.fileOffset),
            rows: tokens, heads: Self.heads, dimensions: Self.headDimensions,
            output: normalizedQuery
        )
        try normalization.multiheadRMSNormF32(
            input: key, checkpoint: checkpoint.buffer, gammaOffset: Int(kGamma.fileOffset),
            rows: conditioningTokens, heads: Self.heads, dimensions: Self.headDimensions,
            output: normalizedKey
        )
        try primitives.roundBF16F32(input: normalizedQuery, count: queryBytes / 4, output: normalizedQuery)
        try primitives.roundBF16F32(input: normalizedKey, count: keyBytes / 4, output: normalizedKey)
        let attended = try makeBuffer(length: queryBytes, label: "SLat cross attended values")
        try attention.fusedF32(
            queries: normalizedQuery, keys: normalizedKey, values: value,
            queryCount: tokens, keyCount: conditioningTokens, heads: Self.heads,
            dimensions: Self.headDimensions, output: attended
        )
        try primitives.roundBF16F32(input: attended, count: queryBytes / 4, output: attended)
        let output = try makeBuffer(length: queryBytes, label: "SLat cross-attention output")
        try dense.linearBF16WeightsF32Output(
            input: attended, checkpoint: checkpoint.buffer,
            weightOffset: Int(outWeight.fileOffset), biasOffset: Int(outBias.fileOffset),
            rows: tokens, inputChannels: Self.channels, outputChannels: Self.channels,
            output: output
        )
        try primitives.roundBF16F32(input: output, count: queryBytes / 4, output: output)
        return output
    }

    private func projectAndSplitKV(
        conditioning: MTLBuffer, checkpoint: MappedCheckpoint, conditioningTokens: Int,
        weight: TensorDescriptor, bias: TensorDescriptor, tensorBytes: Int
    ) throws -> (MTLBuffer, MTLBuffer) {
        let packedBytes = tensorBytes.multipliedReportingOverflow(by: 2)
        guard !packedBytes.overflow else {
            throw NativeRuntimeError.invalidArgument("SLat cross KV buffer size overflows Int")
        }
        let packed = try makeBuffer(
            length: packedBytes.partialValue, label: "SLat cross KV projection"
        )
        try dense.linearBF16WeightsF32Output(
            input: conditioning, checkpoint: checkpoint.buffer,
            weightOffset: Int(weight.fileOffset), biasOffset: Int(bias.fileOffset),
            rows: conditioningTokens, inputChannels: Self.contextChannels,
            outputChannels: Self.channels * 2, output: packed
        )
        try primitives.roundBF16F32(input: packed, count: tensorBytes / 2, output: packed)
        let key = try makeBuffer(length: tensorBytes, label: "SLat cross key")
        let value = try makeBuffer(length: tensorBytes, label: "SLat cross value")
        try primitives.splitKVF32(
            input: packed, rows: conditioningTokens, channels: Self.channels,
            key: key, value: value
        )
        return (key, value)
    }

    private func makeBuffer(length: Int, label: String) throws -> MTLBuffer {
        try context.makeBuffer(length: length, label: label)
    }
}

private func crossByteCount(_ rows: Int, _ channels: Int) throws -> Int {
    let elements = rows.multipliedReportingOverflow(by: channels)
    let bytes = elements.partialValue.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
    guard rows > 0, channels > 0, !elements.overflow, !bytes.overflow else {
        throw NativeRuntimeError.invalidArgument("SLat cross-attention buffer size overflows Int")
    }
    return bytes.partialValue
}
