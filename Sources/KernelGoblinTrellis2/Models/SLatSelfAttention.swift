import Metal

public final class SLatSelfAttention: @unchecked Sendable {
    public static let channels = 1536
    public static let heads = 12
    public static let headDimensions = 128

    private let context: MetalContext
    private let dense: DenseKernel
    private let primitives: PrimitiveKernel
    private let normalization: NormalizationKernel
    private let rotaryPosition: RotaryPositionKernel
    private let attention: AttentionKernel

    public init(context: MetalContext) throws {
        self.context = context
        self.dense = try DenseKernel(context: context)
        self.primitives = try PrimitiveKernel(context: context)
        self.normalization = try NormalizationKernel(context: context)
        self.rotaryPosition = try RotaryPositionKernel(context: context)
        self.attention = try AttentionKernel(context: context)
    }

    public func forwardF32(
        input: MTLBuffer,
        checkpoint: MappedCheckpoint,
        block: Int,
        tokens: Int,
        coordinates: MTLBuffer?
    ) throws -> MTLBuffer {
        // `tokens` is one sparse sample. Segmented multi-sample attention is a
        // separate contract and must not concatenate batches into this call.
        if let coordinates {
            try requireSingleSequence(coordinates: coordinates, tokens: tokens)
        }
        guard block >= 0, tokens > 0 else {
            throw NativeRuntimeError.invalidArgument("block and token count must be nonnegative")
        }
        let prefix = "blocks.\(block).self_attn"
        let qkvWeight = try checkpoint.descriptor(named: "\(prefix).to_qkv.weight")
        let qkvBias = try checkpoint.descriptor(named: "\(prefix).to_qkv.bias")
        let qGamma = try checkpoint.descriptor(named: "\(prefix).q_rms_norm.gamma")
        let kGamma = try checkpoint.descriptor(named: "\(prefix).k_rms_norm.gamma")
        let outWeight = try checkpoint.descriptor(named: "\(prefix).to_out.weight")
        let outBias = try checkpoint.descriptor(named: "\(prefix).to_out.bias")
        let tensorBytes = try byteCount(tokens, Self.channels)
        let tensorElements = tensorBytes / MemoryLayout<Float>.stride
        guard qkvWeight.dtype == .bf16, qkvWeight.shape == [4608, 1536],
              qkvBias.dtype == .bf16, qkvBias.shape == [4608],
              qGamma.dtype == .bf16, qGamma.shape == [12, 128],
              kGamma.dtype == .bf16, kGamma.shape == [12, 128],
              outWeight.dtype == .bf16, outWeight.shape == [1536, 1536],
              outBias.dtype == .bf16, outBias.shape == [1536],
              input.length >= tensorBytes else {
            throw NativeRuntimeError.invalidArgument("incompatible SLat self-attention checkpoint")
        }

        let (query, key, value) = try projectAndSplit(
            input: input, checkpoint: checkpoint, tokens: tokens,
            weight: qkvWeight, bias: qkvBias
        )
        let normalizedQuery = try makeBuffer(length: tensorBytes, label: "SLat normalized query")
        let normalizedKey = try makeBuffer(length: tensorBytes, label: "SLat normalized key")
        try normalization.multiheadRMSNormF32(
            input: query, checkpoint: checkpoint.buffer, gammaOffset: Int(qGamma.fileOffset),
            rows: tokens, heads: Self.heads, dimensions: Self.headDimensions,
            output: normalizedQuery
        )
        try normalization.multiheadRMSNormF32(
            input: key, checkpoint: checkpoint.buffer, gammaOffset: Int(kGamma.fileOffset),
            rows: tokens, heads: Self.heads, dimensions: Self.headDimensions,
            output: normalizedKey
        )
        try primitives.roundBF16F32(
            input: normalizedQuery, count: tensorElements, output: normalizedQuery
        )
        try primitives.roundBF16F32(
            input: normalizedKey, count: tensorElements, output: normalizedKey
        )
        let attentionQuery: MTLBuffer
        let attentionKey: MTLBuffer
        if let coordinates {
            let rotatedQuery = try makeBuffer(length: tensorBytes, label: "SLat rotated query")
            let rotatedKey = try makeBuffer(length: tensorBytes, label: "SLat rotated key")
            try rotaryPosition.apply3DF32(
                input: normalizedQuery, coordinates: coordinates, tokens: tokens,
                heads: Self.heads, dimensions: Self.headDimensions, output: rotatedQuery
            )
            try rotaryPosition.apply3DF32(
                input: normalizedKey, coordinates: coordinates, tokens: tokens,
                heads: Self.heads, dimensions: Self.headDimensions, output: rotatedKey
            )
            try primitives.roundBF16F32(
                input: rotatedQuery, count: tensorElements, output: rotatedQuery
            )
            try primitives.roundBF16F32(
                input: rotatedKey, count: tensorElements, output: rotatedKey
            )
            attentionQuery = rotatedQuery
            attentionKey = rotatedKey
        } else {
            attentionQuery = normalizedQuery
            attentionKey = normalizedKey
        }
        let attended = try makeBuffer(length: tensorBytes, label: "SLat attended values")
        try attention.fusedF32(
            queries: attentionQuery, keys: attentionKey, values: value,
            queryCount: tokens, keyCount: tokens, heads: Self.heads,
            dimensions: Self.headDimensions, output: attended
        )
        try primitives.roundBF16F32(
            input: attended, count: tensorElements, output: attended
        )
        let output = try makeBuffer(length: tensorBytes, label: "SLat self-attention output")
        try dense.linearBF16WeightsF32Output(
            input: attended, checkpoint: checkpoint.buffer,
            weightOffset: Int(outWeight.fileOffset), biasOffset: Int(outBias.fileOffset),
            rows: tokens, inputChannels: Self.channels, outputChannels: Self.channels,
            output: output
        )
        try primitives.roundBF16F32(
            input: output, count: tensorElements, output: output
        )
        return output
    }

    private func requireSingleSequence(coordinates: MTLBuffer, tokens: Int) throws {
        let elements = tokens.multipliedReportingOverflow(by: 4)
        let bytes = elements.partialValue.multipliedReportingOverflow(
            by: MemoryLayout<Int32>.stride
        )
        guard tokens > 0, !elements.overflow, !bytes.overflow,
              coordinates.length >= bytes.partialValue,
              coordinates.storageMode != .private else {
            throw NativeRuntimeError.invalidArgument(
                "SLat coordinates must be CPU-readable [tokens,4] data"
            )
        }
        let values = coordinates.contents().assumingMemoryBound(to: Int32.self)
        let batch = values[0]
        for token in 1..<tokens where values[token * 4] != batch {
            throw NativeRuntimeError.invalidArgument(
                "SLat attention accepts one sparse sequence; use segmented attention for batches"
            )
        }
    }

    private func projectAndSplit(
        input: MTLBuffer, checkpoint: MappedCheckpoint, tokens: Int,
        weight: TensorDescriptor, bias: TensorDescriptor
    ) throws -> (MTLBuffer, MTLBuffer, MTLBuffer) {
        let qkvBytes = try byteCount(tokens, Self.channels * 3)
        let qkvElements = qkvBytes / MemoryLayout<Float>.stride
        let qkv = try makeBuffer(length: qkvBytes, label: "SLat QKV projection")
        try dense.linearBF16WeightsF32Output(
            input: input, checkpoint: checkpoint.buffer,
            weightOffset: Int(weight.fileOffset), biasOffset: Int(bias.fileOffset),
            rows: tokens, inputChannels: Self.channels,
            outputChannels: Self.channels * 3, output: qkv
        )
        try primitives.roundBF16F32(
            input: qkv, count: qkvElements, output: qkv
        )
        let tensorBytes = try byteCount(tokens, Self.channels)
        let query = try makeBuffer(length: tensorBytes, label: "SLat query")
        let key = try makeBuffer(length: tensorBytes, label: "SLat key")
        let value = try makeBuffer(length: tensorBytes, label: "SLat value")
        try primitives.splitQKVF32(
            input: qkv, rows: tokens, channels: Self.channels,
            query: query, key: key, value: value
        )
        return (query, key, value)
    }

    private func makeBuffer(length: Int, label: String) throws -> MTLBuffer {
        try context.makeBuffer(length: length, label: label)
    }
}

private func byteCount(_ rows: Int, _ channels: Int) throws -> Int {
    let elements = rows.multipliedReportingOverflow(by: channels)
    let bytes = elements.partialValue.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
    guard rows > 0, channels > 0, !elements.overflow, !bytes.overflow else {
        throw NativeRuntimeError.invalidArgument("SLat buffer size overflows Int")
    }
    return bytes.partialValue
}
