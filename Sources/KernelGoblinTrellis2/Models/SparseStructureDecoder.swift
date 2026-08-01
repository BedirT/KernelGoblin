import Metal

public final class SparseStructureDecoder: @unchecked Sendable {
    public typealias Trace = (_ name: String, _ buffer: MTLBuffer, _ count: Int) -> Void

    private let context: MetalContext
    private let sparse: SparseStructureKernel
    private let normalization: NormalizationKernel
    private let primitives: PrimitiveKernel

    public init(context: MetalContext) throws {
        self.context = context
        self.sparse = try SparseStructureKernel(context: context)
        self.normalization = try NormalizationKernel(context: context)
        self.primitives = try PrimitiveKernel(context: context)
    }

    public func decodeF32(
        latent: MTLBuffer, inputResolution: Int = 16,
        checkpoint: MappedCheckpoint, trace: Trace? = nil
    ) throws -> MTLBuffer {
        let inputElements = try decoderElements(inputResolution, 8)
        let inputBytes = try decoderBytes(inputElements)
        guard latent.length >= inputBytes else {
            throw NativeRuntimeError.invalidArgument(
                "sparse-structure decoder latent has an incompatible shape"
            )
        }
        var hidden = try convolution(
            input: latent, resolution: inputResolution,
            inputChannels: 8, outputChannels: 512,
            prefix: "input_layer", weightType: .f32, checkpoint: checkpoint
        )
        try roundF16(hidden, count: try decoderElements(inputResolution, 512))
        trace?("input_layer", hidden, try decoderElements(inputResolution, 512))

        hidden = try residualBlock(
            hidden, resolution: inputResolution, channels: 512,
            prefix: "middle_block.0", checkpoint: checkpoint, trace: trace
        )
        hidden = try residualBlock(
            hidden, resolution: inputResolution, channels: 512,
            prefix: "middle_block.1", checkpoint: checkpoint, trace: trace
        )
        hidden = try residualBlock(
            hidden, resolution: inputResolution, channels: 512,
            prefix: "blocks.0", checkpoint: checkpoint, trace: trace
        )
        hidden = try residualBlock(
            hidden, resolution: inputResolution, channels: 512,
            prefix: "blocks.1", checkpoint: checkpoint, trace: trace
        )
        hidden = try upsample(
            hidden, resolution: inputResolution, inputChannels: 512,
            outputChannels: 128, prefix: "blocks.2", checkpoint: checkpoint,
            trace: trace
        )
        let middleResolution = try decoderMultiply(inputResolution, 2)
        hidden = try residualBlock(
            hidden, resolution: middleResolution, channels: 128,
            prefix: "blocks.3", checkpoint: checkpoint, trace: trace
        )
        hidden = try residualBlock(
            hidden, resolution: middleResolution, channels: 128,
            prefix: "blocks.4", checkpoint: checkpoint, trace: trace
        )
        hidden = try upsample(
            hidden, resolution: middleResolution, inputChannels: 128,
            outputChannels: 32, prefix: "blocks.5", checkpoint: checkpoint,
            trace: trace
        )
        let outputResolution = try decoderMultiply(middleResolution, 2)
        hidden = try residualBlock(
            hidden, resolution: outputResolution, channels: 32,
            prefix: "blocks.6", checkpoint: checkpoint, trace: trace
        )
        hidden = try residualBlock(
            hidden, resolution: outputResolution, channels: 32,
            prefix: "blocks.7", checkpoint: checkpoint, trace: trace
        )

        let outputVoxels = try decoderElements(outputResolution, 1)
        let normalized = try makeBuffer(
            elements: try decoderElements(outputResolution, 32),
            label: "Sparse decoder output normalization"
        )
        let normWeight = try requireTensor(
            checkpoint, "out_layer.0.weight", .f32, [32]
        )
        let normBias = try requireTensor(
            checkpoint, "out_layer.0.bias", .f32, [32]
        )
        try normalization.layerNormF32WeightsF32(
            input: hidden, checkpoint: try checkpoint.acquireBuffer(),
            rows: outputVoxels, channels: 32,
            weightOffset: Int(normWeight.fileOffset),
            biasOffset: Int(normBias.fileOffset), epsilon: 1e-5, output: normalized
        )
        let activated = try makeBuffer(
            elements: try decoderElements(outputResolution, 32),
            label: "Sparse decoder output SiLU"
        )
        try primitives.siluF32(
            input: normalized, count: try decoderElements(outputResolution, 32),
            output: activated
        )
        let output = try convolution(
            input: activated, resolution: outputResolution,
            inputChannels: 32, outputChannels: 1,
            prefix: "out_layer.2", weightType: .f32, checkpoint: checkpoint
        )
        trace?("output", output, outputVoxels)
        return output
    }

    private func residualBlock(
        _ input: MTLBuffer, resolution: Int, channels: Int,
        prefix: String, checkpoint: MappedCheckpoint, trace: Trace?
    ) throws -> MTLBuffer {
        let elements = try decoderElements(resolution, channels)
        let voxels = try decoderElements(resolution, 1)
        let firstNorm = try makeBuffer(
            elements: elements, label: "\(prefix) norm1"
        )
        let norm1Weight = try requireTensor(
            checkpoint, "\(prefix).norm1.weight", .f32, [UInt64(channels)]
        )
        let norm1Bias = try requireTensor(
            checkpoint, "\(prefix).norm1.bias", .f32, [UInt64(channels)]
        )
        try normalization.layerNormF32WeightsF32(
            input: input, checkpoint: try checkpoint.acquireBuffer(),
            rows: voxels, channels: channels,
            weightOffset: Int(norm1Weight.fileOffset),
            biasOffset: Int(norm1Bias.fileOffset), epsilon: 1e-5, output: firstNorm
        )
        try roundF16(firstNorm, count: elements)
        let firstActivated = try makeBuffer(
            elements: elements, label: "\(prefix) SiLU 1"
        )
        try primitives.siluF32(input: firstNorm, count: elements, output: firstActivated)
        try roundF16(firstActivated, count: elements)
        var branch = try convolution(
            input: firstActivated, resolution: resolution,
            inputChannels: channels, outputChannels: channels,
            prefix: "\(prefix).conv1", weightType: .f16, checkpoint: checkpoint
        )
        try roundF16(branch, count: elements)

        let secondNorm = try makeBuffer(
            elements: elements, label: "\(prefix) norm2"
        )
        let norm2Weight = try requireTensor(
            checkpoint, "\(prefix).norm2.weight", .f32, [UInt64(channels)]
        )
        let norm2Bias = try requireTensor(
            checkpoint, "\(prefix).norm2.bias", .f32, [UInt64(channels)]
        )
        try normalization.layerNormF32WeightsF32(
            input: branch, checkpoint: try checkpoint.acquireBuffer(),
            rows: voxels, channels: channels,
            weightOffset: Int(norm2Weight.fileOffset),
            biasOffset: Int(norm2Bias.fileOffset), epsilon: 1e-5, output: secondNorm
        )
        try roundF16(secondNorm, count: elements)
        let secondActivated = try makeBuffer(
            elements: elements, label: "\(prefix) SiLU 2"
        )
        try primitives.siluF32(input: secondNorm, count: elements, output: secondActivated)
        try roundF16(secondActivated, count: elements)
        branch = try convolution(
            input: secondActivated, resolution: resolution,
            inputChannels: channels, outputChannels: channels,
            prefix: "\(prefix).conv2", weightType: .f16, checkpoint: checkpoint
        )
        try roundF16(branch, count: elements)
        let output = try makeBuffer(elements: elements, label: "\(prefix) residual")
        try primitives.residualF32(
            residual: input, branch: branch, modulation: input,
            rows: voxels, channels: channels, output: output
        )
        try roundF16(output, count: elements)
        trace?(prefix, output, elements)
        return output
    }

    private func upsample(
        _ input: MTLBuffer, resolution: Int,
        inputChannels: Int, outputChannels: Int,
        prefix: String, checkpoint: MappedCheckpoint, trace: Trace?
    ) throws -> MTLBuffer {
        let packedChannels = try decoderMultiply(outputChannels, 8)
        let packedElements = try decoderElements(resolution, packedChannels)
        let packed = try convolution(
            input: input, resolution: resolution,
            inputChannels: inputChannels, outputChannels: packedChannels,
            prefix: "\(prefix).conv", weightType: .f16, checkpoint: checkpoint
        )
        try roundF16(packed, count: packedElements)
        let outputResolution = try decoderMultiply(resolution, 2)
        let elements = try decoderElements(outputResolution, outputChannels)
        let output = try makeBuffer(elements: elements, label: "\(prefix) pixel shuffle")
        try sparse.pixelShuffle3DF32(
            input: packed, inputResolution: resolution,
            outputChannels: outputChannels, output: output
        )
        trace?(prefix, output, elements)
        return output
    }

    private func convolution(
        input: MTLBuffer, resolution: Int,
        inputChannels: Int, outputChannels: Int,
        prefix: String, weightType: ConvolutionWeightType,
        checkpoint: MappedCheckpoint
    ) throws -> MTLBuffer {
        let dtype: TensorDataType = weightType == .f16 ? .f16 : .f32
        let weight = try requireTensor(
            checkpoint, "\(prefix).weight", dtype,
            [UInt64(outputChannels), UInt64(inputChannels), 3, 3, 3]
        )
        let bias = try requireTensor(
            checkpoint, "\(prefix).bias", dtype, [UInt64(outputChannels)]
        )
        let output = try makeBuffer(
            elements: try decoderElements(resolution, outputChannels),
            label: "\(prefix) Conv3D"
        )
        try sparse.conv3DF32(
            input: input, checkpoint: try checkpoint.acquireBuffer(),
            weightOffset: Int(weight.fileOffset), biasOffset: Int(bias.fileOffset),
            inputResolution: resolution, inputChannels: inputChannels,
            outputChannels: outputChannels, weightType: weightType, output: output
        )
        return output
    }

    private func roundF16(_ buffer: MTLBuffer, count: Int) throws {
        try sparse.roundF16F32(input: buffer, count: count, output: buffer)
    }

    private func requireTensor(
        _ checkpoint: MappedCheckpoint, _ name: String,
        _ dtype: TensorDataType, _ shape: [UInt64]
    ) throws -> TensorDescriptor {
        let tensor = try checkpoint.descriptor(named: name)
        guard tensor.dtype == dtype, tensor.shape == shape else {
            throw NativeRuntimeError.invalidArgument(
                "incompatible sparse decoder tensor \(name)"
            )
        }
        return tensor
    }

    private func makeBuffer(elements: Int, label: String) throws -> MTLBuffer {
        try context.makeBuffer(length: try decoderBytes(elements), label: label)
    }
}

private func decoderElements(_ resolution: Int, _ channels: Int) throws -> Int {
    try decoderMultiply(
        try decoderMultiply(try decoderMultiply(resolution, resolution), resolution),
        channels
    )
}

private func decoderBytes(_ elements: Int) throws -> Int {
    try decoderMultiply(elements, MemoryLayout<Float>.stride)
}

private func decoderMultiply(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard lhs > 0, rhs > 0, !result.overflow else {
        throw NativeRuntimeError.invalidArgument("sparse decoder size overflows Int")
    }
    return result.partialValue
}
