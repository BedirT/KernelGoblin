import Foundation
import Metal

public struct DINOv3Configuration: Equatable, Sendable {
    public var patchSize = 16
    public var inputChannels = 3
    public var hiddenSize = 1024
    public var layerCount = 24
    public var headCount = 16
    public var mlpHiddenSize = 4096
    public var prefixTokenCount = 5
    public var layerNormEpsilon: Float = 1e-5
    public var ropeTheta: Float = 100

    public init() {}
}

public struct DINOv3ConditioningResult: @unchecked Sendable {
    public let conditioning: MTLBuffer
    public let tokenCount: Int
    public let hiddenSize: Int
}

public typealias DINOv3Trace = (_ stage: String, _ values: [Float]) -> Void

public final class DINOv3Conditioner: @unchecked Sendable {
    private let context: MetalContext
    private let configuration: DINOv3Configuration
    private let dense: DenseKernel
    private let normalization: NormalizationKernel
    private let attention: AttentionKernel
    private let primitive: PrimitiveKernel
    private let dino: DINOv3Kernel

    public init(
        context: MetalContext,
        configuration: DINOv3Configuration = DINOv3Configuration()
    ) throws {
        guard context.arena != nil,
              configuration.patchSize > 0, configuration.inputChannels > 0,
              configuration.hiddenSize > 0, configuration.layerCount > 0,
              configuration.headCount > 0,
              configuration.hiddenSize % configuration.headCount == 0,
              (configuration.hiddenSize / configuration.headCount) % 4 == 0,
              configuration.mlpHiddenSize > 0, configuration.prefixTokenCount > 0,
              configuration.layerNormEpsilon > 0, configuration.ropeTheta > 0 else {
            throw NativeRuntimeError.invalidArgument(
                "DINOv3 conditioning requires a valid configuration and bounded Metal arena"
            )
        }
        self.context = context
        self.configuration = configuration
        dense = try DenseKernel(context: context)
        normalization = try NormalizationKernel(context: context)
        attention = try AttentionKernel(context: context)
        primitive = try PrimitiveKernel(context: context)
        dino = try DINOv3Kernel(context: context)
    }

    public func encodeNormalizedImageF32(
        image: MTLBuffer,
        imageHeight: Int,
        imageWidth: Int,
        checkpoint: MappedCheckpoint,
        trace: DINOv3Trace? = nil
    ) throws -> DINOv3ConditioningResult {
        let config = configuration
        guard imageHeight > 0, imageWidth > 0,
              imageHeight % config.patchSize == 0,
              imageWidth % config.patchSize == 0 else {
            throw NativeRuntimeError.invalidArgument(
                "DINOv3 image dimensions must be positive patch-size multiples"
            )
        }
        try validateCheckpoint(checkpoint)
        let patchesH = imageHeight / config.patchSize
        let patchesW = imageWidth / config.patchSize
        let patchCount = try modelProduct(patchesH, patchesW)
        let tokenCount = try modelSum(config.prefixTokenCount, patchCount)
        let tokenElements = try modelProduct(tokenCount, config.hiddenSize)
        let tokenBytes = try modelBytes(tokenElements)
        let imageElements = try modelProduct(
            try modelProduct(imageHeight, imageWidth), config.inputChannels
        )
        let imageBytes = try modelBytes(imageElements)
        guard image.length >= imageBytes else {
            throw NativeRuntimeError.invalidArgument("DINOv3 normalized image buffer is too small")
        }

        let embeddings = try context.makeBuffer(
            length: tokenBytes, label: "DINOv3 patch and prefix embeddings"
        )
        try copyPrefixTokens(checkpoint: checkpoint, output: embeddings)
        let patchWeight = try tensor(
            checkpoint, "embeddings.patch_embeddings.weight",
            shape: [config.hiddenSize, config.inputChannels, config.patchSize, config.patchSize]
        )
        let patchBias = try tensor(
            checkpoint, "embeddings.patch_embeddings.bias", shape: [config.hiddenSize]
        )
        try dino.patchEmbedF32(
            image: image, checkpoint: try checkpoint.acquireBuffer(),
            weightOffset: try tensorOffset(patchWeight),
            biasOffset: try tensorOffset(patchBias),
            imageHeight: imageHeight, imageWidth: imageWidth,
            patchSize: config.patchSize, inputChannels: config.inputChannels,
            outputChannels: config.hiddenSize,
            outputTokenOffset: config.prefixTokenCount, output: embeddings
        )
        snapshot(stage: "embeddings", buffer: embeddings, count: tokenElements, trace: trace)

        var hidden = embeddings
        for layer in 0..<config.layerCount {
            hidden = try runBlock(
                layer: layer, hidden: hidden, tokenCount: tokenCount,
                patchesH: patchesH, patchesW: patchesW,
                checkpoint: checkpoint
            )
            snapshot(
                stage: "block_\(layer)", buffer: hidden,
                count: tokenElements, trace: trace
            )
        }

        let output = try context.makeBuffer(
            length: tokenBytes, label: "DINOv3 final parameter-free LayerNorm"
        )
        try normalization.layerNormF32WeightsF32(
            input: hidden, checkpoint: try checkpoint.acquireBuffer(),
            rows: tokenCount, channels: config.hiddenSize,
            epsilon: config.layerNormEpsilon, output: output
        )
        snapshot(
            stage: "final_parameter_free_layer_norm", buffer: output,
            count: tokenElements, trace: trace
        )
        return DINOv3ConditioningResult(
            conditioning: output, tokenCount: tokenCount, hiddenSize: config.hiddenSize
        )
    }

    private func runBlock(
        layer: Int,
        hidden: MTLBuffer,
        tokenCount: Int,
        patchesH: Int,
        patchesW: Int,
        checkpoint: MappedCheckpoint
    ) throws -> MTLBuffer {
        let config = configuration
        let tokenElements = try modelProduct(tokenCount, config.hiddenSize)
        let tokenBytes = try modelBytes(tokenElements)
        let prefix = "layer.\(layer)"

        let norm1 = try context.makeBuffer(length: tokenBytes, label: "DINOv3 norm1")
        try normalization.layerNormF32WeightsF32(
            input: hidden, checkpoint: try checkpoint.acquireBuffer(),
            rows: tokenCount, channels: config.hiddenSize,
            weightOffset: try offset(checkpoint, "\(prefix).norm1.weight", [config.hiddenSize]),
            biasOffset: try offset(checkpoint, "\(prefix).norm1.bias", [config.hiddenSize]),
            epsilon: config.layerNormEpsilon, output: norm1
        )

        let query = try linear(
            input: norm1, checkpoint: checkpoint, name: "\(prefix).attention.q_proj",
            rows: tokenCount, inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize, hasBias: true, label: "DINOv3 query"
        )
        let key = try linear(
            input: norm1, checkpoint: checkpoint, name: "\(prefix).attention.k_proj",
            rows: tokenCount, inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize, hasBias: false, label: "DINOv3 key"
        )
        let value = try linear(
            input: norm1, checkpoint: checkpoint, name: "\(prefix).attention.v_proj",
            rows: tokenCount, inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize, hasBias: true, label: "DINOv3 value"
        )
        let rotatedQuery = try context.makeBuffer(
            length: tokenBytes, label: "DINOv3 rotated query"
        )
        let rotatedKey = try context.makeBuffer(
            length: tokenBytes, label: "DINOv3 rotated key"
        )
        let headDimension = config.hiddenSize / config.headCount
        try dino.ropeF32(
            input: query, tokenCount: tokenCount,
            prefixTokens: config.prefixTokenCount,
            patchesH: patchesH, patchesW: patchesW,
            heads: config.headCount, dimensions: headDimension,
            theta: config.ropeTheta, output: rotatedQuery
        )
        try dino.ropeF32(
            input: key, tokenCount: tokenCount,
            prefixTokens: config.prefixTokenCount,
            patchesH: patchesH, patchesW: patchesW,
            heads: config.headCount, dimensions: headDimension,
            theta: config.ropeTheta, output: rotatedKey
        )
        let attended = try context.makeBuffer(length: tokenBytes, label: "DINOv3 attention")
        try attention.fusedF32(
            queries: rotatedQuery, keys: rotatedKey, values: value,
            queryCount: tokenCount, keyCount: tokenCount,
            heads: config.headCount, dimensions: headDimension, output: attended
        )
        let projected = try linear(
            input: attended, checkpoint: checkpoint,
            name: "\(prefix).attention.o_proj", rows: tokenCount,
            inputChannels: config.hiddenSize, outputChannels: config.hiddenSize,
            hasBias: true, label: "DINOv3 attention output"
        )
        let afterAttention = try context.makeBuffer(
            length: tokenBytes, label: "DINOv3 attention residual"
        )
        try primitive.layerScaleResidualF32(
            residual: hidden, branch: projected, checkpoint: try checkpoint.acquireBuffer(),
            scaleOffset: try offset(
                checkpoint, "\(prefix).layer_scale1.lambda1", [config.hiddenSize]
            ),
            rows: tokenCount, channels: config.hiddenSize, output: afterAttention
        )

        let norm2 = try context.makeBuffer(length: tokenBytes, label: "DINOv3 norm2")
        try normalization.layerNormF32WeightsF32(
            input: afterAttention, checkpoint: try checkpoint.acquireBuffer(),
            rows: tokenCount, channels: config.hiddenSize,
            weightOffset: try offset(checkpoint, "\(prefix).norm2.weight", [config.hiddenSize]),
            biasOffset: try offset(checkpoint, "\(prefix).norm2.bias", [config.hiddenSize]),
            epsilon: config.layerNormEpsilon, output: norm2
        )
        let mlpHidden = try linear(
            input: norm2, checkpoint: checkpoint, name: "\(prefix).mlp.up_proj",
            rows: tokenCount, inputChannels: config.hiddenSize,
            outputChannels: config.mlpHiddenSize, hasBias: true,
            label: "DINOv3 MLP hidden"
        )
        try primitive.geluErfF32(
            input: mlpHidden,
            count: try modelProduct(tokenCount, config.mlpHiddenSize),
            output: mlpHidden
        )
        let mlpOutput = try linear(
            input: mlpHidden, checkpoint: checkpoint, name: "\(prefix).mlp.down_proj",
            rows: tokenCount, inputChannels: config.mlpHiddenSize,
            outputChannels: config.hiddenSize, hasBias: true,
            label: "DINOv3 MLP output"
        )
        let output = try context.makeBuffer(length: tokenBytes, label: "DINOv3 block output")
        try primitive.layerScaleResidualF32(
            residual: afterAttention, branch: mlpOutput, checkpoint: try checkpoint.acquireBuffer(),
            scaleOffset: try offset(
                checkpoint, "\(prefix).layer_scale2.lambda1", [config.hiddenSize]
            ),
            rows: tokenCount, channels: config.hiddenSize, output: output
        )
        return output
    }

    private func linear(
        input: MTLBuffer,
        checkpoint: MappedCheckpoint,
        name: String,
        rows: Int,
        inputChannels: Int,
        outputChannels: Int,
        hasBias: Bool,
        label: String
    ) throws -> MTLBuffer {
        let output = try context.makeBuffer(
            length: try modelBytes(try modelProduct(rows, outputChannels)), label: label
        )
        try dense.linearF32(
            input: input, checkpoint: try checkpoint.acquireBuffer(),
            weightOffset: try offset(
                checkpoint, "\(name).weight", [outputChannels, inputChannels]
            ),
            biasOffset: hasBias
                ? try offset(checkpoint, "\(name).bias", [outputChannels])
                : nil,
            rows: rows, inputChannels: inputChannels,
            outputChannels: outputChannels, output: output
        )
        return output
    }

    private func copyPrefixTokens(
        checkpoint: MappedCheckpoint,
        output: MTLBuffer
    ) throws {
        let hidden = configuration.hiddenSize
        let cls = try tensor(checkpoint, "embeddings.cls_token", shape: [1, 1, hidden])
        let registers = try tensor(
            checkpoint, "embeddings.register_tokens",
            shape: [1, configuration.prefixTokenCount - 1, hidden]
        )
        let clsBytes = try modelBytes(hidden)
        let registerBytes = try modelBytes(
            try modelProduct(configuration.prefixTokenCount - 1, hidden)
        )
        let prefixBytes = try modelSum(clsBytes, registerBytes)
        guard output.length >= prefixBytes else {
            throw NativeRuntimeError.invalidArgument("DINOv3 prefix output is too small")
        }
        let mappedBuffer = try checkpoint.acquireBuffer()
        output.contents().copyMemory(
            from: mappedBuffer.contents().advanced(by: try tensorOffset(cls)),
            byteCount: clsBytes
        )
        output.contents().advanced(by: clsBytes).copyMemory(
            from: mappedBuffer.contents().advanced(by: try tensorOffset(registers)),
            byteCount: registerBytes
        )
        withExtendedLifetime(mappedBuffer) {}
    }

    private func validateCheckpoint(_ checkpoint: MappedCheckpoint) throws {
        let config = configuration
        _ = try tensor(
            checkpoint, "embeddings.patch_embeddings.weight",
            shape: [config.hiddenSize, config.inputChannels, config.patchSize, config.patchSize]
        )
        _ = try tensor(
            checkpoint, "embeddings.patch_embeddings.bias", shape: [config.hiddenSize]
        )
        _ = try tensor(checkpoint, "embeddings.cls_token", shape: [1, 1, config.hiddenSize])
        _ = try tensor(
            checkpoint, "embeddings.register_tokens",
            shape: [1, config.prefixTokenCount - 1, config.hiddenSize]
        )
        for layer in 0..<config.layerCount {
            let prefix = "layer.\(layer)"
            for name in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
                         "attention.q_proj.bias", "attention.v_proj.bias",
                         "attention.o_proj.bias", "layer_scale1.lambda1",
                         "layer_scale2.lambda1", "mlp.down_proj.bias"] {
                _ = try tensor(checkpoint, "\(prefix).\(name)", shape: [config.hiddenSize])
            }
            _ = try tensor(
                checkpoint, "\(prefix).mlp.up_proj.bias", shape: [config.mlpHiddenSize]
            )
            for name in ["attention.q_proj.weight", "attention.k_proj.weight",
                         "attention.v_proj.weight", "attention.o_proj.weight"] {
                _ = try tensor(
                    checkpoint, "\(prefix).\(name)",
                    shape: [config.hiddenSize, config.hiddenSize]
                )
            }
            _ = try tensor(
                checkpoint, "\(prefix).mlp.up_proj.weight",
                shape: [config.mlpHiddenSize, config.hiddenSize]
            )
            _ = try tensor(
                checkpoint, "\(prefix).mlp.down_proj.weight",
                shape: [config.hiddenSize, config.mlpHiddenSize]
            )
        }
    }

    private func tensor(
        _ checkpoint: MappedCheckpoint,
        _ name: String,
        shape: [Int]
    ) throws -> TensorDescriptor {
        let descriptor = try checkpoint.descriptor(named: name)
        guard descriptor.dtype == .f32,
              descriptor.shape == shape.map(UInt64.init) else {
            throw NativeRuntimeError.invalidArgument(
                "DINOv3 tensor \(name) has the wrong dtype or shape"
            )
        }
        return descriptor
    }

    private func offset(
        _ checkpoint: MappedCheckpoint,
        _ name: String,
        _ shape: [Int]
    ) throws -> Int {
        try tensorOffset(try tensor(checkpoint, name, shape: shape))
    }

    private func tensorOffset(_ descriptor: TensorDescriptor) throws -> Int {
        guard descriptor.fileOffset <= UInt64(Int.max) else {
            throw NativeRuntimeError.invalidArgument("DINOv3 tensor offset exceeds Int")
        }
        return Int(descriptor.fileOffset)
    }

    private func snapshot(
        stage: String,
        buffer: MTLBuffer,
        count: Int,
        trace: DINOv3Trace?
    ) {
        guard let trace else { return }
        trace(stage, Array(UnsafeBufferPointer(
            start: buffer.contents().assumingMemoryBound(to: Float.self), count: count
        )))
    }
}

private func modelProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
    let value = lhs.multipliedReportingOverflow(by: rhs)
    guard !value.overflow else {
        throw NativeRuntimeError.invalidArgument("DINOv3 tensor size overflow")
    }
    return value.partialValue
}

private func modelSum(_ lhs: Int, _ rhs: Int) throws -> Int {
    let value = lhs.addingReportingOverflow(rhs)
    guard !value.overflow else {
        throw NativeRuntimeError.invalidArgument("DINOv3 tensor range overflow")
    }
    return value.partialValue
}

private func modelBytes(_ elements: Int) throws -> Int {
    try modelProduct(elements, MemoryLayout<Float>.stride)
}
