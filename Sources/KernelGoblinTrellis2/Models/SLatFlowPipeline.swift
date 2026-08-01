import Metal

public struct SLatFlowSampleResult: @unchecked Sendable {
    public let latent: MTLBuffer
    public let modelCallCount: Int
    public let crossKVCacheStats: SLatCrossKVCacheStats
}

public final class SLatFlowPipeline: @unchecked Sendable {
    private let context: MetalContext
    private let pipelineMath: SLatPipelineMath
    private let primitives: PrimitiveKernel

    public init(context: MetalContext) throws {
        guard context.arena != nil else {
            throw NativeRuntimeError.invalidArgument(
                "SLat sampling requires a hard-bounded Metal arena"
            )
        }
        self.context = context
        self.pipelineMath = SLatPipelineMath(context: context)
        self.primitives = try PrimitiveKernel(context: context)
    }

    public func sampleSparseStructureF32(
        noise: MTLBuffer, coordinates: MTLBuffer,
        positiveConditioning: MTLBuffer, negativeConditioning: MTLBuffer,
        checkpoint: MappedCheckpoint, conditioningTokens: Int,
        parameters: FlowEulerParameters,
        cacheCrossKV: Bool = false,
        modelTrace: ((_ call: Int, _ pass: FlowConditioningPass, _ output: MTLBuffer) -> Void)? = nil,
        samplerTrace: ((_ step: Int, _ state: MTLBuffer) -> Void)? = nil
    ) throws -> SLatFlowSampleResult {
        let tokens = 16 * 16 * 16
        let flow = try SLatFlow(context: context, configuration: .sparseStructure)
        let roundedPositive = try roundedConditioning(
            positiveConditioning, tokens: conditioningTokens
        )
        let roundedNegative = try roundedConditioning(
            negativeConditioning, tokens: conditioningTokens
        )
        let positiveCache = SLatCachedConditioning(
            buffer: roundedPositive, checkpoint: checkpoint, tokens: conditioningTokens
        )
        let negativeCache = SLatCachedConditioning(
            buffer: roundedNegative, checkpoint: checkpoint, tokens: conditioningTokens
        )
        let layout = try AttentionSegments(offsets: [0, tokens])
        var call = 0
        let result = try FlowEulerSampler(
            context: context, parameters: parameters
        ).sampleF32(
            noise: noise, layout: layout, channels: 8,
            trace: samplerTrace.map { callback in
                { step, state, _ in callback(step, state) }
            },
            predictor: { state, modelTimestep, pass in
                let conditioning = pass == .positive
                    ? roundedPositive : roundedNegative
                let cachedConditioning = pass == .positive
                    ? positiveCache : negativeCache
                let output = try flow.forwardF32(
                    input: state, timestep: try self.scalarBuffer(modelTimestep),
                    conditioning: conditioning, coordinates: coordinates,
                    checkpoint: checkpoint, tokens: tokens,
                    conditioningTokens: conditioningTokens,
                    conditioningIsRounded: true,
                    cachedConditioning: cacheCrossKV ? cachedConditioning : nil
                )
                modelTrace?(call, pass, output)
                call += 1
                return output
            }
        )
        return SLatFlowSampleResult(
            latent: result.samples, modelCallCount: result.modelCallCount,
            crossKVCacheStats: flow.crossKVCacheStats
        )
    }

    public func sampleShapeF32(
        noise: MTLBuffer, coordinates: MTLBuffer,
        positiveConditioning: MTLBuffer, negativeConditioning: MTLBuffer,
        checkpoint: MappedCheckpoint, tokens: Int, conditioningTokens: Int,
        parameters: FlowEulerParameters,
        cacheCrossKV: Bool = false,
        modelTrace: ((_ call: Int, _ pass: FlowConditioningPass, _ output: MTLBuffer) -> Void)? = nil,
        samplerTrace: ((_ step: Int, _ state: MTLBuffer) -> Void)? = nil
    ) throws -> SLatFlowSampleResult {
        let flow = try SLatFlow(context: context, configuration: .shape)
        let roundedPositive = try roundedConditioning(
            positiveConditioning, tokens: conditioningTokens
        )
        let roundedNegative = try roundedConditioning(
            negativeConditioning, tokens: conditioningTokens
        )
        let positiveCache = SLatCachedConditioning(
            buffer: roundedPositive, checkpoint: checkpoint, tokens: conditioningTokens
        )
        let negativeCache = SLatCachedConditioning(
            buffer: roundedNegative, checkpoint: checkpoint, tokens: conditioningTokens
        )
        let layout = try AttentionSegments(offsets: [0, tokens])
        var call = 0
        let result = try FlowEulerSampler(
            context: context, parameters: parameters
        ).sampleF32(
            noise: noise, layout: layout, channels: 32,
            trace: samplerTrace.map { callback in
                { step, state, _ in callback(step, state) }
            },
            predictor: { state, modelTimestep, pass in
                let timestep = try self.scalarBuffer(modelTimestep)
                let conditioning = pass == .positive
                    ? roundedPositive : roundedNegative
                let cachedConditioning = pass == .positive
                    ? positiveCache : negativeCache
                let output = try flow.forwardF32(
                    input: state, timestep: timestep, conditioning: conditioning,
                    coordinates: coordinates, checkpoint: checkpoint, tokens: tokens,
                    conditioningTokens: conditioningTokens,
                    conditioningIsRounded: true,
                    cachedConditioning: cacheCrossKV ? cachedConditioning : nil
                )
                modelTrace?(call, pass, output)
                call += 1
                return output
            }
        )
        return SLatFlowSampleResult(
            latent: try pipelineMath.denormalizeShapeF32(result.samples, tokens: tokens),
            modelCallCount: result.modelCallCount,
            crossKVCacheStats: flow.crossKVCacheStats
        )
    }

    public func sampleTextureF32(
        noise: MTLBuffer, shapeLatent: MTLBuffer, coordinates: MTLBuffer,
        positiveConditioning: MTLBuffer, checkpoint: MappedCheckpoint,
        tokens: Int, conditioningTokens: Int,
        parameters: FlowEulerParameters,
        cacheCrossKV: Bool = false,
        modelTrace: ((_ call: Int, _ output: MTLBuffer) -> Void)? = nil,
        samplerTrace: ((_ step: Int, _ state: MTLBuffer) -> Void)? = nil
    ) throws -> SLatFlowSampleResult {
        guard parameters.guidanceStrength == 1 else {
            throw NativeRuntimeError.invalidArgument(
                "the pinned texture sampler requires guidance strength 1"
            )
        }
        let flow = try SLatFlow(context: context, configuration: .texture)
        let roundedPositive = try roundedConditioning(
            positiveConditioning, tokens: conditioningTokens
        )
        let positiveCache = SLatCachedConditioning(
            buffer: roundedPositive, checkpoint: checkpoint, tokens: conditioningTokens
        )
        let layout = try AttentionSegments(offsets: [0, tokens])
        var call = 0
        let result = try FlowEulerSampler(
            context: context, parameters: parameters
        ).sampleF32(
            noise: noise, layout: layout, channels: 32,
            trace: samplerTrace.map { callback in
                { step, state, _ in callback(step, state) }
            },
            predictor: { state, modelTimestep, pass in
                guard pass == .positive else {
                    throw NativeRuntimeError.invalidArgument(
                        "the pinned texture sampler is positive-only"
                    )
                }
                let input = try self.pipelineMath.makeTextureInputF32(
                    noise: state, shape: shapeLatent, tokens: tokens
                )
                let output = try flow.forwardF32(
                    input: input, timestep: try self.scalarBuffer(modelTimestep),
                    conditioning: roundedPositive, coordinates: coordinates,
                    checkpoint: checkpoint, tokens: tokens,
                    conditioningTokens: conditioningTokens,
                    conditioningIsRounded: true,
                    cachedConditioning: cacheCrossKV ? positiveCache : nil
                )
                modelTrace?(call, output)
                call += 1
                return output
            }
        )
        return SLatFlowSampleResult(
            latent: try pipelineMath.denormalizeTextureF32(result.samples, tokens: tokens),
            modelCallCount: result.modelCallCount,
            crossKVCacheStats: flow.crossKVCacheStats
        )
    }

    private func scalarBuffer(_ value: Float) throws -> MTLBuffer {
        var value = value
        let buffer = try context.makeBuffer(
            length: MemoryLayout<Float>.stride, label: "Flow Euler timestep"
        )
        buffer.contents().copyMemory(from: &value, byteCount: MemoryLayout<Float>.stride)
        return buffer
    }

    private func roundedConditioning(_ input: MTLBuffer, tokens: Int) throws -> MTLBuffer {
        let elements = tokens.multipliedReportingOverflow(by: SLatCrossAttention.contextChannels)
        let bytes = elements.partialValue.multipliedReportingOverflow(by: 4)
        guard tokens > 0, !elements.overflow, !bytes.overflow,
              input.length >= bytes.partialValue else {
            throw NativeRuntimeError.invalidArgument("invalid flow conditioning buffer")
        }
        let output = try context.makeBuffer(
            length: bytes.partialValue, label: "Flow cached rounded conditioning"
        )
        try primitives.roundBF16F32(
            input: input, count: elements.partialValue, output: output
        )
        return output
    }
}
