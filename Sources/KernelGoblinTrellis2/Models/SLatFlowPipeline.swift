import Metal

public struct SLatFlowSampleResult: @unchecked Sendable {
    public let latent: MTLBuffer
    public let modelCallCount: Int
}

public final class SLatFlowPipeline: @unchecked Sendable {
    private let context: MetalContext
    private let pipelineMath: SLatPipelineMath

    public init(context: MetalContext) throws {
        guard context.arena != nil else {
            throw NativeRuntimeError.invalidArgument(
                "SLat sampling requires a hard-bounded Metal arena"
            )
        }
        self.context = context
        self.pipelineMath = SLatPipelineMath(context: context)
    }

    public func sampleSparseStructureF32(
        noise: MTLBuffer, coordinates: MTLBuffer,
        positiveConditioning: MTLBuffer, negativeConditioning: MTLBuffer,
        checkpoint: MappedCheckpoint, conditioningTokens: Int,
        parameters: FlowEulerParameters,
        modelTrace: ((_ call: Int, _ pass: FlowConditioningPass, _ output: MTLBuffer) -> Void)? = nil,
        samplerTrace: ((_ step: Int, _ state: MTLBuffer) -> Void)? = nil
    ) throws -> SLatFlowSampleResult {
        let tokens = 16 * 16 * 16
        let flow = try SLatFlow(context: context, configuration: .sparseStructure)
        let layout = try AttentionSegments(offsets: [0, tokens])
        var call = 0
        let result = try FlowEulerSampler(
            context: context, parameters: parameters
        ).sampleF32(
            noise: noise, layout: layout, channels: 8,
            trace: { step, state, _ in samplerTrace?(step, state) },
            predictor: { state, modelTimestep, pass in
                let conditioning = pass == .positive
                    ? positiveConditioning : negativeConditioning
                let output = try flow.forwardF32(
                    input: state, timestep: try self.scalarBuffer(modelTimestep),
                    conditioning: conditioning, coordinates: coordinates,
                    checkpoint: checkpoint, tokens: tokens,
                    conditioningTokens: conditioningTokens
                )
                modelTrace?(call, pass, output)
                call += 1
                return output
            }
        )
        return SLatFlowSampleResult(
            latent: result.samples, modelCallCount: result.modelCallCount
        )
    }

    public func sampleShapeF32(
        noise: MTLBuffer, coordinates: MTLBuffer,
        positiveConditioning: MTLBuffer, negativeConditioning: MTLBuffer,
        checkpoint: MappedCheckpoint, tokens: Int, conditioningTokens: Int,
        parameters: FlowEulerParameters,
        modelTrace: ((_ call: Int, _ pass: FlowConditioningPass, _ output: MTLBuffer) -> Void)? = nil,
        samplerTrace: ((_ step: Int, _ state: MTLBuffer) -> Void)? = nil
    ) throws -> SLatFlowSampleResult {
        let flow = try SLatFlow(context: context, configuration: .shape)
        let layout = try AttentionSegments(offsets: [0, tokens])
        var call = 0
        let result = try FlowEulerSampler(
            context: context, parameters: parameters
        ).sampleF32(
            noise: noise, layout: layout, channels: 32,
            trace: { step, state, _ in samplerTrace?(step, state) },
            predictor: { state, modelTimestep, pass in
                let timestep = try self.scalarBuffer(modelTimestep)
                let conditioning = pass == .positive
                    ? positiveConditioning : negativeConditioning
                let output = try flow.forwardF32(
                    input: state, timestep: timestep, conditioning: conditioning,
                    coordinates: coordinates, checkpoint: checkpoint, tokens: tokens,
                    conditioningTokens: conditioningTokens
                )
                modelTrace?(call, pass, output)
                call += 1
                return output
            }
        )
        return SLatFlowSampleResult(
            latent: try pipelineMath.denormalizeShapeF32(result.samples, tokens: tokens),
            modelCallCount: result.modelCallCount
        )
    }

    public func sampleTextureF32(
        noise: MTLBuffer, shapeLatent: MTLBuffer, coordinates: MTLBuffer,
        positiveConditioning: MTLBuffer, checkpoint: MappedCheckpoint,
        tokens: Int, conditioningTokens: Int,
        parameters: FlowEulerParameters,
        modelTrace: ((_ call: Int, _ output: MTLBuffer) -> Void)? = nil,
        samplerTrace: ((_ step: Int, _ state: MTLBuffer) -> Void)? = nil
    ) throws -> SLatFlowSampleResult {
        guard parameters.guidanceStrength == 1 else {
            throw NativeRuntimeError.invalidArgument(
                "the pinned texture sampler requires guidance strength 1"
            )
        }
        let flow = try SLatFlow(context: context, configuration: .texture)
        let layout = try AttentionSegments(offsets: [0, tokens])
        var call = 0
        let result = try FlowEulerSampler(
            context: context, parameters: parameters
        ).sampleF32(
            noise: noise, layout: layout, channels: 32,
            trace: { step, state, _ in samplerTrace?(step, state) },
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
                    conditioning: positiveConditioning, coordinates: coordinates,
                    checkpoint: checkpoint, tokens: tokens,
                    conditioningTokens: conditioningTokens
                )
                modelTrace?(call, output)
                call += 1
                return output
            }
        )
        return SLatFlowSampleResult(
            latent: try pipelineMath.denormalizeTextureF32(result.samples, tokens: tokens),
            modelCallCount: result.modelCallCount
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
}
