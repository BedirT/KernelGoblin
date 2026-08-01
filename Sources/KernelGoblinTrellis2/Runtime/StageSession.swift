import Foundation
import Metal

public enum StageLifecycleEvent: Equatable, Sendable {
    case queueDrained
    case arenaReleased
    case checkpointUnmapped
}

public enum StageSessionError: Error, CustomStringConvertible, Equatable {
    case alreadyClosed
    case arenaStillLive(Int)
    case arenaNotReleased
    case cleanupAfterBodyFailure(body: String, cleanup: String)
    case initializationAndCleanupFailure(initialization: String, cleanup: String)

    public var description: String {
        switch self {
        case .alreadyClosed:
            "stage session is already closed"
        case .arenaStillLive(let bytes):
            "stage arena still owns \(bytes) bytes"
        case .arenaNotReleased:
            "stage arena remained alive after its context was detached"
        case .cleanupAfterBodyFailure(let body, let cleanup):
            "stage body failed (\(body)) and cleanup also failed (\(cleanup))"
        case .initializationAndCleanupFailure(let initialization, let cleanup):
            "stage initialization failed (\(initialization)) and cleanup also failed (\(cleanup))"
        }
    }
}

public struct StageSample: @unchecked Sendable {
    public let latent: MTLBuffer
    public let modelCallCount: Int
}

public struct StageConditioning: @unchecked Sendable {
    public let conditioning: MTLBuffer
    public let tokenCount: Int
    public let hiddenSize: Int
}

public struct SparseStructureStageResult: @unchecked Sendable {
    public let logits: MTLBuffer
    public let occupancy: SparseOccupancyGrid
    public let pooledOccupancy: SparseOccupancyGrid

    public var coordinates: [SparseStructureCoordinate] {
        pooledOccupancy.coordinates()
    }

    public var highResolutionCoordinates: [SparseStructureCoordinate] {
        occupancy.coordinates()
    }
}

public struct ShapeDecoderStageResult: @unchecked Sendable {
    public let rawHead: MTLBuffer
    public let coordinates: [SparseStructureCoordinate]
    public let spatialShape: SparseSpatialShape
    public let subdivisionGuides: [SparseSubdivision2x]
}

public struct TextureDecoderStageResult: @unchecked Sendable {
    public let pbrFields: MTLBuffer
    public let coordinates: [SparseStructureCoordinate]
    public let spatialShape: SparseSpatialShape
}

public struct ShapeEncoderStageResult: @unchecked Sendable {
    public let latent: MTLBuffer
    public let coordinates: [SparseStructureCoordinate]
    public let spatialShape: SparseSpatialShape
    public let subdivisionGuides: [SparseSubdivision2x]
}

public typealias ShapeStageModelTrace = (
    _ call: Int, _ pass: FlowConditioningPass, _ values: [Float]
) -> Void
public typealias TextureStageModelTrace = (_ call: Int, _ values: [Float]) -> Void
public typealias StageSamplerTrace = (_ step: Int, _ values: [Float]) -> Void

public final class StageSession {
    public let device: MTLDevice

    private var context: MetalContext?
    private var checkpoint: MappedCheckpoint?
    private var flowPipeline: SLatFlowPipeline?
    private var detachedSnapshot: MetalMemorySnapshot?
    private var detachedArenaWitness: WeakArena?
    private var closeSnapshot: MetalMemorySnapshot?
    private let lifecycleObserver: ((StageLifecycleEvent) -> Void)?

    public init(
        checkpointURL: URL,
        expectedCheckpointSHA256: String? = nil,
        arenaCapacity: Int,
        lifecycleObserver: ((StageLifecycleEvent) -> Void)? = nil
    ) throws {
        let context = try MetalContext(arenaCapacity: arenaCapacity)
        let checkpoint = try MappedCheckpoint(url: checkpointURL, device: context.device)
        let flowPipeline: SLatFlowPipeline
        do {
            if let expectedCheckpointSHA256 {
                let actualSHA256 = try autoreleasepool { try checkpoint.sha256() }
                if actualSHA256 != expectedCheckpointSHA256 {
                    throw NativeRuntimeError.invalidArgument("stage checkpoint SHA-256 mismatch")
                }
            }
            flowPipeline = try SLatFlowPipeline(context: context)
        } catch let initializationError {
            do {
                try checkpoint.close()
                lifecycleObserver?(.checkpointUnmapped)
            } catch let cleanupError {
                throw StageSessionError.initializationAndCleanupFailure(
                    initialization: String(describing: initializationError),
                    cleanup: String(describing: cleanupError)
                )
            }
            throw initializationError
        }
        self.device = context.device
        self.context = context
        self.checkpoint = checkpoint
        self.flowPipeline = flowPipeline
        self.lifecycleObserver = lifecycleObserver
    }

    public static func withSession<Result>(
        checkpointURL: URL,
        expectedCheckpointSHA256: String? = nil,
        arenaCapacity: Int,
        lifecycleObserver: ((StageLifecycleEvent) -> Void)? = nil,
        _ body: (StageSession) throws -> Result
    ) throws -> Result {
        let session = try StageSession(
            checkpointURL: checkpointURL,
            expectedCheckpointSHA256: expectedCheckpointSHA256,
            arenaCapacity: arenaCapacity,
            lifecycleObserver: lifecycleObserver
        )
        let result: Result
        do {
            result = try body(session)
        } catch let bodyError {
            do {
                try session.close()
            } catch let cleanupError {
                throw StageSessionError.cleanupAfterBodyFailure(
                    body: String(describing: bodyError),
                    cleanup: String(describing: cleanupError)
                )
            }
            throw bodyError
        }
        try session.close()
        return result
    }

    public func encodeDINOv3F32(
        normalizedImage: MTLBuffer,
        imageHeight: Int,
        imageWidth: Int,
        trace: DINOv3Trace? = nil
    ) throws -> StageConditioning {
        try autoreleasepool {
            let (context, checkpoint, _) = try activeRuntime()
            let result = try DINOv3Conditioner(context: context).encodeNormalizedImageF32(
                image: normalizedImage,
                imageHeight: imageHeight,
                imageWidth: imageWidth,
                checkpoint: checkpoint,
                trace: trace
            )
            return StageConditioning(
                conditioning: try standaloneCopy(
                    result.conditioning,
                    byteCount: try stageByteCount(result.tokenCount, result.hiddenSize),
                    label: "DINOv3 standalone conditioning"
                ),
                tokenCount: result.tokenCount,
                hiddenSize: result.hiddenSize
            )
        }
    }

    public func sampleShapeF32(
        noise: MTLBuffer,
        coordinates: MTLBuffer,
        positiveConditioning: MTLBuffer,
        negativeConditioning: MTLBuffer,
        tokens: Int,
        conditioningTokens: Int,
        parameters: FlowEulerParameters,
        modelTrace: ShapeStageModelTrace? = nil,
        samplerTrace: StageSamplerTrace? = nil
    ) throws -> StageSample {
        try autoreleasepool {
            let (_, checkpoint, pipeline) = try activeRuntime()
            let elementCount = try stageElementCount(tokens, 32)
            let result = try pipeline.sampleShapeF32(
                noise: noise,
                coordinates: coordinates,
                positiveConditioning: positiveConditioning,
                negativeConditioning: negativeConditioning,
                checkpoint: checkpoint,
                tokens: tokens,
                conditioningTokens: conditioningTokens,
                parameters: parameters,
                modelTrace: { call, pass, output in
                    modelTrace?(call, pass, Self.values(output, count: elementCount))
                },
                samplerTrace: { step, state in
                    samplerTrace?(step, Self.values(state, count: elementCount))
                }
            )
            return StageSample(
                latent: try standaloneCopy(
                    result.latent,
                    byteCount: try stageByteCount(tokens, 32),
                    label: "Shape stage standalone latent"
                ),
                modelCallCount: result.modelCallCount
            )
        }
    }

    public func sampleSparseStructureF32(
        noise: MTLBuffer,
        positiveConditioning: MTLBuffer,
        negativeConditioning: MTLBuffer,
        conditioningTokens: Int,
        parameters: FlowEulerParameters,
        modelTrace: ShapeStageModelTrace? = nil,
        samplerTrace: StageSamplerTrace? = nil
    ) throws -> StageSample {
        try autoreleasepool {
            let (context, checkpoint, pipeline) = try activeRuntime()
            let resolution = 16
            let tokens = resolution * resolution * resolution
            var coordinates = [Int32]()
            coordinates.reserveCapacity(tokens * 4)
            for x in 0..<resolution {
                for y in 0..<resolution {
                    for z in 0..<resolution {
                        coordinates.append(contentsOf: [0, Int32(x), Int32(y), Int32(z)])
                    }
                }
            }
            let coordinateBuffer = try context.makeBuffer(
                length: coordinates.count * MemoryLayout<Int32>.stride,
                label: "Sparse-structure 16-cubed coordinates"
            )
            coordinates.withUnsafeBytes { bytes in
                coordinateBuffer.contents().copyMemory(
                    from: bytes.baseAddress!, byteCount: bytes.count
                )
            }
            let elementCount = try stageElementCount(tokens, 8)
            let result = try pipeline.sampleSparseStructureF32(
                noise: noise, coordinates: coordinateBuffer,
                positiveConditioning: positiveConditioning,
                negativeConditioning: negativeConditioning,
                checkpoint: checkpoint, conditioningTokens: conditioningTokens,
                parameters: parameters,
                modelTrace: { call, pass, output in
                    modelTrace?(call, pass, Self.values(output, count: elementCount))
                },
                samplerTrace: { step, state in
                    samplerTrace?(step, Self.values(state, count: elementCount))
                }
            )
            return StageSample(
                latent: try standaloneCopy(
                    result.latent,
                    byteCount: try stageByteCount(tokens, 8),
                    label: "Sparse-structure standalone latent"
                ),
                modelCallCount: result.modelCallCount
            )
        }
    }

    public func sampleTextureF32(
        noise: MTLBuffer,
        shapeLatent: MTLBuffer,
        coordinates: MTLBuffer,
        positiveConditioning: MTLBuffer,
        tokens: Int,
        conditioningTokens: Int,
        parameters: FlowEulerParameters,
        modelTrace: TextureStageModelTrace? = nil,
        samplerTrace: StageSamplerTrace? = nil
    ) throws -> StageSample {
        try autoreleasepool {
            let (_, checkpoint, pipeline) = try activeRuntime()
            let elementCount = try stageElementCount(tokens, 32)
            let result = try pipeline.sampleTextureF32(
                noise: noise,
                shapeLatent: shapeLatent,
                coordinates: coordinates,
                positiveConditioning: positiveConditioning,
                checkpoint: checkpoint,
                tokens: tokens,
                conditioningTokens: conditioningTokens,
                parameters: parameters,
                modelTrace: { call, output in
                    modelTrace?(call, Self.values(output, count: elementCount))
                },
                samplerTrace: { step, state in
                    samplerTrace?(step, Self.values(state, count: elementCount))
                }
            )
            return StageSample(
                latent: try standaloneCopy(
                    result.latent,
                    byteCount: try stageByteCount(tokens, 32),
                    label: "Texture stage standalone latent"
                ),
                modelCallCount: result.modelCallCount
            )
        }
    }

    public func decodeSparseStructureF32(
        latent: MTLBuffer, inputResolution: Int = 16
    ) throws -> SparseStructureStageResult {
        try autoreleasepool {
            let (context, checkpoint, _) = try activeRuntime()
            let outputResolution = try stageProduct(inputResolution, 4)
            let outputCount = try stageElementCount(
                try stageProduct(try stageProduct(outputResolution, outputResolution),
                                 outputResolution),
                1
            )
            let decoded = try SparseStructureDecoder(context: context).decodeF32(
                latent: latent, inputResolution: inputResolution, checkpoint: checkpoint
            )
            let standalone = try standaloneCopy(
                decoded, byteCount: try stageProduct(outputCount, 4),
                label: "Sparse-structure standalone logits"
            )
            let values = Self.values(standalone, count: outputCount)
            let occupancy = try SparseStructureOccupancy.threshold(
                logits: values, resolution: outputResolution
            )
            return SparseStructureStageResult(
                logits: standalone,
                occupancy: occupancy,
                pooledOccupancy: try SparseStructureOccupancy.downsampleMax2(occupancy)
            )
        }
    }

    public func decodeShapeF32(
        latent: MTLBuffer,
        coordinates: [SparseStructureCoordinate],
        spatialShape: SparseSpatialShape
    ) throws -> ShapeDecoderStageResult {
        try autoreleasepool {
            let (context, checkpoint, _) = try activeRuntime()
            let result = try ShapeSparseDecoder(context: context)(
                latent: latent, coordinates: coordinates,
                spatialShape: spatialShape, checkpoint: checkpoint
            )
            return ShapeDecoderStageResult(
                rawHead: try standaloneCopy(
                    result.rawHead,
                    byteCount: try stageByteCount(result.coordinates.count, 7),
                    label: "Shape decoder standalone raw head"
                ),
                coordinates: result.coordinates,
                spatialShape: result.spatialShape,
                subdivisionGuides: result.subdivisionGuides
            )
        }
    }

    public func decodeTextureF32(
        latent: MTLBuffer,
        coordinates: [SparseStructureCoordinate],
        spatialShape: SparseSpatialShape,
        subdivisionGuides: [SparseSubdivision2x]
    ) throws -> TextureDecoderStageResult {
        try autoreleasepool {
            let (context, checkpoint, _) = try activeRuntime()
            let result = try TextureSparseDecoder(context: context)(
                latent: latent, coordinates: coordinates,
                spatialShape: spatialShape,
                subdivisionGuides: subdivisionGuides, checkpoint: checkpoint
            )
            return TextureDecoderStageResult(
                pbrFields: try standaloneCopy(
                    result.pbrFields,
                    byteCount: try stageByteCount(result.coordinates.count, 6),
                    label: "Texture decoder standalone PBR fields"
                ),
                coordinates: result.coordinates,
                spatialShape: result.spatialShape
            )
        }
    }

    public func encodeShapeF32(
        input: MTLBuffer,
        coordinates: [SparseStructureCoordinate],
        spatialShape: SparseSpatialShape
    ) throws -> ShapeEncoderStageResult {
        try autoreleasepool {
            let (context, checkpoint, _) = try activeRuntime()
            let result = try ShapeSparseEncoder(context: context)(
                input: input, coordinates: coordinates,
                spatialShape: spatialShape, checkpoint: checkpoint
            )
            return ShapeEncoderStageResult(
                latent: try standaloneCopy(
                    result.latent,
                    byteCount: try stageByteCount(result.coordinates.count, 32),
                    label: "Shape encoder standalone latent"
                ),
                coordinates: result.coordinates,
                spatialShape: result.spatialShape,
                subdivisionGuides: result.subdivisionGuides
            )
        }
    }

    @discardableResult
    public func close() throws -> MetalMemorySnapshot {
        if let closeSnapshot { return closeSnapshot }
        let snapshot: MetalMemorySnapshot
        if let detachedSnapshot {
            snapshot = detachedSnapshot
        } else {
            let (newSnapshot, arenaWitness) = try autoreleasepool { try detachContext() }
            detachedSnapshot = newSnapshot
            detachedArenaWitness = arenaWitness
            snapshot = newSnapshot
        }
        try finishArenaRelease()
        guard let checkpoint else {
            throw StageSessionError.alreadyClosed
        }
        try checkpoint.close()
        self.checkpoint = nil
        lifecycleObserver?(.checkpointUnmapped)
        closeSnapshot = snapshot
        return snapshot
    }

    private func activeRuntime() throws -> (
        MetalContext, MappedCheckpoint, SLatFlowPipeline
    ) {
        guard closeSnapshot == nil,
              let context, let checkpoint, let flowPipeline else {
            throw StageSessionError.alreadyClosed
        }
        return (context, checkpoint, flowPipeline)
    }

    private func detachContext() throws -> (MetalMemorySnapshot, WeakArena) {
        guard let context, let arena = context.arena else {
            throw StageSessionError.alreadyClosed
        }
        try context.waitUntilIdle()
        lifecycleObserver?(.queueDrained)
        flowPipeline = nil
        let snapshot = arena.snapshot()
        guard snapshot.usedBytes == 0 else {
            throw StageSessionError.arenaStillLive(snapshot.usedBytes)
        }
        let witness = WeakArena(arena)
        self.context = nil
        return (snapshot, witness)
    }

    private func finishArenaRelease() throws {
        guard let detachedArenaWitness else { return }
        guard detachedArenaWitness.value == nil else {
            throw StageSessionError.arenaNotReleased
        }
        lifecycleObserver?(.arenaReleased)
        self.detachedArenaWitness = nil
    }

    private func standaloneCopy(
        _ source: MTLBuffer,
        byteCount: Int,
        label: String
    ) throws -> MTLBuffer {
        guard source.length >= byteCount, source.storageMode != .private,
              let output = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw NativeRuntimeError.allocationFailed("could not copy standalone stage output")
        }
        output.label = label
        output.contents().copyMemory(from: source.contents(), byteCount: byteCount)
        return output
    }

    private static func values(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().assumingMemoryBound(to: Float.self),
            count: count
        ))
    }

#if DEBUG
    func withRuntimeForTesting<Result>(
        _ body: (MetalContext, MappedCheckpoint) throws -> Result
    ) throws -> Result {
        let (context, checkpoint, _) = try activeRuntime()
        return try autoreleasepool { try body(context, checkpoint) }
    }

    func standaloneCopyForTesting(_ source: MTLBuffer, byteCount: Int) throws -> MTLBuffer {
        try standaloneCopy(source, byteCount: byteCount, label: "Stage test standalone copy")
    }

    var checkpointIsMappedForTesting: Bool { checkpoint?.isMapped == true }
    var checkpointBufferForTesting: MTLBuffer? {
        guard let checkpoint else { return nil }
        return try? checkpoint.acquireBuffer()
    }
#endif
}

private final class WeakArena {
    weak var value: MetalBufferArena?

    init(_ value: MetalBufferArena) {
        self.value = value
    }
}

private func stageElementCount(_ rows: Int, _ channels: Int) throws -> Int {
    let result = rows.multipliedReportingOverflow(by: channels)
    guard rows > 0, channels > 0, !result.overflow else {
        throw NativeRuntimeError.invalidArgument("stage tensor size overflows Int")
    }
    return result.partialValue
}

private func stageByteCount(_ rows: Int, _ channels: Int) throws -> Int {
    let elements = try stageElementCount(rows, channels)
    let result = elements.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
    guard !result.overflow else {
        throw NativeRuntimeError.invalidArgument("stage tensor byte count overflows Int")
    }
    return result.partialValue
}

private func stageProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard lhs > 0, rhs > 0, !result.overflow else {
        throw NativeRuntimeError.invalidArgument("stage size overflows Int")
    }
    return result.partialValue
}
