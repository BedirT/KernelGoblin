import Foundation
import Metal

public struct Trellis2CheckpointSet: Sendable {
    public let dino: URL
    public let sparseStructureFlow: URL
    public let sparseStructureDecoder: URL
    public let shapeFlow: URL
    public let textureFlow: URL
    public let shapeDecoder: URL
    public let textureDecoder: URL

    public init(
        dino: URL,
        sparseStructureFlow: URL,
        sparseStructureDecoder: URL,
        shapeFlow: URL,
        textureFlow: URL,
        shapeDecoder: URL,
        textureDecoder: URL
    ) {
        self.dino = dino
        self.sparseStructureFlow = sparseStructureFlow
        self.sparseStructureDecoder = sparseStructureDecoder
        self.shapeFlow = shapeFlow
        self.textureFlow = textureFlow
        self.shapeDecoder = shapeDecoder
        self.textureDecoder = textureDecoder
    }

    public static func huggingFaceCache(
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
    ) throws -> Self {
        let hub = homeDirectory.appendingPathComponent(
            ".cache/huggingface/hub", isDirectory: true
        )
        let trellis = hub
            .appendingPathComponent("models--microsoft--TRELLIS.2-4B/snapshots")
            .appendingPathComponent(NativeTrellis2Pipeline.trellisWeightsRevision)
            .appendingPathComponent("ckpts")
        let sparseDecoder = hub
            .appendingPathComponent("models--microsoft--TRELLIS-image-large/snapshots")
            .appendingPathComponent("25e0d31ffbebe4b5a97464dd851910efc3002d96")
            .appendingPathComponent("ckpts/ss_dec_conv3d_16l8_fp16.safetensors")
        let dino = hub
            .appendingPathComponent(
                "models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots"
            )
            .appendingPathComponent("ea8dc2863c51be0a264bab82070e3e8836b02d51")
            .appendingPathComponent("model.safetensors")
        let result = Self(
            dino: dino,
            sparseStructureFlow: trellis.appendingPathComponent(
                "ss_flow_img_dit_1_3B_64_bf16.safetensors"
            ),
            sparseStructureDecoder: sparseDecoder,
            shapeFlow: trellis.appendingPathComponent(
                "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors"
            ),
            textureFlow: trellis.appendingPathComponent(
                "slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors"
            ),
            shapeDecoder: trellis.appendingPathComponent(
                "shape_dec_next_dc_f16c32_fp16.safetensors"
            ),
            textureDecoder: trellis.appendingPathComponent(
                "tex_dec_next_dc_f16c32_fp16.safetensors"
            )
        )
        let missing = result.allURLs.filter {
            !FileManager.default.fileExists(atPath: $0.path)
        }
        guard missing.isEmpty else {
            throw NativeRuntimeError.invalidArgument(
                "missing pinned native checkpoint(s): "
                    + missing.map(\.path).joined(separator: ", ")
            )
        }
        return result
    }

    public var allURLs: [URL] {
        [
            dino, sparseStructureFlow, sparseStructureDecoder, shapeFlow,
            textureFlow, shapeDecoder, textureDecoder,
        ]
    }
}

public struct Trellis2TexturingCheckpointSet: Sendable {
    public let dino: URL
    public let shapeEncoder: URL
    public let textureFlow: URL
    public let textureDecoder: URL

    public init(
        dino: URL, shapeEncoder: URL,
        textureFlow: URL, textureDecoder: URL
    ) {
        self.dino = dino
        self.shapeEncoder = shapeEncoder
        self.textureFlow = textureFlow
        self.textureDecoder = textureDecoder
    }

    public static func huggingFaceCache(
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
    ) throws -> Self {
        let generation = try Trellis2CheckpointSet.huggingFaceCache(
            homeDirectory: homeDirectory
        )
        let encoder = homeDirectory.appendingPathComponent(
            ".cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/"
                + NativeTrellis2Pipeline.trellisWeightsRevision
                + "/ckpts/shape_enc_next_dc_f16c32_fp16.safetensors"
        )
        guard FileManager.default.fileExists(atPath: encoder.path) else {
            throw NativeRuntimeError.invalidArgument(
                "missing pinned native shape encoder: \(encoder.path)"
            )
        }
        return Self(
            dino: generation.dino, shapeEncoder: encoder,
            textureFlow: generation.textureFlow,
            textureDecoder: generation.textureDecoder
        )
    }
}

public struct Trellis2MemoryBudget: Equatable, Sendable {
    public var dinoBytes: Int
    public var sparseStructureFlowBytes: Int
    public var sparseStructureDecoderBytes: Int
    public var shapeFlowBytes: Int
    public var textureFlowBytes: Int
    public var shapeEncoderBytes: Int
    public var shapeDecoderBytes: Int
    public var textureDecoderBytes: Int
    public var pbrExportBytes: Int

    public init(
        dinoBytes: Int = 256 * 1024 * 1024,
        sparseStructureFlowBytes: Int = 640 * 1024 * 1024,
        sparseStructureDecoderBytes: Int = 320 * 1024 * 1024,
        shapeFlowBytes: Int = 4 * 1024 * 1024 * 1024,
        textureFlowBytes: Int = 4 * 1024 * 1024 * 1024,
        shapeEncoderBytes: Int = 6 * 1024 * 1024 * 1024,
        shapeDecoderBytes: Int = 6 * 1024 * 1024 * 1024,
        textureDecoderBytes: Int = 6 * 1024 * 1024 * 1024,
        pbrExportBytes: Int = 512 * 1024 * 1024
    ) {
        self.dinoBytes = dinoBytes
        self.sparseStructureFlowBytes = sparseStructureFlowBytes
        self.sparseStructureDecoderBytes = sparseStructureDecoderBytes
        self.shapeFlowBytes = shapeFlowBytes
        self.textureFlowBytes = textureFlowBytes
        self.shapeEncoderBytes = shapeEncoderBytes
        self.shapeDecoderBytes = shapeDecoderBytes
        self.textureDecoderBytes = textureDecoderBytes
        self.pbrExportBytes = pbrExportBytes
    }
}

public struct Trellis2GenerationOptions: Equatable, Sendable {
    public var seed: UInt64
    public var steps: Int
    public var textureSize: Int
    public var uvPolicy: UVPreparationPolicy
    public var alphaMode: String
    public var memory: Trellis2MemoryBudget

    public init(
        seed: UInt64 = 42,
        steps: Int = 12,
        textureSize: Int = 2048,
        uvPolicy: UVPreparationPolicy = .regenerate,
        alphaMode: String = "OPAQUE",
        memory: Trellis2MemoryBudget = Trellis2MemoryBudget()
    ) {
        self.seed = seed
        self.steps = steps
        self.textureSize = textureSize
        self.uvPolicy = uvPolicy
        self.alphaMode = alphaMode
        self.memory = memory
    }
}

public struct NativeStageEvidence: Codable, Equatable, Sendable {
    public let name: String
    public let checkpointSHA256: String
    public let elapsedSeconds: Double
    public let arenaCapacityBytes: Int
    public let arenaPeakBytes: Int
    public let arenaLiveAfterCloseBytes: Int
    public let cumulativeRequestedBytes: Int
    public let allocationCount: Int
}

public struct Trellis2GenerationEvidence: Codable, Equatable, Sendable {
    public let runtime: String
    public let pipeline: String
    public let upstreamSourceRevision: String
    public let weightsRevision: String
    public let device: String
    public let operatingSystem: String
    public let inputSHA256: String?
    public let imagePreprocessing: String
    public let usedMeaningfulAlpha: Bool?
    public let seed: UInt64
    public let noiseAlgorithm: String
    public let steps: Int
    public let conditioningTokens: Int
    public let sparseCoordinateCount: Int
    public let shapeCoordinateCount: Int
    public let meshVertexCount: Int
    public let meshFaceCount: Int
    public let outputVertexCount: Int
    public let outputFaceCount: Int
    public let uvImplementation: String
    public let uvExactUpstreamTier: Bool
    public let uvFingerprintSHA256: String
    public let textureSize: Int
    public let coveredTexels: Int
    public let glbBytes: Int
    public let glbReloadValidated: Bool
    public let outputSHA256: String
    public let stages: [NativeStageEvidence]
}

public struct Trellis2TexturingOptions: Equatable, Sendable {
    public var seed: UInt64
    public var steps: Int
    public var textureSize: Int
    public var uvPolicy: UVPreparationPolicy
    public var alphaMode: String
    public var memory: Trellis2MemoryBudget

    public init(
        seed: UInt64 = 42, steps: Int = 12, textureSize: Int = 2048,
        uvPolicy: UVPreparationPolicy = .preserveOrGenerate,
        alphaMode: String = "OPAQUE",
        memory: Trellis2MemoryBudget = Trellis2MemoryBudget()
    ) {
        self.seed = seed
        self.steps = steps
        self.textureSize = textureSize
        self.uvPolicy = uvPolicy
        self.alphaMode = alphaMode
        self.memory = memory
    }
}

public struct Trellis2TexturingEvidence: Codable, Equatable, Sendable {
    public let runtime: String
    public let pipeline: String
    public let upstreamSourceRevision: String
    public let weightsRevision: String
    public let device: String
    public let operatingSystem: String
    public let imageSHA256: String
    public let meshSHA256: String
    public let imagePreprocessing: String
    public let seed: UInt64
    public let noiseAlgorithm: String
    public let steps: Int
    public let sourceVertexCount: Int
    public let sourceFaceCount: Int
    public let outputVertexCount: Int
    public let outputFaceCount: Int
    public let voxelCoordinateCount: Int
    public let latentCoordinateCount: Int
    public let pbrCoordinateCount: Int
    public let uvImplementation: String
    public let uvExactUpstreamTier: Bool
    public let uvFingerprintSHA256: String
    public let textureSize: Int
    public let coveredTexels: Int
    public let glbBytes: Int
    public let glbReloadValidated: Bool
    public let outputSHA256: String
    public let stages: [NativeStageEvidence]
}

public final class NativeTrellis2Pipeline: @unchecked Sendable {
    public static let trellisSourceRevision =
        "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    public static let trellisWeightsRevision =
        "af44b45f2e35a493886929c6d786e563ec68364d"

    public init() {}

    public func generate512(
        imageURL: URL,
        opaquePolicy: TrellisOpaqueImagePolicy = .requireMeaningfulAlpha,
        checkpoints: Trellis2CheckpointSet,
        options: Trellis2GenerationOptions = Trellis2GenerationOptions(),
        outputURL: URL,
        progress: ((String) -> Void)? = nil
    ) throws -> Trellis2GenerationEvidence {
        progress?("image-preprocessing")
        let image = try TrellisImagePreprocessor().load(
            from: imageURL, targetSize: 512, opaquePolicy: opaquePolicy
        )
        return try generate512(
            normalizedImageCHW: image.chw,
            imageWidth: image.width,
            imageHeight: image.height,
            checkpoints: checkpoints,
            options: options,
            outputURL: outputURL,
            sourceInputSHA256: try fileSHA256(at: imageURL),
            imagePreprocessing: image.backgroundRemoval
                + "+crop-premultiply-lanczos-imagenet",
            usedMeaningfulAlpha: image.usedMeaningfulAlpha,
            progress: progress
        )
    }

    public func texture512(
        meshURL: URL,
        imageURL: URL,
        opaquePolicy: TrellisOpaqueImagePolicy = .requireMeaningfulAlpha,
        checkpoints: Trellis2TexturingCheckpointSet,
        options: Trellis2TexturingOptions = Trellis2TexturingOptions(),
        outputURL: URL,
        progress: ((String) -> Void)? = nil
    ) throws -> Trellis2TexturingEvidence {
        guard options.steps > 0, options.textureSize > 1,
              ["OPAQUE", "BLEND", "MASK"].contains(options.alphaMode) else {
            throw NativeRuntimeError.invalidArgument("invalid native texturing options")
        }
        try validateMemoryBudget(options.memory)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        progress?("mesh-loading")
        let sourceMesh = try ModelIOMeshLoader.load(url: meshURL)
        let mesh = try sourceMesh.normalizedForTrellis()
        let uv = try UVPreparation.prepare(
            positions: mesh.positions, faces: mesh.faces,
            suppliedUVs: mesh.suppliedUVs, policy: options.uvPolicy
        )
        progress?("mesh-voxelization")
        let voxelizationStarted = Date()
        let voxelization = try FlexibleDualGridVoxelizer.voxelize(
            vertices: mesh.positions, faces: mesh.faces,
            gridSize: SIMD3(repeating: 512),
            aabbMinimum: SIMD3(repeating: -0.5),
            aabbMaximum: SIMD3(repeating: 0.5)
        )
        guard !voxelization.coordinates.isEmpty else {
            throw NativeRuntimeError.invalidArgument("mesh voxelization produced no coordinates")
        }
        var stageEvidence: [NativeStageEvidence] = [
            NativeStageEvidence(
                name: "mesh-voxelization", checkpointSHA256: "none",
                elapsedSeconds: Date().timeIntervalSince(voxelizationStarted),
                arenaCapacityBytes: 0, arenaPeakBytes: 0,
                arenaLiveAfterCloseBytes: 0, cumulativeRequestedBytes: 0,
                allocationCount: 0
            )
        ]
        progress?("image-preprocessing")
        let image = try TrellisImagePreprocessor().load(
            from: imageURL, targetSize: 512, opaquePolicy: opaquePolicy
        )
        progress?("image-conditioning")
        let dino = try runStage(
            name: "dino-v3", checkpointURL: checkpoints.dino,
            checkpointSHA256: CheckpointHashes.dino,
            arenaCapacity: options.memory.dinoBytes
        ) { session in
            var values = image.chw
            guard let input = session.device.makeBuffer(
                bytes: &values, length: values.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ) else {
                throw NativeRuntimeError.allocationFailed("could not upload normalized image")
            }
            return try session.encodeDINOv3F32(
                normalizedImage: input, imageHeight: 512, imageWidth: 512
            )
        }
        stageEvidence.append(dino.evidence)

        var encoderInputValues: [Float] = []
        encoderInputValues.reserveCapacity(
            try checkedElements(voxelization.coordinates.count, 6)
        )
        for index in voxelization.coordinates.indices {
            let coordinate = voxelization.coordinates[index]
            let offset = voxelization.dualVertices[index] * 512
                - SIMD3<Float>(Float(coordinate.x), Float(coordinate.y), Float(coordinate.z))
                - SIMD3<Float>(repeating: 0.5)
            let flags = voxelization.intersections[index]
            encoderInputValues.append(contentsOf: [
                offset.x, offset.y, offset.z,
                Float(flags.x) - 0.5, Float(flags.y) - 0.5, Float(flags.z) - 0.5,
            ])
        }
        let encoderCoordinates = voxelization.coordinates.map {
            SparseStructureCoordinate(x: $0.x, y: $0.y, z: $0.z)
        }
        guard let encoderInput = dino.value.conditioning.device.makeBuffer(
            bytes: &encoderInputValues,
            length: encoderInputValues.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw NativeRuntimeError.allocationFailed("could not upload shape encoder input")
        }
        progress?("shape-encoder")
        let encoded = try runStage(
            name: "shape-encoder", checkpointURL: checkpoints.shapeEncoder,
            checkpointSHA256: CheckpointHashes.shapeEncoder,
            arenaCapacity: options.memory.shapeEncoderBytes
        ) { session in
            try requireSameDevice(session.device, [encoderInput])
            return try session.encodeShapeF32(
                input: encoderInput, coordinates: encoderCoordinates,
                spatialShape: SparseSpatialShape(cubic: 512)
            )
        }
        stageEvidence.append(encoded.evidence)
        let compactCoordinates = encoded.value.coordinates
        let coordinateBuffer = try makeCoordinateBuffer(
            device: dino.value.conditioning.device,
            coordinates: compactCoordinates
        )
        var noise = SeededGaussianNoise(seed: options.seed)
        let textureNoise = try noise.makeBuffer(
            device: dino.value.conditioning.device,
            count: try checkedElements(compactCoordinates.count, 32),
            label: "TRELLIS existing-mesh texture noise"
        )
        progress?("texture-flow")
        let textureSample = try runStage(
            name: "texture-flow", checkpointURL: checkpoints.textureFlow,
            checkpointSHA256: CheckpointHashes.textureFlow,
            arenaCapacity: options.memory.textureFlowBytes
        ) { session in
            try requireSameDevice(
                session.device,
                [textureNoise, encoded.value.latent, coordinateBuffer,
                 dino.value.conditioning]
            )
            return try session.sampleTextureF32(
                noise: textureNoise, shapeLatent: encoded.value.latent,
                coordinates: coordinateBuffer,
                positiveConditioning: dino.value.conditioning,
                tokens: compactCoordinates.count,
                conditioningTokens: dino.value.tokenCount,
                parameters: .texture512(steps: options.steps)
            )
        }
        stageEvidence.append(textureSample.evidence)
        progress?("texture-decoder")
        let decoded = try runStage(
            name: "texture-decoder", checkpointURL: checkpoints.textureDecoder,
            checkpointSHA256: CheckpointHashes.textureDecoder,
            arenaCapacity: options.memory.textureDecoderBytes
        ) { session in
            try requireSameDevice(session.device, [textureSample.value.latent])
            return try session.decodeTextureF32(
                latent: textureSample.value.latent,
                coordinates: compactCoordinates,
                spatialShape: encoded.value.spatialShape,
                subdivisionGuides: encoded.value.subdivisionGuides
            )
        }
        stageEvidence.append(decoded.evidence)
        progress?("pbr-export")
        let exportContext = try MetalContext(arenaCapacity: options.memory.pbrExportBytes)
        try requireSameDevice(exportContext.device, [decoded.value.pbrFields])
        let exportStarted = Date()
        let export = try NativePBRExporter(context: exportContext).export(
            atlas: uv.atlas, sourcePositions: mesh.positions,
            sourceFaces: mesh.faces,
            decodedPBRFields: decoded.value.pbrFields,
            fieldCoordinates: decoded.value.coordinates,
            spatialShape: decoded.value.spatialShape,
            aabbMinimum: SIMD3(repeating: -0.5),
            voxelSize: SIMD3(repeating: 1.0 / 512.0),
            textureWidth: options.textureSize,
            textureHeight: options.textureSize,
            alphaMode: options.alphaMode,
            outputURL: outputURL
        )
        let exportSnapshot = try requireReleasedArena(exportContext, name: "pbr-export")
        stageEvidence.append(NativeStageEvidence(
            name: "pbr-export", checkpointSHA256: "none",
            elapsedSeconds: Date().timeIntervalSince(exportStarted),
            arenaCapacityBytes: exportSnapshot.capacityBytes,
            arenaPeakBytes: exportSnapshot.peakUsedBytes,
            arenaLiveAfterCloseBytes: exportSnapshot.usedBytes,
            cumulativeRequestedBytes: exportSnapshot.cumulativeRequestedBytes,
            allocationCount: exportSnapshot.allocationCount
        ))
        let validation = try PBRGLBValidator.validate(url: outputURL)
        guard validation.vertexCount == uv.atlas.positions.count,
              validation.indexCount == uv.atlas.faces.count * 3,
              validation.textureWidth == options.textureSize,
              validation.textureHeight == options.textureSize else {
            throw NativeRuntimeError.invalidArgument(
                "reloaded textured GLB does not match its source mesh"
            )
        }
        return Trellis2TexturingEvidence(
            runtime: "swift-metal", pipeline: "trellis2-texturing-512",
            upstreamSourceRevision: Self.trellisSourceRevision,
            weightsRevision: Self.trellisWeightsRevision,
            device: exportContext.device.name,
            operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
            imageSHA256: try fileSHA256(at: imageURL),
            meshSHA256: try fileSHA256(at: meshURL),
            imagePreprocessing: image.backgroundRemoval
                + "+crop-premultiply-lanczos-imagenet",
            seed: options.seed, noiseAlgorithm: SeededGaussianNoise.algorithm,
            steps: options.steps, sourceVertexCount: mesh.positions.count,
            sourceFaceCount: mesh.faces.count,
            outputVertexCount: validation.vertexCount,
            outputFaceCount: validation.indexCount / 3,
            voxelCoordinateCount: voxelization.coordinates.count,
            latentCoordinateCount: compactCoordinates.count,
            pbrCoordinateCount: decoded.value.coordinates.count,
            uvImplementation: uv.implementation,
            uvExactUpstreamTier: uv.exactUpstreamTier,
            uvFingerprintSHA256: uv.fingerprintSHA256,
            textureSize: options.textureSize,
            coveredTexels: export.coveredTexels, glbBytes: export.glbBytes,
            glbReloadValidated: true,
            outputSHA256: try fileSHA256(at: outputURL), stages: stageEvidence
        )
    }

    public func generate512(
        normalizedImageCHW: [Float],
        imageWidth: Int,
        imageHeight: Int,
        checkpoints: Trellis2CheckpointSet,
        options: Trellis2GenerationOptions = Trellis2GenerationOptions(),
        outputURL: URL,
        sourceInputSHA256: String? = nil,
        imagePreprocessing: String = "caller-supplied-normalized-chw-f32",
        usedMeaningfulAlpha: Bool? = nil,
        progress: ((String) -> Void)? = nil
    ) throws -> Trellis2GenerationEvidence {
        guard imageWidth == 512, imageHeight == 512,
              normalizedImageCHW.count == 3 * imageWidth * imageHeight,
              normalizedImageCHW.allSatisfy({ $0.isFinite }),
              options.steps > 0,
              options.textureSize > 1,
              ["OPAQUE", "BLEND", "MASK"].contains(options.alphaMode) else {
            throw NativeRuntimeError.invalidArgument(
                "native 512 generation requires finite 3x512x512 input and valid options"
            )
        }
        try validateMemoryBudget(options.memory)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        var stageEvidence: [NativeStageEvidence] = []
        var noise = SeededGaussianNoise(seed: options.seed)

        progress?("image-conditioning")
        let dinoResult = try runStage(
            name: "dino-v3", checkpointURL: checkpoints.dino,
            checkpointSHA256: CheckpointHashes.dino,
            arenaCapacity: options.memory.dinoBytes
        ) { session in
            var image = normalizedImageCHW
            guard let buffer = session.device.makeBuffer(
                bytes: &image, length: image.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ) else {
                throw NativeRuntimeError.allocationFailed("could not upload normalized image")
            }
            return try session.encodeDINOv3F32(
                normalizedImage: buffer, imageHeight: imageHeight,
                imageWidth: imageWidth
            )
        }
        stageEvidence.append(dinoResult.evidence)
        let conditioning = dinoResult.value
        let conditioningBytes = try checkedBytes(
            conditioning.tokenCount, conditioning.hiddenSize
        )
        guard let negativeConditioning = conditioning.conditioning.device.makeBuffer(
            length: conditioningBytes, options: .storageModeShared
        ) else {
            throw NativeRuntimeError.allocationFailed("could not allocate negative conditioning")
        }
        negativeConditioning.contents().initializeMemory(
            as: UInt8.self, repeating: 0, count: conditioningBytes
        )

        progress?("sparse-structure-flow")
        let sparseNoise = try noise.makeBuffer(
            device: conditioning.conditioning.device,
            count: 16 * 16 * 16 * 8,
            label: "TRELLIS sparse-structure noise"
        )
        let sparseSample = try runStage(
            name: "sparse-structure-flow",
            checkpointURL: checkpoints.sparseStructureFlow,
            checkpointSHA256: CheckpointHashes.sparseStructureFlow,
            arenaCapacity: options.memory.sparseStructureFlowBytes
        ) { session in
            try requireSameDevice(
                session.device, [sparseNoise, conditioning.conditioning, negativeConditioning]
            )
            return try session.sampleSparseStructureF32(
                noise: sparseNoise,
                positiveConditioning: conditioning.conditioning,
                negativeConditioning: negativeConditioning,
                conditioningTokens: conditioning.tokenCount,
                parameters: .sparseStructure512(steps: options.steps)
            )
        }
        stageEvidence.append(sparseSample.evidence)

        progress?("sparse-structure-decoder")
        let sparseCoordinates = try runStage(
            name: "sparse-structure-decoder",
            checkpointURL: checkpoints.sparseStructureDecoder,
            checkpointSHA256: CheckpointHashes.sparseStructureDecoder,
            arenaCapacity: options.memory.sparseStructureDecoderBytes
        ) { session -> [SparseStructureCoordinate] in
            try requireSameDevice(session.device, [sparseSample.value.latent])
            let decoded = try session.decodeSparseStructureF32(
                latent: sparseSample.value.latent
            )
            guard !decoded.coordinates.isEmpty else {
                throw NativeRuntimeError.invalidArgument(
                    "sparse-structure decoder produced no occupied coordinates"
                )
            }
            return decoded.coordinates
        }
        stageEvidence.append(sparseCoordinates.evidence)
        let coordinates = sparseCoordinates.value
        let coordinateBuffer = try makeCoordinateBuffer(
            device: conditioning.conditioning.device, coordinates: coordinates
        )

        progress?("shape-flow")
        let shapeNoise = try noise.makeBuffer(
            device: conditioning.conditioning.device,
            count: try checkedElements(coordinates.count, 32),
            label: "TRELLIS shape noise"
        )
        let shapeSample = try runStage(
            name: "shape-flow", checkpointURL: checkpoints.shapeFlow,
            checkpointSHA256: CheckpointHashes.shapeFlow,
            arenaCapacity: options.memory.shapeFlowBytes
        ) { session in
            try requireSameDevice(
                session.device,
                [shapeNoise, coordinateBuffer, conditioning.conditioning,
                 negativeConditioning]
            )
            return try session.sampleShapeF32(
                noise: shapeNoise, coordinates: coordinateBuffer,
                positiveConditioning: conditioning.conditioning,
                negativeConditioning: negativeConditioning,
                tokens: coordinates.count,
                conditioningTokens: conditioning.tokenCount,
                parameters: .shape512(steps: options.steps)
            )
        }
        stageEvidence.append(shapeSample.evidence)

        progress?("texture-flow")
        let textureNoise = try noise.makeBuffer(
            device: conditioning.conditioning.device,
            count: try checkedElements(coordinates.count, 32),
            label: "TRELLIS texture noise"
        )
        let textureSample = try runStage(
            name: "texture-flow", checkpointURL: checkpoints.textureFlow,
            checkpointSHA256: CheckpointHashes.textureFlow,
            arenaCapacity: options.memory.textureFlowBytes
        ) { session in
            try requireSameDevice(
                session.device,
                [textureNoise, shapeSample.value.latent, coordinateBuffer,
                 conditioning.conditioning]
            )
            return try session.sampleTextureF32(
                noise: textureNoise,
                shapeLatent: shapeSample.value.latent,
                coordinates: coordinateBuffer,
                positiveConditioning: conditioning.conditioning,
                tokens: coordinates.count,
                conditioningTokens: conditioning.tokenCount,
                parameters: .texture512(steps: options.steps)
            )
        }
        stageEvidence.append(textureSample.evidence)

        progress?("shape-decoder")
        let shapeDecoded = try runStage(
            name: "shape-decoder", checkpointURL: checkpoints.shapeDecoder,
            checkpointSHA256: CheckpointHashes.shapeDecoder,
            arenaCapacity: options.memory.shapeDecoderBytes
        ) { session in
            try requireSameDevice(session.device, [shapeSample.value.latent])
            return try session.decodeShapeF32(
                latent: shapeSample.value.latent,
                coordinates: coordinates,
                spatialShape: SparseSpatialShape(cubic: 32)
            )
        }
        stageEvidence.append(shapeDecoded.evidence)

        progress?("mesh-extraction")
        let meshExtractionStarted = Date()
        let meshContext = try MetalContext(arenaCapacity: 64 * 1024 * 1024)
        try requireSameDevice(meshContext.device, [shapeDecoded.value.rawHead])
        let meshResult = try autoreleasepool {
            try NativeShapeMeshDecoder(context: meshContext).decode(
                rawHead: shapeDecoded.value.rawHead,
                coordinates: shapeDecoded.value.coordinates,
                gridSize: shapeDecoded.value.spatialShape
            )
        }
        guard !meshResult.mesh.faces.isEmpty else {
            throw NativeRuntimeError.invalidArgument("shape decoder produced an empty mesh")
        }
        let repairedMesh = try MeshHoleFiller.fillTriangleAndQuadHoles(meshResult.mesh).mesh
        let meshSnapshot = try requireReleasedArena(meshContext, name: "mesh-extraction")
        stageEvidence.append(stageEvidenceForUtility(
            name: "mesh-extraction", snapshot: meshSnapshot,
            elapsedSeconds: Date().timeIntervalSince(meshExtractionStarted)
        ))

        progress?("texture-decoder")
        let textureDecoded = try runStage(
            name: "texture-decoder", checkpointURL: checkpoints.textureDecoder,
            checkpointSHA256: CheckpointHashes.textureDecoder,
            arenaCapacity: options.memory.textureDecoderBytes
        ) { session in
            try requireSameDevice(session.device, [textureSample.value.latent])
            return try session.decodeTextureF32(
                latent: textureSample.value.latent,
                coordinates: coordinates,
                spatialShape: SparseSpatialShape(cubic: 32),
                subdivisionGuides: shapeDecoded.value.subdivisionGuides
            )
        }
        stageEvidence.append(textureDecoded.evidence)

        progress?("uv-and-pbr-export")
        let exportStarted = Date()
        let uv = try UVPreparation.prepare(
            positions: repairedMesh.vertices,
            faces: repairedMesh.faces,
            suppliedUVs: nil,
            policy: options.uvPolicy
        )
        let exportContext = try MetalContext(
            arenaCapacity: options.memory.pbrExportBytes
        )
        try requireSameDevice(exportContext.device, [textureDecoded.value.pbrFields])
        let export = try NativePBRExporter(context: exportContext).export(
            atlas: uv.atlas,
            sourcePositions: repairedMesh.vertices,
            sourceFaces: repairedMesh.faces,
            decodedPBRFields: textureDecoded.value.pbrFields,
            fieldCoordinates: textureDecoded.value.coordinates,
            spatialShape: textureDecoded.value.spatialShape,
            aabbMinimum: SIMD3(repeating: -0.5),
            voxelSize: SIMD3(repeating: 1.0 / 512.0),
            textureWidth: options.textureSize,
            textureHeight: options.textureSize,
            alphaMode: options.alphaMode,
            outputURL: outputURL
        )
        let exportSnapshot = try requireReleasedArena(exportContext, name: "pbr-export")
        stageEvidence.append(stageEvidenceForUtility(
            name: "pbr-export", snapshot: exportSnapshot,
            elapsedSeconds: Date().timeIntervalSince(exportStarted)
        ))
        let validation = try PBRGLBValidator.validate(url: outputURL)
        guard validation.vertexCount == uv.atlas.positions.count,
              validation.indexCount == uv.atlas.faces.count * 3,
              validation.textureWidth == options.textureSize,
              validation.textureHeight == options.textureSize,
              validation.alphaMode == options.alphaMode else {
            throw NativeRuntimeError.invalidArgument(
                "reloaded GLB does not match the generated mesh and material"
            )
        }
        let outputSHA = try fileSHA256(at: outputURL)
        return Trellis2GenerationEvidence(
            runtime: "swift-metal",
            pipeline: "trellis2-image-to-3d-512",
            upstreamSourceRevision: Self.trellisSourceRevision,
            weightsRevision: Self.trellisWeightsRevision,
            device: exportContext.device.name,
            operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
            inputSHA256: sourceInputSHA256,
            imagePreprocessing: imagePreprocessing,
            usedMeaningfulAlpha: usedMeaningfulAlpha,
            seed: options.seed,
            noiseAlgorithm: SeededGaussianNoise.algorithm,
            steps: options.steps,
            conditioningTokens: conditioning.tokenCount,
            sparseCoordinateCount: coordinates.count,
            shapeCoordinateCount: shapeDecoded.value.coordinates.count,
            meshVertexCount: repairedMesh.vertices.count,
            meshFaceCount: repairedMesh.faces.count,
            outputVertexCount: validation.vertexCount,
            outputFaceCount: validation.indexCount / 3,
            uvImplementation: uv.implementation,
            uvExactUpstreamTier: uv.exactUpstreamTier,
            uvFingerprintSHA256: uv.fingerprintSHA256,
            textureSize: options.textureSize,
            coveredTexels: export.coveredTexels,
            glbBytes: export.glbBytes,
            glbReloadValidated: true,
            outputSHA256: outputSHA,
            stages: stageEvidence
        )
    }
}

private enum CheckpointHashes {
    static let dino = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
    static let sparseStructureFlow = "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6"
    static let sparseStructureDecoder = "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a"
    static let shapeFlow = "ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f"
    static let textureFlow = "8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d"
    static let shapeDecoder = "e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581"
    static let textureDecoder = "97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c"
    static let shapeEncoder = "f37c5ff5b983b68e9946060000f09bc131f3e84318a2c8b7430a81e4b4636c41"
}

private struct StageRun<Value> {
    let value: Value
    let evidence: NativeStageEvidence
}

private func runStage<Value>(
    name: String,
    checkpointURL: URL,
    checkpointSHA256: String,
    arenaCapacity: Int,
    body: (StageSession) throws -> Value
) throws -> StageRun<Value> {
    let started = Date()
    let session = try StageSession(
        checkpointURL: checkpointURL,
        expectedCheckpointSHA256: checkpointSHA256,
        arenaCapacity: arenaCapacity
    )
    let value: Value
    do {
        value = try body(session)
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
    let snapshot = try session.close()
    return StageRun(
        value: value,
        evidence: NativeStageEvidence(
            name: name,
            checkpointSHA256: checkpointSHA256,
            elapsedSeconds: Date().timeIntervalSince(started),
            arenaCapacityBytes: snapshot.capacityBytes,
            arenaPeakBytes: snapshot.peakUsedBytes,
            arenaLiveAfterCloseBytes: snapshot.usedBytes,
            cumulativeRequestedBytes: snapshot.cumulativeRequestedBytes,
            allocationCount: snapshot.allocationCount
        )
    )
}

private func makeCoordinateBuffer(
    device: MTLDevice, coordinates: [SparseStructureCoordinate]
) throws -> MTLBuffer {
    var packed: [Int32] = []
    packed.reserveCapacity(try checkedElements(coordinates.count, 4))
    for coordinate in coordinates {
        packed.append(contentsOf: [
            coordinate.batch, coordinate.x, coordinate.y, coordinate.z,
        ])
    }
    guard let buffer = device.makeBuffer(
        bytes: &packed, length: packed.count * MemoryLayout<Int32>.stride,
        options: .storageModeShared
    ) else {
        throw NativeRuntimeError.allocationFailed("could not upload sparse coordinates")
    }
    buffer.label = "TRELLIS compact sparse coordinates"
    return buffer
}

private func requireSameDevice(_ device: MTLDevice, _ buffers: [MTLBuffer]) throws {
    guard buffers.allSatisfy({ $0.device.registryID == device.registryID }) else {
        throw NativeRuntimeError.invalidArgument(
            "stage buffers and checkpoint must use the same physical Metal device"
        )
    }
}

private func requireReleasedArena(
    _ context: MetalContext, name: String
) throws -> MetalMemorySnapshot {
    try context.waitUntilIdle()
    guard let arena = context.arena else {
        throw NativeRuntimeError.invalidArgument("\(name) requires a bounded arena")
    }
    let snapshot = arena.snapshot()
    guard snapshot.usedBytes == 0 else {
        throw StageSessionError.arenaStillLive(snapshot.usedBytes)
    }
    return snapshot
}

private func stageEvidenceForUtility(
    name: String, snapshot: MetalMemorySnapshot, elapsedSeconds: Double
) -> NativeStageEvidence {
    NativeStageEvidence(
        name: name, checkpointSHA256: "none", elapsedSeconds: elapsedSeconds,
        arenaCapacityBytes: snapshot.capacityBytes,
        arenaPeakBytes: snapshot.peakUsedBytes,
        arenaLiveAfterCloseBytes: snapshot.usedBytes,
        cumulativeRequestedBytes: snapshot.cumulativeRequestedBytes,
        allocationCount: snapshot.allocationCount
    )
}

private func checkedElements(_ rows: Int, _ columns: Int) throws -> Int {
    let value = rows.multipliedReportingOverflow(by: columns)
    guard rows > 0, columns > 0, !value.overflow else {
        throw NativeRuntimeError.invalidArgument("native pipeline tensor size overflows Int")
    }
    return value.partialValue
}

private func checkedBytes(_ rows: Int, _ columns: Int) throws -> Int {
    let elements = try checkedElements(rows, columns)
    let value = elements.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
    guard !value.overflow else {
        throw NativeRuntimeError.invalidArgument("native pipeline byte count overflows Int")
    }
    return value.partialValue
}

private func validateMemoryBudget(_ budget: Trellis2MemoryBudget) throws {
    let values = [
        budget.dinoBytes, budget.sparseStructureFlowBytes,
        budget.sparseStructureDecoderBytes, budget.shapeFlowBytes,
        budget.textureFlowBytes, budget.shapeDecoderBytes,
        budget.shapeEncoderBytes,
        budget.textureDecoderBytes, budget.pbrExportBytes,
    ]
    guard values.allSatisfy({ $0 > 0 }) else {
        throw NativeRuntimeError.invalidArgument("stage memory budgets must be positive")
    }
}
