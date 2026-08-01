import Foundation
import Metal
import Testing
@testable import KernelGoblinTrellis2

@Suite("Native sparse shape encoder")
struct ShapeSparseEncoderTests {
    @Test("reversed encoder guides validate through every decoder subdivision")
    func chainedGuideOrder() throws {
        var coordinates = [
            SparseStructureCoordinate(x: 31, y: 2, z: 17),
            SparseStructureCoordinate(x: 3, y: 15, z: 8),
            SparseStructureCoordinate(x: 30, y: 3, z: 16),
            SparseStructureCoordinate(x: 2, y: 14, z: 9),
        ]
        var shape = try SparseSpatialShape(cubic: 32)
        var guides: [SparseSubdivision2x] = []
        for _ in 0..<4 {
            let topology = try SparseSpatial2Channel2x(
                coordinates: coordinates, spatialShape: shape
            )
            guides.append(topology.subdivision)
            coordinates = topology.coordinates
            shape = topology.spatialShape
        }
        for guide in guides.reversed() {
            try guide.validate(parentCoordinates: coordinates)
            coordinates = guide.coordinates
        }
        #expect(coordinates == [
            SparseStructureCoordinate(x: 31, y: 2, z: 17),
            SparseStructureCoordinate(x: 3, y: 15, z: 8),
            SparseStructureCoordinate(x: 30, y: 3, z: 16),
            SparseStructureCoordinate(x: 2, y: 14, z: 9),
        ])
    }

    @Test("spatial-to-channel sorts parents and uses x-fastest child slots")
    func spatialToChannelTopology() throws {
        let input = [
            SparseStructureCoordinate(batch: 1, x: 3, y: 1, z: 0),
            SparseStructureCoordinate(x: 1, y: 0, z: 1),
            SparseStructureCoordinate(x: 0, y: 0, z: 0),
            SparseStructureCoordinate(batch: 1, x: 2, y: 0, z: 0),
        ]
        let topology = try SparseSpatial2Channel2x(
            coordinates: input,
            spatialShape: SparseSpatialShape(width: 5, height: 3, depth: 3)
        )
        #expect(topology.spatialShape == (try SparseSpatialShape(
            width: 3, height: 2, depth: 2
        )))
        #expect(topology.coordinates == [
            SparseStructureCoordinate(x: 0, y: 0, z: 0),
            SparseStructureCoordinate(batch: 1, x: 1, y: 0, z: 0),
        ])
        #expect(topology.inputParentIndices == [1, 0, 0, 1])
        #expect(topology.inputChildIndices == [3, 5, 0, 0])
        #expect(topology.sourceIndices == [2, -1, -1, -1, -1, 1, -1, -1,
                                           3, -1, -1, 0, -1, -1, -1, -1])
        #expect(topology.subdivision.coordinates == input)
        #expect(topology.subdivision.parentIndices == [1, 0, 0, 1])
        #expect(topology.subdivision.childIndices == [3, 5, 0, 0])
        try topology.subdivision.validate(parentCoordinates: topology.coordinates)
        #expect(throws: NativeRuntimeError.self) {
            _ = try SparseSpatial2Channel2x(
                coordinates: [input[1], input[1]],
                spatialShape: SparseSpatialShape(cubic: 4)
            )
        }
    }

    @Test("Metal spatial-to-channel packs holes and reduces the skip exactly")
    func spatialToChannelMetal() throws {
        let context = try MetalContext(arenaCapacity: 64 * 1024)
        let coordinates = [
            SparseStructureCoordinate(x: 1, y: 0, z: 1),
            SparseStructureCoordinate(x: 0, y: 0, z: 0),
        ]
        let topology = try SparseSpatial2Channel2x(
            coordinates: coordinates, spatialShape: SparseSpatialShape(cubic: 2)
        )
        var values = Array(0..<16).map(Float.init)
        let input = try #require(context.device.makeBuffer(
            bytes: &values, length: values.count * 4, options: .storageModeShared
        ))
        let map = try topology.makeSourceIndexBuffer(context: context)
        let packed = try context.makeBuffer(length: 64 * 4, label: "S2C packed test")
        let skipped = try context.makeBuffer(length: 16 * 4, label: "S2C skip test")
        let kernel = try SparseSpatialToChannelKernel(context: context)
        try kernel.pack(
            input: input, sourceIndices: map, coarseCount: 1,
            channels: 8, output: packed
        )
        try kernel.skipMean(
            input: input, sourceIndices: map, coarseCount: 1,
            inputChannels: 8, outputChannels: 16, output: skipped
        )
        let packedValues = packed.contents().assumingMemoryBound(to: Float.self)
        for channel in 0..<8 { #expect(packedValues[channel] == values[8 + channel]) }
        for channel in 8..<(5 * 8) { #expect(packedValues[channel] == 0) }
        for channel in 0..<8 { #expect(packedValues[5 * 8 + channel] == values[channel]) }
        for channel in (6 * 8)..<64 { #expect(packedValues[channel] == 0) }
        let skipValues = skipped.contents().assumingMemoryBound(to: Float.self)
        #expect(skipValues[0] == 9.5)
        #expect(skipValues[1] == 13.5)
        for channel in 2..<10 { #expect(skipValues[channel] == 0) }
        #expect(skipValues[10] == 1.5)
        #expect(skipValues[11] == 5.5)
        for channel in 12..<16 { #expect(skipValues[channel] == 0) }
    }

    @Test("S2C residual block preserves sorted topology and exact skip reshape")
    func s2cBlockSynthetic() throws {
        let context = try MetalContext(arenaCapacity: 2 * 1024 * 1024)
        let inputChannels = 8, outputChannels = 16
        var payload: [UInt16] = []
        func append(_ values: [Float]) -> Int {
            let offset = payload.count * 2
            payload += values.map(shapeEncoderF16)
            return offset
        }
        let normWeight = append([Float](repeating: 1, count: inputChannels))
        let normBias = append([Float](repeating: 0, count: inputChannels))
        let downWeight = append([Float](
            repeating: 0, count: (outputChannels / 8) * 27 * inputChannels
        ))
        let downBias = append([Float](repeating: 0, count: outputChannels / 8))
        let outputWeight = append([Float](
            repeating: 0, count: outputChannels * 27 * outputChannels
        ))
        let outputBias = append([Float](repeating: 0, count: outputChannels))
        let layout = SparseS2CCheckpointLayout(
            normWeight: normWeight, normBias: normBias,
            convolutionDownWeight: downWeight, convolutionDownBias: downBias,
            convolutionOutputWeight: outputWeight,
            convolutionOutputBias: outputBias
        )
        let coordinates = [
            SparseStructureCoordinate(x: 3, y: 1, z: 0),
            SparseStructureCoordinate(x: 0, y: 0, z: 0),
            SparseStructureCoordinate(x: 1, y: 0, z: 1),
            SparseStructureCoordinate(x: 2, y: 0, z: 0),
        ]
        var values = (0..<(coordinates.count * inputChannels)).map {
            Float($0) + 0.25
        }
        let input = try #require(context.device.makeBuffer(
            bytes: &values, length: values.count * 4, options: .storageModeShared
        ))
        let neighborhood = try SparseNeighborhood3x3(
            coordinates: coordinates, spatialShape: SparseSpatialShape(cubic: 4)
        )
        let neighbors = try #require(try neighborhood.makeMetalBuffer(context: context))
        let checkpoint = try #require(context.device.makeBuffer(
            bytes: &payload, length: payload.count * 2, options: .storageModeShared
        ))
        let result = try SparseS2CBlock(context: context)(
            input: input, coordinates: coordinates,
            spatialShape: SparseSpatialShape(cubic: 4), neighborBuffer: neighbors,
            checkpoint: checkpoint, layout: layout,
            inputChannels: inputChannels, outputChannels: outputChannels
        )
        #expect(result.coordinates == [
            SparseStructureCoordinate(x: 0, y: 0, z: 0),
            SparseStructureCoordinate(x: 1, y: 0, z: 0),
        ])
        #expect(result.subdivision.coordinates == coordinates)
        let output = result.features.contents().assumingMemoryBound(to: Float.self)
        // Parent 0 child 0 is source row 1, followed by three channels at a time.
        #expect(output[0] == Float(Float16((values[8] + values[9] + values[10] + values[11]) / 4)))
        #expect(output[1] == Float(Float16((values[12] + values[13] + values[14] + values[15]) / 4)))
        for channel in 2..<10 { #expect(output[channel] == 0) }
        #expect(output[10] == Float(Float16((values[16] + values[17] + values[18] + values[19]) / 4)))
    }

    @Test(
        "complete encoder matches the pinned real-checkpoint MPS fixture",
        .enabled(
            if: ProcessInfo.processInfo.environment[
                "KG_TRELLIS2_SHAPE_ENCODER_CHECKPOINT"
            ] != nil,
            "Set KG_TRELLIS2_SHAPE_ENCODER_CHECKPOINT for encoder conformance"
        )
    )
    func completeRealCheckpoint() throws {
        let path = try #require(ProcessInfo.processInfo.environment[
            "KG_TRELLIS2_SHAPE_ENCODER_CHECKPOINT"
        ])
        let fixtureURL = try #require(Bundle.module.url(
            forResource: "shape-encoder-full-mps", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(fileSHA256(at: fixtureURL) ==
            "1890fa1b2f7d1ab10e6be502937ad938300494b17d74d6b239204217167309bd")
        let metadataURL = fixtureURL.appendingPathExtension("json")
        try #require(fileSHA256(at: metadataURL) ==
            "9e5bf864abc0ae3a49b5c27c6bae692a774e8d7654a0b97e3167b9ce083ff33b")
        let metadata = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL))
                as? [String: Any]
        )
        #expect(metadata["oracle_call"] as? String ==
            "SparseUnetVaeEncoder.forward with pinned MPS conv overlay")
        let expected = try Data(contentsOf: fixtureURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        var context: MetalContext? = try MetalContext(
            arenaCapacity: 512 * 1024 * 1024
        )
        let checkpoint = try MappedCheckpoint(
            url: URL(fileURLWithPath: path), device: context!.device
        )
        let checkpointDigest = try autoreleasepool { try checkpoint.sha256() }
        #expect(checkpointDigest ==
            "f37c5ff5b983b68e9946060000f09bc131f3e84318a2c8b7430a81e4b4636c41")
        var values = Array(expected[0..<6])
        guard let input = context?.device.makeBuffer(
            bytes: &values, length: values.count * 4, options: .storageModeShared
        ) else {
            throw NativeRuntimeError.allocationFailed(
                "could not allocate shape encoder fixture input"
            )
        }
        let native: (
            latent: [Float], coordinates: [SparseStructureCoordinate],
            shape: SparseSpatialShape, guides: [[SparseStructureCoordinate]]
        ) = try autoreleasepool {
            let result = try ShapeSparseEncoder(context: context!)(
                input: input,
                coordinates: [SparseStructureCoordinate(x: 15, y: 15, z: 15)],
                spatialShape: SparseSpatialShape(cubic: 16), checkpoint: checkpoint
            )
            let pointer = result.latent.contents().assumingMemoryBound(to: Float.self)
            return (
                Array(UnsafeBufferPointer(start: pointer, count: 32)),
                result.coordinates, result.spatialShape,
                result.subdivisionGuides.map(\.coordinates)
            )
        }
        #expect(native.coordinates == [SparseStructureCoordinate(x: 0, y: 0, z: 0)])
        #expect(native.shape == (try SparseSpatialShape(cubic: 1)))
        #expect(native.guides.count == 4)
        #expect(native.guides == [
            [SparseStructureCoordinate(x: 1, y: 1, z: 1)],
            [SparseStructureCoordinate(x: 3, y: 3, z: 3)],
            [SparseStructureCoordinate(x: 7, y: 7, z: 7)],
            [SparseStructureCoordinate(x: 15, y: 15, z: 15)],
        ])
        var squaredError = 0.0
        var squaredExpected = 0.0
        var maximumError: Float = 0
        var maximumMagnitude: Float = 0
        for channel in 0..<32 {
            let reference = expected[1926 + channel]
            let error = abs(native.latent[channel] - reference)
            squaredError += Double(error * error)
            squaredExpected += Double(reference * reference)
            maximumError = max(maximumError, error)
            maximumMagnitude = max(maximumMagnitude, abs(reference))
        }
        let normalizedRMS = sqrt(squaredError / max(squaredExpected, 1e-30))
        let scaleRatio = maximumError / max(maximumMagnitude, 1e-20)
        print(
            "shape encoder full: normalized_rms=\(normalizedRMS) scale=\(scaleRatio)"
        )
        #expect(normalizedRMS <= 0.003)
        #expect(scaleRatio <= 0.006)
        try context!.waitUntilIdle()
        #expect(context?.arena?.snapshot().usedBytes == 0)
        context = nil
        autoreleasepool {}
        try checkpoint.close()
    }
}

private func shapeEncoderF16(_ value: Float) -> UInt16 {
    Float16(value).bitPattern
}
