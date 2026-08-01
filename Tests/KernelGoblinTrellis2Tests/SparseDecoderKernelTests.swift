import Metal
import Testing
@testable import KernelGoblinTrellis2

@Suite("Native sparse decoder kernels")
struct SparseDecoderKernelTests {
    @Test("flexible dual-grid head preserves sigmoid, strict intersections, and softplus")
    func flexibleDualGridHead() throws {
        let context = try MetalContext(arenaCapacity: 64 * 1024)
        var raw: [Float] = [
            -2, 0, 2, -Float.ulpOfOne, 0, Float.ulpOfOne, -2,
            20, -20, 1, 1, -1, 0, 25,
        ]
        let rawBuffer = try #require(context.device.makeBuffer(
            bytes: &raw, length: raw.count * 4, options: .storageModeShared
        ))
        let result = try FlexibleDualGridHeadKernel(context: context)(
            rawHead: rawBuffer, count: 2
        )
        let dual = result.dualOffsets.contents().assumingMemoryBound(to: Float.self)
        let intersections = result.intersections.contents().assumingMemoryBound(to: UInt8.self)
        let split = result.splitWeights.contents().assumingMemoryBound(to: Float.self)
        for token in 0..<2 {
            for axis in 0..<3 {
                let expected = 2 / (1 + exp(-raw[token * 7 + axis])) - 0.5
                #expect(abs(dual[token * 3 + axis] - expected) <= 2e-6)
                #expect(intersections[token * 3 + axis] ==
                    (raw[token * 7 + 3 + axis] > 0 ? 1 : 0))
            }
            let value = raw[token * 7 + 6]
            let expectedSplit = value > 20 ? value : log1p(exp(value))
            #expect(abs(split[token] - expectedSplit) <= 2e-6)
        }
    }

    @Test("flexible dual-grid mesh preserves axis connectivity and diagonal tie rule")
    func flexibleDualGridMesh() throws {
        let context = try MetalContext()
        let coordinates = [
            SparseStructureCoordinate(x: 0, y: 0, z: 0),
            SparseStructureCoordinate(x: 0, y: 0, z: 1),
            SparseStructureCoordinate(x: 0, y: 1, z: 1),
            SparseStructureCoordinate(x: 0, y: 1, z: 0),
        ]
        var dual = [Float](repeating: 0.5, count: coordinates.count * 3)
        var flags = [UInt8](repeating: 0, count: coordinates.count * 3)
        flags[0] = 1
        var split: [Float] = [2, 1, 2, 1]
        let dualBuffer = try #require(context.device.makeBuffer(
            bytes: &dual, length: dual.count * 4, options: .storageModeShared
        ))
        let flagBuffer = try #require(context.device.makeBuffer(
            bytes: &flags, length: flags.count, options: .storageModeShared
        ))
        let splitBuffer = try #require(context.device.makeBuffer(
            bytes: &split, length: split.count * 4, options: .storageModeShared
        ))
        var mesh = try FlexibleDualGridMeshExtractor.extract(
            coordinates: coordinates, dualOffsets: dualBuffer,
            intersections: flagBuffer, splitWeights: splitBuffer,
            gridSize: SparseSpatialShape(cubic: 2)
        )
        #expect(mesh.faces == [SIMD3(0, 1, 2), SIMD3(0, 2, 3)])
        #expect(mesh.vertices[0] == SIMD3(-0.25, -0.25, -0.25))
        splitBuffer.contents().assumingMemoryBound(to: Float.self)
            .initialize(repeating: 1, count: split.count)
        mesh = try FlexibleDualGridMeshExtractor.extract(
            coordinates: coordinates, dualOffsets: dualBuffer,
            intersections: flagBuffer, splitWeights: splitBuffer,
            gridSize: SparseSpatialShape(cubic: 2)
        )
        #expect(mesh.faces == [SIMD3(0, 1, 3), SIMD3(3, 1, 2)])
        let incomplete = try FlexibleDualGridMeshExtractor.extract(
            coordinates: Array(coordinates.dropLast()), dualOffsets: dualBuffer,
            intersections: flagBuffer, splitWeights: splitBuffer,
            gridSize: SparseSpatialShape(cubic: 2)
        )
        #expect(incomplete.faces.isEmpty)
        #expect(throws: NativeRuntimeError.self) {
            _ = try FlexibleDualGridMeshExtractor.extract(
                coordinates: [SparseStructureCoordinate(x: Int32.max, y: 0, z: 0)],
                dualOffsets: dualBuffer, intersections: flagBuffer,
                splitWeights: splitBuffer, gridSize: SparseSpatialShape(cubic: 2)
            )
        }
    }

    @Test("channel-to-spatial subdivision preserves upstream child ordering and layout")
    func channelToSpatialSubdivision() throws {
        let context = try MetalContext(arenaCapacity: 16 * 1024)
        let parents = [
            SparseStructureCoordinate(x: 1, y: 2, z: 3),
            SparseStructureCoordinate(x: 0, y: 0, z: 0),
        ]
        var logits: [Float] = [
            1, -1, -1, -1, -1, 1, -1, 1,
            -1, -1, 1, -1, 1, -1, -1, -1,
        ]
        let logitsBuffer = try #require(context.device.makeBuffer(
            bytes: &logits, length: logits.count * 4, options: .storageModeShared
        ))
        let subdivision = try SparseSubdivision2x(
            parentCoordinates: parents, logits: logitsBuffer
        )
        #expect(throws: NativeRuntimeError.self) {
            _ = try SparseSubdivision2x(
                parentCoordinates: [SparseStructureCoordinate(x: Int32.min, y: 0, z: 0)],
                logits: logitsBuffer
            )
        }
        #expect(subdivision.parentIndices == [0, 0, 0, 1, 1])
        #expect(subdivision.childIndices == [0, 5, 7, 2, 4])
        #expect(subdivision.coordinates == [
            SparseStructureCoordinate(x: 2, y: 4, z: 6),
            SparseStructureCoordinate(x: 3, y: 4, z: 7),
            SparseStructureCoordinate(x: 3, y: 5, z: 7),
            SparseStructureCoordinate(x: 0, y: 1, z: 0),
            SparseStructureCoordinate(x: 0, y: 0, z: 1),
        ])
        try subdivision.validate(parentCoordinates: parents)
        #expect(throws: NativeRuntimeError.self) {
            try subdivision.validate(parentCoordinates: [
                SparseStructureCoordinate(x: 0, y: 0, z: 0),
                SparseStructureCoordinate(x: 0, y: 0, z: 0),
            ])
        }
        let (parentMap, childMap) = try #require(
            try subdivision.makeMetalBuffers(context: context)
        )
        let kernel = try SparseSubdivisionKernel(context: context)
        let outputChannels = 4, inputChannels = 16
        var convolved = (0..<(parents.count * 8 * outputChannels)).map(Float.init)
        var skip = (0..<(parents.count * inputChannels)).map { Float($0) + 0.25 }
        let convolvedBuffer = try #require(context.device.makeBuffer(
            bytes: &convolved, length: convolved.count * 4, options: .storageModeShared
        ))
        let skipBuffer = try #require(context.device.makeBuffer(
            bytes: &skip, length: skip.count * 4, options: .storageModeShared
        ))
        let selected = try context.makeBuffer(
            length: subdivision.coordinates.count * outputChannels * 4,
            label: "subdivision selection test"
        )
        let skipped = try context.makeBuffer(
            length: subdivision.coordinates.count * outputChannels * 4,
            label: "subdivision skip test"
        )
        try kernel.selectConvolutionFeatures(
            input: convolvedBuffer, parentIndices: parentMap, childIndices: childMap,
            childCount: subdivision.coordinates.count, outputChannels: outputChannels,
            output: selected
        )
        try kernel.selectSkipFeatures(
            input: skipBuffer, parentIndices: parentMap, childIndices: childMap,
            childCount: subdivision.coordinates.count, inputChannels: inputChannels,
            outputChannels: outputChannels, output: skipped
        )
        let actualSelected = selected.contents().assumingMemoryBound(to: Float.self)
        let actualSkipped = skipped.contents().assumingMemoryBound(to: Float.self)
        for child in subdivision.coordinates.indices {
            let sourceRow = Int(subdivision.parentIndices[child] * 8 + subdivision.childIndices[child])
            for channel in 0..<outputChannels {
                #expect(actualSelected[child * outputChannels + channel] ==
                    convolved[sourceRow * outputChannels + channel])
                #expect(actualSkipped[child * outputChannels + channel] ==
                    skip[sourceRow * (inputChannels / 8) + channel / 2])
            }
        }
    }

    @Test("C2S block applies predicted subdivision and exact skip reshaping")
    func sparseC2SBlockSynthetic() throws {
        let context = try MetalContext(arenaCapacity: 2 * 1024 * 1024)
        let inputChannels = 8, outputChannels = 4
        var payload: [UInt16] = []
        func append(_ values: [Float]) -> Int {
            let offset = payload.count * 2
            payload += values.map(f16)
            return offset
        }
        let subdivisionWeight = append([Float](repeating: 0, count: 8 * inputChannels))
        let subdivisionBias = append([1, -1, 1, -1, -1, -1, -1, -1])
        let normWeight = append([Float](repeating: 1, count: inputChannels))
        let normBias = append([Float](repeating: 0, count: inputChannels))
        let convUpWeight = append([Float](
            repeating: 0, count: outputChannels * 8 * 27 * inputChannels
        ))
        let convUpBias = append([Float](repeating: 0, count: outputChannels * 8))
        let convOutWeight = append([Float](
            repeating: 0, count: outputChannels * 27 * outputChannels
        ))
        let convOutBias = append([Float](repeating: 0, count: outputChannels))
        let layout = SparseC2SCheckpointLayout(
            subdivisionWeight: subdivisionWeight, subdivisionBias: subdivisionBias,
            normWeight: normWeight, normBias: normBias,
            convolutionUpWeight: convUpWeight, convolutionUpBias: convUpBias,
            convolutionOutputWeight: convOutWeight, convolutionOutputBias: convOutBias
        )
        var inputValues: [Float] = [
            1, 2, 3, 4, 5, 6, 7, 8,
            -1, -2, -3, -4, -5, -6, -7, -8,
        ]
        let coordinates = [
            SparseStructureCoordinate(x: 0, y: 0, z: 0),
            SparseStructureCoordinate(x: 1, y: 0, z: 0),
        ]
        let neighborhood = try SparseNeighborhood3x3(
            coordinates: coordinates, spatialShape: SparseSpatialShape(cubic: 2)
        )
        let input = try #require(context.device.makeBuffer(
            bytes: &inputValues, length: inputValues.count * 4, options: .storageModeShared
        ))
        let neighbors = try #require(try neighborhood.makeMetalBuffer(context: context))
        let checkpoint = try #require(context.device.makeBuffer(
            bytes: &payload, length: payload.count * 2, options: .storageModeShared
        ))
        let result = try SparseC2SBlock(context: context)(
            input: input, coordinates: coordinates,
            spatialShape: SparseSpatialShape(cubic: 2),
            neighborBuffer: neighbors, checkpoint: checkpoint, layout: layout,
            inputChannels: inputChannels, outputChannels: outputChannels
        )
        #expect(result.coordinates == [
            SparseStructureCoordinate(x: 0, y: 0, z: 0),
            SparseStructureCoordinate(x: 0, y: 1, z: 0),
            SparseStructureCoordinate(x: 2, y: 0, z: 0),
            SparseStructureCoordinate(x: 2, y: 1, z: 0),
        ])
        let output = try #require(result.features)
            .contents().assumingMemoryBound(to: Float.self)
        let expected: [Float] = [1, 1, 1, 1, 3, 3, 3, 3, -1, -1, -1, -1, -3, -3, -3, -3]
        for index in expected.indices { #expect(output[index] == expected[index]) }
    }

    @Test("3x3 neighbor map is deterministic, offset ordered, and validates coordinates")
    func neighborMap() throws {
        let shape = try SparseSpatialShape(cubic: 3)
        let coordinates = [
            SparseStructureCoordinate(x: 1, y: 1, z: 1),
            SparseStructureCoordinate(x: 0, y: 1, z: 1),
            SparseStructureCoordinate(x: 2, y: 2, z: 2),
        ]
        let map = try SparseNeighborhood3x3(
            coordinates: coordinates, spatialShape: shape
        )
        #expect(map.indices.count == coordinates.count * 27)
        #expect(map.indices[4] == 1) // (-1, 0, 0)
        #expect(map.indices[13] == 0) // (0, 0, 0)
        #expect(map.indices[26] == 2) // (1, 1, 1)
        #expect(map == (try SparseNeighborhood3x3(
            coordinates: coordinates, spatialShape: shape
        )))
        #expect(throws: NativeRuntimeError.self) {
            _ = try SparseNeighborhood3x3(
                coordinates: [coordinates[0], coordinates[0]], spatialShape: shape
            )
        }
        let context = try MetalContext(arenaCapacity: 4096)
        let empty = try SparseFeatureTensor(
            features: nil, channels: 32, coordinates: [], spatialShape: shape
        )
        #expect(empty.tokenCount == 0)
        let undersized = try #require(context.device.makeBuffer(
            length: 4, options: .storageModeShared
        ))
        #expect(throws: NativeRuntimeError.self) {
            _ = try SparseFeatureTensor(
                features: undersized, channels: 32,
                coordinates: [coordinates[0]], spatialShape: shape
            )
        }
        _ = try #require(try map.makeMetalBuffer(context: context))
        #expect(context.arena?.snapshot().allocationCount == 1)
        #expect(context.arena?.snapshot().cumulativeRequestedBytes == map.indices.count * 4)
        #expect(throws: NativeRuntimeError.self) {
            _ = try SparseNeighborhood3x3(
                coordinates: [SparseStructureCoordinate(x: 3, y: 0, z: 0)],
                spatialShape: shape
            )
        }
    }

    @Test("physical Metal submanifold 3x3 F16 convolution matches ordered CPU reference")
    func scalarSubmanifoldConvolution() throws {
        let context = try MetalContext()
        let kernel = try SubmanifoldConvolutionKernel(context: context)
        let shape = try SparseSpatialShape(cubic: 4)
        let coordinates = [
            SparseStructureCoordinate(x: 1, y: 1, z: 1),
            SparseStructureCoordinate(x: 0, y: 1, z: 1),
            SparseStructureCoordinate(x: 2, y: 1, z: 1),
            SparseStructureCoordinate(x: 1, y: 2, z: 2),
        ]
        let map = try SparseNeighborhood3x3(
            coordinates: coordinates, spatialShape: shape
        )
        let inputChannels = 3, outputChannels = 2
        var inputValues: [Float] = [
            1, -2, 0.5,
            -1, 0.25, 2,
            0.75, 1.5, -0.5,
            2, -1, 0.125,
        ]
        var checkpoint: [UInt16] = (0..<(outputChannels * 27 * inputChannels)).map {
            f16(Float(($0 % 17) - 8) * 0.03125)
        }
        let biasElementOffset = checkpoint.count
        checkpoint += [f16(0.25), f16(-0.5)]
        let input = try #require(context.device.makeBuffer(
            bytes: &inputValues, length: inputValues.count * 4, options: .storageModeShared
        ))
        let neighborBuffer = try #require(try map.makeMetalBuffer(context: context))
        let checkpointBuffer = try #require(context.device.makeBuffer(
            bytes: &checkpoint, length: checkpoint.count * 2, options: .storageModeShared
        ))
        let output = try #require(context.device.makeBuffer(
            length: coordinates.count * outputChannels * 4, options: .storageModeShared
        ))
        try kernel.convolveF16WeightsF32Output(
            input: input, neighbors: neighborBuffer, checkpoint: checkpointBuffer,
            weightOffset: 0, biasOffset: biasElementOffset * 2,
            tokenCount: coordinates.count, inputChannels: inputChannels,
            outputChannels: outputChannels, output: output
        )
        let actual = output.contents().assumingMemoryBound(to: Float.self)
        for token in coordinates.indices {
            for outputChannel in 0..<outputChannels {
                var accumulated = Float16.zero
                for offset in 0..<27 {
                    let source = Int(map.indices[token * 27 + offset])
                    guard source >= 0 else { continue }
                    var contribution: Float = 0
                    for channel in 0..<inputChannels {
                        contribution.addProduct(
                            inputValues[source * inputChannels + channel],
                            fromF16(checkpoint[
                                (outputChannel * 27 + offset) * inputChannels + channel
                            ])
                        )
                    }
                    accumulated = Float16(Float(accumulated) + contribution)
                }
                let expected = Float(Float16(
                    Float(accumulated) + fromF16(checkpoint[biasElementOffset + outputChannel])
                ))
                #expect(actual[token * outputChannels + outputChannel] == expected)
            }
        }
        neighborBuffer.contents().assumingMemoryBound(to: Int32.self)[0] = 99
        #expect(throws: NativeRuntimeError.self) {
            try kernel.convolveF16WeightsF32Output(
                input: input, neighbors: neighborBuffer, checkpoint: checkpointBuffer,
                weightOffset: 0, biasOffset: biasElementOffset * 2,
                tokenCount: coordinates.count, inputChannels: inputChannels,
                outputChannels: outputChannels, output: output
            )
        }
        let privateNeighbors = try #require(context.device.makeBuffer(
            length: map.indices.count * 4, options: .storageModePrivate
        ))
        #expect(throws: NativeRuntimeError.self) {
            try kernel.convolveF16WeightsF32Output(
                input: input, neighbors: privateNeighbors, checkpoint: checkpointBuffer,
                weightOffset: 0, biasOffset: biasElementOffset * 2,
                tokenCount: coordinates.count, inputChannels: inputChannels,
                outputChannels: outputChannels, output: output
            )
        }
    }

    @Test("sparse SIMD-group convolution matches scalar across tile tails")
    func simdSubmanifoldConvolution() throws {
        let context = try MetalContext()
        let kernel = try SubmanifoldConvolutionKernel(context: context)
        guard kernel.supportsSIMDGroupMatrix else { return }
        var coordinates: [SparseStructureCoordinate] = []
        for index in 0..<11 {
            let x = Int32(index % 4)
            let y = Int32((index / 4) % 3)
            let z = Int32(index % 2)
            coordinates.append(SparseStructureCoordinate(x: x, y: y, z: z))
        }
        let map = try SparseNeighborhood3x3(
            coordinates: coordinates, spatialShape: SparseSpatialShape(cubic: 4)
        )
        let inputChannels = 17, outputChannels = 13
        var inputValues = (0..<(coordinates.count * inputChannels)).map {
            Float(sin(Double($0) * 0.13) * 0.25)
        }
        var checkpoint = (0..<(outputChannels * 27 * inputChannels)).map {
            f16(Float(cos(Double($0) * 0.07) * 0.125))
        }
        let biasOffset = checkpoint.count * 2
        checkpoint += (0..<outputChannels).map { f16(Float($0 - 6) * 0.015625) }
        let input = try #require(context.device.makeBuffer(
            bytes: &inputValues, length: inputValues.count * 4, options: .storageModeShared
        ))
        let neighbors = try #require(try map.makeMetalBuffer(context: context))
        let weights = try #require(context.device.makeBuffer(
            bytes: &checkpoint, length: checkpoint.count * 2, options: .storageModeShared
        ))
        let count = coordinates.count * outputChannels
        let scalar = try #require(context.device.makeBuffer(
            length: count * 4, options: .storageModeShared
        ))
        let simd = try #require(context.device.makeBuffer(
            length: (count + 19) * 4, options: .storageModeShared
        ))
        simd.contents().assumingMemoryBound(to: Float.self)
            .initialize(repeating: -9191.25, count: count + 19)
        for (implementation, output) in [
            (SubmanifoldConvolutionImplementation.scalar, scalar),
            (.simdgroupMatrix, simd),
        ] {
            try kernel.convolveF16WeightsF32Output(
                input: input, neighbors: neighbors, checkpoint: weights,
                weightOffset: 0, biasOffset: biasOffset,
                tokenCount: coordinates.count, inputChannels: inputChannels,
                outputChannels: outputChannels, output: output,
                implementation: implementation
            )
        }
        let expected = scalar.contents().assumingMemoryBound(to: Float.self)
        let actual = simd.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<count {
            #expect(abs(actual[index] - expected[index]) <= 0.001)
        }
        for index in count..<(count + 19) {
            #expect(actual[index] == -9191.25)
        }
    }

    @Test("sparse ConvNeXt block preserves the F16 boundary contract")
    func sparseConvNeXtBlock() throws {
        let context = try MetalContext()
        let block = try SparseConvNeXtBlock(context: context)
        let coordinates = [
            SparseStructureCoordinate(x: 0, y: 0, z: 0),
            SparseStructureCoordinate(x: 1, y: 0, z: 0),
            SparseStructureCoordinate(x: 1, y: 1, z: 0),
        ]
        let map = try SparseNeighborhood3x3(
            coordinates: coordinates, spatialShape: SparseSpatialShape(cubic: 2)
        )
        let channels = 2, expandedChannels = 8
        var payload: [UInt16] = []
        func append(_ count: Int, scale: Float) -> Int {
            let offset = payload.count * 2
            payload += (0..<count).map { index in
                f16(Float((index * 7) % 19 - 9) * scale)
            }
            return offset
        }
        let convWeight = append(channels * 27 * channels, scale: 0.015625)
        let convBias = append(channels, scale: 0.03125)
        let normWeight = payload.count * 2
        payload += [f16(0.75), f16(1.25)]
        let normBias = payload.count * 2
        payload += [f16(0.125), f16(-0.25)]
        let upWeight = append(expandedChannels * channels, scale: 0.03125)
        let upBias = append(expandedChannels, scale: 0.015625)
        let downWeight = append(channels * expandedChannels, scale: 0.03125)
        let downBias = append(channels, scale: 0.015625)
        let offsets = SparseConvNeXtOffsets(
            convolutionWeight: convWeight, convolutionBias: convBias,
            normalizationWeight: normWeight, normalizationBias: normBias,
            mlpUpWeight: upWeight, mlpUpBias: upBias,
            mlpDownWeight: downWeight, mlpDownBias: downBias
        )
        var inputValues: [Float] = [1, -0.5, -1.5, 0.25, 0.75, 2]
        let input = try #require(context.device.makeBuffer(
            bytes: &inputValues, length: inputValues.count * 4, options: .storageModeShared
        ))
        let neighbors = try #require(try map.makeMetalBuffer(context: context))
        let checkpoint = try #require(context.device.makeBuffer(
            bytes: &payload, length: payload.count * 2, options: .storageModeShared
        ))
        func buffer(_ count: Int) throws -> MTLBuffer {
            try #require(context.device.makeBuffer(length: count * 4, options: .storageModeShared))
        }
        let elements = coordinates.count * channels
        let conv = try buffer(elements), norm = try buffer(elements)
        let expanded = try buffer(coordinates.count * expandedChannels)
        let branch = try buffer(elements), output = try buffer(elements)
        try block(
            input: input, neighbors: neighbors, checkpoint: checkpoint, offsets: offsets,
            tokenCount: coordinates.count, channels: channels,
            convolutionOutput: conv, normalized: norm, expanded: expanded,
            branch: branch, output: output
        )
        let actual = output.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<elements {
            #expect(actual[index].isFinite)
            #expect(Float(Float16(actual[index])) == actual[index])
        }
        #expect((0..<elements).contains { actual[$0] != inputValues[$0] })
        #expect(try SparseConvNeXtBlock.workspaceBytes(
            tokenCount: coordinates.count, channels: channels
        ) == coordinates.count * (8 * channels * 4 + 27 * 4))
        #expect(try SparseConvNeXtBlock.workspaceBytes(tokenCount: 0, channels: channels) == 0)
        #expect(throws: NativeRuntimeError.self) {
            try block(
                input: input, neighbors: neighbors, checkpoint: checkpoint,
                offsets: offsets, tokenCount: coordinates.count, channels: channels,
                convolutionOutput: conv, normalized: norm, expanded: expanded,
                branch: input, output: output
            )
        }
        try block(
            input: input, neighbors: nil, checkpoint: checkpoint,
            offsets: offsets, tokenCount: 0, channels: channels,
            convolutionOutput: conv, normalized: norm, expanded: expanded,
            branch: branch, output: output
        )
    }

    @Test(
        "real shape-decoder ConvNeXt block matches the pinned physical-MPS oracle",
        .enabled(
            if: ProcessInfo.processInfo.environment["KG_TRELLIS2_SHAPE_DECODER_CHECKPOINT"] != nil,
            "Set KG_TRELLIS2_SHAPE_DECODER_CHECKPOINT to execute real decoder conformance"
        )
    )
    func realShapeDecoderBlock() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["KG_TRELLIS2_SHAPE_DECODER_CHECKPOINT"]
        )
        let fixtureURL = try #require(Bundle.module.url(
            forResource: "shape-decoder-block0-0-mps", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(fileSHA256(at: fixtureURL) ==
            "546445da98a3e15cc184a6cbc2e32313dd8316ae9f75f7e8d51d10feeb1ac48d")
        let metadataURL = fixtureURL.appendingPathExtension("json")
        try #require(fileSHA256(at: metadataURL) ==
            "da25dceb8093634deb98ad5ae1d0911377057220100abaf711b5db77187862b8")
        let metadata = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL)) as? [String: Any]
        )
        #expect(metadata["source_revision"] as? String ==
            "75fbf0183001ed9876c8dbb35de6b68552ee08bd")
        #expect(metadata["pytorch_enable_mps_fallback"] as? String == "0")
        #expect(metadata["oracle_call"] as? String ==
            "SparseConvNeXtBlock3d(1024).forward with pinned MPS conv overlay")
        let expected = try Data(contentsOf: fixtureURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        let context = try MetalContext()
        let checkpoint = try MappedCheckpoint(
            url: URL(fileURLWithPath: path), device: context.device
        )
        try #require(checkpoint.sha256() ==
            "e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581")
        #expect(throws: NativeRuntimeError.self) {
            _ = try SparseConvNeXtOffsets(
                checkpoint: checkpoint, stage: 0, block: 0, channels: 512
            )
        }
        let offsets = try SparseConvNeXtOffsets(
            checkpoint: checkpoint, stage: 0, block: 0, channels: 1024
        )
        let coordinates = [
            SparseStructureCoordinate(x: 0, y: 0, z: 0),
            SparseStructureCoordinate(x: 1, y: 0, z: 0),
            SparseStructureCoordinate(x: 1, y: 1, z: 0),
            SparseStructureCoordinate(x: 1, y: 1, z: 1),
        ]
        let map = try SparseNeighborhood3x3(
            coordinates: coordinates, spatialShape: SparseSpatialShape(cubic: 2)
        )
        let tokens = 4, channels = 1024, elements = tokens * channels
        let metrics: [(String, Double, Float)] = try autoreleasepool {
            var checkpointBuffer: MTLBuffer? = try checkpoint.acquireBuffer()
            var inputs = Array(expected[0..<elements])
            let input = try #require(context.device.makeBuffer(
                bytes: &inputs, length: elements * 4, options: .storageModeShared
            ))
            let neighbors = try #require(try map.makeMetalBuffer(context: context))
            func buffer(_ count: Int) throws -> MTLBuffer {
                try #require(context.device.makeBuffer(length: count * 4, options: .storageModeShared))
            }
            let conv = try buffer(elements), norm = try buffer(elements)
            let expanded = try buffer(tokens * channels * 4)
            let mlpUpTrace = try buffer(tokens * channels * 4)
            let branch = try buffer(elements), output = try buffer(elements)
            let block = try SparseConvNeXtBlock(context: context)
            try block(
                input: input, neighbors: neighbors, checkpoint: checkpointBuffer!,
                offsets: offsets, tokenCount: tokens, channels: channels,
                convolutionOutput: conv, normalized: norm, expanded: expanded,
                branch: branch, output: output,
                convolutionImplementation: .simdgroupMatrix
            )
            let dense = try DenseKernel(context: context)
            try dense.linearF16WeightsF32Output(
                input: norm, checkpoint: checkpointBuffer!,
                weightOffset: offsets.mlpUpWeight, biasOffset: offsets.mlpUpBias,
                rows: tokens, inputChannels: channels, outputChannels: channels * 4,
                output: mlpUpTrace
            )
            try SparseStructureKernel(context: context).roundF16F32(
                input: mlpUpTrace, count: tokens * channels * 4, output: mlpUpTrace
            )
            let traces: [(String, Int, Int, MTLBuffer)] = [
                ("conv", 4096, elements, conv),
                ("norm", 8192, elements, norm),
                ("mlp_up", 12288, tokens * channels * 4, mlpUpTrace),
                ("silu", 28672, tokens * channels * 4, expanded),
                ("mlp_down", 45056, elements, branch),
                ("output", 49152, elements, output),
            ]
            var metrics: [(String, Double, Float)] = []
            for (name, expectedOffset, count, buffer) in traces {
                let actual = buffer.contents().assumingMemoryBound(to: Float.self)
                var squaredError = 0.0, squaredExpected = 0.0
                var maximumError: Float = 0, maximumMagnitude: Float = 0
                for index in 0..<count {
                    let reference = expected[expectedOffset + index]
                    let error = abs(actual[index] - reference)
                    maximumError = max(maximumError, error)
                    maximumMagnitude = max(maximumMagnitude, abs(reference))
                    squaredError += Double(error * error)
                    squaredExpected += Double(reference * reference)
                }
                metrics.append((
                    name,
                    sqrt(squaredError / max(squaredExpected, 1e-30)),
                    maximumError / max(maximumMagnitude, 1e-20)
                ))
            }
            checkpointBuffer = nil
            return metrics
        }
        try context.waitUntilIdle()
        for (name, normalizedRMS, scaleRatio) in metrics {
            print("shape decoder block \(name): normalized_rms=\(normalizedRMS) scale=\(scaleRatio)")
            // Every oracle boundary is F16. Two unit-roundoffs leave room for
            // different legal reduction orders without masking graph drift.
            #expect(normalizedRMS <= 0.001)
            #expect(scaleRatio <= 0.002)
        }
    }

    @Test(
        "real shape-decoder C2S block matches subdivision and output MPS oracle",
        .enabled(
            if: ProcessInfo.processInfo.environment["KG_TRELLIS2_SHAPE_DECODER_CHECKPOINT"] != nil,
            "Set KG_TRELLIS2_SHAPE_DECODER_CHECKPOINT to execute real C2S conformance"
        )
    )
    func realShapeDecoderC2SBlock() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["KG_TRELLIS2_SHAPE_DECODER_CHECKPOINT"]
        )
        let fixtureURL = try #require(Bundle.module.url(
            forResource: "shape-decoder-c2s0-4-mps", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(fileSHA256(at: fixtureURL) ==
            "d0b1a32552b7a4eaec30c60d990c6263b99ae68c6a2137b5c1b41af7702ed167")
        let metadataURL = fixtureURL.appendingPathExtension("json")
        try #require(fileSHA256(at: metadataURL) ==
            "aaa3b27e0f38047dd7241bc9df9a444a56751d754d8998f57283ed1505595260")
        let metadata = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL)) as? [String: Any]
        )
        #expect(metadata["oracle_call"] as? String ==
            "SparseResBlockC2S3d(1024,512).forward with pinned MPS conv overlay")
        let expected = try Data(contentsOf: fixtureURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        let rawCoordinates = try #require(metadata["coordinates"] as? [[Int]])
        let rawOutputCoordinates = try #require(metadata["output_coordinates"] as? [[Int]])
        func coordinates(_ values: [[Int]]) -> [SparseStructureCoordinate] {
            values.map {
                SparseStructureCoordinate(
                    batch: Int32($0[0]), x: Int32($0[1]), y: Int32($0[2]), z: Int32($0[3])
                )
            }
        }
        let parentCoordinates = coordinates(rawCoordinates)
        let context = try MetalContext(arenaCapacity: 64 * 1024 * 1024)
        let checkpoint = try MappedCheckpoint(
            url: URL(fileURLWithPath: path), device: context.device
        )
        let layout = try SparseC2SCheckpointLayout(
            checkpoint: checkpoint, stage: 0, block: 4,
            inputChannels: 1024, outputChannels: 512
        )
        let result: SparseC2SResult = try autoreleasepool {
            var checkpointBuffer: MTLBuffer? = try checkpoint.acquireBuffer()
            var inputs = Array(expected[0..<(parentCoordinates.count * 1024)])
            let input = try #require(context.device.makeBuffer(
                bytes: &inputs, length: inputs.count * 4, options: .storageModeShared
            ))
            let neighborhood = try SparseNeighborhood3x3(
                coordinates: parentCoordinates, spatialShape: SparseSpatialShape(cubic: 2)
            )
            let neighbors = try #require(try neighborhood.makeMetalBuffer(context: context))
            let result = try SparseC2SBlock(context: context)(
                input: input, coordinates: parentCoordinates,
                spatialShape: SparseSpatialShape(cubic: 2),
                neighborBuffer: neighbors, checkpoint: checkpointBuffer!, layout: layout,
                inputChannels: 1024, outputChannels: 512
            )
            checkpointBuffer = nil
            return result
        }
        #expect(result.coordinates == coordinates(rawOutputCoordinates))
        func compare(
            _ buffer: MTLBuffer, expectedOffset: Int, count: Int, name: String
        ) {
            let actual = buffer.contents().assumingMemoryBound(to: Float.self)
            var squaredError = 0.0, squaredExpected = 0.0
            var maximumError: Float = 0, maximumMagnitude: Float = 0
            for index in 0..<count {
                let reference = expected[expectedOffset + index]
                let error = abs(actual[index] - reference)
                maximumError = max(maximumError, error)
                maximumMagnitude = max(maximumMagnitude, abs(reference))
                squaredError += Double(error * error)
                squaredExpected += Double(reference * reference)
            }
            let normalizedRMS = sqrt(squaredError / max(squaredExpected, 1e-30))
            let scaleRatio = maximumError / max(maximumMagnitude, 1e-20)
            print("shape C2S \(name): normalized_rms=\(normalizedRMS) scale=\(scaleRatio)")
            #expect(normalizedRMS <= 0.002)
            #expect(scaleRatio <= 0.004)
        }
        compare(
            try #require(result.subdivisionLogits), expectedOffset: 3072,
            count: parentCoordinates.count * 8, name: "subdivision"
        )
        compare(
            try #require(result.features), expectedOffset: 40088,
            count: result.coordinates.count * 512, name: "output"
        )
    }

    @Test(
        "complete native sparse shape decoder matches the pinned MPS graph",
        .enabled(
            if: ProcessInfo.processInfo.environment["KG_TRELLIS2_SHAPE_DECODER_CHECKPOINT"] != nil,
            "Set KG_TRELLIS2_SHAPE_DECODER_CHECKPOINT for full decoder conformance"
        )
    )
    func completeShapeDecoder() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["KG_TRELLIS2_SHAPE_DECODER_CHECKPOINT"]
        )
        let fixtureURL = try #require(Bundle.module.url(
            forResource: "shape-decoder-full-mps", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(fileSHA256(at: fixtureURL) ==
            "5dc2be21657b0dc3ed82e4323251fbf0f17850f2c13d5f3e421ffec2e96a586b")
        let metadataURL = fixtureURL.appendingPathExtension("json")
        try #require(fileSHA256(at: metadataURL) ==
            "1e5f410c2614400af8c2292ca6a7445dba3b1ca8b087ad69596512356dd9b0b9")
        let metadata = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL)) as? [String: Any]
        )
        #expect(metadata["oracle_call"] as? String ==
            "SparseUnetVaeDecoder.forward(return_subs=True) with pinned MPS conv overlay")
        let expected = try Data(contentsOf: fixtureURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        let context = try MetalContext(arenaCapacity: 256 * 1024 * 1024)
        let checkpoint = try MappedCheckpoint(
            url: URL(fileURLWithPath: path), device: context.device
        )
        var latentValues = Array(expected[0..<32])
        let latent = try #require(context.device.makeBuffer(
            bytes: &latentValues, length: latentValues.count * 4, options: .storageModeShared
        ))
        let result = try ShapeSparseDecoder(context: context)(
            latent: latent,
            coordinates: [SparseStructureCoordinate(x: 0, y: 0, z: 0)],
            spatialShape: SparseSpatialShape(cubic: 1), checkpoint: checkpoint
        )
        let rawOutputCoordinates = try #require(metadata["output_coordinates"] as? [[Int]])
        let expectedCoordinates = rawOutputCoordinates.map {
            SparseStructureCoordinate(
                batch: Int32($0[0]), x: Int32($0[1]), y: Int32($0[2]), z: Int32($0[3])
            )
        }
        #expect(result.coordinates == expectedCoordinates)
        #expect(result.spatialShape == (try SparseSpatialShape(cubic: 16)))
        let subdivisionOffsets = [32, 40, 104, 200]
        let subdivisionCounts = [8, 64, 96, 176]
        #expect(result.subdivisionLogits.count == 4)
        for stage in 0..<4 {
            compareDecoderBuffer(
                result.subdivisionLogits[stage], expected: expected,
                expectedOffset: subdivisionOffsets[stage], count: subdivisionCounts[stage],
                name: "full subdivision \(stage)", normalizedCap: 0.002, scaleCap: 0.004
            )
        }
        compareDecoderBuffer(
            result.rawHead, expected: expected, expectedOffset: 376,
            count: result.coordinates.count * 7, name: "full raw head",
            normalizedCap: 0.003, scaleCap: 0.006
        )
        let snapshot = try #require(context.arena).snapshot()
        print("shape decoder full: tokens=\(result.coordinates.count) peak=\(snapshot.peakUsedBytes)")
        #expect(snapshot.peakUsedBytes <= 256 * 1024 * 1024)
    }

    @Test(
        "complete guided texture decoder matches the pinned MPS graph",
        .enabled(
            if: ProcessInfo.processInfo.environment["KG_TRELLIS2_TEXTURE_DECODER_CHECKPOINT"] != nil,
            "Set KG_TRELLIS2_TEXTURE_DECODER_CHECKPOINT for texture decoder conformance"
        )
    )
    func completeTextureDecoder() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["KG_TRELLIS2_TEXTURE_DECODER_CHECKPOINT"]
        )
        let fixtureURL = try #require(Bundle.module.url(
            forResource: "texture-decoder-full-mps", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(fileSHA256(at: fixtureURL) ==
            "38cda3cd5ce04b41bccb6f70bc82a744e6df7a45378fbbf02e15e7fd38b63d8a")
        let metadataURL = fixtureURL.appendingPathExtension("json")
        try #require(fileSHA256(at: metadataURL) ==
            "97821c9de194c4c34e31b13e50167b813c03bb8ab616a1fee18a285f6c884aaa")
        let metadata = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL)) as? [String: Any]
        )
        #expect(metadata["oracle_call"] as? String ==
            "SparseUnetVaeDecoder(pred_subdiv=False).forward(guide_subs=shape_subs)")
        #expect(metadata["pipeline_transform"] as? String == "raw * 0.5 + 0.5")
        let expected = try Data(contentsOf: fixtureURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        let shapeFixtureURL = try #require(Bundle.module.url(
            forResource: "shape-decoder-full-mps", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(fileSHA256(at: shapeFixtureURL) ==
            "5dc2be21657b0dc3ed82e4323251fbf0f17850f2c13d5f3e421ffec2e96a586b")
        let shapeFixture = try Data(contentsOf: shapeFixtureURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        let context = try MetalContext(arenaCapacity: 256 * 1024 * 1024)
        let checkpoint = try MappedCheckpoint(
            url: URL(fileURLWithPath: path), device: context.device
        )
        try #require(checkpoint.sha256() ==
            "97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c")
        var coordinates = [SparseStructureCoordinate(x: 0, y: 0, z: 0)]
        let subdivisionOffsets = [32, 40, 104, 200]
        let subdivisionParentCounts = [1, 8, 12, 22]
        var guides: [SparseSubdivision2x] = []
        for stage in 0..<4 {
            var logits = Array(shapeFixture[
                subdivisionOffsets[stage]..<(subdivisionOffsets[stage] + subdivisionParentCounts[stage] * 8)
            ])
            let buffer = try #require(context.device.makeBuffer(
                bytes: &logits, length: logits.count * 4, options: .storageModeShared
            ))
            let guide = try SparseSubdivision2x(
                parentCoordinates: coordinates, logits: buffer
            )
            guides.append(guide)
            coordinates = guide.coordinates
        }
        var latentValues = Array(expected[0..<32])
        let latent = try #require(context.device.makeBuffer(
            bytes: &latentValues, length: latentValues.count * 4,
            options: .storageModeShared
        ))
        let result = try TextureSparseDecoder(context: context)(
            latent: latent,
            coordinates: [SparseStructureCoordinate(x: 0, y: 0, z: 0)],
            spatialShape: SparseSpatialShape(cubic: 1),
            subdivisionGuides: guides, checkpoint: checkpoint
        )
        let rawOutputCoordinates = try #require(metadata["output_coordinates"] as? [[Int]])
        let expectedCoordinates = rawOutputCoordinates.map {
            SparseStructureCoordinate(
                batch: Int32($0[0]), x: Int32($0[1]), y: Int32($0[2]), z: Int32($0[3])
            )
        }
        #expect(result.coordinates == expectedCoordinates)
        #expect(result.spatialShape == (try SparseSpatialShape(cubic: 16)))
        compareDecoderBuffer(
            result.rawHead, expected: expected, expectedOffset: 32,
            count: result.coordinates.count * 6, name: "texture raw head",
            normalizedCap: 0.002, scaleCap: 0.004
        )
        compareDecoderBuffer(
            result.pbrFields, expected: expected, expectedOffset: 386,
            count: result.coordinates.count * 6, name: "texture PBR transform",
            normalizedCap: 0.002, scaleCap: 0.004
        )
        let snapshot = try #require(context.arena).snapshot()
        print("texture decoder full: tokens=\(result.coordinates.count) peak=\(snapshot.peakUsedBytes)")
        #expect(snapshot.peakUsedBytes <= 256 * 1024 * 1024)
    }
}

private func compareDecoderBuffer(
    _ buffer: MTLBuffer, expected: [Float], expectedOffset: Int,
    count: Int, name: String, normalizedCap: Double, scaleCap: Float
) {
    let actual = buffer.contents().assumingMemoryBound(to: Float.self)
    var squaredError = 0.0, squaredExpected = 0.0
    var maximumError: Float = 0, maximumMagnitude: Float = 0
    for index in 0..<count {
        let reference = expected[expectedOffset + index]
        let error = abs(actual[index] - reference)
        maximumError = max(maximumError, error)
        maximumMagnitude = max(maximumMagnitude, abs(reference))
        squaredError += Double(error * error)
        squaredExpected += Double(reference * reference)
    }
    let normalizedRMS = sqrt(squaredError / max(squaredExpected, 1e-30))
    let scaleRatio = maximumError / max(maximumMagnitude, 1e-20)
    print("shape decoder \(name): normalized_rms=\(normalizedRMS) scale=\(scaleRatio)")
    #expect(normalizedRMS <= normalizedCap)
    #expect(scaleRatio <= scaleCap)
}

private func f16(_ value: Float) -> UInt16 {
    Float16(value).bitPattern
}

private func fromF16(_ value: UInt16) -> Float {
    Float(Float16(bitPattern: value))
}
