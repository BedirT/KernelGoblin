import Metal
import Testing
@testable import KernelGoblinTrellis2

@Suite("Native sparse decoder kernels")
struct SparseDecoderKernelTests {
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
}

private func f16(_ value: Float) -> UInt16 {
    Float16(value).bitPattern
}

private func fromF16(_ value: UInt16) -> Float {
    Float(Float16(bitPattern: value))
}
