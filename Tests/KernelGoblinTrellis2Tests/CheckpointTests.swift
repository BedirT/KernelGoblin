import Foundation
import Metal
import Testing
@testable import KernelGoblinTrellis2

@Suite("Native TRELLIS.2 checkpoint and memory contracts")
struct CheckpointTests {
    @Test("safetensors validates shape, dtype, and absolute payload offset")
    func parsesSafeTensors() throws {
        let headerObject: [String: Any] = [
            "weight": [
                "dtype": "F16",
                "shape": [2, 3],
                "data_offsets": [0, 12],
            ]
        ]
        var header = try JSONSerialization.data(withJSONObject: headerObject, options: [.sortedKeys])
        while header.count % 8 != 0 { header.append(0x20) }
        var length = UInt64(header.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(header)
        data.append(Data(repeating: 0x2A, count: 12))
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("kg-safe-\(UUID().uuidString).safetensors")
        try data.write(to: url, options: .atomic)
        defer { try? FileManager.default.removeItem(at: url) }

        let index = try SafeTensorsIndex.read(from: url)
        let weight = try #require(index.tensors["weight"])
        #expect(weight.dtype == .f16)
        #expect(weight.shape == [2, 3])
        #expect(weight.byteCount == 12)
        #expect(weight.fileOffset == UInt64(8 + header.count))
    }

    @Test("safetensors follows a checkpoint-cache symlink")
    func followsSymlink() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("kg-safe-link-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let header = try JSONSerialization.data(withJSONObject: [
            "value": ["dtype": "U8", "shape": [1], "data_offsets": [0, 1]]
        ])
        var length = UInt64(header.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(header)
        data.append(7)
        let target = directory.appendingPathComponent("blob")
        let link = directory.appendingPathComponent("model.safetensors")
        try data.write(to: target)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        #expect(try SafeTensorsIndex.read(from: link).tensors["value"]?.byteCount == 1)
    }

    @Test("checkpoint is mapped into a no-copy Metal buffer")
    func mapsCheckpointIntoMetal() throws {
        let header = try JSONSerialization.data(withJSONObject: [
            "weight": ["dtype": "F32", "shape": [2], "data_offsets": [0, 8]]
        ])
        var length = UInt64(header.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(header)
        var values: [Float] = [1.25, -2.5]
        data.append(values.withUnsafeBytes { Data($0) })
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("kg-map-\(UUID().uuidString).safetensors")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let checkpoint = try MappedCheckpoint(url: url, device: device)
        #expect(checkpoint.validByteCount == UInt64(data.count))
        #expect(checkpoint.mappedByteCount % UInt64(getpagesize()) == 0)
        let range = try checkpoint.byteRange(for: "weight")
        let pointer = checkpoint.buffer.contents().advanced(by: range.lowerBound)
            .assumingMemoryBound(to: Float.self)
        #expect(pointer[0] == 1.25)
        #expect(pointer[1] == -2.5)
        #expect(try checkpoint.sha256() == fileSHA256(at: url))
        values.removeAll()
    }

    @Test("safetensors rejects shape and range disagreement")
    func rejectsBadSafeTensors() throws {
        let header = try JSONSerialization.data(withJSONObject: [
            "bad": ["dtype": "F32", "shape": [4], "data_offsets": [0, 4]]
        ])
        var length = UInt64(header.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(header)
        data.append(Data(repeating: 0, count: 4))
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("kg-bad-\(UUID().uuidString).safetensors")
        try data.write(to: url, options: .atomic)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: CheckpointError.self) { try SafeTensorsIndex.read(from: url) }
    }

    @Test("safetensors rejects non-integer dimensions and overlapping ranges")
    func rejectsMalformedMetadata() throws {
        let cases: [[String: Any]] = [
            ["weight": ["dtype": "F32", "shape": [1.5], "data_offsets": [0, 4]]],
            [
                "first": ["dtype": "F32", "shape": [1], "data_offsets": [0, 4]],
                "second": ["dtype": "F32", "shape": [1], "data_offsets": [0, 4]],
            ],
        ]
        for headerObject in cases {
            let header = try JSONSerialization.data(withJSONObject: headerObject)
            var length = UInt64(header.count).littleEndian
            var data = withUnsafeBytes(of: &length) { Data($0) }
            data.append(header)
            data.append(Data(repeating: 0, count: 8))
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("kg-malformed-\(UUID().uuidString).safetensors")
            try data.write(to: url)
            defer { try? FileManager.default.removeItem(at: url) }
            #expect(throws: CheckpointError.self) { try SafeTensorsIndex.read(from: url) }
        }
    }

    @Test("bounded Metal scratch refuses growth")
    func boundedScratch() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let scratch = try BoundedScratch(device: device, capacity: 1024, label: "test")
        #expect(try scratch.range(offset: 128, count: 512) == 128..<640)
        #expect(throws: NativeRuntimeError.self) {
            try scratch.range(offset: 900, count: 200)
        }
    }

    @Test("Metal resources compile on the physical device")
    func compilesMetalLibrary() throws {
        let context = try MetalContext()
        let library = try context.library(named: "identity")
        #expect(library.makeFunction(name: "kg_identity_f32") != nil)
    }

    @Test("Metal float32 dense projection matches CPU")
    func denseProjection() throws {
        let context = try MetalContext()
        let kernel = try DenseKernel(context: context)
        var inputValues: [Float] = [1, -2, 0.5, 3, 1, -1]
        var checkpointValues: [Float] = [
            1, 2, 3, -1, 0.5, 4,
            0.25, -0.75,
        ]
        let input = try #require(context.device.makeBuffer(
            bytes: &inputValues, length: inputValues.count * 4, options: .storageModeShared
        ))
        let checkpoint = try #require(context.device.makeBuffer(
            bytes: &checkpointValues, length: checkpointValues.count * 4, options: .storageModeShared
        ))
        let output = try #require(context.device.makeBuffer(length: 4 * 4, options: .storageModeShared))
        try kernel.linearF32(
            input: input, checkpoint: checkpoint, weightOffset: 0,
            biasOffset: 6 * 4, rows: 2, inputChannels: 3,
            outputChannels: 2, output: output
        )
        let actual = Array(UnsafeBufferPointer(
            start: output.contents().assumingMemoryBound(to: Float.self), count: 4
        ))
        #expect(abs(actual[0] - -1.25) < 1e-6)
        #expect(abs(actual[1] - -0.75) < 1e-6)
        #expect(abs(actual[2] - 2.25) < 1e-6)
        #expect(abs(actual[3] - -7.25) < 1e-6)
    }

    @Test("Metal BF16 checkpoint projection widens weights to float32")
    func bf16DenseProjection() throws {
        let context = try MetalContext()
        let kernel = try DenseKernel(context: context)
        var inputValues: [Float] = [1, -2, 0.5]
        var checkpointValues: [UInt16] = [
            bf16(1), bf16(2), bf16(3),
            bf16(-1), bf16(0.5), bf16(4),
            bf16(0.25), bf16(-0.75),
        ]
        let input = try #require(context.device.makeBuffer(
            bytes: &inputValues, length: inputValues.count * 4, options: .storageModeShared
        ))
        let checkpoint = try #require(context.device.makeBuffer(
            bytes: &checkpointValues, length: checkpointValues.count * 2, options: .storageModeShared
        ))
        let output = try #require(context.device.makeBuffer(length: 2 * 4, options: .storageModeShared))
        try kernel.linearBF16WeightsF32Output(
            input: input, checkpoint: checkpoint, weightOffset: 0,
            biasOffset: 6 * 2, rows: 1, inputChannels: 3,
            outputChannels: 2, output: output
        )
        let actual = output.contents().assumingMemoryBound(to: Float.self)
        #expect(abs(actual[0] - -1.25) < 1e-6)
        #expect(abs(actual[1] - -0.75) < 1e-6)
    }

    @Test("Metal timestep embedding and SiLU match CPU formulas")
    func primitiveMath() throws {
        let context = try MetalContext()
        let kernel = try PrimitiveKernel(context: context)
        var timesteps: [Float] = [0.5, 1000]
        let timestepBuffer = try #require(context.device.makeBuffer(
            bytes: &timesteps, length: timesteps.count * 4, options: .storageModeShared
        ))
        let embedding = try #require(context.device.makeBuffer(
            length: timesteps.count * 5 * 4, options: .storageModeShared
        ))
        try kernel.timestepEmbeddingF32(
            timesteps: timestepBuffer, rows: 2, dimensions: 5, output: embedding
        )
        let actualEmbedding = embedding.contents().assumingMemoryBound(to: Float.self)
        for row in 0..<2 {
            for column in 0..<5 {
                let expected: Float
                if column == 4 {
                    expected = 0
                } else {
                    let frequencyIndex = column % 2
                    let frequency = exp(-log(Float(10_000)) * Float(frequencyIndex) / 2)
                    let phase = timesteps[row] * frequency
                    expected = column < 2 ? cos(phase) : sin(phase)
                }
                #expect(abs(actualEmbedding[row * 5 + column] - expected) < 2e-6)
            }
        }

        var values: [Float] = [-4, -1, 0, 1, 4]
        let input = try #require(context.device.makeBuffer(
            bytes: &values, length: values.count * 4, options: .storageModeShared
        ))
        let output = try #require(context.device.makeBuffer(
            length: values.count * 4, options: .storageModeShared
        ))
        try kernel.siluF32(input: input, count: values.count, output: output)
        let actualSiLU = output.contents().assumingMemoryBound(to: Float.self)
        for index in values.indices {
            let expected = values[index] / (1 + exp(-values[index]))
            #expect(abs(actualSiLU[index] - expected) < 2e-6)
        }


        var packed: [Float] = [
            1.003, 2.007, 3.011, 4.015,
            5.019, 6.023, 7.027, 8.031,
            9.035, 10.039, 11.043, 12.047,
        ]
        let packedBuffer = try #require(context.device.makeBuffer(
            bytes: &packed, length: packed.count * 4, options: .storageModeShared
        ))
        let rounded = try #require(context.device.makeBuffer(
            length: packed.count * 4, options: .storageModeShared
        ))
        try kernel.roundBF16F32(input: packedBuffer, count: packed.count, output: rounded)
        let actualRounded = rounded.contents().assumingMemoryBound(to: Float.self)
        for index in packed.indices {
            #expect(actualRounded[index].bitPattern == fromBF16(roundedBF16(packed[index])).bitPattern)
        }
        let q = try #require(context.device.makeBuffer(length: 4 * 4, options: .storageModeShared))
        let k = try #require(context.device.makeBuffer(length: 4 * 4, options: .storageModeShared))
        let v = try #require(context.device.makeBuffer(length: 4 * 4, options: .storageModeShared))
        try kernel.splitQKVF32(
            input: packedBuffer, rows: 1, channels: 4, query: q, key: k, value: v
        )
        for (buffer, offset) in [(q, 0), (k, 4), (v, 8)] {
            let values = buffer.contents().assumingMemoryBound(to: Float.self)
            for index in 0..<4 { #expect(values[index] == packed[offset + index]) }
        }
        let kvKey = try #require(context.device.makeBuffer(length: 4 * 4, options: .storageModeShared))
        let kvValue = try #require(context.device.makeBuffer(length: 4 * 4, options: .storageModeShared))
        try kernel.splitKVF32(
            input: packedBuffer, rows: 1, channels: 4, key: kvKey, value: kvValue
        )
        for (buffer, offset) in [(kvKey, 0), (kvValue, 4)] {
            let values = buffer.contents().assumingMemoryBound(to: Float.self)
            for index in 0..<4 { #expect(values[index] == packed[offset + index]) }
        }

        var modulationValues: [Float] = [
            0.1, -0.2, 0.3, -0.4,
            0.5, 0.25, -0.5, -0.25,
            2, 3, 4, 5,
        ]
        let modulation = try #require(context.device.makeBuffer(
            bytes: &modulationValues, length: modulationValues.count * 4,
            options: .storageModeShared
        ))
        let modulated = try #require(context.device.makeBuffer(
            length: values.count * 4, options: .storageModeShared
        ))
        try kernel.modulateF32(
            input: input, modulation: modulation, rows: 1, channels: 4,
            shiftOffset: 0, scaleOffset: 4, output: modulated
        )
        let actualModulated = modulated.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<4 {
            let expected = values[index] * (1 + modulationValues[4 + index])
                + modulationValues[index]
            #expect(abs(actualModulated[index] - expected) < 1e-6)
        }
        let residual = try #require(context.device.makeBuffer(
            length: values.count * 4, options: .storageModeShared
        ))
        try kernel.residualF32(
            residual: input, branch: modulated, modulation: modulation,
            rows: 1, channels: 4, gateOffset: 8, output: residual
        )
        let actualResidual = residual.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<4 {
            let expected = values[index] + actualModulated[index] * modulationValues[8 + index]
            #expect(abs(actualResidual[index] - expected) < 1e-6)
        }
        let gelu = try #require(context.device.makeBuffer(
            length: values.count * 4, options: .storageModeShared
        ))
        try kernel.geluTanhF32(input: input, count: values.count, output: gelu)
        let actualGELU = gelu.contents().assumingMemoryBound(to: Float.self)
        for index in values.indices {
            let x = values[index]
            let expected = 0.5 * x * (1 + tanh(
                sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)
            ))
            #expect(abs(actualGELU[index] - expected) < 2e-6)
        }
        var extremeValues: [Float] = [-20, -13.5625, 13.5625, 20]
        let extremes = try #require(context.device.makeBuffer(
            bytes: &extremeValues, length: extremeValues.count * 4,
            options: .storageModeShared
        ))
        try kernel.geluTanhF32(input: extremes, count: extremeValues.count, output: extremes)
        let actualExtremes = extremes.contents().assumingMemoryBound(to: Float.self)
        #expect(actualExtremes[0] == 0 && actualExtremes[1] == 0)
        #expect(actualExtremes[2] == extremeValues[2] && actualExtremes[3] == extremeValues[3])
    }

    @Test("Metal LayerNorm32 and multi-head RMSNorm match CPU formulas")
    func normalizationMath() throws {
        let context = try MetalContext()
        let kernel = try NormalizationKernel(context: context)
        var inputValues: [Float] = [1, 2, 4, 8, -3, 0.5, 2.5, 9]
        var checkpointValues: [UInt16] = [
            bf16(1), bf16(0.5), bf16(2), bf16(-1),
            bf16(0.25), bf16(-0.5), bf16(1), bf16(2),
        ]
        let input = try #require(context.device.makeBuffer(
            bytes: &inputValues, length: inputValues.count * 4, options: .storageModeShared
        ))
        let checkpoint = try #require(context.device.makeBuffer(
            bytes: &checkpointValues, length: checkpointValues.count * 2,
            options: .storageModeShared
        ))
        let layerOutput = try #require(context.device.makeBuffer(
            length: inputValues.count * 4, options: .storageModeShared
        ))
        try kernel.layerNormF32(
            input: input, checkpoint: checkpoint, rows: 2, channels: 4,
            weightOffset: 0, biasOffset: 8, output: layerOutput
        )
        let actualLayer = layerOutput.contents().assumingMemoryBound(to: Float.self)
        for row in 0..<2 {
            let values = Array(inputValues[(row * 4)..<(row * 4 + 4)])
            let mean = values.reduce(0, +) / 4
            let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / 4
            for channel in 0..<4 {
                let normalized = (values[channel] - mean) / sqrt(variance + 1e-6)
                let expected = normalized * fromBF16(checkpointValues[channel])
                    + fromBF16(checkpointValues[4 + channel])
                #expect(abs(actualLayer[row * 4 + channel] - expected) < 2e-6)
            }
        }

        let rmsOutput = try #require(context.device.makeBuffer(
            length: inputValues.count * 4, options: .storageModeShared
        ))
        try kernel.multiheadRMSNormF32(
            input: input, checkpoint: checkpoint, gammaOffset: 0,
            rows: 2, heads: 2, dimensions: 2, output: rmsOutput
        )
        let actualRMS = rmsOutput.contents().assumingMemoryBound(to: Float.self)
        for group in 0..<4 {
            let x0 = inputValues[group * 2]
            let x1 = inputValues[group * 2 + 1]
            let inverse = 1 / max(sqrt(x0 * x0 + x1 * x1), 1e-12)
            let head = group % 2
            for dimension in 0..<2 {
                let expected = inputValues[group * 2 + dimension] * inverse
                    * fromBF16(checkpointValues[head * 2 + dimension]) * sqrt(2)
                #expect(abs(actualRMS[group * 2 + dimension] - expected) < 2e-6)
            }
        }
    }

    @Test("Metal fused self and cross attention match stable CPU softmax")
    func fusedAttention() throws {
        let context = try MetalContext()
        let kernel = try AttentionKernel(context: context)
        let queryCount = 2, keyCount = 3, heads = 2, dimensions = 4
        var queries = (0..<(queryCount * heads * dimensions)).map {
            Float(sin(Double($0) * 0.31))
        }
        var keys = (0..<(keyCount * heads * dimensions)).map {
            Float(cos(Double($0) * 0.17) * 0.8)
        }
        var values = (0..<(keyCount * heads * dimensions)).map {
            Float(sin(Double($0) * 0.11) - 0.2)
        }
        let queryBuffer = try #require(context.device.makeBuffer(
            bytes: &queries, length: queries.count * 4, options: .storageModeShared
        ))
        let keyBuffer = try #require(context.device.makeBuffer(
            bytes: &keys, length: keys.count * 4, options: .storageModeShared
        ))
        let valueBuffer = try #require(context.device.makeBuffer(
            bytes: &values, length: values.count * 4, options: .storageModeShared
        ))
        let output = try #require(context.device.makeBuffer(
            length: queries.count * 4, options: .storageModeShared
        ))
        try kernel.fusedF32(
            queries: queryBuffer, keys: keyBuffer, values: valueBuffer,
            queryCount: queryCount, keyCount: keyCount, heads: heads,
            dimensions: dimensions, output: output
        )
        let actual = output.contents().assumingMemoryBound(to: Float.self)
        let scale = 1 / sqrt(Float(dimensions))
        for query in 0..<queryCount {
            for head in 0..<heads {
                let queryBase = (query * heads + head) * dimensions
                var scores = [Float](repeating: 0, count: keyCount)
                for key in 0..<keyCount {
                    let keyBase = (key * heads + head) * dimensions
                    for dimension in 0..<dimensions {
                        scores[key].addProduct(
                            queries[queryBase + dimension], keys[keyBase + dimension]
                        )
                    }
                    scores[key] *= scale
                }
                let maximum = scores.max()!
                let weights = scores.map { exp($0 - maximum) }
                let denominator = weights.reduce(0, +)
                for dimension in 0..<dimensions {
                    var expected: Float = 0
                    for key in 0..<keyCount {
                        let keyBase = (key * heads + head) * dimensions
                        expected += weights[key] / denominator * values[keyBase + dimension]
                    }
                    #expect(abs(actual[queryBase + dimension] - expected) < 3e-6)
                }
            }
        }
    }

    @Test(
        "real TRELLIS.2 no-RoPE block core matches the pinned Torch BF16 oracle",
        .enabled(
            if: ProcessInfo.processInfo.environment["KG_TRELLIS2_SHAPE_FLOW_CHECKPOINT"] != nil,
            "Set KG_TRELLIS2_SHAPE_FLOW_CHECKPOINT to execute real-weight conformance"
        )
    )
    func realSLatBlockGolden() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["KG_TRELLIS2_SHAPE_FLOW_CHECKPOINT"]
        )
        let checkpointURL = URL(fileURLWithPath: path)
        let context = try MetalContext()
        let checkpoint = try MappedCheckpoint(url: checkpointURL, device: context.device)
        try #require(checkpoint.sha256() == "ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f")
        let tokens = 2
        let stageElements = tokens * 1536
        var features = (0..<stageElements).map {
            fromBF16(roundedBF16(Float(sin(Double($0) * 0.013) * 0.35)))
        }
        var modulationValues = (0..<9216).map {
            fromBF16(roundedBF16(Float(sin(Double($0) * 0.007) * 0.2)))
        }
        var contextValues = (0..<(2 * 1024)).map {
            fromBF16(roundedBF16(Float(sin(Double($0) * 0.015) * 0.25)))
        }
        let featureBuffer = try #require(context.device.makeBuffer(
            bytes: &features, length: features.count * 4, options: .storageModeShared
        ))
        let modulationBuffer = try #require(context.device.makeBuffer(
            bytes: &modulationValues, length: modulationValues.count * 4,
            options: .storageModeShared
        ))
        let contextBuffer = try #require(context.device.makeBuffer(
            bytes: &contextValues, length: contextValues.count * 4, options: .storageModeShared
        ))
        var tracedBuffers: [String: MTLBuffer] = [:]
        let actualBuffer = try SLatBlock(context: context).forwardF32(
            input: featureBuffer, sharedModulation: modulationBuffer,
            conditioning: contextBuffer, checkpoint: checkpoint,
            block: 0, tokens: tokens, conditioningTokens: 2, trace: { name, buffer in
            tracedBuffers[name] = buffer
        })
        let fixtureURL = try #require(Bundle.module.url(
            forResource: "slat-block0-tiny", withExtension: "bf16", subdirectory: "Fixtures"
        ))
        try #require(fileSHA256(at: fixtureURL) == "6c33a91ee7588407ef84926d42cf5181127126e72640b1fc71d0e4ae68dbb857")
        let golden = try Data(contentsOf: fixtureURL).withUnsafeBytes { raw in
            Array(raw.bindMemory(to: UInt16.self))
        }
        try #require(golden.count == stageElements)
        let actual = actualBuffer.contents().assumingMemoryBound(to: Float.self)
        var mismatches = 0
        var maximumULPDistance = 0
        var nonFiniteValues = 0
        var squaredError: Double = 0
        var maximumAbsoluteError: Float = 0
        for index in golden.indices {
            let expectedBits = UInt16(littleEndian: golden[index])
            let expected = fromBF16(expectedBits)
            guard actual[index].isFinite else {
                nonFiniteValues += 1
                continue
            }
            maximumAbsoluteError = max(maximumAbsoluteError, abs(actual[index] - expected))
            let error = abs(actual[index] - expected)
            squaredError += Double(error * error)
            let actualBits = roundedBF16(actual[index])
            maximumULPDistance = max(
                maximumULPDistance, bf16ULPDistance(actualBits, expectedBits)
            )
            if actualBits != expectedBits {
                mismatches += 1
            }
        }
        let traceURL = try #require(Bundle.module.url(
            forResource: "slat-block0-tiny", withExtension: "bf16.trace",
            subdirectory: "Fixtures"
        ))
        try #require(fileSHA256(at: traceURL) == "6316473db08d5cd5aac341685b5e6066cd4a6c79d4841c915a810a40fafdc2ef")
        let trace = try Data(contentsOf: traceURL).withUnsafeBytes { raw in
            Array(raw.bindMemory(to: UInt16.self))
        }
        let stages: [(String, Int)] = [
            ("norm1", stageElements), ("self_input", stageElements),
            ("self_output", stageElements), ("after_self", stageElements),
            ("norm2", stageElements), ("cross_output", stageElements),
            ("after_cross", stageElements), ("norm3", stageElements),
            ("mlp_input", stageElements), ("mlp_hidden_linear", tokens * 8192),
            ("mlp_hidden_gelu", tokens * 8192), ("mlp_output", stageElements),
            ("output", stageElements),
        ]
        try #require(trace.count == stages.reduce(0) { $0 + $1.1 })
        var traceOffset = 0
        for (name, elements) in stages {
            let buffer = try #require(tracedBuffers[name])
            let values = buffer.contents().assumingMemoryBound(to: Float.self)
            var stageMaximum: Float = 0
            var stageMismatches = 0
            var stageSquaredError: Double = 0
            var stageMaximumRelative: Float = 0
            var stageMagnitude: Float = 0
            var stageMaximumULP = 0
            var stageNonFinite = 0
            var reportedNonFinite = false
            for index in 0..<elements {
                let expectedBits = UInt16(littleEndian: trace[traceOffset + index])
                let expected = fromBF16(expectedBits)
                if !values[index].isFinite, !reportedNonFinite {
                    let linear = tracedBuffers["mlp_hidden_linear"]?.contents()
                        .assumingMemoryBound(to: Float.self)[index]
                    print("trace_nonfinite stage=\(name) index=\(index) previous_linear=\(String(describing: linear)) expected=\(expected)")
                    reportedNonFinite = true
                }
                guard values[index].isFinite else {
                    stageNonFinite += 1
                    continue
                }
                let error = abs(values[index] - expected)
                stageMaximum = max(stageMaximum, error)
                stageSquaredError += Double(error * error)
                stageMaximumRelative = max(
                    stageMaximumRelative, error / max(abs(expected), 0.125)
                )
                stageMagnitude = max(stageMagnitude, abs(expected))
                let actualBits = roundedBF16(values[index])
                stageMaximumULP = max(
                    stageMaximumULP, bf16ULPDistance(actualBits, expectedBits)
                )
                if actualBits != expectedBits { stageMismatches += 1 }
            }
            traceOffset += elements
            let rms = sqrt(stageSquaredError / Double(elements))
            print("trace_stage=\(name) max_abs=\(stageMaximum) max_rel_floor_0.125=\(stageMaximumRelative) rms=\(rms) max_expected=\(stageMagnitude) bf16_mismatches=\(stageMismatches) max_ulp=\(stageMaximumULP)")
            #expect(stageNonFinite == 0)
            #expect(stageMaximum <= 0.25)
            #expect(rms <= 0.02)
        }
        let rootMeanSquareError = sqrt(squaredError / Double(golden.count))
        print("block_max_abs=\(maximumAbsoluteError) block_rms=\(rootMeanSquareError) bf16_mismatches=\(mismatches) max_ulp=\(maximumULPDistance) non_finite=\(nonFiniteValues)")
        #expect(nonFiniteValues == 0)
        #expect(maximumAbsoluteError <= 0.25)
        #expect(rootMeanSquareError <= 0.02)
    }
}

private func bf16(_ value: Float) -> UInt16 {
    UInt16(truncatingIfNeeded: value.bitPattern >> 16)
}

private func fromBF16(_ value: UInt16) -> Float {
    Float(bitPattern: UInt32(value) << 16)
}

private func roundedBF16(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    return UInt16(truncatingIfNeeded: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) >> 16)
}

private func bf16ULPDistance(_ lhs: UInt16, _ rhs: UInt16) -> Int {
    func ordered(_ bits: UInt16) -> Int {
        let value = Int(bits)
        return bits & 0x8000 == 0 ? value + 0x8000 : 0x7FFF - (value & 0x7FFF)
    }
    return abs(ordered(lhs) - ordered(rhs))
}
