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

    @Test("Metal arena records peak use, rejects overflow, and releases buffers")
    func boundedMetalArena() throws {
        let context = try MetalContext(arenaCapacity: 64 * 1024)
        let arena = try #require(context.arena)
        var buffer: MTLBuffer? = try context.makeBuffer(
            length: 16 * 1024, label: "arena lifetime test"
        )
        let live = arena.snapshot()
        #expect(live.capacityBytes == 64 * 1024)
        #expect(live.usedBytes >= 16 * 1024)
        #expect(live.peakUsedBytes == live.usedBytes)
        #expect(live.cumulativeRequestedBytes == 16 * 1024)
        #expect(live.allocationCount == 1)
        #expect(throws: NativeRuntimeError.self) {
            try context.makeBuffer(length: 128 * 1024, label: "arena overflow")
        }
        buffer = nil
        #expect(arena.snapshot().usedBytes == 0)
        #expect(arena.snapshot().peakUsedBytes == live.peakUsedBytes)
        _ = buffer
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

    @Test("Metal segmented attention isolates sparse samples")
    func segmentedAttention() throws {
        let context = try MetalContext()
        let kernel = try AttentionKernel(context: context)
        let heads = 2, dimensions = 8
        let querySegments = try AttentionSegments(offsets: [0, 2, 5])
        let keySegments = try AttentionSegments(offsets: [0, 3, 5])
        var queries = (0..<(querySegments.totalCount * heads * dimensions)).map {
            Float(cos(Double($0) * 0.17) * 0.8)
        }
        var keys = (0..<(keySegments.totalCount * heads * dimensions)).map {
            Float(sin(Double($0) * 0.13) * 0.7)
        }
        var values = (0..<(keySegments.totalCount * heads * dimensions)).map {
            Float(cos(Double($0) * 0.11) - 0.2)
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
        try kernel.segmentedF32(
            queries: queryBuffer, keys: keyBuffer, values: valueBuffer,
            querySegments: querySegments, keySegments: keySegments,
            heads: heads, dimensions: dimensions, output: output
        )
        let actual = output.contents().assumingMemoryBound(to: Float.self)
        let scale = 1 / sqrt(Float(dimensions))
        for segment in 0..<querySegments.segmentCount {
            let queryRange = querySegments.offsets[segment]..<querySegments.offsets[segment + 1]
            let keyRange = keySegments.offsets[segment]..<keySegments.offsets[segment + 1]
            for query in queryRange {
                for head in 0..<heads {
                    let queryBase = (query * heads + head) * dimensions
                    var scores: [Float] = []
                    for key in keyRange {
                        let keyBase = (key * heads + head) * dimensions
                        var score: Float = 0
                        for dimension in 0..<dimensions {
                            score.addProduct(
                                queries[queryBase + dimension], keys[keyBase + dimension]
                            )
                        }
                        scores.append(score * scale)
                    }
                    let maximum = scores.max()!
                    let weights = scores.map { exp($0 - maximum) }
                    let denominator = weights.reduce(0, +)
                    for dimension in 0..<dimensions {
                        var expected: Float = 0
                        for (localKey, key) in keyRange.enumerated() {
                            let keyBase = (key * heads + head) * dimensions
                            expected += weights[localKey] / denominator * values[keyBase + dimension]
                        }
                        #expect(abs(actual[queryBase + dimension] - expected) < 3e-6)
                    }
                }
            }
        }
        #expect(throws: NativeRuntimeError.self) {
            try AttentionSegments(offsets: [0, 2, 1])
        }
        let layoutWithEmptyPrefix = try AttentionSegments(offsets: [0, 0, 2])
        #expect(layoutWithEmptyPrefix.segmentCount == 2)
        let wrongKeySegments = try AttentionSegments(offsets: [0, 5])
        #expect(throws: NativeRuntimeError.self) {
            try kernel.segmentedF32(
                queries: queryBuffer, keys: keyBuffer, values: valueBuffer,
                querySegments: querySegments, keySegments: wrongKeySegments,
                heads: heads, dimensions: dimensions, output: output
            )
        }
        #expect(throws: NativeRuntimeError.self) {
            try kernel.segmentedF32(
                queries: queryBuffer, keys: keyBuffer, values: valueBuffer,
                querySegments: querySegments, keySegments: keySegments,
                heads: heads, dimensions: dimensions, output: keyBuffer
            )
        }
        #expect(throws: NativeRuntimeError.self) {
            try kernel.segmentedF32(
                queries: queryBuffer, keys: keyBuffer, values: valueBuffer,
                querySegments: querySegments, keySegments: keySegments,
                heads: heads, dimensions: dimensions, output: valueBuffer
            )
        }
    }

    @Test("native sparse Flow Euler and CFG match the pinned Torch oracle")
    func sparseFlowEuler() throws {
        let context = try MetalContext()
        let parameters = try FlowEulerParameters.shape512()
        let expectedSchedule = [
            1.0, 0.9705882352941178, 0.9374999999999999, 0.9,
            0.8571428571428571, 0.8076923076923076, 0.75,
            0.6818181818181819, 0.6, 0.5, 0.3750000000000001,
            0.21428571428571436, 0.0,
        ]
        let schedule = parameters.schedule()
        try #require(schedule.count == 12)
        for index in schedule.indices {
            #expect(abs(schedule[index].time - expectedSchedule[index]) < 2e-15)
            #expect(abs(schedule[index].previousTime - expectedSchedule[index + 1]) < 2e-15)
        }

        let layout = try AttentionSegments(offsets: [0, 2, 5])
        let channels = 4
        let noiseCount = layout.totalCount * channels
        var noise = (0..<noiseCount).map { index -> Float in
            let periodic = Double((index % 7) - 3) * 0.11
            let trend = Double(index) * 0.003
            return Float(periodic + trend)
        }
        let noiseBuffer = try #require(context.device.makeBuffer(
            bytes: &noise, length: noise.count * 4, options: .storageModeShared
        ))
        var calls: [(String, Float)] = []
        var previousTrace: [[Float]] = []
        var x0Trace: [[Float]] = []
        let result = try FlowEulerSampler(
            context: context, parameters: parameters
        ).sampleF32(
            noise: noiseBuffer, layout: layout, channels: channels,
            trace: { _, previous, x0 in
                let previousValues = previous.contents().assumingMemoryBound(to: Float.self)
                let x0Values = x0.contents().assumingMemoryBound(to: Float.self)
                previousTrace.append((0..<noise.count).map { previousValues[$0] })
                x0Trace.append((0..<noise.count).map { x0Values[$0] })
            },
            predictor: { state, timestep, pass in
                let name: String
                let bias: Float
                switch pass {
                case .positive:
                    name = "positive"
                    bias = 0.075
                case .negative:
                    name = "negative"
                    bias = -0.125
                }
                calls.append((name, timestep))
                let output = try #require(context.device.makeBuffer(
                    length: noise.count * 4, options: .storageModeShared
                ))
                let inputValues = state.contents().assumingMemoryBound(to: Float.self)
                let outputValues = output.contents().assumingMemoryBound(to: Float.self)
                for index in noise.indices {
                    let stateTerm = inputValues[index] * 0.125
                    let timestepTerm = timestep * 0.0001
                    let indexTerm = Float(index) * 0.002
                    outputValues[index] = stateTerm + timestepTerm + indexTerm + bias
                }
                return output
            }
        )
        #expect(result.modelCallCount == 21)
        #expect(calls.count == 21)
        for index in 0..<18 {
            #expect(calls[index].0 == (index.isMultiple(of: 2) ? "positive" : "negative"))
        }
        for index in 18..<21 { #expect(calls[index].0 == "positive") }

        let fixtureURL = try #require(Bundle.module.url(
            forResource: "flow-euler-sparse", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: fixtureURL) ==
                "4a104bffebb6e6164732f12d8f186dbd9fbd5b99889623c6f842703c7c9b3c47"
        )
        let golden = try Data(contentsOf: fixtureURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        try #require(golden.count == noise.count)
        let actual = result.samples.contents().assumingMemoryBound(to: Float.self)
        for index in golden.indices {
            #expect(abs(actual[index] - golden[index]) < 2e-5)
        }
        let traceURL = try #require(Bundle.module.url(
            forResource: "flow-euler-sparse", withExtension: "f32.trace",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: traceURL) ==
                "c39475758c9f4bdebe1de863d4a5a5346f5724f7c81e09e3d7adbe5a53eef00d"
        )
        let trace = try Data(contentsOf: traceURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        try #require(previousTrace.count == 12 && x0Trace.count == 12)
        try #require(trace.count == 24 * noise.count)
        for step in 0..<12 {
            for index in noise.indices {
                #expect(abs(previousTrace[step][index] - trace[step * noise.count + index]) < 2e-5)
                let x0Offset = (12 + step) * noise.count + index
                #expect(abs(x0Trace[step][index] - trace[x0Offset]) < 2e-5)
            }
        }

        let textureResult = try FlowEulerSampler(
            context: context, parameters: .texture512()
        ).sampleF32(
            noise: noiseBuffer, layout: layout, channels: channels,
            predictor: { state, _, pass in
                if case .negative = pass {
                    Issue.record("texture guidance strength 1 must not request negative conditioning")
                }
                return state
            }
        )
        #expect(textureResult.modelCallCount == 12)
    }

    @Test("texture-flow input preserves noise-first normalized-shape layout")
    func textureFlowInputLayout() throws {
        let context = try MetalContext()
        let tokens = 2
        var noise = (0..<(tokens * 32)).map { Float($0) * 0.01 - 0.2 }
        var shape = (0..<(tokens * 32)).map { index in
            let channel = index % 32
            return SLatPipelineMath.shapeMean[channel]
                + SLatPipelineMath.shapeStandardDeviation[channel]
                    * Float(index / 32 + channel) * 0.02
        }
        let noiseBuffer = try #require(context.device.makeBuffer(
            bytes: &noise, length: noise.count * 4, options: .storageModeShared
        ))
        let shapeBuffer = try #require(context.device.makeBuffer(
            bytes: &shape, length: shape.count * 4, options: .storageModeShared
        ))
        let math = SLatPipelineMath(context: context)
        let input = try math.makeTextureInputF32(
            noise: noiseBuffer, shape: shapeBuffer, tokens: tokens
        )
        let values = input.contents().assumingMemoryBound(to: Float.self)
        for token in 0..<tokens {
            for channel in 0..<32 {
                #expect(values[token * 64 + channel] == noise[token * 32 + channel])
                let expected = Float(token + channel) * 0.02
                #expect(abs(values[token * 64 + 32 + channel] - expected) < 2e-6)
            }
        }
        var normalizedShape = (0..<(tokens * 32)).map {
            Float($0 / 32 + $0 % 32) * 0.02
        }
        let normalizedShapeBuffer = try #require(context.device.makeBuffer(
            bytes: &normalizedShape, length: normalizedShape.count * 4,
            options: .storageModeShared
        ))
        let denormalizedShape = try math.denormalizeShapeF32(
            normalizedShapeBuffer, tokens: tokens
        )
        let shapeValues = denormalizedShape.contents().assumingMemoryBound(to: Float.self)
        for index in shape.indices {
            #expect(abs(shapeValues[index] - shape[index]) < 2e-6)
        }
    }

    @Test("Metal 3D RoPE matches the pinned TRELLIS coordinate formula")
    func rotaryPosition3D() throws {
        let context = try MetalContext()
        let kernel = try RotaryPositionKernel(context: context)
        let tokens = 2, heads = 2, dimensions = 128
        var values = (0..<(tokens * heads * dimensions)).map {
            Float(sin(Double($0) * 0.019) * 0.8)
        }
        var coordinates: [Int32] = [0, 0, 0, 0, 0, 1, 2, 3]
        let input = try #require(context.device.makeBuffer(
            bytes: &values, length: values.count * 4, options: .storageModeShared
        ))
        let coordinateBuffer = try #require(context.device.makeBuffer(
            bytes: &coordinates, length: coordinates.count * 4, options: .storageModeShared
        ))
        let output = try #require(context.device.makeBuffer(
            length: values.count * 4, options: .storageModeShared
        ))
        #expect(throws: NativeRuntimeError.self) {
            try kernel.apply3DF32(
                input: input, coordinates: coordinateBuffer, tokens: tokens,
                heads: heads, dimensions: dimensions, output: input
            )
        }
        try kernel.apply3DF32(
            input: input, coordinates: coordinateBuffer, tokens: tokens,
            heads: heads, dimensions: dimensions, output: output
        )
        let actual = output.contents().assumingMemoryBound(to: Float.self)
        let frequencyDimensions = dimensions / 2 / 3
        for token in 0..<tokens {
            for head in 0..<heads {
                let headBase = (token * heads + head) * dimensions
                for pair in 0..<(dimensions / 2) {
                    let real = values[headBase + pair * 2]
                    let imaginary = values[headBase + pair * 2 + 1]
                    let expectedReal: Float
                    let expectedImaginary: Float
                    if pair < frequencyDimensions * 3 {
                        let axis = pair / frequencyDimensions
                        let frequencyIndex = pair % frequencyDimensions
                        let frequency = 1 / pow(
                            10_000, Float(frequencyIndex) / Float(frequencyDimensions)
                        )
                        let angle = Float(coordinates[token * 4 + axis + 1]) * frequency
                        expectedReal = real * cos(angle) - imaginary * sin(angle)
                        expectedImaginary = real * sin(angle) + imaginary * cos(angle)
                    } else {
                        expectedReal = real
                        expectedImaginary = imaginary
                    }
                    #expect(abs(actual[headBase + pair * 2] - expectedReal) < 2e-6)
                    #expect(abs(actual[headBase + pair * 2 + 1] - expectedImaginary) < 2e-6)
                }
            }
        }
    }

    @Test(
        "complete real DINOv3 conditioning stage matches the pinned TRELLIS oracle",
        .enabled(
            if: ProcessInfo.processInfo.environment["KG_TRELLIS2_DINO_CHECKPOINT"] != nil,
            "Set KG_TRELLIS2_DINO_CHECKPOINT to execute real DINOv3 conformance"
        )
    )
    func realDINOv3StageGolden() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["KG_TRELLIS2_DINO_CHECKPOINT"]
        )
        let context = try MetalContext(arenaCapacity: 64 * 1024 * 1024)
        let checkpoint = try MappedCheckpoint(
            url: URL(fileURLWithPath: path), device: context.device
        )
        try #require(
            checkpoint.sha256() ==
                "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
        )
        let traceURL = try #require(Bundle.module.url(
            forResource: "dino-stage-tiny", withExtension: "f32.trace",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: traceURL) ==
                "bbeb6807e7bf276ee9828ae651e0d9e063ce008a7582fe34ddbbfce1df07c72b"
        )
        let oracle = try Data(contentsOf: traceURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        let imageElements = 3 * 32 * 32
        let stageElements = 9 * 1024
        try #require(oracle.count == imageElements + 26 * stageElements)
        var imageValues = Array(oracle[0..<imageElements])
        let image = try #require(context.device.makeBuffer(
            bytes: &imageValues, length: imageValues.count * 4,
            options: .storageModeShared
        ))
        var traces: [(String, [Float])] = []
        let result = try DINOv3Conditioner(context: context).encodeNormalizedImageF32(
            image: image, imageHeight: 32, imageWidth: 32,
            checkpoint: checkpoint,
            trace: { name, values in traces.append((name, values)) }
        )
        #expect(result.tokenCount == 9)
        #expect(result.hiddenSize == 1024)
        try #require(traces.count == 26)
        let expectedNames = ["embeddings"]
            + (0..<24).map { "block_\($0)" }
            + ["final_parameter_free_layer_norm"]
        for traceIndex in traces.indices {
            let (name, values) = traces[traceIndex]
            #expect(name == expectedNames[traceIndex])
            try #require(values.count == stageElements)
            let expectedStart = imageElements + traceIndex * stageElements
            var maximumError: Float = 0
            var squaredError: Double = 0
            var expectedSquaredMagnitude: Double = 0
            var mixedToleranceFailures = 0
            var maximumMixedRatio: Float = 0
            var maximumMixedExpected: Float = 0
            var maximumMixedError: Float = 0
            let absoluteTolerance: Float
            if traceIndex == 0 {
                absoluteTolerance = 1e-5
            } else if traceIndex == traces.count - 1 {
                absoluteTolerance = 1e-4
            } else {
                // Each residual block adds another sequence of F32 reductions.
                // Bound accumulated low-magnitude drift per executed block.
                absoluteTolerance = Float(traceIndex) * 2e-5
            }
            let relativeTolerance: Float = traceIndex == traces.count - 1
                ? 1e-4 : 8e-6
            for index in values.indices {
                try #require(values[index].isFinite)
                let expected = oracle[expectedStart + index]
                let error = abs(values[index] - expected)
                maximumError = max(maximumError, error)
                squaredError += Double(error * error)
                expectedSquaredMagnitude += Double(expected * expected)
                let allowed = absoluteTolerance + relativeTolerance * abs(expected)
                let mixedRatio = error / allowed
                if mixedRatio > maximumMixedRatio {
                    maximumMixedRatio = mixedRatio
                    maximumMixedExpected = expected
                    maximumMixedError = error
                }
                if error > allowed { mixedToleranceFailures += 1 }
            }
            let rms = sqrt(squaredError / Double(values.count))
            let expectedRMS = sqrt(expectedSquaredMagnitude / Double(values.count))
            let normalizedRMS = rms / max(expectedRMS, 1e-12)
            print(
                "DINO trace \(name): max=\(maximumError) rms=\(rms) " +
                "normalized_rms=\(normalizedRMS) mixed_ratio=\(maximumMixedRatio) " +
                "mixed_expected=\(maximumMixedExpected) mixed_error=\(maximumMixedError)"
            )
            // The mixed bound prevents large oracle outliers from diluting
            // ordinary-token errors while allowing F32 reduction drift to
            // scale with the magnitude of the value being compared.
            #expect(mixedToleranceFailures == 0)
            #expect(maximumMixedRatio <= 1)
        }
        let outputFixture = try #require(Bundle.module.url(
            forResource: "dino-stage-tiny", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: outputFixture) ==
                "fb924d7aa23c1340325f05de116ee8788dc2e5ca164997ab2acc089aad8c1462"
        )
        let expectedOutput = try Data(contentsOf: outputFixture).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        let actual = result.conditioning.contents().assumingMemoryBound(to: Float.self)
        try #require(expectedOutput.count == stageElements)
        for index in expectedOutput.indices {
            #expect(abs(actual[index] - expectedOutput[index]) <= 1e-4)
        }
        let memory = try #require(context.arena).snapshot()
        print(
            "DINO arena: peak=\(memory.peakUsedBytes) " +
            "live=\(memory.usedBytes) allocations=\(memory.allocationCount)"
        )
        #expect(memory.peakUsedBytes < memory.capacityBytes)
    }

    @Test(
        "production-token DINOv3 512 conditioning matches the pinned TRELLIS oracle",
        .enabled(
            if: ProcessInfo.processInfo.environment["KG_TRELLIS2_DINO_CHECKPOINT"] != nil,
            "Set KG_TRELLIS2_DINO_CHECKPOINT to execute production DINOv3 conformance"
        )
    )
    func realDINOv3Stage512Golden() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["KG_TRELLIS2_DINO_CHECKPOINT"]
        )
        let context = try MetalContext(arenaCapacity: 256 * 1024 * 1024)
        let checkpoint = try MappedCheckpoint(
            url: URL(fileURLWithPath: path), device: context.device
        )
        try #require(
            checkpoint.sha256() ==
                "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
        )
        let inputFixture = try #require(Bundle.module.url(
            forResource: "dino-stage-512", withExtension: "f32.trace",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: inputFixture) ==
                "b4b82db5b357a16c7b832876a8e0ac6a4f3adfe1e312235b944ffc90e995e4c6"
        )
        var imageValues = try Data(contentsOf: inputFixture).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        try #require(imageValues.count == 3 * 512 * 512)
        let image = try #require(context.device.makeBuffer(
            bytes: &imageValues, length: imageValues.count * 4,
            options: .storageModeShared
        ))
        let started = ContinuousClock.now
        let result = try DINOv3Conditioner(context: context).encodeNormalizedImageF32(
            image: image, imageHeight: 512, imageWidth: 512,
            checkpoint: checkpoint
        )
        let elapsed = started.duration(to: .now)
        #expect(result.tokenCount == 1029)
        #expect(result.hiddenSize == 1024)

        let outputFixture = try #require(Bundle.module.url(
            forResource: "dino-stage-512", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: outputFixture) ==
                "e701080d00baff4f526e2615812c8f0a9d24e13e964c1920a1fcdac90538c7ad"
        )
        let expected = try Data(contentsOf: outputFixture).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        try #require(expected.count == 1029 * 1024)
        let actual = result.conditioning.contents().assumingMemoryBound(to: Float.self)
        var maximumError: Float = 0
        var squaredError: Double = 0
        var expectedSquaredMagnitude: Double = 0
        for index in expected.indices {
            try #require(actual[index].isFinite)
            let error = abs(actual[index] - expected[index])
            maximumError = max(maximumError, error)
            squaredError += Double(error * error)
            expectedSquaredMagnitude += Double(expected[index] * expected[index])
        }
        let rms = sqrt(squaredError / Double(expected.count))
        let expectedRMS = sqrt(expectedSquaredMagnitude / Double(expected.count))
        let normalizedRMS = rms / expectedRMS
        let memory = try #require(context.arena).snapshot()
        print(
            "DINO 512: max=\(maximumError) rms=\(rms) " +
            "normalized_rms=\(normalizedRMS) elapsed=\(elapsed) " +
            "arena_peak=\(memory.peakUsedBytes) arena_live=\(memory.usedBytes)"
        )
        #expect(maximumError <= 1e-4)
        #expect(normalizedRMS <= 5e-6)
        #expect(memory.peakUsedBytes < memory.capacityBytes)
    }

    @Test(
        "native shape sampler drives the complete real TRELLIS.2 flow",
        .enabled(
            if: ProcessInfo.processInfo.environment["KG_TRELLIS2_SHAPE_FLOW_CHECKPOINT"] != nil,
            "Set KG_TRELLIS2_SHAPE_FLOW_CHECKPOINT to execute sampler integration"
        )
    )
    func realSLatShapeSamplerGolden() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["KG_TRELLIS2_SHAPE_FLOW_CHECKPOINT"]
        )
        let context = try MetalContext(arenaCapacity: 16 * 1024 * 1024)
        let checkpoint = try MappedCheckpoint(
            url: URL(fileURLWithPath: path), device: context.device
        )
        try #require(
            checkpoint.sha256() ==
                "ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f"
        )
        let tokens = 2
        let conditioningTokens = 2
        var noise = (0..<(tokens * 32)).map {
            Float(sin(Double($0) * 0.021) * 0.30)
        }
        var positiveConditioning = (0..<(conditioningTokens * 1024)).map {
            Float(sin(Double($0) * 0.015) * 0.25)
        }
        var negativeConditioning = [Float](
            repeating: 0, count: conditioningTokens * 1024
        )
        var coordinates: [Int32] = [0, 0, 0, 0, 0, 1, 2, 3]
        let noiseBuffer = try #require(context.device.makeBuffer(
            bytes: &noise, length: noise.count * 4, options: .storageModeShared
        ))
        let positiveBuffer = try #require(context.device.makeBuffer(
            bytes: &positiveConditioning, length: positiveConditioning.count * 4,
            options: .storageModeShared
        ))
        let negativeBuffer = try #require(context.device.makeBuffer(
            bytes: &negativeConditioning, length: negativeConditioning.count * 4,
            options: .storageModeShared
        ))
        let coordinateBuffer = try #require(context.device.makeBuffer(
            bytes: &coordinates, length: coordinates.count * 4,
            options: .storageModeShared
        ))
        var modelTrace: [[Float]] = []
        var samplerTrace: [[Float]] = []

        let result = try SLatFlowPipeline(context: context).sampleShapeF32(
            noise: noiseBuffer, coordinates: coordinateBuffer,
            positiveConditioning: positiveBuffer,
            negativeConditioning: negativeBuffer,
            checkpoint: checkpoint, tokens: tokens,
            conditioningTokens: conditioningTokens,
            parameters: .shape512(steps: 2),
            modelTrace: { call, _, output in
                #expect(call == modelTrace.count)
                let values = output.contents().assumingMemoryBound(to: Float.self)
                modelTrace.append((0..<noise.count).map { values[$0] })
            },
            samplerTrace: { step, state in
                #expect(step == samplerTrace.count)
                let values = state.contents().assumingMemoryBound(to: Float.self)
                samplerTrace.append((0..<noise.count).map { values[$0] })
            }
        )
        #expect(result.modelCallCount == 4)
        let memory = try #require(context.arena).snapshot()
        #expect(memory.peakUsedBytes < memory.capacityBytes)
        print(
            "shape sampler arena: peak=\(memory.peakUsedBytes) " +
            "live=\(memory.usedBytes) allocations=\(memory.allocationCount)"
        )

        let fixtureURL = try #require(Bundle.module.url(
            forResource: "slat-shape-sampler-2step", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: fixtureURL) ==
                "e8c2fe1c4f1cd549b7d6b406930f73204d6b1626df03a2e4c921928ee5c7288d"
        )
        let golden = try Data(contentsOf: fixtureURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        try #require(golden.count == noise.count)
        let actual = result.latent.contents().assumingMemoryBound(to: Float.self)
        var maximumAbsoluteError: Float = 0
        var squaredError: Double = 0
        var goldenSquaredMagnitude: Double = 0
        var goldenMaximumMagnitude: Float = 0
        for index in golden.indices {
            try #require(actual[index].isFinite)
            let error = abs(actual[index] - golden[index])
            maximumAbsoluteError = max(maximumAbsoluteError, error)
            squaredError += Double(error * error)
            goldenSquaredMagnitude += Double(golden[index] * golden[index])
            goldenMaximumMagnitude = max(goldenMaximumMagnitude, abs(golden[index]))
        }
        let rmsError = sqrt(squaredError / Double(golden.count))
        let goldenRMS = sqrt(goldenSquaredMagnitude / Double(golden.count))
        let traceURL = try #require(Bundle.module.url(
            forResource: "slat-shape-sampler-2step", withExtension: "f32.trace",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: traceURL) ==
                "049b3f9bd2df257975e48a269a15d720134193a21528974fd9c131f3ac808d7f"
        )
        let expectedTrace = try Data(contentsOf: traceURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        try #require(modelTrace.count == 4 && samplerTrace.count == 2)
        try #require(expectedTrace.count == 6 * noise.count)
        let maximumCaps: [Float] = [0.04, 0.02, 0.30, 0.25, 0.07, 0.55]
        let rmsCaps: [Double] = [0.016, 0.008, 0.11, 0.10, 0.03, 0.23]
        for traceIndex in 0..<6 {
            let values = traceIndex < 4 ? modelTrace[traceIndex] : samplerTrace[traceIndex - 4]
            var traceSquaredError: Double = 0
            var traceMaximumError: Float = 0
            for index in values.indices {
                let error = abs(values[index] - expectedTrace[traceIndex * noise.count + index])
                traceMaximumError = max(traceMaximumError, error)
                traceSquaredError += Double(error * error)
            }
            let traceRMS = sqrt(traceSquaredError / Double(values.count))
            print("shape sampler trace \(traceIndex): max=\(traceMaximumError) rms=\(traceRMS)")
            #expect(traceMaximumError <= maximumCaps[traceIndex])
            #expect(traceRMS <= rmsCaps[traceIndex])
        }
        let maximumScaleRatio = maximumAbsoluteError / goldenMaximumMagnitude
        let normalizedRMS = rmsError / goldenRMS
        print(
            "two-step shape sampler final: max=\(maximumAbsoluteError) rms=\(rmsError) " +
            "normalized_rms=\(normalizedRMS) max_scale_ratio=\(maximumScaleRatio)"
        )
        // The first CFG pair is the strict graph-parity gate; later caps bound
        // deterministic trajectory drift after feeding Metal results back in.
        #expect(maximumScaleRatio <= 0.22)
        #expect(normalizedRMS <= 0.22)
    }

    @Test(
        "native texture sampler drives the complete real TRELLIS.2 flow",
        .enabled(
            if: ProcessInfo.processInfo.environment["KG_TRELLIS2_TEXTURE_FLOW_CHECKPOINT"] != nil,
            "Set KG_TRELLIS2_TEXTURE_FLOW_CHECKPOINT to execute sampler integration"
        )
    )
    func realSLatTextureSamplerGolden() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["KG_TRELLIS2_TEXTURE_FLOW_CHECKPOINT"]
        )
        let context = try MetalContext(arenaCapacity: 16 * 1024 * 1024)
        let checkpoint = try MappedCheckpoint(
            url: URL(fileURLWithPath: path), device: context.device
        )
        try #require(
            checkpoint.sha256() ==
                "8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d"
        )
        let tokens = 2
        let conditioningTokens = 2
        var noise = (0..<(tokens * 32)).map {
            Float(sin(Double($0) * 0.021) * 0.30)
        }
        var shape = (0..<(tokens * 32)).map { index in
            let channel = index % 32
            let normalized = Float(cos(Double(index) * 0.017) * 0.20)
            return normalized * SLatPipelineMath.shapeStandardDeviation[channel]
                + SLatPipelineMath.shapeMean[channel]
        }
        var conditioning = (0..<(conditioningTokens * 1024)).map {
            Float(sin(Double($0) * 0.015) * 0.25)
        }
        var coordinates: [Int32] = [0, 0, 0, 0, 0, 1, 2, 3]
        let noiseBuffer = try #require(context.device.makeBuffer(
            bytes: &noise, length: noise.count * 4, options: .storageModeShared
        ))
        let shapeBuffer = try #require(context.device.makeBuffer(
            bytes: &shape, length: shape.count * 4, options: .storageModeShared
        ))
        let conditioningBuffer = try #require(context.device.makeBuffer(
            bytes: &conditioning, length: conditioning.count * 4,
            options: .storageModeShared
        ))
        let coordinateBuffer = try #require(context.device.makeBuffer(
            bytes: &coordinates, length: coordinates.count * 4,
            options: .storageModeShared
        ))
        var modelTrace: [[Float]] = []
        var samplerTrace: [[Float]] = []
        let pipeline = try SLatFlowPipeline(context: context)
        let result = try pipeline.sampleTextureF32(
            noise: noiseBuffer, shapeLatent: shapeBuffer,
            coordinates: coordinateBuffer, positiveConditioning: conditioningBuffer,
            checkpoint: checkpoint, tokens: tokens,
            conditioningTokens: conditioningTokens,
            parameters: .texture512(steps: 2),
            modelTrace: { call, output in
                #expect(call == modelTrace.count)
                let values = output.contents().assumingMemoryBound(to: Float.self)
                modelTrace.append((0..<noise.count).map { values[$0] })
            },
            samplerTrace: { step, state in
                #expect(step == samplerTrace.count)
                let values = state.contents().assumingMemoryBound(to: Float.self)
                samplerTrace.append((0..<noise.count).map { values[$0] })
            }
        )
        #expect(result.modelCallCount == 2)
        let memory = try #require(context.arena).snapshot()
        #expect(memory.peakUsedBytes < memory.capacityBytes)
        print(
            "texture sampler arena: peak=\(memory.peakUsedBytes) " +
            "live=\(memory.usedBytes) allocations=\(memory.allocationCount)"
        )

        let fixtureURL = try #require(Bundle.module.url(
            forResource: "slat-texture-sampler-2step", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: fixtureURL) ==
                "640bacbb2b3c8ba00498cd5e89eb2d17aee62259128b92d01866622ef0329fba"
        )
        let golden = try Data(contentsOf: fixtureURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        try #require(golden.count == noise.count)
        let actual = result.latent.contents().assumingMemoryBound(to: Float.self)
        var maximumAbsoluteError: Float = 0
        var squaredError: Double = 0
        var goldenSquaredMagnitude: Double = 0
        var goldenMaximumMagnitude: Float = 0
        for index in golden.indices {
            try #require(actual[index].isFinite)
            let error = abs(actual[index] - golden[index])
            maximumAbsoluteError = max(maximumAbsoluteError, error)
            squaredError += Double(error * error)
            goldenSquaredMagnitude += Double(golden[index] * golden[index])
            goldenMaximumMagnitude = max(goldenMaximumMagnitude, abs(golden[index]))
        }
        let rmsError = sqrt(squaredError / Double(golden.count))
        let goldenRMS = sqrt(goldenSquaredMagnitude / Double(golden.count))

        let traceURL = try #require(Bundle.module.url(
            forResource: "slat-texture-sampler-2step", withExtension: "f32.trace",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: traceURL) ==
                "2bca46d5a7d66a9cde0b68a4d88e3b800856fc1e74b55c401b47ca2eb15bc8b5"
        )
        let expectedTrace = try Data(contentsOf: traceURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        try #require(modelTrace.count == 2 && samplerTrace.count == 2)
        try #require(expectedTrace.count == 4 * noise.count)
        let maximumCaps: [Float] = [0.005, 0.035, 0.002, 0.025]
        let rmsCaps: [Double] = [0.002, 0.015, 0.001, 0.012]
        for traceIndex in 0..<4 {
            let values = traceIndex < 2 ? modelTrace[traceIndex] : samplerTrace[traceIndex - 2]
            var traceSquaredError: Double = 0
            var traceMaximumError: Float = 0
            for index in values.indices {
                let error = abs(values[index] - expectedTrace[traceIndex * noise.count + index])
                traceMaximumError = max(traceMaximumError, error)
                traceSquaredError += Double(error * error)
            }
            let traceRMS = sqrt(traceSquaredError / Double(values.count))
            print("texture sampler trace \(traceIndex): max=\(traceMaximumError) rms=\(traceRMS)")
            #expect(traceMaximumError <= maximumCaps[traceIndex])
            #expect(traceRMS <= rmsCaps[traceIndex])
        }
        let maximumScaleRatio = maximumAbsoluteError / goldenMaximumMagnitude
        let normalizedRMS = rmsError / goldenRMS
        print(
            "two-step texture sampler final: max=\(maximumAbsoluteError) rms=\(rmsError) " +
            "normalized_rms=\(normalizedRMS) max_scale_ratio=\(maximumScaleRatio)"
        )
        #expect(maximumScaleRatio <= 0.01)
        #expect(normalizedRMS <= 0.01)
    }

    @Test(
        "complete real TRELLIS.2 texture flow matches the pinned Torch oracle",
        .enabled(
            if: ProcessInfo.processInfo.environment["KG_TRELLIS2_TEXTURE_FLOW_CHECKPOINT"] != nil,
            "Set KG_TRELLIS2_TEXTURE_FLOW_CHECKPOINT to execute real-weight conformance"
        )
    )
    func realSLatTextureFlowGolden() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["KG_TRELLIS2_TEXTURE_FLOW_CHECKPOINT"]
        )
        let context = try MetalContext()
        let checkpoint = try MappedCheckpoint(
            url: URL(fileURLWithPath: path), device: context.device
        )
        try #require(
            checkpoint.sha256() ==
                "8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d"
        )
        let tokens = 2
        var features = [Float](repeating: 0, count: tokens * 64)
        for token in 0..<tokens {
            for channel in 0..<32 {
                let index = token * 32 + channel
                features[token * 64 + channel] = Float(sin(Double(index) * 0.021) * 0.30)
                features[token * 64 + 32 + channel] = Float(cos(Double(index) * 0.017) * 0.20)
            }
        }
        var timestep: [Float] = [650.25]
        var conditioning = (0..<(2 * 1024)).map {
            Float(sin(Double($0) * 0.015) * 0.25)
        }
        var coordinates: [Int32] = [0, 0, 0, 0, 0, 1, 2, 3]
        let inputBuffer = try #require(context.device.makeBuffer(
            bytes: &features, length: features.count * 4, options: .storageModeShared
        ))
        let timestepBuffer = try #require(context.device.makeBuffer(
            bytes: &timestep, length: 4, options: .storageModeShared
        ))
        let conditioningBuffer = try #require(context.device.makeBuffer(
            bytes: &conditioning, length: conditioning.count * 4, options: .storageModeShared
        ))
        let coordinateBuffer = try #require(context.device.makeBuffer(
            bytes: &coordinates, length: coordinates.count * 4, options: .storageModeShared
        ))
        var blockOutputs: [MTLBuffer] = []
        let output = try SLatFlow(
            context: context, configuration: .texture
        ).forwardF32(
            input: inputBuffer, timestep: timestepBuffer,
            conditioning: conditioningBuffer, coordinates: coordinateBuffer,
            checkpoint: checkpoint, tokens: tokens, conditioningTokens: 2,
            trace: { _, buffer in blockOutputs.append(buffer) }
        )
        let fixtureURL = try #require(Bundle.module.url(
            forResource: "slat-texture-flow-tiny", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: fixtureURL) ==
                "93ecb91b95fa1c80eccf2783c034419219495e895ac0223f75776bb4ed7e7661"
        )
        let golden = try Data(contentsOf: fixtureURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        try #require(golden.count == tokens * 32)
        let actual = output.contents().assumingMemoryBound(to: Float.self)
        var maximumAbsoluteError: Float = 0
        var squaredError: Double = 0
        for index in golden.indices {
            try #require(actual[index].isFinite)
            let error = abs(actual[index] - golden[index])
            maximumAbsoluteError = max(maximumAbsoluteError, error)
            squaredError += Double(error * error)
        }
        let rmsError = sqrt(squaredError / Double(golden.count))

        let traceURL = try #require(Bundle.module.url(
            forResource: "slat-texture-flow-tiny", withExtension: "f32.trace",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: traceURL) ==
                "243bf35b8fc49a3a393a4dd6f311a654f54782f6610f9623a0375dd8a9a70ead"
        )
        let trace = try Data(contentsOf: traceURL).withUnsafeBytes {
            Array($0.bindMemory(to: UInt16.self))
        }
        let elementsPerBlock = tokens * 1536
        try #require(blockOutputs.count == SLatFlow.blockCount)
        try #require(trace.count == blockOutputs.count * elementsPerBlock)
        var worstBlockMaximum: Float = 0
        var worstBlockRMS: Double = 0
        var worstBlockNormalizedRMS: Double = 0
        var worstBlockMaximumScaleRatio: Float = 0
        for blockIndex in blockOutputs.indices {
            let values = blockOutputs[blockIndex].contents().assumingMemoryBound(to: Float.self)
            var blockSquaredError: Double = 0
            var expectedSquaredMagnitude: Double = 0
            var blockMaximum: Float = 0
            var expectedMaximum: Float = 0
            for index in 0..<elementsPerBlock {
                try #require(values[index].isFinite)
                let expected = fromBF16(UInt16(
                    littleEndian: trace[blockIndex * elementsPerBlock + index]
                ))
                let error = abs(values[index] - expected)
                blockMaximum = max(blockMaximum, error)
                expectedMaximum = max(expectedMaximum, abs(expected))
                blockSquaredError += Double(error * error)
                expectedSquaredMagnitude += Double(expected * expected)
            }
            try #require(expectedMaximum > 0)
            let blockRMS = sqrt(blockSquaredError / Double(elementsPerBlock))
            let expectedRMS = sqrt(expectedSquaredMagnitude / Double(elementsPerBlock))
            worstBlockMaximum = max(worstBlockMaximum, blockMaximum)
            worstBlockRMS = max(worstBlockRMS, blockRMS)
            worstBlockNormalizedRMS = max(
                worstBlockNormalizedRMS, blockRMS / max(expectedRMS, 1e-12)
            )
            worstBlockMaximumScaleRatio = max(
                worstBlockMaximumScaleRatio, blockMaximum / expectedMaximum
            )
        }
        print(
            "30-block texture flow: max=\(maximumAbsoluteError) rms=\(rmsError) " +
            "block_max=\(worstBlockMaximum) block_rms=\(worstBlockRMS) " +
            "block_normalized_rms=\(worstBlockNormalizedRMS) " +
            "block_max_scale_ratio=\(worstBlockMaximumScaleRatio)"
        )
        #expect(maximumAbsoluteError <= 0.025)
        #expect(rmsError <= 0.01)
        #expect(worstBlockMaximum <= 64)
        #expect(worstBlockRMS <= 1)
        #expect(worstBlockNormalizedRMS <= 0.02)
        #expect(worstBlockMaximumScaleRatio <= 0.04)
    }

    @Test(
        "complete real TRELLIS.2 shape flow matches the pinned Torch oracle",
        .enabled(
            if: ProcessInfo.processInfo.environment["KG_TRELLIS2_SHAPE_FLOW_CHECKPOINT"] != nil,
            "Set KG_TRELLIS2_SHAPE_FLOW_CHECKPOINT to execute real-weight conformance"
        )
    )
    func realSLatShapeFlowGolden() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["KG_TRELLIS2_SHAPE_FLOW_CHECKPOINT"]
        )
        let context = try MetalContext()
        let checkpoint = try MappedCheckpoint(
            url: URL(fileURLWithPath: path), device: context.device
        )
        try #require(
            checkpoint.sha256() ==
                "ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f"
        )
        let tokens = 2
        var features = (0..<(tokens * 32)).map {
            Float(sin(Double($0) * 0.021) * 0.30)
        }
        var timestep: [Float] = [650.25]
        var conditioning = (0..<(2 * 1024)).map {
            Float(sin(Double($0) * 0.015) * 0.25)
        }
        var coordinates: [Int32] = [0, 0, 0, 0, 0, 1, 2, 3]
        let inputBuffer = try #require(context.device.makeBuffer(
            bytes: &features, length: features.count * 4, options: .storageModeShared
        ))
        let timestepBuffer = try #require(context.device.makeBuffer(
            bytes: &timestep, length: timestep.count * 4, options: .storageModeShared
        ))
        let conditioningBuffer = try #require(context.device.makeBuffer(
            bytes: &conditioning, length: conditioning.count * 4, options: .storageModeShared
        ))
        let coordinateBuffer = try #require(context.device.makeBuffer(
            bytes: &coordinates, length: coordinates.count * 4, options: .storageModeShared
        ))
        var mixedBatchCoordinates: [Int32] = [0, 0, 0, 0, 1, 1, 2, 3]
        let mixedBatchBuffer = try #require(context.device.makeBuffer(
            bytes: &mixedBatchCoordinates, length: mixedBatchCoordinates.count * 4,
            options: .storageModeShared
        ))
        let flow = try SLatShapeFlow(context: context)
        #expect(throws: NativeRuntimeError.self) {
            try flow.forwardF32(
                input: inputBuffer, timestep: timestepBuffer,
                conditioning: conditioningBuffer, coordinates: mixedBatchBuffer,
                checkpoint: checkpoint, tokens: tokens, conditioningTokens: 2
            )
        }
        var blockOutputs: [MTLBuffer] = []
        let output = try flow.forwardF32(
            input: inputBuffer, timestep: timestepBuffer,
            conditioning: conditioningBuffer, coordinates: coordinateBuffer,
            checkpoint: checkpoint, tokens: tokens, conditioningTokens: 2,
            trace: { index, buffer in
                #expect(index == blockOutputs.count)
                blockOutputs.append(buffer)
            }
        )

        let fixtureURL = try #require(Bundle.module.url(
            forResource: "slat-shape-flow-tiny", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: fixtureURL) ==
                "7f519fae4186b4bdd043ca91eea50eff3600246733bd433a378ad36a9d3659ce"
        )
        let golden = try Data(contentsOf: fixtureURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        try #require(golden.count == tokens * 32)
        let actual = output.contents().assumingMemoryBound(to: Float.self)
        var maximumAbsoluteError: Float = 0
        var squaredError: Double = 0
        for index in golden.indices {
            try #require(actual[index].isFinite)
            let error = abs(actual[index] - golden[index])
            maximumAbsoluteError = max(maximumAbsoluteError, error)
            squaredError += Double(error * error)
        }
        let rmsError = sqrt(squaredError / Double(golden.count))

        let traceURL = try #require(Bundle.module.url(
            forResource: "slat-shape-flow-tiny", withExtension: "f32.trace",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: traceURL) ==
                "55c77fa8a47e2d31fcff75dfa5968cb52d7ecfb830c0d520136aeee0f26e3d94"
        )
        let trace = try Data(contentsOf: traceURL).withUnsafeBytes {
            Array($0.bindMemory(to: UInt16.self))
        }
        let elementsPerBlock = tokens * 1536
        try #require(blockOutputs.count == SLatShapeFlow.blockCount)
        try #require(trace.count == blockOutputs.count * elementsPerBlock)
        var worstBlockRMS: Double = 0
        var worstBlockNormalizedRMS: Double = 0
        var worstBlockMaximumScaleRatio: Float = 0
        var worstBlockMaximum: Float = 0
        for blockIndex in blockOutputs.indices {
            let values = blockOutputs[blockIndex].contents().assumingMemoryBound(to: Float.self)
            var blockSquaredError: Double = 0
            var expectedSquaredMagnitude: Double = 0
            var expectedMaximum: Float = 0
            var blockMaximum: Float = 0
            for index in 0..<elementsPerBlock {
                try #require(values[index].isFinite)
                let expected = fromBF16(UInt16(littleEndian: trace[blockIndex * elementsPerBlock + index]))
                let error = abs(values[index] - expected)
                blockMaximum = max(blockMaximum, error)
                expectedMaximum = max(expectedMaximum, abs(expected))
                blockSquaredError += Double(error * error)
                expectedSquaredMagnitude += Double(expected * expected)
            }
            worstBlockMaximum = max(worstBlockMaximum, blockMaximum)
            try #require(expectedMaximum > 0)
            worstBlockMaximumScaleRatio = max(
                worstBlockMaximumScaleRatio, blockMaximum / expectedMaximum
            )
            let blockRMS = sqrt(blockSquaredError / Double(elementsPerBlock))
            let expectedRMS = sqrt(expectedSquaredMagnitude / Double(elementsPerBlock))
            worstBlockRMS = max(worstBlockRMS, blockRMS)
            worstBlockNormalizedRMS = max(
                worstBlockNormalizedRMS, blockRMS / max(expectedRMS, 1e-12)
            )
        }
        print(
            "30-block shape flow: max=\(maximumAbsoluteError) rms=\(rmsError) " +
            "block_max=\(worstBlockMaximum) block_rms=\(worstBlockRMS) " +
            "block_normalized_rms=\(worstBlockNormalizedRMS) " +
            "block_max_scale_ratio=\(worstBlockMaximumScaleRatio)"
        )
        #expect(maximumAbsoluteError <= 0.025)
        #expect(rmsError <= 0.01)
        #expect(worstBlockMaximum <= 64)
        #expect(worstBlockRMS <= 1)
        #expect(worstBlockNormalizedRMS <= 0.015)
        #expect(worstBlockMaximumScaleRatio <= 0.03)
    }

    @Test(
        "real TRELLIS.2 RoPE block matches the pinned Torch BF16 oracle",
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
        var coordinateValues: [Int32] = [0, 0, 0, 0, 0, 1, 2, 3]
        let coordinateBuffer = try #require(context.device.makeBuffer(
            bytes: &coordinateValues, length: coordinateValues.count * 4,
            options: .storageModeShared
        ))
        var tracedBuffers: [String: MTLBuffer] = [:]
        let actualBuffer = try SLatBlock(context: context).forwardF32(
            input: featureBuffer, sharedModulation: modulationBuffer,
            conditioning: contextBuffer, checkpoint: checkpoint,
            block: 0, tokens: tokens, conditioningTokens: 2,
            coordinates: coordinateBuffer, trace: { name, buffer in
            tracedBuffers[name] = buffer
        })
        let fixtureURL = try #require(Bundle.module.url(
            forResource: "slat-block0-tiny", withExtension: "bf16", subdirectory: "Fixtures"
        ))
        try #require(fileSHA256(at: fixtureURL) == "8cc519f7df166b5722f822ce231f1396a8f857d299074804677cba18547bb80d")
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
        try #require(fileSHA256(at: traceURL) == "ab8c01605829d930f73d79aaf16d5f524aec788e701d7809b5c86f052194c17b")
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
