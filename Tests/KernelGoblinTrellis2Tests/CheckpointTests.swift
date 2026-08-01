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
    }
}

private func bf16(_ value: Float) -> UInt16 {
    UInt16(truncatingIfNeeded: value.bitPattern >> 16)
}
