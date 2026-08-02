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
        let mappedBuffer = try checkpoint.acquireBuffer()
        let pointer = mappedBuffer.contents().advanced(by: range.lowerBound)
            .assumingMemoryBound(to: Float.self)
        #expect(pointer[0] == 1.25)
        #expect(pointer[1] == -2.5)
        withExtendedLifetime(mappedBuffer) {}
        #expect(try checkpoint.sha256() == fileSHA256(at: url))
        values.removeAll()
    }

    @Test("stage session drains Metal and releases arena before checkpoint")
    func stageSessionLifecycle() throws {
        let url = try makeStageTestCheckpoint(weight: 2)
        defer { try? FileManager.default.removeItem(at: url) }
        var events: [StageLifecycleEvent] = []
        let session = try StageSession(
            checkpointURL: url,
            arenaCapacity: 1024 * 1024,
            lifecycleObserver: { events.append($0) }
        )
        var inputValue: Float = 3
        let input = try #require(session.device.makeBuffer(
            bytes: &inputValue,
            length: MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let standalone = try session.withRuntimeForTesting { context, checkpoint in
            let output = try context.makeBuffer(
                length: MemoryLayout<Float>.stride,
                label: "Stage lifecycle dense output"
            )
            let weight = try checkpoint.descriptor(named: "weight")
            let dense = try DenseKernel(context: context)
            for _ in 0..<8 {
                try dense.linearF32(
                    input: input,
                    checkpoint: try checkpoint.acquireBuffer(),
                    weightOffset: Int(weight.fileOffset),
                    rows: 1,
                    inputChannels: 1,
                    outputChannels: 1,
                    output: output
                )
            }
            #expect(output.contents().assumingMemoryBound(to: Float.self)[0] == 6)
            return try session.standaloneCopyForTesting(
                output,
                byteCount: MemoryLayout<Float>.stride
            )
        }
        let snapshot = try session.close()
        #expect(snapshot.usedBytes == 0)
        #expect(events == [.queueDrained, .arenaReleased, .checkpointUnmapped])
        #expect(standalone.contents().assumingMemoryBound(to: Float.self)[0] == 6)
        #expect(try session.close() == snapshot)
        #expect(events == [.queueDrained, .arenaReleased, .checkpointUnmapped])
    }

    @Test("stage session cleans up when its body throws")
    func stageSessionThrowCleanup() throws {
        let url = try makeStageTestCheckpoint(weight: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        var events: [StageLifecycleEvent] = []
        do {
            _ = try StageSession.withSession(
                checkpointURL: url,
                arenaCapacity: 1024 * 1024,
                lifecycleObserver: { events.append($0) }
            ) { _ -> Int in
                throw StageTestFailure.expected
            }
            Issue.record("expected the stage body to throw")
        } catch StageTestFailure.expected {
            // The original body error is preserved when cleanup succeeds.
        }
        #expect(events == [.queueDrained, .arenaReleased, .checkpointUnmapped])
    }

    @Test("stage session unmaps its checkpoint when digest verification fails")
    func stageSessionDigestFailureCleanup() throws {
        let url = try makeStageTestCheckpoint(weight: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        var events: [StageLifecycleEvent] = []
        #expect(
            throws: NativeRuntimeError.invalidArgument(
                "stage checkpoint SHA-256 mismatch"
            )
        ) {
            _ = try StageSession(
                checkpointURL: url,
                expectedCheckpointSHA256: String(repeating: "0", count: 64),
                arenaCapacity: 1024 * 1024,
                lifecycleObserver: { events.append($0) }
            )
        }
        #expect(events == [.checkpointUnmapped])
    }

    @Test("stage session refuses teardown while an arena buffer escapes")
    func stageSessionRejectsEscapedArenaBuffer() throws {
        let url = try makeStageTestCheckpoint(weight: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        var events: [StageLifecycleEvent] = []
        let session = try StageSession(
            checkpointURL: url,
            arenaCapacity: 1024 * 1024,
            lifecycleObserver: { events.append($0) }
        )
        var escaped: MTLBuffer? = try session.withRuntimeForTesting { context, _ in
            try context.makeBuffer(length: 4096, label: "Deliberately escaped arena buffer")
        }
        #expect(escaped != nil)
        #expect(throws: StageSessionError.arenaStillLive(4096)) {
            try session.close()
        }
        #expect(session.checkpointIsMappedForTesting)
        #expect(events == [.queueDrained])
        escaped = nil
        let snapshot = try session.close()
        #expect(snapshot.usedBytes == 0)
        #expect(events == [
            .queueDrained, .queueDrained, .arenaReleased, .checkpointUnmapped,
        ])
    }

    @Test("stage session retries checkpoint release without repeating arena teardown")
    func stageSessionRetriesEscapedCheckpointBuffer() throws {
        let url = try makeStageTestCheckpoint(weight: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        var events: [StageLifecycleEvent] = []
        let session = try StageSession(
            checkpointURL: url,
            arenaCapacity: 1024 * 1024,
            lifecycleObserver: { events.append($0) }
        )
        var escaped = session.checkpointBufferForTesting
        #expect(escaped != nil)
        #expect(throws: MappedCheckpointError.mappingStillReferenced) {
            try session.close()
        }
        #expect(events == [.queueDrained, .arenaReleased])
        escaped = nil
        let snapshot = try session.close()
        #expect(snapshot.usedBytes == 0)
        #expect(events == [.queueDrained, .arenaReleased, .checkpointUnmapped])
    }

    @Test("mapped checkpoint close detects an escaped Metal buffer")
    func mappedCheckpointRejectsEscapedBuffer() throws {
        let url = try makeStageTestCheckpoint(weight: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let checkpoint = try MappedCheckpoint(url: url, device: device)
        var escaped: MTLBuffer? = try checkpoint.acquireBuffer()
        #expect(escaped != nil)
        #expect(throws: MappedCheckpointError.mappingStillReferenced) {
            try checkpoint.close(releaseTimeout: 0)
        }
        #expect(checkpoint.isMapped)
        escaped = nil
        try checkpoint.close()
        #expect(!checkpoint.isMapped)
        #expect(throws: MappedCheckpointError.closed) {
            try checkpoint.acquireBuffer()
        }
    }

    @Test("stage session retries after an externally retained empty arena")
    func stageSessionRetriesRetainedArena() throws {
        let url = try makeStageTestCheckpoint(weight: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        var events: [StageLifecycleEvent] = []
        let session = try StageSession(
            checkpointURL: url,
            arenaCapacity: 1024 * 1024,
            lifecycleObserver: { events.append($0) }
        )
        var escapedContext: MetalContext? = try session.withRuntimeForTesting {
            context, _ in context
        }
        #expect(escapedContext != nil)
        #expect(throws: StageSessionError.arenaNotReleased) {
            try session.close()
        }
        #expect(events == [.queueDrained])
        escapedContext = nil
        let snapshot = try session.close()
        #expect(snapshot.usedBytes == 0)
        #expect(events == [.queueDrained, .arenaReleased, .checkpointUnmapped])
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

    @Test("sparse occupancy uses strict threshold, z-fast coordinates, and 2x max pooling")
    func sparseOccupancyContract() throws {
        var logits = [Float](repeating: -.infinity, count: 64 * 64 * 64)
        func index(_ x: Int, _ y: Int, _ z: Int) -> Int {
            (x * 64 + y) * 64 + z
        }
        logits[index(0, 0, 0)] = 1
        logits[index(1, 1, 1)] = .infinity
        logits[index(1, 1, 2)] = .nan
        logits[index(2, 3, 4)] = 0
        logits[index(2, 3, 5)] = -0.0
        logits[index(63, 63, 63)] = 0.25
        let grid = try SparseStructureOccupancy.threshold(
            logits: logits, resolution: 64
        )
        #expect(grid.contains(x: 0, y: 0, z: 0))
        #expect(grid.contains(x: 1, y: 1, z: 1))
        #expect(!grid.contains(x: 1, y: 1, z: 2))
        #expect(!grid.contains(x: 2, y: 3, z: 4))
        #expect(!grid.contains(x: 2, y: 3, z: 5))
        #expect(grid.contains(x: 63, y: 63, z: 63))
        #expect(grid.coordinates() == [
            SparseStructureCoordinate(x: 0, y: 0, z: 0),
            SparseStructureCoordinate(x: 1, y: 1, z: 1),
            SparseStructureCoordinate(x: 63, y: 63, z: 63),
        ])

        let pooled = try SparseStructureOccupancy.downsampleMax2(grid)
        #expect(pooled.resolution == 32)
        #expect(pooled.coordinates() == [
            SparseStructureCoordinate(x: 0, y: 0, z: 0),
            SparseStructureCoordinate(x: 31, y: 31, z: 31),
        ])
        #expect(try SparseStructureOccupancy.threshold(
            logits: [Float](repeating: -1, count: 8), resolution: 2
        ).coordinates().isEmpty)
        #expect(try SparseStructureOccupancy.threshold(
            logits: [Float](repeating: 1, count: 8), resolution: 2
        ).coordinates().count == 8)
        #expect(throws: NativeRuntimeError.self) {
            try SparseStructureOccupancy.threshold(logits: [], resolution: 64)
        }
        #expect(throws: NativeRuntimeError.self) {
            try SparseOccupancyGrid(resolution: Int.max, packedBits: [])
        }
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

    @Test("Metal BF16 SIMD-group projection matches tiled and CPU tails")
    func bf16DenseProjectionSIMDGroupTails() throws {
        let context = try MetalContext()
        let kernel = try DenseKernel(context: context)
        guard kernel.supportsSIMDGroupMatrix else { return }
        var state: UInt32 = 0x5eed1234
        func randomFloat() -> Float {
            state = state &* 1_664_525 &+ 1_013_904_223
            return Float(Int32(bitPattern: state)) / Float(Int32.max) * 0.25
        }
        let shapes = [
            (rows: 7, outputs: 7, inputs: 7),
            (rows: 8, outputs: 8, inputs: 8),
            (rows: 9, outputs: 9, inputs: 9),
            (rows: 15, outputs: 17, inputs: 31),
        ]
        for (shapeIndex, shape) in shapes.enumerated() {
            let prefixElements = 5
            let hasBias = shapeIndex.isMultiple(of: 2)
            var inputValues = (0..<(shape.rows * shape.inputs)).map { _ in randomFloat() }
            var checkpointValues = [UInt16](repeating: 0x7fc1, count: prefixElements)
            checkpointValues += (0..<(shape.outputs * shape.inputs)).map { _ in
                bf16(randomFloat())
            }
            let biasElementOffset = checkpointValues.count
            checkpointValues += (0..<shape.outputs).map { _ in bf16(randomFloat()) }
            let input = try #require(context.device.makeBuffer(
                bytes: &inputValues,
                length: inputValues.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ))
            let checkpoint = try #require(context.device.makeBuffer(
                bytes: &checkpointValues,
                length: checkpointValues.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared
            ))
            let outputElements = shape.rows * shape.outputs
            let suffixElements = 17
            let outputBytes = (outputElements + suffixElements) * MemoryLayout<Float>.stride
            let tiled = try #require(context.device.makeBuffer(
                length: outputBytes, options: .storageModeShared
            ))
            let simdgroup = try #require(context.device.makeBuffer(
                length: outputBytes, options: .storageModeShared
            ))
            let mpsGraph = try #require(context.device.makeBuffer(
                length: outputBytes, options: .storageModeShared
            ))
            let canary: Float = -9876.5
            for buffer in [tiled, simdgroup, mpsGraph] {
                buffer.contents().assumingMemoryBound(to: Float.self)
                    .initialize(repeating: canary, count: outputElements + suffixElements)
            }
            let weightOffset = prefixElements * MemoryLayout<UInt16>.stride
            let biasOffset = hasBias
                ? biasElementOffset * MemoryLayout<UInt16>.stride : nil
            try kernel.linearBF16WeightsF32Output(
                input: input, checkpoint: checkpoint, weightOffset: weightOffset,
                biasOffset: biasOffset, rows: shape.rows, inputChannels: shape.inputs,
                outputChannels: shape.outputs, output: tiled, implementation: .tiled
            )
            try kernel.linearBF16WeightsF32Output(
                input: input, checkpoint: checkpoint, weightOffset: weightOffset,
                biasOffset: biasOffset, rows: shape.rows, inputChannels: shape.inputs,
                outputChannels: shape.outputs, output: simdgroup,
                implementation: .simdgroupMatrix
            )
            try kernel.linearBF16WeightsF32Output(
                input: input, checkpoint: checkpoint, weightOffset: weightOffset,
                biasOffset: biasOffset, rows: shape.rows, inputChannels: shape.inputs,
                outputChannels: shape.outputs, output: mpsGraph,
                implementation: .mpsGraph
            )
            let tiledValues = tiled.contents().assumingMemoryBound(to: Float.self)
            let simdgroupValues = simdgroup.contents().assumingMemoryBound(to: Float.self)
            let mpsGraphValues = mpsGraph.contents().assumingMemoryBound(to: Float.self)
            for row in 0..<shape.rows {
                for outputChannel in 0..<shape.outputs {
                    var expected = hasBias ? Float(bitPattern: UInt32(checkpointValues[
                        biasElementOffset + outputChannel
                    ]) << 16) : 0
                    for inputChannel in 0..<shape.inputs {
                        let weight = Float(bitPattern: UInt32(checkpointValues[
                            prefixElements + outputChannel * shape.inputs + inputChannel
                        ]) << 16)
                        expected = expected.addingProduct(
                            inputValues[row * shape.inputs + inputChannel], weight
                        )
                    }
                    let index = row * shape.outputs + outputChannel
                    #expect(abs(tiledValues[index] - expected) <= 5e-6)
                    #expect(abs(simdgroupValues[index] - expected) <= 5e-6)
                    #expect(abs(simdgroupValues[index] - tiledValues[index]) <= 5e-6)
                    #expect(abs(mpsGraphValues[index] - expected) <= 5e-6)
                    #expect(abs(mpsGraphValues[index] - tiledValues[index]) <= 5e-6)
                }
            }
            for index in outputElements..<(outputElements + suffixElements) {
                #expect(tiledValues[index] == canary)
                #expect(simdgroupValues[index] == canary)
                #expect(mpsGraphValues[index] == canary)
            }
        }
    }

    @Test("BF16 dense automatic routing preserves the tiled fallback")
    func bf16DenseProjectionFallbackAndDimensionBounds() throws {
        let context = try MetalContext()
        let fallback = try DenseKernel(context: context, enableSIMDGroupMatrix: false)
        #expect(!fallback.supportsSIMDGroupMatrix)
        var inputValues: [Float] = [1, -2, 0.5]
        var checkpointValues: [UInt16] = [bf16(1), bf16(2), bf16(3)]
        let input = try #require(context.device.makeBuffer(
            bytes: &inputValues, length: inputValues.count * 4, options: .storageModeShared
        ))
        let checkpoint = try #require(context.device.makeBuffer(
            bytes: &checkpointValues, length: checkpointValues.count * 2,
            options: .storageModeShared
        ))
        let output = try #require(context.device.makeBuffer(length: 4, options: .storageModeShared))
        try fallback.linearBF16WeightsF32Output(
            input: input, checkpoint: checkpoint, weightOffset: 0,
            rows: 1, inputChannels: 3, outputChannels: 1, output: output,
            implementation: .automatic
        )
        #expect(abs(output.contents().assumingMemoryBound(to: Float.self)[0] - -1.5) < 1e-6)
        try fallback.linearBF16WeightsF32Output(
            input: input, checkpoint: checkpoint, weightOffset: 0,
            rows: 1, inputChannels: 3, outputChannels: 1, output: output,
            implementation: .automaticFloat32
        )
        #expect(abs(output.contents().assumingMemoryBound(to: Float.self)[0] - -1.5) < 1e-6)
        #expect(throws: NativeRuntimeError.self) {
            try fallback.linearBF16WeightsF32Output(
                input: input, checkpoint: checkpoint, weightOffset: 0,
                rows: 1, inputChannels: Int(UInt32.max) - 15,
                outputChannels: 1, output: output, implementation: .tiled
            )
        }
        let kernel = try DenseKernel(context: context)
        if kernel.supportsSIMDGroupMatrix {
            #expect(throws: NativeRuntimeError.self) {
                try kernel.linearBF16WeightsF32Output(
                    input: input, checkpoint: checkpoint, weightOffset: 0,
                    rows: 1, inputChannels: Int(UInt32.max) - 7,
                    outputChannels: 1, output: output, implementation: .simdgroupMatrix
                )
            }
        }
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

    @Test("Metal voxel-major Conv3D matches F32 and F16 CPU references")
    func sparseStructureConv3D() throws {
        let context = try MetalContext()
        let kernel = try SparseStructureKernel(context: context)
        let resolution = 3, inputChannels = 2, outputChannels = 2
        let inputCount = resolution * resolution * resolution * inputChannels
        var inputValues = (0..<inputCount).map {
            Float(sin(Double($0) * 0.17) * 0.4)
        }
        let weightCount = outputChannels * inputChannels * 3 * 3 * 3
        let f32Weights = (0..<weightCount).map {
            Float(cos(Double($0) * 0.11) * 0.2)
        }
        let f32Bias: [Float] = [0.125, -0.25]
        var f32Checkpoint = f32Weights
        f32Checkpoint.append(contentsOf: f32Bias)
        let f16Weights = f32Weights.map(Float16.init)
        let f16Bias = f32Bias.map(Float16.init)
        var f16Checkpoint = f16Weights
        f16Checkpoint.append(contentsOf: f16Bias)
        let input = try #require(context.device.makeBuffer(
            bytes: &inputValues, length: inputValues.count * 4,
            options: .storageModeShared
        ))
        let f32CheckpointBuffer = try #require(context.device.makeBuffer(
            bytes: &f32Checkpoint, length: f32Checkpoint.count * 4,
            options: .storageModeShared
        ))
        let f16CheckpointBuffer = try #require(context.device.makeBuffer(
            bytes: &f16Checkpoint, length: f16Checkpoint.count * 2,
            options: .storageModeShared
        ))
        let outputCount = resolution * resolution * resolution * outputChannels
        let f32Output = try #require(context.device.makeBuffer(
            length: outputCount * 4, options: .storageModeShared
        ))
        let f16Output = try #require(context.device.makeBuffer(
            length: outputCount * 4, options: .storageModeShared
        ))
        try kernel.conv3DF32(
            input: input, checkpoint: f32CheckpointBuffer,
            weightOffset: 0, biasOffset: weightCount * 4,
            inputResolution: resolution, inputChannels: inputChannels,
            outputChannels: outputChannels, weightType: .f32, output: f32Output
        )
        try kernel.conv3DF32(
            input: input, checkpoint: f16CheckpointBuffer,
            weightOffset: 0, biasOffset: weightCount * 2,
            inputResolution: resolution, inputChannels: inputChannels,
            outputChannels: outputChannels, weightType: .f16, output: f16Output
        )
        #expect(throws: NativeRuntimeError.self) {
            try kernel.conv3DF32(
                input: input, checkpoint: f32CheckpointBuffer,
                weightOffset: 0, biasOffset: weightCount * 4,
                inputResolution: resolution, inputChannels: inputChannels,
                outputChannels: outputChannels, padding: Int.max,
                weightType: .f32, output: f32Output
            )
        }
        let actualF32 = f32Output.contents().assumingMemoryBound(to: Float.self)
        let actualF16 = f16Output.contents().assumingMemoryBound(to: Float.self)
        for x in 0..<resolution {
            for y in 0..<resolution {
                for z in 0..<resolution {
                    for outputChannel in 0..<outputChannels {
                        var expectedF32 = f32Bias[outputChannel]
                        var expectedF16 = Float(f16Bias[outputChannel])
                        for inputChannel in 0..<inputChannels {
                            for kx in 0..<3 {
                                let ix = x + kx - 1
                                guard ix >= 0, ix < resolution else { continue }
                                for ky in 0..<3 {
                                    let iy = y + ky - 1
                                    guard iy >= 0, iy < resolution else { continue }
                                    for kz in 0..<3 {
                                        let iz = z + kz - 1
                                        guard iz >= 0, iz < resolution else { continue }
                                        let inputIndex = ((ix * resolution + iy) * resolution
                                            + iz) * inputChannels + inputChannel
                                        let weightIndex = ((((outputChannel * inputChannels
                                            + inputChannel) * 3 + kx) * 3 + ky) * 3 + kz)
                                        expectedF32.addProduct(
                                            inputValues[inputIndex], f32Weights[weightIndex]
                                        )
                                        expectedF16.addProduct(
                                            inputValues[inputIndex], Float(f16Weights[weightIndex])
                                        )
                                    }
                                }
                            }
                        }
                        let outputIndex = ((x * resolution + y) * resolution + z)
                            * outputChannels + outputChannel
                        #expect(abs(actualF32[outputIndex] - expectedF32) < 2e-6)
                        #expect(abs(actualF16[outputIndex] - expectedF16) < 2e-6)
                    }
                }
            }
        }
    }

    @Test("Metal F16 boundary and PixelShuffle3D preserve exact decoder layout")
    func sparseStructureF16AndPixelShuffle() throws {
        let context = try MetalContext()
        let kernel = try SparseStructureKernel(context: context)
        var values: [Float] = [
            1.0001, -2.0009, Float.leastNonzeroMagnitude, .infinity, -.infinity,
        ]
        let input = try #require(context.device.makeBuffer(
            bytes: &values, length: values.count * 4, options: .storageModeShared
        ))
        let rounded = try #require(context.device.makeBuffer(
            length: values.count * 4, options: .storageModeShared
        ))
        try kernel.roundF16F32(input: input, count: values.count, output: rounded)
        let actualRounded = rounded.contents().assumingMemoryBound(to: Float.self)
        for index in values.indices {
            #expect(actualRounded[index].bitPattern == Float(Float16(values[index])).bitPattern)
        }

        let inputResolution = 2, outputChannels = 2, factor = 2
        let inputChannels = outputChannels * factor * factor * factor
        var shuffledInput = (0..<(inputResolution * inputResolution * inputResolution
            * inputChannels)).map(Float.init)
        let shuffledInputBuffer = try #require(context.device.makeBuffer(
            bytes: &shuffledInput, length: shuffledInput.count * 4,
            options: .storageModeShared
        ))
        let outputResolution = inputResolution * factor
        let shuffledOutput = try #require(context.device.makeBuffer(
            length: outputResolution * outputResolution * outputResolution
                * outputChannels * 4,
            options: .storageModeShared
        ))
        try kernel.pixelShuffle3DF32(
            input: shuffledInputBuffer, inputResolution: inputResolution,
            outputChannels: outputChannels, factor: factor, output: shuffledOutput
        )
        let actual = shuffledOutput.contents().assumingMemoryBound(to: Float.self)
        for x in 0..<outputResolution {
            for y in 0..<outputResolution {
                for z in 0..<outputResolution {
                    for channel in 0..<outputChannels {
                        let inputVoxel = ((x / factor * inputResolution + y / factor)
                            * inputResolution + z / factor)
                        let shuffledChannel = channel * factor * factor * factor
                            + ((x % factor) * factor + y % factor) * factor + z % factor
                        let expected = shuffledInput[inputVoxel * inputChannels + shuffledChannel]
                        let outputIndex = ((x * outputResolution + y) * outputResolution + z)
                            * outputChannels + channel
                        #expect(actual[outputIndex] == expected)
                    }
                }
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
        let metalExpected = Array(UnsafeBufferPointer(start: actual, count: queries.count))
        try kernel.fusedF32(
            queries: queryBuffer, keys: keyBuffer, values: valueBuffer,
            queryCount: queryCount, keyCount: keyCount, heads: heads,
            dimensions: dimensions, output: output,
            implementation: .automaticFloat32
        )
        for index in 0..<queries.count {
            #expect(actual[index] == metalExpected[index])
        }
    }

    @Test("production 4,096-token Metal attention matches sampled MPS SDPA")
    func fusedAttentionProductionGolden() throws {
        let fixtureURL = try #require(Bundle.module.url(
            forResource: "attention-r4096-d128-mps", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(fileSHA256(at: fixtureURL) ==
            "96cefd6c6245983e08af9a5831638cc7d48b52222370e6d50fbf869bd570974f")
        let metadataURL = fixtureURL.appendingPathExtension("json")
        try #require(fileSHA256(at: metadataURL) ==
            "d7439483e6d2094d5f714da6010d5d2afd31d4c33007d0700cabd6bad0b1f251")
        let metadata = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL))
                as? [String: Any]
        )
        #expect(metadata["source_revision"] as? String ==
            "75fbf0183001ed9876c8dbb35de6b68552ee08bd")
        #expect(metadata["source_sha256"] as? String ==
            "64c43354780dcbc3dcf7612ac5e53d6e21c2081234ea63cd329a77f4185dadfc")
        #expect(metadata["pytorch_enable_mps_fallback"] as? String == "0")
        let sampledQueries = try #require(metadata["sampled_queries"] as? [Int])
        let expected = try Data(contentsOf: fixtureURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        let tokens = 4096, heads = 12, dimensions = 128
        let count = tokens * heads * dimensions
        try #require(expected.count == sampledQueries.count * heads * dimensions)
        func fixtureValues(multiplier: UInt32, increment: UInt32, scale: Float) -> [Float] {
            (0..<count).map { index in
                let bits = UInt32(truncatingIfNeeded: index) &* multiplier &+ increment
                let value = Float(Int32(bitPattern: bits)) / Float(Int32.max) * scale
                return fromBF16(roundedBF16(value))
            }
        }
        var queries = fixtureValues(
            multiplier: 1_664_525, increment: 1_013_904_223, scale: 1
        )
        var keys = fixtureValues(multiplier: 22_695_477, increment: 1, scale: 1)
        var values = fixtureValues(
            multiplier: 1_103_515_245, increment: 12_345, scale: 0.5
        )
        let context = try MetalContext()
        let kernel = try AttentionKernel(context: context)
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
            length: count * 4, options: .storageModeShared
        ))
        try kernel.fusedF32(
            queries: queryBuffer, keys: keyBuffer, values: valueBuffer,
            queryCount: tokens, keyCount: tokens, heads: heads,
            dimensions: dimensions, output: output
        )
        let actual = output.contents().assumingMemoryBound(to: Float.self)
        var sampledActual: [Float] = []
        sampledActual.reserveCapacity(expected.count)
        for query in sampledQueries {
            let start = query * heads * dimensions
            sampledActual += (0..<(heads * dimensions)).map { actual[start + $0] }
        }
        let metrics = try compareFixtureValues(actual: sampledActual, expected: expected)
        print(
            "production attention: max=\(metrics.maximumError) rms=\(metrics.rms) " +
                "normalized_rms=\(metrics.normalizedRMS) " +
                "max_scale_ratio=\(metrics.maximumScaleRatio)"
        )
        #expect(metrics.normalizedRMS <= 0.005)
        #expect(metrics.maximumScaleRatio <= 0.01)

        let float32Output = try #require(context.device.makeBuffer(
            length: count * 4, options: .storageModeShared
        ))
        try kernel.fusedF32(
            queries: queryBuffer, keys: keyBuffer, values: valueBuffer,
            queryCount: tokens, keyCount: tokens, heads: heads,
            dimensions: dimensions, output: float32Output,
            implementation: .mpsGraphFloat32
        )
        let float32Actual = float32Output.contents().assumingMemoryBound(to: Float.self)
        var sampledFloat32: [Float] = []
        sampledFloat32.reserveCapacity(expected.count)
        for query in sampledQueries {
            let start = query * heads * dimensions
            sampledFloat32 += (0..<(heads * dimensions)).map { float32Actual[start + $0] }
        }
        let float32Metrics = try compareFixtureValues(
            actual: sampledFloat32, expected: expected
        )
        print(
            "production F32 attention: max=\(float32Metrics.maximumError) " +
                "rms=\(float32Metrics.rms) " +
                "normalized_rms=\(float32Metrics.normalizedRMS) " +
                "max_scale_ratio=\(float32Metrics.maximumScaleRatio)"
        )
        #expect(float32Metrics.normalizedRMS <= 0.005)
        #expect(float32Metrics.maximumScaleRatio <= 0.01)
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

    @Test("Metal SIMD-group attention matches 128-wide sparse-structure heads")
    func simdgroupAttention128() throws {
        let context = try MetalContext()
        let kernel = try AttentionKernel(context: context)
        let queryCount = 17, keyCount = 19, heads = 3, dimensions = 128
        var queries = (0..<(queryCount * heads * dimensions)).map {
            Float(sin(Double($0) * 0.013) * 0.4)
        }
        var keys = (0..<(keyCount * heads * dimensions)).map {
            Float(cos(Double($0) * 0.017) * 0.35)
        }
        var values = (0..<(keyCount * heads * dimensions)).map {
            Float(sin(Double($0) * 0.019) * 0.3 - 0.1)
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
                        expected += weights[key] / denominator
                            * values[keyBase + dimension]
                    }
                    #expect(abs(actual[queryBase + dimension] - expected) < 5e-6)
                }
            }
        }
    }

    @Test("native sparse Flow Euler and CFG match the pinned Torch oracle")
    func sparseFlowEuler() throws {
        let context = try MetalContext()
        let parameters = try FlowEulerParameters.sparseStructure512()
        let expectedSchedule = [
            1.0, 0.9821428571428572, 0.9615384615384615, 0.9375,
            0.9090909090909092, 0.875, 0.8333333333333334,
            0.7812500000000001, 0.7142857142857144, 0.625,
            0.5000000000000001, 0.3125000000000001, 0.0,
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
        #expect(result.modelCallCount == 22)
        #expect(calls.count == 22)
        for index in 0..<20 {
            #expect(calls[index].0 == (index.isMultiple(of: 2) ? "positive" : "negative"))
        }
        for index in 20..<22 { #expect(calls[index].0 == "positive") }

        let fixtureURL = try #require(Bundle.module.url(
            forResource: "flow-euler-sparse", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: fixtureURL) ==
                "c91f49c9ad4682abc9b754fea7d210f26da32741839ac1ea3680050f70fa4f4d"
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
                "ab6be0a656f73539e6faff1ca47fa63c949a9af29300914f6a000a5c6699c796"
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

    @Test("SIMD-group normalization matches scalar production widths")
    func simdgroupNormalizationProductionWidths() throws {
        let context = try MetalContext()
        let kernel = try NormalizationKernel(context: context)
        let rows = 4, channels = 1536, heads = 12, dimensions = 128
        var inputValues = (0..<(rows * channels)).map {
            Float(sin(Double($0) * 0.013) * 0.7 + cos(Double($0) * 0.007) * 0.2)
        }
        var checkpointValues = (0..<channels).map {
            bf16(0.8 + Float($0 % 17) * 0.01)
        }
        checkpointValues += (0..<channels).map {
            bf16(-0.05 + Float($0 % 11) * 0.005)
        }
        let gammaOffset = checkpointValues.count * MemoryLayout<UInt16>.stride
        checkpointValues += (0..<(heads * dimensions)).map {
            bf16(0.9 + Float($0 % 13) * 0.008)
        }
        let input = try #require(context.device.makeBuffer(
            bytes: &inputValues,
            length: inputValues.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let checkpoint = try #require(context.device.makeBuffer(
            bytes: &checkpointValues,
            length: checkpointValues.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared
        ))
        let outputBytes = inputValues.count * MemoryLayout<Float>.stride
        let scalar = try #require(context.device.makeBuffer(
            length: outputBytes, options: .storageModeShared
        ))
        let simdgroup = try #require(context.device.makeBuffer(
            length: outputBytes, options: .storageModeShared
        ))

        try kernel.layerNormF32(
            input: input, checkpoint: checkpoint, rows: rows, channels: channels,
            weightOffset: 0,
            biasOffset: channels * MemoryLayout<UInt16>.stride,
            output: scalar, implementation: .scalar
        )
        try kernel.layerNormF32(
            input: input, checkpoint: checkpoint, rows: rows, channels: channels,
            weightOffset: 0,
            biasOffset: channels * MemoryLayout<UInt16>.stride,
            output: simdgroup, implementation: .simdgroup
        )
        var scalarValues = scalar.contents().assumingMemoryBound(to: Float.self)
        var simdValues = simdgroup.contents().assumingMemoryBound(to: Float.self)
        for index in inputValues.indices {
            #expect(abs(scalarValues[index] - simdValues[index]) <= 2e-5)
        }

        try kernel.multiheadRMSNormF32(
            input: input, checkpoint: checkpoint, gammaOffset: gammaOffset,
            rows: rows, heads: heads, dimensions: dimensions,
            output: scalar, implementation: .scalar
        )
        try kernel.multiheadRMSNormF32(
            input: input, checkpoint: checkpoint, gammaOffset: gammaOffset,
            rows: rows, heads: heads, dimensions: dimensions,
            output: simdgroup, implementation: .simdgroup
        )
        scalarValues = scalar.contents().assumingMemoryBound(to: Float.self)
        simdValues = simdgroup.contents().assumingMemoryBound(to: Float.self)
        for index in inputValues.indices {
            #expect(abs(scalarValues[index] - simdValues[index]) <= 2e-5)
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
        var lifecycle: [StageLifecycleEvent] = []
        let session = try StageSession(
            checkpointURL: URL(fileURLWithPath: path),
            expectedCheckpointSHA256:
                "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
            arenaCapacity: 64 * 1024 * 1024,
            lifecycleObserver: { lifecycle.append($0) }
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
        let image = try #require(session.device.makeBuffer(
            bytes: &imageValues, length: imageValues.count * 4,
            options: .storageModeShared
        ))
        var traces: [(String, [Float])] = []
        let result = try session.encodeDINOv3F32(
            normalizedImage: image, imageHeight: 32, imageWidth: 32,
            trace: { name, values in traces.append((name, values)) }
        )
        let memory = try session.close()
        #expect(memory.usedBytes == 0)
        #expect(lifecycle == [.queueDrained, .arenaReleased, .checkpointUnmapped])
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
        var lifecycle: [StageLifecycleEvent] = []
        let session = try StageSession(
            checkpointURL: URL(fileURLWithPath: path),
            expectedCheckpointSHA256:
                "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
            arenaCapacity: 256 * 1024 * 1024,
            lifecycleObserver: { lifecycle.append($0) }
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
        let image = try #require(session.device.makeBuffer(
            bytes: &imageValues, length: imageValues.count * 4,
            options: .storageModeShared
        ))
        let started = ContinuousClock.now
        let result = try session.encodeDINOv3F32(
            normalizedImage: image, imageHeight: 512, imageWidth: 512
        )
        let elapsed = started.duration(to: .now)
        let memory = try session.close()
        #expect(memory.usedBytes == 0)
        #expect(lifecycle == [.queueDrained, .arenaReleased, .checkpointUnmapped])
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
        var lifecycle: [StageLifecycleEvent] = []
        let session = try StageSession(
            checkpointURL: URL(fileURLWithPath: path),
            expectedCheckpointSHA256:
                "ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f",
            arenaCapacity: 16 * 1024 * 1024,
            lifecycleObserver: { lifecycle.append($0) }
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
        let noiseBuffer = try #require(session.device.makeBuffer(
            bytes: &noise, length: noise.count * 4, options: .storageModeShared
        ))
        let positiveBuffer = try #require(session.device.makeBuffer(
            bytes: &positiveConditioning, length: positiveConditioning.count * 4,
            options: .storageModeShared
        ))
        let negativeBuffer = try #require(session.device.makeBuffer(
            bytes: &negativeConditioning, length: negativeConditioning.count * 4,
            options: .storageModeShared
        ))
        let coordinateBuffer = try #require(session.device.makeBuffer(
            bytes: &coordinates, length: coordinates.count * 4,
            options: .storageModeShared
        ))
        var modelTrace: [[Float]] = []
        var samplerTrace: [[Float]] = []

        let result = try session.sampleShapeF32(
            noise: noiseBuffer, coordinates: coordinateBuffer,
            positiveConditioning: positiveBuffer,
            negativeConditioning: negativeBuffer,
            tokens: tokens, conditioningTokens: conditioningTokens,
            parameters: .shape512(steps: 2),
            cacheCrossKV: true,
            modelTrace: { call, _, values in
                #expect(call == modelTrace.count)
                modelTrace.append(values)
            },
            samplerTrace: { step, values in
                #expect(step == samplerTrace.count)
                samplerTrace.append(values)
            }
        )
        #expect(result.modelCallCount == 4)
        #expect(result.crossKVCacheStats == SLatCrossKVCacheStats(hits: 60, misses: 60))
        let memory = try session.close()
        #expect(memory.usedBytes == 0)
        #expect(memory.peakUsedBytes < memory.capacityBytes)
        #expect(lifecycle == [.queueDrained, .arenaReleased, .checkpointUnmapped])
        print(
            "shape sampler arena: peak=\(memory.peakUsedBytes) " +
            "live=\(memory.usedBytes) allocations=\(memory.allocationCount)"
        )

        let uncachedSession = try StageSession(
            checkpointURL: URL(fileURLWithPath: path),
            expectedCheckpointSHA256:
                "ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f",
            arenaCapacity: 16 * 1024 * 1024
        )
        let uncachedNoise = try #require(uncachedSession.device.makeBuffer(
            bytes: &noise, length: noise.count * 4, options: .storageModeShared
        ))
        let uncachedPositive = try #require(uncachedSession.device.makeBuffer(
            bytes: &positiveConditioning, length: positiveConditioning.count * 4,
            options: .storageModeShared
        ))
        let uncachedNegative = try #require(uncachedSession.device.makeBuffer(
            bytes: &negativeConditioning, length: negativeConditioning.count * 4,
            options: .storageModeShared
        ))
        let uncachedCoordinates = try #require(uncachedSession.device.makeBuffer(
            bytes: &coordinates, length: coordinates.count * 4,
            options: .storageModeShared
        ))
        let uncached = try uncachedSession.sampleShapeF32(
            noise: uncachedNoise, coordinates: uncachedCoordinates,
            positiveConditioning: uncachedPositive,
            negativeConditioning: uncachedNegative,
            tokens: tokens, conditioningTokens: conditioningTokens,
            parameters: .shape512(steps: 2), cacheCrossKV: false
        )
        #expect(uncached.crossKVCacheStats == SLatCrossKVCacheStats(hits: 0, misses: 0))
        let cachedPointer = result.latent.contents().assumingMemoryBound(to: Float.self)
        let uncachedPointer = uncached.latent.contents().assumingMemoryBound(to: Float.self)
        let cachedValues = (0..<noise.count).map { cachedPointer[$0] }
        let uncachedValues = (0..<noise.count).map { uncachedPointer[$0] }
        #expect(cachedValues == uncachedValues)
        let uncachedMemory = try uncachedSession.close()
        #expect(uncachedMemory.usedBytes == 0)

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
        var lifecycle: [StageLifecycleEvent] = []
        let session = try StageSession(
            checkpointURL: URL(fileURLWithPath: path),
            expectedCheckpointSHA256:
                "8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d",
            arenaCapacity: 16 * 1024 * 1024,
            lifecycleObserver: { lifecycle.append($0) }
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
        let noiseBuffer = try #require(session.device.makeBuffer(
            bytes: &noise, length: noise.count * 4, options: .storageModeShared
        ))
        let shapeBuffer = try #require(session.device.makeBuffer(
            bytes: &shape, length: shape.count * 4, options: .storageModeShared
        ))
        let conditioningBuffer = try #require(session.device.makeBuffer(
            bytes: &conditioning, length: conditioning.count * 4,
            options: .storageModeShared
        ))
        let coordinateBuffer = try #require(session.device.makeBuffer(
            bytes: &coordinates, length: coordinates.count * 4,
            options: .storageModeShared
        ))
        var modelTrace: [[Float]] = []
        var samplerTrace: [[Float]] = []
        let result = try session.sampleTextureF32(
            noise: noiseBuffer, shapeLatent: shapeBuffer,
            coordinates: coordinateBuffer, positiveConditioning: conditioningBuffer,
            tokens: tokens, conditioningTokens: conditioningTokens,
            parameters: .texture512(steps: 2),
            modelTrace: { call, values in
                #expect(call == modelTrace.count)
                modelTrace.append(values)
            },
            samplerTrace: { step, values in
                #expect(step == samplerTrace.count)
                samplerTrace.append(values)
            }
        )
        #expect(result.modelCallCount == 2)
        let memory = try session.close()
        #expect(memory.usedBytes == 0)
        #expect(memory.peakUsedBytes < memory.capacityBytes)
        #expect(lifecycle == [.queueDrained, .arenaReleased, .checkpointUnmapped])
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
        "real sparse-structure block matches the pinned dense Torch oracle",
        .enabled(
            if: ProcessInfo.processInfo.environment[
                "KG_TRELLIS2_SPARSE_STRUCTURE_FLOW_CHECKPOINT"
            ] != nil,
            "Set KG_TRELLIS2_SPARSE_STRUCTURE_FLOW_CHECKPOINT for real-weight conformance"
        )
    )
    func realSparseStructureBlockGolden() throws {
        let path = try #require(ProcessInfo.processInfo.environment[
            "KG_TRELLIS2_SPARSE_STRUCTURE_FLOW_CHECKPOINT"
        ])
        let context = try MetalContext()
        let checkpoint = try MappedCheckpoint(
            url: URL(fileURLWithPath: path), device: context.device
        )
        try #require(
            checkpoint.sha256() ==
                "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6"
        )
        let tokens = 2
        let stageElements = tokens * 1536
        var features = (0..<stageElements).map {
            fromBF16(roundedBF16(Float(sin(Double($0) * 0.013) * 0.35)))
        }
        var modulationValues = (0..<9216).map {
            fromBF16(roundedBF16(Float(sin(Double($0) * 0.007) * 0.2)))
        }
        var conditioningValues = (0..<(2 * 1024)).map {
            fromBF16(roundedBF16(Float(sin(Double($0) * 0.015) * 0.25)))
        }
        var coordinateValues: [Int32] = [0, 0, 0, 0, 0, 0, 0, 1]
        let featureBuffer = try #require(context.device.makeBuffer(
            bytes: &features, length: features.count * 4, options: .storageModeShared
        ))
        let modulationBuffer = try #require(context.device.makeBuffer(
            bytes: &modulationValues, length: modulationValues.count * 4,
            options: .storageModeShared
        ))
        let conditioningBuffer = try #require(context.device.makeBuffer(
            bytes: &conditioningValues, length: conditioningValues.count * 4,
            options: .storageModeShared
        ))
        let coordinateBuffer = try #require(context.device.makeBuffer(
            bytes: &coordinateValues, length: coordinateValues.count * 4,
            options: .storageModeShared
        ))
        var traces: [String: MTLBuffer] = [:]
        let output = try SLatBlock(context: context).forwardF32(
            input: featureBuffer, sharedModulation: modulationBuffer,
            conditioning: conditioningBuffer, checkpoint: checkpoint,
            block: 0, tokens: tokens, conditioningTokens: 2,
            coordinates: coordinateBuffer,
            trace: { name, buffer in traces[name] = buffer }
        )
        let fixtureURL = try #require(Bundle.module.url(
            forResource: "ss-block0-tiny", withExtension: "bf16",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: fixtureURL) ==
                "f1f436bdfd034e90cc968ae8bee17b96b35d9ad30d605ecf4eba2d7e209adf3e"
        )
        let expected = try Data(contentsOf: fixtureURL).withUnsafeBytes {
            Array($0.bindMemory(to: UInt16.self))
        }
        try #require(expected.count == stageElements)
        let actual = output.contents().assumingMemoryBound(to: Float.self)
        var maximumError: Float = 0
        var maximumMixedToleranceRatio: Float = 0
        var squaredError: Double = 0
        var expectedSquaredMagnitude: Double = 0
        for index in expected.indices {
            try #require(actual[index].isFinite)
            let expectedValue = fromBF16(UInt16(littleEndian: expected[index]))
            let error = abs(actual[index] - expectedValue)
            maximumError = max(maximumError, error)
            squaredError += Double(error * error)
            expectedSquaredMagnitude += Double(expectedValue * expectedValue)
            maximumMixedToleranceRatio = max(
                maximumMixedToleranceRatio,
                error / (0.25 + 0.008 * abs(expectedValue))
            )
        }
        let rms = sqrt(squaredError / Double(expected.count))
        let expectedRMS = sqrt(expectedSquaredMagnitude / Double(expected.count))
        let normalizedRMS = rms / expectedRMS
        print(
            "sparse-structure block: max=\(maximumError) rms=\(rms) " +
                "normalized_rms=\(normalizedRMS) mixed_ratio=\(maximumMixedToleranceRatio)"
        )
        // BF16 contributes one 0.78125% relative ULP per rounded projection;
        // the absolute floor covers cancellation near zero after residuals.
        #expect(maximumMixedToleranceRatio <= 1)
        #expect(normalizedRMS <= 0.002)

        let traceURL = try #require(Bundle.module.url(
            forResource: "ss-block0-tiny", withExtension: "bf16.trace",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: traceURL) ==
                "1df17f811f4b20e74b6c2e615c415ec8a63db112a7aa0365c0098af4797f3bac"
        )
        let trace = try Data(contentsOf: traceURL).withUnsafeBytes {
            Array($0.bindMemory(to: UInt16.self))
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
        var offset = 0
        for (name, count) in stages {
            let values = try #require(traces[name]).contents()
                .assumingMemoryBound(to: Float.self)
            var stageMaximum: Float = 0
            var stageMixedToleranceRatio: Float = 0
            var stageSquaredError: Double = 0
            var stageExpectedSquaredMagnitude: Double = 0
            for index in 0..<count {
                try #require(values[index].isFinite)
                let expectedValue = fromBF16(UInt16(littleEndian: trace[offset + index]))
                let error = abs(values[index] - expectedValue)
                stageMaximum = max(stageMaximum, error)
                stageSquaredError += Double(error * error)
                stageExpectedSquaredMagnitude += Double(expectedValue * expectedValue)
                stageMixedToleranceRatio = max(
                    stageMixedToleranceRatio,
                    error / (0.25 + 0.008 * abs(expectedValue))
                )
            }
            offset += count
            let stageRMS = sqrt(stageSquaredError / Double(count))
            let stageExpectedRMS = sqrt(stageExpectedSquaredMagnitude / Double(count))
            let stageNormalizedRMS = stageRMS / max(stageExpectedRMS, 1e-12)
            print(
                "sparse_trace=\(name) max=\(stageMaximum) rms=\(stageRMS) " +
                    "normalized_rms=\(stageNormalizedRMS) " +
                    "mixed_ratio=\(stageMixedToleranceRatio)"
            )
            #expect(stageMixedToleranceRatio <= 1)
            #expect(stageNormalizedRMS <= 0.002)
        }
    }

    @Test(
        "production sparse-structure sampler runs 4,096 tokens through all 30 blocks",
        .enabled(
            if: ProcessInfo.processInfo.environment[
                "KG_TRELLIS2_SPARSE_STRUCTURE_FLOW_CHECKPOINT"
            ] != nil,
            "Set KG_TRELLIS2_SPARSE_STRUCTURE_FLOW_CHECKPOINT for real-weight conformance"
        )
    )
    func realSparseStructureSamplerProductionGolden() throws {
        let path = try #require(ProcessInfo.processInfo.environment[
            "KG_TRELLIS2_SPARSE_STRUCTURE_FLOW_CHECKPOINT"
        ])
        let fixtureURL = try #require(Bundle.module.url(
            forResource: "ss-sampler-r16-1step-mps", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: fixtureURL) ==
                "9c7cd37e4c3abe1575ba93b1afbb3bee12e8d5295d8905a113bd13a2f0b50936"
        )
        let metadataURL = fixtureURL.appendingPathExtension("json")
        try #require(
            fileSHA256(at: metadataURL) ==
                "ca3384b3c46b8ca32d578ebd87212412d51e31fca32d8bf787902df015caaac9"
        )
        let metadata = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL))
                as? [String: Any]
        )
        #expect(metadata["source_revision"] as? String ==
            "75fbf0183001ed9876c8dbb35de6b68552ee08bd")
        #expect(metadata["weight_revision"] as? String ==
            "af44b45f2e35a493886929c6d786e563ec68364d")
        #expect(metadata["weight_sha256"] as? String ==
            "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6")
        #expect(metadata["payload_sha256"] as? String ==
            "9c7cd37e4c3abe1575ba93b1afbb3bee12e8d5295d8905a113bd13a2f0b50936")
        #expect(metadata["tokens"] as? Int == 4096)
        #expect(metadata["context_tokens"] as? Int == 1029)
        #expect(metadata["steps"] as? Int == 1)
        #expect(metadata["model_calls"] as? Int == 2)
        let fixture = try Data(contentsOf: fixtureURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        let tokens = 16 * 16 * 16
        let latentCount = tokens * 8
        let contextTokens = 1029
        let contextCount = contextTokens * 1024
        try #require(fixture.count == latentCount * 4 + contextCount)
        var noise = Array(fixture[0..<latentCount])
        var positive = Array(fixture[latentCount..<(latentCount + contextCount)])
        var negative = [Float](repeating: 0, count: contextCount)
        let traceStart = latentCount + contextCount
        let expectedTrace = Array(fixture[traceStart..<(traceStart + latentCount * 2)])
        let expectedOutput = Array(fixture[(traceStart + latentCount * 2)...])

        var lifecycle: [StageLifecycleEvent] = []
        let session = try StageSession(
            checkpointURL: URL(fileURLWithPath: path),
            expectedCheckpointSHA256:
                "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6",
            arenaCapacity: 1536 * 1024 * 1024,
            lifecycleObserver: { lifecycle.append($0) }
        )
        let noiseBuffer = try #require(session.device.makeBuffer(
            bytes: &noise, length: noise.count * 4, options: .storageModeShared
        ))
        let positiveBuffer = try #require(session.device.makeBuffer(
            bytes: &positive, length: positive.count * 4, options: .storageModeShared
        ))
        let negativeBuffer = try #require(session.device.makeBuffer(
            bytes: &negative, length: negative.count * 4, options: .storageModeShared
        ))
        var traces: [[Float]] = []
        let started = ContinuousClock.now
        let result = try session.sampleSparseStructureF32(
            noise: noiseBuffer,
            positiveConditioning: positiveBuffer,
            negativeConditioning: negativeBuffer,
            conditioningTokens: contextTokens,
            parameters: .sparseStructure512(steps: 1),
            cacheCrossKV: true,
            modelTrace: { call, pass, values in
                #expect(call == traces.count)
                #expect(pass == (call == 0 ? .positive : .negative))
                traces.append(values)
            }
        )
        let elapsed = started.duration(to: .now)
        #expect(result.modelCallCount == 2)
        #expect(result.crossKVCacheStats == SLatCrossKVCacheStats(hits: 0, misses: 60))
        try #require(traces.count == 2)
        let memory = try session.close()
        #expect(memory.usedBytes == 0)
        // Exact positive and negative cross-K/V caches intentionally trade
        // memory for avoiding invariant work across sampler calls.
        #expect(memory.peakUsedBytes <= 1400 * 1024 * 1024)
        #expect(lifecycle == [.queueDrained, .arenaReleased, .checkpointUnmapped])

        for call in 0..<2 {
            let expected = Array(
                expectedTrace[(call * latentCount)..<((call + 1) * latentCount)]
            )
            let metrics = try compareFixtureValues(actual: traces[call], expected: expected)
            print(
                "sparse sampler call \(call): max=\(metrics.maximumError) " +
                    "rms=\(metrics.rms) normalized_rms=\(metrics.normalizedRMS) " +
                    "max_scale_ratio=\(metrics.maximumScaleRatio)"
            )
            #expect(metrics.normalizedRMS <= 0.05)
            #expect(metrics.maximumScaleRatio <= 0.10)
        }
        let actualOutput = result.latent.contents().assumingMemoryBound(to: Float.self)
        let actualValues = (0..<latentCount).map { actualOutput[$0] }
        let expectedFromNativeCalls = oneStepSparseSamplerReference(
            noise: noise, positive: traces[0], negative: traces[1]
        )
        let orchestrationMetrics = try compareFixtureValues(
            actual: actualValues, expected: expectedFromNativeCalls
        )
        #expect(orchestrationMetrics.maximumError <= 1e-6)
        let finalMetrics = try compareFixtureValues(
            actual: actualValues, expected: expectedOutput
        )
        print(
            "sparse sampler production: max=\(finalMetrics.maximumError) " +
                "rms=\(finalMetrics.rms) normalized_rms=\(finalMetrics.normalizedRMS) " +
                "max_scale_ratio=\(finalMetrics.maximumScaleRatio) " +
                "elapsed=\(elapsed) arena_peak=\(memory.peakUsedBytes)"
        )
        // CFG multiplies the two independently bounded model errors by 7.5
        // and -6.5 before rescaling. Absolute bounds are meaningful here;
        // normalized error is unstable because the one-step result is close
        // to zero after subtracting the guided velocity from the input noise.
        #expect(finalMetrics.maximumError <= 0.25)
        #expect(finalMetrics.rms <= 0.05)
    }

    @Test("production sparse CFG arithmetic matches the one-step MPS oracle")
    func sparseStructureCFGProductionGolden() throws {
        let fixtureURL = try #require(Bundle.module.url(
            forResource: "ss-sampler-r16-1step-mps", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(fileSHA256(at: fixtureURL) ==
            "9c7cd37e4c3abe1575ba93b1afbb3bee12e8d5295d8905a113bd13a2f0b50936")
        let fixture = try Data(contentsOf: fixtureURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        let latentCount = 16 * 16 * 16 * 8
        let contextCount = 1029 * 1024
        let traceStart = latentCount + contextCount
        let noise = Array(fixture[..<latentCount])
        let positive = Array(fixture[traceStart..<(traceStart + latentCount)])
        let negative = Array(
            fixture[(traceStart + latentCount)..<(traceStart + latentCount * 2)]
        )
        let expected = Array(fixture[(traceStart + latentCount * 2)...])
        let actual = oneStepSparseSamplerReference(
            noise: noise, positive: positive, negative: negative
        )
        let metrics = try compareFixtureValues(actual: actual, expected: expected)
        print(
            "sparse CFG oracle-only: max=\(metrics.maximumError) rms=\(metrics.rms) " +
                "normalized_rms=\(metrics.normalizedRMS) " +
                "max_scale_ratio=\(metrics.maximumScaleRatio)"
        )
        #expect(metrics.maximumError <= 1e-5)
        #expect(metrics.rms <= 1e-6)
    }

    @Test(
        "full 12-step sparse trajectory hands off to the production decoder",
        .enabled(
            if: ProcessInfo.processInfo.environment[
                "KG_TRELLIS2_SPARSE_STRUCTURE_FLOW_CHECKPOINT"
            ] != nil && ProcessInfo.processInfo.environment[
                "KG_TRELLIS2_SPARSE_STRUCTURE_DECODER_CHECKPOINT"
            ] != nil,
            "Set both sparse-structure checkpoint variables for full trajectory conformance"
        )
    )
    func realSparseStructureFullTrajectoryAndHandoffGolden() throws {
        let flowPath = try #require(ProcessInfo.processInfo.environment[
            "KG_TRELLIS2_SPARSE_STRUCTURE_FLOW_CHECKPOINT"
        ])
        let decoderPath = try #require(ProcessInfo.processInfo.environment[
            "KG_TRELLIS2_SPARSE_STRUCTURE_DECODER_CHECKPOINT"
        ])
        let fixtureURL = try #require(Bundle.module.url(
            forResource: "ss-full-r16-12step-mps", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(
            fileSHA256(at: fixtureURL) ==
                "8bf56697d1c2cd76758d36b92350dfa4ef305e86e75e8344784327a6a9143fd5"
        )
        let metadataURL = fixtureURL.appendingPathExtension("json")
        try #require(
            fileSHA256(at: metadataURL) ==
                "fde7f7bdd9e685a76b5124b565e2f2785bfb13aa38acf21c09b2ed8009bcaa86"
        )
        let metadata = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL))
                as? [String: Any]
        )
        #expect(metadata["source_revision"] as? String ==
            "75fbf0183001ed9876c8dbb35de6b68552ee08bd")
        #expect(metadata["flow_weight_revision"] as? String ==
            "af44b45f2e35a493886929c6d786e563ec68364d")
        #expect(metadata["flow_weight_sha256"] as? String ==
            "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6")
        #expect(metadata["decoder_weight_revision"] as? String ==
            "25e0d31ffbebe4b5a97464dd851910efc3002d96")
        #expect(metadata["decoder_weight_sha256"] as? String ==
            "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a")
        #expect(metadata["steps"] as? Int == 12)
        #expect(metadata["model_calls"] as? Int == 22)
        #expect(metadata["occupancy_count"] as? Int == 91_584)
        let sourceFiles = try #require(metadata["source_files"] as? [String: String])
        #expect(sourceFiles == [
            "trellis2/models/sparse_structure_flow.py":
                "52182c80687e6544d97fc0fcbede738503623376ba29349004d7cc796c6b8efa",
            "trellis2/models/sparse_structure_vae.py":
                "d32e4490b2f5356b72229ee7c8fb25702238b4f0877c49f1cae15ddc9ca615bc",
            "trellis2/pipelines/samplers/flow_euler.py":
                "b4bd235874adfc47fd3bce3d596249b1ccfa6644983a9b4562c9295a463bc0fd",
            "trellis2/pipelines/samplers/classifier_free_guidance_mixin.py":
                "1780182cd7d3c7af3f82b7904aa9c40eb3f632064d77456565c5acccd9598bee",
            "trellis2/pipelines/samplers/guidance_interval_mixin.py":
                "633f97c48a811a835d3b894b3e0de794407f60774f6c60a2c6a61e7c0351c6f2",
        ])
        let fixture = try Data(contentsOf: fixtureURL)
        func range(_ name: String) throws -> [Int] {
            let value = try #require(metadata[name] as? [Int])
            try #require(value.count == 2 && value[0] >= 0 && value[1] >= value[0])
            try #require(value[1] <= fixture.count)
            return value
        }
        func floats(_ name: String) throws -> [Float] {
            let value = try range(name)
            try #require((value[1] - value[0]).isMultiple(of: MemoryLayout<Float>.stride))
            return fixture.subdata(in: value[0]..<value[1]).withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }
        }
        let tokens = 16 * 16 * 16
        let latentCount = tokens * 8
        let contextTokens = 1029
        var noise = try floats("input_range")
        var positive = try floats("context_range")
        var negative = [Float](repeating: 0, count: positive.count)
        let expectedStates = try floats("state_trace_range")
        let expectedFinal = try floats("final_latent_range")
        let expectedLogits = try floats("logits_range")
        try #require(noise.count == latentCount)
        try #require(positive.count == contextTokens * 1024)
        try #require(expectedStates.count == 12 * latentCount)
        try #require(expectedFinal.count == latentCount)
        try #require(expectedLogits.count == 64 * 64 * 64)
        #expect(Array(expectedStates.suffix(latentCount)) == expectedFinal)

        let cacheCrossKV = ProcessInfo.processInfo.environment[
            "KG_TRELLIS2_ENABLE_CROSS_KV_CACHE"
        ] == "1"
        var flowLifecycle: [StageLifecycleEvent] = []
        let flowSession = try StageSession(
            checkpointURL: URL(fileURLWithPath: flowPath),
            expectedCheckpointSHA256:
                "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6",
            arenaCapacity: (cacheCrossKV ? 1536 : 768) * 1024 * 1024,
            lifecycleObserver: { flowLifecycle.append($0) }
        )
        let noiseBuffer = try #require(flowSession.device.makeBuffer(
            bytes: &noise, length: noise.count * 4, options: .storageModeShared
        ))
        let positiveBuffer = try #require(flowSession.device.makeBuffer(
            bytes: &positive, length: positive.count * 4, options: .storageModeShared
        ))
        let negativeBuffer = try #require(flowSession.device.makeBuffer(
            bytes: &negative, length: negative.count * 4, options: .storageModeShared
        ))
        var nativeStates: [[Float]] = []
        let flowStarted = ContinuousClock.now
        let sample = try flowSession.sampleSparseStructureF32(
            noise: noiseBuffer,
            positiveConditioning: positiveBuffer,
            negativeConditioning: negativeBuffer,
            conditioningTokens: contextTokens,
            parameters: .sparseStructure512(),
            cacheCrossKV: cacheCrossKV,
            samplerTrace: { step, state in
                #expect(step == nativeStates.count)
                nativeStates.append(state)
            }
        )
        let flowElapsed = flowStarted.duration(to: .now)
        #expect(sample.modelCallCount == 22)
        #expect(sample.crossKVCacheStats == (cacheCrossKV
            ? SLatCrossKVCacheStats(hits: 600, misses: 60)
            : SLatCrossKVCacheStats(hits: 0, misses: 0)))
        try #require(nativeStates.count == 12)
        let flowMemory = try flowSession.close()
        #expect(flowMemory.usedBytes == 0)
        #expect(flowMemory.peakUsedBytes <= (cacheCrossKV ? 1400 : 600) * 1024 * 1024)
        #expect(flowLifecycle == [.queueDrained, .arenaReleased, .checkpointUnmapped])
        for step in 0..<12 {
            let expected = Array(
                expectedStates[(step * latentCount)..<((step + 1) * latentCount)]
            )
            let metrics = try compareFixtureValues(
                actual: nativeStates[step], expected: expected
            )
            print(
                "sparse full step=\(step) max=\(metrics.maximumError) " +
                    "rms=\(metrics.rms) normalized_rms=\(metrics.normalizedRMS) " +
                    "max_scale_ratio=\(metrics.maximumScaleRatio)"
            )
            if step <= 4 {
                #expect(metrics.normalizedRMS <= 0.10)
                #expect(metrics.maximumScaleRatio <= 0.20)
            }
        }
        let nativeFinalPointer = sample.latent.contents().assumingMemoryBound(to: Float.self)
        let nativeFinal = (0..<latentCount).map { nativeFinalPointer[$0] }
        let finalMetrics = try compareFixtureValues(
            actual: nativeFinal, expected: expectedFinal
        )

        var decoderLifecycle: [StageLifecycleEvent] = []
        let decoderSession = try StageSession(
            checkpointURL: URL(fileURLWithPath: decoderPath),
            expectedCheckpointSHA256:
                "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a",
            arenaCapacity: 512 * 1024 * 1024,
            lifecycleObserver: { decoderLifecycle.append($0) }
        )
        let decoderStarted = ContinuousClock.now
        let decoded = try decoderSession.decodeSparseStructureF32(latent: sample.latent)
        let decoderElapsed = decoderStarted.duration(to: .now)
        let decoderMemory = try decoderSession.close()
        #expect(decoderMemory.usedBytes == 0)
        #expect(decoderMemory.peakUsedBytes <= 256 * 1024 * 1024)
        #expect(decoderLifecycle == [.queueDrained, .arenaReleased, .checkpointUnmapped])
        let actualLogitsPointer = decoded.logits.contents().assumingMemoryBound(to: Float.self)
        let actualLogits = (0..<expectedLogits.count).map { actualLogitsPointer[$0] }
        let logitsMetrics = try compareFixtureValues(
            actual: actualLogits, expected: expectedLogits
        )
        let occupancyRange = try range("occupancy_range")
        let expectedOccupancy = try SparseOccupancyGrid(
            resolution: 64,
            packedBits: Array(fixture[occupancyRange[0]..<occupancyRange[1]])
        )
        var intersection = 0
        var union = 0
        var actualOccupancyCount = 0
        for index in expectedOccupancy.packedBits.indices {
            intersection += Int(
                expectedOccupancy.packedBits[index]
                    & decoded.occupancy.packedBits[index]
            ).nonzeroBitCount
            union += Int(
                expectedOccupancy.packedBits[index]
                    | decoded.occupancy.packedBits[index]
            ).nonzeroBitCount
            actualOccupancyCount += Int(
                decoded.occupancy.packedBits[index]
            ).nonzeroBitCount
        }
        let occupancyIOU = Double(intersection) / Double(union)
        let occupancyCountRatio = Double(actualOccupancyCount) / 91_584
        print(
            "sparse full handoff: calls=\(sample.modelCallCount) " +
                "flow_elapsed=\(flowElapsed) decoder_elapsed=\(decoderElapsed) " +
                "final_normalized_rms=\(finalMetrics.normalizedRMS) " +
                "logits_normalized_rms=\(logitsMetrics.normalizedRMS) " +
                "occupancy_iou=\(occupancyIOU) occupancy_count=\(actualOccupancyCount) " +
                "occupancy_count_ratio=\(occupancyCountRatio) " +
                "flow_peak=\(flowMemory.peakUsedBytes) " +
                "decoder_peak=\(decoderMemory.peakUsedBytes)"
        )
        // The teacher-forced test bounds the model itself at every sampled
        // trajectory region. This separate gate is a same-seed structural
        // stability sentinel after 22 BF16 calls, where feedback makes exact
        // cross-backend tensors and occupancy non-invariant.
        #expect(occupancyIOU >= 0.70)
        #expect(occupancyCountRatio >= 0.70 && occupancyCountRatio <= 1.30)
        #expect(!decoded.coordinates.isEmpty)
    }

    @Test(
        "sparse flow remains conformant on teacher-forced late trajectory states",
        .enabled(
            if: ProcessInfo.processInfo.environment[
                "KG_TRELLIS2_SPARSE_STRUCTURE_FLOW_CHECKPOINT"
            ] != nil,
            "Set KG_TRELLIS2_SPARSE_STRUCTURE_FLOW_CHECKPOINT for teacher-forced conformance"
        )
    )
    func realSparseStructureTeacherForcedTrajectoryGolden() throws {
        let path = try #require(ProcessInfo.processInfo.environment[
            "KG_TRELLIS2_SPARSE_STRUCTURE_FLOW_CHECKPOINT"
        ])
        let fixtureURL = try #require(Bundle.module.url(
            forResource: "ss-full-r16-12step-mps", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(fileSHA256(at: fixtureURL) ==
            "8bf56697d1c2cd76758d36b92350dfa4ef305e86e75e8344784327a6a9143fd5")
        let metadataURL = fixtureURL.appendingPathExtension("json")
        try #require(fileSHA256(at: metadataURL) ==
            "fde7f7bdd9e685a76b5124b565e2f2785bfb13aa38acf21c09b2ed8009bcaa86")
        let metadata = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL))
                as? [String: Any]
        )
        let fixture = try Data(contentsOf: fixtureURL)
        func floats(_ name: String) throws -> [Float] {
            let range = try #require(metadata[name] as? [Int])
            try #require(range.count == 2 && range[0] >= 0 && range[1] <= fixture.count)
            return fixture.subdata(in: range[0]..<range[1]).withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }
        }
        let tokens = 16 * 16 * 16
        let latentCount = tokens * 8
        let contextTokens = 1029
        let noise = try floats("input_range")
        let conditioning = try floats("context_range")
        let states = try floats("state_trace_range")
        let modelOutputs = try floats("model_trace_range")
        let timesteps = try #require(metadata["model_timesteps"] as? [Double])
        try #require(states.count == 12 * latentCount)
        try #require(modelOutputs.count == 22 * latentCount)
        try #require(timesteps.count == 22)
        let context = try MetalContext()
        let checkpoint = try MappedCheckpoint(
            url: URL(fileURLWithPath: path), device: context.device
        )
        try #require(checkpoint.sha256() ==
            "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6")
        var conditioningValues = conditioning
        let conditioningBuffer = try #require(context.device.makeBuffer(
            bytes: &conditioningValues, length: conditioningValues.count * 4,
            options: .storageModeShared
        ))
        var coordinates: [Int32] = []
        coordinates.reserveCapacity(tokens * 4)
        for x in 0..<16 {
            for y in 0..<16 {
                for z in 0..<16 {
                    coordinates.append(contentsOf: [0, Int32(x), Int32(y), Int32(z)])
                }
            }
        }
        let coordinateBuffer = try #require(context.device.makeBuffer(
            bytes: &coordinates, length: coordinates.count * 4,
            options: .storageModeShared
        ))
        let flow = try SLatFlow(context: context, configuration: .sparseStructure)
        let probes: [(call: Int, precedingState: Int?)] = [
            (0, nil), (8, 3), (16, 7), (21, 10),
        ]
        for probe in probes {
            var inputValues = probe.precedingState.map { step in
                Array(states[(step * latentCount)..<((step + 1) * latentCount)])
            } ?? noise
            var timestep = Float(timesteps[probe.call])
            let inputBuffer = try #require(context.device.makeBuffer(
                bytes: &inputValues, length: inputValues.count * 4,
                options: .storageModeShared
            ))
            let timestepBuffer = try #require(context.device.makeBuffer(
                bytes: &timestep, length: 4, options: .storageModeShared
            ))
            let output = try flow.forwardF32(
                input: inputBuffer, timestep: timestepBuffer,
                conditioning: conditioningBuffer, coordinates: coordinateBuffer,
                checkpoint: checkpoint, tokens: tokens,
                conditioningTokens: contextTokens
            )
            let pointer = output.contents().assumingMemoryBound(to: Float.self)
            let actual = (0..<latentCount).map { pointer[$0] }
            let expected = Array(
                modelOutputs[(probe.call * latentCount)..<((probe.call + 1) * latentCount)]
            )
            let metrics = try compareFixtureValues(actual: actual, expected: expected)
            print(
                "sparse teacher call=\(probe.call) timestep=\(timestep) " +
                    "max=\(metrics.maximumError) rms=\(metrics.rms) " +
                    "normalized_rms=\(metrics.normalizedRMS) " +
                    "max_scale_ratio=\(metrics.maximumScaleRatio)"
            )
            // Each probe supplies the exact oracle state, separating model
            // conformance from feedback sensitivity across denoising steps.
            #expect(metrics.normalizedRMS <= 0.02)
            #expect(metrics.maximumScaleRatio <= 0.05)
        }
    }

    @Test(
        "real sparse-structure decoder executes every pinned weight on Metal",
        .enabled(
            if: ProcessInfo.processInfo.environment[
                "KG_TRELLIS2_SPARSE_STRUCTURE_DECODER_CHECKPOINT"
            ] != nil,
            "Set KG_TRELLIS2_SPARSE_STRUCTURE_DECODER_CHECKPOINT for decoder conformance"
        )
    )
    func realSparseStructureDecoderGolden() throws {
        let path = try #require(ProcessInfo.processInfo.environment[
            "KG_TRELLIS2_SPARSE_STRUCTURE_DECODER_CHECKPOINT"
        ])
        let metrics = try verifySparseStructureDecoderFixture(
            checkpointPath: path,
            fixtureName: "ss-decoder-r2",
            fixtureSHA256:
                "e5fb37ddf9086981afec269c55f53ca6dac76a182c0adcb1edd4fe2dd6b0f8b3",
            metadataSHA256:
                "ec1219d2162c6471d87e4a37bc5c69feebf34845095372c718e7ddd7bb3c1af3",
            inputResolution: 2,
            arenaCapacity: 64 * 1024 * 1024
        )
        print(
            "sparse decoder r2: max=\(metrics.maximumError) rms=\(metrics.rms) " +
                "normalized_rms=\(metrics.normalizedRMS) " +
                "mixed_ratio=\(metrics.maximumMixedToleranceRatio) " +
                "arena_peak=\(metrics.arenaPeakBytes)"
        )
        #expect(metrics.maximumMixedToleranceRatio <= 1)
        #expect(metrics.normalizedRMS <= 0.001)
    }

    @Test(
        "production sparse-structure decoder executes 16-to-64 on Metal",
        .enabled(
            if: ProcessInfo.processInfo.environment[
                "KG_TRELLIS2_SPARSE_STRUCTURE_DECODER_CHECKPOINT"
            ] != nil,
            "Set KG_TRELLIS2_SPARSE_STRUCTURE_DECODER_CHECKPOINT for decoder conformance"
        )
    )
    func realSparseStructureDecoderProductionGolden() throws {
        let path = try #require(ProcessInfo.processInfo.environment[
            "KG_TRELLIS2_SPARSE_STRUCTURE_DECODER_CHECKPOINT"
        ])
        let metrics = try verifySparseStructureDecoderFixture(
            checkpointPath: path,
            fixtureName: "ss-decoder-r16-mps",
            fixtureSHA256:
                "904b84bf376e2cf09c1a0bb483e6d9776aeb6a7d0af9523257ceb20d2cd18b2e",
            metadataSHA256:
                "5ab8d1d09fa2f8ae4185e3eb530c40bec840d9d79a03c6d14d08a074bc19fe88",
            inputResolution: 16,
            arenaCapacity: 512 * 1024 * 1024
        )
        print(
            "sparse decoder r16: max=\(metrics.maximumError) rms=\(metrics.rms) " +
                "normalized_rms=\(metrics.normalizedRMS) " +
                "mixed_ratio=\(metrics.maximumMixedToleranceRatio) " +
                "arena_peak=\(metrics.arenaPeakBytes)"
        )
        // The relative term is one F16 ULP; the absolute floor covers
        // cancellation across ten residual/Conv3D blocks near zero.
        #expect(metrics.maximumMixedToleranceRatio <= 1)
        #expect(metrics.normalizedRMS <= 0.001)
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

private struct SparseStructureDecoderMetrics {
    let maximumError: Float
    let rms: Double
    let normalizedRMS: Double
    let maximumMixedToleranceRatio: Float
    let arenaPeakBytes: Int
}

private struct FixtureComparisonMetrics {
    let maximumError: Float
    let rms: Double
    let normalizedRMS: Double
    let maximumScaleRatio: Float
}

private func compareFixtureValues(
    actual: [Float], expected: [Float]
) throws -> FixtureComparisonMetrics {
    try #require(actual.count == expected.count && !expected.isEmpty)
    var maximumError: Float = 0
    var maximumMagnitude: Float = 0
    var squaredError: Double = 0
    var expectedSquaredMagnitude: Double = 0
    for index in expected.indices {
        try #require(actual[index].isFinite)
        let error = abs(actual[index] - expected[index])
        maximumError = max(maximumError, error)
        maximumMagnitude = max(maximumMagnitude, abs(expected[index]))
        squaredError += Double(error * error)
        expectedSquaredMagnitude += Double(expected[index] * expected[index])
    }
    let rms = sqrt(squaredError / Double(expected.count))
    let expectedRMS = sqrt(expectedSquaredMagnitude / Double(expected.count))
    return FixtureComparisonMetrics(
        maximumError: maximumError,
        rms: rms,
        normalizedRMS: rms / max(expectedRMS, 1e-12),
        maximumScaleRatio: maximumError / max(maximumMagnitude, 1e-12)
    )
}

private func oneStepSparseSamplerReference(
    noise: [Float], positive: [Float], negative: [Float]
) -> [Float] {
    precondition(noise.count == positive.count && positive.count == negative.count)
    let stateScale: Float = 1 - 1e-5
    let sigmaScale: Float = 1
    let strength: Float = 7.5
    var guided = [Float](repeating: 0, count: noise.count)
    var positiveSum: Float = 0
    var positiveSquareSum: Float = 0
    var guidedSum: Float = 0
    var guidedSquareSum: Float = 0
    for index in noise.indices {
        guided[index] = strength * positive[index] + (1 - strength) * negative[index]
        let positiveX0 = stateScale * noise[index] - sigmaScale * positive[index]
        let guidedX0 = stateScale * noise[index] - sigmaScale * guided[index]
        positiveSum += positiveX0
        positiveSquareSum += positiveX0 * positiveX0
        guidedSum += guidedX0
        guidedSquareSum += guidedX0 * guidedX0
    }
    let count = Float(noise.count)
    let positiveMean = positiveSum / count
    let guidedMean = guidedSum / count
    let positiveStandardDeviation = sqrt(
        positiveSquareSum / count - positiveMean * positiveMean
    )
    let guidedStandardDeviation = sqrt(
        guidedSquareSum / count - guidedMean * guidedMean
    )
    let ratio = positiveStandardDeviation / guidedStandardDeviation
    for index in noise.indices {
        let guidedX0 = stateScale * noise[index] - sigmaScale * guided[index]
        let rescaledX0 = guidedX0 * ratio
        let blendedX0: Float = 0.7 * rescaledX0 + 0.3 * guidedX0
        let velocity = (stateScale * noise[index] - blendedX0) / sigmaScale
        guided[index] = noise[index] - velocity
    }
    return guided
}

private func verifySparseStructureDecoderFixture(
    checkpointPath: String,
    fixtureName: String,
    fixtureSHA256: String,
    metadataSHA256: String,
    inputResolution: Int,
    arenaCapacity: Int
) throws -> SparseStructureDecoderMetrics {
    let fixtureURL = try #require(Bundle.module.url(
        forResource: fixtureName, withExtension: "f32", subdirectory: "Fixtures"
    ))
    try #require(fileSHA256(at: fixtureURL) == fixtureSHA256)
    let metadataURL = fixtureURL.appendingPathExtension("json")
    try #require(fileSHA256(at: metadataURL) == metadataSHA256)
    let metadata = try #require(
        JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL))
            as? [String: Any]
    )
    #expect(metadata["source_revision"] as? String ==
        "75fbf0183001ed9876c8dbb35de6b68552ee08bd")
    #expect(metadata["weight_revision"] as? String ==
        "25e0d31ffbebe4b5a97464dd851910efc3002d96")
    #expect(metadata["weight_sha256"] as? String ==
        "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a")
    #expect(metadata["payload_sha256"] as? String == fixtureSHA256)
    #expect(metadata["input_resolution"] as? Int == inputResolution)
    #expect(metadata["output_resolution"] as? Int == inputResolution * 4)
    let fixture = try Data(contentsOf: fixtureURL).withUnsafeBytes {
        Array($0.bindMemory(to: Float.self))
    }
    let inputVoxels = inputResolution * inputResolution * inputResolution
    let outputResolution = inputResolution * 4
    let outputVoxels = outputResolution * outputResolution * outputResolution
    let inputCount = inputVoxels * 8
    try #require(fixture.count == inputCount + outputVoxels)

    var inputValues = Array(fixture[..<inputCount])
    let expectedValues = Array(fixture[inputCount...])
    var lifecycle: [StageLifecycleEvent] = []
    let session = try StageSession(
        checkpointURL: URL(fileURLWithPath: checkpointPath),
        expectedCheckpointSHA256:
            "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a",
        arenaCapacity: arenaCapacity,
        lifecycleObserver: { lifecycle.append($0) }
    )
    let input = try #require(session.device.makeBuffer(
        bytes: &inputValues,
        length: inputValues.count * MemoryLayout<Float>.stride,
        options: .storageModeShared
    ))
    let result = try session.decodeSparseStructureF32(
        latent: input, inputResolution: inputResolution
    )
    let standalone = result.logits
    let expectedOccupancy = try SparseStructureOccupancy.threshold(
        logits: expectedValues, resolution: outputResolution
    )
    #expect(result.occupancy == expectedOccupancy)
    #expect(
        result.pooledOccupancy ==
            (try SparseStructureOccupancy.downsampleMax2(expectedOccupancy))
    )
    #expect(result.coordinates == result.pooledOccupancy.coordinates())
    #expect(result.highResolutionCoordinates == result.occupancy.coordinates())
    let memory = try session.close()
    #expect(memory.usedBytes == 0)
    #expect(lifecycle == [.queueDrained, .arenaReleased, .checkpointUnmapped])

    let actual = standalone.contents().assumingMemoryBound(to: Float.self)
    var maximumError: Float = 0
    var maximumMixedToleranceRatio: Float = 0
    var squaredError: Double = 0
    var expectedSquaredMagnitude: Double = 0
    for index in expectedValues.indices {
        try #require(actual[index].isFinite)
        let expected = expectedValues[index]
        let error = abs(actual[index] - expected)
        maximumError = max(maximumError, error)
        maximumMixedToleranceRatio = max(
            maximumMixedToleranceRatio,
            error / (0.01 + 0.001 * abs(expected))
        )
        squaredError += Double(error * error)
        expectedSquaredMagnitude += Double(expected * expected)
    }
    let rms = sqrt(squaredError / Double(outputVoxels))
    let expectedRMS = sqrt(expectedSquaredMagnitude / Double(outputVoxels))
    return SparseStructureDecoderMetrics(
        maximumError: maximumError,
        rms: rms,
        normalizedRMS: rms / max(expectedRMS, 1e-12),
        maximumMixedToleranceRatio: maximumMixedToleranceRatio,
        arenaPeakBytes: memory.peakUsedBytes
    )
}

private func bf16(_ value: Float) -> UInt16 {
    UInt16(truncatingIfNeeded: value.bitPattern >> 16)
}

private enum StageTestFailure: Error {
    case expected
}

private func makeStageTestCheckpoint(weight: Float) throws -> URL {
    var header = try JSONSerialization.data(withJSONObject: [
        "weight": ["dtype": "F32", "shape": [1, 1], "data_offsets": [0, 4]],
    ], options: [.sortedKeys])
    while header.count % 8 != 0 { header.append(0x20) }
    var length = UInt64(header.count).littleEndian
    var data = withUnsafeBytes(of: &length) { Data($0) }
    data.append(header)
    var weight = weight
    data.append(withUnsafeBytes(of: &weight) { Data($0) })
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("kg-stage-\(UUID().uuidString).safetensors")
    try data.write(to: url, options: .atomic)
    return url
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
