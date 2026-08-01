import Foundation
import Metal
import Testing
@testable import KernelGoblinTrellis2

@Suite("Mesh to sparse shape encoder handoff")
struct MeshShapeEncoderHandoffTests {
    @Test(
        "production mesh handoff matches the pinned O-Voxel and MPS encoder oracle",
        .enabled(
            if: ProcessInfo.processInfo.environment[
                "KG_TRELLIS2_SHAPE_ENCODER_CHECKPOINT"
            ] != nil,
            "Set KG_TRELLIS2_SHAPE_ENCODER_CHECKPOINT for encoder conformance"
        )
    )
    func completeHandoff() throws {
        let checkpointPath = try #require(ProcessInfo.processInfo.environment[
            "KG_TRELLIS2_SHAPE_ENCODER_CHECKPOINT"
        ])
        let fixtureURL = try #require(Bundle.module.url(
            forResource: "mesh-shape-encoder-mps", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(fileSHA256(at: fixtureURL) ==
            "952050a4ffe810aadd224f0bd3ee8045803484c1acb741e563539b6bd45aad55")
        let metadataURL = fixtureURL.appendingPathExtension("json")
        try #require(fileSHA256(at: metadataURL) ==
            "6565dc2e242ad931feb675a749480b87ba4f6429598e355eec8e2b5be26a60ab")
        let metadata = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL))
                as? [String: Any]
        )
        #expect(metadata["source_revision"] as? String ==
            "75fbf0183001ed9876c8dbb35de6b68552ee08bd")
        #expect(metadata["checkpoint_sha256"] as? String ==
            "f37c5ff5b983b68e9946060000f09bc131f3e84318a2c8b7430a81e4b4636c41")
        #expect(metadata["oracle_call"] as? String ==
            "O-Voxel CPU then FlexiDualGridVaeEncoder.forward on MPS")

        let fixture = try Data(contentsOf: fixtureURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        #expect(fixture.count == 107)
        let vertices = stride(from: 0, to: 12, by: 3).map {
            SIMD3<Float>(fixture[$0], fixture[$0 + 1], fixture[$0 + 2])
        }
        let faces = [
            SIMD3<UInt32>(0, 2, 1), SIMD3<UInt32>(0, 1, 3),
            SIMD3<UInt32>(0, 3, 2), SIMD3<UInt32>(1, 2, 3),
        ]
        let voxelization = try FlexibleDualGridVoxelizer.voxelize(
            vertices: vertices, faces: faces,
            gridSize: SIMD3(repeating: 512),
            aabbMinimum: SIMD3(repeating: -0.5),
            aabbMaximum: SIMD3(repeating: 0.5)
        )
        let expectedCoordinates = [
            SIMD3<Int32>(250, 250, 250), SIMD3<Int32>(250, 251, 250),
            SIMD3<Int32>(251, 250, 250), SIMD3<Int32>(251, 251, 250),
            SIMD3<Int32>(250, 250, 251), SIMD3<Int32>(251, 250, 251),
            SIMD3<Int32>(250, 251, 251),
        ]
        #expect(voxelization.coordinates == expectedCoordinates)
        #expect(voxelization.intersections == [
            SIMD3<UInt8>(1, 1, 1), SIMD3<UInt8>(repeating: 0),
            SIMD3<UInt8>(repeating: 0), SIMD3<UInt8>(repeating: 0),
            SIMD3<UInt8>(repeating: 0), SIMD3<UInt8>(repeating: 0),
            SIMD3<UInt8>(repeating: 0),
        ])
        for index in 0..<21 {
            #expect(abs(voxelization.dualVertices[index / 3][index % 3]
                - fixture[12 + index]) <= 2e-5)
        }

        let handoff = try voxelization.shapeEncoderInput(
            spatialShape: SparseSpatialShape(cubic: 512)
        )
        #expect(handoff.coordinates == expectedCoordinates.map {
            SparseStructureCoordinate(x: $0.x, y: $0.y, z: $0.z)
        })
        for index in handoff.values.indices {
            let tolerance: Float = index % 6 < 3 ? 0.01024 : 0
            #expect(abs(handoff.values[index] - fixture[33 + index]) <= tolerance)
        }

        var lifecycle: [StageLifecycleEvent] = []
        var inputValues = handoff.values
        let result = try StageSession.withSession(
            checkpointURL: URL(fileURLWithPath: checkpointPath),
            expectedCheckpointSHA256:
                "f37c5ff5b983b68e9946060000f09bc131f3e84318a2c8b7430a81e4b4636c41",
            arenaCapacity: 512 * 1024 * 1024,
            lifecycleObserver: { lifecycle.append($0) }
        ) { session in
            guard let input = session.device.makeBuffer(
                bytes: &inputValues, length: inputValues.count * 4,
                options: .storageModeShared
            ) else {
                throw NativeRuntimeError.allocationFailed(
                    "could not upload mesh shape encoder fixture"
                )
            }
            return try session.encodeShapeF32(
                input: input, coordinates: handoff.coordinates,
                spatialShape: handoff.spatialShape
            )
        }
        #expect(lifecycle == [.queueDrained, .arenaReleased, .checkpointUnmapped])
        #expect(result.coordinates == [
            SparseStructureCoordinate(x: 15, y: 15, z: 15)
        ])
        #expect(result.subdivisionGuides.count == 4)
        #expect(result.subdivisionGuides[0].coordinates == [
            SparseStructureCoordinate(x: 31, y: 31, z: 31)
        ])
        #expect(result.subdivisionGuides[0].childIndices == [7])
        #expect(result.subdivisionGuides[1].coordinates == [
            SparseStructureCoordinate(x: 62, y: 62, z: 62)
        ])
        #expect(result.subdivisionGuides[1].childIndices == [0])
        #expect(result.subdivisionGuides[2].coordinates == [
            SparseStructureCoordinate(x: 125, y: 125, z: 125)
        ])
        #expect(result.subdivisionGuides[2].childIndices == [7])
        #expect(result.subdivisionGuides[3].coordinates == handoff.coordinates)
        #expect(result.subdivisionGuides[3].parentIndices == [0, 0, 0, 0, 0, 0, 0])
        #expect(result.subdivisionGuides[3].childIndices == [0, 2, 1, 3, 4, 5, 6])
        var squaredError = 0.0
        var squaredExpected = 0.0
        var maximumError: Float = 0
        var maximumMagnitude: Float = 0
        let values = result.latent.contents().assumingMemoryBound(to: Float.self)
        for channel in 0..<32 {
            let reference = fixture[75 + channel]
            let error = abs(values[channel] - reference)
            squaredError += Double(error * error)
            squaredExpected += Double(reference * reference)
            maximumError = max(maximumError, error)
            maximumMagnitude = max(maximumMagnitude, abs(reference))
        }
        let normalizedRMS = sqrt(squaredError / max(squaredExpected, 1e-30))
        let scaleRatio = maximumError / max(maximumMagnitude, 1e-20)
        print("mesh shape encoder: normalized_rms=\(normalizedRMS) scale=\(scaleRatio)")
        #expect(normalizedRMS <= 0.003)
        #expect(scaleRatio <= 0.006)
    }
}
