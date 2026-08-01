import Foundation
import Metal
import Testing
@testable import KernelGoblinTrellis2

@Suite("Native TRELLIS.2 PBR kernels")
struct PBRKernelTests {
    @Test("UV raster and sparse field sampling form a masked object-space bake")
    func integratedSparseTextureBake() throws {
        let context = try MetalContext()
        let baker = try SparsePBRTextureBaker(context: context)
        var coordinates: [SparseStructureCoordinate] = []
        var fields: [Float] = []
        for x in 0..<2 {
            for y in 0..<2 {
                for z in 0..<2 {
                    coordinates.append(SparseStructureCoordinate(
                        x: Int32(x), y: Int32(y), z: Int32(z)
                    ))
                    fields.append(contentsOf: [
                        Float(x), Float(y), Float(z), 0.25, 0.5, 0.75,
                    ])
                }
            }
        }
        let fieldBuffer = try #require(context.device.makeBuffer(
            bytes: &fields, length: fields.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let bake = try baker.bake(
            positions: [
                SIMD3<Float>(0, 0, 0),
                SIMD3<Float>(1, 0, 0),
                SIMD3<Float>(0, 1, 0),
            ],
            uvs: [SIMD2<Float>(0, 0), SIMD2<Float>(1, 0), SIMD2<Float>(0, 1)],
            faces: [SIMD3<UInt32>(0, 1, 2)],
            decodedPBRFields: fieldBuffer, fieldCoordinates: coordinates,
            spatialShape: SparseSpatialShape(cubic: 2),
            aabbMinimum: SIMD3<Float>(repeating: -0.5),
            voxelSize: SIMD3<Float>(repeating: 1), width: 4, height: 4
        )
        let actual = bake.fields.contents().assumingMemoryBound(to: Float.self)
        let mask = bake.mask.contents().assumingMemoryBound(to: UInt8.self)
        var covered = 0
        var uncovered = 0
        for y in 0..<4 {
            for x in 0..<4 {
                let pixel = y * 4 + x
                if mask[pixel] == 0 {
                    uncovered += 1
                    for channel in 0..<6 { #expect(actual[pixel * 6 + channel] == 0) }
                    continue
                }
                covered += 1
                #expect(abs(actual[pixel * 6] - (Float(x) + 0.5) / 4) <= 2e-6)
                #expect(abs(actual[pixel * 6 + 1] - (Float(y) + 0.5) / 4) <= 2e-6)
                #expect(abs(actual[pixel * 6 + 2]) <= 2e-6)
                #expect(abs(actual[pixel * 6 + 3] - 0.25) <= 2e-6)
                #expect(abs(actual[pixel * 6 + 4] - 0.5) <= 2e-6)
                #expect(abs(actual[pixel * 6 + 5] - 0.75) <= 2e-6)
            }
        }
        #expect(covered > 0 && uncovered > 0)
        #expect(throws: NativeRuntimeError.self) {
            _ = try baker.bake(
                positions: [
                    SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 0, 0),
                    SIMD3<Float>(0, 1, 0),
                ],
                uvs: [SIMD2<Float>(0, 0), SIMD2<Float>(1, 0), SIMD2<Float>(0, 1)],
                faces: [SIMD3<UInt32>(0, 1, 2)],
                decodedPBRFields: fieldBuffer, fieldCoordinates: coordinates,
                spatialShape: SparseSpatialShape(cubic: 2),
                aabbMinimum: .zero,
                voxelSize: SIMD3<Float>(repeating: .leastNonzeroMagnitude),
                width: 2, height: 2
            )
        }
    }

    @Test("sparse trilinear PBR sampling preserves half-voxel renormalization")
    func sparseTrilinearSampling() throws {
        let context = try MetalContext(arenaCapacity: 64 * 1024)
        let sampler = try SparsePBRSampler(context: context)
        let coordinates = [
            SparseStructureCoordinate(x: 0, y: 0, z: 0),
            SparseStructureCoordinate(x: 0, y: 0, z: 1),
            SparseStructureCoordinate(x: 0, y: 1, z: 0),
            SparseStructureCoordinate(x: 0, y: 1, z: 1),
            SparseStructureCoordinate(x: 1, y: 0, z: 0),
            SparseStructureCoordinate(x: 1, y: 0, z: 1),
            SparseStructureCoordinate(x: 1, y: 1, z: 0),
            SparseStructureCoordinate(x: 1, y: 1, z: 1),
        ]
        var features: [Float] = coordinates.indices.flatMap {
            [Float($0), Float($0) + 10]
        }
        var positions: [Float] = [
            0.5, 0.5, 0.5,
            1.0, 1.0, 1.0,
            -10, -10, -10,
            .nan, 0.5, 0.5,
            .greatestFiniteMagnitude, 0.5, 0.5,
        ]
        let featureBuffer = try #require(context.device.makeBuffer(
            bytes: &features, length: features.count * 4, options: .storageModeShared
        ))
        let positionBuffer = try #require(context.device.makeBuffer(
            bytes: &positions, length: positions.count * 4, options: .storageModeShared
        ))
        let output = try context.makeBuffer(length: (5 * 2 + 3) * 4, label: "PBR sample test")
        output.contents().assumingMemoryBound(to: Float.self)
            .initialize(repeating: -9191.25, count: 5 * 2 + 3)
        try sampler.sampleTrilinearF32(
            features: featureBuffer, coordinates: coordinates,
            spatialShape: SparseSpatialShape(cubic: 2),
            positions: positionBuffer, queryCount: 5, channels: 2,
            output: output
        )
        let values = output.contents().assumingMemoryBound(to: Float.self)
        let expected: [Float] = [0, 10, 3.5, 13.5, 0, 0]
        for index in expected.indices {
            #expect(abs(values[index] - expected[index]) <= 1e-6)
        }
        #expect(values[6].isNaN && values[7].isNaN)
        #expect(values[8] == 0 && values[9] == 0)
        #expect(values[10] == -9191.25 && values[11] == -9191.25 && values[12] == -9191.25)

        var missingFeatures = Array(features.dropLast(2))
        let missingFeatureBuffer = try #require(context.device.makeBuffer(
            bytes: &missingFeatures, length: missingFeatures.count * 4,
            options: .storageModeShared
        ))
        let missingOutput = try context.makeBuffer(
            length: 2 * 4, label: "PBR missing-neighbor test"
        )
        var midpoint: [Float] = [1, 1, 1]
        let midpointBuffer = try #require(context.device.makeBuffer(
            bytes: &midpoint, length: midpoint.count * 4, options: .storageModeShared
        ))
        try sampler.sampleTrilinearF32(
            features: missingFeatureBuffer,
            coordinates: Array(coordinates.dropLast()),
            spatialShape: SparseSpatialShape(cubic: 2),
            positions: midpointBuffer, queryCount: 1, channels: 2,
            output: missingOutput
        )
        let missing = missingOutput.contents().assumingMemoryBound(to: Float.self)
        #expect(abs(missing[0] - 3) <= 1e-6)
        #expect(abs(missing[1] - 13) <= 1e-6)
        #expect(throws: NativeRuntimeError.self) {
            try sampler.sampleTrilinearF32(
                features: featureBuffer,
                coordinates: [coordinates[0], coordinates[0]],
                spatialShape: SparseSpatialShape(cubic: 2),
                positions: positionBuffer, queryCount: 1, channels: 2,
                output: missingOutput
            )
        }
        let emptyLookup = try sampler.makeLookup(
            coordinates: [], spatialShape: SparseSpatialShape(cubic: 2)
        )
        let emptyOutput = try context.makeBuffer(length: 2 * 4, label: "empty PBR output")
        try sampler.sampleTrilinearF32(
            features: featureBuffer, lookup: emptyLookup,
            positions: midpointBuffer, queryCount: 1, channels: 2,
            output: emptyOutput
        )
        let empty = emptyOutput.contents().assumingMemoryBound(to: Float.self)
        #expect(empty[0] == 0 && empty[1] == 0)
        try sampler.sampleTrilinearF32(
            features: featureBuffer, lookup: emptyLookup,
            positions: midpointBuffer, queryCount: 0, channels: 2,
            output: emptyOutput
        )
        #expect(throws: NativeRuntimeError.self) {
            try sampler.sampleTrilinearF32(
                features: featureBuffer, lookup: emptyLookup,
                positions: midpointBuffer, queryCount: 0, channels: 17,
                output: emptyOutput
            )
        }
        #expect(throws: NativeRuntimeError.self) {
            try sampler.sampleTrilinearF32(
                features: featureBuffer, lookup: emptyLookup,
                positions: midpointBuffer, queryCount: 1, channels: 2,
                output: emptyLookup.buffer
            )
        }
        #expect(throws: NativeRuntimeError.self) {
            _ = try sampler.makeLookup(
                coordinates: [coordinates[0]],
                spatialShape: SparseSpatialShape(
                    width: Int(Int32.max), height: Int(Int32.max),
                    depth: Int(Int32.max)
                )
            )
        }
        #expect(throws: NativeRuntimeError.self) {
            try sampler.sampleTrilinearF32(
                features: featureBuffer, coordinates: coordinates,
                spatialShape: SparseSpatialShape(cubic: 2),
                positions: positionBuffer, queryCount: 1, channels: 17,
                output: missingOutput
            )
        }
    }

    @Test("sparse PBR sampler matches a deterministic six-channel CPU differential")
    func sparseTrilinearDifferential() throws {
        let context = try MetalContext(arenaCapacity: 256 * 1024)
        let sampler = try SparsePBRSampler(context: context)
        let shape = try SparseSpatialShape(cubic: 4)
        var coordinates: [SparseStructureCoordinate] = []
        for x in 0..<4 {
            for y in 0..<4 {
                for z in 0..<4 where (x * 17 + y * 7 + z * 3) % 4 != 0 {
                    coordinates.append(SparseStructureCoordinate(
                        x: Int32(x), y: Int32(y), z: Int32(z)
                    ))
                }
            }
        }
        coordinates.reverse()
        let channels = 6
        var features: [Float] = []
        for index in 0..<(coordinates.count * channels) {
            let first = sin(Double(index) * 0.173) * 0.75
            let second = cos(Double(index) * 0.071) * 0.25
            features.append(Float(first + second))
        }
        let queryCount = 37
        var positions: [Float] = []
        for index in 0..<queryCount {
            positions.append(contentsOf: [
                Float(index % 11) * 0.41 - 0.2,
                Float((index * 5) % 13) * 0.34 - 0.3,
                Float((index * 7) % 17) * 0.27 - 0.4,
            ])
        }
        let featureBuffer = try #require(context.device.makeBuffer(
            bytes: &features, length: features.count * 4, options: .storageModeShared
        ))
        let positionBuffer = try #require(context.device.makeBuffer(
            bytes: &positions, length: positions.count * 4, options: .storageModeShared
        ))
        let output = try context.makeBuffer(
            length: queryCount * channels * 4, label: "PBR differential output"
        )
        try sampler.sampleTrilinearF32(
            features: featureBuffer, coordinates: coordinates, spatialShape: shape,
            positions: positionBuffer, queryCount: queryCount, channels: channels,
            output: output
        )
        let rows = Dictionary(uniqueKeysWithValues: coordinates.enumerated().map { ($1, $0) })
        let actual = output.contents().assumingMemoryBound(to: Float.self)
        let fixtureURL = try #require(Bundle.module.url(
            forResource: "sparse-pbr-sampler", withExtension: "f32",
            subdirectory: "Fixtures"
        ))
        try #require(fileSHA256(at: fixtureURL) ==
            "c08285dc3ac695b3a9810f1784ef645c187483c0ca3c469cd7e7954d1c2b5f22")
        let metadataURL = fixtureURL.appendingPathExtension("json")
        try #require(fileSHA256(at: metadataURL) ==
            "57ffd3d1a7370542ee91d45eabbdde77eba9fb4286dabffae121bb5f69fc9f7b")
        let metadata = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL))
                as? [String: Any]
        )
        #expect(metadata["source_revision"] as? String ==
            "75fbf0183001ed9876c8dbb35de6b68552ee08bd")
        #expect(metadata["upstream_source_sha256"] as? String ==
            "ef51a1ba0f2748ffb4c265b47d382cee956f23c6a52d0f3587e6d8beccb7e54a")
        #expect(metadata["shim_source_sha256"] as? String ==
            "87a6c7182bbfa5bf9deb713d003c1b972e7acc8e917d63257639c88856d55754")
        let golden = try Data(contentsOf: fixtureURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        try #require(golden.count == queryCount * channels)
        for query in 0..<queryCount {
            let point = SIMD3<Float>(
                positions[query * 3], positions[query * 3 + 1], positions[query * 3 + 2]
            )
            let base = SIMD3<Int32>(
                Int32(floor(point.x - 0.5)),
                Int32(floor(point.y - 0.5)),
                Int32(floor(point.z - 0.5))
            )
            var weights: [(Int, Float)] = []
            for dx in 0...1 {
                for dy in 0...1 {
                    for dz in 0...1 {
                        let coordinate = SparseStructureCoordinate(
                            x: base.x + Int32(dx),
                            y: base.y + Int32(dy),
                            z: base.z + Int32(dz)
                        )
                        guard let row = rows[coordinate] else { continue }
                        let difference = point - SIMD3<Float>(
                            Float(coordinate.x) + 0.5,
                            Float(coordinate.y) + 0.5,
                            Float(coordinate.z) + 0.5
                        )
                        let delta = SIMD3<Float>(
                            abs(difference.x), abs(difference.y), abs(difference.z)
                        )
                        weights.append((row, (1 - delta.x) * (1 - delta.y) * (1 - delta.z)))
                    }
                }
            }
            let denominator = max(weights.reduce(Float.zero) { $0 + $1.1 }, 1e-12)
            for channel in 0..<channels {
                let expected = weights.reduce(Float.zero) {
                    $0 + features[$1.0 * channels + channel] * $1.1
                } / denominator
                #expect(abs(actual[query * channels + channel] - expected) <= 2e-5)
                #expect(abs(actual[query * channels + channel]
                            - golden[query * channels + channel]) <= 2e-5)
            }
        }
    }
}
