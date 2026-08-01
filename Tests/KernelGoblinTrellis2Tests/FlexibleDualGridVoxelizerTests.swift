import Foundation
import Testing
@testable import KernelGoblinTrellis2

@Suite("O-Voxel flexible dual-grid voxelizer")
struct FlexibleDualGridVoxelizerTests {
    @Test("matches pinned O-Voxel ordering, flags, and QEF solutions")
    func pinnedOracleDifferential() throws {
        let url = try #require(Bundle.module.url(
            forResource: "flexible-dual-grid-ovoxel", withExtension: "json",
            subdirectory: "Fixtures"
        ))
        #expect(try fileSHA256(at: url) ==
            "97f90ac5c31c5a3217fa8a628d813e11430feea880a045b9644f7fafbf2a50a5")
        let fixture = try JSONDecoder().decode(
            FlexibleDualGridFixture.self, from: Data(contentsOf: url)
        )
        #expect(fixture.upstreamRevision ==
            "75fbf0183001ed9876c8dbb35de6b68552ee08bd")
        #expect(fixture.upstreamSourceSHA256 ==
            "95ebcdec3818539c52504cd4a89f409287afc51b051086b9092e11fc7308063f")
        for testCase in fixture.cases {
            let grid = int3(testCase.gridSize)
            let result = try FlexibleDualGridVoxelizer.voxelize(
                vertices: testCase.vertices.map(float3),
                faces: testCase.faces.map(uint3),
                gridSize: grid,
                aabbMinimum: float3(testCase.aabb[0]),
                aabbMaximum: float3(testCase.aabb[1])
            )
            #expect(result.coordinates == testCase.coords.map(int3),
                    "coordinate mismatch for \(testCase.name)")
            #expect(result.intersections == testCase.intersections.map(uint8x3),
                    "intersection mismatch for \(testCase.name)")
            #expect(result.dualVertices.count == testCase.dualVertices.count)
            for (actual, expected) in zip(
                result.dualVertices, testCase.dualVertices.map(float3)
            ) {
                #expect(abs(actual.x - expected.x) <= 2e-5)
                #expect(abs(actual.y - expected.y) <= 2e-5)
                #expect(abs(actual.z - expected.z) <= 2e-5)
            }
            let offsets = result.dualOffsets()
            #expect(offsets.allSatisfy {
                $0.x >= -2e-4 && $0.x <= 1.0002
                    && $0.y >= -2e-4 && $0.y <= 1.0002
                    && $0.z >= -2e-4 && $0.z <= 1.0002
            })
            let repeated = try FlexibleDualGridVoxelizer.voxelize(
                vertices: testCase.vertices.map(float3),
                faces: testCase.faces.map(uint3), gridSize: grid,
                aabbMinimum: float3(testCase.aabb[0]),
                aabbMaximum: float3(testCase.aabb[1])
            )
            #expect(repeated == result)
        }
    }

    @Test("closed and open meshes retain distinct boundary contracts")
    func analyticTopology() throws {
        let tetrahedron = try FlexibleDualGridVoxelizer.voxelize(
            vertices: [
                SIMD3(0.08, 0.10, 0.12), SIMD3(0.91, 0.17, 0.22),
                SIMD3(0.19, 0.88, 0.25), SIMD3(0.24, 0.31, 0.93),
            ],
            faces: [SIMD3(0, 2, 1), SIMD3(0, 1, 3),
                    SIMD3(0, 3, 2), SIMD3(1, 2, 3)],
            gridSize: SIMD3(7, 6, 5),
            aabbMinimum: .zero, aabbMaximum: SIMD3(repeating: 1)
        )
        #expect(tetrahedron.coordinates.count == 72)
        #expect(Set(tetrahedron.coordinates).count == tetrahedron.coordinates.count)
        let flagSums = tetrahedron.intersections.reduce(SIMD3<Int>(repeating: 0)) {
            $0 &+ SIMD3(Int($1.x), Int($1.y), Int($1.z))
        }
        #expect(flagSums == SIMD3(18, 24, 29))
    }

    @Test("invalid meshes are rejected before QEF accumulation")
    func rejectsInvalidInput() {
        #expect(throws: NativeRuntimeError.self) {
            _ = try FlexibleDualGridVoxelizer.voxelize(
                vertices: [.zero, SIMD3(1, 0, 0), SIMD3(2, 0, 0)],
                faces: [SIMD3(0, 1, 2)], gridSize: SIMD3(repeating: 8),
                aabbMinimum: .zero, aabbMaximum: SIMD3(repeating: 1)
            )
        }
        #expect(throws: NativeRuntimeError.self) {
            _ = try FlexibleDualGridVoxelizer.voxelize(
                vertices: [.zero, SIMD3(1, 0, 0), SIMD3(0, 1, .infinity)],
                faces: [SIMD3(0, 1, 2)], gridSize: SIMD3(repeating: 8),
                aabbMinimum: .zero, aabbMaximum: SIMD3(repeating: 1)
            )
        }
    }

    @Test("voxel-scale triangles are not mistaken for degenerates")
    func acceptsSmallFiniteTriangle() throws {
        let result = try FlexibleDualGridVoxelizer.voxelize(
            vertices: [
                SIMD3(0.1, 0.1, 0.1),
                SIMD3(0.10002, 0.1, 0.1),
                SIMD3(0.1, 0.10002, 0.1),
            ],
            faces: [SIMD3(0, 1, 2)], gridSize: SIMD3(repeating: 100_000),
            aabbMinimum: .zero, aabbMaximum: SIMD3(repeating: 1)
        )
        #expect(!result.coordinates.isEmpty)
        #expect(result.dualVertices.allSatisfy {
            $0.x.isFinite && $0.y.isFinite && $0.z.isFinite
        })
    }
}

private struct FlexibleDualGridFixture: Decodable {
    let upstreamRevision: String
    let upstreamSourceSHA256: String
    let cases: [FlexibleDualGridCase]

    enum CodingKeys: String, CodingKey {
        case upstreamRevision = "upstream_revision"
        case upstreamSourceSHA256 = "upstream_source_sha256"
        case cases
    }
}

private struct FlexibleDualGridCase: Decodable {
    let name: String
    let vertices: [[Float]]
    let faces: [[UInt32]]
    let gridSize: [Int32]
    let aabb: [[Float]]
    let coords: [[Int32]]
    let dualVertices: [[Float]]
    let intersections: [[UInt8]]

    enum CodingKeys: String, CodingKey {
        case name, vertices, faces, aabb, coords, intersections
        case gridSize = "grid_size"
        case dualVertices = "dual_vertices"
    }
}

private func float3(_ values: [Float]) -> SIMD3<Float> {
    SIMD3(values[0], values[1], values[2])
}

private func int3(_ values: [Int32]) -> SIMD3<Int32> {
    SIMD3(values[0], values[1], values[2])
}

private func uint3(_ values: [UInt32]) -> SIMD3<UInt32> {
    SIMD3(values[0], values[1], values[2])
}

private func uint8x3(_ values: [UInt8]) -> SIMD3<UInt8> {
    SIMD3(values[0], values[1], values[2])
}
