import Foundation
import Testing
@testable import KernelGoblinTrellis2

private struct HoleFixture: Decodable {
    struct Case: Decodable { let faces: [[UInt32]] }
    let format: String
    let triangleHole: Case
    let quadHole: Case

    enum CodingKeys: String, CodingKey {
        case format
        case triangleHole = "triangle_hole"
        case quadHole = "quad_hole"
    }
}

private func holeFixture() throws -> HoleFixture {
    let url = try #require(Bundle.module.url(
        forResource: "trimesh-fill-holes", withExtension: "json",
        subdirectory: "Fixtures"
    ))
    return try JSONDecoder().decode(HoleFixture.self, from: Data(contentsOf: url))
}

@Suite("Native Trimesh-compatible small-hole repair")
struct MeshHoleFillerTests {
    @Test("missing tetrahedron face is restored with opposite boundary winding")
    func triangleHole() throws {
        let fixture = try holeFixture()
        #expect(fixture.format == "KernelGoblin-trimesh-fill-holes-v1")
        let vertices = [
            SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 0, 0),
            SIMD3<Float>(0, 1, 0), SIMD3<Float>(0, 0, 1),
        ]
        let source = FlexibleDualGridMesh(
            vertices: vertices,
            faces: [SIMD3(0, 3, 1), SIMD3(1, 3, 2), SIMD3(2, 3, 0)]
        )
        let result = try MeshHoleFiller.fillTriangleAndQuadHoles(source)
        #expect(result.addedFaceCount == 1)
        #expect(result.mesh.faces.map { [$0.x, $0.y, $0.z] } == fixture.triangleHole.faces)
        #expect(try MeshHoleFiller.fillTriangleAndQuadHoles(result.mesh).addedFaceCount == 0)
    }

    @Test("quad boundary is triangulated while larger boundaries remain open")
    func quadAndLargeHoles() throws {
        let fixture = try holeFixture()
        let quadVertices = [
            SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 0, 0),
            SIMD3<Float>(1, 1, 0), SIMD3<Float>(0, 1, 0),
            SIMD3<Float>(0.5, 0.5, 1),
        ]
        let quad = FlexibleDualGridMesh(
            vertices: quadVertices,
            faces: [SIMD3(0, 4, 1), SIMD3(1, 4, 2), SIMD3(2, 4, 3), SIMD3(3, 4, 0)]
        )
        let filled = try MeshHoleFiller.fillTriangleAndQuadHoles(quad)
        #expect(filled.addedFaceCount == 2)
        #expect(filled.mesh.faces.map { [$0.x, $0.y, $0.z] } == fixture.quadHole.faces)

        let pentagon = FlexibleDualGridMesh(
            vertices: (0..<5).map { index in
                let angle = Float(index) * 2 * .pi / 5
                return SIMD3<Float>(cos(angle), sin(angle), 0)
            } + [SIMD3<Float>(0, 0, 1)],
            faces: (0..<5).map { index in
                SIMD3(UInt32(index), 5, UInt32((index + 1) % 5))
            }
        )
        #expect(try MeshHoleFiller.fillTriangleAndQuadHoles(pentagon).addedFaceCount == 0)
    }
}
