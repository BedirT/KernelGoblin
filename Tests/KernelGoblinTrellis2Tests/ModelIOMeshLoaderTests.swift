import Foundation
import Testing
import simd
@testable import KernelGoblinTrellis2

@Suite("Native Model I/O mesh loading")
struct ModelIOMeshLoaderTests {
    @Test("OBJ positions, triangles, UVs, and TRELLIS normalization survive loading")
    func objRoundTrip() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "kg-mesh-\(UUID().uuidString).obj"
        )
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("""
        v 10 20 30
        v 14 20 30
        v 10 22 30
        vt 0 0
        vt 1 0
        vt 0 1
        f 1/1 2/2 3/3
        """.utf8).write(to: url)
        let loaded = try ModelIOMeshLoader.load(url: url)
        #expect(loaded.positions.count == 3)
        #expect(loaded.faces == [SIMD3<UInt32>(0, 1, 2)])
        #expect(loaded.suppliedUVs?.count == 3)
        let normalized = try loaded.normalizedForTrellis()
        let minimum = normalized.positions.reduce(
            SIMD3<Float>(repeating: .greatestFiniteMagnitude), simd_min
        )
        let maximum = normalized.positions.reduce(
            SIMD3<Float>(repeating: -.greatestFiniteMagnitude), simd_max
        )
        #expect(minimum.x >= -0.5 && maximum.x <= 0.5)
        #expect(minimum.y >= -0.5 && maximum.y <= 0.5)
        #expect(minimum.z >= -0.5 && maximum.z <= 0.5)
        #expect(abs((maximum.x - minimum.x) - 0.99999) < 1e-5)
    }

    @Test("face-varying OBJ UV seams are represented by unique loaded vertices")
    func uvSeams() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "kg-seam-\(UUID().uuidString).obj"
        )
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("""
        v 0 0 0
        v 1 0 0
        v 1 1 0
        v 0 1 0
        vt 0 0
        vt 1 0
        vt 1 1
        vt 0 1
        vt 0.25 0.25
        f 1/1 2/2 3/3
        f 1/5 3/3 4/4
        """.utf8).write(to: url)
        let loaded = try ModelIOMeshLoader.load(url: url)
        #expect(loaded.faces.count == 2)
        #expect(loaded.suppliedUVs != nil)
        let originIndices = loaded.positions.indices.filter {
            loaded.positions[$0] == SIMD3<Float>(0, 0, 0)
        }
        #expect(originIndices.count == 2)
        let originUVs = Set(originIndices.map { loaded.suppliedUVs![$0] })
        #expect(originUVs == Set([SIMD2<Float>(0, 0), SIMD2<Float>(0.25, 0.25)]))
    }
}
