import Testing
@testable import KernelGoblinTrellis2

@Suite("Native TRELLIS.2 UV preparation")
struct UVUnwrapTests {
    @Test("deterministic fallback preserves every valid source face")
    func perFaceTopologyPreservation() throws {
        let positions = [
            SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 0, 0),
            SIMD3<Float>(1, 1, 0), SIMD3<Float>(0, 1, 0),
            SIMD3<Float>(0.5, 0.5, 1),
        ]
        let faces = [
            SIMD3<UInt32>(0, 1, 4), SIMD3<UInt32>(1, 2, 4),
            SIMD3<UInt32>(2, 3, 4), SIMD3<UInt32>(3, 0, 4),
            SIMD3<UInt32>(0, 3, 2), SIMD3<UInt32>(0, 2, 1),
        ]
        let first = try PerFaceUVUnwrapper.unwrap(positions: positions, faces: faces)
        let second = try PerFaceUVUnwrapper.unwrap(positions: positions, faces: faces)
        #expect(first == second)
        #expect(first.faces.count == faces.count)
        #expect(first.positions.count == faces.count * 3)
        #expect(first.vertexMap.count == faces.count * 3)
        #expect(Set(first.vertexMap) == Set(0..<UInt32(positions.count)))
        #expect(first.uvs.allSatisfy { (0...1).contains($0.x) && (0...1).contains($0.y) })
    }

    @Test("Model I/O produces deterministic seam topology and normalized UVs")
    func deterministicCubeAtlas() throws {
        let positions = [
            SIMD3<Float>(-1, -1, -1), SIMD3<Float>(1, -1, -1),
            SIMD3<Float>(1, 1, -1), SIMD3<Float>(-1, 1, -1),
            SIMD3<Float>(-1, -1, 1), SIMD3<Float>(1, -1, 1),
            SIMD3<Float>(1, 1, 1), SIMD3<Float>(-1, 1, 1),
        ]
        let faces = [
            SIMD3<UInt32>(0, 2, 1), SIMD3<UInt32>(0, 3, 2),
            SIMD3<UInt32>(4, 5, 6), SIMD3<UInt32>(4, 6, 7),
            SIMD3<UInt32>(0, 1, 5), SIMD3<UInt32>(0, 5, 4),
            SIMD3<UInt32>(2, 3, 7), SIMD3<UInt32>(2, 7, 6),
            SIMD3<UInt32>(0, 4, 7), SIMD3<UInt32>(0, 7, 3),
            SIMD3<UInt32>(1, 2, 6), SIMD3<UInt32>(1, 6, 5),
        ]
        let first = try ModelIOUVUnwrapper.unwrap(positions: positions, faces: faces)
        let second = try ModelIOUVUnwrapper.unwrap(positions: positions, faces: faces)
        #expect(first == second)
        #expect(first.faces.count == faces.count)
        #expect(first.positions.count >= positions.count)
        #expect(first.vertexMap.allSatisfy { Int($0) < positions.count })
        #expect(first.uvs.allSatisfy {
            $0.x.isFinite && $0.y.isFinite
                && $0.x >= 0 && $0.x <= 1 && $0.y >= 0 && $0.y <= 1
        })
        for index in first.positions.indices {
            #expect(first.positions[index] == positions[Int(first.vertexMap[index])])
        }
        let context = try MetalContext()
        let raster = try UVRasterizer(context: context).rasterize(
            positions: first.positions, uvs: first.uvs, faces: first.faces,
            width: 128, height: 128
        )
        let mask = raster.mask.contents().assumingMemoryBound(to: UInt8.self)
        let covered = (0..<(128 * 128)).reduce(0) { $0 + (mask[$1] == 0 ? 0 : 1) }
        #expect(covered > 0 && covered < 128 * 128)
        let prepared = try UVPreparation.prepare(
            positions: positions, faces: faces, suppliedUVs: nil,
            policy: .preserveOrGenerate
        )
        #expect(prepared.atlas == first)
        #expect(prepared.exactUpstreamTier == false)
        #expect(prepared.fingerprintSHA256 == UVPreparation.fingerprint(first))
        #expect(prepared.fingerprintSHA256.count == 64)
    }

    @Test("degenerate faces are removed and invalid meshes are rejected")
    func degenerateAndInvalid() throws {
        let positions = [
            SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0),
        ]
        let atlas = try ModelIOUVUnwrapper.unwrap(
            positions: positions,
            faces: [SIMD3<UInt32>(0, 1, 2), SIMD3<UInt32>(0, 0, 1)]
        )
        #expect(atlas.faces.count == 1)
        let supplied = try UVPreparation.prepare(
            positions: positions, faces: [SIMD3<UInt32>(0, 1, 2)],
            suppliedUVs: [SIMD2<Float>(0, 0), SIMD2<Float>(1, 0), SIMD2<Float>(0, 1)],
            policy: .preserve
        )
        #expect(supplied.exactUpstreamTier)
        #expect(supplied.implementation == "supplied-uv-preservation")
        #expect(throws: NativeRuntimeError.self) {
            _ = try ModelIOUVUnwrapper.unwrap(
                positions: positions, faces: [SIMD3<UInt32>(0, 1, 3)]
            )
        }
    }

    @Test("normalized small triangles are not mistaken for degenerate faces")
    func smallTriangleTopologyPreservation() throws {
        let positions = [
            SIMD3<Float>(0, 0, 0),
            SIMD3<Float>(0.0001, 0, 0),
            SIMD3<Float>(0, 0.0001, 0),
        ]
        let faces = [SIMD3<UInt32>(0, 1, 2)]
        let fallback = try PerFaceUVUnwrapper.unwrap(
            positions: positions, faces: faces
        )
        #expect(fallback.faces.count == faces.count)
        #expect(throws: NativeRuntimeError.self) {
            _ = try ModelIOUVUnwrapper.unwrap(
                positions: positions, faces: faces
            )
        }
        let prepared = try UVPreparation.prepare(
            positions: positions, faces: faces, suppliedUVs: nil,
            policy: .regenerate
        )
        #expect(prepared.atlas.faces.count == faces.count)
        #expect(prepared.implementation == "deterministic-native-per-face-atlas-fallback")
    }
}
