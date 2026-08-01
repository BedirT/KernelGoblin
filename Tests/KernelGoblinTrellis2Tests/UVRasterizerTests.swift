import Testing
@testable import KernelGoblinTrellis2

@Suite("Native TRELLIS.2 UV rasterizer")
struct UVRasterizerTests {
    @Test("physical Metal matches the analytic triangle contract in both windings")
    func triangleDifferential() throws {
        let context = try MetalContext()
        let rasterizer = try UVRasterizer(context: context)
        let positions = [
            SIMD3<Float>(0, 0, 0),
            SIMD3<Float>(1, 0, 0),
            SIMD3<Float>(0, 1, 1),
        ]
        let uvs = [
            SIMD2<Float>(0.125, 0.125),
            SIMD2<Float>(0.875, 0.125),
            SIMD2<Float>(0.125, 0.875),
        ]
        for faces in [
            [SIMD3<UInt32>(0, 1, 2)],
            [SIMD3<UInt32>(2, 1, 0)],
        ] {
            let actual = try rasterizer.rasterize(
                positions: positions, uvs: uvs, faces: faces, width: 8, height: 8
            )
            let expected = uvReference(
                positions: positions, uvs: uvs, faces: faces, width: 8, height: 8
            )
            let faceIDs = actual.faceIDs.contents().assumingMemoryBound(to: UInt32.self)
            let sampled = actual.positions.contents().assumingMemoryBound(to: Float.self)
            let mask = actual.mask.contents().assumingMemoryBound(to: UInt8.self)
            for pixel in 0..<64 {
                #expect(faceIDs[pixel] == expected.faceIDs[pixel])
                #expect(mask[pixel] == (expected.faceIDs[pixel] == 0 ? 0 : 1))
                guard expected.faceIDs[pixel] != 0 else { continue }
                for component in 0..<3 {
                    #expect(abs(sampled[pixel * 3 + component]
                                - expected.positions[pixel * 3 + component]) <= 2e-6)
                }
            }
        }
    }

    @Test("shared edges are crack-free and degenerate triangles cover nothing")
    func edgeAndDegenerateCoverage() throws {
        let context = try MetalContext()
        let rasterizer = try UVRasterizer(context: context)
        let quad = try rasterizer.rasterize(
            positions: [
                SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 0, 0),
                SIMD3<Float>(1, 1, 0), SIMD3<Float>(0, 1, 0),
            ],
            uvs: [
                SIMD2<Float>(0, 0), SIMD2<Float>(1, 0),
                SIMD2<Float>(1, 1), SIMD2<Float>(0, 1),
            ],
            faces: [SIMD3<UInt32>(0, 1, 2), SIMD3<UInt32>(0, 2, 3)],
            width: 16, height: 16
        )
        let quadMask = quad.mask.contents().assumingMemoryBound(to: UInt8.self)
        for pixel in 0..<(16 * 16) { #expect(quadMask[pixel] == 1) }

        let degenerate = try rasterizer.rasterize(
            positions: [
                SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 0, 0), SIMD3<Float>(2, 0, 0),
            ],
            uvs: [
                SIMD2<Float>(0.1, 0.1), SIMD2<Float>(0.5, 0.5), SIMD2<Float>(0.9, 0.9),
            ],
            faces: [SIMD3<UInt32>(0, 1, 2)], width: 8, height: 8
        )
        let degenerateMask = degenerate.mask.contents().assumingMemoryBound(to: UInt8.self)
        for pixel in 0..<64 { #expect(degenerateMask[pixel] == 0) }
    }

    @Test("empty meshes are zeroed and invalid geometry is rejected")
    func emptyAndInvalidInputs() throws {
        let context = try MetalContext()
        let rasterizer = try UVRasterizer(context: context)
        let empty = try rasterizer.rasterize(
            positions: [SIMD3<Float>(0, 0, 0)],
            uvs: [SIMD2<Float>(0, 0)], faces: [], width: 3, height: 2
        )
        let mask = empty.mask.contents().assumingMemoryBound(to: UInt8.self)
        let positions = empty.positions.contents().assumingMemoryBound(to: Float.self)
        let ids = empty.faceIDs.contents().assumingMemoryBound(to: UInt32.self)
        for pixel in 0..<6 {
            #expect(mask[pixel] == 0 && ids[pixel] == 0)
            for component in 0..<3 { #expect(positions[pixel * 3 + component] == 0) }
        }
        #expect(throws: NativeRuntimeError.self) {
            _ = try rasterizer.rasterize(
                positions: [SIMD3<Float>(0, 0, 0)], uvs: [SIMD2<Float>(0, 0)],
                faces: [SIMD3<UInt32>(1, 1, 1)], width: 8, height: 8
            )
        }
        #expect(throws: NativeRuntimeError.self) {
            _ = try rasterizer.rasterize(
                positions: [SIMD3<Float>(0, 0, 0)], uvs: [SIMD2<Float>(0, 0)],
                faces: [], width: 0, height: 8
            )
        }
        #expect(throws: NativeRuntimeError.self) {
            _ = try rasterizer.rasterize(
                positions: [SIMD3<Float>(0, 0, 0)], uvs: [SIMD2<Float>(-0.1, 0)],
                faces: [], width: 8, height: 8
            )
        }
    }
}

private struct UVReferenceResult {
    var positions: [Float]
    var faceIDs: [UInt32]
}

private func uvReference(
    positions: [SIMD3<Float>], uvs: [SIMD2<Float>], faces: [SIMD3<UInt32>],
    width: Int, height: Int
) -> UVReferenceResult {
    var result = UVReferenceResult(
        positions: [Float](repeating: 0, count: width * height * 3),
        faceIDs: [UInt32](repeating: 0, count: width * height)
    )
    func edge(_ a: SIMD2<Float>, _ b: SIMD2<Float>, _ point: SIMD2<Float>) -> Float {
        (point.x - a.x) * (b.y - a.y) - (point.y - a.y) * (b.x - a.x)
    }
    for (faceIndex, face) in faces.enumerated() {
        let indices = [Int(face.x), Int(face.y), Int(face.z)]
        let triangle = indices.map { uvs[$0] }
        let area = edge(triangle[0], triangle[1], triangle[2])
        if abs(area) <= Float.ulpOfOne { continue }
        for y in 0..<height {
            for x in 0..<width {
                let point = SIMD2<Float>(
                    (Float(x) + 0.5) / Float(width),
                    (Float(y) + 0.5) / Float(height)
                )
                let w0 = edge(triangle[1], triangle[2], point) / area
                let w1 = edge(triangle[2], triangle[0], point) / area
                let w2 = 1 - w0 - w1
                guard w0 > 1e-6, w1 > 1e-6, w2 > 1e-6 else { continue }
                let pixel = y * width + x
                let interpolated = positions[indices[0]] * w0
                    + positions[indices[1]] * w1 + positions[indices[2]] * w2
                result.positions[pixel * 3] = interpolated.x
                result.positions[pixel * 3 + 1] = interpolated.y
                result.positions[pixel * 3 + 2] = interpolated.z
                result.faceIDs[pixel] = UInt32(faceIndex + 1)
            }
        }
    }
    return result
}
