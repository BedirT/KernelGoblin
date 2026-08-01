import simd

public enum PerFaceUVUnwrapper {
    public static func unwrap(
        positions: [SIMD3<Float>], faces: [SIMD3<UInt32>]
    ) throws -> UVAtlasMesh {
        guard !positions.isEmpty, !faces.isEmpty,
              positions.count <= Int(UInt32.max) else {
            throw NativeRuntimeError.invalidArgument("invalid mesh for per-face UV unwrapping")
        }
        let validFaces = try faces.filter { face in
            guard Int(face.x) < positions.count,
                  Int(face.y) < positions.count,
                  Int(face.z) < positions.count else {
                throw NativeRuntimeError.invalidArgument("UV unwrap face is out of range")
            }
            guard face.x != face.y, face.y != face.z, face.z != face.x else { return false }
            let normal = simd_cross(
                positions[Int(face.y)] - positions[Int(face.x)],
                positions[Int(face.z)] - positions[Int(face.x)]
            )
            return simd_length_squared(normal) > 0
        }
        guard !validFaces.isEmpty,
              validFaces.count <= Int(UInt32.max) / 3 else {
            throw NativeRuntimeError.invalidArgument("UV unwrap mesh has no usable faces")
        }
        let cellCount = (validFaces.count + 1) / 2
        let columns = Int(ceil(sqrt(Double(cellCount))))
        let rows = (cellCount + columns - 1) / columns
        let inset: Float = 0.08
        var outputPositions: [SIMD3<Float>] = []
        var outputFaces: [SIMD3<UInt32>] = []
        var outputUVs: [SIMD2<Float>] = []
        var vertexMap: [UInt32] = []
        outputPositions.reserveCapacity(validFaces.count * 3)
        outputFaces.reserveCapacity(validFaces.count)
        outputUVs.reserveCapacity(validFaces.count * 3)
        vertexMap.reserveCapacity(validFaces.count * 3)
        for (index, face) in validFaces.enumerated() {
            let base = UInt32(index * 3)
            outputFaces.append(SIMD3(base, base + 1, base + 2))
            let source = [face.x, face.y, face.z]
            for vertex in source {
                outputPositions.append(positions[Int(vertex)])
                vertexMap.append(vertex)
            }
            let cell = index / 2
            let column = cell % columns
            let row = cell / columns
            let local: [SIMD2<Float>] = index.isMultiple(of: 2)
                ? [SIMD2(inset, inset), SIMD2(1 - inset, inset), SIMD2(inset, 1 - inset)]
                : [SIMD2(1 - inset, 1 - inset), SIMD2(inset, 1 - inset), SIMD2(1 - inset, inset)]
            outputUVs.append(contentsOf: local.map { uv in
                SIMD2(
                    (Float(column) + uv.x) / Float(columns),
                    (Float(row) + uv.y) / Float(rows)
                )
            })
        }
        return try UVAtlasMesh(
            positions: outputPositions, faces: outputFaces,
            uvs: outputUVs, vertexMap: vertexMap
        )
    }
}
