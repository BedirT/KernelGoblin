import Metal

public struct FlexibleDualGridMesh: Equatable, Sendable {
    public let vertices: [SIMD3<Float>]
    public let faces: [SIMD3<UInt32>]
}

public enum FlexibleDualGridMeshExtractor {
    public static func extract(
        coordinates: [SparseStructureCoordinate],
        dualOffsets: MTLBuffer,
        intersections: MTLBuffer,
        splitWeights: MTLBuffer,
        gridSize: SparseSpatialShape,
        aabbMinimum: SIMD3<Float> = SIMD3(repeating: -0.5),
        aabbMaximum: SIMD3<Float> = SIMD3(repeating: 0.5)
    ) throws -> FlexibleDualGridMesh {
        let count = coordinates.count
        let dualBytes = try flexibleDualGridMeshBytes(count, 3, MemoryLayout<Float>.stride)
        let intersectionBytes = try flexibleDualGridMeshBytes(count, 3, 1)
        let splitBytes = try flexibleDualGridMeshBytes(
            count, 1, MemoryLayout<Float>.stride
        )
        guard count <= Int(UInt32.max),
              dualOffsets.storageMode != .private,
              intersections.storageMode != .private,
              splitWeights.storageMode != .private,
              dualOffsets.length >= dualBytes,
              intersections.length >= intersectionBytes,
              splitWeights.length >= splitBytes,
              aabbMaximum.x > aabbMinimum.x,
              aabbMaximum.y > aabbMinimum.y,
              aabbMaximum.z > aabbMinimum.z else {
            throw NativeRuntimeError.invalidArgument("invalid flexible dual-grid fields")
        }
        var lookup: [SparseStructureCoordinate: UInt32] = [:]
        lookup.reserveCapacity(count)
        for (index, coordinate) in coordinates.enumerated() {
            guard coordinate.batch == 0,
                  coordinate.x >= 0, coordinate.x < gridSize.width,
                  coordinate.y >= 0, coordinate.y < gridSize.height,
                  coordinate.z >= 0, coordinate.z < gridSize.depth,
                  lookup.updateValue(UInt32(index), forKey: coordinate) == nil else {
                throw NativeRuntimeError.invalidArgument(
                    "flexible dual-grid coordinates must be unique and single-batch"
                )
            }
        }
        let dual = dualOffsets.contents().assumingMemoryBound(to: Float.self)
        let flags = intersections.contents().assumingMemoryBound(to: UInt8.self)
        let split = splitWeights.contents().assumingMemoryBound(to: Float.self)
        let extent = aabbMaximum - aabbMinimum
        let grid = SIMD3<Float>(
            Float(gridSize.width), Float(gridSize.height), Float(gridSize.depth)
        )
        var vertices: [SIMD3<Float>] = []
        vertices.reserveCapacity(count)
        for (index, coordinate) in coordinates.enumerated() {
            guard dual[index * 3].isFinite,
                  dual[index * 3 + 1].isFinite,
                  dual[index * 3 + 2].isFinite,
                  split[index].isFinite else {
                throw NativeRuntimeError.invalidArgument(
                    "flexible dual-grid fields must be finite"
                )
            }
            let position = SIMD3<Float>(
                Float(coordinate.x) + dual[index * 3],
                Float(coordinate.y) + dual[index * 3 + 1],
                Float(coordinate.z) + dual[index * 3 + 2]
            )
            vertices.append(position / grid * extent + aabbMinimum)
        }
        let axes: [[SIMD3<Int32>]] = [
            [SIMD3(0, 0, 0), SIMD3(0, 0, 1), SIMD3(0, 1, 1), SIMD3(0, 1, 0)],
            [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(1, 0, 1), SIMD3(0, 0, 1)],
            [SIMD3(0, 0, 0), SIMD3(0, 1, 0), SIMD3(1, 1, 0), SIMD3(1, 0, 0)],
        ]
        var faces: [SIMD3<UInt32>] = []
        for (index, coordinate) in coordinates.enumerated() {
            let origin = SIMD3(coordinate.x, coordinate.y, coordinate.z)
            for axis in 0..<3 where flags[index * 3 + axis] != 0 {
                let quad = axes[axis].compactMap { offset -> UInt32? in
                    lookup[SparseStructureCoordinate(
                        batch: coordinate.batch,
                        x: origin.x + offset.x,
                        y: origin.y + offset.y,
                        z: origin.z + offset.z
                    )]
                }
                guard quad.count == 4 else { continue }
                if split[Int(quad[0])] * split[Int(quad[2])]
                    > split[Int(quad[1])] * split[Int(quad[3])] {
                    faces.append(SIMD3(quad[0], quad[1], quad[2]))
                    faces.append(SIMD3(quad[0], quad[2], quad[3]))
                } else {
                    faces.append(SIMD3(quad[0], quad[1], quad[3]))
                    faces.append(SIMD3(quad[3], quad[1], quad[2]))
                }
            }
        }
        return FlexibleDualGridMesh(vertices: vertices, faces: faces)
    }
}

private func flexibleDualGridMeshBytes(
    _ count: Int, _ channels: Int, _ width: Int
) throws -> Int {
    let elements = count.multipliedReportingOverflow(by: channels)
    let bytes = elements.partialValue.multipliedReportingOverflow(by: width)
    guard count >= 0, channels > 0, width > 0, !elements.overflow, !bytes.overflow else {
        throw NativeRuntimeError.invalidArgument("flexible dual-grid size overflows Int")
    }
    return bytes.partialValue
}
