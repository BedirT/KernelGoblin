import Metal

public struct SparseSubdivision2x: Sendable {
    public let coordinates: [SparseStructureCoordinate]
    public let parentIndices: [UInt32]
    public let childIndices: [UInt32]

    public init(
        parentCoordinates: [SparseStructureCoordinate], logits: MTLBuffer
    ) throws {
        let count = try sparseSubdivisionProduct(parentCoordinates.count, 8)
        let logitBytes = try sparseSubdivisionProduct(count, MemoryLayout<Float>.stride)
        guard logits.storageMode != .private,
              logits.length >= logitBytes else {
            throw NativeRuntimeError.invalidArgument("invalid sparse subdivision logits")
        }
        let values = logits.contents().assumingMemoryBound(to: Float.self)
        var coordinates: [SparseStructureCoordinate] = []
        var parents: [UInt32] = []
        var children: [UInt32] = []
        coordinates.reserveCapacity(count)
        parents.reserveCapacity(count)
        children.reserveCapacity(count)
        for (parentIndex, parent) in parentCoordinates.enumerated() {
            guard parentIndex <= Int(UInt32.max),
                  parent.batch >= 0,
                  parent.x >= 0,
                  parent.y >= 0,
                  parent.z >= 0,
                  parent.x <= (Int32.max - 1) / 2,
                  parent.y <= (Int32.max - 1) / 2,
                  parent.z <= (Int32.max - 1) / 2 else {
                throw NativeRuntimeError.invalidArgument("sparse subdivision coordinate overflows")
            }
            for child in 0..<8 where values[parentIndex * 8 + child] > 0 {
                coordinates.append(SparseStructureCoordinate(
                    batch: parent.batch,
                    x: parent.x * 2 + Int32(child & 1),
                    y: parent.y * 2 + Int32((child >> 1) & 1),
                    z: parent.z * 2 + Int32((child >> 2) & 1)
                ))
                parents.append(UInt32(parentIndex))
                children.append(UInt32(child))
            }
        }
        self.coordinates = coordinates
        self.parentIndices = parents
        self.childIndices = children
    }

    public func makeMetalBuffers(context: MetalContext) throws -> (MTLBuffer, MTLBuffer)? {
        guard !coordinates.isEmpty else { return nil }
        func upload(_ values: [UInt32], label: String) throws -> MTLBuffer {
            let bytes = try sparseSubdivisionProduct(
                values.count, MemoryLayout<UInt32>.stride
            )
            let buffer = try context.makeBuffer(length: bytes, label: label)
            values.withUnsafeBytes { source in
                buffer.contents().copyMemory(from: source.baseAddress!, byteCount: bytes)
            }
            return buffer
        }
        return try (
            upload(parentIndices, label: "TRELLIS subdivision parents"),
            upload(childIndices, label: "TRELLIS subdivision children")
        )
    }
}

private func sparseSubdivisionProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard lhs >= 0, rhs >= 0, !result.overflow else {
        throw NativeRuntimeError.invalidArgument("sparse subdivision size overflows Int")
    }
    return result.partialValue
}
