import Metal

public struct SparseSpatial2Channel2x: Sendable {
    public let inputCoordinates: [SparseStructureCoordinate]
    public let coordinates: [SparseStructureCoordinate]
    public let inputParentIndices: [UInt32]
    public let inputChildIndices: [UInt32]
    public let sourceIndices: [Int32]
    public let subdivision: SparseSubdivision2x
    public let spatialShape: SparseSpatialShape

    public init(
        coordinates inputCoordinates: [SparseStructureCoordinate],
        spatialShape inputShape: SparseSpatialShape
    ) throws {
        guard !inputCoordinates.isEmpty,
              inputCoordinates.count <= Int(UInt32.max) else {
            throw NativeRuntimeError.invalidArgument(
                "spatial-to-channel requires active sparse coordinates"
            )
        }
        // SparseNeighborhood3x3 owns the common bounds and uniqueness contract.
        _ = try SparseNeighborhood3x3(
            coordinates: inputCoordinates, spatialShape: inputShape
        )

        let coarseSet = Set(inputCoordinates.map {
            SparseStructureCoordinate(
                batch: $0.batch, x: $0.x / 2, y: $0.y / 2, z: $0.z / 2
            )
        })
        let coarseCoordinates = coarseSet.sorted(by: sparseCoordinateLess)
        guard coarseCoordinates.count <= Int(UInt32.max) else {
            throw NativeRuntimeError.invalidArgument(
                "spatial-to-channel coarse coordinate count exceeds UInt32"
            )
        }
        let coarseLookup = Dictionary(
            uniqueKeysWithValues: coarseCoordinates.enumerated().map { ($1, UInt32($0)) }
        )
        let slotCount = try sparseS2CProduct(coarseCoordinates.count, 8)
        var sourceIndices = [Int32](repeating: -1, count: slotCount)
        var inputParents = [UInt32]()
        var inputChildren = [UInt32]()
        inputParents.reserveCapacity(inputCoordinates.count)
        inputChildren.reserveCapacity(inputCoordinates.count)

        for (sourceIndex, coordinate) in inputCoordinates.enumerated() {
            let parent = SparseStructureCoordinate(
                batch: coordinate.batch,
                x: coordinate.x / 2, y: coordinate.y / 2, z: coordinate.z / 2
            )
            guard let parentIndex = coarseLookup[parent],
                  sourceIndex <= Int(Int32.max) else {
                throw NativeRuntimeError.invalidArgument(
                    "spatial-to-channel coordinate map exceeds Metal index space"
                )
            }
            // Upstream uses sum(subidx[axis] * 2**axis): x is the low bit.
            let child = UInt32(coordinate.x & 1)
                | (UInt32(coordinate.y & 1) << 1)
                | (UInt32(coordinate.z & 1) << 2)
            let slot = Int(parentIndex) * 8 + Int(child)
            guard sourceIndices[slot] == -1 else {
                throw NativeRuntimeError.invalidArgument(
                    "spatial-to-channel child slot is duplicated"
                )
            }
            sourceIndices[slot] = Int32(sourceIndex)
            inputParents.append(parentIndex)
            inputChildren.append(child)
        }

        let outputShape = try SparseSpatialShape(
            width: sparseS2CCeilHalf(inputShape.width),
            height: sparseS2CCeilHalf(inputShape.height),
            depth: sparseS2CCeilHalf(inputShape.depth)
        )
        self.inputCoordinates = inputCoordinates
        self.coordinates = coarseCoordinates
        self.inputParentIndices = inputParents
        self.inputChildIndices = inputChildren
        self.sourceIndices = sourceIndices
        self.subdivision = SparseSubdivision2x(
            coordinates: inputCoordinates,
            parentIndices: inputParents,
            childIndices: inputChildren
        )
        self.spatialShape = outputShape
    }

    public func makeSourceIndexBuffer(context: MetalContext) throws -> MTLBuffer {
        let byteCount = try sparseS2CProduct(
            sourceIndices.count, MemoryLayout<Int32>.stride
        )
        let buffer = try context.makeBuffer(
            length: byteCount, label: "TRELLIS spatial-to-channel source map"
        )
        sourceIndices.withUnsafeBytes { source in
            buffer.contents().copyMemory(from: source.baseAddress!, byteCount: byteCount)
        }
        return buffer
    }
}

extension SparseSubdivision2x {
    init(
        coordinates: [SparseStructureCoordinate],
        parentIndices: [UInt32],
        childIndices: [UInt32]
    ) {
        self.coordinates = coordinates
        self.parentIndices = parentIndices
        self.childIndices = childIndices
    }
}

private func sparseCoordinateLess(
    _ lhs: SparseStructureCoordinate, _ rhs: SparseStructureCoordinate
) -> Bool {
    if lhs.batch != rhs.batch { return lhs.batch < rhs.batch }
    if lhs.x != rhs.x { return lhs.x < rhs.x }
    if lhs.y != rhs.y { return lhs.y < rhs.y }
    return lhs.z < rhs.z
}

private func sparseS2CCeilHalf(_ value: Int) throws -> Int {
    guard value > 0 else {
        throw NativeRuntimeError.invalidArgument("invalid spatial-to-channel shape")
    }
    return value / 2 + value % 2
}

func sparseS2CProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard lhs >= 0, rhs >= 0, !result.overflow else {
        throw NativeRuntimeError.invalidArgument(
            "spatial-to-channel tensor size overflows Int"
        )
    }
    return result.partialValue
}
