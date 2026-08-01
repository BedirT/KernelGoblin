import Metal

public struct SparseFeatureTensor: @unchecked Sendable {
    public let features: MTLBuffer?
    public let channels: Int
    public let neighborhood: SparseNeighborhood3x3

    public var coordinates: [SparseStructureCoordinate] { neighborhood.coordinates }
    public var spatialShape: SparseSpatialShape { neighborhood.spatialShape }
    public var tokenCount: Int { coordinates.count }

    public init(
        features: MTLBuffer?,
        channels: Int,
        coordinates: [SparseStructureCoordinate],
        spatialShape: SparseSpatialShape
    ) throws {
        guard channels > 0, channels <= Int(UInt32.max) else {
            throw NativeRuntimeError.invalidArgument("invalid sparse feature channel count")
        }
        let elements = coordinates.count.multipliedReportingOverflow(by: channels)
        let bytes = elements.partialValue.multipliedReportingOverflow(
            by: MemoryLayout<Float>.stride
        )
        guard !elements.overflow, !bytes.overflow,
              coordinates.isEmpty ? features == nil : features?.length ?? 0 >= bytes.partialValue
        else {
            throw NativeRuntimeError.invalidArgument("sparse feature buffer is too small")
        }
        self.features = features
        self.channels = channels
        self.neighborhood = try SparseNeighborhood3x3(
            coordinates: coordinates, spatialShape: spatialShape
        )
    }
}

public struct SparseSpatialShape: Equatable, Sendable {
    public let width: Int
    public let height: Int
    public let depth: Int

    public init(width: Int, height: Int, depth: Int) throws {
        guard width > 0, height > 0, depth > 0,
              width <= Int(Int32.max), height <= Int(Int32.max),
              depth <= Int(Int32.max) else {
            throw NativeRuntimeError.invalidArgument("invalid sparse spatial shape")
        }
        self.width = width
        self.height = height
        self.depth = depth
    }

    public init(cubic resolution: Int) throws {
        try self.init(width: resolution, height: resolution, depth: resolution)
    }
}

public struct SparseNeighborhood3x3: Equatable, Sendable {
    public static let neighborCount = 27
    public let coordinates: [SparseStructureCoordinate]
    public let spatialShape: SparseSpatialShape
    public let indices: [Int32]

    public init(
        coordinates: [SparseStructureCoordinate], spatialShape: SparseSpatialShape
    ) throws {
        var lookup: [SparseStructureCoordinate: Int32] = [:]
        lookup.reserveCapacity(coordinates.count)
        for (index, coordinate) in coordinates.enumerated() {
            guard coordinate.batch >= 0,
                  coordinate.x >= 0, coordinate.x < spatialShape.width,
                  coordinate.y >= 0, coordinate.y < spatialShape.height,
                  coordinate.z >= 0, coordinate.z < spatialShape.depth,
                  index <= Int(Int32.max) else {
                throw NativeRuntimeError.invalidArgument(
                    "sparse coordinate is outside its spatial shape"
                )
            }
            guard lookup.updateValue(Int32(index), forKey: coordinate) == nil else {
                throw NativeRuntimeError.invalidArgument("sparse coordinates must be unique")
            }
        }
        var indices = [Int32]()
        indices.reserveCapacity(coordinates.count * Self.neighborCount)
        for coordinate in coordinates {
            for dx in -1...1 {
                for dy in -1...1 {
                    for dz in -1...1 {
                        let neighbor = SparseStructureCoordinate(
                            batch: coordinate.batch,
                            x: coordinate.x + Int32(dx),
                            y: coordinate.y + Int32(dy),
                            z: coordinate.z + Int32(dz)
                        )
                        indices.append(lookup[neighbor] ?? -1)
                    }
                }
            }
        }
        self.coordinates = coordinates
        self.spatialShape = spatialShape
        self.indices = indices
    }

    public func makeMetalBuffer(context: MetalContext) throws -> MTLBuffer? {
        guard !indices.isEmpty else { return nil }
        let byteCount = indices.count * MemoryLayout<Int32>.stride
        let buffer = try context.makeBuffer(length: byteCount, label: "TRELLIS sparse 3x3 neighbor map")
        indices.withUnsafeBytes { bytes in
            buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: byteCount)
        }
        return buffer
    }
}
