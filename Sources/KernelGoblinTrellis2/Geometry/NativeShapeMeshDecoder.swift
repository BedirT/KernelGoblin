import Metal

public struct NativeShapeMeshResult: Equatable, Sendable {
    public let mesh: FlexibleDualGridMesh
    public let gridSize: SparseSpatialShape
    public let sourceCoordinateCount: Int
}

public final class NativeShapeMeshDecoder: @unchecked Sendable {
    private let context: MetalContext
    private let head: FlexibleDualGridHeadKernel

    public init(context: MetalContext) throws {
        self.context = context
        self.head = try FlexibleDualGridHeadKernel(context: context)
    }

    public func decode(
        rawHead: MTLBuffer,
        coordinates: [SparseStructureCoordinate],
        gridSize: SparseSpatialShape,
        aabbMinimum: SIMD3<Float> = SIMD3(repeating: -0.5),
        aabbMaximum: SIMD3<Float> = SIMD3(repeating: 0.5)
    ) throws -> NativeShapeMeshResult {
        let fields = try head(rawHead: rawHead, count: coordinates.count)
        let mesh = try FlexibleDualGridMeshExtractor.extract(
            coordinates: coordinates,
            dualOffsets: fields.dualOffsets,
            intersections: fields.intersections,
            splitWeights: fields.splitWeights,
            gridSize: gridSize,
            aabbMinimum: aabbMinimum,
            aabbMaximum: aabbMaximum
        )
        return NativeShapeMeshResult(
            mesh: mesh, gridSize: gridSize,
            sourceCoordinateCount: coordinates.count
        )
    }
}
