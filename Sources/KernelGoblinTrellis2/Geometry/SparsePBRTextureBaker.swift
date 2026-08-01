import Metal

public struct SparsePBRTextureBake: @unchecked Sendable {
    public let fields: MTLBuffer
    public let faceIDs: MTLBuffer
    public let mask: MTLBuffer
    public let width: Int
    public let height: Int
}

public final class SparsePBRTextureBaker: @unchecked Sendable {
    public static let channelCount = 6

    private let context: MetalContext
    private let rasterizer: UVRasterizer
    private let sampler: SparsePBRSampler

    public init(context: MetalContext) throws {
        self.context = context
        self.rasterizer = try UVRasterizer(context: context)
        self.sampler = try SparsePBRSampler(context: context)
    }

    public func bake(
        positions: [SIMD3<Float>],
        uvs: [SIMD2<Float>],
        faces: [SIMD3<UInt32>],
        decodedPBRFields: MTLBuffer,
        fieldCoordinates: [SparseStructureCoordinate],
        spatialShape: SparseSpatialShape,
        aabbMinimum: SIMD3<Float>,
        voxelSize: SIMD3<Float>,
        width: Int,
        height: Int
    ) throws -> SparsePBRTextureBake {
        let raster = try rasterizer.rasterize(
            positions: positions, uvs: uvs, faces: faces, width: width, height: height
        )
        let pixelCount = width.multipliedReportingOverflow(by: height)
        let elementCount = pixelCount.partialValue.multipliedReportingOverflow(
            by: Self.channelCount
        )
        let byteCount = elementCount.partialValue.multipliedReportingOverflow(
            by: MemoryLayout<Float>.stride
        )
        guard !pixelCount.overflow, !elementCount.overflow, !byteCount.overflow else {
            throw NativeRuntimeError.invalidArgument("PBR texture dimensions overflow Int")
        }
        let fields = try context.makeBuffer(
            length: byteCount.partialValue, label: "TRELLIS sampled UV-space PBR fields"
        )
        let lookup = try sampler.makeLookup(
            coordinates: fieldCoordinates, spatialShape: spatialShape
        )
        try sampler.sampleSurfaceTrilinearF32(
            features: decodedPBRFields, lookup: lookup,
            objectPositions: raster.positions, mask: raster.mask,
            queryCount: pixelCount.partialValue, channels: Self.channelCount,
            aabbMinimum: aabbMinimum, voxelSize: voxelSize, output: fields
        )
        return SparsePBRTextureBake(
            fields: fields, faceIDs: raster.faceIDs, mask: raster.mask,
            width: width, height: height
        )
    }
}
