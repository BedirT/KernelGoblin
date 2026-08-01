import Foundation
import Metal

public struct NativePBRExportEvidence: Equatable, Sendable {
    public let rasterBackend: String
    public let textureWidth: Int
    public let textureHeight: Int
    public let coveredTexels: Int
    public let totalTexels: Int
    public let glbBytes: Int
    public let alphaMode: String
    public let doubleSided: Bool
}

public final class NativePBRExporter: @unchecked Sendable {
    private let baker: SparsePBRTextureBaker

    public init(context: MetalContext) throws {
        self.baker = try SparsePBRTextureBaker(context: context)
    }

    @discardableResult
    public func export(
        atlas: UVAtlasMesh,
        sourcePositions: [SIMD3<Float>],
        sourceFaces: [SIMD3<UInt32>],
        decodedPBRFields: MTLBuffer,
        fieldCoordinates: [SparseStructureCoordinate],
        spatialShape: SparseSpatialShape,
        aabbMinimum: SIMD3<Float>,
        voxelSize: SIMD3<Float>,
        textureWidth: Int,
        textureHeight: Int,
        alphaMode: String = "OPAQUE",
        doubleSided: Bool = true,
        outputURL: URL
    ) throws -> NativePBRExportEvidence {
        try autoreleasepool {
            let bake = try baker.bake(
            positions: atlas.positions, uvs: atlas.uvs, faces: atlas.faces,
            decodedPBRFields: decodedPBRFields,
            fieldCoordinates: fieldCoordinates, spatialShape: spatialShape,
            aabbMinimum: aabbMinimum, voxelSize: voxelSize,
            width: textureWidth, height: textureHeight
        )
            let total = textureWidth.multipliedReportingOverflow(by: textureHeight)
            guard !total.overflow else {
                throw NativeRuntimeError.invalidArgument("PBR export texture size overflows Int")
            }
            let mask = bake.mask.contents().assumingMemoryBound(to: UInt8.self)
            var covered = 0
            for pixel in 0..<total.partialValue where mask[pixel] != 0 { covered += 1 }
            let textures = try PBRTextureAssembler.finalize(bake)
            let glb = try PBRGLBWriter.encode(
                atlas: atlas, sourcePositions: sourcePositions,
                sourceFaces: sourceFaces, textures: textures,
                alphaMode: alphaMode, doubleSided: doubleSided
            )
            try glb.write(to: outputURL, options: .atomic)
            return NativePBRExportEvidence(
                rasterBackend: "Metal", textureWidth: textureWidth,
                textureHeight: textureHeight, coveredTexels: covered,
                totalTexels: total.partialValue, glbBytes: glb.count,
                alphaMode: alphaMode, doubleSided: doubleSided
            )
        }
    }
}
