import Foundation
import simd

public enum GeometryGLBWriter {
    public static func write(
        to url: URL,
        positions: [SIMD3<Float>],
        faces: [SIMD3<UInt32>]
    ) throws -> PBRGLBValidation {
        let atlas = try UVAtlasMesh.preserving(
            positions: positions,
            faces: faces,
            uvs: []
        )
        try PBRGLBWriter.write(
            to: url,
            atlas: atlas,
            sourcePositions: positions,
            sourceFaces: faces,
            textures: nil
        )
        return try PBRGLBValidator.validate(url: url)
    }
}
