import CryptoKit
import Foundation

public enum UVPreparationPolicy: String, Sendable {
    case preserve
    case preserveOrGenerate = "preserve-or-generate"
    case regenerate
}

public struct UVPreparationResult: Equatable, Sendable {
    public let atlas: UVAtlasMesh
    public let implementation: String
    public let exactUpstreamTier: Bool
    public let fingerprintSHA256: String
    public let operatingSystem: String
}

public enum UVPreparation {
    public static func prepare(
        positions: [SIMD3<Float>],
        faces: [SIMD3<UInt32>],
        suppliedUVs: [SIMD2<Float>]?,
        policy: UVPreparationPolicy
    ) throws -> UVPreparationResult {
        let atlas: UVAtlasMesh
        let implementation: String
        let exact: Bool
        switch policy {
        case .preserve:
            guard let suppliedUVs else {
                throw NativeRuntimeError.invalidArgument(
                    "UV preserve policy requires source texture coordinates"
                )
            }
            atlas = try .preserving(
                positions: positions, faces: faces, uvs: suppliedUVs
            )
            implementation = "supplied-uv-preservation"
            exact = true
        case .preserveOrGenerate where suppliedUVs != nil:
            atlas = try .preserving(
                positions: positions, faces: faces, uvs: suppliedUVs!
            )
            implementation = "supplied-uv-preservation"
            exact = true
        case .preserveOrGenerate, .regenerate:
            atlas = try ModelIOUVUnwrapper.unwrap(
                positions: positions, faces: faces
            )
            implementation = "Apple-ModelIO-addUnwrappedTextureCoordinates"
            exact = false
        }
        return UVPreparationResult(
            atlas: atlas, implementation: implementation,
            exactUpstreamTier: exact,
            fingerprintSHA256: fingerprint(atlas),
            operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString
        )
    }

    public static func fingerprint(_ atlas: UVAtlasMesh) -> String {
        var data = Data()
        appendCount(atlas.positions.count, to: &data)
        for value in atlas.positions {
            appendWord(value.x.bitPattern, to: &data)
            appendWord(value.y.bitPattern, to: &data)
            appendWord(value.z.bitPattern, to: &data)
        }
        appendCount(atlas.faces.count, to: &data)
        for value in atlas.faces {
            appendWord(value.x, to: &data)
            appendWord(value.y, to: &data)
            appendWord(value.z, to: &data)
        }
        appendCount(atlas.uvs.count, to: &data)
        for value in atlas.uvs {
            appendWord(value.x.bitPattern, to: &data)
            appendWord(value.y.bitPattern, to: &data)
        }
        for value in atlas.vertexMap { appendWord(value, to: &data) }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

private func appendCount(_ value: Int, to data: inout Data) {
    var littleEndian = UInt64(value).littleEndian
    withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
}

private func appendWord(_ value: UInt32, to data: inout Data) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
}
