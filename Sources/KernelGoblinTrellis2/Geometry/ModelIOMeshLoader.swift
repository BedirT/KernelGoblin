import Foundation
import ModelIO
import simd

public struct NativeMeshInput: Equatable, Sendable {
    public let positions: [SIMD3<Float>]
    public let faces: [SIMD3<UInt32>]
    public let suppliedUVs: [SIMD2<Float>]?
    public let sourceMeshCount: Int

    public init(
        positions: [SIMD3<Float>], faces: [SIMD3<UInt32>],
        suppliedUVs: [SIMD2<Float>]?, sourceMeshCount: Int
    ) throws {
        guard !positions.isEmpty, !faces.isEmpty,
              sourceMeshCount > 0, positions.count <= Int(UInt32.max),
              suppliedUVs == nil || suppliedUVs?.count == positions.count,
              positions.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }),
              suppliedUVs?.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) ?? true,
              faces.allSatisfy({
                  Int($0.x) < positions.count && Int($0.y) < positions.count
                      && Int($0.z) < positions.count
              }) else {
            throw NativeRuntimeError.invalidArgument("invalid native mesh input")
        }
        self.positions = positions
        self.faces = faces
        self.suppliedUVs = suppliedUVs
        self.sourceMeshCount = sourceMeshCount
    }

    public func normalizedForTrellis() throws -> NativeMeshInput {
        var minimum = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maximum = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for position in positions {
            minimum = simd_min(minimum, position)
            maximum = simd_max(maximum, position)
        }
        let extent = maximum - minimum
        let largest = max(extent.x, max(extent.y, extent.z))
        guard largest.isFinite, largest > Float.ulpOfOne else {
            throw NativeRuntimeError.invalidArgument(
                "mesh must have a finite nonzero bounding-box extent"
            )
        }
        let center = (minimum + maximum) * 0.5
        let scale: Float = 0.99999 / largest
        let transformed = positions.map { position -> SIMD3<Float> in
            let centered = (position - center) * scale
            return SIMD3(centered.x, -centered.z, centered.y)
        }
        guard transformed.allSatisfy({
            $0.x >= -0.5 && $0.x <= 0.5
                && $0.y >= -0.5 && $0.y <= 0.5
                && $0.z >= -0.5 && $0.z <= 0.5
        }) else {
            throw NativeRuntimeError.invalidArgument(
                "normalized TRELLIS mesh escaped the unit AABB"
            )
        }
        return try NativeMeshInput(
            positions: transformed, faces: faces,
            suppliedUVs: suppliedUVs, sourceMeshCount: sourceMeshCount
        )
    }
}

public enum ModelIOMeshLoader {
    public static func load(url: URL) throws -> NativeMeshInput {
        let allocator = MDLMeshBufferDataAllocator()
        let asset = MDLAsset(
            url: url, vertexDescriptor: nil, bufferAllocator: allocator,
            preserveTopology: true, error: nil
        )
        let meshes = asset.childObjects(of: MDLMesh.self) as? [MDLMesh] ?? []
        guard !meshes.isEmpty else {
            throw NativeRuntimeError.invalidArgument(
                "Model I/O found no mesh in \(url.lastPathComponent)"
            )
        }
        var positions: [SIMD3<Float>] = []
        var faces: [SIMD3<UInt32>] = []
        var textureCoordinates: [SIMD2<Float>] = []
        var allMeshesHaveUV = true
        for mesh in meshes {
            guard let positionData = mesh.vertexAttributeData(
                forAttributeNamed: MDLVertexAttributePosition, as: .float3
            ) else {
                throw NativeRuntimeError.invalidArgument(
                    "Model I/O mesh has no Float32 position attribute"
                )
            }
            let uvData = mesh.vertexAttributeData(
                forAttributeNamed: MDLVertexAttributeTextureCoordinate, as: .float2
            )
            allMeshesHaveUV = allMeshesHaveUV && uvData != nil
            let transform = worldTransform(mesh)
            let base = positions.count
            guard base <= Int(UInt32.max) - mesh.vertexCount else {
                throw NativeRuntimeError.invalidArgument("combined mesh exceeds UInt32 indexing")
            }
            for index in 0..<mesh.vertexCount {
                let pointer = positionData.dataStart.advanced(
                    by: index * positionData.stride
                )
                let local = SIMD4<Float>(
                    meshFloat(pointer, 0), meshFloat(pointer, 4),
                    meshFloat(pointer, 8), 1
                )
                let world = transform * local
                guard world.w.isFinite, abs(world.w) > Float.ulpOfOne else {
                    throw NativeRuntimeError.invalidArgument(
                        "mesh transform produced an invalid homogeneous position"
                    )
                }
                positions.append(SIMD3(world.x, world.y, world.z) / world.w)
                if let uvData {
                    let uv = uvData.dataStart.advanced(by: index * uvData.stride)
                    textureCoordinates.append(SIMD2(
                        meshFloat(uv, 0), meshFloat(uv, 4)
                    ))
                } else {
                    textureCoordinates.append(.zero)
                }
            }
            guard let submeshes = mesh.submeshes as? [MDLSubmesh],
                  !submeshes.isEmpty else {
                throw NativeRuntimeError.invalidArgument("Model I/O mesh has no submesh")
            }
            for submesh in submeshes {
                guard submesh.geometryType == .triangles,
                      submesh.indexCount.isMultiple(of: 3) else {
                    throw NativeRuntimeError.invalidArgument(
                        "native mesh loading requires triangulated submeshes"
                    )
                }
                let map = submesh.indexBuffer.map()
                for triangle in stride(from: 0, to: submesh.indexCount, by: 3) {
                    let a = try index(at: triangle, submesh: submesh, map: map)
                    let b = try index(at: triangle + 1, submesh: submesh, map: map)
                    let c = try index(at: triangle + 2, submesh: submesh, map: map)
                    guard Int(a) < mesh.vertexCount,
                          Int(b) < mesh.vertexCount,
                          Int(c) < mesh.vertexCount else {
                        throw NativeRuntimeError.invalidArgument(
                            "Model I/O submesh index is out of range"
                        )
                    }
                    faces.append(SIMD3(
                        UInt32(base) + a, UInt32(base) + b, UInt32(base) + c
                    ))
                }
            }
        }
        return try NativeMeshInput(
            positions: positions, faces: faces,
            suppliedUVs: allMeshesHaveUV ? textureCoordinates : nil,
            sourceMeshCount: meshes.count
        )
    }

    private static func index(
        at offset: Int, submesh: MDLSubmesh, map: MDLMeshBufferMap
    ) throws -> UInt32 {
        switch submesh.indexType {
        case .uInt8:
            UInt32(map.bytes.load(fromByteOffset: offset, as: UInt8.self))
        case .uInt16:
            UInt32(UInt16(littleEndian: map.bytes.loadUnaligned(
                fromByteOffset: offset * 2, as: UInt16.self
            )))
        case .uInt32:
            UInt32(littleEndian: map.bytes.loadUnaligned(
                fromByteOffset: offset * 4, as: UInt32.self
            ))
        default:
            throw NativeRuntimeError.invalidArgument(
                "Model I/O returned an unsupported mesh index type"
            )
        }
    }

    private static func worldTransform(_ object: MDLObject) -> simd_float4x4 {
        var chain: [simd_float4x4] = []
        var current: MDLObject? = object
        while let value = current {
            if let transform = value.transform {
                chain.append(transform.matrix)
            }
            current = value.parent
        }
        return chain.reversed().reduce(matrix_identity_float4x4, *)
    }
}

private func meshFloat(_ pointer: UnsafeMutableRawPointer, _ offset: Int) -> Float {
    Float(bitPattern: UInt32(littleEndian: pointer.loadUnaligned(
        fromByteOffset: offset, as: UInt32.self
    )))
}
