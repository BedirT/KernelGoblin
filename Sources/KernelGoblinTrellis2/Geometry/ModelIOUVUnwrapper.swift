import Foundation
import ModelIO
import simd

public enum ModelIOUVUnwrapper {
    private static let vertexMapAttribute = "kg_vertex_map"

    public static func unwrap(
        positions: [SIMD3<Float>], faces: [SIMD3<UInt32>]
    ) throws -> UVAtlasMesh {
        guard !positions.isEmpty, !faces.isEmpty,
              positions.count <= Int(UInt32.max) else {
            throw NativeRuntimeError.invalidArgument("invalid mesh for UV unwrapping")
        }
        for position in positions {
            guard position.x.isFinite, position.y.isFinite, position.z.isFinite else {
                throw NativeRuntimeError.invalidArgument("UV unwrap positions must be finite")
            }
        }
        var containsSmallFace = false
        let filteredFaces = try faces.filter { face in
            guard Int(face.x) < positions.count,
                  Int(face.y) < positions.count,
                  Int(face.z) < positions.count else {
                throw NativeRuntimeError.invalidArgument("UV unwrap face is out of range")
            }
            guard face.x != face.y, face.y != face.z, face.z != face.x else { return false }
            let normal = simd_cross(
                positions[Int(face.y)] - positions[Int(face.x)],
                positions[Int(face.z)] - positions[Int(face.x)]
            )
            let areaSquared = simd_length_squared(normal)
            if areaSquared > 0, areaSquared <= Float.ulpOfOne {
                containsSmallFace = true
            }
            return areaSquared > 0
        }
        guard !filteredFaces.isEmpty else {
            throw NativeRuntimeError.invalidArgument("UV unwrap mesh has no nondegenerate faces")
        }
        guard !containsSmallFace else {
            throw NativeRuntimeError.invalidArgument(
                "Model I/O UV unwrap is unsafe for normalized small faces"
            )
        }
        let allocator = MDLMeshBufferDataAllocator()
        var vertexBytes = Data(capacity: positions.count * 16)
        for (index, position) in positions.enumerated() {
            appendFloat(position.x, to: &vertexBytes)
            appendFloat(position.y, to: &vertexBytes)
            appendFloat(position.z, to: &vertexBytes)
            appendUInt32UV(UInt32(index), to: &vertexBytes)
        }
        var indexBytes = Data(capacity: filteredFaces.count * 12)
        for face in filteredFaces {
            appendUInt32UV(face.x, to: &indexBytes)
            appendUInt32UV(face.y, to: &indexBytes)
            appendUInt32UV(face.z, to: &indexBytes)
        }
        let vertexBuffer = allocator.newBuffer(with: vertexBytes, type: .vertex)
        let indexBuffer = allocator.newBuffer(with: indexBytes, type: .index)
        let descriptor = MDLVertexDescriptor()
        descriptor.attributes[0] = MDLVertexAttribute(
            name: MDLVertexAttributePosition, format: .float3,
            offset: 0, bufferIndex: 0
        )
        descriptor.attributes[1] = MDLVertexAttribute(
            name: vertexMapAttribute, format: .uInt,
            offset: 12, bufferIndex: 0
        )
        descriptor.layouts[0] = MDLVertexBufferLayout(stride: 16)
        let submesh = MDLSubmesh(
            indexBuffer: indexBuffer, indexCount: filteredFaces.count * 3,
            indexType: .uInt32, geometryType: .triangles, material: nil
        )
        let mesh = MDLMesh(
            vertexBuffer: vertexBuffer, vertexCount: positions.count,
            descriptor: descriptor, submeshes: [submesh]
        )
        mesh.addUnwrappedTextureCoordinates(
            forAttributeNamed: MDLVertexAttributeTextureCoordinate
        )
        guard let positionData = mesh.vertexAttributeData(
                forAttributeNamed: MDLVertexAttributePosition, as: .float3
              ),
              let uvData = mesh.vertexAttributeData(
                forAttributeNamed: MDLVertexAttributeTextureCoordinate, as: .float2
              ),
              let mapData = mesh.vertexAttributeData(
                forAttributeNamed: vertexMapAttribute, as: .uInt
              ),
              let outputSubmesh = mesh.submeshes?.firstObject as? MDLSubmesh else {
            throw NativeRuntimeError.invalidArgument("Model I/O did not produce a UV atlas")
        }
        var outputPositions: [SIMD3<Float>] = []
        var outputUVs: [SIMD2<Float>] = []
        var vertexMap: [UInt32] = []
        outputPositions.reserveCapacity(mesh.vertexCount)
        outputUVs.reserveCapacity(mesh.vertexCount)
        vertexMap.reserveCapacity(mesh.vertexCount)
        for index in 0..<mesh.vertexCount {
            let position = positionData.dataStart.advanced(by: index * positionData.stride)
            let uv = uvData.dataStart.advanced(by: index * uvData.stride)
            let mapping = mapData.dataStart.advanced(by: index * mapData.stride)
            outputPositions.append(SIMD3<Float>(
                loadFloat(position, offset: 0), loadFloat(position, offset: 4),
                loadFloat(position, offset: 8)
            ))
            outputUVs.append(SIMD2<Float>(
                loadFloat(uv, offset: 0), loadFloat(uv, offset: 4)
            ))
            vertexMap.append(loadUInt32(mapping, offset: 0))
        }
        let indexMap = outputSubmesh.indexBuffer.map()
        var outputIndices: [UInt32] = []
        outputIndices.reserveCapacity(outputSubmesh.indexCount)
        for index in 0..<outputSubmesh.indexCount {
            switch outputSubmesh.indexType {
            case .uInt8:
                outputIndices.append(UInt32(indexMap.bytes.load(
                    fromByteOffset: index, as: UInt8.self
                )))
            case .uInt16:
                outputIndices.append(UInt32(UInt16(littleEndian: indexMap.bytes.loadUnaligned(
                    fromByteOffset: index * 2, as: UInt16.self
                ))))
            case .uInt32:
                outputIndices.append(UInt32(littleEndian: indexMap.bytes.loadUnaligned(
                    fromByteOffset: index * 4, as: UInt32.self
                )))
            default:
                throw NativeRuntimeError.invalidArgument("Model I/O returned an invalid index type")
            }
        }
        guard outputIndices.count.isMultiple(of: 3) else {
            throw NativeRuntimeError.invalidArgument("Model I/O returned non-triangle indices")
        }
        let outputFaces = stride(from: 0, to: outputIndices.count, by: 3).map {
            SIMD3<UInt32>(outputIndices[$0], outputIndices[$0 + 1], outputIndices[$0 + 2])
        }
        guard outputFaces.count == filteredFaces.count else {
            throw NativeRuntimeError.invalidArgument(
                "Model I/O UV unwrap changed face count from \(filteredFaces.count) "
                    + "to \(outputFaces.count)"
            )
        }
        return try UVAtlasMesh(
            positions: outputPositions, faces: outputFaces,
            uvs: outputUVs, vertexMap: vertexMap
        )
    }
}

private func appendFloat(_ value: Float, to data: inout Data) {
    appendUInt32UV(value.bitPattern, to: &data)
}

private func appendUInt32UV(_ value: UInt32, to data: inout Data) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
}

private func loadFloat(_ pointer: UnsafeMutableRawPointer, offset: Int) -> Float {
    Float(bitPattern: loadUInt32(pointer, offset: offset))
}

private func loadUInt32(_ pointer: UnsafeMutableRawPointer, offset: Int) -> UInt32 {
    UInt32(littleEndian: pointer.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
}
