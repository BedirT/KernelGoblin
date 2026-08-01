import Foundation
import ImageIO
import Metal
import Testing
@testable import KernelGoblinTrellis2

@Suite("Native TRELLIS.2 PBR export")
struct PBRExportTests {
    @Test("geometry-only GLB has valid indices, no UVs, and a neutral material")
    func geometryOnlyRoundTrip() throws {
        let positions = [
            SIMD3<Float>(0, 0, 0),
            SIMD3<Float>(1, 0, 0),
            SIMD3<Float>(0, 1, 0),
            SIMD3<Float>(0, 0, 1),
        ]
        let faces = [
            SIMD3<UInt32>(0, 2, 1),
            SIMD3<UInt32>(0, 1, 3),
            SIMD3<UInt32>(0, 3, 2),
            SIMD3<UInt32>(1, 2, 3),
        ]
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = directory.appendingPathComponent("geometry.glb")
        let validation = try GeometryGLBWriter.write(
            to: output, positions: positions, faces: faces
        )
        #expect(validation == PBRGLBValidation(
            vertexCount: 4, indexCount: 12, imageCount: 0,
            textureWidth: 0, textureHeight: 0,
            alphaMode: "OPAQUE", doubleSided: true
        ))
        #expect(try Data(contentsOf: output).count > 0)
        let document = try glbDocument(at: output)
        let meshes = try #require(document["meshes"] as? [[String: Any]])
        let primitives = try #require(meshes[0]["primitives"] as? [[String: Any]])
        let attributes = try #require(primitives[0]["attributes"] as? [String: Any])
        #expect(attributes["TEXCOORD_0"] == nil)
        #expect(document["images"] == nil && document["textures"] == nil)
    }

    @Test("supplied-UV sparse fields export through the complete native PBR path")
    func nativeExportCallSite() throws {
        let context = try MetalContext(arenaCapacity: 4 * 1024 * 1024)
        let exporter = try NativePBRExporter(context: context)
        let positions = [
            SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0),
        ]
        let faces = [SIMD3<UInt32>(0, 1, 2)]
        let atlas = try UVAtlasMesh.preserving(
            positions: positions, faces: faces,
            uvs: [SIMD2<Float>(0, 0), SIMD2<Float>(1, 0), SIMD2<Float>(0, 1)]
        )
        var fields: [Float] = [0.2, 0.4, 0.6, 0.8, 0.3, 0.5]
        let fieldBuffer = try #require(context.device.makeBuffer(
            bytes: &fields, length: fields.count * 4, options: .storageModeShared
        ))
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = directory.appendingPathComponent("native-pbr.glb")
        let evidence = try exporter.export(
            atlas: atlas, sourcePositions: positions, sourceFaces: faces,
            decodedPBRFields: fieldBuffer,
            fieldCoordinates: [SparseStructureCoordinate(x: 0, y: 0, z: 0)],
            spatialShape: SparseSpatialShape(cubic: 1),
            aabbMinimum: .zero, voxelSize: SIMD3<Float>(repeating: 1),
            textureWidth: 16, textureHeight: 16, outputURL: output
        )
        let glb = try Data(contentsOf: output)
        let validation = try PBRGLBValidator.validate(data: glb)
        #expect(evidence.rasterBackend == "Metal")
        #expect(evidence.coveredTexels > 0 && evidence.coveredTexels < 256)
        #expect(evidence.totalTexels == 256 && evidence.glbBytes == glb.count)
        #expect(readUInt32(glb, at: 0) == 0x46546C67)
        #expect(validation.vertexCount == 3)
        #expect(validation.indexCount == 3)
        #expect(validation.textureWidth == 16 && validation.textureHeight == 16)
        #expect(context.arena?.snapshot().usedBytes == 0)
    }

    @Test("covered fields use glTF channel packing and truncating uint8 conversion")
    func texturePacking() throws {
        let context = try MetalContext()
        var fields: [Float] = [
            0.2, 0.4, 0.6, 0.8, 0.3, 0.5,
            1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1,
        ]
        var mask: [UInt8] = [1, 0, 0, 0]
        var ids: [UInt32] = [1, 0, 0, 0]
        let fieldBuffer = try #require(context.device.makeBuffer(
            bytes: &fields, length: fields.count * 4, options: .storageModeShared
        ))
        let maskBuffer = try #require(context.device.makeBuffer(
            bytes: &mask, length: mask.count, options: .storageModeShared
        ))
        let idBuffer = try #require(context.device.makeBuffer(
            bytes: &ids, length: ids.count * 4, options: .storageModeShared
        ))
        let packed = try PBRTextureAssembler.packCovered(SparsePBRTextureBake(
            fields: fieldBuffer, faceIDs: idBuffer, mask: maskBuffer,
            width: 2, height: 2
        ))
        #expect(packed.baseColorRGBA == [
            51, 102, 153, 127, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ])
        #expect(packed.metallicRoughnessRGB == [
            0, 76, 204, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ])
        let finalized = try PBRTextureAssembler.finalize(SparsePBRTextureBake(
            fields: fieldBuffer, faceIDs: idBuffer, mask: maskBuffer,
            width: 2, height: 2
        ))
        #expect(finalized.baseColorRGBA == [
            51, 102, 153, 127, 52, 102, 154, 128,
            52, 102, 154, 128, 54, 104, 156, 128,
        ])
        #expect(finalized.metallicRoughnessRGB == [
            0, 76, 204, 0, 76, 204, 0, 76, 204, 0, 76, 204,
        ])
    }

    @Test("GLB embeds reloadable base-color and metallic-roughness PNG materials")
    func glbRoundTrip() throws {
        let atlas = try UVAtlasMesh.preserving(
            positions: [
                SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0),
            ],
            faces: [SIMD3<UInt32>(0, 1, 2)],
            uvs: [SIMD2<Float>(0, 0), SIMD2<Float>(1, 0), SIMD2<Float>(0, 1)]
        )
        let textures = PBRTextureSet(
            baseColorRGBA: [
                255, 0, 0, 255, 0, 255, 0, 255,
                0, 0, 255, 255, 255, 255, 255, 255,
            ],
            metallicRoughnessRGB: [
                0, 64, 128, 0, 64, 128, 0, 64, 128, 0, 64, 128,
            ],
            width: 2, height: 2
        )
        let glb = try PBRGLBWriter.encode(
            atlas: atlas, sourcePositions: atlas.positions,
            sourceFaces: atlas.faces, textures: textures
        )
        let validation = try PBRGLBValidator.validate(data: glb)
        #expect(validation == PBRGLBValidation(
            vertexCount: 3, indexCount: 3, imageCount: 2,
            textureWidth: 2, textureHeight: 2,
            alphaMode: "OPAQUE", doubleSided: true
        ))
        #expect(readUInt32(glb, at: 0) == 0x46546C67)
        #expect(readUInt32(glb, at: 4) == 2)
        #expect(Int(readUInt32(glb, at: 8)) == glb.count)
        let jsonLength = Int(readUInt32(glb, at: 12))
        #expect(readUInt32(glb, at: 16) == 0x4E4F534A)
        let json = glb.subdata(in: 20..<(20 + jsonLength))
        let document = try #require(
            JSONSerialization.jsonObject(with: json) as? [String: Any]
        )
        let materials = try #require(document["materials"] as? [[String: Any]])
        #expect(materials.count == 1)
        #expect(materials[0]["alphaMode"] as? String == "OPAQUE")
        #expect(materials[0]["doubleSided"] as? Bool == true)
        let images = try #require(document["images"] as? [[String: Any]])
        let views = try #require(document["bufferViews"] as? [[String: Any]])
        #expect(images.count == 2)
        let binaryHeader = 20 + jsonLength
        #expect(readUInt32(glb, at: binaryHeader + 4) == 0x004E4942)
        let binaryStart = binaryHeader + 8
        for image in images {
            #expect(image["mimeType"] as? String == "image/png")
            let viewIndex = try #require(image["bufferView"] as? Int)
            let view = views[viewIndex]
            let offset = view["byteOffset"] as? Int ?? 0
            let length = try #require(view["byteLength"] as? Int)
            let png = glb.subdata(in: (binaryStart + offset)..<(binaryStart + offset + length))
            #expect(Array(png.prefix(8)) == [137, 80, 78, 71, 13, 10, 26, 10])
            let source = try #require(CGImageSourceCreateWithData(png as CFData, nil))
            let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
            #expect(image.width == 2 && image.height == 2)
        }
    }
}

private func readUInt32(_ data: Data, at offset: Int) -> UInt32 {
    data[offset..<(offset + 4)].enumerated().reduce(UInt32.zero) {
        $0 | UInt32($1.element) << UInt32($1.offset * 8)
    }
}

private func glbDocument(at url: URL) throws -> [String: Any] {
    let data = try Data(contentsOf: url)
    let jsonLength = Int(readUInt32(data, at: 12))
    return try #require(
        JSONSerialization.jsonObject(
            with: data.subdata(in: 20..<(20 + jsonLength))
        ) as? [String: Any]
    )
}
