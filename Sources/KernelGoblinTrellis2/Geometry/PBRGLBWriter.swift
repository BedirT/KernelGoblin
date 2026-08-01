import CoreGraphics
import Foundation
import ImageIO
import simd
import UniformTypeIdentifiers

public struct UVAtlasMesh: Equatable, Sendable {
    public let positions: [SIMD3<Float>]
    public let faces: [SIMD3<UInt32>]
    public let uvs: [SIMD2<Float>]
    public let vertexMap: [UInt32]

    public init(
        positions: [SIMD3<Float>], faces: [SIMD3<UInt32>],
        uvs: [SIMD2<Float>], vertexMap: [UInt32]
    ) throws {
        guard !positions.isEmpty, positions.count == uvs.count,
              positions.count == vertexMap.count, !faces.isEmpty,
              positions.count <= Int(UInt32.max) else {
            throw NativeRuntimeError.invalidArgument("invalid UV atlas mesh shape")
        }
        for index in positions.indices {
            let position = positions[index]
            let uv = uvs[index]
            guard position.x.isFinite, position.y.isFinite, position.z.isFinite,
                  uv.x.isFinite, uv.y.isFinite,
                  uv.x >= 0, uv.x <= 1, uv.y >= 0, uv.y <= 1 else {
                throw NativeRuntimeError.invalidArgument("invalid UV atlas vertex")
            }
        }
        for face in faces {
            guard Int(face.x) < positions.count,
                  Int(face.y) < positions.count,
                  Int(face.z) < positions.count else {
                throw NativeRuntimeError.invalidArgument("UV atlas face is out of range")
            }
        }
        self.positions = positions
        self.faces = faces
        self.uvs = uvs
        self.vertexMap = vertexMap
    }

    public static func preserving(
        positions: [SIMD3<Float>], faces: [SIMD3<UInt32>], uvs: [SIMD2<Float>]
    ) throws -> UVAtlasMesh {
        try UVAtlasMesh(
            positions: positions, faces: faces, uvs: uvs,
            vertexMap: positions.indices.map(UInt32.init)
        )
    }
}

public enum PBRGLBWriter {
    public static func encode(
        atlas: UVAtlasMesh,
        sourcePositions: [SIMD3<Float>],
        sourceFaces: [SIMD3<UInt32>],
        textures: PBRTextureSet,
        alphaMode: String = "OPAQUE",
        doubleSided: Bool = true
    ) throws -> Data {
        guard ["OPAQUE", "BLEND", "MASK"].contains(alphaMode),
              !sourcePositions.isEmpty, !sourceFaces.isEmpty,
              atlas.vertexMap.allSatisfy({ Int($0) < sourcePositions.count }) else {
            throw NativeRuntimeError.invalidArgument("invalid PBR GLB mesh contract")
        }
        let sourceNormals = try computeVertexNormals(
            positions: sourcePositions, faces: sourceFaces
        )
        let normals = atlas.vertexMap.map { sourceNormals[Int($0)] }
        let convertedPositions = atlas.positions.map { SIMD3<Float>($0.x, $0.z, -$0.y) }
        let convertedNormals = normals.map { SIMD3<Float>($0.x, $0.z, -$0.y) }
        let convertedUVs = atlas.uvs.map { SIMD2<Float>($0.x, 1 - $0.y) }
        let basePNG = try encodePNG(
            bytes: textures.baseColorRGBA, width: textures.width,
            height: textures.height, components: 4
        )
        let materialPNG = try encodePNG(
            bytes: textures.metallicRoughnessRGB, width: textures.width,
            height: textures.height, components: 3
        )
        var binary = Data()
        var views: [[String: Any]] = []
        func appendSection(_ data: Data, target: Int? = nil) -> Int {
            while binary.count % 4 != 0 { binary.append(0) }
            let offset = binary.count
            binary.append(data)
            var view: [String: Any] = ["buffer": 0, "byteOffset": offset, "byteLength": data.count]
            if let target { view["target"] = target }
            views.append(view)
            return views.count - 1
        }
        let positionView = appendSection(floatData(convertedPositions), target: 34962)
        let normalView = appendSection(floatData(convertedNormals), target: 34962)
        let uvView = appendSection(floatData(convertedUVs), target: 34962)
        let indexValues = atlas.faces.flatMap { [$0.x, $0.y, $0.z] }
        let indexView = appendSection(integerData(indexValues), target: 34963)
        let baseView = appendSection(basePNG)
        let materialView = appendSection(materialPNG)
        while binary.count % 4 != 0 { binary.append(0) }
        let minimum = SIMD3<Float>(
            convertedPositions.map(\.x).min()!, convertedPositions.map(\.y).min()!,
            convertedPositions.map(\.z).min()!
        )
        let maximum = SIMD3<Float>(
            convertedPositions.map(\.x).max()!, convertedPositions.map(\.y).max()!,
            convertedPositions.map(\.z).max()!
        )
        let accessors: [[String: Any]] = [
            ["bufferView": positionView, "componentType": 5126,
             "count": convertedPositions.count, "type": "VEC3",
             "min": [minimum.x, minimum.y, minimum.z],
             "max": [maximum.x, maximum.y, maximum.z]],
            ["bufferView": normalView, "componentType": 5126,
             "count": convertedNormals.count, "type": "VEC3"],
            ["bufferView": uvView, "componentType": 5126,
             "count": convertedUVs.count, "type": "VEC2"],
            ["bufferView": indexView, "componentType": 5125,
             "count": indexValues.count, "type": "SCALAR"],
        ]
        var material: [String: Any] = [
            "name": "TRELLIS.2 PBR",
            "pbrMetallicRoughness": [
                "baseColorFactor": [1, 1, 1, 1],
                "baseColorTexture": ["index": 0],
                "metallicFactor": 1,
                "roughnessFactor": 1,
                "metallicRoughnessTexture": ["index": 1],
            ],
            "alphaMode": alphaMode,
            "doubleSided": doubleSided,
        ]
        if alphaMode == "MASK" { material["alphaCutoff"] = 0.5 }
        let document: [String: Any] = [
            "asset": ["version": "2.0", "generator": "KernelGoblin Swift+Metal"],
            "scene": 0,
            "scenes": [["nodes": [0]]],
            "nodes": [["mesh": 0]],
            "meshes": [["name": "TRELLIS.2 mesh", "primitives": [[
                "attributes": ["POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2],
                "indices": 3, "material": 0,
            ]]]],
            "materials": [material],
            "textures": [["sampler": 0, "source": 0], ["sampler": 0, "source": 1]],
            "samplers": [["magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497]],
            "images": [
                ["bufferView": baseView, "mimeType": "image/png", "name": "baseColor"],
                ["bufferView": materialView, "mimeType": "image/png", "name": "metallicRoughness"],
            ],
            "accessors": accessors,
            "bufferViews": views,
            "buffers": [["byteLength": binary.count]],
        ]
        var json = try JSONSerialization.data(withJSONObject: document, options: [.sortedKeys])
        while json.count % 4 != 0 { json.append(0x20) }
        var result = Data()
        appendUInt32(0x46546C67, to: &result)
        appendUInt32(2, to: &result)
        appendUInt32(UInt32(12 + 8 + json.count + 8 + binary.count), to: &result)
        appendUInt32(UInt32(json.count), to: &result)
        appendUInt32(0x4E4F534A, to: &result)
        result.append(json)
        appendUInt32(UInt32(binary.count), to: &result)
        appendUInt32(0x004E4942, to: &result)
        result.append(binary)
        return result
    }

    public static func write(
        to url: URL, atlas: UVAtlasMesh,
        sourcePositions: [SIMD3<Float>], sourceFaces: [SIMD3<UInt32>],
        textures: PBRTextureSet, alphaMode: String = "OPAQUE",
        doubleSided: Bool = true
    ) throws {
        try encode(
            atlas: atlas, sourcePositions: sourcePositions, sourceFaces: sourceFaces,
            textures: textures, alphaMode: alphaMode, doubleSided: doubleSided
        ).write(to: url, options: .atomic)
    }
}

private func computeVertexNormals(
    positions: [SIMD3<Float>], faces: [SIMD3<UInt32>]
) throws -> [SIMD3<Float>] {
    var normals = [SIMD3<Float>](repeating: .zero, count: positions.count)
    for face in faces {
        let a = Int(face.x), b = Int(face.y), c = Int(face.z)
        guard a < positions.count, b < positions.count, c < positions.count else {
            throw NativeRuntimeError.invalidArgument("source face is out of range")
        }
        let normal = simd_cross(positions[b] - positions[a], positions[c] - positions[a])
        guard normal.x.isFinite, normal.y.isFinite, normal.z.isFinite else {
            throw NativeRuntimeError.invalidArgument("source geometry is non-finite")
        }
        normals[a] += normal
        normals[b] += normal
        normals[c] += normal
    }
    return normals.map {
        let length = simd_length($0)
        return length > Float.ulpOfOne ? $0 / length : SIMD3<Float>(0, 0, 1)
    }
}

private func encodePNG(
    bytes: [UInt8], width: Int, height: Int, components: Int
) throws -> Data {
    let count = width.multipliedReportingOverflow(by: height)
    let expected = count.partialValue.multipliedReportingOverflow(by: components)
    guard width > 0, height > 0, [3, 4].contains(components),
          !count.overflow, !expected.overflow, bytes.count == expected.partialValue else {
        throw NativeRuntimeError.invalidArgument("invalid PBR image dimensions")
    }
    let data = Data(bytes)
    guard let provider = CGDataProvider(data: data as CFData),
          let image = CGImage(
            width: width, height: height, bitsPerComponent: 8,
            bitsPerPixel: components * 8, bytesPerRow: width * components,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: components == 4
                ? CGImageAlphaInfo.last.rawValue : CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false,
            intent: .defaultIntent
          ) else {
        throw NativeRuntimeError.invalidArgument("could not construct PBR image")
    }
    let output = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(
        output, UTType.png.identifier as CFString, 1, nil
    ) else {
        throw NativeRuntimeError.allocationFailed("could not create PNG encoder")
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
        throw NativeRuntimeError.allocationFailed("could not encode PBR PNG")
    }
    return output as Data
}

private func floatData(_ values: [SIMD3<Float>]) -> Data {
    floatData(values.flatMap { [$0.x, $0.y, $0.z] })
}

private func floatData(_ values: [SIMD2<Float>]) -> Data {
    floatData(values.flatMap { [$0.x, $0.y] })
}

private func floatData(_ values: [Float]) -> Data {
    var result = Data(capacity: values.count * 4)
    for value in values { appendUInt32(value.bitPattern, to: &result) }
    return result
}

private func integerData(_ values: [UInt32]) -> Data {
    var result = Data(capacity: values.count * 4)
    for value in values { appendUInt32(value, to: &result) }
    return result
}

private func appendUInt32(_ value: UInt32, to data: inout Data) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
}
