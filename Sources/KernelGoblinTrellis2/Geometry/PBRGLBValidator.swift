import Foundation
import ImageIO

public struct PBRGLBValidation: Codable, Equatable, Sendable {
    public let vertexCount: Int
    public let indexCount: Int
    public let imageCount: Int
    public let textureWidth: Int
    public let textureHeight: Int
    public let alphaMode: String
    public let doubleSided: Bool
}

public enum PBRGLBValidator {
    public static func validate(url: URL) throws -> PBRGLBValidation {
        try validate(data: Data(contentsOf: url, options: [.mappedIfSafe]))
    }

    public static func validate(data: Data) throws -> PBRGLBValidation {
        guard data.count >= 28,
              try word(data, 0) == 0x46546C67,
              try word(data, 4) == 2,
              Int(try word(data, 8)) == data.count else {
            throw NativeRuntimeError.invalidArgument("invalid GLB header")
        }
        let jsonLength = Int(try word(data, 12))
        guard try word(data, 16) == 0x4E4F534A,
              jsonLength >= 2,
              jsonLength <= data.count - 28 else {
            throw NativeRuntimeError.invalidArgument("invalid GLB JSON chunk")
        }
        let binaryHeader = try checkedSum(20, jsonLength)
        let binaryLength = Int(try word(data, binaryHeader))
        guard try word(data, binaryHeader + 4) == 0x004E4942 else {
            throw NativeRuntimeError.invalidArgument("missing GLB binary chunk")
        }
        let binaryStart = try checkedSum(binaryHeader, 8)
        guard binaryLength == data.count - binaryStart else {
            throw NativeRuntimeError.invalidArgument("invalid GLB binary chunk length")
        }
        let object = try JSONSerialization.jsonObject(
            with: data.subdata(in: 20..<(20 + jsonLength))
        )
        guard let document = object as? [String: Any],
              let asset = document["asset"] as? [String: Any],
              asset["version"] as? String == "2.0",
              let buffers = document["buffers"] as? [[String: Any]],
              buffers.count == 1,
              buffers[0]["byteLength"] as? Int == binaryLength,
              let views = document["bufferViews"] as? [[String: Any]],
              let accessors = document["accessors"] as? [[String: Any]],
              accessors.count >= 3,
              let meshes = document["meshes"] as? [[String: Any]],
              meshes.count == 1,
              let primitives = meshes[0]["primitives"] as? [[String: Any]],
              primitives.count == 1,
              let attributes = primitives[0]["attributes"] as? [String: Any],
              attributes["POSITION"] as? Int == 0,
              attributes["NORMAL"] as? Int == 1,
              let indexAccessor = primitives[0]["indices"] as? Int,
              accessors.indices.contains(indexAccessor),
              primitives[0]["material"] as? Int == 0 else {
            throw NativeRuntimeError.invalidArgument("invalid GLB scene or mesh contract")
        }
        for view in views {
            guard view["buffer"] as? Int == 0,
                  let length = view["byteLength"] as? Int,
                  length >= 0 else {
                throw NativeRuntimeError.invalidArgument("invalid GLB buffer view")
            }
            let offset = view["byteOffset"] as? Int ?? 0
            let end = offset.addingReportingOverflow(length)
            guard offset >= 0, !end.overflow, end.partialValue <= binaryLength else {
                throw NativeRuntimeError.invalidArgument("GLB buffer view is out of range")
            }
        }
        for accessor in accessors {
            guard let viewIndex = accessor["bufferView"] as? Int,
                  views.indices.contains(viewIndex),
                  let count = accessor["count"] as? Int,
                  count >= 0,
                  let componentType = accessor["componentType"] as? Int,
                  let type = accessor["type"] as? String,
                  let componentBytes = [5125: 4, 5126: 4][componentType],
                  let componentCount = ["SCALAR": 1, "VEC2": 2, "VEC3": 3][type],
                  let viewLength = views[viewIndex]["byteLength"] as? Int else {
                throw NativeRuntimeError.invalidArgument("invalid GLB accessor layout")
            }
            let required = count.multipliedReportingOverflow(
                by: componentBytes * componentCount
            )
            let accessorOffset = accessor["byteOffset"] as? Int ?? 0
            let accessorEnd = accessorOffset.addingReportingOverflow(required.partialValue)
            guard !required.overflow, accessorOffset >= 0, !accessorEnd.overflow,
                  accessorEnd.partialValue <= viewLength else {
                throw NativeRuntimeError.invalidArgument("GLB accessor exceeds its buffer view")
            }
        }
        let uvAccessor = attributes["TEXCOORD_0"] as? Int
        guard accessors[0]["componentType"] as? Int == 5126,
              accessors[0]["type"] as? String == "VEC3",
              accessors[1]["componentType"] as? Int == 5126,
              accessors[1]["type"] as? String == "VEC3",
              accessors[indexAccessor]["componentType"] as? Int == 5125,
              accessors[indexAccessor]["type"] as? String == "SCALAR",
              let vertexCount = accessors[0]["count"] as? Int,
              let normalCount = accessors[1]["count"] as? Int,
              let indexCount = accessors[indexAccessor]["count"] as? Int,
              vertexCount > 0, normalCount == vertexCount,
              indexCount > 0, indexCount % 3 == 0 else {
            throw NativeRuntimeError.invalidArgument("invalid GLB accessor contract")
        }
        if let uvAccessor {
            guard accessors.indices.contains(uvAccessor),
                  accessors[uvAccessor]["componentType"] as? Int == 5126,
                  accessors[uvAccessor]["type"] as? String == "VEC2",
                  accessors[uvAccessor]["count"] as? Int == vertexCount else {
                throw NativeRuntimeError.invalidArgument("invalid GLB UV accessor")
            }
        }
        let indices = try uint32Accessor(
            accessors[indexAccessor], views: views, data: data,
            binaryStart: binaryStart
        )
        guard indices.allSatisfy({ Int($0) < vertexCount }) else {
            throw NativeRuntimeError.invalidArgument("GLB index is out of range")
        }
        guard let materials = document["materials"] as? [[String: Any]],
              materials.count == 1,
              let pbr = materials[0]["pbrMetallicRoughness"] as? [String: Any],
              let alphaMode = materials[0]["alphaMode"] as? String,
              ["OPAQUE", "BLEND", "MASK"].contains(alphaMode),
              let doubleSided = materials[0]["doubleSided"] as? Bool else {
            throw NativeRuntimeError.invalidArgument("invalid GLB material contract")
        }
        let images = document["images"] as? [[String: Any]] ?? []
        if images.isEmpty {
            guard uvAccessor == nil,
                  pbr["baseColorTexture"] == nil,
                  pbr["metallicRoughnessTexture"] == nil,
                  numericArray(pbr["baseColorFactor"]) == [1, 1, 1, 1],
                  number(pbr["metallicFactor"]) == 0,
                  number(pbr["roughnessFactor"]) == 1,
                  document["textures"] == nil,
                  document["samplers"] == nil else {
                throw NativeRuntimeError.invalidArgument(
                    "geometry-only GLB references missing textures"
                )
            }
            return PBRGLBValidation(
                vertexCount: vertexCount, indexCount: indexCount,
                imageCount: 0, textureWidth: 0, textureHeight: 0,
                alphaMode: alphaMode, doubleSided: doubleSided
            )
        }
        guard uvAccessor != nil,
              images.count == 2,
              let base = pbr["baseColorTexture"] as? [String: Any],
              base["index"] as? Int == 0,
              let packed = pbr["metallicRoughnessTexture"] as? [String: Any],
              packed["index"] as? Int == 1 else {
            throw NativeRuntimeError.invalidArgument("invalid GLB PBR material contract")
        }
        var dimensions: (Int, Int)?
        for image in images {
            guard image["mimeType"] as? String == "image/png",
                  let viewIndex = image["bufferView"] as? Int,
                  views.indices.contains(viewIndex),
                  let length = views[viewIndex]["byteLength"] as? Int else {
                throw NativeRuntimeError.invalidArgument("invalid embedded GLB image")
            }
            let offset = views[viewIndex]["byteOffset"] as? Int ?? 0
            let start = try checkedSum(binaryStart, offset)
            let end = try checkedSum(start, length)
            guard end <= data.count else {
                throw NativeRuntimeError.invalidArgument("embedded GLB image is out of range")
            }
            let png = data.subdata(in: start..<end)
            guard png.starts(with: [137, 80, 78, 71, 13, 10, 26, 10]),
                  let source = CGImageSourceCreateWithData(png as CFData, nil),
                  let decoded = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                throw NativeRuntimeError.invalidArgument("embedded GLB PNG cannot be decoded")
            }
            let current = (decoded.width, decoded.height)
            if let dimensions {
                guard dimensions == current else {
                    throw NativeRuntimeError.invalidArgument(
                        "embedded GLB textures have different dimensions"
                    )
                }
            } else {
                dimensions = current
            }
        }
        guard let dimensions else {
            throw NativeRuntimeError.invalidArgument("GLB has no embedded texture dimensions")
        }
        return PBRGLBValidation(
            vertexCount: vertexCount, indexCount: indexCount,
            imageCount: images.count, textureWidth: dimensions.0,
            textureHeight: dimensions.1, alphaMode: alphaMode,
            doubleSided: doubleSided
        )
    }
}

private func word(_ data: Data, _ offset: Int) throws -> UInt32 {
    guard offset >= 0, offset <= data.count - 4 else {
        throw NativeRuntimeError.invalidArgument("truncated GLB word")
    }
    return data[offset..<(offset + 4)].enumerated().reduce(UInt32.zero) {
        $0 | UInt32($1.element) << UInt32($1.offset * 8)
    }
}

private func checkedSum(_ lhs: Int, _ rhs: Int) throws -> Int {
    let value = lhs.addingReportingOverflow(rhs)
    guard lhs >= 0, rhs >= 0, !value.overflow else {
        throw NativeRuntimeError.invalidArgument("GLB offset overflows Int")
    }
    return value.partialValue
}

private func uint32Accessor(
    _ accessor: [String: Any],
    views: [[String: Any]],
    data: Data,
    binaryStart: Int
) throws -> [UInt32] {
    guard let viewIndex = accessor["bufferView"] as? Int,
          views.indices.contains(viewIndex),
          let count = accessor["count"] as? Int else {
        throw NativeRuntimeError.invalidArgument("invalid GLB index accessor")
    }
    let viewOffset = views[viewIndex]["byteOffset"] as? Int ?? 0
    let accessorOffset = accessor["byteOffset"] as? Int ?? 0
    let start = try checkedSum(binaryStart, try checkedSum(viewOffset, accessorOffset))
    let byteCount = count.multipliedReportingOverflow(by: 4)
    guard !byteCount.overflow else {
        throw NativeRuntimeError.invalidArgument("GLB index accessor overflows Int")
    }
    let end = try checkedSum(start, byteCount.partialValue)
    guard end <= data.count else {
        throw NativeRuntimeError.invalidArgument("GLB index accessor is out of range")
    }
    return stride(from: start, to: end, by: 4).map { offset in
        data[offset..<(offset + 4)].enumerated().reduce(UInt32.zero) {
            $0 | UInt32($1.element) << UInt32($1.offset * 8)
        }
    }
}

private func number(_ value: Any?) -> Double? {
    (value as? NSNumber)?.doubleValue
}

private func numericArray(_ value: Any?) -> [Double]? {
    (value as? [Any])?.compactMap(number)
}
