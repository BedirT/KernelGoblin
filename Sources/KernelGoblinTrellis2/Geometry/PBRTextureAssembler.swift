import Metal

public struct PBRTextureSet: Equatable, Sendable {
    public let baseColorRGBA: [UInt8]
    public let metallicRoughnessRGB: [UInt8]
    public let width: Int
    public let height: Int

}

public enum PBRTextureAssembler {
    public static func finalize(_ bake: SparsePBRTextureBake) throws -> PBRTextureSet {
        let packed = try packCovered(bake)
        let pixelCount = packed.width * packed.height
        let maskValues = bake.mask.contents().assumingMemoryBound(to: UInt8.self)
        var unknown = [UInt8](repeating: 0, count: pixelCount)
        var covered = 0
        for pixel in 0..<pixelCount {
            if maskValues[pixel] == 0 {
                unknown[pixel] = 1
            } else {
                covered += 1
            }
        }
        guard covered > 0 else {
            throw NativeRuntimeError.invalidArgument("UV raster produced no covered PBR texels")
        }
        var baseRGB = [UInt8](repeating: 0, count: pixelCount * 3)
        var alpha = [UInt8](repeating: 0, count: pixelCount)
        var roughness = [UInt8](repeating: 0, count: pixelCount)
        var metallic = [UInt8](repeating: 0, count: pixelCount)
        for pixel in 0..<pixelCount {
            baseRGB[pixel * 3] = packed.baseColorRGBA[pixel * 4]
            baseRGB[pixel * 3 + 1] = packed.baseColorRGBA[pixel * 4 + 1]
            baseRGB[pixel * 3 + 2] = packed.baseColorRGBA[pixel * 4 + 2]
            alpha[pixel] = packed.baseColorRGBA[pixel * 4 + 3]
            roughness[pixel] = packed.metallicRoughnessRGB[pixel * 3 + 1]
            metallic[pixel] = packed.metallicRoughnessRGB[pixel * 3 + 2]
        }
        baseRGB = try TeleaInpaint.fill(
            baseRGB, mask: unknown, width: packed.width, height: packed.height,
            channels: 3, radius: 3
        )
        alpha = try TeleaInpaint.fill(
            alpha, mask: unknown, width: packed.width, height: packed.height,
            channels: 1, radius: 1
        )
        roughness = try TeleaInpaint.fill(
            roughness, mask: unknown, width: packed.width, height: packed.height,
            channels: 1, radius: 1
        )
        metallic = try TeleaInpaint.fill(
            metallic, mask: unknown, width: packed.width, height: packed.height,
            channels: 1, radius: 1
        )
        var base = [UInt8](repeating: 0, count: pixelCount * 4)
        var material = [UInt8](repeating: 0, count: pixelCount * 3)
        for pixel in 0..<pixelCount {
            base[pixel * 4] = baseRGB[pixel * 3]
            base[pixel * 4 + 1] = baseRGB[pixel * 3 + 1]
            base[pixel * 4 + 2] = baseRGB[pixel * 3 + 2]
            base[pixel * 4 + 3] = alpha[pixel]
            material[pixel * 3] = 0
            material[pixel * 3 + 1] = roughness[pixel]
            material[pixel * 3 + 2] = metallic[pixel]
        }
        return PBRTextureSet(
            baseColorRGBA: base, metallicRoughnessRGB: material,
            width: packed.width, height: packed.height
        )
    }

    public static func packCovered(_ bake: SparsePBRTextureBake) throws -> PBRTextureSet {
        let pixels = bake.width.multipliedReportingOverflow(by: bake.height)
        let elements = pixels.partialValue.multipliedReportingOverflow(by: 6)
        let fieldBytes = elements.partialValue.multipliedReportingOverflow(
            by: MemoryLayout<Float>.stride
        )
        guard bake.width > 0, bake.height > 0,
              !pixels.overflow, !elements.overflow, !fieldBytes.overflow,
              bake.fields.storageMode != .private,
              bake.mask.storageMode != .private,
              bake.fields.length >= fieldBytes.partialValue,
              bake.mask.length >= pixels.partialValue else {
            throw NativeRuntimeError.invalidArgument("invalid sampled PBR texture buffers")
        }
        let baseCount = pixels.partialValue.multipliedReportingOverflow(by: 4)
        let materialCount = pixels.partialValue.multipliedReportingOverflow(by: 3)
        guard !baseCount.overflow, !materialCount.overflow else {
            throw NativeRuntimeError.invalidArgument("PBR packed texture size overflows Int")
        }
        var base = [UInt8](repeating: 0, count: baseCount.partialValue)
        var material = [UInt8](repeating: 0, count: materialCount.partialValue)
        let fields = bake.fields.contents().assumingMemoryBound(to: Float.self)
        let mask = bake.mask.contents().assumingMemoryBound(to: UInt8.self)
        for pixel in 0..<pixels.partialValue where mask[pixel] != 0 {
            base[pixel * 4] = pbrUInt8(fields[pixel * 6])
            base[pixel * 4 + 1] = pbrUInt8(fields[pixel * 6 + 1])
            base[pixel * 4 + 2] = pbrUInt8(fields[pixel * 6 + 2])
            base[pixel * 4 + 3] = pbrUInt8(fields[pixel * 6 + 5])
            material[pixel * 3] = 0
            material[pixel * 3 + 1] = pbrUInt8(fields[pixel * 6 + 4])
            material[pixel * 3 + 2] = pbrUInt8(fields[pixel * 6 + 3])
        }
        return PBRTextureSet(
            baseColorRGBA: base, metallicRoughnessRGB: material,
            width: bake.width, height: bake.height
        )
    }
}

private func pbrUInt8(_ value: Float) -> UInt8 {
    guard value.isFinite else { return 0 }
    let clipped = min(max(value * 255, 0), 255)
    return UInt8(clipped.rounded(.towardZero))
}
