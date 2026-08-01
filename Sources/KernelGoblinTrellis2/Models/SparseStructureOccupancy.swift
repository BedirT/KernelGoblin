import Foundation

public struct SparseStructureCoordinate: Equatable, Hashable, Sendable {
    public let batch: Int32
    public let x: Int32
    public let y: Int32
    public let z: Int32

    public init(batch: Int32 = 0, x: Int32, y: Int32, z: Int32) {
        self.batch = batch
        self.x = x
        self.y = y
        self.z = z
    }
}

public struct SparseOccupancyGrid: Equatable, Sendable {
    public let resolution: Int
    public let packedBits: [UInt8]

    public init(resolution: Int, packedBits: [UInt8]) throws {
        let voxels = try sparseVoxelCount(resolution)
        guard packedBits.count == (try sparsePackedByteCount(voxels)) else {
            throw NativeRuntimeError.invalidArgument(
                "occupancy bit count does not match its resolution"
            )
        }
        self.resolution = resolution
        self.packedBits = packedBits
    }

    public func contains(x: Int, y: Int, z: Int) -> Bool {
        guard x >= 0, y >= 0, z >= 0,
              x < resolution, y < resolution, z < resolution else {
            return false
        }
        let index = (x * resolution + y) * resolution + z
        return packedBits[index >> 3] & (1 << UInt8(index & 7)) != 0
    }

    public func coordinates(batch: Int32 = 0) -> [SparseStructureCoordinate] {
        var result: [SparseStructureCoordinate] = []
        for x in 0..<resolution {
            for y in 0..<resolution {
                for z in 0..<resolution where contains(x: x, y: y, z: z) {
                    result.append(SparseStructureCoordinate(
                        batch: batch, x: Int32(x), y: Int32(y), z: Int32(z)
                    ))
                }
            }
        }
        return result
    }
}

public enum SparseStructureOccupancy {
    public static func threshold(
        logits: [Float], resolution: Int
    ) throws -> SparseOccupancyGrid {
        let voxels = try sparseVoxelCount(resolution)
        guard logits.count == voxels else {
            throw NativeRuntimeError.invalidArgument(
                "sparse-structure logits do not match their resolution"
            )
        }
        var bits = [UInt8](repeating: 0, count: try sparsePackedByteCount(voxels))
        for index in logits.indices where logits[index] > 0 {
            bits[index >> 3] |= 1 << UInt8(index & 7)
        }
        return try SparseOccupancyGrid(resolution: resolution, packedBits: bits)
    }

    public static func downsampleMax2(
        _ source: SparseOccupancyGrid
    ) throws -> SparseOccupancyGrid {
        guard source.resolution.isMultiple(of: 2) else {
            throw NativeRuntimeError.invalidArgument(
                "2x occupancy downsampling requires an even resolution"
            )
        }
        let resolution = source.resolution / 2
        let voxels = try sparseVoxelCount(resolution)
        var bits = [UInt8](repeating: 0, count: try sparsePackedByteCount(voxels))
        for x in 0..<resolution {
            for y in 0..<resolution {
                for z in 0..<resolution {
                    var occupied = false
                    for dx in 0..<2 {
                        for dy in 0..<2 {
                            for dz in 0..<2 where source.contains(
                                x: x * 2 + dx, y: y * 2 + dy, z: z * 2 + dz
                            ) {
                                occupied = true
                            }
                        }
                    }
                    if occupied {
                        let index = (x * resolution + y) * resolution + z
                        bits[index >> 3] |= 1 << UInt8(index & 7)
                    }
                }
            }
        }
        return try SparseOccupancyGrid(resolution: resolution, packedBits: bits)
    }
}

private func sparseVoxelCount(_ resolution: Int) throws -> Int {
    let squared = resolution.multipliedReportingOverflow(by: resolution)
    let cubed = squared.partialValue.multipliedReportingOverflow(by: resolution)
    guard resolution > 0, !squared.overflow, !cubed.overflow else {
        throw NativeRuntimeError.invalidArgument(
            "sparse occupancy resolution overflows Int"
        )
    }
    return cubed.partialValue
}

private func sparsePackedByteCount(_ voxels: Int) throws -> Int {
    let rounded = voxels.addingReportingOverflow(7)
    guard voxels > 0, !rounded.overflow else {
        throw NativeRuntimeError.invalidArgument(
            "sparse occupancy bit count overflows Int"
        )
    }
    return rounded.partialValue / 8
}
