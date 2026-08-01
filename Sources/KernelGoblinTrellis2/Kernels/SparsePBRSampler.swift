import Metal

private struct SparsePBRLookupEntry {
    var key: UInt64
    var row: UInt32
    var padding: UInt32 = 0
}

public struct SparsePBRLookup: @unchecked Sendable {
    let buffer: MTLBuffer
    let entryCount: Int
    let coordinateCount: Int
    let spatialShape: SparseSpatialShape
}

public final class SparsePBRSampler: @unchecked Sendable {
    private struct Parameters {
        var queryCount: UInt32
        var channels: UInt32
        var width: UInt32
        var height: UInt32
        var depth: UInt32
        var entryCount: UInt32
        var originX: Float
        var originY: Float
        var originZ: Float
        var inverseVoxelX: Float
        var inverseVoxelY: Float
        var inverseVoxelZ: Float
        var hasMask: UInt32
    }

    private let context: MetalContext
    private let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.library(named: "sparse_pbr_sample")
        guard let function = library.makeFunction(name: "kg_sparse_pbr_trilinear_f32") else {
            throw NativeRuntimeError.invalidArgument("sparse PBR sampler function is missing")
        }
        self.pipeline = try context.device.makeComputePipelineState(function: function)
    }

    public func sampleTrilinearF32(
        features: MTLBuffer,
        coordinates: [SparseStructureCoordinate],
        spatialShape: SparseSpatialShape,
        positions: MTLBuffer,
        queryCount: Int,
        channels: Int,
        output: MTLBuffer
    ) throws {
        let lookup = try makeLookup(
            coordinates: coordinates, spatialShape: spatialShape
        )
        try sampleTrilinearF32(
            features: features, lookup: lookup, positions: positions,
            queryCount: queryCount, channels: channels, output: output
        )
    }

    public func makeLookup(
        coordinates: [SparseStructureCoordinate],
        spatialShape: SparseSpatialShape
    ) throws -> SparsePBRLookup {
        _ = try sparsePBRVolume(spatialShape)
        guard coordinates.count <= Int(UInt32.max) else {
            throw NativeRuntimeError.invalidArgument("too many sparse PBR coordinates")
        }
        var entries: [SparsePBRLookupEntry] = []
        entries.reserveCapacity(coordinates.count)
        for (index, coordinate) in coordinates.enumerated() {
            guard coordinate.batch == 0,
                  coordinate.x >= 0, coordinate.x < spatialShape.width,
                  coordinate.y >= 0, coordinate.y < spatialShape.height,
                  coordinate.z >= 0, coordinate.z < spatialShape.depth else {
                throw NativeRuntimeError.invalidArgument(
                    "sparse PBR coordinates must be in-bounds batch zero"
                )
            }
            let key = try sparsePBRKey(
                x: coordinate.x, y: coordinate.y, z: coordinate.z,
                spatialShape: spatialShape
            )
            entries.append(SparsePBRLookupEntry(key: key, row: UInt32(index)))
        }
        entries.sort { $0.key < $1.key }
        if entries.count > 1 {
            for index in 1..<entries.count where entries[index - 1].key == entries[index].key {
                throw NativeRuntimeError.invalidArgument("sparse PBR coordinates must be unique")
            }
        }
        let allocatedCount = max(entries.count, 1)
        let entryBytes = try sparsePBRSampleProduct(
            allocatedCount, MemoryLayout<SparsePBRLookupEntry>.stride
        )
        let buffer = try context.makeBuffer(
            length: entryBytes, label: "TRELLIS sparse PBR lookup"
        )
        if entries.isEmpty {
            buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: entryBytes)
        } else {
            entries.withUnsafeBytes { bytes in
                buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: entryBytes)
            }
        }
        return SparsePBRLookup(
            buffer: buffer, entryCount: entries.count,
            coordinateCount: coordinates.count, spatialShape: spatialShape
        )
    }

    public func sampleTrilinearF32(
        features: MTLBuffer,
        lookup: SparsePBRLookup,
        positions: MTLBuffer,
        queryCount: Int,
        channels: Int,
        output: MTLBuffer
    ) throws {
        try sampleTrilinearF32(
            features: features, lookup: lookup, positions: positions,
            mask: nil, queryCount: queryCount, channels: channels,
            origin: .zero, inverseVoxel: SIMD3<Float>(repeating: 1), output: output
        )
    }

    public func sampleSurfaceTrilinearF32(
        features: MTLBuffer,
        lookup: SparsePBRLookup,
        objectPositions: MTLBuffer,
        mask: MTLBuffer,
        queryCount: Int,
        channels: Int,
        aabbMinimum: SIMD3<Float>,
        voxelSize: SIMD3<Float>,
        output: MTLBuffer
    ) throws {
        guard aabbMinimum.x.isFinite, aabbMinimum.y.isFinite, aabbMinimum.z.isFinite,
              voxelSize.x.isFinite, voxelSize.y.isFinite, voxelSize.z.isFinite,
              voxelSize.x > 0, voxelSize.y > 0, voxelSize.z > 0 else {
            throw NativeRuntimeError.invalidArgument("invalid sparse PBR object-to-grid transform")
        }
        let inverseVoxel = SIMD3<Float>(
            1 / voxelSize.x, 1 / voxelSize.y, 1 / voxelSize.z
        )
        guard inverseVoxel.x.isFinite, inverseVoxel.y.isFinite,
              inverseVoxel.z.isFinite else {
            throw NativeRuntimeError.invalidArgument("sparse PBR inverse voxel size is not finite")
        }
        try sampleTrilinearF32(
            features: features, lookup: lookup, positions: objectPositions,
            mask: mask, queryCount: queryCount, channels: channels,
            origin: aabbMinimum,
            inverseVoxel: inverseVoxel,
            output: output
        )
    }

    private func sampleTrilinearF32(
        features: MTLBuffer,
        lookup: SparsePBRLookup,
        positions: MTLBuffer,
        mask: MTLBuffer?,
        queryCount: Int,
        channels: Int,
        origin: SIMD3<Float>,
        inverseVoxel: SIMD3<Float>,
        output: MTLBuffer
    ) throws {
        _ = try sparsePBRVolume(lookup.spatialShape)
        let featureElements = try sparsePBRSampleProduct(lookup.coordinateCount, channels)
        let featureBytes = try sparsePBRSampleProduct(
            featureElements, MemoryLayout<Float>.stride
        )
        let positionElements = try sparsePBRSampleProduct(queryCount, 3)
        let positionBytes = try sparsePBRSampleProduct(
            positionElements, MemoryLayout<Float>.stride
        )
        let outputElements = try sparsePBRSampleProduct(queryCount, channels)
        let outputBytes = try sparsePBRSampleProduct(
            outputElements, MemoryLayout<Float>.stride
        )
        guard queryCount >= 0, queryCount <= Int(UInt32.max),
              channels > 0, channels <= 16,
              lookup.entryCount <= Int(UInt32.max),
              lookup.spatialShape.width <= Int(UInt32.max),
              lookup.spatialShape.height <= Int(UInt32.max),
              lookup.spatialShape.depth <= Int(UInt32.max),
              features.length >= featureBytes,
              positions.length >= positionBytes,
              mask == nil || mask!.length >= queryCount,
              output.length >= outputBytes,
              output !== features, output !== positions,
              output !== lookup.buffer, output !== mask else {
            throw NativeRuntimeError.invalidArgument("invalid sparse PBR sampling buffers")
        }
        if queryCount == 0 { return }
        var parameters = Parameters(
            queryCount: UInt32(queryCount), channels: UInt32(channels),
            width: UInt32(lookup.spatialShape.width),
            height: UInt32(lookup.spatialShape.height),
            depth: UInt32(lookup.spatialShape.depth),
            entryCount: UInt32(lookup.entryCount),
            originX: origin.x, originY: origin.y, originZ: origin.z,
            inverseVoxelX: inverseVoxel.x,
            inverseVoxelY: inverseVoxel.y,
            inverseVoxelZ: inverseVoxel.z,
            hasMask: mask == nil ? 0 : 1
        )
        try context.runCompute(label: "TRELLIS sparse PBR trilinear sampling") { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(features, offset: 0, index: 0)
            encoder.setBuffer(lookup.buffer, offset: 0, index: 1)
            encoder.setBuffer(positions, offset: 0, index: 2)
            encoder.setBuffer(output, offset: 0, index: 3)
            encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 4)
            encoder.setBuffer(mask ?? lookup.buffer, offset: 0, index: 5)
            let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(
                MTLSize(width: queryCount, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
            )
        }
    }
}

private func sparsePBRVolume(_ spatialShape: SparseSpatialShape) throws -> UInt64 {
    let widthHeight = UInt64(spatialShape.width).multipliedReportingOverflow(
        by: UInt64(spatialShape.height)
    )
    let volume = widthHeight.partialValue.multipliedReportingOverflow(
        by: UInt64(spatialShape.depth)
    )
    guard !widthHeight.overflow, !volume.overflow,
          volume.partialValue <= UInt64(Int64.max) else {
        throw NativeRuntimeError.invalidArgument("sparse PBR key space exceeds Int64")
    }
    return volume.partialValue
}

private func sparsePBRKey(
    x: Int32, y: Int32, z: Int32, spatialShape: SparseSpatialShape
) throws -> UInt64 {
    let xHeight = UInt64(x).multipliedReportingOverflow(
        by: UInt64(spatialShape.height)
    )
    let xy = xHeight.partialValue.addingReportingOverflow(UInt64(y))
    let xyDepth = xy.partialValue.multipliedReportingOverflow(
        by: UInt64(spatialShape.depth)
    )
    let key = xyDepth.partialValue.addingReportingOverflow(UInt64(z))
    guard !xHeight.overflow, !xy.overflow, !xyDepth.overflow, !key.overflow else {
        throw NativeRuntimeError.invalidArgument("sparse PBR coordinate key overflows UInt64")
    }
    return key.partialValue
}

private func sparsePBRSampleProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard lhs >= 0, rhs >= 0, !result.overflow else {
        throw NativeRuntimeError.invalidArgument("sparse PBR tensor size overflows Int")
    }
    return result.partialValue
}
