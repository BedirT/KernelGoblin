import Metal

public struct UVRasterResult: @unchecked Sendable {
    public let positions: MTLBuffer
    public let faceIDs: MTLBuffer
    public let mask: MTLBuffer
    public let width: Int
    public let height: Int
}

public final class UVRasterizer: @unchecked Sendable {
    private let context: MetalContext
    private let renderPipeline: MTLRenderPipelineState
    private let compactPipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.library(named: "uv_raster")
        guard let vertex = library.makeFunction(name: "kg_uv_raster_vertex"),
              let fragment = library.makeFunction(name: "kg_uv_raster_fragment"),
              let compact = library.makeFunction(name: "kg_uv_raster_compact") else {
            throw NativeRuntimeError.invalidArgument("UV raster Metal functions are missing")
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment
        descriptor.colorAttachments[0].pixelFormat = .rgba32Float
        descriptor.colorAttachments[1].pixelFormat = .r32Uint
        self.renderPipeline = try context.device.makeRenderPipelineState(
            descriptor: descriptor
        )
        self.compactPipeline = try context.device.makeComputePipelineState(function: compact)
    }

    public func rasterize(
        positions: [SIMD3<Float>],
        uvs: [SIMD2<Float>],
        faces: [SIMD3<UInt32>],
        width: Int,
        height: Int
    ) throws -> UVRasterResult {
        let pixelCount = try uvRasterProduct(width, height)
        let positionBytes = try uvRasterProduct(
            try uvRasterProduct(pixelCount, 3), MemoryLayout<Float>.stride
        )
        let faceBytes = try uvRasterProduct(pixelCount, MemoryLayout<UInt32>.stride)
        guard !positions.isEmpty, positions.count == uvs.count,
              positions.count <= Int(UInt32.max),
              width > 0, height > 0 else {
            throw NativeRuntimeError.invalidArgument("invalid UV raster dimensions")
        }
        for index in positions.indices {
            let position = positions[index]
            let uv = uvs[index]
            guard position.x.isFinite, position.y.isFinite, position.z.isFinite,
                  uv.x.isFinite, uv.y.isFinite,
                  uv.x >= 0, uv.x <= 1, uv.y >= 0, uv.y <= 1 else {
                throw NativeRuntimeError.invalidArgument("UV raster vertices must be finite")
            }
        }
        var flatFaces: [UInt32] = []
        flatFaces.reserveCapacity(try uvRasterProduct(faces.count, 3))
        for face in faces {
            guard Int(face.x) < positions.count,
                  Int(face.y) < positions.count,
                  Int(face.z) < positions.count else {
                throw NativeRuntimeError.invalidArgument("UV raster face index is out of range")
            }
            flatFaces.append(contentsOf: [face.x, face.y, face.z])
        }
        guard flatFaces.count <= Int(UInt32.max) else {
            throw NativeRuntimeError.invalidArgument("UV raster vertex count exceeds UInt32")
        }
        let outputPositions = try context.makeBuffer(
            length: positionBytes, label: "TRELLIS UV raster positions"
        )
        let outputFaceIDs = try context.makeBuffer(
            length: faceBytes, label: "TRELLIS UV raster face IDs"
        )
        let outputMask = try context.makeBuffer(
            length: pixelCount, label: "TRELLIS UV raster mask"
        )
        guard !faces.isEmpty else {
            outputPositions.contents().initializeMemory(as: UInt8.self, repeating: 0, count: positionBytes)
            outputFaceIDs.contents().initializeMemory(as: UInt8.self, repeating: 0, count: faceBytes)
            outputMask.contents().initializeMemory(as: UInt8.self, repeating: 0, count: pixelCount)
            return UVRasterResult(
                positions: outputPositions, faceIDs: outputFaceIDs, mask: outputMask,
                width: width, height: height
            )
        }
        var flatPositions: [Float] = []
        flatPositions.reserveCapacity(try uvRasterProduct(positions.count, 3))
        var flatUVs: [Float] = []
        flatUVs.reserveCapacity(try uvRasterProduct(uvs.count, 2))
        for index in positions.indices {
            flatPositions.append(contentsOf: [
                positions[index].x, positions[index].y, positions[index].z,
            ])
            flatUVs.append(contentsOf: [uvs[index].x, uvs[index].y])
        }
        let positionInput = try uvRasterUpload(
            flatPositions, context: context, label: "TRELLIS UV positions"
        )
        let uvInput = try uvRasterUpload(
            flatUVs, context: context, label: "TRELLIS UV coordinates"
        )
        let faceInput = try uvRasterUpload(
            flatFaces, context: context, label: "TRELLIS UV faces"
        )
        let positionTexture = try uvRasterTexture(
            device: context.device, width: width, height: height,
            format: .rgba32Float, label: "TRELLIS UV position target"
        )
        let idTexture = try uvRasterTexture(
            device: context.device, width: width, height: height,
            format: .r32Uint, label: "TRELLIS UV face-ID target"
        )
        try autoreleasepool {
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = positionTexture
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
            pass.colorAttachments[1].texture = idTexture
            pass.colorAttachments[1].loadAction = .clear
            pass.colorAttachments[1].storeAction = .store
            pass.colorAttachments[1].clearColor = MTLClearColorMake(0, 0, 0, 0)
            guard let command = context.queue.makeCommandBuffer(),
                  let render = command.makeRenderCommandEncoder(descriptor: pass) else {
                throw NativeRuntimeError.allocationFailed("could not create UV raster command")
            }
            command.label = "TRELLIS native UV raster"
            render.setRenderPipelineState(renderPipeline)
            render.setCullMode(.none)
            render.setVertexBuffer(positionInput, offset: 0, index: 0)
            render.setVertexBuffer(uvInput, offset: 0, index: 1)
            render.setVertexBuffer(faceInput, offset: 0, index: 2)
            render.drawPrimitives(
                type: .triangle, vertexStart: 0, vertexCount: flatFaces.count
            )
            render.endEncoding()
            guard let compute = command.makeComputeCommandEncoder() else {
                throw NativeRuntimeError.allocationFailed("could not compact UV raster output")
            }
            compute.setComputePipelineState(compactPipeline)
            compute.setTexture(positionTexture, index: 0)
            compute.setTexture(idTexture, index: 1)
            compute.setBuffer(outputPositions, offset: 0, index: 0)
            compute.setBuffer(outputFaceIDs, offset: 0, index: 1)
            compute.setBuffer(outputMask, offset: 0, index: 2)
            let side = min(
                16,
                max(1, Int(Double(compactPipeline.maxTotalThreadsPerThreadgroup).squareRoot()))
            )
            compute.dispatchThreads(
                MTLSize(width: width, height: height, depth: 1),
                threadsPerThreadgroup: MTLSize(width: side, height: side, depth: 1)
            )
            compute.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            guard command.status == .completed else {
                throw NativeRuntimeError.allocationFailed(
                    "UV raster failed: \(command.error?.localizedDescription ?? "unknown error")"
                )
            }
        }
        return UVRasterResult(
            positions: outputPositions, faceIDs: outputFaceIDs, mask: outputMask,
            width: width, height: height
        )
    }
}

private func uvRasterProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard lhs >= 0, rhs >= 0, !result.overflow else {
        throw NativeRuntimeError.invalidArgument("UV raster size overflows Int")
    }
    return result.partialValue
}

private func uvRasterUpload<T>(
    _ values: [T], context: MetalContext, label: String
) throws -> MTLBuffer {
    let bytes = try uvRasterProduct(values.count, MemoryLayout<T>.stride)
    let buffer = try context.makeBuffer(length: bytes, label: label)
    values.withUnsafeBytes { source in
        buffer.contents().copyMemory(from: source.baseAddress!, byteCount: bytes)
    }
    return buffer
}

private func uvRasterTexture(
    device: MTLDevice, width: Int, height: Int,
    format: MTLPixelFormat, label: String
) throws -> MTLTexture {
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: format, width: width, height: height, mipmapped: false
    )
    descriptor.storageMode = .private
    descriptor.usage = [.renderTarget, .shaderRead]
    guard let texture = device.makeTexture(descriptor: descriptor) else {
        throw NativeRuntimeError.allocationFailed("could not allocate \(label)")
    }
    texture.label = label
    return texture
}
