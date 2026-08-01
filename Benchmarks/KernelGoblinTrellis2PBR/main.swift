import Foundation
import KernelGoblinTrellis2
import Metal

struct Options {
    var textureSize = 2048
    var resolution = 64
    var warmup = 5
    var iterations = 20

    init(arguments: [String]) throws {
        var index = 1
        while index < arguments.count {
            guard index + 1 < arguments.count, let value = Int(arguments[index + 1]) else {
                throw NativeRuntimeError.invalidArgument("benchmark options require integer values")
            }
            switch arguments[index] {
            case "--texture-size": textureSize = value
            case "--resolution": resolution = value
            case "--warmup": warmup = value
            case "--iterations": iterations = value
            default:
                throw NativeRuntimeError.invalidArgument("unknown option \(arguments[index])")
            }
            index += 2
        }
        guard textureSize > 0, resolution > 0,
              resolution <= Int(Int32.max), warmup >= 0, iterations > 0 else {
            throw NativeRuntimeError.invalidArgument("benchmark values must be positive")
        }
    }
}

func quantileR7(_ sorted: [Double], _ fraction: Double) -> Double {
    let position = Double(sorted.count - 1) * fraction
    let lower = Int(position.rounded(.down))
    let upper = Int(position.rounded(.up))
    let weight = position - Double(lower)
    return sorted[lower] * (1 - weight) + sorted[upper] * weight
}

func run() throws {
    let options = try Options(arguments: CommandLine.arguments)
    let arenaCapacity = 384 * 1024 * 1024
    let context = try MetalContext(arenaCapacity: arenaCapacity)
    let baker = try SparsePBRTextureBaker(context: context)
    let shape = try SparseSpatialShape(cubic: options.resolution)
    let planeCount = try checkedProduct(options.resolution, options.resolution)
    let tokenCount = try checkedProduct(planeCount, options.resolution)
    let fieldCount = try checkedProduct(tokenCount, 6)
    var coordinates: [SparseStructureCoordinate] = []
    coordinates.reserveCapacity(tokenCount)
    var fields = [Float]()
    fields.reserveCapacity(fieldCount)
    let denominator = Float(max(options.resolution - 1, 1))
    for x in 0..<options.resolution {
        for y in 0..<options.resolution {
            for z in 0..<options.resolution {
                coordinates.append(SparseStructureCoordinate(
                    x: Int32(x), y: Int32(y), z: Int32(z)
                ))
                let fx = options.resolution == 1 ? 0 : Float(x) / denominator
                let fy = options.resolution == 1 ? 0 : Float(y) / denominator
                let fz = options.resolution == 1 ? 0 : Float(z) / denominator
                fields.append(contentsOf: [fx, fy, fz, (fx + fy + fz) / 3, 0.3, 0.5])
            }
        }
    }
    let fieldBytes = try checkedProduct(fieldCount, MemoryLayout<Float>.stride)
    guard let fieldBuffer = context.device.makeBuffer(
        bytes: &fields, length: fieldBytes,
        options: .storageModeShared
    ) else {
        throw NativeRuntimeError.allocationFailed("could not allocate benchmark fields")
    }
    let positions = [
        SIMD3<Float>(-0.5, -0.5, 0), SIMD3<Float>(0.5, -0.5, 0),
        SIMD3<Float>(0.5, 0.5, 0), SIMD3<Float>(-0.5, 0.5, 0),
    ]
    let uvs = [
        SIMD2<Float>(0, 0), SIMD2<Float>(1, 0),
        SIMD2<Float>(1, 1), SIMD2<Float>(0, 1),
    ]
    let faces = [SIMD3<UInt32>(0, 1, 2), SIMD3<UInt32>(0, 2, 3)]
    func execute() throws -> SparsePBRTextureBake {
        try baker.bake(
            positions: positions, uvs: uvs, faces: faces,
            decodedPBRFields: fieldBuffer, fieldCoordinates: coordinates,
            spatialShape: shape, aabbMinimum: SIMD3<Float>(repeating: -0.5),
            voxelSize: SIMD3<Float>(repeating: 1 / Float(options.resolution)),
            width: options.textureSize, height: options.textureSize
        )
    }
    func verify(_ output: SparsePBRTextureBake) throws {
        let pixelCount = try checkedProduct(options.textureSize, options.textureSize)
        let mask = output.mask.contents().assumingMemoryBound(to: UInt8.self)
        let values = output.fields.contents().assumingMemoryBound(to: Float.self)
        var hash: UInt64 = 1_469_598_103_934_665_603
        for pixel in 0..<pixelCount {
            guard mask[pixel] == 1 else {
                throw NativeRuntimeError.invalidArgument("benchmark quad has an uncovered texel")
            }
            let x = pixel % options.textureSize
            let y = pixel / options.textureSize
            let gridX = (Float(x) + 0.5) / Float(options.textureSize)
                * Float(options.resolution)
            let gridY = (Float(y) + 0.5) / Float(options.textureSize)
                * Float(options.resolution)
            let expectedX = options.resolution == 1 ? 0
                : min(max((gridX - 0.5) / denominator, 0), 1)
            let expectedY = options.resolution == 1 ? 0
                : min(max((gridY - 0.5) / denominator, 0), 1)
            let gridZ = Float(options.resolution) * 0.5
            let expectedZ = options.resolution == 1 ? 0
                : min(max((gridZ - 0.5) / denominator, 0), 1)
            let expected: [Float] = [
                expectedX, expectedY, expectedZ,
                (expectedX + expectedY + expectedZ) / 3, 0.3, 0.5,
            ]
            for channel in 0..<6 {
                let actual = values[pixel * 6 + channel]
                guard abs(actual - expected[channel]) <= 3e-5 else {
                    throw NativeRuntimeError.invalidArgument(
                        "benchmark correctness failed at pixel \(pixel), channel \(channel)"
                    )
                }
                hash = (hash ^ UInt64(actual.bitPattern)) &* 1_099_511_628_211
            }
        }
        print(String(format: "correctness=full-output mask_covered=%d output_fnv1a64=%016llx",
                     pixelCount, hash))
    }
    for iteration in 0..<max(options.warmup, 1) {
        try autoreleasepool {
            let output = try execute()
            if iteration == 0 { try verify(output) }
        }
    }
    func timedExecute() throws -> Double {
        let start = ContinuousClock.now
        let result = try execute()
        let elapsed = start.duration(to: .now)
        _ = result.fields.length
        return Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
    }
    var durations: [Double] = []
    durations.reserveCapacity(options.iterations)
    for _ in 0..<options.iterations {
        durations.append(try autoreleasepool { try timedExecute() })
    }
    let orderedDurations = durations
    durations.sort()
    let memory = context.arena!.snapshot()
    let process = ProcessInfo.processInfo
    #if DEBUG
    let configuration = "debug"
    #else
    let configuration = "release"
    #endif
    print("benchmark_scope=pbr_bake_stage_not_full_export")
    print("backend=Metal device=\"\(context.device.name)\" build=\(configuration)")
    print("os=\"\(process.operatingSystemVersionString)\" physical_memory_bytes=\(process.physicalMemory)")
    print("workload=analytic_dense_volume_full_atlas_quad resolution=\(options.resolution) "
        + "tokens=\(tokenCount) coordinate_density=1.0 faces=2 atlas_coverage=1.0 "
        + "texture=\(options.textureSize)x\(options.textureSize) channels=6")
    print("warmup=\(max(options.warmup, 1)) iterations=\(options.iterations) synchronization=per-stage")
    print("timing_boundary=includes_host_coordinate_lookup_sort+per_iteration_arena_allocations+coordinate_upload+Metal_raster+Metal_sparse_sample+GPU_wait")
    print("timing_excludes=context_queue_heap_creation+shader_compilation+pipeline_creation+dense_field_allocation_upload+analytic_mesh_coordinate_field_construction+correctness_check+packing+inpaint+PNG_GLB_encoding+file_IO+GLB_validation")
    print("arena_capacity_bytes=\(memory.capacityBytes) arena_peak_bytes=\(memory.peakUsedBytes) "
        + "arena_live_bytes=\(memory.usedBytes) device_current_after_run_bytes=\(memory.currentDeviceAllocatedBytes)")
    print("memory_disclosure=arena_peak_excludes_direct_field_buffer_and_direct_raster_textures;device_value_is_not_peak")
    print(String(format: "median_ms=%.3f p95_r7_ms=%.3f min_ms=%.3f max_ms=%.3f",
                 quantileR7(durations, 0.5) * 1000,
                 quantileR7(durations, 0.95) * 1000,
                 durations[0] * 1000, durations.last! * 1000))
    print("raw_samples_ms=" + orderedDurations.map {
        String(format: "%.6f", $0 * 1000)
    }.joined(separator: ","))
}

func checkedProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
    let value = lhs.multipliedReportingOverflow(by: rhs)
    guard lhs >= 0, rhs >= 0, !value.overflow else {
        throw NativeRuntimeError.invalidArgument("benchmark size overflows Int")
    }
    return value.partialValue
}

do {
    try run()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
