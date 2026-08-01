import Foundation
import Metal

public final class MetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let arena: MetalBufferArena?
    let mpsGraphDense: MPSGraphDenseKernel

    public init(arenaCapacity: Int? = nil) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NativeRuntimeError.allocationFailed("no Metal device is available")
        }
        guard let queue = device.makeCommandQueue() else {
            throw NativeRuntimeError.allocationFailed("could not create Metal command queue")
        }
        self.device = device
        self.queue = queue
        self.arena = try arenaCapacity.map {
            try MetalBufferArena(device: device, capacity: $0, label: "KernelGoblin arena")
        }
        self.mpsGraphDense = MPSGraphDenseKernel(queue: queue)
    }

    public func library(named name: String) throws -> MTLLibrary {
        guard let url = Bundle.module.url(forResource: name, withExtension: "metal", subdirectory: "Metal") else {
            throw NativeRuntimeError.invalidArgument("missing Metal resource \(name).metal")
        }
        let source = try String(contentsOf: url, encoding: .utf8)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_0
        options.fastMathEnabled = true
        return try device.makeLibrary(source: source, options: options)
    }

    public func makeBuffer(length: Int, label: String) throws -> MTLBuffer {
        if let arena {
            return try arena.makeBuffer(length: length, label: label)
        }
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw NativeRuntimeError.allocationFailed("could not allocate \(label)")
        }
        buffer.label = label
        return buffer
    }

    func runCompute(
        label: String,
        _ encode: (MTLComputeCommandEncoder) throws -> Void
    ) throws {
        // Every current kernel waits before returning, so stage-owned buffers
        // safely outlive unretained GPU references. The nested pool also
        // destroys completed encoders before stage teardown.
        try autoreleasepool {
            guard let command = queue.makeCommandBufferWithUnretainedReferences(),
                  let encoder = command.makeComputeCommandEncoder() else {
                throw NativeRuntimeError.allocationFailed(
                    "could not create \(label) Metal command"
                )
            }
            command.label = label
            do {
                try encode(encoder)
            } catch {
                encoder.endEncoding()
                throw error
            }
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            guard command.status == .completed else {
                throw NativeRuntimeError.allocationFailed(
                    "\(label) failed: " +
                        (command.error?.localizedDescription ?? "unknown error")
                )
            }
        }
    }

    public func waitUntilIdle() throws {
        guard let sentinel = queue.makeCommandBuffer() else {
            throw NativeRuntimeError.allocationFailed(
                "could not create Metal queue-drain sentinel"
            )
        }
        sentinel.label = "KernelGoblin stage queue drain"
        sentinel.commit()
        sentinel.waitUntilCompleted()
        guard sentinel.status == .completed else {
            throw NativeRuntimeError.allocationFailed(
                "Metal queue drain failed: \(sentinel.error?.localizedDescription ?? "unknown error")"
            )
        }
    }
}
