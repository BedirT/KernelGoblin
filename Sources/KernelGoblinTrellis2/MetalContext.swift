import Foundation
import Metal

public final class MetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let arena: MetalBufferArena?

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
    }

    public func library(named name: String) throws -> MTLLibrary {
        guard let url = Bundle.module.url(forResource: name, withExtension: "metal", subdirectory: "Metal") else {
            throw NativeRuntimeError.invalidArgument("missing Metal resource \(name).metal")
        }
        let source = try String(contentsOf: url, encoding: .utf8)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_0
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
}
