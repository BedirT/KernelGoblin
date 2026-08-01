import Foundation
import Metal

public final class MetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let queue: MTLCommandQueue

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NativeRuntimeError.allocationFailed("no Metal device is available")
        }
        guard let queue = device.makeCommandQueue() else {
            throw NativeRuntimeError.allocationFailed("could not create Metal command queue")
        }
        self.device = device
        self.queue = queue
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
}
