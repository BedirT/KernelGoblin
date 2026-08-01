import Metal

public final class BoundedScratch: @unchecked Sendable {
    public let capacity: Int
    public let buffer: MTLBuffer

    public init(device: MTLDevice, capacity: Int, label: String) throws {
        guard capacity > 0 else { throw NativeRuntimeError.invalidArgument("scratch capacity must be positive") }
        guard let buffer = device.makeBuffer(length: capacity, options: .storageModeShared) else {
            throw NativeRuntimeError.allocationFailed("could not allocate \(capacity)-byte scratch buffer")
        }
        self.capacity = capacity
        self.buffer = buffer
        self.buffer.label = label
    }

    public func range(offset: Int, count: Int) throws -> Range<Int> {
        guard offset >= 0, count >= 0, offset <= capacity, count <= capacity - offset else {
            throw NativeRuntimeError.capacityExceeded(
                requested: count, available: max(0, capacity - max(0, offset))
            )
        }
        return offset..<(offset + count)
    }
}

public enum NativeRuntimeError: Error, CustomStringConvertible, Equatable {
    case allocationFailed(String)
    case capacityExceeded(requested: Int, available: Int)
    case invalidArgument(String)

    public var description: String {
        switch self {
        case .allocationFailed(let detail): detail
        case .capacityExceeded(let requested, let available):
            "bounded allocation requested \(requested) bytes with \(available) available"
        case .invalidArgument(let detail): detail
        }
    }
}
