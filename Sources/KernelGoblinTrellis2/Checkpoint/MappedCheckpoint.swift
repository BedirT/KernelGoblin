import Darwin
import CryptoKit
import Foundation
import Metal

public final class MappedCheckpoint: @unchecked Sendable {
    public let index: SafeTensorsIndex
    public let validByteCount: UInt64
    public let mappedByteCount: UInt64

    private let lock = NSLock()
    private let mappingLifetime: MappingLifetime
    private var mappedBuffer: MTLBuffer?

    public var isMapped: Bool {
        mappingLifetime.isMappingAlive
    }

    public init(url: URL, device: MTLDevice) throws {
        let resolvedURL = url.resolvingSymlinksInPath()
        let descriptor = open(resolvedURL.path, O_RDONLY)
        guard descriptor >= 0 else {
            throw MappedCheckpointError.posix("open", errno)
        }
        defer { Darwin.close(descriptor) }
        let index = try SafeTensorsIndex.read(fileDescriptor: descriptor)
        let pageSize = UInt64(getpagesize())
        let rounded = try Self.roundUp(index.fileSize, alignment: pageSize)
        guard rounded <= UInt64(Int.max) else {
            throw MappedCheckpointError.tooLarge(index.fileSize)
        }
        let pointer = mmap(nil, Int(rounded), PROT_READ, MAP_PRIVATE, descriptor, 0)
        let savedErrno = errno
        guard pointer != MAP_FAILED, let base = pointer else {
            throw MappedCheckpointError.posix("mmap", savedErrno)
        }
        let mappingLifetime = MappingLifetime()
        nonisolated(unsafe) let capturedBase = base
        let capturedLength = Int(rounded)
        guard let buffer = device.makeBuffer(
            bytesNoCopy: base,
            length: capturedLength,
            options: .storageModeShared,
            deallocator: { _, _ in
                let result = munmap(capturedBase, capturedLength)
                mappingLifetime.record(result: result, errno: result == 0 ? 0 : Darwin.errno)
            }
        ) else {
            let result = munmap(base, capturedLength)
            if result != 0 {
                throw MappedCheckpointError.posix("munmap", errno)
            }
            throw MappedCheckpointError.metalBufferCreationFailed
        }
        buffer.label = "Mapped safetensors: \(resolvedURL.lastPathComponent)"
        self.index = index
        self.mappingLifetime = mappingLifetime
        self.mappedBuffer = buffer
        self.validByteCount = index.fileSize
        self.mappedByteCount = rounded
    }

    public func close(releaseTimeout: TimeInterval = 1) throws {
        guard releaseTimeout >= 0 else {
            throw NativeRuntimeError.invalidArgument(
                "checkpoint release timeout must be nonnegative"
            )
        }
        autoreleasepool { detachMappedBuffer() }
        switch mappingLifetime.waitForRelease(timeout: releaseTimeout) {
        case .pending:
            throw MappedCheckpointError.mappingStillReferenced
        case .succeeded:
            return
        case .failed(let code):
            throw MappedCheckpointError.posix("munmap", code)
        }
    }

    public func acquireBuffer() throws -> MTLBuffer {
        lock.lock()
        defer { lock.unlock() }
        guard let mappedBuffer else {
            throw MappedCheckpointError.closed
        }
        return mappedBuffer
    }

    private func detachMappedBuffer() {
        lock.lock()
        mappedBuffer = nil
        lock.unlock()
    }

    public func descriptor(named name: String) throws -> TensorDescriptor {
        guard let descriptor = index.tensors[name] else {
            throw MappedCheckpointError.missingTensor(name)
        }
        return descriptor
    }

    public func byteRange(for name: String) throws -> Range<Int> {
        let tensor = try descriptor(named: name)
        guard tensor.fileOffset <= UInt64(Int.max), tensor.byteCount <= UInt64(Int.max),
              tensor.fileOffset <= validByteCount,
              tensor.byteCount <= validByteCount - tensor.fileOffset else {
            throw MappedCheckpointError.tooLarge(tensor.byteCount)
        }
        let start = Int(tensor.fileOffset)
        return start..<(start + Int(tensor.byteCount))
    }

    public func sha256(chunkSize: Int = 8 * 1024 * 1024) throws -> String {
        guard chunkSize > 0, validByteCount <= UInt64(Int.max) else {
            throw NativeRuntimeError.invalidArgument("invalid mapped SHA-256 byte range")
        }
        lock.lock()
        guard let buffer = mappedBuffer else {
            lock.unlock()
            throw MappedCheckpointError.closed
        }
        lock.unlock()
        var hasher = SHA256()
        let total = Int(validByteCount)
        var offset = 0
        while offset < total {
            let count = min(chunkSize, total - offset)
            hasher.update(bufferPointer: UnsafeRawBufferPointer(
                start: buffer.contents().advanced(by: offset), count: count
            ))
            offset += count
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func roundUp(_ value: UInt64, alignment: UInt64) throws -> UInt64 {
        let remainder = value % alignment
        if remainder == 0 { return value }
        let addition = alignment - remainder
        let result = value.addingReportingOverflow(addition)
        guard !result.overflow else { throw MappedCheckpointError.tooLarge(value) }
        return result.partialValue
    }
}

private enum MappingReleaseState {
    case pending
    case succeeded
    case failed(Int32)
}

private final class MappingLifetime: @unchecked Sendable {
    private let condition = NSCondition()
    private var state: MappingReleaseState = .pending

    var isMappingAlive: Bool {
        condition.lock()
        defer { condition.unlock() }
        return switch state {
        case .succeeded:
            false
        case .pending, .failed:
            true
        }
    }

    func record(result: Int32, errno code: Int32) {
        condition.lock()
        state = result == 0 ? .succeeded : .failed(code)
        condition.broadcast()
        condition.unlock()
    }

    func waitForRelease(timeout: TimeInterval) -> MappingReleaseState {
        condition.lock()
        defer { condition.unlock() }
        let deadline = Date(timeIntervalSinceNow: timeout)
        while case .pending = state, condition.wait(until: deadline) {}
        return state
    }
}

public enum MappedCheckpointError: Error, CustomStringConvertible, Equatable {
    case closed
    case mappingStillReferenced
    case metalBufferCreationFailed
    case missingTensor(String)
    case posix(String, Int32)
    case tooLarge(UInt64)

    public var description: String {
        switch self {
        case .closed: "mapped checkpoint is closed"
        case .mappingStillReferenced:
            "mapped checkpoint buffer is still referenced"
        case .metalBufferCreationFailed: "Metal could not wrap the mapped checkpoint"
        case .missingTensor(let name): "checkpoint is missing tensor \(name)"
        case .posix(let call, let code): "\(call) failed with errno \(code)"
        case .tooLarge(let count): "checkpoint byte count \(count) exceeds host limits"
        }
    }
}
