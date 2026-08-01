import Darwin
import Foundation
import Metal

public final class MappedCheckpoint: @unchecked Sendable {
    public let index: SafeTensorsIndex
    public let buffer: MTLBuffer
    public let validByteCount: UInt64
    public let mappedByteCount: UInt64

    public init(url: URL, device: MTLDevice) throws {
        let resolvedURL = url.resolvingSymlinksInPath()
        let descriptor = open(resolvedURL.path, O_RDONLY)
        guard descriptor >= 0 else {
            throw MappedCheckpointError.posix("open", errno)
        }
        defer { close(descriptor) }
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
        nonisolated(unsafe) let capturedBase = base
        let capturedLength = Int(rounded)
        guard let buffer = device.makeBuffer(
            bytesNoCopy: base,
            length: capturedLength,
            options: .storageModeShared,
            deallocator: { _, _ in munmap(capturedBase, capturedLength) }
        ) else {
            munmap(base, capturedLength)
            throw MappedCheckpointError.metalBufferCreationFailed
        }
        buffer.label = "Mapped safetensors: \(resolvedURL.lastPathComponent)"
        self.index = index
        self.buffer = buffer
        self.validByteCount = index.fileSize
        self.mappedByteCount = rounded
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

    private static func roundUp(_ value: UInt64, alignment: UInt64) throws -> UInt64 {
        let remainder = value % alignment
        if remainder == 0 { return value }
        let addition = alignment - remainder
        let result = value.addingReportingOverflow(addition)
        guard !result.overflow else { throw MappedCheckpointError.tooLarge(value) }
        return result.partialValue
    }
}

public enum MappedCheckpointError: Error, CustomStringConvertible, Equatable {
    case metalBufferCreationFailed
    case missingTensor(String)
    case posix(String, Int32)
    case tooLarge(UInt64)

    public var description: String {
        switch self {
        case .metalBufferCreationFailed: "Metal could not wrap the mapped checkpoint"
        case .missingTensor(let name): "checkpoint is missing tensor \(name)"
        case .posix(let call, let code): "\(call) failed with errno \(code)"
        case .tooLarge(let count): "checkpoint byte count \(count) exceeds host limits"
        }
    }
}
