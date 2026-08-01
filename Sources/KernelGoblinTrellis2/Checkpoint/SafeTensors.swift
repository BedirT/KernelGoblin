import CoreFoundation
import Darwin
import Foundation

public enum TensorDataType: String, Codable, Sendable {
    case bf16 = "BF16"
    case f16 = "F16"
    case f32 = "F32"
    case i32 = "I32"
    case i64 = "I64"
    case u8 = "U8"

    public var byteWidth: UInt64 {
        switch self {
        case .u8: 1
        case .bf16, .f16: 2
        case .f32, .i32: 4
        case .i64: 8
        }
    }
}

public struct TensorDescriptor: Equatable, Sendable {
    public let name: String
    public let dtype: TensorDataType
    public let shape: [UInt64]
    public let fileOffset: UInt64
    public let byteCount: UInt64
}

public struct SafeTensorsIndex: Sendable {
    public static let maximumHeaderBytes: UInt64 = 16 * 1024 * 1024

    public let fileSize: UInt64
    public let payloadOffset: UInt64
    public let tensors: [String: TensorDescriptor]

    public static func read(from url: URL) throws -> SafeTensorsIndex {
        let resolvedURL = url.resolvingSymlinksInPath()
        let descriptor = open(resolvedURL.path, O_RDONLY)
        guard descriptor >= 0 else {
            throw CheckpointError.invalidFile("open failed with errno \(errno)")
        }
        defer { close(descriptor) }
        return try read(fileDescriptor: descriptor)
    }

    static func read(fileDescriptor descriptor: Int32) throws -> SafeTensorsIndex {
        var status = stat()
        guard fstat(descriptor, &status) == 0, (status.st_mode & S_IFMT) == S_IFREG,
              status.st_size >= 8 else {
            throw CheckpointError.invalidFile("safetensors input is not a regular file of at least 8 bytes")
        }
        let fileSize = UInt64(status.st_size)
        let duplicate = dup(descriptor)
        guard duplicate >= 0 else {
            throw CheckpointError.invalidFile("dup failed with errno \(errno)")
        }
        let handle = FileHandle(fileDescriptor: duplicate, closeOnDealloc: true)
        defer { try? handle.close() }
        let prefix = try handle.read(upToCount: 8) ?? Data()
        guard prefix.count == 8 else { throw CheckpointError.truncated("missing header length") }
        let headerSize = prefix.withUnsafeBytes { bytes in
            bytes.loadUnaligned(as: UInt64.self).littleEndian
        }
        guard headerSize > 0, headerSize <= maximumHeaderBytes, headerSize <= fileSize - 8 else {
            throw CheckpointError.invalidFile("safetensors header length is out of bounds")
        }
        let header = try handle.read(upToCount: Int(headerSize)) ?? Data()
        guard header.count == Int(headerSize) else {
            throw CheckpointError.truncated("safetensors JSON header")
        }
        let object: Any
        do {
            object = try JSONSerialization.jsonObject(with: header)
        } catch {
            throw CheckpointError.invalidFile("safetensors header is not valid JSON: \(error)")
        }
        guard let dictionary = object as? [String: Any] else {
            throw CheckpointError.invalidFile("safetensors header must be a JSON object")
        }
        let payloadOffset = 8 + headerSize
        var tensors: [String: TensorDescriptor] = [:]
        for (name, rawEntry) in dictionary where name != "__metadata__" {
            guard let entry = rawEntry as? [String: Any],
                  let rawDType = entry["dtype"] as? String,
                  let dtype = TensorDataType(rawValue: rawDType),
                  let rawShape = entry["shape"] as? [Any],
                  let rawOffsets = entry["data_offsets"] as? [Any], rawOffsets.count == 2
            else {
                throw CheckpointError.invalidTensor(name, "missing dtype, shape, or offsets")
            }
            let shape = try rawShape.map { raw -> UInt64 in
                guard let number = exactInteger(raw), number.int64Value >= 0 else {
                    throw CheckpointError.invalidTensor(name, "shape contains a negative or non-integer value")
                }
                return number.uint64Value
            }
            guard let beginNumber = exactInteger(rawOffsets[0]),
                  let endNumber = exactInteger(rawOffsets[1]),
                  beginNumber.int64Value >= 0, endNumber.int64Value >= 0
            else {
                throw CheckpointError.invalidTensor(name, "offsets must be nonnegative integers")
            }
            let begin = beginNumber.uint64Value
            let end = endNumber.uint64Value
            guard begin <= end, payloadOffset <= fileSize,
                  end <= fileSize - payloadOffset else {
                throw CheckpointError.invalidTensor(name, "payload range is outside the file")
            }
            let expectedElements = try checkedProduct(shape, tensor: name)
            let expectedBytes = try checkedMultiply(expectedElements, dtype.byteWidth, tensor: name)
            guard expectedBytes == end - begin else {
                throw CheckpointError.invalidTensor(
                    name,
                    "shape requires \(expectedBytes) bytes but range contains \(end - begin)"
                )
            }
            tensors[name] = TensorDescriptor(
                name: name,
                dtype: dtype,
                shape: shape,
                fileOffset: payloadOffset + begin,
                byteCount: end - begin
            )
        }
        guard !tensors.isEmpty else {
            throw CheckpointError.invalidFile("safetensors file contains no tensors")
        }
        let ranges = tensors.values.sorted {
            if $0.fileOffset != $1.fileOffset { return $0.fileOffset < $1.fileOffset }
            if $0.byteCount != $1.byteCount { return $0.byteCount < $1.byteCount }
            return $0.name < $1.name
        }
        var expectedOffset = payloadOffset
        for tensor in ranges {
            guard tensor.fileOffset == expectedOffset else {
                throw CheckpointError.invalidTensor(
                    tensor.name,
                    "tensor ranges must be contiguous and non-overlapping"
                )
            }
            let next = expectedOffset.addingReportingOverflow(tensor.byteCount)
            guard !next.overflow else {
                throw CheckpointError.invalidTensor(tensor.name, "tensor range overflows UInt64")
            }
            expectedOffset = next.partialValue
        }
        guard expectedOffset == fileSize else {
            throw CheckpointError.invalidFile("tensor ranges do not consume the complete payload")
        }
        return SafeTensorsIndex(fileSize: fileSize, payloadOffset: payloadOffset, tensors: tensors)
    }
}

private func exactInteger(_ value: Any) -> NSNumber? {
    guard let number = value as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID(),
          !CFNumberIsFloatType(number) else {
        return nil
    }
    return number
}

private func checkedProduct(_ values: [UInt64], tensor: String) throws -> UInt64 {
    try values.reduce(1) { partial, value in
        let result = partial.multipliedReportingOverflow(by: value)
        guard !result.overflow else {
            throw CheckpointError.invalidTensor(tensor, "shape product overflows UInt64")
        }
        return result.partialValue
    }
}

private func checkedMultiply(_ lhs: UInt64, _ rhs: UInt64, tensor: String) throws -> UInt64 {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard !result.overflow else {
        throw CheckpointError.invalidTensor(tensor, "byte count overflows UInt64")
    }
    return result.partialValue
}

public enum CheckpointError: Error, CustomStringConvertible, Equatable {
    case invalidFile(String)
    case invalidTensor(String, String)
    case truncated(String)

    public var description: String {
        switch self {
        case .invalidFile(let detail): "invalid checkpoint: \(detail)"
        case .invalidTensor(let name, let detail): "invalid tensor \(name): \(detail)"
        case .truncated(let detail): "truncated checkpoint: \(detail)"
        }
    }
}
