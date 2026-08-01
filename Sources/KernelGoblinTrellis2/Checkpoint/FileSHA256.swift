import CryptoKit
import Darwin
import Foundation

public func fileSHA256(at url: URL, chunkSize: Int = 8 * 1024 * 1024) throws -> String {
    guard chunkSize > 0 else {
        throw NativeRuntimeError.invalidArgument("SHA-256 chunk size must be positive")
    }
    let path = url.resolvingSymlinksInPath().path
    let descriptor = Darwin.open(path, O_RDONLY)
    guard descriptor >= 0 else {
        throw NativeRuntimeError.invalidArgument(
            "could not open \(path) for SHA-256: \(String(cString: strerror(errno)))"
        )
    }
    defer { Darwin.close(descriptor) }
    var hasher = SHA256()
    var chunk = [UInt8](repeating: 0, count: chunkSize)
    while true {
        let count = chunk.withUnsafeMutableBytes { bytes in
            Darwin.read(descriptor, bytes.baseAddress, bytes.count)
        }
        guard count >= 0 else {
            throw NativeRuntimeError.invalidArgument(
                "could not read \(path) for SHA-256: \(String(cString: strerror(errno)))"
            )
        }
        guard count > 0 else { break }
        chunk.withUnsafeBytes { bytes in
            hasher.update(bufferPointer: UnsafeRawBufferPointer(rebasing: bytes[..<count]))
        }
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}
