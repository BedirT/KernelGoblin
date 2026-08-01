import CryptoKit
import Foundation

public func fileSHA256(at url: URL, chunkSize: Int = 8 * 1024 * 1024) throws -> String {
    guard chunkSize > 0 else {
        throw NativeRuntimeError.invalidArgument("SHA-256 chunk size must be positive")
    }
    let handle = try FileHandle(forReadingFrom: url.resolvingSymlinksInPath())
    defer { try? handle.close() }
    var hasher = SHA256()
    while let chunk = try handle.read(upToCount: chunkSize), !chunk.isEmpty {
        hasher.update(data: chunk)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}
