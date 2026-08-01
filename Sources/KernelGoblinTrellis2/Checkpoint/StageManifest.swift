import Foundation

public struct StageFile: Codable, Equatable, Sendable {
    public let path: String
    public let size: UInt64
    public let sha256: String
}

public struct StageManifest: Codable, Equatable, Sendable {
    public static let magic = "KGSTAGE"
    public static let supportedMajorVersion = 1

    public let format: String
    public let versionMajor: Int
    public let versionMinor: Int
    public let model: String
    public let component: String
    public let sourceRepository: String
    public let sourceRevision: String
    public let files: [StageFile]

    public func validate() throws {
        guard format == Self.magic else {
            throw StageManifestError.invalid("format must be \(Self.magic)")
        }
        guard versionMajor == Self.supportedMajorVersion else {
            throw StageManifestError.invalid("unsupported major version \(versionMajor)")
        }
        guard sourceRevision.count == 40,
              sourceRevision.allSatisfy({ $0.isHexDigit && !$0.isUppercase }) else {
            throw StageManifestError.invalid("source revision must be a full lowercase Git SHA")
        }
        guard !component.isEmpty, !files.isEmpty else {
            throw StageManifestError.invalid("component and files must be nonempty")
        }
        var paths: Set<String> = []
        for file in files {
            guard file.size > 0, file.sha256.count == 64,
                  file.sha256.allSatisfy({ $0.isHexDigit && !$0.isUppercase }),
                  !file.path.hasPrefix("/"), !file.path.contains(".."),
                  paths.insert(file.path).inserted else {
                throw StageManifestError.invalid("invalid or duplicate stage file \(file.path)")
            }
        }
    }

    public static func read(from url: URL, maximumBytes: Int = 4 * 1024 * 1024) throws -> Self {
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])
        guard data.count <= maximumBytes else {
            throw StageManifestError.invalid("manifest exceeds \(maximumBytes) bytes")
        }
        let manifest = try JSONDecoder().decode(Self.self, from: data)
        try manifest.validate()
        return manifest
    }
}

public enum StageManifestError: Error, CustomStringConvertible, Equatable {
    case invalid(String)

    public var description: String {
        switch self { case .invalid(let detail): "invalid stage manifest: \(detail)" }
    }
}
