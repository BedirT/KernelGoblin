import Foundation

public struct NativeCheckpointComponent: Codable, Equatable, Sendable {
    public let role: String
    public let repository: String
    public let revision: String
    public let path: String
    public let bytes: UInt64
    public let sha256: String

    public var downloadURL: URL {
        URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(path)")!
    }
}

public struct NativeInstallReceipt: Codable, Equatable, Sendable {
    public let format: String
    public let runtime: String
    public let installedAt: String
    public let components: [NativeCheckpointComponent]
}

public enum Trellis2InstallFeature: String, Codable, Sendable {
    case geometry
    case generate
    case texture
    case all
}

public enum Trellis2NativeInstaller {
    public static let components: [NativeCheckpointComponent] = [
        .init(
            role: "dino", repository: "facebook/dinov3-vitl16-pretrain-lvd1689m",
            revision: "ea8dc2863c51be0a264bab82070e3e8836b02d51",
            path: "model.safetensors", bytes: 1_212_559_808,
            sha256: "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
        ),
        .init(
            role: "sparse-structure-flow", repository: "microsoft/TRELLIS.2-4B",
            revision: NativeTrellis2Pipeline.trellisWeightsRevision,
            path: "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
            bytes: 2_584_426_920,
            sha256: "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6"
        ),
        .init(
            role: "sparse-structure-decoder", repository: "microsoft/TRELLIS-image-large",
            revision: "25e0d31ffbebe4b5a97464dd851910efc3002d96",
            path: "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
            bytes: 147_591_972,
            sha256: "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a"
        ),
        .init(
            role: "shape-flow", repository: "microsoft/TRELLIS.2-4B",
            revision: NativeTrellis2Pipeline.trellisWeightsRevision,
            path: "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
            bytes: 2_584_574_424,
            sha256: "ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f"
        ),
        .init(
            role: "texture-flow", repository: "microsoft/TRELLIS.2-4B",
            revision: NativeTrellis2Pipeline.trellisWeightsRevision,
            path: "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
            bytes: 2_584_672_728,
            sha256: "8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d"
        ),
        .init(
            role: "shape-decoder", repository: "microsoft/TRELLIS.2-4B",
            revision: NativeTrellis2Pipeline.trellisWeightsRevision,
            path: "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
            bytes: 948_490_494,
            sha256: "e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581"
        ),
        .init(
            role: "texture-decoder", repository: "microsoft/TRELLIS.2-4B",
            revision: NativeTrellis2Pipeline.trellisWeightsRevision,
            path: "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
            bytes: 948_458_812,
            sha256: "97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c"
        ),
        .init(
            role: "shape-encoder", repository: "microsoft/TRELLIS.2-4B",
            revision: NativeTrellis2Pipeline.trellisWeightsRevision,
            path: "ckpts/shape_enc_next_dc_f16c32_fp16.safetensors",
            bytes: 708_797_208,
            sha256: "f37c5ff5b983b68e9946060000f09bc131f3e84318a2c8b7430a81e4b4636c41"
        ),
    ]

    public static var defaultRoot: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/KernelGoblin", isDirectory: true)
            .appendingPathComponent("trellis2-512", isDirectory: true)
    }

    public static func checkpointSet(
        root: URL = defaultRoot,
        geometryOnly: Bool = false
    ) throws -> Trellis2CheckpointSet {
        let required = components(for: geometryOnly ? .geometry : .generate)
        let files = Dictionary(uniqueKeysWithValues: required.map {
            ($0.role, installedURL(for: $0, root: root))
        })
        let missing = required.filter {
            !FileManager.default.fileExists(atPath: installedURL(for: $0, root: root).path)
        }
        guard missing.isEmpty else {
            throw NativeRuntimeError.invalidArgument(
                "native TRELLIS.2 install is incomplete at \(root.path); run `kg-trellis2 install`"
            )
        }
        return Trellis2CheckpointSet(
            dino: files["dino"]!,
            sparseStructureFlow: files["sparse-structure-flow"]!,
            sparseStructureDecoder: files["sparse-structure-decoder"]!,
            shapeFlow: files["shape-flow"]!,
            textureFlow: files["texture-flow"],
            shapeDecoder: files["shape-decoder"]!,
            textureDecoder: files["texture-decoder"]
        )
    }

    public static func texturingCheckpointSet(
        root: URL = defaultRoot
    ) throws -> Trellis2TexturingCheckpointSet {
        let required = components(for: .texture)
        let files = Dictionary(uniqueKeysWithValues: required.map {
            ($0.role, installedURL(for: $0, root: root))
        })
        guard required.allSatisfy({
            FileManager.default.fileExists(atPath: installedURL(for: $0, root: root).path)
        }) else {
            throw NativeRuntimeError.invalidArgument(
                "native TRELLIS.2 texturing install is incomplete at \(root.path); "
                    + "run `kg-trellis2 install --feature texture`"
            )
        }
        return Trellis2TexturingCheckpointSet(
            dino: files["dino"]!, shapeEncoder: files["shape-encoder"]!,
            textureFlow: files["texture-flow"]!,
            textureDecoder: files["texture-decoder"]!
        )
    }

    public static func install512(
        root: URL = defaultRoot,
        feature: Trellis2InstallFeature = .all,
        huggingFaceToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"],
        progress: @Sendable (String) -> Void = { _ in }
    ) async throws -> NativeInstallReceipt {
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true
        )
        var cachedByRole: [String: URL] = [:]
        for component in components {
            let candidate = huggingFaceCacheURL(for: component)
            if FileManager.default.fileExists(atPath: candidate.path) {
                cachedByRole[component.role] = candidate
            }
        }
        let selected = components(for: feature)
        for component in selected {
            let destination = installedURL(for: component, root: root)
            if try validateExisting(destination, component: component) {
                progress("verified \(component.role)")
                continue
            }
            if let source = cachedByRole[component.role],
               try validateFile(source, component: component) {
                try FileManager.default.createDirectory(
                    at: destination.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try? FileManager.default.removeItem(at: destination)
                try FileManager.default.createSymbolicLink(
                    at: destination, withDestinationURL: source.resolvingSymlinksInPath()
                )
                progress("linked verified Hugging Face cache for \(component.role)")
                continue
            }
            progress("downloading \(component.role) (\(component.bytes) bytes)")
            var request = URLRequest(url: component.downloadURL)
            request.timeoutInterval = 24 * 60 * 60
            if let token = huggingFaceToken, !token.isEmpty {
                request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }
            let (temporary, response) = try await URLSession.shared.download(for: request)
            guard let http = response as? HTTPURLResponse,
                  (200..<300).contains(http.statusCode) else {
                let status = (response as? HTTPURLResponse)?.statusCode ?? -1
                throw NativeRuntimeError.invalidArgument(
                    "download for \(component.role) failed with HTTP \(status); "
                        + "DINOv3 requires accepted terms and HF_TOKEN"
                )
            }
            let attributes = try FileManager.default.attributesOfItem(
                atPath: temporary.path
            )
            guard let size = attributes[.size] as? NSNumber,
                  size.uint64Value == component.bytes else {
                throw NativeRuntimeError.invalidArgument(
                    "downloaded \(component.role) has the wrong byte count"
                )
            }
            let actualHash = try fileSHA256(at: temporary)
            guard actualHash == component.sha256 else {
                throw NativeRuntimeError.invalidArgument(
                    "downloaded \(component.role) failed SHA-256 verification"
                )
            }
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let partial = destination.appendingPathExtension("partial")
            try? FileManager.default.removeItem(at: partial)
            try FileManager.default.moveItem(at: temporary, to: partial)
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: partial, to: destination)
            progress("installed \(component.role)")
        }
        let receipt = NativeInstallReceipt(
            format: "KernelGoblin-native-install-v1",
            runtime: "swift-metal",
            installedAt: ISO8601DateFormatter().string(from: Date()),
            components: selected
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(receipt).write(
            to: root.appendingPathComponent("install.json"), options: .atomic
        )
        return receipt
    }

    public static func components(
        for feature: Trellis2InstallFeature
    ) -> [NativeCheckpointComponent] {
        switch feature {
        case .all:
            components
        case .generate:
            components.filter { $0.role != "shape-encoder" }
        case .geometry:
            components.filter {
                !["shape-encoder", "texture-flow", "texture-decoder"]
                    .contains($0.role)
            }
        case .texture:
            components.filter {
                ["dino", "shape-encoder", "texture-flow", "texture-decoder"]
                    .contains($0.role)
            }
        }
    }

    static func huggingFaceCacheURL(
        for component: NativeCheckpointComponent,
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> URL {
        let repositoryDirectory = "models--"
            + component.repository.replacingOccurrences(of: "/", with: "--")
        return homeDirectory
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
            .appendingPathComponent(repositoryDirectory, isDirectory: true)
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent(component.revision, isDirectory: true)
            .appendingPathComponent(component.path)
    }

    private static func installedURL(
        for component: NativeCheckpointComponent, root: URL
    ) -> URL {
        root.appendingPathComponent(component.role, isDirectory: true)
            .appendingPathComponent((component.path as NSString).lastPathComponent)
    }

    private static func validateExisting(
        _ url: URL, component: NativeCheckpointComponent
    ) throws -> Bool {
        guard FileManager.default.fileExists(atPath: url.path) else { return false }
        guard try validateFile(url, component: component) else {
            try FileManager.default.removeItem(at: url)
            return false
        }
        return true
    }

    private static func validateFile(
        _ url: URL, component: NativeCheckpointComponent
    ) throws -> Bool {
        let resolved = url.resolvingSymlinksInPath()
        let attributes = try FileManager.default.attributesOfItem(atPath: resolved.path)
        guard (attributes[.size] as? NSNumber)?.uint64Value == component.bytes else {
            return false
        }
        return try fileSHA256(at: resolved) == component.sha256
    }
}
