import Foundation
import Testing
@testable import KernelGoblinTrellis2

@Suite("Native TRELLIS.2 production pipeline")
struct NativePipelineTests {
    @Test("native installer supports generation-only and texturing-only selections")
    func selectiveInstallManifest() {
        let generation = Trellis2NativeInstaller.components(for: .generate)
        let texturing = Trellis2NativeInstaller.components(for: .texture)
        let all = Trellis2NativeInstaller.components(for: .all)
        #expect(generation.count == 7)
        #expect(!generation.contains { $0.role == "shape-encoder" })
        #expect(Set(texturing.map(\.role)) == Set([
            "dino", "shape-encoder", "texture-flow", "texture-decoder",
        ]))
        #expect(all.count == 8)
        #expect(Set(all.map(\.role)).count == all.count)
        #expect(all.allSatisfy {
            $0.revision.count == 40 && $0.sha256.count == 64 && $0.bytes > 0
        })
        let home = URL(fileURLWithPath: "/tmp/kg-home", isDirectory: true)
        let dino = all.first { $0.role == "dino" }!
        #expect(Trellis2NativeInstaller.huggingFaceCacheURL(
            for: dino, homeDirectory: home
        ).path == "/tmp/kg-home/.cache/huggingface/hub/"
            + "models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/"
            + "ea8dc2863c51be0a264bab82070e3e8836b02d51/model.safetensors")
    }

    @Test("checkpoint resolver uses only pinned files")
    func checkpointResolver() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "kg-checkpoints-\(UUID().uuidString)", isDirectory: true
        )
        defer { try? FileManager.default.removeItem(at: root) }
        let hub = root.appendingPathComponent(".cache/huggingface/hub")
        let expected = [
            "models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
            "models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
            "models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
            "models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
            "models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
            "models--microsoft--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
            "models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/ea8dc2863c51be0a264bab82070e3e8836b02d51/model.safetensors",
        ]
        for relative in expected {
            let url = hub.appendingPathComponent(relative)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try Data([0]).write(to: url)
        }
        let checkpoints = try Trellis2CheckpointSet.huggingFaceCache(
            homeDirectory: root
        )
        #expect(Set(checkpoints.allURLs.map(\.standardizedFileURL.path)) == Set(
            expected.map { hub.appendingPathComponent($0).standardizedFileURL.path }
        ))
        try FileManager.default.removeItem(at: checkpoints.textureDecoder)
        #expect(throws: NativeRuntimeError.self) {
            _ = try Trellis2CheckpointSet.huggingFaceCache(homeDirectory: root)
        }
    }

    @Test("generation rejects non-512 and malformed input before loading weights")
    func inputValidation() {
        let nowhere = URL(fileURLWithPath: "/does-not-exist")
        let checkpoints = Trellis2CheckpointSet(
            dino: nowhere, sparseStructureFlow: nowhere,
            sparseStructureDecoder: nowhere, shapeFlow: nowhere,
            textureFlow: nowhere, shapeDecoder: nowhere,
            textureDecoder: nowhere
        )
        #expect(throws: NativeRuntimeError.self) {
            _ = try NativeTrellis2Pipeline().generate512(
                normalizedImageCHW: [0], imageWidth: 1, imageHeight: 1,
                checkpoints: checkpoints,
                outputURL: FileManager.default.temporaryDirectory
                    .appendingPathComponent("never.glb")
            )
        }
    }
}
