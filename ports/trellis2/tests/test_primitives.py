from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F


PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PORT))

import runtime_env
import streaming_loader
from streaming_loader import lazy_from_pretrained, normalize_checkpoint_path


runtime_env.configure_backends()

from pbr import bake_pbr_mesh, prepare_surface, rasterize_metal, unwrap_mesh, validate_mesh
from flex_gemm.ops.grid_sample import grid_sample_3d
from flex_gemm.ops.grid_sample import grid_sample as grid_sample_module
from o_voxel.convert.flexible_dual_grid import _lookup as voxel_lookup
from o_voxel.convert.flexible_dual_grid import flexible_dual_grid_to_mesh
from o_voxel.convert.flexible_dual_grid import mesh_to_flexible_dual_grid
from o_voxel.postprocess import to_glb
from trellis2.modules.sparse import SparseTensor, VarLenTensor
from trellis2.modules.sparse.attention.full_attn import (
    sparse_scaled_dot_product_attention,
)
from trellis2.modules.sparse.attention.windowed_attn import (
    sparse_windowed_scaled_dot_product_self_attention,
)


def devices() -> list[str]:
    result = ["cpu"]
    if torch.backends.mps.is_available():
        result.append("mps")
    return result


class PrimitiveTests(unittest.TestCase):
    def test_runtime_rejects_mps_cpu_fallback(self) -> None:
        with mock.patch.dict("os.environ", {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
            with self.assertRaisesRegex(RuntimeError, "must be 0"):
                runtime_env.require_no_cpu_fallback()

    def test_mesh_validation_rejects_degenerate_and_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonzero spatial extent"):
            validate_mesh([[0, 0, 0]] * 3, [[0, 1, 2]])
        with self.assertRaisesRegex(ValueError, "face index"):
            validate_mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 3]])

    def test_xatlas_unwrap_returns_valid_seam_topology(self) -> None:
        vertices = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ], dtype=torch.float32).numpy()
        faces = torch.tensor([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ], dtype=torch.int32).numpy()
        atlas = unwrap_mesh(vertices, faces, texture_size=64, padding=2)
        self.assertEqual(atlas.faces.shape, (12, 3))
        self.assertGreater(len(atlas.positions), len(vertices))
        self.assertTrue((atlas.vertex_map < len(vertices)).all())
        self.assertTrue(((atlas.uvs >= 0) & (atlas.uvs <= 1)).all())

    def test_physical_metal_uv_raster_interpolates_positions(self) -> None:
        result = rasterize_metal(
            [[0, 0, 0], [1, 0, 0], [0, 1, 1]], [[0, 1, 2]],
            [[0.125, 0.125], [0.875, 0.125], [0.125, 0.875]],
            width=8,
        )
        self.assertIn("backend=Metal", result.backend)
        self.assertGreater(int((result.face_ids != 0).sum()), 0)
        covered = result.positions[result.face_ids != 0]
        self.assertTrue((covered[:, 3] == 1).all())
        self.assertTrue(((covered[:, :3] >= 0) & (covered[:, :3] <= 1)).all())

    def test_pbr_bake_packs_channels_and_round_trips_glb(self) -> None:
        surface = prepare_surface(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]],
            texture_size=16, decimation_target=10,
        )
        mesh, evidence = bake_pbr_mesh(
            surface,
            torch.tensor([[0.2, 0.4, 0.6, 0.8, 0.3, 0.5]]),
            torch.tensor([[0, 0, 0]], dtype=torch.int32),
            {
                "base_color": slice(0, 3), "metallic": slice(3, 4),
                "roughness": slice(4, 5), "alpha": slice(5, 6),
            },
            aabb=[[0, 0, 0], [1, 1, 1]], grid_size=[1, 1, 1],
            texture_size=16,
        )
        material = mesh.visual.material
        base = torch.from_numpy(__import__("numpy").array(material.baseColorTexture))
        mr = torch.from_numpy(__import__("numpy").array(material.metallicRoughnessTexture))
        self.assertEqual(base.shape, (16, 16, 4))
        self.assertEqual(mr.shape, (16, 16, 3))
        self.assertTrue(bool(((base[..., :3].to(torch.int16) - torch.tensor(
            [51, 102, 153], dtype=torch.int16
        )).abs() <= 4).all()))
        self.assertTrue(bool(((base[..., 3].to(torch.int16) - 127).abs() <= 1).all()))
        self.assertTrue(bool((mr[..., 0] == 0).all()))
        self.assertTrue(bool(((mr[..., 1].to(torch.int16) - 76).abs() <= 2).all()))
        self.assertTrue(bool(((mr[..., 2].to(torch.int16) - 204).abs() <= 4).all()))
        self.assertEqual(evidence["alpha_mode"], "OPAQUE")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pbr.glb"
            mesh.export(output)
            loaded = __import__("trimesh").load(output, force="mesh", process=False)
            self.assertEqual(loaded.visual.uv.shape[1], 2)
            self.assertIsNotNone(loaded.visual.material.baseColorTexture)
            self.assertIsNotNone(loaded.visual.material.metallicRoughnessTexture)

    def test_failed_checkpoint_download_cleans_local_cache(self) -> None:
        import huggingface_hub

        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory) / "downloads"
            config = Path(directory) / "model.json"
            config.write_text("{}")

            def download(_repo_id, filename, **_kwargs):
                if filename.endswith(".json"):
                    return str(config)
                cache = scratch / ".cache" / "huggingface" / "download"
                cache.mkdir(parents=True)
                (cache / "partial").write_bytes(b"partial")
                raise OSError("simulated full disk")

            with (
                mock.patch.object(streaming_loader, "DOWNLOADS", scratch),
                mock.patch.object(huggingface_hub, "hf_hub_download", download),
                self.assertRaisesRegex(OSError, "simulated full disk"),
            ):
                streaming_loader.streaming_from_pretrained(
                    "microsoft/TRELLIS.2-4B/ckpts/test"
                )

            self.assertFalse((scratch / ".cache").exists())

    def test_lazy_model_materializes_once_without_recursive_registration(self) -> None:
        model = torch.nn.Linear(2, 1)
        with mock.patch.object(
            streaming_loader, "streaming_from_pretrained", return_value=model
        ) as load:
            lazy = lazy_from_pretrained("microsoft/TRELLIS.2-4B/ckpts/test")
            lazy.eval()
            load.assert_not_called()

            first = lazy(torch.ones(1, 2))
            second = lazy(torch.ones(1, 2))
            lazy.unload()
            third = lazy(torch.ones(1, 2))

        self.assertEqual(first.shape, (1, 1))
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first, third)
        self.assertEqual(load.call_count, 2)
        load.assert_called_with("microsoft/TRELLIS.2-4B/ckpts/test")

    def test_external_checkpoint_path_is_normalized_for_lazy_loading(self) -> None:
        self.assertEqual(
            normalize_checkpoint_path(
                "microsoft/TRELLIS.2-4B/microsoft/TRELLIS-image-large/ckpts/decoder"
            ),
            "microsoft/TRELLIS-image-large/ckpts/decoder",
        )

    def test_model_attention_call_site_uses_mps_replacement(self) -> None:
        from trellis2.modules.sparse.attention import modules

        self.assertIs(
            modules.sparse_scaled_dot_product_attention,
            sparse_scaled_dot_product_attention,
        )
        self.assertIs(
            modules.sparse_windowed_scaled_dot_product_self_attention,
            sparse_windowed_scaled_dot_product_self_attention,
        )

    def test_submanifold_convolution_matches_dense_reference(self) -> None:
        from trellis2.modules import sparse as sp
        from trellis2.modules.sparse.conv import conv_mps

        coords_cpu = torch.tensor(
            [[0, x, y, z] for x in range(3) for y in range(3) for z in range(3)],
            dtype=torch.int32,
        )
        feats_cpu = torch.arange(54, dtype=torch.float32).reshape(27, 2) / 17
        weight = torch.arange(3 * 3 * 3 * 3 * 2, dtype=torch.float32).reshape(
            3, 3, 3, 3, 2
        ) / 101
        bias = torch.tensor([0.25, -0.5, 0.75])

        dense = feats_cpu.T.reshape(1, 2, 3, 3, 3)
        reference = F.conv3d(
            dense,
            weight.permute(0, 4, 1, 2, 3),
            bias,
            padding=1,
        )[0].permute(1, 2, 3, 0).reshape(-1, 3)

        for device in devices():
            with self.subTest(device=device):
                with (
                    mock.patch.object(conv_mps, "_MAP_CHUNK_SIZE", 2),
                    mock.patch.object(conv_mps, "_CONV_CHUNK_SIZE", 2),
                ):
                    layer = sp.SparseConv3d(2, 3, 3).to(device)
                    layer.weight.data.copy_(weight.to(device))
                    layer.bias.data.copy_(bias.to(device))
                    tensor = SparseTensor(
                        feats_cpu.to(device),
                        coords_cpu.to(device),
                        shape=torch.Size((1, 2)),
                    )
                    actual = layer(tensor).feats.cpu()
                torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)

    def test_fp16_sparse_convolution_matches_irregular_reference(self) -> None:
        from trellis2.modules import sparse as sp

        torch.manual_seed(19)
        coords = torch.tensor([
            [0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0],
            [0, 3, 2, 1], [0, 3, 3, 1], [0, 2, 3, 2],
        ], dtype=torch.int32)
        feats = torch.randn(6, 3, dtype=torch.float16)
        weight = torch.randn(4, 3, 3, 3, 3, dtype=torch.float16) / 4
        bias = torch.randn(4, dtype=torch.float16) / 4
        coord_map = {tuple(coord.tolist()): index for index, coord in enumerate(coords)}
        reference = torch.zeros(6, 4, dtype=torch.float32)
        for output_index, coord in enumerate(coords):
            reference[output_index] = bias.float()
            for ix in range(3):
                for iy in range(3):
                    for iz in range(3):
                        query = coord.clone()
                        query[1:] += torch.tensor([ix - 1, iy - 1, iz - 1])
                        source = coord_map.get(tuple(query.tolist()))
                        if source is not None:
                            reference[output_index] += (
                                feats[source].float()
                                @ weight[:, ix, iy, iz, :].float().T
                            )

        for device in devices():
            with self.subTest(device=device):
                layer = sp.SparseConv3d(3, 4, 3).to(device=device, dtype=torch.float16)
                layer.weight.data.copy_(weight.to(device))
                layer.bias.data.copy_(bias.to(device))
                actual = layer(SparseTensor(
                    feats.to(device), coords.to(device), shape=torch.Size((1, 3))
                )).feats.float().cpu()
                torch.testing.assert_close(actual, reference, atol=2e-2, rtol=2e-2)

    def test_variable_length_attention_matches_individual_sdpa(self) -> None:
        torch.manual_seed(7)
        layout = [slice(0, 3), slice(3, 5)]
        for device in devices():
            with self.subTest(device=device):
                qkv = torch.randn(5, 3, 2, 4, device=device)
                value = VarLenTensor(qkv, layout)
                actual = sparse_scaled_dot_product_attention(value).feats
                expected = []
                for segment in layout:
                    q, k, v = qkv[segment].unbind(1)
                    expected.append(
                        F.scaled_dot_product_attention(
                            q.transpose(0, 1).unsqueeze(0),
                            k.transpose(0, 1).unsqueeze(0),
                            v.transpose(0, 1).unsqueeze(0),
                        ).squeeze(0).transpose(0, 1)
                    )
                torch.testing.assert_close(actual, torch.cat(expected), atol=2e-5, rtol=2e-5)

    def test_bf16_sparse_to_dense_cross_attention(self) -> None:
        torch.manual_seed(23)
        layout = [slice(0, 2), slice(2, 5)]
        for device in devices():
            with self.subTest(device=device):
                q = torch.randn(5, 2, 4, device=device, dtype=torch.bfloat16)
                k = torch.randn(2, 3, 2, 4, device=device, dtype=torch.bfloat16)
                v = torch.randn(2, 3, 2, 5, device=device, dtype=torch.bfloat16)
                actual = sparse_scaled_dot_product_attention(
                    VarLenTensor(q, layout), k, v
                ).feats
                expected = []
                for index, segment in enumerate(layout):
                    expected.append(F.scaled_dot_product_attention(
                        q[segment].transpose(0, 1).unsqueeze(0),
                        k[index].transpose(0, 1).unsqueeze(0),
                        v[index].transpose(0, 1).unsqueeze(0),
                    ).squeeze(0).transpose(0, 1))
                torch.testing.assert_close(
                    actual.float().cpu(), torch.cat(expected).float().cpu(),
                    atol=3e-2, rtol=3e-2,
                )

    def test_variable_length_reduction_stays_on_backend(self) -> None:
        feats = torch.tensor([[1.0, 3.0], [5.0, 7.0], [10.0, 14.0]])
        layout = [slice(0, 2), slice(2, 3)]
        expected_mean = torch.tensor([[4.0], [12.0]])
        expected_std = torch.tensor([[math.sqrt(5.0)], [2.0]])
        for device in devices():
            with self.subTest(device=device):
                value = VarLenTensor(feats.to(device), layout)
                actual_mean = value.mean(dim=[1], keepdim=True)
                actual_std = value.std(dim=[1], keepdim=True)
                self.assertEqual(actual_mean.device.type, device)
                torch.testing.assert_close(actual_mean.cpu(), expected_mean)
                torch.testing.assert_close(actual_std.cpu(), expected_std)

    def test_windowed_attention_does_not_mix_windows(self) -> None:
        coords = torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 4, 0, 0], [0, 5, 0, 0]],
            dtype=torch.int32,
        )
        for device in devices():
            with self.subTest(device=device):
                qkv = torch.zeros(4, 3, 1, 1, device=device)
                qkv[:, 2, 0, 0] = torch.tensor([1, 3, 10, 14], device=device)
                tensor = SparseTensor(qkv, coords.to(device), shape=torch.Size((1, 3, 1, 1)))
                actual = sparse_windowed_scaled_dot_product_self_attention(
                    tensor, window_size=2
                ).feats[:, 0, 0]
                torch.testing.assert_close(
                    actual.cpu(), torch.tensor([2.0, 2.0, 12.0, 12.0])
                )

    def test_sparse_grid_sampling_matches_hand_computation(self) -> None:
        coords = torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.int32
        )
        feats = torch.tensor([[1.0], [3.0], [5.0]])
        grid = torch.tensor([[[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]])
        for device in devices():
            with self.subTest(device=device):
                actual = grid_sample_3d(
                    feats.to(device), coords.to(device), torch.Size((1, 1, 2, 2, 2)),
                    grid.to(device), mode="trilinear",
                ).cpu()
                torch.testing.assert_close(actual, torch.tensor([[[1.0], [2.0]]]))

    def test_sparse_grid_sampling_chunks_without_changing_results(self) -> None:
        coords = torch.tensor(
            [[0, x, 0, 0] for x in range(4)], dtype=torch.int32
        )
        feats = torch.tensor([[1.0], [3.0], [7.0], [15.0]])
        grid = torch.tensor([[[0.5, 0.5, 0.5], [1.0, 0.5, 0.5],
                              [2.0, 0.5, 0.5], [3.0, 0.5, 0.5]]])
        for device in devices():
            with self.subTest(device=device), mock.patch.object(
                grid_sample_module, "_SAMPLE_CHUNK_SIZE", 1
            ):
                actual = grid_sample_3d(
                    feats.to(device), coords.to(device),
                    torch.Size((1, 1, 4, 1, 1)), grid.to(device),
                )
                torch.testing.assert_close(
                    actual.cpu(), torch.tensor([[[1.0], [2.0], [5.0], [11.0]]])
                )

    def test_sparse_grid_sampling_renormalizes_missing_neighbors(self) -> None:
        coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
        feats = torch.tensor([[6.0]])
        grid = torch.tensor([[[0.75, 0.75, 0.75], [8.0, 8.0, 8.0]]])
        for device in devices():
            with self.subTest(device=device):
                actual = grid_sample_3d(
                    feats.to(device), coords.to(device),
                    torch.Size((1, 1, 2, 2, 2)), grid.to(device),
                )
                torch.testing.assert_close(
                    actual.cpu(), torch.tensor([[[6.0], [0.0]]])
                )

    def test_sparse_grid_sampling_accumulates_low_precision_in_float32(self) -> None:
        coords = torch.tensor(
            [[0, x, y, z] for x in range(2) for y in range(2) for z in range(2)],
            dtype=torch.int32,
        )
        feats = torch.tensor(
            [[1000.0], [0.125], [-999.0], [0.5], [1.0], [-0.25], [3.0], [0.75]]
        )
        grid = torch.tensor([[[0.9, 0.8, 0.7]]])
        shape = torch.Size((1, 1, 2, 2, 2))
        for device in devices():
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(device=device, dtype=dtype):
                    expected = grid_sample_3d(
                        feats.to(dtype).float(), coords, shape, grid
                    ).to(dtype).float()
                    actual = grid_sample_3d(
                        feats.to(device=device, dtype=dtype), coords.to(device), shape,
                        grid.to(device),
                    )
                    torch.testing.assert_close(
                        actual.float().cpu(), expected, atol=0, rtol=0
                    )

    def test_sparse_grid_sampling_rejects_feature_coordinate_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "matching coords"):
            grid_sample_3d(
                torch.ones(2, 1), torch.zeros(1, 4, dtype=torch.int32),
                torch.Size((1, 1, 2, 2, 2)), torch.zeros(1, 1, 3),
            )

    def test_sparse_nearest_sampling_truncates_like_cuda(self) -> None:
        coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32)
        feats = torch.tensor([[2.0], [9.0]])
        grid = torch.tensor([[[0.9, 0.0, 0.0], [1.1, 0.0, 0.0]]])
        for device in devices():
            with self.subTest(device=device):
                actual = grid_sample_3d(
                    feats.to(device), coords.to(device), torch.Size((1, 1, 2, 1, 1)),
                    grid.to(device), mode="nearest",
                )
                torch.testing.assert_close(actual.cpu(), torch.tensor([[[2.0], [9.0]]]))

    def test_sparse_convolution_rejects_even_kernel_size(self) -> None:
        from trellis2.modules import sparse as sp

        with self.assertRaisesRegex(ValueError, "positive odd kernel"):
            sp.SparseConv3d(2, 3, 2)

    def test_voxel_lookup_marks_missing_coordinates(self) -> None:
        coords = torch.tensor([[0, 0, 0], [1, 2, 3], [3, 3, 3]], dtype=torch.int32)
        queries = torch.tensor([[3, 3, 3], [0, 0, 1], [1, 2, 3]], dtype=torch.int32)
        for device in devices():
            with self.subTest(device=device):
                actual = voxel_lookup(coords.to(device), queries.to(device), (4, 4, 4))
                torch.testing.assert_close(actual.cpu(), torch.tensor([2, -1, 1]))

    def test_dual_grid_extracts_an_intersected_quad(self) -> None:
        coords = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=torch.int32
        )
        flags = torch.zeros((4, 3), dtype=torch.bool)
        flags[0, 0] = True
        for device in devices():
            with self.subTest(device=device):
                vertices, faces = flexible_dual_grid_to_mesh(
                    coords.to(device), torch.full((4, 3), 0.5, device=device),
                    flags.to(device), None,
                    aabb=[[0, 0, 0], [2, 2, 2]], grid_size=2,
                )
                self.assertEqual(vertices.shape, (4, 3))
                self.assertEqual(faces.shape, (2, 3))
                self.assertTrue(bool(((faces >= 0) & (faces < 4)).all()))

    def test_cpu_mesh_to_dual_grid_tetrahedron_contract(self) -> None:
        vertices = torch.tensor([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ])
        faces = torch.tensor([
            [0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3],
        ], dtype=torch.int32)
        coords, dual, intersected = mesh_to_flexible_dual_grid(
            vertices, faces, grid_size=8, aabb=[[0, 0, 0], [1, 1, 1]],
        )
        self.assertEqual(coords.shape, (152, 3))
        self.assertEqual(dual.shape, coords.shape)
        self.assertEqual(intersected.shape, coords.shape)
        self.assertEqual(coords.dtype, torch.int32)
        self.assertEqual(dual.dtype, torch.float32)
        self.assertEqual(intersected.dtype, torch.bool)
        self.assertTrue(bool(torch.isfinite(dual).all()))
        self.assertTrue(bool(((coords >= 0) & (coords < 8)).all()))
        torch.testing.assert_close(
            intersected.sum(dim=0), torch.tensor([47, 47, 47])
        )

    def test_cpu_mesh_to_dual_grid_rejects_bad_faces(self) -> None:
        with self.assertRaisesRegex(ValueError, "face index"):
            mesh_to_flexible_dual_grid(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 3]],
                grid_size=8, aabb=[[0, 0, 0], [1, 1, 1]],
            )

    def test_voxel_lookup_rejects_out_of_bounds_key_collisions(self) -> None:
        coords = torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.int32)
        queries = torch.tensor([[0, 0, 0], [-1, 2, 0]], dtype=torch.int32)
        for device in devices():
            with self.subTest(device=device):
                actual = voxel_lookup(coords.to(device), queries.to(device), 2)
                torch.testing.assert_close(actual.cpu(), torch.tensor([0, -1]))

    def test_portable_glb_export_preserves_predicted_vertex_color(self) -> None:
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        attrs = torch.tensor([[0.25, 0.5, 0.75, 1.0]])
        coords = torch.tensor([[0, 0, 0]], dtype=torch.int32)
        mesh = to_glb(
            vertices, faces, attrs, coords,
            attr_layout={"base_color": slice(0, 3), "alpha": slice(3, 4)},
            aabb=[[0, 0, 0], [1, 1, 1]], grid_size=[1, 1, 1],
        )
        self.assertEqual(mesh.visual.vertex_colors.shape, (3, 4))
        self.assertTrue((mesh.visual.vertex_colors[0] == [63, 127, 191, 255]).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.glb"
            mesh.export(path)
            self.assertGreater(path.stat().st_size, 100)

    def test_complete_pbr_layout_stays_vertex_color_without_opt_in(self) -> None:
        mesh = to_glb(
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([[0, 1, 2]], dtype=torch.int32),
            torch.tensor([[0.25, 0.5, 0.75, 0.1, 0.9, 1.0]]),
            torch.tensor([[0, 0, 0]], dtype=torch.int32),
            attr_layout={
                "base_color": slice(0, 3), "metallic": slice(3, 4),
                "roughness": slice(4, 5), "alpha": slice(5, 6),
            },
            aabb=[[0, 0, 0], [1, 1, 1]], grid_size=[1, 1, 1],
        )
        self.assertNotIn("kernel_goblin_pbr", mesh.metadata)
        self.assertEqual(mesh.visual.vertex_colors.shape, (3, 4))

    def test_portable_glb_uses_upstream_axis_conversion(self) -> None:
        vertices = torch.tensor(
            [[0.0, 1.0, 2.0], [1.0, 1.0, 2.0], [0.0, 2.0, 2.0]]
        )
        mesh = to_glb(
            vertices, torch.tensor([[0, 1, 2]], dtype=torch.int32),
            decimation_target=10,
        )
        torch.testing.assert_close(
            torch.from_numpy(mesh.vertices.copy()),
            torch.tensor([[0.0, 2.0, -1.0], [1.0, 2.0, -1.0], [0.0, 2.0, -2.0]], dtype=torch.float64),
        )


if __name__ == "__main__":
    unittest.main()
