"""Portable mesh export fallback for generated TRELLIS.2 results."""

from __future__ import annotations

import numpy as np
import torch
import trimesh

from flex_gemm.ops.grid_sample import grid_sample_3d


def to_glb(
    vertices, faces, attr_volume=None, coords=None, attr_layout=None,
    aabb=((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)), voxel_size=None,
    grid_size=None, decimation_target=1_000_000, **kwargs,
):
    vertices_np = vertices.detach().cpu().numpy().copy()
    vertices_np[:, 1], vertices_np[:, 2] = (
        vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
    )
    mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces.detach().cpu().numpy(),
        process=False,
    )
    if attr_volume is not None and coords is not None and len(vertices):
        aabb_tensor = torch.as_tensor(aabb, device=vertices.device, dtype=torch.float32)
        if voxel_size is None:
            grid_size = torch.as_tensor(grid_size, device=vertices.device, dtype=torch.int64)
            voxel = (aabb_tensor[1] - aabb_tensor[0]) / grid_size
        else:
            voxel = torch.as_tensor(voxel_size, device=vertices.device, dtype=torch.float32)
            if voxel.ndim == 0:
                voxel = voxel.repeat(3)
            grid_size = ((aabb_tensor[1] - aabb_tensor[0]) / voxel).round().to(torch.int64)
        queries = ((vertices - aabb_tensor[0]) / voxel).reshape(1, -1, 3)
        batched_coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
        shape = torch.Size([1, attr_volume.shape[1], *grid_size.tolist()])
        attrs = grid_sample_3d(attr_volume, batched_coords, shape, queries)[0]
        color_slice = (attr_layout or {}).get("base_color", slice(0, 3))
        rgb = attrs[:, color_slice].clamp(0, 1).detach().cpu().numpy()
        alpha_slice = (attr_layout or {}).get("alpha")
        alpha = (
            attrs[:, alpha_slice].clamp(0, 1).detach().cpu().numpy()
            if alpha_slice is not None else np.ones((len(vertices), 1), dtype=np.float32)
        )
        mesh.visual.vertex_colors = np.clip(
            np.concatenate([rgb, alpha], axis=1) * 255, 0, 255
        ).astype(np.uint8)
    if len(mesh.faces) > decimation_target:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=decimation_target)
        except BaseException:
            pass
    return mesh
