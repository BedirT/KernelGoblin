"""CPU mesh conversion and PyTorch extraction for O-Voxel flexible dual grids."""

from __future__ import annotations

import torch
import importlib.util
from pathlib import Path


EDGE_OFFSETS = torch.tensor([
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
], dtype=torch.int64)
SPLIT_1 = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.int64)
SPLIT_2 = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.int64)


def _cpu_extension():
    root = Path(__file__).resolve().parents[5]
    matches = list((root / "build" / "trellis2" / "fdg").glob("kg_trellis2_fdg*.so"))
    if len(matches) != 1:
        raise RuntimeError(
            "CPU dual-grid extension is not built; run `./kg model setup trellis2`"
        )
    spec = importlib.util.spec_from_file_location("kg_trellis2_fdg", matches[0])
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load CPU dual-grid extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def mesh_to_flexible_dual_grid(
    vertices, faces, voxel_size=None, grid_size=None, aabb=None,
    face_weight=1.0, boundary_weight=1.0, regularization_weight=0.1,
    timing=False,
):
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device="cpu").contiguous()
    faces = torch.as_tensor(faces, dtype=torch.int32, device="cpu").contiguous()
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("vertices must be a nonempty [V, 3] tensor")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("faces must be a nonempty [F, 3] tensor")
    if not bool(torch.isfinite(vertices).all()):
        raise ValueError("vertices must be finite")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError("face index exceeds vertex count")
    if voxel_size is None and grid_size is None:
        raise ValueError("voxel_size or grid_size is required")
    if aabb is None:
        minimum = vertices.min(dim=0).values
        maximum = vertices.max(dim=0).values
        if bool(((maximum - minimum) <= torch.finfo(torch.float32).eps).all()):
            raise ValueError("mesh must have nonzero spatial extent")
        if voxel_size is not None:
            voxel = torch.as_tensor(voxel_size, dtype=torch.float32)
            if voxel.ndim == 0:
                voxel = voxel.repeat(3)
            padding = torch.ceil((maximum - minimum) / voxel) * voxel - (maximum - minimum)
            minimum -= padding * 0.5
            maximum += padding * 0.5
        else:
            grid = torch.as_tensor(grid_size, dtype=torch.int32)
            if grid.ndim == 0:
                grid = grid.repeat(3)
            if bool((grid <= 1).any()):
                raise ValueError("automatic AABB requires grid_size greater than one")
            padding = (maximum - minimum) / (grid - 1)
            minimum -= padding * 0.5
            maximum += padding * 0.5
        aabb = torch.stack([minimum, maximum])
    else:
        aabb = torch.as_tensor(aabb, dtype=torch.float32, device="cpu")
    if aabb.shape != (2, 3) or not bool(torch.isfinite(aabb).all()):
        raise ValueError("aabb must be a finite [2, 3] tensor")
    if voxel_size is None:
        grid = torch.as_tensor(grid_size, dtype=torch.int32, device="cpu")
        if grid.ndim == 0:
            grid = grid.repeat(3)
        voxel = (aabb[1] - aabb[0]) / grid
    else:
        voxel = torch.as_tensor(voxel_size, dtype=torch.float32, device="cpu")
        if voxel.ndim == 0:
            voxel = voxel.repeat(3)
        grid = ((aabb[1] - aabb[0]) / voxel).round().to(torch.int32)
    if voxel.shape != (3,) or grid.shape != (3,) or bool((voxel <= 0).any()) or bool((grid <= 0).any()):
        raise ValueError("voxel_size and grid_size must be positive 3-vectors")
    local_vertices = (vertices - aabb[0]).contiguous()
    grid_range = torch.stack([torch.zeros_like(grid), grid]).to(torch.int32).contiguous()
    return _cpu_extension().mesh_to_flexible_dual_grid_cpu(
        local_vertices, faces, voxel.contiguous(), grid_range,
        float(face_weight), float(boundary_weight), float(regularization_weight),
        bool(timing),
    )


def _lookup(coords, queries, grid_size):
    device = coords.device
    grid_size = torch.as_tensor(grid_size, device=device, dtype=torch.int64)
    if grid_size.ndim == 0:
        grid_size = grid_size.repeat(3)
    multipliers = torch.tensor(
        [int(grid_size[1]) * int(grid_size[2]), int(grid_size[2]), 1],
        device=device,
        dtype=torch.int64,
    )
    keys = (coords.to(torch.int64) * multipliers).sum(dim=-1)
    sorted_keys, order = torch.sort(keys)
    query_keys = (queries.to(torch.int64) * multipliers).sum(dim=-1)
    positions = torch.searchsorted(sorted_keys, query_keys)
    safe = positions.clamp(max=max(sorted_keys.numel() - 1, 0))
    valid = positions < sorted_keys.numel()
    bounds = torch.as_tensor(grid_size, device=device, dtype=torch.int64)
    valid &= ((queries >= 0) & (queries < bounds)).all(dim=-1)
    if sorted_keys.numel():
        valid &= sorted_keys[safe] == query_keys
    indices = torch.full(query_keys.shape, -1, device=device, dtype=torch.int64)
    if sorted_keys.numel():
        indices[valid] = order[safe[valid]]
    return indices


def flexible_dual_grid_to_mesh(
    coords, dual_vertices, intersected_flag, split_weight,
    aabb, voxel_size=None, grid_size=None, train=False,
):
    if train:
        raise NotImplementedError("the MPS port implements inference extraction only")
    device = coords.device
    aabb = torch.as_tensor(aabb, dtype=torch.float32, device=device)
    if voxel_size is None:
        grid_size = torch.as_tensor(grid_size, dtype=torch.int64, device=device)
        if grid_size.ndim == 0:
            grid_size = grid_size.repeat(3)
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    else:
        voxel_size = torch.as_tensor(voxel_size, dtype=torch.float32, device=device)
        if voxel_size.ndim == 0:
            voxel_size = voxel_size.repeat(3)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().to(torch.int64)

    vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0]
    offsets = EDGE_OFFSETS.to(device)
    connected = (coords.to(torch.int64)[:, None, None, :] + offsets[None])[intersected_flag.bool()]
    if connected.numel() == 0:
        return vertices, torch.empty((0, 3), dtype=torch.int32, device=device)
    quads = _lookup(coords, connected.reshape(-1, 3), grid_size).reshape(-1, 4)
    quads = quads[(quads >= 0).all(dim=1)]
    if quads.numel() == 0:
        return vertices, torch.empty((0, 3), dtype=torch.int32, device=device)

    split1 = SPLIT_1.to(device)
    split2 = SPLIT_2.to(device)
    if split_weight is None:
        triangles0 = quads[:, split1]
        triangles1 = quads[:, split2]
        normal00 = torch.cross(vertices[triangles0[:, 1]] - vertices[triangles0[:, 0]], vertices[triangles0[:, 2]] - vertices[triangles0[:, 0]], dim=1)
        normal01 = torch.cross(vertices[triangles0[:, 2]] - vertices[triangles0[:, 1]], vertices[triangles0[:, 3]] - vertices[triangles0[:, 1]], dim=1)
        normal10 = torch.cross(vertices[triangles1[:, 1]] - vertices[triangles1[:, 0]], vertices[triangles1[:, 2]] - vertices[triangles1[:, 0]], dim=1)
        normal11 = torch.cross(vertices[triangles1[:, 2]] - vertices[triangles1[:, 1]], vertices[triangles1[:, 3]] - vertices[triangles1[:, 1]], dim=1)
        choose_first = (normal00 * normal01).sum(1).abs() > (normal10 * normal11).sum(1).abs()
    else:
        weights = split_weight[quads]
        choose_first = (weights[:, 0] * weights[:, 2]).squeeze(-1) > (weights[:, 1] * weights[:, 3]).squeeze(-1)
    faces = torch.where(choose_first[:, None], quads[:, split1], quads[:, split2]).reshape(-1, 3)
    return vertices, faces.to(torch.int32)
