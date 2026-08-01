"""Pure PyTorch inference extraction for O-Voxel flexible dual grids."""

from __future__ import annotations

import torch


EDGE_OFFSETS = torch.tensor([
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
], dtype=torch.int64)
SPLIT_1 = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.int64)
SPLIT_2 = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.int64)


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
