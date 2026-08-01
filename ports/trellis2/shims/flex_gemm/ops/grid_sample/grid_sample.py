"""Sparse nearest/trilinear sampling implemented with PyTorch searchsorted."""

from __future__ import annotations

import torch


def _lookup(coords, queries, shape):
    _, width, height, depth = shape[-4:]
    factors = torch.tensor([width * height * depth, height * depth, depth, 1], device=coords.device, dtype=torch.int64)
    keys = (coords.to(torch.int64) * factors).sum(-1)
    sorted_keys, order = torch.sort(keys)
    query_keys = (queries.to(torch.int64) * factors).sum(-1)
    positions = torch.searchsorted(sorted_keys, query_keys)
    safe = positions.clamp(max=max(sorted_keys.numel() - 1, 0))
    valid = positions < sorted_keys.numel()
    bounds = torch.tensor([shape[-3], shape[-2], shape[-1]], device=coords.device)
    valid &= (
        (queries[:, 0] >= 0) & (queries[:, 0] < shape[0])
        & ((queries[:, 1:] >= 0) & (queries[:, 1:] < bounds)).all(dim=-1)
    )
    if sorted_keys.numel():
        valid &= sorted_keys[safe] == query_keys
    result = torch.full(query_keys.shape, -1, device=coords.device, dtype=torch.int64)
    if sorted_keys.numel():
        result[valid] = order[safe[valid]]
    return result


def grid_sample_3d(feats, coords, shape, grid, mode="trilinear"):
    batch, length = grid.shape[:2]
    channels = feats.shape[1]
    if mode == "nearest":
        queries = torch.cat([
            torch.arange(batch, device=grid.device)[:, None, None].expand(-1, length, 1),
            grid.to(torch.int64),
        ], dim=-1)
        indices = _lookup(coords, queries.reshape(-1, 4), shape)
        output = torch.zeros((batch * length, channels), device=feats.device, dtype=feats.dtype)
        valid = indices >= 0
        output[valid] = feats[indices[valid]]
        return output.reshape(batch, length, channels)

    base = torch.floor(grid - 0.5).to(torch.int64)
    offsets = torch.tensor([
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
    ], device=grid.device)
    spatial = base[:, :, None, :] + offsets
    batches = torch.arange(batch, device=grid.device)[:, None, None, None].expand(-1, length, 8, 1)
    queries = torch.cat([batches, spatial], dim=-1)
    indices = _lookup(coords, queries.reshape(-1, 4), shape).reshape(batch, length, 8)
    delta = (grid[:, :, None, :] - spatial - 0.5).abs()
    weights = (1 - delta).prod(dim=-1)
    valid = indices >= 0
    weights = weights * valid
    gathered = torch.zeros((batch, length, 8, channels), device=feats.device, dtype=feats.dtype)
    gathered[valid] = feats[indices[valid]]
    denominator = weights.sum(-1, keepdim=True).clamp_min(1e-12)
    return (gathered * weights.unsqueeze(-1)).sum(-2) / denominator
