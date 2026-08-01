"""Sparse nearest/trilinear sampling implemented with PyTorch searchsorted."""

from __future__ import annotations

import torch


_SAMPLE_CHUNK_SIZE = 131_072


def _prepare_lookup(coords, shape):
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("coords must have shape [N, 4]")
    if len(shape) != 5:
        raise ValueError("shape must be [batch, channels, width, height, depth]")
    _, _, width, height, depth = shape
    factors = torch.tensor(
        [width * height * depth, height * depth, depth, 1],
        device=coords.device,
        dtype=torch.int64,
    )
    keys = (coords.to(torch.int64) * factors).sum(-1)
    sorted_keys, order = torch.sort(keys)
    return factors, sorted_keys, order


def _lookup_prepared(queries, shape, factors, sorted_keys, order):
    batch, _, width, height, depth = shape
    query_keys = (queries.to(torch.int64) * factors).sum(-1)
    positions = torch.searchsorted(sorted_keys, query_keys)
    safe = positions.clamp(max=max(sorted_keys.numel() - 1, 0))
    valid = positions < sorted_keys.numel()
    bounds = torch.tensor([width, height, depth], device=queries.device)
    valid &= (
        (queries[:, 0] >= 0) & (queries[:, 0] < batch)
        & ((queries[:, 1:] >= 0) & (queries[:, 1:] < bounds)).all(dim=-1)
    )
    if sorted_keys.numel():
        valid &= sorted_keys[safe] == query_keys
    result = torch.full(query_keys.shape, -1, device=queries.device, dtype=torch.int64)
    if sorted_keys.numel():
        result[valid] = order[safe[valid]]
    return result


def _lookup(coords, queries, shape):
    return _lookup_prepared(queries, shape, *_prepare_lookup(coords, shape))


def grid_sample_3d(feats, coords, shape, grid, mode="trilinear"):
    if mode not in {"nearest", "trilinear"}:
        raise ValueError(f"unsupported sampling mode: {mode}")
    if feats.device != coords.device or feats.device != grid.device:
        raise ValueError("feats, coords, and grid must be on the same device")
    if feats.ndim != 2 or feats.shape[0] != coords.shape[0]:
        raise ValueError("feats must have shape [N, C] matching coords")
    if grid.ndim != 3 or grid.shape[-1] != 3:
        raise ValueError("grid must have shape [B, L, 3]")
    batch, length = grid.shape[:2]
    if batch != shape[0]:
        raise ValueError("grid batch dimension must match sparse tensor shape")
    channels = feats.shape[1]
    output = torch.zeros((batch * length, channels), device=feats.device, dtype=feats.dtype)
    flat_grid = grid.reshape(-1, 3)
    flat_batches = torch.arange(batch, device=grid.device).repeat_interleave(length)
    prepared = _prepare_lookup(coords, shape)
    offsets = torch.tensor([
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
    ], device=grid.device)

    for start in range(0, flat_grid.shape[0], _SAMPLE_CHUNK_SIZE):
        end = min(start + _SAMPLE_CHUNK_SIZE, flat_grid.shape[0])
        chunk_grid = flat_grid[start:end]
        chunk_batches = flat_batches[start:end]
        if mode == "nearest":
            # CUDA static_cast<int> truncates toward zero.
            queries = torch.cat(
                [chunk_batches[:, None], chunk_grid.to(torch.int64)], dim=-1
            )
            indices = _lookup_prepared(queries, shape, *prepared)
            valid = indices >= 0
            output[start:end][valid] = feats[indices[valid]]
            continue

        base = torch.floor(chunk_grid - 0.5).to(torch.int64)
        spatial = base[:, None, :] + offsets
        batches = chunk_batches[:, None, None].expand(-1, 8, 1)
        queries = torch.cat([batches, spatial], dim=-1)
        indices = _lookup_prepared(
            queries.reshape(-1, 4), shape, *prepared
        ).reshape(-1, 8)
        delta = (chunk_grid[:, None, :] - spatial - 0.5).abs()
        weights = (1 - delta).prod(dim=-1)
        valid = indices >= 0
        weights = weights * valid
        numerator = torch.zeros(
            (end - start, channels), device=feats.device, dtype=torch.float32
        )
        for neighbor in range(8):
            present = valid[:, neighbor]
            if bool(present.any()):
                numerator[present] += (
                    feats[indices[present, neighbor]].float()
                    * weights[present, neighbor, None]
                )
        denominator = weights.sum(-1, keepdim=True).clamp_min(1e-12)
        output[start:end] = (numerator / denominator).to(feats.dtype)
    return output.reshape(batch, length, channels)
