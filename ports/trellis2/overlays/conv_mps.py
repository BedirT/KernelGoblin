"""Inference-oriented submanifold sparse convolution for PyTorch MPS."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .. import SparseTensor


_MAP_CHUNK_SIZE = 131_072
_CONV_CHUNK_SIZE = 32_768


def sparse_conv3d_init(
    self,
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    padding=None,
    bias=True,
    indice_key=None,
):
    assert stride == 1 and padding is None, (
        "MPS implementation supports submanifold convolution only"
    )
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = (
        tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
    )
    self.stride = (1, 1, 1)
    self.dilation = (
        tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3
    )
    weight = torch.empty((out_channels, in_channels, *self.kernel_size))
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    self.weight = nn.Parameter(weight.permute(0, 2, 3, 4, 1).contiguous())
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in else 0
        nn.init.uniform_(self.bias, -bound, bound)
    else:
        self.register_parameter("bias", None)
    self.indice_key = indice_key


def _neighbor_map(coords, spatial_shape, kernel_size, dilation):
    device = coords.device
    coords64 = coords.to(torch.int64)
    width, height, depth = (int(value) for value in spatial_shape)
    multipliers = torch.tensor(
        [width * height * depth, height * depth, depth, 1],
        device=device,
        dtype=torch.int64,
    )
    keys = (coords64 * multipliers).sum(dim=-1)
    sorted_keys, sorted_indices = torch.sort(keys)

    axes = [
        torch.arange(-(size // 2) * dil, (size // 2) * dil + 1, dil, device=device)
        for size, dil in zip(kernel_size, dilation)
    ]
    offsets = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    result = torch.full(
        (coords.shape[0], offsets.shape[0]), -1, device=device, dtype=torch.int32
    )
    if not sorted_keys.numel():
        return result

    for offset_index, offset in enumerate(offsets):
        for start in range(0, coords.shape[0], _MAP_CHUNK_SIZE):
            end = min(start + _MAP_CHUNK_SIZE, coords.shape[0])
            query = coords64[start:end].clone()
            query[:, 1:] += offset
            valid = (
                (query[:, 1] >= 0) & (query[:, 1] < width)
                & (query[:, 2] >= 0) & (query[:, 2] < height)
                & (query[:, 3] >= 0) & (query[:, 3] < depth)
            )
            query_keys = (query * multipliers).sum(dim=-1)
            positions = torch.searchsorted(sorted_keys, query_keys)
            safe_positions = positions.clamp(max=sorted_keys.numel() - 1)
            valid &= positions < sorted_keys.numel()
            valid &= sorted_keys[safe_positions] == query_keys
            chunk = result[start:end, offset_index]
            chunk[valid] = sorted_indices[safe_positions[valid]].to(torch.int32)
    return result


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    co, kd, kh, kw, ci = self.weight.shape
    cache_key = f"MPSSubMConv3d_{kw}x{kh}x{kd}_dilation{self.dilation}"
    neighbors = x.get_spatial_cache(cache_key)
    if neighbors is None:
        neighbors = _neighbor_map(
            x.coords, x.spatial_shape, (kd, kh, kw), self.dilation
        )
        x.register_spatial_cache(cache_key, neighbors)

    output = torch.zeros(
        (x.feats.shape[0], co), device=x.feats.device, dtype=x.feats.dtype
    )
    flat_weight = self.weight.reshape(co, -1, ci)
    for offset in range(neighbors.shape[1]):
        for start in range(0, neighbors.shape[0], _CONV_CHUNK_SIZE):
            end = min(start + _CONV_CHUNK_SIZE, neighbors.shape[0])
            source = neighbors[start:end, offset]
            valid = source >= 0
            if bool(valid.any()):
                contribution = (
                    x.feats[source[valid].to(torch.int64)]
                    @ flat_weight[:, offset, :].transpose(0, 1)
                )
                output[start:end][valid] += contribution
    if self.bias is not None:
        output += self.bias
    return x.replace(output)


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError("SparseInverseConv3d is not used by TRELLIS.2 inference")


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError("SparseInverseConv3d is not used by TRELLIS.2 inference")
