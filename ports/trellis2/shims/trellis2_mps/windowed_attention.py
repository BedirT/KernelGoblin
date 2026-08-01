"""Windowed sparse SDPA implementations for MPS."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def calc_window_partition(tensor, window_size, shift_window=0):
    dims = tensor.coords.shape[1] - 1
    shift = (shift_window,) * dims if isinstance(shift_window, int) else shift_window
    window = (window_size,) * dims if isinstance(window_size, int) else window_size
    coords = tensor.coords.clone().to(torch.int64)
    coords[:, 1:] += torch.tensor(shift, device=tensor.device)
    maxima = [value + delta for value, delta in zip(tensor.spatial_shape, shift)]
    counts = [math.ceil((value + 1) / size) for value, size in zip(maxima, window)]
    offsets = torch.cumprod(torch.tensor([1] + counts[::-1]), dim=0).tolist()[::-1]
    coords[:, 1:] //= torch.tensor(window, device=tensor.device)
    codes = (coords * torch.tensor(offsets, device=tensor.device)).sum(dim=1)
    forward = torch.argsort(codes)
    backward = torch.empty_like(forward)
    backward[forward] = torch.arange(forward.shape[0], device=tensor.device)
    _, lengths = torch.unique_consecutive(codes[forward], return_counts=True)
    return forward, backward, lengths, {}


def _segments(q, k, v, q_lengths, kv_lengths):
    outputs = []
    q_start = kv_start = 0
    for q_length, kv_length in zip(q_lengths.tolist(), kv_lengths.tolist()):
        qs = q[q_start:q_start + q_length].transpose(0, 1).unsqueeze(0)
        ks = k[kv_start:kv_start + kv_length].transpose(0, 1).unsqueeze(0)
        vs = v[kv_start:kv_start + kv_length].transpose(0, 1).unsqueeze(0)
        outputs.append(F.scaled_dot_product_attention(qs, ks, vs).squeeze(0).transpose(0, 1))
        q_start += q_length
        kv_start += kv_length
    return torch.cat(outputs, dim=0)


def sparse_windowed_scaled_dot_product_self_attention(qkv, window_size, shift_window=(0, 0, 0)):
    key = f"windowed_attention_{window_size}_{shift_window}"
    cached = qkv.get_spatial_cache(key)
    if cached is None:
        cached = calc_window_partition(qkv, window_size, shift_window)
        qkv.register_spatial_cache(key, cached)
    forward, backward, lengths, _ = cached
    q, k, v = qkv.feats[forward].unbind(dim=1)
    return qkv.replace(_segments(q, k, v, lengths, lengths)[backward])


def sparse_windowed_scaled_dot_product_cross_attention(
    q, kv, q_window_size, kv_window_size,
    q_shift_window=(0, 0, 0), kv_shift_window=(0, 0, 0),
):
    q_part = calc_window_partition(q, q_window_size, q_shift_window)
    kv_part = calc_window_partition(kv, kv_window_size, kv_shift_window)
    q_forward, q_backward, q_lengths, _ = q_part
    kv_forward, _, kv_lengths, _ = kv_part
    k, v = kv.feats[kv_forward].unbind(dim=1)
    out = _segments(q.feats[q_forward], k, v, q_lengths, kv_lengths)
    return q.replace(out[q_backward])
