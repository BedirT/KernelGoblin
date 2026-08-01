"""Variable-length SDPA implementation backed by PyTorch MPS."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _sdpa(q, k, v):
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    return F.scaled_dot_product_attention(q, k, v).squeeze(0).transpose(0, 1)


def sparse_scaled_dot_product_attention(*args, **kwargs):
    from trellis2.modules.sparse import VarLenTensor

    values = list(args)
    names = {1: ("qkv",), 2: ("q", "kv"), 3: ("q", "k", "v")}
    count = len(args) + len(kwargs)
    if count not in names:
        raise ValueError(f"expected 1, 2, or 3 attention inputs, got {count}")
    for name in names[count][len(args):]:
        values.append(kwargs[name])

    template = next((value for value in values if isinstance(value, VarLenTensor)), None)
    batch = values[0].shape[0]
    outputs = []
    for index in range(batch):
        sliced = []
        for value in values:
            if isinstance(value, VarLenTensor):
                sliced.append(value.feats[value.layout[index]])
            else:
                sliced.append(value[index])
        if count == 1:
            q, k, v = sliced[0].unbind(dim=1)
        elif count == 2:
            q = sliced[0]
            k, v = sliced[1].unbind(dim=1)
        else:
            q, k, v = sliced
        outputs.append(_sdpa(q, k, v))

    result = torch.cat(outputs, dim=0)
    return template.replace(result) if template is not None else torch.stack(outputs)
