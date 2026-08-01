#!/usr/bin/env python3
"""Export a real shape-decoder ConvNeXt block oracle from physical Torch MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


PORT = Path(__file__).resolve().parents[1]
ROOT = PORT.parents[1]
UPSTREAM = ROOT / "build" / "trellis2" / "upstream"
REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
CHECKPOINT_SHA256 = "e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581"
MODEL_SOURCE = UPSTREAM / "trellis2" / "models" / "sc_vaes" / "sparse_unet_vae.py"
OVERLAY_SOURCE = PORT / "overlays" / "conv_mps.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip()
    if revision != REVISION:
        raise SystemExit(f"upstream checkout is {revision}, expected {REVISION}")
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise SystemExit("shape decoder checkpoint SHA-256 does not match the pinned model")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise SystemExit("set PYTORCH_ENABLE_MPS_FALLBACK=0")
    if not torch.backends.mps.is_available():
        raise SystemExit("Torch MPS is unavailable")

    device = torch.device("mps")
    prefix = "blocks.0.0."
    state = load_file(str(args.checkpoint), device="cpu")
    block_state = {key.removeprefix(prefix): value for key, value in state.items()
                   if key.startswith(prefix)}
    coordinates = np.asarray(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1]],
        dtype=np.int32,
    )
    lookup = {tuple(value): index for index, value in enumerate(coordinates.tolist())}
    neighbors = np.full((len(coordinates), 27), -1, dtype=np.int32)
    offset = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for index, coordinate in enumerate(coordinates):
                    query = tuple((coordinate + np.asarray([0, dx, dy, dz])).tolist())
                    neighbors[index, offset] = lookup.get(query, -1)
                offset += 1

    indices = np.arange(len(coordinates) * 1024, dtype=np.uint32)
    bits = indices * np.uint32(1_664_525) + np.uint32(1_013_904_223)
    values = bits.view(np.int32).astype(np.float32) / np.float32(2_147_483_647)
    x = torch.from_numpy(values.reshape(len(coordinates), 1024) * np.float32(0.125))
    x = x.to(device=device, dtype=torch.float16)
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    os.environ["SPARSE_CONV_BACKEND"] = "mps"
    from trellis2.modules import sparse as sp
    from trellis2.modules.sparse import config as sparse_config
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseConvNeXtBlock3d
    from trellis2.modules.utils import convert_module_to_f16

    sparse_config.set_conv_backend("mps")
    installed_overlay = (
        UPSTREAM / "trellis2" / "modules" / "sparse" / "conv" / "conv_mps.py"
    )
    if sha256(installed_overlay) != sha256(OVERLAY_SOURCE):
        raise SystemExit("installed MPS sparse-convolution overlay is not pinned")
    block = SparseConvNeXtBlock3d(1024)
    block.load_state_dict(block_state, strict=True)
    block.apply(convert_module_to_f16)
    block.to(device).eval()
    traces: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            traces[name] = (output.feats if isinstance(output, sp.SparseTensor) else output).detach()
        return hook

    hooks = [
        block.conv.register_forward_hook(capture("conv")),
        block.norm.register_forward_hook(capture("norm")),
        block.mlp[0].register_forward_hook(capture("mlp_up")),
        block.mlp[1].register_forward_hook(capture("silu")),
        block.mlp[2].register_forward_hook(capture("mlp_down")),
    ]
    with torch.inference_mode():
        sparse_input = sp.SparseTensor(
            feats=x, coords=torch.from_numpy(coordinates).to(device)
        )
        output = block(sparse_input).feats
    for hook in hooks:
        hook.remove()
    torch.mps.synchronize()

    arrays = [
        ("input", x), ("conv", traces["conv"]), ("norm", traces["norm"]),
        ("mlp_up", traces["mlp_up"]), ("silu", traces["silu"]),
        ("mlp_down", traces["mlp_down"]), ("output", output),
    ]
    payload = bytearray()
    traces = {}
    for name, tensor in arrays:
        value = tensor.float().cpu().contiguous().numpy().astype("<f4", copy=False)
        traces[name] = {"offset_f32": len(payload) // 4, "shape": list(value.shape)}
        payload.extend(value.tobytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    metadata = {
        "format": "KernelGoblin TRELLIS.2 shape decoder ConvNeXt block 0.0 MPS oracle v1",
        "source_revision": REVISION,
        "model_source": str(MODEL_SOURCE.relative_to(UPSTREAM)),
        "model_source_sha256": sha256(MODEL_SOURCE),
        "overlay_source": str(OVERLAY_SOURCE.relative_to(ROOT)),
        "overlay_source_sha256": sha256(OVERLAY_SOURCE),
        "oracle_call": "SparseConvNeXtBlock3d(1024).forward with pinned MPS conv overlay",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "torch_version": torch.__version__,
        "pytorch_enable_mps_fallback": "0",
        "device": "mps",
        "coordinates": coordinates.tolist(),
        "spatial_shape": [2, 2, 2],
        "neighbor_offset_order": "x-outer,y-middle,z-inner over -1,0,1",
        "traces": traces,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
