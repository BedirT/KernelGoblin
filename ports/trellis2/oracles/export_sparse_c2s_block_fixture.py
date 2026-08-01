#!/usr/bin/env python3
"""Export real TRELLIS.2 shape-decoder C2S block boundaries on Torch MPS."""

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
    if revision != REVISION or sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise SystemExit("upstream or shape decoder checkpoint is not pinned")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise SystemExit("set PYTORCH_ENABLE_MPS_FALLBACK=0")
    if not torch.backends.mps.is_available():
        raise SystemExit("Torch MPS is unavailable")
    installed_overlay = (
        UPSTREAM / "trellis2" / "modules" / "sparse" / "conv" / "conv_mps.py"
    )
    if sha256(installed_overlay) != sha256(OVERLAY_SOURCE):
        raise SystemExit("installed MPS sparse-convolution overlay is not pinned")

    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    os.environ["SPARSE_CONV_BACKEND"] = "mps"
    from trellis2.modules import sparse as sp
    from trellis2.modules.sparse import config as sparse_config
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseResBlockC2S3d
    from trellis2.modules.utils import convert_module_to_f16

    sparse_config.set_conv_backend("mps")
    device = torch.device("mps")
    prefix = "blocks.0.4."
    state = load_file(str(args.checkpoint), device="cpu")
    block_state = {key.removeprefix(prefix): value for key, value in state.items()
                   if key.startswith(prefix)}
    block = SparseResBlockC2S3d(1024, 512, pred_subdiv=True)
    block.load_state_dict(block_state, strict=True)
    block.apply(convert_module_to_f16)
    block.to(device).eval()
    coordinates = np.asarray(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0]], dtype=np.int32
    )
    indices = np.arange(len(coordinates) * 1024, dtype=np.uint32)
    bits = indices * np.uint32(22_695_477) + np.uint32(1)
    values = bits.view(np.int32).astype(np.float32) / np.float32(2_147_483_647)
    x = torch.from_numpy(values.reshape(len(coordinates), 1024) * np.float32(0.125))
    x = x.to(device=device, dtype=torch.float16)
    traces: dict[str, torch.Tensor] = {}
    upsampled: list[torch.Tensor] = []

    def capture(name: str):
        def hook(_module, _inputs, output):
            value = output.feats if isinstance(output, sp.SparseTensor) else output
            traces[name] = value.detach()
        return hook

    def capture_up(_module, _inputs, output):
        upsampled.append(output.feats.detach())

    hooks = [
        block.to_subdiv.register_forward_hook(capture("subdivision")),
        block.norm1.register_forward_hook(capture("norm1")),
        block.conv1.register_forward_hook(capture("conv1")),
        block.updown.register_forward_hook(capture_up),
        block.norm2.register_forward_hook(capture("norm2")),
        block.conv2.register_forward_hook(capture("conv2")),
    ]
    with torch.inference_mode():
        result, subdivision = block(sp.SparseTensor(
            feats=x, coords=torch.from_numpy(coordinates).to(device)
        ))
    for hook in hooks:
        hook.remove()
    torch.mps.synchronize()
    arrays = [
        ("input", x), ("subdivision", subdivision.feats),
        ("norm1", traces["norm1"]), ("conv1", traces["conv1"]),
        ("upsampled_conv", upsampled[0]), ("upsampled_skip", upsampled[1]),
        ("norm2", traces["norm2"]), ("conv2", traces["conv2"]),
        ("output", result.feats),
    ]
    payload = bytearray()
    trace_layout = {}
    for name, tensor in arrays:
        value = tensor.float().cpu().contiguous().numpy().astype("<f4", copy=False)
        trace_layout[name] = {"offset_f32": len(payload) // 4, "shape": list(value.shape)}
        payload.extend(value.tobytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    metadata = {
        "format": "KernelGoblin TRELLIS.2 shape decoder C2S block 0.4 MPS oracle v1",
        "source_revision": REVISION,
        "model_source": str(MODEL_SOURCE.relative_to(UPSTREAM)),
        "model_source_sha256": sha256(MODEL_SOURCE),
        "overlay_source": str(OVERLAY_SOURCE.relative_to(ROOT)),
        "overlay_source_sha256": sha256(OVERLAY_SOURCE),
        "oracle_call": "SparseResBlockC2S3d(1024,512).forward with pinned MPS conv overlay",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "torch_version": torch.__version__,
        "pytorch_enable_mps_fallback": "0",
        "device": "mps",
        "coordinates": coordinates.tolist(),
        "output_coordinates": result.coords.int().cpu().tolist(),
        "trace_layout": trace_layout,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
