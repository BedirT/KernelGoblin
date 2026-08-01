#!/usr/bin/env python3
"""Export the complete real TRELLIS.2 sparse shape encoder on Torch MPS."""

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
CHECKPOINT_SHA256 = "f37c5ff5b983b68e9946060000f09bc131f3e84318a2c8b7430a81e4b4636c41"
MODEL_SOURCE = UPSTREAM / "trellis2" / "models" / "sc_vaes" / "sparse_unet_vae.py"
SPATIAL_SOURCE = (
    UPSTREAM / "trellis2" / "modules" / "sparse" / "spatial" / "spatial2channel.py"
)
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
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip() != REVISION or sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise SystemExit("upstream or shape encoder checkpoint is not pinned")
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
    from trellis2.models.sc_vaes.sparse_unet_vae import (
        SparseResBlockS2C3d,
        SparseUnetVaeEncoder,
    )

    sparse_config.set_conv_backend("mps")
    device = torch.device("mps")
    encoder = SparseUnetVaeEncoder(
        in_channels=6,
        model_channels=[64, 128, 256, 512, 1024],
        latent_channels=32,
        num_blocks=[0, 4, 8, 16, 4],
        block_type=["SparseConvNeXtBlock3d"] * 5,
        down_block_type=["SparseResBlockS2C3d"] * 4,
        block_args=[{}] * 5,
        use_fp16=True,
    )
    encoder.load_state_dict(load_file(str(args.checkpoint), device="cpu"), strict=True)
    encoder.to(device).eval()
    values = np.asarray(
        [[-0.25, 0.125, -0.375, 0.5, -0.5, 0.5]], dtype=np.float32
    )
    features = torch.from_numpy(values).to(device)
    coords = torch.tensor([[0, 15, 15, 15]], dtype=torch.int32, device=device)
    downsamples: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture_downsample(_module, _inputs, output):
        downsamples.append((output.coords.detach(), output.feats.detach()))

    hooks = [
        module.register_forward_hook(capture_downsample)
        for module in encoder.modules()
        if isinstance(module, SparseResBlockS2C3d)
    ]
    with torch.inference_mode():
        latent = encoder(sp.SparseTensor(feats=features, coords=coords))
    for hook in hooks:
        hook.remove()
    torch.mps.synchronize()

    arrays = [("input", features)]
    arrays += [
        (f"downsample_{index}", tensor) for index, (_coords, tensor) in enumerate(downsamples)
    ]
    arrays.append(("latent", latent.feats))
    payload = bytearray()
    layout = {}
    for name, tensor in arrays:
        value = tensor.float().cpu().contiguous().numpy().astype("<f4", copy=False)
        layout[name] = {"offset_f32": len(payload) // 4, "shape": list(value.shape)}
        payload.extend(value.tobytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    metadata = {
        "format": "KernelGoblin complete TRELLIS.2 sparse shape encoder MPS oracle v1",
        "source_revision": REVISION,
        "model_source": str(MODEL_SOURCE.relative_to(UPSTREAM)),
        "model_source_sha256": sha256(MODEL_SOURCE),
        "spatial_source": str(SPATIAL_SOURCE.relative_to(UPSTREAM)),
        "spatial_source_sha256": sha256(SPATIAL_SOURCE),
        "overlay_source": str(OVERLAY_SOURCE.relative_to(ROOT)),
        "overlay_source_sha256": sha256(OVERLAY_SOURCE),
        "oracle_call": "SparseUnetVaeEncoder.forward with pinned MPS conv overlay",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "torch_version": torch.__version__,
        "pytorch_enable_mps_fallback": "0",
        "device": "mps",
        "input_coordinates": coords.int().cpu().tolist(),
        "downsample_coordinates": [
            value.int().cpu().tolist() for value, _features in downsamples
        ],
        "output_coordinates": latent.coords.int().cpu().tolist(),
        "trace_layout": layout,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
