#!/usr/bin/env python3
"""Export a complete real TRELLIS.2 sparse shape-decoder oracle on Torch MPS."""

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
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip() != REVISION or sha256(args.checkpoint) != CHECKPOINT_SHA256:
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
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder

    sparse_config.set_conv_backend("mps")
    device = torch.device("mps")
    decoder = SparseUnetVaeDecoder(
        out_channels=7,
        model_channels=[1024, 512, 256, 128, 64],
        latent_channels=32,
        num_blocks=[4, 16, 8, 4, 0],
        block_type=["SparseConvNeXtBlock3d"] * 5,
        up_block_type=["SparseResBlockC2S3d"] * 4,
        block_args=[{}] * 5,
        use_fp16=True,
        pred_subdiv=True,
    )
    decoder.load_state_dict(load_file(str(args.checkpoint), device="cpu"), strict=True)
    decoder.to(device).eval()
    indices = np.arange(32, dtype=np.uint32)
    bits = indices * np.uint32(1_103_515_245) + np.uint32(12_345)
    values = bits.view(np.int32).astype(np.float32) / np.float32(2_147_483_647)
    latent = torch.from_numpy(values.reshape(1, 32) * np.float32(0.25)).to(device)
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32, device=device)
    with torch.inference_mode():
        output, subdivisions = decoder(
            sp.SparseTensor(feats=latent, coords=coords), return_subs=True
        )
    torch.mps.synchronize()
    arrays = [("input", latent)]
    arrays += [(f"subdivision_{index}", value.feats) for index, value in enumerate(subdivisions)]
    arrays += [("raw_head", output.feats)]
    payload = bytearray()
    layout = {}
    for name, tensor in arrays:
        value = tensor.float().cpu().contiguous().numpy().astype("<f4", copy=False)
        layout[name] = {"offset_f32": len(payload) // 4, "shape": list(value.shape)}
        payload.extend(value.tobytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    metadata = {
        "format": "KernelGoblin complete TRELLIS.2 sparse shape decoder MPS oracle v1",
        "source_revision": REVISION,
        "model_source": str(MODEL_SOURCE.relative_to(UPSTREAM)),
        "model_source_sha256": sha256(MODEL_SOURCE),
        "overlay_source": str(OVERLAY_SOURCE.relative_to(ROOT)),
        "overlay_source_sha256": sha256(OVERLAY_SOURCE),
        "oracle_call": "SparseUnetVaeDecoder.forward(return_subs=True) with pinned MPS conv overlay",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "torch_version": torch.__version__,
        "pytorch_enable_mps_fallback": "0",
        "device": "mps",
        "input_coordinates": coords.int().cpu().tolist(),
        "output_coordinates": output.coords.int().cpu().tolist(),
        "trace_layout": layout,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
