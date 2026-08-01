#!/usr/bin/env python3
"""Export the guided TRELLIS.2 sparse texture-decoder graph on Torch MPS."""

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
SHAPE_CHECKPOINT_SHA256 = "e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581"
TEXTURE_CHECKPOINT_SHA256 = "97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c"
MODEL_SOURCE = UPSTREAM / "trellis2" / "models" / "sc_vaes" / "sparse_unet_vae.py"
PIPELINE_SOURCE = UPSTREAM / "trellis2" / "pipelines" / "trellis2_image_to_3d.py"
OVERLAY_SOURCE = PORT / "overlays" / "conv_mps.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoder(*, out_channels: int, pred_subdiv: bool):
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
    return SparseUnetVaeDecoder(
        out_channels=out_channels,
        model_channels=[1024, 512, 256, 128, 64],
        latent_channels=32,
        num_blocks=[4, 16, 8, 4, 0],
        block_type=["SparseConvNeXtBlock3d"] * 5,
        up_block_type=["SparseResBlockC2S3d"] * 4,
        block_args=[{}] * 5,
        use_fp16=True,
        pred_subdiv=pred_subdiv,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape-checkpoint", type=Path, required=True)
    parser.add_argument("--texture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip() != REVISION:
        raise SystemExit("upstream checkout is not pinned")
    if sha256(args.shape_checkpoint) != SHAPE_CHECKPOINT_SHA256:
        raise SystemExit("shape decoder checkpoint is not pinned")
    if sha256(args.texture_checkpoint) != TEXTURE_CHECKPOINT_SHA256:
        raise SystemExit("texture decoder checkpoint is not pinned")
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

    sparse_config.set_conv_backend("mps")
    device = torch.device("mps")
    shape = decoder(out_channels=7, pred_subdiv=True)
    texture = decoder(out_channels=6, pred_subdiv=False)
    shape.load_state_dict(load_file(str(args.shape_checkpoint), device="cpu"), strict=True)
    texture.load_state_dict(
        load_file(str(args.texture_checkpoint), device="cpu"), strict=True
    )
    shape.to(device).eval()
    texture.to(device).eval()
    shape_indices = np.arange(32, dtype=np.uint32)
    shape_bits = shape_indices * np.uint32(1_103_515_245) + np.uint32(12_345)
    shape_values = shape_bits.view(np.int32).astype(np.float32) / np.float32(2_147_483_647)
    shape_latent = torch.from_numpy(
        shape_values.reshape(1, 32) * np.float32(0.25)
    ).to(device)
    texture_indices = np.arange(32, dtype=np.uint32)
    texture_bits = texture_indices * np.uint32(747_796_405) + np.uint32(2_891_336_453)
    texture_values = texture_bits.view(np.int32).astype(np.float32) / np.float32(2_147_483_647)
    texture_latent = torch.from_numpy(
        texture_values.reshape(1, 32) * np.float32(0.375)
    ).to(device)
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32, device=device)
    with torch.inference_mode():
        _, guides = shape(
            sp.SparseTensor(feats=shape_latent, coords=coords), return_subs=True
        )
        raw = texture(
            sp.SparseTensor(feats=texture_latent, coords=coords), guide_subs=guides
        )
        pbr = raw * 0.5 + 0.5
    torch.mps.synchronize()
    arrays = [("input", texture_latent), ("raw_head", raw.feats), ("pbr", pbr.feats)]
    payload = bytearray()
    layout = {}
    for name, tensor in arrays:
        value = tensor.float().cpu().contiguous().numpy().astype("<f4", copy=False)
        layout[name] = {"offset_f32": len(payload) // 4, "shape": list(value.shape)}
        payload.extend(value.tobytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    metadata = {
        "format": "KernelGoblin complete TRELLIS.2 guided texture decoder MPS oracle v1",
        "source_revision": REVISION,
        "model_source": str(MODEL_SOURCE.relative_to(UPSTREAM)),
        "model_source_sha256": sha256(MODEL_SOURCE),
        "pipeline_source": str(PIPELINE_SOURCE.relative_to(UPSTREAM)),
        "pipeline_source_sha256": sha256(PIPELINE_SOURCE),
        "overlay_source": str(OVERLAY_SOURCE.relative_to(ROOT)),
        "overlay_source_sha256": sha256(OVERLAY_SOURCE),
        "oracle_call": "SparseUnetVaeDecoder(pred_subdiv=False).forward(guide_subs=shape_subs)",
        "pipeline_transform": "raw * 0.5 + 0.5",
        "shape_checkpoint_sha256": SHAPE_CHECKPOINT_SHA256,
        "texture_checkpoint_sha256": TEXTURE_CHECKPOINT_SHA256,
        "torch_version": torch.__version__,
        "pytorch_enable_mps_fallback": "0",
        "device": "mps",
        "input_coordinates": coords.int().cpu().tolist(),
        "output_coordinates": raw.coords.int().cpu().tolist(),
        "trace_layout": layout,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
