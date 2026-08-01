#!/usr/bin/env python3
"""Capture a full-weight tiny spatial fixture for the sparse decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


PORT = Path(__file__).resolve().parents[1]
ROOT = PORT.parents[1]
UPSTREAM = ROOT / "build" / "trellis2" / "upstream"
sys.path.insert(0, str(PORT / "shims"))
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(PORT))

import runtime_env

runtime_env.activate()

SOURCE_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
WEIGHT_REVISION = "25e0d31ffbebe4b5a97464dd851910efc3002d96"
WEIGHT_SHA256 = "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_pinned_source() -> None:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise SystemExit(f"upstream checkout is {revision}, expected {SOURCE_REVISION}")
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], cwd=UPSTREAM
    ).returncode != 0:
        raise SystemExit("upstream checkout has tracked modifications")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-resolution", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    require_pinned_source()
    if sha256(args.checkpoint) != WEIGHT_SHA256:
        raise SystemExit("checkpoint SHA-256 does not match the pinned sparse decoder")
    if args.input_resolution <= 0:
        raise SystemExit("input resolution must be positive")

    from trellis2.models.sparse_structure_vae import SparseStructureDecoder

    model = SparseStructureDecoder(
        out_channels=1,
        latent_channels=8,
        num_res_blocks=2,
        num_res_blocks_middle=2,
        channels=[512, 128, 32],
        use_fp16=True,
    ).eval()
    model.load_state_dict(load_file(args.checkpoint), strict=True)
    model.to(torch.device(args.device))
    resolution = args.input_resolution
    token_major = torch.tensor(
        [
            math.sin(index * 0.071) * 0.35 + math.cos(index * 0.019) * 0.1
            for index in range(resolution**3 * 8)
        ],
        dtype=torch.float32,
    ).reshape(resolution, resolution, resolution, 8)
    latent = token_major.permute(3, 0, 1, 2).unsqueeze(0).contiguous().to(args.device)
    with torch.inference_mode():
        logits = model(latent).float().contiguous()
    output_resolution = resolution * 4
    assert list(logits.shape) == [1, 1, output_resolution, output_resolution, output_resolution]

    latent_bytes = token_major.contiguous().numpy().tobytes()
    output_bytes = logits.cpu().numpy().tobytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(latent_bytes + output_bytes)
    metadata = {
        "format": "KernelGoblin TRELLIS.2 sparse-structure decoder F32 oracle v1",
        "source_revision": SOURCE_REVISION,
        "weight_revision": WEIGHT_REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "input_resolution": resolution,
        "output_resolution": output_resolution,
        "input_layout": "voxel-major [x,y,z,channel] F32",
        "output_layout": "voxel-major [x,y,z,channel=1] F32",
        "input_formula": "sin(i * 0.071) * 0.35 + cos(i * 0.019) * 0.1",
        "oracle_device": args.device,
        "input_range": [0, len(latent_bytes)],
        "output_range": [len(latent_bytes), len(latent_bytes) + len(output_bytes)],
        "payload_sha256": hashlib.sha256(latent_bytes + output_bytes).hexdigest(),
        "oracle": f"pinned Torch {args.device.upper()} SparseStructureDecoder with FP16 torso",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
