#!/usr/bin/env python3
"""Export a pinned Torch oracle for TRELLIS.2 sparse PBR trilinear sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


PORT = Path(__file__).resolve().parents[1]
ROOT = PORT.parents[1]
UPSTREAM = ROOT / "build" / "trellis2" / "upstream"
REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
UPSTREAM_SOURCE = UPSTREAM / "o-voxel" / "o_voxel" / "postprocess.py"
UPSTREAM_SOURCE_SHA256 = "ef51a1ba0f2748ffb4c265b47d382cee956f23c6a52d0f3587e6d8beccb7e54a"
SHIM_SOURCE = PORT / "shims" / "flex_gemm" / "ops" / "grid_sample" / "grid_sample.py"
SHIM_SOURCE_SHA256 = "87a6c7182bbfa5bf9deb713d003c1b972e7acc8e917d63257639c88856d55754"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip()
    if revision != REVISION:
        raise SystemExit("TRELLIS.2 upstream checkout is not pinned")
    if sha256(UPSTREAM_SOURCE) != UPSTREAM_SOURCE_SHA256:
        raise SystemExit("upstream O-Voxel PBR source is not pinned")
    if sha256(SHIM_SOURCE) != SHIM_SOURCE_SHA256:
        raise SystemExit("sparse grid-sample shim is not pinned")
    sys.path.insert(0, str(PORT / "shims"))
    from flex_gemm.ops.grid_sample import grid_sample_3d

    coordinates = []
    for x in range(4):
        for y in range(4):
            for z in range(4):
                if (x * 17 + y * 7 + z * 3) % 4 != 0:
                    coordinates.append([0, x, y, z])
    coordinates.reverse()
    coords = torch.tensor(coordinates, dtype=torch.int32)
    indices = np.arange(len(coordinates) * 6, dtype=np.float64)
    features = (
        np.sin(indices * 0.173) * 0.75 + np.cos(indices * 0.071) * 0.25
    ).astype(np.float32).reshape(-1, 6)
    query_values = []
    for index in range(37):
        query_values.append([
            (index % 11) * 0.41 - 0.2,
            ((index * 5) % 13) * 0.34 - 0.3,
            ((index * 7) % 17) * 0.27 - 0.4,
        ])
    queries = torch.tensor([query_values], dtype=torch.float32)
    output = grid_sample_3d(
        torch.from_numpy(features), coords,
        torch.Size([1, 6, 4, 4, 4]), queries, mode="trilinear"
    ).contiguous().numpy().astype("<f4", copy=False)
    payload = output.tobytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    metadata = {
        "format": "KernelGoblin TRELLIS.2 sparse PBR sampler Torch oracle v1",
        "source_revision": REVISION,
        "upstream_source": str(UPSTREAM_SOURCE.relative_to(UPSTREAM)),
        "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
        "shim_source": str(SHIM_SOURCE.relative_to(ROOT)),
        "shim_source_sha256": SHIM_SOURCE_SHA256,
        "oracle_call": "grid_sample_3d(..., mode='trilinear')",
        "torch_version": torch.__version__,
        "device": "cpu",
        "coordinate_generation": "reverse lexicographic 4^3 grid excluding (17x+7y+3z)%4==0",
        "feature_generation": "sin(i*0.173)*0.75 + cos(i*0.071)*0.25, float32",
        "query_generation": "37 deterministic affine modular positions",
        "shape": [1, 6, 4, 4, 4],
        "output_shape": list(output.shape),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
