#!/usr/bin/env python3
"""Capture sampled outputs from production-size BF16 SDPA on Torch MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PORT = Path(__file__).resolve().parents[1]
ROOT = PORT.parents[1]
UPSTREAM = ROOT / "build" / "trellis2" / "upstream"
SOURCE_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
SOURCE_PATH = UPSTREAM / "trellis2" / "modules" / "attention" / "full_attn.py"
TORCH_VERSION = "2.13.0"
SAMPLED_QUERIES = [0, 1, 2, 7, 31, 127, 511, 1023, 2047, 3071, 4088, 4089, 4090, 4091, 4094, 4095]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise SystemExit(f"upstream checkout is {revision}, expected {SOURCE_REVISION}")
    if torch.__version__ != TORCH_VERSION:
        raise SystemExit(f"Torch is {torch.__version__}, expected {TORCH_VERSION}")
    if not torch.backends.mps.is_available():
        raise SystemExit("PyTorch MPS is unavailable")
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    if fallback != "0":
        raise SystemExit("set PYTORCH_ENABLE_MPS_FALLBACK=0 to prove physical MPS execution")

    tokens = 4096
    heads = 12
    dimensions = 128
    count = tokens * heads * dimensions
    device = torch.device("mps")
    indices = np.arange(count, dtype=np.uint64)

    def values(multiplier: int, increment: int, scale: float) -> torch.Tensor:
        bits = ((indices * multiplier + increment) & 0xFFFFFFFF).astype(np.uint32)
        signed = bits.view(np.int32).astype(np.float32)
        return torch.from_numpy(signed / np.float32(2_147_483_647) * scale).to(
            device=device, dtype=torch.bfloat16
        )

    q = values(1_664_525, 1_013_904_223, 1.0)
    k = values(22_695_477, 1, 1.0)
    v = values(1_103_515_245, 12_345, 0.5)
    q = q.reshape(1, tokens, heads, dimensions).permute(0, 2, 1, 3)
    k = k.reshape(1, tokens, heads, dimensions).permute(0, 2, 1, 3)
    v = v.reshape(1, tokens, heads, dimensions).permute(0, 2, 1, 3)
    with torch.inference_mode():
        output = F.scaled_dot_product_attention(q, k, v)
    sampled = output.permute(0, 2, 1, 3)[0, SAMPLED_QUERIES].float().cpu().contiguous()
    torch.mps.synchronize()
    payload = sampled.numpy().astype("<f4", copy=False).tobytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    metadata = {
        "format": "KernelGoblin production-size BF16 Torch MPS SDPA sampled oracle v1",
        "source_revision": SOURCE_REVISION,
        "source_path": str(SOURCE_PATH.relative_to(UPSTREAM)),
        "source_sha256": sha256(SOURCE_PATH),
        "torch_version": torch.__version__,
        "device": "mps",
        "pytorch_enable_mps_fallback": fallback,
        "tokens": tokens,
        "heads": heads,
        "dimensions": dimensions,
        "sampled_queries": SAMPLED_QUERIES,
        "query_formula": "bf16_rne(i32(i*1664525+1013904223)/2147483647*1.0)",
        "key_formula": "bf16_rne(i32(i*22695477+1)/2147483647*1.0)",
        "value_formula": "bf16_rne(i32(i*1103515245+12345)/2147483647*0.5)",
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "oracle": "torch.nn.functional.scaled_dot_product_attention on physical MPS",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
