#!/usr/bin/env python3
"""Capture a pinned Torch oracle for the complete 30-block shape SLat flow."""

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
WEIGHT_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
WEIGHT_SHA256 = "ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic(count: int, multiplier: float, scale: float) -> torch.Tensor:
    return torch.tensor(
        [math.sin(index * multiplier) * scale for index in range(count)],
        dtype=torch.float32,
    )


def require_pinned_source() -> None:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise SystemExit(f"upstream checkout is {revision}, expected {SOURCE_REVISION}")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=UPSTREAM).returncode != 0:
        raise SystemExit("upstream checkout has tracked modifications")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_pinned_source()
    if sha256(args.checkpoint) != WEIGHT_SHA256:
        raise SystemExit("checkpoint SHA-256 does not match the pinned 512 shape-flow file")

    from trellis2.models.structured_latent_flow import SLatFlowModel
    from trellis2.modules.sparse import SparseTensor

    runtime_env.configure_backends()
    model = SLatFlowModel(
        resolution=32,
        in_channels=32,
        model_channels=1536,
        cond_channels=1024,
        out_channels=32,
        num_blocks=30,
        num_heads=12,
        mlp_ratio=5.3334,
        pe_mode="rope",
        share_mod=True,
        initialization="scaled",
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        dtype="bfloat16",
    ).eval()
    model.load_state_dict(load_file(args.checkpoint, device="cpu"), strict=True)

    tokens, context_tokens = 2, 2
    features = deterministic(tokens * 32, 0.021, 0.30).reshape(tokens, 32)
    timestep = torch.tensor([650.25], dtype=torch.float32)
    context = deterministic(context_tokens * 1024, 0.015, 0.25).reshape(
        1, context_tokens, 1024
    )
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=torch.int32)
    block_outputs: list[torch.Tensor] = []
    hooks = [
        block.register_forward_hook(
            lambda _module, _inputs, output: block_outputs.append(output.feats.detach().clone())
        )
        for block in model.blocks
    ]
    with torch.inference_mode():
        output = model(SparseTensor(features, coords), timestep, context).feats
    for hook in hooks:
        hook.remove()

    output_payload = output.contiguous().cpu().numpy().astype("<f4", copy=False).tobytes()
    trace_payload = b"".join(
        tensor.contiguous().view(torch.uint16).cpu().numpy().astype("<u2", copy=False).tobytes()
        for tensor in block_outputs
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output_payload)
    trace_path = args.output.with_suffix(args.output.suffix + ".trace")
    trace_path.write_bytes(trace_payload)
    metadata = {
        "format": "KernelGoblin TRELLIS.2 30-block shape SLat flow F32 oracle v1",
        "source_revision": SOURCE_REVISION,
        "weight_revision": WEIGHT_REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "batch_count": 1,
        "tokens": tokens,
        "context_tokens": context_tokens,
        "input_channels": 32,
        "model_channels": 1536,
        "output_channels": 32,
        "blocks": len(block_outputs),
        "use_rope": True,
        "timestep": 650.25,
        "input_formula": "f32(sin(i * 0.021) * 0.30)",
        "context_formula": "f32(sin(i * 0.015) * 0.25)",
        "coordinates": coords.tolist(),
        "output_bytes": len(output_payload),
        "output_sha256": hashlib.sha256(output_payload).hexdigest(),
        "trace_file": trace_path.name,
        "trace_dtype": "BF16",
        "trace_elements_per_block": tokens * 1536,
        "trace_sha256": hashlib.sha256(trace_payload).hexdigest(),
        "oracle": "pinned Torch CPU SLatFlowModel",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
