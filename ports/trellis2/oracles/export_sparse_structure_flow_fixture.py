#!/usr/bin/env python3
"""Capture a one-step production-token sparse-structure sampler oracle."""

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
WEIGHT_SHA256 = "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6"
SOURCE_PATH = UPSTREAM / "trellis2" / "models" / "sparse_structure_flow.py"


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
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], cwd=UPSTREAM
    ).returncode != 0:
        raise SystemExit("upstream checkout has tracked modifications")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    require_pinned_source()
    if sha256(args.checkpoint) != WEIGHT_SHA256:
        raise SystemExit("checkpoint SHA-256 does not match the pinned sparse flow")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("PyTorch MPS is unavailable")

    from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
    from trellis2.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler

    runtime_env.configure_backends()
    model = SparseStructureFlowModel(
        resolution=16,
        in_channels=8,
        model_channels=1536,
        cond_channels=1024,
        out_channels=8,
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
    state = load_file(args.checkpoint, device="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys != ["rope_phases"] or incompatible.unexpected_keys:
        raise SystemExit(f"incompatible sparse-flow checkpoint: {incompatible}")
    del state
    device = torch.device(args.device)
    model.to(device)

    resolution = 16
    tokens = resolution**3
    context_tokens = 1029
    token_major = deterministic(tokens * 8, 0.021, 0.30).reshape(tokens, 8)
    context = deterministic(context_tokens * 1024, 0.015, 0.25).reshape(
        1, context_tokens, 1024
    )
    negative = torch.zeros_like(context)
    volume = (
        token_major.reshape(resolution, resolution, resolution, 8)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .contiguous()
    )
    model_outputs: list[torch.Tensor] = []

    def counted_model(state, timestep, condition):
        result = model(state, timestep, condition)
        model_outputs.append(
            result.float().cpu().squeeze(0).permute(1, 2, 3, 0).contiguous()
        )
        return result

    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    with torch.inference_mode():
        sampler_result = sampler.sample(
            counted_model,
            volume.to(device),
            cond=context.to(device),
            neg_cond=negative.to(device),
            steps=1,
            rescale_t=5,
            guidance_strength=7.5,
            guidance_rescale=0.7,
            guidance_interval=(0.6, 1.0),
            verbose=False,
        )
    if args.device == "mps":
        torch.mps.synchronize()
    if len(model_outputs) != 2:
        raise SystemExit(f"expected two CFG model calls, got {len(model_outputs)}")
    output_token_major = (
        sampler_result.samples.float().cpu().squeeze(0)
        .permute(1, 2, 3, 0).contiguous()
    )

    input_payload = token_major.numpy().astype("<f4", copy=False).tobytes()
    context_payload = context.numpy().astype("<f4", copy=False).tobytes()
    trace_payload = b"".join(
        item.numpy().astype("<f4", copy=False).tobytes() for item in model_outputs
    )
    output_payload = output_token_major.numpy().astype("<f4", copy=False).tobytes()
    payload = input_payload + context_payload + trace_payload + output_payload
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    metadata = {
        "format": "KernelGoblin TRELLIS.2 production sparse sampler F32 oracle v1",
        "source_revision": SOURCE_REVISION,
        "source_path": str(SOURCE_PATH.relative_to(UPSTREAM)),
        "source_sha256": sha256(SOURCE_PATH),
        "weight_revision": WEIGHT_REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "device": args.device,
        "torch_version": torch.__version__,
        "cpu_fallback_enabled": False,
        "resolution": resolution,
        "tokens": tokens,
        "context_tokens": context_tokens,
        "input_channels": 8,
        "model_channels": 1536,
        "output_channels": 8,
        "blocks": 30,
        "steps": 1,
        "model_calls": 2,
        "model_timesteps": [1000.0, 1000.0],
        "rescale_t": 5.0,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.7,
        "guidance_interval": [0.6, 1.0],
        "coordinate_order": "(x * 16 + y) * 16 + z; z-fast",
        "input_formula": "f32(sin(i * 0.021) * 0.30)",
        "context_formula": "f32(sin(i * 0.015) * 0.25)",
        "input_range": [0, len(input_payload)],
        "context_range": [len(input_payload), len(input_payload) + len(context_payload)],
        "trace_range": [
            len(input_payload) + len(context_payload),
            len(input_payload) + len(context_payload) + len(trace_payload),
        ],
        "trace_layout": "positive then negative model output; each [4096,8] F32",
        "output_range": [
            len(input_payload) + len(context_payload) + len(trace_payload), len(payload)
        ],
        "output_space": "one-step Flow Euler sparse latent",
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "oracle": "pinned Torch SparseStructureFlowModel plus FlowEulerGuidanceIntervalSampler",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
