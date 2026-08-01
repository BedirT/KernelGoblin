#!/usr/bin/env python3
"""Capture a real-checkpoint two-step texture SLat sampler oracle."""

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
WEIGHT_SHA256 = "8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d"
SHAPE_MEAN = [
    0.781296, 0.018091, -0.495192, -0.558457, 1.060530, 0.093252, 1.518149, -0.933218,
    -0.732996, 2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944,
    0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
    -0.059621, 2.220780, 1.621212, 0.877230, 0.567247, -3.175944, -3.186688, 1.578665,
]
SHAPE_STD = [
    5.972266, 4.706852, 5.445010, 5.209927, 5.320220, 4.547237, 5.020802, 5.444004,
    5.226681, 5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578,
    4.805265, 5.124013, 5.530808, 5.619001, 5.103930, 5.417670, 5.269677, 5.547194,
    5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691,
]
TEXTURE_MEAN = [
    3.501659, 2.212398, 2.226094, 0.251093, -0.026248, -0.687364, 0.439898, -0.928075,
    0.029398, -0.339596, -0.869527, 1.038479, -0.972385, 0.126042, -1.129303, 0.455149,
    -1.209521, 2.069067, 0.544735, 2.569128, -0.323407, 2.293000, -1.925608, -1.217717,
    1.213905, 0.971588, -0.023631, 0.106750, 2.021786, 0.250524, -0.662387, -0.768862,
]
TEXTURE_STD = [
    2.665652, 2.743913, 2.765121, 2.595319, 3.037293, 2.291316, 2.144656, 2.911822,
    2.969419, 2.501689, 2.154811, 3.163343, 2.621215, 2.381943, 3.186697, 3.021588,
    2.295916, 3.234985, 3.233086, 2.260140, 2.874801, 2.810596, 3.292720, 2.674999,
    2.680878, 2.372054, 2.451546, 2.353556, 2.995195, 2.379849, 2.786195, 2.775190,
]


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
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=UPSTREAM).returncode:
        raise SystemExit("upstream checkout has tracked modifications")


def deterministic(count: int, multiplier: float, scale: float, trig=math.sin) -> torch.Tensor:
    return torch.tensor(
        [trig(index * multiplier) * scale for index in range(count)], dtype=torch.float32
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_pinned_source()
    if sha256(args.checkpoint) != WEIGHT_SHA256:
        raise SystemExit("checkpoint SHA-256 does not match the pinned 512 texture-flow file")

    from trellis2.models.structured_latent_flow import SLatFlowModel
    from trellis2.modules.sparse import SparseTensor
    from trellis2.pipelines.samplers.flow_euler import FlowEulerSampler

    runtime_env.configure_backends()
    model = SLatFlowModel(
        resolution=32, in_channels=64, model_channels=1536, cond_channels=1024,
        out_channels=32, num_blocks=30, num_heads=12, mlp_ratio=5.3334,
        pe_mode="rope", share_mod=True, initialization="scaled",
        qk_rms_norm=True, qk_rms_norm_cross=True, dtype="bfloat16",
    ).eval()
    model.load_state_dict(load_file(args.checkpoint, device="cpu"), strict=True)
    tokens, context_tokens = 2, 2
    noise = deterministic(tokens * 32, 0.021, 0.30).reshape(tokens, 32)
    normalized_shape = deterministic(tokens * 32, 0.017, 0.20, math.cos).reshape(tokens, 32)
    shape = normalized_shape * torch.tensor(SHAPE_STD) + torch.tensor(SHAPE_MEAN)
    context = deterministic(context_tokens * 1024, 0.015, 0.25).reshape(
        1, context_tokens, 1024
    )
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=torch.int32)
    calls: list[float] = []
    model_outputs: list[torch.Tensor] = []

    def counted_model(state, timestep, condition):
        calls.append(float(timestep[0].item()))
        features = torch.cat([state.feats, normalized_shape], dim=1)
        output = model(SparseTensor(features, state.coords), timestep, condition)
        model_outputs.append(output.feats.float().cpu())
        return output

    sampler = FlowEulerSampler(sigma_min=1e-5)
    with torch.inference_mode():
        sampler_result = sampler.sample(
            counted_model, SparseTensor(noise, coords), cond=context,
            steps=2, rescale_t=3, verbose=False,
        )
        output = sampler_result.samples.feats * torch.tensor(TEXTURE_STD) + torch.tensor(TEXTURE_MEAN)
    output_payload = output.contiguous().numpy().astype("<f4", copy=False).tobytes()
    trace = torch.stack(
        [*model_outputs, *[sample.feats.float().cpu() for sample in sampler_result.pred_x_t]]
    )
    trace_payload = trace.contiguous().numpy().astype("<f4", copy=False).tobytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output_payload)
    trace_path = args.output.with_suffix(args.output.suffix + ".trace")
    trace_path.write_bytes(trace_payload)
    metadata = {
        "format": "KernelGoblin TRELLIS.2 two-step texture SLat sampler F32 oracle v1",
        "source_revision": SOURCE_REVISION,
        "weight_revision": WEIGHT_REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "tokens": tokens,
        "context_tokens": context_tokens,
        "steps": 2,
        "model_calls": len(calls),
        "model_timesteps": calls,
        "noise_formula": "f32(sin(i * 0.021) * 0.30)",
        "normalized_shape_formula": "f32(cos(i * 0.017) * 0.20)",
        "context_formula": "f32(sin(i * 0.015) * 0.25)",
        "coordinates": coords.tolist(),
        "input_layout": "[32-channel sampler state, 32-channel normalized shape]",
        "output_space": "denormalized texture latent",
        "output_sha256": hashlib.sha256(output_payload).hexdigest(),
        "trace_layout": "two model outputs followed by two pred_x_t states, all [2,32] F32",
        "trace_sha256": hashlib.sha256(trace_payload).hexdigest(),
        "oracle": "pinned Torch CPU SLatFlowModel plus FlowEulerSampler",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
