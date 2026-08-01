#!/usr/bin/env python3
"""Capture the pinned sparse Flow-Euler/CFG behavior without model weights."""

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
sys.path.insert(0, str(PORT / "shims"))
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(PORT))

import runtime_env

runtime_env.activate()

SOURCE_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"


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


class DeterministicVelocityModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, float | str]] = []

    def __call__(self, state, timestep: torch.Tensor, condition: str):
        batch_timestep = timestep[state.batch_boardcast_map, None]
        indices = torch.arange(state.feats.numel(), dtype=torch.float32).reshape_as(
            state.feats
        )
        bias = 0.075 if condition == "positive" else -0.125
        self.calls.append(
            {"pass": condition, "model_timestep": float(timestep[0].item())}
        )
        velocity = (
            state.feats * 0.125
            + batch_timestep * 0.0001
            + indices * 0.002
            + bias
        )
        return state.replace(velocity)


def payload(tensors: list[torch.Tensor]) -> bytes:
    array = torch.cat([tensor.feats.reshape(-1) for tensor in tensors])
    return array.numpy().astype("<f4", copy=False).tobytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_pinned_source()
    runtime_env.configure_backends()

    from trellis2.modules.sparse import SparseTensor
    from trellis2.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler

    coords = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 2, 0],
            [1, 0, 3, 0],
        ],
        dtype=torch.int32,
    )
    noise_values = torch.tensor(
        [((index % 7) - 3) * 0.11 + index * 0.003 for index in range(20)],
        dtype=torch.float32,
    ).reshape(5, 4)
    noise = SparseTensor(noise_values, coords)
    model = DeterministicVelocityModel()
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    result = sampler.sample(
        model,
        noise,
        cond="positive",
        neg_cond="negative",
        steps=12,
        rescale_t=5,
        guidance_strength=7.5,
        guidance_rescale=0.7,
        guidance_interval=(0.6, 1.0),
        verbose=False,
    )

    final_payload = result.samples.feats.numpy().astype("<f4", copy=False).tobytes()
    previous_payload = payload(result.pred_x_t)
    x0_payload = payload(result.pred_x_0)
    trace_payload = previous_payload + x0_payload
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(final_payload)
    trace_path = args.output.with_suffix(args.output.suffix + ".trace")
    trace_path.write_bytes(trace_payload)
    unit = np.linspace(1, 0, 13)
    times = (5 * unit / (1 + 4 * unit)).tolist()
    metadata = {
        "format": "KernelGoblin TRELLIS.2 sparse Flow Euler F32 oracle v1",
        "source_revision": SOURCE_REVISION,
        "sampler": {
            "sigma_minimum": 1e-5,
            "steps": 12,
            "time_rescale": 5.0,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.7,
            "guidance_interval": [0.6, 1.0],
        },
        "schedule_f64": times,
        "layout_offsets": [0, 2, 5],
        "channels": 4,
        "noise_f32": noise_values.reshape(-1).tolist(),
        "velocity_formula": "state*0.125 + model_timestep*0.0001 + i*0.002 + pass_bias",
        "positive_bias": 0.075,
        "negative_bias": -0.125,
        "call_trace": model.calls,
        "final_sha256": hashlib.sha256(final_payload).hexdigest(),
        "trace_file": trace_path.name,
        "trace_layout": "12 xprev tensors followed by 12 x0 tensors",
        "trace_sha256": hashlib.sha256(trace_payload).hexdigest(),
        "oracle": "pinned Torch CPU FlowEulerGuidanceIntervalSampler with SparseTensor",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
