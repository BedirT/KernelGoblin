#!/usr/bin/env python3
"""Capture the pinned 12-step sparse flow and decoder handoff oracle."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import safetensors
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
FLOW_WEIGHT_SHA256 = "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6"
DECODER_WEIGHT_REVISION = "25e0d31ffbebe4b5a97464dd851910efc3002d96"
DECODER_WEIGHT_SHA256 = "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a"
EXPECTED_VERSIONS = {
    "torch": "2.13.0",
    "safetensors": "0.7.0",
    "numpy": "2.4.6",
}
SOURCE_PATHS = [
    Path("trellis2/models/sparse_structure_flow.py"),
    Path("trellis2/models/sparse_structure_vae.py"),
    Path("trellis2/pipelines/samplers/flow_euler.py"),
    Path("trellis2/pipelines/samplers/classifier_free_guidance_mixin.py"),
    Path("trellis2/pipelines/samplers/guidance_interval_mixin.py"),
]


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


def append(payload: bytearray, value: bytes) -> list[int]:
    start = len(payload)
    payload.extend(value)
    return [start, len(payload)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    require_pinned_source()
    if sha256(args.flow_checkpoint) != FLOW_WEIGHT_SHA256:
        raise SystemExit("flow checkpoint SHA-256 does not match the pinned payload")
    if sha256(args.decoder_checkpoint) != DECODER_WEIGHT_SHA256:
        raise SystemExit("decoder checkpoint SHA-256 does not match the pinned payload")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("PyTorch MPS is unavailable")
    actual_versions = {
        "torch": torch.__version__,
        "safetensors": safetensors.__version__,
        "numpy": np.__version__,
    }
    if actual_versions != EXPECTED_VERSIONS:
        raise SystemExit(
            f"oracle dependency versions are {actual_versions}, expected {EXPECTED_VERSIONS}"
        )

    from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
    from trellis2.models.sparse_structure_vae import SparseStructureDecoder
    from trellis2.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler

    runtime_env.configure_backends()
    device = torch.device(args.device)
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
    incompatible = model.load_state_dict(
        load_file(args.flow_checkpoint, device="cpu"), strict=False
    )
    if incompatible.missing_keys != ["rope_phases"] or incompatible.unexpected_keys:
        raise SystemExit(f"incompatible sparse-flow checkpoint: {incompatible}")
    model.to(device)

    resolution = 16
    tokens = resolution**3
    context_tokens = 1029
    steps = 12
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
    model_calls = 0
    model_outputs: list[np.ndarray] = []

    def counted_model(state, timestep, condition):
        nonlocal model_calls
        model_calls += 1
        result = model(state, timestep, condition)
        model_outputs.append(
            result.float().cpu().squeeze(0).permute(1, 2, 3, 0).contiguous().numpy()
        )
        return result

    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    with torch.inference_mode():
        sampled = sampler.sample(
            counted_model,
            volume.to(device),
            cond=context.to(device),
            neg_cond=negative.to(device),
            steps=steps,
            rescale_t=5,
            guidance_strength=7.5,
            guidance_rescale=0.7,
            guidance_interval=(0.6, 1.0),
            verbose=False,
        )
    if args.device == "mps":
        torch.mps.synchronize()
    if model_calls != 22 or len(sampled.pred_x_t) != steps:
        raise SystemExit(
            f"expected 22 model calls and {steps} states, got "
            f"{model_calls} and {len(sampled.pred_x_t)}"
        )
    state_arrays = [
        state.float().cpu().squeeze(0).permute(1, 2, 3, 0).contiguous().numpy()
        for state in sampled.pred_x_t
    ]
    final_token_major = state_arrays[-1]

    del sampled, sampler, model
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()

    decoder = SparseStructureDecoder(
        out_channels=1,
        latent_channels=8,
        num_res_blocks=2,
        num_res_blocks_middle=2,
        channels=[512, 128, 32],
        use_fp16=True,
    ).eval()
    decoder.load_state_dict(load_file(args.decoder_checkpoint), strict=True)
    decoder.to(device)
    latent = torch.from_numpy(final_token_major.copy()).permute(3, 0, 1, 2).unsqueeze(0)
    with torch.inference_mode():
        logits = decoder(latent.to(device)).float().cpu().contiguous()
    if args.device == "mps":
        torch.mps.synchronize()
    if list(logits.shape) != [1, 1, 64, 64, 64]:
        raise SystemExit(f"unexpected decoder output shape {list(logits.shape)}")
    logits_array = logits.numpy().reshape(-1).astype("<f4", copy=False)
    occupancy = logits_array > 0

    payload = bytearray()
    input_range = append(payload, token_major.numpy().astype("<f4", copy=False).tobytes())
    context_range = append(payload, context.numpy().astype("<f4", copy=False).tobytes())
    model_trace_range = append(
        payload,
        b"".join(output.astype("<f4", copy=False).tobytes() for output in model_outputs),
    )
    state_trace_range = append(
        payload,
        b"".join(state.astype("<f4", copy=False).tobytes() for state in state_arrays),
    )
    final_latent_range = append(
        payload, final_token_major.astype("<f4", copy=False).tobytes()
    )
    logits_range = append(payload, logits_array.tobytes())
    occupancy_range = append(
        payload, np.packbits(occupancy, bitorder="little").tobytes()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)

    times = np.linspace(1, 0, steps + 1)
    times = 5 * times / (1 + 4 * times)
    model_timesteps = [
        float(1000 * time)
        for time in times[:-1]
        for _ in range(2 if 0.6 <= time <= 1.0 else 1)
    ]
    metadata = {
        "format": "KernelGoblin TRELLIS.2 full sparse trajectory and handoff oracle v1",
        "source_revision": SOURCE_REVISION,
        "source_files": {
            str(path): sha256(UPSTREAM / path) for path in SOURCE_PATHS
        },
        "flow_weight_revision": WEIGHT_REVISION,
        "flow_weight_sha256": FLOW_WEIGHT_SHA256,
        "decoder_weight_revision": DECODER_WEIGHT_REVISION,
        "decoder_weight_sha256": DECODER_WEIGHT_SHA256,
        "device": args.device,
        "torch_version": torch.__version__,
        "safetensors_version": safetensors.__version__,
        "cpu_fallback_enabled": False,
        "resolution": resolution,
        "tokens": tokens,
        "context_tokens": context_tokens,
        "steps": steps,
        "model_calls": model_calls,
        "model_timesteps": model_timesteps,
        "rescale_t": 5.0,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.7,
        "guidance_interval": [0.6, 1.0],
        "coordinate_order": "(x * 16 + y) * 16 + z; z-fast",
        "input_formula": "f32(sin(i * 0.021) * 0.30)",
        "context_formula": "f32(sin(i * 0.015) * 0.25)",
        "input_range": input_range,
        "context_range": context_range,
        "model_trace_range": model_trace_range,
        "model_trace_layout": "22 consecutive token-major [4096,8] F32 model outputs",
        "state_trace_range": state_trace_range,
        "state_trace_layout": "12 consecutive token-major [4096,8] F32 post-step states",
        "final_latent_range": final_latent_range,
        "logits_range": logits_range,
        "occupancy_range": occupancy_range,
        "occupancy_bit_order": "little; flat x-major/y-middle/z-fast 64-cubed grid",
        "occupancy_count": int(occupancy.sum()),
        "stage_boundary": "flow tensors copied to CPU before model release and decoder load",
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "oracle": "pinned Torch flow sampler, explicit stage release, then pinned decoder",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
